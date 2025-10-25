
import copy
import math
from dataclasses import dataclass
import os
import pdb
import sys
from typing import List, Optional, Tuple, Union

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, LayerNorm
from torch.nn import MSELoss
import numpy as np
import torch.distributed as dist
import itertools, torch

from functools import partial

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    ModelOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .configuration_qwen2_vl import Qwen2VLConfig, Qwen2VLVisionConfig
from .modeling_bitransformer import GenBiTransformer, BiT_config


if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func

    from transformers.modeling_flash_attention_utils import _flash_attention_forward
else:
    flash_attn_varlen_func = None

from timm.models.layers import DropPath

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "Qwen2VLConfig"


@dataclass
class Qwen2VLCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Qwen2VL causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


class Qwen2VLRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[Qwen2VLConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`Qwen2VLRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block. In contrast to other models, Qwen2_VL has different position ids for thw grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension seperately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.to(orig_dtype)
    return output


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class PatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


class VisionMlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = ACT2FN[hidden_act]
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class VisionAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor = None
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attention_mask = torch.full(
            [1, seq_length, seq_length], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class VisionFlashAttention2(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor = None
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output


class VisionSdpaAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor = None
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


QWEN2_VL_VISION_ATTENTION_CLASSES = {
    "eager": VisionAttention,
    "flash_attention_2": VisionFlashAttention2,
    "sdpa": VisionSdpaAttention,
}


class Qwen2VLVisionBlock(nn.Module):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = LayerNorm(config.embed_dim, eps=1e-6)
        self.norm2 = LayerNorm(config.embed_dim, eps=1e-6)
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)

        self.attn = QWEN2_VL_VISION_ATTENTION_CLASSES[attn_implementation](
            config.embed_dim, num_heads=config.num_heads
        )
        self.mlp = VisionMlp(dim=config.embed_dim, hidden_dim=mlp_hidden_dim, hidden_act=config.hidden_act)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


# Copied from transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# Copied from transformers.models.qwen2.modeling_qwen2.Qwen2MLP
class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2VLAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2VLConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        # pdb.set_trace()
        self.rope_scaling = config.rope_scaling

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2VLRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        # pdb.set_trace()
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Fix precision issues in Qwen2-VL float16 inference
        # Replace inf values with zeros in attention weights to prevent NaN propagation
        if query_states.dtype == torch.float16:
            attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen2VLFlashAttention2(Qwen2VLAttention):
    """
    Qwen2VL flash attention module, following Qwen2VL attention module. This module inherits from `Qwen2VLAttention`
    as the weights of the module stays untouched. The only required change would be on the forward pass
    where it needs to correctly call the public API of flash attention and deal with padding tokens
    in case the input contains any of them. Additionally, for sliding window attention, we apply SWA only to the bottom
    config.max_window_layers layers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # pdb.set_trace()
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        # pdb.set_trace()
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        # pdb.set_trace()
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class BiQwen2VLFlashAttention2(Qwen2VLAttention):
    """
    Qwen2VL flash attention module, following Qwen2VL attention module. This module inherits from `Qwen2VLAttention`
    as the weights of the module stays untouched. The only required change would be on the forward pass
    where it needs to correctly call the public API of flash attention and deal with padding tokens
    in case the input contains any of them. Additionally, for sliding window attention, we apply SWA only to the bottom
    config.max_window_layers layers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_causal = False
        if self.config.hidden_size == 2048:
            self.rope_scaling["mrope_section"] = [8, 12, 12]

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.config.hidden_size == 1024:
            self.rope_scaling["mrope_section"] = [4,6,6]
        elif self.config.hidden_size == 2048:
            self.rope_scaling["mrope_section"] = [8,12,12]

        cur_rope_scaling = self.rope_scaling["mrope_section"]

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        # pdb.set_trace()
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, cur_rope_scaling
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            # logger.warning_once(
            #     f"The input hidden states seems to be silently casted in float32, this might be related to"
            #     f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            #     f" {target_dtype}."
            # )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        # pdb.set_trace()
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    


class Qwen2VLSdpaAttention(Qwen2VLAttention):
    """
    Qwen2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Qwen2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from Qwen2Attention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Qwen2VLModel is using Qwen2VLSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


QWEN2_VL_ATTENTION_CLASSES = {
    "eager": Qwen2VLAttention,
    "flash_attention_2": Qwen2VLFlashAttention2,
    "sdpa": Qwen2VLSdpaAttention,
}


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->Qwen2VLBiTransformer
class Qwen2VLBiTransformerIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->Qwen2VLBiTransformer
class Qwen2VLBiTransformerOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    

class Qwen2VLCrossAttention(Qwen2VLAttention):
    """
    Qwen2VL flash attention module, following Qwen2VL attention module. This module inherits from `Qwen2VLAttention`
    as the weights of the module stays untouched. The only required change would be on the forward pass
    where it needs to correctly call the public API of flash attention and deal with padding tokens
    in case the input contains any of them. Additionally, for sliding window attention, we apply SWA only to the bottom
    config.max_window_layers layers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

        self.is_causal = False
        if self.config.hidden_size == 1024:
            self.rope_scaling["mrope_section"] = [4,6,6]
        elif self.config.hidden_size == 2048:
            self.rope_scaling["mrope_section"] = [8,12,12]


        self.k_proj = nn.Linear(self.config.encoder_hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.config.encoder_hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        # pdb.set_trace()
        # self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        

        self.rotary_emb = Qwen2VLRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    ):
        bsz, q_len, _ = hidden_states.size()
        bsz_1, q_len_1, _ = encoder_hidden_states.size()
        # pdb.set_trace()
        # 判断是否进行交叉注意力
        is_cross_attention = encoder_hidden_states is not None

        # 查询（Q）始终来自于 hidden_states
        query_states = self.q_proj(hidden_states)
        # pdb.set_trace()
        if is_cross_attention:
            # 键（K）和值（V）来自于 encoder_hidden_states
            key_states = self.k_proj(encoder_hidden_states)
            value_states = self.v_proj(encoder_hidden_states)
            # 注意力掩码也使用 encoder_attention_mask
            attention_mask = encoder_attention_mask
        else:
            # 键（K）和值（V）来自于 hidden_states（自注意力）
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz_1, q_len_1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz_1, q_len_1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # pdb.set_trace()
        if self.config.hidden_size == 1024:
            self.rope_scaling["mrope_section"] = [4,6,6]
        elif self.config.hidden_size == 2048:
            self.rope_scaling["mrope_section"] = [8,12,12]
        cur_rope_scaling = self.rope_scaling["mrope_section"]
        # Because the input can be padded, the absolute sequence length depends on the max position id.
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        # pdb.set_trace()

        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, cur_rope_scaling
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            # logger.warning_once(
            #     f"The input hidden states seems to be silently casted in float32, this might be related to"
            #     f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            #     f" {target_dtype}."
            # )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    
    
    

class Qwen2VLDecoderLayerWithCrossAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = BiQwen2VLFlashAttention2(config, layer_idx)
        # pdb.set_trace()
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # if layer_idx % config.cross_attention_frequency == 0:
        # # if layer_idx == 0:
        #     self.crossattention = Qwen2VLCrossAttention(config, layer_idx)
        #     self.has_cross_attention = True
        # else:
        #     self.has_cross_attention = False

        self.crossattention = Qwen2VLCrossAttention(config, layer_idx)
        self.has_cross_attention = True

        # if True:
        #     self.intermediate = Qwen2VLBiTransformerIntermediate(config)
        #     self.output = Qwen2VLBiTransformerOutput(config)
        # self.intermediate_query = Qwen2VLBiTransformerIntermediate(config)
        # self.output_query = Qwen2VLBiTransformerOutput(config)



    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        query_length=0,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # pdb.set_trace()
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states.to(torch.bfloat16),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        
        hidden_states = residual + hidden_states

        if query_length > 0:
            hidden_states = hidden_states[:, :query_length, :]
            corss_residual = hidden_states

            if self.has_cross_attention:
                if encoder_hidden_states is None:
                    raise ValueError("encoder_hidden_states must be given for cross-attention layers")

                cross_hidden_states, cross_attn_weights, cross_present_key_value = self.crossattention(
                    hidden_states=hidden_states.to(torch.bfloat16),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                # pdb.set_trace()
                hidden_states = corss_residual + cross_hidden_states

            # # pdb.set_trace()
            # hidden_states = apply_chunking_to_forward(
            #     self.feed_forward_chunk_query,
            #     self.chunk_size_feed_forward,
            #     self.seq_len_dim,
            #     hidden_states,
            # )

            # if hidden_states.shape[1] > query_length:
            #     layer_output_text = apply_chunking_to_forward(
            #         self.feed_forward_chunk,
            #         self.chunk_size_feed_forward,
            #         self.seq_len_dim,
            #         hidden_states[:, query_length:, :],
            #     )
            #     hidden_states = torch.cat([hidden_states, layer_output_text], dim=1)
        # else:
        #     hidden_states = apply_chunking_to_forward(
        #         self.feed_forward_chunk,
        #         self.chunk_size_feed_forward,
        #         self.seq_len_dim,
        #         hidden_states,
        #     )



        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    
    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def feed_forward_chunk_query(self, attention_output):
        intermediate_output = self.intermediate_query(attention_output)
        layer_output = self.output_query(intermediate_output, attention_output)
        return layer_output



class Qwen2VLDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2VLConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = QWEN2_VL_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


QWEN2VL_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen2VLConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen2VL Model outputting raw hidden-states without any specific head on top.",
    QWEN2VL_START_DOCSTRING,
)
class Qwen2VLPreTrainedModel(PreTrainedModel):
    config_class = Qwen2VLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2VLDecoderLayer", "Qwen2VLVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Qwen2VisionTransformerPretrainedModel(Qwen2VLPreTrainedModel):
    config_class = Qwen2VLVisionConfig
    _no_split_modules = ["Qwen2VLVisionBlock"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [Qwen2VLVisionBlock(config, config._attn_implementation) for _ in range(config.depth)]
        )
        self.merger = PatchMerger(
            dim=config.hidden_size, context_dim=config.embed_dim, spatial_merge_size=config.spatial_merge_size
        )
        self.gradient_checkpointing = False

    def get_dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    def get_device(self) -> torch.device:
        return self.blocks[0].mlp.fc2.weight.device

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens, rotary_pos_emb
                )
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

        return self.merger(hidden_states)


@add_start_docstrings(
    "The bare Qwen2VL Model outputting raw hidden-states without any specific head on top.",
    QWEN2VL_START_DOCSTRING,
)
class Qwen2VLModel(Qwen2VLPreTrainedModel):
    def __init__(self, config: Qwen2VLConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2VLRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        # pdb.set_trace()
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # Copied from transformers.models.phi3.modeling_phi3.Phi3Model._update_causal_mask with Phi3->Qwen2VL
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen2VL. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    # Copied from transformers.models.mistral.modeling_mistral.MistralModel._prepare_4d_causal_attention_mask_with_cache_position with Mistral->Qwen2VL
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen2VLConfig,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen2VLConfig`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device) <= (
                        cache_position.reshape(-1, 1) - config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask


@dataclass
class BiTransformerConfig:
    architectures: List[str]
    attention_dropout: float
    rms_norm_eps: float


    bos_token_id: int
    eos_token_id: int
    image_token_id: int
    video_token_id: int
    vision_end_token_id: int
    vision_start_token_id: int
    hidden_act: str
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    max_position_embeddings: int
    model_type: str
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    num_layers: int
    max_window_layers: int
    rope_scaling: dict
    rope_theta: float
    sliding_window: int
    tie_word_embeddings:bool
    torch_dtype: str

    transformers_version: str
    use_cache: bool
    use_sliding_window: bool
    vision_config: dict
    vocab_size: int

    num_query_tokens: int
    encoder_hidden_size: int
    layer_norm_eps: int
    hidden_dropout_prob: float
    attention_probs_dropout_prob: float
    cross_attention_frequency: int
    chunk_size_feed_forward: int 
    _attn_implementation: str
    use_qformer_text_input: bool
    has_learned_query: bool
    position_embedding_type: str


    @classmethod
    def from_dict(cls, config_dict: dict) -> 'BiTransformerConfig':
        return cls(**config_dict)
    
BiTransformer_Config = {
  "architectures": [
    "Qwen2VLForConditionalGeneration"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 3584,
  "image_token_id": 151655,
  "initializer_range": 0.02,
  "intermediate_size": 18944,
  "max_position_embeddings": 32768,
  "max_window_layers": 32,
  "model_type": "qwen2_vl",
  "num_attention_heads": 32,
  "num_hidden_layers": 3,
  "num_layers": 3,
  "num_key_value_heads": 4,
  "rms_norm_eps": 1e-06,
  "rope_scaling": {
    "mrope_section": [
      16,
      24,
      24
    ],
    "rope_type": "default",
    "type": "default"
  },
  "rope_theta": 1000000.0,
  "sliding_window": 32768,
  "tie_word_embeddings": False,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.44.2",
  "use_cache": True,
  "use_sliding_window": False,
  "video_token_id": 151656,
  "vision_config": {
    "in_chans": 3,
    "model_type": "qwen2_vl",
    "spatial_patch_size": 14
  },
  "vision_end_token_id": 151653,
  "vision_start_token_id": 151652,
#   "vision_token_id": 151654,
  "vocab_size": 152064,


  "num_query_tokens": 256,
  "encoder_hidden_size": 3584,
  "layer_norm_eps": 1e-12,
  "hidden_dropout_prob": 0.1,
  "attention_probs_dropout_prob": 0.1,
  "cross_attention_frequency": 2,
  "chunk_size_feed_forward": 0,
  "_attn_implementation": "flash_attention_2",
  "use_qformer_text_input": False,
  "has_learned_query": False,
  "position_embedding_type":'absolute',

#   "bi_intermediate_size": 3072,
}

class BiTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([Qwen2VLDecoderLayerWithCrossAttention(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

        self.has_learned_query = config.has_learned_query
        if config.has_learned_query:
            self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.hidden_size))
            
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


        self.rotary_emb = Qwen2VLRotaryEmbedding(config=config)


    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        all_cross_attentions = None
        # pdb.set_trace()
        use_cache = False

        def pad_or_truncate_with_F_pad(hidden_states, padding, max_len=256):
            batch_size, current_len, hidden_size = hidden_states.size()

            if current_len < max_len:
                pad_size = max_len - current_len
                # F.pad 的 pad 参数是从最后一个维度开始，依次为 (左, 右, 上, 下, ...)
                # 在第二维度（len）上右侧填充 pad_size
                hidden_states_padded = F.pad(hidden_states, (0, 0, 0, pad_size), "constant", padding)
                return hidden_states_padded
            elif current_len > max_len:
                hidden_states_truncated = hidden_states[:, :max_len, :]
                return hidden_states_truncated
            else:
                return hidden_states
            
        def pad_or_truncate_attention_mask(attention_mask, max_len=333):
            """
            对 attention_mask 在第二维度进行右侧填充或截断，使其长度固定为 max_len。
            
            参数:
            - attention_mask (torch.Tensor): 输入张量，形状为 (batch_size, len)
            - max_len (int): 目标长度，默认为 333
            
            返回:
            - torch.Tensor: 处理后的张量，形状为 (batch_size, max_len)
            """
            batch_size, current_len = attention_mask.size()
            
            if current_len < max_len:
                # 计算需要填充的长度
                pad_size = max_len - current_len
                # 创建一个全零的填充张量，形状为 (batch_size, pad_size)
                padding = torch.zeros(batch_size, pad_size, device=attention_mask.device, dtype=attention_mask.dtype)
                # 在第二维度上拼接填充张量
                attention_mask_padded = torch.cat([attention_mask, padding], dim=1)
                return attention_mask_padded
            elif current_len > max_len:
                # 截断到 max_len
                attention_mask_truncated = attention_mask[:, :max_len]
                return attention_mask_truncated
            else:
                # 如果已经是目标长度，则返回原始张量
                return attention_mask
            
        encoder_hidden_states = pad_or_truncate_with_F_pad(encoder_hidden_states, 0.0, 256)
        encoder_attention_mask = pad_or_truncate_attention_mask(encoder_attention_mask, 256)
        input_shape = hidden_states.size()[:-1]

        batch_size, seq_length = input_shape

        # step 1: forward the images through the vision encoder,
        # image_attention_mask = torch.ones(hidden_states.size()[:-1], dtype=torch.long, device=hidden_states.device)
        if self.has_learned_query:
            query_tokens = self.query_tokens.expand(hidden_states.shape[0], -1, -1)
        elif hidden_states is not None:
            query_tokens = hidden_states

        query_length = None
        query_length = (
            query_length if query_length is not None else query_tokens.shape[1] if query_tokens is not None else 0
        )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + query_tokens.shape[1], device=query_tokens.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, query_tokens.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)


        embedding_output = self.layernorm(query_tokens)
        embedding_output = self.dropout(embedding_output)

        input_shape = embedding_output.size()[:-1]
        batch_size, seq_length = input_shape
        device = embedding_output.device

        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        encoder_extended_attention_mask = encoder_attention_mask


        # pdb.set_trace()

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] - self.config.query_length if past_key_values is not None else 0
        )
        attention_mask = None
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        query_attention_mask = torch.ones(((batch_size, seq_length)), device=device)


        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(embedding_output, position_ids)

        all_hidden_states = None
        all_self_attns = None
        next_decoder_cache = None
        

        for layer in self.layers:

            # if self.gradient_checkpointing and self.training:
            #     layer_outputs = self._gradient_checkpointing_func(
            #         layer.forward,
            #         embedding_output,
            #         attention_mask=query_attention_mask,
            #         encoder_hidden_states=encoder_hidden_states,
            #         encoder_attention_mask=encoder_extended_attention_mask,
            #         query_length=query_length,
            #         position_ids=position_ids,
            #         past_key_value=past_key_values,
            #         output_attentions=output_attentions,
            #         use_cache=use_cache,
            #         cache_position=cache_position,
            #         position_embeddings=position_embeddings,
            #     )
            # else:
            #     layer_outputs = layer(
            #         embedding_output,
            #         attention_mask=query_attention_mask,
            #         encoder_hidden_states=encoder_hidden_states,
            #         encoder_attention_mask=encoder_extended_attention_mask,
            #         query_length=query_length,
            #         position_ids=position_ids,
            #         past_key_value=past_key_values,
            #         output_attentions=output_attentions,
            #         use_cache=use_cache,
            #         cache_position=cache_position,
            #         position_embeddings=position_embeddings,
            #     )

            layer_outputs = layer(
                    embedding_output,
                    attention_mask=query_attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    query_length=query_length,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            embedding_output = layer_outputs[0]

        return embedding_output
    
    # def invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor:
    #     """
    #     Invert an attention mask (e.g., switches 0. and 1.).

    #     Args:
    #         encoder_attention_mask (`torch.Tensor`): An attention mask.

    #     Returns:
    #         `torch.Tensor`: The inverted attention mask.
    #     """
    #     if encoder_attention_mask.dim() == 3:
    #         encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
    #     if encoder_attention_mask.dim() == 2:
    #         encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    #     # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
    #     # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
    #     # /transformer/transformer_layers.py#L270
    #     # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
    #     # encoder_extended_attention_mask.transpose(-1, -2))
    #     encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
    #     encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(self.dtype).min

    #     return encoder_extended_attention_mask


class FP32_SiLU(nn.SiLU):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(inputs.float(), inplace=False).to(inputs.dtype)

QWEN2_VL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        pixel_values (`torch.FloatTensor` of shape `(seq_length, num_channels * image_size * image_size)):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`Qwen2VLImageProcessor.__call__`] for details. [`Qwen2VLProcessor`] uses
            [`Qwen2VLImageProcessor`] for processing images.
        pixel_values_videos (`torch.FloatTensor` of shape `(seq_length, num_channels * temporal_size * image_size * image_size)):
            The tensors corresponding to the input videos. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`Qwen2VLImageProcessor.__call__`] for details. [`Qwen2VLProcessor`] uses
            [`Qwen2VLImageProcessor`] for processing videos.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
"""


class CustomLMHead(nn.Module):
    def __init__(self, original_lm_head: nn.Linear):
        super(CustomLMHead, self).__init__()
        in_features = original_lm_head.in_features
        out_features = original_lm_head.out_features
        bias = original_lm_head.bias is not None

        assert out_features >= 2, "vocab_size must be at least 2 to allow training of last two rows."

        # 冻结的权重部分：前 N-2 行
        self.fixed_weight = nn.Parameter(
            original_lm_head.weight.data[:151657, :].clone(), 
            requires_grad=False
        )

        # 可训练的权重部分：最后两行
        self.trainable_weight = nn.Parameter(
            original_lm_head.weight.data[151657:, :].clone(), 
            requires_grad=True
        )

        if bias:
            # 冻结的偏置部分：前 N-2 项
            self.fixed_bias = nn.Parameter(
                original_lm_head.bias.data[:151657].clone(), 
                requires_grad=False
            )
            # 可训练的偏置部分：最后两项
            self.trainable_bias = nn.Parameter(
                original_lm_head.bias.data[151657:].clone(), 
                requires_grad=True
            )
        else:
            self.fixed_bias = None
            self.trainable_bias = None

    def forward(self, x):
        # 拼接固定和可训练的权重
        weight = torch.cat([self.fixed_weight, self.trainable_weight], dim=0)
        if self.trainable_bias is not None:
            bias = torch.cat([self.fixed_bias, self.trainable_bias], dim=0)
        else:
            bias = None
        return F.linear(x, weight, bias)


class CustomEmbedding(nn.Module):
    def __init__(self, original_embedding: nn.Embedding, num_fixed: int = 2):
        super(CustomEmbedding, self).__init__()
        in_features = original_embedding.embedding_dim
        num_embeddings = original_embedding.num_embeddings
        padding_idx = original_embedding.padding_idx
        max_norm = original_embedding.max_norm
        norm_type = original_embedding.norm_type
        scale_grad_by_freq = original_embedding.scale_grad_by_freq
        sparse = original_embedding.sparse

        assert num_embeddings >= num_fixed, "Number of trainable embeddings must be less than total embeddings."

        # 冻结的嵌入部分：前 N - num_trainable 行
        self.fixed_weight = nn.Parameter(
            original_embedding.weight.data[:num_fixed, :].clone(),
            requires_grad=False
        )

        # 可训练的嵌入部分：最后 num_trainable 行
        self.trainable_weight = nn.Parameter(
            original_embedding.weight.data[num_fixed:, :].clone(),
            requires_grad=True
        )

        self.num_fixed = num_fixed
        self.total_embeddings = num_embeddings

        # 其他参数
        self.embedding_dim = in_features
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

    def forward(self, input):
        # 拼接固定和可训练的嵌入权重
        weight = torch.cat([self.fixed_weight, self.trainable_weight], dim=0)
        return F.embedding(
            input,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse
        )

    def extra_repr(self):
        return 'num_embeddings={}, embedding_dim={}, padding_idx={}, trainable_embeddings={}'.format(
            self.total_embeddings, self.embedding_dim, self.padding_idx, self.fixed_weight
        )

class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation.
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return torch.stack(output, 0)

    @staticmethod
    def backward(ctx, grads):
        input, = ctx.saved_tensors
        dist.all_reduce(grads)
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class CrossAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        assert all_head_dim == dim

        self.q = nn.Linear(dim, all_head_dim, bias=True)
        self.k = nn.Linear(dim, all_head_dim, bias=True)
        self.v = nn.Linear(dim, all_head_dim, bias=True)

        # if qkv_bias:
        #     self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
        #     self.k_bias = nn.Parameter(torch.zeros(all_head_dim))
        #     self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        # else:
        #     self.q_bias = None
        #     self.k_bias = None
        #     self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, k=None, v=None):
        B, N, C = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]

        # q_bias, k_bias, v_bias = None, None, None
        # if self.q_bias is not None:
        #     q_bias = self.q_bias
        #     k_bias = self.k_bias
        #     v_bias = self.v_bias

        # q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = self.q(x)
        q = q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)  # (B, N_head, N_q, dim)

        # if self.q.weight.isnan().any().item(): print("q weight contains NaN ")
        # print("q weight min/max/std:", 
        #     self.q.weight.min().item(), 
        #     self.q.weight.max().item(), 
        #     self.q.weight.std().item())
        # if self.k.weight.isnan().any().item(): print("k weight contains NaN ")
        # print("k weight contains NaN? ", self.k.weight.isnan().any().item())
        # print("k weight min/max/std:", 
        #     self.k.weight.min().item(), 
        #     self.k.weight.max().item(), 
        #     self.k.weight.std().item())
        # if self.v.weight.isnan().any().item(): print("v weight contains NaN ")
        # print("v weight contains NaN? ", self.v.weight.isnan().any().item())
        # print("v weight min/max/std:", 
        #     self.v.weight.min().item(), 
        #     self.v.weight.max().item(), 
        #     self.v.weight.std().item())

        

        # k = F.linear(input=k, weight=self.k.weight, bias=k_bias)
        k = self.k(k)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        # v = F.linear(input=v, weight=self.v.weight, bias=v_bias)
        v = self.v(v)
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        q = q * self.scale

        # ✅ 插入调试代码
        if torch.isnan(q).any(): print("NaN in q")
        if torch.isnan(k).any(): print("NaN in k")
        # import pdb
        # pdb.set_trace()

        attn = (q @ k.transpose(-2, -1))  # [B, heads, N_q, N_k]
        # if torch.isnan(attn).any(): print("NaN in attn scores before softmax")

        attn = attn.softmax(dim=-1)
        # if torch.isnan(attn).any(): print("NaN in attn after softmax")

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        # if torch.isnan(x).any(): print("NaN in output after attention")

        x = self.proj(x)
        x = self.proj_drop(x)
        # if torch.isnan(x).any(): print("NaN in final output")


        # attn = (q @ k.transpose(-2, -1))  # (B, N_head, N_q, N_k)

        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        # x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        # x = self.proj(x)
        # x = self.proj_drop(x)
        # pdb.set_trace()

        return x




class StableCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_head_dim=None, qkv_bias=True, qk_scale=None, out_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads if attn_head_dim is None else attn_head_dim
        all_head_dim = head_dim * num_heads

        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, all_head_dim, bias=qkv_bias)
        self.k = nn.Linear(dim, all_head_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, all_head_dim, bias=qkv_bias)

        self.proj = nn.Linear(all_head_dim, out_dim or dim)

    def forward(self, q, k, v):
        B, N_k, C = k.shape
        N_q = q.shape[1]

        q = self.q(q).view(B, N_q, self.num_heads, -1).transpose(1, 2)  # [B, heads, N_q, d]
        k = self.k(k).view(B, N_k, self.num_heads, -1).transpose(1, 2)
        v = self.v(v).view(B, N_k, self.num_heads, -1).transpose(1, 2)

        q = q * self.scale
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # [B, heads, N_q, N_k]
        attn_scores = attn_scores.clamp(min=-50, max=50)    # 防止 softmax 溢出
        attn = F.softmax(attn_scores, dim=-1)

        out = torch.matmul(attn, v)  # [B, heads, N_q, d]
        out = out.transpose(1, 2).reshape(B, N_q, -1)
        return self.proj(out)

class StableAttentionPoolingBlock(nn.Module):
    def __init__(self, dim, num_heads=8, attn_head_dim=None, out_dim=None):
        super().__init__()
        self.cross_attn = StableCrossAttention(
            dim=dim,
            num_heads=num_heads,
            attn_head_dim=attn_head_dim,
            qk_scale=1.0 / math.sqrt(attn_head_dim or (dim // num_heads)),
            qkv_bias=True,
            out_dim=out_dim
        )

    def forward(self, x):
        # x: [B, N, D]
        q = x.mean(dim=1, keepdim=True)  # 不加 LayerNorm，避免 std=0
        # 可选：扰动 q 防止恒定值（若你发现极端情况）
        q = q + torch.randn_like(q) * 1e-6
        x_out = self.cross_attn(q, k=x, v=x)
        return x_out.squeeze(1)  # [B, D]


class AttentiveBlock(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, attn_head_dim=None, out_dim=None):
        super().__init__()

        self.norm1_q = nn.LayerNorm(dim, eps=1e-5)
        self.norm1_k = nn.LayerNorm(dim, eps=1e-5)
        self.norm1_v = nn.LayerNorm(dim, eps=1e-5)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop, attn_head_dim=attn_head_dim, out_dim=out_dim)

        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = DropPath(0.1)

    def forward(self, x_q, x_kv, pos_q, pos_k, bool_masked_pos, rel_pos_bias=None):
        # if torch.isnan(x_q).any(): print("NaN in x_q")
        # print(f"pos_q:{pos_q}, pos_k:{pos_k}")

        # mean = x_q.mean(-1, keepdim=True)            # [B,1,1]
        # var  = (x_q - mean).pow(2).mean(-1, keepdim=True)  # [B,1,1]
        # print("LayerNorm input var:", var.flatten()[:5])

        # with torch.no_grad():
        #     print("LN weight NaN? ", self.norm1_q.weight.isnan().any().item())
        #     print("LN bias   NaN? ", self.norm1_q.bias.isnan().any().item())
        #     print("LN weight NaN? ", self.norm1_k.weight.isnan().any().item())
        #     print("LN bias   NaN? ", self.norm1_k.bias.isnan().any().item())
        #     print("LN weight NaN? ", self.norm1_v.weight.isnan().any().item())
        #     print("LN bias   NaN? ", self.norm1_v.bias.isnan().any().item())
        #     print("LN weight[:5]:", self.norm1_q.weight.flatten()[:5])
        #     print("LN bias[:5]:",   self.norm1_q.bias.flatten()[:5])

        # if torch.isnan(x_q).any(): print("NaN in x_q")
        x_q = self.norm1_q(x_q + pos_q)
        # if torch.isnan(x_q).any(): print("NaN in x_q_processed")
        # if torch.isnan(x_kv).any(): print("NaN in x_kv")
        x_k = self.norm1_k(x_kv + pos_k)
        # if torch.isnan(x_k).any(): print("NaN in x_k_processed")
        x_v = self.norm1_v(x_kv + pos_k)
        # if torch.isnan(x_v).any(): print("NaN in x_v_processed")
        x = self.cross_attn(x_q, k=x_k, v=x_v)

        # x = self.drop_path(x) + x_q

        # if torch.isnan(x).any(): print("NaN in x_processed")

        return x



class AttentionPoolingBlock(AttentiveBlock):

    def forward(self, x):

        if torch.isnan(x).any(): print("NaN in x----AttentionPoolingBlock")
        x_q = x.mean(1, keepdim=True)
        if torch.isnan(x_q).any(): print("NaN in x_q-----AttentionPoolingBlock")
        # x_q = x_q + torch.randn_like(x_q) * 1e-6

        # if torch.isnan(x_q).any(): print("NaN in x_q-111----AttentionPoolingBlock")

        x_kv, pos_q, pos_k = x, 0, 0
        x = super().forward(x_q, x_kv, pos_q, pos_k, bool_masked_pos=None, rel_pos_bias=None)
        x = x.squeeze(1)
        return x


class MLPAdapterHead(nn.Module):
    def __init__(self,
                 in_dim: int = 2048,
                 adapter_dim: int = 128,
                 out_dim: int = 256):
        """
        - in_dim: 输入特征维度（mt5_ar_head 输出维度）
        - adapter_dim: Adapter 瓶颈维度
        - out_dim: 最终对齐到的特征维度（和 BLIP‑2 一致）
        """
        super().__init__()
        # ① Adapter 部分：LayerNorm → down-project → GELU → up-project
        self.norm    = nn.LayerNorm(in_dim, eps=1e-5)
        self.down    = nn.Linear(in_dim, adapter_dim, bias=True)
        self.act     = nn.GELU()
        self.up      = nn.Linear(adapter_dim, in_dim, bias=True)
        # ② 最终投影
        self.proj    = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim]
        h = self.norm(x)
        h = self.up(self.act(self.down(h)))  # Adapter 残差支路
        h = x + h                            # 残差连接
        z = self.proj(h)                     # 最终投影到 out_dim
        return F.normalize(z, dim=-1)        # 归一化对齐



class Qwen2VLForConditionalGeneration(Qwen2VLPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.model = Qwen2VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.rope_deltas = None  # cache rope_deltas here

        self.bilm_hidden_size = 2048

        if hasattr(config, "mm_added_mlp"):
            self.downsampling_linear_layer = nn.Linear(self.config.hidden_size, self.bilm_hidden_size, bias=False)

            from transformers.models.t5.configuration_t5 import T5Config
            bilmConfig = T5Config(**BiT_config)
            bilmConfig.num_query_tokens = 256
            
            bilmConfig.has_query_tokens = True
            bilmConfig.mask_random_number = 0
            bilmConfig.cross_attn_layer_ids = [0, 2, 4, 8, 12, 16, 20]



            self.bilm = GenBiTransformer(bilmConfig)

            self.logit_scale = nn.Parameter(torch.ones(1,dtype=torch.float32) * np.log(1 / 0.07))  # trainable 

            init_std = 0.02

            self.image_queries = nn.Parameter(
                torch.randn(1, 128, 2048) * init_std
            )
            self.text_queries = nn.Parameter(
                torch.randn(1, 96, 2048) * init_std
            )

            self.image_proj = AttentionPoolingBlock(  # trainable
            dim=2048, num_heads=16, qkv_bias=True, qk_scale=None,
            drop=0., attn_drop=0., norm_layer=partial(nn.LayerNorm, eps=1e-5), out_dim=1024)

            self.image_proj_1 = AttentionPoolingBlock(  # trainable
            dim=2048, num_heads=16, qkv_bias=True, qk_scale=None,
            drop=0., attn_drop=0., norm_layer=partial(nn.LayerNorm, eps=1e-5), out_dim=1024)

            self.text_proj = AttentionPoolingBlock(  # trainable
            dim=2048, num_heads=16, qkv_bias=True, qk_scale=None,
            drop=0., attn_drop=0., norm_layer=partial(nn.LayerNorm, eps=1e-5), out_dim=1024)
            self.text_proj_1 = AttentionPoolingBlock(  # trainable
            dim=2048, num_heads=16, qkv_bias=True, qk_scale=None,
            drop=0., attn_drop=0., norm_layer=partial(nn.LayerNorm, eps=1e-5), out_dim=1024)
            
            self.fusion_alpha_image = nn.Parameter(torch.zeros(1,))
            self.fusion_alpha_text = nn.Parameter(torch.zeros(1,))

        self.post_init()

    @torch.no_grad()
    def dequeue_and_enqueue(self, new_embeds: torch.Tensor, new_img_ids: torch.Tensor):
        """
        new_embeds:  [B, D]  当前 batch 的 text_itc 或 image_itc
        new_img_ids: [B]     对应的 全局 图像 ID
        """
        B = new_embeds.size(0)
        ptr = self.bank_ptr
        end = (ptr + B) % self.memory_size

        if end > ptr:
            self.memory_bank[ptr:end] = new_embeds
            self.bank_img_ids[ptr:end] = new_img_ids
        else:
            split = self.memory_size - ptr
            # wrap around
            self.memory_bank[ptr:] = new_embeds[:split]
            self.memory_bank[:end]  = new_embeds[split:]
            self.bank_img_ids[ptr:] = new_img_ids[:split]
            self.bank_img_ids[:end] = new_img_ids[split:]

        self.bank_ptr = end
    
    def initialize_embedding_modules(self, model_args):
        self.qwen2vl_embedding = CustomEmbedding(self.model.embed_tokens, 151657)

        self.lm_head = CustomLMHead(self.lm_head)

    def load_embedding_modules(self, trainable_embedding_path, lm_head_path):
        device = self.lm_head.weight.device
    
        trainable_embedding_weight = torch.load(trainable_embedding_path, map_location=device, weights_only=True)
        lm_head_weight = torch.load(lm_head_path, map_location=device, weights_only=True)

        trainable_embedding_weight = trainable_embedding_weight["trainable_weight"].to(device)
        trainable_lm_head_weight = lm_head_weight["trainable_weight"].to(device)

        self.model.embed_tokens.weight.data[151657:] = trainable_embedding_weight
        self.lm_head.weight.data[151657:] = trainable_lm_head_weight




    def initialize_dit_modules(self, pretrain_dit):
        
        import sys

        from dit.hydit.config import get_args_without
        from dit.hydit.inference import End2End

        def build_captioner_embedding():
            args = get_args_without()
            if not (pretrain_dit.endswith("t2i") or pretrain_dit.endswith("t2i/")):
                hunyuan_dit_path = pretrain_dit.split("t2i")[0]
            else:
                hunyuan_dit_path = pretrain_dit
            
            print(hunyuan_dit_path)
            captioner_embedding = End2End(args, hunyuan_dit_path)

            enhancer = None

            return args, captioner_embedding, enhancer

        args, gen, enhancer = build_captioner_embedding()

        self.dit_model = gen.to(self.device)
        self.dit_args = args

        if enhancer:
            logger.info("Prompt Enhancement...")
            success, enhanced_prompt = enhancer(args.prompt)
            if not success:
                logger.info("Sorry, the prompt is not compliant, refuse to draw.")
                exit()
            logger.info(f"Enhanced prompt: {enhanced_prompt}")
        else:
            enhanced_prompt = None


    def initialize_embedder_dit_modules(self, pretrain_dit, is_image_generation_or_retrieval, is_vl_training = True):
        
        if not (pretrain_dit.endswith("t2i") or pretrain_dit.endswith("t2i/")):
            t5_text_encoder_path = os.path.join(pretrain_dit, 't2i/mt5')
        else:
            t5_text_encoder_path = os.path.join(pretrain_dit, 'mt5')

        assert os.path.exists(t5_text_encoder_path), f"Path {t5_text_encoder_path} does not exist!"

        from dit.hydit.modules.text_encoder import MT5Embedder
            
        embedder_dit = MT5Embedder(t5_text_encoder_path, torch_dtype=torch.float16, max_length=256, is_vl_training = is_vl_training).to(self.device)

        if is_image_generation_or_retrieval:
            self.embedder_dit = embedder_dit
            self.dit_embedding = self.embedder_dit.model.encoder.embed_tokens
            self.dit_tokenizer = self.embedder_dit.tokenizer



    def initialize_ar_modules(self, model_args):
        
        pretrain_ar_modules = None 
        if "pretrain_ar_modules" in model_args:
            pretrain_ar_modules = model_args.pretrain_ar_modules
        # pdb.set_trace()
        if getattr(self, 'downsampling_linear_layer', None) is None:
            self.downsampling_linear_layer = nn.Linear(self.config.hidden_size, self.bilm_hidden_size, bias=False)
        else:
            for p in self.downsampling_linear_layer.parameters():
                p.requires_grad = True


        self.config.mm_added_mlp = True
        if getattr(self, 'mt5_ar_head', None) is None:

            from transformers.models.t5.configuration_t5 import T5Config
            bilmConfig = T5Config(**BiT_config)
            bilmConfig.num_query_tokens = 256
            bilmConfig.has_query_tokens = False
            bilmConfig.mask_random_number = 0
            bilmConfig.cross_attn_layer_ids = [0, 2, 4, 8, 12, 16, 20]

            self.bilm = GenBiTransformer(bilmConfig)
            # pdb.set_trace()
            t5_weights = self.embedder_t5.model.encoder.state_dict()


            bilm_state_dict = self.bilm.state_dict()

            for name, param in t5_weights.items():
                if name in bilm_state_dict:
                    bilm_state_dict[name].copy_(param)

            self.bilm.load_state_dict(bilm_state_dict)


        else:
            for p in self.bilm.parameters():
                p.requires_grad = True

        if "has_learned_query" in model_args and model_args['has_learned_query']:
            self.config.has_learned_query = True


        if pretrain_ar_modules is not None:
            pretrain_ar_modules_weights = torch.load(pretrain_ar_modules, weights_only=True,map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.bilm.load_state_dict(get_w(pretrain_ar_modules_weights, 'bilm'))

            
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embeddin for text part.
            Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas


    @add_start_docstrings_to_model_forward(QWEN2_VL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Qwen2VLCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        is_generate: Optional[bool] = False,
        images: Optional[List] = None,
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        >>> model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        if inputs_embeds is None:                   
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)
        
        # pdb.set_trace()
        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if not (hasattr(self, "hy_ar_head") or hasattr(self, "bilm")) and position_ids is None and input_ids is not None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        # pdb.set_trace()


        
        has_hy_ar_head = hasattr(self, "hy_ar_head")


        if (hasattr(self, "logit_scale") or hasattr(self, "image_queries")) and labels is not None:
            input_ids = labels

            image_token_id = 151655
            split_token_id = 77091
            pad_token_id = 151643
            _token_id = -100

            def is_cpu_obj(x) -> bool:
                if isinstance(x, nn.Module):
                    for t in itertools.chain(x.parameters(recurse=True), x.buffers()):
                        if t.device.type != "cpu":
                            return False
                    return True  # 无参数/缓冲也视为在 CPU
                elif isinstance(x, (torch.Tensor, nn.Parameter)):
                    return x.device.type == "cpu"
                else:
                    raise TypeError(f"Expected nn.Module or Tensor/Parameter, got {type(x)}")

            if not is_cpu_obj(self.lm_head):
                self.lm_head.to("cpu")
            if not is_cpu_obj(self.bilm.query_tokens):
                self.bilm.query_tokens.to("cpu")

            with torch.no_grad():
                
                seq_attention_mask = (input_ids != pad_token_id)
                seq_num_non_pads = seq_attention_mask.sum(dim=1)

                indices = torch.where(input_ids == split_token_id)[1]   # 取出assistant之后的，即取出文本部分
                

                # pdb.set_trace()

                # attention_mask = torch.zeros(input_ids.shape[0], input_ids.shape[1], dtype=torch.bool).to(input_ids.device)

                # for batch_idx in range(input_ids.shape[0]):
                #     attention_mask[batch_idx][:indices[batch_idx].item()+2] = 1

                model_outputs = self.model(
                    input_ids=None,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    cache_position=cache_position,
                )

                model_hidden_states = model_outputs[0]

                image_mask = (input_ids == image_token_id)
                image_first_index = (image_mask[0] == 1).nonzero(as_tuple=True)[0][0].item()

                image_part = model_hidden_states[:, image_first_index:, :]
                image_mask = image_mask[:, image_first_index:, ]

                batch_max_image_tokens = image_mask.sum(dim=1).max().item()

                image_mask = image_mask[:,:batch_max_image_tokens]
                image_part = image_part[:,:batch_max_image_tokens,:]

                
                image_mask_expanded = image_mask.unsqueeze(-1).to(image_part.device)
                masked_image_part = image_part * (image_mask_expanded.float())

                retrieval_texts = []
                t5_retrieval_texts = []
                image_ids = []
                for index_i in range(indices.shape[0]):

                    assistant_prompt_ids = labels[index_i][indices[index_i]+2:]
                    assistant_prompt = self.tokenizer.batch_decode(assistant_prompt_ids.unsqueeze(0), skip_special_tokens=True)[0]
                    assistant_prompt = assistant_prompt.split("</s>")
                    image_ids.append(int(assistant_prompt[0][3:]))
                    cur_caption = assistant_prompt[1]

                    # input_text = "</s>" + cur_caption
                    input_text = cur_caption

                    input_text = f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{cur_caption}<|im_end|>\n<|im_start|>assistant\n'

                    retrieval_texts.append(input_text)
                    t5_retrieval_texts.append(cur_caption)


                # pdb.set_trace()

                # label_inputs = self.blip2_processor(images=images, text=t5_retrieval_texts, return_tensors="pt", padding=True).to(self.device, torch.float16)
                # itc_out = self.blip2_model(**label_inputs, use_image_text_matching_head=False)
                # image_itc_label = itc_out.image_embeds
                # text_itc_label = itc_out.text_embeds

                # t5_text_inputs = self.mt5_tokenizer(
                #     t5_retrieval_texts,
                #     padding="max_length",
                #     max_length=256,
                #     truncation=True,
                #     return_attention_mask=True,
                #     return_tensors="pt",
                # )
                # t5_attention_mask = t5_text_inputs.attention_mask.to(masked_image_part.device)
                # t5_input_ids = t5_text_inputs.input_ids.to(masked_image_part.device)


                # t5_embedding = self.embedder_t5.model.encoder.embed_tokens(t5_input_ids)
                

                # qwen2vl 获取文本部分的嵌入
                self.tokenizer.padding_side = "right"

                cur_inputs = self.tokenizer(retrieval_texts,return_tensors="pt",padding=True)
                pure_cur_inputs = self.tokenizer(t5_retrieval_texts,return_tensors="pt",padding=True)
                self.tokenizer.padding_side = "left"
                cur_inputs = {k: v.to(image_mask.device) for k, v in cur_inputs.items()}
                pure_cur_inputs = {k: v.to(image_mask.device) for k, v in pure_cur_inputs.items()}

                # pdb.set_trace()

                batch_max_text_tokens = pure_cur_inputs['attention_mask'].sum(dim=1).max().item()

                text_model_outputs = self.model(
                                **cur_inputs,
                                output_hidden_states=True,
                                return_dict=True,
                            )
                            
                text_hidden_states = text_model_outputs[0]
                text_part = text_hidden_states[:,14:,:]

                text_mask = pure_cur_inputs['attention_mask'][:,:batch_max_text_tokens]
                text_part = text_part[:,:batch_max_text_tokens,:]

                
                text_mask_expanded = text_mask.unsqueeze(-1).to(text_part.device)
                masked_text_part = text_part * (text_mask_expanded.float())

                # text_mask = cur_inputs["attention_mask"][:, 14:, ]
                # text_mask_expanded = text_mask.unsqueeze(-1)
                # masked_text_part = text_part * text_mask_expanded.float()



        elif labels is not None and -100 not in labels and (has_hy_ar_head or hasattr(self, "bilm")):
            
            input_ids = labels

            image_token_id = 151655
            split_token_id = 77091
            pad_token_id = 151643
            _token_id = -100

            image_mask = (input_ids == image_token_id) 
            seq_attention_mask = (input_ids != pad_token_id)    # .to(input_ids.device)
            seq_num_non_pads = seq_attention_mask.sum(dim=1)  # 形状为 (batch_size,)


            indices = torch.where(input_ids == split_token_id)[1] 

            attention_mask = torch.zeros(input_ids.shape[0], input_ids.shape[1], dtype=torch.bool).to(input_ids.device)

            edit_condition_ids = []
            for batch_idx in range(input_ids.shape[0]):
                # 输入部分可见的注意力掩码
                attention_mask[batch_idx][:indices[batch_idx].item()+2] = 1
                # 从label中提取修改指令
                edit_condition_ids.append(input_ids[batch_idx][indices[batch_idx].item()+2:seq_num_non_pads[batch_idx].item()-1])

            # 输入部分的token长度
            num_non_pads = attention_mask.sum(dim=1)  # 形状为 (batch_size,)

            model_outputs = self.model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

            model_hidden_states = model_outputs[0]

            image_embeddings_list = []
            image_embeddings_masks = []
            
            # 获取每个样本的图片嵌入
            for batch_idx in range(model_hidden_states.shape[0]):
                # 获取当前样本的图片嵌入，image_mask[batch_idx] 为该样本的掩码
                mask = image_mask[batch_idx]
                image_embed = model_hidden_states[batch_idx, mask]  # 根据掩码提取图片嵌入

                # 将提取出的图片嵌入放入列表
                image_embeddings_list.append(image_embed)
            
            # 计算批次中最大的图片嵌入长度
            max_image_embed_length = max([img_embed.size(0) for img_embed in image_embeddings_list])

            # 创建一个新的列表用于填充
            padded_image_embeddings = []

            actual_lengths = torch.tensor([img_embed.size(0) for img_embed in image_embeddings_list])

            retrieval_index = [False for _ in range(actual_lengths.size(0))]
            image_ids = [_i-256 for _i in range(actual_lengths.size(0))]

            # 填充每个样本的图片嵌入到最大长度
            for img_embed in image_embeddings_list:
                # 计算当前样本需要填充的数量
                padding_length = max_image_embed_length - img_embed.size(0)

                # 使用零向量填充
                padding = torch.zeros(padding_length, img_embed.size(1)).to(img_embed.device)  # 填充零向量
                padded_img_embed = torch.cat([img_embed, padding], dim=0)  # 拼接原图片嵌入与填充

                padded_image_embeddings.append(padded_img_embed)
            
            # 将所有样本的填充后图片嵌入拼接成一个新的张量
            padded_image_embeddings_tensor = torch.stack(padded_image_embeddings).to(model_hidden_states.device)

            image_masks = torch.zeros(padded_image_embeddings_tensor.shape[0], padded_image_embeddings_tensor.shape[1], dtype=torch.bool).to(model_hidden_states.device)

            for batch_idx, length in enumerate(actual_lengths):
                image_masks[batch_idx, :length] = 1  # 设置非填充部分为 True


            batch_size, max_length = input_ids.size()
            max_new_token = 512
            token_eos = 151645

            batch_qwen2vl_hidden_states = []
            batch_qwen2vl_attention_mask = []
            batch_retrieval_attention_mask = []
            batch_t5_labels = []
            batch_t5_embeddings = []
            batch_qwen2vl_ids = [] 
            batch_t5_attention_mask = []

            batch_retrieval_hidden_states = [] 
            batch_retrieval_last_hidden_states = []

            texts = []
            edit_texts = []

            def pad_sequence(sequences, padding_side='right', padding_value=0):
                """
                Pad a list of sequences to the same length.
                sequences: list of tensors in [seq_len, *] shape
                """
                assert padding_side in ['right', 'left']
                max_size = sequences[0].size()
                trailing_dims = max_size[1:]
                max_len = max(len(seq) for seq in sequences)
                batch_size = len(sequences)
                output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
                for i, seq in enumerate(sequences):
                    length = seq.size(0)
                    if padding_side == 'right':
                        output.data[i, :length] = seq
                    else:
                        output.data[i, -length:] = seq
                return output

            self.model.eval()
            # self.clip_text_encoder.eval()
            self.embedder_t5.model.eval()

            input_ids = input_ids.to(self.model.device)
            for i in range(batch_size):
                # input_ids = input_ids.to
                n = num_non_pads[i]
                if n > 0:
                    new_input_ids = input_ids[i, :n].unsqueeze(0)
                    
                    cur_attention_mask = torch.full_like(new_input_ids, 1).to(torch.long)

                    qwen_hidden_states = torch.empty((0,3584)) if actual_lengths[i]==0 else None 


                    max_new_token = 512  + new_input_ids.shape[1]
                    kvcache = DynamicCache()
                    output_ids = new_input_ids.clone().detach().to(self.model.device)

                    with torch.no_grad():
                        
                        inputs_embeds = self.model.embed_tokens(new_input_ids)

                        past_seen_tokens = kvcache.get_seq_length() if kvcache is not None else 0
                        cache_position = torch.arange(
                            past_seen_tokens, past_seen_tokens + new_input_ids.shape[1], device=new_input_ids.device
                        )
                        position_ids, _ = self.get_rope_index(
                            new_input_ids, None, None, cur_attention_mask
                        )

                        outputs = self.model(
                            inputs_embeds=inputs_embeds,
                            position_ids=position_ids,
                            attention_mask = cur_attention_mask,
                            use_cache = True,
                            output_hidden_states = True, 
                            past_key_values = kvcache,
                            output_attentions=False,
                            return_dict=True,
                            # cache_position=cache_position,
                        )

                        kvcache = outputs.past_key_values
                        hidden_states = outputs.last_hidden_state

                        logits = self.lm_head(hidden_states)

                        out = logits[...,-1, :]
                        out_token = torch.argmax(out, dim=-1, keepdim=True).to(self.model.device)

                        output_ids=torch.cat([output_ids,out_token],dim=-1)
                        if actual_lengths[i]==0:
                            qwen_hidden_state = hidden_states[...,-1, :]
                            qwen_hidden_states=torch.cat([qwen_hidden_states, qwen_hidden_state.clone().detach().cpu()],dim=0)

                        inputs_embeds = self.model.embed_tokens(out_token)

                        while True:
                            
                            cache_position = cache_position[-1]+1
                            cache_position = cache_position.unsqueeze(0)

                            outputs = self.model(
                            inputs_embeds=inputs_embeds,
                            # position_ids=position_ids,
                            attention_mask = torch.full_like(output_ids, 1).to(torch.long),
                                use_cache = True,
                                output_hidden_states = False, 
                                past_key_values = kvcache,
                                output_attentions=False,
                                return_dict=True,
                                cache_position=cache_position,
                            )

                            kvcache = outputs.past_key_values
                            hidden_states = outputs.last_hidden_state

                            logits = self.lm_head(hidden_states)

                            out = logits[...,-1, :]
                            out_token = torch.argmax(out, dim=-1, keepdim=True).to(self.model.device)

                            output_ids=torch.cat([output_ids,out_token],dim=-1)

                            if actual_lengths[i]==0:
                                qwen_hidden_state = hidden_states[...,-1, :]
                                qwen_hidden_states=torch.cat([qwen_hidden_states, qwen_hidden_state.clone().detach().cpu()],dim=0)

                            inputs_embeds =  self.model.embed_tokens(out_token)

                            if output_ids.shape[1]>max_new_token:
                                break

                            if out_token == token_eos:
                                break
                        
                        if actual_lengths[i]==0:
                            # pdb.set_trace()
                            generate_ids = output_ids[:, new_input_ids.size(1)+1:-2]
                            generate_text = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
                            generate_text = generate_text.replace("<img> ", "").replace(" </img>", "")
                            qwen_hidden_states = qwen_hidden_states[1:-2,:].to(torch.bfloat16).to(new_input_ids.device)
                            batch_qwen2vl_hidden_states.append(qwen_hidden_states)
                            batch_qwen2vl_attention_mask.append(torch.ones(qwen_hidden_states.size(0), ).to(new_input_ids.device))

                            use_label_text = generate_text

                            batch_retrieval_hidden_states.append(torch.zeros((1, 3584)).to(new_input_ids.device))
                            batch_retrieval_attention_mask.append(torch.zeros(1, ).to(new_input_ids.device))
                            batch_retrieval_last_hidden_states.append(torch.zeros((1, 3584)).to(new_input_ids.device))

                        else:
                            # pdb.set_trace()
                            assistant_prompt_ids = labels[i][new_input_ids.size(1):]
                            assistant_prompt = self.tokenizer.batch_decode(assistant_prompt_ids.unsqueeze(0), skip_special_tokens=True)[0]
                            
                            if assistant_prompt.startswith("<s>"):
                                # pdb.set_trace()
                                assistant_prompt = assistant_prompt.split("</s>")
                                image_index = int(assistant_prompt[0].split("<s>")[1])
                                image_ids[i] = image_index
                                cur_caption = assistant_prompt[1]
                                assistant_prompt = "</s>" + assistant_prompt[1]
                                use_label_text = assistant_prompt
                                generate_text = assistant_prompt
                                retrieval_index[i] = True

                                input_text = f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{cur_caption}<|im_end|>\n<|im_start|>assistant\n'
                                cur_inputs = self.tokenizer(input_text,return_tensors="pt",)
                                cur_inputs = {k: v.cuda() for k, v in cur_inputs.items()}

                                model_outputs = self.model(
                                            **cur_inputs,
                                            output_hidden_states=True,
                                            return_dict=True,
                                        )
                                model_hidden_states = model_outputs[0]
                                model_hidden_states = model_hidden_states[:,14:-5,:]
                                batch_retrieval_hidden_states.append(model_hidden_states.squeeze(0).to(new_input_ids.device))
                                last_model_hidden_states = model_hidden_states[:,-5:-4,:]
                                batch_retrieval_last_hidden_states.append(last_model_hidden_states.squeeze(0).to(new_input_ids.device))
                                batch_retrieval_attention_mask.append(torch.ones(model_hidden_states.size(1), ).to(new_input_ids.device))

                                # pdb.set_trace()
                            else:
                                batch_retrieval_hidden_states.append(torch.zeros((1, 3584)).to(new_input_ids.device))
                                batch_retrieval_last_hidden_states.append(torch.zeros((1, 3584)).to(new_input_ids.device))
                                batch_retrieval_attention_mask.append(torch.zeros(1, ).to(new_input_ids.device))


                                generate_ids = output_ids[:, new_input_ids.size(1):]

                                edit_condition_text = self.tokenizer.batch_decode(edit_condition_ids[i].unsqueeze(0), skip_special_tokens=True)[0]
                                edit_texts.append(edit_condition_text)

                                generate_text = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]

                                
                                # pdb.set_trace()

                                if generate_text.startswith("I'm unable to see the image"):
                                    generate_text = edit_condition_text

                                else:
                                    edit_condition_text = generate_text

                                use_label_text = edit_condition_text
                                
                            batch_qwen2vl_hidden_states.append(padded_image_embeddings_tensor[i])

                            batch_qwen2vl_attention_mask.append(image_masks[i])

                        texts.append(generate_text)

                        # pdb.set_trace()

                        t5_text_inputs = self.embedder_t5.tokenizer(
                            use_label_text,
                            padding="max_length",
                            max_length=256,
                            truncation=True,
                            return_attention_mask=True,
                            return_tensors="pt",
                        )
                        t5_attention_mask = t5_text_inputs.attention_mask.to(new_input_ids.device)
                        t5_input_ids = t5_text_inputs.input_ids.to(new_input_ids.device)


                        t5_label_inputs = self.embedder_t5.tokenizer(
                            generate_text,
                            padding="max_length",
                            max_length=256,
                            truncation=True,
                            return_attention_mask=True,
                            return_tensors="pt",
                        )
                        t5_label_attention_mask = t5_label_inputs.attention_mask.to(new_input_ids.device)
                        t5_label_input_ids = t5_label_inputs.input_ids.to(new_input_ids.device)
                        t5_prompt_embeds = self.embedder_t5.model(
                            t5_label_input_ids,
                            attention_mask=t5_label_attention_mask,
                        )
                        t5_embedding = self.embedder_t5.model.encoder.embed_tokens(t5_input_ids)
                        batch_t5_embeddings.append(t5_embedding.squeeze(0))
                        # pdb.set_trace()
                        batch_t5_labels.append(t5_prompt_embeds[0].squeeze(0))
                        # batch_qwen2vl_ids.append(generate_ids.squeeze(0))
                        
                        batch_t5_attention_mask.append(t5_attention_mask.squeeze(0))

            qwen2vl_states = pad_sequence(batch_qwen2vl_hidden_states, padding_side='right', padding_value=0.0)
            t5_ar_labels = pad_sequence(batch_t5_labels, padding_side='right', padding_value=0.0)
            t5_attention_masks = pad_sequence(batch_t5_attention_mask, padding_side='right', padding_value=0)
            t5_embeddings = pad_sequence(batch_t5_embeddings, padding_side='right', padding_value=0.0)

            qwen2vl_attention_masks = pad_sequence(batch_qwen2vl_attention_mask, padding_side='right', padding_value=0)
            retrieval_attention_mask = pad_sequence(batch_retrieval_attention_mask, padding_side='right', padding_value=0)
            retrieval_hidden_states = pad_sequence(batch_retrieval_hidden_states, padding_side='right', padding_value=0)
            retrieval_last_hidden_states = pad_sequence(batch_retrieval_last_hidden_states, padding_side='right', padding_value=0)
            # pdb.set_trace()

        elif (has_hy_ar_head or hasattr(self, "bilm")) and labels is not None:
            pdb.set_trace()
            input_ids = labels
            pad_token_id = 151643
            _token_id = -100
            attention_mask = (input_ids != _token_id).to(input_ids.device)
            num_non_pads = attention_mask.sum(dim=1)  # 形状为 (batch_size,)

            batch_size, max_length = input_ids.size()
            max_new_token = 256
            token_eos = 151645

            batch_qwen2vl_hidden_states = []
            batch_t5_labels = []
            batch_t5_embeddings = []
            batch_qwen2vl_ids = []
            batch_clip_labels = []
            batch_t5_attention_mask = []
            batch_clip_attention_mask = []
            batch_qwen2vl_attention_mask = []
            texts = []

            def pad_sequence(sequences, padding_side='right', padding_value=0):
                """
                Pad a list of sequences to the same length.
                sequences: list of tensors in [seq_len, *] shape
                """
                assert padding_side in ['right', 'left']
                max_size = sequences[0].size()
                trailing_dims = max_size[1:]
                max_len = max(len(seq) for seq in sequences)
                batch_size = len(sequences)
                output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
                for i, seq in enumerate(sequences):
                    length = seq.size(0)
                    if padding_side == 'right':
                        output.data[i, :length] = seq
                    else:
                        output.data[i, -length:] = seq
                return output

            self.model.eval()
            self.clip_text_encoder.eval()
            self.embedder_t5.model.eval()

            input_ids = input_ids.to(self.model.device)
            for i in range(batch_size):
                # input_ids = input_ids.to
                n = num_non_pads[i]
                if n > 0:
                    new_input_ids = input_ids[i, :n].unsqueeze(0)
                    
                    cur_attention_mask = torch.full_like(new_input_ids, 1).to(torch.long)
                    qwen_hidden_states = torch.empty((0,3584))
                    max_new_token = 256  + new_input_ids.shape[1]
                    kvcache = DynamicCache()
                    output_ids = new_input_ids.clone().detach().to(self.model.device)

                    with torch.no_grad():
                        # pdb.set_trace()
                        # generated_ids = self.generate(
                        #     input_ids=new_input_ids,
                        #     attention_mask=cur_attention_mask,
                        #     max_length=max_new_token,
                        #     return_dict_in_generate=True,
                        #     output_hidden_states = True
                        # )
                        # pdb.set_trace()
                        inputs_embeds = self.model.embed_tokens(new_input_ids)

                        past_seen_tokens = kvcache.get_seq_length() if kvcache is not None else 0
                        cache_position = torch.arange(
                            past_seen_tokens, past_seen_tokens + new_input_ids.shape[1], device=new_input_ids.device
                        )
                        position_ids, _ = self.get_rope_index(
                            new_input_ids, None, None, cur_attention_mask
                        )

                        outputs = self.model(
                            inputs_embeds=inputs_embeds,
                            position_ids=position_ids,
                            attention_mask = cur_attention_mask,
                            use_cache = True,
                            output_hidden_states = False, 
                            past_key_values = kvcache,
                            output_attentions=False,
                            return_dict=True,
                            # cache_position=cache_position,
                        )

                        kvcache = outputs.past_key_values
                        hidden_states = outputs.last_hidden_state

                        logits = self.lm_head(hidden_states)

                        out = logits[...,-1, :]
                        out_token = torch.argmax(out, dim=-1, keepdim=True).to(self.model.device)

                        output_ids=torch.cat([output_ids,out_token],dim=-1)

                        qwen_hidden_state = hidden_states[...,-1, :]
                        qwen_hidden_states=torch.cat([qwen_hidden_states, qwen_hidden_state.clone().detach().cpu()],dim=0)
                        inputs_embeds =  self.model.embed_tokens(out_token)

                        while True:
                            # position_ids = torch.arange(output_ids.size(1), dtype=torch.long).unsqueeze(0).to(qwen_hidden_states.device)
                            cache_position = cache_position[-1]+1
                            cache_position = cache_position.unsqueeze(0)

                            outputs = self.model(
                            inputs_embeds=inputs_embeds,
                            # position_ids=position_ids,
                            attention_mask = torch.full_like(output_ids, 1).to(torch.long),
                                use_cache = True,
                                output_hidden_states = False, 
                                past_key_values = kvcache,
                                output_attentions=False,
                                return_dict=True,
                                cache_position=cache_position,
                            )

                            kvcache = outputs.past_key_values
                            hidden_states = outputs.last_hidden_state

                            logits = self.lm_head(hidden_states)

                            out = logits[...,-1, :]
                            out_token = torch.argmax(out, dim=-1, keepdim=True).to(self.model.device)

                            output_ids=torch.cat([output_ids,out_token],dim=-1)

                            qwen_hidden_state = hidden_states[...,-1, :]
                            qwen_hidden_states=torch.cat([qwen_hidden_states, qwen_hidden_state.clone().detach().cpu()],dim=0)
                            inputs_embeds =  self.model.embed_tokens(out_token)

                            if output_ids.shape[1]>max_new_token:
                                break

                            if out_token == token_eos:
                                break

                        # pdb.set_trace()
                        generate_ids = output_ids[:, new_input_ids.size(1)+1:-2]
                        generate_text = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
                        generate_text = generate_text.replace("<img> ", "").replace(" </img>", "")
                        qwen_hidden_states = qwen_hidden_states[1:-2,:].to(torch.bfloat16).to(new_input_ids.device)
                        batch_qwen2vl_hidden_states.append(qwen_hidden_states)
                        # pdb.set_trace()

                        texts.append(generate_text)

                        if has_hy_ar_head:
                            clip_text_inputs = self.clip_tokenizer(
                                generate_text,
                                padding="max_length",
                                max_length=77,
                                truncation=True,
                                return_attention_mask=True,
                                return_tensors="pt",
                            )
                            clip_attention_mask = clip_text_inputs.attention_mask.to(new_input_ids.device)
                            clip_input_ids = clip_text_inputs.input_ids.to(new_input_ids.device)

                            clip_prompt_embeds = self.clip_text_encoder(
                                clip_input_ids,
                                attention_mask=clip_attention_mask,
                            )

                        t5_text_inputs = self.embedder_t5.tokenizer(
                            generate_text,
                            padding="max_length",
                            max_length=256,
                            truncation=True,
                            return_attention_mask=True,
                            return_tensors="pt",
                        )
                        t5_attention_mask = t5_text_inputs.attention_mask.to(new_input_ids.device)
                        t5_input_ids = t5_text_inputs.input_ids.to(new_input_ids.device)

                        t5_prompt_embeds = self.embedder_t5.model(
                            t5_input_ids,
                            attention_mask=t5_attention_mask,
                        )
                        t5_embedding = self.embedder_t5.model.encoder.embed_tokens(t5_input_ids)
                        batch_t5_embeddings.append(t5_embedding.squeeze(0))
                        # pdb.set_trace()
                        batch_t5_labels.append(t5_prompt_embeds[0].squeeze(0))
                        # batch_qwen2vl_ids.append(generate_ids.squeeze(0))
                        
                        batch_t5_attention_mask.append(t5_attention_mask.squeeze(0))

                        if has_hy_ar_head:
                            batch_clip_labels.append(clip_prompt_embeds[0].squeeze(0))
                            batch_clip_attention_mask.append(clip_attention_mask.squeeze(0))

                        batch_qwen2vl_attention_mask.append(torch.ones(qwen_hidden_states.size(0), ).to(new_input_ids.device))

            qwen2vl_states = pad_sequence(batch_qwen2vl_hidden_states, padding_side='right', padding_value=0.0)
            t5_ar_labels = pad_sequence(batch_t5_labels, padding_side='right', padding_value=0.0)
            # qwen2vl_ids = pad_sequence(batch_qwen2vl_ids, padding_side='right', padding_value=self.tokenizer.pad_token_id)
            t5_attention_masks = pad_sequence(batch_t5_attention_mask, padding_side='right', padding_value=0)
            t5_embeddings = pad_sequence(batch_t5_embeddings, padding_side='right', padding_value=0.0)

            if has_hy_ar_head:
                clip_ar_labels = pad_sequence(batch_clip_labels, padding_side='right', padding_value=0.0)
                clip_attention_masks = pad_sequence(batch_clip_attention_mask, padding_side='right', padding_value=0)

                hy_ar_labels = self.text_state_encoder(clip_ar_labels, clip_attention_masks, t5_ar_labels, t5_attention_masks)
            # else:
            #     t5_ar_labels = self.embedder_t5.mlp_t5(t5_ar_labels)

            qwen2vl_attention_masks = pad_sequence(batch_qwen2vl_attention_mask, padding_side='right', padding_value=0)
            
            # pdb.set_trace()
            # if qwen2vl_ids is not None and (qwen2vl_attention_masks is None or qwen2vl_attention_masks.ndim == 2):
            # # calculate RoPE index once per generation in the pre-fill stage only
            #     if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
            #         position_ids, rope_deltas = self.get_rope_index(
            #             qwen2vl_ids, None, None, qwen2vl_attention_masks
            #         )
            #         self.rope_deltas = rope_deltas
            #     # then use the prev pre-calculated rope-deltas to get the correct position ids
            #     else:
            #         batch_size, seq_length, _ = inputs_embeds.shape
            #         delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
            #         position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            #         position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            #         if cache_position is not None:  # otherwise `deltas` is an int `0`
            #             delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            #         position_ids = position_ids.add(delta)
            #         position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)


        # pdb.set_trace()
        # t5_embedding = t5_ar_labels.detach().cpu().numpy()
        # with h5py.File("/data/xiaot/Methods/VLLMs/EmbAR/data/batch_hidden_states.h5", 'w') as f:
        #     f.create_dataset('t5_ar_labels', data=t5_embedding, compression = 'gzip', compression_opts = 9)

        #     # self.model.train()
        # pdb.set_trace()
        if labels is not None and -100 not in labels and (hasattr(self, "logit_scale") or hasattr(self, "image_queries")):

            # with torch.no_grad():
            image_qwen2vl_states = self.downsampling_linear_layer(masked_image_part)
            text_qwen2vl_states = self.downsampling_linear_layer(masked_text_part)

            image_ids = torch.tensor(image_ids, dtype=torch.long, device=masked_image_part.device)  # [B]

            # text_embeddings = torch.zeros(masked_text_part.size(0), 256, 2048).to(text_qwen2vl_states.device)

            image_queries = self.image_queries.expand(image_qwen2vl_states.size(0), -1, -1)
            text_queries = self.text_queries.expand(text_qwen2vl_states.size(0), -1, -1)
            # pdb.set_trace()
            

            def pad_seq_lastdim(x, target_len):
                """
                x: (B, L, D) or (B, L)
                target_len: int
                """
                pad_len = target_len - x.size(1)
                if pad_len <= 0:
                    return x
                if x.dim() == 3:  # (B, L, D)
                    # pad on L dimension: (last_dim_left, last_dim_right, L_left, L_right)
                    return F.pad(x, (0, 0, 0, pad_len), value=0)
                elif x.dim() == 2:  # (B, L)
                    # pad on L dimension: (L_left, L_right)
                    return F.pad(x, (0, pad_len), value=0)
                else:
                    raise ValueError("Unsupported shape for padding")
                
            # 记录 batch 大小与原始长度
            Bt, Bi = text_queries.size(0), image_queries.size(0)
            Lt_q = text_queries.size(1)
            Li_q = image_queries.size(1)
            Lt_enc = text_qwen2vl_states.size(1)
            Li_enc = image_qwen2vl_states.size(1)

            # 计算各自的 max 长度
            Lq_max   = max(Lt_q,  Li_q)    # decoder query 的 max len
            Lenc_max = max(Lt_enc, Li_enc) # encoder states 的 max len

            # 先分别对 query 和 encoder 侧做 padding
            text_queries_pad   = pad_seq_lastdim(text_queries, Lq_max)
            image_queries_pad  = pad_seq_lastdim(image_queries, Lq_max)

            text_states_pad    = pad_seq_lastdim(text_qwen2vl_states, Lenc_max)
            image_states_pad   = pad_seq_lastdim(image_qwen2vl_states, Lenc_max)

            text_mask_pad      = pad_seq_lastdim(text_mask,   Lenc_max)   # 注意 mask 跟 encoder 长度对齐
            image_mask_pad     = pad_seq_lastdim(image_mask,  Lenc_max)

            # # 拼接
            inputs_embeds = torch.cat([text_queries_pad,  image_queries_pad],  dim=0)  # (Bt+Bi, Lq_max,  D_q)
            enc_states    = torch.cat([text_states_pad,   image_states_pad],   dim=0)  # (Bt+Bi, Lenc_max, D_enc)
            enc_mask      = torch.cat([text_mask_pad,     image_mask_pad],     dim=0)  # (Bt+Bi, Lenc_max)
           
            def make_query_mask(B, L_real, L_max, device, dtype=torch.long):
                """
                生成 decoder query 的 mask: 真实长度内为1，padding为0，形状 (B, L_max)。
                """
                mask = torch.zeros(B, L_max, dtype=dtype, device=device)
                mask[:, :L_real] = 1
                return mask
            device = text_queries.device
            text_q_attn_mask  = make_query_mask(Bt, Lt_q, Lq_max, device, dtype=torch.long)   # (Bt, Lq_max)
            image_q_attn_mask = make_query_mask(Bi, Li_q, Lq_max, device, dtype=torch.long)   # (Bi, Lq_max)
            q_attn_mask = torch.cat([text_q_attn_mask, image_q_attn_mask], dim=0)
            
            # 一次 forward
            ar_logits_pad = self.bilm(
                inputs_embeds=inputs_embeds,
                attention_mask=q_attn_mask, 
                encoder_hidden_states=enc_states,
                encoder_attention_mask=enc_mask,
                is_use_query_tokens=False,
            )  # 形状 (Bt+Bi, Lq_max, D_out) —— 解码器输出沿着 query 长度

            # 按 batch 切回两段，并去掉多余的 query padding
            language_ar_logits = ar_logits_pad[:Bt, :Lt_q]   # (Bt, Lt_q,  D_out)
            vision_ar_logits   = ar_logits_pad[Bt:, :Li_q]   # (Bi, Li_q, D_out)

            backbone_image_embeds = self.image_proj_1(image_qwen2vl_states)
            backbone_text_embeds = self.text_proj_1(text_qwen2vl_states)

            vision_embeds = self.image_proj(vision_ar_logits)
            vision_embeds = F.normalize(vision_embeds, dim=-1)

            text_embeds = self.text_proj(language_ar_logits)
            text_embeds = F.normalize(text_embeds, dim=-1)

            # ensemble
            backbone_image_embeds = F.normalize(backbone_image_embeds, dim=-1)
            backbone_text_embeds = F.normalize(backbone_text_embeds, dim=-1)
            
            # image_itc = vision_embeds + backbone_image_embeds
            # text_itc = text_embeds + backbone_text_embeds


            alpha_image = torch.sigmoid(self.fusion_alpha_image).to(vision_embeds.device)
            alpha_text = torch.sigmoid(self.fusion_alpha_text).to(text_embeds.device)
            image_itc = alpha_image * vision_embeds + (1 - alpha_image) * backbone_image_embeds
            text_itc = alpha_text * text_embeds + (1 - alpha_text) * backbone_text_embeds

            image_itc = F.normalize(image_itc, dim=-1)
            text_itc = F.normalize(text_itc, dim=-1)

            image_itc = image_itc.float()
            text_itc = text_itc.float()

            # print("image_itc.shape", image_itc.shape, "   text_itc.shape:", text_itc.shape)
            image_itc_all = GatherLayer.apply(image_itc).flatten(0, 1)
            image_itc_all = image_itc_all.float()
            text_itc_all = GatherLayer.apply(text_itc).flatten(0, 1)
            text_itc_all = text_itc_all.float()
            # print("image_itc_all.shape", image_itc_all.shape, "   text_itc_all.shape:", text_itc_all.shape)

            print("logit_scale: ", self.logit_scale.item(),"   current scale:", self.logit_scale.exp().item())

            # with torch.no_grad():
            # # clip_value 对应 log(100)
            #     self.logit_scale.clamp_(min=0.0, max=4.6052)

            logit_scale = self.logit_scale.exp().to(image_itc.device)    

            logits_per_image = logit_scale * (image_itc @ text_itc_all.t())
            logits_per_text = logit_scale * (text_itc @ image_itc_all.t())

            logits_text_text = logit_scale * (text_itc @ text_itc_all.t())
            # logits_per_text = logits_per_image.t()


            # # 计算 bank 中的相似度
            # bank = self.memory_bank.clone().detach().to(image_itc.device)  # [M,D]
            # logits_mb = logit_scale * (image_itc @ bank.t())  # [B, M]

            # # 合并 logits, 拼成一个大 logits 矩阵
            # logits = torch.cat([logits_per_image, logits_mb], dim=1)  # [B, B+M]

            # # 构造“全正例”分布 sim_targets_full

            # ## in-batch 正例
            # pos_ib = (image_ids.unsqueeze(1) == image_ids.unsqueeze(0)).float()   # [B, B]
            # sim_ib = pos_ib / pos_ib.sum(1, keepdim=True).clamp(min=1e-8)         # [B, B]
            # ## bank 中的正例：只要 bank_img_ids[j] == image_ids[i] 就当正
            # pos_mb = (self.bank_img_ids.unsqueeze(0) == image_ids.unsqueeze(1)).float()  # [B, M]
            # sim_mb = pos_mb / pos_mb.sum(1, keepdim=True).clamp(min=1e-8)           # [B, M]

            # ## 拼接 in-batch + bank 部分
            # sim_targets_full = torch.cat([sim_ib, sim_mb], dim=1)  # [B, B+M]

            # # 统一 softmax + InfoNCE
            # log_probs = F.log_softmax(logits, dim=1)              # [B, B+M]
            # loss_i2t  = - (sim_targets_full * log_probs).sum(1).mean()

            # # 文→图 对称处理
            # logits_t2i = logits.t()                               # [B+M, B]
            # sim_targets_t2i = sim_targets_full.t()                # [B+M, B]
            # log_probs_t2i  = F.log_softmax(logits_t2i, dim=1)     # [B+M, B]
            
            # per_row_loss = - (sim_targets_t2i * log_probs_t2i).sum(1)
            # # 找出哪些行有正样本（行和 > 0）
            # valid_rows = sim_targets_t2i.sum(dim=1) > 0               # [B+M] bool
            # # 只对这些行取平均
            # loss_t2i = per_row_loss[valid_rows].mean()

            # # 1) 得到正样本在 logits 矩阵中的位置列表
            # pos_mask_i2t = sim_targets_full > 0 
            # # 2) 计算 Image→Text Recall@K
            # def recall_at_k_i2t(logits, pos_mask, Ks=(1,5,10)):
            #     """
            #     logits: [B, B+M]
            #     pos_mask: [B, B+M]   True 表示该位置是正样本
            #     """
            #     recalls = {}
            #     # 对每张图，取 top-K 文本的索引
            #     for K in Ks:
            #         topk = logits.topk(K, dim=1).indices            # [B, K]
            #         # 看 topk 里是否有任意一个是正样本
            #         hits = pos_mask.gather(1, topk).any(dim=1)      # [B]
            #         recalls[K] = hits.float().mean().item()
            #     return recalls

            # rec_i2t = recall_at_k_i2t(logits, pos_mask_i2t, Ks=(1,5,10))
            # print(f"Batch I→T  R@1={rec_i2t[1]:.3f}, R@5={rec_i2t[5]:.3f}, R@10={rec_i2t[10]:.3f}")

            # # 3) 计算 Text→Image
            # #    先转置 logits 和 pos_mask，然后同理
            # pos_mask_t2i = pos_mask_i2t.t()              # [B+M, B]

            # # def recall_at_k_t2i(logits, pos_mask, Ks=(1,5,10)):
            # #     recalls = {}
            # #     # 先找出哪些行是“有至少一个正样本”的
            # #     valid = pos_mask.sum(dim=1) > 0        # [B+M]，in-batch的行一定是True，bank如果跨batch正例也会是True
            # #     for K in Ks:
            # #         topk = logits.topk(K, dim=1).indices    # [B+M, K]
            # #         hits = pos_mask.gather(1, topk).any(dim=1)
            # #         # 只在 valid 行上算平均
            # #         recalls[K] = hits[valid].float().mean().item()
            # #     return recalls

            # # rec_t2i = recall_at_k_t2i(logits_t2i, pos_mask_t2i, Ks=(1,5,10))
            # # print(f"Batch T→I  R@1={rec_t2i[1]:.3f}, R@5={rec_t2i[5]:.3f}, R@10={rec_t2i[10]:.3f}")

            # # **只**截 in-batch 文本对应的那 B 行
            # B = logits_t2i.size(1)
            # logits_ib_t2i = logits_t2i[:B]       # [B, B]
            # mask_ib_t2i   = pos_mask_t2i[:B]     # [B, B]

            # recalls = {}
            # for K in (1,5,10):
            #     topk = logits_ib_t2i.topk(K, dim=1).indices   # [B, K]
            #     hits = mask_ib_t2i.gather(1, topk).any(dim=1) # [B]
            #     recalls[K] = hits.float().mean().item()

            # print(f"Batch T→I (in-batch)  R@1={recalls[1]:.3f}, R@5={recalls[5]:.3f}, R@10={recalls[10]:.3f}")


            # self.dequeue_and_enqueue(text_itc.detach(), image_ids)

            # pdb.set_trace()

            # print("image_itc + text_itc cos sim:", F.cosine_similarity(image_itc, text_itc, dim=-1).mean().item())
            # print("vision_embeds cos sim:", F.cosine_similarity(vision_embeds, backbone_image_embeds, dim=-1).mean().item())
            # print("text_embeds cos sim:", F.cosine_similarity(text_embeds, backbone_text_embeds, dim=-1).mean().item())
            # backbone_i2t = logit_scale * (backbone_embeds @ text_itc_all.T)
            # backbone_t2i = logit_scale * (text_itc @ backbone_embeds_all.T) 

            # 构建 ground-truth 对齐标签（基于 image_ids）
            # pos_idx = (image_ids.view(-1, 1) == image_ids.view(1, -1)).float()
            # sim_targets = pos_idx / pos_idx.sum(dim=1, keepdim=True).clamp(min=1e-8)
            # sim_targets_t2i = sim_targets.t()

            image_ids = image_ids.view(-1, 1)
            image_ids_all = GatherLayer.apply(image_ids).flatten(0, 1)
            pos_idx = torch.eq(image_ids, image_ids_all.t()).float()
            sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
            # sim_targets_t2i = sim_targets.t()

            # ranks = torch.argsort(torch.argsort(-logits_per_image, dim=1), dim=1)
            # r1 = (ranks.diag()==0).float().mean().item()
            # r5 = (ranks.diag()<5).float().mean().item()
            with torch.no_grad():
                k = 10 # 先取一个最大的 k 值
                B = logits_per_image.shape[0] # local batch size

                # 1. 获取 Top-k 预测的索引
                #    topk_indices 的维度是 [B, k]
                _, topk_indices = torch.topk(logits_per_image, k=k, dim=1)

                # 2. 创建一个标识所有正例的布尔掩码 (boolean mask)
                #    sim_targets > 0 会将所有正例位置标记为 True，其余为 False
                #    positive_mask 的维度是 [B, 2*B] (全局 batch size)
                positive_mask = (sim_targets > 0)

                # 3. 检查 Top-k 预测中有多少是真正的正例
                #    使用 gather 函数，根据 topk_indices 从 positive_mask 中“提取”出相应的值。
                #    matches 的维度是 [B, k]，如果 topk_indices[i,j] 是一个正例，则 matches[i,j] 为 True
                matches = torch.gather(positive_mask, 1, topk_indices)

                # 4. 计算 Recall@k
                #    对于每个图像（每一行），只要 Top-k 中有任意一个命中（any(dim=1)），就算这个图像预测成功
                recall_at_k = matches.any(dim=1).float().mean().item()

                # 分别计算 R@1, R@5, R@10
                # R@1: 检查第一个预测是否为 True
                r1 = matches[:, 0].float().mean().item()
                
                # R@5: 检查前5个预测中是否有任意一个为 True
                r5 = matches[:, :5].any(dim=1).float().mean().item()

                # R@10: 检查前10个预测中是否有任意一个为 True
                r10 = matches[:, :10].any(dim=1).float().mean().item() # 这等同于 recall_at_k
                
                try:
                    gpu_rank = torch.distributed.get_rank()
                except:
                    gpu_rank = 0
                
                print(f"GPU {gpu_rank} | Multi-Positive Recall: R@1={r1:.3f}, R@5={r5:.3f}, R@10={r10:.3f}\n")


            gpu_rank = torch.distributed.get_rank()
            local_bs = text_itc.size(0)
            diag_mask = torch.zeros_like(pos_idx)
            diag_mask[:, gpu_rank*local_bs : (gpu_rank+1)*local_bs].fill_diagonal_(1)
            pos_idx_t2t = pos_idx - diag_mask

            # local_bs = text_itc.size(0) 
            # # 【关键步骤】排除自身与自身的匹配
            # # 在DDP中，一个样本“自己”的位置是在全局张量中的特定偏移量的对角线上。
            # # 我们可以创建一个对角线为1，其余为0的掩码，然后从pos_idx_t2t中减去它。
            # diag_mask = torch.zeros_like(pos_idx)
            # # 定位到当前GPU的样本在全局矩阵中对应的块
            # # 并在这个块的对角线上填充1
            # diag_mask[:, gpu_rank*local_bs : (gpu_rank+1)*local_bs].fill_diagonal_(1)

            # # 从正样本矩阵中剔除自己
            # pos_idx_t2t = pos_idx - diag_mask

            # 2.4 计算软标签 sim_targets_t2t
            # clamp(min=1e-8) 是为了防止分母为0 (当一个文本在全局batch中没有其他正例时)
            pos_idx_sum = pos_idx_t2t.sum(1)

            # logits_text_text.masked_fill_(diag_mask.bool(), -torch.inf)

            # 2. 创建一个布尔掩码，标记出那些“非孤独”的、有正样本的行
            valid_rows_mask = (pos_idx_sum > 0)

            # 3. （安全检查）如果当前全局批次中没有任何有效的正样本对，直接返回0损失
            if valid_rows_mask.sum() == 0:
                # 这种情况很少见，但可能在批次很小或数据特殊时发生
                print(f"Rank {gpu_rank} WARNING: No valid positive pairs found in this entire global batch for T2T loss.")
                loss_t2t = torch.tensor(0.0, device=text_itc.device, requires_grad=True)
            else:
                # 4. 只针对“非孤独”的行计算软标签
                #    注意：我们只从未过滤的 pos_idx_sum 中取值，因为它对应着 pos_idx_t2t
                sim_targets_t2t_filtered = pos_idx_t2t[valid_rows_mask] / pos_idx_sum[valid_rows_mask].unsqueeze(1)

                # 5. 从 logits 中也只选出这些有效的行
                logits_text_text_filtered = logits_text_text[valid_rows_mask]

                logits_text_text_f32 = logits_text_text_filtered.float()
                sim_targets_t2t_f32 = sim_targets_t2t_filtered.float()
                # 6. 在过滤后的、完全安全的数据上计算损失
                loss_t2t = -torch.sum(
                    F.log_softmax(logits_text_text_f32, dim=1) * sim_targets_t2t_f32,
                    dim=1
                ).mean()

            # 对比损失（InfoNCE）
            loss_t2i = -torch.sum(F.log_softmax(logits_per_text, dim=1) * sim_targets, dim=1).mean()
            loss_i2t = -torch.sum(F.log_softmax(logits_per_image, dim=1) * sim_targets, dim=1).mean()

            loss_itc = (loss_t2i + loss_i2t + 0.05*loss_t2t) 

            loss = loss_itc


            if torch.isnan(loss_t2t):
                # 如果执行到这里，说明问题比我们想象的更复杂
                rank = gpu_rank
                print(f"\n--- Rank {rank} CRITICAL DEBUG DUMP ---")
                print(f"NaN detected even after float32 casting!")
                
                # 打印导致NaN的原始输入（float16版本）
                print("Logits (filtered, fp16) stats:", logits_text_text_filtered.min().item(), logits_text_text_filtered.max().item(), logits_text_text_filtered.mean().item())
                print("Targets (filtered, fp16) stats:", sim_targets_t2t_filtered.min().item(), sim_targets_t2t_filtered.max().item(), sim_targets_t2t_filtered.mean().item())
                
                # 检查原始logits中是否已经包含了 inf
                if torch.isinf(logits_text_text_filtered).any():
                    print("!!! DETECTED 'inf' in filtered logits BEFORE log_softmax !!!")

            print(f"loss_itc: {loss_itc.item():.6f}  loss_t2i: {loss_t2i.item():.6f}   loss_i2t: {loss_i2t.item():.6f}   loss_t2t: {loss_t2t.item():.6f}\n")
            # print(f"mse_loss: {mse_loss}    text_loss: {text_loss}  image_loss: {image_loss}")


            # pdb.set_trace()

            # if not return_dict:
            #     output = (logits,) + outputs[1:]
            #     return (loss,) + output if loss is not None else output

            return Qwen2VLCausalLMOutputWithPast(
                loss=loss,
                logits=language_ar_logits,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
                rope_deltas=self.rope_deltas,
            )

        
        elif not (hasattr(self, "hy_ar_head") or hasattr(self, "bilm")) or labels is None:
            # pdb.set_trace()
            outputs = self.model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)

            loss = None
            if labels is not None:
                # Upcast to float if we need to compute the loss to avoid potential precision issues
                logits = logits.float()
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return Qwen2VLCausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                rope_deltas=self.rope_deltas,
            )
        
        else:
            

            ds_qwen2vl_states = self.downsampling_linear_layer(qwen2vl_states)
            t5_embeddings = t5_embeddings.to(torch.bfloat16)
            # t5_embeddings.requires_grad_()

            image_ids = torch.tensor(image_ids, dtype=torch.long, device=t5_embeddings.device)  # [B]

            retrieval_mask = torch.tensor(retrieval_index, dtype=torch.bool, device=t5_embeddings.device)  # shape [B]

            mask = retrieval_mask.unsqueeze(-1).unsqueeze(-1)   # 变成 [B,1,1] 以广播到 [B,N,D]
            masked_states = ds_qwen2vl_states.masked_fill(mask, 0.0)


            hy_ar_logits = self.bilm(
                # hidden_states = t5_embeddings,
                inputs_embeds = t5_embeddings,
                encoder_hidden_states = masked_states,
                encoder_attention_mask = qwen2vl_attention_masks,
                edit_indexs = actual_lengths,
            )

            vision_ar_logits = self.bilm(
                # hidden_states = t5_embeddings,
                inputs_embeds = t5_embeddings,
                encoder_hidden_states = masked_states,
                encoder_attention_mask = qwen2vl_attention_masks,
                edit_indexs = actual_lengths,
                is_retrieval_text = True,
            )

            # pdb.set_trace()
            ds_retrieval_qwen2vl_states = self.downsampling_linear_layer(retrieval_hidden_states)
            text_ar_logits = self.bilm(
                # hidden_states = t5_embeddings,
                inputs_embeds = t5_embeddings,
                encoder_hidden_states = ds_retrieval_qwen2vl_states,
                encoder_attention_mask = retrieval_attention_mask,
                edit_indexs = actual_lengths,
            )

            # retrieval_index

            if any(element is not False for element in retrieval_index):
                hy_ar_logits = hy_ar_logits.float().contiguous()
                hy_ar_labels = t5_ar_labels.float().contiguous()

                non_retrieval_mask = ~retrieval_mask
                hy_ar_logits_sel = hy_ar_logits[non_retrieval_mask]
                hy_ar_labels_sel = hy_ar_labels[non_retrieval_mask]
                if hy_ar_logits_sel.numel() == 0:
                    t5_loss = torch.tensor(0.0, device=hy_ar_logits.device)
                else:
                    hy_ar_loss_fct = nn.MSELoss()
                    t5_loss = hy_ar_loss_fct(hy_ar_logits_sel, hy_ar_labels_sel)
                

                text_itc = text_ar_logits[:, 0, :]


                text_itc = self.text_proj(text_itc)
                
                img_h = self.image_projector(vision_ar_logits)
                # img_h = self.image_projector(masked_states)

                image_itc = self.image_proj(img_h)

                image_itc = F.normalize(image_itc, dim=1)
                text_itc = F.normalize(text_itc, dim=1)

                # 筛选有效样本
                image_itc = image_itc[retrieval_mask]
                # backbone_embeds = backbone_embeds[retrieval_mask]
                text_itc = text_itc[retrieval_mask]
                image_ids = image_ids[retrieval_mask]

                # 多卡 all-gather（使用 GatherLayer）
                if dist.is_available() and dist.is_initialized():
                    image_itc_all = GatherLayer.apply(image_itc).flatten(0, 1)
                    # backbone_embeds_all = GatherLayer.apply(backbone_embeds).flatten(0, 1)
                    text_itc_all = GatherLayer.apply(text_itc).flatten(0, 1)
                    image_ids_all = GatherLayer.apply(image_ids.view(-1, 1)).flatten(0, 1)
                else:
                    image_itc_all = image_itc
                    text_itc_all = text_itc
                    image_ids_all = image_ids
                    # backbone_embeds_all = backbone_embeds
                # pdb.set_trace()
                # 相似度计算
                logit_scale = self.logit_scale.exp()
                sim_i2t = logit_scale * (image_itc @ text_itc_all.T)
                sim_t2i = logit_scale * (text_itc @ image_itc_all.T)
                # backbone_i2t = logit_scale * (backbone_embeds @ text_itc_all.T)
                # backbone_t2i = logit_scale * (text_itc @ backbone_embeds_all.T)

                # 构建 ground-truth 对齐标签（基于 image_ids）
                pos_idx = (image_ids.view(-1, 1) == image_ids_all.view(1, -1)).float()
                sim_targets = pos_idx / pos_idx.sum(dim=1, keepdim=True).clamp(min=1e-8)

                # 对比损失（InfoNCE）
                loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_targets, dim=1).mean()
                loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1).mean()
                # loss_backbone_t2i = -torch.sum(F.log_softmax(backbone_t2i, dim=1) * sim_targets, dim=1).mean()
                # loss_backbone_i2t = -torch.sum(F.log_softmax(backbone_i2t, dim=1) * sim_targets, dim=1).mean()
                loss_itc = (loss_t2i + loss_i2t) / 2 

                print(f"t5_loss: {t5_loss}    loss_itc: {loss_itc}   loss_t2i: {loss_t2i}   loss_i2t: {loss_i2t}")

                t5_loss = t5_loss + loss_itc

            else:

                hy_ar_logits = hy_ar_logits.float().contiguous()
                hy_ar_labels = t5_ar_labels.float().contiguous()

                hy_ar_loss_fct = MSELoss()

                t5_loss = hy_ar_loss_fct(hy_ar_logits, hy_ar_labels)

            # pdb.set_trace()

            # t5_loss = (shift_hy_ar_logits - shift_hy_ar_labels).pow(2).mean(dim=-1)  # 对特征维度取平均
            # t5_loss = t5_loss.mean(dim=-1)  # 对 token 维度取平均
            # t5_loss = t5_loss.sum()  # 对整个 batch 求和

            # hy_ar_loss_fct = MSELoss(reduction='none')
            # elementwise_loss = hy_ar_loss_fct(shift_hy_ar_logits, shift_hy_ar_labels)

            # # 对每个 token 取平均，沿着最后一个维度（2048）取平均
            # # 结果形状为 (batch_size, 256)
            # tokenwise_loss = elementwise_loss.mean(dim=-1)

            # # 对整个 batch 的 token 损失取平均
            # # final_loss = tokenwise_loss.mean()
            # total_loss = tokenwise_loss.sum()
            
            # # pdb.set_trace()

            # 定义 KL 散度损失函数
            # def kl_loss_fct(p, q):
            #     # 确保 p 和 q 是概率分布（即每一行的和为 1）
            #     p = F.softmax(p, dim=-1)
            #     q = F.log_softmax(q, dim=-1)
            #     return F.kl_div(q, p, reduction='batchmean')  # 使用 'batchmean' 对 batch 取平均
            
            # # # 计算 KL 散度损失
            # kl_loss = kl_loss_fct(shift_hy_ar_logits, shift_hy_ar_labels)

            # # cos_similarity = F.cosine_similarity(shift_hy_ar_logits, shift_hy_ar_labels, dim=-1, eps=1e-8)

            # # loss_matrix = 1.0 - cos_similarity
            # # cos_similarity_loss = loss_matrix.mean()

            # print(f"t5_loss: {total_loss}  kl_loss: {kl_loss}")

            if hasattr(self, "query_ar_head"):
                pdb.set_trace()
                query_ar_logits = self.query_ar_head(
                    hidden_states = None,
                    encoder_hidden_states = qwen2vl_states,
                    encoder_attention_mask = qwen2vl_attention_masks,
                    query_length = None
                )
                query_ar_logits = query_ar_logits.float()
                query_ar_labels = t5_ar_labels.float()

                # Shift so that tokens < n predict n
                shift_query_ar_logits = query_ar_logits.contiguous()
                shift_query_ar_labels = query_ar_labels.contiguous()

                query_ar_loss_fct = MSELoss()

                query_loss = query_ar_loss_fct(shift_query_ar_logits, shift_query_ar_labels)

                #   计算对比损失
                # 1. 归一化
                query_logits = F.normalize(query_ar_logits, p=2, dim=-1)  # 归一化query特征
                t5_logits = F.normalize(hy_ar_logits, p=2, dim=-1)  # 归一化t5特征
                # 2. 计算点积相似度
                similarity_matrix = torch.matmul(query_logits, t5_logits.transpose(1, 2))
                # 3. 计算对比损失
                # 目标是使得正样本的相似度最大化，负样本的相似度最小化
                temperature = 0.07
                logits = similarity_matrix / temperature  # 对相似度进行缩放
                # 4. 使用Softmax和交叉熵计算损失
                contrastive_labels = torch.arange(query_logits.size(0)).to(query_logits.device)  # 正样本标签

                contrastive_loss = F.cross_entropy(logits, contrastive_labels)

                print(f"t5_loss: {t5_loss}  query_loss: {query_loss}  contrastive_loss: {contrastive_loss}")
                loss = t5_loss + t5_loss + contrastive_loss
            else:
                # loss = total_loss + 0.5*kl_loss
                loss = t5_loss 
            

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return Qwen2VLCausalLMOutputWithPast(
                loss=loss,
                logits=hy_ar_logits,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
                rope_deltas=self.rope_deltas,
            )



    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            attention_mask = self.model._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=self.lm_head.weight.dtype,
                device=device,
                cache_position=cache_position,
                batch_size=batch_size,
                config=self.config,
                past_key_values=past_key_values,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "cache_position": cache_position,
            }
        )
        return model_inputs


__all__ = ["Qwen2VLForConditionalGeneration", "Qwen2VLModel", "Qwen2VLPreTrainedModel"]
