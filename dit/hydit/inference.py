import pdb
import random
import time
from pathlib import Path

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch

from diffusers import schedulers
from diffusers.models import AutoencoderKL
from diffusers.models.lora import adjust_lora_scale_text_encoder
from loguru import logger
from transformers import BertModel, BertTokenizer
from transformers.modeling_utils import logger as tf_logger
import torch.nn as nn
import torch.nn.functional as F
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin

from peft import LoraConfig

from .constants import SAMPLER_FACTORY, NEGATIVE_PROMPT, TRT_MAX_WIDTH, TRT_MAX_HEIGHT, TRT_MAX_BATCH_SIZE
from .diffusion.pipeline import StableDiffusionPipeline, StableDiffusionPipelineWithoutEncoder
from .modules.models import HunYuanDiT, HUNYUAN_DIT_CONFIG, FP32_SiLU
from .modules.posemb_layers import get_2d_rotary_pos_embed, get_fill_resize_and_crop
from .modules.text_encoder import MT5Embedder
from .utils.tools import set_seeds


class Resolution:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __str__(self):
        return f'{self.height}x{self.width}'


class ResolutionGroup:
    def __init__(self):
        self.data = [
            Resolution(1024, 1024), # 1:1
            Resolution(1280, 1280), # 1:1
            Resolution(1024, 768),  # 4:3
            Resolution(1152, 864),  # 4:3
            Resolution(1280, 960),  # 4:3
            Resolution(768, 1024),  # 3:4
            Resolution(864, 1152),  # 3:4
            Resolution(960, 1280),  # 3:4
            Resolution(1280, 768),  # 16:9
            Resolution(768, 1280),  # 9:16
        ]
        self.supported_sizes = set([(r.width, r.height) for r in self.data])


def is_valid(self, width, height):
    return (width, height) in self.supported_sizes


STANDARD_RATIO = np.array([
    1.0,        # 1:1
    4.0 / 3.0,  # 4:3
    3.0 / 4.0,  # 3:4
    16.0 / 9.0, # 16:9
    9.0 / 16.0, # 9:16
])
STANDARD_SHAPE = [
    [(1024, 1024), (1280, 1280)],   # 1:1
    [(1280, 960)],                # 4:3
    [(960, 1280)],                   # 3:4
    [(1280, 768)],                              # 16:9
    [(768, 1280)],                              # 9:16
]
STANDARD_AREA = [
    np.array([w * h for w, h in shapes])
    for shapes in STANDARD_SHAPE
]


def get_standard_shape(target_width, target_height):
    """
    Map image size to standard size.
    """
    target_ratio = target_width / target_height
    closest_ratio_idx = np.argmin(np.abs(STANDARD_RATIO - target_ratio))
    closest_area_idx = np.argmin(np.abs(STANDARD_AREA[closest_ratio_idx] - target_width * target_height))
    width, height = STANDARD_SHAPE[closest_ratio_idx][closest_area_idx]
    return width, height

def _to_tuple(val):
    if isinstance(val, (list, tuple)):
        if len(val) == 1:
            val = [val[0], val[0]]
        elif len(val) == 2:
            val = tuple(val)
        else:
            raise ValueError(f"Invalid value: {val}")
    elif isinstance(val, (int, float)):
        val = (val, val)
    else:
        raise ValueError(f"Invalid value: {val}")
    return val


def get_pipeline(args, vae, text_encoder, tokenizer, model, device, rank,
                 embedder_t5, infer_mode, sampler=None):
    """
    Get scheduler and pipeline for sampling. The sampler and pipeline are both
    based on diffusers and make some modifications.

    Returns
    -------
    pipeline: StableDiffusionPipeline
    sampler_name: str
    """
    sampler = sampler or args.sampler

    # Load sampler from factory
    kwargs = SAMPLER_FACTORY[sampler]['kwargs']
    scheduler = SAMPLER_FACTORY[sampler]['scheduler']

    # Update sampler according to the arguments
    kwargs['beta_schedule'] = args.noise_schedule
    kwargs['beta_start'] = args.beta_start
    kwargs['beta_end'] = args.beta_end
    kwargs['prediction_type'] = args.predict_type

    # Build scheduler according to the sampler.
    scheduler_class = getattr(schedulers, scheduler)
    scheduler = scheduler_class(**kwargs)
    logger.debug(f"Using sampler: {sampler} with scheduler: {scheduler}")

    # Set timesteps for inference steps.
    scheduler.set_timesteps(args.infer_steps, device)

    # Only enable progress bar for rank 0
    progress_bar_config = {} if rank == 0 else {'disable': True}

    pipeline = StableDiffusionPipeline(vae=vae,
                                       text_encoder=text_encoder,
                                       tokenizer=tokenizer,
                                       unet=model,
                                       scheduler=scheduler,
                                       feature_extractor=None,
                                       safety_checker=None,
                                       requires_safety_checker=False,
                                       progress_bar_config=progress_bar_config,
                                       embedder_t5=embedder_t5,
                                       infer_mode=infer_mode,
                                       )

    pipeline = pipeline.to(device)

    return pipeline, sampler


def get_pipeline_without_encoder(args, vae, model, device, rank,
                 infer_mode, sampler=None):
    """
    Get scheduler and pipeline for sampling. The sampler and pipeline are both
    based on diffusers and make some modifications.

    Returns
    -------
    pipeline: StableDiffusionPipeline
    sampler_name: str
    """
    sampler = sampler or args.sampler

    # Load sampler from factory
    kwargs = SAMPLER_FACTORY[sampler]['kwargs']
    scheduler = SAMPLER_FACTORY[sampler]['scheduler']

    # Update sampler according to the arguments
    kwargs['beta_schedule'] = args.noise_schedule
    kwargs['beta_start'] = args.beta_start
    kwargs['beta_end'] = args.beta_end
    kwargs['prediction_type'] = args.predict_type

    # Build scheduler according to the sampler.
    scheduler_class = getattr(schedulers, scheduler)
    scheduler = scheduler_class(**kwargs)
    logger.debug(f"Using sampler: {sampler} with scheduler: {scheduler}")

    # Set timesteps for inference steps.
    scheduler.set_timesteps(args.infer_steps, device)

    # Only enable progress bar for rank 0
    progress_bar_config = {} if rank == 0 else {'disable': True}

    pipeline = StableDiffusionPipelineWithoutEncoder(vae=vae,
                                       unet=model,
                                       scheduler=scheduler,
                                       feature_extractor=None,
                                       safety_checker=None,
                                       requires_safety_checker=False,
                                       progress_bar_config=progress_bar_config,
                                       infer_mode=infer_mode,
                                       )

    pipeline = pipeline.to(device)

    return pipeline, sampler


class Captioner2Embedding(object):
    def __init__(self, args, models_root_path, pre_mlp_t5_path, device):
        self.args = args

        # Check arguments
        t2i_root_path = Path(models_root_path) / "t2i"
        self.root = t2i_root_path
        logger.info(f"Got text-to-image model root path: {t2i_root_path}")

        # Set device and disable gradient
        self.device = device
        # torch.set_grad_enabled(False)
        # Disable BertModel logging checkpoint info
        tf_logger.setLevel('ERROR')

        # ========================================================================
        text_encoder_path = self.root / "clip_text_encoder"
        self.clip_text_encoder = BertModel.from_pretrained(str(text_encoder_path), False, revision=None).to(self.device)

        self.clip_text_encoder.requires_grad_(False)
        # ========================================================================
        tokenizer_path = self.root / "tokenizer"
        self.tokenizer = BertTokenizer.from_pretrained(str(tokenizer_path))

        # ========================================================================
        t5_text_encoder_path = self.root / 'mt5'
        embedder_t5 = MT5Embedder(t5_text_encoder_path, torch_dtype=torch.float16, max_length=256)
        self.embedder_t5 = embedder_t5
        self.embedder_t5.model.to(self.device)  # Only move encoder to device

        self.embedder_t5.requires_grad_(False)
        # ========================================================================
        self.text_states_dim = args.text_states_dim
        self.text_states_dim_t5 = args.text_states_dim_t5
        self.text_len = args.text_len
        self.text_len_t5 = args.text_len_t5
        # self.norm = args.norm

        self.mlp_t5 = nn.Sequential(
            nn.Linear(self.text_states_dim_t5, self.text_states_dim_t5 * 4, bias=True),
            FP32_SiLU(),
            nn.Linear(self.text_states_dim_t5 * 4, self.text_states_dim, bias=True),
        )
        self.mlp_t5 = self.mlp_t5.to(device=self.device, dtype=torch.float16)

        state_dict = torch.load(pre_mlp_t5_path, map_location=lambda storage, loc: storage)
        self.mlp_t5.load_state_dict(state_dict, strict=True)

        self.text_embedding_padding = nn.Parameter(
            torch.randn(self.text_len + self.text_len_t5, self.text_states_dim, dtype=torch.float32), requires_grad=False)
        self.text_embedding_padding = self.text_embedding_padding.to(self.device)

        self.mlp_t5.requires_grad_(False)

        # ========================================================================
        self.default_negative_prompt = NEGATIVE_PROMPT
        logger.info("==================================================")
        logger.info(f"                Model is ready.                  ")
        logger.info("==================================================")

    def generate_embedding(self,
                user_prompt,
                prompt_embeds = None,
                enhanced_prompt=None,
                negative_prompt=None,
                guidance_scale=6,
                batch_size=1,
                num_images_per_prompt = 1
                ):
        # ========================================================================

        # ========================================================================
        # Arguments: prompt, new_prompt, negative_prompt
        # ========================================================================
        if user_prompt is not None:

            if not isinstance(user_prompt, str):
                raise TypeError(f"`user_prompt` must be a string, but got {type(user_prompt)}")
            user_prompt = user_prompt.strip()
            prompt = user_prompt
        else:
            prompt = None

        if enhanced_prompt is not None:
            if not isinstance(enhanced_prompt, str):
                raise TypeError(f"`enhanced_prompt` must be a string, but got {type(enhanced_prompt)}")
            enhanced_prompt = enhanced_prompt.strip()
            prompt = enhanced_prompt
        

        # negative prompt
        if negative_prompt is None or negative_prompt == '':
            negative_prompt = self.default_negative_prompt
        if not isinstance(negative_prompt, str):
            raise TypeError(f"`negative_prompt` must be a string, but got {type(negative_prompt)}")

        # ========================================================================
        # Arguments: style. (A fixed argument. Don't Change it.)
        # ========================================================================
        
        device = self.device

        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        cross_attention_kwargs = None
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds_t5 = None
        negative_prompt_embeds = None
        negative_prompt_embeds_t5 = None

        prompt_embeds, negative_prompt_embeds, attention_mask, uncond_attention_mask = \
        self.encode_prompt(prompt,
                            device,
                            num_images_per_prompt,
                            do_classifier_free_guidance,
                            negative_prompt,
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=negative_prompt_embeds,
                            lora_scale=text_encoder_lora_scale,
                            )
        
        prompt_embeds_t5, negative_prompt_embeds_t5, attention_mask_t5, uncond_attention_mask_t5 = \
        self.encode_prompt(prompt,
                            device,
                            num_images_per_prompt,
                            do_classifier_free_guidance,
                            negative_prompt,
                            prompt_embeds=prompt_embeds_t5,
                            negative_prompt_embeds=negative_prompt_embeds_t5,
                            lora_scale=text_encoder_lora_scale,
                            embedder=self.embedder_t5,
                            )
        
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        attention_mask = torch.cat([uncond_attention_mask, attention_mask])
        prompt_embeds_t5 = torch.cat([negative_prompt_embeds_t5, prompt_embeds_t5])
        attention_mask_t5 = torch.cat([uncond_attention_mask_t5, 
           attention_mask_t5])
        # pdb.set_trace()
        text_states = prompt_embeds                     # 2,77,1024
        text_states = text_states.half()
        text_states_t5 = prompt_embeds_t5               # 2,256,2048
        text_states_mask = attention_mask.bool()           # 2,77
        text_states_t5_mask = attention_mask_t5.bool()     # 2,256
        b_t5, l_t5, c_t5 = text_states_t5.shape
        text_states_t5 = self.mlp_t5(text_states_t5.view(-1, c_t5))
        text_states = torch.cat([text_states, text_states_t5.view(b_t5, l_t5, -1)], dim=1)  # 2,205，1024

        # text_states_mask = torch.zeros_like(text_states_mask)  # 第三种
        # text_states_t5_mask = torch.zeros_like(text_states_t5_mask)  # 第四种
        clip_t5_mask = torch.cat([text_states_mask, text_states_t5_mask], dim=-1)

        clip_t5_mask = clip_t5_mask


        text_states = torch.where(clip_t5_mask.unsqueeze(2), text_states, self.text_embedding_padding.to(text_states))
        return text_states[1]


    def encode_prompt(
            self,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            lora_scale: Optional[float] = None,
            embedder=None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            embedder:
                T5 embedder (including text encoder and tokenizer): 256 token
        """
        # pdb.set_trace()
        if embedder is None:
            text_encoder = self.clip_text_encoder
            tokenizer = self.tokenizer
            max_length = self.tokenizer.model_max_length
        else:
            text_encoder = embedder.model
            tokenizer = embedder.tokenizer
            max_length = embedder.max_length

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(
                    untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {tokenizer.model_max_length} tokens: {removed_text}"
                )

            attention_mask = text_inputs.attention_mask.to(device)
            prompt_embeds = text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]
            attention_mask = attention_mask.repeat(num_images_per_prompt, 1)
        else:
            attention_mask = None

        if text_encoder is not None:
            prompt_embeds_dtype = text_encoder.dtype
        # elif self.unet is not None:
        #     prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            uncond_attention_mask = uncond_input.attention_mask.to(device)
            negative_prompt_embeds = text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=uncond_attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]
            uncond_attention_mask = uncond_attention_mask.repeat(num_images_per_prompt, 1)
        else:
            uncond_attention_mask = None

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds, attention_mask, uncond_attention_mask


class End2End(object):
    def __init__(self, args, models_root_path):
        self.args = args

        # Check arguments
        t2i_root_path = Path(models_root_path) / "t2i"
        self.root = t2i_root_path
        logger.info(f"Got text-to-image model root path: {t2i_root_path}")
        
        # Set device and disable gradient
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_grad_enabled(False)
        # Disable BertModel logging checkpoint info
        tf_logger.setLevel('ERROR')

        # ========================================================================
        text_encoder_path = self.root / "clip_text_encoder"
        self.clip_text_encoder = BertModel.from_pretrained(str(text_encoder_path), False, revision=None).to(self.device)

        # ========================================================================
        tokenizer_path = self.root / "tokenizer"
        self.tokenizer = BertTokenizer.from_pretrained(str(tokenizer_path))

        # pdb.set_trace()
        # ========================================================================
        t5_text_encoder_path = self.root / 'mt5'
        embedder_t5 = MT5Embedder(t5_text_encoder_path, torch_dtype=torch.float16, max_length=256)
        self.embedder_t5 = embedder_t5
        self.embedder_t5.model.to(self.device)  # Only move encoder to device

        # pdb.set_trace()
        # ========================================================================
        logger.info(f"Loading VAE...")
        vae_path = self.root / "sdxl-vae-fp16-fix"
        self.vae = AutoencoderKL.from_pretrained(str(vae_path)).to(self.device)
        logger.info(f"Loading VAE finished")

        # ========================================================================
        # Create model structure and load the checkpoint
        logger.info(f"Building HunYuan-DiT model...")
        model_config = HUNYUAN_DIT_CONFIG[self.args.model]
        self.patch_size = model_config['patch_size']
        self.head_size = model_config['hidden_size'] // model_config['num_heads']
        self.resolutions, self.freqs_cis_img = self.standard_shapes()   # Used for TensorRT models
        self.image_size = _to_tuple(self.args.image_size)
        latent_size = (self.image_size[0] // 8, self.image_size[1] // 8)

        self.infer_mode = self.args.infer_mode
        if self.infer_mode in ['fa', 'torch']:
            # Build model structure
            self.model = HunYuanDiT(self.args,
                                    input_size=latent_size,
                                    **model_config,
                                    log_fn=logger.info,
                                    ).half().to(self.device)    # Force to use fp16

            # Load model checkpoint
            self.load_torch_weights()

            lora_ckpt = args.lora_ckpt
            if lora_ckpt is not None and lora_ckpt != "":
                logger.info(f"Loading Lora checkpoint {lora_ckpt}...")

                self.model.load_adapter(lora_ckpt)
                self.model.merge_and_unload()

            self.model.eval()
            logger.info(f"Loading torch model finished")
        elif self.infer_mode == 'trt':
            from .modules.trt.hcf_model import TRTModel

            trt_dir = self.root / "model_trt"
            engine_dir = trt_dir / "engine"
            plugin_path = trt_dir / "fmha_plugins/9.2_plugin_cuda11/fMHAPlugin.so"
            model_name = "model_onnx"

            logger.info(f"Loading TensorRT model {engine_dir}/{model_name}...")
            self.model = TRTModel(model_name=model_name,
                                  engine_dir=str(engine_dir),
                                  image_height=TRT_MAX_HEIGHT,
                                  image_width=TRT_MAX_WIDTH,
                                  text_maxlen=args.text_len,
                                  embedding_dim=args.text_states_dim,
                                  plugin_path=str(plugin_path),
                                  max_batch_size=TRT_MAX_BATCH_SIZE,
                                  )
            logger.info(f"Loading TensorRT model finished")
        else:
            raise ValueError(f"Unknown infer_mode: {self.infer_mode}")

        # ========================================================================
        # Build inference pipeline. We use a customized StableDiffusionPipeline.
        logger.info(f"Loading inference pipeline...")
        self.pipeline, self.sampler = self.load_sampler()
        logger.info(f'Loading pipeline finished')

        # ========================================================================
        self.default_negative_prompt = NEGATIVE_PROMPT
        logger.info("==================================================")
        logger.info(f"                Model is ready.                  ")
        logger.info("==================================================")

    def to(self, device):
        """
        将所有 nn.Module 类型的子模块移动到指定设备。
        """
        for attr_name in self.__dict__:
            # 获取属性值
            attr_value = getattr(self, attr_name)
            # 检查属性是否是 nn.Module 的实例
            if isinstance(attr_value, nn.Module):
                # 如果是，就将其移动到 device
                setattr(self, attr_name, attr_value.to(device))
        
        print("All model attributes moved to device.")
        # 返回 self 以支持链式调用，例如 model.to(device).eval()
        return self

        
    def load_torch_weights(self):

        load_key = self.args.load_key
        if self.args.dit_weight is not None:
            dit_weight = Path(self.args.dit_weight)
            if dit_weight.is_dir():
                files = list(dit_weight.glob("*.pt"))
                if len(files) == 0:
                    raise ValueError(f"No model weights found in {dit_weight}")
                if str(files[0]).startswith('pytorch_model_'):
                    model_path = dit_weight / f"pytorch_model_{load_key}.pt"
                    bare_model = True
                elif any(str(f).endswith('_model_states.pt') for f in files):
                    files = [f for f in files if str(f).endswith('_model_states.pt')]
                    model_path = files[0]
                    if len(files) > 1:
                        logger.warning(f"Multiple model weights found in {dit_weight}, using {model_path}")
                    bare_model = False
                else:
                    raise ValueError(f"Invalid model path: {dit_weight} with unrecognized weight format: "
                                     f"{list(map(str, files))}. When given a directory as --dit-weight, only "
                                     f"`pytorch_model_*.pt`(provided by HunyuanDiT official) and "
                                     f"`*_model_states.pt`(saved by deepspeed) can be parsed. If you want to load a "
                                     f"specific weight file, please provide the full path to the file.")
            elif dit_weight.is_file():
                model_path = dit_weight
                bare_model = 'unknown'
            else:
                raise ValueError(f"Invalid model path: {dit_weight}")
        else:
            model_dir = self.root / "model"
            model_path = model_dir / f"pytorch_model_{load_key}.pt"
            bare_model = True

        if not model_path.exists():
            raise ValueError(f"model_path not exists: {model_path}")
        logger.info(f"Loading torch model {model_path}...")
        if model_path.suffix == '.safetensors':
            raise NotImplementedError(f"Loading safetensors is not supported yet.")
        else:
            # Assume it's a single weight file in the *.pt format.
            state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

        if bare_model == 'unknown' and ('ema' in state_dict or 'module' in state_dict):
            bare_model = False
        if bare_model is False:
            if load_key in state_dict:
                state_dict = state_dict[load_key]
            else:
                raise KeyError(f"Missing key: `{load_key}` in the checkpoint: {model_path}. The keys in the checkpoint "
                               f"are: {list(state_dict.keys())}.")

        if 'style_embedder.weight' in state_dict and not hasattr(self.model, 'style_embedder'):
            raise ValueError(f"You might be attempting to load the weights of HunYuanDiT version <= 1.1. You need "
                             f"to set `--use-style-cond --size-cond 1024 1024 --beta-end 0.03` to adapt to these weights."
                             f"Alternatively, you can use weights of version >= 1.2, which no longer depend on "
                             f"these two parameters.")
        if 'style_embedder.weight' not in state_dict and hasattr(self.model, 'style_embedder'):
            raise ValueError(f"You might be attempting to load the weights of HunYuanDiT version >= 1.2. You need "
                             f"to remove `--use-style-cond` and `--size-cond 1024 1024` to adapt to these weights.")


        # # 提取第一个 Linear 层的权重和偏置 (in_features=2048, out_features=8192)
        # linear1_weight = state_dict['mlp_t5.0.weight']
        # linear1_bias = state_dict['mlp_t5.0.bias']

        # # 提取第二个 Linear 层的权重和偏置 (in_features=8192, out_features=1024)
        # linear2_weight = state_dict['mlp_t5.2.weight']
        # linear2_bias = state_dict['mlp_t5.2.bias']

        # weights_to_save = {
        #     '0.weight': linear1_weight,
        #     '0.bias': linear1_bias,
        #     '2.weight': linear2_weight,
        #     '2.bias': linear2_bias
        # }

        # # 保存为 .pt 文件
        # torch.save(weights_to_save, '/data/xiaot/LLaVA/HunyuanDiT/ckpts/t2i/model/mlp_t5_weights.pt')
        # Don't set strict=False. Always explicitly check the state_dict.

        self.model.load_state_dict(state_dict, strict=True)

    def load_sampler(self, sampler=None):
        pipeline, sampler = get_pipeline(self.args,
                                         self.vae,
                                         self.clip_text_encoder,
                                         self.tokenizer,
                                         self.model,
                                         device=self.device,
                                         rank=0,
                                         embedder_t5=self.embedder_t5,
                                         infer_mode=self.infer_mode,
                                         sampler=sampler,
                                         )
        return pipeline, sampler

    def load_sampler_without_encoder(self, sampler=None):
        pipeline, sampler = get_pipeline(self.args,
                                         self.vae,
                                         self.model,
                                         device=self.device,
                                         rank=0,
                                         infer_mode=self.infer_mode,
                                         sampler=sampler,
                                         )
        return pipeline, sampler

    def calc_rope(self, height, width):
        th = height // 8 // self.patch_size
        tw = width // 8 // self.patch_size
        base_size = 512 // 8 // self.patch_size
        start, stop = get_fill_resize_and_crop((th, tw), base_size)
        sub_args = [start, stop, (th, tw)]
        rope = get_2d_rotary_pos_embed(self.head_size, *sub_args)
        return rope

    def standard_shapes(self):
        resolutions = ResolutionGroup()
        freqs_cis_img = {}
        for reso in resolutions.data:
            freqs_cis_img[str(reso)] = self.calc_rope(reso.height, reso.width)
        return resolutions, freqs_cis_img

    def predict(self,
                user_prompt,
                prompt_embeds = None,
                height=1024,
                width=1024,
                seed=None,
                enhanced_prompt=None,
                negative_prompt=None,
                infer_steps=100,
                guidance_scale=6,
                batch_size=1,
                src_size_cond=(1024, 1024),
                sampler=None,
                use_style_cond=False,
                has_cfg=True,
                is_get_embedding = False,
                from_embedd = None
                ):
        # ========================================================================
        # Arguments: seed
        # ========================================================================
        if seed is None:
            seed = random.randint(0, 1_000_000)
        if not isinstance(seed, int):
            raise TypeError(f"`seed` must be an integer, but got {type(seed)}")
        generator = set_seeds(seed, device=self.device)
        # ========================================================================
        # Arguments: target_width, target_height
        # ========================================================================
        if width <= 0 or height <= 0:
            raise ValueError(f"`height` and `width` must be positive integers, got height={height}, width={width}")
        logger.info(f"Input (height, width) = ({height}, {width})")
        if self.infer_mode in ['fa', 'torch']:
            # We must force height and width to align to 16 and to be an integer.
            target_height = int((height // 16) * 16)
            target_width = int((width // 16) * 16)
            logger.info(f"Align to 16: (height, width) = ({target_height}, {target_width})")
        elif self.infer_mode == 'trt':
            target_width, target_height = get_standard_shape(width, height)
            logger.info(f"Align to standard shape: (height, width) = ({target_height}, {target_width})")
        else:
            raise ValueError(f"Unknown infer_mode: {self.infer_mode}")

        # ========================================================================
        # Arguments: prompt, new_prompt, negative_prompt
        # ========================================================================
        if user_prompt is not None:

            if not isinstance(user_prompt, str):
                raise TypeError(f"`user_prompt` must be a string, but got {type(user_prompt)}")
            user_prompt = user_prompt.strip()
            prompt = user_prompt
        else:
            prompt = None

        if enhanced_prompt is not None:
            if not isinstance(enhanced_prompt, str):
                raise TypeError(f"`enhanced_prompt` must be a string, but got {type(enhanced_prompt)}")
            enhanced_prompt = enhanced_prompt.strip()
            prompt = enhanced_prompt
        

        # negative prompt
        if negative_prompt is None or negative_prompt == '':
            negative_prompt = self.default_negative_prompt
        if not isinstance(negative_prompt, str):
            raise TypeError(f"`negative_prompt` must be a string, but got {type(negative_prompt)}")

        # ========================================================================
        # Arguments: style. (A fixed argument. Don't Change it.)
        # ========================================================================
        if use_style_cond:
            # Only for hydit <= 1.1
            style = torch.as_tensor([0, 0] * batch_size, device=self.device)
        else:
            style = None

        # ========================================================================
        # Inner arguments: image_meta_size (Please refer to SDXL.)
        # ========================================================================
        if src_size_cond is None:
            size_cond = None
            image_meta_size = None
        else:
            # Only for hydit <= 1.1
            if isinstance(src_size_cond, int):
                src_size_cond = [src_size_cond, src_size_cond]
            if not isinstance(src_size_cond, (list, tuple)):
                raise TypeError(f"`src_size_cond` must be a list or tuple, but got {type(src_size_cond)}")
            if len(src_size_cond) != 2:
                raise ValueError(f"`src_size_cond` must be a tuple of 2 integers, but got {len(src_size_cond)}")
            size_cond = list(src_size_cond) + [target_width, target_height, 0, 0]
            image_meta_size = torch.as_tensor([size_cond] * 2 * batch_size, device=self.device)

        # ========================================================================
        start_time = time.time()
        logger.debug(f"""
                       prompt: {user_prompt}
              enhanced prompt: {enhanced_prompt}
                         seed: {seed}
              (height, width): {(target_height, target_width)}
              negative_prompt: {negative_prompt}
                   batch_size: {batch_size}
               guidance_scale: {guidance_scale}
                  infer_steps: {infer_steps}
              image_meta_size: {size_cond}
        """)
        reso = f'{target_height}x{target_width}'
        if reso in self.freqs_cis_img:
            freqs_cis_img = self.freqs_cis_img[reso]
        else:
            freqs_cis_img = self.calc_rope(target_height, target_width)

        if sampler is not None and sampler != self.sampler:
            self.pipeline, self.sampler = self.load_sampler(sampler)
        
        if is_get_embedding:
            text_states = self.pipeline(
                height=target_height,
                width=target_width,
                prompt=prompt,
                prompt_embeds = prompt_embeds,
                negative_prompt=negative_prompt,
                num_images_per_prompt=batch_size,
                guidance_scale=guidance_scale,
                num_inference_steps=infer_steps,
                image_meta_size=image_meta_size,
                style=style,
                return_dict=False,
                generator=generator,
                freqs_cis_img=freqs_cis_img,
                use_fp16=self.args.use_fp16,
                learn_sigma=self.args.learn_sigma,
                has_cfg=has_cfg,
                is_get_embedding = True,
                from_embedd = from_embedd
            )
            return text_states
        else:
            samples = self.pipeline(
                height=target_height,
                width=target_width,
                prompt=prompt,
                prompt_embeds = prompt_embeds,
                negative_prompt=negative_prompt,
                num_images_per_prompt=batch_size,
                guidance_scale=guidance_scale,
                num_inference_steps=infer_steps,
                image_meta_size=image_meta_size,
                style=style,
                return_dict=False,
                generator=generator,
                freqs_cis_img=freqs_cis_img,
                use_fp16=self.args.use_fp16,
                learn_sigma=self.args.learn_sigma,
                has_cfg=has_cfg,
                is_get_embedding = False,
                from_embedd = from_embedd
            )[0]
            gen_time = time.time() - start_time
            logger.debug(f"Success, time: {gen_time}")

            return {
                'images': samples,
                'seed': seed,
            }


class End2EndWithoutEncoder(object):
    def __init__(self, args, models_root_path):
        self.args = args

        # Check arguments
        t2i_root_path = Path(models_root_path) / "t2i"
        self.root = t2i_root_path
        logger.info(f"Got text-to-image model root path: {t2i_root_path}")
        
        # Set device and disable gradient
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_grad_enabled(False)
        # Disable BertModel logging checkpoint info
        tf_logger.setLevel('ERROR')

        # ========================================================================


        # ========================================================================
        logger.info(f"Loading VAE...")
        vae_path = self.root / "sdxl-vae-fp16-fix"
        self.vae = AutoencoderKL.from_pretrained(str(vae_path)).to(self.device)
        logger.info(f"Loading VAE finished")

        # ========================================================================
        # Create model structure and load the checkpoint
        logger.info(f"Building HunYuan-DiT model...")
        model_config = HUNYUAN_DIT_CONFIG[self.args.model]
        self.patch_size = model_config['patch_size']
        self.head_size = model_config['hidden_size'] // model_config['num_heads']
        self.resolutions, self.freqs_cis_img = self.standard_shapes()   # Used for TensorRT models
        self.image_size = _to_tuple(self.args.image_size)
        latent_size = (self.image_size[0] // 8, self.image_size[1] // 8)

        self.infer_mode = self.args.infer_mode
        if self.infer_mode in ['fa', 'torch']:
            # Build model structure
            self.model = HunYuanDiT(self.args,
                                    input_size=latent_size,
                                    **model_config,
                                    log_fn=logger.info,
                                    ).half().to(self.device)    # Force to use fp16

            # Load model checkpoint
            self.load_torch_weights()

            lora_ckpt = args.lora_ckpt
            if lora_ckpt is not None and lora_ckpt != "":
                logger.info(f"Loading Lora checkpoint {lora_ckpt}...")

                self.model.load_adapter(lora_ckpt)
                self.model.merge_and_unload()

            self.model.eval()
            logger.info(f"Loading torch model finished")
        elif self.infer_mode == 'trt':
            from .modules.trt.hcf_model import TRTModel

            trt_dir = self.root / "model_trt"
            engine_dir = trt_dir / "engine"
            plugin_path = trt_dir / "fmha_plugins/9.2_plugin_cuda11/fMHAPlugin.so"
            model_name = "model_onnx"

            logger.info(f"Loading TensorRT model {engine_dir}/{model_name}...")
            self.model = TRTModel(model_name=model_name,
                                  engine_dir=str(engine_dir),
                                  image_height=TRT_MAX_HEIGHT,
                                  image_width=TRT_MAX_WIDTH,
                                  text_maxlen=args.text_len,
                                  embedding_dim=args.text_states_dim,
                                  plugin_path=str(plugin_path),
                                  max_batch_size=TRT_MAX_BATCH_SIZE,
                                  )
            logger.info(f"Loading TensorRT model finished")
        else:
            raise ValueError(f"Unknown infer_mode: {self.infer_mode}")

        # ========================================================================
        # Build inference pipeline. We use a customized StableDiffusionPipeline.
        logger.info(f"Loading inference pipeline...")
        self.pipeline, self.sampler = self.load_sampler()
        logger.info(f'Loading pipeline finished')

        # ========================================================================
        self.default_negative_prompt = NEGATIVE_PROMPT
        logger.info("==================================================")
        logger.info(f"                Model is ready.                  ")
        logger.info("==================================================")

    def load_torch_weights(self):

        load_key = self.args.load_key
        if self.args.dit_weight is not None:
            dit_weight = Path(self.args.dit_weight)
            if dit_weight.is_dir():
                files = list(dit_weight.glob("*.pt"))
                if len(files) == 0:
                    raise ValueError(f"No model weights found in {dit_weight}")
                if str(files[0]).startswith('pytorch_model_'):
                    model_path = dit_weight / f"pytorch_model_{load_key}.pt"
                    bare_model = True
                elif any(str(f).endswith('_model_states.pt') for f in files):
                    files = [f for f in files if str(f).endswith('_model_states.pt')]
                    model_path = files[0]
                    if len(files) > 1:
                        logger.warning(f"Multiple model weights found in {dit_weight}, using {model_path}")
                    bare_model = False
                else:
                    raise ValueError(f"Invalid model path: {dit_weight} with unrecognized weight format: "
                                     f"{list(map(str, files))}. When given a directory as --dit-weight, only "
                                     f"`pytorch_model_*.pt`(provided by HunyuanDiT official) and "
                                     f"`*_model_states.pt`(saved by deepspeed) can be parsed. If you want to load a "
                                     f"specific weight file, please provide the full path to the file.")
            elif dit_weight.is_file():
                model_path = dit_weight
                bare_model = 'unknown'
            else:
                raise ValueError(f"Invalid model path: {dit_weight}")
        else:
            model_dir = self.root / "model"
            model_path = model_dir / f"pytorch_model_{load_key}.pt"
            bare_model = True

        if not model_path.exists():
            raise ValueError(f"model_path not exists: {model_path}")
        logger.info(f"Loading torch model {model_path}...")
        if model_path.suffix == '.safetensors':
            raise NotImplementedError(f"Loading safetensors is not supported yet.")
        else:
            # Assume it's a single weight file in the *.pt format.
            state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

        if bare_model == 'unknown' and ('ema' in state_dict or 'module' in state_dict):
            bare_model = False
        if bare_model is False:
            if load_key in state_dict:
                state_dict = state_dict[load_key]
            else:
                raise KeyError(f"Missing key: `{load_key}` in the checkpoint: {model_path}. The keys in the checkpoint "
                               f"are: {list(state_dict.keys())}.")

        if 'style_embedder.weight' in state_dict and not hasattr(self.model, 'style_embedder'):
            raise ValueError(f"You might be attempting to load the weights of HunYuanDiT version <= 1.1. You need "
                             f"to set `--use-style-cond --size-cond 1024 1024 --beta-end 0.03` to adapt to these weights."
                             f"Alternatively, you can use weights of version >= 1.2, which no longer depend on "
                             f"these two parameters.")
        if 'style_embedder.weight' not in state_dict and hasattr(self.model, 'style_embedder'):
            raise ValueError(f"You might be attempting to load the weights of HunYuanDiT version >= 1.2. You need "
                             f"to remove `--use-style-cond` and `--size-cond 1024 1024` to adapt to these weights.")


        # # 提取第一个 Linear 层的权重和偏置 (in_features=2048, out_features=8192)
        # linear1_weight = state_dict['mlp_t5.0.weight']
        # linear1_bias = state_dict['mlp_t5.0.bias']

        # # 提取第二个 Linear 层的权重和偏置 (in_features=8192, out_features=1024)
        # linear2_weight = state_dict['mlp_t5.2.weight']
        # linear2_bias = state_dict['mlp_t5.2.bias']

        # weights_to_save = {
        #     '0.weight': linear1_weight,
        #     '0.bias': linear1_bias,
        #     '2.weight': linear2_weight,
        #     '2.bias': linear2_bias
        # }

        # # 保存为 .pt 文件
        # torch.save(weights_to_save, '/data/xiaot/LLaVA/HunyuanDiT/ckpts/t2i/model/mlp_t5_weights.pt')
        # Don't set strict=False. Always explicitly check the state_dict.

        self.model.load_state_dict(state_dict, strict=True)
    
    def load_sampler(self, sampler=None):
        pipeline, sampler = get_pipeline_without_encoder(self.args,
                                         self.vae,
                                         self.model,
                                         device=self.device,
                                         rank=0,
                                         infer_mode=self.infer_mode,
                                         sampler=sampler,
                                         )
        return pipeline, sampler

    def calc_rope(self, height, width):
        th = height // 8 // self.patch_size
        tw = width // 8 // self.patch_size
        base_size = 512 // 8 // self.patch_size
        start, stop = get_fill_resize_and_crop((th, tw), base_size)
        sub_args = [start, stop, (th, tw)]
        rope = get_2d_rotary_pos_embed(self.head_size, *sub_args)
        return rope

    def standard_shapes(self):
        resolutions = ResolutionGroup()
        freqs_cis_img = {}
        for reso in resolutions.data:
            freqs_cis_img[str(reso)] = self.calc_rope(reso.height, reso.width)
        return resolutions, freqs_cis_img

    def predict(self,
                prompt_embeds = None,
                height=1024,
                width=1024,
                seed=None,
                enhanced_prompt=None,
                negative_prompt=None,
                infer_steps=100,
                guidance_scale=6,
                batch_size=1,
                src_size_cond=(1024, 1024),
                sampler=None,
                use_style_cond=False,
                has_cfg=True,
                is_get_embedding = False,
                from_embedd = None
                ):
        # ========================================================================
        # Arguments: seed
        # ========================================================================
        if seed is None:
            seed = random.randint(0, 1_000_000)
        if not isinstance(seed, int):
            raise TypeError(f"`seed` must be an integer, but got {type(seed)}")
        generator = set_seeds(seed, device=self.device)
        # ========================================================================
        # Arguments: target_width, target_height
        # ========================================================================
        if width <= 0 or height <= 0:
            raise ValueError(f"`height` and `width` must be positive integers, got height={height}, width={width}")
        logger.info(f"Input (height, width) = ({height}, {width})")
        if self.infer_mode in ['fa', 'torch']:
            # We must force height and width to align to 16 and to be an integer.
            target_height = int((height // 16) * 16)
            target_width = int((width // 16) * 16)
            logger.info(f"Align to 16: (height, width) = ({target_height}, {target_width})")
        elif self.infer_mode == 'trt':
            target_width, target_height = get_standard_shape(width, height)
            logger.info(f"Align to standard shape: (height, width) = ({target_height}, {target_width})")
        else:
            raise ValueError(f"Unknown infer_mode: {self.infer_mode}")

        # ========================================================================
        # Arguments: prompt, new_prompt, negative_prompt
        # ========================================================================
        if user_prompt is not None:

            if not isinstance(user_prompt, str):
                raise TypeError(f"`user_prompt` must be a string, but got {type(user_prompt)}")
            user_prompt = user_prompt.strip()
            prompt = user_prompt
        else:
            prompt = None

        if enhanced_prompt is not None:
            if not isinstance(enhanced_prompt, str):
                raise TypeError(f"`enhanced_prompt` must be a string, but got {type(enhanced_prompt)}")
            enhanced_prompt = enhanced_prompt.strip()
            prompt = enhanced_prompt
        

        # negative prompt
        if negative_prompt is None or negative_prompt == '':
            negative_prompt = self.default_negative_prompt
        if not isinstance(negative_prompt, str):
            raise TypeError(f"`negative_prompt` must be a string, but got {type(negative_prompt)}")

        # ========================================================================
        # Arguments: style. (A fixed argument. Don't Change it.)
        # ========================================================================
        if use_style_cond:
            # Only for hydit <= 1.1
            style = torch.as_tensor([0, 0] * batch_size, device=self.device)
        else:
            style = None

        # ========================================================================
        # Inner arguments: image_meta_size (Please refer to SDXL.)
        # ========================================================================
        if src_size_cond is None:
            size_cond = None
            image_meta_size = None
        else:
            # Only for hydit <= 1.1
            if isinstance(src_size_cond, int):
                src_size_cond = [src_size_cond, src_size_cond]
            if not isinstance(src_size_cond, (list, tuple)):
                raise TypeError(f"`src_size_cond` must be a list or tuple, but got {type(src_size_cond)}")
            if len(src_size_cond) != 2:
                raise ValueError(f"`src_size_cond` must be a tuple of 2 integers, but got {len(src_size_cond)}")
            size_cond = list(src_size_cond) + [target_width, target_height, 0, 0]
            image_meta_size = torch.as_tensor([size_cond] * 2 * batch_size, device=self.device)

        # ========================================================================
        start_time = time.time()
        logger.debug(f"""
                       prompt: {user_prompt}
              enhanced prompt: {enhanced_prompt}
                         seed: {seed}
              (height, width): {(target_height, target_width)}
              negative_prompt: {negative_prompt}
                   batch_size: {batch_size}
               guidance_scale: {guidance_scale}
                  infer_steps: {infer_steps}
              image_meta_size: {size_cond}
        """) 
        reso = f'{target_height}x{target_width}'
        if reso in self.freqs_cis_img:
            freqs_cis_img = self.freqs_cis_img[reso]
        else:
            freqs_cis_img = self.calc_rope(target_height, target_width)

        if sampler is not None and sampler != self.sampler:
            self.pipeline, self.sampler = self.load_sampler(sampler)
        
        if is_get_embedding:
            text_states = self.pipeline(
                height=target_height,
                width=target_width,
                prompt=prompt,
                prompt_embeds = prompt_embeds,
                negative_prompt=negative_prompt,
                num_images_per_prompt=batch_size,
                guidance_scale=guidance_scale,
                num_inference_steps=infer_steps,
                image_meta_size=image_meta_size,
                style=style,
                return_dict=False,
                generator=generator,
                freqs_cis_img=freqs_cis_img,
                use_fp16=self.args.use_fp16,
                learn_sigma=self.args.learn_sigma,
                has_cfg=has_cfg,
                is_get_embedding = True,
                from_embedd = from_embedd
            )
            return text_states
        else:
            samples = self.pipeline(
                height=target_height,
                width=target_width,
                prompt=prompt,
                prompt_embeds = prompt_embeds,
                negative_prompt=negative_prompt,
                num_images_per_prompt=batch_size,
                guidance_scale=guidance_scale,
                num_inference_steps=infer_steps,
                image_meta_size=image_meta_size,
                style=style,
                return_dict=False,
                generator=generator,
                freqs_cis_img=freqs_cis_img,
                use_fp16=self.args.use_fp16,
                learn_sigma=self.args.learn_sigma,
                has_cfg=has_cfg,
                is_get_embedding = False,
                from_embedd = from_embedd
            )[0]
            gen_time = time.time() - start_time
            logger.debug(f"Success, time: {gen_time}")

            return {
                'images': samples,
                'seed': seed,
            }
