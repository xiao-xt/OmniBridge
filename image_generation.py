from typing import Any, Dict, List, Literal, Optional, Tuple, TypeVar, Union
from pathlib import Path

from tqdm import tqdm

import re

from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification,
                          AutoTokenizer, GenerationConfig, PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase)

from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
import torch


from swift.utils import get_dist_setting, get_env_args, get_logger


from  utils.qwen_util import patch_qwen_vl_utils


from utils.vision_utils import load_file, load_image


import torch
from omnibridge.modeling_qwen2_vl import Qwen2VLForConditionalGeneration

import pdb
cuda_device_map = "cuda:0"

model_path = "/data/xiaot/arXiv/OmniBridge/save/omnibridge_retrieval_finetuned"

pretrain_dit = '/data/xiaot/Methods/VLLMs/UMUG/base/HunyuanDiT-v1.2'

save_image_path = "/data/xiaot/arXiv/OmniBridge/images/"

cur_prompt = "A steaming basket of Goubuli buns on a tabletop, a high-definition photograph with a close-up shot."

is_using_learned_queries = True

is_image_generation_or_retrieval = True

from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(model_path, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28, trust_remote_code=True)
tokenizer = processor.tokenizer

model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, attn_implementation="flash_attention_2",device_map=cuda_device_map)
model.bilm = model.bilm.to(model.device)

if is_using_learned_queries:
    model.bilm.mask_random_number = 2
else:
    model.bilm.mask_random_number = 0

for name, p in model.named_parameters():
    if p.dtype != torch.bfloat16 and 'logit_scale' not in name:
        p.data = p.data.to(dtype=torch.bfloat16)

if is_image_generation_or_retrieval:
    model.initialize_embedder_dit_modules(
        pretrain_dit=pretrain_dit,
        is_image_generation_or_retrieval=is_image_generation_or_retrieval
    )
    model.initialize_dit_modules(
        pretrain_dit=pretrain_dit
    )

model.bilm.eval()

cur_device = model.device
user_prompt = f"Please rewrite the image caption of \"{cur_prompt}\". Generate a single-paragraph English image description (under 70 words) that professionally and concisely identifies: 1) Object types with exact counts. 2) Distinct color attributes bound to specific objects. 3) Clear spatial positioning. 4) Explicit property-object associations. Ensure visual unambiguity through absolute color assignments (e.g., '[color] [object]') and positional clarity (left/right/foreground/background). Exclude interpretations, metaphors, and subjective terms. Example structure: 'A [position] [color] [object] adjacent to a [position] [color] [object]...' with strict property binding. 5) There is no need to repeat an instruction. No additional output is required."
input_text = f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n'

cur_inputs = tokenizer(input_text, return_attention_mask=True, return_tensors="pt")
input_ids = cur_inputs.input_ids.to(cur_device)
attention_mask = cur_inputs.attention_mask.to(cur_device)


generate_kwargs={
    "input_ids": input_ids,
    "attention_mask":attention_mask,
    "max_length":2048,
    "max_new_tokens": 256,
    'return_dict_in_generate':True,
    "output_hidden_states":True
}

generated_outputs = model.generate(**generate_kwargs)

generated_ids = generated_outputs.sequences[:,generate_kwargs["input_ids"].size(-1)+1:-2]
cur_generate_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

if is_using_learned_queries:
    prompt_embeddings = torch.zeros(generate_kwargs["input_ids"].size(0), 256, 2048).to(cur_device)

else:
    dit_text_inputs = model.dit_tokenizer(
        cur_generate_text,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    dit_input_ids = dit_text_inputs.input_ids.to(cur_device)
    prompt_embeddings = model.dit_embedding(dit_input_ids).to(torch.bfloat16)

all_hidden_states = generated_outputs.hidden_states

last_hidden_states = []

for hidden_state in all_hidden_states:
    cur_hidden_state = hidden_state[-1].squeeze(0)[-1:]
    last_hidden_states.append(cur_hidden_state) 

last_hidden_states = last_hidden_states[1:-2]

qwen2_vl_hidden_states = torch.cat(last_hidden_states, dim=0).unsqueeze(0)

qwen_hidden_states = model.downsampling_linear_layer(qwen2_vl_hidden_states)

bilm_hidden_states = model.bilm(inputs_embeds = prompt_embeddings, encoder_hidden_states= qwen_hidden_states, encoder_attention_mask = torch.ones(qwen_hidden_states.size(1), ).unsqueeze(0).to(cur_device)).to(torch.float32)

height, width = model.dit_args.image_size

results = model.dit_model.predict(cur_generate_text,
                        height=height,
                        width=width,
                        seed=model.dit_args.seed,
                        enhanced_prompt=None,
                        negative_prompt=model.dit_args.negative,
                        infer_steps=100,
                        guidance_scale=model.dit_args.cfg_scale,
                        batch_size=model.dit_args.batch_size,
                        src_size_cond=model.dit_args.size_cond,
                        use_style_cond=model.dit_args.use_style_cond,
                        from_embedd = bilm_hidden_states,
                        )
images = results['images']

# Save images
save_dir = Path(save_image_path)
save_dir.mkdir(exist_ok=True)
# Find the first available index
all_files = list(save_dir.glob('*.png'))
if all_files:
    start = max([int(f.stem) for f in all_files]) + 1
else:
    start = 0

save_results = []
for idx, pil_img in enumerate(images):
    save_path = save_dir / f"{idx + start}.png"
    pil_img.save(save_path)
    print(f"Save to {save_path}")
    save_results.append(str(save_path))
