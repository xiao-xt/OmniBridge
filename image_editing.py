from typing import Any, Dict, List, Literal, Optional, Tuple, TypeVar, Union
from pathlib import Path

from tqdm import tqdm

import re

from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification,
                          AutoTokenizer, GenerationConfig, PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase)

from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
import torch

from PIL import Image

from qwen_vl_utils import fetch_image

from  utils.qwen_util import patch_qwen_vl_utils
from utils.vision_utils import load_file, load_image

from omnibridge.modeling_qwen2_vl import Qwen2VLForConditionalGeneration

import pdb
cuda_device_map = "cuda:0"
model_path = "/data/xiaot/arXiv/OmniBridge/save/omnibridge_retrieval_finetuned"



pretrain_dit = '/data/xiaot/Methods/VLLMs/UMUG/base/HunyuanDiT-v1.2'

save_image_path = "/data/xiaot/arXiv/OmniBridge/images/"


image_path = "/data/xiaot/arXiv/OmniBridge/images/0.png"
editing_prompt = "change the background of the image to a blue sky."

is_using_learned_queries = False

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

user_input = "<image>Please generate a new description (under 80 words) for the image based on the provided instruction. The description should reflect the changes as specified, without mentioning the instruction itself. Focus on accurately describing the updated visual content while ensuring clarity and detail. Do not reference the instruction used to modify the image in the final description.\n\n\nModification instruction: \n" + editing_prompt

if image_path is not None:
    image = Image.open(image_path)
    image = load_image(image)

inputs = {
      "user": user_input,
      "image": [image],
      "media_type": "images",
      'pixel_values': None,
      "pixel_values_videos": None
  }

for index, item in enumerate(inputs["image"]):
    inputs["image"][index] = fetch_image({'image': inputs["image"][index]})


media_inputs = processor.image_processor(
                    images=inputs["image"], videos=None, return_tensors='pt', do_resize=False)
image_grid_thw = media_inputs['image_grid_thw'].to(cur_device)
inputs.update(media_inputs)

inputs["user"] = inputs["user"].replace("<image>", '<|vision_start|><|image_pad|><|vision_end|>')
input_text = f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{inputs["user"]}<|im_end|>\n<|im_start|>assistant\n'
inputs["user"] = input_text

cur_inputs = processor.tokenizer(input_text, return_attention_mask=True, return_tensors="pt")

input_ids = cur_inputs.input_ids.squeeze(0).to(cur_device)
attention_mask = cur_inputs.attention_mask.to(cur_device)

image_token_id = 151655
idx_list = (input_ids == image_token_id).nonzero(as_tuple=True)[0].tolist()

if idx_list:
    idx_media_pairs = sorted(zip(idx_list, image_grid_thw), key=lambda x: x[0], reverse=True)
    merge_length = processor.image_processor.merge_size ** 2

    input_ids = input_ids.squeeze(0)

    for idx, media_grid in idx_media_pairs:
        token_len = (media_grid.prod() // merge_length)
        
        input_ids = torch.cat([
            input_ids[:idx],  
            torch.full((token_len,), image_token_id, dtype=input_ids.dtype, device=input_ids.device),  # 插入的 Token
            input_ids[idx + 1:]  
        ])
    input_ids = input_ids.unsqueeze(0).to(cur_device)

inputs_embeds = model.model.embed_tokens(input_ids)

pixel_values = inputs['pixel_values'].type(model.visual.get_dtype()).to(cur_device)
image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw.to(cur_device))
image_mask = (input_ids == model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

cur_attention_mask = torch.full_like(input_ids, 1).to(torch.long)
position_ids, _ = model.get_rope_index(input_ids, image_grid_thw, None, cur_attention_mask)
position_ids = position_ids.contiguous()

image_mask = input_ids == image_token_id

model_outputs = model.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=cur_attention_mask,
        past_key_values=None,
        inputs_embeds=inputs_embeds,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=True,
        return_dict=True,
    )

model_hidden_states = model_outputs[0]
mask = image_mask[0]
image_embed = model_hidden_states[0, mask].unsqueeze(0)

image_masks = torch.ones(image_embed.shape[0], image_embed.shape[1], dtype=torch.bool).to(model_hidden_states.device)

inputs = {
        "input_ids":input_ids,
        "attention_mask": cur_attention_mask,
        "image_grid_thw":image_grid_thw,
        "pixel_values":pixel_values
        
    }

generated_ids = model.generate(**inputs, max_new_tokens=512,)
generate_text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].split("assistant\n")[1]
edit_condition_text = editing_prompt

generate_text=generate_text.split("\n")[0]

if generate_text.startswith("I'm unable to see the image"):
    generate_text = edit_condition_text
else:
    edit_condition_text = generate_text

dit_text_inputs = model.dit_tokenizer(
                        edit_condition_text,
                        padding="max_length",
                        max_length=256,
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors="pt",
                    )

dit_input_ids = dit_text_inputs.input_ids.to(input_ids.device)

model.dit_embedding.to(input_ids.device)
dit_embeddings = model.dit_embedding(dit_input_ids)
qwen2vl_states = model.downsampling_linear_layer(image_embed)
dit_embeddings = dit_embeddings.to(torch.bfloat16)

bilm_hidden_states = model.bilm(
        inputs_embeds = dit_embeddings,
        encoder_hidden_states = qwen2vl_states,
        encoder_attention_mask = image_masks,
    )

height, width = model.dit_args.image_size

results = model.dit_model.predict(generate_text,
            height=height,
            width=width,
            seed=model.dit_args.seed,
            enhanced_prompt=None,
            negative_prompt=model.dit_args.negative,
            infer_steps=100,
            guidance_scale=model.dit_args.cfg_scale,
            batch_size=1,
            src_size_cond=model.dit_args.size_cond,
            use_style_cond=model.dit_args.use_style_cond,
            from_embedd = bilm_hidden_states,
            # from_embedd = None
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
