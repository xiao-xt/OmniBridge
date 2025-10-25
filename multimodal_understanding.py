from typing import Any, Dict, List, Literal, Optional, Tuple, TypeVar, Union
from pathlib import Path

from tqdm import tqdm

import re

from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification,
                          AutoTokenizer, GenerationConfig, PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase)

from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
import torch

from qwen_vl_utils import process_vision_info

from  utils.qwen_util import patch_qwen_vl_utils


from utils.vision_utils import load_file, load_image

from omnibridge.modeling_qwen2_vl import Qwen2VLForConditionalGeneration

import pdb

# model_path = "/data2/xiaot/umug/checkpoints/Qwen2-VL-7B-Instruct-t2all-bitransformers-training-queried/v412-20250821-115859/checkpoint-800"
model_path = "/data/xiaot/arXiv/OmniBridge/save/omnibridge_retrieval_finetuned"

pretrain_dit = '/data/xiaot/Methods/VLLMs/UMUG/base/HunyuanDiT-v1.2'

is_image_generation_or_retrieval = True

image_path = "/data/xiaot/arXiv/OmniBridge/images/0.png"
question = "Please describe the image?"

is_deep_research = True

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Please describe the image"},
        ],
    }
]

from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(model_path, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28, trust_remote_code=True)
tokenizer = processor.tokenizer

model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, attn_implementation="flash_attention_2",device_map="auto")

# pdb.set_trace()

for name, p in model.named_parameters():
    if p.dtype != torch.bfloat16 and 'logit_scale' not in name:
        p.data = p.data.to(dtype=torch.bfloat16)


cur_device = model.device

user_prompt = """Generate an image description based on the question.
Then, provide a rationale to analyze the question.
Next, generate a step-by-step reasoning process to solve the problem. Ensure the steps are logical and concise.
Finally, provide a concise summary of the final answer in the following format: 'The final answer is: xxx.

Format your response with the following sections, separated by ###:
### Image Description:
### Rationales:
### Let's think step by step.
### Step 1:
### Step 2:
...
### The final answer is: 

{question}"""


if is_deep_research:
    messages[0]["content"][1]["text"] = user_prompt.format(question=messages[0]["content"][1]["text"])


# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

















# cur_inputs = model.tokenizer(input_text, return_attention_mask=True, return_tensors="pt")
# input_ids = cur_inputs.input_ids.to(cur_device)
# attention_mask = cur_inputs.attention_mask.to(cur_device)


# generate_kwargs={
#     "input_ids": input_ids,
#     "attention_mask":attention_mask,
#     "max_length":2048,
#     'return_dict_in_generate':True,
#     "output_hidden_states":True
# }

# generated_outputs = model.generate(**generate_kwargs)
