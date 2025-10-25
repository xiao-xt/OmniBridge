from contextlib import suppress

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from torchvision.transforms import ToPILImage, ToTensor
from torch.nn.utils.rnn import pad_sequence
from omnibridge.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from eval.flickr import build_dataset, get_dataset_collate_fn

import pdb

def pad_sequence_(sequences, padding_side='right', padding_value=0):
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


def dataloader_with_indices_train(dataloader):
    start = 0
    for x, y, z in dataloader:
        end = start + len(x)
        inds = torch.arange(start, end)
        yield x, y, z, inds
        start = end



def dataloader_with_indices(dataloader):
    start = 0
    for x, y in dataloader:
        end = start + len(x)
        inds = torch.arange(start, end)
        yield x, y, inds
        start = end

def evaluate(model, processor, dataloader, tokenizer, device, is_train=False, amp=True, recall_k_list=[5]):

    # list of batch of images embedding
    batch_images_emb_list = []
    # list of batch of text embedding
    batch_texts_emb_list = []
    # for each text, we collect the corresponding image index, as each image can have multiple corresponding texts
    texts_image_index = []

    backbone_image_embeds_list = []
    backbone_text_embeds_list = []
    vision_embeds_list = []

    text_embeds_list = []

    if is_train:
        dataloader = dataloader_with_indices_train(dataloader)
    else:
        dataloader = dataloader_with_indices(dataloader)
    
    autocast = torch.cuda.amp.autocast if amp else suppress

    input_text = f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n'
        # <|vision_start|><|image_pad|><|vision_end|>
    cur_device = model.device
    from qwen_vl_utils import fetch_image, fetch_video

    image_token_id = 151655

    model.tokenizer.padding_side = "right"

    # for batch_images, batch_texts, batch_image_ids, inds in tqdm(dataloader):
    for cur_data in tqdm(dataloader):
        if is_train:
            batch_images, batch_texts, batch_image_ids, inds = cur_data
            # 训练时的处理逻辑
        else:
            batch_images, batch_texts, inds = cur_data
        
        input_texts = [input_text for _ in range(len(batch_texts))]

        batch_images = [ToPILImage()(item) for item in batch_images]

        with torch.no_grad():
            inputs = processor(
                images=batch_images,
                text=input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            inputs = {k: v.cuda() for k, v in inputs.items()}

            mask = inputs['input_ids'] == 151643 
            batch_idx, seq_idx = mask.nonzero(as_tuple=True)

            cur_attention_mask = torch.full_like(inputs['input_ids'], 1).to(torch.long)
            position_ids, _ = model.get_rope_index(inputs['input_ids'], inputs['image_grid_thw'], None, cur_attention_mask)

            inputs_embeds = model.model.embed_tokens(inputs['input_ids'])
            pixel_values = inputs['pixel_values'].type(model.visual.get_dtype()).to(cur_device)
            image_embeds = model.visual(pixel_values, grid_thw=inputs['image_grid_thw'].to(cur_device))
            image_mask = (inputs['input_ids'] == model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)


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

            image_mask = (inputs['input_ids'] == image_token_id)

            image_first_index = (image_mask[0] == 1).nonzero(as_tuple=True)[0][0].item()

            image_mask = image_mask[:, image_first_index:, ]
            image_embeds = inputs_embeds[:,image_first_index:,:]
            batch_max_image_tokens = image_mask.sum(dim=1).max().item()
            image_mask = image_mask[:,:batch_max_image_tokens]
            image_embeds = image_embeds[:,:batch_max_image_tokens,:]

            image_mask_expanded = image_mask.unsqueeze(-1)
            image_part = model_hidden_states[:, image_first_index:, :]
            image_part = image_part[:,:batch_max_image_tokens,:]
            masked_image_part = image_part * image_mask_expanded.float().to(torch.bfloat16)


            bilm_retrieval_texts = [text for i, texts in enumerate(batch_texts) for text in texts]

            model.tokenizer.padding_side = "right"
            cur_texts = [f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n' for text in bilm_retrieval_texts]

            cur_inputs = model.tokenizer(cur_texts,return_tensors="pt",padding=True)
            pure_cur_inputs = model.tokenizer(bilm_retrieval_texts,return_tensors="pt",padding=True)
            model.tokenizer.padding_side = "left"
            cur_inputs = {k: v.to(image_mask.device) for k, v in cur_inputs.items()}
            pure_cur_inputs = {k: v.to(image_mask.device) for k, v in pure_cur_inputs.items()}

            batch_max_text_tokens = pure_cur_inputs['attention_mask'].sum(dim=1).max().item()

            text_model_outputs = model.model(
                            **cur_inputs,
                            output_hidden_states=True,
                            return_dict=True,
                        )
                        
            text_hidden_states = text_model_outputs[0]
            text_part = text_hidden_states[:,14:,:]

            text_mask = pure_cur_inputs['attention_mask'][:,:batch_max_text_tokens]
            text_part = text_part[:,:batch_max_text_tokens,:]

            
            text_mask_expanded = text_mask.unsqueeze(-1)
            masked_text_part = text_part * (text_mask_expanded.float()).to(torch.bfloat16)

            image_qwen2vl_states = model.downsampling_linear_layer(masked_image_part)

            text_qwen2vl_states = model.downsampling_linear_layer(masked_text_part)

            image_queries = model.image_queries.expand(image_qwen2vl_states.size(0), -1, -1).to(image_qwen2vl_states.device)
        
            text_queries = model.text_queries.expand(text_qwen2vl_states.size(0), -1, -1).to(image_qwen2vl_states.device)
            # pdb.set_trace()
            vision_ar_logits = model.bilm(
                inputs_embeds = image_queries,
                encoder_hidden_states = image_qwen2vl_states,
                encoder_attention_mask = image_mask,
                is_use_query_tokens = False,
            )

            language_ar_logits = model.bilm(
                    inputs_embeds = text_queries,
                    encoder_hidden_states = text_qwen2vl_states,
                    encoder_attention_mask = text_mask,
                    is_use_query_tokens = False,
                )

            backbone_image_embeds = model.image_proj_1(image_qwen2vl_states)
            backbone_text_embeds = model.text_proj_1(text_qwen2vl_states)

            vision_embeds = model.image_proj(vision_ar_logits)

            backbone_image_embeds_list.append(backbone_image_embeds.cpu())
            backbone_text_embeds_list.append(backbone_text_embeds.cpu())
            vision_embeds_list.append(vision_embeds.cpu())

            vision_embeds = F.normalize(vision_embeds, dim=-1)

            text_embeds = model.text_proj(language_ar_logits)
            text_embeds_list.append(text_embeds.cpu())


            text_embeds = F.normalize(text_embeds, dim=-1)

            backbone_image_embeds = F.normalize(backbone_image_embeds, dim=-1)
            backbone_text_embeds = F.normalize(backbone_text_embeds, dim=-1)
            
            alpha_image = torch.sigmoid(model.fusion_alpha_image).to(vision_embeds.device)
            alpha_text = torch.sigmoid(model.fusion_alpha_text).to(text_embeds.device)
            image_itc = alpha_image * vision_embeds + (1 - alpha_image) * backbone_image_embeds
            text_itc = alpha_text * text_embeds + (1 - alpha_text) * backbone_text_embeds

            image_itc_ = F.normalize(image_itc, dim=-1)
            text_itc_ = F.normalize(text_itc, dim=-1)

            image_itc_ = image_itc_.float()
            text_itc_ = text_itc_.float()

            logit_scale = model.logit_scale.exp()

            logits_per_image = logit_scale * (image_itc_ @ text_itc_.t())

            scores = logits_per_image.t()

            num_caps = 5
            ks=(1,5,10)
            recalls = {'i2t': {}, 't2i': {}}
            B, D = image_itc.shape
            lens = [len(texts_i) for texts_i in batch_texts]          # 每张图对应几个文本
            cur_texts_image_idx = torch.cat([
                torch.full((l,), i, dtype=torch.long, device=image_itc.device)
                for i, l in enumerate(lens)
            ], dim=0)  # [总文本数量]

            img_inds = torch.arange(B, device=image_itc.device)
            # import pdb
            # pdb.set_trace()
            if B > 10:
                for K in ks:
                    topk_inds = logits_per_image.topk(K, dim=1).indices     # [B, K]

                    hits = (cur_texts_image_idx[topk_inds] == img_inds.unsqueeze(1))  # [B, K] -> bool
                    # 只要有一个 True 就算命中：
                    r_at_k = hits.any(dim=1).float().mean().item()
                    recalls['i2t'][K] = r_at_k

                txt_inds = cur_texts_image_idx
                for K in ks:
                    topk_img = scores.topk(K, dim=1).indices    # [N_txt, K]
                    hits_t2i = (topk_img == txt_inds.unsqueeze(1))
                    recalls['t2i'][K] = hits_t2i.any(dim=1).float().mean().item()
                print(f"I2T R@1={recalls['i2t'][1]:.3f}, R@5={recalls['i2t'][5]:.3f}, R@10={recalls['i2t'][10]:.3f}")
                print(f"T2I R@1={recalls['t2i'][1]:.3f}, R@5={recalls['t2i'][5]:.3f}, R@10={recalls['t2i'][10]:.3f}")

        batch_texts_image_index = [ind for ind, texts in zip(inds, batch_texts) for text in texts]

        batch_images_emb_list.append(image_itc.cpu())
        batch_texts_emb_list.append(text_itc.cpu())
        texts_image_index.extend(batch_texts_image_index)

    batch_size = len(batch_images_emb_list[0])


    vision_embeds = torch.cat(vision_embeds_list)
    text_embeds = torch.cat(text_embeds_list)
    backbone_image_embeds = torch.cat(backbone_image_embeds_list)
    backbone_text_embeds = torch.cat(backbone_text_embeds_list)


    vision_embeds = F.normalize(vision_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    backbone_image_embeds = F.normalize(backbone_image_embeds, dim=-1)
    backbone_text_embeds = F.normalize(backbone_text_embeds, dim=-1)
    
    alpha_image = torch.sigmoid(model.fusion_alpha_image).to(vision_embeds.device)
    alpha_text = torch.sigmoid(model.fusion_alpha_text).to(text_embeds.device)
    images_emb = alpha_image * vision_embeds + (1 - alpha_image) * backbone_image_embeds
    texts_emb = alpha_text * text_embeds + (1 - alpha_text) * backbone_text_embeds


    images_emb = F.normalize(images_emb, dim=-1)
    texts_emb = F.normalize(texts_emb, dim=-1)

    logit_scale = model.logit_scale.exp().cpu()

    logits_per_image = logit_scale * (images_emb @ texts_emb.t())

    scores = logits_per_image.t()

    #  Compute R@1
    top1 = torch.argmax(logits_per_image, dim=1)     # [N_img]
    texts_image_index_ = torch.tensor(texts_image_index, dtype=torch.long)  # [N_txt]
    img_inds = torch.arange(images_emb.size(0), device=top1.device)
    r1 = (texts_image_index_[top1] == img_inds).float().mean().item()

    #  Compute R@5
    top5 = torch.topk(logits_per_image, k=5, dim=1).indices    # [N_img,5]
    correct5 = (texts_image_index_[top5] == img_inds.unsqueeze(1))
    r5 = correct5.any(dim=1).float().mean().item()

    #  Compute R@10
    top10 = torch.topk(logits_per_image, k=10, dim=1).indices    # [N_img,5]
    correct10 = (texts_image_index_[top10] == img_inds.unsqueeze(1))
    r10 = correct10.any(dim=1).float().mean().item()

    print(f"ALL Image→Text    R@1={r1:.3f}, R@5={r5:.3f}, R@10={r10:.3f}")


    txt2img = texts_image_index_                      # [N_txt]

    #  Recall@1
    top1 = torch.argmax(scores, dim=1)      # [N_txt]
    r1   = (top1 == txt2img).float().mean().item()

    #  Recall@5
    top5 = torch.topk(scores, k=5, dim=1).indices  # [N_txt,5]
    #    看这 5 张图里有没有正确的那张
    hits5 = (top5 == txt2img.unsqueeze(1))          # [N_txt,5] bool
    r5   = hits5.any(dim=1).float().mean().item()

    #  Recall@10
    top10 = torch.topk(scores, k=10, dim=1).indices  # [N_txt,10]
    hits10 = (top10 == txt2img.unsqueeze(1))      # [N_txt,10]
    r10    = hits10.any(dim=1).float().mean().item()

    print(f"ALL Text→Image  R@1={r1:.3f}, R@5={r5:.3f}, R@10={r10:.3f}")

    logit_scale = model.logit_scale.exp().cpu()

    logits_per_text = logit_scale * (texts_emb @ texts_emb.t())

    logits_per_text.fill_diagonal_(-torch.inf)

    gt_t2t = (texts_image_index_.unsqueeze(1) == texts_image_index_.unsqueeze(0))

 
    ks = [1, 5, 10]
    recalls_t2t = {}

    num_queries = texts_emb.size(0)
    query_indices = torch.arange(num_queries).unsqueeze(1)

    for k in ks:
        topk_indices = torch.topk(logits_per_text, k=k, dim=1).indices  # [N_txt, k]

        hits = gt_t2t[query_indices, topk_indices]

        recall_at_k = hits.any(dim=1).float().mean().item()
        recalls_t2t[k] = recall_at_k

    print(f"ALL Text→Text    R@1={recalls_t2t[1]:.3f}, R@5={recalls_t2t[5]:.3f}, R@10={recalls_t2t[10]:.3f}")





def main():
    models = []
    datasets = ["flickr30k"]
    languages = ["en"]

    models = ""
    dataset_name = "flickr30k"
    languages = "en"
    task = "zeroshot_retrieval"
    batch_size = 64
    num_workers = 8
    recall_k = [1, 5, 10]
    device = "cuda:0"
    amp = True

    model_path = "/data2/xiaot/umug/checkpoints/Qwen2-VL-7B-Instruct-t2all-bitransformers-training-queried/v413-20250821-161210/checkpoint-1000"            
    is_omnibridge = True

    cuda_device_map = "cuda:0"

    dataset_root = "/data2/xiaot/umug/data/images/flickr30k_EN"
    dataset_root = "/data2/xiaot/umug/data/images/flickr30k_en"
    # dataset_root = "/data2/xiaot/umug/data/images"

    model, transform, collate_fn, dataloader = None, None, None, None
    tokenizer = None

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28, trust_remote_code=True)
    tokenizer = processor.tokenizer

    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, attn_implementation="flash_attention_2",device_map=cuda_device_map)
    model.tokenizer = processor.tokenizer
    model.bilm = model.bilm.to(model.device)

    for name, p in model.named_parameters():
        if p.dtype != torch.bfloat16 and 'logit_scale' not in name:
            p.data = p.data.to(dtype=torch.bfloat16)

    from torchvision import transforms
    transform = transforms.ToTensor()

    dataset = build_dataset(
            dataset_name=dataset_name,
            root=dataset_root,
            transform=transform,
            split="test",
            annotation_file="/data/xiaot/Methods/VLLMs/EmbAR/data/flickr30k/flickr30k_test_karpathy.txt",
            download=True,
            language=languages,
            task=task,
            cupl=False,
            wds_cache_dir=None,
        )
    is_train = False
    collate_fn = get_dataset_collate_fn(dataset_name, is_train=is_train)

    dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size,
                shuffle=False, num_workers=num_workers,
                collate_fn=collate_fn
            )
    # pdb.set_trace()
    if is_omnibridge:
        evaluate(
            model,
            processor,
            dataloader,
            tokenizer,
            is_train=is_train,
            recall_k_list=recall_k,
            device=device,
            amp=amp
        )


if __name__ == '__main__':

    main()
