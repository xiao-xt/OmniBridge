"""
Adapted from https://github.com/pytorch/vision/blob/main/torchvision/datasets/flickr.py
Thanks to the authors of torchvision
"""
import os
from collections import defaultdict
from typing import Any, Callable, Optional, Tuple

from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision import transforms

class Flickr(VisionDataset):

    def __init__(
            self,
            root: str,
            ann_file: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.ann_file = os.path.expanduser(ann_file)
        data = defaultdict(list)

        if ann_file.endswith(".jsonl"):
            import json
            datas = []
            with open(ann_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)  # 每行就是一个 dict
                    except json.JSONDecodeError as e:
                        print(f"解析失败：{e}")
                        continue
                    # 在这里就可以按需处理 obj，比如：
                    # print(obj['某个键'])
                    datas.append(obj)

            for item in datas:
                img = item["image"][0]
                caption = item["messages"][1]["content"].split("</s>")[1]
                data[img].append(caption)
        else:

            with open(ann_file) as fd:
                fd.readline()
                for line in fd:
                    line = line.strip()
                    if line:
                        # some lines have comma in the caption, se we make sure we do the split correctly
                        img, caption = line.strip().split('.jpg,')
                        img = img + '.jpg'
                        data[img].append(caption)
        self.data = list(data.items())

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        img, captions = self.data[index]

        cur_img_id = img
        cur_img_id = os.path.join(self.root, img)

        # Image
        img = Image.open(os.path.join(self.root, img)).convert('RGB')
        # img = load_image(img)
        
        if self.transform is not None:
            img = self.transform(img)

        # transform = transforms.ToTensor()
        # img_tensor = transform(img)

        # Captions
        target = captions
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, cur_img_id

    def __len__(self) -> int:
        return len(self.data)


import re


def process_single_caption(caption, max_words=50):
    caption = re.sub(r"([.!\"()*#:;~])", ' ', caption.lower())
    caption = re.sub(r'\s{2,}', ' ', caption)
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[: max_words])
    return caption


def pre_caption(caption, max_words=50):
    if type(caption) == str:
        caption = process_single_caption(caption, max_words)
    else:
        caption = [process_single_caption(c, max_words) for c in caption]
    return caption
    

def build_dataset(dataset_name, root='root', transform=None, split='test', download=True, annotation_file=None,
                  language='en', task='zeroshot_classification', cupl=False, wds_cache_dir=None, **kwargs):
    ds = Flickr(root=f'{root}/', ann_file=annotation_file, transform=transform,
                           target_transform=pre_caption, **kwargs)
    return ds


def get_dataset_collate_fn(dataset_name, is_train=False):
    if dataset_name in ('flickr30k', 'flickr8k'):
        if is_train:
            return image_captions_collate_fn_train
        return image_captions_collate_fn


def image_captions_collate_fn_train(batch):
    
    transposed = list(zip(*batch))
    imgs = transposed[0]
    texts = transposed[1]
    imgs_ids = transposed[2]
    return imgs, texts, imgs_ids



def image_captions_collate_fn(batch):
    
    transposed = list(zip(*batch))
    imgs = transposed[0]
    texts = transposed[1]
    # imgs_ids = transposed[2]
    # return imgs, texts, imgs_ids

    return imgs, texts
