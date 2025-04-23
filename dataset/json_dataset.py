from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import json
import numpy as np
from PIL import Image
import os
import torch
from dataset.HarmaDataUtils import pre_caption
from dataset.HarmaDataUtils import *
from dataset.reaugment import RandomAugment
from torchvision import transforms
from glob import glob
from loguru import logger

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(256, scale=(0.5, 1.0),
                                 interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                          'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
    transforms.ToTensor(),
    normalize,
])

test_transform = transforms.Compose([
    transforms.Resize((256,256), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    normalize,
])
def get_dataset(args):
    jsons = glob(os.path.join(args.json_dir, '*.json'))

    train_json = None
    val_json = None
    test_json = None
    for json in jsons:
        if 'train' in json:
            train_json = json
        elif 'val' in json:
            val_json = json
        elif 'test' in json:
            test_json = json
        else:
            pass
    assert train_json is not None
    assert val_json is not None
    assert test_json is not None
    logger.info(f'get train json file at {train_json}')
    logger.info(f'get val json file at {val_json}')
    logger.info(f'get test json file at {test_json}')
    train_set = JsonDataset([train_json],
                            transform=train_transform,
                            image_root=args.img_dir)
    val_set = JsonDataset([val_json],
                            transform=test_transform,
                            image_root=args.img_dir)
    test_set = JsonDataset([test_json],
                          transform=test_transform,
                          image_root=args.img_dir)
    return train_set, val_set, test_set
class JsonDataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.img_ids = {}

        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = pre_caption(ann['caption'], self.max_words)

        # t = analyse.extract_tags(caption, topK=4, withWeight=False)
        # ii = caption.split(' ')
        # k = ""
        # fl = 0
        # for j in range(len(ii)):
        #     if fl == 1:
        #         k += " "
        #     fl = 1
        #     if ii[j] not in t:
        #         k += "[MASK]"
        #     else:
        #         k += ii[j]
        #
        # mask_text = pre_caption(k, self.max_words)
        # print('caption: {}'.format(caption))
        # print('mask_texts: {}'.format(mask_texts))

        label = torch.tensor(ann['label'])

        ## if no need label, set value to zero or others:
        # label = 0
        # return image, caption, mask_text, self.img_ids[ann['image_id']], label
        return image, caption, self.img_ids[ann['image_id']], label

