import torch
import json
import os

from tqdm import tqdm

if __name__ == '__main__':
    caption_dir = '/dataset/rsitmd_precomp/val_caps_verify.txt'
    with open(caption_dir, 'r') as f:
        captions = []
        for line in tqdm(f.readlines()):
            captions.append(line)

    print(len(captions))
    print(captions[0])
    print(captions[0])
