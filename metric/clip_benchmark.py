import torch
from torch.utils.data import default_collate

def dataloader_with_indices(dataloader):
    '''
    https://github.com/LAION-AI/CLIP_benchmark/blob/main/clip_benchmark/metrics/zeroshot_retrieval.py
    '''
    start = 0
    for imgs, texts, img_id, labels in dataloader:
        end = start + len(imgs)
        inds = torch.arange(start, end)
        yield imgs, texts, img_id, labels, inds
        start = end

def image_captions_collate_fn(batch):
    transposed = list(zip(*batch))
    imgs = default_collate(transposed[0])
    texts = transposed[1]
    img_id = transposed[2]
    labels = transposed[3]
    return imgs, texts, img_id, labels
