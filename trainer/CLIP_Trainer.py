import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from dataset.json_dataset import JsonDataset
from metric.clip_benchmark import dataloader_with_indices, image_captions_collate_fn
from tqdm import tqdm
from open_clip.tokenizer import tokenize


class CLIPTrainer:
    def __init__(self, model, args, train_set, val_set, test_set, optimizer, loss):
        self.model = model
        self.args = args
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.optimizer = optimizer
        self.loss = loss
        self.get_dataloader()



    def get_dataloader(self):
        self.train_loader = DataLoader(self.train_set,
                                       batch_size=self.args.batch_size,
                                       shuffle=True,
                                       num_workers=self.args.num_workers,
                                       collate_fn=image_captions_collate_fn)
        self.val_loader = DataLoader(self.val_set,
                                  batch_size=self.args.batch_size,
                                  shuffle=False,
                                  num_workers=self.args.num_workers,
                                 collate_fn=image_captions_collate_fn)
        self.test_loader = DataLoader(self.test_set,
                                   batch_size=self.args.batch_size,
                                   shuffle=False,
                                   num_workers=self.args.num_workers,
                                      collate_fn=image_captions_collate_fn)
        self.len_train_batch = len(self.train_loader)
        self.len_val_batch = len(self.val_loader)
        self.len_test_batch = len(self.test_loader)
        self.train_loader = dataloader_with_indices(self.train_loader)
        self.val_loader = dataloader_with_indices(self.val_loader)
        self.test_loader = dataloader_with_indices(self.test_loader)

    def train(self,):
        for epoch in range(self.args.epochs):
            self.train_one_epoch(epoch)

    def train_one_epoch(self, epoch):
        torch.cuda.empty_cache()
        for imgs, texts, img_id, labels, index in tqdm(self.train_loader, total=self.len_train_batch):
            batch_image = imgs.to(self.args.device)
            batch_text = tokenize([text for i, text in enumerate(texts)]).to(self.args.device)

            batch_image_features = self.model.encode_image(batch_image)
            batch_text_features = self.model.encode_text(batch_text)
            batch_image_features = F.normalize(batch_image_features,dim=-1)
            batch_text_features = F.normalize(batch_text_features,dim=-1)

            if self.args.loss_type == 'cliploss':
                tot_loss = self.loss(batch_image_features, batch_text_features)
            else:
                raise NotImplementedError("not implemented loss!")
            self.optimizer.zero_grad()
            tot_loss.backward()
            self.optimizer.step()

    def validate(self,):
        pass
    def test_one_epoch(self,):
        pass