import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
import numpy as np
import yaml
import argparse

from transformers.models import opt

from PIL import Image
import dataset.data_utils as utils
import random
import json

class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def deserialize_vocab(src):
    with open(src) as f:
        d = json.load(f)
    vocab = Vocabulary()
    vocab.word2idx = d['word2idx']
    vocab.idx2word = d['idx2word']
    vocab.idx = d['idx']
    return vocab

class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    仅当使用RNN作为输入网络时使用这个数据集
    """

    def __init__(self, data_split, vocab, config):
        self.vocab = vocab
        self.loc = config.data['data_path']
        self.img_path = config.data['data_path'] + '/images/'

        # Captions
        self.captions = []
        self.maxlength = 0

        if data_split != 'test':
            with open(self.loc + '/%s_caps_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    self.captions.append(line.strip())

            self.images = []
            with open(self.loc + '/%s_filename_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    self.images.append(line.strip())
        else:
            with open(self.loc + '/%s_caps.txt' % data_split, 'rb') as f:
                for line in f:
                    self.captions.append(line.strip())

            self.images = []
            with open(self.loc + '/%s_filename.txt' % data_split, 'rb') as f:
                for line in f:
                    self.images.append(line.strip())

        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if len(self.images) != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        if data_split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((278, 278)),
                # transforms.RandomRotation(0, 90),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index // self.im_div
        caption = self.captions[index]

        vocab = self.vocab

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            caption.lower().decode('utf-8'))
        punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
        tokens = [k for k in tokens if k not in punctuations]
        tokens_UNK = [k if k in vocab.word2idx.keys() else '<unk>' for k in tokens]

        caption = []
        caption.extend([vocab(token) for token in tokens_UNK])
        caption = torch.LongTensor(caption)

        image = Image.open(self.img_path + str(self.images[img_id])[2:-1]).convert('RGB')
        image = self.transform(image)  # torch.Size([3, 256, 256])

        return image, caption, tokens_UNK, index, img_id

    def __len__(self):
        return self.length

def generate_random_samples(config):
    # load all anns
    caps = utils.load_from_txt(config.data['data_path']+'train_caps.txt')
    fnames = utils.load_from_txt(config.data['data_path']+'train_filename.txt')

    # merge
    assert len(caps) // 5 == len(fnames)
    all_infos = []
    for img_id in range(len(fnames)):
        cap_id = [img_id * 5 ,(img_id+1) * 5]
        all_infos.append([caps[cap_id[0]:cap_id[1]], fnames[img_id]])

    # shuffle
    random.shuffle(all_infos)

    # split_trainval
    percent = 0.8
    train_infos = all_infos[:int(len(all_infos)*percent)]
    val_infos = all_infos[int(len(all_infos)*percent):]

    # save to txt
    train_caps = []
    train_fnames = []
    for item in train_infos:
        for cap in item[0]:
            train_caps.append(cap)
        train_fnames.append(item[1])

    utils.log_to_txt(train_caps, config.data['data_path']+'train_caps_verify.txt',mode='w')
    utils.log_to_txt(train_fnames, config.data['data_path']+'train_filename_verify.txt',mode='w')

    val_caps = []
    val_fnames = []
    for item in val_infos:
        for cap in item[0]:
            val_caps.append(cap)
        val_fnames.append(item[1])
    utils.log_to_txt(val_caps, config.data['data_path']+'val_caps_verify.txt',mode='w')
    utils.log_to_txt(val_fnames, config.data['data_path']+'val_filename_verify.txt',mode='w')

    print("Generate random samples to {} complete.".format(config.data['data_path']))



class TestSet(data.Dataset):
    """
    Load precomputed captions and image features
    """

    def __init__(self,root_dir ,data_split, vocab=None,):
        self.vocab = vocab
        self.loc = root_dir
        self.img_path = root_dir + '/images/'

        # Captions
        self.captions = []
        self.maxlength = 0

        if data_split != 'test':
            with open(self.loc + '/%s_caps_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    self.captions.append(line.strip())

            self.images = []
            with open(self.loc + '/%s_filename_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    self.images.append(line.strip())
        else:
            with open(self.loc + '/%s_caps.txt' % data_split, 'rb') as f:
                for line in f:
                    self.captions.append(line.strip())

            self.images = []
            with open(self.loc + '/%s_filename.txt' % data_split, 'rb') as f:
                for line in f:
                    self.images.append(line.strip())

        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if len(self.images) != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        if data_split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((278, 278)),
                # transforms.RandomRotation(0, 90),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index // self.im_div
        caption = self.captions[index]

        vocab = self.vocab

        # Convert caption (string) to word ids.
        # tokens = nltk.tokenize.word_tokenize(
        #     caption.lower().decode('utf-8'))
        # punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
        # tokens = [k for k in tokens if k not in punctuations]
        # tokens_UNK = [k if k in vocab.word2idx.keys() else '<unk>' for k in tokens]

        # caption = []
        # caption.extend([vocab(token) for token in tokens_UNK])
        # caption = torch.LongTensor(caption)
        #
        image = Image.open(self.img_path + str(self.images[img_id])[2:-1]).convert('RGB')
        image = self.transform(image)  # torch.Size([3, 256, 256])

        # return image, caption, tokens_UNK, index, img_id
        return image, caption, index, img_id

    def __len__(self):
        return self.length

def generate_random_samples(config):
    # load all anns
    caps = utils.load_from_txt(config.data['data_path']+'train_caps.txt')
    fnames = utils.load_from_txt(config.data['data_path']+'train_filename.txt')

    # merge
    assert len(caps) // 5 == len(fnames)
    all_infos = []
    for img_id in range(len(fnames)):
        cap_id = [img_id * 5 ,(img_id+1) * 5]
        all_infos.append([caps[cap_id[0]:cap_id[1]], fnames[img_id]])

    # shuffle
    random.shuffle(all_infos)

    # split_trainval
    percent = 0.8
    train_infos = all_infos[:int(len(all_infos)*percent)]
    val_infos = all_infos[int(len(all_infos)*percent):]

    # save to txt
    train_caps = []
    train_fnames = []
    for item in train_infos:
        for cap in item[0]:
            train_caps.append(cap)
        train_fnames.append(item[1])

    utils.log_to_txt(train_caps, config.data['data_path']+'train_caps_verify.txt',mode='w')
    utils.log_to_txt(train_fnames, config.data['data_path']+'train_filename_verify.txt',mode='w')

    val_caps = []
    val_fnames = []
    for item in val_infos:
        for cap in item[0]:
            val_caps.append(cap)
        val_fnames.append(item[1])
    utils.log_to_txt(val_caps, config.data['data_path']+'val_caps_verify.txt',mode='w')
    utils.log_to_txt(val_fnames, config.data['data_path']+'val_filename_verify.txt',mode='w')

    print("Generate random samples to {} complete.".format(config.data['data_path']))

