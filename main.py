import pdb

import torch
import argparse
import yaml
import json
import os

from loguru import logger
from datetime import datetime
from loguru import logger

from metric.tensorboard_logger import MyLogger
from open_clip import ClipLoss
from trainer.CLIP_Trainer import CLIPTrainer
from trainer.trainer_utils import get_model, get_optimizer
from dataset.json_dataset import get_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Model Parameter Config")

    parser.add_argument("--epoch", default=20, type=int)
    parser.add_argument("--lr", default=0.0001, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--device", default="cuda:0", type=str, help="Training device (cuda/cpu)")
    parser.add_argument("--optimizer", default="adam", type=str)

    parser.add_argument("--model_name", default="RN50",choices=["resnet50", "ViT-B32"],
                        type=str, help="Image Backbone for CLIP")

    parser.add_argument("--loss_type", default="cliploss",choices=["cliploss"], type=str, help="Configure loss")
    parser.add_argument("--loss_param", default=True, type=str, help="if loss function has a trainable param")

    parser.add_argument("--vocab_path", default='/home/dhm04/PycharmProjects/RSCR-baseline/vocab/rsitmd_splits_vocab.json',
                        help='go to vocab directory',type=str)
    parser.add_argument("--dataset", default='rsitmd', choices=["rsitmd", "RSITMD"], type=str)
    parser.add_argument("--num_workers", default=8, type=int)

    parser.add_argument("--json_dir", default='/home/dhm04/PycharmProjects/RSCR-baseline/dataset/AMNFNFinetune', type=str)
    parser.add_argument("--img_dir", default='/home/dhm04/PycharmProjects/RSCR-baseline/dataset/AMNFNFinetune/rsitmd/images', type=str)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    current_time = datetime.now()
    time_str = current_time.strftime("%Y%m%d_%H%M")
    os.makedirs('tensorboard_log', exist_ok=True)
    logger.add(
        'logs/{time}' + '-' + args.model_name + args.loss_type + '-' + args.dataset + '-' +
        args.optimizer + '.log',
        rotation='50 MB', level='DEBUG')

    logger.info(args)


    writer = MyLogger(logdir=os.path.join('tensorboard_log', time_str))

    train_set, val_set, test_set = get_dataset(args)


    # # ### GET TRAINER ###
    model = get_model(args)
    optmizer = get_optimizer(args, model)

    if args.loss_opt == True:
        if args.loss_type == 'cliploss':
            loss = ClipLoss()
            optmizer.param_groups.append({'params': loss.parameters()}) # CLIP loss has a trainable param
        else:
            raise NotImplementedError

    trainer = CLIPTrainer(model=model,
                          args=args,
                          train_set=train_set,
                          val_set=val_set,
                          test_set=test_set,
                          optimizer=optmizer)

    trainer.train_one_epoch(0)