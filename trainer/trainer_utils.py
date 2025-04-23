import json

from safetensors import torch
from loguru import logger
from torch import optim

from open_clip.model import CLIP

def get_model(args):
    model_name = args.model_name

    if model_name == 'RN50':
        model_config_path = './open_clip/model_configs/' + model_name + '.json'
        with open(model_config_path, 'r') as f:
            model_config = json.load(f)

        return CLIP(**model_config)
    else:
        raise NotImplementedError

def get_optimizer(args, model):
    """根据配置选择优化器"""
    optimizer_name = args.optimizer

    if optimizer_name == "sgd":
        logger.info('Using SGD Optimizer')
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
        )
    elif optimizer_name == "adam":
        logger.info('Using Adam Optimizer')
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
        )
    elif optimizer_name == "adamw":
        logger.info('Using AdamW Optimizer')

        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
        )
    elif optimizer_name == "rmsprop":
        logger.info('Using RMSprop Optimizer')

        optimizer = optim.RMSprop(
            model.parameters(),
            lr=args.lr,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    print(f"Using optimizer: {optimizer}")
    return optimizer
