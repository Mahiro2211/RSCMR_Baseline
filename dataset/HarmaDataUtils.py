import re
import json
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from tqdm import tqdm

# from utils.hdfs_io import hexists, hcopy, hopen
# from vqaTools.vqaEval import VQAEval
# from refTools.evaluation.refEvaluation import RefEvaluation


def pre_question(question, max_ques_words):
    question = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        question.lower(),
    ).replace('-', ' ').replace('/', ' ')
    question = question.rstrip(' ')

    # truncate question
    question_words = question.split(' ')
    if len(question_words) > max_ques_words:
        question = ' '.join(question_words[:max_ques_words])

    return question


def pre_caption(caption, max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])
    # print(caption)
    if not len(caption):
        # print('=========')
        # print(caption)
        # print('=========')
        raise ValueError("pre_caption yields invalid text")

    return caption
