from __future__ import print_function

import json
import random
import os
import argparse
from collections import deque

import numpy as np
import pandas as pd
import torch

from alana_learning_to_rank.kvmemnn_agent import KvmemnnAgent
from .config import get_config, DEFAULT_CONFIG


random.seed(273)
np.random.seed(273)
torch.manual_seed(273)

MODEL_FILENAME = 'learning_to_rank.ckpt'
CONFIG = get_config(os.path.join(os.path.dirname(__file__), DEFAULT_CONFIG))


def create_model(in_config):
    result = KvmemnnAgent(in_config)
    return result


def train(in_model, in_trainset, batchsize, num_epochs, max_history_length, **kwargs):
    for epoch in range(num_epochs):
        print('Epoch {}'.format(epoch))
        batch_q = deque([], maxlen=batchsize)
        for dialog in in_trainset:
            context_q = []
            for turn in dialog:
                if turn['agent'] == 'sys':
                    batch_q.append({'mem': turn['persona_self'] + context_q[:-1],
                                    'query': context_q[-1],
                                    'labels': [turn['response']],
                                    'label_candidates': turn['response_cands']})
                context_q.append(turn['utterance'])
                if len(batch_q) == batchsize:
                    in_model.batch_act(batch_q)
        in_model.batch_act(batch_q)


def build_argument_parser():
    result = argparse.ArgumentParser()
    result.add_argument('trainset')
    result.add_argument('devset')
    result.add_argument('testset')
    result.add_argument('model_folder')
    result.add_argument('--config', default=os.path.join(os.path.dirname(__file__), DEFAULT_CONFIG))
    result.add_argument('--evaluate', action='store_true', default=False, help='Only evaluate a trained model')
    return result


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()
    trainset = pd.read_json(args.trainset).sample(frac=1).reset_index(drop=True)
    devset = pd.read_json(args.devset).sample(frac=1).reset_index(drop=True)
    testset = pd.read_json(args.testset).sample(frac=1).reset_index(drop=True)

    if not os.path.exists(args.model_folder):
        os.makedirs(args.model_folder)

    model = create_model(CONFIG)
    print('Training with config "{}" :'.format(args.config))
    print(json.dumps(CONFIG, indent=2))

    train(model, trainset, **CONFIG)
