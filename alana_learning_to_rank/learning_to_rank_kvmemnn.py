from __future__ import print_function

import json
import random
import os
import argparse
from collections import deque
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ParlAI'))

from alana_learning_to_rank.kvmemnn import Kvmemnn
from .data_utils import build_vocabulary
from .config import get_config, DEFAULT_CONFIG


random.seed(273)
np.random.seed(273)
torch.manual_seed(273)

MODEL_FILENAME = 'learning_to_rank.ckpt'
CONFIG = get_config(os.path.join(os.path.dirname(__file__), DEFAULT_CONFIG))


def create_model(in_config):
    result = Kvmemnn(in_config)
    return result


def train(in_model, in_trainset, in_vocab, atch_size, num_epochs, max_history_length, **kwargs):
    for epoch in range(num_epochs):
        print('Epoch {}'.format(epoch))
        batch_q = deque([], maxlen=batch_size)
        for dialog in in_trainset:
            context_q = []
            for turn in dialog['turns']:
                if turn['agent'] == 'sys':
                    batch_q.append({'text': '\n'.join(dialog['persona_self'] + context_q[:-1]),
                                    'dialog': tokenize_sequences([dialog['persona_self'] + context_q[:-1]])
                                    'persona': [Variable(torch.LongTensor(tokenize_sequence(persona_i)).unsqueeze(0) for persona_i in persona],
                                    'mem': [Variable(torch.LongTensor(tokenize_sequence(persona_i)).unsqueeze(0) for persona_i in persona],
                                    'query': tokenize_sequences([dialog['persona_self'] + context_q[:-1]]),
                                    'labels': [turn['utterance']],
                                    'label_candidates': turn['response_cands']})
                context_q.append(turn['utterance'])
                if len(batch_q) == batch_size:
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
    CONFIG = get_config(args.config)

    with open(args.trainset) as trainset_in:
        trainset = json.load(trainset_in)
    dialogs_raw = []
    for dialog in trainset:
        dialogs_raw.append([turn['utterance'] for turn in dialog['turns']])
    vocab, rev_vocab = build_vocabulary(dialogs_raw, max_size=CONFIG['max_vocab_size'])
    trainset = pd.read_json(args.trainset).sample(frac=1).reset_index(drop=True)
    devset = pd.read_json(args.devset).sample(frac=1).reset_index(drop=True)
    testset = pd.read_json(args.testset).sample(frac=1).reset_index(drop=True)

    if not os.path.exists(args.model_folder):
        os.makedirs(args.model_folder)

    model = create_model(CONFIG)
    print('Training with config "{}" :'.format(args.config))
    print(json.dumps(CONFIG, indent=2))

    train(model, trainset, vocab, **CONFIG)
