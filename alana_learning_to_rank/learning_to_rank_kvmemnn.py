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

from alana_learning_to_rank.util.training_utils import batch_generator

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ParlAI'))

from alana_learning_to_rank.kvmemnn import Kvmemnn
from .data_utils import build_vocabulary, tokenize_utterance, vectorize_sequences
from .config import get_config, DEFAULT_CONFIG


random.seed(273)
np.random.seed(273)
torch.manual_seed(273)

MODEL_FILENAME = 'learning_to_rank.ckpt'
CONFIG = get_config(os.path.join(os.path.dirname(__file__), DEFAULT_CONFIG))


def create_model(in_config):
    result = Kvmemnn(in_config)
    return result


def train(in_model, in_trainset, in_vocab, batch_size, num_epochs, **kwargs):
    for epoch in range(num_epochs):
        batch_gen = batch_generator(in_trainset, batch_size=batch_size)
        print('Epoch {}'.format(epoch))
        for batch in batch_gen:
            xs, mems, ys, cands = batch
            in_model.forward(xs, mems, ys, cands)



def make_dataset(in_table, in_rev_vocab, config, use_sample_weights=True):
    xs, mems, ys, cands = [], [], [], []

    all_cands = set(in_table.response)

    for context_i in in_table.context:
        context_i_tokenized = [tokenize_utterance(ctx_i_j, remove_stopwords=False) for ctx_i_j in context_i]
        xs.append(vectorize_sequences(context_i_tokenized), in_rev_vocab)

    for persona_i in in_table.persona:
        persona_i_tokenized = [tokenize_utterance(persona_i_j, remove_stopwords=False)
                               for persona_i_j in persona_i]
        mems.append(vectorize_sequences(persona_i_tokenized), in_rev_vocab)

    ys = vectorize_sequences([tokenize_utterance(y_i) for y_i in in_table.y], in_rev_vocab)

    for i in range(len(ys)):
        cands_i = [neg for neg in np.random.choice(all_cands, size=config['neg_number'])
                   if neg != in_table.iloc[i].response]
        cands.append(vectorize_sequences([tokenize_utterance(cand_i_j) for cand_i_j in cands_i], in_rev_vocab))
    return xs, mems, ys, cands


def make_training_data(in_train, in_dev, in_test, in_config):
    utterances_tokenized = []
    for context_utterances in in_train.context:
        utterances_tokenized += [
            tokenize_utterance(utt, add_special_symbols=False, remove_stopwords=False)
            for utt in context_utterances]
    utterances_tokenized += list(
        map(lambda x: tokenize_utterance(x, add_special_symbols=False, remove_stopwords=False),
            in_train.response))

    word_vocab, rev_word_vocab = build_vocabulary(
        in_train.context.values.tolist() + in_train.response.values.tolist(),
        max_size=in_config['max_vocab_size'])

    in_config['vocab_size'] = len(rev_word_vocab)
    xs, mems, ys, cands = make_dataset(in_train, rev_word_vocab, in_config)
    return word_vocab, rev_word_vocab, (xs, mems, ys, cands)


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

    trainset = pd.read_json(args.trainset).sample(frac=1).reset_index(drop=True)
    devset = pd.read_json(args.devset).sample(frac=1).reset_index(drop=True)
    testset = pd.read_json(args.testset).sample(frac=1).reset_index(drop=True)

    if not os.path.exists(args.model_folder):
        os.makedirs(args.model_folder)

    word_vocab, rev_word_vocab, (xs, mems, ys, cands) = make_training_data(trainset, devset, testset, CONFIG)
    model = create_model(CONFIG)
    print('Training with config "{}" :'.format(args.config))
    print(json.dumps(CONFIG, indent=2))

    train(model, (xs, mems, ys, cands), vocab, **CONFIG)
