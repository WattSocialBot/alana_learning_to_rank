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
from torch.autograd import Variable
import tensorflow as tf

from alana_learning_to_rank.util.training_utils import batch_generator

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ParlAI'))

from alana_learning_to_rank.kvmemnn import Kvmemnn
from .data_utils import build_vocabulary, tokenize_utterance, vectorize_sequence, vectorize_sequences, pad_3d_batch
from .config import get_config, DEFAULT_CONFIG


random.seed(273)
np.random.seed(273)
torch.manual_seed(273)

MODEL_FILENAME = 'learning_to_rank.ckpt'
CONFIG = get_config(os.path.join(os.path.dirname(__file__), DEFAULT_CONFIG))


def create_model(in_config):
    result = Kvmemnn(in_config, in_config['vocab_size'], None)
    if in_config['use_gpu']:
        result.cuda()
    return result


def long_tensor_type(use_gpu, **kwargs):
    return torch.cuda.LongTensor if use_gpu else torch.LongTensor


def float_tensor_type(use_gpu, **kwargs):
    return torch.cuda.FloatTensor if use_gpu else torch.FloatTensor


def train(in_model, in_trainset, in_vocab, batch_size, num_epochs, truncate, cosine_loss_margin, learning_rate, optimizer, **kwargs):
    long_tensor_t = long_tensor_type(**kwargs)
    float_tensor_t = float_tensor_type(**kwargs)
    opt = getattr(torch.optim, optimizer)(in_model.parameters(), lr=learning_rate)
    in_model.train()
    for epoch in range(num_epochs):
        batch_gen = batch_generator(in_trainset, batch_size=batch_size)
        print('Epoch {}'.format(epoch))
        for batch_idx, batch in enumerate(batch_gen):
            print('Processing batch {}\r'.format(batch_idx), end='')
            opt.zero_grad()
            xs, mems, ys, cands = batch

            xs = tf.keras.preprocessing.sequence.pad_sequences(xs, truncate)
            mems = pad_3d_batch(mems, truncate, **kwargs)
            cands = pad_3d_batch(cands, truncate, **kwargs)
            ys = tf.keras.preprocessing.sequence.pad_sequences(ys, truncate)
            xe, ye = in_model(Variable(long_tensor_t(xs)),
                              Variable(long_tensor_t(mems)),
                              Variable(long_tensor_t(ys)),
                              Variable(long_tensor_t(cands)))
            y = Variable(float_tensor_t(xe.shape[:-1]).fill_(-1.0))
            y[:,0]= 1
            emb_size = xe.shape[-1]
            loss = torch.nn.CosineEmbeddingLoss(margin=cosine_loss_margin, reduce='sum')(xe.view(-1, emb_size), ye.view(-1, emb_size), y.view(-1))
            loss.backward()
            opt.step()
    print('')


def make_dataset(in_table, in_rev_vocab, config, use_sample_weights=True):
    xs, mems, ys, cands = [], [], [], []

    all_cands = list(set(in_table.response))

    for context_i in in_table.context:
        context_i_tokenized = tokenize_utterance('\n'.join(context_i), remove_stopwords=False)
        xs.append(vectorize_sequence(context_i_tokenized, in_rev_vocab))

    for persona_i in in_table.persona:
        persona_i_tokenized = [tokenize_utterance(persona_i_j, remove_stopwords=False)
                               for persona_i_j in persona_i]
        mems.append(vectorize_sequences(persona_i_tokenized, in_rev_vocab))

    ys = vectorize_sequences([tokenize_utterance(y_i) for y_i in in_table.response], in_rev_vocab)

    for i in range(len(ys)):
        cands_i = [neg for neg in np.random.choice(all_cands, size=config['neg_number'])
                   if neg != in_table.iloc[i].response]
        cands.append(vectorize_sequences([tokenize_utterance(cand_i_j) for cand_i_j in cands_i], in_rev_vocab))
    return xs, mems, ys, cands


def make_training_data(in_train, in_dev, in_test, in_config):
    utterances_tokenized = []
    for context_utterances in in_train.context:
        utterances_tokenized += [
            tokenize_utterance(utt, add_special_symbols=True, remove_stopwords=False)
            for utt in context_utterances]
    utterances_tokenized += list(
        map(lambda x: tokenize_utterance(x, add_special_symbols=True, remove_stopwords=False),
            in_train.response))
    word_vocab, rev_word_vocab = build_vocabulary(utterances_tokenized, max_size=in_config['max_vocab_size'])

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
    gold_trainset = trainset[trainset.target == 1.0][:100]

    if not os.path.exists(args.model_folder):
        os.makedirs(args.model_folder)
    word_vocab, rev_word_vocab, (xs, mems, ys, cands) = make_training_data(gold_trainset, gold_trainset, gold_trainset, CONFIG)
    model = create_model(CONFIG)
    print('Training with config "{}" :'.format(args.config))
    print(json.dumps(CONFIG, indent=2))

    train(model, (xs, mems, ys, cands), word_vocab, **CONFIG)

