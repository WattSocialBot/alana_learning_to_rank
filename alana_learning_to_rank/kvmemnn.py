#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
import json
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd

from .config import get_config, DEFAULT_CONFIG


class Kvmemnn(nn.Module):
    def __init__(self, opt, num_features, dict):
        super().__init__()
        self.lt = nn.Embedding(num_features, opt['embeddingsize'], 0,
                               sparse=True, max_norm=opt['embeddingnorm'])
        if not opt['tfidf']:
            dict = None
        self.encoder = Encoder(self.lt, dict)
        if not opt['share_embeddings']:
            self.lt2 = nn.Embedding(num_features, opt['embeddingsize'], 0,
                                   sparse=True, max_norm=opt['embeddingnorm'])
            self.encoder2 = Encoder(self.lt2, dict)
        else:
            self.encoder2 = self.encoder
        self.opt = opt
        self.softmax = nn.Softmax(dim=1)
        self.cosine = nn.CosineSimilarity()

        self.lin1 = nn.Linear(opt['embeddingsize'], opt['embeddingsize'], bias=False)
        self.lin2 = nn.Linear(opt['embeddingsize'], opt['embeddingsize'], bias=False)
        self.hops = 1
        self.lins = 0
        if 'hops' in opt:
            self.hops = opt['hops']
        if 'lins' in opt:
            self.lins = opt['lins']
        self.cosineEmbedding = True
        if opt['loss'] == 'nll':
            self.cosineEmbedding = False
            
    def forward(self, xs, mems, ys=None, cands=None):
        xs_enc = []
        xs_emb = self.encoder(xs)

        if len(mems) > 0 and self.hops > 0:
            mem_enc = []
            for m in mems:
                mem_enc.append(self.encoder(m))
            mem_enc.append(xs_emb)
            mems_enc = torch.cat(mem_enc)
            self.layer_mems = mems
            layer2 = self.cosine(xs_emb, mems_enc).unsqueeze(0)
            self.layer2 = layer2
            layer3 = self.softmax(layer2)
            self.layer3 = layer3
            lhs_emb = torch.mm(layer3, mems_enc)

            if self.lins > 0:
                lhs_emb = self.lin1(lhs_emb)
            if self.hops > 1:
                layer4 = self.cosine(lhs_emb, mems_enc).unsqueeze(0)
                layer5 = self.softmax(layer4)
                self.layer5 = layer5
                lhs_emb = torch.mm(layer5, mems_enc)
                if self.lins > 1:
                    lhs_emb = self.lin2(lhs_emb)
        else:
            if self.lins > 0:
                lhs_emb = self.lin1(xs_emb)
            else:
                lhs_emb = xs_emb
        if ys is not None:
            # training
            if self.cosineEmbedding:
                ys_enc = []
                xs_enc.append(lhs_emb)
                ys_enc.append(self.encoder2(ys))
                for c in cands:
                    xs_enc.append(lhs_emb)
                    c_emb = self.encoder2(c)
                    ys_enc.append(c_emb)
            else:
                xs_enc.append(lhs_emb.dot(self.encoder2(ys)))
                for c in cands:
                    c_emb = self.encoder2(c)
                    xs_enc.append(lhs_emb.dot(c_emb))
        else:
            # test
            if self.cosineEmbedding:
                ys_enc = []
                for c in cands:
                    xs_enc.append(lhs_emb)
                    c_emb = self.encoder2(c)
                    ys_enc.append(c_emb)
            else:
                for c in cands:
                    c_emb = self.encoder2(c)
                    xs_enc.append(lhs_emb.dot(c_emb))
        if self.cosineEmbedding:
            return torch.cat(xs_enc), torch.cat(ys_enc)
        else:
            return torch.cat(xs_enc)

        
class Encoder(nn.Module):
    def __init__(self, shared_lt, dict):
        super().__init__()
        self.lt = shared_lt
        if dict is not None:
            l = len(dict)
            freqs = torch.Tensor(l)
            for i in range(l):
                ind = dict.ind2tok[i]
                freq = dict.freq[ind]
                freqs[i] = 1.0 / (1.0 + math.log(1.0 + freq))
            self.freqs = freqs
        else:
            self.freqs = None

    def forward(self, xs):
        xs_emb = self.lt(xs)
        if self.freqs is not None:
            # tfidf embeddings
            l = xs.size(1)
            w = Variable(torch.Tensor(l))
            for i in range(l):
                w[i] = self.freqs[xs.data[0][i]]
            w = w.mul(1 / w.norm())
            xs_emb = xs_emb.squeeze(0).t().matmul(w.unsqueeze(1)).t()
        else:
            # basic embeddings (faster)
            xs_emb = xs_emb.mean(1)
        return xs_emb


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

    train_data, dev_data, test_data, rev_vocab = make_training_data(trainset,
                                                                    devset,
                                                                    testset,
                                                                    {},
                                                                    CONFIG)
    X, y, X_w = train_data
    X_dev, y_dev, X_dev_w = dev_data
    X_test, y_test, X_test_w = test_data

    if not os.path.exists(args.model_folder):
        os.makedirs(args.model_folder)

    save_vocabulary(rev_vocab, os.path.join(args.model_folder, 'rev_vocab'))
    with open(os.path.join(args.model_folder, 'config.json'), 'w') as config_out:
        json.dump(CONFIG, config_out)

    print('Training with config "{}" :'.format(args.config))
    print(json.dumps(CONFIG, indent=2))
    kv = Kvmemnn(CONFIG, 128, {})
