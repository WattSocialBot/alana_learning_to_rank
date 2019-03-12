#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Basic template of training loop.
We create an agent that will train on the training task, and be evaluated
on the validation version of the task.

We then do one iteration over ten training examples and one validation example,
printing reports from those tasks after completing those iterations.

This code is meant as a basic template: more advanced loops can iterate over
a validation dataset for exactly one epoch, can take in command-line arguments
using the argument parser in the core library, or generate a dictionary before
processing the data.
"""

import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'ParlAI'))

from parlai.core.params import ParlaiParser
from learning_to_rank_agent import LearningToRankAgent
from parlai.core.worlds import create_task


def main():
    # Get command line arguments
    parser = ParlaiParser()
    parser.add_argument('-n', '--num-iters', default=10, type=int)
    parser.add_argument('-a', '--num-agents', default=1, type=int)
    parser.add_argument('-hsz', '--hiddensize', default=256, type=int)
    parser.add_argument('-nl', '--numlayers', default=3, type=int)
    parser.add_argument('-lr', '--learningrate', default=0.001, type=float)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--no_cuda', default=False, type=bool)
    parser.add_argument('--history_size', default=10, type=int)
    parser.add_argument('--truncate', default=150, type=int)
    parser.add_argument('--rank_candidates', default=True, type=bool)
    opt = parser.parse_args()

    agents = []
    for _ in range(opt['num_agents']):
        agents.append(LearningToRankAgent(opt))

    opt['datatype'] = 'train'
    world_train = create_task(opt, agents)

    opt['datatype'] = 'valid'
    world_valid = create_task(opt, agents)

    start = time.time()
    # train / valid loop
    for _ in range(1):
        print('[ training ]')
        for _ in range(opt['num_iters']):  # train for a bit
            world_train.parley()

        print('[ training summary. ]')
        print(world_train.report())

        print('[ validating ]')
        for _ in range(1):  # check valid accuracy
            world_valid.parley()

        print('[ validation summary. ]')
        print(world_valid.report())

    print('finished in {} s'.format(round(time.time() - start, 2)))


if __name__ == '__main__':
    main()
