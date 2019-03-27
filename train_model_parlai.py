#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Train a model using parlai's standard training loop.
For documentation, see parlai.scripts.train_model.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'ParlAI'))
from parlai.scripts.train_model import TrainLoop, setup_args

if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(task='convai2:self',
                        evaltask='convai2:self',
                        model='learning_to_rank',
                        dict_file='learning_to_rank_model/vocab',
                        batchsize=1,
                        numthreads=1,
                        validation_every_n_epochs=1,
                        log_every_n_secs=60)

    opt = parser.parse_args()
    TrainLoop(opt).train()

