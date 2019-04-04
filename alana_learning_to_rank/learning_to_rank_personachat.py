from __future__ import print_function

import json
import random
import os
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

from .config import get_config, DEFAULT_CONFIG
from .data_utils import build_vocabulary, tokenize_utterance
from .learning_to_rank import (make_dataset,
                               load,
                               save_vocabulary,
                               create_model_personachat,
                               train,
                               evaluate)

random.seed(273)
np.random.seed(273)
tf.set_random_seed(273)

MODEL_FILENAME = 'learning_to_rank.ckpt'
CONFIG = get_config(os.path.join(os.path.dirname(__file__), DEFAULT_CONFIG))


def make_training_data(in_train, in_dev, in_test, in_sample_weight, in_config):
    utterances_tokenized = []
    for context_utterances in in_train.context:
        utterances_tokenized += [tokenize_utterance(utt, add_special_symbols=False, remove_stopwords=False)
                                 for utt in context_utterances]
    utterances_tokenized += list(map(lambda x: tokenize_utterance(x, add_special_symbols=False, remove_stopwords=False),
                                     in_train.answer))

    context_nes = []
    for ne_list in in_train.context_ne:
        context_nes += ne_list
    word_vocab, rev_word_vocab = build_vocabulary(in_train.context.values.tolist() + in_train.answer.values.tolist(),
                                                  max_size=in_config['max_vocab_size'] - in_config['max_ne_vocab_size'])
    ne_vocab, ne_rev_vocab = build_vocabulary(context_nes + in_train.answer_ne.values.tolist(),
                                              max_size=in_config['max_ne_vocab_size'],
                                              add_special_symbols=False)
    unified_vocab = list(set(word_vocab + ne_vocab))
    unified_rev_vocab = {word: index for index, word in enumerate(unified_vocab)}

    in_config['vocab_size'] = len(unified_rev_vocab)
    X, y, X_weight = make_dataset(in_train, unified_rev_vocab, in_config, use_sample_weights=in_sample_weight)
    X_dev, y_dev, X_dev_weight = make_dataset(in_dev,
                                              unified_rev_vocab,
                                              in_config,
                                              use_sample_weights=in_sample_weight)
    X_test, y_test, X_test_weight = make_dataset(in_test,
                                                 unified_rev_vocab,
                                                 in_config,
                                                 use_sample_weights=in_sample_weight)
    return ((X, y, X_weight),
            (X_dev, y_dev, X_dev_weight),
            (X_test, y_test, X_test_weight),
            unified_rev_vocab)


def build_argument_parser():
    result = argparse.ArgumentParser()
    result.add_argument('trainset')
    result.add_argument('devset')
    result.add_argument('testset')
    result.add_argument('model_folder')
    result.add_argument('--bot_sample_weight', action='store_true')
    result.add_argument('--config', default=os.path.join(os.path.dirname(__file__), DEFAULT_CONFIG))
    result.add_argument('--evaluate', action='store_true', default=False, help='Only evaluate a trained model')
    return result


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()
    trainset = pd.read_json(args.trainset).sample(frac=1).reset_index(drop=True)
    devset = pd.read_json(args.devset).sample(frac=1).reset_index(drop=True)
    testset = pd.read_json(args.testset).sample(frac=1).reset_index(drop=True)

    with tf.Session() as sess:
        if args.evaluate:
            model, config, _ = load(args.model_folder, sess)
            evaluate(model, testset, config)
        else:
            CONFIG = get_config(args.config)

            train_data, dev_data, test_data, rev_vocab = make_training_data(trainset,
                                                                            devset,
                                                                            testset,
                                                                            args.bot_sample_weight,
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
            model = create_model_personachat(**CONFIG)
            checkpoint_file = os.path.join(args.model_folder, MODEL_FILENAME)
            init = tf.global_variables_initializer()
            sess.run(init)
            train(model,
                  (X, y, X_w),
                  (X_dev, y_dev, X_dev_w),
                  (X_test, y_test, X_test_w),
                  checkpoint_file,
                  sess,
                  **CONFIG)
