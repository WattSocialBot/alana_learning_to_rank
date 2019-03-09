import argparse
import random
from operator import itemgetter

import pandas as pd
import numpy as np
import tensorflow as tf

from learning_to_rank import load, make_dataset, predict
from config import get_config

random.seed(273)
np.random.seed(273)


def eval_accuracy(in_pred_true, in_pred_fake):
    accuracy = sum(map(lambda x: 0 < x, in_pred_true - in_pred_fake)) / float(len(in_pred_true))
    return accuracy


def main(in_model_folder, in_eval_set):
    with tf.Session() as sess:
        model, config, rev_vocab = load(in_model_folder, sess)

        eval_set = pd.read_json(in_eval_set)

        gold_qa_pairs = pd.DataFrame({'context': eval_set.context,
                                      'answer': eval_set.answer,
                                      'context_ne': eval_set.context_ne,
                                      'answer_ne': eval_set.answer_ne,
                                      'context_sentiment': eval_set.context_sentiment,
                                      'answer_sentiment': eval_set.answer_sentiment,
                                      'timestamp': eval_set.timestamp,
                                      'target': eval_set.length_target})
        fake_qa_pairs = pd.DataFrame({'context': eval_set.context,
                                      'answer': eval_set.fake_answer,
                                      'context_ne': eval_set.context_ne,
                                      'answer_ne': eval_set.fake_answer_ne,
                                      'context_sentiment': eval_set.context_sentiment,
                                      'answer_sentiment': eval_set.fake_answer_sentiment,
                                      'timestamp': eval_set.timestamp,
                                      'target': eval_set.length_target})
        data_true = make_dataset(gold_qa_pairs, rev_vocab, config, use_sample_weights=False)
        data_fake = make_dataset(fake_qa_pairs, rev_vocab, config, use_sample_weights=False)

        pred_true = predict(model, data_true, sess)
        pred_fake = predict(model, data_fake, sess)

        accuracy = eval_accuracy(pred_true, pred_fake)
        print 'Precision@1: {:.3f}'.format(accuracy)


def build_argument_parser():
    result = argparse.ArgumentParser()
    result.add_argument('model_folder')
    result.add_argument('eval_dataset')
    return result


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()
    main(args.model_folder, args.eval_dataset)

