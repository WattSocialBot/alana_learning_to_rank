from __future__ import print_function

import tensorflow as tf


def get_loss_function(in_preds,
                      in_labels,
                      in_sample_weights,
                      l2_coef=0.00):
    loss_main = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=in_labels, logits=in_preds)
    # loss_main = tf.losses.log_loss(in_labels, in_preds)
    # loss_main = tf.losses.mean_squared_error(in_labels, in_preds)
    loss_l2 = tf.reduce_sum([tf.nn.l2_loss(v)
                             for v in tf.trainable_variables()
                             if 'bias' not in v.name]) * l2_coef
    cost = tf.reduce_mean(tf.add(loss_main, loss_l2), name='cost')
    return cost


def batch_generator(data, batch_size, rotate=False):
    batch_start_idx = 0
    while True:
        if not rotate and data[0].shape[0] <= batch_start_idx:
            break
        batch = ([data_i[batch_start_idx: batch_start_idx + batch_size] for data_i in data])
        batch_start_idx += batch_size
        if rotate:
            batch_start_idx %= data[0].shape[0]
        yield batch

