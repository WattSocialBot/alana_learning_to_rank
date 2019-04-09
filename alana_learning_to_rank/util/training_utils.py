from __future__ import print_function

import tensorflow as tf


def get_loss_function(in_preds,
                      in_labels,
                      in_sample_weights,
                      l2_coef=0.00):
    # loss_xe = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=in_labels, logits=in_preds)
    loss_main = tf.losses.mean_squared_error(in_labels, in_preds)
    loss_l2 = tf.reduce_sum([tf.nn.l2_loss(v)
                             for v in tf.trainable_variables()
                             if 'bias' not in v.name]) * l2_coef
    cost = tf.reduce_mean(tf.add(loss_main, loss_l2), name='cost')
    return cost


def batch_generator(X, y, sample_weights, batch_size, rotate=False):
    batch_start_idx = 0
    total_batches_number = y.shape[0] // batch_size
    batch_counter = 0
    loop_pred = True if rotate else batch_start_idx < y.shape[0]
    while loop_pred:
        if batch_counter % 100 == 0:
            print('\rProcessed {} out of {} batches'.format(batch_counter, total_batches_number), end='')
        batch = ([X_i[batch_start_idx: batch_start_idx + batch_size] for X_i in X],
                 y[batch_start_idx: batch_start_idx + batch_size],
                 sample_weights[batch_start_idx: batch_start_idx + batch_size])
        batch_start_idx += batch_size
        if rotate:
            batch_start_idx %= y.shape[0]
        batch_counter += 1
        yield batch
    print('\n')
