import random
import copy

import numpy as np
import torch
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from alana_learning_to_rank.learning_to_rank import create_model
from alana_learning_to_rank.util.training_utils import get_loss_function

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent


class LearningToRankAgent(Agent):
    def __init__(self, opt, shared=None):
        # initialize defaults first
        super().__init__(opt, shared)
        self.sess = tf.Session()

        if not shared:
            # this is not a shared instance of this class, so do full
            # initialization. if shared is set, only set up shared members.
            self.id = 'LearningToRank'
            self.dict = DictionaryAgent(opt)
            self.EOS = self.dict.end_token
            self.observation = {'text': self.EOS, 'episode_done': True}
            self.learning_to_rank_config = {'max_context_turns': 10,
                                            'max_sequence_length': 60,
                                            'embedding_size': 256,
                                            'vocab_size': len(self.dict),
                                            'rnn_cell': 'GRUCell',
                                            'dropout_prob': 0.3,
                                            'mlp_sizes': [16],
                                            'l2_coef': 1e-5,
                                            'lr': 0.0001,
                                            'optimizer': 'AdamOptimizer',
                                            'answer_candidates_number': 20}

            self.X, self.pred, self.y = create_model(**(self.learning_to_rank_config))
        self.episode_done = True

    def batchify(self, observations):
        """Convert a list of observations into input & target tensors."""
        context_turns = [[] for _ in range(self.learning_to_rank_config['max_context_turns'])]
        answers = [[] for _ in range(len(observations[0]['label_candidates']))]
        y = []
        max_seq_len = self.learning_to_rank_config['max_sequence_length']
        for observation_i in observations:
            context_turns_i = list(map(self.parse, observation_i['text'].split('\n')))
            context_turns_i = pad_sequences(context_turns_i,
                                            maxlen=max_seq_len) 
            context_turns_i = pad_sequences([context_turns_i],
                                             maxlen=len(context_turns),
                                             value=np.zeros(max_seq_len))[0]
            for j, context_turn in enumerate(context_turns_i):
                context_turns[j].append(context_turn)
            answers_i = list(map(self.parse, observation_i['label_candidates']))
            answers_i = pad_sequences(answers_i,
                                      maxlen=max_seq_len) 
            answers_i = pad_sequences([answers_i],
                                      maxlen=len(answers),  
                                      value=np.zeros(max_seq_len))[0]
            for j, answer in enumerate(answers_i):
                answers[j].append(answer)
            y.append(observation_i['label_candidates'].index(observation_i['labels'][0]))
        X = list(map(np.array, [context_turns, answers]))
        y = np.array(y)
        return X, y

    def predict(self, xs, ys=None):
        """Produce a prediction from our model. Update the model using the
        targets if available.
        """
        batchsize = self.opt['batchsize'] 

        if ys is not None:
            # update the model based on the labels
            loss = 0
            # keep track of longest label we've ever seen
            sample_weights = np.expand_dims(np.ones(ys.shape[0]), -1)
            with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
                batch_sample_weight = tf.placeholder(tf.float32, [None, 1], name='sample_weight')

                # Define loss and optimizer
                loss_op = get_loss_function(self.pred, self.y, batch_sample_weight, l2_coef=self.learning_to_rank_config['l2_coef'])

                global_step = tf.Variable(0, trainable=False)
                self.sess.run(tf.assign(global_step, 0))
                learning_rate = tf.train.cosine_decay(self.learning_to_rank_config['lr'], global_step, 2000000, alpha=0.001)
                optimizer_class = getattr(tf.train, self.learning_to_rank_config['optimizer'])
                optimizer = optimizer_class(learning_rate=learning_rate)
                train_op = optimizer.minimize(loss_op, global_step)

                saver = tf.train.Saver(tf.global_variables())

                feed_dict = {X_i: batch_x_i for X_i, batch_x_i in zip(self.X, xs)}
                feed_dict.update({self.y: ys, batch_sample_weight: np.ones((batchsize, 1))})
                _, train_batch_loss = self.sess.run([train_op, loss_op], feed_dict=feed_dict)
            return ['I don\'t know'] * batchsize
        else:
            # just produce a prediction without training the model
            done = [False for _ in range(batchsize)]
            total_done = 0
            max_len = 0

            while(total_done < batchsize) and max_len < self.longest_label:
                # keep producing tokens until we hit EOS or max length for each
                # example in the batch
                output, hn = self.decoder(xes, hn)
                preds, scores = self.hidden_to_idx(output, drop=False)
                xes = self.lt(preds.t())
                max_len += 1
                for b in range(batchsize):
                    if not done[b]:
                        # only add more tokens for examples that aren't done yet
                        token = self.v2t(preds.data[b])
                        if token == self.EOS:
                            # if we produced EOS, we're done
                            done[b] = True
                            total_done += 1
                        else:
                            output_lines[b].append(token)

        return output_lines

    def parse(self, text):
        """Convert string to token indices."""
        return self.dict.txt2vec(text)

    def v2t(self, vec):
        """Convert token indices to string of tokens."""
        return self.dict.vec2txt(vec)

    def hidden_to_idx(self, hidden, drop=False):
        """Converts hidden state vectors into indices into the dictionary."""
        if hidden.size(0) > 1:
            raise RuntimeError('bad dimensions of tensor:', hidden)
        hidden = hidden.squeeze(0)
        scores = self.d2o(hidden)
        if drop:
            scores = self.dropout(scores)
        scores = self.softmax(scores)
        _max_score, idx = scores.max(1)
        return idx, scores

    def observe(self, observation):
        observation = copy.deepcopy(observation)
        if not self.episode_done:
            # if the last example wasn't the end of an episode, then we need to
            # recall what was said in that example
            prev_dialogue = self.observation['text']
            observation['text'] = prev_dialogue + '\n' + observation['text']
        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

    def act(self):
        # call batch_act with this batch of one
        return self.batch_act([self.observation])[0]

    def batch_act(self, observations):
        batchsize = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        # convert the observations into batches of inputs and targets
        # valid_inds tells us the indices of all valid examples
        # e.g. for input [{}, {'text': 'hello'}, {}, {}], valid_inds is [1]
        # since the other three elements had no 'text' field
        xs, ys = self.batchify(observations)

        if len(xs) == 0:
            # no valid examples, just return the empty responses we set up
            return batch_reply

        # produce predictions either way, but use the targets if available
        predictions = self.predict(xs, ys)

        for i in range(len(predictions)):
            # map the predictions back to non-empty examples in the batch
            # we join with spaces since we produce tokens one at a time
            batch_reply[valid_inds[i]]['text'] = ' '.join(
                c for c in predictions[i] if c != self.EOS)

        return batch_reply

