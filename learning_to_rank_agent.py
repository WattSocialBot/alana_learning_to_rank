import random
import copy

from alana_learning_to_rank.learning_to_rank import create_model

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent


class LearningToRankAgent(Agent):
    def __init__(self, opt, shared=None):
        # initialize defaults first
        super().__init__(opt, shared)

        if not shared:
            # this is not a shared instance of this class, so do full
            # initialization. if shared is set, only set up shared members.
            self.id = 'LearningToRank'
            self.observation = {'text': self.EOS, 'episode_done': True}
            self.dict = DictionaryAgent(opt)

            self.learning_to_rank_config = {'sentiment_features_number': 0,
                                            'max_context_turns': 10,
                                            'max_sequence_length': 60,
                                            'embedding_size': 256,
                                            'vocab_size': len(self.dict),
                                            'rnn_cell': 'GRUCell',
                                            'bidirectional': False,
                                            'dropout_prob': 0.3,
                                            'mlp_sizes': [32]}

            self.X, self.pred, self.y = create_model(**(self.learning_to_rank_config))

        self.episode_done = True

    def batchify(self, observations):
        """Convert a list of observations into input & target tensors."""
        # valid examples
        exs = [ex for ex in observations if 'text' in ex]
        # the indices of the valid (non-empty) tensors
        valid_inds = [i for i, ex in enumerate(observations) if 'text' in ex]

        # set up the input tensors
        batchsize = len(exs)
        # tokenize the text
        parsed = [self.dict.parse(ex['text']) for ex in exs]
        max_x_len = max([len(x) for x in parsed])
        xs = torch.LongTensor(batchsize, max_x_len).fill_(0)
        # pack the data to the right side of the tensor for this model
        for i, x in enumerate(parsed):
            offset = max_x_len - len(x)
            for j, idx in enumerate(x):
                xs[i][j + offset] = idx
        if self.use_cuda:
            xs = xs.cuda(async=True)
        xs = Variable(xs)
        if self.use_cuda:
            xs.cuda()

        # set up the target tensors
        ys = None
        if 'labels' in exs[0]:
            # randomly select one of the labels to update on, if multiple
            # append EOS to each label
            labels = [random.choice(ex['labels']) + ' ' + self.EOS for ex in exs]
            parsed = [self.dict.parse(y) for y in labels]
            max_y_len = max(len(y) for y in parsed)
            ys = torch.LongTensor(batchsize, max_y_len).fill_(0)
            for i, y in enumerate(parsed):
                for j, idx in enumerate(y):
                    ys[i][j] = idx
            if self.use_cuda:
                ys = ys.cuda(async=True)
            ys = Variable(ys)
            if self.use_cuda:
                ys.cuda()
        return xs, ys, valid_inds

    def predict(self, xs, ys=None):
        """Produce a prediction from our model. Update the model using the
        targets if available.
        """
        batchsize = len(xs)

        import pdb; pdb.set_trace()
        # first encode context
        xes = self.lt(xs.t())
        h0 = torch.zeros(self.num_layers, batchsize, self.hidden_size)
        if self.use_cuda:
            h0 = h0.cuda(async=True)
        h0 = Variable(h0)
        _output, hn = self.encoder(xes, h0)

        # next we use EOS as an input to kick off our decoder
        x = Variable(self.EOS_TENSOR)
        xe = self.lt(x).unsqueeze(1)
        xes = xe.expand(xe.size(0), batchsize, xe.size(2))

        # list of output tokens for each example in the batch
        output_lines = [[] for _ in range(batchsize)]

        if ys is not None:
            # update the model based on the labels
            self.zero_grad()
            loss = 0
            # keep track of longest label we've ever seen
            self.longest_label = max(self.longest_label, ys.size(1))
            for i in range(ys.size(1)):
                output, hn = self.decoder(xes, hn)
                preds, scores = self.hidden_to_idx(output, drop=True)
                y = ys.select(1, i)
                loss += self.criterion(scores, y)
                # use the true token as the next input instead of predicted
                # this produces a biased prediction but better training
                xes = self.lt(y).unsqueeze(0)
                for b in range(batchsize):
                    # convert the output scores to tokens
                    token = self.v2t([preds.data[b][0]])
                    output_lines[b].append(token)

            loss.backward()
            self.update_params()
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
        xs, ys, valid_inds = self.batchify(observations)

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
