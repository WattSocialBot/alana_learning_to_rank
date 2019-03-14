from collections import defaultdict, deque
from operator import itemgetter
import time

import numpy as np
import nltk

nltk.download('stopwords')
STOP_LIST = nltk.corpus.stopwords.words('english')


def tokenize_utterance(in_utterance, add_special_symbols=True, remove_stopwords=True):
    utterance_tokenized = nltk.RegexpTokenizer('\w+').tokenize(in_utterance.lower())
    if remove_stopwords:
        utterance_tokenized = filter(lambda token: token not in STOP_LIST, utterance_tokenized)
    if add_special_symbols:
        utterance_tokenized = ['^'] + utterance_tokenized + ['$']
    return utterance_tokenized


def build_vocabulary(in_sequences, max_size=10000, max_ngram_length=1, add_special_symbols=True):
    vocabulary = defaultdict(lambda: 0)

    for seq in in_sequences:
        ngram_windows = [
            deque([], maxlen=length)
            for length in xrange(1, max_ngram_length + 1)
        ]
        for token in seq:
            for ngram_window in ngram_windows:
                ngram_window.append(token)
                if len(ngram_window) == max_ngram_length:
                    vocabulary[' '.join(ngram_window)] += 1
    if add_special_symbols:
        vocabulary['_PAD'] = 999999999  # hack
        vocabulary['_UNK'] = 999999998  # hack
    vocab = map(itemgetter(0), sorted(vocabulary.items(), key=itemgetter(1), reverse=True))[:max_size]
    rev_vocab = {word: index for index, word in enumerate(vocab)}
    return vocab, rev_vocab

def vectorize_sequences(in_sequences, in_rev_vocab):
    return [vectorize_sequence(seq, in_rev_vocab) for seq in in_sequences]


def vectorize_sequence(in_sequence, in_rev_vocab):
    unk_id = in_rev_vocab['_UNK']
    return [in_rev_vocab.get(word, unk_id) for word in in_sequence]
