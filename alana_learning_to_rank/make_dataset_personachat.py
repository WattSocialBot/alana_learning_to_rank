import json

from collections import deque
import os
import argparse
import random
from operator import itemgetter

import pandas as pd
import numpy as np
import nltk
from nltk import sentiment
from .config import get_config, DEFAULT_CONFIG

random.seed(273)
np.random.seed(273)

nltk.download('vader_lexicon')

CONFIG = get_config(os.path.join(os.path.dirname(__file__), DEFAULT_CONFIG))
SENT = sentiment.vader.SentimentIntensityAnalyzer()


def get_sentiment(in_utterance):
    sentiment_dict = SENT.polarity_scores(in_utterance)
    sentiment = list(map(itemgetter(1), sorted(sentiment_dict.items())))
    return sentiment


def process_dataset(in_json, fake_responses_number, randomize_fake_responses):
    context, response, persona, target, c_sentiment, a_sentiment = [], [], [], [], [], []
    for dialog in in_json:
        context_queue = deque(maxlen=CONFIG['max_context_turns'])
        current_persona = [turn.strip() for turn in dialog['persona_self']]
        last_sentiment = None
        for turn in dialog['turns']:
            sentiment = get_sentiment(turn['utterance'])
            if turn['agent'] == 'sys':
                context.append(list(context_queue))
                response.append(turn['utterance'])
                persona.append(current_persona)
                target.append(1.0)
                c_sentiment.append(last_sentiment)
                a_sentiment.append(sentiment)
                fake_responses = \
                    np.random.choice(turn['response_cands'][:-1], fake_responses_number, replace=False) \
                    if randomize_fake_responses \
                    else turn['response_cands'][:min(fake_responses_number, 19)]
                for fake_response in fake_responses:
                    context.append(list(context_queue))
                    response.append(fake_response)
                    persona.append(current_persona)
                    target.append(0.0)
                    c_sentiment.append(last_sentiment)
                    a_sentiment.append(get_sentiment(fake_response))
            context_queue.append(turn['utterance'])
            last_sentiment = sentiment
    assert len(response) % (fake_responses_number + 1) == 0
    return pd.DataFrame({'context': context,
                         'response': response,
                         'persona': persona,
                         'target': target,
                         'c_sentiment': c_sentiment,
                         'a_sentiment': a_sentiment})


def build_argument_parser():
    result = argparse.ArgumentParser()
    result.add_argument('dataset_file')
    result.add_argument('output_file')
    result.add_argument('--fake_responses_number', default=1, type=int)
    result.add_argument('--randomize_fake_responses', action='store_true')
    result.add_argument('--config', default=os.path.join(os.path.dirname(__file__), DEFAULT_CONFIG))
    return result


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()
    CONFIG = get_config(args.config)

    with open(args.dataset_file) as f_in:
        dataset_json = json.load(f_in)
    pd_dataset = process_dataset(dataset_json, args.fake_responses_number, args.randomize_fake_responses)
    pd_dataset.to_json(args.output_file)

