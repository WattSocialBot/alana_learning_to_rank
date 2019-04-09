import json
import sys
from collections import deque
import os
import argparse
import random
from operator import itemgetter

import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from .config import get_config, DEFAULT_CONFIG

random.seed(273)
np.random.seed(273)

CONFIG = get_config(os.path.join(os.path.dirname(__file__), DEFAULT_CONFIG))
SENT = SentimentIntensityAnalyzer()

import nltk
nltk.download('vader_lexicon')

def get_sentiment(in_utterance):
    sentiment_dict = SENT.polarity_scores(in_utterance)
    sentiment = list(map(itemgetter(1), sorted(sentiment_dict.items())))
    return sentiment


def process_dataset(in_json):
    sentiment_analyzer = SentimentIntensityAnalyzer()

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
                fake_response = random.choice(turn['response_cands'][:-1])
                context.append(list(context_queue))
                response.append(fake_response)
                persona.append(current_persona)
                target.append(0.0)
                c_sentiment.append(last_sentiment)
                a_sentiment.append(get_sentiment(fake_response))
            context_queue.append(turn['utterance'])
            last_sentiment = sentiment
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
    result.add_argument('--config', default=os.path.join(os.path.dirname(__file__), DEFAULT_CONFIG))
    return result


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()
    CONFIG = get_config(args.config)

    with open(args.dataset_file) as f_in:
        dataset_json = json.load(f_in)
    pd_dataset = process_dataset(dataset_json)
    pd_dataset.to_json(args.output_file)

