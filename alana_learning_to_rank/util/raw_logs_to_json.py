import json
import sys
from collections import deque
from operator import itemgetter
from os import path
import re

import pandas as pd

DEFAULT_CONTEXT_LENGTH = 3


def main(in_sessions, in_context_length):
    result = {
        'context': [],
        'answer': [],
        'context_ne': [],
        'answer_ne': [],
        'context_bots': [],
        'bot': [],
        'turn_number': [],
        'score': [],
        'timestamp': []
    }
    for session in in_sessions:
        if 'rating' not in session:
            continue
        context = deque([], maxlen=in_context_length)
        session_rating = session['rating']
        score_for_ranking = int(float(session_rating))
        partial_result = []
        for turn_index, turn in enumerate(session['dialogue']):
            if in_context_length == len(context) and turn['actor'] == 'system':
                answer = turn['utterance']
                result['context'].append(' '.join(map(itemgetter('utterance'), context)))
                result['answer'].append(answer)
                result['context_ne'].append(context[-1]['ne'])
                result['answer_ne'].append(turn['ne'])
                result['score'].append(score_for_ranking)
                result['context_bots'].append(' '.join(map(lambda x: x['bot'], context)))
                result['turn_number'].append(turn_index)
                result['bot'].append(turn['bot'])
                result['timestamp'].append(session['timestamp'].partition('T')[2].partition(':')[0])
            context.append(turn)
    return pd.DataFrame(result)


def load_dialogue_logs(in_stream):
    dialogues = []
    for line in in_stream:
        line = line.strip()
        if line.startswith('+++++++RATING:'):  # new dialogue
            rating = re.search(r'[0-9]', line).group(0)
            dialogues.append({'rating': int(rating), 'dialogue': []})
        elif line.startswith('=======session:'):  # session ID, time
            sessionID, time = re.search(
                'session: (.*)time: (.*)\=+', line).groups()
            dialogues[-1]['sessionID'] = sessionID
            time = time.rstrip('=')
        elif line.startswith('user '):
            dialogues[-1]['dialogue'].append(
                {
                    'actor': 'user',
                    'utterance': re.sub(r'^.*score\(1\): ', '', line),
                    'ne': [],
                    'bot': '0',
                    'timestamp': time
                }
            )
        elif line.startswith('system '):
            bot, reply = re.search(
                'bot: ([^-)]*).* score[^:]*: (.*)$', line).groups()
            dialogues[-1]['dialogue'].append(
                {
                    'actor': 'system',
                    'utterance': reply,
                    'ne': [],
                    'bot': bot,
                    'timestamp': time
                }
            )
    return dialogues


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'Usage: {} <rated dialogues file> <result file>' \
            ' [--context_length={}]'.format(
                path.basename(__file__),
                DEFAULT_CONTEXT_LENGTH
            )
        exit()
    with open(sys.argv[1]) as dialogues_in:
        rated_dialogues = load_dialogue_logs(dialogues_in)
    with open(sys.argv[2], 'w') as dialogues_out:
        json.dump(rated_dialogues, dialogues_out)

