import json
from collections import defaultdict
import argparse

import pandas as pd

CORRECTIONS_MAP = defaultdict(lambda: [])


def turn_to_be_filtered(in_turn, in_blacklist):
    for bot in in_blacklist:
        if bot in in_turn['actor']:
            return True
    return False


def filter_turns(in_history, in_bot_list):
    filtered_history = []
    for dialogue in in_history:
        if 'dialogue' not in dialogue:
            continue
        filtered_dialogue = dict(dialogue)
        turns = filtered_dialogue['dialogue']
        filtered_turns = []
        for turn_index, turn in enumerate(turns):
            if turn_to_be_filtered(turn, in_bot_list):
                continue
            if (
                turn_index != len(turns) - 1
                and turn_to_be_filtered(turns[turn_index + 1], in_bot_list)
            ):
                continue
            filtered_turns.append(turn)
        filtered_dialogue['dialogue'] = filtered_turns
        filtered_history.append(filtered_dialogue)
    return filtered_history


def load_corrections(in_file_name):
    global CORRECTIONS_MAP
    corrections_csv = pd.read_csv(in_file_name, delimiter=',')
    for index, row in corrections_csv.iterrows():
        c_id, turn_number = row['conversation_id'], int(row['turn_number'])
        CORRECTIONS_MAP[c_id].append(turn_number)


def main(
    in_history_file,
    in_events_file,
    in_ratings_file,
    in_result_file,
    in_mode,
    in_blacklist
):
    with open(in_history_file) as history_in:
        dialogue_history = json.load(history_in)
    with open(in_events_file) as events_in:
        event_log = json.load(events_in)
    dialogue_history = filter_turns(dialogue_history, in_blacklist)
    ratings = pd.read_csv(in_ratings_file, delimiter=',')
    session_id_to_conversation_id = {}
    for item in event_log:
        event_string = item['event']
        if type(event_string) != dict:
            event = json.loads(event_string)
        else:
            event = event_string
        session_id = event['session']['sessionId']
        conversation_id = event['request'].get('body', {}).get('conversationId', None)
        if not conversation_id:
            continue
        session_id_to_conversation_id[session_id] = conversation_id
    conversation_rating = {}
    for index, row in ratings.iterrows():
        if 0.0 < row['Rating']:  # not nan
            conversation_rating[row['Conversation ID']] = row['Rating']
    history_with_ratings = []
    for item in dialogue_history:
        conversation_id = session_id_to_conversation_id.get(item['sessionID'], None)
        if not conversation_id:
            continue
        rating = conversation_rating.get(conversation_id, None)
        if not rating:
            continue
        history_with_ratings.append(item)
        history_with_ratings[-1]['conversationID'] = conversation_id
        for turn in history_with_ratings[-1]['dialogue']:
            turn['rating'] = rating \
                if in_mode == 'rating' \
                else len(history_with_ratings[-1]['dialogue'])
        for turn_index in CORRECTIONS_MAP.get(conversation_id, []):
            history_with_ratings[-1]['dialogue'][turn_index - 1]['rating'] = 1.0
    with open(in_result_file, 'w') as result_out:
        json.dump(history_with_ratings, result_out)


def build_argument_parser():
    result = argparse.ArgumentParser()
    result.add_argument('history_file')
    result.add_argument('events_file')
    result.add_argument('ratings_file')
    result.add_argument('result_file')
    result.add_argument('mode', help='rating or length')
    result.add_argument('--corrections_file', default=None)
    result.add_argument('--bots_to_filter', default='quiz')
    return result


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()
    if args.corrections_file:
        load_corrections(args.corrections_file)
    main(
        args.history_file,
        args.events_file,
        args.ratings_file,
        args.result_file,
        args.mode,
        ','.split(args.bots_to_filter)
    )

