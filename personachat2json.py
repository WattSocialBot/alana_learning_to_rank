import json
import sys

def process_file(in_file):
    dialogs = []
    for line in in_file:
        if not len(line):
            continue
        idx, _, turn = line.partition(' ')
        if int(idx) == 1:
            dialogs.append({'persona_self': [], 'persona_other': [], 'turns': []})
            continue
        if turn.startswith('your persona:'):
            _, _, persona_turn = turn.partition('your persona: ')
            dialogs[-1]['persona_self'].append(persona_turn)
            continue
        if turn.startswith('partner\'s persona:'):
            _, _, persona_turn = turn.partition('partner\'s persona: ')
            dialogs[-1]['persona_other'].append(persona_turn)
            continue
        context, response, _, response_cands = turn.split('\t')
        dialogs[-1]['turns'].append({'agent': 'usr', 'utterance': context})
        dialogs[-1]['turns'].append({'agent': 'sys', 'utterance': response, 'response_cands': response_cands.split('|')})
    return dialogs


if __name__ == '__main__':
    input_file, output_file = sys.argv[1:3]
    with open(input_file) as f_in:
        dialogs_json = process_file(f_in)
    with open(output_file, 'w') as f_out:
        json.dump(dialogs_json, f_out)
