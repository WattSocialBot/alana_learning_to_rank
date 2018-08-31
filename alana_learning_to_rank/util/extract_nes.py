import sys
from operator import itemgetter
import json
import subprocess
import time
from os import path

from ner_util import StanfordNERWrapper

ner = StanfordNERWrapper()


def extract_nes_batch(in_sequences):
    stanford_ner = None
    try:
        stanford_ner = subprocess.Popen(
            ' '.join([
                'java',
                '-mx1000m',
                '-cp Stanford_NER/stanford-ner-2016-10-31/stanford-ner-3.7.0.jar edu.stanford.nlp.ie.NERServer',
                '-loadClassifier Stanford_NER/stanford-ner-2016-10-31/classifiers/english.all.3class.distsim.crf.ser.gz',
                '-port 7878',
                '-outputFormat inlineXML'
            ]).split()
        )
        time.sleep(10)
        result = []
        for index, sequence in enumerate(in_sequences):
            result.append(extract_nes(sequence, 7878))
            if index % 100 == 0:
                print 'Processed {}/{} sequences'.format(
                    index + 1,
                    len(in_sequences)
                )
    finally:
        if stanford_ner is not None:
            stanford_ner.terminate()
    return result


def main(in_history_file, in_result_file):
    with open(in_history_file) as history_in:
        dialogue_history = json.load(history_in)
    utterances_flat = []

    for index, item in enumerate(dialogue_history):
        if index % 100 == 0:
            print 'Processed {}/{} dialogues'.format(
                index,
                len(dialogue_history)
            )
        for turn in item.get('dialogue', []):
            turn['ne'] = ner.get_entities(turn['utterance'])

    with open(in_result_file, 'w') as result_out:
        json.dump(dialogue_history, result_out)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'Usage: {} <dialogue history file> <result file>'.format(
            path.basename(__file__)
        )
        exit()
    main(sys.argv[1], sys.argv[2])

