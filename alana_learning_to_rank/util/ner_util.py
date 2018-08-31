import sys
import os
import json
from collections import defaultdict

import ner
from configobj import ConfigObj

NER_CONFIG_FILES = [
    'config_caseless.txt',
    'config_case_sensitive.txt'
]
NER_CONFIGS = [
    ConfigObj(os.path.join(os.path.dirname(__file__), 'Stanford_NER', config_file))
    for config_file in NER_CONFIG_FILES
]


class StanfordNERWrapper(object):
    def __init__(self):
        ner_configs = [
            ConfigObj(os.path.join(os.path.dirname(__file__), 'Stanford_NER', config_file))
            for config_file in NER_CONFIG_FILES
        ]
        self.ners = [
            ner.SocketNER(host=ner_config['NER_HOST'], port=int(ner_config['NER_PORT']))
            for ner_config in ner_configs
        ]

    def get_entities(self, in_string):
        all_values = set([])
        result = defaultdict(lambda: set([]))
        for ner_tagger in self.ners:
            entity_map = ner_tagger.get_entities(in_string)
            for entity_type, entities in entity_map.iteritems():
                new_values = filter(
                    lambda x: x not in all_values,
                    [entity.lower() for entity in entities]
                )
                all_values = all_values.union(new_values)
                result[entity_type] = result[entity_type].union(new_values)
        return {key: list(values) for key, values in result.iteritems()}


if __name__ == '__main__':
    print StanfordNERWrapper().get_entities(' '.join(sys.argv[1:]))
 
