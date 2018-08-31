import json

DEFAULT_CONFIG = 'config.json'


def get_config(in_config_file):
    with open(in_config_file) as config_in:
        config = json.load(config_in)
    return config
