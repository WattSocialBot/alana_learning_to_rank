import json
import sys
from os import path

import time
from scipy import stats
from boto3.dynamodb.types import Decimal

from aws_util.db import DynamoDBWrapper

DIALOGUE_HISTORY_DB_NAME = 'DialogueHistory'
EVENT_LOG_DB_NAME = 'event_log'


class number_str(float):
    def __init__(self, o):
        self.o = o

    def __repr__(self):
        return str(self.o)


def decimal_serializer(o):
    if isinstance(o, Decimal):
        return number_str(o)
    raise TypeError(repr(o) + " is not JSON serializable")


def iter_table_with_timeout(in_table_name):
    timeout_lower, timeout_upper = 0.0, 0.2
    timeout_mu, timeout_sigma = 0.1, 0.05
    timeout_random_source = stats.truncnorm(
        (timeout_lower - timeout_mu) / timeout_sigma,
        (timeout_upper - timeout_mu) / timeout_sigma,
        loc=timeout_mu,
        scale=timeout_sigma
    )
    result = []
    for item in DynamoDBWrapper(in_table_name).iteritems():
        timeout = timeout_random_source.rvs(1)[0]
        time.sleep(timeout)
        print >>sys.stderr, item
        yield item


def main():
    dialogue_history_temp = []
    for item in iter_table_with_timeout(DIALOGUE_HISTORY_DB_NAME):
        dialogue_history_temp.append(item)
    with open('DialogueHistory.json', 'w') as dialogue_history_temp_out:
        json.dump(dialogue_history_temp, dialogue_history_temp_out, default=decimal_serializer)

    event_log = []
    for item in iter_table_with_timeout(EVENT_LOG_DB_NAME):
        event_log.append(item)
    with open('event_log.json', 'w') as event_log_out:
        json.dump(event_log, event_log_out)


if __name__ == '__main__':
    main()

