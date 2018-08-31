import json
import sys


def main(in_log_files, in_output_file):
    session_map = set([])
    logs_joined = []
    for log_file in in_log_files:
        with open(log_file) as log_in:
            history = json.load(log_in)
        for item in history:
            if item['sessionID'] not in session_map:
                logs_joined.append(item)
                session_map.add(item['sessionID'])
    with open(in_output_file, 'w') as logs_out:
        json.dump(logs_joined, logs_out)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'Usage: merge_logs.py <log_file(,log_file,...)> <output_file>'
        exit()
    main(sys.argv[1].split(','), sys.argv[2])

