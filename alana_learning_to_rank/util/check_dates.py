import json
import sys
from dateutil import parser

with open(sys.argv[1]) as hist_in:
    history = json.load(hist_in)
dates = set([])

for item in history:
    if 'dialogue' not in item or not len(item['dialogue']) or 'timestamp' not in item['dialogue'][0]:
        continue
    date = parser.parse(item['dialogue'][0]['timestamp'])
    dates.add((date.year, date.month, date.day))
for date in sorted(dates):
    print date
