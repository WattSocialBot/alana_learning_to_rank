#!/usr/bin/env sh

wget "http://parl.ai/downloads/personachat/personachat.tgz"

tar -xvzf personachat.tgz

mkdir -p data/personachat_json

for filename in `ls personachat`; do
  filenamenoext="${filename%.*}"
  python personachat2json.py "personachat/$filename" "data/personachat_json/$filenamenoext.json"
done

rm -rf personachat personachat.tgz