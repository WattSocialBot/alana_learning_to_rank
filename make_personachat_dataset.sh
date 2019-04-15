#!/usr/bin/env sh

if [ $# -lt 4 ]; then
  echo "Usage: make_personachat_dataset.sh <train json> <dev json> <test json> <output folder>"
  return 1
fi

TRAIN=$1
DEV=$2
TEST=$3
DST_FOLDER=$4

mkdir -p $DST_FOLDER
python -m alana_learning_to_rank.make_dataset_personachat \
  $TRAIN \
  $DST_FOLDER/$(basename $TRAIN) \
  --config personachat.json \
  --fake_responses_number 1 \
  --randomize_fake_responses
python -m alana_learning_to_rank.make_dataset_personachat \
  $DEV \
  $DST_FOLDER/$(basename $DEV) \
  --config personachat.json \
  --fake_responses_number 19
python -m alana_learning_to_rank.make_dataset_personachat \
  $TEST \
  $DST_FOLDER/$(basename $TEST) \
  --config personachat.json \
  --fake_responses_number 19

