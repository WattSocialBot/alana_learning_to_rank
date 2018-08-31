#!/usr/bin/env bash

if [ ! -d "stanford-ner-2018-02-27" ]; then
    wget https://nlp.stanford.edu/software/stanford-ner-2018-02-27.zip --no-check-certificate
    unzip stanford-ner-2018-02-27.zip
    rm stanford-ner-2018-02-27.zip
fi

if [ ! -d "stanford-english-corenlp-2018-02-27-models" ]; then
    wget http://nlp.stanford.edu/software/stanford-english-corenlp-2018-02-27-models.jar --no-check-certificate
    unzip stanford-english-corenlp-2018-02-27-models.jar -d stanford-english-corenlp-2018-02-27-models
    rm stanford-english-corenlp-2018-02-27-models.jar
fi
