#!/bin/bash
NER_FOLDER='stanford-ner-2018-02-27'
MODELS_FOLDER='stanford-english-corenlp-2018-02-27-models'
RUNNING=`ps ax | awk '/stanford-n[e]r/ { print $1 }'`
if [ -n "$RUNNING" ]; then
    echo "NER running ($RUNNING). Trying to kill and restart."
    kill -9 $RUNNING
    sleep 2
fi

./init_ner.sh
. ./config_caseless.txt
nohup java -mx1000m -cp $NER_FOLDER/stanford-ner.jar edu.stanford.nlp.ie.NERServer -loadClassifier $MODELS_FOLDER/edu/stanford/nlp/models/ner/english.conll.4class.caseless.distsim.crf.ser.gz -port $NER_PORT -outputFormat inlineXML > ner_caseless.log 2>&1 &

. ./config_case_sensitive.txt
nohup java -mx1000m -cp $NER_FOLDER/stanford-ner.jar edu.stanford.nlp.ie.NERServer -loadClassifier $MODELS_FOLDER/edu/stanford/nlp/models/ner/english.all.3class.distsim.crf.ser.gz -port $NER_PORT -outputFormat inlineXML > ner_case_sensitive.log 2>&1 &
