export MODEL_PATH=$3
export VOCAB_PATH="phoneme"
allennlp make-vocab -s $VOCAB_PATH --include-package stt $1
python3 -W ignore run.py train $2 -s $MODEL_PATH --include-package stt
