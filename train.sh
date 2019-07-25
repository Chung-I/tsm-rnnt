export MODEL_PATH="phoneme"
#allennlp make-vocab -s $VOCAB_PATH --include-package stt $1
python3 -W ignore run.py train $1 -s $MODEL_PATH --include-package stt
