export VOCAB_PATH="data/vocabulary/ipa"
allennlp make-vocab -s $VOCAB_PATH --include-package stt $1

