export FISHER_PATH="data/fisher_callhome_spanish/translation"
export LEXICON_PATH="/home/nlpmaster/lexicon.txt"
export FISHER_TRAIN_PATH="data/fisher_callhome_spanish/s5/data/train/feats.scp"
export FISHER_VAL_PATH="data/fisher_callhome_spanish/s5/data/dev/feats.scp"
export FISHER_VOCAB_PATH="data/fisher_callhome_spanish/vocabulary"
python3 -W ignore run.py train $1 -s $2 --include-package stt
