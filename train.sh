export MODEL_PATH=$2
python3 -W ignore run.py train $1 -s $MODEL_PATH --include-package stt
