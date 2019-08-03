export MODEL_PATH=$2
python3 -W ignore run.py train runs/$1/config.json -s $MODEL_PATH --include-package stt -r runs/$1 --file-friendly-logging
