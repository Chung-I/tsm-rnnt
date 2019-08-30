allennlp fine-tune -m runs/$1 -c experiments/phn_level.jsonnet \
  -s runs/$2 --include-package stt
