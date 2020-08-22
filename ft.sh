ARCHIVE_DIR="runs/uttcmvn-pure-attn-2"
VOCAB_PATH="data/vocab.txt"
NEW_ARCHIVE_DIR="runs/uttcmvn-pure-attn-2-ft"
CKPT_NAME="best.th"
rm -r $NEW_ARCHIVE_DIR
cp -r $ARCHIVE_DIR $NEW_ARCHIVE_DIR
python3 tools/merge_vocab.py $ARCHIVE_DIR/vocabulary/characters.txt $VOCAB_PATH $NEW_ARCHIVE_DIR/vocabulary/characters.txt
python3 tools/extend_embedding.py $ARCHIVE_DIR/$CKPT_NAME "_target_embedder.weight" $ARCHIVE_DIR/vocabulary/characters.txt $NEW_ARCHIVE_DIR/vocabulary/characters.txt $NEW_ARCHIVE_DIR/$CKPT_NAME
python3 tools/extend_embedding.py $NEW_ARCHIVE_DIR/$CKPT_NAME "_output_projection_layer.weight" $ARCHIVE_DIR/vocabulary/characters.txt $NEW_ARCHIVE_DIR/vocabulary/characters.txt $NEW_ARCHIVE_DIR/$CKPT_NAME
python3 tools/extend_embedding.py $NEW_ARCHIVE_DIR/$CKPT_NAME "_output_projection_layer.bias" $ARCHIVE_DIR/vocabulary/characters.txt $NEW_ARCHIVE_DIR/vocabulary/characters.txt $NEW_ARCHIVE_DIR/tmp.th
mv $NEW_ARCHIVE_DIR/tmp.th $NEW_ARCHIVE_DIR/$CKPT_NAME

TRAIN_ROOT="$DATA_ROOT/TAT-Vol1-train-lavalier-train" VAL_ROOT="$DATA_ROOT/TAT-Vol1-train-lavalier-dev"  python3 run.py fine-tune -m $NEW_ARCHIVE_DIR -c experiments/ft.jsonnet -s $1 --include-package stt
