ARCHIVE_DIR="runs/uttcmvn-pure-attn-2"
NEW_ARCHIVE_DIR="runs/uttcmvn-pure-attn-2-ft-$1"
CKPT_NAME="best.th"
FIELD="$1"
python3 tools/make_trns.py $DATA_ROOT/TAT-Vol1-train-lavalier-train $DATA_ROOT/TAT-Vol1-train-lavalier-train --field $FIELD --tokenizer char
python3 tools/make_trns.py $DATA_ROOT/TAT-Vol1-train-lavalier-dev $DATA_ROOT/TAT-Vol1-train-lavalier-dev --field $FIELD --tokenizer char
rm -r $NEW_ARCHIVE_DIR
cp -r $ARCHIVE_DIR $NEW_ARCHIVE_DIR
python3 tools/make_vocab.py $DATA_ROOT/TAT-Vol1-train-lavalier-train/$FIELD.txt $NEW_ARCHIVE_DIR/vocabulary/characters.txt
python3 tools/resize_embedding.py $ARCHIVE_DIR/$CKPT_NAME "_target_embedder.weight" $NEW_ARCHIVE_DIR/vocabulary/characters.txt $NEW_ARCHIVE_DIR/$CKPT_NAME
python3 tools/resize_embedding.py $NEW_ARCHIVE_DIR/$CKPT_NAME "_output_projection_layer.weight" $NEW_ARCHIVE_DIR/vocabulary/characters.txt $NEW_ARCHIVE_DIR/$CKPT_NAME
python3 tools/resize_embedding.py $NEW_ARCHIVE_DIR/$CKPT_NAME "_output_projection_layer.bias" $NEW_ARCHIVE_DIR/vocabulary/characters.txt $NEW_ARCHIVE_DIR/$CKPT_NAME
python3 tools/reinit_module.py $NEW_ARCHIVE_DIR/$CKPT_NAME experiments/decoder_reinit.json $NEW_ARCHIVE_DIR/tmp.th
mv $NEW_ARCHIVE_DIR/tmp.th $NEW_ARCHIVE_DIR/$CKPT_NAME
FIELD=$FIELD TRAIN_ROOT="$DATA_ROOT/TAT-Vol1-train-lavalier-train" VAL_ROOT="$DATA_ROOT/TAT-Vol1-train-lavalier-dev"  python3 run.py fine-tune -m $NEW_ARCHIVE_DIR -c experiments/ft.jsonnet -s $2 --include-package stt
rm -r $NEW_ARCHIVE_DIR
