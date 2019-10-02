export FISHER_PATH="data/fisher_callhome_spanish/translation"
export LEXICON_PATH="/home/nlpmaster/lexicon.txt"
export FISHER_TRAIN_PATH="data/fisher_callhome_spanish/s5/data/train/feats.scp"
export FISHER_VAL_PATH="data/fisher_callhome_spanish/s5/data/dev/feats.scp"
export FISHER_VOCAB_PATH="data/fisher_callhome_spanish/vocabulary"
rm -r $FISHER_VOCAB_PATH
mkdir $FISHER_VOCAB_PATH
allennlp make-vocab $1 -s $FISHER_VOCAB_PATH --include-package stt
mv $FISHER_VOCAB_PATH/vocabulary/* $FISHER_VOCAB_PATH
rm -r $FISHER_VOCAB_PATH/vocabulary
REMOTE_PATH="/groups/public/fisher_callhome_spanish"
mkdir mnt_tmp
cp $FISHER_VOCAB_PATH/tokens.txt $FISHER_VOCAB_PATH/full_tokens.txt || exit 1;
cp $FISHER_VOCAB_PATH/characters.txt $FISHER_VOCAB_PATH/full_characters.txt || exit 1;
sed -i '10002,$d' $FISHER_VOCAB_PATH/tokens.txt || cp $FISHER_VOCAB_PATH/full_tokens.txt $FISHER_VOCAB_PATH/tokens.txt;
sed -i 's/\(@@UNKNOWN@@\)/\1\n@start@\n@end@/g' $FISHER_VOCAB_PATH/tokens.txt || cp $FISHER_VOCAB_PATH/full_tokens.txt $FISHER_VOCAB_PATH/tokens.txt;  
sed -i 's/\(@@UNKNOWN@@\)/\1\n@start@\n@end@/g' $FISHER_VOCAB_PATH/characters.txt || cp $FISHER_VOCAB_PATH/full_characters.txt $FISHER_VOCAB_PATH/characters.txt;  
sshfs -p 3122 storage:$REMOTE_PATH mnt_tmp
cp $FISHER_VOCAB_PATH/* mnt_tmp/vocabulary
umount mnt_tmp && rm -r mnt_tmp || exit 1;
