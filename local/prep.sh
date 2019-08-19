DATA_DIR=$1

cd $DATA_DIR
subword-nmt learn-bpe -s 32000 < *.de-en.* > bpe.vocab
for file in *.de-en.*;
do
  subword-nmt apply-bpe -c bpe.vocab < $file > bpe.${file}
done

