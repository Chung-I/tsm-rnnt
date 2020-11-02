# Speech Translation from Taiwanese Southern Min(TSM) to Mandarin

## Installation
### Main Dependencies
python >= 3.6
pytorch == 1.1.0
allennlp == 0.8.4
```=bash
pip install -r requirements.txt
```
### Extra Dependencies
#### warp-transducer
```=bash
export CUDA_HOME=[where include/cuda.h and lib/libcudart.so live, e.g. /usr/local/cuda]
git clone https://github.com/HawkAaron/warp-transducer
cd warp-transducer
mkdir build
cd build
cmake -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME ..
make
cd ../pytorch_binding
python setup.py install
```
### torchaudio
```=bash
pip install torchaudio -f https://download.pytorch.org/whl/torch_stable.html
```

## Usage

### data preparation
The file structure of training/validation datas should be arranged as follows:
```
TRAIN_ROOT
├── refs.txt
├── trn.txt
│── corpus_1
│   ├── 1.wav
│   └── 2.wav
└── corpus_2
    ├── 1.wav
    └── 2.wav
```
Dataset readers take two files: `refs.txt` and `trn.txt`:
The `refs.txt` should store paths of audio files:
```
corpus_1/1.wav
corpus_1/2.wav
corpus_2/1.wav
corpus_2/2.wav
```
And the `trn.txt` should store their corresponding transcripts:
```
<transcript of corpus_1/1.wav>
<transcript of corpus_1/2.wav>
<transcript of corpus_2/1.wav>
<transcript of corpus_2/2.wav>
```

Feature extraction is performed by [`torchaudio.compliance.kaldi.fbank`](https://pytorch.org/audio/compliance.kaldi.html#torchaudio.compliance.kaldi.fbank).

If you are NTU Speech Lab users and want to use the 1500hrs TSMDrama data,
extracted filterbanks are readily available at `/groups/public/TSM/trains/data.npy`; don't forget to set the value of the key `online` to `false` in `data_readers` and `validation_data_readers` for `experiments/att.jsonnet`.
```
/groups/public/TSM/trains
├── refs.txt
├── trn.txt
├── data.npy
└── lens.npy
```
### training
```=bash
export TRAIN_ROOT=/where/training/data/lies
export VAL_ROOT=/where/validation/data/lies
allennlp train experiments/att.jsonnet -s SERIALIZATION_DIR --include-package stt
```
For NTU Speech Lab users, `TRAIN_ROOT=/groups/public/TSM/trains`;
`VAL_ROOT=/groups/public/TSM/valids`.
`SERIALIZATION_DIR`: directory in which to save the model and its logs
For more options, run `allennlp train -h`.
### evaluation
```=bash
allennlp evaluate SERIALIZATION_DIR VAL_ROOT --output-file OUTPUT_FILE --include-package stt
```
For NTU Speech Lab users, `VAL_ROOT=/groups/public/TSM/valids`; a
pretrained model is readily available at `SERIALIZATION_DIR=/groups/public/tsm-rnnt-runs/uttcmvn-pure-attn-2`.
For more options, run `allennlp evaluate -h`.

### prediction
```=bash
source path.sh
echo "WAVFILE REF"| allennlp predict --predictor online_stt --output-file OUTPUT_FILE --include-package stt SERIALIZATION_DIR -
```
`WAVFILE`: e.g. `speech.wav`.
`REF`: reference transcript. Currently this only has effect on attention-based model where reference transcript is used when doing in inference through teacher forcing, but still has to be provided when using online models like RNN Transducer and CTC. Will fixed it in the near future.
## Performance
results were evaluated on 平凡很幸福：
| Model | CER | character BLEU | model checkpoint |
| ----- | --- | ----- | ----- |
| Offline (Attn) | 37.8% | 48.9 | [link](https://drive.google.com/file/d/11vLwmOYxfR0w72HgEmhdj3SV5Nt5yPYb/view?usp=sharing) |
| Online (CTC) | 45.5% | 34.4 | - |
| Online(RNNT) | 45.1% | 40.7 | - |

## Todos
- 補上 Offline models

## 劇名

### 民視 
- 阿布拉跟他的三個女人
- 風水世家
- 龍飛鳳舞
- 外鄉女
- 春風望露
- 大時代(第1-160集左右)
- 嫁妝
- 幸福來了

### 大愛
- 我綿一家人
- 真愛緣起
- 碧海映藍天
- 愛主的心願
- 路邊董事長
- 仙女不下凡
- 黃金大天團
- 長盤決勝
- 幸福魔法師
- 竹音深處
- 同學早安
- 我的尪我的某
- 愛上ㄆㄤˋ滋味
- 若是來恆春
- 我和我母親
- 四重奏
- 清風無痕
- 阿寬
- 有你陪伴
- 讓愛飛翔
- 在愛之外
- 生命桃花源

### Performance on TAT-train-lavalier-dev
We fine-tuned the  [Offline (Attn) model](https://drive.google.com/file/d/11vLwmOYxfR0w72HgEmhdj3SV5Nt5yPYb/view?usp=sharing) mentioned above on [TAT-train-lavalier](https://sites.google.com/speech.ntut.edu.tw/fsw/home/challenge-2020), a Taiwanese speech recognition challenge corpus. We split it further into TAT-train-lavalier-train (`/groups/public/TAT-Vol1-train-lavalier-train` on battleship) and TAT-train-lavalier-dev (`/groups/public/TAT-Vol1-train-lavalier-dev` on battleship) and report score on the latter split.
| Model      | CER on 漢羅台文 (no punct)  |  CER on 台羅數字調 |
| -----      | -----------                | ---------------- |
| FT-SpecAug | 27.4% （繼承原模型embedding) ([model](https://drive.google.com/drive/folders/1kXT9KHdsTkj6anrIDFKegVzd1YNQ5ZQd?usp=sharing)) | 12.6% ([model](https://drive.google.com/drive/folders/10h23s4NwoFSJOon0grnd1uJRvWyVDkYq?usp=sharing)) |
| SpecAug    | 51.8% ([model](https://drive.google.com/drive/folders/1auxXgJyliqwHLeLKJ20kMRQf48GULu77?usp=sharing)) | 13.8% ([model](https://drive.google.com/drive/folders/14mXqSZBGPEMAgYQLPXwhhatJFcO1w0vD?usp=sharing))|

FT-SpecAug: [SpecAugment](https://arxiv.org/abs/1904.08779), decoder parameter transfer(including both input and output character embedding)
(On battleship)
gold: `/groups/public/TAT-Vol1-train-lavalier-dev/trn.txt`
pred: `/groups/public/tsm-rnnt-runs/TAT-ft-specaug/val_trn.txt`

### fine-tuning on TAT-Vol1-lavalier from pretrained encoder and decoder character embeddings
put the [Offline (Attn) model](https://drive.google.com/file/d/11vLwmOYxfR0w72HgEmhdj3SV5Nt5yPYb/view?usp=sharing) in `SERIALIZATION_DIR` and run:
```=bash
export DATA_ROOT=/where/data/lies
bash ft.sh SERIALIZATION_DIR
```
For NTU Speech Lab users, `DATA_ROOT=/groups/public`;

### fine-tuning on TAT-Vol1-lavalier from pretrained encoder (decoder is reinitialized)
put the [Offline (Attn) model](https://drive.google.com/file/d/11vLwmOYxfR0w72HgEmhdj3SV5Nt5yPYb/view?usp=sharing) in `SERIALIZATION_DIR` and run:
```=bash
export DATA_ROOT=/where/data/lies
bash ft-reinit.sh SERIALIZATION_DIR
```
For NTU Speech Lab users, `DATA_ROOT=/groups/public`;

### train on TAT-Vol1-lavalier from scratch
```=bash
python3 tools/make_trns.py $DATA_ROOT/TAT-Vol1-train-lavalier-train $DATA_ROOT/TAT-Vol1-train-lavalier-train --field $FIELD --tokenizer char
python3 tools/make_trns.py $DATA_ROOT/TAT-Vol1-train-lavalier-dev $DATA_ROOT/TAT-Vol1-train-lavalier-dev --field $FIELD --tokenizer char
TRAIN_ROOT="$DATA_ROOT/TAT-Vol1-train-lavalier-train" VAL_ROOT="$DATA_ROOT/TAT-Vol1-train-lavalier-dev" FIELD="漢羅台文"  python3 run.py train experiments/ft.jsonnet -s runs/TAT-specaug --include-package stt
```

For NTU Speech Lab users, `DATA_ROOT=/groups/public`;
`FIELD` can be 漢羅台文or台羅數字調
