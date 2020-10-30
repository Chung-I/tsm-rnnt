# Speech Translation from Taiwanese Southern Min(TSM) to Mandarin

## Installation
### Main Dependencies
python >= 3.6
pytorch >= 1.2.0
allennlp >= 0.9.0
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

### training

```=bash
source path.sh
allennlp train experiments/config.jsonnet -s SERIALIZATION_DIR --include-package stt
```
`SERIALIZATION_DIR`: directory in which to save the model and its logs
For more options, run `allennlp train -h`.
### evaluation
```=bash
source path.sh
allennlp evaluate SERIALIZATION_DIR VAL_DATA_PATH --output-file OUTPUT_FILE --include-package stt
```
For more options, run `allennlp evaluate -h`.

### prediction
```=bash
source path.sh
echo "WAVFILE REF"| allennlp predict --predictor online_stt --output-file OUTPUT_FILE --include-package stt SERIALIZATION_DIR -
```
`WAVFILE`: e.g. `speech.wav`.
`REF`: reference transcript. Currently this only has effect on attention-based model where reference transcript is used when doing in inference through teacher forcing, but still has to be provided when using online models like RNN Transducer and CTC. Will fixed it in the near future.
## Performance (evaluated on 平凡很幸福)
| Model | CER | character BLEU | model checkpoint |
| ----- | --- | ----- | ----- |
| Offline (Attn) | 37.8% | 48.9 | [link](https://drive.google.com/file/d/11vLwmOYxfR0w72HgEmhdj3SV5Nt5yPYb/view?usp=sharing) |
| Online (CTC) | 45.5% | 34.4 | - |
| Online(RNNT) | 45.1% | 40.7 | - |

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
