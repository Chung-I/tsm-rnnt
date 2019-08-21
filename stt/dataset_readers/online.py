from typing import Dict, Iterable, Callable, Union, Tuple
import logging
from pathlib import Path
from overrides import overrides

import torchaudio
from torchaudio.transforms import Resample, MelSpectrogram

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ArrayField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer, Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from stt.dataset_readers.utils import pad_and_stack
from stt.data.tensor_field import TensorField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("online")
class SpeechToTextDatasetReader(DatasetReader):
    """
    Read a tsv file containing paired sequences, and create a dataset suitable for a
    ``SimpleSeq2Seq`` model, or any model with a matching API.

    Expected format for each input line: <source_sequence_string>\t<target_sequence_string>

    The output of ``read`` is a list of ``Instance`` s with the fields:
        source_tokens: ``TextField`` and
        target_tokens: ``TextField``

    `START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.

    Parameters
    ----------
    source_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    target_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to ``source_tokenizer``.
    target_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define output (target side) token representations. Defaults to
        ``source_token_indexers``.
    """

    def __init__(self,
                 sample_rate: int = 16000,
                 input_stack_rate: int = 1,
                 model_stack_rate: int = 1,
                 max_frames: int = 3000,
                 target_tokenizer: Tokenizer = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 target_add_start_end_token: bool = False,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._target_tokenizer = target_tokenizer or WordTokenizer()
        self._target_token_indexers = target_token_indexers or {
            "tokens": SingleIdTokenIndexer()}
        self.input_stack_rate = input_stack_rate
        self.model_stack_rate = model_stack_rate
        self.stack_rate = input_stack_rate * model_stack_rate
        self._target_add_start_end_token = target_add_start_end_token
        self._pad_mode = "wrap" if input_stack_rate == 1 else "constant"
        self._max_frames = max_frames
        self._epoch_num = 0
        self._sample_rate = sample_rate
        win_length = int(sample_rate * 0.025)
        hop_length = int(sample_rate * 0.01)
        n_fft = win_length
        self._mel_spectrogram = MelSpectrogram(sample_rate, n_fft,
                                               win_length=win_length,
                                               hop_length=hop_length,
                                               n_mels=80)

    @overrides
    def _read(self, *args) -> Iterable[Instance]:
        instance = self.text_to_instance(*args)
        src_len = instance.fields['source_lengths'].label
        tgt_len = instance.fields['target_tokens'].sequence_length()
        if tgt_len < 1 or src_len > self._max_frames \
                or (src_len//self.stack_rate) <= tgt_len:
            print("source length {} smaller than target length {}, skipping".format(
                src_len//self.stack_rate, tgt_len))
            yield None
        else:
            yield instance
        # pylint: disable=arguments-differ

    @overrides
    def text_to_instance(self,
                         data: Tuple[str, str]) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        wav_file, text = data
        if callable(wav_file):
            y = wav_file()
        else:
            y, orig_freq = torchaudio.load(wav_file)
            if orig_freq != self._sample_rate:
                resample = Resample(orig_freq=orig_freq, new_freq=16000)
                y = resample(y)

        source_array = torchaudio.compliance.kaldi.fbank(
            y, num_mel_bins=80, use_energy=True).detach()
        #source_array = self._mel_spectrogram(y).detach()
        source_array, src_len = pad_and_stack(source_array,
                                              self.input_stack_rate,
                                              self.model_stack_rate,
                                              pad_mode=self._pad_mode)
        source_length_field = LabelField(src_len, skip_indexing=True)
        source_field = TensorField(source_array)

        if text is not None:
            target = self._target_tokenizer.tokenize(text)
            if self._target_add_start_end_token:
                target.insert(0, Token(START_SYMBOL))
                target.append(Token(END_SYMBOL))

            target_field = TextField(
                target, self._target_token_indexers)
            return Instance({"source_features": source_field,
                             "target_tokens": target_field,
                             "source_lengths": source_length_field})
        else:
            return Instance({"source_features": source_field,
                             "source_lengths": source_length_field})
