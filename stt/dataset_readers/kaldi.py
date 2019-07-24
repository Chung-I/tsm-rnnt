import csv
from typing import Dict, Iterable, List, Tuple
import logging
import random
import glob
import re
import numpy as np
from overrides import overrides
from opencc import OpenCC

import kaldi_io
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ArrayField, MetadataField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer, Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from stt.dataset_readers.utils import pad_and_stack

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def process_phone(phone, remove_tone=True):
    if remove_tone:
        phone = re.sub("\d+", "", phone)
    return phone

def word_to_phones(lexicon):
    def w2p(word):
        phones = []
        try:
            phones.extend(re.split("\s+", lexicon[word]))
        except KeyError:
            for char in word:
                try:
                    phones.extend(re.split("\s+", lexicon[char]))
                except KeyError:
                    pass
        phones = [process_phone(phone) for phone in phones]
        return phones

    return w2p

@DatasetReader.register("kaldi-stt")
class KaldiSpeechToTextDatasetReader(DatasetReader):
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
    delimiter : str, (optional, default="\t")
        Set delimiter for tsv/csv file.
    """
    def __init__(self,
                 shard_size: int,
                 lexicon_path: str,
                 transcript_path: str,
                 input_stack_rate: int = 1,
                 model_stack_rate: int = 1,
                 target_tokenizer: Tokenizer = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 delimiter: str = "\t",
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        transcript_files = glob.glob(transcript_path)
        self.transcripts: Dict[str, str] = {}
        for transcript_file in transcript_files:
            with open(transcript_file) as f:
                for line in f.read().splitlines():
                    end, start = re.search(r'\s+', line).span()
                    self.transcripts[line[:end]] = line[start:]

        self.lexicon: Dict[str, str] = {}
        with open(lexicon_path) as f:
            for line in f.read().splitlines():
                end, start = re.search(r'\s+', line).span()
                self.lexicon[line[:end]] = line[start:]

        self.cc = OpenCC('s2t')
        self.w2p = word_to_phones(self.lexicon)
        self._target_tokenizer = target_tokenizer or WordTokenizer()
        self._target_token_indexers = target_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._delimiter = delimiter
        self._shard_size = shard_size
        self.input_stack_rate = input_stack_rate
        self.model_stack_rate = model_stack_rate

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # pylint: disable=arguments-differ
        logger.info('Loading data from %s', file_path)
        dropped_instances = 0

        feats: List[Tuple[str, np.ndarray]] = [(key, mat) for key, mat in
                                               kaldi_io.read_mat_scp(file_path)]
        feats = sorted(feats, key=lambda pair: pair[1].shape[0])

        dataset_size = len(feats)

        batched_indices = list(range(0, dataset_size, self._shard_size))
        random.shuffle(batched_indices)
        for start_idx in batched_indices:
            end_idx = min(start_idx + self._shard_size, dataset_size)
            for idx in range(start_idx, end_idx):
                key, src = feats[idx]
                tgt = self.transcripts[key]
                instance = self.text_to_instance(src, tgt)
                tgt_len = instance.fields['target_tokens'].sequence_length()
                if tgt_len < 1:
                    dropped_instances += 1
                else:
                    yield instance
                del src
                del instance
        if not dropped_instances:
            logger.info("No instances dropped from {}.".format(file_path))
        else:
            logger.warning("Dropped {} instances from {}.".format(dropped_instances,
                                                                  file_path))

    @overrides
    def text_to_instance(self,
                         source_array: np.ndarray,
                         target_string: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        source_array, src_len = pad_and_stack(source_array,
                                              self.input_stack_rate,
                                              self.model_stack_rate)
        source_length_field = LabelField(src_len, skip_indexing=True)
        source_field = ArrayField(source_array)
        if target_string is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            phonemized_target: List[str] = []
            for word in tokenized_target:
                word = self.cc.convert(word.text)
                phonemized_target.extend(self.w2p(word))

            # print(target_string, phonemized_target)
            target_field = TextField([Token(x) for x in phonemized_target],
                            self._target_token_indexers)
            return Instance({"source_features": source_field,
                             "target_tokens": target_field,
                             "source_lengths": source_length_field})
        else:
            return Instance({"source_features": source_field,
                             "source_lengths": source_length_field})
