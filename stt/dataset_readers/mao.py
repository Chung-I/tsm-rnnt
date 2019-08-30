from typing import Dict, Iterable, List, Tuple, Any
import logging
import os
import re

import numpy as np
from opencc import OpenCC
from overrides import overrides
from conllu import parse_incr

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ArrayField, MetadataField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer, Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from stt.dataset_readers.utils import pad_and_stack, word_to_phones


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def read_dependencies(file_path: str, use_language_specific_pos=False):

    annotations = []
    with open(os.path.join(file_path, "dep.txt"), "r") as conllu_file:
        for annotation in parse_incr(conllu_file):
            annotation = [x for x in annotation if isinstance(x["id"], int)]

            heads = [x["head"] for x in annotation]
            tags = [x["deprel"] for x in annotation]
            words = [x["form"] for x in annotation]
            if use_language_specific_pos:
                pos_tags = [x["xpostag"] for x in annotation]
            else:
                pos_tags = [x["upostag"] for x in annotation]
            annotations.append([heads, tags, words, pos_tags])

    return annotations

@DatasetReader.register("mao-stt")
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
    delimiter : str, (optional, default="\t")
        Set delimiter for tsv/csv file.
    """

    def __init__(self,
                 shard_size: int,
                 lexicon_path: str = None,
                 is_phone: bool = False,
                 to_char: bool = False,
                 input_stack_rate: int = 1,
                 model_stack_rate: int = 1,
                 max_frames: int = 3000,
                 target_tokenizer: Tokenizer = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 target_add_start_end_token: bool = False,
                 delimiter: str = "\t",
                 curriculum: List[Tuple[int, int]] = None,
                 mmap: bool = True,
                 dep: bool = False,
                 bucket: bool = False,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._target_tokenizer = target_tokenizer or WordTokenizer()
        self._target_token_indexers = target_token_indexers or {
            "tokens": SingleIdTokenIndexer()}
        self._delimiter = delimiter
        self._shard_size = shard_size
        self.input_stack_rate = input_stack_rate
        self.model_stack_rate = model_stack_rate
        self.stack_rate = input_stack_rate * model_stack_rate
        self._target_add_start_end_token = target_add_start_end_token
        self._pad_mode = "wrap" if input_stack_rate == 1 else "constant"
        self._bucket = bucket
        self._max_frames = max_frames
        self._curriculum = curriculum
        self._epoch_num = 0
        self._mmap = mmap
        self._dep = dep

        self.lexicon: Dict[str, str] = {}
        if lexicon_path is not None:
            with open(lexicon_path) as f:
                for line in f.read().splitlines():
                    end, start = re.search(r'\s+', line).span()
                    self.lexicon[line[:end]] = line[start:]

        self.cc = OpenCC('s2t')
        self.w2p = word_to_phones(self.lexicon)
        self._is_phone = is_phone
        self._to_char = to_char

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # pylint: disable=arguments-differ
        logger.info('Loading data from %s', file_path)
        dropped_instances = 0
        if self._mmap:
            source_datas = np.load(os.path.join(
                file_path, 'data.npy'), mmap_mode='r')
        else:
            source_datas = np.load(os.path.join(
                file_path, 'data.npy'))
        source_lens = np.load(os.path.join(file_path, 'lens.npy'))
        source_positions = np.pad(source_lens, pad_width=(1, 0), mode='constant') \
            .cumsum()
        with open(os.path.join(file_path, "trn.txt")) as f:
            target_datas = f.read().splitlines()

        annotations = []
        if self._dep:
            annotations = read_dependencies(file_path)

        max_src_len = self.get_max_src_len()
        curriculum = max_src_len < np.inf

        source_orders = np.argsort(source_lens)

        source_orders = [
            idx for idx in source_orders if source_lens[idx] < max_src_len]

        batched_indices = list(range(0, len(source_orders), self._shard_size))
        np.random.shuffle(batched_indices)

        annotation = None

        for start_idx in batched_indices:
            end_idx = min(start_idx + self._shard_size, len(source_orders))
            for idx in range(start_idx, end_idx):
                if annotations:
                    annotation = annotations[idx]
                if self._bucket or curriculum:
                    idx = source_orders[idx]
                start, end = source_positions[idx], source_positions[idx+1]
                src = source_datas[start:end]
                tgt = target_datas[idx]
                instance = self.text_to_instance(src, tgt, annotation)
                tgt_len = instance.fields['target_tokens'].sequence_length()
                if tgt_len < 1 or src.shape[0] > self._max_frames \
                        or (src.shape[0]//self.stack_rate) < tgt_len:
                    dropped_instances += 1
                else:
                    yield instance
                del instance
        if not dropped_instances:
            logger.info("No instances dropped from {}.".format(file_path))
        else:
            logger.warning("Dropped {} instances from {}.".format(dropped_instances,
                                                                  file_path))
        self._epoch_num += 1

    @overrides
    def text_to_instance(self,
                         source_array: np.ndarray,
                         target_string: str = None,
                         annotation: List[Any] = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        source_array, src_len = pad_and_stack(source_array,
                                              self.input_stack_rate,
                                              self.model_stack_rate,
                                              pad_mode=self._pad_mode)
        source_length_field = LabelField(src_len, skip_indexing=True)
        source_field = ArrayField(source_array)

        if target_string is not None:
            char_target = [Token(x) for x in list(re.sub(r"\s+", "", target_string))]
            target = self._target_tokenizer.tokenize(target_string)

            if self.lexicon:
                phn_target: List[str] = []
                tokenized_target = [word.text for word in target]
                for word in tokenized_target:
                    word = self.cc.convert(word)
                    phn_target.extend(self.w2p(word))

                # if self._target_add_start_end_token:
                #     phn_target.insert(0, START_SYMBOL)
                #     phn_target.append(END_SYMBOL)
                phn_target_field = TextField([Token(x) for x in phn_target], {
                    "tokens": SingleIdTokenIndexer()})

            if self._target_add_start_end_token:
                target.insert(0, Token(START_SYMBOL))
                target.append(Token(END_SYMBOL))

            target_field = TextField(
                target, self._target_token_indexers)
            char_target_field = TextField(char_target, {
                "tokens": SingleIdTokenIndexer()})
            print(target, char_target, phn_target)
            return Instance({"source_features": source_field,
                             "target_tokens": target_field,
                             #"target_phn_tokens": phn_target_field,
                             #"target_char_tokens": char_target_field,
                             "transcripts": MetadataField(target_string),
                             "source_lengths": source_length_field})

        else:
            return Instance({"source_features": source_field,
                             "source_lengths": source_length_field})

    def get_max_src_len(self):
        max_len = np.inf
        if self._curriculum is not None:
            for epoch, cur_max_len in self._curriculum:
                if self._epoch_num < epoch:
                    break
                max_len = cur_max_len

        return max_len
