from typing import Dict, Iterable, List, Tuple, Any, Union, Callable
import logging
import os
import re
import functools
import tqdm

import numpy as np
from opencc import OpenCC
from overrides import overrides
from conllu import parse_incr
import torch
import torchaudio
import kaldi_io

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, ArrayField, MetadataField
from allennlp.data.fields import LabelField, SequenceLabelField, ListField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer, Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from stt.dataset_readers.utils import pad_and_stack, word_to_phones, get_fisher_callhome_transcripts
from stt.data.characters_indexer import CharactersIndexer
from stt.data.phoneme_tokenizer import PhonemeTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


CTC_SRC_TGT_LEN_RATIO = 2

def get_iterator(dataset_size: int, shard_size: int, bucket: bool = True):
    if bucket:
        batch_indices = list(range(0, dataset_size, shard_size))
        np.random.shuffle(batch_indices)
        for batch_idx in batch_indices:
            end = min(batch_idx + shard_size, dataset_size)
            for idx in range(batch_idx, end):
                yield idx
    else:
        indices = list(range(dataset_size))
        np.random.shuffle(indices)
        for idx in indices:
            yield idx


def read_dependencies(file_path: str, use_language_specific_pos=False):

    annotations = []
    with open(os.path.join(file_path, "dep.conll"), "r") as conllu_file:
        for annotation in tqdm.tqdm(parse_incr(conllu_file)):
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

def data_func_factory(data_gen, orders):
    def data_func(idx):
        if orders is not None:
            idx = orders[idx]
        return data_gen(idx)

    return data_func

def mao_get_datas(file_path: str, online: bool = False, trn_file: str = "trn.txt",
                  mmap: bool = True, dep: bool = False) -> Tuple[Callable, Callable, Callable, int]:
    source_orders = None
    try:
        source_lens = np.load(os.path.join(file_path, 'lens.npy'))
        source_positions = np.pad(source_lens, pad_width=(1, 0), mode='constant') \
            .cumsum()
        source_orders = np.argsort(source_lens)

    except FileNotFoundError:
        logger.warning(f"lens.npy not found under directory {file_path}; bucketing won't take effect.")

    if not online:
        if mmap:
            source_datas = np.load(os.path.join(
                file_path, 'data.npy'), mmap_mode='r')
        else:
            source_datas = np.load(os.path.join(
                file_path, 'data.npy'))

        src_data_gen = lambda idx: source_datas[source_positions[idx]:source_positions[idx+1], :]

    else:
        with open(os.path.join(file_path, "refs.txt")) as f:
            source_datas = [os.path.join(file_path, wav) for wav in f.read().splitlines()]
            src_data_gen = lambda idx: source_datas[idx]

    with open(os.path.join(file_path, trn_file)) as f:
        target_datas = f.read().splitlines()

    src_data_func = data_func_factory(src_data_gen, source_orders)
    tgt_data_func = data_func_factory(lambda idx: target_datas[idx], source_orders)

    annotations = None
    if dep:
        annotations = read_dependencies(file_path)

    anno_data_func = data_func_factory(lambda idx: annotations[idx] if annotations else None,
                                       source_orders)

    return src_data_func, tgt_data_func, anno_data_func, len(target_datas)

def kaldi_get_datas(file_path: str, targets: List[Tuple[str]],
                    dep: bool = False) -> Tuple[Callable, Callable, Callable, int]:
    raw_src_datas: Dict[str, np.array] = {k: m for k, m in kaldi_io.read_mat_scp(file_path)}
    source_datas = []
    target_datas = []
    for utt_ids, src_trns, *tgt_trns in targets:
        utt_datas = []
        for utt_id in utt_ids:
            utt_datas.append(raw_src_datas[utt_id])
        source_data = np.concatenate((utt_datas), axis=-1)
        source_datas.append(source_data)
        target_datas.append(tgt_trns[0])

    src_lens = [src.shape[0] for src in source_datas]
    source_orders = np.argsort(src_lens)

    src_data_func = data_func_factory(lambda idx: source_datas[idx], source_orders)
    tgt_data_func = data_func_factory(lambda idx: target_datas[idx], source_orders)

    annotations = None
    if dep:
        annotations = read_dependencies(file_path)

    anno_data_func = data_func_factory(lambda idx: annotations[idx] if annotations else None,
                                       source_orders)

    return src_data_func, tgt_data_func, anno_data_func, len(source_orders)

def flatten(l):
    return [item for sublist in l for item in sublist]

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
                 fisher_ch: Tuple[str, str, str] = None,
                 word_level: bool = False,
                 discard_energy_dim: bool = False,
                 input_stack_rate: int = 1,
                 model_stack_rate: int = 1,
                 max_frames: int = 3000,
                 target_tokenizer: Tokenizer = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 target_add_start_end_token: bool = False,
                 delimiter: str = "\t",
                 curriculum: List[Tuple[int, int]] = None,
                 online: bool = False,
                 num_mel_bins: int = 80,
                 mmap: bool = True,
                 dep: bool = False,
                 bucket: bool = False,
                 noskip: bool = False,
                 trn_file: str = "trn.txt",
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._target_tokenizer = target_tokenizer or WordTokenizer()
        self._target_token_indexers = target_token_indexers or {
            "tokens": SingleIdTokenIndexer(start_tokens=[START_SYMBOL], end_tokens=[END_SYMBOL]),
            "token_characters": CharactersIndexer()
        }
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
        self._discard_energy_dim = discard_energy_dim
        self._online = online
        self._num_mel_bins = num_mel_bins
        self._noskip = noskip
        self._trn_file = trn_file

        cc = OpenCC('s2t')
        if lexicon_path is not None:
            lexicon: Dict[str, str] = {}
            with open(lexicon_path) as f:
                for line in f.read().splitlines():
                    end, start = re.search(r'\s+', line).span()
                    lexicon[line[:end]] = line[start:]
            w2p = word_to_phones(lexicon)
            phonemizer = lambda word: w2p(cc.convert(word))
            phn_tokenizer = PhonemeTokenizer(phonemizer)
            self._target_token_indexers["phonemes"] = \
                CharactersIndexer(namespace='phonemes',
                                  character_tokenizer=phn_tokenizer)

        self._word_level = word_level
        self._fisher_callhome_datas = None

        if fisher_ch is not None:
            root, corpus, split = fisher_ch
            self._fisher_callhome_datas = \
                get_fisher_callhome_transcripts(root, corpus, split)

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # pylint: disable=arguments-differ
        logger.info('Loading data from %s', file_path)
        dropped_instances = 0

        if self._fisher_callhome_datas is None:
            source_datas, target_datas, annotations, dataset_size = \
                mao_get_datas(file_path, self._online, self._trn_file,
                              self._mmap, self._dep)
        else:
            utt_ids, src_trns, list_of_tgt_trns = self._fisher_callhome_datas
            targets = zip(utt_ids, src_trns, *list_of_tgt_trns)
            source_datas, target_datas, annotations, dataset_size = \
                kaldi_get_datas(file_path, targets, self._dep)

        iterator = get_iterator(dataset_size, self._shard_size, self._bucket)

        for idx in iterator:
            drop = False
            src = source_datas(idx)
            tgt = target_datas(idx)
            annotation = annotations(idx)
            if not isinstance(src, str) and src.shape[0] == 0 or not tgt.strip():
                drop = True
            else:
                instance = self.text_to_instance(src, tgt, annotation)
                source_tensor = instance.fields["source_features"].array
                src_len = source_tensor.shape[0]
                tgt_len = instance.fields['target_tokens'].sequence_length()
                if tgt_len < 1 or src_len > self._max_frames \
                        or (src_len // self.stack_rate) < CTC_SRC_TGT_LEN_RATIO * tgt_len \
                        or np.isnan(source_tensor).any():
                    drop = True
            if not self._noskip and drop:
                dropped_instances += 1
                continue
            yield instance
            del instance

        if not dropped_instances:
            logger.info("No instances dropped from {}.".format(file_path))
        else:
            logger.warning("Dropped {} instances from {}.".format(dropped_instances,
                                                                  file_path))
        self._epoch_num += 1

    def _text_to_dep_fields(self,
                            words: List[str],
                            upos_tags: List[str],
                            dependencies: List[Tuple[str, int]] = None) -> Instance:

        fields: Dict[str, Field] = {}

        tokens = [Token(t) for t in words]

        text_field = TextField(tokens, self._target_token_indexers)
        fields["words"] = text_field

        fields["pos_tags"] = SequenceLabelField(upos_tags, text_field, label_namespace="pos")
        if dependencies is not None:
            # We don't want to expand the label namespace with an additional dummy token, so we'll
            # always give the 'ROOT_HEAD' token a label of 'root'.
            fields["head_tags"] = SequenceLabelField([x[0] for x in dependencies],
                                                        text_field,
                                                        label_namespace="head_tags")
            fields["head_indices"] = SequenceLabelField([x[1] for x in dependencies],
                                                        text_field,
                                                        label_namespace="head_index_tags")

        fields["metadata"] = MetadataField({"words": words, "pos": upos_tags})

        return fields

    @overrides
    def text_to_instance(self,
                         source: Union[np.ndarray, str],
                         target_string: str = None,
                         annotation: List[Any] = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        if isinstance(source, str):
            source, _ = torchaudio.load(source)
            source = torchaudio.compliance.kaldi.fbank(source,
                                                       use_energy=(not self._discard_energy_dim),
                                                       num_mel_bins=self._num_mel_bins,
                                                       dither=0.0, energy_floor=1.0).numpy()

        source, src_len = pad_and_stack(source,
                                        self.input_stack_rate,
                                        self.model_stack_rate,
                                        pad_mode=self._pad_mode)

        source_length_field = LabelField(src_len, skip_indexing=True)
        source_field = ArrayField(source)

        if target_string is not None:
            target = self._target_tokenizer.tokenize(target_string)

            target_field = TextField(target,
                                     self._target_token_indexers)

            fields = {"source_features": source_field,
                      "target_tokens": target_field,
                      "source_lengths": source_length_field}

            if self._dep and annotation is not None:
                heads, tags, words, pos_tags = annotation
                dep_fields = self._text_to_dep_fields(words, pos_tags, list(zip(tags, heads)))
                segments = [0] + flatten([[idx + 1] * len(word) for idx, word in enumerate(words)]) + [0]
                fields["segments"] = SequenceLabelField(segments, target_field,
                                                        label_namespace="segment_labels")
                fields.update(dep_fields)

            return Instance(fields)

        else:
            return Instance({"source_features": source_field,
                             "source_lengths": source_length_field})
