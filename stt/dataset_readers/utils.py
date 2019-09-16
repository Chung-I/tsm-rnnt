from typing import Tuple, Union, List
from pathlib import Path

import math
import numpy as np
import re

import torch
import torch.nn.functional as F


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


def pad_and_stack(array: Union[np.ndarray, torch.Tensor],
                  input_stack_rate: int = 1,
                  model_stack_rate: int = 1,
                  pad_mode: str = 'wrap') -> Tuple[np.ndarray, int]:
    """
        Pad to lengths with multiples of input_stack_rate * model_stack_rate.
        Parameters
        ----------
        array : np.ndarray, required
            Array to be padded and stacked.
        input_stack_rate : int, optional
            stack every this many frames upon input.
        model_stack_rate : int, optional
            stack every this many frames later in model.

        Returns
        -------
        array: np.ndarray
            New padded array.
        array_length: int
            New array length where padding doesn't count.

    """
    if input_stack_rate == 1 and model_stack_rate == 1:
        return array, array.shape[0]
    else:
        total_rate = input_stack_rate * model_stack_rate
        frame_len, feat_dim = array.shape
        padded_array = array
        pad_width = 0
        if frame_len % total_rate != 0:
            pad_width = (-frame_len) % total_rate
            if isinstance(array, np.ndarray):
                padded_array = np.pad(
                    array, ((0, pad_width), (0, 0)), mode=pad_mode)
            else:
                pad_mode = "circular" if pad_mode == "wrap" else pad_mode
                padded_array = F.pad(array.transpose(1, 0).unsqueeze(0),
                                     pad=(0, pad_width), mode=pad_mode).squeeze(0).transpose(1, 0)

        new_shape = ((frame_len + pad_width) // input_stack_rate,
                     feat_dim * input_stack_rate)
        new_len = math.ceil(frame_len / input_stack_rate)
        stacked_array = padded_array.reshape(new_shape)
        return stacked_array, new_len

def get_fisher_callhome_transcripts(root: str, corpus: str, split: str,
                                    src_lang: str = 'es', tgt_lang: str = 'en',
                                    num_tgt_trns: int = 4
                                    ) -> Tuple[List[List[str]], List[str], List[List[str]]]:
    root = Path(root)
    utt_ids: List[List[str]] = []
    mapping_dir = "kaldi_mapping"
    corpus_dir = "corpus"
    with open(root.joinpath(mapping_dir, f"{corpus}_{split}")) as fp:
        utt_ids = [line.split(" ") for line in fp.read().splitlines()]

    with open(root.joinpath(corpus_dir, "ldc", f"{corpus}_{split}.{src_lang}")) as fp:
        src_transcripts = fp.read().splitlines()

    list_of_tgt_transcripts = []
    if split != 'train':
        for idx in range(num_tgt_trns):
            with open(root.joinpath(corpus_dir, "ldc", f"{corpus}_{split}.{tgt_lang}.{idx}")) as fp:
                tgt_transcripts = fp.read().splitlines()
                list_of_tgt_transcripts.append(tgt_transcripts)
    else:
        with open(root.joinpath(corpus_dir, "ldc", f"{corpus}_{split}.{tgt_lang}")) as fp:
            tgt_transcripts = fp.read().splitlines()
            list_of_tgt_transcripts.append(tgt_transcripts)

    return utt_ids, src_transcripts, list_of_tgt_transcripts
