from typing import Tuple

import math
import numpy as np
import re


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


def pad_and_stack(array: np.ndarray,
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
            padded_array = np.pad(
                array, ((0, pad_width), (0, 0)), mode=pad_mode)
        new_shape = ((frame_len + pad_width) // input_stack_rate,
                     feat_dim * input_stack_rate)
        new_len = math.ceil(frame_len / input_stack_rate)
        stacked_array = padded_array.reshape(new_shape)
        return stacked_array, new_len
