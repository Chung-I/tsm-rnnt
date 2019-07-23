from typing import Tuple

import math
import numpy as np


def pad_and_stack(array: np.ndarray,
                  input_stack_rate: int = 1,
                  model_stack_rate: int = 1) -> Tuple[np.ndarray, int]:
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
        padded_frame_len = frame_len
        if frame_len % total_rate != 0:
            padded_frame_len = frame_len + (-frame_len) % total_rate
            padded_array = np.zeros((padded_frame_len, feat_dim))
            padded_array[:frame_len, :] = array
        new_shape = (padded_frame_len // input_stack_rate, feat_dim * input_stack_rate)
        new_len = math.ceil(frame_len / input_stack_rate)
        stacked_array = padded_array.reshape(new_shape)
        return stacked_array, new_len
        