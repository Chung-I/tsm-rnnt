from typing import Optional, List

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric

from stt.data.text import levenshtein


@Metric.register("word_error_rate")
class WordErrorRate(Metric):
    """
    Calculates word error rate(WER).

    Parameters
    ----------
    ignore_classes : ``List[int]``, optional (default = None)
        A list of label ids to ignore when computing metrics.
    """

    def __init__(self) -> None:
        self._total_errors = 0.
        self._total_words = 0.

    def __call__(self,  # type: ignore
                 predicted_indices: List[List[int]],
                 gold_indices: List[List[int]],
                 target_lengths: List[int]):
        """
        Parameters
        ----------
        predicted_indices : ``List[List[int]]``, required.
            A tensor of head index predictions of shape (batch_size, timesteps).
        gold_indices : ``List[List[int]]``, required.
            A tensor of the same shape as ``predicted_indices``.
        lengths: ``List[int]``, required.
            A tensor of the same shape as ``predicted_indices``.
        """
        for predicted, gold, length in zip(predicted_indices, gold_indices, target_lengths):
            self._total_errors += levenshtein(predicted,
                                              gold[:length])
            self._total_words += length

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated metrics as a dictionary.
        """
        wer = self._total_errors / self._total_words
        if reset:
            self.reset()
        return {
            "WER": wer
        }

    @overrides
    def reset(self):
        self._total_errors = 0.
        self._total_words = 0.
