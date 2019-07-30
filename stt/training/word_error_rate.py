from typing import Optional, List, Set

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

    def __init__(self, exclude_indices: Set[int] = None) -> None:
        self._exclude_indices = exclude_indices or set()
        self._total_errors = 0.
        self._total_words = 0.

    def __call__(self,  # type: ignore
                 predictions: torch.LongTensor,
                 gold_targets: torch.LongTensor) -> None:
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
        predictions, gold_targets = self.unwrap_to_tensors(predictions, gold_targets)

        predictions = [list(filter(lambda idx: idx not in self._exclude_indices, prediction))
            for prediction in predictions.tolist()]

        gold_targets = [list(filter(lambda idx: idx not in self._exclude_indices, gold_target))
            for gold_target in gold_targets.tolist()]

        for prediction, target in zip(predictions, gold_targets):
            self._total_errors += levenshtein(prediction,
                                              target)
            self._total_words += len(target)

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
