from typing import List, Iterator, Dict, Tuple, Any
from overrides import overrides
import numpy as np
import torch
from contextlib import contextmanager

from allennlp.models import Model
from allennlp.data import DatasetReader, Instance
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.dataset_readers.interleaving_dataset_reader import InterleavingDatasetReader

from stt.predictors.utils import wavfile_to_feature


@Predictor.register('online_stt')
class OnlineSpeechToTextPredictor(Predictor):
    """
    Predictor for sequence to sequence models, including
    :class:`~allennlp.models.encoder_decoder.simple_seq2seq` and
    :class:`~allennlp.models.encoder_decoder.copynet_seq2seq`.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super(OnlineSpeechToTextPredictor, self).__init__(model, dataset_reader)
        self._model = model
        self._dataset_reader = dataset_reader

    def load_line(self, line: str) -> JsonDict:
        idx = line[:-1].index(" ")
        wav = line[:idx]
        target_string = line[idx+1:]
        return {"wav": wav, "target_string": target_string}

    def predict(self,
                wavfile: str,
                transcript: str = None) -> JsonDict:
        with self.capture_model_internals() as internals:
            results = self.predict_json({
                "wav": wavfile,
                "target_string": transcript
            })
            return {**results, "model_internals": internals}

    @contextmanager
    def capture_model_internals(self) -> Iterator[dict]:
        """
        Context manager that captures the internal-module outputs of
        this predictor's model. The idea is that you could use it as follows:

        .. code-block:: python

            with predictor.capture_model_internals() as internals:
                outputs = predictor.predict_json(inputs)

            return {**outputs, "model_internals": internals}
        """
        results = {}
        hooks = []

        # First we'll register hooks to add the outputs of each module to the results dict.
        def add_output(idx: int):
            def _add_output(mod, _, outputs):
                if idx in results:
                    prev = torch.Tensor(results[idx])
                    outputs = torch.cat((prev, outputs), dim=0)
                outputs = outputs.unsqueeze(1)
                results[idx] = {"name": str(mod), "output": sanitize(outputs)}
            return _add_output

        for idx, module in enumerate(self._model.modules()):
            if module != self._model:
                hook = module.register_forward_hook(add_output(idx))
                hooks.append(hook)

        # If you capture the return value of the context manager, you get the results dict.
        yield results

        # And then when you exit the context we remove all the hooks.
        for hook in hooks:
            hook.remove()

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"source": "..."}``.
        """
        feature = wavfile_to_feature(json_dict["wav"])
        target_string = json_dict["target_string"]
        dataset = "target_tokens"
        if isinstance(self._dataset_reader, InterleavingDatasetReader):
            return self._dataset_reader._readers[dataset].text_to_instance(
                source=feature,
                target_string=target_string)
        else:
            return self._dataset_reader.text_to_instance(
                source=feature,
                target_string=target_string)
