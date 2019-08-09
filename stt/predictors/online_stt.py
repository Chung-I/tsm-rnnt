from overrides import overrides
import numpy as np

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

from stt.predictors.utils import wavfile_to_feature


@Predictor.register('online_stt')
class OnlineSpeechToTextPredictor(Predictor):
    """
    Predictor for sequence to sequence models, including
    :class:`~allennlp.models.encoder_decoder.simple_seq2seq` and
    :class:`~allennlp.models.encoder_decoder.copynet_seq2seq`.
    """

    def predict(self,
                wavfile: str,
                transcript: str = None) -> JsonDict:
        feature = wavfile_to_feature(wavfile)
        return self.predict_json({
            "source_array": feature.tolist(),
            "target_string": transcript
        })

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"source": "..."}``.
        """
        source_array = np.array(json_dict["source_array"])
        target_string = json_dict["target_string"]

        return self._dataset_reader.text_to_instance(
            source_array,
            target_string)
