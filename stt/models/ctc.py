from typing import Dict, Optional, List, Any

import os
import numpy
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
import torch.nn.functional as F
from torch.nn import CTCLoss

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits, \
    get_lengths_from_binary_sequence_mask, get_mask_from_sequence_lengths
from allennlp.training.metrics import BLEU
from ds_ctcdecoder import ctc_beam_search_decoder_batch

from stt.data.text import Alphabet
from stt.training.word_error_rate import WordErrorRate


@Model.register("ctc")
class CTCModel(Model):
    """
    This ``SimpleTagger`` simply encodes a sequence of text with a stacked ``Seq2SeqEncoder``, then
    predicts a tag for each token in the sequence.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and predicting output tags.
    calculate_span_f1 : ``bool``, optional (default=``None``)
        Calculate span-level F1 metrics during training. If this is ``True``, then
        ``label_encoding`` is required. If ``None`` and
        label_encoding is specified, this is set to ``True``.
        If ``None`` and label_encoding is not specified, it defaults
        to ``False``.
    label_encoding : ``str``, optional (default=``None``)
        Label encoding to use when calculating span f1.
        Valid options are "BIO", "BIOUL", "IOB1", "BMES".
        Required if ``calculate_span_f1`` is true.
    verbose_metrics : ``bool``, optional (default = False)
        If true, metrics will be returned per label class in addition
        to the overall statistics.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary,
                 encoder: Seq2SeqEncoder,
                 loss_type: str = "ctc",
                 beam_size: int = 5,
                 target_namespace: str = "target_tokens",
                 vocab_path: str = "phonemes/vocabulary",
                 verbose_metrics: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(CTCModel, self).__init__(vocab, regularizer)

        self._target_namespace = target_namespace
        self.num_classes = self.vocab.get_vocab_size(target_namespace)
        self.encoder = encoder
        self._verbose_metrics = verbose_metrics
        self.projection_layer = TimeDistributed(Linear(encoder.get_output_dim(),
                                                       self.num_classes))

        self._blank_idx = self.vocab.get_token_index(DEFAULT_PADDING_TOKEN)
        self._loss_type = loss_type
        if self._loss_type == "ctc":
            self._loss = CTCLoss(blank=self._blank_idx, zero_infinity=True)
        elif self._loss_type == "rnnt":
            from warprnnt_pytorch import RNNTLoss
            self._loss = RNNTLoss(blank=self._blank_idx)

        self._decoder = ctc_beam_search_decoder_batch
        self._alphabet = Alphabet(os.path.join(
            vocab_path, target_namespace + ".txt"))
        self._beam_size = beam_size
        self._num_processes = beam_size
        self._wer = WordErrorRate()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                source_features: torch.FloatTensor,
                source_lengths: torch.LongTensor,
                target_tokens: Dict[str, torch.LongTensor] = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            metadata containing the original words in the sentence to be tagged under a 'words' key.

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """
        target_mask = get_text_field_mask(target_tokens)
        encoded_features, _, source_lengths = self.encoder(
            source_features, source_lengths)
        batch_size, src_out_len, _ = encoded_features.size()

        logits = self.projection_layer(encoded_features)
        if self._loss_type == "ctc":
            reshaped_logits = logits.view(-1, self.num_classes)
            log_probs = F.log_softmax(reshaped_logits, dim=-1).view([batch_size,
                                                                     src_out_len,
                                                                     self.num_classes])

        output_dict = {}
        target_lengths = get_lengths_from_binary_sequence_mask(target_mask)
        if target_tokens is not None:
            inputs = log_probs if self._loss_type == "ctc" else logits
            loss = self._loss(inputs.transpose(1, 0),
                              target_tokens["tokens"],
                              source_lengths,
                              target_lengths)
            output_dict["loss"] = loss
            if not self.training:
                probs = torch.exp(log_probs)
                batch_beam_results = \
                    self._decoder(probs.tolist(), source_lengths.tolist(),
                                  self._alphabet, self._beam_size,
                                  self._num_processes)
                batch_best_results = [
                    beam_results[0][1] for beam_results in batch_beam_results
                ]
                self._wer(batch_best_results, target_tokens["tokens"].tolist(),
                          target_lengths.tolist())

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple position-wise argmax over each token, converts indices to string labels, and
        adds a ``"tags"`` key to the dictionary with the result.
        """
        all_predictions = output_dict['class_probabilities']
        all_predictions = all_predictions.cpu().data.numpy()
        if all_predictions.ndim == 3:
            predictions_list = [all_predictions[i]
                                for i in range(all_predictions.shape[0])]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        for predictions in predictions_list:
            argmax_indices = numpy.argmax(predictions, axis=-1)
            tags = [self.vocab.get_token_from_index(x, namespace="labels")
                    for x in argmax_indices]
            all_tags.append(tags)
        output_dict['tags'] = all_tags
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if not self.training:
            all_metrics.update(self._wer.get_metric(reset=reset))
        return all_metrics
