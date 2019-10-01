from typing import Dict, List, Tuple, Any
from itertools import groupby

import numpy
from overrides import overrides

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import START_SYMBOL, END_SYMBOL, sanitize
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.attention import LegacyAttention
from allennlp.modules import Attention, Seq2SeqEncoder
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
from allennlp.nn.beam_search import BeamSearch
from allennlp.training.metrics import UnigramRecall, Average, BLEU
from allennlp.nn import InitializerApplicator

from stt.models.awd_rnn import AWDRNN
from stt.training.word_error_rate import WordErrorRate as WER
from stt.modules.losses import OCDLoss, EDOCDLoss, maybe_sample_from_candidates
from stt.modules.losses import target_to_candidates
from stt.models.util import averaging_tensor_of_same_label, remove_sentence_boundaries, char_to_word
from stt.models.util import remove_eos as _remove_eos
from stt.modules.attention import MonotonicAttention, differentiable_average_lagging
from stt.modules.stateful_attention import StatefulAttention
from stt.modules.specaugment import TimeMask, FreqMask
from stt.modules.audio import Delta

@Model.register("old_phn_mocha")
class PhnMoChA(Model):
    """
    This ``SimpleSeq2Seq`` class is a :class:`Model` which takes a sequence, encodes it, and then
    uses the encoded representations to decode another sequence.  You can use this as the basis for
    a neural machine translation system, an abstractive summarization system, or any other common
    seq2seq problem.  The model here is simple, but should be a decent starting place for
    implementing recent models for these tasks.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    source_embedder : ``TextFieldEmbedder``, required
        Embedder for source side sequences
    encoder : ``Seq2SeqEncoder``, required
        The encoder of the "encoder/decoder" model
    max_decoding_steps : ``int``
        Maximum length of decoded sequences.
    target_namespace : ``str``, optional (default = 'tokens')
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    target_embedding_dim : ``int``, optional (default = source_embedding_dim)
        You can specify an embedding dimensionality for the target side. If not, we'll use the same
        value as the source embedder's.
    attention : ``Attention``, optional (default = None)
        If you want to use attention to get a dynamic summary of the encoder outputs at each step
        of decoding, this is the function used to compute similarity between the decoder hidden
        state and encoder outputs.
    attention_function: ``SimilarityFunction``, optional (default = None)
        This is if you want to use the legacy implementation of attention. This will be deprecated
        since it consumes more memory than the specialized attention modules.
    beam_size : ``int``, optional (default = None)
        Width of the beam for beam search. If not specified, greedy decoding is used.
    scheduled_sampling_ratio : ``float``, optional (default = 0.)
        At each timestep during training, we sample a random number between 0 and 1, and if it is
        not less than this value, we use the ground truth labels for the whole batch. Else, we use
        the predictions from the previous time step for the whole batch. If this value is 0.0
        (default), this corresponds to teacher forcing, and if it is 1.0, it corresponds to not
        using target side ground truth labels.  See the following paper for more information:
        `Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. Bengio et al.,
        2015 <https://arxiv.org/abs/1506.03099>`_.
    use_bleu : ``bool``, optional (default = True)
        If True, the BLEU metric will be calculated during validation.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 encoder: Seq2SeqEncoder,
                 input_size: int,
                 target_embedding_dim: int,
                 max_decoding_steps: int,
                 max_decoding_ratio: float = 1.5,
                 dep_parser: Model = None,
                 pos_tagger: Model = None,
                 cmvn: str = 'none',
                 delta: bool = False,
                 acceleration: bool = False,
                 time_mask_width: int = 0,
                 freq_mask_width: int = 0,
                 time_mask_max_ratio: float = 0.0,
                 rnnt_dim: int = 512,
                 dec_layers: int = 1,
                 layerwise_pretraining: List[Tuple[int, int]] = None,
                 cnn: Seq2SeqEncoder = None,
                 train_at_phn_level: bool = False,
                 joint_ctc_ratio: float = 0.0,
                 dep_ratio: float = 0.0,
                 pos_ratio: float = 0.0,
                 ctc_keep_eos: bool = False,
                 attention: Attention = None,
                 attention_function: SimilarityFunction = None,
                 latency_penalty: float = 0.0,
                 loss_type: str = "nll",
                 beam_size: int = 1,
                 target_namespace: str = "tokens",
                 phoneme_target_namespace: str = "phonemes",
                 n_pretrain_ctc_epochs: int = 10,
                 dropout: float = 0.0,
                 sampling_strategy: str = "max",
                 from_candidates: bool = False,
                 scheduled_sampling_ratio: float = 0.,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(PhnMoChA, self).__init__(vocab)
        self._input_size = input_size
        self._target_namespace = target_namespace
        self._phn_target_namespace = phoneme_target_namespace
        self._scheduled_sampling_ratio = scheduled_sampling_ratio
        self._sampling_strategy = sampling_strategy
        self._train_at_phn_level = train_at_phn_level

        self._dep_parser = dep_parser
        self._pos_tagger = pos_tagger

        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self.vocab.get_token_index(
            START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(
            END_SYMBOL, self._target_namespace)

        self._pad_index = self.vocab.get_token_index(
            self.vocab._padding_token, self._target_namespace)  # pylint: disable=protected-access
        self._phn_pad_index = self.vocab.get_token_index(
            self.vocab._padding_token, self._phn_target_namespace)  # pylint: disable=protected-access

        self._bleu = BLEU(
            exclude_indices={self._pad_index, self._end_index, self._start_index})
        self._wer = WER(exclude_indices={
            self._pad_index, self._end_index, self._start_index})
        self._ctc_wer = WER(exclude_indices={
            self._pad_index, self._end_index, self._start_index})
        self._phn_wer = WER(exclude_indices={
            self._phn_pad_index})
        self._unigram_recall = UnigramRecall()
        self._logs = {
            "phn_ctc_loss": Average(),
            "joint_ctc_loss": Average(),
            "att_loss": Average(),
            "dal_loss": (Average() if latency_penalty > 0.0 else None),
            "dep_loss": (Average() if self._dep_parser else None),
            "pos_loss": (Average() if self._pos_tagger else None),
            "tag_loss": (Average() if self._dep_parser else None),
            "arc_loss": (Average() if self._dep_parser else None)
        }

        # At prediction time, we use a beam search to find the most likely sequence of target tokens.
        self._max_decoding_steps = max_decoding_steps
        self._max_decoding_ratio = max_decoding_ratio
        self._beam_search = BeamSearch(
            self._end_index, max_steps=max_decoding_steps, beam_size=beam_size)

        # Encodes the sequence of source embeddings into a sequence of hidden states.
        self._encoder = encoder

        self._cnn = cnn

        num_classes = self.vocab.get_vocab_size(self._target_namespace)
        self._num_classes = num_classes

        # Attention mechanism applied to the encoder output for each step.
        if attention:
            if attention_function:
                raise ConfigurationError("You can only specify an attention module or an "
                                         "attention function, but not both.")
            self._attention = attention
        elif attention_function:
            self._attention = LegacyAttention(attention_function)
        else:
            self._attention = None

        # Dense embedding of vocab words in the target space.
        self._target_embedder = Embedding(num_classes, target_embedding_dim)

        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with the final hidden state of the encoder.
        self._encoder_output_dim = self._encoder.get_output_dim()
        #self._decoder_output_dim = self._encoder_output_dim
        self._decoder_output_dim = target_embedding_dim
        self._dec_layers = dec_layers
        if self._decoder_output_dim != self._encoder_output_dim:
            self.bridge = nn.Linear(
                self._encoder_output_dim, self._dec_layers * self._decoder_output_dim, bias=False)

        if self._attention:
            # If using attention, a weighted average over encoder outputs will be concatenated
            # to the previous target embedding to form the input to the decoder at each
            # time step.
            self._decoder_input_dim = self._decoder_output_dim + target_embedding_dim
            self.att_out = Linear(self._decoder_output_dim + self._encoder_output_dim,
                                  self._decoder_output_dim, bias=True)
        else:
            # Otherwise, the input to the decoder is just the previous target embedding.
            self._decoder_input_dim = target_embedding_dim

        # We'll use an LSTM cell as the recurrent cell that produces a hidden state
        # for the decoder at each time step.
        # TODO (pradeep): Do not hardcode decoder cell type.
        self._decoder = nn.LSTM(self._decoder_input_dim, self._decoder_output_dim,
                                num_layers=self._dec_layers, batch_first=True)

        self._num_phn_classes = self.vocab.get_vocab_size(self._phn_target_namespace)
        self._ctc_projection_layer = nn.Linear(self._encoder_output_dim, self._num_phn_classes)
        self._ctc_loss = nn.CTCLoss(blank=self._phn_pad_index)
        self._n_pretrain_ctc_epochs = n_pretrain_ctc_epochs
        self._ctc_proj_drop = nn.Dropout(p=dropout)

        self._dep_ratio = dep_ratio
        self._pos_ratio = pos_ratio
        self._joint_ctc_ratio = joint_ctc_ratio
        self._loss_type = loss_type
        if self._joint_ctc_ratio > 0.0:
            self._joint_ctc_projection_layer = nn.Linear(self._encoder_output_dim,
                                                         self._num_classes)
            if  loss_type == 'rnnt':
                import warprnnt_pytorch
                self._joint_ctc_loss = warprnnt_pytorch.RNNTLoss(blank=self._pad_index,
                                                                 reduction='mean')

                self._rnnt_recurrency = nn.LSTM(input_size=target_embedding_dim,
                                                hidden_size=self._encoder_output_dim,
                                                num_layers=self._dec_layers,
                                                batch_first=True)
                self.w_enc = Linear(self._encoder_output_dim, rnnt_dim, bias=True)
                self.w_dec = Linear(self._encoder_output_dim, rnnt_dim, bias=False)
                self._rnnt_proj = nn.Linear(rnnt_dim, self._num_classes)
            else:
                self._joint_ctc_loss = nn.CTCLoss(blank=self._pad_index)
            self._joint_ctc_proj_drop = nn.Dropout(p=dropout)
        self._ctc_keep_eos = ctc_keep_eos

        # We project the hidden state from the decoder into the output vocabulary space
        # in order to get log probabilities of each target token, at each time step.
        self._output_projection_layer = Linear(
            self._decoder_output_dim, num_classes)

        self._input_norm = lambda x: x
        if cmvn == 'global':
            self._input_norm = nn.BatchNorm1d(self._input_size)
        elif cmvn == 'utt':
            self._input_norm = nn.InstanceNorm1d(self._input_size)

        self._delta = None
        if delta:
            raise NotImplementedError
            # self._delta = Delta(order=1)
        self._acceleration = None
        if acceleration:
            raise NotImplementedError
            # self._acceleration = Delta(order=2)

        self._epoch_num = float("inf")
        self._layerwise_pretraining = layerwise_pretraining
        try:
            if isinstance(self._encoder, PytorchSeq2SeqWrapper):
                self._num_layers = self._encoder._module.num_layers
            else:
                self._num_layers = self._encoder.num_layers
        except AttributeError:
            self._num_layers = float("inf")

        self._output_layer_num = self._num_layers

        self._loss = None

        self._from_candidates = from_candidates
        if loss_type == "ocd":
            self._loss = OCDLoss(
                self._end_index, 1e-7, 1e-7, 5)
        elif loss_type == "edocd":
            self._loss = EDOCDLoss(
                self._end_index, 1e-7, 1e-7, 5)    

        self._latency_penalty = latency_penalty
        self._target_granularity = self._target_namespace

        self.time_mask = TimeMask(time_mask_width, time_mask_max_ratio)
        self.freq_mask = FreqMask(freq_mask_width)

        initializer(self)

    def take_step(self,
                  last_predictions: torch.Tensor,
                  state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take a decoding step. This is called by the beam search class.

        Parameters
        ----------
        last_predictions : ``torch.Tensor``
            A tensor of shape ``(group_size,)``, which gives the indices of the predictions
            during the last time step.
        state : ``Dict[str, torch.Tensor]``
            A dictionary of tensors that contain the current state information
            needed to predict the next step, which includes the encoder outputs,
            the source mask, and the decoder hidden state and context. Each of these
            tensors has shape ``(group_size, *)``, where ``*`` can be any other number
            of dimensions.

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            A tuple of ``(log_probabilities, updated_state)``, where ``log_probabilities``
            is a tensor of shape ``(group_size, num_classes)`` containing the predicted
            log probability of each class for the next step, for each item in the group,
            while ``updated_state`` is a dictionary of tensors containing the encoder outputs,
            source mask, and updated decoder hidden state and context.

        Notes
        -----
            We treat the inputs as a batch, even though ``group_size`` is not necessarily
            equal to ``batch_size``, since the group may contain multiple states
            for each source sentence in the batch.
        """
        # shape: (group_size, num_classes)
        output_projections, state = self._prepare_output_projections(
            last_predictions, state)

        # shape: (group_size, num_classes)
        class_log_probabilities = F.log_softmax(output_projections, dim=-1)

        return class_log_probabilities, state

    def _rnnt_joint(self, eouts, douts, non_linear=torch.tanh):
        """Combine encoder outputs and prediction network outputs.
        Args:
            eouts (FloatTensor): `[B, T, n_units]`
            douts (FloatTensor): `[B, L, n_units]`
        Returns:
            out (FloatTensor): `[B, T, L, vocab]`
        """
        # broadcast
        eouts = eouts.unsqueeze(2)  # `[B, T, 1, n_units]`
        douts = douts.unsqueeze(1)  # `[B, 1, L, n_units]`
        out = non_linear(self.w_enc(eouts) + self.w_dec(douts))
        out = self._rnnt_proj(out)
        return out

    def _get_ctc_output(self, projection: nn.Module,
                        dropout: nn.Module,
                        num_classes: int,
                        state: Dict[str, torch.Tensor]) -> torch.Tensor:

        logits = projection(dropout(state["encoder_outputs"]))
        batch_size, src_out_len, _ = logits.size()
        reshaped_logits = logits.view(-1, num_classes)
        ctc_log_probs = F.log_softmax(reshaped_logits, dim=-1).view(batch_size,
                                                                    src_out_len,
                                                                    num_classes)
        return ctc_log_probs

    def _get_ctc_loss(self, loss_func: nn.Module,
                      log_probs: torch.Tensor,
                      source_lengths: torch.Tensor,
                      targets: Dict[str, torch.Tensor],
                      remove_sos: bool = False,
                      remove_eos: bool = False,
                      pad_index=0) -> torch.Tensor:

        mask = (targets != pad_index).bool()
        if remove_sos and remove_eos:
            targets, mask = remove_sentence_boundaries(targets, mask)
        elif remove_sos:
            targets = targets[:, 1:]
            mask = mask[:, 1:]
        else:
            raise NotImplementedError

        target_lengths = util.get_lengths_from_binary_sequence_mask(mask)
        ctc_loss = loss_func(log_probs.transpose(1, 0),
                             targets,
                             source_lengths,
                             target_lengths)
        return ctc_loss

    def _ctc_greedy_decode(self, ctc_log_probs: torch.Tensor,
                           source_lengths: torch.LongTensor,
                           pad_index: int) -> List[List[int]]:
        _, raw_predictions = ctc_log_probs.max(dim=-1)
        predictions = [[k for k, g in groupby(prediction[:length]) if k != pad_index]
                       for prediction, length in
                       zip(raw_predictions.tolist(), source_lengths.tolist())]
        return predictions
    def _rnnt_greedy_decode(self, eouts, elens,
                            exclude_eos=False, oracle=False,
                            refs_id=None):
        """Greedy decoding in the inference stage.
        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
            elens (IntTensor): `[B]`
            max_len_ratio (int): maximum sequence length of tokens
            idx2token (): converter from index to token
            exclude_eos (bool): exclude <eos> from hypothesis
            oracle (bool): teacher-forcing mode
            refs_id (list): reference list
            utt_ids (list): utterance id list
            speakers (list): speaker list
        Returns:
            hyps (list): A list of length `[B]`, which contains arrays of size `[L]`
            aw: dummy
        """
        bs = eouts.size(0)

        hyps = []
        for b in range(bs):
            best_hyp_b = []
            # Initialization
            y = elens.new_zeros((1,1)).fill_(self._start_index)
            dout, dstate = self._rnnt_recurrency(self._target_embedder(y), None)

            t = 0
            while t < elens[b]:
                # Pick up 1-best per frame
                out = self._rnnt_joint(eouts[b:b + 1, t:t + 1], dout)
                y = out.squeeze(2).argmax(-1)
                idx = y[0].item()

                # Update prediction network only when predicting non-blank labels
                if idx != self._pad_index:
                    # early stop
                    if idx == self._end_index:
                        if not exclude_eos:
                            best_hyp_b += [idx]
                        break

                    best_hyp_b += [idx]
                    if oracle:
                        raise NotImplementedError
                        y = eouts.new_zeros(1, 1).fill_(refs_id[b, len(best_hyp_b) - 1])
                    dout, dstate = self._rnnt_recurrency(self._target_embedder(y), dstate)
                else:
                    t += 1

            hyps += [best_hyp_b]

        return hyps

    def my_rnnt_greedy_decode(self, eouts: torch.Tensor, elens: torch.Tensor,
                              dlens: torch.Tensor = None):
        """
        Batch decoding for RNN Transducer.

        Parameters
        ----------
        eouts: ``torch.FloatTensor``
           encoder outputs, of shape (batch_size, max_enc_len, hidden_dim).
        elens : ``torch.LongTensor``
           encoder output sequence lengths, of shape (batch_size).
        dlens : ``torch.LongTensor``, optional (default = None)
           largest possible target sequence lengths, of shape (batch_size).
           default to have the same value as elens.

        Returns
        -------
        Dict[str, torch.Tensor]
        """
        bs, max_enc_len, _ = eouts.size()
        if dlens is None:
            dlens = elens.new_full(elens.size(), max_enc_len)
        # Initialize target predictions with the start index.
        # shape: (batch_size,)
        y = elens.new_zeros((bs,1)).fill_(self._start_index)
        dout, dstate = self._rnnt_recurrency(self._target_embedder(y), None)
        # current source position t for each element in a batch.
        cur_enc_indices = elens.new_zeros((bs,))
        end_indices = elens
        # decoding finished when each current source position equals to the
        # end_indices.
        finished = (cur_enc_indices == end_indices)
        batch_indices = torch.arange(bs)
        # total steps traversed in the latttices, including horizontal steps
        # (blanks).
        hyp_lens = elens.new_zeros((bs,))
        hypotheses = []
        is_pad = lambda y, shape, tensor: y.view(*shape) \
            .expand_as(tensor) == self._pad_index
        while not finished.all():
            hyp_lens += (~finished).long()
            # for finished batch elements with cur_enc_indices == end_indices,
            # we clamped it to prevent IndexError.
            clamped_indices = torch.min(cur_enc_indices, end_indices - 1)
            out = self._rnnt_joint(eouts[batch_indices, clamped_indices].unsqueeze(1), dout)
            y = out.argmax(-1).squeeze(1)
            hypotheses.append(y.squeeze(1))
            maybe_dout, maybe_dstate = self._rnnt_recurrency(self._target_embedder(y),
                                                             dstate)
            # state will not be updated for those who output blanks (horizontal
            # steps).
            dout = torch.where(is_pad(y, (-1, 1, 1), dout), dout, maybe_dout)
            dstate = tuple(torch.where(is_pad(y, (1, -1, 1), dstate[i]), dstate[i], maybe_dstate[i])
                           for i in range(len(maybe_dstate)))
            cur_enc_indices = torch.min(end_indices,
                                        cur_enc_indices + (y.view(-1) == self._pad_index).long())
            finished = (cur_enc_indices == (end_indices)) | (y.view(-1) == self._end_index) | \
                        (hyp_lens > (elens + dlens))

        hypotheses = torch.stack(hypotheses, dim=-1)
        hypotheses = [list(filter(lambda idx: idx != self._pad_index, hypothesis[:hyp_len]))
                      for hypothesis, hyp_len in zip(hypotheses, hyp_lens)]
        return hypotheses

    @overrides
    def forward(self,  # type: ignore
                source_features: torch.FloatTensor,
                source_lengths: torch.LongTensor,
                target_tokens: Dict[str, torch.LongTensor] = None,
                words: Dict[str, torch.LongTensor] = None,
                segments: torch.LongTensor = None,
                pos_tags: torch.LongTensor = None,
                head_tags: torch.LongTensor = None,
                head_indices: torch.LongTensor = None,
                epoch_num: int = None,
                dataset: str = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Make foward pass with decoder logic for producing the entire target sequence.

        Parameters
        ----------
        source_tokens : ``Dict[str, torch.LongTensor]``
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.
        target_tokens : ``Dict[str, torch.LongTensor]``, optional (default = None)
           Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
           target tokens are also represented as a `TextField`.

        Returns
        -------
        Dict[str, torch.Tensor]
        """
        output_dict = {}
        if dataset is not None:
            self._target_granularity = dataset[0]

        if epoch_num is not None:
            self._epoch_num = epoch_num[0]
        self.set_output_layer_num()

        source_mask = util.get_mask_from_sequence_lengths(source_lengths,
                                                          source_features.size(1)).bool()
        source_features = self._input_norm(
            source_features.transpose(-2, -1)).transpose(-2, -1)
        source_features = self.time_mask(source_features, source_mask)
        source_features = self.freq_mask(source_features, source_mask)

        source_features = source_features.masked_fill(
            ~source_mask.unsqueeze(-1).expand_as(source_features), 0.0)
        state = self._encode(source_features, source_lengths)
        source_lengths = util.get_lengths_from_binary_sequence_mask(state["source_mask"])
        target_tokens["mask"] = (target_tokens[self._target_namespace] != self._pad_index).bool()

        if self._phn_target_namespace in self._target_granularity or self._train_at_phn_level:
            targets = target_tokens[self._phn_target_namespace]
            ctc_log_probs = self._get_ctc_output(self._ctc_projection_layer,
                                                 self._ctc_proj_drop,
                                                 self._num_phn_classes,
                                                 state)
            ctc_loss = self._get_ctc_loss(self._ctc_loss, ctc_log_probs,
                                          source_lengths, targets,
                                          pad_index=self._phn_pad_index)
            output_dict["phn_ctc_loss"] = ctc_loss
            if not self._train_at_phn_level:
                predictions = self._ctc_greedy_decode(ctc_log_probs, source_lengths,
                                                      self._phn_pad_index)
                self._phn_wer(predictions, targets)

        if self._joint_ctc_ratio > 0.0:
            targets = target_tokens[self._target_namespace]
            if self._loss_type == "rnnt":
                rnnt_dec_outs, _ = self._rnnt_recurrency(self._target_embedder(targets), None)
                rnnt_logits = self._rnnt_joint(state["encoder_outputs"], rnnt_dec_outs)
                rnnt_log_probs = F.log_softmax(rnnt_logits, dim=-1)
                target_mask = util.get_text_field_mask(target_tokens)
                relevant_mask = target_mask[:, 1:]
                relevant_targets = targets[:, 1:]
                target_lengths = util.get_lengths_from_binary_sequence_mask(relevant_mask)
                joint_ctc_loss = self._joint_ctc_loss(rnnt_log_probs, relevant_targets.int(),
                                                      source_lengths.int(), target_lengths.int())
                # with torch.no_grad():
                #     ctc_predictions = self.my_rnnt_greedy_decode(state["encoder_outputs"],
                #                                                  source_lengths,
                #                                                  target_lengths)
            else:
                joint_ctc_log_probs = self._get_ctc_output(self._joint_ctc_projection_layer,
                                                           self._joint_ctc_proj_drop,
                                                           self._num_classes,
                                                           state)
                joint_ctc_loss = self._get_ctc_loss(self._joint_ctc_loss, joint_ctc_log_probs,
                                                    source_lengths, targets, remove_sos=True,
                                                    remove_eos=(not self._ctc_keep_eos),
                                                    pad_index=self._pad_index)

                ctc_predictions = self._ctc_greedy_decode(joint_ctc_log_probs,
                                                          source_lengths,
                                                          self._pad_index)
                self._ctc_wer(ctc_predictions, targets)
            output_dict["joint_ctc_loss"] = joint_ctc_loss

        if target_tokens and self._joint_ctc_ratio < 1.0 and \
            self._target_granularity == self._target_namespace:
            targets = target_tokens[self._target_namespace]
            output_dict["target_tokens"] = targets
            target_mask = util.get_text_field_mask(target_tokens)
            if self._train_at_phn_level:
                state = self._get_phn_level_representations(
                    state["encoder_outputs"].detach().requires_grad_(True),
                    state["source_mask"],
                    ctc_log_probs)

            state = self._init_decoder_state(state)
            output_dict.update(self._forward_loop(state, target_tokens))
            self._wer(output_dict["predictions"], targets)

            if self._dep_parser or self._pos_tagger:
                relevant_mask = target_mask[:, 1:]
                attention_contexts, _ = _remove_eos(output_dict["attention_contexts"],
                                                    relevant_mask)
                if segments is not None:
                    segments, _ = remove_sentence_boundaries(segments, target_mask)
                    attention_contexts, _ = \
                        char_to_word(attention_contexts, segments)
                contexts = {"tokens": attention_contexts}
                if self._dep_parser:
                    parser_outputs = self._dep_parser(contexts, pos_tags, metadata,
                                                      head_tags, head_indices)
                    parser_outputs["dep_loss"] = parser_outputs.pop("loss")
                    output_dict.update(parser_outputs)
                if self._pos_tagger:
                    tagger_outputs = self._pos_tagger(contexts, pos_tags, metadata)
                    tagger_outputs["pos_loss"] = tagger_outputs.pop("loss")
                    output_dict.update(tagger_outputs)

        self._update_metrics(output_dict)

        if not self.training:
            if self._target_granularity == self._target_namespace:
                if self._joint_ctc_ratio < 1.0:
                    state = self._init_decoder_state(state)
                    predictions = self._forward_beam_search(state)
                    output_dict.update(predictions)
                    if target_tokens:
                        targets = target_tokens[self._target_namespace]
                        # shape: (batch_size, beam_size, max_sequence_length)
                        top_k_predictions = output_dict["predictions"]
                        # shape: (batch_size, max_predicted_sequence_length)
                        best_predictions = top_k_predictions[:, 0, :]
                        self._bleu(best_predictions, targets)
                        self._wer(best_predictions, targets)
                        self._unigram_recall(top_k_predictions, targets,
                                             target_mask.long())
                if self._joint_ctc_ratio > 0.0:
                    with torch.no_grad():
                        if self._loss_type == 'rnnt':
                            ctc_predictions = self.my_rnnt_greedy_decode(state["encoder_outputs"],
                                                                        source_lengths)
                        else:
                            ctc_predictions = self._ctc_greedy_decode(joint_ctc_log_probs,
                                                                    source_lengths,
                                                                    self._pad_index)
                    if target_tokens:
                        self._ctc_wer(ctc_predictions, target_tokens[self._target_namespace])
                    output_dict["ctc_predictions"] = ctc_predictions

            elif self._phn_target_namespace in self._target_granularity:
                predictions = self._ctc_greedy_decode(ctc_log_probs, source_lengths,
                                                      self._phn_pad_index)
                self._phn_wer(predictions, target_tokens[self._phn_target_namespace])
            else:
                raise NotImplementedError

        if self.training:
            att_ratio = 1 - self._joint_ctc_ratio - self._dep_ratio - self._pos_ratio
            output_dict = self._collect_losses(output_dict,
                                               joint_ctc=self._joint_ctc_ratio,
                                               att=att_ratio,
                                               dal=self._latency_penalty,
                                               dep=self._dep_ratio,
                                               pos=self._pos_ratio)
            if torch.isnan(output_dict["loss"]).any() or \
                    (torch.abs(output_dict["loss"]) == float('inf')).any():
                for key, _ in output_dict.items():
                    if "loss" in key:
                        output_dict[key] = output_dict[key].new_zeros(size=(), requires_grad=True).clone()

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Finalize predictions.

        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``forward`` method.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        def _indices_to_tokens(indices):
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]
            predicted_tokens = [self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                                for x in indices]
            return predicted_tokens

        def _decode_predictions(input_key: str, output_key: str, beam=False):
            if input_key in output_dict:
                if beam:
                    all_predicted_tokens = [list(map(_indices_to_tokens, beams)) 
                                            for beams in sanitize(output_dict[input_key])]
                else:
                    all_predicted_tokens = list(map(_indices_to_tokens, sanitize(output_dict[input_key])))
                output_dict[output_key] = all_predicted_tokens

        _decode_predictions("predictions", "predicted_tokens", beam=True)
        _decode_predictions("ctc_predictions", "ctc_predicted_tokens")

        return output_dict

    def _encode(self,
                source_features: torch.FloatTensor,
                source_lengths: torch.LongTensor) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        if self._cnn is not None:
            source_features, source_lengths = self._cnn(
                source_features, source_lengths)
        if not isinstance(self._encoder, AWDRNN):
            source_mask = util.get_mask_from_sequence_lengths(
                source_lengths, source_features.size(1))
            encoder_outputs = self._encoder(source_features, source_mask)
        else:
            encoder_outputs, _, source_lengths = self._encoder(
                source_features, source_lengths, self._output_layer_num)
            source_mask = util.get_mask_from_sequence_lengths(
                source_lengths, encoder_outputs.size(1))
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        return {
            "source_mask": source_mask,
            "encoder_outputs": encoder_outputs
        }

    def _get_phn_level_representations(self,
                                       features: torch.FloatTensor,
                                       mask: torch.BoolTensor,
                                       phn_log_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        phn_enc_outs, segment_lengths = averaging_tensor_of_same_label(features,
                                                                       phn_log_probs,
                                                                       mask=mask)
        state = {"encoder_outputs": phn_enc_outs,
                 "source_mask": util.get_mask_from_sequence_lengths(
                     segment_lengths, int(max(segment_lengths)))
                }
        return state

    def _init_decoder_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = state["source_mask"].size(0)
        # shape: (batch_size, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]
        source_mask = state["source_mask"]
        final_encoder_output = util.get_final_encoder_states(
            encoder_outputs,
            source_mask,
            self._encoder.is_bidirectional())
        if self._encoder_output_dim != self._dec_layers * self._decoder_output_dim:
            final_encoder_output = self.bridge(final_encoder_output)
        initial_decoder_input = final_encoder_output.view(-1, self._dec_layers,
                                                          self._decoder_output_dim) \
                                                          .contiguous()
        # Initialize the decoder hidden state with the final output of the encoder.
        # shape: (batch_size, decoder_output_dim)
        state["decoder_hidden"] = initial_decoder_input
        state["decoder_output"] = initial_decoder_input[:, 0]
        # shape: (batch_size, decoder_output_dim)
        state["decoder_context"] = encoder_outputs.new_zeros(
            batch_size, self._dec_layers, self._decoder_output_dim)
        state["attention"] = None
        if isinstance(self._attention, StatefulAttention):
            state["att_keys"], state["att_values"] = \
                self._attention.init_state(encoder_outputs)

        return state

    def _forward_loop(self,
                      state: Dict[str, torch.Tensor],
                      target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        """
        Make forward pass during training or do greedy search during prediction.

        Notes
        -----
        We really only use the predictions from the method to test that beam search
        with a beam size of 1 gives the same results.
        """
        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        batch_size = source_mask.size()[0]

        candidates = None

        if target_tokens:
            # shape: (batch_size, max_target_sequence_length)
            targets = target_tokens[self._target_namespace]

            _, target_sequence_length = targets.size()

            if self._loss is not None:
                candidates = target_to_candidates(
                    targets, self._num_classes, ignore_indices=[self._pad_index,
                                                                self._start_index,
                                                                self._end_index])

            # The last input from the target is either padding or the end symbol.
            # Either way, we don't have to process it.
            if isinstance(self._loss, EDOCDLoss):
                num_decoding_steps = int(target_sequence_length * self._max_decoding_ratio) - 1
            else:
                num_decoding_steps = target_sequence_length - 1
        else:
            num_decoding_steps = self._max_decoding_steps

        # Initialize target predictions with the start index.
        # shape: (batch_size,)
        last_predictions = source_mask.new_full(
            (batch_size,), fill_value=self._start_index)

        step_logits: List[torch.Tensor] = []
        step_predictions: List[torch.Tensor] = []
        step_attns: List[torch.Tensor] = []
        step_attn_cxts: List[torch.Tensor] = []

        for timestep in range(num_decoding_steps):
            if self.training and torch.rand(1).item() < self._scheduled_sampling_ratio:
                # Use gold tokens at test time and at a rate of 1 - _scheduled_sampling_ratio
                # during training.
                # shape: (batch_size,)
                input_choices = last_predictions
            elif self._loss is not None:
                # shape: (batch_size,)
                input_choices = last_predictions
            elif not target_tokens:
                # shape: (batch_size,)
                input_choices = last_predictions
            else:
                # shape: (batch_size,)
                input_choices = targets[:, timestep]

            # shape: (batch_size, num_classes)
            output_projections, state = self._prepare_output_projections(
                input_choices, state)

            # list of tensors, shape: (batch_size, 1, num_classes)
            step_logits.append(output_projections.unsqueeze(1))

            # list of tensors, shape: (batch_size, 1, num_encoding_steps)
            if self._attention:
                step_attns.append(state["attention"].unsqueeze(1))
                step_attn_cxts.append(state["attention_contexts"].unsqueeze(1))

            # shape: (batch_size, num_classes)
            class_probabilities = F.softmax(output_projections, dim=-1)

            # shape (predicted_classes): (batch_size,)

            predicted_classes = maybe_sample_from_candidates(class_probabilities,
                                                             candidates=(candidates
                                                                         if self._from_candidates
                                                                         else None),
                                                             strategy=self._sampling_strategy)

            # shape (predicted_classes): (batch_size,)
            last_predictions = predicted_classes

            step_predictions.append(last_predictions.unsqueeze(1))

        # shape: (batch_size, num_decoding_steps)
        predictions = torch.cat(step_predictions, 1)

        output_dict = {
            "predictions": predictions,
        }

        # shape: (batch_size, num_decoding_steps, num_encoding_steps)
        if self._attention:
            output_dict["attentions"] = torch.cat(step_attns, 1)
            output_dict["attention_contexts"] = torch.cat(step_attn_cxts, 1)

        if target_tokens:
            # shape: (batch_size, num_decoding_steps, num_classes)
            logits = torch.cat(step_logits, 1)

            # Compute loss.
            target_mask = util.get_text_field_mask(target_tokens)
            loss = self._get_loss(logits, predictions,
                                  targets, target_mask, candidates)

            output_dict["att_loss"] = loss

            if self._latency_penalty > 0.0:
                DAL = differentiable_average_lagging(output_dict["attentions"], source_mask,
                                                     target_mask[:, 1:])
                output_dict["dal"] = DAL

        return output_dict

    def _forward_beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Make forward pass during prediction using a beam search."""
        batch_size = state["source_mask"].size()[0]
        start_predictions = state["source_mask"].new_full(
            (batch_size,), fill_value=self._start_index)

        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        all_top_k_predictions, log_probabilities = self._beam_search.search(
            start_predictions, state, self.take_step)

        output_dict = {
            "class_log_probabilities": log_probabilities,
            "predictions": all_top_k_predictions
        }
        return output_dict

    def _prepare_output_projections(self,
                                    last_predictions: torch.Tensor,
                                    state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:  # pylint: disable=line-too-long
        """
        Decode current state and last prediction to produce produce projections
        into the target space, which can then be used to get probabilities of
        each target token for the next step.

        Inputs are the same as for `take_step()`.
        """
        # shape: (group_size, decoder_output_dim)
        decoder_hidden = state["decoder_hidden"]

        # shape: (group_size, decoder_output_dim)
        decoder_context = state["decoder_context"]

        # shape: (group_size, decoder_output_dim)
        decoder_output = state["decoder_output"]

        attention = state["attention"]

        # shape: (group_size, target_embedding_dim)
        embedded_input = self._target_embedder(last_predictions)

        # shape: (group_size, decoder_output_dim + target_embedding_dim)
        decoder_input = torch.cat((embedded_input, decoder_output), -1)

        # shape (decoder_hidden): (batch_size, decoder_output_dim)
        # shape (decoder_context): (batch_size, decoder_output_dim)
        outputs, (decoder_hidden, decoder_context) = self._decoder(
            decoder_input.unsqueeze(1),
            (decoder_hidden.transpose(1, 0).contiguous(),
             decoder_context.transpose(1, 0).contiguous()))

        decoder_hidden = decoder_hidden.transpose(1, 0).contiguous()
        decoder_context = decoder_context.transpose(1, 0).contiguous()
        outputs = outputs.squeeze(1)
        if self._attention:
            # shape: (group_size, encoder_output_dim)
            attended_output, attention = self._prepare_attended_output(outputs, state)

            # shape: (group_size, decoder_output_dim)
            decoder_output = torch.tanh(
                self.att_out(torch.cat((attended_output, outputs), -1))
            )
            state["attention"] = attention
            state["attention_contexts"] = attended_output

        else:
            # shape: (group_size, target_embedding_dim)
            decoder_output = outputs

        state["decoder_hidden"] = decoder_hidden
        state["decoder_context"] = decoder_context
        state["decoder_output"] = decoder_output

        # shape: (group_size, num_classes)
        output_projections = self._output_projection_layer(decoder_output)

        return output_projections, state

    def _prepare_attended_output(self,
                                 decoder_hidden_state: torch.Tensor,
                                 state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply attention over encoder outputs and decoder state."""
        # Ensure mask is also a FloatTensor. Or else the multiplication within
        # attention will complain.
        # shape: (batch_size, max_input_sequence_length)

        encoder_outputs = state["encoder_outputs"]
        source_mask = state["source_mask"]
        prev_attention = state["attention"]
        att_keys = state["att_keys"]
        att_values = state["att_values"]

        # shape: (batch_size, max_input_sequence_length)
        mode = "soft" if self.training else "hard"
        if isinstance(self._attention, MonotonicAttention):
            encoder_outs: Dict[str, torch.Tensor] = {
                "value": state["encoder_outputs"],
                "mask": state["source_mask"]
            }

            monotonic_attention, chunk_attention = self._attention(
                encoder_outs, decoder_hidden_state, prev_attention, mode=mode)
            # shape: (batch_size, encoder_output_dim)
            attended_output = util.weighted_sum(
                encoder_outputs, chunk_attention)
            attention = monotonic_attention
        elif isinstance(self._attention, StatefulAttention):
            attended_output, attention = self._attention(decoder_hidden_state,
                                                         att_keys, att_values, source_mask)
        else:
            attention = self._attention(
                decoder_hidden_state, source_mask)
            attended_output = util.weighted_sum(
                encoder_outputs, attention)

        return attended_output, attention

    # @staticmethod
    def _get_loss(self,
                  logits: torch.FloatTensor,
                  predictions: torch.LongTensor,
                  targets: torch.LongTensor,
                  target_mask: torch.LongTensor,
                  candidates: torch.LongTensor = None) -> torch.Tensor:
        """
        Compute loss.

        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.

        The length of ``targets`` is expected to be greater than that of ``logits`` because the
        decoder does not need to compute the output corresponding to the last timestep of
        ``targets``. This method aligns the inputs appropriately to compute the loss.

        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        # shape: (batch_size, num_decoding_steps)
        relevant_targets = targets[:, 1:].contiguous()

        # shape: (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()

        if self._loss is not None:
            if isinstance(self._loss, OCDLoss) or isinstance(self._loss, EDOCDLoss):
                self._loss.update_temperature(self._epoch_num)

            if isinstance(self._loss, EDOCDLoss):
                log_probs = F.log_softmax(logits, dim=-1)
                return self._loss(log_probs, predictions, relevant_targets, relevant_mask)
            else:
                raise NotImplementedError

        return util.sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)

    def _collect_losses(self,
                        output_dict: Dict[str, torch.Tensor],
                        phn_ctc: float = 1.0,
                        joint_ctc: float = 1.0,
                        att: float = 1.0,
                        dep: float = 1.0,
                        pos: float = 1.0,
                        dal: float = 1.0) -> torch.Tensor:
        loss = 0.0
        if "phn_ctc_loss" in output_dict:
            loss += phn_ctc * output_dict["phn_ctc_loss"]
        if "joint_ctc_loss" in output_dict:
            loss += joint_ctc * output_dict["joint_ctc_loss"]
        if "att_loss" in output_dict:
            loss += att * output_dict["att_loss"]
        if "dep_loss" in output_dict:
            loss += dep * output_dict["dep_loss"]
        if "pos_loss" in output_dict:
            loss += pos * output_dict["pos_loss"]
        if "dal" in output_dict:
            loss += dal * output_dict["dal"]

        output_dict["loss"] = loss
        return output_dict

    def _update_metrics(self, output_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        for key, track_func in self._logs.items():
            try:
                value = output_dict[key]
                value = value.item() if isinstance(value, torch.Tensor) else value
                track_func(value)
            except KeyError:
                continue

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        all_metrics: Dict[str, float] = {}
        if self._phn_target_namespace in self._target_granularity:
            all_metrics["phn_ctc_loss"] = self._logs["phn_ctc_loss"].get_metric(reset=reset)
            all_metrics["phn_wer"] = self._phn_wer.get_metric(reset=reset)
        if self._target_granularity == self._target_namespace:
            all_metrics["att_loss"] = self._logs["att_loss"].get_metric(reset=reset)
            all_metrics["att_wer"] = self._wer.get_metric(reset=reset)
        if self._joint_ctc_ratio > 0.0:
            all_metrics["joint_ctc_loss"] = self._logs["joint_ctc_loss"].get_metric(reset=reset)
            all_metrics["ctc_wer"] = self._ctc_wer.get_metric(reset=reset)
        if self._logs["dal_loss"] is not None:
            all_metrics["dal_loss"] = self._logs["dal_loss"].get_metric(reset=reset)
        if self._logs["pos_loss"] is not None:
            all_metrics["pos_loss"] = self._logs["pos_loss"].get_metric(reset=reset)
        if self._logs["dep_loss"] is not None:
            all_metrics["dep_loss"] = self._logs["dep_loss"].get_metric(reset=reset)

        if not self.training:
            all_metrics.update(self._bleu.get_metric(reset=reset))
            all_metrics["UR"] = self._unigram_recall.get_metric(reset=reset)
            if self._dep_parser:
                all_metrics.update(self._dep_parser.get_metrics(reset=reset))
            if self._pos_tagger:
                all_metrics.update(self._pos_tagger.get_metrics(reset=reset))
        return all_metrics

    def set_output_layer_num(self):
        output_layer_num = self._num_layers
        if self._layerwise_pretraining is not None:
            for epoch, layer_num in self._layerwise_pretraining:
                if self._epoch_num < epoch:
                    break
                output_layer_num = layer_num
        self._output_layer_num = output_layer_num
        return output_layer_num
