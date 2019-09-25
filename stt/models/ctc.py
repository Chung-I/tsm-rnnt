from typing import Dict, Optional, List, Tuple, Any, Callable
from itertools import groupby

import os
import numpy
from overrides import overrides
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CTCLoss
from torchaudio.transforms import MelSpectrogram
import itertools

from allennlp.common.util import START_SYMBOL, END_SYMBOL, sanitize
from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.attention import Attention
from allennlp.training.metrics import Average, Metric, BLEU

from stt.models.util import remove_sentence_boundaries, list_to_tensor
from stt.training.word_error_rate import WordErrorRate as WER
from stt.models.util import prepare_attended_output_factory, maybe_update_state
from stt.modules.residual_lstm import ResidualLSTM

@Model.register("ctc")
class CTCLayer(Model):
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
                 loss_ratio: float = 1.0,
                 remove_sos: bool = True,
                 remove_eos: bool = False,
                 target_namespace: str = "tokens",
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(CTCLayer, self).__init__(vocab, regularizer)
        self.loss_ratio = loss_ratio
        self._remove_sos = remove_sos
        self._remove_eos = remove_eos
        self._target_namespace = target_namespace
        self._num_classes = self.vocab.get_vocab_size(target_namespace)
        self._pad_index = self.vocab.get_token_index(
            DEFAULT_PADDING_TOKEN, self._target_namespace)
        self._loss = CTCLoss(blank=self._pad_index)
        self._start_index = self.vocab.get_token_index(
            START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(
            END_SYMBOL, self._target_namespace)
        exclude_indices = {self._pad_index, self._end_index, self._start_index}
        self._wer: Metric = WER(exclude_indices=exclude_indices)
        self._bleu: Metric = BLEU(exclude_indices=exclude_indices)
        self._dal: Metric = Average()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                logits: torch.FloatTensor,
                source_lengths: torch.LongTensor,
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        batch_size, src_out_len, _ = logits.size()
        reshaped_logits = logits.view(-1, self._num_classes)
        log_probs = F.log_softmax(reshaped_logits, dim=-1).view(batch_size,
                                                                src_out_len,
                                                                self._num_classes)
        loss = self._get_loss(log_probs, target_tokens, source_lengths)
        predictions = self._greedy_decode(log_probs, source_lengths)
        relevant_targets = target_tokens[self._target_namespace][:, 1:]
        self._wer(predictions, relevant_targets)
        prediction_tensor = list_to_tensor(predictions, relevant_targets) 
        self._bleu(prediction_tensor, relevant_targets)

        output_dict: Dict[str, torch.FloatTensor] = {}
        output_dict["predictions"] = predictions
        output_dict["loss"] = loss
        return output_dict

    def _get_loss(self, log_probs: torch.FloatTensor, target_tokens: Dict[str, torch.LongTensor],
                  source_lengths: torch.LongTensor) -> torch.FloatTensor:
        targets = target_tokens[self._target_namespace]
        mask = (targets != self._pad_index).bool()
        if self._remove_sos and self._remove_eos:
            targets, mask = remove_sentence_boundaries(targets, mask)
        elif self._remove_sos:
            targets = targets[:, 1:]
            mask = mask[:, 1:]
        else:
            raise NotImplementedError

        target_lengths = util.get_lengths_from_binary_sequence_mask(mask)
        loss = self._loss(log_probs.transpose(1, 0),
                          targets,
                          source_lengths,
                          target_lengths)

        return loss

    def _greedy_decode(self, log_probs: torch.Tensor,
                       source_lengths: torch.LongTensor) -> List[List[int]]:
        _, raw_predictions = log_probs.max(dim=-1)
        predictions = [[k for k, g in groupby(prediction[:length]) if k != self._pad_index]
                       for prediction, length in
                       zip(raw_predictions.tolist(), source_lengths.tolist())]
        return predictions

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        all_metrics["wer"] = self._wer.get_metric(reset=reset)
        all_metrics.update(self._bleu.get_metric(reset=reset))
        return all_metrics

@Model.register("rnnt")
class RNNTLayer(Model):
    def __init__(self, vocab: Vocabulary,
                 input_size: int,
                 hidden_size: int,
                 loss_ratio: float = 1.0,
                 recurrency: nn.LSTM = None,
                 residual: bool = False,
                 num_layers: int = None,
                 remove_sos: bool = True,
                 remove_eos: bool = False,
                 target_embedder: Embedding = None,
                 target_embedding_dim: int = None,
                 target_namespace: str = "tokens",
                 attention: Attention = None,
                 slow_decode: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(RNNTLayer, self).__init__(vocab, regularizer)
        import warprnnt_pytorch
        self.loss_ratio = loss_ratio
        self._remove_sos = remove_sos
        self._remove_eos = remove_eos
        self._hidden_size = hidden_size
        self._slow_decode = slow_decode
        self._target_namespace = target_namespace
        self._num_classes = self.vocab.get_vocab_size(target_namespace)
        self._pad_index = self.vocab.get_token_index(
            DEFAULT_PADDING_TOKEN, self._target_namespace)
        self._start_index = self.vocab.get_token_index(
            START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(
            END_SYMBOL, self._target_namespace)

        self._loss = warprnnt_pytorch.RNNTLoss(blank=self._pad_index,
                                               reduction='mean')

        self._attention = attention
        self._input_size = target_embedding_dim
        self._att_out = None
        self.joint_network = self._joint
        self._nonlinear = torch.tanh
        if attention:
            self.joint_network = self._attn_joint
            self._input_size = target_embedding_dim + hidden_size
            self._att_out = nn.Linear(self._input_size, hidden_size, bias=True)
            self._init_hidden = nn.Parameter(torch.randn([1, hidden_size]))
        self._decoder = recurrency
        if self._decoder is None:
            if residual:
                self._decoder = nn.LSTM(input_size=self._input_size,
                                        hidden_size=hidden_size,
                                        num_layers=num_layers,
                                        batch_first=True)
            else:
                self._decoder = ResidualLSTM(input_size=self._input_size,
                                             hidden_size=hidden_size,
                                             num_layers=num_layers)
        num_directions = 2 if self._decoder.bidirectional else 1
        self._num_layers_dirs = self._decoder.num_layers * num_directions

        self._target_embedder = target_embedder or Embedding(self._num_classes,
                                                             target_embedding_dim)
        self.w_enc = nn.Linear(input_size, hidden_size, bias=True)
        self.w_dec = nn.Linear(input_size, hidden_size, bias=False)
        self._proj = nn.Linear(hidden_size, self._num_classes)

        self._prepare_attended_output = prepare_attended_output_factory(self)

        exclude_indices = {self._pad_index, self._end_index, self._start_index}
        self._wer: Metric = WER(exclude_indices=exclude_indices)
        self._bleu: Metric = BLEU(exclude_indices=exclude_indices)
        self._dal = Average()

        initializer(self)

    def set_projection_layer(self, projection_layer: nn.Module) -> None:
        self._proj = projection_layer

    @overrides
    def forward(self,  # type: ignore
                state: Dict[str, torch.Tensor],
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        output_dict: Dict[str, torch.FloatTensor] = {}
        targets = target_tokens[self._target_namespace]
        if self._attention:
            state["att_keys"], state["att_values"] = \
                self._attention.init_state(state["encoder_outputs"])
        self._init_state(state, attn=(self.joint_network == self._attn_joint))
        logits, _ = self.joint_network(state, targets)
        source_lengths = util.get_lengths_from_binary_sequence_mask(state["source_mask"])
        log_probs = F.log_softmax(logits, dim=-1)
        loss = self._get_loss(log_probs, target_tokens, source_lengths)[0]
        if not self.training:
            self._init_state(state, attn=(self.joint_network == self._attn_joint))
            predictions = self._greedy_decode(state, source_lengths)
            relevant_targets = targets[:, 1:]
            self._wer(predictions, relevant_targets)
            prediction_tensor = list_to_tensor(predictions, relevant_targets)
            self._bleu(prediction_tensor, relevant_targets)
            output_dict["predictions"] = predictions

        output_dict["loss"] = loss
        return output_dict

    def _init_state(self, state: Dict[str, torch.Tensor], attn: bool = False):
        enc_outs = state["encoder_outputs"]
        batch_size, enc_steps, _ = enc_outs.size()
        fields = ["decoder_hidden", "decoder_context"]
        for field in fields:
            state[field] = enc_outs.new_zeros(batch_size, self._num_layers_dirs, self._hidden_size)
            # state[f"prev_{field}"] = enc_outs.new_zeros(batch_size, self._num_layers_dirs, self._hidden_size)
        if attn:
            state["decoder_output"] = self._init_hidden.expand(batch_size, -1)
            for field in fields + ["decoder_output"]:
                state[field] = state[field].repeat_interleave(enc_steps, dim=0)\
                    .view(batch_size, enc_steps, *state[field].size()[1:])

    def _get_loss(self, log_probs: torch.FloatTensor, target_tokens: Dict[str, torch.LongTensor],
                  source_lengths: torch.LongTensor) -> torch.FloatTensor:
        targets = target_tokens[self._target_namespace]
        mask = (targets != self._pad_index).bool()

        relevant_mask = mask[:, 1:]
        relevant_targets = targets[:, 1:]
        target_lengths = util.get_lengths_from_binary_sequence_mask(relevant_mask)
        loss = self._loss(log_probs, relevant_targets.int(),
                          source_lengths.int(), target_lengths.int())
        return loss

    def _joint(self, state: Dict[str, torch.Tensor], targets: torch.Tensor,
               enc_indices: torch.Tensor = None):
        """Combine encoder outputs and prediction network outputs.
        Args:
            eouts (FloatTensor): `[B, T, n_units]`
            douts (FloatTensor): `[B, L, n_units]`
        Returns:
            out (FloatTensor): `[B, T, L, vocab]`
        """
        # broadcast
        eouts = state["encoder_outputs"]
        if enc_indices is not None:
            eouts = eouts[torch.arange(enc_indices.size(0)), enc_indices].unsqueeze(1)
        state["prev_decoder_hidden"] = state["decoder_hidden"]
        state["prev_decoder_context"] = state["decoder_context"]
        decoder_hidden = state["decoder_hidden"].transpose(1, 0).contiguous()
        decoder_context = state["decoder_context"].transpose(1, 0).contiguous()
        douts, (decoder_hidden, decoder_context) = \
            self._decoder(self._target_embedder(targets), (decoder_hidden, decoder_context))
        state["decoder_hidden"] = decoder_hidden.transpose(1, 0)
        state["decoder_context"] = decoder_context.transpose(1, 0)
        eouts = eouts.unsqueeze(2)  # `[B, T, 1, n_units]`
        douts = douts.unsqueeze(1)  # `[B, 1, L, n_units]`
        out = self._nonlinear(self.w_enc(eouts) + self.w_dec(douts))
        out = self._proj(out)
        output_dict: Dict[str, torch.Tensor] = {}
        output_dict["logits"] = out
        output_dict["states"] = state
        return output_dict

    def _attn_joint(self, state: Dict[str, torch.Tensor], targets: torch.Tensor,
                    enc_indices: torch.Tensor = None):
        _, dec_steps = targets.size()
        step_outputs: List[torch.Tensor] = []
        step_attns: List[torch.Tensor] = []
        for i in range(dec_steps):
            decoder_output, state = self._prepare_decoder_output(targets[:, i], state)
            step_outputs.append(decoder_output)
            step_attns.append(state["attentions"])
        decoder_outputs = torch.stack(step_outputs, dim=2)
        logits = self._proj(decoder_outputs)
        attentions = torch.stack(step_attns, dim=1)
        if enc_indices is not None:
            logits = logits[torch.arange(enc_indices.size(0)), enc_indices]
            attentions = attentions[torch.arange(enc_indices.size(0)), :, enc_indices]
        return logits, state

    def _prepare_decoder_output(self, last_predictions: torch.Tensor,
                                state: Dict[str, torch.Tensor]):
        decoder_output = state["decoder_output"]
        batch_size, enc_steps, _ = decoder_output.size()
        group_size = batch_size * enc_steps
        if state["decoder_hidden"] is None or state["decoder_context"] is None:
            hx = None
        else:
            state["prev_decoder_hidden"] = state["decoder_hidden"]
            state["prev_decoder_context"] = state["decoder_context"]
            state["prev_decoder_output"] = state["decoder_output"]
            decoder_hidden = state["decoder_hidden"].reshape(group_size, -1, self._hidden_size).transpose(1, 0)
            decoder_context = state["decoder_context"].reshape(group_size, -1, self._hidden_size).transpose(1, 0)
            hx = (decoder_hidden.contiguous(), decoder_context.contiguous())
        keys = state["att_keys"]#.repeat(enc_steps, 1, 1).view(group_size, enc_steps, -1)
        values = state["att_values"]#.repeat(enc_steps, 1, 1).view(group_size, enc_steps, -1)
        source_mask = state["source_mask"]#.repeat(enc_steps, 1).view(group_size, -1)

        embedded_input = self._target_embedder(last_predictions)
        queries = self._attention._query_projection(decoder_output)
        flattened_attns = torch.bmm(queries, keys.transpose(-2, -1))
        #flattened_attns = torch.einsum('bnlh,bhk->bnl', queries.unsqueeze(1).expand(-1, enc_steps, -1, -1),
        #                               keys.transpose(-2, -1))
        #_, flattened_attns, flattened_values = self._attention(decoder_output.view(group_size, -1),
        #                                                       att_keys,
        #                                                       att_values,
        #                                                       source_mask)

        #values = flattened_values.view(batch_size, enc_steps, enc_steps, -1)
        attns = flattened_attns.view(batch_size, enc_steps, -1)
        lookback_mask = util.get_mask_from_sequence_lengths(torch.arange(1, enc_steps + 1, device=source_mask.device),
                                                            enc_steps)
        attn_mask = source_mask.view(batch_size, enc_steps, -1) * lookback_mask.unsqueeze(0).expand(batch_size, -1, -1)
        attns = util.masked_softmax(attns, attn_mask)
        attn_cxts = util.weighted_sum(values, attns)
        attn_outputs = self._attention._output_projection(attn_cxts)

        decoder_input = torch.cat((embedded_input.repeat(enc_steps, 1).view(batch_size, enc_steps, -1),
                                   attn_outputs), -1)
        _, (decoder_hidden, decoder_context) = self._decoder(
            decoder_input.view(batch_size*enc_steps, -1).unsqueeze(1),
            hx)

        decoder_hidden = decoder_hidden.transpose(1, 0).reshape(batch_size, enc_steps, -1, self._hidden_size)
        decoder_context = decoder_context.transpose(1, 0).reshape(batch_size, enc_steps, -1, self._hidden_size)
        #outputs = outputs.reshape(batch_size, enc_steps, -1, self._hidden_size)

        state["decoder_hidden"] = decoder_hidden
        state["decoder_context"] = decoder_context
        state["decoder_output"] = decoder_hidden[..., 0, :]
        state["attentions"] = attns

        return decoder_hidden[..., -1, :], state

    def _slow_greedy_decode(self, eouts, elens,
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
            y = elens.new_zeros((1, 1)).fill_(self._start_index)
            dout, dstate = self._decoder(self._target_embedder(y), None)

            t = 0
            while t < elens[b]:
                # Pick up 1-best per frame
                out = self._joint(eouts[b:b + 1, t:t + 1], dout)
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
                    dout, dstate = self._decoder(self._target_embedder(y), dstate)
                else:
                    t += 1

            hyps += [best_hyp_b]

        return hyps

    def _greedy_decode(self, state: Dict[str, torch.Tensor], elens: torch.Tensor,
                       dlens: torch.Tensor = None) -> List[List[int]]:
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
        eouts = state["encoder_outputs"]
        batch_size, max_enc_len, _ = eouts.size()
        if dlens is None:
            dlens = elens.new_full(elens.size(), max_enc_len)
        # Initialize target predictions with the start index.
        # shape: (batch_size,)
        last_predictions = elens.new_zeros((batch_size, 1)).fill_(self._start_index)
        # current source position t for each element in a batch.
        cur_enc_indices = elens.new_zeros((batch_size,))
        end_indices = elens
        # decoding finished when each current source position equals to the
        # end_indices.
        finished = (cur_enc_indices == end_indices)
        # total steps traversed in the latttices, including horizontal steps
        # (blanks).
        hyp_lens = elens.new_zeros((batch_size,))
        hypotheses = []

        while not finished.all():
            hyp_lens += (~finished).long()
            # for finished batch elements with cur_enc_indices == end_indices,
            # we clamped it to prevent IndexError.
            clamped_indices = torch.min(cur_enc_indices, end_indices - 1)
            out, state = self.joint_network(state,
                                            last_predictions,
                                            clamped_indices)

            y = out.argmax(-1).squeeze(1)
            hypotheses.append(y.squeeze(-1))
            # state will not be updated for those who output blanks (horizontal
            # steps).
            state = maybe_update_state(state, y, self._pad_index)

            cur_enc_indices = torch.min(end_indices,
                                        cur_enc_indices + (y.view(-1) == self._pad_index).long())
            finished = (cur_enc_indices == (end_indices)) | (y.view(-1) == self._end_index) | \
                        (hyp_lens > (elens + dlens))

        raw_hypotheses = torch.stack(hypotheses, dim=-1)
        hypotheses_for_logging = [hypothesis[:hyp_len]
                      for hypothesis, hyp_len in zip(raw_hypotheses.tolist(), hyp_lens.tolist())]
        #print(hypotheses_for_logging)
        hypotheses = [list(filter(lambda idx: idx != self._pad_index, hypothesis[:hyp_len]))
                      for hypothesis, hyp_len in zip(raw_hypotheses.tolist(), hyp_lens.tolist())]
        return hypotheses

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        all_metrics["wer"] = self._wer.get_metric(reset=reset)
        all_metrics.update(self._bleu.get_metric(reset=reset))
        return all_metrics
