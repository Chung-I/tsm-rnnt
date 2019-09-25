from typing import Tuple, List, Union, Dict, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from allennlp.nn.util import get_mask_from_sequence_lengths, weighted_sum, masked_mean

from stt.modules.attention import MonotonicAttention
from stt.modules.stateful_attention import StatefulAttention

def is_nan_or_inf(x): return (x == np.inf) | (x != x)

# def masked_mean(vector: torch.Tensor,
#                 mask: torch.Tensor,
#                 dim: int,
#                 keepdim: bool = False,
#                 eps: float = 1e-8) -> torch.Tensor:

#     one_minus_mask = ~mask
#     replaced_vector = vector.masked_fill(one_minus_mask, 0.0)

#     value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
#     value_count = torch.sum(mask.float(), dim=dim, keepdim=keepdim)
#     return value_sum / value_count.clamp(min=eps)


def averaging_tensor_of_same_label(enc_outs: torch.Tensor,
                                   phn_log_probs: torch.Tensor,
                                   lengths=None, mask=None):
    assert (lengths is None) != (
        mask is None), "exactly one of lengths or mask is to be provided"
    batch_size, _, feat_dim = enc_outs.size()
    _, phn_labels = phn_log_probs.max(-1)
    boundaries = phn_labels[:,:-1] != phn_labels[:,1:]
    boundaries = F.pad(boundaries.unsqueeze(1), pad=(1, 0)).squeeze(1)
    segments = torch.cumsum(boundaries, dim=-1)
    segment_lengths, _ = (segments + 1).max(dim=-1)
    max_len = segment_lengths.max()
    phn_enc_outs = enc_outs.new_zeros((batch_size, max_len, feat_dim))
    for i in range(max_len):
        phn_enc_outs[:, i] = masked_mean(enc_outs, mask=(
            (segments == i).unsqueeze(-1).expand_as(enc_outs)), dim=1)

    phn_enc_outs.masked_fill(is_nan_or_inf(phn_enc_outs), value=0)
    return phn_enc_outs, segment_lengths

def remove_sentence_boundaries(tensor: torch.Tensor,
                               mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()
    tensor_shape = list(tensor.data.shape)
    new_shape = list(tensor_shape)
    new_shape[1] = tensor_shape[1] - 2
    tensor_without_boundary_tokens = tensor.new_zeros(*new_shape)
    new_mask = tensor.new_zeros((new_shape[0], new_shape[1]), dtype=torch.bool)
    for i, j in enumerate(sequence_lengths):
        if j > 2:
            tensor_without_boundary_tokens[i, :(j - 2)] = tensor[i, 1:(j - 1)]
            new_mask[i, :(j - 2)] = 1

    return tensor_without_boundary_tokens, new_mask

def remove_eos(tensor: torch.Tensor,
               mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()
    tensor_shape = list(tensor.data.shape)
    new_shape = list(tensor_shape)
    new_shape[1] = tensor_shape[1] - 1
    tensor_without_boundary_tokens = tensor.new_zeros(*new_shape)
    new_mask = tensor.new_zeros((new_shape[0], new_shape[1]), dtype=torch.bool)
    for i, j in enumerate(sequence_lengths):
        if j > 1:
            tensor_without_boundary_tokens[i, :(j - 1), ...] = tensor[i, :(j - 1), ...]
            new_mask[i, :(j - 1)] = 1

    return tensor_without_boundary_tokens, new_mask

def char_to_word(tensor: torch.Tensor,
                 segments: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, _, feat_dim = tensor.size()
    segment_lengths, _ = segments.max(dim=-1)
    max_len = segment_lengths.max()
    new_tensor = tensor.new_zeros((batch_size, max_len, feat_dim))
    for i in range(max_len):
        new_tensor[:, i] = masked_mean(tensor, mask=(
            (segments == (i+1)).unsqueeze(-1).expand_as(tensor)), dim=1)

    new_tensor.masked_fill(is_nan_or_inf(new_tensor), value=0)
    return new_tensor, segment_lengths

def list_to_tensor(list_of_tensors: List[Union[List[int], torch.Tensor]],
                   placeholder: torch.Tensor) -> torch.Tensor:
    batch_size = len(list_of_tensors)
    max_len = max(list(map(len, list_of_tensors)))
    new_tensor = placeholder.new_zeros((batch_size, max_len))
    for b, tensor in enumerate(list_of_tensors):
        for i, value in enumerate(tensor):
            new_tensor[b, i] = value
    return new_tensor

def sequence_cross_entropy_with_log_probs(log_probs: torch.FloatTensor,
                                          targets: torch.LongTensor,
                                          weights: torch.FloatTensor,
                                          average: str = "batch",
                                          label_smoothing: float = None,
                                          gamma: float = None,
                                          alpha: Union[float, List[float], torch.FloatTensor] = None
                                         ) -> torch.FloatTensor:
    """
    Computes the cross entropy loss of a sequence, weighted with respect to
    some user provided weights. Note that the weighting here is not the same as
    in the :func:`torch.nn.CrossEntropyLoss()` criterion, which is weighting
    classes; here we are weighting the loss contribution from particular elements
    in the sequence. This allows loss computations for models which use padding.

    Parameters
    ----------
    logits : ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of size (batch_size, sequence_length, num_classes)
        which contains the unnormalized probability for each class.
    targets : ``torch.LongTensor``, required.
        A ``torch.LongTensor`` of size (batch, sequence_length) which contains the
        index of the true class for each corresponding step.
    weights : ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of size (batch, sequence_length)
    average: str, optional (default = "batch")
        If "batch", average the loss across the batches. If "token", average
        the loss across each item in the input. If ``None``, return a vector
        of losses per batch element.
    label_smoothing : ``float``, optional (default = None)
        Whether or not to apply label smoothing to the cross-entropy loss.
        For example, with a label smoothing value of 0.2, a 4 class classification
        target would look like ``[0.05, 0.05, 0.85, 0.05]`` if the 3rd class was
        the correct label.
    gamma : ``float``, optional (default = None)
        Focal loss[*] focusing parameter ``gamma`` to reduces the relative loss for
        well-classified examples and put more focus on hard. The greater value
        ``gamma`` is, the more focus on hard examples.
    alpha : ``float`` or ``List[float]``, optional (default = None)
        Focal loss[*] weighting factor ``alpha`` to balance between classes. Can be
        used independently with ``gamma``. If a single ``float`` is provided, it
        is assumed binary case using ``alpha`` and ``1 - alpha`` for positive and
        negative respectively. If a list of ``float`` is provided, with the same
        length as the number of classes, the weights will match the classes.
        [*] T. Lin, P. Goyal, R. Girshick, K. He and P. Dollár, "Focal Loss for
        Dense Object Detection," 2017 IEEE International Conference on Computer
        Vision (ICCV), Venice, 2017, pp. 2999-3007.

    Returns
    -------
    A torch.FloatTensor representing the cross entropy loss.
    If ``average=="batch"`` or ``average=="token"``, the returned loss is a scalar.
    If ``average is None``, the returned loss is a vector of shape (batch_size,).

    """
    if average not in {None, "token", "batch"}:
        raise ValueError("Got average f{average}, expected one of "
                         "None, 'token', or 'batch'")

    # make sure weights are float
    weights = weights.float()
    # sum all dim except batch
    non_batch_dims = tuple(range(1, len(weights.shape)))
    # shape : (batch_size,)
    weights_batch_sum = weights.sum(dim=non_batch_dims)
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = log_probs.view(-1, log_probs.size(-1))
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()
    # focal loss coefficient
    if gamma:
        # shape : (batch * sequence_length, num_classes)
        probs_flat = log_probs_flat.exp()
        # shape : (batch * sequence_length,)
        probs_flat = torch.gather(probs_flat, dim=1, index=targets_flat)
        # shape : (batch * sequence_length,)
        focal_factor = (1. - probs_flat) ** gamma
        # shape : (batch, sequence_length)
        focal_factor = focal_factor.view(*targets.size())
        weights = weights * focal_factor

    if alpha is not None:
        # shape : () / (num_classes,)
        if isinstance(alpha, (float, int)):
            # pylint: disable=not-callable
            # shape : (2,)
            alpha_factor = torch.tensor([1. - float(alpha), float(alpha)],
                                        dtype=weights.dtype, device=weights.device)
            # pylint: enable=not-callable
        elif isinstance(alpha, (list, np.ndarray, torch.Tensor)):
            # pylint: disable=not-callable
            # shape : (c,)
            alpha_factor = torch.tensor(alpha, dtype=weights.dtype, device=weights.device)
            # pylint: enable=not-callable
            if not alpha_factor.size():
                # shape : (1,)
                alpha_factor = alpha_factor.view(1)
                # shape : (2,)
                alpha_factor = torch.cat([1 - alpha_factor, alpha_factor])
        else:
            raise TypeError(('alpha must be float, list of float, or torch.FloatTensor, '
                             '{} provided.').format(type(alpha)))
        # shape : (batch, max_len)
        alpha_factor = torch.gather(alpha_factor, dim=0, index=targets_flat.view(-1)).view(*targets.size())
        weights = weights * alpha_factor

    if label_smoothing is not None and label_smoothing > 0.0:
        num_classes = log_probs.size(-1)
        smoothing_value = label_smoothing / num_classes
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = - log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # Contribution to the negative log likelihood only comes from the exact indices
        # of the targets, as the target distributions are one-hot. Here we use torch.gather
        # to extract the indices of the num_classes dimension which contribute to the loss.
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood * weights

    if average == "batch":
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (weights_batch_sum + 1e-13)
        num_non_empty_sequences = ((weights_batch_sum > 0).float().sum() + 1e-13)
        return per_batch_loss.sum() / num_non_empty_sequences
    elif average == "token":
        return negative_log_likelihood.sum() / (weights_batch_sum.sum() + 1e-13)
    else:
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (weights_batch_sum + 1e-13)
        return per_batch_loss

def prepare_attended_output_factory(self) -> Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
    """Apply attention over encoder outputs and decoder state."""
    # Ensure mask is also a FloatTensor. Or else the multiplication within
    # attention will complain.
    def prepare_attended_output(decoder_hidden_state: torch.Tensor,
                                state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
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
            attended_output = weighted_sum(encoder_outputs, chunk_attention)
            attns = monotonic_attention
        elif isinstance(self._attention, StatefulAttention):
            attended_output, attns = self._attention(decoder_hidden_state,
                                                     att_keys, att_values, source_mask)
        else:
            attns = self._attention(
                decoder_hidden_state, source_mask)
            attended_output = weighted_sum(encoder_outputs, attns)

        return attended_output, attns

    return prepare_attended_output

def prepare_decoder_output_factory(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]: # pylint: disable=line-too-long
    """
    Decode current state and last prediction to produce produce projections
    into the target space, which can then be used to get probabilities of
    each target token for the next step.

    Inputs are the same as for `take_step()`.
    """
    def prepare_decoder_output(last_predictions: torch.Tensor,
                               state: Dict[str, torch.Tensor]):
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
            attended_output, attns = self._prepare_attended_output(outputs, state)

            # shape: (group_size, decoder_output_dim)
            decoder_output = torch.tanh(
                self._att_out(torch.cat((attended_output, outputs), -1))
            )
            state["attention"] = attns
            state["attention_contexts"] = attended_output

        else:
            # shape: (group_size, target_embedding_dim)
            decoder_output = outputs

        state["decoder_hidden"] = decoder_hidden
        state["decoder_context"] = decoder_context
        state["decoder_output"] = decoder_output

        return decoder_output, state

    return prepare_decoder_output

def maybe_update_state(state: Dict[str, torch.Tensor],
                       predictions: torch.Tensor, pad_index: int) -> Dict[str, torch.Tensor]:
    is_pad = lambda y, tensor: y.view(-1, *(1,)*(tensor.ndim-1)) \
        .expand_as(tensor) == pad_index
    for field in ["decoder_hidden", "decoder_context", "decoder_output"]:
        if field in state:
            state[field] = torch.where(is_pad(predictions, state[field]),
                                       state[f"prev_{field}"],
                                       state[field])
    return state
