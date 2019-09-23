from typing import Tuple, List, Union

import torch
import torch.nn.functional as F
import numpy as np

from allennlp.nn.util import get_mask_from_sequence_lengths


def is_nan_or_inf(x): return (x == np.inf) | (x != x)

def masked_mean(vector: torch.Tensor,
                mask: torch.Tensor,
                dim: int,
                keepdim: bool = False,
                eps: float = 1e-8) -> torch.Tensor:

    one_minus_mask = ~mask
    replaced_vector = vector.masked_fill(one_minus_mask, 0.0)

    value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
    value_count = torch.sum(mask.float(), dim=dim, keepdim=keepdim)
    return value_sum / value_count.clamp(min=eps)


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
