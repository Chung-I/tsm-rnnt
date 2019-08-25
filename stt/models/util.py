import torch
import torch.nn.functional as F
import numpy as np
import pdb

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
