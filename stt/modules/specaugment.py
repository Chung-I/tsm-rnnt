import torch
import torch.nn as nn

from allennlp.nn.util import get_lengths_from_binary_sequence_mask, masked_mean

class TimeMask(nn.Module):
    def __init__(self, max_width: int, max_ratio: float, replace_with_zero: bool = True) -> None:
        self.max_width = max_width
        self.max_ratio = max_ratio
        self._replace_with_zero = replace_with_zero
    
    def forward(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.training:
            lengths = get_lengths_from_binary_sequence_mask(mask)
            time_mask = mask.new_zeros(mask.size())
            for idx, length in enumerate(lengths):
                max_width = min(int(self.max_ratio * length), self.max_width)
                start = torch.randint(0, length - max_width)
                width = torch.randint(0, max_width)
                time_mask[idx, start:start + width] = 1

            replaced_value = 0.0
            if not self._replace_with_zero:
                raise NotImplementedError

            tensor = tensor.masked_fill(time_mask.unsqueeze(-1).expand_as(tensor),
                                        replaced_value)

        return tensor


class FreqMask(nn.Module):
    def __init__(self, max_width: int, replace_with_zero: bool = True) -> None:
        self.max_width = max_width
        self._replace_with_zero = replace_with_zero

    def forward(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.training:
            freq_dim = tensor.size(-1)
            freq_mask = mask.new_zeros((freq_dim,))
            start = torch.randint(0, freq_dim - self.max_width)
            width = torch.randint(0, self.max_width)
            freq_mask[start:start + width] = 1

            replaced_value = 0.0
            if not self._replace_with_zero:
                raise NotImplementedError

            tensor = tensor.masked_fill(freq_mask.view(1, 1, -1).expand_as(tensor),
                                        replaced_value)

        return tensor
