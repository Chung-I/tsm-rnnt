from overrides import overrides

import torch
import torch.nn as nn

from allennlp.nn.util import get_lengths_from_binary_sequence_mask, masked_mean

class TimeMask(nn.Module):
    def __init__(self, max_width: int, max_ratio: float, replace_with_zero: bool = True) -> None:
        super(TimeMask, self).__init__()
        self.max_width = max_width
        self.max_ratio = max_ratio
        self._replace_with_zero = replace_with_zero
        self._enable = max_width > 0

    @overrides
    def forward(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.training and self._enable:
            lengths = get_lengths_from_binary_sequence_mask(mask)
            time_mask = mask.new_zeros(mask.size())
            for idx, length in enumerate(lengths.tolist()):
                max_width = min(int(self.max_ratio * float(length)), self.max_width)
                if max_width < 1:
                    continue
                width = torch.randint(low=0, high=max_width, size=()).item()
                start = torch.randint(low=0, high=length - width, size=()).item()
                time_mask[idx, start:start + width] = 1

            replaced_value = 0.0
            if not self._replace_with_zero:
                raise NotImplementedError

            tensor = tensor.masked_fill(time_mask.unsqueeze(1).unsqueeze(-1) \
                                        .expand_as(tensor),
                                        replaced_value)

        return tensor


class FreqMask(nn.Module):
    def __init__(self, max_width: int, replace_with_zero: bool = True) -> None:
        super(FreqMask, self).__init__()
        self.max_width = max_width
        self._replace_with_zero = replace_with_zero
        self._enable = max_width > 0

    @overrides
    def forward(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.training and self._enable:
            freq_dim = tensor.size(-1)
            assert self.max_width < freq_dim, "self.max_width must be strictly \
                lower than freq_dim"
            freq_mask = mask.new_zeros((freq_dim,))
            width = torch.randint(low=0, high=self.max_width, size=()).item()
            start = torch.randint(low=0, high=freq_dim - width, size=()).item()
            freq_mask[start:start + width] = 1

            replaced_value = 0.0
            if not self._replace_with_zero:
                raise NotImplementedError

            tensor = tensor.masked_fill(freq_mask.view(1, 1, 1, -1).expand_as(tensor),
                                        replaced_value)

        return tensor
