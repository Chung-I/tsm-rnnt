import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import math

class Delta(torch.jit.ScriptModule):

    __constants__ = ["order", "window_size", "padding"]

    def __init__(self, order=1, window_size=2):
        # Reference:
        # https://kaldi-asr.org/doc/feature-functions_8cc_source.html
        # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_audio.py
        super(Delta, self).__init__()

        self.order = order
        self.window_size = window_size

        filters = self._create_filters(order, window_size)
        self.register_buffer("filters", filters)
        self.padding = (0, (filters.shape[-1] - 1) // 2)

    @torch.jit.script_method
    def forward(self, x):
        # Unsqueeze batch dim
        #x = x.unsqueeze(0)
        return F.conv2d(x, weight=self.filters, padding=self.padding)

    # TODO(WindQAQ): find more elegant way to create `scales`
    def _create_filters(self, order, window_size):
        scales = [[1.0]]
        for i in range(1, order + 1):
            prev_offset = (len(scales[i-1]) - 1) // 2
            curr_offset = prev_offset + window_size

            curr = [0] * (len(scales[i-1]) + 2 * window_size)
            normalizer = 0.0
            for j in range(-window_size, window_size + 1):
                normalizer += j * j
                for k in range(-prev_offset, prev_offset + 1):
                    curr[j+k+curr_offset] += (j * scales[i-1][k+prev_offset])
            curr = [x / normalizer for x in curr]
            scales.append(curr)

        max_len = len(scales[-1])
        for i, scale in enumerate(scales[:-1]):
            padding = (max_len - len(scale)) // 2
            scales[i] = [0] * padding + scale + [0] * padding

        return torch.Tensor(scales).unsqueeze(1).unsqueeze(1)

    def savgol_coeffs(self, window_length, polyorder, deriv=0, delta=1.0, pos=None,
                      use="conv"):
        if polyorder >= window_length:
            raise ValueError("polyorder must be less than window_length.")

        halflen, rem = divmod(window_length, 2)

        if rem == 0:
            raise ValueError("window_length must be odd.")

        if pos is None:
            pos = halflen

        if not (0 <= pos < window_length):
            raise ValueError("pos must be nonnegative and less than "
                            "window_length.")

        if use not in ['conv', 'dot']:
            raise ValueError("`use` must be 'conv' or 'dot'")

        # Form the design matrix A.  The columns of A are powers of the integers
        # from -pos to window_length - pos - 1.  The powers (i.e. rows) range
        # from 0 to polyorder.  (That is, A is a vandermonde matrix, but not
        # necessarily square.)
        x = torch.arange(-pos, window_length - pos, dtype=torch.float32)
        if use == "conv":
            # Reverse so that result can be used in a convolution.
            x = x[::-1]

        order = torch.arange(polyorder + 1).reshape(-1, 1)
        if order.size == 1:
            raise NotImplementedError
            # Avoid spurious DeprecationWarning in numpy 1.8.0 for
            # ``[1] ** [[2]]``, see numpy gh-4145.
            # A = np.atleast_2d(x ** order[0, 0])
        else:
            A = x ** order

        # y determines which order derivative is returned.
        y = torch.zeros(polyorder + 1)
        # The coefficient assigned to y[deriv] scales the result to take into
        # account the order of the derivative and the sample spacing.
        y[deriv] = math.factorial(deriv) / (delta ** deriv)

        # Find the least-squares solution of A*c = y
        coeffs, _ = torch.lstsq(y, A)

        return coeffs

    def extra_repr(self):
        return "order={}, window_size={}".format(self.order, self.window_size)
