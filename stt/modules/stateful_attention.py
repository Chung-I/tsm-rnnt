from overrides import overrides
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from allennlp.nn.util import masked_softmax, weighted_sum
from allennlp.modules.attention.attention import Attention
from allennlp.nn import Activation


@Attention.register("stateful")
class StatefulAttention(Attention):
    """
    Computes attention between a vector and a matrix using a bilinear attention function.  This
    function has a matrix of weights ``W`` and a bias ``b``, and the similarity between the vector
    ``x`` and the matrix ``y`` is computed as ``x^T W y + b``.

    Parameters
    ----------
    vector_dim : ``int``
        The dimension of the vector, ``x``, described above.  This is ``x.size()[-1]`` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    matrix_dim : ``int``
        The dimension of the matrix, ``y``, described above.  This is ``y.size()[-1]`` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    activation : ``Activation``, optional (default=linear (i.e. no activation))
        An activation function applied after the ``x^T W y + b`` calculation.  Default is no
        activation.
    normalize : ``bool``, optional (default: ``True``)
        If true, we normalize the computed similarities with a softmax, to return a probability
        distribution for your attention.  If false, this is just computing a similarity score.
    """
    def __init__(self,
                 vector_dim: int,
                 matrix_dim: int,
                 attention_dim: int,
                 values_dim: int,
                 num_heads: int = 1,
                 activation: Activation = None,
                 attention_dropout_prob: float = 0.0,
                 normalize: bool = True) -> None:
        super().__init__(normalize)

        self._num_heads = num_heads
        self._attention_dim = attention_dim
        self._values_dim = values_dim
        self._output_dim = matrix_dim

        if attention_dim % num_heads != 0:
            raise ValueError(f"Key size ({attention_dim}) must be divisible by the number of "
                             f"attention heads ({num_heads}).")

        self._combined_projection = nn.Linear(matrix_dim, attention_dim + values_dim)
        self._query_projection = nn.Linear(vector_dim, attention_dim)

        self._scale = (attention_dim // num_heads) ** 0.5
        # self._output_projection = Linear(values_dim, self._output_dim)
        self._attention_dropout = nn.Dropout(attention_dropout_prob)
        self._output_projection = nn.Linear(values_dim, self._output_dim)

        self._activation = activation or Activation.by_name('linear')()
        self._num_heads = num_heads
        self.reset_state()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._weight_matrix)
        self._bias.data.fill_(0)

    def init_state(self, matrix: torch.Tensor, mask: torch.Tensor) -> None:
        combined_projection = self._combined_projection(matrix)
        keys, *values = combined_projection.split(self._attention_dim, -1)
        keys = keys.contiguous()
        values = torch.cat(values, -1).contiguous()
        self.keys = keys
        self.values = values
        self.mask = mask

    def reset_state(self) -> None:
        self.keys = None
        self.values = None
        self.mask = None

    def _view_as_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, timesteps, hidden_dim = tensor.size()
        tensor_per_head = tensor.view(batch_size, timesteps, self._num_heads,
                                      int(hidden_dim/self._num_heads))
        tensor_per_head = tensor_per_head.transpose(1, 2).contiguous()
        tensor_per_head = tensor_per_head.view(batch_size * self._num_heads, timesteps,
                                               int(hidden_dim/self._num_heads))
        return tensor_per_head

    @overrides
    def forward(self, queries: torch.Tensor,
                inputs: torch.Tensor = None,
                mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if inputs is not None:
            self.init_state(inputs, mask)

        batch_size, enc_timesteps, _ = self.keys.size()

        if len(queries.size()) < 3:
            queries = queries.unsqueeze(1)
        
        dec_timesteps = self.queries.size(1)

        queries = self._query_projection(queries)

        if self._num_heads > 1:
            keys_per_head = self._view_as_heads(self.keys)
            values_per_head = self._view_as_heads(self.values)

            queries_per_head = self._view_as_heads(queries.unsqueeze(1)).squeeze(1)
            scaled_similarities = torch.bmm(queries_per_head / self._scale,
                                            keys_per_head.transpose(1, 2))

            attention = masked_softmax(scaled_similarities,
                                       self.mask.repeat(1, self._num_heads) \
                                        .view(batch_size * self._num_heads, enc_timesteps),
                                       memory_efficient=True)

            attention = self._attention_dropout(attention)

            outputs = weighted_sum(values_per_head, attention)
            outputs = outputs.view(batch_size, self._num_heads, dec_timesteps,
                                   int(self._values_dim / self._num_heads))
            outputs = outputs.transpose(1, 2).contiguous()
            outputs = outputs.view(batch_size, dec_timesteps, self._values_dim)
        else:
            scaled_similarities = torch.bmm(queries / self.scale,
                                            self.keys.transpose(1, 2))
            attention = masked_softmax(scaled_similarities,
                                       self.mask,
                                       memory_efficient=True)

            attention = self._attention_dropout(attention)
            outputs = weighted_sum(self.values, attention)

        outputs = self._output_projection(outputs)
        if dec_timesteps == 1:
            outputs = outputs.squeeze(1)

        return outputs, attention
