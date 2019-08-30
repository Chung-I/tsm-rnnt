from overrides import overrides
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from allennlp.modules.attention.attention import Attention
from allennlp.nn import Activation


@Attention.register("multi-head")
class MultiHeadedAttention(Attention):
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
                 keys: torch.Tensor,
                 num_heads: int,
                 activation: Activation = None,
                 attention_dropout_prob: float = 0.0,
                 normalize: bool = True) -> None:
        super().__init__(normalize)

        key_dim = keys.size(-1)
        self._num_heads = num_heads
        self._attention_dim = attention_dim

        if attention_dim % num_heads != 0:
            raise ValueError(f"Key size ({attention_dim}) must be divisible by the number of "
                             f"attention heads ({num_heads}).")

        self._combined_projection = nn.Linear(key_dim, key_dim * 2)
        self._combined_projection = nn.Linear()

        self.keys, self.values = self._combined_projection(keys).split(key_dim, -1)
    

        self._scale = (key_dim // num_heads) ** 0.5
        self._output_projection = Linear(values_dim, self._output_dim)
        self._attention_dropout = Dropout(attention_dropout_prob)

        self._weight_matrix = Parameter(torch.Tensor(vector_dim, matrix_dim))
        self._bias = Parameter(torch.Tensor(1))
        self._activation = activation or Activation.by_name('linear')()
        self._num_heads = num_heads
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._weight_matrix)
        self._bias.data.fill_(0)

    def _forward_internal(self, querys: torch.Tensor) -> torch.Tensor:
        intermediate = vector.mm(self._weight_matrix).unsqueeze(1)
        return self._activation(intermediate.bmm(matrix.transpose(1, 2)).squeeze(1) + self._bias)
