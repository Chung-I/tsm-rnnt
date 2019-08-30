from typing import Optional, Tuple
import torch
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from allennlp.modules.seq2seq_encoders import _Seq2SeqWrapper
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.common.checks import ConfigurationError


class ResidualBidirectionalLstm(torch.nn.Module):
    """
    A standard stacked Bidirectional LSTM where the LSTM layers
    are concatenated between each layer. The only difference between
    this and a regular bidirectional LSTM is the application of
    variational dropout to the hidden states and outputs of each layer apart
    from the last layer of the LSTM. Note that this will be slower, as it
    doesn't use CUDNN.

    Parameters
    ----------
    input_size : int, required
        The dimension of the inputs to the LSTM.
    hidden_size : int, required
        The dimension of the outputs of the LSTM.
    num_layers : int, required
        The number of stacked Bidirectional LSTMs to use.
    recurrent_dropout_probability: float, optional (default = 0.0)
        The recurrent dropout probability to be used in a dropout scheme as
        stated in `A Theoretically Grounded Application of Dropout in Recurrent
        Neural Networks <https://arxiv.org/abs/1512.05287>`_ .
    layer_dropout_probability: float, optional (default = 0.0)
        The layer wise dropout probability to be used in a dropout scheme as
        stated in  `A Theoretically Grounded Application of Dropout in
        Recurrent Neural Networks <https://arxiv.org/abs/1512.05287>`_ .
    use_highway: bool, optional (default = True)
        Whether or not to use highway connections between layers. This effectively involves
        reparameterising the normal output of an LSTM as::

            gate = sigmoid(W_x1 * x_t + W_h * h_t)
            output = gate * h_t  + (1 - gate) * (W_x2 * x_t)
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 layer_dropout_probability: float = 0.0,
                 use_residual: bool = True,
                 use_residual_projection: bool = False) -> None:
        super(ResidualBidirectionalLstm, self).__init__()

        # Required to be wrapped with a :class:`PytorchSeq2SeqWrapper`.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = True
        self.use_residual = use_residual
        self.use_residual_projection = use_residual_projection

        layers = []
        lstm_input_size = input_size
        for layer_index in range(num_layers):

            layer = torch.nn.LSTM(lstm_input_size, hidden_size,
                                  num_layers=1,
                                  batch_first=True,
                                  bidirectional=True)

            if use_residual and layer_index < (self.num_layers - 1):
                if use_residual_projection or lstm_input_size != hidden_size * 2:
                    residual_projection = torch.nn.Linear(lstm_input_size, hidden_size * 2, bias=False)
                else:
                    residual_projection = torch.nn.Identity()
                self.add_module('res_proj_{}'.format(layer_index), residual_projection)                

            lstm_input_size = hidden_size * 2
            self.add_module('layer_{}'.format(layer_index), layer)


            layers.append(layer)
        self.lstm_layers = layers
        self.layer_dropout = InputVariationalDropout(layer_dropout_probability)

    def forward(self,  # pylint: disable=arguments-differ
                inputs: PackedSequence,
                initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        inputs : ``PackedSequence``, required.
            A batch first ``PackedSequence`` to run the stacked LSTM over.
        initial_state : Tuple[torch.Tensor, torch.Tensor], optional, (default = None)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. Each tensor has shape (num_layers, batch_size, output_dimension * 2).

        Returns
        -------
        output_sequence : PackedSequence
            The encoded sequence of shape (batch_size, sequence_length, hidden_size * 2)
        final_states: torch.Tensor
            The per-layer final (state, memory) states of the LSTM, each with shape
            (num_layers * 2, batch_size, hidden_size * 2).
        """
        if not initial_state:
            hidden_states = [None] * len(self.lstm_layers)
        elif initial_state[0].size()[0] != len(self.lstm_layers):
            raise ConfigurationError("Initial states were passed to forward() but the number of "
                                     "initial states does not match the number of layers.")
        else:
            hidden_states = list(zip(initial_state[0].split(1, 0),
                                     initial_state[1].split(1, 0)))

        output_sequence = inputs
        prev_sequence, _ = pad_packed_sequence(output_sequence, batch_first=True)
        final_h = []
        final_c = []
        for i, state in enumerate(hidden_states):
            layer = getattr(self, 'layer_{}'.format(i))
            # The state is duplicated to mirror the Pytorch API for LSTMs.
            output, final_state = layer(output_sequence, state)

            output_sequence, lengths = pad_packed_sequence(output, batch_first=True)

            # Apply layer wise dropout on each output sequence apart from the
            # first (input) and last
            if i < (self.num_layers - 1):
                output_sequence = self.layer_dropout(output_sequence)
                if self.use_residual:
                    res_proj = getattr(self, 'res_proj_{}'.format(i))
                    tmp = output_sequence
                    output_sequence = output_sequence + res_proj(prev_sequence)
                    prev_sequence = tmp

            output_sequence = pack_padded_sequence(output_sequence, lengths, batch_first=True)

            final_h.append(final_state[0])
            final_c.append(final_state[1])

        final_h = torch.cat(final_h, dim=0)
        final_c = torch.cat(final_c, dim=0)
        final_state_tuple = (final_h, final_c)
        return output_sequence, final_state_tuple


Seq2SeqEncoder.register("residual_bidirectional_lstm")(_Seq2SeqWrapper(ResidualBidirectionalLstm))
