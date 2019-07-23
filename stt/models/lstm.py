"""
A stacked LSTM with LSTM layers which alternate between going forwards over
the sequence and going backwards.
"""

from typing import Optional, Tuple, Union
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
from allennlp.common.checks import ConfigurationError
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.seq2seq_encoders import _Seq2SeqWrapper

from stt.models.custom_lstms import script_lstm, script_lnlstm

class StackedCustomLstm(torch.nn.Module):
    """
    A stacked LSTM with LSTM layers which alternate between going forwards over
    the sequence and going backwards. This implementation is based on the
    description in `Deep Semantic Role Labelling - What works and what's next
    <https://homes.cs.washington.edu/~luheng/files/acl2017_hllz.pdf>`_ .

    Parameters
    ----------
    input_size : int, required
        The dimension of the inputs to the LSTM.
    hidden_size : int, required
        The dimension of the outputs of the LSTM.
    num_layers : int, required
        The number of stacked LSTMs to use.
    recurrent_dropout_probability: float, optional (default = 0.0)
        The dropout probability to be used in a dropout scheme as stated in
        `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
        <https://arxiv.org/abs/1512.05287>`_ .
    use_input_projection_bias : bool, optional (default = True)
        Whether or not to use a bias on the input projection layer. This is mainly here
        for backwards compatibility reasons and will be removed (and set to False)
        in future releases.

    Returns
    -------
    output_accumulator : PackedSequence
        The outputs of the interleaved LSTMs per timestep. A tensor of shape
        (batch_size, max_timesteps, hidden_size) where for a given batch
        element, all outputs past the sequence length for that batch are
        zero tensors.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 layer_norm: bool = False) -> None:
        super(StackedCustomLstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if layer_norm:
            self.rnn = script_lnlstm(input_size, hidden_size, num_layers)
        else:
            self.rnn = script_lstm(input_size, hidden_size, num_layers)

    def forward(self,  # pylint: disable=arguments-differ
                inputs: PackedSequence,
                initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> \
            Tuple[Union[torch.Tensor, PackedSequence], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        inputs : ``PackedSequence``, required.
            A batch first ``PackedSequence`` to run the stacked LSTM over.
        initial_state : Tuple[torch.Tensor, torch.Tensor], optional, (default = None)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. Each tensor has shape (1, batch_size, output_dimension).

        Returns
        -------
        output_sequence : PackedSequence
            The encoded sequence of shape (batch_size, sequence_length, hidden_size)
        final_states: Tuple[torch.Tensor, torch.Tensor]
            The per-layer final (state, memory) states of the LSTM, each with shape
            (num_layers, batch_size, hidden_size).
        """
        sequence_tensor, batch_lengths = pad_packed_sequence(inputs, batch_first=True)
        batch_size, _, _ = sequence_tensor.size()
        if not initial_state:
            hidden_states = [(sequence_tensor.new_zeros(batch_size, self.hidden_size),
                              sequence_tensor.new_zeros(batch_size, self.hidden_size))
                             for _ in range(self.num_layers)]
        elif initial_state[0].size()[0] != self.num_layers:
            raise ConfigurationError("Initial states were passed to forward() but the number of "
                                     "initial states does not match the number of layers.")
        else:
            hidden_states = list(zip(initial_state[0].split(1, 0),
                                     initial_state[1].split(1, 0)))
        outputs, states = self.rnn(sequence_tensor.transpose(1, 0), hidden_states)
        outputs = pack_padded_sequence(outputs, batch_lengths)
        return outputs, states



Seq2SeqEncoder.register("stacked_custom_lstm")(_Seq2SeqWrapper(StackedCustomLstm))
