from typing import List

import torch
import torch.nn as nn
from torch.autograd import Variable

from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from stt.models.weight_drop import WeightDrop


class pRNNLayer(nn.Module):
    def __init__(self, rnn, input_size, hidden_size, stack_rate: int = 1, wdrop: float = 0):
        super(pRNNLayer, self).__init__()
        self.stack_rate = stack_rate
        self.module = rnn(input_size * stack_rate,
                          hidden_size, num_layers=1, dropout=0)
        if wdrop:
            self.module = WeightDrop(
                self.module, ['weight_hh_l0'], dropout=wdrop)

    def forward(self, inputs, states, lengths):
        seq_len, batch_size, feat_dim = inputs.size()
        if self.stack_rate != 1:
            inputs = inputs.transpose(1, 0) \
                .reshape(batch_size,
                         seq_len // self.stack_rate,
                         feat_dim * self.stack_rate) \
                .transpose(1, 0)
            lengths = torch.floor(
                torch.clamp(
                    lengths.float() / self.stack_rate, 1.
                )
            ).long()
        return self.module.forward(inputs, states), lengths


class LockedDropout(nn.Module):

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


@Seq2SeqEncoder.register('awd-rnn')
class AWDRNN(Seq2SeqEncoder):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self,
                 input_size: int, hidden_size: int, num_layers: int,
                 rnn_type: str = 'LSTM', dropout: float = 0.5, dropouth: float = 0.5,
                 dropouti: float = 0.5, dropoute: float = 0.1, wdrop: float = 0,
                 stack_rates: List[int] = None):
        super(AWDRNN, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)

        assert rnn_type in ['LSTM', 'GRU'], 'RNN type is not supported'
        assert len(stack_rates) == num_layers

        rnn = getattr(nn, rnn_type)
        self.rnns = [pRNNLayer(rnn,
                               input_size if l == 0 else hidden_size,
                               hidden_size, stack_rates[l],
                               wdrop)
                     for l in range(num_layers)]

        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462

        self.init_weights()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.stack_rates = stack_rates

    def reset(self):
        if self.rnn_type == 'QRNN':
            [r.reset() for r in self.rnns]

    def init_weights(self):
        pass

    def forward(self,
                emb: torch.Tensor,
                lengths: torch.Tensor,
                output_layer_num: int = None):
        emb = emb.transpose(1, 0)
        batch_size = emb.size(1)
        hidden = self.init_hidden(batch_size)
        if output_layer_num is None:
            output_layer_num = len(self.rnns)

        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        # raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns[:output_layer_num]):
            rnn_out, lengths = rnn(raw_output, hidden[l], lengths)
            raw_output, new_h = rnn_out
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            # if l != self.num_layers - 1:
            if l != output_layer_num - 1:
                # self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        return output.transpose(1, 0), hidden, lengths

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.hidden_size).zero_(),
                     weight.new(1, bsz, self.hidden_size).zero_())
                    for l in range(self.num_layers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.hidden_size).zero_()
                    for l in range(self.num_layers)]

    def get_input_dim(self) -> int:
        return self.input_size

    def get_output_dim(self) -> int:
        return self.hidden_size

    def is_bidirectional(self) -> bool:
        return False
