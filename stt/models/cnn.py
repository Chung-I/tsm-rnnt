from typing import Tuple
from overrides import overrides
from collections import OrderedDict
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.nn.activations import Activation
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder


class LengthAwareWrapper(nn.Module):
    def __init__(self, module, pass_through: str = False):
        super(LengthAwareWrapper, self).__init__()
        self.module = module
        self._pass_through = pass_through
        if not pass_through:
            self.padding = self.module.padding[0] \
                if isinstance(self.module.padding, tuple) \
                else self.module.padding

            self.dilation = self.module.dilation[0] \
                if isinstance(self.module.dilation, tuple) \
                else self.module.dilation

            self.kernel_size = self.module.kernel_size[0] \
                if isinstance(self.module.kernel_size, tuple) \
                else self.module.kernel_size

            self.stride = self.module.stride[0] \
                if isinstance(self.module.stride, tuple) \
                else self.module.stride

    @overrides
    def forward(self, inputs_and_lengths: Tuple[torch.FloatTensor, torch.LongTensor]):
        # pylint: disable=arguments-differ
        """
        Expect inputs of (N, C, T, D) dimension.
        There's something peculiar here in that we made the padding affects our length
        only at the start position, so that 2 * self.padding becomes 1 * self.padding.
        """
        inputs, lengths = inputs_and_lengths
        if not self._pass_through:
            lengths = torch.floor(
                torch.clamp(
                    (lengths.float() + 2 * self.padding - self.dilation *
                     (self.kernel_size - 1) - 1) / self.stride + 1,
                    1.
                )
            ).long()
        outputs = self.module.forward(inputs)
        return (outputs, lengths)

@Seq2SeqEncoder.register("vgg")
class VGGExtractor(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(VGGExtractor, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.hidden_channel = self.out_channel // 2

        self.conv1 = LengthAwareWrapper(
            nn.Conv2d(in_channel, self.hidden_channel, 3, stride=1, padding=1))
        self.conv2 = LengthAwareWrapper(
            nn.Conv2d(self.hidden_channel, self.hidden_channel, 3, stride=1, padding=1))
        self.pool1 = LengthAwareWrapper(
            nn.MaxPool2d(2, stride=2))  # Half-time dimension
        self.conv3 = LengthAwareWrapper(
            nn.Conv2d(self.hidden_channel, self.out_channel, 3, stride=1, padding=1))
        self.conv4 = LengthAwareWrapper(
            nn.Conv2d(self.out_channel, self.out_channel, 3, stride=1, padding=1))
        self.pool2 = LengthAwareWrapper(
            nn.MaxPool2d(2, stride=2))  # Half-time dimension

    @overrides
    def forward(self, feature, lengths):
        # pylint: disable=arguments-differ
        # Feature shape BSxTxD -> BS x CH(num of delta) x T x D(acoustic feature dim)
        batch_size, _, feat_dim = feature.size()
        feature = feature.unsqueeze(1)
        feature, lengths = self.conv1(feature, lengths)
        feature = F.relu(feature)
        feature, lengths = self.conv2(feature, lengths)
        feature = F.relu(feature)
        feature, lengths = self.pool1(feature, lengths)  # BSx32xT/2xD/2
        feature, lengths = self.conv3(feature, lengths)
        feature = F.relu(feature)
        feature, lengths = self.conv4(feature, lengths)
        feature = F.relu(feature)
        feature, lengths = self.pool2(feature, lengths)  # BSx64xT/4xD/4
        # BSx64xT/4xD/4 -> BSxT/4x64xD/4
        feature = feature.transpose(1, 2)
        #  BS x T/4 x 64 x D/4 -> BS x T/4 x 16D
        feature = feature.reshape(
            batch_size, -1, feat_dim * self.out_channel // 4)
        return feature, lengths

@Seq2SeqEncoder.register("cnn")
class CNN(Seq2SeqEncoder):
    def __init__(self, num_layers: int, in_channel: int, hidden_channel: int,
                 kernel_size: int = 3, stride: int = 2,
                 nonlinearity: Activation = Activation.by_name('linear')()):
        super(CNN, self).__init__()
        self._in_channel = in_channel
        self._hidden_channel = hidden_channel
        self._kernel_size = kernel_size
        self._stride = stride
        self._num_layers = num_layers
        layers = []
        for l in range(num_layers):
            in_channel = self._in_channel if l == 0 else self._hidden_channel
            conv = LengthAwareWrapper(nn.Conv2d(in_channel, self._hidden_channel,
                                                self._kernel_size, stride=self._stride,
                                                padding=1))
            layers.append((f"conv{l}", conv))
            layers.append((f"nonlinear{l}", LengthAwareWrapper(nonlinearity, pass_through=True)))
        self.module = nn.Sequential(OrderedDict(layers))
        strides = [self.module[idx].stride for idx in range(len(self.module))
                   if hasattr(self.module[idx], "stride")]
        self._downsample_rate = reduce(lambda x, y: x * y, strides)

    @overrides
    def get_input_dim(self) -> int:
        return self._in_channel

    @overrides
    def get_output_dim(self) -> int:
        return self._hidden_channel

    @overrides
    def is_bidirectional(self) -> bool:
        return False

    @overrides
    def forward(self, feature, lengths):
        # Feature shape BSxTxD -> BS x CH(num of delta) x T x D(acoustic feature dim)
        batch_size, _, feat_dim = feature.size()
        feature = feature.unsqueeze(1)
        feature, lengths = self.module((feature, lengths))
        feature = feature.transpose(1, 2)
        #  BS x T/4 x 64 x D/4 -> BS x T/4 x 16D
        feature = feature.reshape(
            batch_size, -1, feat_dim * self._hidden_channel // self._downsample_rate)
        return feature, lengths
