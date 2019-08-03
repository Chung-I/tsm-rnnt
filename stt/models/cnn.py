import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.nn.util import get_mask_from_sequence_lengths


class LengthAwareWrapper(nn.Module):
    def __init__(self, module):
        super(LengthAwareWrapper, self).__init__()
        self.module = module

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

    def forward(self, inputs: torch.FloatTensor, lengths: torch.LongTensor):
        """
        Expect inputs of (N, C, T, D) dimension.
        There's something peculiar here in that we made the padding affects our length
        only at the start position, so that 2 * self.padding becomes 1 * self.padding.
        """
        lengths = torch.floor(
            torch.clamp(
                (lengths.float() + 1 * self.padding - self.dilation *
                 (self.kernel_size - 1) - 1) / self.stride + 1,
                1.
            )
        ).long()
        outputs = self.module.forward(inputs)
        return outputs, lengths


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

    def forward(self, feature, lengths):
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
