import torch


class MaskedBatchNorm1d(torch.nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super(MaskedBatchNorm1d).__init__(*args, **kwargs)

    def forward(self, inputs: torch.FloatTensor, mask: torch.ByteTensor):
        """
        Perform masked batch normalization for padded inputs.

        Parameters
        ----------
        inputs : ``torch.FloatTensor``
            A tensor of shape (batch_size, sequence_length, channel_size).
        mask : ``torch.ByteTensor``
            A tensor of shape (batch_size, sequence_length).
        """
        _, _, channel_size = inputs.size()
        inputs = inputs.masked_select(mask.unsqueeze(2).expand_as(inputs)) \
            .view(-1, channel_size, 1)  # (N*L, C)
        outputs = super().forward(inputs)
        return outputs
