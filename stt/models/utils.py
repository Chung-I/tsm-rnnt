import torch

def pad_to_multiples(array: torch.Tensor, multiple: int):
    batch_size, sequence_length, feat_size = array.size()
    if sequence_length % multiple == 0:
        return array, sequence_length
    else:
        padding_length = sequence_length + (-sequence_length) % multiple
        new_array = array.new_zeros((batch_size, padding_length, feat_size))
        new_array[:, :sequence_length, :] = array
    return new_array, padding_length
