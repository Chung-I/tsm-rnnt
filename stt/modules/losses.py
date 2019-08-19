import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.autograd import Variable


def as_set(targets: torch.LongTensor):
    """
    shape of targets: (batch_size, label_size)
    """
    return targets > 0


def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y
    y_tensor = y_tensor.long().reshape(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = y.new_zeros(y_tensor.size(0),
                            n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot


def target_to_candidates(targets, label_size, ignore_indices):
    mask = targets.new_zeros((label_size))
    for index in ignore_indices:
        mask[index] = 1
    targets = to_one_hot(targets, label_size).sum(dim=1)
    one_hot_mask = mask.byte().unsqueeze(0).expand_as(targets)
    targets = targets.masked_fill(one_hot_mask, 0)
    return targets


def maybe_sample_from_candidates(probs: torch.FloatTensor,
                                 candidates: torch.LongTensor = None,
                                 strategy="sample"):
    assert strategy in ["sample", "max"], "strategy must be one of [sample, max], \
        got {} instead".format(strategy)
    if candidates is not None:
        mask = (1 - (candidates > 0)).byte().to(probs.device)
        probs = probs.masked_fill(mask, 0)
    if strategy == "sample":
        predicted_classes = probs.multinomial(1)
    else:
        _, predicted_classes = probs.max(1)

    return predicted_classes


class OCDLoss(nn.Module):
    def __init__(self, eos_id, init_temp, end_temp, final_epoch):

        self.eos_id = eos_id
        self.temperature = float(init_temp)
        self.init_temp = float(init_temp)
        self.end_temp = float(end_temp)
        self.final_epoch = final_epoch  # num of epoch to reduce temperature to zero
        super(OCDLoss, self).__init__()

    def __call__(self, outputs, output_symbols, targets):
        '''
        Inputs:
            outputs: (seq_len, batch_size, label_size)
            output_symbols : (seq_len, batch_size) index of output symbol (sampling from policy)
            targets: (batch_size, label_size)
        '''
        # some details:
        # directly minize specific score
        # give sos low score
        # outputs = torch.stack(outputs)
        # targets = targets.to(outputs.device)

        # output_symbols = torch.stack(output_symbols).squeeze(2)
        seq_len, batch_size, label_size = outputs.size()

        outputs_one_hot = to_one_hot(
            output_symbols, label_size)
        q_values = outputs.new_zeros(outputs.shape)

        mask = outputs.new_ones((seq_len, batch_size))

        q_values[0, :, :] = -1 + targets
        for i in range(1, seq_len):
            # batch_size * label_size
            is_correct = (targets > 0).float() * \
                outputs_one_hot[i-1, :, :].float()
            targets = targets.float() - is_correct
            q_values[i, :, :] = q_values[i-1, :, :] - is_correct + \
                torch.sum(is_correct, dim=1).unsqueeze(1) - 1

            # check if all targets are sucessfully predicted
            is_end_batch = torch.sum(targets, dim=1).eq(0)
            q_values[i, :, self.eos_id] += is_end_batch.float()

            # check eos in output token
            eos_batches = output_symbols[i-1, :].data.eq(self.eos_id)
            eos_batches = eos_batches.float()
            mask[i, :] = (1 - eos_batches) * mask[i-1, :]

        optimal_policy = torch.softmax(q_values / self.temperature, dim=2)
        losses = torch.mean(
            optimal_policy * (torch.log(optimal_policy+1e-8) - outputs), dim=2) * mask
        loss = torch.sum(losses) / torch.sum(mask)
        return loss

    def update_temperature(self, epoch):
        if epoch >= self.final_epoch:
            self.temperature = self.end_temp
        else:
            self.temperature = self.init_temp - \
                (self.init_temp - self.end_temp) / \
                self.final_epoch * float(epoch)


class OrderFreeLoss(nn.Module):
    def __init__(self, eos_id):
        super(OrderFreeLoss, self).__init__()
        self.eos_id = eos_id
        self.criterion = nn.NLLLoss()

    def __call__(self, outputs, output_symbols, targets):
        '''
        Inputs:
            outputs: (seq_len, batch_size, label_size)
            output_symbols : (seq_len, batch_size) index of output symbol (sampling from policy)
            targets: (batch_size, label_size)
        '''
        '''
        outputs = torch.stack(outputs)

        output_symbols = torch.stack(output_symbols).squeeze(2)

        seq_len, batch_size, label_size = outputs.shape

        outputs = outputs.transpose(0,1) # batch_size * seq_len * label_size
        outputs = outputs.transpose(1,2) # batch_size * label_size * seq_len

        mask = torch.ones((seq_len, batch_size),
                          dtype = torch.float32, device = outputs.device)
        mask[1:,:] = 1 - output_symbols[:-1,:].data.eq(self.eos_id).float()

        losses = self.criterion(outputs, output_symbols.transpose(0,1)) * mask
        loss = torch.sum(losses) / torch.sum(mask)
        '''

        outputs = torch.stack(outputs)
        targets = targets.to(outputs.device)

        output_symbols = torch.stack(output_symbols).squeeze(2)
        seq_len, batch_size, label_size = outputs.shape

        # dtype=torch.long, device=outputs.device)
        target_time = targets.new_zeros((seq_len, batch_size))
        mask = outputs.new_ones((seq_len, batch_size))
        # dtype=torch.float32, device=outputs.device)

        for i in range(0, seq_len):
            target_time[i] = (torch.exp(outputs[i]) *
                              targets).topk(1, dim=1)[1].squeeze(1)
            targets = targets - to_one_hot(target_time[i], label_size)

            # check if all targets are sucessfully predicted
            is_end_batch = torch.sum(targets, dim=1).eq(0)
            targets[:, self.eos_id] += is_end_batch.float()

            # check eos in output token
            if i > 0:
                eos_batches = target_time[i-1, :].data.eq(self.eos_id)
                eos_batches = eos_batches.float()
                mask[i, :] = (1 - eos_batches) * mask[i-1, :]

        losses = self.criterion(outputs.permute(1, 2, 0), target_time.transpose(
            0, 1)) * mask  # (batch_size, label_size, seq_len)
        loss = torch.sum(losses) / torch.sum(mask)
        return loss
