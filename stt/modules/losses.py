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
        predicted_classes = probs.multinomial(1).squeeze(1)
    else:
        _, predicted_classes = probs.max(1)

    return predicted_classes


class OCDLoss(nn.Module):
    def __init__(self, eos_id, init_temp, end_temp, final_epoch):

        super(OCDLoss, self).__init__()
        self.eos_id = eos_id
        self.temperature = float(init_temp)
        self.init_temp = float(init_temp)
        self.end_temp = float(end_temp)
        self.final_epoch = final_epoch  # num of epoch to reduce temperature to zero

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
        print("todo: make this function batch first")
        raise NotImplementedError
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

class EDOCDLoss(nn.Module):
    def __init__(self, eos_id: int, init_temp: float, end_temp: float, final_epoch: int,
                 average: str = 'batch'):

        super(EDOCDLoss, self).__init__()
        if average not in {None, "token", "batch"}:
            raise ValueError("Got average f{average}, expected one of "
                             "None, 'token', or 'batch'")
        self.average = average
        self.eos_id = eos_id
        self.temperature = float(init_temp)
        self.init_temp = float(init_temp)
        self.end_temp = float(end_temp)
        self.dsub = 1.0
        self.ddel = 1.0
        self.dins = 1.0
        self.kl_div = nn.KLDivLoss(reduction='none')
        self.final_epoch = final_epoch  # num of epoch to reduce temperature to zero

    def __call__(self, outputs, output_symbols, targets, mask):
        mask = mask.bool()
        q_values = self._construct_q_values(outputs, output_symbols, targets, mask)
        # pred_mask = self._get_pred_mask(output_symbols)
        pred_mask = mask
        optimal_policy = torch.softmax(q_values / self.temperature, dim=-1)
        #loss = torch.sum(
        #    optimal_policy * (torch.log(optimal_policy+1e-8) - outputs), dim=-1) * pred_mask.float()
        loss = self.kl_div(outputs, optimal_policy).sum(dim=-1) * pred_mask.float()
        weights_batch_sum = pred_mask.float().sum(dim=-1)
        if self.average == 'batch':
            per_batch_loss = loss.sum(dim=-1) / (weights_batch_sum + 1e-13)
            num_non_empty_sequences = ((weights_batch_sum > 0).float().sum() + 1e-13)
            return per_batch_loss.sum() / num_non_empty_sequences
        elif self.average == 'token':
            return loss.sum() / (weights_batch_sum.sum() + 1e-13)
        else:
            per_batch_loss = loss.sum(dim=-1) / (weights_batch_sum + 1e-13)
            return per_batch_loss

    def _get_pred_mask(self, output_symbols: torch.LongTensor) -> torch.BoolTensor:
        batch_size, max_pred_len = output_symbols.size()
        pred_mask = output_symbols.new_ones((batch_size, max_pred_len), dtype=torch.bool)
        for i in range(1, max_pred_len):
            pred_mask[:, i] = pred_mask[:, i - 1] * ~(output_symbols[:, i - 1] == self.eos_id)
        return pred_mask

    def _construct_q_values(self, outputs: torch.FloatTensor, output_symbols: torch.LongTensor,
                            targets: torch.LongTensor, mask: torch.BoolTensor) -> torch.FloatTensor:
        batch_size, max_pred_len, nlabels = outputs.size()
        _, max_len = targets.size()
        q_values = outputs.new_zeros((batch_size, max_pred_len, nlabels))
        distances = self._calculate_edit_distance(output_symbols, targets, mask)
        # distances = distances[:, :-1, :-1]
        min_dists, _ = torch.min(distances, dim=-1)
        truth_values = distances == min_dists.unsqueeze(-1)
        indices = truth_values.nonzero()
        extended_targets = targets.repeat(1, max_pred_len) \
            .view(-1, max_pred_len, max_len)
        # next_indices = indices.clone()
        gold_next_tokens = extended_targets[indices.split(1, dim=1)]
        indices[:, -1] = gold_next_tokens.squeeze(dim=1)
        q_values[indices.split(1, dim=1)] = 1
        q_values = q_values - (1 + min_dists).unsqueeze(-1)
        return q_values

    def _calculate_edit_distance(self, output_symbols: torch.LongTensor,
                                 targets: torch.LongTensor,
                                 mask: torch.BoolTensor) -> torch.FloatTensor:
        batch_size, max_pred_len = output_symbols.size()
        _, max_len = targets.size()

        distances = output_symbols.new_zeros(batch_size, max_pred_len, max_len)
        distances[:, :, 0] = torch.arange(max_pred_len)
        distances[:, 0, :] = torch.arange(max_len)
        distances = distances.float()

        for i in range(1, max_pred_len):
            for j in range(1, max_len):
                diagonal = distances[:, i-1, j-1] + \
                    self.dsub * (output_symbols[:, i-1] != targets[:, j-1]).float()
                comp = torch.stack(
                    (diagonal,
                     distances[:, i-1, j] + self.dins,
                     distances[:, i, j-1] + self.ddel), dim=-1)
                distances[:, i, j], _ = torch.min(comp, dim=-1)

        #edit_distance_mask = self._get_edit_distance_mask(mask, output_symbols)
        distances = distances.masked_fill(~mask.unsqueeze(1), float('inf'))
        return distances

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
