import torch
from torch import nn
import torch.nn.functional as F

from allennlp.nn.util import replace_masked_values
from allennlp.modules.attention import Attention
from allennlp.common.registrable import Registrable


def safe_cumprod(x):
    """Numerically stable cumulative product by cumulative sum in log-space"""
    return torch.exp(torch.cumsum(torch.log(torch.clamp(x, min=1e-10, max=1)), dim=-1))


def moving_sum(x, back, forward):
    """Parallel moving sum with 1D Convolution"""
    # Pad window before applying convolution
    # [batch_size,    back + sequence_length + forward]
    x_padded = F.pad(x, pad=(back, forward))

    # Fake channel dimension for conv1d
    # [batch_size, 1, back + sequence_length + forward]
    x_padded = x_padded.unsqueeze(1)

    # Apply conv1d with filter of all ones for moving sum
    filters = x.new_ones(1, 1, back + forward + 1)

    x_sum = F.conv1d(x_padded, filters)

    # Remove fake channel dimension
    # [batch_size, sequence_length]
    return x_sum.squeeze(1)


def moving_max(x, w):
    """Compute the moving sum of x over a window with the provided bounds.

    x is expected to be of shape (batch_size, sequence_length).
    The returned tensor x_max is computed as
    x_max[i, j] = max(x[i, j - window + 1], ..., x[i, j])
    """
    # Pad x with -inf at the start
    x = F.pad(x, pad=(w - 1, 0), value=float("-inf"))
    # Add "channel" dimension (max_pool operates on 1D)
    x = x.unsqueeze(1)
    x = F.max_pool1d(x, kernel_size=w, stride=1).squeeze(1)
    return x


def frame(tensor, chunk_size, pad_end=False, value=0):
    if pad_end:
        padded_tensor = F.pad(tensor, pad=(0, chunk_size - 1), value=value)
    else:
        padded_tensor = F.pad(tensor, pad=(chunk_size - 1, 0), value=value)
    framed_tensor = F.unfold(padded_tensor.unsqueeze(1).unsqueeze(-1),
                             kernel_size=(chunk_size, 1)).transpose(-2, -1)
    return framed_tensor


class Energy(nn.Module):
    def __init__(self,
                 enc_dim: int,
                 dec_dim: int,
                 att_dim: int,
                 init_r: float = -0.1) -> None:
        """
        [Modified Bahdahnau attention] from
        "Online and Linear-Time Attention by Enforcing Monotonic Alignment" (ICML 2017)
        http://arxiv.org/abs/1704.00784

        Used for Monotonic Attention and Chunk Attention
        """
        super().__init__()
        self.tanh = nn.Tanh()
        self.W = nn.Linear(enc_dim, att_dim, bias=False)
        self.V = nn.Linear(dec_dim, att_dim, bias=False)
        self.b = nn.Parameter(torch.Tensor(att_dim).normal_())

        self.v = nn.utils.weight_norm(nn.Linear(att_dim, 1))
        self.v.weight_g.data = torch.Tensor([1 / att_dim]).sqrt()

        self.r = nn.Parameter(torch.Tensor([init_r]))

    def forward(self, encoder_outputs, decoder_h):
        """
        Args:
            encoder_outputs: [batch_size, sequence_length, enc_dim]
            decoder_h: [batch_size, dec_dim]
        Return:
            Energy [batch_size, sequence_length]
        """
        mask = encoder_outputs["mask"]
        encoder_outputs = encoder_outputs["value"]
        batch_size, sequence_length, enc_dim = encoder_outputs.size()
        encoder_outputs = encoder_outputs.reshape(-1, enc_dim)
        energy = self.tanh(self.W(encoder_outputs) +
                           self.V(decoder_h).repeat(sequence_length, 1) +
                           self.b)
        energy = self.v(energy).squeeze(-1) + self.r

        energy = energy.view(batch_size, sequence_length)
        return replace_masked_values(energy, mask, float('-inf'))


@Attention.register("monotonic")
class MonotonicAttention(Attention, Registrable):
    def __init__(self,
                 enc_dim: int,
                 dec_dim: int,
                 att_dim: int):
        """
        [Monotonic Attention] from
        "Online and Linear-Time Attention by Enforcing Monotonic Alignment" (ICML 2017)
        http://arxiv.org/abs/1704.00784
        """
        super().__init__()

        self.monotonic_energy = Energy(enc_dim, dec_dim, att_dim)

    def gaussian_noise(self, tensor):
        """Additive gaussian nosie to encourage discreteness"""
        return tensor.new_empty(tensor.size()).normal_()

    def exclusive_cumprod(self, x):
        """Exclusive cumulative product [a, b, c] => [1, a, a * b]
        * TensorFlow: https://www.tensorflow.org/api_docs/python/tf/cumprod
        * PyTorch: https://discuss.pytorch.org/t/cumprod-exclusive-true-equivalences/2614
        """
        batch_size, sequence_length = x.size()
        if torch.cuda.is_available():
            one_x = torch.cat(
                [torch.ones(batch_size, 1).cuda(), x], dim=1)[:, :-1]
        else:
            one_x = torch.cat([torch.ones(batch_size, 1), x], dim=1)[:, :-1]
        return torch.cumprod(one_x, dim=1)

    def soft_recursive(self, encoder_outputs, decoder_h, previous_alpha=None):
        """
        Soft monotonic attention (Train)
        Args:
            encoder_outputs [batch_size, sequence_length, enc_dim]
            decoder_h [batch_size, dec_dim]
            previous_alpha [batch_size, sequence_length]
        Return:
            alpha [batch_size, sequence_length]
        """
        batch_size, sequence_length, _ = encoder_outputs["value"].size()
        mask = encoder_outputs["mask"]
        end_mask = mask * F.pad((1 - mask), pad=(0, 1), value=1.)[:, 1:]

        monotonic_energy = self.monotonic_energy(encoder_outputs, decoder_h)
        # p_select = F.sigmoid(monotonic_energy)
        p_select = F.sigmoid(monotonic_energy +
                             self.gaussian_noise(monotonic_energy))
        p_select = torch.where(end_mask.byte(), end_mask, p_select)

        # p_select[:,-1] = 1.0
        shifted_1mp_choose_i = F.pad(
            1 - p_select[:, :-1], pad=(1, 0, 0, 0), value=1.0)

        if previous_alpha is None:
            # cumprod_1_minus_p = self.safe_cumprod(1 - p_select)
            # shifted_cum_1mp_choose_i = F.pad(
            #     cumprod_1_minus_p[:, :-1], pad=(1, 0, 0, 0), value=1.0)
            # alpha = p_select * shifted_cum_1mp_choose_i
            alpha = decoder_h.new_zeros(batch_size, sequence_length)
            # alpha[:, 0] = decoder_h.new_ones(batch_size)
            alpha[:, 0] = 1.0

        else:
            alpha_div_ps = []
            alpha_div_p = previous_alpha.new_zeros(batch_size)
            for j in range(sequence_length):
                alpha_div_p = shifted_1mp_choose_i[:, j] * \
                    alpha_div_p + previous_alpha[:, j]
                alpha_div_ps.append(alpha_div_p)
            alpha = p_select * torch.stack(alpha_div_ps, -1)

        return alpha

    def soft(self, encoder_outputs, decoder_h, previous_alpha=None):
        """
        Soft monotonic attention (Train)
        Args:
            encoder_outputs [batch_size, sequence_length, enc_dim]
            decoder_h [batch_size, dec_dim]
            previous_alpha [batch_size, sequence_length]
        Return:
            alpha [batch_size, sequence_length]
        """
        batch_size, sequence_length, enc_dim = encoder_outputs["value"].size()
        mask = encoder_outputs["mask"]
        end_mask = mask * F.pad((1 - mask), pad=(0, 1), value=1.)[:, 1:]

        monotonic_energy = self.monotonic_energy(encoder_outputs, decoder_h)
        # p_select = F.sigmoid(monotonic_energy)
        p_select = F.sigmoid(monotonic_energy +
                             self.gaussian_noise(monotonic_energy))
        p_select = torch.where(end_mask.byte(), end_mask, p_select)

        cumprod_1_minus_p = safe_cumprod(1 - p_select)

        if previous_alpha is None:
            # First iteration => alpha = [1, 0, 0 ... 0]
            # shifted_cum_1mp_choose_i = F.pad(
            #    cumprod_1_minus_p[:, :-1], pad=(1, 0, 0, 0), value=1.0)
            # alpha = p_select * shifted_cum_1mp_choose_i
            alpha = decoder_h.new_zeros(batch_size, sequence_length)
            alpha[:, 0] = decoder_h.new_ones(batch_size)

        else:
            # alpha = p_select * cumprod_1_minus_p * \
            #    torch.cumsum(previous_alpha / cumprod_1_minus_p, dim=1)
            alpha = p_select * cumprod_1_minus_p * \
                torch.cumsum(previous_alpha /
                             torch.clamp(cumprod_1_minus_p, 1e-10, 1.), dim=1)

        return alpha

    def hard(self, encoder_outputs, decoder_h, previous_attention=None):
        """
        Hard monotonic attention (Test)
        Args:
            encoder_outputs [batch_size, sequence_length, enc_dim]
            decoder_h [batch_size, dec_dim]
            previous_attention [batch_size, sequence_length]
        Return:
            alpha [batch_size, sequence_length]
        """
        batch_size, sequence_length, enc_dim = encoder_outputs["value"].size()

        if previous_attention is None:
            # First iteration => alpha = [1, 0, 0 ... 0]
            attention = decoder_h.new_zeros(batch_size, sequence_length)
            attention[:, 0] = decoder_h.new_ones(batch_size)
        else:
            # TODO: Linear Time Decoding
            # It's not clear if authors' TF implementation decodes in linear time.
            # https://github.com/craffel/mad/blob/master/example_decoder.py#L235
            # They calculate energies for whole encoder outputs
            # instead of scanning from previous attended encoder output.
            monotonic_energy = self.monotonic_energy(
                encoder_outputs, decoder_h)

            # Hard Sigmoid
            # Attend when monotonic energy is above threshold (Sigmoid > 0.5)
            above_threshold = (monotonic_energy > 0).float()

            p_select = above_threshold * \
                torch.cumsum(previous_attention, dim=1)
            attention = p_select * self.exclusive_cumprod(1 - p_select)

            # Not attended => attend at last encoder output
            # Assume that encoder outputs are not padded
            # attended = attention.sum(dim=1)
            # for batch_i in range(batch_size):
            #     if not attended[batch_i]:
            #         attention[batch_i, -1] = 1

            # Ex)
            # p_select                        = [0, 0, 0, 1, 1, 0, 1, 1]
            # 1 - p_select                    = [1, 1, 1, 0, 0, 1, 0, 0]
            # exclusive_cumprod(1 - p_select) = [1, 1, 1, 1, 0, 0, 0, 0]
            # attention: product of above     = [0, 0, 0, 1, 0, 0, 0, 0]
        return attention


@Attention.register("mocha")
class MoChA(MonotonicAttention):
    def __init__(self,
                 chunk_size: int,
                 enc_dim: int,
                 dec_dim: int,
                 att_dim: int) -> None:
        """
        [Monotonic Chunkwise Attention] from
        "Monotonic Chunkwise Attention" (ICLR 2018)
        https://openreview.net/forum?id=Hko85plCW
        """
        super().__init__(enc_dim, dec_dim, att_dim)
        self.chunk_size = chunk_size
        self.chunk_energy = Energy(enc_dim, dec_dim, att_dim)
        self.unfold = nn.Unfold(kernel_size=(self.chunk_size, 1))
        self.softmax = nn.Softmax(dim=1)

    def my_stable_chunkwise_attention_soft(self, emit_probs, chunk_energy):
        """
        PyTorch version of stable_chunkwise_attention in author's TF Implementation:
        https://github.com/craffel/mocha/blob/master/Demo.ipynb.
        Compute chunkwise attention distribution stably by subtracting logit max.
        """
        batch_size, _ = emit_probs.size()
        framed_chunk_energy = frame(
            chunk_energy, self.chunk_size, value=float("-inf"))

        chunk_probs = F.softmax(framed_chunk_energy, dim=-1)

        non_inf_mask = 1 - (framed_chunk_energy == float("-inf"))
        chunk_probs = torch.where(
            non_inf_mask, chunk_probs, non_inf_mask.float())

        weighted_chunk_probs = emit_probs.unsqueeze(-1) * chunk_probs
        kernel = torch.eye(
            self.chunk_size, dtype=chunk_probs.dtype, device=chunk_probs.device)
        kernel = torch.flip(kernel, dims=(-1,)).flatten()
        padded_chunk_probs = F.pad(
            weighted_chunk_probs, pad=(0, 0, 0, self.chunk_size - 1))
        beta = F.conv1d(padded_chunk_probs.view(batch_size, 1, -1),
                        kernel.view(1, 1, -1),
                        stride=self.chunk_size).squeeze(1)
        return beta

    def stable_chunkwise_attention_soft(self, emit_probs, chunk_energy):
        """
        PyTorch version of stable_chunkwise_attention in author's TF Implementation:
        https://github.com/craffel/mocha/blob/master/Demo.ipynb.
        Compute chunkwise attention distribution stably by subtracting logit max.
        """
        # Compute length-chunk_size sliding max of sequences in softmax_logits (m)
        # (batch_size, sequence_length)
        chunk_energy_max = moving_max(chunk_energy, self.chunk_size)
        framed_chunk_energy = frame(
            chunk_energy, self.chunk_size, value=float("-inf"))
        framed_chunk_energy = framed_chunk_energy - \
            chunk_energy_max.unsqueeze(-1)
        softmax_denominators = torch.sum(torch.exp(framed_chunk_energy), -1)
        # Construct matrix of framed denominators, padding at the end so the final
        # frame is [softmax_denominators[-1], inf, inf, ..., inf] (E)
        framed_denominators = frame(
            softmax_denominators, self.chunk_size, pad_end=True, value=float("inf"))
        framed_chunk_energy_max = frame(chunk_energy_max, self.chunk_size, pad_end=True,
                                        value=float("inf"))
        softmax_numerators = torch.exp(
            chunk_energy.unsqueeze(-1) - framed_chunk_energy_max)

        framed_probs = frame(emit_probs, self.chunk_size, pad_end=True)
        beta = torch.sum(framed_probs * softmax_numerators /
                         framed_denominators, dim=-1)
        return beta

    def chunkwise_attention_soft(self, alpha, u):
        """
        Args:
            alpha [batch_size, sequence_length]: emission probability in monotonic attention
            u [batch_size, sequence_length]: chunk energy
            chunk_size (int): window size of chunk
        Return
            beta [batch_size, sequence_length]: MoChA weights
        """

        # Numerical stability
        # Divide by same exponent => doesn't affect softmax
        u -= torch.max(u, dim=1, keepdim=True)[0]
        exp_u = torch.exp(u)
        # Limit range of logit
        exp_u = torch.clamp(exp_u, min=1e-5)

        # Moving sum:
        # Zero-pad (chunk size - 1) on the left + 1D conv with filters of 1s.
        # [batch_size, sequence_length]
        denominators = moving_sum(exp_u,
                                  back=self.chunk_size - 1, forward=0)

        # Compute beta (MoChA weights)
        beta = exp_u * moving_sum(alpha / denominators,
                                  back=0, forward=self.chunk_size - 1)
        return beta

    def chunkwise_attention_hard(self, monotonic_attention, chunk_energy):
        """
        Mask non-attended area with '-inf'
        Args:
            monotonic_attention [batch_size, sequence_length]
            chunk_energy [batch_size, sequence_length]
        Return:
            masked_energy [batch_size, sequence_length]
        """
        batch_size, sequence_length = monotonic_attention.size()

        mask = monotonic_attention.new_tensor(monotonic_attention)
        for i in range(1, self.chunk_size):
            mask[:, :-i] += monotonic_attention[:, i:]

        # mask '-inf' energy before softmax
        masked_energy = chunk_energy.masked_fill_(
            (1 - mask).byte(), -float('inf'))
        return masked_energy

    def soft(self, encoder_outputs, decoder_h, previous_alpha=None):
        """
        Soft monotonic chunkwise attention (Train)
        Args:
            encoder_outputs [batch_size, sequence_length, enc_dim]
            decoder_h [batch_size, dec_dim]
            previous_alpha [batch_size, sequence_length]
        Return:
            alpha [batch_size, sequence_length]
            beta [batch_size, sequence_length]
        """
        alpha = super().soft_recursive(encoder_outputs, decoder_h, previous_alpha)
        # alpha_fast = super().soft(encoder_outputs, decoder_h, previous_alpha)
        # print("recursive: {}".format(torch.sum(alpha, dim=-1)))
        # print("fast: {}".format(torch.sum(alpha_fast, dim=-1)))
        # alpha = super().soft(encoder_outputs, decoder_h, previous_alpha)
        # assert torch.allclose(torch.sum(alpha, dim=-1),
        #                       alpha.new_ones(alpha.size(0))), "{}".format(torch.sum(alpha, dim=-1))
        # assert torch.allclose(torch.sum(alpha_rec, dim=-1),
        #                      alpha.new_ones(alpha.size(0)),
        #                      rtol=1e-3,
        #                      atol=1e-3), "{}".format(torch.sum(alpha_rec, dim=-1))
        # assert torch.allclose(alpha_rec, alpha,e
        #                      rtol=1e-3,
        #                      atol=1e-3),  "{} {}".format(alpha, alpha_rec)
        chunk_energy = self.chunk_energy(encoder_outputs, decoder_h)
        # complex_beta = self.stable_chunkwise_attention_soft(
        #    alpha, chunk_energy)
        beta = self.my_stable_chunkwise_attention_soft(alpha, chunk_energy)
        # assert torch.allclose(
        #   complex_beta, beta),  "{} {}".format(complex_beta, beta)
        return alpha, beta

    def hard(self, encoder_outputs, decoder_h, previous_attention=None):
        """
        Hard monotonic chunkwise attention (Test)
        Args:
            encoder_outputs [batch_size, sequence_length, enc_dim]
            decoder_h [batch_size, dec_dim]
            previous_attention [batch_size, sequence_length]
        Return:
            monotonic_attention [batch_size, sequence_length]: hard alpha
            chunkwise_attention [batch_size, sequence_length]: hard beta
        """
        # hard attention (one-hot)
        # [batch_size, sequence_length]
        monotonic_attention = super().hard(encoder_outputs, decoder_h, previous_attention)
        chunk_energy = self.chunk_energy(encoder_outputs, decoder_h)
        masked_energy = self.chunkwise_attention_hard(
            monotonic_attention, chunk_energy)
        chunkwise_attention = self.softmax(masked_energy)
        chunkwise_attention.masked_fill_(
            chunkwise_attention != chunkwise_attention,
            0)  # a trick to replace nan value with 0
        return monotonic_attention, chunkwise_attention


@Attention.register("milk")
class MILk(MonotonicAttention):
    def __init__(self,
                 chunk_size: int,
                 enc_dim: int,
                 dec_dim: int,
                 att_dim: int) -> None:
        """
        [Monotonic Chunkwise Attention] from
        "Monotonic Chunkwise Attention" (ICLR 2018)
        https://openreview.net/forum?id=Hko85plCW
        """
        super().__init__(enc_dim, dec_dim, att_dim)
        self.chunk_size = chunk_size
        self.chunk_energy = Energy(enc_dim, dec_dim, att_dim)
        self.unfold = nn.Unfold(kernel_size=(self.chunk_size, 1))
        self.softmax = nn.Softmax(dim=1)

    def my_stable_chunkwise_attention_soft(self, emit_probs, chunk_energy):
        """
        PyTorch version of stable_chunkwise_attention in author's TF Implementation:
        https://github.com/craffel/mocha/blob/master/Demo.ipynb.
        Compute chunkwise attention distribution stably by subtracting logit max.
        """
        # framed_chunk_energy = frame(
        #    chunk_energy, self.chunk_size, value=float("-inf"))
        # framed_chunk_probs = F.softmax(framed_chunk_energy, dim=-1)
        # framed_emit_probs = frame(emit_probs, self.chunk_size, pad_end=True, value)
        pass

    def stable_chunkwise_attention_soft(self, emit_probs, chunk_energy):
        """
        PyTorch version of stable_chunkwise_attention in author's TF Implementation:
        https://github.com/craffel/mocha/blob/master/Demo.ipynb.
        Compute chunkwise attention distribution stably by subtracting logit max.
        """
        # Compute length-chunk_size sliding max of sequences in softmax_logits (m)
        # (batch_size, sequence_length)
        batch_size, sequence_length = chunk_energy
        exp_energy = torch.exp(chunk_energy)
        cum_exp_energy = torch.cumsum(exp_energy)
        chunk_probs = exp_energy.unsqueeze(1).repeat(
            1,  sequence_length, 1) / cum_exp_energy.unsqueeze(1)
        return beta
        pass

    def chunkwise_attention_soft(self, alpha, u):
        """
        Args:
            alpha [batch_size, sequence_length]: emission probability in monotonic attention
            u [batch_size, sequence_length]: chunk energy
            chunk_size (int): window size of chunk
        Return
            beta [batch_size, sequence_length]: MoChA weights
        """

        # Numerical stability
        # Divide by same exponent => doesn't affect softmax
        u -= torch.max(u, dim=1, keepdim=True)[0]
        exp_u = torch.exp(u)
        # Limit range of logit
        exp_u = torch.clamp(exp_u, min=1e-5)

        # Moving sum:
        # Zero-pad (chunk size - 1) on the left + 1D conv with filters of 1s.
        # [batch_size, sequence_length]
        denominators = moving_sum(exp_u,
                                  back=self.chunk_size - 1, forward=0)

        # Compute beta (MoChA weights)
        beta = exp_u * moving_sum(alpha / denominators,
                                  back=0, forward=self.chunk_size - 1)
        return beta

    def chunkwise_attention_hard(self, monotonic_attention, chunk_energy):
        """
        Mask non-attended area with '-inf'
        Args:
            monotonic_attention [batch_size, sequence_length]
            chunk_energy [batch_size, sequence_length]
        Return:
            masked_energy [batch_size, sequence_length]
        """
        batch_size, sequence_length = monotonic_attention.size()

        mask = monotonic_attention.new_tensor(monotonic_attention)
        for i in range(1, self.chunk_size):
            mask[:, :-i] += monotonic_attention[:, i:]

        # mask '-inf' energy before softmax
        masked_energy = chunk_energy.masked_fill_(
            (1 - mask).byte(), -float('inf'))
        return masked_energy

    def soft(self, encoder_outputs, decoder_h, previous_alpha=None):
        """
        Soft monotonic chunkwise attention (Train)
        Args:
            encoder_outputs [batch_size, sequence_length, enc_dim]
            decoder_h [batch_size, dec_dim]
            previous_alpha [batch_size, sequence_length]
        Return:
            alpha [batch_size, sequence_length]
            beta [batch_size, sequence_length]
        """
        alpha = super().soft_recursive(encoder_outputs, decoder_h, previous_alpha)
        chunk_energy = self.chunk_energy(encoder_outputs, decoder_h)
        beta = self.stable_chunkwise_attention_soft(alpha, chunk_energy)
        return alpha, beta

    def hard(self, encoder_outputs, decoder_h, previous_attention=None):
        """
        Hard monotonic chunkwise attention (Test)
        Args:
            encoder_outputs [batch_size, sequence_length, enc_dim]
            decoder_h [batch_size, dec_dim]
            previous_attention [batch_size, sequence_length]
        Return:
            monotonic_attention [batch_size, sequence_length]: hard alpha
            chunkwise_attention [batch_size, sequence_length]: hard beta
        """
        # hard attention (one-hot)
        # [batch_size, sequence_length]
        monotonic_attention = super().hard(encoder_outputs, decoder_h, previous_attention)
        chunk_energy = self.chunk_energy(encoder_outputs, decoder_h)
        masked_energy = self.chunkwise_attention_hard(
            monotonic_attention, chunk_energy)
        chunkwise_attention = self.softmax(masked_energy)
        chunkwise_attention.masked_fill_(
            chunkwise_attention != chunkwise_attention,
            0)  # a trick to replace nan value with 0
        return monotonic_attention, chunkwise_attention
