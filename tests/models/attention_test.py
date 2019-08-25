from unittest import TestCase
from numpy.testing import assert_almost_equal
import torch

from allennlp.common import Params

from stt.modules.attention import soft_efficient, MoChA, MILk
from allennlp.common.testing import AllenNlpTestCase


class AttentionTest(AllenNlpTestCase):

    def test_monotonic_soft(self):
        p_select = torch.FloatTensor(
            [[0.2, 0.1, 0.15, 0.7, 0.3, 1.0]])
        prev_alpha = torch.FloatTensor(
            [[0.1, 0.2, 0.3, 0.15, 0.25, 0.0]])
        alpha = soft_efficient(p_select, prev_alpha)
        assert_almost_equal(
            alpha, [[0.02, 0.028, 0.0828, 0.43344, 0.130728, 0.305032]])
        assert_almost_equal(torch.sum(alpha, -1), [1])

    def test_my_chunkwise_soft(self):
        params = Params({
            "chunk_size": 2,
            "enc_dim": 10,
            "dec_dim": 10,
            "att_dim": 10
        })
        att = MoChA.from_params(params)
        emit_probs = torch.FloatTensor(
            [[0.1, 0.2, 0.3, 0.15, 0.25, 0.0],
             [0.3, 0.7, 0.0,  0.0,  0.0, 0.0]])
        chunk_probs = torch.FloatTensor([[0.15, 0.15, 0.3, 0.3, 0.1, 0.0],
                                         [0.5,   0.5, 0.0, 0.0, 0.0, 0.0]])
        chunk_energy = torch.log(chunk_probs)
        beta = att.my_soft(emit_probs, chunk_energy)
        assert_almost_equal(
            beta, [[0.2,   0.2, 0.275, 0.2625, 0.0625, 0.0],
                   [0.65, 0.35,   0.0,    0.0,    0.0, 0.0]])
        assert_almost_equal(torch.sum(beta, -1), [1, 1])

    def test_author_chunkwise_soft(self):
        params = Params({
            "chunk_size": 2,
            "enc_dim": 10,
            "dec_dim": 10,
            "att_dim": 10
        })
        att = MoChA.from_params(params)
        emit_probs = torch.FloatTensor(
            [[0.1, 0.2, 0.3, 0.15, 0.25, 0.0],
             [0.3, 0.7, 0.0,  0.0,  0.0, 0.0]])
        chunk_probs = torch.FloatTensor([[0.15, 0.15, 0.3, 0.3, 0.1, 0.0],
                                         [0.5,   0.5, 0.0, 0.0, 0.0, 0.0]])
        chunk_energy = torch.log(chunk_probs)
        beta = att.stable_soft(emit_probs, chunk_energy)
        assert_almost_equal(
            beta, [[0.2,   0.2, 0.275, 0.2625, 0.0625, 0.0],
                   [0.65, 0.35,   0.0,    0.0,    0.0, 0.0]])
        assert_almost_equal(torch.sum(beta, -1), [1, 1])

    def test_milk_soft(self):
        params = Params({
            "enc_dim": 10,
            "dec_dim": 10,
            "att_dim": 10
        })
        att = MILk.from_params(params)
        emit_probs = torch.FloatTensor(
            [[0.1, 0.2, 0.3, 0.15, 0.25, 0.0],
             [0.3, 0.7, 0.0,  0.0,  0.0, 0.0]])
        chunk_probs = torch.FloatTensor([[0.15, 0.15, 0.3, 0.3, 0.1, 0.0],
                                         [0.5,   0.5, 0.0, 0.0, 0.0, 0.0]])
        chunk_energy = torch.log(chunk_probs)
        beta = att.soft(emit_probs, chunk_energy)
        assert_almost_equal(
            beta, [[0.3375, 0.2375, 0.275, 0.125, 0.025, 0.0],
                   [0.65, 0.35,   0.0,    0.0,    0.0, 0.0]])
        assert_almost_equal(torch.sum(beta, -1), [1, 1])
