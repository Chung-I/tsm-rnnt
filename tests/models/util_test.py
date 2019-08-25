import unittest
import torch
import torch.nn.functional as F
from stt.models.util import averaging_tensor_of_same_label
from allennlp.nn.util import get_mask_from_sequence_lengths
import pdb


class TestUtil(unittest.TestCase):
    def setUp(self):
        super(TestUtil, self).__init__()

    def test_average_tensor_of_same_labels(self):
        batch_size = 10
        max_len = 16
        feat_dim = 32
        label_dim = 4
        for _ in range(10):
            phn_logits = torch.randn(batch_size, max_len, label_dim)
            phn_log_probs = F.log_softmax(phn_logits)
            lengths = torch.randint(label_dim, (batch_size,))
            mask = get_mask_from_sequence_lengths(lengths, int(max(lengths)))
            enc_outs = torch.randn((batch_size, max_len, feat_dim))
            _, max_ids = phn_log_probs.max(dim=-1)
            phn_enc_out_list = []
            for b in range(batch_size):
                count = 1
                phn_enc_out = []
                feat = enc_outs[b, 0].clone()
                prev_id = None
                for t, max_id in enumerate(max_ids[b]):
                    if prev_id is None:
                        pass
                    elif max_id == prev_id:
                        feat += enc_outs[b, t].clone()
                        count += 1
                    else:
                        phn_enc_out.append(feat.div(count))
                        feat = enc_outs[b, t].clone()
                        count = 1
                    prev_id = max_id
                phn_enc_out.append(feat/float(count))
                phn_enc_out_list.append(phn_enc_out)
            phn_max_len = len(max(phn_enc_out_list, key=lambda x: len(x)))
            phn_enc_outs = enc_outs.new_zeros(batch_size, phn_max_len, feat_dim)
            for idx, phn_enc_out in enumerate(phn_enc_out_list):
                phn_enc_outs[idx, :len(phn_enc_out)] = torch.stack(phn_enc_out)
                        
            len_phn_enc_outs, _ = averaging_tensor_of_same_label(enc_outs, phn_log_probs, lengths)
            torch.testing.assert_allclose(phn_enc_outs, len_phn_enc_outs)
            mask_phn_enc_outs, _ = averaging_tensor_of_same_label(enc_outs, phn_log_probs, mask)
            torch.testing.assert_allclose(phn_enc_outs, mask_phn_enc_outs)

