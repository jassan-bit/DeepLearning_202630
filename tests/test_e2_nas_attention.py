"""Synthetic mechanism checks for the two new attention NAS families."""
import sys
from pathlib import Path
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from e2_nas_common import NASConfig
from e2_nas_attention import PatchTSTNAS, HybridNAS


def config(widths=(16, 32, 64), bits=(0, 1, 0)):
    return NASConfig(widths=widths, dropout_pattern=bits)


class AttentionFactoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(821)
        self.x = torch.randn(2, 7, 5)

    def test_asymmetric_widths_preserve_positive_scalar_shapes_and_gradients(self):
        for cls in (PatchTSTNAS, HybridNAS):
            for widths in ((16, 32, 64), (64, 16, 32)):
                with self.subTest(model=cls.__name__, widths=widths):
                    model = cls(config(widths))
                    y = model(self.x)
                    self.assertEqual(y.shape, (2,))
                    self.assertTrue(bool(torch.isfinite(y).all()))
                    self.assertTrue(bool((y > 0).all()))
                    y.sum().backward()
                    for name, p in model.named_parameters():
                        self.assertIsNotNone(p.grad, name)
                        self.assertTrue(bool(torch.isfinite(p.grad).all()), name)

    def test_patch_normalizes_real_window_before_replication(self):
        model = PatchTSTNAS(config()).eval()
        x = torch.arange(70, dtype=torch.float32).reshape(2, 7, 5)
        expected = (x - x.mean(1, keepdim=True)) / torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
        patches = model.normalize_and_patch(x)
        self.assertEqual(patches.shape, (2, 5, 2, 16))
        # First patch positions9:16 contain exactly the seven real normalized days.
        torch.testing.assert_close(patches[:, :, 0, 9:], expected.transpose(1, 2))
        torch.testing.assert_close(patches[:, :, 0, :9], expected[:, :1].transpose(1, 2).expand(-1, -1, 9))
        torch.testing.assert_close(patches[:, :, 1, -8:], expected[:, -1:].transpose(1, 2).expand(-1, -1, 8))
        self.assertEqual(model.head.in_features, 5 * 2 * 64 + 10)

    def test_patch_heads_feedforward_dropout_and_no_cross_layer_attention(self):
        model = PatchTSTNAS(config((16, 64, 32), (1, 0, 1)))
        self.assertEqual([b.heads for b in model.blocks], [1, 4, 2])
        self.assertEqual([b.feedforward_width for b in model.blocks], [128, 512, 256])
        self.assertEqual([b.attention_dropout.p for b in model.blocks], [0.2, 0.0, 0.2])
        self.assertEqual([b.feedforward_dropout.p for b in model.blocks], [0.2, 0.0, 0.2])
        self.assertTrue(all(b.attention.dropout == 0 for b in model.blocks))
        self.assertTrue(all(not b.residual_attention for b in model.blocks))
        self.assertEqual(sum(isinstance(m, nn.Dropout) for m in model.head.modules()), 0)

    def test_eval_is_batch_independent_and_batchnorm_stats_are_frozen(self):
        for cls in (PatchTSTNAS, HybridNAS):
            with self.subTest(model=cls.__name__):
                model = cls(config(bits=(1, 1, 1)))
                model(self.x)  # Update BN, when present, from synthetic training input.
                model.eval()
                states = {k: v.clone() for k, v in model.state_dict().items()}
                with torch.no_grad():
                    solo = model(self.x[:1])
                    combined = model(torch.cat((self.x[:1], 100 * self.x[1:]), 0))[:1]
                torch.testing.assert_close(solo, combined, rtol=1e-5, atol=1e-6)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, states[name])

    def test_singleton_training_is_supported(self):
        for cls in (PatchTSTNAS, HybridNAS):
            with self.subTest(model=cls.__name__):
                y = cls(config())(self.x[:1])
                self.assertEqual(y.shape, (1,))
                y.sum().backward()

    def test_hybrid_attention_is_simplex_and_state_is_reset(self):
        model = HybridNAS(config((64, 16, 32), (1, 1, 1))).eval()
        with torch.no_grad():
            y, weights = model.forward_with_attention(self.x)
            model(200 * self.x)
            again, _ = model.forward_with_attention(self.x)
        self.assertEqual(weights.shape, (2, 7))
        self.assertTrue(bool((weights >= 0).all()))
        torch.testing.assert_close(weights.sum(1), torch.ones(2))
        torch.testing.assert_close(y, again)
        self.assertEqual(model.head.in_features, 16)
        self.assertEqual((model.cnn_dropout.p, model.sequence_dropout.p, model.context_dropout.p), (0.2, 0.2, 0.2))

    def test_hybrid_prefix_causality_for_all_kernel_candidates(self):
        for kernel in (2, 3, 5):
            with self.subTest(kernel=kernel):
                model = HybridNAS(config(), kernel=kernel).eval()
                x2 = self.x.clone()
                x2[:, 4:] = torch.randn_like(x2[:, 4:]) * 100
                torch.testing.assert_close(model.encode_steps(self.x)[:, :4], model.encode_steps(x2)[:, :4])
                torch.testing.assert_close(model.encode_steps(self.x[:, :4]), model.encode_steps(self.x)[:, :4])

    def test_bad_input_or_kernel_is_rejected(self):
        for bad in (1, 4, True, 3.0):
            with self.assertRaises(ValueError):
                HybridNAS(config(), kernel=bad)
        for cls in (PatchTSTNAS, HybridNAS):
            model = cls(config())
            for x in (torch.zeros(2, 8, 5), torch.zeros(2, 7, 4), torch.zeros(2, 7, 5, dtype=torch.int64)):
                with self.assertRaises((ValueError, TypeError)):
                    model(x)
            invalid = self.x.clone()
            invalid[0, 0, 0] = float("nan")
            with self.assertRaises((ValueError, TypeError)):
                model(invalid)


if __name__ == "__main__":
    unittest.main()
