"""Numerical mechanism checks for the new spectral NAS families."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import torch
from e2_nas_common import NASConfig
from e2_nas_spectral import (
    TimesNetNAS, TimeMixerNAS, _OddKernelInception, _NASPeriodBlock,
    _NASPDMBlock, _moving_average,
)


class SpectralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(281230)

    def test_inception_fusion_preserves_all_kernels_outputs_and_gradients(self):
        fused = _OddKernelInception(2, 4, 6).double()
        literal = deepcopy(fused)
        a = torch.randn(2, 2, 1, 8, dtype=torch.float64, requires_grad=True)
        b = a.detach().clone().requires_grad_(True)
        actual = fused(a)
        expected = torch.stack([kernel(b) for kernel in literal.kernels]).mean(0)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-11, atol=1e-12)
        self.assertEqual(len(list(fused.kernels)), 6)
        for one, two in zip(fused.parameters(), literal.parameters()):
            self.assertIsNotNone(one.grad)
            torch.testing.assert_close(one.grad, two.grad, rtol=1e-11, atol=1e-12)

    def test_period_selection_is_per_example_without_dc(self):
        t = torch.arange(8, dtype=torch.float64)
        low = 3*torch.cos(2*torch.pi*t/8)+2*torch.cos(2*torch.pi*2*t/8)
        high = 3*torch.cos(2*torch.pi*3*t/8)+2*torch.cos(torch.pi*t)
        values = torch.stack((low, high))[:, :, None].repeat(1, 1, 16)
        block = _NASPeriodBlock(16, 2, 2).double()
        weights = block.period_weights(values)
        torch.testing.assert_close(weights.sum(1), torch.ones(2, dtype=torch.float64))
        self.assertTrue(bool((weights[0, :2] > 0).all()))
        self.assertTrue(bool((weights[0, 2:] == 0).all()))
        self.assertTrue(bool((weights[1, :2] == 0).all()))
        self.assertTrue(bool((weights[1, 2:] > 0).all()))
        self.assertEqual(block.periods, (8, 4, 2, 2))
        for k in (2, 3, 4):
            block.top_k = k
            self.assertTrue(bool(((block.period_weights(values)>0).sum(1) == k).all()))

    def test_timesnet_projection_shapes_positive_output_and_finite_connected_gradients(self):
        config = NASConfig((16, 64, 32), (1, 0, 1))
        model = TimesNetNAS(config, top_k=3, num_kernels=2)
        self.assertEqual([block.width for block in model.blocks], [16, 64, 32])
        self.assertEqual([projection.in_features for projection in model.channel_projections], [32, 16, 64])
        self.assertEqual([drop.p for drop in model.dropouts], [0.2, 0.0, 0.2])
        x = torch.randn(2, 7, 5, requires_grad=True)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2,))
        self.assertTrue(bool((y>0).all()))
        y.sum().backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(bool(torch.isfinite(param.grad).all()), name)

    def test_both_families_eval_are_batch_independent_including_singleton(self):
        x = torch.randn(3, 7, 5)
        for model in [TimesNetNAS(NASConfig((16, 32, 16), (1, 1, 1)), num_kernels=2),
                      TimeMixerNAS(NASConfig((16, 32, 16), (1, 1, 1)))]:
            with self.subTest(model=type(model).__name__):
                model.eval()
                with torch.no_grad():
                    alone = model(x[:1])
                    batched = model(x)
                    changed = x.clone()
                    changed[1:] = 200*changed[1:]+500
                    perturbed = model(changed)
                torch.testing.assert_close(alone[0], batched[0], rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(alone[0], perturbed[0], rtol=2e-5, atol=2e-6)

    def test_timemixer_preserves_scales_and_normalization_degeneracy(self):
        model = TimeMixerNAS(NASConfig((16, 32, 64), (0, 0, 0)))
        x = torch.arange(35, dtype=torch.float32).reshape(1, 7, 5)
        levels = model.input_scales(x)
        self.assertEqual([level.shape[1] for level in levels], [7, 3, 1])
        torch.testing.assert_close(levels[1], (x[:, :6:2]+x[:, 1:6:2])/2)
        torch.testing.assert_close(levels[2], levels[1][:, :2].mean(1, keepdim=True))
        # A one-step normalization has no temporal contrast. This is a declared
        # source-geometry limitation, not extra observed history.
        torch.testing.assert_close(model.normalizers[2](levels[2]), torch.zeros_like(levels[2]))

    def test_moving_average_reconstruction_and_directional_scale_dependencies(self):
        for window in (3, 7, 25):
            block = _NASPDMBlock(16, window)
            levels = [torch.randn(2, length, 16) for length in [7, 3, 1]]
            season, trend = block.decompose(levels)
            for values, s, t in zip(levels, season, trend):
                torch.testing.assert_close(s+t, values)
            torch.testing.assert_close(season[-1], torch.zeros_like(season[-1]), atol=1e-6, rtol=0)
            constant = torch.ones(2, 1, 16)*3
            torch.testing.assert_close(_moving_average(constant, window), constant)
            mixed_season = block.mix_seasons(season)
            changed_season = [v.clone() for v in season]
            changed_season[0] += 1
            self.assertFalse(torch.allclose(mixed_season[-1], block.mix_seasons(changed_season)[-1]))
            mixed_trend = block.mix_trends(trend)
            changed_trend = [v.clone() for v in trend]
            changed_trend[-1] += 1
            self.assertFalse(torch.allclose(mixed_trend[0], block.mix_trends(changed_trend)[0]))

    def test_timemixer_width_projections_fmm_and_connected_gradients(self):
        model = TimeMixerNAS(NASConfig((64, 16, 32), (0, 1, 1)), moving_average=25)
        self.assertEqual([block.width for block in model.blocks], [64, 16, 32])
        self.assertEqual([layer.in_features for layer in model.fmm], [7, 3, 1])
        self.assertEqual(model.readout.in_features, 5*32+10)
        x = torch.randn(2, 7, 5, requires_grad=True)
        y = model(x)
        self.assertTrue(bool((y>0).all()))
        y.sum().backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(bool(torch.isfinite(param.grad).all()), name)

    def test_declared_hpo_domains_and_invalid_inputs(self):
        config = NASConfig((16, 16, 16), (0, 0, 0))
        for keyword, value in [("top_k", 1), ("top_k", True), ("num_kernels", 3)]:
            with self.assertRaises(ValueError):
                TimesNetNAS(config, **{keyword: value})
        with self.assertRaises(ValueError):
            TimeMixerNAS(config, moving_average=5)
        for model in (TimesNetNAS(config, num_kernels=2), TimeMixerNAS(config)):
            for x in (torch.zeros(1, 8, 5), torch.full((1, 7, 5), float("nan"))):
                with self.assertRaises(ValueError):
                    model(x)

    def test_meta_extreme_and_asymmetric_width_geometry_and_autograd(self):
        for widths in [(256, 256, 256), (16, 256, 32), (256, 32, 16)]:
            for cls in (TimesNetNAS, TimeMixerNAS):
                with self.subTest(widths=widths, family=cls.__name__), torch.device("meta"):
                    model = cls(NASConfig(widths, (1, 0, 1)))
                    x = torch.empty(1, 7, 5, requires_grad=True)
                    y = model(x)
                    self.assertEqual(tuple(y.shape), (1,))
                    y.sum().backward()
                    self.assertTrue(all(p.grad is not None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
