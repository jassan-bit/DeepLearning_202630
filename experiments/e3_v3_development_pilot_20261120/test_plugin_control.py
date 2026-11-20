"""Synthetic software checks only; no market reads, model fitting or scoring."""
from __future__ import annotations

import copy
import io
import math
import os
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
_source_env = os.environ.get("E3_PILOT_SOURCE_ROOT") or os.environ.get("E3_SCIENTIFIC_SOURCE")
_source_candidates = [Path(_source_env)] if _source_env else list(Path(__file__).resolve().parents)
SOURCE = next((candidate for candidate in _source_candidates
               if (candidate / "src" / "e3_overlap_model_v3.py").is_file()), None)
if SOURCE is None:
    raise RuntimeError("Set E3_PILOT_SOURCE_ROOT to the frozen source checkout when testing staged files")
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(HERE))

import torch
from src.e3_overlap_model_v3 import C, OverlapDistributionModel, loss_components
from plugin_control import PluginSecondMomentModel, second_moment_prediction, load_control_checkpoint


class PluginControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def law(self):
        return (torch.tensor([0., .01, .04], dtype=torch.float64),
                torch.tensor([[.1, .6, 1.5], [.2, .7, 1.2], [.3, 1., 2.]], dtype=torch.float64),
                torch.tensor([[.2, .5, .3], [.4, .1, .5], [.3, .4, .3]], dtype=torch.float64))

    def inputs(self):
        generator = torch.Generator().manual_seed(91)
        return torch.randn(5, 7, 5, generator=generator, dtype=torch.float32), torch.linspace(.001, .01, 5, dtype=torch.float64)

    def model(self, **kwargs):
        return PluginSecondMomentModel(u_scale=.03, y_scale=.02, seed=37, **kwargs)

    def test_jensen_order_and_expected_square_identity(self):
        A, atoms, p = self.law()
        exact = (p * torch.sqrt(A[:, None] + C * atoms.square())).sum(-1)
        result = second_moment_prediction(A, atoms, p)
        self.assertTrue(bool((result > exact).all()))
        torch.testing.assert_close(result.square(), A + C * (p * atoms.square()).sum(-1), rtol=2e-15, atol=2e-15)

    def test_single_atom_and_zero_mass_degeneracy(self):
        A = torch.tensor([0., .04], dtype=torch.float64)
        atoms = torch.tensor([[.3, .8, 1.7], [.5, 1., 2.]], dtype=torch.float64)
        p = torch.tensor([[0., 1., 0.], [1., 0., 0.]], dtype=torch.float64)
        actual = second_moment_prediction(A, atoms, p)
        expected = torch.sqrt(A + C * torch.tensor([.8 ** 2, .5 ** 2], dtype=torch.float64))
        torch.testing.assert_close(actual, expected, rtol=1e-15, atol=1e-15)

    def test_constant_atoms_equal_exact_mean(self):
        A, atoms, p = self.law()
        atoms = torch.ones_like(atoms) * .7
        result = second_moment_prediction(A, atoms, p)
        torch.testing.assert_close(result, torch.sqrt(A + C * .7 ** 2), rtol=1e-15, atol=1e-15)

    def test_large_representable_atoms_avoid_square_overflow(self):
        atoms = torch.tensor([[1e200, 2e200, 3e200]], dtype=torch.float64)
        p = torch.tensor([[.2, .5, .3]], dtype=torch.float64)
        result = second_moment_prediction(torch.tensor([1e300], dtype=torch.float64), atoms, p)
        self.assertTrue(bool(torch.isfinite(result).all()))
        expected = 1e200 * math.sqrt(C * (.2 + .5 * 4 + .3 * 9))
        self.assertAlmostEqual(float(result) / expected, 1., places=14)

    def test_zero_norm_is_finite_and_has_zero_atom_gradient(self):
        atoms = torch.zeros((1, 3), dtype=torch.float64, requires_grad=True)
        p = torch.tensor([[.2, .5, .3]], dtype=torch.float64, requires_grad=True)
        result = second_moment_prediction(torch.zeros(1, dtype=torch.float64), atoms, p)
        self.assertEqual(float(result.detach()), 0.)
        result.sum().backward()
        self.assertTrue(bool(torch.isfinite(atoms.grad).all()))
        self.assertTrue(bool(torch.isfinite(p.grad).all()))

    def test_detected_dynamic_range_underflow_is_not_a_zero_forecast(self):
        atoms = torch.tensor([[1e-200, 1., 1e200]], dtype=torch.float64)
        p = torch.tensor([[1., 0., 0.]], dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, 'underflow'):
            second_moment_prediction(torch.zeros(1, dtype=torch.float64), atoms, p)

    def test_zero_mass_has_finite_probability_gradient(self):
        atoms = torch.tensor([[.1, .7, 1.]], dtype=torch.float64, requires_grad=True)
        p = torch.tensor([[0., .4, .6]], dtype=torch.float64, requires_grad=True)
        result = second_moment_prediction(torch.tensor([.03], dtype=torch.float64), atoms, p)
        result.sum().backward()
        self.assertTrue(bool(torch.isfinite(p.grad).all()))
        self.assertTrue(bool(torch.isfinite(atoms.grad).all()))

    def test_float32_law_returns_float64_above_floor(self):
        A = torch.tensor([0., .04], dtype=torch.float32)
        atoms = torch.tensor([[.1, .7, 1.], [.3, .5, .9]], dtype=torch.float32)
        p = torch.tensor([[.25, .5, .25], [.5, .25, .25]], dtype=torch.float32)
        result = second_moment_prediction(A, atoms, p)
        self.assertEqual(result.dtype, torch.float64)
        self.assertEqual(result.device, atoms.device)
        self.assertTrue(bool((result >= A.to(torch.float64).sqrt()).all()))

    def test_bad_laws_and_shapes_are_rejected(self):
        A, atoms, p = self.law()
        cases = [(A[:1], atoms, p), (A, atoms[:, :2], p), (-A - 1, atoms, p),
                 (A, -atoms, p), (A, atoms, p * 2), (A, atoms, p - 1),
                 (A, atoms * float('nan'), p)]
        for args in cases:
            with self.subTest(args_shapes=[tuple(x.shape) for x in args]):
                with self.assertRaises(ValueError): second_moment_prediction(*args)

    def test_same_seed_same_law_initialization_and_parameter_count(self):
        control = self.model()
        original = OverlapDistributionModel('atom_crps', u_scale=.03, y_scale=.02, seed=37)
        self.assertEqual(sum(p.numel() for p in control.parameters()), 1262)
        self.assertEqual(set(control.state_dict()), set(original.state_dict()))
        for name, value in control.state_dict().items():
            self.assertTrue(torch.equal(value, original.state_dict()[name]), name)
        x, A = self.inputs()
        first, second = control.forward_components(x, A), original.forward_components(x, A)
        for name in ('atoms', 'probabilities', 'sigma_atoms', 'raw_head'):
            self.assertTrue(torch.equal(first[name], second[name]), name)
        self.assertTrue(torch.equal(first['finite_law_mean_prediction'], second['prediction']))
        self.assertTrue(bool((first['prediction'] >= second['prediction'] - 1e-15).all()))

    def test_inherited_forward_and_loss_use_plugin_not_exact_readout(self):
        model = self.model()
        x, A = self.inputs()
        outputs = model.forward_components(x, A)
        torch.testing.assert_close(model(x, A), outputs['prediction'], rtol=0, atol=0)
        Y = torch.tensor([.08, .07, .06, .05, .04], dtype=torch.float64)
        U = torch.tensor([.025, .03, .04, .045, .06], dtype=torch.float64)
        loss = model.loss_components(outputs, Y, U)
        expected = ((outputs['prediction'] - Y) / .02).square().mean()
        torch.testing.assert_close(loss['mse_scaled'], expected, rtol=0, atol=0)
        original = dict(outputs, prediction=outputs['finite_law_mean_prediction'])
        exact = loss_components(original, Y, U, y_scale=.02, u_scale=.03, lambda_crps=.1)
        self.assertNotEqual(float(loss['mse_scaled'].detach()), float(exact['mse_scaled'].detach()))
        torch.testing.assert_close(loss['crps_u'], exact['crps_u'], rtol=0, atol=0)
        loss['loss'].backward()
        for label, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, label)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), label)
        self.assertTrue(bool((model.head.bias.grad != 0).all()))
        self.assertTrue(bool((model.probability_head.bias.grad != 0).all()))

    def test_configuration_is_a_distinct_checkpoint_identity(self):
        model = self.model()
        configuration = model.configuration()
        original = OverlapDistributionModel('atom_crps', u_scale=.03, y_scale=.02, seed=37)
        self.assertEqual(configuration['variant'], 'atom_plugin')
        self.assertEqual(configuration['lambda_crps'], .1)
        self.assertFalse(configuration['exact_mean_under_finite_atomic_law'])
        self.assertNotEqual(configuration, original.configuration())
        before = copy.deepcopy(model.state_dict())
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            load_control_checkpoint(model, {'configuration': original.configuration(), 'state_dict': original.state_dict()})
        for name, value in model.state_dict().items(): self.assertTrue(torch.equal(value, before[name]))

    def test_checked_checkpoint_roundtrip_and_schema_rejection(self):
        model = self.model()
        x, A = self.inputs()
        checkpoint = {'configuration': model.configuration(), 'state_dict': model.state_dict()}
        stream = io.BytesIO()
        torch.save(checkpoint, stream)
        stream.seek(0)
        restored = self.model()
        saved = torch.load(stream, map_location='cpu', weights_only=True)
        load_control_checkpoint(restored, saved)
        torch.testing.assert_close(model(x, A), restored(x, A), rtol=0, atol=0)
        broken = copy.deepcopy(saved)
        broken['state_dict']['head.bias'] = torch.zeros(5, dtype=torch.float64)
        before = copy.deepcopy(restored.state_dict())
        with self.assertRaisesRegex(ValueError, 'Invalid checkpoint tensor'):
            load_control_checkpoint(restored, broken)
        for name, value in restored.state_dict().items(): self.assertTrue(torch.equal(value, before[name]))

    def test_configuration_scale_or_weight_change_rejects_resume(self):
        saved = self.model()
        for different in (PluginSecondMomentModel(u_scale=.031, y_scale=.02, seed=37),
                          PluginSecondMomentModel(u_scale=.03, y_scale=.021, seed=37),
                          self.model(lambda_crps=.2)):
            with self.assertRaisesRegex(ValueError, 'configuration differs'):
                load_control_checkpoint(different, {'configuration': saved.configuration(), 'state_dict': saved.state_dict()})


if __name__ == '__main__':
    unittest.main(verbosity=2)
