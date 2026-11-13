"""Synthetic checks of frozen inference, perturbations, accounting and resume."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import pickle
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e2_frozen_posthoc_gpu as gpu


class SyntheticFrozenARIMA:
    """A state-only predictor: there is deliberately no fit method."""
    def __init__(self):
        self.current = pd.Timestamp("2024-12-30")
        self.origin = pd.Timestamp("2024-11-08")
        self.receipt = {"parameters": [.5]}
        self.last = .01
        self.pending = None

    def forecast(self, target):
        if target != self.current + pd.Timedelta(days=1) or self.pending is not None:
            raise ValueError("forecast order")
        self.pending = target
        return SimpleNamespace(value=self.last)

    def observe(self, target, value):
        if self.pending != target or target != self.current + pd.Timedelta(days=1):
            raise ValueError("observation before forecast")
        self.current, self.last, self.pending = target, float(value), None


class FrozenDiagnosticsTests(unittest.TestCase):
    def test_projection_uses_absolute_gate_and_rejects_broadcast_nonfinite(self):
        self.assertTrue(gpu.projection_check([.01, .02], [.01, .02])["baseline_projection_passed"])
        for actual, expected in (([1., 2.], [1.]), ([1e5+.01], [1e5]), ([float('nan')], [1.])):
            with self.assertRaises(ValueError):
                gpu.projection_check(actual, expected)

    def test_training_reference_preserves_invariant_coordinates(self):
        training = np.random.default_rng(1).normal(size=(40, 7, 5))
        training[:, -1, 3] = 0
        before = training.copy()
        reference = gpu.fit_reference(training)
        np.testing.assert_array_equal(training, before)
        self.assertEqual(reference["scale"][-1][3], 0)
        for kind, levels in gpu.SEVERITIES.items():
            for level in levels:
                changed = gpu.perturb(training, reference, kind, level, break_index=20)
                np.testing.assert_array_equal(changed[:, -1, 3], 0)

    def test_perturbations_are_deterministic_aligned_and_do_not_mutate_inputs(self):
        x = np.random.default_rng(2).normal(size=(80, 7, 5)).astype(np.float32)
        saved = x.copy()
        reference = gpu.fit_reference(x[:40])
        truth = np.arange(80, dtype=float)
        for kind, levels in gpu.SEVERITIES.items():
            changed = gpu.perturb(x, reference, kind, levels[-1], break_index=40)
            repeated = gpu.perturb(x, reference, kind, levels[-1], break_index=40)
            np.testing.assert_array_equal(changed, repeated)
            self.assertEqual(changed.shape, x.shape)
            self.assertTrue(np.isfinite(changed).all())
        np.testing.assert_array_equal(x, saved)
        np.testing.assert_array_equal(truth, np.arange(80))

    def test_structural_break_is_fixed_at_declared_calendar_index(self):
        x = np.zeros((608, 7, 5), dtype=np.float32)
        ref = {"median": np.zeros((7, 5)), "scale": np.ones((7, 5))}
        changed = gpu.perturb(x, ref, 'structural_break', .5)
        np.testing.assert_array_equal(changed[:365], x[:365])
        np.testing.assert_array_equal(changed[365:], .5)

    def test_missing_values_use_training_reference_and_stay_finite(self):
        x = np.ones((1000, 7, 5), dtype=np.float32)
        ref = {"median": np.full((7, 5), -3), "scale": np.ones((7, 5))}
        altered = gpu.perturb(x, ref, 'missing', .3)
        self.assertEqual(set(np.unique(altered)), {-3, 1})
        with self.assertRaises(ValueError):
            gpu.perturb(x, ref, 'missing', .9)

    def test_bootstrap_keeps_contiguous_circular_blocks(self):
        index = gpu.block_indices(31, block=7)
        self.assertEqual(index.shape, (2000, 31))
        np.testing.assert_array_equal((np.diff(index[:, :7], axis=1) % 31), 1)
        np.testing.assert_array_equal(index, gpu.block_indices(31, block=7))
        with self.assertRaises(ValueError):
            gpu.block_indices(31, draws=1999)

    def test_paired_bootstrap_known_constant_effect_is_exact(self):
        truth = np.zeros(40)
        base = np.array([np.ones(40), np.full(40, 2.)])
        changed = base + 1
        result = gpu.paired_rmse_change(truth, base, changed, gpu.block_indices(40))
        self.assertEqual(result['baseline_mean_seed_RMSE'], 1.5)
        self.assertEqual(result['perturbed_mean_seed_RMSE'], 2.5)
        self.assertEqual(result['paired_date_block_CI95'], [1., 1.])
        self.assertEqual(result['RMSE_degradation'], 1.)

    def test_occlusion_recovers_linear_cell_effects_and_fidelity_endpoints(self):
        x = np.ones((4, 7, 5))
        weights = np.arange(1, 36).reshape(7, 5)
        predict = lambda values: (np.asarray(values)*weights).sum(axis=(1, 2))
        result = gpu.occlusion(predict, x, np.zeros((7, 5)))
        np.testing.assert_allclose(result['signed_effects'], np.tile(weights.flatten(), (4, 1)))
        np.testing.assert_array_equal(result['deletion_predictions'][0], predict(x))
        np.testing.assert_array_equal(result['deletion_predictions'][-1], 0)
        np.testing.assert_array_equal(result['insertion_predictions'][0], 0)
        np.testing.assert_array_equal(result['insertion_predictions'][-1], predict(x))

    def test_stability_ties_and_deterministic_system_are_explicit(self):
        self.assertEqual(gpu.stability([{'mean_absolute_effect': [1, 2, 3]}])['mean_spearman'], None)
        same = gpu.stability([{'mean_absolute_effect': [1, 1, 3]}, {'mean_absolute_effect': [1, 1, 3]}])
        self.assertAlmostEqual(same['mean_spearman'], 1.)
        undefined = gpu.stability([{'mean_absolute_effect': [0, 0]}, {'mean_absolute_effect': [0, 0]}])
        self.assertIsNone(undefined['mean_spearman'])

    def test_causal_arima_local_occlusion_replays_saved_state_without_fitting(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            checkpoint = output / 'final/predictor.pkl'
            checkpoint.parent.mkdir()
            raw = pickle.dumps(SyntheticFrozenARIMA())
            checkpoint.write_bytes(raw)
            history = pd.Series(np.arange(1, 763)/10000,
                                index=pd.date_range('2023-01-01', periods=762))
            dates = pd.date_range('2025-01-01', periods=20, tz='UTC')
            context = {'dates': dates, 'panel': None,
                       'final_module': SimpleNamespace(_history=lambda panel: history)}
            baseline, predict, lags, center = gpu.causal_arima_baseline(
                context, output, {'checkpoint': 'predictor.pkl', 'checkpoint_sha256': gpu.sha(raw)},
                np.array([7, 19]), time.monotonic()+20)
            np.testing.assert_array_equal(predict(lags), baseline[[7, 19]])
            self.assertEqual(center.shape, (7,))
            self.assertEqual(checkpoint.read_bytes(), raw)

    def test_completed_receipt_is_reused_only_if_every_artifact_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gpu.immutable_json(root / 'data.json', {'synthetic': True})
            receipt = {'status': 'completed', 'run_id': 'test',
                       'artifacts_sha256': {'data.json': gpu.sha((root / 'data.json').read_bytes())}}
            gpu.immutable_json(root / 'result.json', receipt)
            self.assertEqual(gpu.completed_record(root, 'test'), receipt)
            (root / 'data.json').write_text('{}')
            with self.assertRaises(ValueError):
                gpu.completed_record(root, 'test')

    def test_accounting_reserves_then_charges_full_invocation_even_on_failure(self):
        budget = Mock()
        budget.persistence = object()
        with patch.object(gpu, '_run_impl', side_effect=ValueError('synthetic stop')):
            with self.assertRaisesRegex(ValueError, 'synthetic stop'):
                gpu.run('source', 'output', 'destination', Mock(), budget=budget, configuration_commit='a'*40)
        budget.reserve.assert_called_once()
        budget.charge.assert_called_once()
        self.assertEqual(budget.reserve.call_args.args[1], 5400)
        self.assertEqual(budget.charge.call_args.kwargs['status'], 'failed')
        self.assertGreaterEqual(budget.charge.call_args.args[1], 0.)

    def test_interruption_retains_accounting_reservation(self):
        budget = Mock()
        budget.persistence = object()
        with patch.object(gpu, '_run_impl', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                gpu.run('source', 'output', 'destination', Mock(), budget=budget, configuration_commit='a'*40)
        self.assertEqual(budget.charge.call_args.kwargs['status'], 'interrupted')

    def test_backup_retries_do_not_execute_compute(self):
        client = Mock()
        client.persist_result.side_effect = [RuntimeError('temporary'), {'status': 'backed_up'}]
        with patch.object(gpu, '_run_impl') as compute:
            result = gpu.backup_result(Path('/synthetic'), {'status': 'completed'}, client)
        self.assertEqual(result['status'], 'backed_up')
        self.assertEqual(client.persist_result.call_count, 2)
        compute.assert_not_called()

    def test_launcher_dry_run_does_not_materialize_or_read_secret(self):
        spec = importlib.util.spec_from_file_location('posthoc_launcher', ROOT/'notebooks/run_e2_frozen_posthoc_colab.py')
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        with patch.object(launcher, 'materialize') as materialize:
            result = launcher.run_e2_frozen_posthoc_colab('a'*40)
        materialize.assert_not_called()
        self.assertEqual(result['status'], 'dry_run')
        self.assertEqual(result['batch_size'], 1)

    def test_synthetic_complete_flow_preserves_projection_batchone_and_backup_resume(self):
        import torch
        import e2_published_pilot_training as inference
        x = np.random.default_rng(8).normal(size=(608, 7, 5)).astype(np.float32)
        original = (torch.as_tensor(x).sum(dim=(1, 2))).numpy()
        scaler = SimpleNamespace(transform=lambda raw: raw, target_scale=1.)
        context = {'training': SimpleNamespace(scaler=scaler), 'reference': gpu.fit_reference(x[:100]),
                   'raw_windows': x, 'y': original+.1,
                   'bundle': {'provenance': {'pinned_source_sha256': {'freeze': 'f'*64}},
                              'predictors': [{'model': 'GRU32', 'seed': 271801,
                                              'prediction_original': original.tolist(), 'calibrator_q': .2}]},
                   'freeze': {'models': [{'model': 'GRU32', 'seed': 271801, 'checkpoint_sha256': 'e'*64}]}}
        calls = []

        def synthetic_predict(model, tensor, *, batch_size, device):
            calls.append(batch_size)
            return tensor.sum(dim=(1, 2))

        client = Mock()
        client.persist_result.return_value = {'status': 'backed_up', 'remote_commit': 'a'*40, 'manifest_path': 'synthetic/manifest.json'}
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(torch, '__version__', '2.8.0+cu126'), \
                patch.object(gpu.sys, 'version_info', (3, 12)), \
                patch.object(torch.cuda, 'is_available', return_value=True), \
                patch.object(torch.cuda, 'get_device_name', return_value='Synthetic A100'), \
                patch.object(gpu, 'original_context', return_value=context), \
                patch.object(gpu, 'load_neural', return_value=(object(), scaler)), \
                patch.object(gpu, 'profile_predictor', return_value={'parameters': 1}), \
                patch.object(inference, 'predict_pilot', side_effect=synthetic_predict), \
                patch.object(gpu, 'heartbeat'):
            root = Path(temporary)
            destination = root / 'posthoc'
            result = gpu._run_impl(root/'source', root/'original', destination, client, configuration_commit='a'*40)
            self.assertEqual(result['status'], 'backed_up')
            summary = json.loads((destination/'summary.json').read_bytes())
            self.assertTrue(summary['all_baseline_projections_passed'])
            self.assertEqual(len(summary['robustness']), 36)
            self.assertEqual(json.loads((destination/'registration.json').read_bytes())['configuration_commit'], 'a'*40)
            self.assertTrue(all(call == 1 for call in calls))
            self.assertTrue(all(row['mean_interval_width90_frozen_q'] == .4 for row in summary['robustness']))
            call_count = len(calls)
            resumed = gpu._run_impl(root/'source', root/'original', destination, client, configuration_commit='a'*40)
            self.assertEqual(resumed['status'], 'backed_up')
            self.assertEqual(len(calls), call_count)
            with self.assertRaisesRegex(ValueError, 'differs'):
                gpu._run_impl(root/'source', root/'original', destination, client, configuration_commit='b'*40)

    def test_production_requires_configuration_commit_and_durable_shared_budget(self):
        for commit in (None, 'main', 'A'*40, 'a'*39):
            with self.assertRaisesRegex(ValueError, '40-hex'):
                gpu.run('source', 'output', 'destination', Mock(), configuration_commit=commit)
        for budget in (None, SimpleNamespace(persistence=None), SimpleNamespace()):
            with self.assertRaisesRegex(ValueError, 'persistencia remota'):
                gpu.run('source', 'output', 'destination', Mock(), budget=budget, configuration_commit='a'*40)

    def test_standalone_execution_is_disabled_before_any_source_or_secret_access(self):
        spec = importlib.util.spec_from_file_location('posthoc_launcher_disabled', ROOT/'notebooks/run_e2_frozen_posthoc_colab.py')
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        with patch.object(launcher, 'materialize') as materialize:
            with self.assertRaisesRegex(ValueError, 'unificado'):
                launcher.run_e2_frozen_posthoc_colab('a'*40, execute=True)
        materialize.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'unificado'):
            gpu.main([])

    def test_registered_launcher_rejects_source_tampering_before_materialization(self):
        spec = importlib.util.spec_from_file_location('posthoc_launcher_integrity', ROOT/'notebooks/run_e2_frozen_posthoc_colab.py')
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        payloads = {name: b'synthetic verified source' for name in launcher.FILES}
        registration = {'engine_commit': gpu.ENGINE_COMMIT, 'protocol_id': 'e2-reduced-delivery-v1',
                        'training_performed': False,
                        'files_sha256': {name: gpu.sha(raw) for name, raw in payloads.items()}}
        registered = gpu.canonical(registration)
        payloads[launcher.FILES[0]] = b'tampered source'
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(launcher, 'WORKSPACE', Path(temporary)), \
                patch.object(launcher, '_download', side_effect=lambda commit, name: registered if name == launcher.REGISTRATION else payloads[name]):
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                launcher.materialize('a'*40)
            self.assertEqual(list(Path(temporary).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
