"""Synthetic-only checks: never invokes an experiment, auditor main, or models."""
import ast
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
from scipy import stats

AUDITOR = Path(__file__).resolve().parents[1] / 'scripts/verify_filipe_confirmatory_d7.py'
spec = importlib.util.spec_from_file_location('independent_confirmatory_synthetic_fixture', AUDITOR)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@contextmanager
def fixture_root():
    original = module.ROOT, module.OUT
    with tempfile.TemporaryDirectory(prefix='confirmatory_auditor_fixture_', dir='/tmp') as name:
        module.ROOT = Path(name)
        module.OUT = Path(name)/'results'/module.STUDY
        try:
            yield Path(name)
        finally:
            module.ROOT, module.OUT = original


class SyntheticChecks(unittest.TestCase):
    def test_no_experiment_or_torch_import(self):
        modules = []
        for node in ast.walk(ast.parse(AUDITOR.read_text())):
            if isinstance(node, ast.Import):
                modules.extend(item.name for item in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module)
        self.assertFalse(any(name and name.startswith(('torch', 'run_filipe', 'analyze_filipe')) for name in modules))

    def test_integer_R_minimal_and_uncapped(self):
        power, alpha = module.TTestPower(), .05/3
        n = module.independent_R(power, .0002, .001, alpha, .8, 30, 10000)
        self.assertEqual(n, 265)
        self.assertGreaterEqual(module.verified_power(power, .2, n, alpha), .8)
        self.assertLess(module.verified_power(power, .2, n-1, alpha), .8)

    def test_floor_and_unidentifiable_design(self):
        power = module.TTestPower()
        self.assertEqual(module.independent_R(power, .01, .0001, .05/3, .8, 30, 10000), 30)
        self.assertIsNone(module.independent_R(power, 1e-12, 1., .05/3, .8, 30, 10000))
        self.assertIsNone(module.independent_R(power, .01, 0., .05/3, .8, 30, 10000))

    def test_MDE_inverse(self):
        power = module.TTestPower()
        for n in (20, 30, 265):
            effect = module.independent_mde_effect(power, n, .05/3, .8)
            self.assertAlmostEqual(module.verified_power(power, effect, n, .05/3), .8, places=9)

    def test_nonfinite_power_quadrature(self):
        nonfinite = SimpleNamespace(power=lambda *args, **kwargs: float('nan'))
        for effect, n in ((.2, 30), (.3, 100), (2., 30)):
            own = module.verified_power(nonfinite, effect, n, .05/3)
            critical = stats.t.isf(.05/6, n-1)
            reference = (stats.nct.sf(critical, n-1, effect*np.sqrt(n))
                         + stats.nct.sf(critical, n-1, -effect*np.sqrt(n)))
            self.assertAlmostEqual(own, reference, places=9)

    def test_paired_bootstrap_cancels_shared_seed_noise(self):
        rng = np.random.default_rng(42)
        scores = rng.normal(.005, .0002, size=(30, 3))
        shared = rng.normal(size=30)
        before = module.independent_bootstrap(scores, 4999, 20261105)
        after = module.independent_bootstrap(scores+shared[:, None], 4999, 20261105)
        self.assertEqual(before[0].shape, (3, 4999))
        for left, right in zip(before, after):
            np.testing.assert_allclose(left, right, rtol=1e-10, atol=1e-12)

    def test_metrics_known_answer(self):
        frame = pd.DataFrame({'y_true': [1., 2., 4.], 'y_pred': [1., 3., 2.],
                              'mase_train_scale': [2., 2., 2.]})
        result = module.metric_values(frame)
        self.assertAlmostEqual(result['mse'], 5/3)
        self.assertAlmostEqual(result['mae'], 1.)
        self.assertAlmostEqual(result['mase'], .5)

    def test_manifests_reject_missing_extra_and_tampered_bytes(self):
        with fixture_root() as root:
            directory = root/'phase'
            directory.mkdir()
            path = directory/'model.pt'
            path.write_bytes(b'synthetic-only-no-model')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            audit = module.Audit()
            audit.manifest(directory, {'model.pt': digest}, excluded=set())
            for manifest in ({}, {'../escape': digest}):
                with self.assertRaises(AssertionError):
                    audit.manifest(directory, manifest, excluded=set())
            path.write_bytes(b'changed-synthetic-only-no-model')
            with self.assertRaises(AssertionError):
                audit.manifest(directory, {'model.pt': digest}, excluded=set())

    def test_boundaries_and_duplicate_JSON(self):
        with fixture_root() as root:
            for path in (Path('/etc/passwd'), root/'results/old/test/predictions.csv', root/'market.zip'):
                with self.assertRaises(AssertionError):
                    module.permitted(path)
            path = root/'duplicate.json'
            path.write_text('{"a":1,"a":2}')
            with self.assertRaises(AssertionError):
                module.read(path)

    def test_curve_selection_and_negative_MSE_rejection(self):
        with fixture_root() as root:
            inner = pd.DataFrame({'epoch': [1, 2, 3], 'train_mse_scaled_eval': [4., 3., 2.],
                'validation_mse_scaled_eval': [3., 2., 2.5], 'best_epoch_so_far': [1, 2, 2],
                'learning_rate': [.001]*3})
            refit = pd.DataFrame({'epoch': [1, 2], 'train_mse_scaled_eval': [4., 3.], 'learning_rate': [.001]*2})
            configuration, training = {'best_epoch': 2, 'learning_rate': .001}, {'max_epochs': 3, 'patience': 1, 'min_delta': 0.}
            inner.to_csv(root/'history_inner.csv', index=False)
            refit.to_csv(root/'history_refit.csv', index=False)
            module.curve_checks(module.Audit(), root, configuration, training)
            inner.loc[1, 'best_epoch_so_far'] = 1
            inner.to_csv(root/'history_inner.csv', index=False)
            with self.assertRaises(AssertionError):
                module.curve_checks(module.Audit(), root, configuration, training)
            inner.loc[1, 'best_epoch_so_far'] = 2
            inner.loc[0, 'train_mse_scaled_eval'] = -1.
            inner.to_csv(root/'history_inner.csv', index=False)
            with self.assertRaises(AssertionError):
                module.curve_checks(module.Audit(), root, configuration, training)

    def test_final_rechecks_every_extension_start(self):
        with fixture_root():
            directory = module.OUT/'variance_extension'
            starts = directory/'trajectory_start_receipts'
            starts.mkdir(parents=True)
            base = datetime(2026, 11, 5, tzinfo=timezone.utc)
            timestamp = lambda seconds: (base+timedelta(seconds=seconds)).isoformat()
            context = {'protocol_sha256': 'fixture-protocol', 'code_signature': 'fixture-code', 'registered_at_utc': timestamp(0)}
            plan = {'models': {model: {'width': 16, 'learning_rate': .001} for model in module.MODELS}}
            (directory/'run_started.json').write_text(json.dumps({'protocol_sha256': 'fixture-protocol',
                'phase': 'variance_extension', 'locked_design_sha256': None, 'started_at_utc': timestamp(2)}))
            (directory/'execution_summary.json').write_text(json.dumps({'completed_at_utc': timestamp(120),
                'first_new_trajectory_started_at_utc': timestamp(3)}))
            ordinal = 3
            for model in module.MODELS:
                for seed in module.EXTENSION_SEEDS:
                    (starts/f'D7_{model}_s{seed}.json').write_text(json.dumps({
                        'job': {'model': model, 'seed': seed, 'history_days': 7, 'variant': 'relative_window', **plan['models'][model]},
                        'started_at_utc': timestamp(ordinal), 'phase': 'variance_extension',
                        'protocol_sha256': 'fixture-protocol', 'code_signature': 'fixture-code'}))
                    ordinal += 1
            arguments = (module.Audit(), plan, context, {'published_at_utc': timestamp(1)}, {'locked_at_utc': timestamp(180)})
            module.recheck_extension_starts(*arguments)
            path = starts/'D7_CNN1D_s6339.json'
            changed = json.loads(path.read_text())
            changed['started_at_utc'] = timestamp(-1)
            path.write_text(json.dumps(changed))
            with self.assertRaises(AssertionError):
                module.recheck_extension_starts(*arguments)


if __name__ == '__main__':
    unittest.main(verbosity=2)
