"""ARIMA receipt compatibility on synthetic data; no market fit or campaign writes."""
from dataclasses import FrozenInstanceError, dataclass, fields
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import e2_temporal_controls as controls
import e2_reduced_final as final
import e2_arima_prediction_compat as compatibility
import e2_arima_compat_worker as worker


class WalkModel:
    def __init__(self, output):
        self.origin = controls.day('2024-01-01')
        self.current = self.origin - controls.DAY
        self.output = Path(output)
        self.forecasts = []
        self.observations = []

    def forecast(self, target):
        raw = -0.002 if not self.forecasts else 0.004
        value = controls.output(target, raw)
        self.forecasts.append(value)
        return value

    def observe(self, target, value):
        path = self.output / (target.strftime('%Y-%m-%d') + '.json')
        if not path.is_file():
            raise AssertionError('A same-day observation was revealed before its forecast receipt')
        self.observations.append((target, value, json.loads(path.read_text())))
        self.current = target


class PredictionCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.prediction = controls.Prediction
        self.previous = self.prediction.__dict__.get('raw_value')
        if 'raw_value' in self.prediction.__dict__:
            delattr(self.prediction, 'raw_value')

    def tearDown(self):
        if 'raw_value' in self.prediction.__dict__:
            delattr(self.prediction, 'raw_value')
        if self.previous is not None:
            setattr(self.prediction, 'raw_value', self.previous)

    def test_actual_frozen_walk_reproduces_original_failure_before_receipt(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as td:
            model = WalkModel(td)
            history = pd.Series([0.01, 0.02], index=pd.date_range('2024-01-01', periods=2))
            with self.assertRaisesRegex(AttributeError, 'raw_value'):
                final._arima_walk(model, history, '2024-01-02', td)
            self.assertEqual(len(model.forecasts), 1)
            self.assertEqual(model.observations, [])
            self.assertEqual(list(Path(td).glob('*.json')), [])

    def test_negative_raw_and_clipped_value_remain_distinct_and_alias_readonly(self):
        receipt = compatibility.install_prediction_raw_alias(ROOT)
        prediction = controls.output('2024-01-01', -0.002)
        self.assertEqual(prediction.raw_value, -0.002)
        self.assertEqual(prediction.raw, -0.002)
        self.assertEqual(prediction.value, 0.)
        self.assertTrue(prediction.truncated)
        self.assertIsNone(type(prediction).__dict__['raw_value'].fset)
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            prediction.raw_value = 0.01
        self.assertFalse(receipt['numeric_predictions_changed'])
        self.assertEqual([f.name for f in fields(type(prediction))], ['date', 'raw', 'value', 'truncated'])
        self.assertEqual(compatibility.install_prediction_raw_alias(ROOT), receipt)

    def test_actual_frozen_walk_persists_unclipped_receipt_before_every_observation(self):
        import pandas as pd
        compatibility.install_prediction_raw_alias(ROOT)
        with tempfile.TemporaryDirectory() as td:
            model = WalkModel(td)
            history = pd.Series([0.01, 0.02], index=pd.date_range('2024-01-01', periods=2))
            values = final._arima_walk(model, history, '2024-01-02', td)
            self.assertEqual(values, {'2024-01-01': 0., '2024-01-02': 0.004})
            self.assertEqual(len(model.observations), 2)
            first = model.observations[0][2]
            self.assertEqual(first['prediction'], 0.)
            self.assertEqual(first['raw_prediction'], -0.002)
            self.assertFalse(first['coefficients_refitted'])

    def test_changed_source_wrong_class_and_existing_foreign_alias_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for relative in compatibility.ENGINE_FILES:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, target)
            (root / 'scripts/e2_temporal_controls.py').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'source differs'):
                compatibility.install_prediction_raw_alias(root)
        @dataclass(frozen=True)
        class Other:
            date: str
            raw: float
            value: float
            truncated: bool
        with patch.object(controls, 'Prediction', Other):
            with self.assertRaisesRegex(ValueError, 'dataclass contract'):
                compatibility.install_prediction_raw_alias(ROOT)
        setattr(self.prediction, 'raw_value', property(lambda value: 0.))
        with self.assertRaisesRegex(ValueError, 'already active'):
            compatibility.install_prediction_raw_alias(ROOT)

    def test_neural_selection_and_non_arima_classical_cannot_enter_shim(self):
        for mode, spec in [('selection', {'kind': 'classical', 'model': 'ARIMA'}),
                ('final', {'kind': 'neural-final', 'model': 'GRU32'}),
                ('final', {'kind': 'classical', 'model': 'Persistence'}),
                ('final', {'kind': 'unknown'})]:
            with self.assertRaisesRegex(ValueError, 'cannot execute'):
                worker.validate_opportunity(spec, mode)
        worker.validate_opportunity({'kind': 'classical', 'model': 'ARIMA'}, 'final')
        worker.validate_opportunity({'kind': 'final-score', 'model': 'all16'}, 'final')

    def test_admission_is_separate_immutable_and_preserves_specification(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            spec = root / 'worker_spec.json'
            spec.write_text('{"old":"unchanged"}\n')
            before = spec.read_bytes()
            worker.write_admission(root, {'repair': 'synthetic'})
            worker.write_admission(root, {'repair': 'synthetic'})
            with self.assertRaisesRegex(ValueError, 'immutable'):
                worker.write_admission(root, {'repair': 'different'})
            self.assertEqual(spec.read_bytes(), before)

    def test_actual_git_registration_binds_runtime_bytes_and_original_authorities(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); repo = root / 'git'; runtime = root / 'runtime'
            repo.mkdir(); runtime.mkdir()
            for relative in worker.FILES:
                for base in (repo, runtime):
                    target = base / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / relative, target)
            record = {'protocol_id': 'e2-reduced-delivery-v1', 'engine_commit': worker.ENGINE_COMMIT,
                'resource_commit': worker.RESOURCE_COMMIT, 'files_sha256': {
                    relative: hashlib.sha256((repo / relative).read_bytes()).hexdigest() for relative in worker.FILES}}
            raw = worker.canonical(record) + b'\n'
            (runtime / 'registration.json').write_bytes(raw)
            target = repo / worker.REGISTRATION_PATH
            target.parent.mkdir(parents=True); target.write_bytes(raw)
            subprocess.run(['git', 'init', '-q', str(repo)], check=True)
            subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
            subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Fixture', '-c',
                'user.email=fixture@example.invalid', 'commit', '-qm', 'Synthetic source proof'], check=True)
            commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
            loader = importlib.util.spec_from_file_location('synthetic_arima_worker', runtime / worker.FILES[1])
            module = importlib.util.module_from_spec(loader); loader.loader.exec_module(module)
            self.assertEqual(module.verify_registration(runtime, commit, repo), record)
            (runtime / worker.FILES[0]).write_text('changed')
            with self.assertRaisesRegex(ValueError, 'source differs'):
                module.verify_registration(runtime, commit, repo)


if __name__ == '__main__':
    unittest.main()
