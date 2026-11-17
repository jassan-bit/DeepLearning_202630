"""Scientific admission/inference checks; all price arrays and weights are synthetic."""
from contextlib import redirect_stdout
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import sys
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('E3_EVAL_SOURCE_ROOT', str(HERE.parents[1]))).resolve()
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('frozen_worker_tested', HERE / 'worker.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
from src.e3_innovation_model import InnovationModel


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(w.canonical(value) + b'\n')


def fixture(directory):
    inputs = directory / 'inputs'
    inputs.mkdir()
    old = ROOT / 'experiments/e3_cierre_v2_20261116/audit/inputs'
    partition = json.loads((ROOT / 'book_e2/_static/e2_eda_current_v1/partition_audit.json').read_text())
    train_dates = partition['final_partition']['training']['training_sample_IDs']
    eval_dates = partition['final_partition']['evaluation']['sample_IDs']
    adm = json.loads((old / 'data_admission.json').read_text())
    scaler = adm['scaler']
    scaler.update(mean=[0.] * 5, std=[1.] * 5, y_scale=.02, target_scale=.02, u_scale=.02)
    rng = np.random.default_rng(9123)
    def raw(n):
        close = np.exp(4 + rng.normal(0, .001, size=(n, 7)))
        return np.stack((close * 1.0001, close * 1.01, close * .99, close,
                         rng.uniform(1, 2, size=(n, 7))), axis=-1)
    train_raw, test_raw = raw(655), raw(608)
    _, train_a = w.transform(train_raw, scaler)
    _, test_a = w.transform(test_raw, scaler)
    train_y, test_y = np.sqrt(train_a) + .002, np.sqrt(test_a) + .002
    # Unused scaled/object members must never be indexed by the worker.
    np.savez(inputs / 'current_cohort_arrays.npz', training_dates=np.array(train_dates),
             training_raw=train_raw, training_y_original=train_y,
             evaluation_dates=np.array(eval_dates), evaluation_raw=test_raw,
             evaluation_y_original=test_y, evaluation_X=np.array([object()]))
    w.NPZ_SHA256 = w.digest(inputs / 'current_cohort_arrays.npz')
    adm['npz_sha256'] = w.NPZ_SHA256
    model = InnovationModel(w.VARIANT, seed=20261116, u_scale=.02, y_scale=.02)
    source_hashes = json.loads((old / 'manifest.json').read_text())['source_hashes']
    manifest = {'run_id': '6cc32b1a91224c4f9144a1c831394746', 'claim_id': 'synthetic-only',
                'source_hashes': source_hashes, 'plan_sha256': 'synthetic-only',
                'npz_sha256': w.NPZ_SHA256,
                'data_fingerprint': hashlib.sha256(w.canonical(adm)).hexdigest()}
    cp = {'version': 'e3-innovation-development-v2', 'model_name': w.VARIANT,
          'epoch': 38, 'best_epoch': 38, 'model_state': model.state_dict(),
          **{k: manifest[k] for k in ('run_id', 'claim_id', 'source_hashes',
                                     'plan_sha256', 'data_fingerprint')}}
    (inputs / 'additive_positive_head').mkdir()
    torch.save(cp, inputs / 'additive_positive_head/best.pt')
    w.BEST_SHA256 = w.digest(inputs / 'additive_positive_head/best.pt')
    positions = {d: i for i, d in enumerate(train_dates)}
    selected = [positions[d] for d in adm['validation_dates']]
    predicted, floor = w.infer(model, train_raw[selected], scaler, 'cpu')
    rows = [{'date': d, 'target_sigma7': float(train_y[i]), 'prediction_sigma7': float(p)}
            for d, i, p in zip(adm['validation_dates'], selected, predicted)]
    (inputs / 'additive_positive_head/validation_predictions.csv').write_bytes(w.csv_bytes(rows))
    save_json(inputs / 'manifest.json', manifest)
    save_json(inputs / 'data_admission.json', adm)
    files = {n: {'sha256': w.digest(inputs / n), 'bytes': (inputs / n).stat().st_size}
             for n in ('manifest.json', 'data_admission.json', 'additive_positive_head/best.pt',
                       'additive_positive_head/validation_predictions.csv')}
    snapshot = {'files': files, 'progress': {'state_status': 'completed'},
                'scientific_identity': manifest}
    save_json(inputs / 'snapshot_manifest.json', snapshot)
    w.SNAPSHOT_SHA256 = w.digest(inputs / 'snapshot_manifest.json')
    save_json(inputs / 'e2_partition.json', partition)
    e2_admission = {'target': 'population sigma7 of daily Close log returns, ddof0, unannualized; h1',
                    'common_eligible_dates': eval_dates}
    save_json(inputs / 'e2_data_admission.json', e2_admission)
    rows = [{'model': name, 'RMSE': .01, 'MAE': .005,
             'R': 1 if name in w.DETERMINISTIC else 6, 'n_common_dates': 608}
            for name in w.TOP10 + w.NEURAL_BASELINES + w.DETERMINISTIC]
    (inputs / 'e2_benchmark_A2.csv').write_bytes(w.csv_bytes(rows))
    w.E2_HASHES = {n: w.digest(inputs / n) for n in w.E2_HASHES}
    return inputs


class Checks(unittest.TestCase):
    def test_synthetic_full_worker_and_immutable_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            inputs = fixture(directory)
            output = directory / 'output'
            with redirect_stdout(io.StringIO()):
                summary = w.evaluate(ROOT, inputs, output, 'cpu')
            self.assertEqual(summary['evaluation_n'], 608)
            self.assertTrue(summary['no_training'])
            self.assertFalse(summary['winner_declared'])
            self.assertEqual(summary['roundtrip_max_absolute_error'], 0)
            self.assertEqual(len(w.read_csv(output / 'predictions.csv')), 608)
            rows = w.read_csv(output / 'comparison.csv')
            self.assertEqual(len(rows), 17)
            self.assertTrue(all(r['training_examples'] == '' for r in rows if r['model'] in w.DETERMINISTIC))
            self.assertEqual(rows[-1]['training_examples'], '532')
            with self.assertRaisesRegex(RuntimeError, 'preserve existing'):
                w.evaluate(ROOT, inputs, output, 'cpu')

    def test_roundtrip_only_never_opens_test_members(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            inputs = fixture(directory)
            original = np.load
            opened = []
            class Guard:
                def __enter__(self):
                    self.archive = original(inputs / 'current_cohort_arrays.npz', allow_pickle=False)
                    return self
                def __exit__(self, *args):
                    self.archive.close()
                def __getitem__(self, key):
                    opened.append(key)
                    if key.startswith('evaluation_'):
                        raise AssertionError('Test member opened during validation-only')
                    return self.archive[key]
            np.load = lambda *args, **kwargs: Guard()
            try:
                with redirect_stdout(io.StringIO()):
                    result = w.evaluate(ROOT, inputs, directory / 'output', 'cpu', roundtrip_only=True)
            finally:
                np.load = original
            self.assertFalse(result['evaluation_members_opened'])
            self.assertEqual(set(opened), {'training_dates', 'training_raw', 'training_y_original'})
            self.assertFalse((directory / 'output').exists())

    def test_bad_ohlc_and_bad_dates_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'OHLCV'):
            w.transform(np.zeros((2, 7, 5)), {})
        with self.assertRaisesRegex(RuntimeError, 'dates/count'):
            w.dates(['2025-01-02', '2025-01-01'], 2, 'fixture')

    def test_nonfinite_target_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Invalid target'):
            w.validate_target([float('nan')], 1)


if __name__ == '__main__':
    unittest.main()
