"""Recheck the immutable A100 preflight archive without running models."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import urllib.request
import zipfile
from datetime import datetime, timezone

RESULTS = '47d2f316a50d9268cb6484a547e8a2f0cc0413eb'
SOURCE = '1ff136f71c0ffcd1dbd918efbc078a8d9653ed21'
RUN = '20261118T172624131741Z'
PINS = {
    'receipts.zip': (43860, '44b7fb93a3f9f667bbe3cd7b6a8d4b1a190c0742ac62e78c0dbd127dfa62fd31'),
    'storage_manifest.json': (None, '61dc57e55c3385c398bf663754320d8dc53f46d15345c17be9762f172a2203c7'),
}
BASE = f'https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/{RESULTS}/runs/colab_preflight/{RUN}/'

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def gitsha(raw):
    return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()

def verify(inputs, output, download=False):
    inputs.mkdir(parents=True, exist_ok=True)
    raw_files = {}
    for name, (size, digest) in PINS.items():
        path = inputs / name
        if not path.exists():
            if not download:
                raise FileNotFoundError(f'{path}: use --download or provide the pinned files')
            with urllib.request.urlopen(BASE + name, timeout=40) as response:
                raw = response.read(1024 * 1024 + 1)
            assert sha(raw) == digest and (size is None or len(raw) == size), name
            path.write_bytes(raw)
        raw = path.read_bytes()
        assert sha(raw) == digest and (size is None or len(raw) == size), name
        raw_files[name] = raw
    manifest = json.loads(raw_files['storage_manifest.json'])
    assert manifest['source_commit'] == SOURCE and manifest['run_id'] == RUN
    assert manifest['preflight_status'] == 'passed_synthetic_only'
    assert len(manifest['files']) == 41
    zip_record = manifest['files']['receipts.zip']
    assert zip_record == {'bytes': 43860, 'git_blob_sha1': gitsha(raw_files['receipts.zip']), 'sha256': PINS['receipts.zip'][1]}
    selected = {p.removeprefix('files/'): row for p, row in manifest['files'].items() if p.startswith('files/')}
    members = {}
    with zipfile.ZipFile(io.BytesIO(raw_files['receipts.zip'])) as package:
        assert len(package.infolist()) == len(selected) == 40
        assert len(set(package.namelist())) == 40 and set(package.namelist()) == set(selected)
        assert not package.comment and sum(e.file_size for e in package.infolist()) < 20 * 1024 * 1024
        for entry in package.infolist():
            p = PurePosixPath(entry.filename)
            assert not p.is_absolute() and '..' not in p.parts and 'source_package' not in p.parts
            assert not entry.is_dir() and not stat.S_ISLNK(entry.external_attr >> 16)
            assert p.suffix in {'.json', '.log'}
            raw = package.read(entry)
            row = selected[entry.filename]
            assert len(raw) == row['bytes'] and sha(raw) == row['sha256'] and gitsha(raw) == row['git_blob_sha1']
            members[entry.filename] = raw
    receipt = json.loads(members['preflight_receipt.json'])
    assert receipt['repository_HEAD'] == SOURCE and receipt['status'] == 'passed_synthetic_only'
    assert receipt['GPU_execution_verified'] is True and receipt['R1_R3_verified'] is False
    assert all(type(receipt[k]) is int and receipt[k] == 0 for k in ['new_real_data_fits', 'new_market_predictions', 'new_hypothesis_tests'])
    assert receipt['campaign_guard']['allowed'] is False
    assert receipt['source_gate']['passed'] is True
    assert len(receipt['source_gate']['files']) == 7 and all(r['verified'] for r in receipt['source_gate']['files'])
    env = receipt['environment']
    assert env['approved'] is True and env['blockers'] == []
    assert env['GPU']['name'] == 'NVIDIA A100-SXM4-40GB' and env['cuda_available'] is True
    assert (env['python'], env['torch'], env['cuda_runtime']) == ('3.12.12', '2.8.0+cu126', '12.6')
    models = receipt['models']
    assert len(models) == len({r['model'] for r in models}) == 17
    neural = []
    rows = []
    for index, row in enumerate(models, 1):
        single = json.loads(members[f'model_{index:02}/receipt.json'])
        assert {k: v for k, v in row.items() if k != 'worker_wall_seconds'} == single
        assert row['status'] == 'passed' and row['synthetic_only'] is True
        assert row['real_data_model_fits'] == row['optimizer_steps'] == 0
        assert row['finite_nonnegative'] is True and row['checkpoint_reload_identical'] is True
        if row['device'] == 'cuda:0':
            neural.append(row)
            assert row['fixture_shape'] == [4, 7, 5] and row['output_shape'] == [4]
            assert row['synthetic_backward_executed'] is True and row['finite_gradient_tensors'] > 0
            assert row['GPU_peak_memory_bytes'] > 0
        else:
            assert row['device'] == 'cpu_classical' and row['model'] in {'Persistence', 'SeasonalNaive', 'ARIMA'}
        rows.append({'model': row['model'], 'status': row['status'], 'device': row['device'],
                     'parameters': row.get('parameters'), 'reported_finite_gradient_tensors': row.get('finite_gradient_tensors', 0),
                     'GPU_peak_memory_bytes': row.get('GPU_peak_memory_bytes'),
                     'reported_checkpoint_reload_identical': row['checkpoint_reload_identical'],
                     'real_data_model_fits': row['real_data_model_fits'], 'synthetic_ARIMA_fits': row.get('synthetic_ARIMA_fits', 0)})
    assert len(neural) == receipt['synthetic_neural_backward_count'] == 14
    gradient_count = sum(r['finite_gradient_tensors'] for r in neural)
    assert gradient_count == 456 and receipt['synthetic_ARIMA_fit_count'] == 1
    original = next(r for r in models if r['model'] == 'CNN-LSTM-atencion-aditiva E3')
    assert original['parameters'] == 1210 and original['configuration']['variant'] == 'additive_positive_head'
    arima = next(r for r in models if r['model'] == 'ARIMA')
    assert arima['order'] == [2, 0, 1] and arima['synthetic_ARIMA_fits'] == 1
    assert arima['forecast_before_observe_guard_passed'] is True and arima['fit_receipt']['converged'] is True
    assert all(r['causal_exact_lag_verified'] for r in models if r['model'] in {'Persistence', 'SeasonalNaive'})
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'per_model.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {'status': 'passed', 'audited_at_utc': datetime.now(timezone.utc).isoformat(),
              'results_commit': RESULTS, 'executed_source_commit': SOURCE, 'run_id': RUN,
              'ZIP_members_verified': 40, 'manifest_artifacts_verified': 41, 'persisted_files_including_manifest': 42,
              'zip_bytes': 43860, 'zip_sha256': PINS['receipts.zip'][1], 'manifest_sha256': PINS['storage_manifest.json'][1],
              'models_passed': 17, 'neural_CUDA_receipts': 14, 'classical_CPU_receipts': 3,
              'reported_finite_gradient_tensors': gradient_count, 'individual_aggregate_discrepancies': 0,
              'reported_synthetic_ARIMA_fits': 1, 'reported_real_data_fits': 0, 'reported_market_predictions': 0,
              'reported_new_hypothesis_tests': 0, 'real_campaign_allowed': False,
              'checkpoint_or_gradient_tensor_bytes_recomputed': False, 'new_GPU_executions_by_auditor': 0,
              'independent_human_reproduction_accredited': False, 'R1_R3_accredited': False,
              'scope': 'Archive integrity and receipt consistency; model operations are reported by the hashed runner, not reexecuted from weights.'}
    (output / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in {'scope', 'audited_at_utc'}}))
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    verify(args.inputs, args.output, args.download)
