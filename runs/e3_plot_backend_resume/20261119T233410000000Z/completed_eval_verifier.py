"""Read-only completion gate for this already published E3 campaign.

Only the original, completed evaluation may bypass its worker.  Statistical
analysis remains the frozen analyze.py, whose existing final-receipt reuse path
checks every input and table. This verifier never imports model code, fits,
predicts, writes an artifact, changes a receipt, or accredits R1/R3.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


SOURCE_COMMIT = '5a07d6157a9796ca54416dc5dea1e1527e731597'
PROTOCOL_SHA256 = '5d3e5c0f7440efaa4993e193526d0d53628b7bb053d87a2324975d673379136a'
EVALUATION_CODE_SHA256 = '55dd1cbde85f565b1aca9d74d44b44b27ab84d2c016677d15223cee5a94f49f3'
SCOPE = 'experiments/e3_evaluacion_ampliada_20261119'
# Bytes independently read from the saved final evaluation at d8133ce...,
# part 189. Its successor, part 190, contains the completed statistical analysis.
FINAL_ARTIFACT_SHA256 = {
    'evaluation_receipt.json': '67e27e5324b6feb3cb0a03fcbd05e196449b3100ee85efb68f0f8a085bcce110',
    'run_manifest.json': '288eadf7430e97e1df76eb96efa9ecdf1f697dba834771f5544ee5be5f99acc1',
    'panel_manifest.json': '5f91f7622b5eb160dd5de14cb38edbc872f0fd9963c94afa24450250bec2aeb5',
}
DATASETS = ('D00', 'D01', 'D02', 'D03')
DETERMINISTIC = {'Persistence', 'SeasonalNaive', 'ARIMA', 'floor_only', 'floor_plus_mean_excess'}
EXPECTED_COUNTS = {
    'dataset_count': 4, 'additional_datasets_evaluated': 3,
    'new_real_data_fit_attempts': 780, 'new_real_data_fits': 780,
    'neural_fit_jobs_attempted': 776, 'ARIMA_fit_jobs_attempted': 4,
    'completed_job_results': 796, 'panel_rows': 1874560,
    'panel_files': 76, 'sample_curve_rows': 32544,
}


def _require(value, message):
    if not value:
        raise ValueError('completed_evaluation_verification: ' + message)


def _regular(path):
    path = Path(path).absolute()
    _require(not any(item.is_symlink() for item in (path, *path.parents)),
             'symlink refused: ' + str(path))
    _require(path.is_file(), 'regular file missing: ' + str(path))
    return path


def _read(path):
    return _regular(path).read_bytes()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(path):
    return json.loads(_read(path))


def _file_proof(path):
    path = _regular(path)
    before = path.stat()
    digest = hashlib.sha256()
    size = newlines = 0
    last = b''
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
            size += len(chunk)
            newlines += chunk.count(b'\n')
            last = chunk[-1:]
    after = path.stat()
    _require((before.st_size, before.st_mtime_ns) ==
             (after.st_size, after.st_mtime_ns) and size == before.st_size,
             'file changed while hashing: ' + str(path))
    # The immutable executor CSV writer emits exactly one newline per record,
    # without embedded newlines. Exact published byte hashes bind this format.
    _require(last == b'\n', 'published CSV must end with newline: ' + str(path))
    return {'sha256': digest.hexdigest(), 'bytes': size, 'rows': newlines - 1}


def verify_completed_evaluation(output, protocol_path, source):
    """Return proof, or raise; ``source`` is the frozen source checkout path.

    This campaign-specific gate requires the published final receipt/manifest
    byte identities and validates the restored panel payloads. A missing or
    changed artifact is an error, never a request to rerun the training worker.
    """
    output = Path(output).absolute()
    source = Path(source).absolute()
    protocol_raw = _read(protocol_path)
    protocol = json.loads(protocol_raw)
    _require(_sha(protocol_raw) == PROTOCOL_SHA256, 'protocol differs from frozen campaign')
    _require(protocol.get('source_commit') == SOURCE_COMMIT, 'protocol source differs')
    _require(_sha(_read(source / SCOPE / 'evaluation/run.py')) == EVALUATION_CODE_SHA256,
             'evaluation worker source differs')
    artifacts = {}
    for name, expected in FINAL_ARTIFACT_SHA256.items():
        raw = _read(output / name)
        _require(_sha(raw) == expected, 'published final artifact differs: ' + name)
        artifacts[name] = json.loads(raw)
    receipt = artifacts['evaluation_receipt.json']
    manifest = artifacts['run_manifest.json']
    panel = artifacts['panel_manifest.json']
    _require(receipt.get('kind') == 'e3_extended_evaluation_receipt' and
             receipt.get('status') == 'completed' and receipt.get('failures') == [],
             'evaluation did not complete without failures')
    for name, expected in EXPECTED_COUNTS.items():
        _require(type(receipt.get(name)) is int and receipt[name] == expected,
                 'final evaluation count differs: ' + name)
    for data in (receipt, manifest, panel):
        _require(data.get('source_commit') == SOURCE_COMMIT and
                 data.get('protocol_sha256') == PROTOCOL_SHA256,
                 'final artifact source/protocol binding differs')
    for data in (receipt, manifest):
        _require(data.get('R1_accredited') is False and data.get('R3_accredited') is False,
                 'exploratory accreditation flags changed')
    _require(manifest.get('kind') == 'e3_extended_audit_run_manifest' and
             manifest.get('panel_manifest') == 'panel_manifest.json' and
             manifest.get('panel_sha256') == FINAL_ARTIFACT_SHA256['panel_manifest.json'],
             'run/panel manifest binding differs')
    _require(manifest.get('datasets') == {k: {'raw_sha256': v}
             for k, v in protocol.get('datasets', {}).items()} and
             set(protocol.get('datasets', {})) == set(DATASETS), 'dataset binding differs')
    _require(manifest.get('family') == panel.get('family') == protocol.get('family') and
             len(protocol['family']) == 19 and len(set(protocol['family'])) == 19,
             'complete fixed model family differs')
    _require(manifest.get('seeds') == panel.get('seeds') == protocol.get('seeds') and
             protocol['seeds'] == list(range(20261118, 20261128)), 'final seed roster differs')
    _require(manifest.get('test_targets_used_for_selection') is False and
             manifest.get('additional_datasets_used_for_mechanism_redesign') is False and
             manifest.get('prospective_power_certified') is False and
             manifest.get('evidence_mode') == protocol.get('evidence_mode') ==
             'prespecified_exploratory_audit', 'exploratory evaluation policy differs')
    freeze_raw = _read(output / 'freeze_receipt.json')
    freeze = json.loads(freeze_raw)
    _require(_sha(freeze_raw) == manifest.get('freeze_receipt_sha256') and
             freeze.get('verified') is True and freeze.get('source_commit') == SOURCE_COMMIT and
             freeze.get('protocol_sha256') == PROTOCOL_SHA256 and
             freeze.get('annotated_tag') == manifest.get('protocol_tag'),
             'freeze receipt binding differs')
    _require(panel.get('kind') == 'e3_extended_panel_manifest' and
             type(panel.get('files')) is list and len(panel['files']) == 76 and
             panel.get('total_rows') == receipt['panel_rows'], 'panel completion differs')
    expected_pairs = {(dataset, model) for dataset in DATASETS for model in protocol['family']}
    actual_pairs = set()
    paths = set()
    total_rows = total_bytes = 0
    for entry in panel['files']:
        name = entry.get('path')
        _require(type(name) is str and name, 'missing shard path')
        relative = Path(name)
        _require(not relative.is_absolute() and '..' not in relative.parts and
                 '\\' not in name and relative.as_posix() == name and
                 relative.suffix == '.csv' and name not in paths, 'unsafe/duplicate shard path')
        paths.add(name)
        pair = (entry.get('dataset'), entry.get('model'))
        _require(pair in expected_pairs and pair not in actual_pairs, 'unknown/duplicate dataset-model panel')
        actual_pairs.add(pair)
        expected_seeds = [-1] if pair[1] in DETERMINISTIC else protocol['seeds']
        _require(entry.get('observed_seeds') == expected_seeds, 'panel seed roster differs')
        target = output / relative
        _require(target.resolve().is_relative_to(output.resolve()), 'shard escaped output')
        proof = _file_proof(target)
        _require(all(type(entry.get(field)) is int and entry[field] > 0
                     for field in ('bytes', 'rows')), 'invalid shard sizes')
        _require(all(proof[field] == entry.get(field) for field in ('sha256', 'bytes', 'rows')),
                 'panel payload differs from final manifest: ' + name)
        total_rows += proof['rows']
        total_bytes += proof['bytes']
    _require(actual_pairs == expected_pairs and total_rows == receipt['panel_rows'],
             'incomplete final panel roster/rows')
    _require(_json(output / 'failures.json') == [], 'failure list is nonempty')
    # Only saved forecasts are needed after the worker completes; no checkpoint
    # is deserialized or used for another forward pass by this completion gate.
    return {'status': 'published_completed_evaluation_verified',
            'source_commit': SOURCE_COMMIT, 'protocol_sha256': PROTOCOL_SHA256,
            'evaluation_receipt_sha256': FINAL_ARTIFACT_SHA256['evaluation_receipt.json'],
            'run_manifest_sha256': FINAL_ARTIFACT_SHA256['run_manifest.json'],
            'panel_manifest_sha256': FINAL_ARTIFACT_SHA256['panel_manifest.json'],
            'completed_job_results': receipt['completed_job_results'],
            'verified_panel_files': len(paths), 'verified_panel_rows': total_rows,
            'verified_panel_bytes': total_bytes, 'new_model_fits': 0,
            'new_model_inference': 0, 'R1_accredited': False, 'R3_accredited': False}
