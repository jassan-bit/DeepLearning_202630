"""Score-only source transport continuation through the verified ARIMA/P child."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ENGINE_COMMIT = '1e8745497b484518f766b6da5c19be8362d9add7'
RESOURCE_COMMIT = 'b352ce7f27cd7b55e4141fbfb1548f506ed80987'
PRIOR_ARIMA_COMMIT = 'f12cc0904efefde6854342ef18a6c447d5572b3e'
PRIOR_WORKER_PATH = 'scripts/e2_arima_compat_worker.py'
PRIOR_WORKER_SHA256 = '7227b30778b9db974b1438071a8b7ec864196cca881ec46a40a0f7ef7e44475a'
FINAL_DATA_PATH = 'scripts/e2_reduced_final_data.py'
FINAL_DATA_SHA256 = '693842a92386cd3c49ed3e10a9296428e900010df7ae7db373a80d625e851867'
AGGREGATOR_PATH = 'scripts/load_filipe_reserved_test.py'
AGGREGATOR_SHA256 = '6f253b063cbb8482e325d7fb846bd58b5480c30bc2e6d4aa2ce4d9c4b8d0b908'
REGISTRATION_PATH = 'experiments/e2_score_source_recovery_20261113/registration.json'
FILES = ('scripts/e2_score_source_worker.py', 'scripts/e2_score_archive_transport.py')
OUTPUT_NAME = 'score-worker-source-recovery-v1'
RAW_NAME = 'private_raw_E1'
SENTINEL_NAME = 'private-known-E1-source-absent/daily_market_test.csv'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()


def _argument(argv, option):
    try:
        return argv[argv.index(option) + 1]
    except (ValueError, IndexError):
        raise ValueError('Missing preserved score child argument: ' + option) from None


def _verified_path(root, relative, checksum):
    root = Path(root).absolute()
    path = root / relative
    if (not path.resolve().is_relative_to(root.resolve())
            or any(p.is_symlink() for p in (path, *path.parents))
            or hashlib.sha256(path.read_bytes()).hexdigest() != checksum):
        raise ValueError('A registered score source differs: ' + relative)
    return path


def verify_registration(root, commit, engine):
    if not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('An immutable score source recovery commit is required')
    root, engine = Path(root).absolute(), Path(engine).absolute()
    raw = (root / 'registration.json').read_bytes()
    published = subprocess.check_output(['git', '-C', str(engine), 'show', commit + ':' + REGISTRATION_PATH])
    if raw != published:
        raise ValueError('The score source registration differs from its published commit')
    record = json.loads(raw)
    expected = {'protocol_id': 'e2-reduced-delivery-v1', 'engine_commit': ENGINE_COMMIT,
        'resource_commit': RESOURCE_COMMIT, 'prior_arima_commit': PRIOR_ARIMA_COMMIT}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError('Keep the original E/P/ARIMA authorities for final scoring')
    for relative in FILES:
        checksum = record.get('files_sha256', {}).get(relative)
        _verified_path(root, relative, checksum)
    if Path(__file__).resolve() != (root / FILES[0]).resolve():
        raise ValueError('Execute the registered score source worker itself')
    return record


def validate_opportunity(spec, mode, spec_path):
    """Accept only the separate registered score attempt; never a training job."""
    if (mode != 'final' or spec.get('kind') != 'final-score'
            or spec.get('model') != 'all16' or spec.get('allow_download') is not True):
        raise ValueError('This source continuation admits only the registered all16 scoring worker')
    unsigned = dict(spec)
    expected = unsigned.pop('worker_spec_sha256', None)
    if hashlib.sha256(canonical(unsigned)).hexdigest() != expected:
        raise ValueError('The score specification checksum differs')
    final = Path(spec.get('final_output', '')).absolute()
    output = Path(spec.get('output', '')).absolute()
    daily = Path(spec.get('daily_csv', '')).absolute()
    raw = Path(spec.get('raw_root', '')).absolute()
    paths = (final, output, daily, raw, Path(spec_path).absolute())
    if any(p.is_symlink() for path in paths for p in (path, *path.parents)):
        raise ValueError('Score source paths must not follow symbolic links')
    if (output != final / OUTPUT_NAME or output.resolve() != Path(spec_path).parent.resolve()
            or raw != final.parent / RAW_NAME or daily != final / SENTINEL_NAME
            or daily.exists() or spec.get('device') != 'cuda:0'
            or spec.get('fit_wall_seconds') != 1800):
        raise ValueError('Keep the registered new score output, raw source and absent daily sentinel')
    return final, raw


def _write_admission(output, value):
    target = Path(output) / 'score_source_admission.json'
    payload = canonical(value) + b'\n'
    if target.exists():
        if target.read_bytes() != payload:
            raise ValueError('The immutable score source admission differs')
    else:
        with target.open('xb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--score-source-root', type=Path, required=True)
    parser.add_argument('--score-source-commit', required=True)
    own, remainder = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    engine = Path(_argument(remainder, '--engine-root')).absolute()
    arima = Path(_argument(remainder, '--arima-root')).absolute()
    if (_argument(remainder, '--arima-commit') != PRIOR_ARIMA_COMMIT
            or _argument(remainder, '--extension-commit') != RESOURCE_COMMIT):
        raise ValueError('Keep the verified original ARIMA and P child authorities')
    record = verify_registration(own.score_source_root, own.score_source_commit, engine)
    prior_path = _verified_path(arima, PRIOR_WORKER_PATH, PRIOR_WORKER_SHA256)
    data_path = _verified_path(engine, FINAL_DATA_PATH, FINAL_DATA_SHA256)
    _verified_path(engine, AGGREGATOR_PATH, AGGREGATOR_SHA256)
    spec_path = Path(_argument(remainder, '--worker-spec')).absolute()
    raw_spec = spec_path.read_bytes()
    spec = json.loads(raw_spec)
    final, raw_root = validate_opportunity(spec, _argument(remainder, '--mode'), spec_path)
    sys.path.insert(0, str(own.score_source_root / 'scripts'))
    sys.path.insert(0, str(engine / 'scripts'))
    data = importlib.import_module('e2_reduced_final_data')
    if Path(data.__file__).resolve() != data_path.resolve():
        raise ValueError('The loaded final decoder is not the verified frozen E module')
    transport_path = Path(own.score_source_root) / FILES[1]
    loader = importlib.util.spec_from_file_location('e2_registered_score_archive_transport', transport_path)
    transport = importlib.util.module_from_spec(loader); loader.loader.exec_module(transport)
    original_download = data._download_archive
    wrapped = transport.bounded_archive_transport(original_download, engine, raw_root)
    admission = {'version': 'e2-score-source-worker-v1',
        'score_source_registration_commit': own.score_source_commit,
        'prior_arima_registration_commit': PRIOR_ARIMA_COMMIT,
        'prior_arima_worker_sha256': PRIOR_WORKER_SHA256,
        'engine_commit': ENGINE_COMMIT, 'resource_commit': RESOURCE_COMMIT,
        'source_files_sha256': {relative: record['files_sha256'][relative] for relative in FILES},
        'original_decoder_sha256': FINAL_DATA_SHA256,
        'original_aggregator_sha256': AGGREGATOR_SHA256,
        'archive_reservation_sha256': transport.RESERVE_SHA256,
        'worker_spec_sha256': spec['worker_spec_sha256'],
        'worker_spec_file_sha256': hashlib.sha256(raw_spec).hexdigest(),
        'maximum_archive_transport_attempts': transport.MAX_ATTEMPTS,
        'market_payloads_opened_before_admission': 0,
        'model_refits_requested': False, 'scientific_retry_requested': False,
        'raw_archive_branch_of_original_decoder': True,
        'frozen_prediction_math_changed': False, 'budget_authorization_changed': False}
    admission['sha256'] = hashlib.sha256(canonical(admission)).hexdigest()
    _write_admission(spec_path.parent, admission)
    loader = importlib.util.spec_from_file_location('e2_preserved_ARIMA_score_worker', prior_path)
    prior = importlib.util.module_from_spec(loader); loader.loader.exec_module(prior)
    data._download_archive = wrapped
    try:
        print(json.dumps({'event': 'registered_final_raw_archive_source_admitted',
            'archive_count': 20, 'scoring_only': True,
            'scientific_retry': False, 'score_source_registration_commit': own.score_source_commit}), flush=True)
        return prior.main(remainder)
    finally:
        data._download_archive = original_download


if __name__ == '__main__':
    raise SystemExit(main())
