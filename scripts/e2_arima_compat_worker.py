"""Registered child shim for future ARIMA opportunities and final scoring.

Only a name used by E's raw forecast receipt is repaired. The exact P worker
still owns resource acceptance, data admission, fitting and result creation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ENGINE_COMMIT = '1e8745497b484518f766b6da5c19be8362d9add7'
RESOURCE_COMMIT = 'b352ce7f27cd7b55e4141fbfb1548f506ed80987'
RESOURCE_WORKER_SHA256 = '539f043b8a3fb1b9dccbdf9ec73bde63be175baf076fbb6995930cbf70a49a81'
REGISTRATION_PATH = 'experiments/e2_arima_compat_20261113/registration.json'
FILES = ('scripts/e2_arima_prediction_compat.py', 'scripts/e2_arima_compat_worker.py')
SIDECAR = 'arima_fix_admission.json'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()


def _argument(argv, option):
    try:
        return argv[argv.index(option) + 1]
    except (ValueError, IndexError):
        raise ValueError('Missing preserved P argument: ' + option) from None


def verify_registration(root, commit, engine):
    if not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('An immutable ARIMA repair registration is required')
    root, engine = Path(root).absolute(), Path(engine).absolute()
    raw = (root / 'registration.json').read_bytes()
    published = subprocess.check_output(['git', '-C', str(engine), 'show', commit + ':' + REGISTRATION_PATH])
    if raw != published:
        raise ValueError('The ARIMA repair registration differs from its published commit')
    record = json.loads(raw)
    expected = {'protocol_id': 'e2-reduced-delivery-v1', 'engine_commit': ENGINE_COMMIT,
        'resource_commit': RESOURCE_COMMIT}
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError('The original E/P execution authorities must remain unchanged')
    for relative in FILES:
        path = root / relative
        if (not path.resolve().is_relative_to(root.resolve())
                or any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(root))
                or hashlib.sha256(path.read_bytes()).hexdigest() != record.get('files_sha256', {}).get(relative)):
            raise ValueError('A registered ARIMA repair source differs: ' + relative)
    if Path(__file__).resolve() != (root / FILES[1]).resolve():
        raise ValueError('Execute the registered ARIMA worker itself')
    return record


def validate_opportunity(spec, mode):
    if (mode != 'final' or spec.get('kind') not in ('classical', 'final-score')
            or spec.get('kind') == 'classical' and spec.get('model') != 'ARIMA'):
        raise ValueError('The ARIMA repair cannot execute selection or neural fits')
    return spec


def write_admission(directory, value):
    directory = Path(directory).absolute()
    target = directory / SIDECAR
    if any(p.is_symlink() for p in (target, *target.parents)):
        raise ValueError('The ARIMA repair admission must not follow a symlink')
    payload = canonical(value) + b'\n'
    directory.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != payload:
            raise ValueError('An immutable ARIMA repair admission differs')
    else:
        with target.open('xb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--arima-root', type=Path, required=True)
    parser.add_argument('--arima-commit', required=True)
    own, remainder = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    engine = Path(_argument(remainder, '--engine-root')).absolute()
    extension = Path(_argument(remainder, '--extension-root')).absolute()
    resource_commit = _argument(remainder, '--extension-commit')
    if resource_commit != RESOURCE_COMMIT:
        raise ValueError('Retain the immutable P resource worker')
    record = verify_registration(own.arima_root, own.arima_commit, engine)
    worker_path = extension / 'scripts/e2_resource_worker.py'
    if worker_path.is_symlink() or hashlib.sha256(worker_path.read_bytes()).hexdigest() != RESOURCE_WORKER_SHA256:
        raise ValueError('The immutable P resource worker source differs')
    loader = importlib.util.spec_from_file_location('e2_original_P_arima_worker', worker_path)
    worker = importlib.util.module_from_spec(loader); loader.loader.exec_module(worker)
    # P verifies its original Git-bound sources and 1800-second policy before
    # this adapter adds its separate admission. P.main repeats that acceptance.
    worker.verify_extension(engine, extension, resource_commit)
    spec_path = Path(_argument(remainder, '--worker-spec'))
    raw_spec = spec_path.read_bytes(); spec = json.loads(raw_spec)
    validate_opportunity(spec, _argument(remainder, '--mode'))
    expected = spec.get('worker_spec_sha256'); unsigned = dict(spec)
    unsigned.pop('worker_spec_sha256', None)
    if hashlib.sha256(canonical(unsigned)).hexdigest() != expected:
        raise ValueError('The original E worker specification checksum differs')
    if Path(spec.get('output', '')).resolve() != spec_path.parent.resolve():
        raise ValueError('The ARIMA worker output differs from its existing specification')
    sys.path.insert(0, str(engine / 'scripts'))
    sys.path.insert(0, str(own.arima_root / 'scripts'))
    from e2_arima_prediction_compat import install_prediction_raw_alias
    admission = install_prediction_raw_alias(engine)
    admission.update(repair_registration_commit=own.arima_commit,
        repair_registration_path=REGISTRATION_PATH,
        repair_source_sha256=record['files_sha256'][FILES[0]],
        repair_worker_sha256=record['files_sha256'][FILES[1]],
        resource_registration_commit=RESOURCE_COMMIT,
        resource_worker_sha256=RESOURCE_WORKER_SHA256,
        worker_spec_sha256=expected, worker_spec_file_sha256=hashlib.sha256(raw_spec).hexdigest(),
        kind=spec['kind'], model=spec.get('model'), mode=spec.get('mode'),
        market_data_opened_before_admission=False)
    admission['sha256'] = hashlib.sha256(canonical(admission)).hexdigest()
    write_admission(spec_path.parent, admission)
    print(json.dumps({'event': 'arima_prediction_receipt_compatibility_applied',
        'kind': spec['kind'], 'model': spec.get('model'),
        'repair_registration_commit': own.arima_commit,
        'numeric_predictions_changed': False, 'scientific_fit_repeated': False}), flush=True)
    return worker.main(remainder)


if __name__ == '__main__':
    raise SystemExit(main())
