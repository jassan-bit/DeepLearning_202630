#!/content/entregable2_gpu_workspace/reduced-runtime-py312-cu126-v1/venv-py312-cu126/bin/python
"""Transport-only interpreter adapter; registered science bytes are unchanged."""
from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import time

ORIGINAL_COMMIT = 'ae1a2d4f96807a20ea92c313297a5c09512a4817'
WORKSPACE = Path('/content/entregable2_gpu_workspace')
REAL_PYTHON = WORKSPACE/'reduced-runtime-py312-cu126-v1/venv-py312-cu126/bin/python'
NEW_RELEASE = 'e2-guide-closure-posthoc-20261113-v1'
OLD_RELEASE = 'e2-reduced-results-20261112-v1'
REGISTRATION = 'experiments/e2_posthoc_fit_reproduction_20261113/registration.json'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def regular(path):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Transport paths must not use symlinks')
    return path


def argument(args, name):
    if args.count(name) != 1:
        raise ValueError('Exactly one registered '+name+' is required')
    index = args.index(name)
    if index+1 >= len(args):
        raise ValueError('Missing registered argument '+name)
    return args[index+1]


def validate_invocation(args):
    if not args or '--orchestrate' not in args:
        return None
    expected = WORKSPACE/('guide-closure-extension-'+ORIGINAL_COMMIT)
    if regular(args[0]).absolute() != expected/'scripts/e2_posthoc_fit_reproduction.py':
        raise ValueError('Only the unchanged published orchestrator may use this adapter')
    if argument(args, '--configuration-commit') != ORIGINAL_COMMIT:
        raise ValueError('Preserve the original published configuration commit')
    extension = regular(argument(args, '--extension-root'))
    if extension.absolute() != expected:
        raise ValueError('Preserve the original extension root')
    destination = regular(argument(args, '--destination'))
    if destination.absolute() != WORKSPACE/'e2-guide-closure-posthoc-v1':
        raise ValueError('Preserve the same additive campaign and shared budget')
    registration = json.loads(regular(extension/REGISTRATION).read_bytes())
    for relative, expected_sha in registration['files_sha256'].items():
        path = regular(extension/relative)
        if not path.resolve().is_relative_to(extension.resolve()) or sha(path.read_bytes()) != expected_sha:
            raise ValueError('Original registered bytes changed: '+relative)
    return extension, destination, registration


def immutable_json(path, value):
    path = regular(path)
    raw = canonical(value)+b'\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('Preserve the existing transport amendment for review')
        return
    with path.open('xb') as stream:
        stream.write(raw)


def install_release_amendment(extension, destination, registration, token):
    sys.path.insert(0, str(extension/'scripts'))
    module = importlib.import_module('e2_reduced_github_backup')
    if Path(module.__file__).resolve() != (extension/'scripts/e2_reduced_github_backup.py').resolve():
        raise ValueError('Registered backup module was shadowed')
    if module.RELEASE_TAG != OLD_RELEASE:
        raise ValueError('Unexpected original release transport')
    module.RELEASE_TAG = NEW_RELEASE
    client = module.GitHubBackup(token, timeout=20, max_conflicts=2)
    client._branch_head()
    client._checked = True
    release = client._ensure_release()
    proof = {'version': 'e2-posthoc-release-capacity-amendment-v1', 'status': 'completed',
        'original_configuration_commit': ORIGINAL_COMMIT, 'original_release_tag': OLD_RELEASE,
        'new_release_tag': NEW_RELEASE, 'new_release_id': release['id'],
        'registered_backup_sha256': registration['files_sha256']['scripts/e2_reduced_github_backup.py'],
        'adapter_sha256': sha(Path(__file__).read_bytes()), 'original_source_files_modified': False,
        'model_inputs_outputs_or_optimizer_changed': False, 'budget_reset': False,
        'common_control_namespace': 'control/e2-guide-closure-posthoc-v1',
        'reason': 'Separate posthoc release: historic release has510assets; about576new assets would exceed1000.'}
    folder = destination/'transport_amendments/release-capacity-v1'
    immutable_json(folder/'amendment.json', proof)
    receipt = {'status': 'completed', 'attempt_id': 'e2-posthoc-transport-release-v1',
        'scientific_result': False, 'original_configuration_commit': ORIGINAL_COMMIT,
        'artifacts_sha256': {'amendment.json': sha((folder/'amendment.json').read_bytes())}}
    immutable_json(folder/'result.json', receipt)
    backed = client.persist_result(folder, receipt)
    if backed.get('status') != 'backed_up':
        raise RuntimeError('Transport amendment must be backed up before resuming')
    print(json.dumps({'event': 'posthoc_transport_amendment_backed_up',
        'release_tag': NEW_RELEASE, 'manifest_path': backed['manifest_path']}), flush=True)
    return module


def main(args=None):
    args = list(sys.argv[1:] if args is None else args)
    invocation = validate_invocation(args)
    if invocation is None:
        os.execv(str(REAL_PYTHON), [str(REAL_PYTHON), *args])
        raise AssertionError('exec returned')
    started = time.monotonic()
    extension, destination, registration = invocation
    token = sys.stdin.readline().rstrip('\r\n')
    if not token or any(c.isspace() for c in token):
        raise ValueError('Existing GitHub credential must arrive only through STDIN')
    install_release_amendment(extension, destination, registration, token)
    index = args.index('--launcher-elapsed-seconds')
    args[index+1] = str(float(args[index+1])+time.monotonic()-started)
    sys.stdin = io.StringIO(token+'\n')
    sys.argv = args
    runpy.run_path(args[0], run_name='__main__')


if __name__ == '__main__':
    main()
