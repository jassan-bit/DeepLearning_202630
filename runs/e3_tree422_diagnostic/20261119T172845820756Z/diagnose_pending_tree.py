"""Observe one pending Git tree save without training or changing audit state.

AI assistance declared. Reconstructs the original immutable pending tree,
checks its snapshot, and asks GitHub to create that content-addressed object.
No commit/ref is created; output, snapshots, protocol and state are unchanged.
"""
from getpass import getpass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.error import HTTPError

SOURCE = '5a07d6157a9796ca54416dc5dea1e1527e731597'
MASTER = '20261119T031104515520Z'
PROTOCOL = '5d3e5c0f7440efaa4993e193526d0d53628b7bb053d87a2324975d673379136a'
REPO = 'jassan-bit/DeepLearning_202630'
SCOPE = 'experiments/e3_evaluacion_ampliada_20261119/orchestration'
TREE_PATH = '/repos/' + REPO + '/git/trees'
TREE_URL = 'https://api.github.com' + TREE_PATH


class TreeValidationRejected(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _clean(value, token, limit=2000):
    return str(value).replace(token, '[REDACTED]')[:limit]


class Observe422:
    def __init__(self, original, token, report):
        self.original, self.token, self.report = original, token, report

    def open(self, request, **kwargs):
        try:
            return self.original.open(request, **kwargs)
        except HTTPError as error:
            if (error.code != 422 or request.get_method() != 'POST'
                    or request.full_url != TREE_URL):
                raise
            try:
                raw = error.read(65537)
                try:
                    body = json.loads(raw) if len(raw) <= 65536 else {}
                except (ValueError, UnicodeError):
                    body = {}
                if not isinstance(body, dict):
                    body = {}
                errors = body.get('errors', [])
                if not isinstance(errors, list):
                    errors = [errors]
                safe_errors = []
                for item in errors[:20]:
                    if isinstance(item, dict):
                        safe_errors.append({k: _clean(item[k], self.token, 1000)
                                            for k in ('resource', 'field', 'code', 'message') if k in item})
                    else:
                        safe_errors.append(_clean(item, self.token, 1000))
                self.report.update(http=422, status='tree_validation_rejected',
                    message=_clean(body.get('message', 'No readable JSON message'), self.token),
                    errors=safe_errors,
                    request_id=_clean(error.headers.get('X-GitHub-Request-Id', '')
                                      if error.headers is not None else '', self.token, 200))
            finally:
                error.close()
            raise TreeValidationRejected('GitHub validation details captured safely') from None


def token_from(namespace):
    token = (os.environ.get('GITHUB_TOKEN') or namespace.get('_E3_GITHUB_TOKEN') or '').strip()
    if not token:
        try:
            from google.colab import userdata
            token = (userdata.get('GITHUB_TOKEN') or '').strip()
        except Exception:
            token = ''
    if not token:
        token = getpass('GITHUB_TOKEN (entrada oculta): ').strip()
    require(bool(token), 'A GitHub token is required')
    return token


def main(namespace, *, base=None):
    base = Path('/content') if base is None else Path(base).absolute()
    checkout = base / ('e3-extended-source-' + SOURCE)
    require(not any(p.is_symlink() for p in (checkout, base, *base.parents)), 'Symlink checkout refused')
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip()
    require(actual == SOURCE, 'Scientific checkout SHA differs')
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD'], cwd=checkout, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    token = token_from(namespace)
    api = load(checkout / 'experiments/e3_remediacion_20261118/evaluation/colab_github.py', 'tree422_legacy')
    api = load(checkout / SCOPE / 'github_transport.py', 'tree422_transport').wrap(api)
    continuation = load(checkout / SCOPE / 'continuation.py', 'tree422_continuation')
    statepath = api._regular_path(base / ('e3-extended-continuation-' + MASTER + '.json'))
    state_raw = statepath.read_bytes()
    state = json.loads(state_raw)
    continuation._compatible(state, SOURCE, PROTOCOL, MASTER, api)
    require(state.get('kind') == 'e3_audit_continuation_state' and state.get('schema_version') == 1,
            'Unexpected continuation state')
    protocol = api._regular_path(base / ('e3-extended-audit-' + MASTER) / 'protocol.json')
    require(hashlib.sha256(protocol.read_bytes()).hexdigest() == PROTOCOL, 'Frozen protocol differs')
    pending = state.get('pending')
    if pending is None:
        report = {'status': 'no_pending_snapshot', 'master_run_id': MASTER,
                  'new_real_data_fits': 0, 'state_unchanged': True}
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        namespace['_E3_TREE422_DIAGNOSTIC'] = report
        return report
    require(isinstance(pending, dict), 'Invalid pending snapshot')
    snapshot = api._regular_path(pending['snapshot'], directory=True)
    receipt = json.loads((snapshot / 'stage_receipt.json').read_bytes())
    part = pending.get('part_id')
    require(isinstance(part, str) and re.fullmatch(r'[0-9]{8}T[0-9]{12}Z', part)
            and receipt.get('part_id') == part and receipt.get('source_commit') == SOURCE
            and receipt.get('master_run_id') == MASTER and receipt.get('protocol_sha256') == PROTOCOL
            and receipt.get('sequence') == state.get('sequence', -1) + 1
            and receipt.get('previous_part') == {'commit_sha': state.get('head_commit'),
                'part_id': state.get('head_part'), 'receipt_sha256': state.get('head_receipt_sha256')},
            'Pending snapshot does not extend this exact saved head')
    files = continuation.STORE.snapshot(snapshot, token, SOURCE, part, api, 'audit')
    require(sum(map(len, files.values())) <= continuation.MAX_PART, 'Pending snapshot exceeds its limit')
    payload = {name.removeprefix('payload/'): raw for name, raw in files.items() if name.startswith('payload/')}
    require(isinstance(receipt.get('files'), dict) and set(payload) == set(receipt['files'])
            and all(continuation._meta(raw, api) == receipt['files'][name] for name, raw in payload.items()),
            'Pending payload differs from its original receipt')
    parent = continuation._commit(api, token, state['head_commit'])
    prefix = 'runs/e3_extended_audit/' + part
    entries = [{'path': prefix + '/' + name, 'mode': '100644', 'type': 'blob',
                'sha': api._git_sha(raw)} for name, raw in sorted(files.items())]
    request_payload = {'base_tree': parent['tree']['sha'], 'tree': entries}
    report = {'master_run_id': MASTER, 'pending_part_id': part, 'pending_sequence': receipt['sequence'],
              'saved_head': state['head_commit'], 'base_tree': parent['tree']['sha'],
              'entry_count': len(entries), 'snapshot_bytes': sum(map(len, files.values())),
              'pending_payload_hashes_verified': True, 'new_real_data_fits': 0,
              'training_started': False, 'commit_or_ref_created': False}
    existing = api._request('GET', '/repos/' + REPO + '/git/ref/heads/e3-extended-audit-' + part,
                            token, allow_missing=True)
    if existing is not None:
        saved = continuation._read_part(api, token, existing['object']['sha'], part)
        continuation._validate_part_files(saved, api, token)
        require(saved['receipt_raw'] == files['stage_receipt.json'], 'Existing part receipt differs')
        report.update(status='pending_part_already_saved', commit=existing['object']['sha'])
    else:
        api._opener = Observe422(api._opener, token, report)
        try:
            tree = api._request('POST', TREE_PATH, token, request_payload)
            require(isinstance(tree, dict) and re.fullmatch(r'[0-9a-f]{40}', tree.get('sha', '')),
                    'GitHub returned an invalid tree identity')
            report.update(status='tree_creation_accepted', http=201, tree_sha=tree['sha'])
        except TreeValidationRejected:
            pass
    require(statepath.read_bytes() == state_raw, 'State changed concurrently; preserve this session')
    report['state_unchanged'] = True
    namespace['_E3_TREE422_DIAGNOSTIC'] = report
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == '__main__':
    main(globals())
