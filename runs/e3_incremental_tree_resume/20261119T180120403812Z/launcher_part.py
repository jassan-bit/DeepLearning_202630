"""Publish the exact pending snapshot with incremental Git trees, then resume.

AI assistance declared. Scientific source, protocol, job bytes and deadlines
remain unchanged. Authoritative commits, refs and state are still handled by
the original frozen continuation implementation.
"""
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request
from urllib.error import HTTPError
from getpass import getpass

SOURCE = '5a07d6157a9796ca54416dc5dea1e1527e731597'
MASTER = '20261119T031104515520Z'
PROTOCOL = '5d3e5c0f7440efaa4993e193526d0d53628b7bb053d87a2324975d673379136a'
REPO = 'jassan-bit/DeepLearning_202630'
SCOPE = 'experiments/e3_evaluacion_ampliada_20261119/orchestration'
ROOT_API = '/repos/' + REPO
SEED_SHA256 = '32a90c1eee50dfb1571dfee51b342923addbe34045a454b6b0871ba0803b5ae8'
ALLOWED_READ = re.compile(r'/repos/jassan-bit/DeepLearning_202630/git/'
    r'(?:trees/[0-9a-f]{40}(?:\?recursive=1)?|ref/heads/e3-extended-audit-[0-9]{8}T[0-9]{12}Z)')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def emit(event):
    print(json.dumps(event, ensure_ascii=False, sort_keys=True), flush=True)


class CaptureTreeError:
    def __init__(self, original, token):
        self.original, self.token = original, token

    def open(self, request, **kwargs):
        try:
            return self.original.open(request, **kwargs)
        except HTTPError as error:
            if (error.code == 422 and request.get_method() == 'POST'
                    and request.full_url == 'https://api.github.com' + ROOT_API + '/git/trees'):
                raw = error.read(65537)
                try:
                    body = json.loads(raw) if len(raw) <= 65536 else {}
                except (ValueError, UnicodeError):
                    body = {}
                if not isinstance(body, dict):
                    body = {}
                message = str(body.get('message', 'No readable JSON message')).replace(self.token, '[REDACTED]')[:2000]
                emit({'event': 'incremental_tree_rejected', 'http': 422, 'message': message})
            raise


class TreeAdapter:
    def __init__(self, api, known_blobs):
        self.api, self.original, self.known_blobs = api, api._request, known_blobs
        self.trees = {}
        self.reused_blobs = 0

    def request(self, method, path, token, payload=None, **kwargs):
        if method == 'POST' and path == ROOT_API + '/git/blobs' and isinstance(payload, dict):
            if set(payload) == {'encoding', 'content'} and payload.get('encoding') == 'base64':
                self.api._inputs(method, path, token, payload)
                try:
                    raw = base64.b64decode(payload['content'], validate=True)
                except (ValueError, TypeError):
                    return self.original(method, path, token, payload, **kwargs)
                sha = self.api._git_sha(raw)
                if sha in self.known_blobs:
                    self.reused_blobs += 1
                    if self.reused_blobs == 1 or self.reused_blobs % 64 == 0:
                        emit({'event': 'verified_pending_blobs_reused', 'count': self.reused_blobs})
                    return {'sha': sha}
        if method == 'POST' and path == ROOT_API + '/git/trees' and isinstance(payload, dict):
            rows = payload.get('tree')
            if ('base_tree' in payload and isinstance(rows, list)
                    and any(isinstance(r, dict) and isinstance(r.get('path'), str)
                            and r['path'].startswith('runs/e3_extended_audit/') for r in rows)):
                self.api._inputs(method, path, token, payload)
                key = hashlib.sha256(self.api._canonical(payload)).hexdigest()
                if key not in self.trees:
                    self.trees[key] = build_incremental_tree(self.original, token, payload, emit=emit)
                return self.trees[key]
        return self.original(method, path, token, payload, **kwargs)


def install_api(api, token, known_blobs):
    if getattr(api, '_e3_incremental_tree_installed', False):
        return api
    api._opener = CaptureTreeError(api._opener, token)
    adapter = TreeAdapter(api, known_blobs)
    api._request = adapter.request
    api._e3_incremental_tree_installed = True
    api._e3_incremental_tree_adapter = adapter
    return api


def prepare_api(checkout, token, known_blobs):
    legacy = load(checkout / 'experiments/e3_remediacion_20261118/evaluation/colab_github.py', 'incremental_legacy')
    transport = load(checkout / SCOPE / 'github_transport.py', 'incremental_transport')
    return install_api(transport.wrap(legacy), token, known_blobs)


def finish_pending(base, token, api, continuation, known_blobs):
    statepath = api._regular_path(base / ('e3-extended-continuation-' + MASTER + '.json'))
    if not statepath.is_file():
        return {'status': 'no_local_state'}
    state_raw = statepath.read_bytes()
    state = json.loads(state_raw)
    continuation._compatible(state, SOURCE, PROTOCOL, MASTER, api)
    require(state.get('kind') == 'e3_audit_continuation_state' and state.get('schema_version') == 1,
            'Unexpected continuation state')
    protocol = api._regular_path(base / ('e3-extended-audit-' + MASTER) / 'protocol.json')
    require(hashlib.sha256(protocol.read_bytes()).hexdigest() == PROTOCOL, 'Frozen protocol differs')
    pending = state.get('pending')
    if pending is None:
        return {'status': 'no_pending_snapshot'}
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
            'Pending snapshot does not extend this saved head')
    files = continuation.STORE.snapshot(snapshot, token, SOURCE, part, api, 'audit')
    require(sum(map(len, files.values())) <= continuation.MAX_PART, 'Pending snapshot exceeds its limit')
    payload = {n.removeprefix('payload/'): b for n, b in files.items() if n.startswith('payload/')}
    require(isinstance(receipt.get('files'), dict) and set(payload) == set(receipt['files'])
            and all(continuation._meta(raw, api) == receipt['files'][name] for name, raw in payload.items()),
            'Pending payload differs from its immutable receipt')
    prefix = 'runs/e3_extended_audit/' + part
    reference = api._request('GET', ROOT_API + '/git/ref/heads/e3-extended-audit-' + part,
                             token, allow_missing=True)
    if reference is None:
        parent = continuation._commit(api, token, state['head_commit'])
        expected = {name: api._git_sha(raw) for name, raw in files.items()}
        request = {'base_tree': parent['tree']['sha'], 'tree': [
            {'path': prefix + '/' + name, 'mode': '100644', 'type': 'blob', 'sha': sha}
            for name, sha in sorted(expected.items())]}
        print('Construyendo por carpetas la misma parte pendiente:', part, flush=True)
        tree = api._request('POST', ROOT_API + '/git/trees', token, request)
        leaf = continuation._subtree(api, token, tree['sha'], prefix)
        rows = continuation._tree(api, token, leaf, recursive=True)
        blobs = {name: row for name, row in rows.items() if row.get('type') == 'blob'}
        require(set(blobs) == set(files) and all(
            blobs[name].get('mode') == '100644' and blobs[name].get('sha') == expected[name]
            and blobs[name].get('size') == len(raw) for name, raw in files.items()),
            'Remote incremental tree differs from the exact pending snapshot')
        # Cache only blobs whose actual remote tree inventory and sizes were
        # checked. The frozen publisher will still verify the complete tree.
        known_blobs.update(expected.values())
    require(statepath.read_bytes() == state_raw, 'Continuation state changed concurrently')
    saved = continuation._finish_pending(state, statepath, api, token)
    committed = json.loads(statepath.read_bytes())
    require(committed.get('pending') is None and committed.get('head_part') == part
            and committed.get('source_commit') == SOURCE and committed.get('protocol_sha256') == PROTOCOL,
            'Original publisher did not complete this exact pending snapshot')
    print('Parte pendiente guardada y verificada:', saved['run_url'], flush=True)
    return {'status': 'pending_saved_by_original_publisher', **saved}


def install_controller(controller, checkout, token, known_blobs):
    original_load = controller.load
    target = (checkout / SCOPE / 'github_transport.py').resolve()

    def installed_load(path, name):
        module = original_load(path, name)
        if name == 'extended_controller_transport' and Path(path).resolve() == target:
            original_wrap = module.wrap

            def wrap(legacy, **kwargs):
                return install_api(original_wrap(legacy, **kwargs), token, known_blobs)
            module.wrap = wrap
        return module
    controller.load = installed_load
    return controller


def retryable_read(error, checkout):
    if type(error).__name__ != 'StorageError':
        return False
    if str(error).startswith('github_transport_network_read_failed '):
        return True
    if str(error) != 'github_transport_http_error operation=git_trees_read status=404 attempts=1':
        return False
    target = (checkout / SCOPE / 'github_transport.py').resolve()
    trace = error.__traceback__
    while trace is not None:
        frame, values = trace.tb_frame, trace.tb_frame.f_locals
        if (frame.f_code.co_name == '_perform' and Path(frame.f_code.co_filename).resolve() == target
                and values.get('method') == 'GET' and type(values.get('status')) is int
                and values['status'] == 404 and values.get('allow_missing') is False
                and isinstance(values.get('path'), str) and ALLOWED_READ.fullmatch(values['path'])):
            return True
        trace = trace.tb_next
    return False


def checkout_source(base):
    checkout = base / ('e3-extended-source-' + SOURCE)
    require(not any(p.is_symlink() for p in (checkout, base, *base.parents)), 'Symlink checkout refused')
    env = dict(os.environ)
    for key in list(env):
        if re.search(r'(?:^|_)(?:TOKEN|PAT|SECRET|PASSWORD|CREDENTIAL)(?:_|$)', key.upper()):
            env.pop(key, None)
    env.update(GIT_TERMINAL_PROMPT='0', PYTHONNOUSERSITE='1')
    if not checkout.exists():
        sparse = ['src', 'scripts', 'experiments/e3_evaluacion_ampliada_20261119', 'deployment/science',
            'experiments/e3_cierre_integral_20261118/campaign', 'experiments/e3_cierre_integral_20261118/data',
            'experiments/e3_cierre_integral_20261118/orchestration', 'experiments/e3_remediacion_20261118/evaluation',
            'experiments/e3_dataset_admission_20261118/inputs', 'experiments/e3_d01_admission_20261118',
            'experiments/e3_ficha_cnn_lstm_atencion_v2_20261116/verification', 'experiments/e3_estadistica_posterior_20261117']
        subprocess.run(['git', 'clone', '--depth', '1', '--filter=blob:none', '--no-checkout',
                        'https://github.com/' + REPO + '.git', str(checkout)], env=env, check=True)
        for args in (['sparse-checkout', 'init', '--cone'], ['sparse-checkout', 'set', *sparse],
                     ['fetch', '--depth', '1', 'origin', SOURCE], ['checkout', '--detach', SOURCE]):
            subprocess.run(['git', *args], cwd=checkout, env=env, check=True)
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, env=env, text=True).strip()
    require(actual == SOURCE, 'Scientific checkout SHA differs')
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD'], cwd=checkout, env=env, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return checkout


def main(namespace, *, base=None):
    base = Path('/content') if base is None else Path(base).absolute()
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
    print('GPU asignada:', gpu, flush=True)
    require(gpu == 'NVIDIA A100-SXM4-80GB', 'Resume requires the original A100 80 GB cohort')
    checkout = checkout_source(base)
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
    namespace['_E3_GITHUB_TOKEN'] = token
    seed_url = ('https://raw.githubusercontent.com/' + REPO + '/88aa4a7bdd20f6485880a895cb251b21ae049fcc/'
                'runs/e3_view_reseed_resume/20261119T151114001132Z/resume_cell.py')
    with urllib.request.urlopen(seed_url, timeout=60) as response:
        seed_source = response.read()
    require(hashlib.sha256(seed_source).hexdigest() == SEED_SHA256, 'Immutable view helper hash differs')
    seed = {'__name__': 'incremental_view_helper'}
    exec(compile(seed_source, 'e3_verified_view_seed_helper', 'exec'), seed)
    known_blobs = set()
    view_retry_used = False
    read_retries = 0
    while True:
        try:
            api = prepare_api(checkout, token, known_blobs)
            continuation = load(checkout / SCOPE / 'continuation.py', 'incremental_continuation')
            namespace['_E3_INCREMENTAL_PENDING_SAVE'] = finish_pending(base, token, api, continuation, known_blobs)
            seed['seed_view'](base)
            controller = load(checkout / SCOPE / 'controller.py', 'incremental_original_controller')
            install_controller(controller, checkout, token, known_blobs)
            result = controller.run(SOURCE, token, base=base, start_new=False, master_run_id=MASTER)
            namespace['_E3_EXTENDED_AUDIT_RESULT'] = result
            return result
        except Exception as error:
            if not view_retry_used and seed['_expected_view_error'](error, base):
                view_retry_used = True
                seed['seed_view'](base, required=True)
                continue
            if read_retries == 3 or not retryable_read(error, checkout):
                raise
            read_retries += 1
            print(f'Lectura de GitHub interrumpida; reanudación del mismo master en 30 s ({read_retries}/3).', flush=True)
            time.sleep(30)


if __name__ == '__main__':
    main(globals())
