"""Offline integration of incremental I/O with unchanged audit publication.

Uses the real frozen publisher and synthetic Git objects. No HTTP, model
initialization, fitting or remote mutation is performed.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import unquote
from urllib.request import Request

HERE = Path(__file__).resolve().parent
FROZEN = '5a07d6157a9796ca54416dc5dea1e1527e731597'


def checkout():
    requested = os.environ.get('E3_SCIENTIFIC_CHECKOUT')
    candidates = ([Path(requested)] if requested else
                  [*HERE.parents, Path('/content/e3-extended-source-' + FROZEN), Path('/workspace/e3_review9')])
    for candidate in candidates:
        if (candidate/'experiments/e3_evaluacion_ampliada_20261119/orchestration/continuation.py').is_file():
            return candidate
    raise RuntimeError('Set E3_SCIENTIFIC_CHECKOUT to the pinned scientific checkout')


SCIENCE = checkout()
SCOPE = SCIENCE/'experiments/e3_evaluacion_ampliada_20261119/orchestration'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


C = load(SCOPE/'continuation.py', 'incremental_resume_real_continuation')
A = load(SCIENCE/'experiments/e3_remediacion_20261118/evaluation/colab_github.py',
         'incremental_resume_real_legacy_api')
TOKEN = 'offline_integration_token'


def git_sha(kind, raw):
    return hashlib.sha1(kind.encode() + b' ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


class FakeGitServer:
    """Small Git object store with real object identities and immutable refs."""
    REPOSITORY = C.REPOSITORY
    StorageError = A.StorageError
    StorageConflict = A.StorageConflict
    _token = staticmethod(A._token)
    _canonical = staticmethod(A._canonical)
    _git_sha = staticmethod(A._git_sha)
    _safe_relative = staticmethod(A._safe_relative)
    _regular_path = staticmethod(A._regular_path)

    def __init__(self):
        self.blobs = {}
        self.trees = {}
        self.commits = {}
        self.refs = {}
        self.calls = []
        self._opener = SimpleNamespace(open=lambda *args, **kwargs: None)

    def validate_credentials(self, token):
        assert token == TOKEN
        return {'authenticated': True}

    def _inputs(self, method, path, token, payload):
        assert token == TOKEN
        if payload is not None:
            assert TOKEN.encode() not in self._canonical(payload)
        return token, None if payload is None else self._canonical(payload)

    def blob(self, raw):
        sha = git_sha('blob', raw)
        self.blobs[sha] = raw
        return sha

    def direct_tree(self, entries):
        rows = [{k: e[k] for k in ('path', 'mode', 'type', 'sha')} for e in entries]
        rows.sort(key=lambda e: e['path'].encode() + (b'/' if e['type'] == 'tree' else b''))
        raw = b''
        for row in rows:
            mode = '40000' if row['mode'] == '040000' else row['mode']
            raw += mode.encode() + b' ' + row['path'].encode() + b'\0' + bytes.fromhex(row['sha'])
            if row['type'] == 'blob':
                if row['sha'] not in self.blobs:
                    raise self.StorageError('missing referenced blob')
            elif row['type'] == 'tree' and row['sha'] not in self.trees:
                raise self.StorageError('missing referenced tree')
        sha = git_sha('tree', raw)
        self.trees[sha] = rows
        return sha

    def flat(self, sha):
        result = {}
        def visit(current, prefix=''):
            for row in self.trees[current]:
                name = prefix + row['path']
                if row['type'] == 'tree':
                    visit(row['sha'], name + '/')
                else:
                    result[name] = {k: row[k] for k in ('mode', 'type', 'sha')}
        visit(sha)
        return result

    def from_flat(self, flat):
        nested = {}
        for path, value in flat.items():
            current = nested
            parts = path.split('/')
            for name in parts[:-1]:
                current = current.setdefault(name, {})
            current[parts[-1]] = dict(value)
        def build(node):
            entries = []
            for name, value in node.items():
                if isinstance(value, dict) and 'sha' in value:
                    entries.append({'path': name, **value})
                else:
                    entries.append({'path': name, 'mode': '040000', 'type': 'tree', 'sha': build(value)})
            return self.direct_tree(entries)
        return build(nested)

    def rows(self, sha, recursive=False):
        result = []
        def visit(current, prefix=''):
            for original in self.trees[current]:
                row = copy.deepcopy(original)
                row['path'] = prefix + row['path']
                if row['type'] == 'blob':
                    row['size'] = len(self.blobs[row['sha']])
                result.append(row)
                if recursive and row['type'] == 'tree':
                    visit(row['sha'], row['path'] + '/')
        visit(sha)
        return result

    def _request(self, method, path, token, payload=None, allow_missing=False):
        self._inputs(method, path, token, payload)
        self.calls.append((method, path, copy.deepcopy(payload)))
        prefix = '/repos/' + self.REPOSITORY
        assert path.startswith(prefix)
        route = path[len(prefix):]
        if method == 'GET' and route.startswith('/git/trees/'):
            sha = route.split('/git/trees/', 1)[1].split('?')[0]
            if sha not in self.trees:
                if allow_missing: return None
                raise self.StorageError('missing tree')
            return {'sha': sha, 'truncated': False, 'tree': self.rows(sha, '?recursive=1' in route)}
        if method == 'GET' and route.startswith('/git/commits/'):
            return copy.deepcopy(self.commits[route.rsplit('/', 1)[1]])
        if method == 'GET' and route.startswith('/git/ref/heads/'):
            ref = 'refs/heads/' + route.split('/git/ref/heads/', 1)[1]
            if ref not in self.refs:
                if allow_missing: return None
                raise self.StorageError('missing ref')
            return copy.deepcopy(self.refs[ref])
        if method == 'GET' and route.startswith('/contents/'):
            name, commit = route[len('/contents/'):].rsplit('?ref=', 1)
            name = unquote(name)
            row = self.flat(self.commits[commit]['tree']['sha'])[name]
            raw = self.blobs[row['sha']]
            return {'type': 'file', 'path': name, 'sha': row['sha'], 'size': len(raw),
                    'encoding': 'base64', 'content': base64.b64encode(raw).decode()}
        if method == 'POST' and route == '/git/blobs':
            return {'sha': self.blob(base64.b64decode(payload['content'], validate=True))}
        if method == 'POST' and route == '/git/trees':
            if 'base_tree' in payload:
                flat = self.flat(payload['base_tree'])
                for row in payload['tree']:
                    flat[row['path']] = {k: row[k] for k in ('mode', 'type', 'sha')}
                sha = self.from_flat(flat)
            else:
                assert all('/' not in row['path'] for row in payload['tree'])
                sha = self.direct_tree(payload['tree'])
            return {'sha': sha, 'truncated': False, 'tree': self.rows(sha)}
        if method == 'POST' and route == '/git/commits':
            sha = hashlib.sha1(self._canonical(payload)).hexdigest()
            self.commits[sha] = {'sha': sha, 'tree': {'sha': payload['tree']},
                                 'parents': [{'sha': p} for p in payload['parents']]}
            return {'sha': sha}
        if method == 'POST' and route == '/git/refs':
            assert payload['ref'] not in self.refs
            value = {'ref': payload['ref'], 'object': {'sha': payload['sha'], 'type': 'commit'}}
            self.refs[payload['ref']] = value
            return copy.deepcopy(value)
        raise AssertionError((method, route))


def fixture(module, base):
    """Synthetic sequence 63 with unchanged scientific bytes and old siblings."""
    api = FakeGitServer()
    protocol_raw = b'{"offline_fixture":true}\n'
    protocol_hash = hashlib.sha256(protocol_raw).hexdigest()
    output = base/('e3-extended-audit-' + module.MASTER)
    output.mkdir()
    (output/'protocol.json').write_bytes(protocol_raw)
    payload = {
        'jobs/D03/MLP16__final__20261118/fit_request.json': b'{"seed":20261118}\n',
        'jobs/D03/MLP16__final__20261118/supervised/checkpoint.pt': b'unchanged serialized checkpoint bytes',
        'jobs/D03/MLP16__final__20261118/supervised/phase_started.json': b'{"deadline":12345}\n',
        'jobs/D03/MLP16__final__20261118/result.json': b'{"status":"completed"}\n',
        'resources.csv': b'job,seconds\nMLP16,120\n',
    }
    for name, raw in payload.items():
        target = output/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    frozen_blob = api.blob(b'frozen scientific file')
    old_blob = api.blob(b'older audit metadata')
    flat = {
        'src/frozen.py': {'mode': '100644', 'type': 'blob', 'sha': frozen_blob},
        'scripts/tool.sh': {'mode': '100755', 'type': 'blob', 'sha': frozen_blob},
        'link': {'mode': '120000', 'type': 'blob', 'sha': frozen_blob},
        'vendor/submodule': {'mode': '160000', 'type': 'commit', 'sha': 'd'*40},
        'runs/e3_extended_audit/20261119T165424079731Z/old.json':
            {'mode': '100644', 'type': 'blob', 'sha': old_blob},
    }
    parent_tree = api.from_flat(flat)
    head = 'c'*40
    api.commits[head] = {'sha': head, 'tree': {'sha': parent_tree}, 'parents': []}
    state = C._new_state(module.SOURCE, protocol_hash, module.MASTER)
    state.update(head_commit=head, head_part='20261119T165424079731Z',
                 head_receipt_sha256='b'*64, sequence=62)
    part = '20261119T170007583476Z'
    snapshot = base/('e3-audit-part-' + part + '-fixture')
    snapshot.mkdir()
    job = 'jobs/D03/MLP16__final__20261118'
    metadata = {name: C._meta(raw, api) for name, raw in payload.items()}
    receipt = {'kind': 'e3_closure_stage_receipt', 'profile': 'audit', 'schema_version': 1,
               'repository': C.REPOSITORY, 'source_commit': module.SOURCE,
               'protocol_sha256': protocol_hash, 'master_run_id': module.MASTER,
               'part_id': part, 'sequence': 63,
               'previous_part': {'commit_sha': head, 'part_id': state['head_part'],
                                 'receipt_sha256': state['head_receipt_sha256']},
               'files': metadata, 'new_attempted_jobs': [job], 'new_real_data_fits': 1,
               'R1_accredited': False, 'R3_accredited': False, 'main_updated': False,
               'controller_status': {'stage_status': 'failed', 'analysis_complete': False}}
    for name, raw in payload.items():
        target = snapshot/'payload'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    (snapshot/'stage_receipt.json').write_bytes(api._canonical(receipt))
    state['pending'] = {'part_id': part, 'snapshot': str(snapshot)}
    statepath = base/('e3-extended-continuation-' + module.MASTER + '.json')
    statepath.write_bytes(api._canonical(state))
    files = C.STORE.snapshot(snapshot, TOKEN, module.SOURCE, part, api, 'audit')
    # Simulates the original failed POST: all blobs already reached GitHub.
    for raw in files.values():
        api.blob(raw)
    return SimpleNamespace(api=api, protocol_hash=protocol_hash, output=output,
        state=state, statepath=statepath, part=part, snapshot=snapshot,
        receipt=receipt, files=files, original_flat=flat, parent_tree=parent_tree,
        original_output={p.relative_to(output).as_posix():p.read_bytes()
                         for p in output.rglob('*') if p.is_file()},
        original_snapshot={p.relative_to(snapshot).as_posix():p.read_bytes()
                           for p in snapshot.rglob('*') if p.is_file()})


class FrozenPublisherIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.algorithm = load(HERE/'incremental_git_trees.py', 'integration_real_incremental_algorithm')
        self.constants = SimpleNamespace(SOURCE=FROZEN, MASTER='20261119T031104515520Z')
        self.f = fixture(self.constants, self.base)

    def tearDown(self):
        self.temp.cleanup()

    def unchanged(self):
        for path, expected in self.f.original_output.items():
            self.assertEqual((self.f.output/path).read_bytes(), expected)
        for path, expected in self.f.original_snapshot.items():
            self.assertEqual((self.f.snapshot/path).read_bytes(), expected)

    def test_frozen_finish_pending_publishes_exact_chain_through_incremental_trees(self):
        api = self.f.api
        original = api._request
        def request(method, path, token, payload=None, **kwargs):
            if method == 'POST' and path == self.algorithm.ENDPOINT:
                return self.algorithm.build_incremental_tree(original, token, payload)
            return original(method, path, token, payload, **kwargs)
        api._request = request
        saved = C._finish_pending(self.f.state, self.f.statepath, api, TOKEN)
        self.assertEqual(saved['sequence'], 63)
        self.assertIsNone(self.f.state['pending'])
        self.assertEqual(json.loads(self.f.statepath.read_bytes()), self.f.state)
        self.assertEqual(self.f.state['files'], self.f.receipt['files'])
        self.assertEqual(self.f.state['counted_jobs'], self.f.receipt['new_attempted_jobs'])
        commit = api.commits[saved['commit_sha']]
        self.assertEqual(commit['parents'], [{'sha': 'c'*40}])
        flattened = api.flat(commit['tree']['sha'])
        for path, row in self.f.original_flat.items():
            self.assertEqual(flattened[path], row)
        expected_flat = dict(self.f.original_flat)
        for name, raw in self.f.files.items():
            path = C.PREFIX + self.f.part + '/' + name
            expected_flat[path] = {'sha': api._git_sha(raw), 'mode': '100644', 'type': 'blob'}
        self.assertEqual(flattened, expected_flat)
        self.assertEqual(commit['tree']['sha'], api.from_flat(expected_flat))
        self.assertTrue(all('base_tree' not in payload and all('/' not in row['path'] for row in payload['tree'])
                            for method, path, payload in api.calls
                            if method == 'POST' and path.endswith('/git/trees')))
        self.unchanged()

    def test_frozen_existing_pending_reference_is_reconciled_without_new_writes(self):
        api = self.f.api
        original = api._request
        def request(method, path, token, payload=None, **kwargs):
            if method == 'POST' and path == self.algorithm.ENDPOINT:
                return self.algorithm.build_incremental_tree(original, token, payload)
            return original(method, path, token, payload, **kwargs)
        api._request = request
        saved_part = C._upload_part(self.f.snapshot, self.f.state, api, TOKEN)
        before = sum(method == 'POST' for method, path, payload in api.calls)
        self.assertIsNotNone(self.f.state['pending'])
        saved = C._finish_pending(self.f.state, self.f.statepath, api, TOKEN)
        self.assertEqual(saved['commit_sha'], saved_part['commit_sha'])
        self.assertEqual(sum(method == 'POST' for method, path, payload in api.calls), before)
        self.assertIsNone(self.f.state['pending'])
        self.unchanged()


class LauncherIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.algorithm = load(HERE/'incremental_git_trees.py', 'launcher_integration_algorithm')
        self.module = load(HERE/'launcher_part.py', 'launcher_under_integration_test')
        self.module.build_incremental_tree = self.algorithm.build_incremental_tree
        self.f = fixture(self.module, self.base)
        self.module.PROTOCOL = self.f.protocol_hash
        self.silence = patch.object(self.module, 'emit', lambda value: None)
        self.silence.start()

    def tearDown(self):
        self.silence.stop()
        self.temp.cleanup()

    def unchanged_payloads(self):
        for path, expected in self.f.original_output.items():
            self.assertEqual((self.f.output/path).read_bytes(), expected)
        for path, expected in self.f.original_snapshot.items():
            self.assertEqual((self.f.snapshot/path).read_bytes(), expected)

    def test_actual_finish_pending_arms_verified_cache_then_original_publisher(self):
        known = set()
        api = self.module.install_api(self.f.api, TOKEN, known)
        with patch('sys.stdout', io.StringIO()):
            result = self.module.finish_pending(self.base, TOKEN, api, C, known)
        self.assertEqual(result['status'], 'pending_saved_by_original_publisher')
        self.assertEqual(result['sequence'], 63)
        self.assertEqual(known, {api._git_sha(raw) for raw in self.f.files.values()})
        self.assertFalse(any(method == 'POST' and path.endswith('/git/blobs')
                             for method, path, payload in api.calls))
        self.assertEqual(sum(method == 'POST' and path.endswith('/git/commits')
                             for method, path, payload in api.calls), 1)
        self.assertEqual(sum(method == 'POST' and path.endswith('/git/refs')
                             for method, path, payload in api.calls), 1)
        state = json.loads(self.f.statepath.read_bytes())
        self.assertIsNone(state['pending'])
        self.assertEqual(state['head_part'], self.f.part)
        self.assertEqual(state['sequence'], 63)
        self.unchanged_payloads()

    def test_remote_inventory_size_failure_never_arms_cache_or_commits(self):
        api = self.f.api
        original = api._request
        def bad_inventory(method, path, token, payload=None, **kwargs):
            result = original(method, path, token, payload, **kwargs)
            if method == 'GET' and '?recursive=1' in path:
                for row in result['tree']:
                    if row['type'] == 'blob':
                        row['size'] += 1
                        break
            return result
        api._request = bad_inventory
        known = set()
        self.module.install_api(api, TOKEN, known)
        before = self.f.statepath.read_bytes()
        with patch('sys.stdout', io.StringIO()), self.assertRaisesRegex(ValueError, 'Remote incremental tree'):
            self.module.finish_pending(self.base, TOKEN, api, C, known)
        self.assertEqual(known, set())
        self.assertEqual(self.f.statepath.read_bytes(), before)
        self.assertFalse(any(method == 'POST' and path.endswith(('/git/blobs', '/git/commits', '/git/refs'))
                             for method, path, payload in api.calls))
        self.unchanged_payloads()

    def test_changed_snapshot_payload_fails_before_any_mutation(self):
        path = self.f.snapshot/'payload/resources.csv'
        path.write_bytes(b'changed payload')
        known = set()
        api = self.module.install_api(self.f.api, TOKEN, known)
        before = self.f.statepath.read_bytes()
        with self.assertRaisesRegex(ValueError, 'Pending payload differs'):
            self.module.finish_pending(self.base, TOKEN, api, C, known)
        self.assertFalse(any(method == 'POST' for method, path, payload in api.calls))
        self.assertEqual(self.f.statepath.read_bytes(), before)
        self.assertEqual(known, set())

    def test_no_pending_does_not_mutate_or_publish(self):
        self.f.state['pending'] = None
        self.f.statepath.write_bytes(self.f.api._canonical(self.f.state))
        api = self.module.install_api(self.f.api, TOKEN, set())
        before = self.f.statepath.read_bytes()
        result = self.module.finish_pending(self.base, TOKEN, api, C, set())
        self.assertEqual(result, {'status': 'no_pending_snapshot'})
        self.assertFalse(api.calls)
        self.assertEqual(self.f.statepath.read_bytes(), before)

    def test_adapter_only_reuses_blob_after_it_is_in_verified_cache(self):
        known = set()
        api = self.module.install_api(self.f.api, TOKEN, known)
        raw = b'new transport body'
        payload = {'encoding': 'base64', 'content': base64.b64encode(raw).decode()}
        path = self.module.ROOT_API + '/git/blobs'
        api._request('POST', path, TOKEN, payload)
        count = sum(method == 'POST' and route == path for method, route, body in api.calls)
        self.assertEqual(count, 1)
        known.add(api._git_sha(raw))
        self.assertEqual(api._request('POST', path, TOKEN, payload), {'sha': api._git_sha(raw)})
        self.assertEqual(sum(method == 'POST' and route == path for method, route, body in api.calls), count)
        other = {'encoding': 'base64', 'content': base64.b64encode(b'other bytes').decode()}
        api._request('POST', path, TOKEN, other)
        self.assertEqual(sum(method == 'POST' and route == path for method, route, body in api.calls), count + 1)

    def test_adapter_delegates_non_audit_tree_and_rejects_mixed_scope(self):
        api = self.module.install_api(self.f.api, TOKEN, set())
        blob = api.blob(b'ordinary metadata')
        payload = {'base_tree': self.f.parent_tree, 'tree': [
            {'path': 'metadata.json', 'mode': '100644', 'type': 'blob', 'sha': blob}]}
        api._request('POST', self.algorithm.ENDPOINT, TOKEN, payload)
        self.assertTrue(any(method == 'POST' and body == payload for method, path, body in api.calls))
        before = len(api.calls)
        mixed = copy.deepcopy(payload)
        mixed['tree'].append({'path': C.PREFIX + self.f.part + '/other.json',
                              'mode': '100644', 'type': 'blob', 'sha': blob})
        with self.assertRaises(ValueError):
            api._request('POST', self.algorithm.ENDPOINT, TOKEN, mixed)
        self.assertEqual(len(api.calls), before)

    def test_controller_hook_is_exact_path_and_module_name(self):
        def original_load(path, name):
            return SimpleNamespace(wrap=lambda legacy, **kwargs: legacy)
        controller = SimpleNamespace(load=original_load)
        checkout_path = self.base/('e3-extended-source-' + self.module.SOURCE)
        target = checkout_path/self.module.SCOPE/'github_transport.py'
        self.module.install_controller(controller, checkout_path, TOKEN, set())
        for path, name in ((target, 'other_module'), (self.base/'other.py', 'extended_controller_transport')):
            api = FakeGitServer()
            result = controller.load(path, name).wrap(api)
            self.assertIs(result, api)
            self.assertFalse(getattr(api, '_e3_incremental_tree_installed', False))
        api = FakeGitServer()
        wrapped = controller.load(target, 'extended_controller_transport').wrap(api)
        self.assertIs(wrapped, api)
        self.assertTrue(api._e3_incremental_tree_installed)

    def run_main_fixture(self, outcomes):
        events = []
        seed_source = b"def seed_view(base, required=False):\n    (base/'seed.called').write_text('ready')\ndef _expected_view_error(error,base):\n    return False\n"
        original_digest = self.module.SEED_SHA256
        self.module.SEED_SHA256 = hashlib.sha256(seed_source).hexdigest()
        outcomes = iter(outcomes)
        def run(source, token, **kwargs):
            self.assertTrue((self.base/'seed.called').is_file())
            self.assertEqual(source, self.module.SOURCE)
            self.assertEqual(token, TOKEN)
            self.assertEqual(kwargs, {'base': self.base, 'start_new': False, 'master_run_id': self.module.MASTER})
            events.append('worker')
            value = next(outcomes)
            if isinstance(value, BaseException): raise value
            return value
        controller = SimpleNamespace(run=run, load=lambda *args:None)
        def module_loader(path, name):
            return controller if Path(path).name == 'controller.py' else C
        def finish(*args):
            events.append('finish')
            return {'status':'test_saved'}
        namespace = {'_E3_GITHUB_TOKEN': TOKEN,
                     'E3_EXTENDED_MASTER_RUN': 'incorrect_prior_master',
                     'E3_EXTENDED_START_NEW': True}
        sleeps = []
        response = io.BytesIO(seed_source)
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(self.module.subprocess, 'check_output', return_value='NVIDIA A100-SXM4-80GB\n'), \
             patch.object(self.module, 'checkout_source', return_value=SCIENCE), \
             patch.object(self.module.urllib.request, 'urlopen', return_value=response), \
             patch.object(self.module, 'prepare_api', return_value=self.f.api), \
             patch.object(self.module, 'finish_pending', side_effect=finish), \
             patch.object(self.module, 'load', side_effect=module_loader), \
             patch.object(self.module.time, 'sleep', side_effect=sleeps.append), \
             patch('sys.stdout', io.StringIO()):
            try:
                result = self.module.main(namespace, base=self.base)
            finally:
                self.module.SEED_SHA256 = original_digest
        return result, events, sleeps, namespace

    def test_main_saves_pending_and_seeds_before_fixed_original_worker(self):
        result, events, sleeps, namespace = self.run_main_fixture([{'status':'completed_fixture'}])
        self.assertEqual(result, {'status':'completed_fixture'})
        self.assertEqual(events, ['finish', 'worker'])
        self.assertEqual(sleeps, [])
        self.assertEqual(namespace['_E3_EXTENDED_AUDIT_RESULT'], result)

    def test_main_read_retry_keeps_same_master_and_has_bounded_waits(self):
        error = A.StorageError('github_transport_network_read_failed operation=git_trees_read attempts=1')
        result, events, sleeps, namespace = self.run_main_fixture([error, error, {'status':'ok'}])
        self.assertEqual(result, {'status':'ok'})
        self.assertEqual(events, ['finish', 'worker'] * 3)
        self.assertEqual(sleeps, [30, 30])

    def test_auth_and_model_errors_never_retry(self):
        errors = [A.StorageError('github_transport_auth_or_contents_write_required operation=git_trees_write status=403 attempts=1'),
                  RuntimeError('model failure')]
        for error in errors:
            with self.subTest(error=str(error)), self.assertRaises(type(error)):
                self.run_main_fixture([error])

    def test_wrong_gpu_prevents_checkout_network_and_publication(self):
        with patch.object(self.module.subprocess, 'check_output', return_value='NVIDIA A100-SXM4-40GB\n'), \
             patch.object(self.module, 'checkout_source') as check, \
             patch.object(self.module.urllib.request, 'urlopen') as http, \
             patch.object(self.module, 'finish_pending') as finish, \
             patch('sys.stdout', io.StringIO()), self.assertRaises(ValueError):
            self.module.main({}, base=self.base)
        check.assert_not_called()
        http.assert_not_called()
        finish.assert_not_called()

    def test_original_view_helper_digest_remains_pinned(self):
        candidates = [Path('/workspace/e3_view_reseed_resume_20261119T151114001132Z/resume_cell.py'),
                      SCIENCE/'runs/e3_view_reseed_resume/20261119T151114001132Z/resume_cell.py']
        helper = next((p for p in candidates if p.is_file()), None)
        if helper is not None:
            self.assertEqual(hashlib.sha256(helper.read_bytes()).hexdigest(), self.module.SEED_SHA256)
        self.assertEqual(self.module.SEED_SHA256,
                         '32a90c1eee50dfb1571dfee51b342923addbe34045a454b6b0871ba0803b5ae8')

    def test_combined_single_file_publishes_with_isolated_exact_algorithm(self):
        combined = load(HERE/'resume_cell.py', 'combined_operational_launcher_tested')
        self.assertIsNot(combined.build_incremental_tree.__globals__, combined.__dict__)
        self.assertIs(combined.build_incremental_tree, combined._e3_tree_module.build_incremental_tree)
        other = self.base/'combined'
        other.mkdir()
        f = fixture(combined, other)
        combined.PROTOCOL = f.protocol_hash
        known = set()
        api = combined.install_api(f.api, TOKEN, known)
        with patch.object(combined, 'emit', lambda value:None), patch('sys.stdout', io.StringIO()):
            result = combined.finish_pending(other, TOKEN, api, C, known)
        self.assertEqual(result['sequence'], 63)
        self.assertIsNone(json.loads(f.statepath.read_bytes())['pending'])
        self.assertFalse(any(method == 'POST' and path.endswith('/git/blobs')
                             for method, path, payload in api.calls))
        self.assertEqual(known, {api._git_sha(raw) for raw in f.files.values()})
        for path, raw in f.original_snapshot.items():
            self.assertEqual((f.snapshot/path).read_bytes(), raw)

    def test_error_observer_redacts_422_but_leaves_rate_limit_body_readable(self):
        url = 'https://api.github.com' + self.module.ROOT_API + '/git/trees'
        for status in (422, 403):
            raw = json.dumps({'message':'validation ' + TOKEN}).encode()
            error = HTTPError(url, status, 'fixture', {}, io.BytesIO(raw))
            original = SimpleNamespace(open=lambda *args, _error=error, **kwargs:(_ for _ in ()).throw(_error))
            capture = self.module.CaptureTreeError(original, TOKEN)
            messages = []
            with patch.object(self.module, 'emit', side_effect=messages.append), self.assertRaises(HTTPError):
                capture.open(Request(url, data=b'{}', method='POST'), timeout=90)
            if status == 422:
                self.assertEqual(len(messages), 1)
                self.assertNotIn(TOKEN, json.dumps(messages))
                self.assertIn('[REDACTED]', messages[0]['message'])
            else:
                self.assertEqual(messages, [])
                self.assertEqual(error.read(), raw)
            error.close()


if __name__ == '__main__':
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    report = {'kind':'e3_incremental_resume_offline_integration_verification',
              'tests_run':result.testsRun, 'failures':len(result.failures), 'errors':len(result.errors),
              'successful':result.wasSuccessful(), 'scientific_source':FROZEN,
              'real_frozen_finish_pending_exercised':True,
              'exact_git_tree_identity_and_preserved_ancestor_modes_verified':True,
              'pending_state_advanced_only_by_frozen_publisher':True,
              'checkpoint_deadline_output_snapshot_bytes_unchanged':True,
              'remote_inventory_verified_before_blob_cache':True,
              'fixed_original_master_and_no_new_campaign_verified':True,
              'HTTP_requests':0, 'real_GPU_fits':0, 'GitHub_mutations':0,
              'limitations':'Synthetic remote Git store and worker; full Colab campaign not executed.',
              'launcher_part_sha256':hashlib.sha256((HERE/'launcher_part.py').read_bytes()).hexdigest(),
              'algorithm_sha256':hashlib.sha256((HERE/'incremental_git_trees.py').read_bytes()).hexdigest(),
              'combined_resume_cell_sha256':hashlib.sha256((HERE/'resume_cell.py').read_bytes()).hexdigest(),
              'details':stream.getvalue()}
    (HERE/'integration_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(stream.getvalue())
    raise SystemExit(0 if result.wasSuccessful() else 1)

