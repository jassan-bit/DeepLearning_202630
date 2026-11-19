"""Offline pending-tree observer contracts; no HTTP, training or GitHub writes."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('tested_tree422', HERE / 'diagnose_pending_tree.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
TOKEN = 'offline-token-not-for-printing'


def scientific_checkout():
    explicit = os.environ.get('E3_SCIENTIFIC_CHECKOUT')
    candidates = ([Path(explicit)] if explicit else list(HERE.parents)
                  + [Path('/content/e3-extended-source-' + helper.SOURCE), Path('/workspace/e3_review9')])
    for root in candidates:
        if (root / helper.SCOPE / 'continuation.py').is_file():
            return root
    raise RuntimeError('Set E3_SCIENTIFIC_CHECKOUT to the frozen scientific checkout')


science = scientific_checkout()
cspec = importlib.util.spec_from_file_location('offline_tree422_continuation', science / helper.SCOPE / 'continuation.py')
continuation = importlib.util.module_from_spec(cspec)
cspec.loader.exec_module(continuation)


class StorageError(Exception):
    pass


class RejectingOpener:
    def __init__(self, error):
        self.error = error
        self.calls = 0

    def open(self, request, **kwargs):
        self.calls += 1
        raise self.error


class FakeAPI:
    REPOSITORY = helper.REPO
    StorageError = StorageError

    def __init__(self):
        self.posts = []
        self.gets = []
        self._opener = None
        self.existing = None

    @staticmethod
    def _regular_path(path, directory=False):
        path = Path(path).absolute()
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise StorageError('symlink')
        return path

    @staticmethod
    def _canonical(value):
        return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()

    @staticmethod
    def _git_sha(raw):
        return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()

    @staticmethod
    def _safe_relative(name):
        p = Path(name)
        return bool(name) and not p.is_absolute() and '..' not in p.parts and '\\' not in name

    def _request(self, method, path, token, payload=None, allow_missing=False):
        if method == 'POST':
            self.posts.append((path, payload))
            if self._opener.original is not None:
                self._opener.open(Request('https://api.github.com' + path,
                                         data=self._canonical(payload), method='POST'))
            return {'sha': 'd' * 40}
        self.gets.append(path)
        if '/git/commits/' in path:
            return {'sha': path.rsplit('/', 1)[-1], 'tree': {'sha': 'c' * 40}}
        if '/git/ref/heads/' in path:
            return self.existing
        raise AssertionError('Unexpected request: ' + method + ' ' + path)


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.api = FakeAPI()
        self.protocol_raw = b'offline protocol fixture\n'
        self.protocol_hash = hashlib.sha256(self.protocol_raw).hexdigest()
        self.protocol = self.base / ('e3-extended-audit-' + helper.MASTER) / 'protocol.json'
        self.protocol.parent.mkdir(parents=True)
        self.protocol.write_bytes(self.protocol_raw)
        self.part = '20261119T170000000001Z'
        self.snapshot = self.base / ('e3-audit-part-' + self.part + '-fixture')
        payload = self.snapshot / 'payload/jobs/D02/fixture/result.json'
        payload.parent.mkdir(parents=True)
        payload.write_bytes(b'{"status":"completed"}\n')
        self.state = {'kind': 'e3_audit_continuation_state', 'schema_version': 1,
                      'repository': helper.REPO, 'source_commit': helper.SOURCE,
                      'protocol_sha256': self.protocol_hash, 'master_run_id': helper.MASTER,
                      'head_commit': 'a' * 40, 'head_part': '20261119T155220632605Z',
                      'head_receipt_sha256': 'b' * 64, 'sequence': 51,
                      'pending': {'part_id': self.part, 'snapshot': str(self.snapshot)}}
        self.receipt = {'kind': 'e3_closure_stage_receipt', 'profile': 'audit',
                        'source_commit': helper.SOURCE, 'master_run_id': helper.MASTER,
                        'protocol_sha256': self.protocol_hash, 'part_id': self.part, 'sequence': 52,
                        'previous_part': {'commit_sha': 'a' * 40, 'part_id': self.state['head_part'],
                                          'receipt_sha256': 'b' * 64},
                        'files': {'jobs/D02/fixture/result.json': continuation._meta(payload.read_bytes(), self.api)},
                        'new_real_data_fits': 0, 'R1_accredited': False, 'R3_accredited': False}
        self.statepath = self.base / ('e3-extended-continuation-' + helper.MASTER + '.json')
        self.save_fixtures()

    def tearDown(self):
        self.temp.cleanup()

    def save_fixtures(self):
        self.statepath.write_bytes(self.api._canonical(self.state))
        (self.snapshot / 'stage_receipt.json').write_bytes(self.api._canonical(self.receipt))

    def run_main(self):
        def load(path, name):
            if name == 'tree422_legacy':
                return self.api
            if name == 'tree422_transport':
                return SimpleNamespace(wrap=lambda api: api)
            if name == 'tree422_continuation':
                return continuation
            raise AssertionError('Unexpected module load')
        with patch.object(helper, 'PROTOCOL', self.protocol_hash), \
                patch.object(helper, 'token_from', return_value=TOKEN), \
                patch.object(helper, 'load', side_effect=load), \
                patch.object(helper.subprocess, 'check_output', return_value=helper.SOURCE + '\n'), \
                patch.object(helper.subprocess, 'run'), \
                patch('sys.stdout', new_callable=io.StringIO):
            return helper.main({}, base=self.base)

    def error(self, body, code=422, headers=None):
        return HTTPError(helper.TREE_URL, code, 'fixture', headers, io.BytesIO(body))

    def test_422_exact_post_captures_allowlist_redacts_and_closes(self):
        body = json.dumps({'message': TOKEN + ' validation',
                           'errors': [{'field': 'sha', 'code': TOKEN, 'secret': TOKEN}],
                           'untrusted': TOKEN}).encode()
        error = self.error(body, headers={'X-GitHub-Request-Id': TOKEN + '-id'})
        report = {}
        with self.assertRaises(helper.TreeValidationRejected):
            helper.Observe422(RejectingOpener(error), TOKEN, report).open(Request(helper.TREE_URL, method='POST'))
        serialized = json.dumps(report)
        self.assertNotIn(TOKEN, serialized)
        self.assertNotIn('secret', serialized)
        self.assertEqual(report['http'], 422)
        self.assertTrue(error.fp.closed)

    def test_non_json_oversized_and_absent_headers_are_bounded(self):
        for body in (b'not json', b'x' * 70000, b'[]'):
            with self.subTest(size=len(body)):
                report = {}
                with self.assertRaises(helper.TreeValidationRejected):
                    helper.Observe422(RejectingOpener(self.error(body)), TOKEN, report).open(
                        Request(helper.TREE_URL, method='POST'))
                self.assertEqual(report['message'], 'No readable JSON message')
                self.assertEqual(report['errors'], [])

    def test_other_routes_methods_and_statuses_propagate_untouched(self):
        for url, method, status in ((helper.TREE_URL, 'GET', 422),
                                    (helper.TREE_URL + '/x', 'POST', 422),
                                    (helper.TREE_URL, 'POST', 403)):
            error = self.error(b'{}', code=status)
            report = {}
            with self.assertRaises(HTTPError) as caught:
                helper.Observe422(RejectingOpener(error), TOKEN, report).open(Request(url, method=method))
            self.assertIs(caught.exception, error)
            self.assertEqual(report, {})

    def test_accepted_tree_only_posts_exact_pending_and_preserves_files(self):
        before = {p: p.read_bytes() for p in self.base.rglob('*') if p.is_file()}
        result = self.run_main()
        self.assertEqual(result['status'], 'tree_creation_accepted')
        self.assertEqual(len(self.api.posts), 1)
        path, payload = self.api.posts[0]
        self.assertEqual(path, helper.TREE_PATH)
        self.assertEqual(payload['base_tree'], 'c' * 40)
        self.assertEqual(result['entry_count'], 3)
        self.assertTrue(all(e['path'].startswith('runs/e3_extended_audit/' + self.part + '/')
                            for e in payload['tree']))
        self.assertEqual(before, {p: p.read_bytes() for p in self.base.rglob('*') if p.is_file()})

    def test_main_rejected_tree_returns_details_without_state_changes(self):
        before = self.statepath.read_bytes()
        self.api._opener = RejectingOpener(self.error(b'{"message":"invalid sha","errors":[]}'))
        result = self.run_main()
        self.assertEqual(result['status'], 'tree_validation_rejected')
        self.assertEqual(result['message'], 'invalid sha')
        self.assertEqual(before, self.statepath.read_bytes())
        self.assertEqual(len(self.api.posts), 1)

    def test_wrong_source_protocol_or_receipt_stops_before_post(self):
        original_state = dict(self.state)
        original_receipt = dict(self.receipt)
        for kind in ('source', 'protocol', 'receipt'):
            self.state = dict(original_state)
            self.receipt = dict(original_receipt)
            if kind == 'source':
                self.state['source_commit'] = 'e' * 40
            elif kind == 'protocol':
                self.state['protocol_sha256'] = 'f' * 64
            else:
                self.receipt['previous_part'] = {}
            self.save_fixtures()
            with self.subTest(kind=kind), self.assertRaises((ValueError, StorageError)):
                self.run_main()
            self.assertEqual(self.api.posts, [])

    def test_modified_payload_stops_before_post(self):
        (self.snapshot / 'payload/jobs/D02/fixture/result.json').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            self.run_main()
        self.assertEqual(self.api.posts, [])

    def test_no_pending_never_posts(self):
        self.state['pending'] = None
        self.save_fixtures()
        result = self.run_main()
        self.assertEqual(result['status'], 'no_pending_snapshot')
        self.assertEqual(self.api.posts, [])
        self.assertEqual(self.api.gets, [])

    def test_already_saved_part_verified_without_post(self):
        self.api.existing = {'object': {'sha': 'e' * 40}}
        before = self.statepath.read_bytes()
        with patch.object(continuation, '_read_part', return_value={
                'receipt_raw': (self.snapshot / 'stage_receipt.json').read_bytes()}), \
                patch.object(continuation, '_validate_part_files') as validate:
            result = self.run_main()
        self.assertEqual(result['status'], 'pending_part_already_saved')
        self.assertEqual(self.api.posts, [])
        self.assertEqual(before, self.statepath.read_bytes())
        validate.assert_called_once()

    def test_checkout_identity_failure_stops_before_loading_or_post(self):
        with patch.object(helper.subprocess, 'check_output', return_value='e' * 40), \
                patch.object(helper, 'load') as load:
            with self.assertRaises(ValueError):
                helper.main({}, base=self.base)
        load.assert_not_called()
        self.assertEqual(self.api.posts, [])


if __name__ == '__main__':
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(DiagnosticTests))
    report = {'kind': 'e3_pending_tree422_diagnostic_offline_verification',
              'tests_run': result.testsRun, 'successful': result.wasSuccessful(),
              'failures': len(result.failures), 'errors': len(result.errors),
              'real_HTTP_requests': 0, 'real_GPU_fits': 0, 'real_GitHub_mutations': 0,
              'helper_sha256': hashlib.sha256((HERE / 'diagnose_pending_tree.py').read_bytes()).hexdigest(),
              'frozen_snapshot_and_state_validators_exercised': True,
              'limitations': 'Synthetic API outcomes; protocol hash is a local fixture, no real pending save is executed.',
              'details': output.getvalue()}
    (HERE / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(output.getvalue())
    raise SystemExit(0 if result.wasSuccessful() else 1)
