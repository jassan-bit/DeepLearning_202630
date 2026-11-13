"""Integrity regression tests for the complete local GitHub mirror path."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from e2_reduced_github_backup import BackupPause, canonical, git_blob_sha
from e2_reduced_github_state import GitHubState
from e2_verified_local_restore import verified_local_restore
import e2_verified_local_restore as local_restore


class Client:
    def __init__(self):
        self.objects = {}
        self.requests = []
        self.head = 'a' * 40
    def _branch_head(self):
        return self.head
    def _content(self, path, head):
        self.requests.append((path, head))
        raw = self.objects.get(path)
        if isinstance(raw, Exception):
            raise raw
        if raw is None:
            return None
        return {'sha': git_blob_sha(raw), 'encoding': 'base64',
            'content': base64.b64encode(raw).decode()}


class Base(GitHubState):
    def _current(self, head):
        if getattr(self, 'advance_after_first_read', False):
            self.reads = getattr(self, 'reads', 0) + 1
            if self.reads > 1:
                return dict(self.fixture_state, state_id='changed')
        return deepcopy(self.fixture_state)
    def restore(self):
        self.full_restore_called = True
        return {'status': 'original_full_restore'}


class VerifiedMirrorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / 'output'
        self.output.mkdir()
        self.directory = self.output / 'jobs' / 'one'
        self.directory.mkdir(parents=True)
        (self.directory / 'result.json').write_bytes(b'{"status":"completed"}\n')
        (self.directory / 'selected_model.pt').write_bytes(b'actual tensor bytes')
        self.client = Client()
        manifest = {'artifacts': []}
        for name in ['result.json', 'selected_model.pt']:
            raw = (self.directory / name).read_bytes()
            manifest['artifacts'].append({'path': name, 'sha256': hashlib.sha256(raw).hexdigest(),
                'bytes': len(raw), 'storage': 'git_blob'})
        self.raw_manifest = canonical(manifest) + b'\n'
        self.manifest_path = 'runs/one/manifest.json'
        self.client.objects[self.manifest_path] = self.raw_manifest
        self.state = verified_local_restore(Base)(self.output, self.client)
        files, _ = self.state._scan()
        self.state.fixture_state = {'state_id': 'valid', 'files': files,
            'output_path': str(self.output), 'metadata_deltas': [{'path': 'never-download-this.zip'}],
            'result_backups': {'jobs/one': {'manifest_path': self.manifest_path,
                'manifest_sha256': hashlib.sha256(self.raw_manifest).hexdigest()}}}

    def test_complete_cache_checks_tensors_and_never_downloads_deltas(self):
        before = {str(p): p.read_bytes() for p in self.directory.iterdir()}
        result = self.state.restore()
        self.assertEqual(result['status'], 'github_complete_local_mirror_verified')
        self.assertEqual(result['existing_identical_files'], 2)
        self.assertEqual(result['metadata_deltas_replayed'], 0)
        self.assertTrue(self.state._restored)
        self.assertEqual(self.client.requests, [(self.manifest_path, 'a' * 40)])
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.directory.iterdir()})

    def test_verified_manifest_cache_is_content_addressed_and_reused(self):
        self.state.restore()
        self.client.requests.clear()
        result = self.state.restore()
        self.assertEqual(result['status'], 'github_complete_local_mirror_verified')
        self.assertEqual(self.client.requests, [])

    def test_missing_tensor_uses_unchanged_full_restore(self):
        (self.directory / 'selected_model.pt').unlink()
        self.assertEqual(self.state.restore()['status'], 'original_full_restore')
        self.assertFalse(self.state._restored)

    def test_corrupt_tensor_fails_closed_without_overwrite(self):
        p = self.directory / 'selected_model.pt'
        p.write_bytes(b'corrupt')
        with self.assertRaisesRegex(BackupPause, 'local_artifact_differs'):
            self.state.restore()
        self.assertEqual(p.read_bytes(), b'corrupt')
        self.assertFalse(self.state._restored)

    def test_missing_metadata_uses_unchanged_full_restore(self):
        (self.directory / 'result.json').unlink()
        self.assertEqual(self.state.restore()['status'], 'original_full_restore')
        self.assertEqual(self.client.requests, [])

    def test_foreign_metadata_fails_closed(self):
        (self.output / 'foreign.json').write_bytes(b'{}')
        with self.assertRaisesRegex(BackupPause, 'local_control_diverges'):
            self.state.restore()
        self.assertEqual(self.client.requests, [])

    def test_active_reservation_blocks_before_artifact_reads(self):
        row = {'sequence': 0, 'previous_sha256': '0' * 64,
            'kind': 'reserved', 'attempt_id': 'unknown', 'reservation_seconds': 1860}
        row['sha256'] = hashlib.sha256(canonical(row)).hexdigest()
        p = self.output / 'shared_budget' / 'pilot_budget.jsonl'
        p.parent.mkdir()
        p.write_bytes(canonical(row) + b'\n')
        self.state.fixture_state['files'] = self.state._scan()[0]
        with self.assertRaisesRegex(BackupPause, 'unclosed_remote_lease'):
            self.state.restore()
        self.assertEqual(self.client.requests, [])

    def test_changed_remote_control_blocks_admission(self):
        self.state.advance_after_first_read = True
        with self.assertRaisesRegex(BackupPause, 'another_session_advanced'):
            self.state.restore()
        self.assertFalse(self.state._restored)

    def test_failed_or_corrupt_manifest_never_admits_state(self):
        for payload in [BackupPause('github_transport_or_response_failed'), b'wrong']:
            self.client.objects[self.manifest_path] = payload
            with self.assertRaises(BackupPause):
                self.state.restore()
            self.assertFalse(self.state._restored)

    def test_interrupted_manifest_cache_write_is_atomic_and_retryable(self):
        with patch.object(local_restore.os, 'replace', side_effect=OSError('synthetic interrupted cache write')):
            with self.assertRaises(OSError):
                self.state.restore()
        cache = self.output / 'assets' / 'verified_manifest_cache'
        self.assertEqual(list(cache.iterdir()), [])
        self.assertFalse(self.state._restored)
        self.assertEqual(self.state.restore()['status'], 'github_complete_local_mirror_verified')


if __name__ == '__main__':
    unittest.main()
