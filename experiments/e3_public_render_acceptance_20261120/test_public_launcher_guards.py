"""Guard tests only: no network, browser, predictor or fitting is executed."""
import importlib.util
import ast
import base64
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('public_gate', HERE / 'run_public_profile14.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
capture_spec = importlib.util.spec_from_file_location('public_captures', HERE / 'emit_public_captures.py')
capture_module = importlib.util.module_from_spec(capture_spec); capture_spec.loader.exec_module(capture_module)
CANONICAL = Path(os.environ.get('E3_ACCEPTANCE_TEST_SOURCE_ROOT', '/workspace/e3_completion_worktree'))
HISTORICAL = Path(os.environ.get('E3_ACCEPTANCE_TEST_FALLBACK_ROOT', '/workspace/e3_review9'))


class GuardTests(unittest.TestCase):
    def ready_row(self):
        return {'origin_HTTP': 200, 'origin_response_received': True,
                'body': {'status': 'ok', 'source_commit': module.SOURCE,
                         'source_commit_source': 'RENDER_GIT_COMMIT', 'profile_predictors': 14,
                         'original_UI_predictors': 5, 'V3_enabled': False,
                         'profile_sha256': module.INPUTS[module.PROFILE][1]}}

    def test_accepted_source_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for relative in module.INPUTS:
                source = CANONICAL / relative
                if not source.exists():source = HISTORICAL / relative
                p = root / relative; p.parent.mkdir(parents=True, exist_ok=True);shutil.copyfile(source, p)
            completed = subprocess.CompletedProcess([], 0, module.SOURCE + '\n', '')
            with patch.object(module.subprocess, 'run', return_value=completed):
                self.assertEqual(module.verify_source(root)['models'], 14)
                p = root / module.CLIENT;p.write_bytes(b'X' + p.read_bytes()[1:])
                with self.assertRaisesRegex(ValueError, 'reference_hash_mismatch'):
                    module.verify_source(root)

    def test_wrong_git_checkout_rejected(self):
        completed = subprocess.CompletedProcess([], 0, '0' * 40 + '\n', '')
        with patch.object(module.subprocess, 'run', return_value=completed):
            with self.assertRaisesRegex(ValueError, 'immutable_source_checkout_mismatch'):
                module.verify_source(Path('/unused'))

    def test_all_five_client_public_asset_paths_are_pinned(self):
        source = CANONICAL / module.CLIENT5
        if not source.exists():source = HISTORICAL / module.CLIENT5
        tree = ast.parse(source.read_text())
        paths = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.For) or not isinstance(node.target, ast.Tuple):continue
            if [x.id for x in node.target.elts if isinstance(x, ast.Name)] != ['route', 'relative']:continue
            for route, relative in ast.literal_eval(node.iter):
                paths.add(relative)
        self.assertEqual(paths, {'MODEL_CARD.md', 'DATA_STATEMENT.md', 'assets/vendor/flatly-5.3.6.min.css'})
        self.assertTrue(paths.issubset(module.INPUTS))

    def test_expected_public_profile_ready(self):
        with patch.object(module, 'health', return_value=self.ready_row()) as get:
            gate = module.public_gate(Path('/unused'), 1)
            self.assertEqual(gate['status'], 'expected_source_and_profile_ready');self.assertEqual(get.call_count, 1)

    def test_missing_profile14_retains_health_and_blocks(self):
        row = self.ready_row();del row['body']['profile_predictors']
        with patch.object(module, 'health', return_value=row):
            gate = module.public_gate(Path('/unused'), 1)
            self.assertEqual(gate['status'], 'expected_source_wrong_profile')
            self.assertEqual(gate['health']['source_commit'], module.SOURCE)

    def test_wrong_profile_hash_blocks(self):
        row = self.ready_row();row['body']['profile_sha256'] = '0' * 64
        with patch.object(module, 'health', return_value=row):
            self.assertEqual(module.public_gate(Path('/unused'), 1)['status'], 'expected_source_wrong_profile')

    def test_proxy_failure_is_not_origin_failure(self):
        row = {'origin_HTTP': None, 'origin_response_received': False, 'status': 'proxy_block_before_origin'}
        with patch.object(module, 'health', return_value=row) as get:
            gate = module.public_gate(Path('/unused'), 1)
            self.assertEqual(gate['status'], 'source_or_profile_not_ready');self.assertEqual(get.call_count, 1)
            self.assertIsNone(gate['attempts'][0]['origin_HTTP'])

    def test_public_capture_chunks_roundtrip_and_ignore_other_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp);folder = root / 'captures';folder.mkdir()
            raw = bytes(range(256)) * 1024
            (folder / 'fixture.jpg').write_bytes(raw);(root / 'not_a_capture.json').write_text('do not log')
            stream = io.StringIO()
            with patch('sys.argv', ['emit', '--output', temp]), contextlib.redirect_stdout(stream):
                capture_module.main()
            chunks = [json.loads(line.split('=', 1)[1]) for line in stream.getvalue().splitlines()
                      if line.startswith('E3_PUBLIC_CAPTURE_CHUNK=')]
            self.assertTrue(all(len(item['data']) <= 49152 for item in chunks))
            recovered = base64.b64decode(''.join(x['data'] for x in sorted(chunks, key=lambda x: x['index'])), validate=True)
            self.assertEqual(recovered, raw)
            self.assertTrue(all(x['sha256'] == hashlib.sha256(raw).hexdigest() for x in chunks))
            self.assertNotIn('not_a_capture.json', stream.getvalue())

    def test_capture_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp);folder = root / 'captures';folder.mkdir()
            target = root / 'not_a_capture.json';target.write_text('do not log')
            (folder / 'not_public.jpg').symlink_to(target)
            with self.assertRaisesRegex(ValueError, 'symlink rejected'):
                capture_module.capture_paths(root)


if __name__ == '__main__':unittest.main()
