import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('transport', ROOT/'scripts/e2_posthoc_transport_resume.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class TransportTests(unittest.TestCase):
    def fixture(self, root):
        extension = root/('guide-closure-extension-'+m.ORIGINAL_COMMIT)
        source = extension/'scripts/e2_posthoc_fit_reproduction.py'
        source.parent.mkdir(parents=True)
        source.write_bytes(b'original registered bytes')
        registration = extension/m.REGISTRATION
        registration.parent.mkdir(parents=True)
        registration.write_text(json.dumps({'files_sha256': {
            'scripts/e2_posthoc_fit_reproduction.py': m.sha(source.read_bytes())}}))
        args = [str(source), '--orchestrate', '--extension-root', str(extension), '--destination',
            str(root/'e2-guide-closure-posthoc-v1'), '--configuration-commit', m.ORIGINAL_COMMIT,
            '--launcher-elapsed-seconds', '3']
        return extension, source, args

    def test_normal_python_invocations_forward_without_loading_science(self):
        for args in (['-I', '-c', 'print(1)'], ['-m', 'venv', 'new'], ['worker.py', '--worker-spec', 'x']):
            with patch.object(m.os, 'execv', side_effect=SystemExit(0)) as call, patch.object(m, 'install_release_amendment') as transport:
                with self.assertRaises(SystemExit):
                    m.main(args)
            self.assertEqual(call.call_args.args[1], [str(m.REAL_PYTHON), *args])
            transport.assert_not_called()

    def test_original_configuration_and_source_bytes_are_required(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(m, 'WORKSPACE', Path(tmp)):
            extension, source, args = self.fixture(Path(tmp))
            self.assertEqual(m.validate_invocation(args)[0], extension)
            bad = args.copy()
            bad[bad.index('--configuration-commit')+1] = 'b'*40
            with self.assertRaisesRegex(ValueError, 'configuration'):
                m.validate_invocation(bad)
            source.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'bytes changed'):
                m.validate_invocation(args)

    def test_duplicate_identity_arguments_fail_before_transport(self):
        with self.assertRaises(ValueError):
            m.argument(['--configuration-commit', 'a', '--configuration-commit', 'b'], '--configuration-commit')

    def test_transport_is_only_in_memory_global_and_proof_has_no_secret(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            extension = root/'extension'
            backup_path = extension/'scripts/e2_reduced_github_backup.py'
            backup_path.parent.mkdir(parents=True)
            backup_path.write_bytes(b'original backup bytes')
            client = Mock()
            client._ensure_release.return_value = {'id': 123}
            client.persist_result.return_value = {'status': 'backed_up', 'manifest_path': 'runs/proof/manifest.json'}
            module = SimpleNamespace(__file__=str(backup_path), RELEASE_TAG=m.OLD_RELEASE,
                GitHubBackup=Mock(return_value=client))
            token = 'synthetic-private-token'
            registration = {'files_sha256': {'scripts/e2_reduced_github_backup.py': m.sha(backup_path.read_bytes())}}
            with patch.object(m.importlib, 'import_module', return_value=module):
                m.install_release_amendment(extension, root/'output', registration, token)
            self.assertEqual(module.RELEASE_TAG, m.NEW_RELEASE)
            self.assertEqual(backup_path.read_bytes(), b'original backup bytes')
            for path in (root/'output').rglob('*.json'):
                self.assertNotIn(token, path.read_text())

    def test_runpy_arguments_and_setup_budget_elapsed_preserved(self):
        args = ['original.py', '--orchestrate', '--configuration-commit', m.ORIGINAL_COMMIT,
            '--launcher-elapsed-seconds', '3']
        with patch.object(m, 'validate_invocation', return_value=(Path('/x'), Path('/y'), {})), \
             patch.object(m, 'install_release_amendment'), \
             patch.object(m.time, 'monotonic', side_effect=[100., 105.]), \
             patch.object(m.sys, 'stdin', io.StringIO('private\n')), \
             patch.object(m.sys, 'argv', []), patch.object(m.runpy, 'run_path') as run:
            m.main(args)
            self.assertEqual(m.sys.argv[-1], '8.0')
            self.assertEqual(m.sys.argv[3], m.ORIGINAL_COMMIT)
            run.assert_called_once_with('original.py', run_name='__main__')


if __name__ == '__main__':
    unittest.main()
