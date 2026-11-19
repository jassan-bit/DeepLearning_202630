"""Offline orchestration tests: no subprocess, fit, prediction or remote write."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('d00_resume_hook_test', HERE / 'resume_part.py')
HOOK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HOOK)
SOURCE = Path('/workspace/e3_review9')
WORKER = (HERE / 'worker_compat.py').read_bytes()


class HookTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.output = self.base / ('e3-extended-audit-' + HOOK.MASTER)
        self.output.mkdir()
        self.protocol = self.output / 'protocol.json'
        self.protocol.write_bytes((HERE / 'fixtures/protocol.json').read_bytes())
        self.frozen_run = SOURCE / HOOK.SCOPE / 'evaluation/run.py'
        self.command = [Path('/isolated/python3.12'), self.frozen_run,
                        '--output', self.output, '--source-commit', HOOK.SOURCE,
                        '--admission', self.output / 'admission/admission.json',
                        '--protocol', self.protocol, '--freeze-receipt', self.output / 'freeze_receipt.json',
                        '--npz', self.base / 'arrays.npz', '--btc-history',
                        self.base / 'daily-dev.csv', self.base / 'daily-eval.csv',
                        '--device', 'cuda:0', '--package-path', self.base / 'package']
        self.env = {'UNCHANGED_ENV': 'preserve', 'NOT_A_REAL_TOKEN': 'private-test-placeholder'}
        self.log = self.output / 'worker.log'
        self.calls, self.events = [], []

    def add_result(self, dataset, name='model'):
        result = self.output / 'jobs' / dataset / name / 'result.json'
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text(json.dumps({'status': 'completed', 'dataset': dataset}) + '\n')
        return result

    def instrument(self, *, preflight_error=None, persist_error=None):
        def stream(command, cwd, env, log, on_job=None):
            self.calls.append((command, cwd, env, log, on_job))
            if '--preflight-only' in command:
                self.events.append('preflight')
                if preflight_error:
                    raise preflight_error
                proof = self.output / 'parser_compatibility/d00_preflight.json'
                proof.write_text('{"status":"passed_data_only"}\n')
                return 'preflight-passed'
            self.events.append('worker')
            return 'worker-completed'

        def persist():
            self.events.append('persist')
            folder = self.output / 'parser_compatibility'
            self.assertTrue((folder / 'amendment_v1.json').is_file())
            self.assertEqual((folder / 'worker_source_v1.txt').read_bytes(), WORKER)
            self.assertTrue((folder / 'd00_preflight.json').is_file())
            if persist_error:
                raise persist_error

        controller = types.SimpleNamespace(stream=stream)
        HOOK.install_worker_compatibility(controller, SOURCE, self.base, WORKER)
        return controller, persist

    def invoke(self, controller, persist):
        with contextlib.redirect_stdout(io.StringIO()):
            return controller.stream(self.command, SOURCE, self.env, self.log, persist)

    def test_exact_worker_forwarding_and_durable_order(self):
        self.add_result('D01')
        controller, persist = self.instrument()
        self.assertEqual(self.invoke(controller, persist), 'worker-completed')
        self.assertEqual(self.events, ['preflight', 'persist', 'worker'])
        self.assertEqual(len(self.calls), 2)
        pre, worker = self.calls
        self.assertIs(pre[0][0], self.command[0])
        self.assertIs(worker[0][0], self.command[0])
        expected_path = self.base / ('e3-d00-parser-worker-' + hashlib.sha256(WORKER).hexdigest() + '.py')
        self.assertEqual(worker[0][1:4], [expected_path, '--frozen-run', self.frozen_run])
        self.assertEqual(worker[0][4:], [str(x) for x in self.command[2:]])
        self.assertEqual(pre[0], [*worker[0][:4], '--preflight-only', *worker[0][4:]])
        for call in self.calls:
            self.assertIs(call[1], SOURCE)
            self.assertIs(call[2], self.env)
        self.assertIsNone(pre[4])
        self.assertIs(worker[4], persist)
        self.assertIs(worker[3], self.log)

    def test_nonworker_commands_pass_through_untouched(self):
        controller, persist = self.instrument()
        commands = [[self.command[0], '-m', 'venv', self.base / 'venv'],
                    [self.command[0], SOURCE / HOOK.SCOPE / 'robustness/analyze.py', '--output', self.output],
                    [self.command[0], self.base / 'other/evaluation/run.py']]
        for command in commands:
            with self.subTest(command=command):
                result = controller.stream(command, SOURCE, self.env, self.log, persist)
                self.assertEqual(result, 'worker-completed')
                self.assertIs(self.calls[-1][0], command)
                self.assertIs(self.calls[-1][2], self.env)
                self.assertIs(self.calls[-1][4], persist)
        self.assertFalse((self.output / 'parser_compatibility').exists())
        self.assertEqual(self.events, ['worker'] * len(commands))

    def test_failed_preflight_prevents_persist_and_worker(self):
        controller, persist = self.instrument(preflight_error=RuntimeError('preflight stopped'))
        with self.assertRaisesRegex(RuntimeError, 'preflight stopped'):
            self.invoke(controller, persist)
        self.assertEqual(self.events, ['preflight'])
        self.assertEqual(len(self.calls), 1)

    def test_failed_persistence_prevents_worker(self):
        controller, persist = self.instrument(persist_error=RuntimeError('publisher stopped'))
        with self.assertRaisesRegex(RuntimeError, 'publisher stopped'):
            self.invoke(controller, persist)
        self.assertEqual(self.events, ['preflight', 'persist'])
        self.assertEqual(len(self.calls), 1)

    def test_amendment_idempotence_growth_and_tamper_rejection(self):
        self.add_result('D01')
        controller, persist = self.instrument()
        self.invoke(controller, persist)
        folder = self.output / 'parser_compatibility'
        receipt_bytes = (folder / 'amendment_v1.json').read_bytes()
        saved = json.loads(receipt_bytes)
        self.assertEqual(saved['completed_jobs_at_first_declaration'], 1)
        self.assertEqual(saved['D00_completed_jobs_at_first_declaration'], 0)
        self.add_result('D00', 'newly_completed_under_amendment')
        self.invoke(controller, persist)
        self.assertEqual((folder / 'amendment_v1.json').read_bytes(), receipt_bytes)
        source_file = folder / 'worker_source_v1.txt'
        source_file.write_bytes(WORKER + b'\n# tampered\n')
        count_before = len(self.calls)
        with self.assertRaisesRegex(ValueError, 'Existing parser amendment file differs'):
            self.invoke(controller, persist)
        self.assertEqual(len(self.calls), count_before)

    def test_existing_jobs_binding_and_credentials_preserved(self):
        for dataset in ('D01', 'D02', 'D03'):
            result = self.add_result(dataset)
            (result.parent / 'selected_model.pt').write_bytes(b'unchanged-selected-weights-' + dataset.encode())
        original_jobs = {p.relative_to(self.output).as_posix(): p.read_bytes()
                         for p in (self.output / 'jobs').rglob('*') if p.is_file()}
        original_run = self.frozen_run.read_bytes()
        controller, persist = self.instrument()
        self.invoke(controller, persist)
        self.assertEqual({p.relative_to(self.output).as_posix(): p.read_bytes()
                          for p in (self.output / 'jobs').rglob('*') if p.is_file()}, original_jobs)
        self.assertEqual(self.frozen_run.read_bytes(), original_run)
        folder = self.output / 'parser_compatibility'
        receipt = json.loads((folder / 'amendment_v1.json').read_bytes())
        self.assertEqual(receipt['worker_source_sha256'], hashlib.sha256(WORKER).hexdigest())
        self.assertEqual(receipt['frozen_evaluation_run_sha256'], hashlib.sha256(original_run).hexdigest())
        self.assertEqual(receipt['completed_jobs_at_first_declaration'], 3)
        self.assertTrue(receipt['effective_executable_amended'])
        self.assertTrue(receipt['AI_assistance_declared'])
        self.assertFalse(receipt['original_job_request_identity_changed'])
        for path in folder.rglob('*'):
            if path.is_file():
                self.assertNotIn(b'private-test-placeholder', path.read_bytes())
        self.assertEqual(self.env['NOT_A_REAL_TOKEN'], 'private-test-placeholder')


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(HookTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    verification = {
        'kind': 'e3_D00_parser_resume_hook_verification',
        'status': 'passed' if result.wasSuccessful() else 'failed',
        'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'scope': ['exact worker interception and argument/environment forwarding',
                  'preflight then durable persistence then worker', 'nonworker passthrough',
                  'preflight failure prevents fitting', 'persistence failure prevents fitting',
                  'immutable source/amendment idempotence and tamper rejection',
                  'existing job bytes/source binding and credentials unchanged'],
        'resume_part_sha256': hashlib.sha256((HERE / 'resume_part.py').read_bytes()).hexdigest(),
        'worker_sha256': hashlib.sha256(WORKER).hexdigest(),
        'new_real_data_fits': 0, 'model_predictions': 0,
        'remote_mutations': 0, 'subprocesses_started': 0,
    }
    (HERE / 'hook_verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
