"""Offline contracts for the separate-cohort operational launcher.

No HTTP, CUDA runtime, model optimization or GitHub mutation is performed.
The real frozen controller is executed only to its unknown-master guard.
"""
from __future__ import annotations

import builtins
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SOURCE = '5a07d6157a9796ca54416dc5dea1e1527e731597'
SCOPE = Path('experiments/e3_evaluacion_ampliada_20261119/orchestration')


def scientific_checkout():
    explicit = os.environ.get('E3_SCIENTIFIC_CHECKOUT')
    candidates = ([Path(explicit)] if explicit else []) + list(HERE.parents) + [
        Path('/content')/('e3-extended-source-'+SOURCE), Path('/workspace/e3_review9')]
    for candidate in candidates:
        if all((candidate/SCOPE/name).is_file() for name in ('launch_colab.py', 'controller.py')):
            return candidate
        if explicit:
            raise RuntimeError('E3_SCIENTIFIC_CHECKOUT lacks the scientific orchestration files')
    raise RuntimeError('Set E3_SCIENTIFIC_CHECKOUT to the checkout of scientific commit '+SOURCE)


SCIENCE = scientific_checkout()
ORCHESTRATION = SCIENCE/'experiments/e3_evaluacion_ampliada_20261119/orchestration'
ORIGINAL_LAUNCHER = (ORCHESTRATION/'launch_colab.py').read_bytes()
ORIGINAL_CONTROLLER = (ORCHESTRATION/'controller.py').read_bytes()
spec = importlib.util.spec_from_file_location('reviewed_a10040_launcher', HERE/'launch_cell.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class StorageError(Exception):
    pass


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.flags = []
        self.contexts = []

    def tearDown(self):
        self.temp.cleanup()

    def controller(self, *, rows=None):
        """Compile actual scientific bytes at their exact expected checkout path."""
        path = (self.base/('e3-extended-source-'+launcher.SOURCE)
                /launcher.SCOPE/'controller.py')
        module = ModuleType('offline_frozen_controller')
        module.__file__ = str(path)
        builtins.exec(compile(ORIGINAL_CONTROLLER, str(path), 'exec'), module.__dict__)
        api = SimpleNamespace(_token=lambda value:value,
                              validate_credentials=lambda token:None)
        continuation = SimpleNamespace(discover_runs=lambda *args:rows or [])

        def load(path, name):
            file = Path(path).name
            if file == 'colab_github.py': return api
            if file == 'github_transport.py': return SimpleNamespace(wrap=lambda api:api)
            if file == 'continuation.py': return continuation
            raise AssertionError('Must stop before admission/runtime/model modules: '+file)

        module.checkout = lambda source, base:path.parents[3]
        module.load = load
        return module

    def original_missing_error(self, **arguments):
        controller = self.controller()
        try:
            controller.run(arguments.get('source_commit', launcher.SOURCE), 'offline-fixture-token',
                           base=self.base, start_new=False,
                           master_run_id=arguments.get('master_run_id', launcher.MASTER))
        except Exception as error:
            return error
        self.fail('Expected actual frozen-controller unknown-master guard')

    def fake_exec(self, action):
        def execute(code, context):
            self.assertEqual(code.co_filename, 'e3_a10040_original_launcher')
            start = context['E3_EXTENDED_START_NEW']
            self.flags.append(start)
            self.contexts.append(dict(context))
            # Mirror original launch_colab.main: consume START_NEW before controller.run.
            context['E3_EXTENDED_START_NEW'] = False
            result = action(len(self.flags), start, context)
            if result is not None:
                context['_E3_EXTENDED_AUDIT_RESULT'] = result
        return execute

    def launch(self, action, namespace=None, gpu=None, payload=None):
        namespace = {} if namespace is None else namespace
        with patch.object(launcher.subprocess, 'check_output', return_value=gpu or launcher.GPU), \
             patch.object(launcher.urllib.request, 'urlopen', return_value=io.BytesIO(
                 ORIGINAL_LAUNCHER if payload is None else payload)) as network, \
             patch.object(launcher.time, 'sleep') as sleep, \
             patch.object(launcher, 'exec', self.fake_exec(action), create=True):
            result = launcher.main(namespace, base=self.base)
        return result, namespace, network.call_count, sleep.call_args_list

    def test_scientific_launcher_digest_matches_frozen_local_bytes(self):
        self.assertEqual(hashlib.sha256(ORIGINAL_LAUNCHER).hexdigest(), launcher.LAUNCHER_SHA256)

    def test_wrong_gpu_stops_before_network_or_output_creation(self):
        with patch.object(launcher.subprocess, 'check_output', return_value='NVIDIA A100-SXM4-80GB'), \
             patch.object(launcher.urllib.request, 'urlopen') as network:
            with self.assertRaisesRegex(ValueError, 'requiere'):
                launcher.main({}, base=self.base)
        network.assert_not_called()
        self.assertFalse((self.base/('e3-extended-audit-'+launcher.MASTER)).exists())

    def test_first_launch_uses_actual_unknown_master_guard_then_single_creation(self):
        def action(number, start, context):
            if number == 1:
                raise self.original_missing_error()
            self.assertIs(start, True)
            return {'status':'running', 'master_run_id':launcher.MASTER}
        result, namespace, requests, sleeps = self.launch(action)
        self.assertEqual(self.flags, [False, True])
        self.assertEqual(requests, 1)
        self.assertEqual(sleeps, [])
        self.assertEqual(namespace['_E3_A10040_AUDIT_RESULT'], result)

    def test_fixed_master_overrides_old_user_globals_and_session(self):
        old = '20261119T031104515520Z'
        old_output = self.base/('e3-extended-audit-'+old)
        old_output.mkdir()
        weights = old_output/'checkpoint.pt'
        weights.write_bytes(b'original-80GB-checkpoint-preserved')
        old_protocol = old_output/'protocol.json'
        old_protocol.write_bytes(b'{"GPU":"NVIDIA A100-SXM4-80GB"}\n')
        (self.base/'e3_extended_audit_session.json').write_text(json.dumps({
            'source_commit':launcher.SOURCE, 'master_run_id':old, 'output':str(old_output)}))
        namespace = {'E3_EXTENDED_MASTER_RUN':old, 'E3_EXTENDED_START_NEW':True,
                     'E3_EXTENDED_SOURCE_COMMIT':'a'*40, '_E3_GITHUB_TOKEN':'offline-fixture-token'}
        def action(number, start, context):
            self.assertEqual(context['E3_EXTENDED_MASTER_RUN'], launcher.MASTER)
            self.assertEqual(context['E3_EXTENDED_SOURCE_COMMIT'], launcher.SOURCE)
            self.assertEqual(context['_E3_GITHUB_TOKEN'], 'offline-fixture-token')
            if number == 1: raise self.original_missing_error()
            return {'status':'running'}
        self.launch(action, namespace)
        self.assertEqual(self.flags, [False, True])
        self.assertEqual(weights.read_bytes(), b'original-80GB-checkpoint-preserved')
        self.assertEqual(old_protocol.read_bytes(), b'{"GPU":"NVIDIA A100-SXM4-80GB"}\n')

    def test_existing_or_completed_campaign_only_resumes(self):
        for status in ('running', 'already_completed'):
            with self.subTest(status=status):
                self.flags.clear()
                result, _, _, _ = self.launch(lambda *args:{'status':status})
                self.assertEqual(result['status'], status)
                self.assertEqual(self.flags, [False])

    def test_nonretryable_model_and_auth_errors_propagate(self):
        for error in (ValueError('model failure'), StorageError('github_transport_auth_or_contents_write_required')):
            with self.subTest(error=str(error)):
                self.flags.clear()
                def action(*args): raise error
                with self.assertRaises(type(error)):
                    self.launch(action)
                self.assertEqual(self.flags, [False])

    def test_identical_missing_message_from_wrong_frame_does_not_create(self):
        def action(*args): raise ValueError(launcher.MISSING)
        with self.assertRaisesRegex(ValueError, 'Requested audit master'):
            self.launch(action)
        self.assertEqual(self.flags, [False])

    def test_real_missing_guard_is_accepted_only_for_expected_master(self):
        self.assertTrue(launcher._missing_master(self.original_missing_error(), self.base))
        for args in ({'source_commit':'b'*40}, {'master_run_id':'20261119T000000000000Z'}):
            with self.subTest(arguments=args):
                self.assertFalse(launcher._missing_master(self.original_missing_error(**args), self.base))

    def synthetic_missing(self, **changes):
        values = dict(source_commit=launcher.SOURCE, master_run_id=launcher.MASTER,
                      requested=launcher.MASTER, start_new=False, previous=None, known_local=False)
        values.update(changes)
        path = self.base/('e3-extended-source-'+launcher.SOURCE)/launcher.SCOPE/'controller.py'
        namespace = {'error_type':ValueError, 'message':launcher.MISSING}
        code = 'def run(source_commit, master_run_id, requested, start_new, previous, known_local):\n    raise error_type(message)\n'
        builtins.exec(compile(code, str(path), 'exec'), namespace)
        try:
            namespace['run'](**values)
        except ValueError as error:
            return error

    def test_every_missing_guard_local_is_enforced(self):
        for changes in ({'source_commit':'b'*40}, {'master_run_id':'20261119T000000000000Z'},
                        {'requested':None}, {'start_new':True}, {'previous':{}}, {'known_local':True}):
            with self.subTest(changes=changes):
                self.assertFalse(launcher._missing_master(self.synthetic_missing(**changes), self.base))

    def test_scope_note_is_immutable_and_declares_known_prior_results(self):
        launcher._write_scope(self.base)
        path = self.base/('e3-extended-audit-'+launcher.MASTER)/'GPU_cohort_reexecution.json'
        original = path.read_bytes()
        launcher._write_scope(self.base)
        self.assertEqual(path.read_bytes(), original)
        note = json.loads(original)
        self.assertEqual(note['previous_80GB_completed_records_at_anchor'], 252)
        self.assertIs(note['previous_results_known'], True)
        self.assertIs(note['cross_GPU_result_pooling_allowed'], False)
        self.assertIs(note['previous_weights_or_results_imported'], False)
        path.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, 'declaration differs'):
            launcher._write_scope(self.base)
        self.assertEqual(path.read_text(), '{"changed":true}')

    def test_scope_note_symlink_is_refused(self):
        output = self.base/('e3-extended-audit-'+launcher.MASTER)
        output.mkdir()
        target = self.base/'target.json'
        target.write_bytes(b'preserved')
        (output/'GPU_cohort_reexecution.json').symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            launcher._write_scope(self.base)
        self.assertEqual(target.read_bytes(), b'preserved')

    def test_scope_note_enters_frozen_stable_view(self):
        launcher._write_scope(self.base)
        controller = self.controller()
        output = self.base/('e3-extended-audit-'+launcher.MASTER)
        view = self.base/'view'
        controller.stable_view(output, view)
        self.assertEqual((view/'GPU_cohort_reexecution.json').read_bytes(),
                         (output/'GPU_cohort_reexecution.json').read_bytes())

    def test_wrong_downloaded_launcher_hash_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'launcher hash differs'):
            self.launch(lambda *args:self.fail('Must not execute modified launcher'), payload=b'changed')
        self.assertEqual(self.flags, [])

    def test_first_creation_network_failure_retries_with_resume_false(self):
        def action(number, start, context):
            if number == 1: raise self.original_missing_error()
            if number == 2: raise StorageError('github_transport_network_read_failed operation=git_blob_read attempts=1')
            return {'status':'running'}
        _, _, _, sleeps = self.launch(action)
        self.assertEqual(self.flags, [False, True, False])
        self.assertEqual([call.args for call in sleeps], [(30,)])

    def test_persistent_network_failure_is_bounded_to_four_attempts(self):
        def action(*args):
            raise StorageError('github_transport_network_read_failed operation=git_blob_read attempts=1')
        with patch.object(launcher.time, 'sleep') as sleeps:
            # launch() supplies its own fake sleep; verify attempt count independently.
            with self.assertRaises(StorageError): self.launch(action)
        self.assertEqual(self.flags, [False]*4)

    def test_second_missing_error_does_not_create_again(self):
        def action(number, start, context):
            if number in (1, 3): raise self.original_missing_error()
            raise StorageError('github_transport_network_read_failed operation=git_blob_read attempts=1')
        with self.assertRaisesRegex(ValueError, 'Requested audit master'):
            self.launch(action)
        self.assertEqual(self.flags, [False, True, False])

    def test_keyboard_interrupt_never_retries(self):
        def action(*args): raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt): self.launch(action)
        self.assertEqual(self.flags, [False])

    def transport_error(self, **changes):
        values = dict(method='GET', path='/repos/jassan-bit/DeepLearning_202630/git/trees/'+'a'*40,
                      status=404, allow_missing=False)
        values.update(changes)
        path = self.base/('e3-extended-source-'+launcher.SOURCE)/launcher.SCOPE/'github_transport.py'
        namespace = {'StorageError':StorageError,
                     'message':'github_transport_http_error operation=git_trees_read status=404 attempts=1'}
        code = 'def _perform(method, path, status, allow_missing):\n    raise StorageError(message)\n'
        builtins.exec(compile(code, str(path), 'exec'), namespace)
        try:
            namespace['_perform'](**values)
        except StorageError as error:
            return error

    def test_tree_and_campaign_ref_reads_can_retry(self):
        paths = ['/repos/jassan-bit/DeepLearning_202630/git/trees/'+'a'*40,
                 '/repos/jassan-bit/DeepLearning_202630/git/trees/'+'a'*40+'?recursive=1',
                 '/repos/jassan-bit/DeepLearning_202630/git/ref/heads/e3-extended-audit-'+launcher.MASTER]
        for path in paths:
            with self.subTest(path=path):
                self.assertEqual(launcher._read_retry(self.transport_error(path=path), self.base), path)

    def test_other_transport_routes_methods_statuses_never_retry_as_404(self):
        cases = [{'method':'POST'}, {'status':403}, {'status':'404'}, {'allow_missing':True},
                 {'path':'/repos/other/repo/git/trees/'+'a'*40},
                 {'path':'/repos/jassan-bit/DeepLearning_202630/git/ref/heads/main'},
                 {'path':'/repos/jassan-bit/DeepLearning_202630/git/blobs/'+'a'*40}]
        for changes in cases:
            with self.subTest(changes=changes):
                self.assertIsNone(launcher._read_retry(self.transport_error(**changes), self.base))


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(LauncherTests)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print(stream.getvalue())
    report = {
        'kind':'e3_a10040_operational_launcher_offline_verification',
        'tests_run':result.testsRun,
        'failures':len(result.failures),
        'errors':len(result.errors),
        'successful':result.wasSuccessful(),
        'scientific_source_commit':launcher.SOURCE,
        'new_master_run_id':launcher.MASTER,
        'launcher_sha256':hashlib.sha256((HERE/'launch_cell.py').read_bytes()).hexdigest(),
        'scientific_launcher_sha256':hashlib.sha256(ORIGINAL_LAUNCHER).hexdigest(),
        'scientific_launcher_hash_matches':hashlib.sha256(ORIGINAL_LAUNCHER).hexdigest()==launcher.LAUNCHER_SHA256,
        'real_controller_unknown_master_guard_exercised':True,
        'HTTP_requests':0,
        'GPU_fits':0,
        'GitHub_mutations':0,
        'limitations':'Synthetic process outcomes; Colab runtime and full GPU campaign are not executed.',
        'details':stream.getvalue(),
    }
    (HERE/'launch_verification.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
