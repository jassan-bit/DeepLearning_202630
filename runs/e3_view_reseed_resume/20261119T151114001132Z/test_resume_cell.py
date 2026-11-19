"""Offline recovery contracts; no HTTP, GPU, unpickling or remote writes.

Set E3_SCIENTIFIC_CHECKOUT when the pinned science checkout is elsewhere.
E3_ORIGINAL_RESUME_WRAPPER optionally supplies the earlier published wrapper.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import builtins
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SOURCE = '5a07d6157a9796ca54416dc5dea1e1527e731597'
SCOPE = Path('experiments/e3_evaluacion_ampliada_20261119/orchestration')


def science_checkout():
    explicit = os.environ.get('E3_SCIENTIFIC_CHECKOUT')
    candidates = ([Path(explicit)] if explicit else [*HERE.parents,
        Path('/content/e3-extended-source-'+SOURCE), Path('/workspace/e3_review9')])
    for candidate in candidates:
        if (candidate/SCOPE/'continuation.py').is_file():
            return candidate
    raise RuntimeError('Set E3_SCIENTIFIC_CHECKOUT to the pinned scientific checkout')


SCIENCE = science_checkout()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


R = load(HERE/'resume_cell.py', 'view_recovery_under_test')
CONTROLLER = load(SCIENCE/SCOPE/'controller.py', 'frozen_controller_for_view_recovery')
API = load(SCIENCE/'experiments/e3_remediacion_20261118/evaluation/colab_github.py',
           'frozen_storage_API_for_view_recovery')
CONTINUATION_BYTES = (SCIENCE/SCOPE/'continuation.py').read_bytes()
TOKEN = 'offline_fixture_secret_never_in_evidence'
PROTOCOL_BYTES = b'{"kind":"synthetic_frozen_protocol","GPU":"A10080"}\n'
PROTOCOL_HASH = hashlib.sha256(PROTOCOL_BYTES).hexdigest()
WRAPPER_FIXTURE = b'# synthetic offline original-wrapper outcome\n'


def metadata(raw):
    return {'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest(),
            'git_blob_sha':hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()}


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.hash_patch = patch.object(R, 'PROTOCOL', PROTOCOL_HASH)
        self.hash_patch.start()
        self.output = self.base/('e3-extended-audit-'+R.MASTER)
        self.view = self.base/('e3-extended-persist-view-'+R.MASTER)
        self.statepath = self.base/('e3-extended-continuation-'+R.MASTER+'.json')
        self.expected_file = (self.base/('e3-extended-source-'+SOURCE)/SCOPE/'continuation.py')
        self.C = ModuleType('actual_frozen_continuation_for_recovery')
        self.C.__file__ = str(SCIENCE/SCOPE/'continuation.py')
        builtins.exec(compile(CONTINUATION_BYTES, str(self.expected_file), 'exec'), self.C.__dict__)
        self.api = SimpleNamespace(REPOSITORY=R.SOURCE and 'jassan-bit/DeepLearning_202630',
            StorageError=API.StorageError, _token=API._token, _canonical=API._canonical,
            _safe_relative=API._safe_relative, _git_sha=API._git_sha,
            _regular_path=API._regular_path, validate_credentials=lambda token:None,
            _request=lambda *a,**kw:self.fail('No HTTP requests are permitted'))

    def tearDown(self):
        self.hash_patch.stop()
        self.temp.cleanup()

    def write(self, path, raw):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw if isinstance(raw,bytes) else
                         (json.dumps(raw,sort_keys=True)+'\n').encode())
        return path

    def fixture(self):
        files = {
            'protocol.json':PROTOCOL_BYTES,
            'resources.csv':b'model,seconds\nknown,12\n',
            'controller_status.json':b'{"stage_status":"failed","analysis_complete":false}\n',
            'jobs/D01/completed/result.json':b'{"status":"completed"}\n',
            'jobs/D01/completed/selected_model.pt':b'original completed model weights',
            'jobs/D02/TimesNet__hpo_lr1/fit_request.json':b'{"seed":20261117}\n',
            'jobs/D02/TimesNet__hpo_lr1/supervised/phase_started.json':
                b'{"deadline":123456789.0,"budget_seconds":120}\n',
            'jobs/D02/TimesNet__hpo_lr1/supervised/checkpoint.pt':b'partial checkpoint exact bytes',
        }
        for name,raw in files.items():
            self.write(self.output/name,raw)
        state = {'kind':'e3_audit_continuation_state','schema_version':1,
            'repository':'jassan-bit/DeepLearning_202630','source_commit':R.SOURCE,
            'master_run_id':R.MASTER,'protocol_sha256':PROTOCOL_HASH,
            'head_commit':'b'*40,'head_part':'20261119T061446338127Z',
            'head_receipt_sha256':'c'*64,'sequence':50,
            'files':{name:metadata(raw) for name,raw in files.items()},
            'counted_jobs':['jobs/D02/TimesNet__hpo_lr1'],'pending':None}
        self.write(self.statepath,state)
        return state

    def seed(self, **kw):
        with redirect_stdout(io.StringIO()):
            return R.seed_view(self.base, **kw)

    def save(self):
        return self.C.save_changed(self.view,TOKEN,R.SOURCE,R.MASTER,PROTOCOL_HASH,
                                   self.api,statefile=self.statepath)

    def invoke(self, action, gpu='NVIDIA A100-SXM4-80GB', wrapper=WRAPPER_FIXTURE):
        calls=[]
        def execute(code,namespace):
            self.assertEqual(code.co_filename,'e3_original_80GB_resume')
            calls.append(namespace)
            return action(len(calls),namespace)
        with patch.object(R.subprocess,'check_output',return_value=gpu), \
             patch.object(R.urllib.request,'urlopen',return_value=io.BytesIO(wrapper)) as opener, \
             patch.object(R,'OLD_WRAPPER_SHA256',hashlib.sha256(WRAPPER_FIXTURE).hexdigest()), \
             patch.object(builtins,'exec',side_effect=execute), redirect_stdout(io.StringIO()):
            R.main({},base=self.base)
        return calls,opener

    def test_actual_frozen_missing_invariant_then_seed_and_save_succeeds(self):
        state=self.fixture()
        CONTROLLER.stable_view(self.output,self.view)
        self.assertFalse((self.view/'resources.csv').exists())
        self.assertFalse((self.view/'jobs/D02/TimesNet__hpo_lr1/supervised/checkpoint.pt').exists())
        with self.assertRaisesRegex(API.StorageError,'continuation_previously_saved_file_removed'):
            self.save()
        report=self.seed()
        self.assertGreater(report['copied_files'],0)
        self.assertTrue(set(state['files']).issubset(self.C._scan(self.view,TOKEN,self.api,self.statepath)))
        saved=self.save()
        self.assertEqual(saved['changed_files'],0)
        self.assertEqual(saved['saved_parts'],[])

    def test_completed_weights_deadlines_output_and_state_are_unchanged(self):
        state=self.fixture()
        before={name:(self.output/name).read_bytes() for name in state['files']}
        state_raw=self.statepath.read_bytes()
        self.seed()
        self.assertEqual(state_raw,self.statepath.read_bytes())
        self.assertEqual(before,{name:(self.output/name).read_bytes() for name in before})
        self.assertEqual(before,{name:(self.view/name).read_bytes() for name in before})

    def test_recovery_is_idempotent(self):
        self.fixture()
        first=self.seed();second=self.seed()
        self.assertEqual(first['copied_files'],8)
        self.assertEqual(second['copied_files'],0)
        self.assertFalse(list(self.view.rglob('.partial-*')))

    def test_existing_modified_view_status_is_not_overwritten(self):
        self.fixture()
        changed=b'{"stage_status":"running","analysis_complete":false}\n'
        self.write(self.view/'controller_status.json',changed)
        self.seed()
        self.assertEqual((self.view/'controller_status.json').read_bytes(),changed)
        self.assertNotEqual((self.output/'controller_status.json').read_bytes(),changed)

    def test_size_sha256_and_git_sha_mismatches_abort_before_copying(self):
        for field,bad in [('bytes',999),('sha256','0'*64),('git_blob_sha','0'*40)]:
            with self.subTest(field=field):
                state=self.fixture()
                state['files']['resources.csv'][field]=bad
                self.write(self.statepath,state)
                with self.assertRaises(ValueError):self.seed()
                self.assertFalse(self.view.exists())

    def test_missing_output_file_aborts_before_any_copy(self):
        self.fixture()
        (self.output/'resources.csv').unlink()
        with self.assertRaisesRegex(ValueError,'also missing'):self.seed()
        self.assertFalse(self.view.exists())

    def test_unsafe_paths_and_metadata_are_rejected(self):
        for name in ['../escape.pt','/tmp/escape.pt','jobs\\escape.pt','jobs/../escape.pt',
                     'jobs/.partial-checkpoint.pt','jobs/checkpoint.pt.tmp']:
            with self.subTest(name=name):
                state=self.fixture()
                state['files'][name]=metadata(b'unsafe')
                self.write(self.statepath,state)
                with self.assertRaises(ValueError):self.seed()
                self.assertFalse(self.view.exists())
        state=self.fixture();state['files']['resources.csv']['bytes']=True
        self.write(self.statepath,state)
        with self.assertRaises(ValueError):self.seed()

    def test_source_and_destination_symlinks_are_rejected(self):
        self.fixture()
        source=self.output/'resources.csv';source.unlink()
        elsewhere=self.write(self.base/'elsewhere.csv',b'not original')
        source.symlink_to(elsewhere)
        with self.assertRaisesRegex(ValueError,'symlinks'):self.seed()
        source.unlink();self.fixture()
        self.view.symlink_to(self.output,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'symlinks'):self.seed()

    def test_other_campaign_protocol_and_pending_are_refused(self):
        for field,value in [('source_commit','a'*40),('master_run_id','20261119T000000000000Z'),
                            ('protocol_sha256','0'*64),('repository','other/repo'),
                            ('kind','other'),('schema_version',2)]:
            with self.subTest(field=field):
                state=self.fixture();state[field]=value;self.write(self.statepath,state)
                with self.assertRaises(ValueError):self.seed()
                self.assertFalse(self.view.exists())
        state=self.fixture();state['pending']={'snapshot':'retained'};self.write(self.statepath,state)
        self.assertEqual(self.seed()['status'],'awaiting_pending_reconciliation')
        with self.assertRaisesRegex(ValueError,'Pending'):self.seed(required=True)
        self.assertFalse(self.view.exists())

    def test_changed_protocol_bytes_are_rejected(self):
        self.fixture();self.write(self.output/'protocol.json',b'changed')
        with self.assertRaisesRegex(ValueError,'protocol bytes differ'):self.seed()
        self.assertFalse(self.view.exists())

    def test_absent_restore_can_be_deferred_but_not_faked(self):
        self.assertEqual(self.seed()['status'],'awaiting_original_restore')
        with self.assertRaisesRegex(ValueError,'finish restoring'):self.seed(required=True)

    def test_wrong_gpu_stops_before_HTTP_or_local_repair(self):
        with patch.object(R.subprocess,'check_output',return_value='NVIDIA A100-SXM4-40GB'), \
             patch.object(R.urllib.request,'urlopen') as opener, redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError,'80 GB'):R.main({},base=self.base)
        opener.assert_not_called();self.assertFalse(self.output.exists())

    def test_wrong_original_wrapper_hash_is_refused(self):
        with self.assertRaisesRegex(ValueError,'wrapper hash differs'):
            self.invoke(lambda *a:self.fail('Unverified code must not run'),wrapper=b'changed')

    def test_fresh_restore_missing_view_is_repaired_and_retried_once(self):
        def action(number,namespace):
            if number==1:
                self.fixture()
                CONTROLLER.stable_view(self.output,self.view)
                return self.save()
            result=self.save()
            self.assertEqual(result['changed_files'],0)
        calls,_=self.invoke(action)
        self.assertEqual(len(calls),2)
        self.assertGreater(calls[-1]['_E3_VIEW_RECOVERY_REPORT']['copied_files'],0)

    def test_auth_model_and_interrupt_errors_are_never_retried(self):
        for error in [API.StorageError('github_transport_auth_or_contents_write_required'),
                      RuntimeError('model failure'),KeyboardInterrupt()]:
            with self.subTest(type=type(error).__name__):
                count=[]
                def action(*args):count.append(1);raise error
                with self.assertRaises(type(error)):self.invoke(action)
                self.assertEqual(len(count),1)

    def test_same_error_text_from_wrong_frame_is_not_repaired(self):
        count=[]
        def action(*args):
            count.append(1)
            raise API.StorageError('continuation_previously_saved_file_removed')
        with self.assertRaises(API.StorageError):self.invoke(action)
        self.assertEqual(len(count),1)

    def generated_error(self, **overrides):
        source='def save_changed(source,master_run_id,protocol_sha256,output):\n'
        source+=' raise StorageError("continuation_previously_saved_file_removed")\n'
        context={'StorageError':API.StorageError}
        builtins.exec(compile(source,str(self.expected_file),'exec'),context)
        arguments={'source':R.SOURCE,'master_run_id':R.MASTER,'protocol_sha256':PROTOCOL_HASH,
                   'output':self.view}|overrides
        try:context['save_changed'](**arguments)
        except Exception as error:return error
        self.fail('Expected generated error')

    def test_expected_frame_requires_correct_source_master_protocol_and_view(self):
        self.assertTrue(R._expected_view_error(self.generated_error(),self.base))
        for arguments in [{'source':'a'*40},{'master_run_id':'other'},
                          {'protocol_sha256':'0'*64},{'output':self.output},
                          {'output':None},{'output':123}]:
            with self.subTest(arguments=arguments):
                self.assertFalse(R._expected_view_error(self.generated_error(**arguments),self.base))

    def test_persistent_expected_error_is_bounded_to_one_repair(self):
        # Generate both errors before patching exec; source frames remain exact.
        first=self.generated_error();second=self.generated_error();count=[]
        def action(number,namespace):
            count.append(1)
            if number==1:self.fixture();raise first
            raise second
        with self.assertRaises(API.StorageError):self.invoke(action)
        self.assertEqual(len(count),2)

    def test_original_published_wrapper_digest_and_master_bindings(self):
        explicit=os.environ.get('E3_ORIGINAL_RESUME_WRAPPER')
        candidates=([Path(explicit)] if explicit else [
            SCIENCE/'runs/e3_tree_read_resume/20261119T061446338127Z/resume_cell.py',
            Path('/workspace/e3_tree_read_resume_20261119T061446338127Z/resume_cell.py')])
        path=next((p for p in candidates if p.is_file()),None)
        if path is None:self.skipTest('Set E3_ORIGINAL_RESUME_WRAPPER for the optional immutable-wrapper check')
        raw=path.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),R.OLD_WRAPPER_SHA256)
        self.assertIn(('E3_EXTENDED_MASTER_RUN = "'+R.MASTER+'"').encode(),raw)
        self.assertIn(b'E3_EXTENDED_START_NEW = False',raw)


if __name__=='__main__':
    stream=io.StringIO()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryTests)
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    report={'kind':'e3_restored_persistence_view_offline_verification',
        'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'skipped':len(result.skipped),'successful':result.wasSuccessful(),
        'resume_cell_sha256':hashlib.sha256((HERE/'resume_cell.py').read_bytes()).hexdigest(),
        'scientific_source_commit':SOURCE,'master_run_id':R.MASTER,
        'real_frozen_stable_view_and_save_changed_exercised':True,
        'HTTP_requests':0,'GPU_fits':0,'GitHub_mutations':0,
        'limitations':'Synthetic byte fixtures and protocol digest; no real Colab restoration, GPU run or external HTTP.',
        'details':stream.getvalue()}
    (HERE/'resume_verification.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(stream.getvalue())
    sys.exit(0 if result.wasSuccessful() else 1)
