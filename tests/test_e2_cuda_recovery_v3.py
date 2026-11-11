"""External recovery boundaries; synthetic metadata only, no CUDA or network."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import hashlib
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

CONTROLLER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTROLLER / 'scripts'))
import run_e2_cuda_recovery_v3 as recovery


TEST_HEAD='e'*40
TEST_RECEIPT_BRANCH='recoveries/e2-cuda-selection-v1/20261111-r3'
TEST_TARGET={'job_id':'c'*64,'job_payload_sha256':'d'*64,'started_event_sha256':'e'*64,
             'sequence':3124,'old_session_id':'f'*32,'reservation_seconds':2100}
TEST_LIMITS={'session_seconds':28800,'initial_campaign_seconds':86400,
             'campaign_seconds':129600,
             'budget_extension_event_sha256':recovery.BUDGET_EXTENSION_EVENT_SHA256,
             'old_session_conservative_allocation_seconds':28800,
             'budget_extension_authorized':False}


class Clock:
    def __init__(self, value=100):
        self.value = float(value)
    def __call__(self):
        return self.value
    def advance(self, seconds):
        self.value += seconds


class Paused(RuntimeError):
    def __init__(self, reason, **details):
        self.reason, self.details = reason, details


class SyncError(RuntimeError):
    def __init__(self, code):
        self.code = code


class Unresolved(RuntimeError):
    pass


class SelectionAborted(RuntimeError):
    pass


def auth_fixture():
    return {'version': 'e2-manual-recovery-authorization-v3',
        'status': 'registered_before_recovery_execution',
        'repository': recovery.REPOSITORY, 'campaign_branch': recovery.BRANCH,
        'campaign_prefix': recovery.BRANCH, 'receipt_branch': TEST_RECEIPT_BRANCH,
        'scientific_source_commit': recovery.SOURCE,
        'scientific_registration_sha256': recovery.REGISTRATION,
        'plan_sha256': recovery.PLAN, 'sealed_campaign_head': TEST_HEAD,
        'previous_completed_jobs': 1552, 'decision': recovery.DECISION, 'target': deepcopy(TEST_TARGET),
        'limits': deepcopy(TEST_LIMITS), 'original_unknown_attempt': deepcopy(recovery.UNKNOWN),
        'implementation_sha256': {}}


class Snapshot:
    def __init__(self, plan, results):
        self.plan_sha256 = recovery.digest(plan)
        self.results = results
        self.reads = []
    def get_result(self, job_id, payload=None):
        self.reads.append(job_id)
        item = self.results.get(job_id)
        if item is not None and payload is not None and item[0] != recovery.digest(payload):
            raise ValueError('cached_payload_changed')
        return None if item is None else item[1]


class RecoveryRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.clock = Clock()

    def tearDown(self):
        self.temporary.cleanup()

    def args(self, *extra):
        return recovery.parser().parse_args(['--execute', '--engine-root', str(self.base/'engine'),
            '--controller-source-commit', 'a'*40, '--authorization', str(self.base/'auth.json'),
            '--expected-authorization-sha256', 'b'*64, '--retry-lost-once',
            '--output', str(self.base/'cache'), '--data-root', str(self.base/'development'), *extra])

    def invoke(self, args=None):
        output=StringIO()
        with redirect_stdout(output):
            code=recovery.execute(self.args() if args is None else args,clock=self.clock)
        final=json.loads(output.getvalue().splitlines()[-1])
        report=None if final['external_report'] is None else json.loads(Path(final['external_report']).read_text())
        return code,final,report

    def fixture_authorization(self, **changes):
        root = self.base / 'controller'
        value = auth_fixture()
        for relative in recovery.CONTROLLER_FILES:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# synthetic controller fixture\n')
            value['implementation_sha256'][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        value.update(changes)
        path = self.base/'auth.json'
        raw = json.dumps(value, indent=2).encode()
        path.write_bytes(raw)
        return root, path, hashlib.sha256(raw).hexdigest(), value

    def test_inspect_subprocess_has_no_engine_scientific_imports_network_or_credentials(self):
        # The actual CLI is launched with an import guard. An invalid token
        # must be irrelevant, and the nonexistent output must remain absent.
        script = CONTROLLER/'scripts/run_e2_cuda_recovery_v3.py'
        output = self.base/'must-not-exist'
        wrapper = '''import sys,runpy,socket
class Block:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','numpy','pandas','e2_search_session','e2_manual_reconciliation_v3','e2_recovery_receipts_v3','run_e2_cuda_search','e2_cuda_search_plan'}:
   raise AssertionError('scientific import during inspection:'+fullname)
sys.meta_path.insert(0,Block())
def forbidden(*a,**kw):raise AssertionError('network during inspection')
socket.socket=forbidden
sys.argv=[sys.argv[1],'--inspect','--output',sys.argv[2]]
runpy.run_path(sys.argv[0],run_name='__main__')
'''
        environment = dict(os.environ, GITHUB_TOKEN='bogus-never-use', GH_TOKEN='bogus-never-use')
        completed = subprocess.run([sys.executable, '-I', '-c', wrapper, str(script), str(output)],
            capture_output=True, text=True, env=environment, check=True)
        value = json.loads(completed.stdout)
        self.assertEqual(value['status'], 'inspection_only')
        self.assertEqual(value['selection_attempts_upper_bound'], 85074)
        self.assertEqual((value['hpo_attempts'], value['nas1_attempts'], value['nas2_attempts_upper_bound']),
                         (5544, 66000, 13530))
        self.assertFalse(value['recovery_enabled'])
        self.assertFalse(output.exists())
        self.assertNotIn('bogus', completed.stdout + completed.stderr)

    def test_plan_only_accepts_draft_without_import_or_implementation_gate(self):
        _, path, sha, _ = self.fixture_authorization(status='draft_before_recovery_execution')
        args = recovery.parser().parse_args(['--plan-only', '--authorization', str(path),
                                             '--expected-authorization-sha256', sha])
        with patch.object(recovery, 'load_engine', side_effect=AssertionError('engine')), \
                patch.object(recovery, 'git_head', side_effect=AssertionError('git')):
            value = recovery.inspection(args)
        self.assertEqual(value['authorization_sha256'], sha)
        self.assertEqual(value['authorization_status'], 'draft_before_recovery_execution')
        self.assertIsNone(value['original_unknown_attempt']['optimizer_steps'])

    def test_exact_raw_authorization_and_closed_controller_manifest(self):
        root, path, sha, expected = self.fixture_authorization()
        with patch.object(recovery, 'ROOT', root):
            raw, value, actual = recovery.load_authorization(path, sha, execute=True)
            self.assertEqual((value, actual), (expected, sha))
            self.assertEqual(hashlib.sha256(raw).hexdigest(), sha)
            (root / recovery.CONTROLLER_FILES[1]).write_text('# changed\n')
            with self.assertRaisesRegex(ValueError, 'recovery_implementation_hash_mismatch'):
                recovery.load_authorization(path, sha, execute=True)

    def test_draft_empty_hash_or_changed_raw_authorization_rejects(self):
        root, path, sha, _ = self.fixture_authorization(status='draft_before_recovery_execution')
        with patch.object(recovery, 'ROOT', root):
            with self.assertRaisesRegex(ValueError, 'published_recovery_authorization_required'):
                recovery.load_authorization(path, sha, execute=True)
        _, path, sha, _ = self.fixture_authorization(implementation_sha256={})
        with self.assertRaisesRegex(ValueError, 'closed_recovery_implementation_required'):
            recovery.load_authorization(path, sha, execute=True)
        path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError, 'authorization_hash_mismatch'):
            recovery.load_authorization(path, sha, execute=True)

    def test_invalid_incident_budget_and_source_reject_but_valid_incident_is_auth_bound(self):
        for field, change in [('target', dict(TEST_TARGET, job_id='invalid')),
                              ('limits', dict(TEST_LIMITS, session_seconds=36000)),
                              ('scientific_source_commit', 'd'*40),
                              ('original_unknown_attempt', dict(recovery.UNKNOWN, optimizer_steps=0))]:
            with self.subTest(field=field):
                _, path, sha, _ = self.fixture_authorization(**{field: change})
                with self.assertRaisesRegex(ValueError, 'authorization_'):
                    recovery.load_authorization(path, sha)
        _,path,sha,_=self.fixture_authorization(target=dict(TEST_TARGET,job_id='1'*64),
            sealed_campaign_head='2'*40,previous_completed_jobs=704,
            receipt_branch='recoveries/e2-cuda-selection-v1/20261111-r3')
        _,value,actual=recovery.load_authorization(path,sha)
        incident=recovery.AuthBoundIncident(value,actual)
        self.assertEqual(incident.target['job_id'],'1'*64)
        self.assertEqual(incident.previous_completed_jobs,704)
        self.assertEqual(incident.sealed_head,'2'*40)
        self.assertEqual(value['limits']['old_session_conservative_allocation_seconds'],28800)

    def test_duplicate_or_nonfinite_json_rejected(self):
        for raw in (b'{"target":1,"target":2}', b'{"value":NaN}'):
            path = self.base/'bad.json'
            path.write_bytes(raw)
            with self.assertRaises(ValueError):
                recovery.read_json(path)

    def test_recovery_cannot_reset_or_extend_existing_global_budget(self):
        for limits in (
                dict(TEST_LIMITS, initial_campaign_seconds=129600),
                dict(TEST_LIMITS, campaign_seconds=86400),
                dict(TEST_LIMITS, campaign_seconds=172800),
                dict(TEST_LIMITS, budget_extension_authorized=True),
                dict(TEST_LIMITS, budget_extension_event_sha256='0'*64),
                dict(TEST_LIMITS, old_session_conservative_allocation_seconds=28799)):
            with self.subTest(limits=limits):
                _, path, sha, _ = self.fixture_authorization(limits=limits)
                with self.assertRaisesRegex(ValueError, 'authorization_budget_binding_invalid'):
                    recovery.load_authorization(path, sha)
        for field, value in (
                ('version', 'e2-manual-recovery-authorization-v2'),
                ('receipt_branch', 'recoveries/e2-cuda-selection-v1/20261110-r2')):
            with self.subTest(field=field):
                _, path, sha, _ = self.fixture_authorization(**{field:value})
                with self.assertRaisesRegex(ValueError, 'authorization_'):
                    recovery.load_authorization(path, sha)

    def test_inspection_distinguishes_original_controls_from_existing_extension(self):
        _, path, sha, _ = self.fixture_authorization()
        args = recovery.parser().parse_args(['--plan-only', '--authorization', str(path),
            '--expected-authorization-sha256', sha])
        value = recovery.inspection(args)
        self.assertEqual((value['campaign_hours'], value['initial_campaign_hours'],
                          value['effective_campaign_hours']), (24, 24, 36))
        self.assertEqual(value['budget_extension_event_sha256'],
                         recovery.BUDGET_EXTENSION_EVENT_SHA256)
        self.assertFalse(value['budget_extension_authorized'])

    def test_explicit_retry_and_original_operational_caps_required_before_import(self):
        for extra in (['--session-hours','9'], ['--campaign-hours','25'],
                      ['--extend-campaign-hours','24'], ['--storage-gib','11'],
                      ['--device','cpu'], ['--github-branch','campaigns/other']):
            with self.subTest(extra=extra), patch.object(recovery, 'load_engine') as load:
                with self.assertRaisesRegex(ValueError, 'sealed_scientific_controls_required'):
                    recovery.preflight(self.args(*extra))
                load.assert_not_called()
        args = self.args()
        args.retry_lost_once = False
        with self.assertRaisesRegex(ValueError, 'explicit_retry_lost_once_required'):
            recovery.preflight(args)

    def test_separate_engine_and_scientific_module_provenance_are_required(self):
        with self.assertRaisesRegex(ValueError,'separate_scientific_checkout_required'):
            recovery.load_engine(CONTROLLER)
        poisoned=SimpleNamespace(__file__=str(CONTROLLER/'scripts/e2_shadow.py'))
        with patch.dict(sys.modules,{'e2_shadow':poisoned}):
            with self.assertRaisesRegex(ValueError,'scientific_import_provenance_changed'):
                recovery.load_engine(self.base/'engine')

    def fake_preflight_engine(self, hardware_error=None):
        reg = self.base/'registration.json'
        raw = json.dumps({'status':'registered_before_scientific_cuda_fits',
                          'implementation_sha256':{'scripts/frozen.py':'a'*64}}).encode()
        reg.write_bytes(raw)
        engine = SimpleNamespace(
            frozen=SimpleNamespace(verify_source=Mock(), cuda_hardware=Mock(
                side_effect=hardware_error, return_value={'device_name':'Tesla T4'})),
            admission=SimpleNamespace(REGISTRATION=reg,
                source_hashes=Mock(return_value={'scripts/frozen.py':'a'*64}),
                build_plan=Mock(return_value={'sealed':'synthetic'})),
            transport=SimpleNamespace(GitCampaignMirror=Mock()),
            receipts=SimpleNamespace(RecoveryReceiptStore=Mock()))
        return engine, hashlib.sha256(raw).hexdigest()

    def test_no_gpu_or_quota_means_zero_receipt_claims_ledger_appends_or_fits(self):
        engine, registration_sha = self.fake_preflight_engine(RuntimeError('cuda_unavailable'))
        auth = auth_fixture()
        with patch.object(recovery,'load_authorization',return_value=(b'{}',auth,'b'*64)), \
                patch.object(recovery,'git_head',return_value='a'*40), \
                patch.object(recovery,'load_engine',return_value=engine), \
                patch.object(recovery,'REGISTRATION',registration_sha):
            code,_,report=self.invoke()
            self.assertEqual(code,2)
            self.assertEqual((report['error_code'],report['error_phase']),('cuda_unavailable','cuda_preflight'))
        engine.transport.GitCampaignMirror.assert_not_called()
        engine.receipts.RecoveryReceiptStore.assert_not_called()
        self.assertFalse((self.base/'cache').exists())
        engine.admission.build_plan.assert_not_called()

    def test_non_t4_or_changed_plan_blocks_before_remote_mutations(self):
        engine, registration_sha = self.fake_preflight_engine()
        with patch.object(recovery,'load_authorization',return_value=(b'{}',auth_fixture(),'b'*64)), \
                patch.object(recovery,'git_head',return_value='a'*40), \
                patch.object(recovery,'load_engine',return_value=engine), \
                patch.object(recovery,'REGISTRATION',registration_sha):
            engine.frozen.cuda_hardware.return_value = {'device_name':'A100'}
            code,_,report=self.invoke()
            self.assertEqual((code,report['error_code']),(2,'sealed_t4_hardware_required'))
            engine.frozen.cuda_hardware.return_value = {'device_name':'Tesla T4'}
            code,_,report=self.invoke()
            self.assertEqual((code,report['error_code']),(2,'sealed_campaign_plan_mismatch'))
        engine.transport.GitCampaignMirror.assert_not_called()
        engine.receipts.RecoveryReceiptStore.assert_not_called()

    def test_admission_keeps_initial_24_hour_plan_without_new_extension(self):
        engine, registration_sha = self.fake_preflight_engine()
        plan = engine.admission.build_plan.return_value
        with patch.object(recovery,'load_authorization',return_value=(b'{}',auth_fixture(),'b'*64)), \
                patch.object(recovery,'git_head',return_value='a'*40), \
                patch.object(recovery,'load_engine',return_value=engine), \
                patch.object(recovery,'REGISTRATION',registration_sha), \
                patch.object(recovery,'PLAN',recovery.digest(plan)):
            recovery.preflight(self.args())
        self.assertEqual(engine.admission.build_plan.call_args.kwargs['campaign_hours'],24)
        self.assertEqual(engine.admission.build_plan.call_args.kwargs['session_hours'],8)
        engine.transport.GitCampaignMirror.assert_not_called()
        engine.receipts.RecoveryReceiptStore.assert_not_called()

    def replay_fixture(self, completed_target=False):
        plan = {'synthetic_plan':'fixed'}
        payload = {'model':'SAE','seed':271301,'stage':'nas1','candidate_id':'nas-123'}
        job_id = recovery.digest({'plan_sha256':recovery.digest(plan),'job':payload})
        target = dict(TEST_TARGET,job_id=job_id,job_payload_sha256=recovery.digest(payload))
        auth = {'target':target}
        result = {'status':'completed','predictions_original':[1.25,1.75],
                  'prediction_dates':['2024-02-01T00:00:00+00:00','2024-02-02T00:00:00+00:00'],
                  'phase_epochs':{'supervised':2},'resource_use':{'optimizer_steps':12}}
        rows = {str(i):(recovery.digest({'cached':i}),deepcopy(result)) for i in range(1552)}
        if completed_target:
            rows[job_id]=(recovery.digest(payload),result)
        view = Snapshot(plan,rows)
        reconciler = SimpleNamespace(recover=Mock())
        return plan,payload,job_id,auth,result,view,reconciler

    def test_1552_cached_outcomes_not_reexecuted_target_callback_exactly_once(self):
        _,payload,job_id,auth,result,view,reconciler = self.replay_fixture()
        worker = Mock(return_value=result)
        order = []
        def reconcile(identity,job,callback,*,reservation_seconds):
            order.append('started_remote_ack')
            value=callback(self.clock()+2100)
            order.append('observed_remote_ack')
            self.assertIs(value,result)
            return {'status':'reconciled','completed_jobs':1553}
        reconciler.recover.side_effect = reconcile
        facade=recovery.ReplayFacade(view,auth,reconciler)
        for i in range(1552):
            job={'cached':i}
            self.assertIsNotNone(facade.get_result(str(i),job))
            facade.run_job(str(i),job,worker,reservation_seconds=2100)
        worker.assert_not_called()
        self.assertIsNone(facade.get_result(job_id,payload))
        with self.assertRaises(recovery.RecoveryStopped) as stopped:
            facade.run_job(job_id,payload,worker,reservation_seconds=2100)
        worker.assert_called_once_with(2200)
        self.assertEqual(stopped.exception.result['completed_jobs'],1553)
        self.assertEqual(order,['started_remote_ack','observed_remote_ack'])
        with self.assertRaisesRegex(ValueError,'recovery_target_payload_changed'):
            facade.run_job(job_id,payload,worker,reservation_seconds=2100)
        self.assertEqual(worker.call_count,1)

    def test_cached_target_still_captured_for_idempotent_receipt_validation_without_fit(self):
        _,payload,job_id,auth,_,view,reconciler=self.replay_fixture(completed_target=True)
        reconciler.recover.return_value={'status':'already_reconciled','physical_reexecution_performed':False}
        facade=recovery.ReplayFacade(view,auth,reconciler)
        worker=Mock(side_effect=AssertionError('second physical fit'))
        self.assertIsNone(facade.get_result(job_id,payload))
        with self.assertRaises(recovery.RecoveryStopped):
            facade.run_job(job_id,payload,worker,reservation_seconds=2100)
        reconciler.recover.assert_called_once()
        worker.assert_not_called()

    def test_wrong_first_uncached_identity_payload_or_reservation_block_without_core(self):
        _,payload,job_id,auth,_,view,reconciler=self.replay_fixture()
        facade=recovery.ReplayFacade(view,auth,reconciler)
        with self.assertRaisesRegex(ValueError,'unexpected_uncached_job_before_recovery'):
            facade.run_job('other',payload,Mock(),reservation_seconds=2100)
        for body,seconds in [(dict(payload,seed=1),2100),(payload,2101)]:
            with self.assertRaisesRegex(ValueError,'recovery_target_payload_changed'):
                facade.run_job(job_id,body,Mock(),reservation_seconds=seconds)
        reconciler.recover.assert_not_called()

    def test_deadline_checks_whole_reservation_before_any_started_event(self):
        events=[]
        session=SimpleNamespace(get_result=lambda *a:None,
            run_job=Mock(side_effect=lambda *a,**k:events.append('started')),
            _pause=Mock(side_effect=lambda reason,**details:(_ for _ in ()).throw(Paused(reason,**details))))
        wrapper=recovery.DeadlineSession(session,200,clock=self.clock)
        with self.assertRaises(Paused) as caught:
            wrapper.run_job('id',{},Mock(),reservation_seconds=101)
        self.assertEqual(caught.exception.reason,'session_budget')
        session.run_job.assert_not_called()
        self.assertEqual(events,[])

    def test_cached_result_after_deadline_is_not_replayed_and_worker_cap_is_clipped(self):
        worker=Mock(return_value={'genuine':'outcome'})
        def run_job(identity,payload,callback,*,reservation_seconds):
            return callback(9999)
        session=SimpleNamespace(get_result=lambda *a:None,run_job=run_job)
        wrapper=recovery.DeadlineSession(session,200,clock=self.clock)
        self.assertEqual(wrapper.run_job('id',{},worker,reservation_seconds=100),{'genuine':'outcome'})
        worker.assert_called_once_with(200)
        session.get_result=lambda *a:{'cached':True}
        session.run_job=Mock(return_value={'cached':True})
        self.clock.advance(500)
        wrapper.run_job('id',{},Mock(side_effect=AssertionError('cached fit')),reservation_seconds=100)
        session.run_job.assert_called_once()

    def test_external_report_keeps_original_unknown_and_does_not_export_outcome_or_secrets(self):
        self.clock.advance(123)
        context={'invocation_id':'c'*32,'status':'paused_session_budget','phase':'normal_continuation',
            'authorization':auth_fixture(),'authorization_sha256':'e'*64,
            'recovery_result':{'status':'reconciled','actual_fit_wall_seconds':21,'physical_reexecution_performed':True,
                'outcome':{'private':'must-not-export'},'authorization_bytes':'secret'},
            'diagnostics':{'logical_completed_jobs':502,'pending_job_ids':[]}}
        report=recovery.external_report(self.args(),context,100,clock=self.clock)
        self.assertEqual(report['physical_session_elapsed_seconds'],123)
        self.assertEqual(report['physical_session_remaining_seconds'],28800-123)
        self.assertIsNone(report['original_unknown_attempt']['elapsed_seconds'])
        self.assertIsNone(report['original_unknown_attempt']['optimizer_steps'])
        self.assertTrue(report['original_physical_attempt_still_unknown'])
        self.assertEqual((report['initial_campaign_limit_seconds'],report['effective_campaign_limit_seconds']),
                         (86400,129600))
        self.assertFalse(report['budget_extension_authorized'])
        self.assertNotIn('secret',json.dumps(report))
        self.assertNotIn('must-not-export',json.dumps(report))

    def fake_execute_engine(self, plan, receipt_state=None, *, prepare_error=None):
        mirror=Mock()
        def restore(directory):
            directory.mkdir()
            (directory/'campaign.json').write_text(json.dumps({'plan':plan,'plan_sha256':recovery.PLAN}))
            self.clock.advance(30)
            return {'remote_commit':TEST_HEAD,'plan_sha256':recovery.PLAN}
        mirror.restore.side_effect=restore
        receipts=Mock()
        receipts.inspect.return_value=receipt_state or {'started':None,'observed':None,'completed':None}
        engine=SimpleNamespace(transport=SimpleNamespace(GitCampaignMirror=Mock(return_value=mirror)),
            receipts=SimpleNamespace(RecoveryReceiptStore=Mock(return_value=receipts)),
            ledger=SimpleNamespace(SearchSession=Mock()),
            core=SimpleNamespace(read_snapshot=Mock(),ManualReconciler=Mock()),
            frozen=SimpleNamespace(prepare_panel=Mock(side_effect=prepare_error,return_value=('panel',{})),
                                   run_selection=Mock(),isolated_trial=Mock()))
        return engine,mirror,receipts

    def test_existing_started_without_observed_blocks_before_data_or_fits(self):
        plan={'synthetic':'plan'}
        engine,mirror,receipts=self.fake_execute_engine(plan,{'started':{'nonce':'prior'},'observed':None})
        with patch.object(recovery,'preflight',return_value=(engine,b'{}',auth_fixture(),'b'*64,plan)), \
                patch.dict(os.environ,{'GITHUB_TOKEN':'synthetic'}):
            code,_,report=self.invoke()
        self.assertEqual((code,report['error_code'],report['error_phase']),
            (2,'recovery_already_started_without_observed_outcome','recovery_receipt_inspection'))
        engine.frozen.prepare_panel.assert_not_called()
        engine.frozen.isolated_trial.assert_not_called()
        receipts.claim_started.assert_not_called()
        mirror.push.assert_not_called()
        mirror.close.assert_called_once()

    def test_changed_remote_head_refuses_recovery_before_data_or_claim(self):
        plan={'synthetic':'plan'}
        engine,mirror,receipts=self.fake_execute_engine(plan)
        mirror.restore.side_effect=None
        mirror.restore.return_value={'remote_commit':'9'*40,'plan_sha256':recovery.PLAN}
        with patch.object(recovery,'preflight',return_value=(engine,b'{}',auth_fixture(),'b'*64,plan)), \
                patch.dict(os.environ,{'GITHUB_TOKEN':'synthetic'}):
            code,_,report=self.invoke()
        self.assertEqual((code,report['error_code']),(2,'sealed_campaign_head_changed_before_recovery'))
        engine.core.read_snapshot.assert_not_called()
        engine.frozen.prepare_panel.assert_not_called()
        engine.frozen.isolated_trial.assert_not_called()
        receipts.claim_started.assert_not_called()
        mirror.push.assert_not_called()

    def test_invalid_existing_extension_blocks_before_data_and_is_public_safe_code(self):
        plan={'synthetic':'plan'}
        engine,mirror,receipts=self.fake_execute_engine(plan)
        engine.core.read_snapshot.side_effect=RuntimeError('sealed_budget_extension_changed')
        with patch.object(recovery,'preflight',return_value=(engine,b'{}',auth_fixture(),'b'*64,plan)), \
                patch.dict(os.environ,{'GITHUB_TOKEN':'synthetic'}):
            code,_,report=self.invoke()
        self.assertEqual((code,report['error_code'],report['error_phase']),
                         (2,'sealed_budget_extension_changed','snapshot_validation'))
        engine.frozen.prepare_panel.assert_not_called()
        receipts.claim_started.assert_not_called()
        mirror.push.assert_not_called()

    def test_execute_uses_entry_clock_for_recovery_worker_and_remaining_continuation(self):
        plan,payload,job_id,auth,trial,view,_=self.replay_fixture()
        engine,mirror,receipts=self.fake_execute_engine(plan)
        engine.core.read_snapshot.return_value=view
        real_auth=auth_fixture()
        real_auth['target']=auth['target']
        reconciler=Mock()
        engine.core.ManualReconciler.return_value=reconciler
        def recover_once(identity,body,callback,*,reservation_seconds):
            self.clock.advance(40)
            value=callback(999999)
            self.assertIs(value,trial)
            return {'status':'reconciled','completed_jobs':1553,'actual_fit_wall_seconds':50,
                    'physical_reexecution_performed':True,'campaign_accounted_through_monotonic':self.clock()}
        reconciler.recover.side_effect=recover_once
        engine.frozen.isolated_trial.return_value=trial
        def replay(plan,facade,panel,*,worker):
            facade.run_job(job_id,payload,lambda deadline:worker(payload,deadline),reservation_seconds=2100)
        engine.frozen.run_selection.side_effect=replay
        seen={}
        def continuation(args,engine,plan,directory,mirror,panel,deadline,*,clock,repair_only,accounted_through,context):
            seen.update(deadline=deadline,remaining=deadline-clock(),accounted_through=accounted_through)
            return 0,'paused_session_budget',1553
        with patch.object(recovery,'preflight',return_value=(engine,b'exact auth bytes',real_auth,'b'*64,plan)), \
                patch.object(recovery,'continue_selection',side_effect=continuation), \
                patch.dict(os.environ,{'GITHUB_TOKEN':'synthetic'}):
            code,_,state=self.invoke()
        self.assertEqual(code,0)
        self.assertEqual(engine.core.ManualReconciler.call_args.kwargs['physical_session_started_at'],100)
        self.assertEqual(engine.core.ManualReconciler.call_args.kwargs['physical_deadline'],28900)
        self.assertEqual(engine.frozen.isolated_trial.call_args.kwargs['deadline'],28900)
        self.assertEqual(seen,{'deadline':28900,'remaining':28730,'accounted_through':170})
        self.assertEqual(engine.receipts.RecoveryReceiptStore.call_args.kwargs['authorization_bytes'],b'exact auth bytes')
        self.assertEqual(state['physical_session_elapsed_seconds'],70)
        self.assertEqual(state['logical_completed_jobs'],1553)
        mirror.close.assert_called_once()

    def test_continuation_eager_session_and_status_ack_precede_transport_close(self):
        events=[]
        plan={'initial_controls':{'session_budget_seconds':28800}}
        session=SimpleNamespace(directory=self.base/'cache',pending_record_paths=('events/new.json',),
            completed_job_ids=tuple(str(i) for i in range(495)),
            _start_session=lambda:events.append('session_started'),
            mark_records_synced=lambda paths:events.append('start_ack'),
            close=lambda:events.append('closed'))
        def select(plan,wrapper,panel,*,worker,mirror,progress):
            self.assertIs(wrapper.session,session)
            self.assertEqual(events,['session_started','start_push','start_ack'])
            raise Paused('session_budget')
        frozen=SimpleNamespace(run_selection=select,report=Mock(return_value={'status':'paused_session_budget'}),
                               write_json=Mock())
        factory=Mock(return_value=session)
        engine=SimpleNamespace(ledger=SimpleNamespace(SearchSession=factory,SessionPaused=Paused,UnresolvedAttempt=Unresolved),
            campaign=SimpleNamespace(CampaignSyncError=SyncError),frozen=frozen)
        mirror=SimpleNamespace(push=lambda directory,*,paths,message:events.append('status_push' if 'status.json' in paths else 'start_push'))
        self.clock.advance(300)
        code,status,completed=recovery.continue_selection(self.args(),engine,plan,self.base/'cache',mirror,
            'panel',100+28800,clock=self.clock)
        self.assertEqual((code,status,completed),(0,'paused_session_budget',495))
        self.assertEqual(factory.call_args.kwargs['session_budget_seconds'],28500)
        self.assertEqual(plan['initial_controls']['session_budget_seconds'],28800)
        self.assertEqual(events,['session_started','start_push','start_ack','closed','status_push'])
        self.assertEqual(frozen.report.call_args.kwargs['pause_reason'],'session_budget')

    def test_real_frozen_ledger_charges_handoff_ack_and_cached_replay_once(self):
        # The sparse controller checkout obtains the unchanged engine path
        # through E2_ENGINE_ROOT/PYTHONPATH; it need not contain old code/data.
        engine_root=Path(os.environ.get('E2_ENGINE_ROOT',CONTROLLER))
        path=engine_root/'scripts/e2_search_session.py'
        if not path.is_file():
            found=importlib.util.find_spec('e2_search_session')
            if found is None:
                self.skipTest('separate pinned A3 ledger checkout unavailable')
            path=Path(found.origin)
        spec=importlib.util.spec_from_file_location('recovery_test_frozen_ledger',path)
        ledger=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ledger)
        directory=self.base/'real-ledger'
        plan={'synthetic':'fixed','initial_controls':{'session_budget_seconds':28800}}
        # A genuine prior session contributes20sec. Its recorded closure is
        # the observed handoff point;180sec of remote ACK delay follow it.
        prior=ledger.SearchSession(directory,plan,session_budget_seconds=28800,
            campaign_budget_seconds=86400,max_storage_bytes=10*1024**3,clock=self.clock)
        prior._start_session()
        self.clock.advance(20)
        prior.close()
        handoff=self.clock()
        self.clock.advance(180)
        recorded={}
        def report(plan,session,status,selections,repository,branch,**kwargs):
            recorded.update(summary=session.summary(),session_cap=session.session_budget_seconds)
            return {'status':status,'budget':session.summary()}
        def cached_replay(plan,session,panel,*,worker,mirror,progress):
            self.clock.advance(45)
            session._pause('session_budget')
        frozen=SimpleNamespace(report=report,write_json=recovery.write_json,run_selection=cached_replay)
        engine=SimpleNamespace(ledger=ledger,campaign=SimpleNamespace(CampaignSyncError=SyncError),frozen=frozen)
        calls=[]
        def push(directory,*,paths,message):
            calls.append(tuple(paths))
            self.clock.advance(5 if 'status.json' not in paths else 7)
        mirror=SimpleNamespace(push=push)
        deadline=100+28800
        result=recovery.continue_selection(self.args(),engine,plan,directory,mirror,'synthetic panel',
            deadline,clock=self.clock,accounted_through=handoff)
        self.assertEqual(result,(0,'paused_session_budget',0))
        #20prior +180ACK +5startACK +45cachedReplay; the last7sec final
        #closure ACK retains the exact finite frozen closure boundary policy.
        self.assertEqual(recorded['summary']['charged_wall_seconds'],250)
        self.assertEqual(recorded['session_cap'],deadline-handoff)
        self.assertEqual(plan['initial_controls']['session_budget_seconds'],28800)
        reopened=ledger.SearchSession(directory,plan,session_budget_seconds=28800,
            campaign_budget_seconds=86400,max_storage_bytes=10*1024**3,clock=self.clock)
        self.assertEqual(reopened.summary()['charged_wall_seconds'],250)
        self.assertEqual(reopened.summary()['conservative_unclosed_session_charges'],{})
        reopened.close()
        events=[json.loads(file.read_text()) for file in sorted((directory/'events').glob('*.json'))]
        self.assertEqual([x['session_budget_seconds'] for x in events if x['kind']=='session_started'],
                         [28800,28780])
        self.assertFalse(any(x['kind']=='job_started' for x in events))

    def test_genuine_fatal_selection_failure_keeps_frozen_abort_policy(self):
        events=[]
        session=SimpleNamespace(pending_record_paths=(),completed_job_ids=('target',),
            _start_session=lambda:events.append('start'),mark_records_synced=lambda paths:None,
            close=lambda:events.append('close'))
        frozen=SimpleNamespace(run_selection=Mock(side_effect=SelectionAborted()),
            report=Mock(return_value={'status':'blocked_selection_failure'}),write_json=Mock())
        engine=SimpleNamespace(ledger=SimpleNamespace(SearchSession=Mock(return_value=session),
            SessionPaused=Paused,UnresolvedAttempt=Unresolved),
            campaign=SimpleNamespace(CampaignSyncError=SyncError),frozen=frozen)
        mirror=Mock()
        temporal=SimpleNamespace(SelectionAborted=SelectionAborted)
        with patch.object(recovery.importlib,'import_module',return_value=temporal):
            result=recovery.continue_selection(self.args(),engine,{},self.base/'cache',mirror,
                'synthetic',100+28800,clock=self.clock)
        self.assertEqual(result,(5,'blocked_selection_failure',1))
        self.assertEqual(events,['start','close'])
        self.assertEqual(mirror.push.call_count,2)

    def test_real_ledger_continuation_preserves_24_to_36_extension_and_global_limit(self):
        engine_root=Path(os.environ.get('E2_ENGINE_ROOT',CONTROLLER))
        path=engine_root/'scripts/e2_search_session.py'
        if not path.is_file():
            found=importlib.util.find_spec('e2_search_session')
            if found is None:self.skipTest('separate pinned A3 ledger unavailable')
            path=Path(found.origin)
        spec=importlib.util.spec_from_file_location('recovery_v3_extended_frozen_ledger',path)
        ledger=importlib.util.module_from_spec(spec);spec.loader.exec_module(ledger)
        directory=self.base/'extended-ledger'
        plan={'synthetic':'fixed-initial-controls'}
        extension_bytes=None
        for index,duration in enumerate([28800,28800,28800,28800,14300]):
            prior=ledger.SearchSession(directory,plan,session_budget_seconds=28800,
                campaign_budget_seconds=86400,max_storage_bytes=10*1024**3,clock=self.clock)
            if index==0:
                prior.extend_campaign_budget(129600,reason='operator_requested_budget_extension')
                extension_bytes=(directory/'events/000000000.json').read_bytes()
            self.assertEqual(prior.campaign_budget_seconds,129600)
            prior._start_session();self.clock.advance(duration);prior.close()
        original_config=(directory/'campaign.json').read_bytes()
        worker=Mock(side_effect=AssertionError('global budget admitted an unaffordable fit'))
        def select(plan,session,panel,*,worker,mirror,progress):
            session.run_job('next',{'same_candidate_and_seed':True},lambda deadline:worker({},deadline),
                            reservation_seconds=2100)
        frozen=SimpleNamespace(run_selection=select,isolated_trial=worker,write_json=recovery.write_json,
            report=lambda plan,session,status,*args,**kwargs:{'status':status,'budget':session.summary()})
        engine=SimpleNamespace(ledger=ledger,campaign=SimpleNamespace(CampaignSyncError=SyncError),frozen=frozen)
        result=recovery.continue_selection(self.args(),engine,plan,directory,Mock(),'synthetic',
            self.clock()+28800,clock=self.clock)
        self.assertEqual(result,(0,'paused_campaign_budget',0))
        worker.assert_not_called()
        status=json.loads((directory/'status.json').read_text())
        self.assertEqual(status['budget']['campaign_budget_seconds'],129600)
        self.assertEqual(status['budget']['remaining_campaign_seconds'],100)
        self.assertEqual((directory/'campaign.json').read_bytes(),original_config)
        self.assertEqual(json.loads(original_config)['controls']['campaign_budget_seconds'],86400)
        self.assertEqual((directory/'events/000000000.json').read_bytes(),extension_bytes)
        events=[json.loads(p.read_text()) for p in sorted((directory/'events').glob('*.json'))]
        self.assertEqual(sum(e['kind']=='campaign_budget_extended' for e in events),1)
        self.assertFalse(any(e['kind']=='job_started' for e in events))

    def metadata_fixture(self, *, completed=1552, target_completed=False, pending=None):
        """Small hashed diagnostic records; scientific admission is tested separately."""
        directory=self.base/'cache'
        (directory/'events').mkdir(parents=True)
        (directory/'results').mkdir()
        plan={'diagnostic_fixture':'same sealed metadata'}
        def seal(path,value):
            value=dict(value,sha256=recovery.digest(value))
            recovery.write_json(path,value)
            return value
        config=seal(directory/'campaign.json',{'plan':plan,'plan_sha256':recovery.digest(plan)})
        previous='0'*64
        sequence=0
        def event(kind,**fields):
            nonlocal sequence,previous
            value=seal(directory/'events'/('%09d.json'%sequence),
                {'kind':kind,'sequence':sequence,'previous_sha256':previous,
                 'plan_sha256':config['plan_sha256'],**fields})
            previous=value['sha256'];sequence+=1
        ids=[hashlib.sha256(('synthetic-completed-%s'%i).encode()).hexdigest() for i in range(completed)]
        if target_completed:
            ids[-1]=TEST_TARGET['job_id']
        for identity in ids:
            event('job_started',job_id=identity)
            seal(directory/'results'/(hashlib.sha256(identity.encode()).hexdigest()+'.json'),{'job_id':identity})
        event('session_closed',session_id='d'*32,paused_reason='manual_recovery_observed_result_committed')
        for identity in ([TEST_TARGET['job_id']] if pending is None else pending):
            event('job_started',job_id=identity)
        return directory,plan

    def test_active_local_lock_is_read_only_and_blocks_before_source_gpu_or_network(self):
        directory,_=self.metadata_fixture()
        lock=(directory/'.campaign.lock').open('wb')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        before={str(p.relative_to(directory)):p.read_bytes() for p in directory.rglob('*.json')}
        try:
            with patch.object(recovery,'preflight',side_effect=AssertionError('source/GPU reached')) as check:
                code,final,report=self.invoke()
            check.assert_not_called()
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        self.assertEqual((code,report['status'],report['error_code'],report['error_phase']),
            (2,'runner_already_active','runner_already_active','local_lock_check'))
        self.assertEqual(report['logical_completed_jobs'],1552)
        self.assertEqual(report['pending_job_ids'],[TEST_TARGET['job_id']])
        self.assertEqual(report['prior_recovery_sessions_completed'],1)
        self.assertEqual(before,{str(p.relative_to(directory)):p.read_bytes() for p in directory.rglob('*.json')})
        self.assertNotIn(directory,Path(final['external_report']).parents)

    def test_explicit_report_is_exclusive_and_second_invocation_cannot_overwrite_or_fit(self):
        destination=self.base/'first-invocation.json'
        args=self.args('--external-report',str(destination))
        with patch.object(recovery,'preflight',side_effect=ValueError('github_secret_required')):
            first,_,report=self.invoke(args)
        original=destination.read_bytes()
        self.assertEqual((first,report['error_code']),(2,'github_secret_required'))
        with patch.object(recovery,'preflight',side_effect=AssertionError('second preflight')) as check:
            code,final,report=self.invoke(args)
        check.assert_not_called()
        self.assertEqual((code,final['error_code']),(2,'external_report_already_exists'))
        self.assertIsNone(report)
        self.assertEqual(destination.read_bytes(),original)

    def test_default_reports_are_unique_and_inside_campaign_path_is_refused(self):
        with patch.object(recovery,'preflight',side_effect=ValueError('github_secret_required')):
            _,first,one=self.invoke()
            _,second,two=self.invoke()
        self.assertNotEqual(first['external_report'],second['external_report'])
        self.assertNotEqual(one['invocation_id'],two['invocation_id'])
        with patch.object(recovery,'preflight') as check:
            code,final,report=self.invoke(self.args('--external-report',str(self.base/'cache'/'report.json')))
        check.assert_not_called()
        self.assertEqual((code,final['error_code']),(2,'external_report_must_be_outside_campaign'))
        self.assertIsNone(report)
        self.assertFalse((self.base/'cache').exists())

    def test_prior_recovery_completed_plus_new_pending_has_precise_diagnostic(self):
        directory,plan=self.metadata_fixture(completed=1553,target_completed=True,pending=['9'*64])
        engine,mirror,receipts=self.fake_execute_engine(plan,{'started':{'yes':True},'observed':{'yes':True}})
        mirror.restore.side_effect=None
        mirror.restore.return_value={'remote_commit':TEST_HEAD,'plan_sha256':recovery.digest(plan)}
        engine.core.read_snapshot.side_effect=RuntimeError('unexpected_additional_unresolved_job')
        with patch.object(recovery,'PLAN',recovery.digest(plan)), \
                patch.object(recovery,'preflight',return_value=(engine,b'raw',auth_fixture(),'b'*64,plan)), \
                patch.dict(os.environ,{'GITHUB_TOKEN':'synthetic'}):
            code,_,report=self.invoke()
        self.assertEqual((code,report['status'],report['error_code'],report['error_phase']),
            (2,'blocked_new_unresolved_job','unexpected_additional_unresolved_job','snapshot_validation'))
        self.assertEqual(report['logical_completed_jobs'],1553)
        self.assertEqual(report['pending_job_ids'],['9'*64])
        self.assertTrue(report['authorized_target_completed'])
        self.assertEqual(report['prior_recovery_sessions_completed'],1)
        engine.frozen.prepare_panel.assert_not_called()
        engine.frozen.isolated_trial.assert_not_called()
        mirror.push.assert_not_called()
        receipts.claim_started.assert_not_called()

    def test_arbitrary_lowercase_token_exception_never_becomes_public_error_code(self):
        private='ghp_abcdefghijklmnop'
        with patch.object(recovery,'preflight',side_effect=RuntimeError(private)):
            code,final,report=self.invoke()
        self.assertEqual((code,report['error_code']),(2,'recovery_runtime_failure'))
        self.assertNotIn(private,json.dumps(final)+json.dumps(report))
        typed=SyncError('github_pat_abcdefghijklmnop')
        self.assertEqual(recovery.safe_error_code(typed),'recovery_runtime_failure')
        self.assertEqual(recovery.safe_error_code(SyncError('github_rate_limit_pause')),'github_rate_limit_pause')

    def test_final_publication_error_retains_confirmed_recovery_and_exact_phase(self):
        session=SimpleNamespace(pending_record_paths=(),completed_job_ids=('confirmed',),
            _start_session=lambda:None,mark_records_synced=lambda paths:None,close=lambda:None)
        frozen=SimpleNamespace(run_selection=Mock(),report=Mock(return_value={'status':'completed_selected_registered_domains'}),
            write_json=Mock())
        engine=SimpleNamespace(ledger=SimpleNamespace(SearchSession=Mock(return_value=session),
            SessionPaused=Paused,UnresolvedAttempt=Unresolved),
            campaign=SimpleNamespace(CampaignSyncError=SyncError),frozen=frozen)
        def push(directory,*,paths,message):
            if 'status.json' in paths:
                raise SyncError('github_rate_limit_pause')
        mirror=SimpleNamespace(push=push)
        context={'phase':'normal_continuation','recovery_result':{'status':'logical_job_reconciled'},
                 'error_code':None,'error_phase':None}
        code,status,_=recovery.continue_selection(self.args(),engine,{},self.base/'cache',mirror,
            'synthetic',100+28800,clock=self.clock,context=context)
        self.assertEqual((code,status),(4,'paused_github_sync'))
        self.assertEqual((context['error_code'],context['error_phase']),
            ('github_rate_limit_pause','continuation_publication'))
        self.assertEqual(context['recovery_result'],{'status':'logical_job_reconciled'})

    def test_graceful_interruption_closes_real_ledger_preserves_unknown_and_blocks_replay(self):
        engine_root=Path(os.environ.get('E2_ENGINE_ROOT',CONTROLLER))
        path=engine_root/'scripts/e2_search_session.py'
        if not path.is_file():
            found=importlib.util.find_spec('e2_search_session')
            if found is None:self.skipTest('separate pinned A3 ledger unavailable')
            path=Path(found.origin)
        spec=importlib.util.spec_from_file_location('recovery_v3_cancel_frozen_ledger',path)
        ledger=importlib.util.module_from_spec(spec);spec.loader.exec_module(ledger)
        directory=self.base/'cancelled-ledger'
        plan={'same_plan':True,'initial_controls':{'session_budget_seconds':28800}}
        cleanup=[]
        def isolated(*args,**kwargs):
            try:raise KeyboardInterrupt()
            finally:cleanup.append('worker_group_cleanup')
        def select(plan,session,panel,*,worker,mirror,progress):
            session.run_job('new-pending',{'synthetic':'same candidate and seed'},
                lambda deadline:worker({'synthetic':'job'},deadline),reservation_seconds=100)
        def report(plan,session,status,selections,repository,branch,**kwargs):
            return {'status':status,'budget':session.summary()}
        frozen=SimpleNamespace(run_selection=select,isolated_trial=isolated,report=report,write_json=recovery.write_json)
        engine=SimpleNamespace(ledger=ledger,campaign=SimpleNamespace(CampaignSyncError=SyncError),frozen=frozen)
        context={'phase':'normal_continuation','error_code':None,'error_phase':None}
        code,status,_=recovery.continue_selection(self.args(),engine,plan,directory,Mock(),
            'synthetic',100+28800,clock=self.clock,context=context)
        self.assertEqual((code,status),(130,'interrupted_between_jobs'))
        self.assertEqual(context['error_code'],'operator_interruption')
        self.assertEqual(cleanup,['worker_group_cleanup'])
        events=[json.loads(path.read_text()) for path in sorted((directory/'events').glob('*.json'))]
        self.assertEqual(sum(event['kind']=='job_started' for event in events),1)
        self.assertEqual(events[-1]['kind'],'session_closed')
        self.assertEqual(list((directory/'results').glob('*.json')),[])
        self.assertIsNone(recovery.UNKNOWN['optimizer_steps'])
        with self.assertRaises(ledger.UnresolvedAttempt):
            ledger.SearchSession(directory,plan,session_budget_seconds=28800,campaign_budget_seconds=86400,
                max_storage_bytes=10*1024**3,clock=self.clock)


if __name__=='__main__':
    unittest.main()
