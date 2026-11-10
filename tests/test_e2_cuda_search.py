"""CUDA search admission, public outcomes, and durable orchestration without GPU fits."""
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
from contextlib import redirect_stdout
from io import StringIO
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import e2_cuda_search_plan as admission
import run_e2_cuda_search as runner
from e2_search_session import SearchSession,SessionPaused,UnresolvedAttempt
from e2_temporal_training import CandidateSpec,TrialRun,prepare_b,select_b
from e2_training_bindings import candidate_for
import e2_training_models as registry
from run_e2_training_synthetic import artificial_panel


class Clock:
    def __init__(self):self.value=0.0
    def __call__(self):return self.value
    def advance(self,seconds):self.value+=seconds


class RecordingMirror:
    """Local acknowledgement model; never calls GitHub or a network client."""
    def __init__(self):
        self.started=set();self.finished=set();self.order=[];self.sealed=False

    def restore(self,directory):
        return None

    def check_capacity(self,*,required_requests):
        if required_requests<1:raise AssertionError('Capacity must reserve positive requests')
        return {'remaining_requests':100000}

    def push(self,directory,*,paths,message):
        from e2_github_campaign import validate_record
        for relative in paths:
            payload=(Path(directory)/relative).read_bytes()
            record=validate_record(relative,payload)
            if relative=='campaign.json':self.sealed=True
            if record.get('kind')=='job_started':
                self.started.add(record['job_id']);self.order.append(('started_ack',record['job_id']))
            if relative.startswith('results/'):
                self.finished.add(record['job_id']);self.order.append(('result_ack',record['job_id']))
        return {'acknowledged':True}


class SearchIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel=artificial_panel()
        cls.prepared=prepare_b(cls.panel,cls.panel.calendar[220],cls.panel.calendar[222])

    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.base=Path(self.temporary.name)
        self.registration=self.base/'admission.json'
        value={'version':'e2-cuda-search-admission-v1','protocol_sha256':admission.PROTOCOL_SHA256,
            'repository':admission.REPOSITORY,'initial_budget':{'session_seconds':28800,'campaign_seconds':86400},
            'status':'registered_before_scientific_cuda_fits',
            'implementation_sha256':{'scripts/run_e2_cuda_search.py':admission.file_sha(ROOT/'scripts/run_e2_cuda_search.py')}}
        self.registration.write_text(json.dumps(value))
        self.reg_patch=patch.object(admission,'REGISTRATION',self.registration);self.reg_patch.start()

    def tearDown(self):
        self.reg_patch.stop();self.temporary.cleanup()

    def plan(self,**kwargs):
        return admission.build_plan(source_commit='a'*40,**kwargs)

    def trial(self,*,run_id='b'*64,predictions=None,dates=None,resource=None,status='completed',fatal=False):
        if resource is None:
            resource={'optimizer_steps':4,'optimizer_steps_known_lower_bound':4,'model_parameters':593,
                'optimization_stages':1,'elapsed_seconds':.2,
                'phase_records':{'supervised':{'epochs_completed':1,'best_epoch':1,'optimizer_steps':4,
                    'elapsed_seconds':.1,'fixed_epochs':None,'selected_state_sha256':'c'*64}},
                'max_cuda_memory_allocated_bytes':1024,'max_cuda_memory_reserved_bytes':2048}
        return TrialRun(status,run_id,resource,
            [float(v)+.001 for v in self.prepared.inner_validation.y_original] if predictions is None and status=='completed' else predictions,
            self.prepared.inner_validation.dates if dates is None and status=='completed' else dates,
            {'supervised':1} if status=='completed' else None,
            None if status=='completed' else 'private raw failure /content/secret',fatal)

    def fixture_plan(self):
        plan=self.plan(models=['MLP16'],origins=['2025-01-01'],hardware={'device_name':'Synthetic T4'})
        # Explicit artificial-date fixture, never a registered scientific run.
        plan['origins']=[str(self.prepared.origin.date())]
        plan['scope']='synthetic_orchestration_fixture'
        return plan

    def test_stdlib_cli_plan_does_not_import_trainers_touch_market_or_authenticate(self):
        output=self.base/'must-not-exist'
        script=ROOT/'scripts/run_e2_cuda_search.py'
        wrapper=f'''
import sys,os,runpy,importlib.abc
from pathlib import Path
class BlockImports(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {{'torch','numpy','pandas','scipy','e2_github_campaign','e2_search_session','e2_development_data','e2_temporal_training'}}:
   raise AssertionError('plan-only imported execution dependency: '+fullname)
sys.meta_path.insert(0,BlockImports())
def audit(event,args):
 if event in ('socket.connect','subprocess.Popen','os.mkdir'):
  raise AssertionError('plan-only performed execution action: '+event)
 if event=='open':
  path,mode,flags=args
  if isinstance(path,str) and path.lower().endswith('.csv'):
   raise AssertionError('plan-only opened market values')
  if flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
   raise AssertionError('plan-only wrote a file')
sys.addaudithook(audit)
sys.path.insert(0,{str(ROOT/'scripts')!r})
sys.argv=[{str(script)!r},'--plan-only','--source-commit',{'a'*40!r},'--output',{str(output)!r},'--data-root','/not-read/market.csv']
runpy.run_path({str(script)!r},run_name='__main__')
'''
        env=dict(os.environ,GITHUB_TOKEN='bogus-plan-only-token',GH_TOKEN='bogus-other-token',PYTHONDONTWRITEBYTECODE='1')
        result=subprocess.run([sys.executable,'-B','-S','-c',wrapper],text=True,capture_output=True,env=env,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        value=json.loads(result.stdout)
        self.assertEqual((value['hpo_attempts'],value['nas1_attempts'],value['nas2_attempts_upper_bound']),(5544,66000,13530))
        self.assertEqual(value['selection_attempts_upper_bound'],85074)
        self.assertFalse(value['scientific_HPO_or_NAS_executed'])
        self.assertFalse(value['final_market_values_read'])
        self.assertFalse(output.exists())
        self.assertNotIn('bogus',result.stdout+result.stderr)

    def test_plan_admits_full_domains_and_marks_explicit_subsets(self):
        full=self.plan();summary=admission.plan_summary(full)
        self.assertEqual(len(full['models']),14);self.assertEqual(len(full['origins']),11)
        self.assertFalse(full['coverage']['partial_models']);self.assertFalse(full['coverage']['partial_origins'])
        self.assertEqual(full['k'],40)
        self.assertEqual(summary['nas_finalists'],'40 or 41 including reference')
        partial=self.plan(models=['MLP16'],origins=['2025-01-01'])
        self.assertTrue(partial['coverage']['partial_models']);self.assertTrue(partial['coverage']['partial_origins'])
        self.assertFalse(partial['coverage']['final_refit_included'])
        self.assertFalse(partial['coverage']['classical_models_included'])

    def test_invalid_source_budget_device_and_registration_are_rejected(self):
        cases=[{'source_commit':'main'},{'source_commit':'A'*40},{'session_hours':0},
            {'session_hours':True},{'campaign_hours':4},{'storage_gib':0},
            {'device':'cpu'},{'models':['missing']},{'origins':['2025-02-01']},
            {'expected_registration_sha256':'0'*64}]
        for item in cases:
            with self.subTest(item=item),self.assertRaises(ValueError):
                options={'source_commit':'a'*40,**item};admission.build_plan(**options)

    def test_implementation_manifest_is_an_exact_subset_and_changed_bytes_fail(self):
        record=json.loads(self.registration.read_text())
        path='scripts/e2_cuda_search_plan.py'
        record['implementation_sha256']={path:admission.file_sha(ROOT/path)}
        self.registration.write_text(json.dumps(record))
        plan=self.plan()
        self.assertEqual(set(plan['source_sha256']),{path})
        record['implementation_sha256'][path]='0'*64
        self.registration.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,'implementation_hash_mismatch'):self.plan()

    def test_plan_and_job_identity_change_with_source_hardware_payload_or_validation(self):
        base=self.plan(models=['MLP16'],origins=['2025-01-01'],hardware={'device_name':'T4'})
        other=admission.build_plan(source_commit='d'*40,models=['MLP16'],origins=['2025-01-01'],hardware={'device_name':'T4'})
        hardware=self.plan(models=['MLP16'],origins=['2025-01-01'],hardware={'device_name':'L4'})
        self.assertEqual(len({admission.digest(p) for p in (base,other,hardware)}),3)
        candidate=candidate_for('MLP16',registry.registered_trials('MLP16')[0],None,'hpo')
        job=runner.job_payload(base,self.prepared,'MLP16',candidate,271201)
        changed=runner.job_payload(hardware,self.prepared,'MLP16',candidate,271201)
        self.assertNotEqual(admission.digest(job),admission.digest(changed))
        fake=SimpleNamespace(origin=self.prepared.origin,signature=self.prepared.signature,
            inner_train=self.prepared.inner_train,inner_validation=SimpleNamespace(signature='e'*64))
        changed=runner.job_payload(base,fake,'MLP16',candidate,271201)
        self.assertNotEqual(admission.digest(job),admission.digest(changed))
        payload=deepcopy(candidate.payload);payload['identity_fixture']=1
        altered=CandidateSpec(candidate.candidate_id,payload)
        changed=runner.job_payload(base,self.prepared,'MLP16',altered,271201)
        self.assertNotEqual(admission.digest(job),admission.digest(changed))

    def test_public_trial_excludes_private_nested_receipts_and_restores_typed_dates(self):
        raw=self.trial()
        raw.resource_use.update(reason='/content/private',checkpoint={'file':'/content/weights.pt'},
            environment={'GITHUB_TOKEN':'not-for-export'},raw_logs=['private'])
        raw.resource_use['phase_records']['supervised']['logs']=[{'path':'/content/internal'}]
        value=runner.public_trial(raw)
        serialized=json.dumps(value)
        for text in ('/content','GITHUB_TOKEN','not-for-export','raw_logs','"logs"','weights.pt'):
            self.assertNotIn(text,serialized)
        self.assertEqual(value['resource_use']['cuda_metrics'],{'max_memory_allocated_bytes':1024,'max_memory_reserved_bytes':2048})
        restored=runner.restore_trial(value)
        self.assertIsInstance(restored,TrialRun)
        self.assertEqual(restored.prediction_dates,self.prepared.inner_validation.dates)
        self.assertEqual(list(restored.predictions_original),list(raw.predictions_original))
        self.assertEqual(restored.phase_epochs,raw.phase_epochs)

    def test_public_trial_invalid_numbers_status_identifiers_and_cache_are_rejected(self):
        for field,value in [('elapsed_seconds',{'path':'/content/private'}),('optimizer_steps',True),
                            ('optimizer_steps',-1),('model_parameters',float('nan'))]:
            raw=self.trial();raw.resource_use[field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):runner.public_trial(raw)
        for field,value in [('status','running'),('fatal','false'),('run_id','/content/run')]:
            raw=self.trial();setattr(raw,field,value)
            with self.subTest(field=field),self.assertRaises(ValueError):runner.public_trial(raw)
        serialized=runner.public_trial(self.trial());serialized['resource_use']['unexpected']='private'
        with self.assertRaises(ValueError):runner.restore_trial(serialized)
        serialized=runner.public_trial(self.trial());serialized['predictions_original'][0]=float('inf')
        with self.assertRaises(ValueError):runner.restore_trial(serialized)

    def test_public_failure_uses_fixed_code_and_keeps_unknown_optimizer_count(self):
        run=self.trial(status='failed',resource={'optimizer_steps':None,'optimizer_steps_known_lower_bound':3,
            'partial_optimizer_update_may_have_occurred':True,'error_code':'external_timeout'},fatal=False)
        value=runner.public_trial(run)
        self.assertEqual(value['reason'],'external_timeout')
        self.assertIsNone(value['resource_use']['optimizer_steps'])
        self.assertEqual(value['resource_use']['optimizer_steps_known_lower_bound'],3)
        self.assertTrue(value['resource_use']['optimizer_steps_unknown'])
        self.assertNotIn('/content',json.dumps(value))

    def fixture_orchestrator(self,*,winner_events):
        candidates=[candidate_for('MLP16',registry.registered_trials('MLP16')[i],None,'hpo') for i in range(3)]
        def orchestrator(prepared,model,adapter,**kwargs):
            self.assertEqual(model,'MLP16')
            self.assertEqual(len(kwargs['hpo_trials']),18)
            self.assertEqual(adapter.full_epochs,200)
            result=select_b(prepared,candidates,[271201],adapter.fit_candidate)
            winner_events.append(result.winner.candidate_id)
            return result,{'hpo':result}
        return candidates,orchestrator

    def test_partial_budget_pauses_without_winner_and_resume_never_replays_committed_fits(self):
        plan=self.fixture_plan();clock=Clock();mirror=RecordingMirror();winners=[];calls=[]
        candidates,orchestrator=self.fixture_orchestrator(winner_events=winners)
        index={candidate.candidate_id:i for i,candidate in enumerate(candidates)}
        directory=self.base/'campaign'
        def worker(job,deadline):
            key=admission.digest({'plan_sha256':admission.digest(plan),'job':job})
            self.assertIn(key,mirror.started)
            self.assertNotIn(key,mirror.finished)
            self.assertGreater(deadline,clock())
            mirror.order.append(('fit',key));calls.append(key);clock.advance(3000)
            value=[v+.003-index[job['candidate_id']]*.001 for v in self.prepared.inner_validation.y_original]
            trial=self.trial(run_id=admission.digest(job),predictions=value)
            trial.resource_use['identity']=runner.identity(job)
            return runner.public_trial(trial)
        with patch('e2_training_bindings.run_neural_selection',side_effect=orchestrator):
            for expected in (1,2):
                with SearchSession(directory,plan,session_budget_seconds=10000,campaign_budget_seconds=30000,
                                   max_storage_bytes=8*1024**2,clock=clock) as session:
                    with self.assertRaises(SessionPaused):runner.run_selection(plan,session,self.panel,worker=worker,mirror=mirror)
                    self.assertEqual(len(session.completed_job_ids),expected)
                    self.assertEqual(len(calls),expected)
                    self.assertEqual(winners,[])
            with SearchSession(directory,plan,session_budget_seconds=10000,campaign_budget_seconds=30000,
                               max_storage_bytes=8*1024**2,clock=clock) as session:
                selected=runner.run_selection(plan,session,self.panel,worker=worker,mirror=mirror)
                self.assertEqual(len(calls),3)
                self.assertEqual(selected[0]['winner_id'],candidates[2].candidate_id)
                self.assertEqual(len(session.completed_job_ids),3)
                self.assertEqual(session.charged_wall_seconds,9000)
            with SearchSession(directory,plan,session_budget_seconds=10000,campaign_budget_seconds=30000,
                               max_storage_bytes=8*1024**2,clock=clock) as session:
                again=runner.run_selection(plan,session,self.panel,worker=worker,mirror=mirror)
                self.assertEqual(again,selected);self.assertEqual(len(calls),3)
        for key in calls:
            self.assertEqual(calls.count(key),1)
            sequence=[kind for kind,identity in mirror.order if identity==key]
            self.assertLess(sequence.index('started_ack'),sequence.index('fit'))
            self.assertLess(sequence.index('fit'),sequence.index('result_ack'))
        for previous,current in zip(calls,calls[1:]):
            self.assertLess(mirror.order.index(('result_ack',previous)),mirror.order.index(('fit',current)))

    def test_failed_started_acknowledgement_starts_no_fit_and_blocks_automatic_retry(self):
        from e2_github_campaign import CampaignSyncError
        plan=self.fixture_plan();clock=Clock();winners=[];_,orchestrator=self.fixture_orchestrator(winner_events=winners)
        class FailedMirror:
            def check_capacity(self,**kwargs):return None
            def push(self,*args,**kwargs):raise CampaignSyncError('synthetic_sync_failure')
        directory=self.base/'campaign'
        with SearchSession(directory,plan,session_budget_seconds=10000,campaign_budget_seconds=30000,
                           max_storage_bytes=8*1024**2,clock=clock) as session:
            with patch('e2_training_bindings.run_neural_selection',side_effect=orchestrator),patch.object(runner,'restore_trial',wraps=runner.restore_trial):
                worker=unittest.mock.Mock()
                with self.assertRaises(UnresolvedAttempt):runner.run_selection(plan,session,self.panel,worker=worker,mirror=FailedMirror())
                worker.assert_not_called();self.assertEqual(winners,[])
        with self.assertRaises(UnresolvedAttempt):
            SearchSession(directory,plan,session_budget_seconds=10000,campaign_budget_seconds=30000,
                          max_storage_bytes=8*1024**2,clock=clock)

    def test_remote_rate_refusal_precedes_started_ledger_and_training(self):
        from e2_github_campaign import CampaignRateLimited
        plan=self.fixture_plan();clock=Clock();winners=[];_,orchestrator=self.fixture_orchestrator(winner_events=winners)
        class RateLimitedMirror:
            def check_capacity(self,**kwargs):raise CampaignRateLimited(60)
            def push(self,*args,**kwargs):raise AssertionError('No upload may follow capacity refusal')
        directory=self.base/'campaign'
        with SearchSession(directory,plan,session_budget_seconds=10000,campaign_budget_seconds=30000,
                           max_storage_bytes=8*1024**2,clock=clock) as session:
            worker=unittest.mock.Mock()
            with patch('e2_training_bindings.run_neural_selection',side_effect=orchestrator),self.assertRaises(CampaignRateLimited):
                runner.run_selection(plan,session,self.panel,worker=worker,mirror=RateLimitedMirror())
            worker.assert_not_called();self.assertEqual(session.completed_job_ids,())
        with SearchSession(directory,plan,session_budget_seconds=10000,campaign_budget_seconds=30000,
                           max_storage_bytes=8*1024**2,clock=clock) as restored:
            self.assertEqual(restored.completed_job_ids,())

    def test_supervisor_strips_github_credentials_and_timeout_is_unknown_not_zero(self):
        plan=self.fixture_plan();candidate=candidate_for('MLP16',registry.registered_trials('MLP16')[0],None,'hpo')
        job=runner.job_payload(plan,self.prepared,'MLP16',candidate,271201)
        seen={}
        def supervised(command,output,**kwargs):
            seen.update(command=command,**kwargs)
            return {'status':'timeout','worker_profile':None,'elapsed_wall_seconds':1.,'sampled_peak_process_group_rss_bytes':1000}
        with patch('profile_e2_training.run_isolated_profile',side_effect=supervised),patch.dict(os.environ,{
                'GITHUB_TOKEN':'private-test-github','GH_TOKEN':'private-test-other','PYTHONPATH':'/not-a-worker-path'}):
            value=runner.isolated_trial(plan,job,data_root=self.base,source_root_itransformer=None,deadline=time.monotonic()+60)
        self.assertNotIn('GITHUB_TOKEN',seen['environment']);self.assertNotIn('GH_TOKEN',seen['environment'])
        self.assertNotIn('PYTHONPATH',seen['environment'])
        self.assertEqual(seen['environment']['CUBLAS_WORKSPACE_CONFIG'],':4096:8')
        self.assertIn('--worker-spec',seen['command'])
        self.assertIsNone(value['resource_use']['optimizer_steps'])
        self.assertEqual(value['resource_use']['optimizer_steps_known_lower_bound'],0)
        self.assertTrue(value['resource_use']['optimizer_steps_unknown'])
        self.assertFalse(value['fatal']);self.assertEqual(value['reason'],'external_timeout')

    def test_successful_worker_result_is_bound_to_exact_plan_and_job(self):
        plan=self.fixture_plan();candidate=candidate_for('MLP16',registry.registered_trials('MLP16')[0],None,'hpo')
        job=runner.job_payload(plan,self.prepared,'MLP16',candidate,271201)
        raw=self.trial();raw.resource_use['identity']=runner.identity(job);value=runner.public_trial(raw)
        def supervised(command,output,**kwargs):
            spec=json.loads(Path(command[-1]).read_text())
            return {'status':'success','worker_profile':{'job_id':spec['job_id'],'plan_sha256':admission.digest(plan),'trial_run':value},
                    'elapsed_wall_seconds':.5,'sampled_peak_process_group_rss_bytes':1000}
        with patch('profile_e2_training.run_isolated_profile',side_effect=supervised):
            outcome=runner.isolated_trial(plan,job,data_root=self.base,source_root_itransformer=None,deadline=time.monotonic()+60)
        self.assertEqual(outcome['status'],'completed')
        self.assertEqual(outcome['resource_use']['optimizer_steps'],4)
        self.assertEqual(outcome['resource_use']['supervisor_rss_bytes'],1000)

    def test_external_timeout_preserves_any_committed_worker_step_lower_bound(self):
        plan=self.fixture_plan();candidate=candidate_for('MLP16',registry.registered_trials('MLP16')[0],None,'hpo')
        job=runner.job_payload(plan,self.prepared,'MLP16',candidate,271201)
        raw=self.trial();raw.resource_use['identity']=runner.identity(job);value=runner.public_trial(raw)
        def supervised(command,output,**kwargs):
            spec=json.loads(Path(command[-1]).read_text())
            return {'status':'timeout','worker_profile':{'job_id':spec['job_id'],'plan_sha256':admission.digest(plan),'trial_run':value},
                    'elapsed_wall_seconds':1.,'sampled_peak_process_group_rss_bytes':1000}
        with patch('profile_e2_training.run_isolated_profile',side_effect=supervised):
            outcome=runner.isolated_trial(plan,job,data_root=self.base,source_root_itransformer=None,deadline=time.monotonic()+60)
        self.assertEqual(outcome['status'],'failed')
        self.assertIsNone(outcome['resource_use']['optimizer_steps'])
        self.assertEqual(outcome['resource_use']['optimizer_steps_known_lower_bound'],4)

    def test_wrong_plan_or_job_worker_envelope_is_fatal_and_not_counted_as_zero_work(self):
        plan=self.fixture_plan();candidate=candidate_for('MLP16',registry.registered_trials('MLP16')[0],None,'hpo')
        job=runner.job_payload(plan,self.prepared,'MLP16',candidate,271201)
        raw=self.trial();raw.resource_use['identity']=runner.identity(job);value=runner.public_trial(raw)
        for changed in ('job_id','plan_sha256'):
            def supervised(command,output,**kwargs):
                spec=json.loads(Path(command[-1]).read_text())
                worker={'job_id':spec['job_id'],'plan_sha256':admission.digest(plan),'trial_run':value}
                worker[changed]='0'*64
                return {'status':'success','worker_profile':worker,'elapsed_wall_seconds':1.,'sampled_peak_process_group_rss_bytes':1000}
            with self.subTest(changed=changed),patch('profile_e2_training.run_isolated_profile',side_effect=supervised):
                outcome=runner.isolated_trial(plan,job,data_root=self.base,source_root_itransformer=None,deadline=time.monotonic()+60)
            self.assertEqual(outcome['status'],'failed');self.assertTrue(outcome['fatal'])
            self.assertEqual(outcome['reason'],'worker_result_invalid')
            self.assertIsNone(outcome['resource_use']['optimizer_steps'])

    def test_execute_seals_before_data_and_reports_fatal_failure_without_a_winner(self):
        plan=self.fixture_plan();mirror=RecordingMirror();winners=[];_,orchestrator=self.fixture_orchestrator(winner_events=winners)
        directory=self.base/'campaign'
        args=SimpleNamespace(github_repository=admission.REPOSITORY,github_branch='campaigns/scientific-test',
            github_prefix='campaigns/scientific-test',source_commit='a'*40,models=['MLP16'],origins=['2025-01-01'],
            session_hours=8,campaign_hours=24,storage_gib=10,device='cuda:0',expected_registration_sha256=None,
            output=directory,extend_campaign_hours=None,data_root=self.base,source_root_itransformer=None)
        def panel(data_root):
            self.assertTrue(mirror.sealed)
            return self.panel,{'synthetic_fixture':True}
        def worker(plan,job,**kwargs):
            trial=self.trial(status='failed',resource={'optimizer_steps':None,'optimizer_steps_known_lower_bound':2,
                'partial_optimizer_update_may_have_occurred':True,'error_code':'geometry','identity':runner.identity(job)},fatal=True)
            return runner.public_trial(trial)
        with (patch.dict(os.environ,{'GITHUB_TOKEN':'not-a-real-secret'}),
              patch.object(runner,'verify_source'),patch.object(runner,'cuda_hardware',return_value=plan['hardware_binding']),
              patch.object(admission,'build_plan',return_value=plan),patch.object(runner,'prepare_panel',side_effect=panel),
              patch.object(runner,'isolated_trial',side_effect=worker),
              patch('e2_github_campaign.GitHubCampaignMirror',return_value=mirror),
              patch('e2_training_bindings.run_neural_selection',side_effect=orchestrator),redirect_stdout(StringIO())):
            code=runner.execute(args)
        self.assertEqual(code,5)
        state=json.loads((directory/'status.json').read_text())
        self.assertEqual(state['status'],'blocked_selection_failure')
        self.assertEqual(state['completed_selections'],[]);self.assertEqual(winners,[])
        self.assertEqual((state['completed_jobs'],state['failed_jobs'],state['known_optimizer_steps'],state['unknown_optimizer_step_jobs']),(1,1,2,1))
        self.assertEqual(len(mirror.finished),1)

    def test_execution_without_secret_or_with_wrong_source_stops_before_market_and_remote(self):
        args=SimpleNamespace(github_repository=admission.REPOSITORY,source_commit='a'*40)
        with (patch.dict(os.environ,{},clear=True),patch.object(runner,'verify_source') as source,
                patch.object(runner,'prepare_panel') as data,patch('e2_github_campaign.GitHubCampaignMirror') as mirror):
            with self.assertRaisesRegex(ValueError,'github_secret_required'):runner.execute(args)
            source.assert_not_called();data.assert_not_called();mirror.assert_not_called()
        with (patch.dict(os.environ,{'GITHUB_TOKEN':'not-a-real-secret'}),patch.object(runner,'verify_source',side_effect=ValueError('pin mismatch')),
                patch.object(runner,'cuda_hardware') as hardware,patch.object(runner,'prepare_panel') as data,
                patch('e2_github_campaign.GitHubCampaignMirror') as mirror):
            with self.assertRaisesRegex(ValueError,'pin mismatch'):runner.execute(args)
            hardware.assert_not_called();data.assert_not_called();mirror.assert_not_called()

    def test_unpublished_or_unfrozen_admission_blocks_before_gpu_data_and_remote(self):
        args=SimpleNamespace(github_repository=admission.REPOSITORY,source_commit='a'*40)
        published=json.loads(self.registration.read_text())
        for issue in ('draft','empty_implementation'):
            record=deepcopy(published)
            if issue=='draft':record['status']='draft_awaiting_implementation_freeze'
            else:record['implementation_sha256']={}
            self.registration.write_text(json.dumps(record))
            with (self.subTest(issue=issue),patch.dict(os.environ,{'GITHUB_TOKEN':'not-a-real-secret'}),
                  patch.object(runner,'verify_source'),patch.object(runner,'cuda_hardware') as hardware,
                  patch.object(runner,'prepare_panel') as data,
                  patch('e2_github_campaign.GitHubCampaignMirror') as mirror):
                with self.assertRaisesRegex(ValueError,'published_scientific_admission_required'):
                    runner.execute(args)
                hardware.assert_not_called();data.assert_not_called();mirror.assert_not_called()


if __name__=='__main__':unittest.main()
