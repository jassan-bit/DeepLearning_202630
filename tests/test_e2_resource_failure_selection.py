"""Synthetic closed-timeout continuation; no GPU, market data, network or retry."""
from pathlib import Path
from types import ModuleType,SimpleNamespace
import hashlib,json,sys,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import e2_resource_failure_selection as R
from e2_amended_pilot_artifacts import PilotArtifactStore
E=R.original

class ResourceContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source={'registered-engine':'unchanged'}
        self.domains=E.current_domains();self.jobs=E.job_plan(['TimesNet'],domains=self.domains)
        self.job=self.jobs[6];self.nextjob=self.jobs[7]
        self.directory=self.root/'jobs'/self.job['logical_job_id'];self.directory.mkdir(parents=True)
        self.budget=SimpleNamespace(_events=[],remaining_seconds=30000,stage_wall_seconds=1800,
            report=lambda:{'remaining_seconds':30000})
        self.runtime=self.root/'runtime.json'
        E.write_json(self.runtime,{'fit_execution_policy_verified':True,'runtime':{'test':'no GPU'}})
        self.args=SimpleNamespace(output=self.root,runtime_profile=self.runtime,models=['TimesNet'],
            data_root='unopened',source_roots='unopened')

    def specimen(self,*,wall=600,spec=None):
        d=self.directory;job=self.job
        if spec is None:
            spec={'version':E.VERSION,'job':job,'output':str(d),'data_root':'unopened',
                'source_roots':'unopened','source_sha256':self.source,'runtime_identity':{'test':'no GPU'},
                'fit_wall_seconds':wall}
            E.write_json(d/'worker_spec.json',spec)
        wall=spec['fit_wall_seconds']
        config={'seed':job['seed'],'max_epochs':200,'patience':20,'wall_limit_seconds':wall,
                'learning_rate':job['trial']['learning_rate'],'weight_decay':job['trial']['weight_decay'],
                'batch_size':job['trial']['batch_size']}
        request={'name':job['model_id'],'trial':job['trial'],'seed':job['seed'],
            'sources':self.source,'config':config}
        E.write_json(d/'request.json',request)
        store=PilotArtifactStore(d/'phases','supervised',
            {'request_sha256':E.digest(request),'seed':job['seed'],'phase':'supervised'})
        store.prepare({'config':config,'source':self.source,'data':{'test':'synthetic'},
            'seed':job['seed'],'resources':{'test':'no GPU'}})
        store.begin_training()
        initial=store.checkpoint(b'initial',label='initial')
        store.event('epoch_completed',{'optimizer_steps':7,'known_cost_seconds':wall-1})
        latest=store.checkpoint(b'latest',label='latest');best=store.checkpoint(b'best',label='best')
        resource={'version':'e2-published-pilot-training-v1','status':'failed','error_type':'TimeoutError',
            'error_code':'resource_limit','message':R.WALL_MESSAGE,'elapsed_seconds':wall+1,
            'known_cost_seconds':wall+1,'optimizer_steps':7,'completed_optimizer_steps':7,
            'epochs_completed':1,'logs':[{'epoch':1}],'seed':job['seed'],'config':config,
            'device':'cuda:0','partial_optimizer_update_may_have_occurred':False,
            'optimizer_update_occurred':True,'resume_exact_supported':False,'retry_performed':False,
            'requires_reconciliation':True,'checkpoint_receipts':{'initial':initial,'latest':latest,'best':best}}
        store.finish('failed',resource);store.close()
        worker={'status':'failed','error_type':'TrainingFailure','reason':R.WALL_MESSAGE,
            'resource_receipt':resource,'automatic_retry':False}
        E.write_json(d/'worker_failure.json',worker);(d/'worker.log').write_text(json.dumps(worker))
        sha=E.digest(spec)
        charge={'kind':'charged','attempt_id':job['logical_job_id'],'status':'failed',
            'reservation_overrun':False,'elapsed_seconds':wall+10,
            'resource_use':{'worker_spec_sha256':sha,'failure':'worker_exit_2','automatic_retry':False}}
        charge['sha256']=E.digest(charge);self.budget._events.append(charge)
        supervision={'worker_spec_sha256':sha,'failure':'worker_exit_2','automatic_retry':False,
            'budget_receipt':charge}
        failure={'version':E.VERSION,'status':'failed','attempt_id':job['logical_job_id'],'job':job,
            'worker_spec_sha256':sha,'supervision':supervision,'automatic_retry':False,
            'artifacts_sha256':{str(p.relative_to(d)):E.file_sha(p) for p in d.rglob('*')
                if p.is_file() and p.name not in ('failure_result.json','.writer.lock')}}
        E.write_json(d/'failure_result.json',failure)
        return failure,resource

    def test_historical_600_keeps_actual_steps_and_cost_under_future_1800(self):
        failure,resource=self.specimen(wall=600)
        before={str(p.relative_to(self.directory)):p.read_bytes() for p in self.directory.rglob('*') if p.is_file()}
        row,result=R.resource_failure_row(self.job,self.directory,self.budget,self.source,range(83))
        self.assertEqual(result,failure);self.assertEqual(row['status'],'failed')
        self.assertEqual(row['resource_use']['optimizer_steps'],7)
        self.assertEqual(row['predictions_produced'],0);self.assertIsNone(row['rmse_original'])
        self.assertFalse(row['automatic_retry']);self.assertFalse(row['replacement_performed'])
        self.assertEqual(self.budget._events[0]['elapsed_seconds'],610)
        self.assertEqual(before,{str(p.relative_to(self.directory)):p.read_bytes() for p in self.directory.rglob('*') if p.is_file()})

    def test_future_1800_is_bound_to_actual_spec_and_receipt(self):
        self.specimen(wall=1800)
        row,_=R.resource_failure_row(self.job,self.directory,self.budget,self.source,range(83))
        self.assertEqual(row['failure_code'],'resource_limit')
        self.assertEqual(self.budget._events[0]['elapsed_seconds'],1810)

    def test_unknown_failure_partial_update_and_changed_charge_are_rejected(self):
        failure,_=self.specimen()
        worker=json.loads((self.directory/'worker_failure.json').read_text())
        worker['error_type']='CUDAKernelError'
        (self.directory/'worker_failure.json').write_text(json.dumps(worker))
        with self.assertRaisesRegex(ValueError,'Unknown'):
            R.resource_failure_row(self.job,self.directory,self.budget,self.source,range(83))
        worker['error_type']='TrainingFailure';worker['resource_receipt']['partial_optimizer_update_may_have_occurred']=True
        (self.directory/'worker_failure.json').write_text(json.dumps(worker))
        with self.assertRaisesRegex(ValueError,'synchronized'):
            R.resource_failure_row(self.job,self.directory,self.budget,self.source,range(83))
        worker['resource_receipt']['partial_optimizer_update_may_have_occurred']=False
        raw=E.canonical(worker)+b'\n';(self.directory/'worker_failure.json').write_bytes(raw)
        self.assertEqual(E.file_sha(self.directory/'worker_failure.json'),failure['artifacts_sha256']['worker_failure.json'])
        self.budget._events[0]['elapsed_seconds']=1
        with self.assertRaisesRegex(ValueError,'charge'):
            R.resource_failure_row(self.job,self.directory,self.budget,self.source,range(83))

    def test_mutated_checkpoint_or_incomplete_phase_is_rejected(self):
        self.specimen();target=next((self.directory/'phases/supervised/checkpoints').glob('*'))
        target.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'artifact changed'):
            R.resource_failure_row(self.job,self.directory,self.budget,self.source,range(83))

    def prepared_modules(self):
        cuda=ModuleType('run_e2_cuda_search');cuda.prepare_panel=lambda _:(None,None)
        temporal=ModuleType('e2_temporal_training')
        temporal.prepare_b=lambda *args:SimpleNamespace(inner_validation=SimpleNamespace(dates=list(range(83))))
        return {'run_e2_cuda_search':cuda,'e2_temporal_training':temporal}

    def cached_next(self):
        job=self.nextjob;d=self.root/'jobs'/job['logical_job_id'];d.mkdir(parents=True)
        result={'rmse_original':.01,'phase_best_epochs':{'supervised':2},'artifacts_sha256':{},
            'cache_reuse':True,'optimizer_steps':0}
        E.write_json(d/'result.json',result);E.write_json(d/'selection_row.json',E._row(job,result,E.file_sha(d/'result.json')))
        return {'fits':[{'model_id':job['model_id'],'origin':job['fold_origin'],
            'configuration_id':job['configuration_id'],'reuse_approved':True}]}

    def test_restored_failure_continues_and_replay_never_supervises_it(self):
        self.specimen();cache=self.cached_next();received=[]
        def selected(rows,*args):
            received.append(rows);return {'winner_configuration_id':'C02'}
        with patch.dict(sys.modules,self.prepared_modules()),patch.object(E,'verify_reduced_registration'), \
                patch.object(E,'source_identity',return_value=self.source),patch.object(E,'job_plan',return_value=[self.job,self.nextjob]), \
                patch.object(E,'select_reduced_hpo',side_effect=selected),patch.object(E,'supervise_command',side_effect=AssertionError('No retry')) as worker:
            first=R.run_selection(self.args,self.budget,cache,'unopened')
            second=R.run_selection(self.args,self.budget,cache,'unopened')
        self.assertEqual(first,second);self.assertFalse(worker.called)
        self.assertEqual([r['status'] for r in received[0]],['failed','completed'])

    def test_new_timeout_is_persisted_and_next_planned_opportunity_runs(self):
        cache=self.cached_next();calls=[];saved=[]
        def supervisor(command,*args,**kwargs):
            calls.append(command)
            spec=json.loads((self.directory/'worker_spec.json').read_text())
            failure,_=self.specimen(spec=spec)
            raise E.ReducedWorkerFailure('worker_exit_2',failure['supervision'])
        with patch.dict(sys.modules,self.prepared_modules()),patch.object(E,'verify_reduced_registration'), \
                patch.object(E,'source_identity',return_value=self.source),patch.object(E,'job_plan',return_value=[self.job,self.nextjob]), \
                patch.object(E,'select_reduced_hpo',return_value={'winner_configuration_id':'C02'}), \
                patch.object(E,'supervise_command',side_effect=supervisor):
            result=R.run_selection(self.args,self.budget,cache,'unopened',persist_result=lambda d,r:saved.append((d,r)))
        self.assertEqual(len(calls),1);self.assertEqual(Path(calls[0][2]),Path(E.__file__))
        self.assertEqual(result['selections']['TimesNet']['winner_configuration_id'],'C02')
        self.assertEqual(saved[0][1]['status'],'failed')
        self.assertEqual(len(saved),2)  # one failed backup, one cached next row
        self.assertEqual(len(self.budget._events),1)

    def test_unknown_new_failure_is_backed_up_then_stops_without_retry(self):
        cache=self.cached_next();saved=[];calls=[]
        def supervisor(command,*args,**kwargs):
            calls.append(command)
            spec=json.loads((self.directory/'worker_spec.json').read_text())
            failure,_=self.specimen(spec=spec)
            raise E.ReducedWorkerFailure('worker_exit_2',failure['supervision'])
        with patch.dict(sys.modules,self.prepared_modules()),patch.object(E,'verify_reduced_registration'), \
                patch.object(E,'source_identity',return_value=self.source),patch.object(E,'job_plan',return_value=[self.job,self.nextjob]), \
                patch.object(E,'supervise_command',side_effect=supervisor), \
                patch.object(R,'resource_failure_row',side_effect=ValueError('Unknown failure requires inspection')):
            with self.assertRaisesRegex(ValueError,'Unknown failure'):
                R.run_selection(self.args,self.budget,cache,'unopened',persist_result=lambda d,r:saved.append((d,r)))
        self.assertEqual(len(calls),1);self.assertEqual(len(saved),1)
        self.assertEqual(saved[0][1]['status'],'failed')
        self.assertFalse((self.directory/'selection_row.json').exists())

    def test_matching_restored_transport_receipt_is_allowed_changed_identity_rejected(self):
        failure,_=self.specimen()
        backup={'status':'backed_up','attempt_id':self.job['logical_job_id'],
            'repository':'jassan-bit/DeepLearning_202630','branch':'results/e2-reduced-delivery-v1',
            'manifest_path':'runs/'+self.job['logical_job_id']+'/manifest.json',
            'artifact_count':len(failure['artifacts_sha256']),'retry_scientific_fit':False,
            'manifest_sha256':'a'*64}
        path=self.directory/'github_backup_receipt.json';E.write_json(path,backup)
        row,_=R.resource_failure_row(self.job,self.directory,self.budget,self.source,range(83))
        self.assertEqual(row['status'],'failed')
        backup['repository']='another/repository';path.write_text(json.dumps(backup))
        with self.assertRaisesRegex(ValueError,'transport receipt'):
            R.resource_failure_row(self.job,self.directory,self.budget,self.source,range(83))

if __name__=='__main__':unittest.main()
