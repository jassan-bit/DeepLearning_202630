"""Controller contracts with synthetic files and mocked HTTP/CUDA workers."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('extended_controller_tested',Path(__file__).with_name('controller.py'))
controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
SOURCE='a'*40;OLD_SOURCE='b'*40;RUN='20261119T020000000000Z';RUN2='20261119T020001000000Z'


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.source=self.root/'source';self.source.mkdir()
        self.api=SimpleNamespace(REPOSITORY=controller.REPOSITORY,_token=lambda value:value,
            validate_credentials=lambda token:None,StorageError=RuntimeError)
    def tearDown(self):self.temp.cleanup()
    def write(self,path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(value if isinstance(value,bytes) else controller.canonical(value));return path

    def test_stable_view_excludes_live_checkpoints_and_partial_csv(self):
        out=self.root/'run';view=self.root/'view'
        self.write(out/'jobs/D00/complete/result.json',{'status':'completed'})
        self.write(out/'jobs/D00/complete/supervised/checkpoint.pt',b'closed-checkpoint')
        self.write(out/'jobs/D00/active/supervised/checkpoint.pt',b'active-checkpoint')
        self.write(out/'resources.csv',b'currently-being-written')
        self.write(out/'admission/datasets/D01.csv',b'closed admitted CSV')
        self.write(out/'inputs/daily_market_development.csv',b'closed known source CSV')
        self.write(out/'worker.log',b'closed log prefix\n')
        self.write(out/'evaluation_receipt.json',{'status':'stale_failed_receipt'})
        controller.stable_view(out,view)
        self.assertEqual((view/'jobs/D00/complete/supervised/checkpoint.pt').read_bytes(),b'closed-checkpoint')
        self.assertFalse((view/'jobs/D00/active').exists())
        self.assertFalse((view/'resources.csv').exists())
        self.assertEqual((view/'admission/datasets/D01.csv').read_bytes(),b'closed admitted CSV')
        self.assertEqual((view/'inputs/daily_market_development.csv').read_bytes(),b'closed known source CSV')
        self.assertFalse((view/'evaluation_receipt.json').exists())
        self.assertEqual((view/'worker.log').read_bytes(),b'closed log prefix\n')
        controller.stable_view(out,view,finished=True)
        self.assertTrue((view/'jobs/D00/active/supervised/checkpoint.pt').is_file())

    def test_resumed_partial_panel_does_not_replace_previously_closed_seed_set(self):
        out=self.root/'run';view=self.root/'view';name='panels/D01/MLP16.csv'
        def manifest(raw,seeds):
            self.write(out/name,raw)
            self.write(out/'panel_manifest.json',{'seeds':[1,2],
                'files':[{'path':name,'sha256':hashlib.sha256(raw).hexdigest(),
                          'bytes':len(raw),'rows':len(seeds),'observed_seeds':seeds}]})
        manifest(b'full two seeds',[1,2]);controller.stable_view(out,view)
        manifest(b'first seed regenerated',[1]);controller.stable_view(out,view)
        self.assertEqual((view/name).read_bytes(),b'full two seeds')
        saved=json.loads((view/'panel_manifest.json').read_bytes())
        self.assertEqual(saved['files'][0]['observed_seeds'],[1,2])
        self.assertEqual(saved['total_rows'],2)

    def test_final_panel_manifest_preserves_exact_bytes_bound_by_worker(self):
        out=self.root/'run';view=self.root/'view';name='panels/D01/MLP16.csv';raw=b'complete panel'
        self.write(out/name,raw)
        manifest={'files':[{'path':name,'sha256':hashlib.sha256(raw).hexdigest(),
                           'bytes':len(raw),'rows':2,'observed_seeds':[1,2]}],
                  'seeds':[1,2],'total_rows':2,'producer_order_is_significant_for_hash':True}
        original=(json.dumps(manifest,indent=4,sort_keys=False)+'\n').encode()
        self.write(out/'panel_manifest.json',original)
        controller.stable_view(out,view)
        self.assertNotEqual((view/'panel_manifest.json').read_bytes(),original)
        controller.stable_view(out,view,finished=True)
        self.assertEqual((view/'panel_manifest.json').read_bytes(),original)
        self.assertEqual(hashlib.sha256((view/'panel_manifest.json').read_bytes()).hexdigest(),
                         hashlib.sha256(original).hexdigest())

    def test_public_ancestry_uses_real_compare_schema_without_head_commit(self):
        self.api._request=lambda *args:{'status':'ahead','base_commit':{'sha':OLD_SOURCE},
                                       'merge_base_commit':{'sha':OLD_SOURCE},'commits':[{'sha':SOURCE}]}
        controller._main_ancestry(self.api,'fixture',OLD_SOURCE,SOURCE)
        self.api._request=lambda *args:{'status':'diverged','base_commit':{'sha':OLD_SOURCE},
                                       'merge_base_commit':{'sha':'c'*40}}
        with self.assertRaisesRegex(ValueError,'ancestor'):
            controller._main_ancestry(self.api,'fixture',OLD_SOURCE,SOURCE)

    def test_storage_failure_stops_only_the_current_worker(self):
        class Worker:
            def __init__(self):
                self.stdout=iter(['JOB_COMPLETED one\n','JOB_COMPLETED two\n','JOB_COMPLETED three\n'])
                self.code=None;self.terminated=False
            def poll(self):return self.code
            def terminate(self):self.terminated=True;self.code=-15
            def wait(self,timeout=None):
                if self.code is None:self.code=0
                return self.code
        worker=Worker()
        def failure():raise RuntimeError('simulated storage outage')
        with patch.object(controller.subprocess,'Popen',return_value=worker):
            with self.assertRaisesRegex(RuntimeError,'storage outage'):
                controller.stream(['/unused/python'],self.root,{},self.root/'worker.log',failure)
        self.assertTrue(worker.terminated)

    def test_completed_remote_run_returns_without_gpu_setup_or_gigabyte_restore(self):
        row={'master_run_id':RUN,'source_commit':SOURCE,'stage_status':'completed',
             'analysis_complete':True,'run_url':'https://github.com/fixture/completed'}
        def forbidden(*args,**kwargs):self.fail('Completed run must not start computation or restore weights')
        continuation=SimpleNamespace(discover_runs=lambda *args:[row],restore=forbidden)
        def load(path,name):
            return self.api if Path(path).name=='colab_github.py' else continuation
        with patch.object(controller,'checkout',return_value=self.source),patch.object(controller,'load',side_effect=load),\
             patch.object(controller,'_gpu_profile',side_effect=forbidden):
            result=controller.run(SOURCE,'fixture-token',base=self.root/'colab')
        self.assertEqual(result['status'],'already_completed')

    def test_original_source_controller_is_used_to_resume_after_main_changes(self):
        calls=[];row={'master_run_id':RUN,'source_commit':OLD_SOURCE,'stage_status':'running',
                      'analysis_complete':False,'protocol_sha256':'d'*64}
        continuation=SimpleNamespace(discover_runs=lambda *args:[row])
        old=SimpleNamespace(run=lambda *args,**kwargs:calls.append((args,kwargs)) or {'status':'delegated'})
        self.api._request=lambda *args:{'status':'ahead','base_commit':{'sha':OLD_SOURCE},
                                       'merge_base_commit':{'sha':OLD_SOURCE}}
        def load(path,name):
            if Path(path).name=='colab_github.py':return self.api
            if Path(path).name=='continuation.py':return continuation
            if Path(path).name=='controller.py':return old
            self.fail('No admission or model code should run before source handoff')
        with patch.object(controller,'checkout',return_value=self.source),patch.object(controller,'load',side_effect=load):
            result=controller.run(SOURCE,'fixture-token',base=self.root/'colab')
        self.assertEqual(result['status'],'delegated')
        self.assertEqual(calls[0][0][0],OLD_SOURCE)
        self.assertEqual(calls[0][1]['master_run_id'],RUN)

    def test_unknown_explicit_master_never_becomes_an_accidental_new_campaign(self):
        continuation=SimpleNamespace(discover_runs=lambda *args:[])
        def load(path,name):
            if Path(path).name=='colab_github.py':return self.api
            if Path(path).name=='continuation.py':return continuation
            self.fail('Unknown requested master must stop before admission or CUDA setup')
        with patch.object(controller,'checkout',return_value=self.source),patch.object(controller,'load',side_effect=load):
            with self.assertRaisesRegex(ValueError,'accidental new campaign'):
                controller.run(SOURCE,'fixture-token',base=self.root/'colab',master_run_id=RUN)

    def test_incomplete_admission_retries_preserve_one_audit_and_create_a_new_data_attempt(self):
        saves=[];continuation=SimpleNamespace(discover_runs=lambda *args:[])
        def admit(source,output,sha):
            self.write(output/'stage_receipt.json',{'datasets_admitted':2})
            return {'datasets_admitted':2}
        admission=SimpleNamespace(run_stage=admit)
        store=SimpleNamespace(save=lambda output,token,sha,run_id,api,profile:
                              saves.append(run_id) or {'run_url':'https://github.com/fixture/'+run_id})
        def load(path,name):
            name=Path(path).name
            if name=='colab_github.py':return self.api
            if name=='continuation.py':return continuation
            if name=='protocol.py':return SimpleNamespace()
            if name=='colab.py':return admission
            if name=='github_store.py':return store
            self.fail('Physical-admission failure must stop before GPU setup')
        base=self.root/'colab'
        with patch.object(controller,'checkout',return_value=self.source),patch.object(controller,'load',side_effect=load),\
             patch.object(controller,'stamp',side_effect=[RUN,RUN2]):
            first=controller.run(SOURCE,'fixture-token',base=base)
            second=controller.run(SOURCE,'fixture-token',base=base)
        self.assertEqual(first['new_real_data_fits'],0)
        self.assertEqual(second['status'],'blocked_at_physical_admission')
        self.assertEqual(saves,[RUN,RUN2])
        self.assertEqual(json.loads((base/'e3_extended_audit_session.json').read_bytes())['master_run_id'],RUN)

    def test_changed_gpu_cohort_stops_before_new_optimization(self):
        base=self.root/'colab';out=base/('e3-extended-audit-'+RUN)
        self.write(out/'protocol.json',{'compute_profile':{'device':'cuda:0','GPU':'original-A100'}})
        self.write(base/('e3-extended-continuation-'+RUN+'.json'),{'fixture':True})
        self.write(out/'jobs/known/checkpoint.pt',b'preserved original weights')
        row={'master_run_id':RUN,'source_commit':SOURCE,'stage_status':'running',
             'analysis_complete':False,'protocol_sha256':'d'*64}
        continuation=SimpleNamespace(discover_runs=lambda *args:[row])
        runtime=SimpleNamespace(prepare_runtime=lambda path:(Path('/unused/gpu-python'),{},{}))
        def reject(protocol,actual):raise ValueError('GPU/software cohort changed')
        protocol=SimpleNamespace(verify_compute_profile=reject)
        def load(path,name):
            name=Path(path).name
            if name=='colab_github.py':return self.api
            if name=='continuation.py':return continuation
            if name=='protocol.py':return protocol
            if name=='colab_runtime.py':return runtime
            self.fail('GPU mismatch must stop before admission restart, freeze or models')
        with patch.object(controller,'checkout',return_value=self.source),patch.object(controller,'load',side_effect=load),\
             patch.object(controller,'_gpu_profile',return_value={'device':'cuda:0','GPU':'other-L4'}):
            with self.assertRaisesRegex(ValueError,'cohort changed'):
                controller.run(SOURCE,'fixture-token',base=base)
        self.assertEqual((out/'jobs/known/checkpoint.pt').read_bytes(),b'preserved original weights')

    def test_protocol_present_without_final_restore_state_is_restored_again(self):
        base=self.root/'colab';out=base/('e3-extended-audit-'+RUN);events=[]
        self.write(out/'protocol.json',{'compute_profile':{'device':'cuda:0','GPU':'original-A100'}})
        row={'master_run_id':RUN,'source_commit':SOURCE,'stage_status':'running',
             'analysis_complete':False,'protocol_sha256':'d'*64,'run_url':'https://github.com/fixture/run'}
        def restore(output,*args,**kwargs):
            events.append('restore')
            self.assertTrue((output/'protocol.json').is_file())
            self.write(output/'resources.csv',b'previously missing trailing restore payload')
            self.write(kwargs['statefile'],{'fixture':'restore completion marker'})
        def runtime(path):
            events.append('runtime')
            self.assertTrue((out/'resources.csv').is_file())
            raise RuntimeError('stop after restoration')
        continuation=SimpleNamespace(discover_runs=lambda *args:[row],restore=restore)
        def load(path,name):
            name=Path(path).name
            if name=='colab_github.py':return self.api
            if name=='continuation.py':return continuation
            if name=='protocol.py':return SimpleNamespace()
            if name=='colab_runtime.py':return SimpleNamespace(prepare_runtime=runtime)
            self.fail('Partial restoration must finish before other scientific work')
        with patch.object(controller,'checkout',return_value=self.source),patch.object(controller,'load',side_effect=load):
            with self.assertRaisesRegex(RuntimeError,'stop after restoration'):
                controller.run(SOURCE,'fixture-token',base=base)
        self.assertEqual(events,['restore','runtime'])

    def test_interrupted_package_download_is_retained_then_reacquired_and_verified(self):
        base=self.root/'colab';base.mkdir();attempts=[];inventories=[]
        raw=b'v'*7839993;expected=hashlib.sha256(raw).hexdigest()
        destination=base/'e3-immutable-source-package-c4a4a1e68e98'
        def acquire(folder):
            attempts.append(folder);folder.mkdir()
            if len(attempts)==1:
                self.write(folder/'02_codigo_modelos.zip',b'interrupted download')
                raise OSError('synthetic interrupted download')
            self.write(folder/'02_codigo_modelos.zip',raw)
            package=folder/'extracted/package'
            self.write(package/'code/scripts/e2_standalone_code.py',b'verified fixture source')
            return package
        def inventory(package):
            inventories.append(package)
            self.assertEqual((package/'code/scripts/e2_standalone_code.py').read_bytes(),b'verified fixture source')
            return {'status':'verified','synthetic_fixture':True},{},{}
        preflight=SimpleNamespace(PACKAGE_SHA256=expected,acquire_package=acquire,portable_inventory=inventory)
        with patch.object(controller,'load',return_value=preflight),patch.object(controller,'stamp',return_value=RUN):
            with self.assertRaisesRegex(OSError,'interrupted download'):
                controller._package_cache(self.source,base)
            receipt=controller._package_cache(self.source,base)
            cached=controller._package_cache(self.source,base)
        quarantine=base/(destination.name+'-quarantine-'+RUN)
        self.assertEqual((quarantine/'02_codigo_modelos.zip').read_bytes(),b'interrupted download')
        self.assertEqual(len(attempts),2);self.assertEqual(len(inventories),2)
        self.assertEqual(receipt['archive_sha256'],expected)
        self.assertEqual(receipt['quarantined_cache'],str(quarantine))
        self.assertTrue(receipt['source_verified']);self.assertEqual(receipt['new_real_data_fits'],0)
        self.assertIsNone(cached['quarantined_cache'])

    def test_unpinned_fresh_archive_is_refused_before_inventory_or_fitting(self):
        base=self.root/'colab';base.mkdir()
        def acquire(folder):
            folder.mkdir();self.write(folder/'02_codigo_modelos.zip',b'w'*7839993)
        def forbidden(*args):self.fail('Inventory must not run on an archive with an unpinned digest')
        preflight=SimpleNamespace(PACKAGE_SHA256='0'*64,acquire_package=acquire,portable_inventory=forbidden)
        with patch.object(controller,'load',return_value=preflight):
            with self.assertRaisesRegex(ValueError,'pinned size/hash'):
                controller._package_cache(self.source,base)
        self.assertTrue((base/'e3-immutable-source-package-c4a4a1e68e98/02_codigo_modelos.zip').is_file())

    def test_package_preparation_uses_the_isolated_scientific_interpreter(self):
        receipt={'path':str(self.root/'verified-package'),'source_verified':True}
        with patch.object(controller.subprocess,'check_output',return_value=json.dumps(receipt)) as call:
            package,saved=controller.prepare_package(self.source,self.root,Path('/isolated/gpu/python'),{'SAFE':'yes'})
        self.assertEqual(call.call_args.args[0][0],'/isolated/gpu/python')
        self.assertEqual(call.call_args.kwargs['env'],{'SAFE':'yes'})
        self.assertEqual(package,self.root/'verified-package');self.assertEqual(saved,receipt)

    def test_scientific_environment_excludes_credentials_and_python_injection(self):
        with patch.dict(os.environ,{'GITHUB_TOKEN':'never-forward','MY_PASSWORD':'never-forward',
                                   'PYTHONPATH':'unsafe','UV_INDEX_URL':'unsafe','NORMAL_FLAG':'keep'}):
            env=controller.environment()
        for name in ('GITHUB_TOKEN','MY_PASSWORD','PYTHONPATH','UV_INDEX_URL'):self.assertNotIn(name,env)
        self.assertEqual(env['NORMAL_FLAG'],'keep')

    def test_analysis_installer_targets_an_isolated_pipless_cpu_venv(self):
        source=self.root/'source';lock=source/'deployment/science/requirements.lock.txt'
        self.write(lock,b'numpy==2.3.5\nmatplotlib==3.10.8\n')
        commands=[]
        with patch.object(controller,'stream',side_effect=lambda args,*rest,**kw:commands.append([str(v) for v in args])),\
             patch.object(controller.subprocess,'run',side_effect=[SimpleNamespace(returncode=1),SimpleNamespace(returncode=0)]):
            python,env,receipt=controller.analysis_runtime(source,self.root/'colab',self.root/'analysis.log')
        self.assertIn('--without-pip',commands[0])
        self.assertEqual(commands[1][1:5],['-m','pip','--python',str(python)])
        self.assertFalse(receipt['GPU_runtime_changed'])
        self.assertIn('MPLCONFIGDIR',env)

    def test_three_jobs_full_controller_order_and_final_analysis_persist(self):
        fixture_path=Path(__file__).with_name('test_protocol.py')
        fixture_spec=importlib.util.spec_from_file_location('controller_tag_fixture',fixture_path)
        fixture=importlib.util.module_from_spec(fixture_spec);fixture_spec.loader.exec_module(fixture)
        api=fixture.FakeAPI();api._token=lambda value:value;api.validate_credentials=lambda token:None
        events=[];snapshots=[];base=self.root/'colab';output=base/('e3-extended-audit-'+RUN)
        def admitted(source,folder,sha):
            rows=[]
            for identity in ('D01','D02','D03'):
                raw=b'date,open,high,low,close,volume\n'
                self.write(folder/'data/datasets'/(identity+'.csv'),raw)
                rows.append({'id':identity,'admitted':True,'path':'datasets/'+identity+'.csv',
                             'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                             'target_values_computed':False})
            self.write(folder/'data/admission.json',{'kind':'e3_additional_OHLCV_admission',
                'source_commit':SOURCE,'datasets_required':3,'datasets_admitted':3,
                'datasets_evaluated_by_models':0,'target_values_computed':False,'datasets':rows})
            return {'datasets_admitted':3}
        def save_changed(view,*args,**kwargs):
            snapshot={p.relative_to(view).as_posix():p.read_bytes() for p in Path(view).rglob('*') if p.is_file()}
            snapshots.append(snapshot);events.append('persist')
            return {'head_part':RUN,'changed_files':len(snapshot),'all_files_verified':True}
        def history(folder):
            paths=(folder/'development.csv',folder/'evaluation.csv')
            for p in paths:self.write(p,b'synthetic-history')
            return paths,{'fixture':True}
        continuation=SimpleNamespace(discover_runs=lambda *args:[],save_changed=save_changed)
        def load(path,name):
            name=Path(path).name
            if name=='colab_github.py':return api
            if name=='continuation.py':return continuation
            if name=='protocol.py':return fixture.protocol
            if name=='colab_runtime.py':return SimpleNamespace(prepare_runtime=lambda path:
                (Path('/isolated/gpu/python'),{'GITHUB_TOKEN':'never-child','SAFE':'yes'},{'fixture':True}))
            if name=='github_store.py':return SimpleNamespace(save=lambda *args,**kwargs:{'run_url':'https://github.com/fixture/admission'})
            if name=='colab.py' and '/admission/' in str(path):return SimpleNamespace(run_stage=admitted)
            if name=='colab.py':return SimpleNamespace(admit_npz=lambda base:base/'fixed.npz')
            if name=='d00_history.py':return SimpleNamespace(acquire=history)
            raise AssertionError(path)
        parent=self
        class Worker:
            def __init__(self,command,**kwargs):
                self.code=None
                parent.assertNotIn('GITHUB_TOKEN',kwargs['env'])
                name=Path(command[1]).name
                def lines():
                    if name=='run.py':
                        events.append('worker')
                        parent.assertEqual(command[command.index('--package-path')+1],str(base/'verified-package'))
                        for i in range(3):
                            job=output/'jobs/D00'/('model'+str(i))
                            parent.write(job/'result.json',{'status':'completed'})
                            parent.write(job/'supervised/checkpoint.pt',b'synthetic-checkpoint')
                            yield 'JOB_COMPLETED '+job.relative_to(output).as_posix()+'\n'
                        parent.write(output/'evaluation_receipt.json',{'status':'completed'})
                        parent.write(output/'run_manifest.json',{'fixture':True})
                        for n in ('resources.csv','sample_curve_predictions.csv','HPO_sensitivity.csv'):
                            parent.write(output/n,b'fixture\n')
                        parent.write(output/'sample_curve_jobs.json',[])
                        parent.write(output/'panel_manifest.json',{'seeds':[1,2], 'files':[], 'total_rows':0})
                    elif name=='analyze.py':
                        events.append('analysis')
                        parent.assertIn('--hpo-sensitivity',command)
                        parent.write(output/'analysis/analysis_receipt.json',{'R3_accredited':False})
                    elif name=='plots.py':
                        events.append('figures')
                        parent.write(output/'figures/figures_manifest.json',{'fixture':True})
                    else:raise AssertionError(name)
                self.stdout=lines()
            def poll(self):return self.code
            def wait(self,timeout=None):self.code=0;return 0
            def terminate(self):self.code=-15
        profile={'device':'cuda:0','GPU':'synthetic-fixture'}
        with patch.object(controller,'checkout',return_value=self.source),patch.object(controller,'load',side_effect=load),\
             patch.object(controller,'stamp',return_value=RUN),patch.object(controller,'_gpu_profile',return_value=profile),\
             patch.object(controller,'prepare_package',return_value=(base/'verified-package',{'source_verified':True})),\
             patch.object(controller,'analysis_runtime',return_value=(Path('/isolated/analysis/python'),{},{})),\
             patch.object(controller.subprocess,'Popen',side_effect=Worker):
            result=controller.run(SOURCE,'fixture-token',base=base)
        self.assertEqual(result['status'],'completed')
        self.assertFalse(result['R3_accredited'])
        self.assertLess(events.index('persist'),events.index('worker'))
        self.assertLess(events.index('worker'),events.index('analysis'))
        self.assertEqual(sum(p.startswith('jobs/') and p.endswith('result.json') for p in snapshots[1]),3)
        self.assertIn('admission/datasets/D03.csv',snapshots[0])
        self.assertIn('inputs/development.csv',snapshots[0])
        self.assertIn('immutable_package.json',snapshots[0])
        self.assertIn('figures/figures_manifest.json',snapshots[-1])
        status=json.loads(snapshots[-1]['controller_status.json'])
        self.assertEqual(status['stage_status'],'completed');self.assertTrue(status['analysis_complete'])


if __name__=='__main__':unittest.main()
