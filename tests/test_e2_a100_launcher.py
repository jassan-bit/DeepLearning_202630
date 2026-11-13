"""CPU orchestration checks; no GPU, model data, optimizer or GitHub write."""
from contextlib import contextmanager
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
launcher=load('a100_launcher','notebooks/run_e2_a100_recovery_colab.py')
driver=load('a100_driver','scripts/run_e2_a100_recovery_delivery.py')


class A100LauncherTests(unittest.TestCase):
    def test_prepare_only_never_opens_funding_or_recovery(self):
        namespace={}
        def prepare(ns):ns['GPU_PYTHON']='/isolated/python'
        with patch.object(launcher,'_prepare_runtime',side_effect=prepare), \
                patch.object(launcher,'_funding_launcher',side_effect=AssertionError('GitHub execution requested')):
            result=launcher.run_e2_a100_recovery_colab('a'*40,namespace=namespace,execute=False)
        self.assertFalse(result['training_started']);self.assertFalse(result['recovery_requested'])
        self.assertEqual(result['GPU_PYTHON'],'/isolated/python')

    def test_existing_environment_is_checked_for_actual_a100_without_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            python=Path(directory)/'python';python.write_bytes(b'fixture')
            ns={'GPU_PYTHON':str(python)}
            with patch.dict(launcher.os.environ,{'GITHUB_TOKEN':'secret-marker','GH_TOKEN':'secret-marker'}), \
                    patch.object(launcher.subprocess,'run',return_value=SimpleNamespace(stdout='NVIDIA A100-SXM4-40GB')) as run:
                launcher._prepare_runtime(ns)
            command=run.call_args.args[0]
            self.assertIn("'A100'",command[2])
            self.assertNotIn('GITHUB_TOKEN',run.call_args.kwargs['env'])
            self.assertNotIn('GH_TOKEN',run.call_args.kwargs['env'])

    def test_corrupt_preparation_download_is_not_executed(self):
        with patch.object(launcher.Path,'is_file',return_value=False), \
                patch.object(launcher,'PREPARATION_COMMIT','a'*40), \
                patch.object(launcher,'PREPARATION_SHA256','0'*64), \
                patch.object(launcher.urllib.request,'urlopen',return_value=io.BytesIO(b'wrong')):
            with self.assertRaisesRegex(RuntimeError,'source differs'):
                launcher._prepare_runtime({})

    def test_launcher_materializes_original_p_and_separate_q_and_routes_parent(self):
        seen=[]
        base={'LOCAL_WORKSPACE':Path('/scratch'),'CONTROLLERS':set(),
            '_checkout':lambda commit,*a:seen.append(('checkout',commit)) or Path('/engine'),
            '_stream':lambda command,*a:seen.append(('stream',command)) or 0}
        def extension(commit,workspace):
            seen.append(('extension',commit,base['EXTENSION_REGISTRATION'],base['EXTENSION_FILES']))
            return workspace/('github-primary-extension-'+commit)
        base['_extension']=extension
        funding={'_base_launcher':lambda:base}
        def funded_run(commit,**kwargs):
            b=funding['_base_launcher']()
            b['EXTENSION_REGISTRATION']=funding['EXTENSION_REGISTRATION']
            b['EXTENSION_FILES']=funding['EXTENSION_FILES']
            b['_checkout'](launcher.ENGINE_COMMIT,None,None)
            b['_extension'](commit,b['LOCAL_WORKSPACE'])
            b['_stream'](['python','/extension/run_e2_reduced_github72_delivery.py','--engine-root','/engine'])
            return {'execute_requested':kwargs['execute']}
        funding['run_reduced_github72_colab']=funded_run
        with patch.object(launcher,'_prepare_runtime'),patch.object(launcher,'_funding_launcher',return_value=funding), \
                patch.object(launcher,'materialize_pinned_metadata',return_value={'sha256':launcher.METADATA_SHA256}):
            result=launcher.run_e2_a100_recovery_colab('b'*40,execute=True)
        self.assertTrue(result['execute_requested'])
        ext=[x for x in seen if x[0]=='extension']
        self.assertEqual(ext[0][1:3],(launcher.RESOURCE_COMMIT,launcher.RESOURCE_REGISTRATION))
        self.assertEqual(ext[0][3],launcher.RESOURCE_FILES)
        self.assertEqual(ext[1][1:3],('b'*40,launcher.EXTENSION_REGISTRATION))
        self.assertEqual(ext[1][3],launcher.EXTENSION_FILES)
        command=next(x[1] for x in seen if x[0]=='stream')
        self.assertEqual(Path(command[1]).name,'run_e2_a100_recovery_delivery.py')
        self.assertEqual(command[-2:],['--resource-extension-root',Path('/scratch/github-primary-extension-'+launcher.RESOURCE_COMMIT)])
        self.assertIn('run_e2_a100_recovery_delivery.py',base['CONTROLLERS'])

    def test_published_registration_rejects_changed_or_escaped_runtime_sources(self):
        with tempfile.TemporaryDirectory() as directory,tempfile.TemporaryDirectory() as outside:
            root=Path(directory);relative='scripts/runtime.py';path=root/relative;path.parent.mkdir()
            path.write_bytes(b'valid');record={'engine_commit':driver.ENGINE_COMMIT,'protocol_id':'e2-reduced-delivery-v1',
                'files_sha256':{relative:hashlib.sha256(b'valid').hexdigest()}}
            raw=json.dumps(record).encode();(root/'registration.json').write_bytes(raw)
            with patch.object(driver.subprocess,'check_output',return_value=raw):
                driver.verify_published_registration(Path('/engine'),root,'c'*40,'registration-path')
                path.write_bytes(b'invalid')
                with self.assertRaisesRegex(ValueError,'runtime source differs'):
                    driver.verify_published_registration(Path('/engine'),root,'c'*40,'registration-path')
                other=Path(outside)/'runtime.py';other.write_bytes(b'valid');path.unlink();path.symlink_to(other)
                with self.assertRaisesRegex(ValueError,'runtime source differs'):
                    driver.verify_published_registration(Path('/engine'),root,'c'*40,'registration-path')

    def test_routed_workers_keep_original_p_proof_not_parent_q(self):
        calls=[]
        route=driver.resource_worker_router(lambda *a,**k:calls.append((a,k)),'/engine','/resource-p',driver.RESOURCE_COMMIT)
        route(['python','-u','/engine/scripts/run_e2_reduced_selection.py','--worker-spec','spec'],
            'budget','job','output','specsha',phases=1)
        command=calls[0][0][0]
        self.assertEqual(command[command.index('--extension-commit')+1],driver.RESOURCE_COMMIT)
        self.assertEqual(command[command.index('--extension-root')+1],Path('/resource-p'))
        self.assertEqual(calls[0][0][1:],('budget','job','output','specsha'))

    def _driver_fixture(self,current_gpu='NVIDIA A100-SXM4-40GB'):
        calls=[];q='d'*40
        args=SimpleNamespace(execute=True,stage='all',output=Path('/output'),data_root='dev',source_roots='roots',
            runtime_profile='runtime',budget_evidence='budget',migration='migration',cache_audit='current-audit',
            reconciliation_root='recon',migration_registration_commit=driver.ENGINE_COMMIT,inherited_ledger=[])
        selected=SimpleNamespace(run_selection='old',supervise_command=lambda *a,**k:None)
        run_selection=lambda *a,**k:None
        def read(path):return {'runtime':{'device_name':current_gpu}} if path=='runtime' else {'file':path}
        scientific=SimpleNamespace(parser=lambda:SimpleNamespace(parse_args=lambda _:args),read_json=read,
            execute_delivery=lambda *a:calls.append(('execute',selected.run_selection)))
        class Budget:
            _closed=False;stage_wall_seconds=1800
            def report(self):return {'reserved_seconds':0}
            def charge_overhead(self,*a,**k):pass
        class Backup:
            def __init__(self):calls.append(('client-created',))
            def preflight(self):calls.append(('preflight',))
        class State:
            def __init__(self,*a):pass
            def restore(self):calls.append(('restore',))
            @contextmanager
            def budget_factory(self,*a,budget_class=None,**k):yield budget_class('/budget',{})
            def snapshot(self,reason,**k):calls.append(('snapshot',reason))
        def funded(policy,**kwargs):
            self.assertEqual(kwargs['funding_registration_commit'],driver.FUNDING_COMMIT)
            calls.append(('funding',driver.FUNDING_COMMIT));return lambda *a,**k:Budget()
        def resources(parent,policy,**kwargs):
            self.assertEqual(kwargs['registration_commit'],driver.RESOURCE_COMMIT)
            self.assertEqual(kwargs['extension_root'],Path('/resource-p'))
            calls.append(('resource',driver.RESOURCE_COMMIT));return parent
        def recovered(*a,**kwargs):
            self.assertEqual(kwargs['registration_commit'],q)
            calls.append(('recover',q));return {'status':'recovered','training_started':False}
        def historical(*a,**kwargs):
            self.assertEqual(kwargs['registration_commit'],q)
            calls.append(('historical',q));return {'cache_audit':'historical-audit'}
        modules={'run_e2_reduced_delivery':scientific,'run_e2_reduced_selection':selected,
            'e2_reduced_protocol':SimpleNamespace(verify_reduced_registration=lambda *a,**k:None),
            'e2_resilient_github_backup':SimpleNamespace(ResilientGitHubBackup=Backup),
            'e2_reduced_github_state':SimpleNamespace(GitHubState=State),
            'e2_registered_budget_extension':SimpleNamespace(make_funded_budget_class=funded),
            'e2_registered_resource_extension':SimpleNamespace(make_resource_budget_class=resources),
            'reconcile_e2_github_completed':SimpleNamespace(reconcile_completed_local=lambda *a:calls.append(('completed-reconciliation',))),
            'e2_expired_lease_recovery':SimpleNamespace(reconcile_expired_lease=recovered),
            'e2_a100_cache_selection':SimpleNamespace(prepare_historical_cache_audit=historical,run_selection=run_selection)}
        record={'funding_policy':{'hours':72},'resource_policy':{'seconds':1800},'expired_lease_policy':{'charge':1860}}
        return q,args,calls,modules,record,run_selection

    def test_driver_recovers_before_restore_and_preserves_original_h_and_p_anchors(self):
        q,args,calls,modules,record,run_selection=self._driver_fixture()
        with patch.dict(sys.modules,modules),patch.object(driver,'_engine_root',return_value=Path('/engine')), \
                patch.object(driver,'_extension_root',return_value=(Path('/extension-q'),record)), \
                patch.object(driver,'verify_published_registration',return_value=record), \
                patch.object(driver.subprocess,'run') as fetch:
            result=driver.main(['--engine-root','/engine','--extension-root','/extension-q','--extension-commit',q,
                '--resource-extension-root','/resource-p'])
        self.assertEqual(result,0)
        self.assertEqual([c.args[0][-1] for c in fetch.call_args_list],[driver.FUNDING_COMMIT,driver.RESOURCE_COMMIT,q])
        self.assertLess(calls.index(('recover',q)),calls.index(('completed-reconciliation',)))
        self.assertLess(calls.index(('recover',q)),calls.index(('restore',)))
        self.assertLess(calls.index(('restore',)),calls.index(('historical',q)))
        self.assertLess(calls.index(('historical',q)),calls.index(('snapshot','hardware_transition_historical_cache_admitted')))
        self.assertEqual(calls[-1],('execute',run_selection))

    def test_driver_rejects_actual_t4_before_any_backup_or_recovery_write(self):
        q,args,calls,modules,record,run_selection=self._driver_fixture('Tesla T4')
        with patch.dict(sys.modules,modules),patch.object(driver,'_engine_root',return_value=Path('/engine')), \
                patch.object(driver,'_extension_root',return_value=(Path('/extension-q'),record)), \
                patch.object(driver,'verify_published_registration',return_value=record),patch.object(driver.subprocess,'run'):
            result=driver.main(['--engine-root','/engine','--extension-root','/extension-q','--extension-commit',q,
                '--resource-extension-root','/resource-p'])
        self.assertEqual(result,3)
        self.assertNotIn(('client-created',),calls)
        self.assertFalse(any(x[0]=='recover' for x in calls))


if __name__=='__main__':unittest.main()
