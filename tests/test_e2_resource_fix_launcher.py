"""Synthetic metadata and orchestration checks; no market, GPU or network."""
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

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


launcher = load('resource_fix_launcher', 'notebooks/run_e2_reduced_resource_fix_colab.py')
driver = load('resource_fix_driver', 'scripts/run_e2_reduced_github72_delivery.py')


class ResourceLauncherTests(unittest.TestCase):
    def test_summary_claims_72_only_for_checksummed_original_funding_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            result={'engine_commit':launcher.ENGINE_COMMIT,'temporary_workspace':directory,'budget_extended':False}
            self.assertNotIn('effective_total_budget_hours',launcher.verified_funding_summary(dict(result)))
            anchor={'version':'e2-registered-budget-extension-v1',
                'funding_policy_sha256':launcher.FUNDING_POLICY_SHA256,
                'registration_proof':{'registration_commit':launcher.FUNDING_COMMIT,
                    'registration_path':'experiments/e2_budget_extension_20261112/registration.json',
                    'funding_policy_sha256':launcher.FUNDING_POLICY_SHA256},
                'initial_total_hours':64,'effective_total_hours':72,'budget_reset':False}
            anchor['sha256']=hashlib.sha256(json.dumps(anchor,ensure_ascii=False,sort_keys=True,
                separators=(',',':'),allow_nan=False).encode()).hexdigest()
            path=Path(directory,'shared_budget/funding_extension.json');path.parent.mkdir()
            path.write_text(json.dumps(anchor))
            confirmed=launcher.verified_funding_summary(dict(result))
            self.assertEqual(confirmed['effective_total_budget_hours'],72)
            self.assertTrue(confirmed['budget_extension_applied'])
            self.assertNotIn('budget_extended',confirmed)
            anchor['registration_proof']['registration_commit']='b'*40
            anchor.pop('sha256')
            anchor['sha256']=hashlib.sha256(json.dumps(anchor,ensure_ascii=False,sort_keys=True,
                separators=(',',':'),allow_nan=False).encode()).hexdigest()
            path.write_text(json.dumps(anchor))
            rejected=launcher.verified_funding_summary(dict(result))
            self.assertFalse(rejected['budget_extension_applied'])
            self.assertNotIn('effective_total_budget_hours',rejected)

    def test_only_pinned_metadata_is_materialized_and_reused(self):
        raw=b'{"synthetic_author_metadata":true}\n';seen=[]
        def opener(url, **kwargs):seen.append(url);return io.BytesIO(raw)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(launcher,'METADATA_BYTES',len(raw)), \
                patch.object(launcher,'METADATA_SHA256',hashlib.sha256(raw).hexdigest()):
            first=launcher.materialize_pinned_metadata(directory,opener=opener)
            second=launcher.materialize_pinned_metadata(directory,opener=lambda *a,**k:self.fail('redownload'))
            self.assertFalse(first['reused']);self.assertTrue(second['reused'])
            self.assertEqual(Path(directory,launcher.METADATA_PATH).read_bytes(),raw)
            self.assertIn('/'+launcher.ENGINE_COMMIT+'/',seen[0])
            self.assertTrue(seen[0].endswith('/'+launcher.METADATA_PATH))
            self.assertFalse(first['market_values_opened'])

    def test_corrupted_existing_or_remote_metadata_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory,launcher.METADATA_PATH);path.parent.mkdir(parents=True)
            path.write_bytes(b'keep this')
            with self.assertRaisesRegex(RuntimeError,'differs'):
                launcher.materialize_pinned_metadata(directory)
            self.assertEqual(path.read_bytes(),b'keep this')
            path.unlink()
            with self.assertRaisesRegex(RuntimeError,'differs'):
                launcher.materialize_pinned_metadata(directory,opener=lambda *a,**k:io.BytesIO(b'wrong'))
            self.assertFalse(path.exists())

    def test_metadata_check_precedes_registered_execution_and_engine_stays_fixed(self):
        calls=[];base={'_checkout':lambda *args:calls.append(('checkout',args[0])) or Path('/engine')}
        funding={'_base_launcher':lambda:base}
        def run(commit, **kwargs):
            selected=funding['_base_launcher']()
            selected['_checkout'](launcher.ENGINE_COMMIT,Path('/content'),{})
            calls.append(('execute',commit));return kwargs
        funding['run_reduced_github72_colab']=run
        with patch.object(launcher,'_funding_launcher',return_value=funding), \
                patch.object(launcher,'materialize_pinned_metadata',side_effect=lambda *a:calls.append(('metadata',str(a[0]))) or {'sha256':launcher.METADATA_SHA256}):
            result=launcher.run_reduced_resource_fix_colab('a'*40,namespace={'GPU_PYTHON':'isolated'},execute=True)
        self.assertEqual(calls,[('checkout',launcher.ENGINE_COMMIT),('metadata','/engine'),('execute','a'*40)])
        self.assertTrue(result['execute'])
        self.assertEqual(funding['EXTENSION_REGISTRATION'],launcher.EXTENSION_REGISTRATION)
        self.assertIn('scripts/e2_known_setup_failure_selection.py',funding['EXTENSION_FILES'])
        self.assertIn('scripts/e2_resource_failure_selection.py',funding['EXTENSION_FILES'])
        self.assertEqual(len(funding['EXTENSION_FILES']),11)
        self.assertIn('experiments/e2_resource_continuation_20261112/resource_policy.json',funding['EXTENSION_FILES'])
        self.assertIn('scripts/e2_resource_worker.py',funding['EXTENSION_FILES'])
        self.assertIn('scripts/e2_registered_resource_extension.py',funding['EXTENSION_FILES'])
        self.assertIn('scripts/e2_resilient_github_backup.py',funding['EXTENSION_FILES'])

    def test_standalone_resource_policy_is_materialized_and_hash_checked(self):
        policy='experiments/e2_resource_continuation_20261112/resource_policy.json'
        self.assertIn(policy,launcher.EXTENSION_FILES)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);hashes={}
            for relative in launcher.EXTENSION_FILES:
                path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
                raw=b'{"synthetic_resource_policy":true}' if relative==policy else relative.encode()
                path.write_bytes(raw);hashes[relative]=hashlib.sha256(raw).hexdigest()
            record={'engine_commit':driver.ENGINE_COMMIT,'protocol_id':'e2-reduced-delivery-v1',
                'files_sha256':hashes}
            (root/'registration.json').write_text(json.dumps(record))
            driver._extension_root(root,'e'*40)
            (root/policy).write_text('{"synthetic_resource_policy":false}')
            with self.assertRaisesRegex(ValueError,'source differs'):
                driver._extension_root(root,'e'*40)
            (root/policy).unlink()
            with self.assertRaises(OSError):
                driver._extension_root(root,'e'*40)

    def test_missing_or_escaped_resource_handler_is_rejected_before_driver_admission(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root=Path(directory)
            hashes={}
            for relative in launcher.EXTENSION_FILES:
                path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
                raw=('fixture '+relative).encode();path.write_bytes(raw)
                hashes[relative]=hashlib.sha256(raw).hexdigest()
            record={'engine_commit':driver.ENGINE_COMMIT,'protocol_id':'e2-reduced-delivery-v1',
                'files_sha256':hashes}
            (root/'registration.json').write_text(json.dumps(record))
            self.assertEqual(driver._extension_root(root,'a'*40)[0],root)
            relative='scripts/e2_resource_failure_selection.py'
            target=root/relative
            target.write_bytes(b'not the registered source')
            with self.assertRaisesRegex(ValueError,'source differs'):
                driver._extension_root(root,'a'*40)
            escaped=Path(outside)/'same-content.py';escaped.write_bytes(('fixture '+relative).encode())
            target.unlink();target.symlink_to(escaped)
            with self.assertRaisesRegex(ValueError,'source differs'):
                driver._extension_root(root,'a'*40)

    def test_router_preserves_spec_and_supervision_for_selection_and_final(self):
        seen=[]
        def supervisor(*args,**kwargs):seen.append((args,kwargs));return 'terminal-receipt'
        router=driver.resource_worker_router(supervisor,Path('/engine'),Path('/extension'),'c'*40)
        for script,mode in [('run_e2_reduced_selection.py','selection'),('e2_reduced_final.py','final')]:
            command=['/isolated/python','-u','/engine/scripts/'+script,'--worker-spec','/output/worker_spec.json']
            result=router(command,'budget','same-attempt','output','same-spec-digest',phases=3,cwd='/engine')
            self.assertEqual(result,'terminal-receipt')
            args,kwargs=seen[-1]
            self.assertEqual(args[1:],('budget','same-attempt','output','same-spec-digest'))
            self.assertEqual(kwargs,{'phases':3,'cwd':'/engine'})
            self.assertEqual(args[0],['/isolated/python','-u',Path('/extension/scripts/e2_resource_worker.py'),
                '--mode',mode,'--engine-root',Path('/engine'),'--extension-root',Path('/extension'),
                '--extension-commit','c'*40,'--worker-spec','/output/worker_spec.json'])

    def test_router_rejects_unregistered_entrypoints_before_supervision(self):
        router=driver.resource_worker_router(lambda *a,**k:self.fail('worker launched'),
            '/engine','/extension','d'*40)
        for command in [
            ['python','-u','/elsewhere/run_e2_reduced_selection.py','--worker-spec','spec'],
            ['python','-u','/engine/scripts/run_e2_reduced_selection.py','--different','spec'],
            ['python','-u','/engine/scripts/run_e2_reduced_selection.py','--worker-spec','spec','--extra'],
        ]:
            with self.assertRaisesRegex(ValueError,'nothing was launched'):
                router(command,'budget')

    def test_new_fix_commit_does_not_reattribute_funding_or_change_child_engine(self):
        calls=[]
        args=SimpleNamespace(execute=True,stage='all',output=Path('/temporary'),data_root='dev',
            source_roots='roots',runtime_profile='runtime',budget_evidence='budget',migration='migration',
            cache_audit='audit',reconciliation_root='recon',migration_registration_commit=driver.ENGINE_COMMIT,
            inherited_ledger=[])
        original_selection=SimpleNamespace(run_selection='original',supervise_command=lambda *a,**k:None)
        new_selection=lambda *a,**k:None
        scientific=SimpleNamespace(parser=lambda:SimpleNamespace(parse_args=lambda unused:args),
            read_json=lambda p:{'file':p},execute_delivery=lambda *a:calls.append(('execute',original_selection.run_selection)))
        class Budget:
            _closed=False
            stage_wall_seconds=1800
            def report(self):return {'reserved_seconds':0}
            def charge_overhead(self,*a,**k):pass
        class Backup:
            def preflight(self):calls.append(('preflight',))
        class State:
            def __init__(self,*a):pass
            def restore(self):calls.append(('restore',))
            @contextmanager
            def budget_factory(self,*a,budget_class=None,**k):yield budget_class('/budget',{})
            def snapshot(self,*a,**k):pass
        def factory(policy,**kwargs):
            self.assertEqual(kwargs['funding_registration_commit'],driver.FUNDING_COMMIT)
            self.assertNotEqual(kwargs['funding_registration_commit'],'b'*40)
            return lambda *a,**k:Budget()
        def resource_factory(funded,policy,**kwargs):
            self.assertEqual(policy,{'phase_seconds':1800})
            self.assertEqual(kwargs['registration_commit'],'b'*40)
            self.assertEqual(kwargs['registration_root'],Path('/engine'))
            self.assertEqual(kwargs['extension_root'],Path('/extension'))
            calls.append(('resource_policy',policy))
            return funded
        modules={'run_e2_reduced_delivery':scientific,'run_e2_reduced_selection':original_selection,
            'e2_reduced_protocol':SimpleNamespace(verify_reduced_registration=lambda *a,**k:None),
            'e2_resilient_github_backup':SimpleNamespace(ResilientGitHubBackup=Backup),
            'e2_reduced_github_state':SimpleNamespace(GitHubState=State),
            'e2_registered_budget_extension':SimpleNamespace(make_funded_budget_class=factory),
            'e2_registered_resource_extension':SimpleNamespace(make_resource_budget_class=resource_factory),
            'reconcile_e2_github_completed':SimpleNamespace(reconcile_completed_local=lambda *a:calls.append(('reconcile',))),
            'e2_resource_failure_selection':SimpleNamespace(run_selection=new_selection)}
        with patch.dict(sys.modules,modules),patch.object(driver,'_engine_root',return_value=Path('/engine')), \
                patch.object(driver,'_extension_root',return_value=(Path('/extension'),{'funding_policy':{'approved':'fixture'},'resource_policy':{'phase_seconds':1800}})), \
                patch.object(driver.subprocess,'run') as fetch:
            result=driver.main(['--engine-root','/engine','--extension-root','/extension','--extension-commit','b'*40])
        self.assertEqual(result,0)
        self.assertEqual([c.args[0][-1] for c in fetch.call_args_list],[driver.FUNDING_COMMIT,'b'*40])
        self.assertEqual(calls[-1],('execute',new_selection))
        self.assertLess(calls.index(('reconcile',)),calls.index(('restore',)))


if __name__=='__main__':unittest.main()
