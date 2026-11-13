"""CPU-only migration checks; no optimizer, market values or network writes."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import e2_a100_cache_selection as M
import audit_e2_reduced_cache as A
E=M.original


def runtime(name):
    return {'python':'3.12.12','torch':'2.8.0+cu126','torch_cuda':'12.6',
        'device':'cuda:0','device_name':name,'total_memory_bytes':1000 if name=='Tesla T4' else 4000,
        'effective_memory_limit_bytes':800 if name=='Tesla T4' else 3200,'cudnn':91002,
        'cuda_kernel_probe_passed':True,'cpu_threads':2,'interop_threads':1,
        'dtype':'float32','amp':False,'tf32':False,'cudnn_tf32':False,'deterministic':True}


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.output=self.root/'output';self.output.mkdir()
        self.source={'frozen-E':'same-bytes'}
        self.policy={'same_registered_policy':True}
        self.actual={'runtime':runtime('NVIDIA A100-SXM4-40GB'),'fit_execution_policy_verified':True,
            'fit_execution_policy':self.policy}
        self.runtime_path=self.root/'a100.json';E.write_json(self.runtime_path,self.actual)
        self.recon=self.root/'recon';self.recon.mkdir()
        self.heads={A.LEGACY_BRANCH:'a'*40,A.OPTIMIZED_BRANCH:'b'*40}
        E.write_json(self.recon/'budget_evidence.json',{'historical_heads':self.heads})
        for branch,head in self.heads.items():
            p=self.recon/'snapshots'/branch.rsplit('/',1)[-1]/head/'campaign.json'
            E.write_json(p,{'plan':{'hardware_binding':runtime('Tesla T4'),'execution':self.policy}})
        self.domains=E.current_domains();self.jobs=E.job_plan(['FCN-1D'],domains=self.domains)
        self.old=self.output/'jobs'/self.jobs[0]['logical_job_id'];self.old.mkdir(parents=True)
        spec={'job':self.jobs[0],'source_sha256':self.source,'runtime_identity':runtime('Tesla T4')}
        E.write_json(self.old/'worker_spec.json',spec)
        result={'status':'completed','cache_reuse':False,'job':self.jobs[0],
            'worker_spec_sha256':E.digest(spec),'artifacts_sha256':{}}
        E.write_json(self.old/'result.json',result)
        self.cache=self.output/'jobs'/self.jobs[1]['logical_job_id'];self.cache.mkdir(parents=True)
        checks={k:True for k in ('candidate_payload','recipe_identity','complete_validation_calendar',
            'known_completed_updates','prepared_inputs','job_payload','worker_runtime')}
        self.admission={'reuse_approved':True,'runtime_facts_match':True,'fit_execution_policy_attested':True,
            'fit_execution_policy_match':True,'checks':checks}
        E.write_json(self.cache/'cache_admission.json',self.admission)
        cached={'status':'completed','cache_reuse':True,'artifacts_sha256':{
            'cache_admission.json':E.file_sha(self.cache/'cache_admission.json')}}
        E.write_json(self.cache/'result.json',cached)
        def completed(identity,*,expected_resource_use):
            self.assertEqual(identity,self.jobs[0]['logical_job_id'])
            self.assertEqual(expected_resource_use,{'worker_spec_sha256':E.digest(spec),
                'result_sha256':E.file_sha(self.old/'result.json')})
            return {'sha256':'c'*64}
        self.budget=SimpleNamespace(completed_receipt=completed,remaining_seconds=30000,stage_wall_seconds=1800,
            report=lambda:{'remaining_seconds':30000})
        self.args=SimpleNamespace(output=self.output,runtime_profile=self.runtime_path,
            reconciliation_root=self.recon,cache_audit=self.root/'new_a100_audit.json',
            models=['FCN-1D'],data_root='not-opened',source_roots='not-opened')
        self.report={'reuse_approved':1,'fits':[],'scope':'synthetic-no-market-audit'}

    def prepared(self):
        captured={}
        def audit(*args,**kwargs):
            captured.update(kwargs);return self.report
        with patch.object(E,'source_identity',return_value=self.source), \
                patch.object(A,'build_development_context',return_value={'synthetic':True}), \
                patch.object(A,'audit',side_effect=audit):
            result=M.prepare_historical_cache_audit(self.args,self.budget,registration_commit='d'*40)
        self.assertEqual(captured['runtime_profile']['runtime']['device_name'],'Tesla T4')
        return result

    def test_audits_historical_T4_without_relabeling_actual_A100(self):
        before=self.runtime_path.read_bytes();result=self.prepared()
        self.assertEqual(before,self.runtime_path.read_bytes())
        actual,historical=M.validate_hardware_transition(self.args,self.report)
        self.assertIn('A100',actual['runtime']['device_name'])
        self.assertEqual(historical['runtime']['device_name'],'Tesla T4')
        self.assertTrue(historical['current_gpu_is_not_claimed_to_be_T4'])
        self.assertEqual(result['historical_cache_reuse_approved'],1)
        before_files={p:p.read_bytes() for p in self.output.rglob('*') if p.is_file()}
        self.prepared()
        self.assertEqual(before_files,{p:p.read_bytes() for p in self.output.rglob('*') if p.is_file()})

    def test_rejects_missing_cost_false_policy_and_preexisting_final_fit(self):
        self.budget.completed_receipt=lambda *a,**k:None
        with patch.object(E,'source_identity',return_value=self.source),self.assertRaisesRegex(ValueError,'completed cost'):
            M.prepare_historical_cache_audit(self.args,self.budget,registration_commit='d'*40)
        self.budget.completed_receipt=lambda *a,**k:{'sha256':'c'*64}
        admission=deepcopy(self.admission);admission['fit_execution_policy_attested']=False
        (self.cache/'cache_admission.json').write_text(json.dumps(admission))
        with patch.object(E,'source_identity',return_value=self.source),self.assertRaisesRegex(ValueError,'attestation'):
            M.prepare_historical_cache_audit(self.args,self.budget,registration_commit='d'*40)
        p=self.output/'final/final-fit/one/worker_spec.json';p.parent.mkdir(parents=True);p.write_text('{}')
        with self.assertRaisesRegex(ValueError,'not to have started'):
            M.prepare_historical_cache_audit(self.args,self.budget,registration_commit='d'*40)

    def test_actual_T4_or_changed_transition_profile_rejected(self):
        bad=deepcopy(self.actual);bad['runtime']=runtime('Tesla T4')
        with self.assertRaisesRegex(ValueError,'A100'):M._strict_current(bad)
        self.prepared()
        old=json.loads(self.args.historical_runtime_profile.read_text());old['runtime']['total_memory_bytes']=42
        self.args.historical_runtime_profile.write_text(json.dumps(old))
        with self.assertRaisesRegex(ValueError,'binding changed'):
            M.validate_hardware_transition(self.args,self.report)

    def loop_patches(self,jobs):
        cuda=ModuleType('run_e2_cuda_search');cuda.prepare_panel=lambda *a:(None,None)
        temporal=ModuleType('e2_temporal_training');temporal.prepare_b=lambda *a:SimpleNamespace(
            inner_validation=SimpleNamespace(dates=list(range(83))))
        return [patch.dict(sys.modules,{'run_e2_cuda_search':cuda,'e2_temporal_training':temporal}),
            patch.object(E,'verify_reduced_registration'),patch.object(E,'source_identity',return_value=self.source),
            patch.object(E,'job_plan',return_value=jobs),
            patch.object(E,'select_reduced_hpo',return_value={'winner_configuration_id':jobs[-1]['configuration_id']})]

    def test_new_cache_opportunity_passes_only_historical_profile(self):
        self.prepared();job=self.jobs[2];self.report['fits']=[{'model_id':job['model_id'],
            'origin':job['fold_origin'],'configuration_id':job['configuration_id'],'reuse_approved':True}]
        # Update the bound synthetic audit before the selector test.
        receipt=json.loads(self.args.hardware_transition_receipt.read_text())
        self.args.cache_audit.write_text(json.dumps(self.report))
        receipt['cache_audit_sha256']=E.file_sha(self.args.cache_audit);receipt.pop('sha256');receipt['sha256']=E.digest(receipt)
        self.args.hardware_transition_receipt.write_text(json.dumps(receipt))
        seen=[]
        def cached(job,match,recon,directory,prepared,profile):
            seen.append(profile['runtime']['device_name'])
            result={'status':'completed','cache_reuse':True,'rmse_original':.01,
                'phase_best_epochs':{'supervised':2},'artifacts_sha256':{}}
            E.write_json(directory/'result.json',result);return result
        patches=self.loop_patches([job])
        for p in patches:p.start();self.addCleanup(p.stop)
        with patch.object(E,'_cached_result',side_effect=cached),patch.object(E,'supervise_command',side_effect=AssertionError('no optimizer')):
            M.run_selection(self.args,self.budget,self.report,self.recon)
        self.assertEqual(seen,['Tesla T4'])
        self.assertIn('A100',json.loads(self.runtime_path.read_text())['runtime']['device_name'])

    def test_new_fit_uses_actual_A100_and_unchanged_entrypoint(self):
        self.prepared();job=self.jobs[2];seen=[]
        def supervise(command,budget,identity,directory,spec_sha,**kwargs):
            spec=json.loads((directory/'worker_spec.json').read_text());seen.append((spec,command))
            result={'status':'completed','rmse_original':.01,'phase_best_epochs':{'supervised':2},'optimizer_steps':1,
                'artifacts_sha256':{},'worker_spec_sha256':spec_sha}
            E.write_json(directory/'result.json',result);return {'result':result}
        patches=self.loop_patches([job])
        for p in patches:p.start();self.addCleanup(p.stop)
        with patch.object(E,'supervise_command',side_effect=supervise):
            M.run_selection(self.args,self.budget,self.report,self.recon)
        self.assertIn('A100',seen[0][0]['runtime_identity']['device_name'])
        self.assertEqual(seen[0][0]['fit_wall_seconds'],1800)
        self.assertEqual(Path(seen[0][1][2]).resolve(),Path(E.__file__).resolve())

    def test_registered_unknown_opportunity_is_not_persisted_as_a_scientific_result(self):
        self.prepared();job={'logical_job_id':M.LOST_JOB,'model_id':'FCN-1D',
            'fold_origin':'2024-07-01','configuration_id':'C01'}
        row={'status':'failed','model_id':'FCN-1D','resource_use':{'cache_reuse':False},'rmse_original':None}
        unknown={'status':'unknown_outcome_not_retried'}
        recovery=ModuleType('e2_expired_lease_recovery');recovery.verify_recovery_disposition=lambda *a:(row,unknown)
        patches=self.loop_patches([job])
        for p in patches:p.start();self.addCleanup(p.stop)
        persisted=[]
        with patch.dict(sys.modules,{'e2_expired_lease_recovery':recovery}), \
                patch.object(E,'supervise_command',side_effect=AssertionError('lost job must never train')):
            M.run_selection(self.args,self.budget,self.report,self.recon,persist_result=lambda *a:persisted.append(a))
        self.assertEqual(persisted,[])
        self.assertEqual(json.loads((self.output/'jobs'/M.LOST_JOB/'selection_row.json').read_text()),row)


if __name__=='__main__':unittest.main()
