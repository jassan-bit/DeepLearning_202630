"""Prospective admission and routing tests; no CUDA, market data, or optimizer."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
loader=importlib.util.spec_from_file_location('resource_worker',ROOT/'scripts/e2_resource_worker.py')
W=importlib.util.module_from_spec(loader);loader.loader.exec_module(W)


class ResourceWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.policy=json.loads((ROOT/W.POLICY_PATH).read_bytes())
        self.runtime={'python':'3.12.12','torch':'2.8.0+cu126','torch_cuda':'12.6',
            'device_name':'Tesla T4','total_memory_bytes':15637086208,
            'cpu_threads':2,'interop_threads':1,'deterministic':True,'tf32':False,'cudnn_tf32':False}
        self.job={'model_id':'TimesNet','logical_job_id':'planned'}
        self.selection=SimpleNamespace(VERSION='frozen-E',ROOT=self.root,
            job_plan=lambda names:[self.job],source_identity=lambda:{'E':'frozen'},
            verify_reduced_registration=lambda *a,**k:None,
            worker_preflight_policy=lambda:{'runtime':self.runtime})
        self.spec={'version':'frozen-E','job':self.job,'fit_wall_seconds':1800,
            'source_sha256':{'E':'frozen'},'runtime_identity':self.runtime,'output':str(self.root/'fit')}
        self.record={'files_sha256':{W.SELF_PATH:W.file_sha(ROOT/W.SELF_PATH)}}

    def test_real_cap_and_unchanged_policy_are_checked(self):
        self.assertEqual(W.verify_policy(self.policy)['stage_wall_seconds'],1800)
        W.validate_hpo(self.spec,self.selection,observed=self.runtime)
        for changed in (600,1801):
            with self.assertRaisesRegex(ValueError,'truthfully'):
                W.validate_hpo({**self.spec,'fit_wall_seconds':changed},self.selection,observed=self.runtime)
        policy={**self.policy,'max_epochs':100};policy['sha256']=W.digest({k:v for k,v in policy.items() if k!='sha256'})
        with self.assertRaisesRegex(ValueError,'policy differs'):W.verify_policy(policy)

    def test_changed_source_or_platform_is_rejected_before_model_data(self):
        with self.assertRaisesRegex(ValueError,'source identity'):
            W.validate_hpo({**self.spec,'source_sha256':{'E':'changed'}},self.selection,observed=self.runtime)
        with self.assertRaisesRegex(ValueError,'Actual HPO platform'):
            W.validate_hpo(self.spec,self.selection,observed={**self.runtime,'torch':'other'})
        bad={**self.runtime,'deterministic':False}
        with self.assertRaisesRegex(ValueError,'strict fitting policy'):
            W.validate_hpo({**self.spec,'runtime_identity':bad},self.selection,observed=bad)

    def test_hpo_calls_frozen_worker_with_truthful_cap_and_bound_sidecar(self):
        original=lambda spec:None;self.selection._validate_spec=original
        called=[]
        def frozen(spec):
            self.selection._validate_spec(spec)
            admission=json.loads((Path(spec['output'])/W.SIDECAR).read_bytes())
            self.assertEqual(admission['resource_phase_wall_seconds'],spec['fit_wall_seconds'])
            self.assertEqual(admission['resource_policy_sha256'],self.policy['sha256'])
            called.append(spec['fit_wall_seconds']);return {'status':'completed'}
        self.selection.run_worker=frozen
        with patch.object(W,'observed_platform',return_value=self.runtime):
            result=W.run_selection(self.spec,self.selection,'a'*40,self.record,self.policy)
        self.assertEqual(called,[1800]);self.assertIs(self.selection._validate_spec,original)
        self.assertEqual(result['status'],'completed')

    def test_sidecar_does_not_relabel_a_prior_admission(self):
        W.write_admission(self.root,'a'*40,self.record,self.policy,'selection','b'*64)
        before=(self.root/W.SIDECAR).read_bytes()
        W.write_admission(self.root,'a'*40,self.record,self.policy,'selection','b'*64)
        self.assertEqual((self.root/W.SIDECAR).read_bytes(),before)
        with self.assertRaisesRegex(ValueError,'previous extension admission'):
            W.write_admission(self.root,'a'*40,self.record,self.policy,'selection','c'*64)

    def final_fixture(self,kind):
        source=self.root/'scripts/source.py';source.parent.mkdir(exist_ok=True);source.write_text('unchanged\n')
        spec={'kind':kind,'fit_wall_seconds':1800,'device':'cuda:0','output':str(self.root/'final-worker'),
            'source_identity':{'source_commit':W.ENGINE_COMMIT,'files_sha256':{'scripts/source.py':W.file_sha(source)}},
            'runtime_identity':self.runtime,'data_root':'not-opened','project_root':'not-opened'}
        self.calls=[]
        def result(label):
            def run(*args,**kwargs):
                self.calls.append(label)
                self.assertTrue((Path(spec['output'])/W.SIDECAR).is_file())
                return {'status':'completed','phase':label,'model':'representative','seed':271801}
            return run
        final=SimpleNamespace(ROOT=self.root,VERSION='frozen-final-E',
            neural_final_worker=result('neural-final'),classical_worker=result('classical'),score_final=result('score'),
            _artifact_hashes=lambda output:{W.SIDECAR:W.file_sha(Path(output)/W.SIDECAR)},
            write_json=lambda p,v:Path(p).write_text(json.dumps(v)))
        return spec,final

    def test_final_and_classical_route_to_original_functions(self):
        for kind in ('neural-final','classical'):
            spec,final=self.final_fixture(kind)
            spec['worker_spec_sha256']=W.digest(spec)
            result=W.run_final(spec,final,self.selection,'a'*40,self.record,self.policy)
            self.assertEqual(self.calls,[kind]);self.assertEqual(result['worker_spec_sha256'],spec['worker_spec_sha256'])
            path=Path(spec['output']);(path/'result.json').unlink();(path/W.SIDECAR).unlink()

    def test_final_runtime_source_and_checksum_are_verified(self):
        spec,final=self.final_fixture('neural-final');expected=W.digest(spec)
        with self.assertRaisesRegex(ValueError,'checksum'):W.validate_final(spec,'a'*64,final,self.selection)
        changed={**spec,'fit_wall_seconds':600}
        with self.assertRaisesRegex(ValueError,'1800-second'):W.validate_final(changed,W.digest(changed),final,self.selection)
        (self.root/'scripts/source.py').write_text('changed')
        with self.assertRaisesRegex(ValueError,'executable bytes'):W.validate_final(spec,expected,final,self.selection)

    def test_scoring_keeps_original_freeze_and_scoring_function(self):
        spec,final=self.final_fixture('final-score');frozen=self.root/'frozen';frozen.mkdir()
        (frozen/'final_freeze.json').write_text('{"all81":true}')
        spec.update(final_output=str(frozen),final_freeze_sha256=W.file_sha(frozen/'final_freeze.json'))
        spec['worker_spec_sha256']=W.digest(spec)
        result=W.run_final(spec,final,self.selection,'a'*40,self.record,self.policy)
        self.assertEqual(self.calls,['score']);self.assertEqual(result['A2'],'scientific_outputs/benchmark_A2.csv')
        self.assertIn(W.SIDECAR,result['artifacts_sha256'])

    def test_extension_needs_actual_publication_and_exact_runtime_bytes(self):
        extension=self.root/'extension';worker=extension/W.SELF_PATH;worker.parent.mkdir(parents=True)
        worker.write_text('published resource worker\n')
        policy=extension/W.POLICY_PATH;policy.parent.mkdir(parents=True);policy.write_text(json.dumps(self.policy))
        record={'engine_commit':W.ENGINE_COMMIT,'protocol_id':'e2-reduced-delivery-v1',
            'resource_policy':self.policy,'files_sha256':{W.SELF_PATH:W.file_sha(worker),W.POLICY_PATH:W.file_sha(policy)}}
        raw=json.dumps(record).encode();(extension/'registration.json').write_bytes(raw)
        def check(command,**kwargs):return W.ENGINE_COMMIT if kwargs.get('text') else raw
        with patch.object(W,'__file__',str(worker)),patch.object(W.subprocess,'check_output',side_effect=check):
            verified=W.verify_extension(self.root,extension,'a'*40)
            self.assertEqual(verified[-1],self.policy)
            worker.write_text('different bytes')
            with self.assertRaisesRegex(ValueError,'execution bytes differ'):
                W.verify_extension(self.root,extension,'a'*40)


if __name__=='__main__':unittest.main()
