"""Synthetic durability, three-stage SAE and supervision tests; no market data."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import torch
import pandas as pd
from e2_reduced_fit import fit_reduced_job,file_sha
from e2_published_registry import registered_trials,build_train_model
from e2_reduced_supervision import supervise_command,ReducedWorkerFailure
from run_e2_reduced_selection import job_plan

torch.set_num_threads(1)


class Scale:
    target_scale=2.
    def record(self):return {'target_scale':self.target_scale,'synthetic':True}


def split(seed,count=8):
    g=torch.Generator().manual_seed(seed)
    x=torch.randn(count,7,5,generator=g)
    y=torch.rand(count,generator=g)+.1
    dates=tuple(pd.date_range('2024-01-01',periods=count,tz='UTC'))
    value=SimpleNamespace(x=x,y=y,y_original=tuple((y*2).tolist()),dates=dates,scaler=Scale(),signature='synthetic-'+str(seed))
    value.validate=lambda:None
    value.record=lambda:{'signature':value.signature,'dates':[str(d.date()) for d in dates],'synthetic':True}
    return value


class Budget:
    def __init__(self,timeout=2.):self.active={};self.charges={};self.timeout=timeout
    def reservation_for(self,phases=1):return self.timeout+1
    def worker_timeout_for(self,phases=1):return self.timeout
    def reserve(self,key,seconds,metadata=None):
        if key in self.active or key in self.charges:raise ValueError('duplicate')
        self.active[key]=seconds
    def charge(self,key,elapsed,status,resource_use=None):
        if key not in self.active:raise ValueError('unbound')
        del self.active[key]
        self.charges[key]={'elapsed_seconds':elapsed,'status':status,'resource_use':resource_use}
        return self.charges[key]
    def completed_receipt(self,key,expected_resource_use=None):
        r=self.charges.get(key)
        if r and r['status']=='completed' and all(r['resource_use'].get(k)==v for k,v in (expected_resource_use or {}).items()):return r
        return None


class ReducedExecutionTests(unittest.TestCase):
    def fit(self,name,directory):
        trial=registered_trials(name)[5]
        phases={'ae1':1,'ae2':1,'supervised':1} if 'autoencoder' in name else {'supervised':1}
        return fit_reduced_job(name,trial,123,split(1),None,directory,fixed_epochs=phases,
            device='cpu',allow_cpu_for_tests=True,fit_wall_seconds=30,
            source_identity={'synthetic':'fixture'},runtime_identity={'device':'cpu','synthetic':True})

    def test_full_model_checkpoint_can_be_reconstructed_and_reused_without_fit(self):
        with tempfile.TemporaryDirectory() as d:
            result=self.fit('MLP16',d)
            saved=torch.load(Path(d)/result['selected_checkpoint']['path'],weights_only=False)
            model=build_train_model('MLP16',registered_trials('MLP16')[5],seed=999)
            model.load_state_dict(saved['state_dict'])
            self.assertEqual(tuple(model(split(1).x).shape),(8,))
            self.assertEqual(self.fit('MLP16',d),result)
            self.assertEqual(result['phase_best_epochs'],{'supervised':1})
            self.assertTrue(result['synthetic'])

    def test_sae_really_preserves_three_durable_phases_and_full_predictor(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.fit('LSTM-autoencoder con predictor',d)
            self.assertEqual(set(r['phase_records']),{'ae1','ae2','supervised'})
            self.assertTrue(r['architecture']['pretraining_executed'])
            saved=torch.load(Path(d)/'selected_model.pt',weights_only=False)
            self.assertTrue(any(k.startswith('encoder1.') for k in saved['state_dict']))
            self.assertTrue(any(k.startswith('encoder2.') for k in saved['state_dict']))
            self.assertFalse(any('decoder' in k for k in saved['state_dict']))
            for phase in r['phase_records']:
                self.assertTrue((Path(d)/'phases'/phase/'events.jsonl').is_file())
                self.assertEqual(r['phase_records'][phase]['optimizer_initial_state_entries'],0)

    def test_tampered_weights_block_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            self.fit('MLP16',d)
            with (Path(d)/'selected_model.pt').open('ab') as stream:stream.write(b'tamper')
            with self.assertRaisesRegex(ValueError,'changed'):self.fit('MLP16',d)

    def test_interrupted_fit_never_restarts_automatically(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'request.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'reconciliation_required'):self.fit('MLP16',d)

    def test_job_plan_exact_six_slots_three_folds_and_single_seed(self):
        jobs=job_plan(['GRU32','TimesNet'])
        self.assertEqual(len(jobs),36)
        self.assertEqual({j['seed'] for j in jobs},{271201})
        self.assertEqual(len({j['logical_job_id'] for j in jobs}),36)
        self.assertEqual({j['configuration_id'] for j in jobs},{'C01','C02','C03','C04','C05','C06'})

    def test_completed_worker_is_charged_and_reused_only_with_its_charge(self):
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d);spec='a'*64
            code="import pathlib,json,hashlib; p=pathlib.Path('.');(p/'artifact.txt').write_text('durable');(p/'result.json').write_text(json.dumps({'status':'completed','worker_spec_sha256':'"+spec+"','artifacts_sha256':{'artifact.txt':hashlib.sha256(b'durable').hexdigest()},'optimizer_steps':1}))"
            budget=Budget()
            first=supervise_command([sys.executable,'-c',code],budget,'job',d,spec,cwd=d)
            self.assertGreater(first['charged_wall_seconds'],0)
            second=supervise_command([sys.executable,'-c','raise Exception()'],budget,'job',d,spec,cwd=d)
            self.assertEqual(first['result_sha256'],second['result_sha256'])
            with self.assertRaisesRegex(RuntimeError,'matching budget'):supervise_command([sys.executable,'-c',code],Budget(),'job',d,spec,cwd=d)

    def test_deadline_terminates_worker_and_does_not_claim_completion(self):
        with tempfile.TemporaryDirectory() as d:
            budget=Budget(.1)
            with self.assertRaises(ReducedWorkerFailure):
                supervise_command([sys.executable,'-c','import time;time.sleep(5)'],budget,'job',d,'b'*64,cwd=d)
            self.assertEqual(budget.charges['job']['status'],'failed')
            self.assertEqual(budget.active,{})
            self.assertFalse((Path(d)/'result.json').exists())


if __name__=='__main__':unittest.main()
