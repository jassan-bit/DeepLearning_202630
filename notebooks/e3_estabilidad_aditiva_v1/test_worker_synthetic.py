"""Synthetic-only trajectory, admission, budget and evaluation tests."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('E3_EVAL_SOURCE_ROOT', str(HERE.parents[1]))).resolve()
spec = importlib.util.spec_from_file_location('tested_stability_worker', HERE/'worker.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
CODE_BYTES = {n:(HERE/n).read_bytes() for n in ('worker.py','bootstrap.py','reference.py')}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(w.canonical(value)+b'\n')


def fixture(root):
    inputs, run = root/'inputs', root/'run'
    inputs.mkdir()
    run.mkdir()
    code = root/'code'; code.mkdir()
    for n,payload in CODE_BYTES.items(): (code/n).write_bytes(payload)
    w.__file__ = str(code/'worker.py')
    plan = json.loads((HERE/'plan.json').read_text())
    old = ROOT/'experiments/e3_cierre_v2_20261116/audit/inputs'
    admission = json.loads((old/'data_admission.json').read_text())
    partition = json.loads((ROOT/'book_e2/_static/e2_eda_current_v1/partition_audit.json').read_text())
    trdates = partition['final_partition']['training']['training_sample_IDs']
    evdates = partition['final_partition']['evaluation']['sample_IDs']
    rng = np.random.default_rng(42)
    def raw(n):
        close = np.exp(4+rng.normal(0,.001,size=(n,7)))
        return np.stack([close,close*1.01,close*.99,close,rng.uniform(30000,60000,(n,7))],axis=-1)
    trraw, evraw = raw(655), raw(608)
    def target(a):
        ret=np.diff(np.log(a[...,3]),axis=1)
        return np.sqrt(((ret-ret.mean(axis=1)[:,None])**2).sum(axis=1)/7)+.002
    # Forbidden object payload proves *_scaled or other members are never loaded.
    np.savez(inputs/'current_cohort_arrays.npz',training_dates=np.array(trdates),training_raw=trraw,
             training_y_original=target(trraw),evaluation_dates=np.array(evdates),evaluation_raw=evraw,
             evaluation_y_original=target(evraw),evaluation_X=np.array([object()]))
    admission['npz_sha256']=w.sha(inputs/'current_cohort_arrays.npz')
    admission['scaler'].update(mean=[0]*5,std=[1]*5,y_scale=.02,target_scale=.02,u_scale=.02,d_fit=.002)
    saved_manifest=json.loads((old/'manifest.json').read_text())
    saved_manifest['data_fingerprint']=hashlib.sha256(w.canonical(admission)).hexdigest()
    dump(inputs/'v2_data_admission.json',admission)
    dump(inputs/'v2_manifest.json',saved_manifest)
    dump(inputs/'e2_partition.json',partition)
    dump(inputs/'e2_data_admission.json',dict(common_eligible_dates=evdates,
        target='population sigma7 of daily Close log returns, ddof0, unannualized; h1'))
    (inputs/'e2_benchmark_A2.csv').write_bytes((ROOT/'book_e2/_static/e2_benchmark_final_20261113/benchmark_A2.csv').read_bytes())
    dump(inputs/'prior_moment_summary.json',dict(status='completed',global_charged_seconds=w.PRIOR))
    dump(inputs/'historical_summary.json',dict(evaluation=dict(RMSE=.01)))
    (inputs/'historical_predictions.csv').write_text('synthetic\n')
    for name in plan['inputs']:
        plan['inputs'][name]['sha256']=w.sha(inputs/name)
    w.PLAN_FINGERPRINT=hashlib.sha256(w.canonical(plan)).hexdigest()
    dump(run/'plan.json',plan)
    identity=dict(version='synthetic_only',run_id='run-synthetic',claim_id='claim-synthetic',
        plan_sha256=w.PLAN_FINGERPRINT,npz_sha256=admission['npz_sha256'],source_hashes=plan['source']['files'],
        worker_sha256=w.sha(code/'worker.py'),bootstrap_sha256=w.sha(code/'bootstrap.py'),reference_sha256=w.sha(code/'reference.py'))
    dump(run/'manifest.json',identity)
    dump(run/'ledger.json',dict(run_id=identity['run_id'],limit_seconds=w.LIMIT,charged_seconds=0.,
        prior_charged_seconds=w.PRIOR,global_limit_seconds=w.GLOBAL_LIMIT,events=[],active_session=None))
    return inputs,run


def reserve(run,phase='train'):
    m=json.loads((run/'manifest.json').read_text())
    l=json.loads((run/'ledger.json').read_text())
    l['active_session']=dict(session_id='synthetic-active',last_wall_utc=datetime.now(timezone.utc).isoformat())
    dump(run/'ledger.json',l)
    claim=dict(run_id=m['run_id'],claim_id=m['claim_id'],plan_sha256=m['plan_sha256'],npz_sha256=m['npz_sha256'],
        phase=phase,active_session_id=l['active_session']['session_id'],ack_commit='a'*40,
        additional_limit_seconds=w.LIMIT,prior_charged_seconds=w.PRIOR,global_limit_seconds=w.GLOBAL_LIMIT)
    if phase=='evaluate':
        claim['training_snapshot_commit']='b'*40
        claim['checkpoint_sha256']={str(s):w.sha(w.checkpoint_path(run,s)) for s in w.SEEDS if w.checkpoint_path(run,s).is_file()}
    dump(run/'claim.json',claim)
    return run/'claim.json'


def execute(inputs,run,phase='train',n=10,clock=None):
    claim=reserve(run,phase)
    kwargs={} if clock is None else dict(clock=clock)
    with redirect_stdout(io.StringIO()):
        return w.execute(ROOT,inputs,run,claim,phase,n,'cpu',**kwargs)


class StabilityChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        w.MAX_EPOCHS=2  # Test-only shortening; original productionplan still fixes40.

    def test_ten_seed_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'one').mkdir(); (root/'two').mkdir()
            ai,a=fixture(root/'one'); bi,b=fixture(root/'two')
            for _ in range(2): execute(ai,a)
            for _ in range(10): execute(bi,b,n=2)
            self.assertEqual(json.loads((a/'state.json').read_text())['status'],'trained')
            self.assertEqual(json.loads((b/'state.json').read_text())['status'],'trained')
            for seed in w.SEEDS:
                ca=torch.load(w.checkpoint_path(a,seed),weights_only=True,map_location='cpu')
                cb=torch.load(w.checkpoint_path(b,seed),weights_only=True,map_location='cpu')
                self.assertEqual(ca['history'],cb['history'])
                self.assertEqual(ca['best_epoch'],cb['best_epoch'])
                self.assertTrue(all(torch.equal(ca['model_state'][k],cb['model_state'][k]) for k in ca['model_state']))
            result=execute(ai,a,phase='evaluate')
            self.assertEqual(result['complete_training_seeds'],10)
            self.assertEqual(len(result['per_seed_metrics']),10)
            self.assertTrue(result['comparison_complete'])
            self.assertEqual(result['historical_seed_excluded'],20261116)
            rmses=[r['RMSE'] for r in result['per_seed_metrics']]
            self.assertAlmostEqual(result['RMSE_mean'],float(np.mean(rmses)),places=15)
            self.assertAlmostEqual(result['RMSE_sample_SD'],float(np.std(rmses,ddof=1)),places=15)
            self.assertEqual([r['seed'] for r in result['per_seed_metrics']],list(w.SEEDS))
            self.assertEqual(len((a/'evaluation_predictions.csv').read_text().splitlines()),6081)
            self.assertGreater(json.loads((b/'ledger.json').read_text())['charged_seconds'],0)

    def test_midbatch_checkpoint_resume_and_metadata_repair(self):
        sys.path.insert(0,str(ROOT))
        from src import e3_innovation as lib
        from src.e3_innovation_model import InnovationModel
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'one').mkdir(); (root/'two').mkdir()
            ai,a=fixture(root/'one'); bi,b=fixture(root/'two')
            m=json.loads((a/'manifest.json').read_text())
            admission=json.loads((ai/'v2_data_admission.json').read_text())
            reference=w.load_reference(m)
            data=w.train_data(ai,admission,reference)
            class Crash(Exception): pass
            class Budget:
                def __init__(self,run,crash=False): self.run=run;self.crash=crash;self.ledger={'charged_seconds':0.}
                def tick(self):
                    self.ledger['charged_seconds']+=.001
                    p=self.run/'seeds'/str(w.SEEDS[0])/'checkpoint.meta.json'
                    if self.crash and p.is_file():
                        meta=json.loads(p.read_text())
                        if meta['epoch']==0 and meta['batch_cursor']>=64: raise Crash()
            with redirect_stdout(io.StringIO()):
                w.train_epoch(w.SEEDS[0],a,data,admission['scaler'],m,ROOT,InnovationModel,lib,Budget(a),'cpu')
                mb=json.loads((b/'manifest.json').read_text())
                with self.assertRaises(Crash):
                    w.train_epoch(w.SEEDS[0],b,data,admission['scaler'],mb,ROOT,InnovationModel,lib,Budget(b,True),'cpu')
                partial=torch.load(w.checkpoint_path(b,w.SEEDS[0]),weights_only=True,map_location='cpu')
                self.assertEqual(partial['batch_cursor'],64)
                w.train_epoch(w.SEEDS[0],b,data,admission['scaler'],mb,ROOT,InnovationModel,lib,Budget(b),'cpu')
            ca=torch.load(w.checkpoint_path(a,w.SEEDS[0]),weights_only=True,map_location='cpu')
            cb=torch.load(w.checkpoint_path(b,w.SEEDS[0]),weights_only=True,map_location='cpu')
            self.assertEqual(ca['history'],cb['history'])
            self.assertTrue(all(torch.equal(ca['model_state'][k],cb['model_state'][k]) for k in ca['model_state']))
            path=w.checkpoint_path(b,w.SEEDS[0]);before=w.sha(path)
            meta=json.loads(path.with_name('checkpoint.meta.json').read_text())
            meta.update(epoch=0,batch_cursor=0,checkpoint_sha256='0'*64)
            dump(path.with_name('checkpoint.meta.json'),meta)
            w.reconcile_metadata(b,mb,Budget(b),lib)
            self.assertEqual(w.sha(path),before)
            self.assertEqual(json.loads(path.with_name('checkpoint.meta.json').read_text())['epoch'],1)

    def test_training_member_access_guard_and_early_eval_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs,run=fixture(Path(tmp))
            original=np.load
            opened=[]
            class Guard:
                def __enter__(self): self.a=original(inputs/'current_cohort_arrays.npz',allow_pickle=False); return self
                def __exit__(self,*args): self.a.close()
                def __getitem__(self,key):
                    opened.append(key)
                    if key.startswith('evaluation_') or key.startswith('calibration_'): raise AssertionError(key)
                    return self.a[key]
            np.load=lambda *args,**kwargs:Guard()
            try:
                execute(inputs,run,n=1)
                with self.assertRaisesRegex(RuntimeError,'Training must complete'):
                    execute(inputs,run,phase='evaluate')
            finally: np.load=original
            self.assertEqual(set(opened),{'training_dates','training_raw','training_y_original'})

    def test_budget_is_terminal_no_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs,run=fixture(Path(tmp))
            now=[0.]
            def clock(): now[0]+=200.; return now[0]
            result=execute(inputs,run,clock=clock)
            self.assertEqual(result['status'],'budget_exhausted')
            old=json.loads((run/'ledger.json').read_text())['charged_seconds']
            result=execute(inputs,run)
            self.assertEqual(result['status'],'budget_exhausted')
            self.assertEqual(json.loads((run/'ledger.json').read_text())['charged_seconds'],old)
            self.assertFalse((run/'evaluation_predictions.csv').exists())


if __name__=='__main__': unittest.main()
