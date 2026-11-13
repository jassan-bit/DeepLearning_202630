"""Synthetic budget/transport/geometry tests; no market fits or CUDA claims."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import e2_posthoc_fit_reproduction as m


def anchor():
    return {'approved_total_seconds':m.TOTAL_SECONDS,'original_ledger_sha256':m.LEDGER_SHA256,
            'original_charged_seconds':m.ORIGINAL_CHARGED_SECONDS,'initial_remaining_seconds':m.REMAINING_SECONDS,
            'budget_reset':False,'original_approval_sha256':'a'*64}


def selection():
    return {'fixed_phase_epochs':{'ae1':1,'ae2':1,'supervised':1},
            'winner_trial':{'dropout':.3,'learning_rate':.001,'weight_decay':.0001,'batch_size':16}}


class MemoryGitHub:
    """Git-data CAS fixture; preserves the original campaign pointer."""
    max_conflicts=2
    _checked=True
    def __init__(self):
        self.head='a'*40
        self.files={'control/github-primary-v1/head.json':b'original-pointer-untouched'}
        self.blobs={};self.trees={};self.commits={}
    def _branch_head(self):return self.head
    def _content(self,path,head):
        raw=self.files.get(path)
        if raw is None:return None
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        return {'content':base64.b64encode(raw).decode(),'encoding':'base64','sha':blob}
    def _request(self,method,path,payload=None):
        if method=='GET' and path.startswith('/git/commits/'):
            return {'tree':{'sha':'tree-current'}}
        if method=='POST' and path=='/git/blobs':
            raw=base64.b64decode(payload['content']);sha=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
            self.blobs[sha]=raw;return {'sha':sha}
        if method=='POST' and path=='/git/trees':
            value=dict(self.files)
            for item in payload['tree']:value[item['path']]=self.blobs[item['sha']]
            sha=hashlib.sha1(repr(sorted(value)).encode()+str(len(self.trees)).encode()).hexdigest()
            self.trees[sha]=value;return {'sha':sha}
        if method=='POST' and path=='/git/commits':
            sha=hashlib.sha1(json.dumps(payload,sort_keys=True).encode()).hexdigest();self.commits[sha]=payload
            return {'sha':sha}
        if method=='PATCH' and path.startswith('/git/refs/'):
            commit=self.commits[payload['sha']]
            if commit['parents']!=[self.head]:raise RuntimeError('CAS conflict')
            self.files=self.trees[commit['tree']];self.head=payload['sha'];return {}
        raise AssertionError((method,path))


class LedgerTests(unittest.TestCase):
    def test_original_credit_preserved_charges_resume_and_duplicate_forbidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            with m.PosthocLedger(Path(tmp)/'budget',anchor()) as ledger:
                self.assertAlmostEqual(ledger.remaining_seconds,18169.090972095262)
                ledger.reserve('one',1860);ledger.charge('one',12,status='completed')
                ledger.charge_overhead(3)
                self.assertAlmostEqual(ledger.remaining_seconds,m.REMAINING_SECONDS-15)
            with m.PosthocLedger(Path(tmp)/'budget',anchor()) as ledger:
                self.assertAlmostEqual(ledger.remaining_seconds,m.REMAINING_SECONDS-15)
                with self.assertRaisesRegex(ValueError,'Duplicate'):ledger.reserve('one',1860)
                with self.assertRaises(ValueError):ledger.reserve('too-large',m.REMAINING_SECONDS)
    def test_unknown_reservation_survives_interrupt_and_blocks_reopen(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger=m.PosthocLedger(tmp,anchor());ledger.reserve('unknown',5460)
            ledger.charge('unknown',7,status='interrupted')
            self.assertEqual(ledger.report()['unresolved_reservations'],{'unknown':5460})
            ledger.close()
            with self.assertRaisesRegex(RuntimeError,'Unresolved'):m.PosthocLedger(tmp,anchor())
    def test_lock_corruption_and_overrun_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with m.PosthocLedger(tmp,anchor()) as ledger:
                with self.assertRaisesRegex(RuntimeError,'Another'):m.PosthocLedger(tmp,anchor())
                ledger.reserve('bounded',10)
                with self.assertRaisesRegex(RuntimeError,'overrun'):ledger.charge('bounded',11,status='failed')
                self.assertAlmostEqual(ledger.remaining_seconds,m.REMAINING_SECONDS-11)
            path=Path(tmp)/'posthoc_budget.jsonl';raw=path.read_text();path.write_text(raw.replace('"elapsed_seconds":11.0','"elapsed_seconds":1.0'))
            with self.assertRaises(ValueError):m.PosthocLedger(tmp,anchor())
    def test_remote_snapshot_failure_retains_reservation_and_prevents_compute(self):
        class State:
            def snapshot(self,reason):
                if reason.endswith('reserved'):raise RuntimeError('network unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            ledger=m.PosthocLedger(tmp,anchor());ledger.set_persistence(State())
            computed=False
            with self.assertRaisesRegex(RuntimeError,'network'):
                ledger.reserve('fit',1860)
                computed=True
            self.assertFalse(computed);self.assertEqual(ledger.active,{'fit':1860})
            ledger.close()
            with self.assertRaisesRegex(RuntimeError,'Unresolved'):m.PosthocLedger(tmp,anchor())
    def test_remote_CAS_new_namespace_and_unknown_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            client=MemoryGitHub();root=Path(tmp)/'posthoc'
            state=m.PosthocRemoteState(root,client,ROOT);state.restore()
            with m.PosthocLedger(root/'budget',anchor()) as ledger:
                ledger.set_persistence(state);ledger.reserve('lost-fit',1860)
            self.assertEqual(client.files['control/github-primary-v1/head.json'],b'original-pointer-untouched')
            self.assertIn('control/e2-guide-closure-posthoc-v1/head.json',client.files)
            shutil.rmtree(root)
            with self.assertRaisesRegex(RuntimeError,'unresolved'):
                m.PosthocRemoteState(root,client,ROOT).restore()
            self.assertTrue((root/'budget/posthoc_budget.jsonl').is_file())
    def test_remote_advanced_state_rejects_stale_local(self):
        with tempfile.TemporaryDirectory() as tmp:
            client=MemoryGitHub();root=Path(tmp)/'posthoc'
            state=m.PosthocRemoteState(root,client,ROOT);state.restore()
            with m.PosthocLedger(root/'budget',anchor()) as ledger:
                ledger.set_persistence(state)
                old=(root/'budget/posthoc_budget.jsonl').read_bytes() if (root/'budget/posthoc_budget.jsonl').exists() else None
                ledger.reserve('done',10);ledger.charge('done',1)
            (root/'budget/posthoc_budget.jsonl').write_bytes(old or b'')
            with self.assertRaisesRegex(Exception,'diverges'):
                m.PosthocRemoteState(root,client,ROOT).restore()


class ScientificHelpersTests(unittest.TestCase):
    def test_exact_ablation_design_and_frozen_dropout_guard(self):
        d=m.ablation_design(selection())
        self.assertEqual(set(d[m.VARIANTS[0]]['fixed_epochs']),{'supervised'})
        self.assertEqual(set(d[m.VARIANTS[1]]['fixed_epochs']),{'ae1','supervised'})
        self.assertEqual(d[m.VARIANTS[2]]['trial']['dropout'],0.)
        invalid=selection();invalid['winner_trial']['dropout']=.1
        with self.assertRaises(ValueError):m.ablation_design(invalid)
    def test_actual_synthetic_three_conditions_save_new_weights_and_expected_phases(self):
        import torch
        torch.set_num_threads(1);torch.manual_seed(991)
        train=SimpleNamespace(x=torch.randn(16,7,5),y=torch.rand(16)+.5,signature='synthetic_only',
            scaler=SimpleNamespace(target_scale=1.,record=lambda:{'target_scale':1.}),
            record=lambda:{'signature':'synthetic_only','dates_scope':'synthetic_no_market'})
        with tempfile.TemporaryDirectory() as tmp:
            states=[]
            for variant,phases in zip(m.VARIANTS,({'supervised'},{'ae1','supervised'},{'ae1','ae2','supervised'})):
                model,result=m.fit_ablation({'variant':variant,'seed':m.SEEDS[0],'selection':selection(),
                    'device':'cpu','phase_wall_seconds':60},train,Path(tmp)/variant,allow_cpu_for_tests=True)
                self.assertEqual(set(result['phases']),phases)
                self.assertTrue((Path(tmp)/variant/'selected_model.pt').is_file())
                self.assertEqual(model(train.x).shape,(16,));self.assertTrue(bool((model(train.x)>0).all()))
                self.assertEqual(result['architecture']['dropout'],0. if variant==m.VARIANTS[2] else .3)
                self.assertNotEqual(result['optimizer_steps'],0);states.append(model.state_dict())
            self.assertNotIn('encoder2.weight_ih_l0',states[1])
            self.assertIn('encoder2.weight_ih_l0',states[0])
            with self.assertRaisesRegex(RuntimeError,'never restart'):
                m.fit_ablation({'variant':m.VARIANTS[0],'seed':m.SEEDS[0],'selection':selection(),'device':'cpu'},
                               train,Path(tmp)/m.VARIANTS[0],allow_cpu_for_tests=True)
    def test_state_comparison_detects_weights_and_geometry(self):
        import torch
        a={'w':torch.tensor([1.,2.]),'n':torch.tensor([1])}
        self.assertTrue(m.compare_states(a,a)['state_tensor_bitwise_equal'])
        b={**a,'w':torch.tensor([1.,2.25])}
        self.assertEqual(m.compare_states(a,b)['maximum_state_tensor_absolute_difference'],.25)
        with self.assertRaises(ValueError):m.compare_states(a,{'w':torch.ones(3)})
    def test_bootstrap_estimand_direction_Holm_and_repeatability(self):
        import numpy as np
        y=np.zeros(28);full=np.zeros((6,28))
        variants={name:np.ones((6,28))*(i+1) for i,name in enumerate(m.VARIANTS)}
        a=m.compare_ablations(y,full,variants);b=m.compare_ablations(y,full,variants)
        self.assertEqual(a,b)
        self.assertEqual([r['delta_mean_seed_RMSE_ablated_minus_full'] for r in a],[1.,2.,3.])
        self.assertTrue(all(r['p_Holm3']>=r['p_two_sided_centered'] for r in a))
        same=m.compare_ablations(y,full,{k:full for k in m.VARIANTS})
        self.assertTrue(all(r['p_Holm3']==1. for r in same))
        with self.assertRaises(ValueError):m.compare_ablations(y,full,variants,draws=9999)
    def test_completed_receipt_binds_code_config_and_artifact_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'weights.pt').write_bytes(b'synthetic')
            result={'status':'completed','controller_source_sha256':'a'*64,'configuration_commit':'b'*40,
                    'artifacts_sha256':{'weights.pt':m.file_sha(p/'weights.pt')}}
            m.immutable_json(p/'result.json',result)
            self.assertIsNotNone(m.verified_completed(p,controller_source_sha256='a'*64,configuration_commit='b'*40))
            with self.assertRaisesRegex(RuntimeError,'different controller'):
                m.verified_completed(p,controller_source_sha256='c'*64)
            (p/'weights.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError):m.verified_completed(p)
    def test_fresh_venv_copies_interpreter_integration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'new-venv'
            subprocess.run([sys.executable,'-m','venv','--without-pip','--copies',path],check=True,
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            executable=m.regular(path/'bin/python')
            profile=json.loads(subprocess.check_output([str(executable),'-I','-c',
                'import json,sys; print(json.dumps({"prefix":sys.prefix,"base":sys.base_prefix}))'],text=True))
            self.assertEqual(Path(profile['prefix']),path);self.assertNotEqual(profile['prefix'],profile['base'])
    def test_bounded_child_is_reaped_on_timeout(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            m.run_bounded([sys.executable,'-c','import time; time.sleep(10)'],timeout=.15)
    def test_launcher_dry_run_never_downloads_or_starts_computation(self):
        spec=importlib.util.spec_from_file_location('synthetic_launcher',ROOT/'notebooks/run_e2_posthoc_fit_reproduction_colab.py')
        launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
        with patch.object(launcher,'download',side_effect=AssertionError('network forbidden')):
            result=launcher.run_e2_posthoc_fit_reproduction_colab('a'*40)
        self.assertFalse(result['execute']);self.assertFalse(result['new_budget_authorization'])
        self.assertEqual(len(result['steps']),5)


if __name__=='__main__':unittest.main()
