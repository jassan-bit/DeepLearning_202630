"""Generated fixtures only. No real key, restore or new market inference."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('reanalysis',HERE/'frozen_v5_reanalysis.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class FixtureModel(torch.nn.Module):
    def __init__(self,seed):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(float(seed%100)/1000,dtype=torch.float64))
    def forward_components(self,X,A,scale):
        if self.training or torch.is_grad_enabled():raise AssertionError('Inference-only fixture requires eval/no gradients')
        return {'prediction':torch.sqrt(A)+.02+self.weight*.01+X[:,0,0].to(torch.float64)*.0001}


def fixture():
    rng=np.random.default_rng(9723);data={'scalers':{m.RECIPE:{'u_scale':.01,'y_scale':.02}}}
    for split,n in (('train',532),('selection',54),('assessment',55)):
        X=rng.normal(size=(n,7,5)).astype(np.float32)
        data[split]={'X':{m.RECIPE:X},'A':np.full(n,.0001),'scale':np.ones(n),
            'y':np.full(n,.031)+rng.normal(scale=.0001,size=n),
            'dates':np.asarray([f'SYNTHETIC-{split}-{i:03d}' for i in range(n)])}
    evidence={'jobs':{},'assessment':{}};states={};calls=[]
    for seed in m.SEEDS:
        model=FixtureModel(seed);model.eval()
        with torch.inference_mode():
            def pred(split):
                s=data[split]
                return model.forward_components(torch.as_tensor(s['X'][m.RECIPE]),torch.as_tensor(s['A']),torch.as_tensor(s['scale']))['prediction'].numpy()
            selection=pred('selection');assessment=pred('assessment')
        request='FIXTURE-REQUEST-'+str(seed);checkpoint='FIXTURE-CHECKPOINT-'+str(seed)
        evidence['jobs'][seed]={'job':{'job_id':f'seed{seed}/{m.RECIPE}'},'request_sha256':request,
            'checkpoint_sha256':checkpoint,'best_epoch':m.BEST_EPOCHS[seed],
            'selection_RMSE':float(np.sqrt(np.mean((selection-data['selection']['y'])**2)))}
        evidence['assessment'][seed]=[{'date':str(date),'target':str(y),'prediction':str(p)}
            for date,y,p in zip(data['assessment']['dates'],data['assessment']['y'],assessment)]
        best=copy.deepcopy(model.state_dict());current=copy.deepcopy(best);current['weight']+=100
        states[seed]={'request_sha256':request,'best_epoch':m.BEST_EPOCHS[seed],'best':best,'current':current}
    def loader(path,request):
        seed=int(path.parent.name[4:]);calls.append(seed)
        return states[seed],{'sha256':evidence['jobs'][seed]['checkpoint_sha256']}
    factory=types.SimpleNamespace(build_model=lambda recipe,seed,**scales:FixtureModel(seed))
    return data,evidence,states,loader,factory,calls


class Tests(unittest.TestCase):
    def test_remote_store_is_GET_only_and_ref_is_immutable(self):
        class Base:
            def __init__(self,**kwargs):self.calls=[]
            def _request(self,method,path,payload=None,**options):self.calls.append((method,path));return {'ok':True}
        store=m.pinned_readonly_store(types.SimpleNamespace(GithubStore=Base),'FIXTURE_TOKEN','FIXTURE_KEY')
        self.assertEqual(store._ref()['object']['sha'],m.RESULT_COMMIT)
        store._request('GET','/fixture')
        for method in ('POST','PATCH','PUT','DELETE'):
            with self.subTest(method=method),self.assertRaisesRegex(m.ReanalysisError,'remote_write_forbidden'):
                store._request(method,'/fixture',{})
        self.assertEqual(store.calls,[('GET','/fixture')])

    def test_existing_private_case_never_requests_credentials_or_remote(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=m.recover(types.SimpleNamespace(),Path(tmp),{})
            self.assertFalse(result['restored']);self.assertEqual(result['remote_GET_operations'],0)

    def test_missing_key_stops_before_any_remote_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(m.ReanalysisError,'read_token'):
                m.recover(types.SimpleNamespace(),Path(tmp)/'fresh',{})

    def test_invalid_source_and_public_results_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'SOURCE_MANIFEST.json').write_text('{}')
            with self.assertRaisesRegex(m.ReanalysisError,'source_manifest'):m.verify_source(root)
            with self.assertRaisesRegex(m.ReanalysisError,'public_results'):m.public_evidence(root/'SOURCE_MANIFEST.json')

    def test_original_crypto_fixture_wrong_key_and_context_rejected(self):
        root=Path('/workspace/e3_seoi_v5_github_clean_cache/source-'+m.SOURCE_COMMIT)
        root=m.verify_source(root)
        _,module=m.import_original(root)
        token='FIXTURE-ONLY-TOKEN-'+'x'*24;secret='FIXTURE-ONLY-BACKUP-KEY-'+'a'*40
        store=m.pinned_readonly_store(module,token,secret)
        context={'scope':'SOFTWARE_FIXTURE_ONLY','source_commit':m.SOURCE_COMMIT}
        payload=store._encrypt(b'fixture-not-market-not-checkpoint',context)
        self.assertEqual(store._decrypt(payload,context),b'fixture-not-market-not-checkpoint')
        wrong=m.pinned_readonly_store(module,token,'FIXTURE-WRONG-BACKUP-KEY-'+'b'*40)
        with self.assertRaisesRegex(module.StorageError,'wrong_secret_or_tampering'):
            wrong._decrypt(payload,context)
        with self.assertRaisesRegex(module.StorageError,'context_or_KDF_differs'):
            store._decrypt(payload,{'scope':'DIFFERENT-FIXTURE'})
        self.assertEqual(store.get_operations,0)

    def test_original_public_evidence_and_best_epochs(self):
        path=Path('/workspace/e3_input_only_candidate_20261121/inputs/public_results.zip')
        evidence=m.public_evidence(path)
        self.assertEqual(evidence['plan']['case_id'],m.CASE_ID)
        self.assertEqual({s:r['best_epoch'] for s,r in evidence['jobs'].items()},m.BEST_EPOCHS)

    def test_original_data_bindings_without_prediction_or_reserved_arrays(self):
        root=m.verify_source(Path('/workspace/e3_seoi_v5_github_clean_cache/source-'+m.SOURCE_COMMIT))
        runner,_=m.import_original(root);data_module,_=runner._modules(root)
        evidence=m.public_evidence(Path('/workspace/e3_input_only_candidate_20261121/inputs/public_results.zip'))
        plan=evidence['plan'];path=Path('/workspace/e3_seoi_v5_github_clean_cache/data/current_cohort_arrays.npz')
        self.assertEqual(m.sha(path.read_bytes()),m.DATA_SHA)
        self.assertEqual(data_module.ALLOWED_KEYS,('training_raw','training_y_original','training_dates'))
        data=data_module.load_data(path)
        self.assertEqual(runner.data_identity(data),plan['data_identities'])
        self.assertEqual(runner.digest(data['admission']),plan['admission_sha256'])
        self.assertEqual(runner.digest(data['scalers']),plan['scalers_sha256'])
        self.assertEqual([len(data[k]['y']) for k in ('train','selection','assessment')],[532,54,55])

    def test_optional_launcher_default_does_not_download_install_or_request_secret(self):
        spec=importlib.util.spec_from_file_location('launcher',HERE/'colab_optional_recovery.py')
        launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
        with patch.object(launcher,'download_verified',side_effect=AssertionError('NETWORK_FORBIDDEN')), \
             patch.object(launcher,'select_python',side_effect=AssertionError('INSTALL_FORBIDDEN')), \
             patch.object(launcher,'private_secret',side_effect=AssertionError('SECRET_FORBIDDEN')):
            result=launcher.execute()
        self.assertEqual(result['status'],'DISABLED_NO_DOWNLOAD_NO_INSTALL_NO_SECRET_NO_INFERENCE')

    def test_best_state_only_eval_mode_no_optimizer_and_exact_counts(self):
        data,evidence,states,loader,factory,calls=fixture()
        with patch.object(torch.optim.Optimizer,'step',side_effect=AssertionError('OPTIMIZER_FORBIDDEN')):
            rows,metrics,parity,counts=m.evaluate_frozen(None,factory,data,Path('/FIXTURE'),evidence,
                load_snapshot=loader,fixture=True)
        self.assertEqual(len(rows),2930);self.assertEqual(len(metrics),10);self.assertEqual(len(parity),5)
        self.assertEqual(counts['status'],'SOFTWARE_FIXTURE_ONLY')
        self.assertEqual(counts['total_forward_points'],3205)
        self.assertEqual(calls,list(m.SEEDS))
        first=next(r for r in rows if r['seed']==m.SEEDS[0] and r['split']=='fit')
        expected=.01+.02+(m.SEEDS[0]%100)/1000*.01+float(data['train']['X'][m.RECIPE][0,0,0])*.0001
        self.assertAlmostEqual(first['prediction'],expected)
        self.assertLess(first['prediction'],.1)  # A wrong 'current' state would add1.0.

    def test_changed_best_epoch_stops_without_outputs(self):
        data,evidence,states,loader,factory,_=fixture();states[m.SEEDS[0]]['best_epoch']+=1
        with self.assertRaisesRegex(m.ReanalysisError,'selected_state_identity'):
            m.evaluate_frozen(None,factory,data,Path('/FIXTURE'),evidence,load_snapshot=loader,fixture=True)

    def test_changed_assessment_predictions_stop_at_frozen_gate(self):
        data,evidence,states,loader,factory,_=fixture()
        evidence['assessment'][m.SEEDS[0]][0]['prediction']='100.0'
        with self.assertRaisesRegex(m.ReanalysisError,'assessment_parity_failed'):
            m.evaluate_frozen(None,factory,data,Path('/FIXTURE'),evidence,load_snapshot=loader,fixture=True)

    def test_changed_selection_metric_stops_at_frozen_gate(self):
        data,evidence,states,loader,factory,_=fixture()
        evidence['jobs'][m.SEEDS[0]]['selection_RMSE']=100.
        with self.assertRaisesRegex(m.ReanalysisError,'selection_metric_parity_failed'):
            m.evaluate_frozen(None,factory,data,Path('/FIXTURE'),evidence,load_snapshot=loader,fixture=True)

    def test_wrong_snapshot_hash_stops_before_model_construction(self):
        data,evidence,states,loader,factory,_=fixture()
        def wrong(path,request):
            state,pointer=loader(path,request);pointer['sha256']='WRONG';return state,pointer
        factory.build_model=lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('must not construct'))
        with self.assertRaisesRegex(m.ReanalysisError,'selected_state_identity'):
            m.evaluate_frozen(None,factory,data,Path('/FIXTURE'),evidence,load_snapshot=wrong,fixture=True)

    def test_outputs_are_private_immutable_and_hash_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'private'
            m.write_outputs(output,[{'fixture':True}],[{'fixture':True}],[{'fixture':True}],{'status':'SOFTWARE_FIXTURE_ONLY'})
            self.assertEqual(output.stat().st_mode & 0o777,0o700)
            self.assertTrue(all(p.stat().st_mode & 0o777==0o600 for p in output.iterdir()))
            self.assertEqual(m.verify_outputs(output)['new_forward_points'],0)
            with self.assertRaisesRegex(m.ReanalysisError,'existing_outputs_preserved'):
                m.write_outputs(output,[{}],[{}],[{}],{})
            p=output/'fit_selection_predictions_PRIVATE.csv';p.write_text('tampered')
            with self.assertRaisesRegex(m.ReanalysisError,'output_hash_differs'):m.verify_outputs(output)


if __name__=='__main__':unittest.main(verbosity=2)
