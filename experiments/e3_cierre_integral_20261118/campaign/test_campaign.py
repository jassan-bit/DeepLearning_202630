"""Synthetic invariants; never opens market fixtures or initiates network IO."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

HERE=Path(__file__).resolve().parent


def module(name):
    spec=importlib.util.spec_from_file_location('campaign_test_'+name,HERE/(name+'.py'))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


pilot=module('pilot');data=module('data');power=module('power');colab=module('colab')
torch.set_num_threads(1)


def fake_npz(path):
    rng=np.random.default_rng(1938)
    dates=np.arange(np.datetime64('2024-06-01'),np.datetime64('2024-07-21')).astype('U10')
    prices=np.exp(rng.normal(0,.03,(len(dates),7,4)))
    raw=prices.copy();raw[:,:,1]=np.maximum.reduce([prices[:,:,0],prices[:,:,1],prices[:,:,3]])
    raw[:,:,2]=np.minimum.reduce([prices[:,:,0],prices[:,:,2],prices[:,:,3]])
    raw=np.concatenate([raw,rng.lognormal(5,1,(len(dates),7,1))],axis=-1)
    known=np.diff(np.log(raw[:,:,3]),axis=1)
    A=((known-known.mean(1)[:,None])**2).sum(1)/7
    U=rng.lognormal(-3,.5,len(dates));y=np.sqrt(A+(6/49)*U**2)
    # Any accidental test-array access with allow_pickle=False raises.
    np.savez(path,training_raw=raw,training_y_original=y,training_dates=dates,
             test_raw=np.asarray([{'forbidden':'sentinel'}],dtype=object))
    return pilot.sha(path)


class CommonDataTests(unittest.TestCase):
    def test_variance_identity_and_purged_raw_support(self):
        rng=np.random.default_rng(77);close=np.exp(rng.normal(0,.01,400).cumsum())
        raw=np.column_stack([close,close*1.01,close*.99,close,np.ones(len(close))])
        panel=data.windows(raw)
        np.testing.assert_allclose(panel['y']**2,panel['A']+(6/49)*panel['U']**2,atol=1e-15)
        proposal=data.propose_split_rows(len(raw));parts=list(proposal['parts'].values())
        for left,right in zip(parts,parts[1:]):
            # Window index i has close support i..i+7, including its target.
            self.assertLess(left['window_stop_exclusive']-1+7,right['window_start'])

    def test_heldout_price_and_volume_do_not_refit_scaler(self):
        rng=np.random.default_rng(2);raw=np.exp(rng.normal(size=(100,5))*.1)
        raw[:,1]=raw[:,:4].max(1);raw[:,2]=raw[:,:4].min(1)
        p=data.windows(raw);_,first=data.training_scaler(p,np.arange(30))
        q={k:v.copy() for k,v in p.items()};q['raw_X'][40:]*=10
        q['y'][40:]*=999;q['U'][40:]*=999
        _,second=data.training_scaler(q,np.arange(30));self.assertEqual(first,second)

    def test_reserved_access_fails_before_reading_path(self):
        with self.assertRaisesRegex(ValueError,'final protocol freeze'):
            data.build_reserved_dataset('/file/not/present.json','D01',frozen_protocol=None)


class TrainingTests(unittest.TestCase):
    def test_attempt_marker_counts_failure_before_first_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'output';job=output/'jobs'/'interrupted__17';phase=job/'supervised'
            x=torch.ones(8,3);y=torch.ones(8,1);model=torch.nn.Linear(3,1)
            config={**pilot.DEFAULT_CONFIG,'max_epochs':1}
            identity={'name':'synthetic_fixture','seed':17,'phase':'supervised',
                      'source_identity':{'source_commit':'a'*40,'fixture_only':True}}
            with patch.object(pilot,'_torch_save',side_effect=InterruptedError('first checkpoint failed')):
                with self.assertRaises(InterruptedError):
                    pilot._phase(model,x,y,x,y,phase,config,identity=identity)
            self.assertFalse((phase/'checkpoint.pt').exists())
            marker=phase/'phase_started.json';before=marker.read_bytes()
            self.assertEqual(json.loads(before)['source_identity']['source_commit'],'a'*40)
            self.assertEqual(colab._fit_attempts([job]),(1,1))
            snapshot=colab.create_snapshot(output,Path(tmp)/'snapshot','a'*40,
                '20261118T203000000000Z',[job],partial=True,prior_saves=[])
            receipt=json.loads((snapshot/'stage_receipt.json').read_text())
            self.assertEqual(receipt['new_real_data_fits'],1)
            self.assertEqual(receipt['attempted_optimizer_phases'],1)
            self.assertEqual(receipt['available_phase_checkpoints'],0)
            pilot._phase(model,x,y,x,y,phase,config,identity=identity)
            self.assertEqual(marker.read_bytes(),before)
            self.assertEqual(colab._fit_attempts([job]),(1,1))

    def test_SAE_executes_both_reconstruction_phases_before_predictor(self):
        from src.e3_innovation_model import load_innovation_data
        with tempfile.TemporaryDirectory() as tmp:
            npz=Path(tmp)/'synthetic.npz';sha=fake_npz(npz)
            panel=load_innovation_data(npz,sha)
            pred,receipt=pilot._fit_neural('LSTM-autoencoder con predictor',17,panel,
                Path(tmp)/'sae',{**pilot.DEFAULT_CONFIG,'max_epochs':2,'patience':2},'cpu',
                rows={'LSTM-autoencoder con predictor':{'trial':{'dropout':.3}}})
            self.assertEqual(set(receipt['phases']),{'ae1','ae2','supervised'})
            self.assertTrue(receipt['sae_pretraining_executed'])
            self.assertTrue(all(p['steps']>0 for p in receipt['phases'].values()))
            self.assertEqual(pred.shape,panel['val_y_original'].shape)

    def test_epoch_checkpoint_resume_matches_uninterrupted_optimization(self):
        with tempfile.TemporaryDirectory() as tmp:
            pilot._seed(21);x=torch.randn(31,3);y=torch.randn(31,1)
            def new_model():return torch.nn.Sequential(torch.nn.Linear(3,8),torch.nn.Dropout(.2),torch.nn.Linear(8,1))
            conf={**pilot.DEFAULT_CONFIG,'max_epochs':3,'patience':3,'batch_size':7}
            pilot._seed(41);a=new_model()
            complete=pilot._phase(a,x,y,x,y,Path(tmp)/'complete',conf,identity={'fixture':'synthetic'})
            pilot._seed(41);b=new_model();save=pilot._torch_save
            def interrupt_after_epoch(path,value):
                save(path,value)
                if value['epoch']==1:raise InterruptedError('simulated interruption AFTER durable epoch')
            with patch.object(pilot,'_torch_save',interrupt_after_epoch):
                with self.assertRaises(InterruptedError):
                    pilot._phase(b,x,y,x,y,Path(tmp)/'interrupted',conf,identity={'fixture':'synthetic'})
            pilot._seed(777);c=new_model()
            resumed=pilot._phase(c,x,y,x,y,Path(tmp)/'interrupted',conf,identity={'fixture':'synthetic'})
            self.assertEqual(complete['steps'],resumed['steps'])
            for key in complete['latest']:self.assertTrue(torch.equal(complete['latest'][key],resumed['latest'][key]))
            for key in complete['best']:self.assertTrue(torch.equal(complete['best'][key],resumed['best'][key]))

    def test_actual_control_fit_uses_training_members_only_and_reuses_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz=Path(tmp)/'synthetic.npz';sha=fake_npz(npz);out=Path(tmp)/'run'
            args={'suite':'controls','seeds':(31,),'device':'cpu',
                  'config':{'max_epochs':2,'patience':2},'allow_cpu_for_synthetic_tests':True}
            first=pilot.run(out,'a'*40,npz,sha,**args)
            self.assertEqual(first['status'],'completed');self.assertEqual(first['jobs_completed'],7)
            self.assertFalse(first['real_campaign_allowed']);self.assertEqual(first['new_test_predictions'],0)
            checkpoints={str(p):pilot.sha(p) for p in out.rglob('selected_model.pt')}
            second=pilot.run(out,'a'*40,npz,sha,**args)
            self.assertEqual(second['status'],'completed')
            self.assertEqual(checkpoints,{str(p):pilot.sha(p) for p in out.rglob('selected_model.pt')})


class PowerTests(unittest.TestCase):
    def test_Holm_is_global_and_all_three_must_reject(self):
        values=np.ones((1,51));values[0,:3]=[.001,.001,.049]
        adjusted=power.holm(values)
        self.assertGreater(adjusted[0,2],.05)
        self.assertGreater(adjusted[0,0],.05)

    def test_incomplete_controls_panel_is_not_power_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);panel=path/'development_predictions.csv';panel.write_text('model,seed,date,target,prediction\n')
            pilot.write_json(path/'request.json',{'test_only':False,'suite':'controls','family':list(pilot.CONTROLS),'seeds':[1,2,3]})
            pilot.write_json(path/'pilot_receipt.json',{'status':'completed','predictions_sha256':pilot.sha(panel)})
            report=power.run(path,path/'power',margin=.02,excess=.02,simulations=1000)
            self.assertFalse(report['power_dimensioned']);self.assertEqual(report['status'],'not_ready')

    def test_joint_simulation_rejects_null_in_fewer_than_family_alpha(self):
        rng=np.random.default_rng(73);gains=rng.normal(0,.1,(20,17))
        null=power.simulate(gains,seeds=20,margin=.02,excess=1e-10,simulations=2000)
        strong=power.simulate(gains,seeds=40,margin=.02,excess=.5,simulations=1000)
        self.assertLess(null['joint_all_51_power'],.05)
        self.assertGreater(strong['joint_all_51_power'],.99)
        self.assertEqual(strong['global_family_size'],51)


class SnapshotTests(unittest.TestCase):
    def test_controller_preserves_initial_explicit_context_and_incremental_parts(self):
        api=colab.load(pilot.ROOT/'experiments/e3_remediacion_20261118/evaluation/colab_github.py','controller_fixture_api')
        store=colab.load(pilot.ROOT/'experiments/e3_cierre_integral_20261118/orchestration/github_store.py','controller_fixture_store')
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'explicit-first-output';saved=[]
            def save(evidence,token,source,part,actual_api,profile):
                files=store.snapshot(evidence,token,source,part,actual_api,profile)
                saved.append(files)
                return {'run_url':'https://github.com/fixture/part/'+part,'saved':True}
            runtime=SimpleNamespace(prepare_runtime=lambda _: (Path('/unused/python'),{}, {'fixture':True}))
            def load(path,name):
                if Path(path).name=='colab_github.py':return api
                if Path(path).name=='github_store.py':return SimpleNamespace(save=save)
                if Path(path).name=='colab_runtime.py':return runtime
                raise AssertionError(path)
            class FakeWorker:
                def __init__(self,*args,**kwargs):
                    for seed in (1,2,3):
                        job=output/'jobs'/f'fixture__{seed}';phase=job/'supervised';phase.mkdir(parents=True)
                        torch.save({'w':torch.ones(2)},phase/'checkpoint.pt')
                        pilot.write_json(job/'result.json',{'status':'completed'})
                    pilot.write_json(output/'pilot_receipt.json',{'status':'completed'})
                    self.stdout=iter(['fixture: completed; development RMSE=.1\n']*3)
                def wait(self,**kwargs):return 0
            with patch.object(colab,'checkout',return_value=pilot.ROOT),patch.object(colab,'load',side_effect=load),\
                 patch.object(colab,'admit_npz',return_value=Path('/unused/fixture.npz')),\
                 patch.object(api,'validate_credentials',return_value=True),\
                 patch.object(colab.subprocess,'Popen',FakeWorker):
                result=colab.run('a'*40,'not-a-real-credential',base=tmp,resume_output=output)
            self.assertEqual(result['worker_exit_code'],0)
            self.assertTrue((output/'isolated_runtime.json').is_file())
            self.assertEqual(len(saved),2)
            first=json.loads(saved[0]['stage_receipt.json']);last=json.loads(saved[-1]['stage_receipt.json'])
            self.assertEqual(first['new_real_data_fits'],3)
            self.assertEqual(last['new_real_data_fits'],0)
            self.assertEqual(len(last['prior_saved_parts']),1)
            self.assertFalse(last['real_campaign_allowed'])

    def test_raw_torch_checkpoints_survive_sharding_byte_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'output';job=out/'jobs'/'fixture__17';phase=job/'supervised'
            phase.mkdir(parents=True);torch.save({'state_dict':{'w':torch.ones(3)}},phase/'checkpoint.pt')
            pilot.write_json(out/'request.json',{'source_commit':'a'*40,'test_only':True})
            snapshot=colab.create_snapshot(out,Path(tmp)/'snapshot','a'*40,
                '20261118T203000000000Z',[job],partial=True,prior_saves=[])
            self.assertEqual(pilot.sha(phase/'checkpoint.pt'),pilot.sha(snapshot/'jobs/fixture__17/supervised/checkpoint.pt'))
            api=colab.load(pilot.ROOT/'experiments/e3_remediacion_20261118/evaluation/colab_github.py','fixture_api')
            store=colab.load(pilot.ROOT/'experiments/e3_cierre_integral_20261118/orchestration/github_store.py','fixture_store')
            files=store.snapshot(snapshot,'not-a-real-credential','a'*40,'20261118T203000000000Z',api,'pilot')
            self.assertIn('jobs/fixture__17/supervised/checkpoint.pt',files)

    def test_GPU_child_environment_cannot_inherit_token_variable(self):
        with patch.dict('os.environ',{'GITHUB_TOKEN':'fixture-secret-value','SOME_PASSWORD':'fixture-password'}):
            env=colab._environment()
            self.assertNotIn('GITHUB_TOKEN',env);self.assertNotIn('SOME_PASSWORD',env)


if __name__=='__main__':unittest.main()
