"""Fixed synthetic contract checks: all 21 real methods, no market observations."""
from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path

import numpy as np
import unittest

_assertion = unittest.TestCase()
import torch

from engine import models

SEED = 2870326772251854041  # A genuine full-63-bit initialization fixture.


def fixture():
    torch.set_num_threads(1)
    rng = np.random.default_rng(20261121)
    close = np.exp(4.5+np.cumsum(rng.normal(0,.001,180)))
    bars = np.stack((close*.999,close*1.01,close*.99,close,np.arange(180)+100.),axis=1)
    raw = np.stack([bars[i-7:i] for i in range(14,174)])
    A = models.candidate_data.known_geometry(raw)[0]
    y = np.sqrt(A+(6/49)*rng.uniform(.003,.03,len(A))**2)
    return raw,y,models.fit_preprocessor(raw[:100],y[:100])


def _check_complete_frozen_family_and_exact_optimization_rows():
    assert len(models.METHODS)==len(set(models.METHODS))==21
    assert len(models.NEURAL)==16 and len(models.DETERMINISTIC)==5
    assert models.METHODS[0]=='input_only_mse' and 'global_mse' in models.METHODS
    assert models.canonical_name(models.CANDIDATE_IDENTITY)=='input_only_mse'
    config = models.model_config('input_only_mse')
    for k,v in models.candidate.TRAINING.items():
        if k!='selection':
            assert config[k]==v
    assert models.model_config('PatchTST')['warmup_fraction']==.1
    assert models.model_config('FCN-1D')['learning_rate']==.0005695394539643588
    assert models.model_config('GRU32')['weight_decay']==.006658367249304354
    assert models.model_config('ARIMA')['trial']['order']==[2,0,1]
    assert models.verify_sources()['files_verified']>=100
    with _assertion.assertRaisesRegex(ValueError,'Unknown'):
        models.canonical_name('undefined_model')


def _check_real_forward_gradient_and_selected_state_factory_roundtrip(name,fixture):
    raw,y,state = fixture
    torch.manual_seed(1641)
    caller_rng = torch.get_rng_state().clone()
    model = models.build_model(name,SEED,state)
    assert torch.equal(torch.get_rng_state(),caller_rng), 'Factory must preserve caller Torch RNG'
    assert isinstance(model,torch.nn.Module)
    model.train()
    loss = models.loss(model,name,raw[:3],y[:3],state)
    assert loss.ndim==0 and bool(torch.isfinite(loss))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
    assert any(bool(g.abs().max()>0) for g in grads)
    # One software-fixture optimizer step makes the roundtrip test distinguish
    # restored selected weights from merely rebuilding the seeded factory.
    opt = torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=0.)
    opt.step()
    before = models.predict(model,name,raw[:3],state)
    stream = io.BytesIO()
    torch.save(model.state_dict(),stream)
    stream.seek(0)
    rebuilt = models.build_model(name,SEED,state)
    rebuilt.load_state_dict(torch.load(stream,weights_only=True),strict=True)
    after = models.predict(rebuilt,name,raw[:3],state)
    for a,b in zip(before,after):
        np.testing.assert_array_equal(a,b)
    assert np.all(before[1]>=before[2])
    assert model.architecture_metadata['selected_recipe']==models._RECIPES['models'][name]['trial']


def _check_candidate_exact_frozen_preprocessing_architecture_and_mse(fixture):
    raw,y,state = fixture
    expected_state = models.candidate.fit_inputs(raw[:100],y[:100])
    assert state['candidate']==expected_state
    model = models.build_model('input_only_mse',SEED,state)
    expected = models.candidate.make_model(SEED,expected_state)
    assert model.state_dict().keys()==expected.state_dict().keys()
    for key,value in model.state_dict().items():
        assert torch.equal(value,expected.state_dict()[key])
    X,A,s = models.transform(raw[:4],state,'input_only_mse')
    ex = models.candidate.transform_known_history(raw[:4],expected_state)
    for a,b in zip((X,A,s),ex[:3]):
        np.testing.assert_array_equal(a,b)
    expected.eval();model.eval()
    out = expected.forward_components(torch.from_numpy(X),torch.from_numpy(A),torch.from_numpy(s))
    U = np.sqrt(np.maximum(y[:4]**2-A,0)/models.candidate_data.C)
    frozen_loss = expected.loss_components(out,y[:4],U)['loss']
    assert torch.equal(models.loss(model,'input_only_mse',raw[:4],y[:4],state),frozen_loss)


def _check_sae_actual_three_stage_graph_and_exact_encoder_transfer(fixture):
    raw,y,state = fixture
    model = models.build_model(models.SAE,SEED,state)
    assert models.training_phases(models.SAE)==('ae1','ae2','supervised')
    assert not model.completed_phases.any()
    with _assertion.assertRaisesRegex(RuntimeError,'AE2 requires'):
        models.phase_loss(model,models.SAE,'ae2',raw[:3],y[:3],state)
    X,_,_ = models.transform(raw[:3],state,models.SAE)
    target = torch.from_numpy(X)/state['sae_auxiliary_scale']
    expected_ae1 = (model.ae1(torch.from_numpy(X))-target).square().mean()
    assert torch.equal(models.phase_loss(model,models.SAE,'ae1',raw[:3],y[:3],state),expected_ae1)
    # Exercise both reconstruction phases and the supervised phase with fresh
    # optimizers, then verify selected pretrained encoders are copied exactly.
    for phase in models.training_phases(models.SAE):
        old1 = {k:v.detach().clone() for k,v in model.ae1.encoder.state_dict().items()}
        optimizer = torch.optim.AdamW(models.phase_parameters(model,models.SAE,phase),lr=.001)
        optimizer.zero_grad()
        objective = models.phase_loss(model,models.SAE,phase,raw[:3],y[:3],state)
        objective.backward();optimizer.step()
        if phase=='ae2':
            assert all(torch.equal(v,model.ae1.encoder.state_dict()[k]) for k,v in old1.items())
        models.transition_phase(model,models.SAE,phase)
        if phase=='ae2':
            for encoder,trained in ((model.predictor.encoder1,model.ae1.encoder),(model.predictor.encoder2,model.ae2.encoder)):
                assert all(torch.equal(v,trained.state_dict()[k]) for k,v in encoder.state_dict().items())
    assert model.completed_phases.all()
    assert model.architecture_metadata['pretraining_executed']
    reloaded = models.build_model(models.SAE,SEED,state)
    reloaded.load_state_dict(model.state_dict())
    assert reloaded.architecture_metadata['pretraining_executed']
    assert not any(p.requires_grad for p in reloaded.ae1.encoder.parameters())


def _check_deterministic_single_trajectory_exact_rule_and_checkpoint(name,fixture):
    raw,y,state = fixture
    # ARMA(2,1) fixture fixed in advance; it contains no market measurements.
    rng = np.random.default_rng(271828)
    sigma = np.empty(170);innovation = rng.normal(0,.0008,len(sigma));sigma[:2]=.03
    for i in range(2,len(sigma)):
        sigma[i] = .03+.45*(sigma[i-1]-.03)-.2*(sigma[i-2]-.03)+innovation[i]+.25*innovation[i-1]
    state = models.fit_preprocessor(raw[:100],sigma[:100])
    model = models.build_model(name,SEED,state)
    fit = {'raw':raw[:100],'y':sigma[:100],'history_end_positions':np.arange(7,107),
           'dates':np.arange(100).astype(str)}
    models.fit_deterministic(model,name,fit,state,budget_seconds=30.)
    record = deepcopy(model.record)
    models.fit_deterministic(model,name,fit,state,budget_seconds=30.)
    assert model.record==record, 'A deterministic seed must never trigger another fit'
    history = np.stack([sigma[i-7:i] for i in range(100,104)])
    partition = {'raw':raw[100:104], 'known_sigma_history':history,
                 'observed_sigma_since_fit':np.r_[np.full(7,.029),sigma[:103]],
                 'history_end_positions':np.arange(107,111)}
    point,projected,floor = models.predict(model,name,partition,state)
    if name=='Persistence':
        np.testing.assert_array_equal(point,history[:,-1])
    elif name=='SeasonalNaive':
        np.testing.assert_array_equal(point,history[:,-7])
    elif name=='floor_only':
        np.testing.assert_array_equal(point,floor)
    elif name=='floor_plus_mean_excess':
        np.testing.assert_array_equal(point,floor+state['mean_excess'])
    else:
        assert model.record['real_data_fits']==1
        assert model.record['order']==[2,0,1]
        assert model.result.model.order==(2,0,1)
        expected=[];running=model.result
        for i in range(4):
            expected.append(float(running.forecast(1)[0])*model.record['scale'])
            if i<3:
                running=running.extend([sigma[100+i]/model.record['scale']])
        np.testing.assert_array_equal(point,np.asarray(expected))
        first={'raw':raw[100:101],'forecast_offsets':np.array([0]),
               'observed_sigma_since_fit':np.array([float('nan'),1e9])}
        # Future contents are not read for offset0.
        np.testing.assert_array_equal(models.predict(model,name,first,state)[0],point[:1])
    assert np.all(projected>=floor)
    other = models.build_model(name,1,state)
    other.load_state_dict(model.state_dict())
    np.testing.assert_array_equal(models.predict(other,name,partition,state)[0],point)
    assert list(model.parameters())==[] and models.training_phases(name)==()


def _check_fit_only_scaling_and_causal_predictions_ignore_unrequested_rows(fixture):
    raw,y,state = fixture
    saved = json.loads(json.dumps(state,allow_nan=False))
    changed = raw[100:].copy();changed[:,:,4]*=1e5
    models.transform(changed,state,'input_only_mse')
    assert state==saved
    for name in ('input_only_mse','global_mse','TimesNet','TimeMixer','PatchTST','iTransformer'):
        model = models.build_model(name,SEED,state)
        a = models.predict(model,name,raw[100:103],state,batch_size=3)[0]
        b = models.predict(model,name,raw[100:103],state,batch_size=1)[0]
        np.testing.assert_allclose(a,b,rtol=2e-6,atol=1e-10)
    with _assertion.assertRaisesRegex(ValueError,'target-history'):
        models.predict(models.build_model('Persistence',SEED,state),'Persistence',raw[:2],state)
    with _assertion.assertRaisesRegex(ValueError,'support'):
        models.fit_preprocessor(raw[:2],np.zeros(2))



class ModelContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = fixture()


def _install_cases():
    cases = [
        ('family',_check_complete_frozen_family_and_exact_optimization_rows,None,False),
        ('candidate_exact',_check_candidate_exact_frozen_preprocessing_architecture_and_mse,None,True),
        ('sae_three_phases',_check_sae_actual_three_stage_graph_and_exact_encoder_transfer,None,True),
        ('causal_preprocessing',_check_fit_only_scaling_and_causal_predictions_ignore_unrequested_rows,None,True),
    ]
    cases += [('neural_'+str(i),_check_real_forward_gradient_and_selected_state_factory_roundtrip,name,True)
              for i,name in enumerate(models.NEURAL)]
    cases += [('deterministic_'+str(i),_check_deterministic_single_trajectory_exact_rule_and_checkpoint,name,True)
              for i,name in enumerate(models.DETERMINISTIC)]
    for suffix,function,name,uses_data in cases:
        def check(self,f=function,n=name,d=uses_data):
            args = ((n,) if n is not None else ()) + ((self.data,) if d else ())
            f(*args)
        check.__doc__ = function.__name__ + (' ['+name+']' if name else '')
        setattr(ModelContracts,'test_'+suffix,check)


_install_cases()

if __name__=='__main__':
    unittest.main()
