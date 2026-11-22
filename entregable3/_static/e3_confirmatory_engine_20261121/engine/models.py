"""The frozen 21-system family: original constructors and causal task adapters.

Nothing trains, downloads observations, or opens a market split on import/build.
Outputs are one original-unit sigma7 per input window. Training/checkpoint
orchestration lives in training.py; this module owns model math and identities.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import importlib
import json
import math
from pathlib import Path
import pickle
import sys
import time
from threading import RLock
import warnings

import numpy as np
import torch
from torch import nn

from vendor.candidate import candidate
from vendor.candidate.vendor.seoi_v5 import data as candidate_data
from vendor.candidate.vendor.seoi_v5.model import ScaleOverlapModel
from vendor.historical.scripts import e2_recipe_models as recipes
from vendor.historical.scripts.e2_final_models_mlp import TSMixerAdapter
from vendor.historical.scripts.e2_final_models_attention import AttentionVolatilityAdapter
from vendor.historical.scripts.e2_published_periodic import build_periodic
from vendor.historical.scripts.e2_published_patchtst import build_patchtst
from vendor.historical.scripts.e2_published_tide import build_tide
from vendor.historical.scripts.dl_models import SmallMLP, SingleLayerLSTM

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'vendor'
HISTORICAL = VENDOR / 'historical'
# This exact historical module contains one absolute sibling import. Register
# the already loaded sibling temporarily while importing; source bytes stay exact.
_dl_key = sys.modules.get('dl_models')
sys.modules['dl_models'] = sys.modules['vendor.historical.scripts.dl_models']
try:
    from vendor.historical.scripts.dl_models_gru_lstm_tcn import SingleLayerGRU
finally:
    if _dl_key is None:
        sys.modules.pop('dl_models', None)
    else:
        sys.modules['dl_models'] = _dl_key

_RECIPES = json.loads(Path(__file__).with_name('recipes.json').read_text())
METHODS = tuple(_RECIPES['method_order'])
DETERMINISTIC = ('Persistence', 'SeasonalNaive', 'ARIMA', 'floor_only', 'floor_plus_mean_excess')
NEURAL = tuple(n for n in METHODS if n not in DETERMINISTIC)
SAE = 'LSTM-autoencoder con predictor'
CANDIDATE = 'input_only_mse'
CANDIDATE_IDENTITY = candidate.IDENTITY
ALIASES = {CANDIDATE_IDENTITY: CANDIDATE, 'additive_mse': 'additive_positive_head',
           'floor_plus_fit_mean': 'floor_plus_mean_excess'}
_IMPORT_LOCK = RLock()
FIXED_COUNTS = {'FCN-1D':268929, 'ResNet-1D':522955, 'Inception-1D':490561,
                'TSMixer':35044, 'iTransformer':795403, 'MLP16':593,
                'LSTM32':5025, 'GRU32':3777, 'additive_positive_head':1210}


def canonical_name(name):
    name = ALIASES.get(name, name)
    if name not in METHODS:
        raise ValueError('Unknown frozen confirmatory method: ' + str(name))
    return name


def is_deterministic(name):
    return canonical_name(name) in DETERMINISTIC


def model_config(name):
    """Frozen per-system settings; no common learning-rate substitution.

    E2 optimizer/dropout rows come from deployment_master.csv. Historical fixed
    phase epochs are preserved as provenance, not silently used as a fresh
    selection result. E2's registered cap/patience are 200/20; the candidate's
    original cap/patience are 40/8. A whole-system wall budget is worker-owned.
    """
    name = canonical_name(name)
    row = deepcopy(_RECIPES['models'][name])
    if name in DETERMINISTIC:
        return {'name':name, 'deterministic':True, 'trial':row['trial'],
                'phases':[], 'seed_replications':1, 'source_recipe':row}
    own = name in (CANDIDATE, 'global_mse', 'additive_positive_head')
    config = {'name':name, 'optimizer':'AdamW', 'max_epochs':40 if own else 200,
              'patience':8 if own else 20, 'gradient_clip_norm':1.,
              'warmup_fraction':0., 'phases':list(training_phases(name)),
              'training_loss':'raw_sigma7_MSE_normalized_by_fit_target_mean',
              'selection':'minimum_projected_selection_RMSE_earliest_tie',
              'deterministic':False, 'source_recipe':row}
    config.update(row['trial'])
    return config


def training_phases(name):
    name = canonical_name(name)
    if name in DETERMINISTIC:
        return ()
    return ('ae1','ae2','supervised') if name == SAE else ('supervised',)


def verify_sources():
    """Verify all copied implementation bytes and original author licenses."""
    manifest = json.loads((VENDOR/'SOURCE_MANIFEST.json').read_text())
    for row in manifest['files']:
        path = ROOT/row['path']
        if path.is_symlink() or not path.is_file() or sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Frozen model source changed: '+row['path'])
    return {'version':manifest['version'], 'files_verified':len(manifest['files']),
            'manifest_sha256':sha256((VENDOR/'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),
            'recipes_sha256':sha256(Path(__file__).with_name('recipes.json').read_bytes()).hexdigest()}


def _raw(raw):
    if isinstance(raw, dict):
        raw = raw.get('raw', raw.get('windows'))
    if isinstance(raw, torch.Tensor):
        raw = raw.detach().cpu().numpy()
    return candidate_data.validate_raw(raw)


def _relative(raw):
    relative = np.empty_like(raw)
    relative[:,:,:4] = np.log(raw[:,:,:4])-np.log(raw[:,-1,3])[:,None,None]
    relative[:,:,4] = np.log1p(raw[:,:,4])
    return relative


def fit_preprocessor(raw_fit, y_fit):
    """Return JSON-only state fitted exclusively on caller-admitted fit rows."""
    raw_fit = _raw(raw_fit)
    fitted = candidate.fit_inputs(raw_fit, y_fit)
    y = np.asarray(y_fit, dtype=np.float64)
    if len(y)<2:
        raise ValueError('At least two explicit fit rows required')
    relative = _relative(raw_fit)
    mean, std = relative.mean((0,1)), relative.std((0,1), ddof=0)
    std = np.where(std==0,1.,std)
    X = ((relative-mean)/std).astype(np.float32)
    A = candidate_data.known_geometry(raw_fit)[0]
    state = {'version':'e3-family-fit-preprocessor-v1', 'candidate':fitted,
             'mean':mean.tolist(), 'std':std.tolist(),
             'y_scale':fitted['y_scale'], 'u_scale':fitted['u_scale'],
             'mean_excess':float(np.mean(y-np.sqrt(A))),
             'sae_auxiliary_scale':max(1.,float(np.abs(X).max())),
             'fit_rows':len(y), 'fit_y':y.tolist(),
             'fit_only_statistics':True, 'fit_label_history_clock':'admitted_observation_order',
             'array_hashes':{'raw':sha256(np.ascontiguousarray(raw_fit).tobytes()).hexdigest(),
                             'y':sha256(np.ascontiguousarray(y).tobytes()).hexdigest()}}
    if state['mean_excess']<0:
        raise ValueError('Negative mean fit excess violates physical support')
    json.dumps(state, allow_nan=False)
    return state


def _validate_state(state):
    if not isinstance(state,dict) or state.get('version')!='e3-family-fit-preprocessor-v1':
        raise ValueError('A bound fit-only family preprocessor is required')
    for key in ('u_scale','y_scale','sae_auxiliary_scale'):
        if not np.isfinite(state[key]) or state[key]<=0:
            raise ValueError('Invalid positive fit statistic: '+key)


def transform(raw, fit_state, name):
    name = canonical_name(name)
    _validate_state(fit_state)
    raw = _raw(raw)
    if name in (CANDIDATE,'global_mse'):
        recipe = 'input_only_mse' if name==CANDIDATE else 'global_mse'
        X,A,scale,_ = candidate_data.transform(raw,fit_state['candidate']['preprocessor'],recipe)
    else:
        X = ((_relative(raw)-np.asarray(fit_state['mean']))/np.asarray(fit_state['std'])).astype(np.float32)
        A = candidate_data.known_geometry(raw)[0]
        scale = np.ones(len(raw),dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError('Nonfinite transformed features')
    return X,A,scale


def _innovation_class():
    # Historical absolute `src.*` imports are isolated to this import, then the
    # caller's module namespace is restored. Objects keep the exact globals.
    with _IMPORT_LOCK:
        keys = [k for k in sys.modules if k=='src' or k.startswith('src.')]
        saved = {k:sys.modules.pop(k) for k in keys}
        sys.path.insert(0,str(HISTORICAL))
        try:
            return importlib.import_module('src.e3_innovation_model').InnovationModel
        finally:
            sys.path.remove(str(HISTORICAL))
            for k in list(sys.modules):
                if k=='src' or k.startswith('src.'):
                    del sys.modules[k]
            sys.modules.update(saved)


class StagedSAE(nn.Module):
    """Checkpointable execution of the original AE1 -> AE2 -> predictor graph.

    Uses the exact historical ReconstructionAE/SAERecipe constructors and
    repeat7/last_state operations. Auxiliary scaling is frozen on fit rows.
    Decoder modules remain in the training checkpoint; forward only invokes
    the transferred predictor. Epoch selection and one shared budget are owned
    by the worker, with a fresh optimizer per phase.
    """
    def __init__(self, seed, dropout, auxiliary_scale):
        super().__init__()
        self.seed, self.dropout = seed,float(dropout)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.ae1 = recipes.ReconstructionAE(5,25)
            self.ae2 = recipes.ReconstructionAE(25,26)
            torch.manual_seed(seed)
            self.predictor = recipes.SAERecipe()
        self.predictor.dropout1.p = self.predictor.dropout2.p = self.dropout
        self.register_buffer('auxiliary_scale',torch.tensor(auxiliary_scale,dtype=torch.float32))
        self.register_buffer('completed_phases',torch.zeros(3,dtype=torch.bool))
        self.architecture_metadata = {'model':SAE,'recipe_id':'SAE-D7-PUBLISHED-v1',
            'published_hidden_units':[25,26], 'input_shape':[7,5],
            'pretraining_required':True,'pretraining_executed':False,
            'phase_sequence':['ae1','ae2','supervised'],
            'source_body_sha256':sha256(Path(recipes.__file__).read_bytes()).hexdigest(),
            'dropout':self.dropout, 'initialization_seed':seed,
            'auxiliary_target':'original fit-scaled X / fit-only max(1,max(abs(X_fit)))',
            'phase_budget':'single whole-system allocation split across three phases',
            'decoders_used_for_prediction':False}

    def forward(self,x):
        return self.predictor(x)

    def load_state_dict(self,state_dict,strict=True,assign=False):
        result = super().load_state_dict(state_dict,strict=strict,assign=assign)
        for p in self.ae1.encoder.parameters():
            p.requires_grad_(not bool(self.completed_phases[0]))
        self.architecture_metadata['pretraining_executed'] = bool(self.completed_phases[:2].all())
        return result


class DeterministicModel:
    """Single trajectory, serializable actual ARIMA state or analytic rule."""
    def __init__(self,name):
        self.name = name
        self.record = {'name':name,'fitted':False,'seed_replications':1}
        self.result = None

    def parameters(self):
        return iter(())

    def train(self,mode=True):
        return self

    def eval(self):
        return self

    def to(self,*args,**kwargs):
        return self

    def state_dict(self):
        return {'record':deepcopy(self.record),
                'statsmodels_result_pickle':pickle.dumps(self.result,protocol=5) if self.result is not None else None}

    def load_state_dict(self,state,strict=True):
        if state['record']['name']!=self.name or set(state)!={'record','statsmodels_result_pickle'}:
            raise ValueError('Deterministic checkpoint method/state differs')
        self.record = deepcopy(state['record'])
        self.result = pickle.loads(state['statsmodels_result_pickle']) if state['statsmodels_result_pickle'] is not None else None
        return self


def build_model(name, seed, fit_state):
    """Construct the exact named body with fresh full-63-bit Torch seed."""
    name = canonical_name(name)
    _validate_state(fit_state)
    if type(seed) is not int or not 0<=seed<2**63:
        raise ValueError('A nonnegative full 63-bit seed-bank ID is required')
    verify_sources()
    if name in DETERMINISTIC:
        return DeterministicModel(name)
    trial = _RECIPES['models'][name]['trial']
    p = float(trial.get('dropout',0.))
    roots = {n:VENDOR/'author_sources'/n for n in ('TimesNet','TimeMixer','PatchTST','iTransformer','TiDE')}
    with _IMPORT_LOCK, torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if name==CANDIDATE:
            model = candidate.make_model(seed,fit_state['candidate'])
        elif name=='global_mse':
            model = ScaleOverlapModel('global_mse',seed,fit_state['u_scale'],fit_state['y_scale'])
        elif name=='additive_positive_head':
            model = _innovation_class()(name,u_scale=fit_state['u_scale'],y_scale=fit_state['y_scale'],seed=seed,
                rule_path=HISTORICAL/'experiments/e3_ficha_cnn_lstm_atencion_v2_20261116/verification/quadrature_k128.json')
        elif name in ('FCN-1D','ResNet-1D','Inception-1D'):
            model = recipes.build_model(name)
        elif name==SAE:
            model = StagedSAE(seed,p,fit_state['sae_auxiliary_scale'])
        elif name=='TSMixer':
            model = TSMixerAdapter(p)
        elif name in ('TimesNet','TimeMixer'):
            model = build_periodic(name,p,roots)
        elif name=='PatchTST':
            model = build_patchtst(dropout=p,source_roots=roots)
        elif name=='TiDE':
            model = build_tide(seed=seed,source_roots=roots)
        elif name=='iTransformer':
            model = AttentionVolatilityAdapter(name,{'parameters':{'dropout':min(p,.3)}},roots)
            for module in model.modules():
                if isinstance(module,nn.Dropout):
                    module.p = p
        elif name=='MLP16':
            model = SmallMLP(7,16,dropout=p,input_features=5)
        elif name=='LSTM32':
            model = SingleLayerLSTM(7,32,dropout=p,input_features=5)
        elif name=='GRU32':
            model = SingleLayerGRU(7,32,dropout=p,input_features=5)
        else:
            raise ValueError('Unimplemented method escaped family validation: '+name)
    count = sum(p.numel() for p in model.parameters())
    if name in FIXED_COUNTS and count!=FIXED_COUNTS[name]:
        raise RuntimeError('Frozen parameter count changed for '+name+': '+str(count))
    metadata = deepcopy(getattr(model,'architecture_metadata',{}))
    metadata.update(confirmatory_method=name,initialization_seed=seed,
        actual_parameter_count=count,selected_recipe=deepcopy(trial),
        historical_hpo_opportunities=deepcopy(_RECIPES['models'][name]),
        parameters_freshly_initialized=True,market_fit_performed=False)
    model.architecture_metadata = metadata
    return model


def _tensors(model,name,raw,state):
    X,A,scale = transform(raw,state,name)
    device = next(model.parameters()).device
    return (torch.as_tensor(X,device=device),torch.as_tensor(A,dtype=torch.float64,device=device),
            torch.as_tensor(scale,dtype=torch.float64,device=device))


def _values(model,name,X,A,scale,state):
    if name in (CANDIDATE,'global_mse'):
        return model(X,A,scale)
    if name=='additive_positive_head':
        return model(X,A)
    return model(X).double()*state['y_scale']


def loss(model,name,raw_batch,y,fit_state):
    name = canonical_name(name)
    if name in DETERMINISTIC:
        raise ValueError('A deterministic trajectory has no optimizer loss')
    X,A,scale = _tensors(model,name,raw_batch,fit_state)
    prediction = _values(model,name,X,A,scale,fit_state)
    target = torch.as_tensor(y,dtype=torch.float64,device=prediction.device)
    if (target.shape!=prediction.shape or not torch.isfinite(target).all() or (target<0).any()
            or (target.square()-A < -1e-12).any()):
        raise ValueError('Aligned finite nonnegative labels required')
    return ((prediction-target)/fit_state['y_scale']).square().mean()


def phase_model(model,name,phase):
    name = canonical_name(name)
    if phase not in training_phases(name):
        raise ValueError('Unknown phase for this method')
    return {'ae1':model.ae1,'ae2':model.ae2,'supervised':model.predictor}[phase] if name==SAE else model


def phase_parameters(model,name,phase):
    return (p for p in phase_model(model,name,phase).parameters() if p.requires_grad)


def phase_loss(model,name,phase,raw_batch,y,fit_state):
    name = canonical_name(name)
    if phase not in training_phases(name):
        raise ValueError('Unknown phase for this method')
    if phase=='supervised':
        return loss(model,name,raw_batch,y,fit_state)
    X,_,_ = _tensors(model,name,raw_batch,fit_state)
    target = X/model.auxiliary_scale
    if phase=='ae1':
        prediction = model.ae1(X)
    else:
        if not bool(model.completed_phases[0]):
            raise RuntimeError('AE2 requires the selected completed AE1')
        with torch.no_grad():
            encoded = recipes.repeat7(recipes.last_state(model.ae1.encoder,X)).detach()
        prediction = model.ae2(encoded)
    return (prediction-target).square().mean()


def transition_phase(model,name,completed_phase):
    """Call after restoring that phase's selected weights, before next phase."""
    name = canonical_name(name)
    if completed_phase not in training_phases(name):
        raise ValueError('Unknown completed phase')
    if name!=SAE:
        return model
    index = training_phases(name).index(completed_phase)
    if index and not bool(model.completed_phases[index-1]):
        raise RuntimeError('SAE phases must complete in their registered order')
    if completed_phase=='ae1':
        for p in model.ae1.encoder.parameters():
            p.requires_grad_(False)
    elif completed_phase=='ae2':
        model.predictor.encoder1.load_state_dict(model.ae1.encoder.state_dict(),strict=True)
        model.predictor.encoder2.load_state_dict(model.ae2.encoder.state_dict(),strict=True)
        for p in model.predictor.parameters():
            p.requires_grad_(True)
    model.completed_phases[index] = True
    model.architecture_metadata['pretraining_executed'] = bool(model.completed_phases[:2].all())
    return model


def fit_deterministic(model,name,fit_data,fit_state,*,budget_seconds=120.):
    """One genuine fit for ARIMA; analytic rules have no repeated seed fits.

    Matches the pinned historical statsmodels invocation. The selected CSV
    order (2,0,1) is supplied explicitly, replacing the helper's default only.
    There is no fallback forecast when convergence or the budget check fails.
    """
    name = canonical_name(name)
    if name not in DETERMINISTIC or model.name!=name:
        raise ValueError('A matching deterministic method is required')
    _validate_state(fit_state)
    if model.record['fitted']:
        return model
    if not isinstance(budget_seconds,(int,float)) or not np.isfinite(budget_seconds) or budget_seconds<=0:
        raise ValueError('Positive whole-system fitting budget required')
    if name!='ARIMA':
        model.record.update(fitted=True,real_data_fits=0,training_seconds=0.,parameters=0,
                            rule=deepcopy(_RECIPES['models'][name]['trial']))
        return model
    from importlib.metadata import version
    if version('statsmodels')!='0.14.5':
        raise RuntimeError('ARIMA requires statsmodels==0.14.5')
    from statsmodels.tsa.arima.model import ARIMA
    y = np.asarray(fit_state['fit_y'],dtype=np.float64)
    axis_metadata = {}
    if isinstance(fit_data,dict) and 'history_end_positions' in fit_data:
        positions = np.asarray(fit_data['history_end_positions'])
        if (positions.shape!=(len(y),) or positions.dtype.kind not in 'iu'
                or (positions<0).any() or (np.diff(positions)<=0).any()):
            raise ValueError('ARIMA fit history positions must be aligned and increasing')
        axis_metadata['fit_history_stop'] = int(positions[-1])+1
        if 'dates' in fit_data:
            axis_metadata['fit_end_date'] = str(np.asarray(fit_data['dates'])[-1])
    if isinstance(fit_data,dict) and 'fit_sigma_axis' in fit_data:
        # Daily-clock panels may retain admitted missing dates as NaN states.
        y = np.asarray(fit_data['fit_sigma_axis'],dtype=np.float64)
        if not np.array_equal(y[np.isfinite(y)],np.asarray(fit_state['fit_y'])):
            raise ValueError('ARIMA observation axis differs from admitted fit labels')
    if y.ndim!=1 or np.isfinite(y).sum()<90 or (y[np.isfinite(y)]<0).any():
        raise ValueError('ARIMA requires at least 90 genuine fit-only labels')
    scale = float(np.nanmean(y))
    order = tuple(_RECIPES['models']['ARIMA']['trial']['order'])
    started = time.perf_counter()
    def budget_check(_parameters):
        if time.perf_counter()-started>=budget_seconds:
            raise RuntimeError('ARIMA fitting budget expired at an optimizer iteration boundary')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        budget_check(None)
        fitted = ARIMA(y/scale,order=order,trend='c' if order[1]==0 else 'n',missing='none',
            enforce_stationarity=True,enforce_invertibility=True).fit(
                method_kwargs={'maxiter':200,'disp':False,'callback':budget_check})
    if not bool(fitted.mle_retvals.get('converged',False)) or not np.isfinite(fitted.params).all():
        raise RuntimeError('ARIMA did not converge; no replacement forecast')
    for part in (fitted,getattr(fitted,'_results',None)):
        settings = getattr(part,'mle_settings',None)
        if isinstance(settings,dict):
            settings.pop('callback',None)
    elapsed = time.perf_counter()-started
    model.result = fitted
    model.record.update(fitted=True,real_data_fits=1,scale=scale,order=list(order),
        statsmodels='0.14.5',training_seconds=elapsed,allocated_seconds=float(budget_seconds),
        deadline_overrun_seconds=max(0.,elapsed-budget_seconds),parameters=len(fitted.params),
        parameters_frozen_during_filtering=True,fit_labels=len(y),fit_observed_labels=int(np.isfinite(y).sum()),
        fit_values_sha256=sha256(np.ascontiguousarray(y).tobytes()).hexdigest(),
        warnings=[str(w.message) for w in caught],hard_process_timeout_enforced=False)
    model.record.update(axis_metadata)
    return model


def _deterministic_predictions(model,name,raw,state,A):
    if name=='floor_only':
        return np.sqrt(A)
    if name=='floor_plus_mean_excess':
        return np.sqrt(A)+state['mean_excess']
    if not isinstance(raw,dict):
        raise ValueError(name+' requires admitted causal target-history metadata')
    if name in ('Persistence','SeasonalNaive'):
        history = np.asarray(raw.get('known_sigma_history'),dtype=np.float64)
        lag = 1 if name=='Persistence' else 7
        if history.ndim!=2 or history.shape[0]!=len(A) or history.shape[1]<lag:
            raise ValueError('Aligned known_sigma_history oldest-to-newest with lag '+str(lag)+' required')
        values = history[:,-lag]
        if not np.isfinite(values).all() or (values<0).any():
            raise ValueError('Required prior observed sigma is unavailable; no substitution')
        return values.copy()
    if name!='ARIMA' or model.result is None or not model.record['fitted']:
        raise RuntimeError('A completed actual ARIMA fit is required')
    if 'arima_in_sample_positions' in raw:
        positions = np.asarray(raw['arima_in_sample_positions'])
        fitted = np.asarray(model.result.fittedvalues,dtype=np.float64)*model.record['scale']
        if positions.shape!=(len(A),) or positions.dtype.kind not in 'iu' or (positions<0).any() or (positions>=len(fitted)).any():
            raise ValueError('Invalid explicitly labeled conditional in-sample positions')
        return fitted[positions]
    observations = np.asarray(raw.get('observed_sigma_since_fit'),dtype=np.float64)
    if 'forecast_offsets' in raw:
        offsets = np.asarray(raw['forecast_offsets'])
    elif 'history_end_positions' in raw and 'fit_history_stop' in model.record:
        # Admission stores a shared global axis starting at the first genuine
        # rolling sigma (bar7). Its compatibility key is historical; subtract
        # the saved fit boundary to obtain the actual post-fit causal prefix.
        offsets = np.asarray(raw['history_end_positions'])-model.record['fit_history_stop']
        observations = observations[model.record['fit_history_stop']:]
    else:
        raise ValueError('ARIMA needs relative forecast_offsets or verified global history_end_positions')
    if (observations.ndim!=1 or offsets.shape!=(len(A),) or offsets.dtype.kind not in 'iu'
            or (offsets<0).any() or len(set(offsets.tolist()))!=len(offsets)
            or int(offsets.max())>len(observations)):
        raise ValueError('ARIMA requires full admitted causal post-fit sigma axis and distinct forecast_offsets')
    observations = observations[:int(offsets.max())]
    if (observations[np.isfinite(observations)]<0).any():
        raise ValueError('Negative observed sigma in causal ARIMA prefix')
    fitted = model.result
    by_offset = {}
    scale = model.record['scale']
    # Local replay never mutates the fitted checkpoint; interrupted evaluation
    # can reproduce precisely the same trajectory without a second fit.
    for k in range(int(offsets.max())+1):
        value = float(np.asarray(fitted.forecast(1))[0])*scale
        by_offset[k] = value
        if k<int(offsets.max()):
            fitted = fitted.extend(np.asarray([observations[k]/scale]))
    return np.asarray([by_offset[int(k)] for k in offsets],dtype=np.float64)


def predict(model,name,raw,fit_state,*,batch_size=64):
    """Return raw sigma7, common support-projected sigma7 and floor arrays."""
    name = canonical_name(name)
    if type(batch_size) is not int or batch_size<1:
        raise ValueError('A positive inference batch size is required')
    X,A,scale = transform(raw,fit_state,name)
    if name in DETERMINISTIC:
        point = _deterministic_predictions(model,name,raw,fit_state,A)
    else:
        device = next(model.parameters()).device
        outputs = []
        model.eval()
        with torch.no_grad():
            for start in range(0,len(X),batch_size):
                xx = torch.as_tensor(X[start:start+batch_size],device=device)
                aa = torch.as_tensor(A[start:start+batch_size],dtype=torch.float64,device=device)
                ss = torch.as_tensor(scale[start:start+batch_size],dtype=torch.float64,device=device)
                out = _values(model,name,xx,aa,ss,fit_state)
                if out.shape!=(len(xx),):
                    raise RuntimeError('Frozen predictor output shape changed')
                outputs.append(out.detach().cpu().numpy())
        point = np.concatenate(outputs).astype(np.float64)
    if point.shape!=(len(A),) or not np.isfinite(point).all():
        raise RuntimeError('Nonfinite or misaligned forecast; no fallback or dropped rows')
    floor = np.sqrt(A)
    return point,np.maximum(point,floor),floor
