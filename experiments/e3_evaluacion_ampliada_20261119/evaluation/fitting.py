"""Common 14-neural-system fitting and frozen-scale predictions.

The existing E2 constructors and aditive E3 formulation are retained. Optimizer
policy and common physical-support projection belong to this new audit. SAE's
AE1, AE2 and supervised stages split one total phase opportunity budget.
"""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
_prior_path=list(sys.path)
for directory in (ROOT, ROOT/'scripts'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

def _module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    sys.modules[name]=result
    spec.loader.exec_module(result)
    return result

common=_module('e3_extended_common_pilot_helpers',ROOT/'experiments/e3_cierre_integral_20261118/campaign/pilot.py')
bounded=_module('e3_extended_bounded_phase',Path(__file__).with_name('phase.py'))
# Importing historical helpers prepends their own scripts path. Keep a caller's
# verified D02 package ahead of that fallback, so its policy JSON remains local.
sys.path[:]=list(dict.fromkeys([*_prior_path,str(ROOT),str(ROOT/'scripts')]))
NEURAL=common.NEURAL
LEGACY=common.LEGACY
SAE='LSTM-autoencoder con predictor'
MODELS=(*NEURAL,LEGACY)


def _factory(name,seed,data,device,rows,roots):
    if name not in MODELS:
        raise ValueError('Unknown frozen neural system: '+str(name))
    common._seed(seed)
    if name==LEGACY:
        from src.e3_innovation_model import InnovationModel
        return InnovationModel('additive_positive_head',u_scale=data['scaler']['u_scale'],
                               y_scale=data['scaler']['y_scale'],seed=seed).to(device)
    if rows is None or name not in rows or 'trial' not in rows[name]:
        raise ValueError('Exact immutable recipe row required for '+name)
    if name==SAE:
        from e2_published_sae import build_sae
        return build_sae(dropout=rows[name]['trial']['dropout'],seed=seed).to(device)
    from e2_reduced_fit import _builder
    return _builder(name,rows[name]['trial'],seed,roots).to(device)


def _values(model,name,x,A,scale):
    return model(x,A) if name==LEGACY else model(x).double()*scale


def transform(raw_windows,scaler):
    raw=np.asarray(raw_windows,dtype=np.float64)
    if (raw.ndim!=3 or raw.shape[1:]!=(7,5) or len(raw)==0
            or not np.isfinite(raw).all() or (raw[:,:,:4]<=0).any() or (raw[:,:,4]<0).any()
            or (raw[:,:,1]<np.maximum(raw[:,:,0],raw[:,:,3])).any()
            or (raw[:,:,2]>np.minimum(raw[:,:,0],raw[:,:,3])).any()
            or (raw[:,:,1]<raw[:,:,2]).any()):
        raise ValueError('Finite coherent N×7×5 positive-price OHLCV windows required')
    mean=np.asarray(scaler['mean'],dtype=np.float64)
    std=np.asarray(scaler['std'],dtype=np.float64)
    if mean.shape!=(5,) or std.shape!=(5,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any():
        raise ValueError('Frozen five-feature scaler required')
    relative=np.empty_like(raw)
    relative[:,:,:4]=np.log(raw[:,:,:4])-np.log(raw[:,-1,3])[:,None,None]
    relative[:,:,4]=np.log1p(raw[:,:,4])
    x=((relative-mean)/std).astype(np.float32)
    known=np.diff(np.log(raw[:,:,3]),axis=1)
    A=((known-known.mean(axis=1)[:,None])**2).sum(axis=1)/7
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite transformed input')
    return x,A


def predict(model,name,raw_windows,scaler,device,batch_size=64):
    """Return raw σ7, common projected σ7, and observed σ7 floors."""
    import torch
    if name not in MODELS or type(batch_size)is not int or batch_size<1:
        raise ValueError('Frozen model and positive batch size required')
    scale=float(scaler['y_scale'])
    if not np.isfinite(scale) or scale<=0:
        raise ValueError('Positive frozen training target scale required')
    x,A=transform(raw_windows,scaler)
    outputs=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(x),batch_size):
            xx=torch.as_tensor(x[start:start+batch_size],device=device)
            aa=torch.as_tensor(A[start:start+batch_size],device=device)
            out=_values(model,name,xx,aa,scale).detach().cpu().numpy()
            if out.shape!=(len(xx),):
                raise ValueError('Frozen predictor output shape changed')
            outputs.append(out)
    raw=np.concatenate(outputs).astype(np.float64)
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite prediction; no zero fallback or row deletion')
    floors=np.sqrt(A)
    return raw,np.maximum(raw,floors),floors


def fit(name,seed,data,directory,config,device,rows=None,roots=None,source_identity=None):
    """Fit one configuration/seed; training is raw MSE, selection projected MSE."""
    import torch
    if type(seed)is not int or seed<0:
        raise ValueError('Explicit nonnegative seed required')
    directory=Path(directory).absolute()
    if any(p.is_symlink() for p in (directory,*directory.parents)):
        raise ValueError('Fitting path cannot traverse symlinks')
    directory.mkdir(parents=True,exist_ok=True)
    source_identity=dict(source_identity or {})
    if not source_identity:
        raise ValueError('Explicit source/protocol/data identity required')
    if 'per_fit_seconds' not in config:
        raise ValueError('One explicit whole-system phase opportunity budget required')
    phases=('ae1','ae2','supervised') if name==SAE else ('supervised',)
    allocations=bounded.allocate_phase_budgets(config['per_fit_seconds'],phases)
    request={'name':name,'seed':seed,'source_identity':source_identity,'config':config,
             'phase_allocations_seconds':allocations,'scaler':data['scaler'],
             'train_X_sha256':data['admission']['arrays_sha256']['train_X'],
             'training_loss':'raw sigma7 MSE normalized by training target mean',
             'validation_selection':'common projected sigma7 MSE normalized by training target mean'}
    request_sha=common.digest(request)
    if (directory/'fit_request.json').exists():
        if json.loads((directory/'fit_request.json').read_text())!=request:
            raise ValueError('Fit request changed; old checkpoint bytes preserved')
    else:
        bounded.write_json(directory/'fit_request.json',request)
    started=time.perf_counter()
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    tensors={key:torch.as_tensor(data[key],device=device) for key in (
        'train_X','val_X','train_A','val_A','train_y_original','val_y_original')}
    x,vx=tensors['train_X'],tensors['val_X']
    scale=float(data['scaler']['y_scale'])
    outcomes={}
    if name==SAE:
        common._seed(seed)
        from e2_recipe_models import SAEPretraining,repeat7,last_state
        from e2_published_sae import _annotate,_verify_source_and_dropout
        if rows is None or name not in rows:
            raise ValueError('Exact SAE source row required')
        trial=rows[name]['trial']
        _verify_source_and_dropout(trial['dropout'])
        manager=SAEPretraining(x,data['admission']['arrays_sha256']['train_X'],accept_auxiliary_scale=True)
        ax,ay=manager.phase1_data()
        val_target=manager.reconstruction_target(vx)
        outcomes['ae1']=bounded.phase(manager.ae1,ax,ay,vx,val_target,directory/'ae1',config,
            identity={**source_identity,'model':name,'seed':seed,'phase':'ae1','request_sha256':request_sha},
            budget_seconds=allocations['ae1'])
        x2,y2=manager.prepare_phase2()
        with torch.no_grad():
            v2=repeat7(last_state(manager.ae1.encoder,vx)).detach()
        outcomes['ae2']=bounded.phase(manager.ae2,x2,y2,v2,val_target,directory/'ae2',config,
            identity={**source_identity,'model':name,'seed':seed,'phase':'ae2','request_sha256':request_sha},
            budget_seconds=allocations['ae2'])
        common._seed(seed)
        model=_annotate(manager.transfer_predictor(),dropout=trial['dropout'],seed=seed).to(device)
    else:
        model=_factory(name,seed,data,device,rows,roots)
    def loss(m,indexes):
        if indexes is None:
            out=_values(m,name,vx,tensors['val_A'],scale)
            target=tensors['val_y_original']
            projected=torch.maximum(out.double(),torch.sqrt(tensors['val_A']))
            if projected.shape!=target.shape:
                raise ValueError('Validation point shape changed')
            return torch.mean(((projected-target)/scale)**2)
        out=_values(m,name,x[indexes],tensors['train_A'][indexes],scale)
        target=tensors['train_y_original'][indexes]
        if out.shape!=target.shape:
            raise ValueError('Training point shape changed')
        return torch.mean(((out-target)/scale)**2)
    outcomes['supervised']=bounded.phase(model,x,tensors['train_y_original']/scale,
        vx,tensors['val_y_original']/scale,directory/'supervised',config,
        identity={**source_identity,'model':name,'seed':seed,'phase':'supervised','request_sha256':request_sha},
        loss_fn=loss,budget_seconds=allocations['supervised'])
    configuration=(model.configuration() if hasattr(model,'configuration')
                   else copy.deepcopy(getattr(model,'architecture_metadata',{'model':name})))
    if name==SAE:
        configuration.update(pretraining_executed=True,pretraining_execution_scope='AE1 and AE2 in this fit')
    point,projected,floor=predict(model,name,data['val_raw_X'],data['scaler'],device)
    metadata={'status':'completed','model':name,'seed':seed,'fit_request_sha256':request_sha,
        'source_identity':source_identity,'configuration':configuration,
        'scaler_sha256':common.digest(data['scaler']),
        'parameters':sum(p.numel() for p in model.parameters()),
        'phase_budget_seconds':allocations,'total_allocated_phase_seconds':sum(allocations.values()),
        'whole_system_budget_seconds':float(config['per_fit_seconds']),
        'phase_resources':{key:value['resource_receipt'] for key,value in outcomes.items()},
        'optimizer_steps':sum(value['steps'] for value in outcomes.values()),
        'training_epoch_seconds':sum(value['measured_completed_epoch_seconds'] for value in outcomes.values()),
        'measured_fit_wall_seconds_this_segment':time.perf_counter()-started,
        'batch_boundary_deadline_overrun_seconds':sum(value['resource_receipt']['deadline_overrun_seconds'] for value in outcomes.values()),
        'hard_process_timeout_enforced':False,
        'SAE_pretraining_executed':name==SAE,'legacy_initial_bias_calibration_executed':False,
        'training_loss':'raw_sigma7_MSE','validation_selection':'common_projected_sigma7_MSE',
        'validation_projected_RMSE':float(np.sqrt(np.mean((projected-data['val_y_original'])**2))),
        'validation_raw_RMSE':float(np.sqrt(np.mean((point-data['val_y_original'])**2))),
        'GPU':torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
        'peak_GPU_allocated_bytes':int(torch.cuda.max_memory_allocated(device)) if str(device).startswith('cuda') else None}
    metadata['GPU_memory_scope']='Process allocated peak after reset; includes any caller-resident models/tensors'
    bounded._torch_save(directory/'selected_model.pt',{'model':name,'seed':seed,
        'state_dict':bounded._clone(model),'configuration':configuration,'scaler':data['scaler'],
        'fit_request_sha256':request_sha,'source_identity':source_identity})
    metadata['selected_model_sha256']=common.sha(directory/'selected_model.pt')
    bounded.write_json(directory/'fit_metadata.json',metadata)
    return model,metadata


def load_selected(name,path,data,device,rows=None,roots=None):
    """Restore trained selected weights without repeating any SAE reconstruction."""
    import torch
    path=Path(path)
    if path.is_symlink() or path.with_name('fit_metadata.json').is_symlink():
        raise ValueError('Selected model and metadata must be regular files')
    metadata=json.loads(path.with_name('fit_metadata.json').read_text())
    if metadata['model']!=name or metadata['selected_model_sha256']!=common.sha(path):
        raise ValueError('Selected model identity or bytes changed')
    record=torch.load(path,map_location='cpu',weights_only=False)
    if (record['model']!=name or record['fit_request_sha256']!=metadata['fit_request_sha256']
            or common.digest(record['scaler'])!=common.digest(data['scaler'])):
        raise ValueError('Selected model source/scaler binding differs')
    model=_factory(name,int(record['seed']),data,device,rows,roots)
    model.load_state_dict(record['state_dict'],strict=True)
    model.eval()
    return model,metadata
