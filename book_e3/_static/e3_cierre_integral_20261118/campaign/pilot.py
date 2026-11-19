"""Executable common BTC development pilot. No confirmation/test-array access.

This stage estimates feasibility and variance. Its validation is already exposed,
so neither model ranking here nor its early stopping constitutes confirmation.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
NEURAL = ('FCN-1D','ResNet-1D','Inception-1D','LSTM-autoencoder con predictor',
          'TimesNet','TSMixer','TiDE','TimeMixer','PatchTST','iTransformer','MLP16','LSTM32','GRU32')
CLASSICAL = ('Persistence','SeasonalNaive','ARIMA')
LEGACY = 'CNN-LSTM-atencion-aditiva E3'
CANDIDATE = 'OverlapDistribution-v3'
FAMILY = (*NEURAL, *CLASSICAL, LEGACY, CANDIDATE)
CONTROLS = (CANDIDATE, 'OverlapDistribution-v3 atom_mse_only',
            'OverlapDistribution-v3 fixed_probs', LEGACY, 'M00_projected',
            'floor_only', 'floor_plus_mean_excess')
DEFAULT_CONFIG = {'learning_rate': .001, 'weight_decay': 0., 'batch_size': 32,
                  'max_epochs': 40, 'patience': 8, 'gradient_clip': 1.,
                  'lambda_crps': .1, 'arima_order': [1,0,1], 'seasonal_period': 7}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',',':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''): h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp,path)


def _torch_save(path, value):
    import torch
    path=Path(path); temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as stream:
        torch.save(value,stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp,path)


def _seed(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def _rng():
    import torch
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    import torch
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']: torch.cuda.set_rng_state_all(state['cuda'])


def _clone(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def _phase_receipt(directory,state,path,request_sha256):
    write_json(Path(directory)/'receipt.json',{
        'status':'completed','best_epoch':state['best_epoch'],'epochs':state['epoch'],
        'optimizer_steps':state['steps'],'request_sha256':request_sha256,
        'checkpoint_sha256':sha(path),'epoch_boundary_resume':True,
        'partial_epoch_policy':'restore previous epoch RNG/model/optimizer and replay',
        'wall_time_is_total_compute':False,
        'measured_completed_epoch_seconds':state['measured_completed_epoch_seconds']})


def _phase(model, x, y, vx, vy, directory, config, *, identity, loss_fn=None):
    """Epoch-boundary optimizer/RNG checkpoint; interrupted epoch is replayed.

    No wall-budget claim is made: elapsed time excludes unclosed crash segments.
    Every restart writes a segment event so this limitation remains observable.
    """
    import torch
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    request={'identity':identity,'config':config,'train_shape':list(x.shape),
             'validation_shape':list(vx.shape)}
    req=digest(request); path=directory/'checkpoint.pt'
    marker=directory/'phase_started.json'
    started={'kind':'e3_optimizer_phase_started','request_sha256':req,
             'source_identity':identity.get('source_identity',identity),
             'phase_identity':identity,
             'counter_scope':'attempted phase; optimizer steps may still be zero'}
    if marker.exists():
        if canonical(json.loads(marker.read_text()))!=canonical(started):
            raise ValueError('Immutable phase-start identity differs')
    else:
        # Durable evidence exists even if optimizer setup or the first checkpoint
        # write fails. It is never overwritten by a retry or successful epoch.
        with marker.open('x') as stream:
            stream.write(json.dumps(started,indent=2,allow_nan=False)+'\n')
            stream.flush();os.fsync(stream.fileno())
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],
                                weight_decay=config['weight_decay'])
    state={'request_sha256':req,'epoch':0,'best_epoch':0,'best_score':None,
           'bad_epochs':0,'steps':0,'best':_clone(model),'latest':_clone(model),
           'optimizer':optimizer.state_dict(),'rng':_rng(),'curve':[],'completed':False,
           'measured_completed_epoch_seconds':0.}
    if path.exists():
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['request_sha256']!=req: raise ValueError('Phase resume identity differs')
        model.load_state_dict(state['latest']); optimizer.load_state_dict(state['optimizer'])
        for optstate in optimizer.state.values():
            for key,value in optstate.items():
                if torch.is_tensor(value): optstate[key]=value.to(x.device)
        _restore_rng(state['rng'])
    else:
        write_json(directory/'request.json',request); _torch_save(path,state)
    if state['completed']:
        model.load_state_dict(state['best'])
        _phase_receipt(directory,state,path,req)
        return state
    with (directory/'segments.log').open('a') as stream:
        stream.write(canonical({'event':'segment_start','resume_epoch':state['epoch'],
                                'unix_time':time.time(),'pid':os.getpid()})+'\n')
    for epoch in range(state['epoch']+1,config['max_epochs']+1):
        start=time.perf_counter(); model.train()
        order=torch.randperm(len(x),device=x.device)
        total=0.
        for offset in range(0,len(x),config['batch_size']):
            indexes=order[offset:offset+config['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            loss=(loss_fn(model,indexes) if loss_fn is not None
                  else torch.mean((model(x[indexes])-y[indexes])**2))
            if loss.ndim or not torch.isfinite(loss): raise ValueError('Nonfinite scalar loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'])
            optimizer.step(); total+=float(loss.detach())*len(indexes);state['steps']+=1
        model.eval()
        with torch.no_grad():
            score=(float(loss_fn(model,None)) if loss_fn is not None
                   else float(torch.mean((model(vx)-vy)**2)))
        if not np.isfinite(score): raise ValueError('Nonfinite validation loss')
        if x.device.type=='cuda': torch.cuda.synchronize(x.device)
        improved=state['best_score'] is None or score<state['best_score']
        if improved:
            state.update(best_score=score,best_epoch=epoch,best=_clone(model),bad_epochs=0)
        else: state['bad_epochs']+=1
        duration=time.perf_counter()-start
        state['curve'].append({'epoch':epoch,'train_loss':total/len(x),
                               'validation_mse_scaled':score,'seconds':duration})
        state.update(epoch=epoch,latest=_clone(model),optimizer=optimizer.state_dict(),rng=_rng())
        state['measured_completed_epoch_seconds']+=duration
        state['completed']=state['bad_epochs']>=config['patience'] or epoch==config['max_epochs']
        _torch_save(path,state)
        write_json(directory/'curve.json',state['curve'])
        if state['completed']:break
    model.load_state_dict(state['best'])
    _phase_receipt(directory,state,path,req)
    return state


def _preflight():
    path=ROOT/'experiments/e3_remediacion_20261118/evaluation/colab_preflight.py'
    spec=importlib.util.spec_from_file_location('e3_common_preflight',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def _package(directory,package_path):
    preflight=_preflight()
    if package_path:
        package=Path(package_path)
    else:
        # Cache vendor code outside evidence: upload only receipts/checkpoints.
        destination=Path(directory).parent/'e3-immutable-source-package-c4a4a1e68e98'
        if not destination.exists():package=preflight.acquire_package(destination)
        else:
            candidates=list((destination/'extracted').rglob('code/scripts/e2_standalone_code.py'))
            if len(candidates)!=1:raise ValueError('Incomplete immutable package; retain it and reconcile')
            package=candidates[0].parents[2]
    inventory,rows,roots=preflight.portable_inventory(package)
    return package,inventory,rows,roots


def _fit_neural(name, seed, data, directory, config, device, rows=None, roots=None,source_identity=None):
    import torch
    from src.e3_overlap_model_v3 import OverlapDistributionModel
    from src.e3_innovation_model import InnovationModel
    from src.e3_development import DevelopmentModel
    _seed(seed)
    tensors={key:torch.as_tensor(data[key],device=device) for key in (
        'train_X','train_A','train_U','train_y_original','val_X','val_A','val_U','val_y_original')}
    x,vx=tensors['train_X'],tensors['val_X'];scale=data['scaler']['y_scale']
    y=tensors['train_y_original']/scale;vy=tensors['val_y_original']/scale
    phases={}; custom=name not in NEURAL
    source_identity=source_identity or {'scope':'caller_supplied_identity_without_production_source_binding'}
    if name.startswith(CANDIDATE):
        variant={'OverlapDistribution-v3':'atom_crps',
                 'OverlapDistribution-v3 atom_mse_only':'atom_mse_only',
                 'OverlapDistribution-v3 fixed_probs':'fixed_probs'}[name]
        model=OverlapDistributionModel(variant,u_scale=data['scaler']['u_scale'],
            y_scale=scale,seed=seed,lambda_crps=config['lambda_crps']).to(device)
    elif name==LEGACY:
        model=InnovationModel('additive_positive_head',u_scale=data['scaler']['u_scale'],
                             y_scale=scale,seed=seed).to(device)
    elif name=='M00_projected':model=DevelopmentModel('M00',seed).to(device)
    elif name=='LSTM-autoencoder con predictor':
        from e2_recipe_models import SAEPretraining,repeat7,last_state
        from e2_published_sae import _annotate,_verify_source_and_dropout
        trial=rows[name]['trial']; _verify_source_and_dropout(trial['dropout'])
        manager=SAEPretraining(x, data['admission']['arrays_sha256']['train_X'],accept_auxiliary_scale=True)
        ax,ay=manager.phase1_data();av=manager.reconstruction_target(vx)
        phases['ae1']=_phase(manager.ae1.to(device),ax,ay,vx,av,Path(directory)/'ae1',config,
                             identity={'name':name,'seed':seed,'phase':'ae1','trial':trial,'source_identity':source_identity})
        x2,y2=manager.prepare_phase2()
        with torch.no_grad():v2=repeat7(last_state(manager.ae1.encoder,vx)).detach()
        phases['ae2']=_phase(manager.ae2.to(device),x2,y2,v2,av,Path(directory)/'ae2',config,
                             identity={'name':name,'seed':seed,'phase':'ae2','trial':trial,'source_identity':source_identity})
        _seed(seed)
        model=_annotate(manager.transfer_predictor(),dropout=trial['dropout'],seed=seed).to(device)
    else:
        from e2_reduced_fit import _builder
        model=_builder(name,rows[name]['trial'],seed,roots).to(device)
    model_config=(model.configuration() if hasattr(model,'configuration')
                  else getattr(model,'architecture_metadata',{'model_name':name}))

    def prediction(m,validation=False,indexes=None):
        prefix='val' if validation else 'train'
        xx=tensors[prefix+'_X'];aa=tensors[prefix+'_A']
        if indexes is not None:xx,aa=xx[indexes],aa[indexes]
        if name=='M00_projected':return torch.maximum(m(xx).double()*scale,torch.sqrt(aa))
        return m(xx,aa) if custom else m(xx).double()*scale

    def supervised_loss(m,indexes):
        if indexes is None:
            return torch.mean(((prediction(m,True)-tensors['val_y_original'])/scale)**2)
        if name.startswith(CANDIDATE):
            outputs=m.forward_components(x[indexes],tensors['train_A'][indexes])
            return m.loss_components(outputs,tensors['train_y_original'][indexes],
                                     tensors['train_U'][indexes])['loss']
        return torch.mean(((prediction(m,indexes=indexes)-tensors['train_y_original'][indexes])/scale)**2)
    phases['supervised']=_phase(model,x,y,vx,vy,Path(directory)/'supervised',config,
        identity={'name':name,'seed':seed,'phase':'supervised','configuration':model_config,'source_identity':source_identity},
        loss_fn=supervised_loss)
    model.eval()
    with torch.no_grad():pred=prediction(model,True).detach().cpu().numpy()
    _torch_save(Path(directory)/'selected_model.pt',{
        'state_dict':_clone(model),'configuration':model_config,'name':name,'seed':seed,
        'scaler':data['scaler'],'source_scope':'development_only'})
    return pred,{'trainable_parameters':sum(p.numel() for p in model.parameters()),
        'configuration':model_config,'phases':{p:{k:s[k] for k in (
            'best_epoch','epoch','steps','measured_completed_epoch_seconds')} for p,s in phases.items()},
        'sae_pretraining_executed':name=='LSTM-autoencoder con predictor',
        'common_optimizer_overrides_historical_optimizer':name in NEURAL,
        'legacy_initial_bias_calibration_executed':False}


def _classical(name,data,npz_path,source_commit,config,directory):
    import pandas as pd
    from e2_temporal_controls import FrozenARIMA,seasonal_naive
    # These are the same two members already admitted by the strict loader.
    # No cal/test/scaled arrays are indexed, even if present in the archive.
    with np.load(npz_path,allow_pickle=False) as archive:
        dates=np.asarray(archive['training_dates']).astype('U10')
        values=np.asarray(archive['training_y_original'],dtype=np.float64)
    if dates.max()>'2024-10-31':raise ValueError('Training era escaped')
    history=pd.Series(values,index=pd.to_datetime(dates))
    if name!='ARIMA':
        period=1 if name=='Persistence' else config['seasonal_period']
        return np.asarray([seasonal_naive(history,d,period).value for d in data['val_dates']]),{
            'real_data_fits':0,'period':period,'seed_replicas':False,'online_truth_lags':True}
    directory=Path(directory)
    if (directory/'arima_fit_started.json').exists():
        raise RuntimeError('Interrupted ARIMA requires reconciliation; it is not automatically fitted again')
    write_json(directory/'arima_fit_started.json',{'unix_time':time.time(),'source_commit':source_commit})
    model=FrozenARIMA.fit(history,data['val_dates'][0],tuple(config['arima_order']),source_commit=source_commit)
    pred=[]
    with (directory/'causal_forecasts.log').open('x') as stream:
        for date in pd.date_range(model.current+pd.Timedelta(days=1),history.index[-1]):
            if date>=model.origin:
                forecast=model.forecast(date).value
                stream.write(canonical({'date':date.date().isoformat(),'prediction':forecast})+'\n')
                stream.flush();os.fsync(stream.fileno())
                pred.append(forecast)
            model.observe(date,float(history.get(date,np.nan)))
    return np.asarray(pred),{'real_data_fits':1,'seed_replicas':False,'fit':model.receipt,
                            'online_filtering_parameters_frozen':True}


def run(output,source_commit,npz_path,expected_sha256,*,suite='controls',
        seeds=(20261118,20261119,20261120),device='cuda:0',package_path=None,
        config=None,allow_cpu_for_synthetic_tests=False):
    """Run or resume completed/epoch-checkpointed DEVELOPMENT jobs in output."""
    import torch
    from src.e3_innovation_model import load_innovation_data
    if not re.fullmatch('[0-9a-f]{40}',source_commit):raise ValueError('Full source commit required')
    if suite not in ('controls','full_family'):raise ValueError('Unknown pilot suite')
    if len(set(seeds))!=len(seeds) or not seeds or any(type(s)is not int or s<0 for s in seeds):
        raise ValueError('Unique explicit nonnegative integer seeds required')
    if device!='cuda:0' and not allow_cpu_for_synthetic_tests:raise ValueError('Use the verified Colab CUDA runtime')
    if device=='cuda:0' and not torch.cuda.is_available():raise ValueError('CUDA not available')
    config={**DEFAULT_CONFIG,**(config or {})}
    if config['max_epochs']<1 or config['patience']<1 or config['batch_size']<1:
        raise ValueError('Positive epoch, patience and batch limits required')
    data=load_innovation_data(npz_path,expected_sha256)
    # The CPU escape hatch is for generated fixtures, never real BTC files.
    if allow_cpu_for_synthetic_tests and data['admission']['n_training_source']>=600:
        raise ValueError('CPU test escape hatch cannot execute the real BTC cohort')
    directory=Path(output);directory.mkdir(parents=True,exist_ok=True)
    names=FAMILY if suite=='full_family' else CONTROLS
    source_files=['src/e3_overlap_model_v3.py','src/e3_innovation_model.py','src/e3_development.py',
                  'src/original_model.py',str(Path(__file__).relative_to(ROOT))]
    request={'kind':'e3_common_development_pilot','source_commit':source_commit,'suite':suite,
             'family':list(names),'seeds':list(seeds),'config':config,'device':device,
             'source_sha256':{p:sha(ROOT/p) for p in source_files},
             'data_admission':data['admission'],'test_only':allow_cpu_for_synthetic_tests,
             'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
             'CUDA_device':torch.cuda.get_device_name(0) if device=='cuda:0' else None,
             'common_policy':'one architecture point; equal epoch policy, SAE three phases counted',
             'HPO_searches':0,'confirmation_protocol_frozen':False,'real_campaign_allowed':False}
    if (directory/'request.json').exists():
        if json.loads((directory/'request.json').read_text())!=request:raise ValueError('Pilot request changed on resume')
    else:write_json(directory/'request.json',request)
    rows=roots=None
    if suite=='full_family':
        package,inventory,rows,roots=_package(directory,package_path)
        sys.path.insert(0,str(package/'code/scripts'))
        write_json(directory/'source_inventory.json',inventory)
    results=[];predictions=[]
    for name in names:
        deterministic=name in CLASSICAL or name.startswith('floor_')
        for seed in ((-1,) if deterministic else seeds):
            job=directory/'jobs'/(re.sub('[^a-zA-Z0-9_-]+','_',name)+f'__{seed}')
            job.mkdir(parents=True,exist_ok=True)
            try:
                if (job/'result.json').exists():
                    result=json.loads((job/'result.json').read_text())
                    if result['request_sha256']!=digest(request):raise ValueError('Completed job request differs')
                    pred=np.asarray(result['predictions'])
                else:
                    start=time.perf_counter()
                    if name in CLASSICAL:pred,extra=_classical(name,data,npz_path,source_commit,config,job)
                    elif name.startswith('floor_'):
                        pred=data['val_floor'].copy()
                        if name=='floor_plus_mean_excess':pred+=data['scaler']['d_fit']
                        extra={'real_data_fits':0,'seed_replicas':False}
                    else:pred,extra=_fit_neural(name,seed,data,job,config,device,rows,roots,
                        source_identity={'source_commit':source_commit,'pilot_request_sha256':digest(request)})
                    if pred.shape!=data['val_y_original'].shape or not np.isfinite(pred).all() or (pred<0).any():
                        raise ValueError('Invalid aligned predictions; no fallback or row deletion')
                    result={'status':'completed','name':name,'seed':seed,'request_sha256':digest(request),
                        'predictions':pred.tolist(),'validation_rmse':float(np.sqrt(np.mean((pred-data['val_y_original'])**2))),
                        'seconds_this_segment':time.perf_counter()-start,**extra}
                    write_json(job/'result.json',result)
                for date,truth,p in zip(data['val_dates'],data['val_y_original'],pred):
                    predictions.append({'model':name,'seed':seed,'date':str(date),'target':float(truth),'prediction':float(p)})
                print(f'{name} seed={seed}: completed; development RMSE={result["validation_rmse"]:.8g}',flush=True)
            except Exception as exc:
                result={'status':'failed','name':name,'seed':seed,'error':f'{type(exc).__name__}: {exc}'}
                write_json(job/'failure.json',result);print(f'{name} seed={seed}: FAILED {exc}',flush=True)
            results.append({k:v for k,v in result.items() if k!='predictions'})
            write_json(directory/'progress.json',results)
    with (directory/'development_predictions.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['model','seed','date','target','prediction'])
        writer.writeheader();writer.writerows(predictions)
    completed=sum(r['status']=='completed' for r in results)
    attempted_jobs=[p for p in (directory/'jobs').iterdir() if p.is_dir() and
                    (any(p.rglob('phase_started.json')) or any(p.rglob('checkpoint.pt')) or (p/'arima_fit_started.json').exists())]
    attempted_phases={p.parent for pattern in ('phase_started.json','checkpoint.pt')
                      for p in (directory/'jobs').rglob(pattern)}
    receipt={'kind':'e3_common_development_pilot_receipt','status':'completed' if completed==len(results) else 'partial_failed',
        'suite':suite,'source_commit':source_commit,'jobs_completed':completed,'jobs_required':len(results),
        'models_with_predictions':len(set(r['model'] for r in predictions)),
        'new_real_data_fits':0 if allow_cpu_for_synthetic_tests else len(attempted_jobs),
        'attempted_model_jobs':len(attempted_jobs),
        'attempted_optimizer_phases':len(attempted_phases),
        'available_phase_checkpoints':len(list((directory/'jobs').rglob('checkpoint.pt'))),
        'fit_counter_scope':'unique attempted model jobs; immutable phase-start markers count before optimizer setup or first checkpoint; prior checkpoints remain a compatibility fallback',
        'expected_full_family':list(FAMILY),'seeds':list(seeds),'development_dates':data['val_dates'].tolist(),
        'predictions_sha256':sha(directory/'development_predictions.csv'),'new_test_predictions':0,
        'additional_datasets_evaluated':0,'R1_accredited':False,'R3_accredited':False,
        'real_campaign_allowed':False,'prospective_power_complete':False,
        'next_step':'Full-family development predictions are required for conditional prospective dimensioning',
        'job_results':results}
    write_json(directory/'pilot_receipt.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--source-commit',required=True)
    parser.add_argument('--npz',required=True);parser.add_argument('--sha256',required=True)
    parser.add_argument('--suite',choices=['controls','full_family'],default='controls')
    parser.add_argument('--package-path');parser.add_argument('--seeds',type=int,nargs='+',default=[20261118,20261119,20261120])
    parser.add_argument('--max-epochs',type=int,default=40);parser.add_argument('--patience',type=int,default=8)
    args=parser.parse_args();result=run(args.output,args.source_commit,args.npz,args.sha256,suite=args.suite,
        seeds=tuple(args.seeds),package_path=args.package_path,
        config={'max_epochs':args.max_epochs,'patience':args.patience})
    print(json.dumps({k:v for k,v in result.items() if k!='job_results'},indent=2))


if __name__=='__main__':main()
