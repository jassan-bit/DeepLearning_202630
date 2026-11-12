"""Reduced E2 final refits, development pilots and sealed historical scoring.

Production fits run only in the bounded workers supervised by the shared
reduced-budget controller. This module never resumes an interrupted optimizer,
changes a recipe from final scores, or treats six seeds as independent markets.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import subprocess
import re

from e2_reduced_protocol import CLASSICAL, FOLDS, NEURAL, PROTOCOL_ID, SYSTEMS, digest

ROOT=Path(__file__).resolve().parents[1]
VERSION='e2-reduced-final-executor-v1'
FINAL_SEEDS=tuple(range(271801,271807))
PILOT_SEEDS=tuple(range(271701,271706))
CALIBRATION_DATES=[(date(2024,11,8)+timedelta(days=i)).isoformat() for i in range(53)]


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path,value):
    from e2_reduced_fit import write_json as durable_write
    durable_write(path,value)


def slug(name):
    return hashlib.sha256(name.encode()).hexdigest()[:16]


def _relative(path,root):
    path,root=Path(path).resolve(),Path(root).resolve()
    if path.is_symlink() or not path.is_relative_to(root):
        raise ValueError('Artifact path must stay inside the private output root')
    return path.relative_to(root).as_posix()


def _artifact_hashes(directory):
    directory=Path(directory)
    return {p.relative_to(directory).as_posix():file_sha(p) for p in directory.rglob('*')
        if p.is_file() and not p.name.startswith('.') and p.name not in
        ('result.json','worker_spec.json','worker.log','worker-spec.json')}


def _trial_builder(name,trial,seed,roots):
    from e2_reduced_fit import _builder
    return _builder(name,trial,seed,roots)


def final_training_split(panel):
    """No calibration/bridge label can enter the final scaler or fit tensors."""
    from e2_temporal_training import _split
    support=panel.support
    dates=support.index[(support.index<=__import__('pandas').Timestamp('2024-10-31',tz='UTC'))&support.D7_target]
    if not len(dates) or dates[-1].strftime('%Y-%m-%d')!='2024-10-31':
        raise ValueError('The common final training cutoff must be October31')
    return _split(panel,'reduced_final_train_through_2024-10-31',tuple(dates))


def _calibration_split(panel,scaler):
    from e2_temporal_training import _split
    import pandas as pd
    dates=pd.date_range('2024-11-08','2024-12-30',tz='UTC')
    if not panel.support.loc[dates,'common'].all():
        raise ValueError('All53 registered calibration dates require common support')
    return _split(panel,'reduced_calibration_no_fit',tuple(dates),scaler)


def validate_selection(name,selection):
    if (selection.get('model_id')!=name or selection.get('all_eighteen_slots_accounted_for') is not True
            or selection.get('run_artifact_bytes_verified') is not True
            or selection.get('source_current_recipe_verified') is not True
            or not isinstance(selection.get('selection_ledger_sha256'),str)
            or len(selection['selection_ledger_sha256'])!=64):
        raise ValueError('A fully audited18-run reduced selection is required: '+name)
    trial=selection.get('winner_trial')
    if name=='TiDE':
        from e2_published_tide import registered_trials
        allowed=registered_trials()
    else:
        from e2_published_registry import registered_trials
        allowed=registered_trials(name)
    if trial not in allowed or trial.get('trial_id')!=selection.get('winner_parent_trial_id'):
        raise ValueError('The winning exact current recipe does not match its selected trial')
    expected={'ae1','ae2','supervised'} if name=='LSTM-autoencoder con predictor' else {'supervised'}
    epochs=selection.get('fixed_phase_epochs')
    if not isinstance(epochs,dict) or set(epochs)!=expected or any(type(v)is not int or not 1<=v<=200 for v in epochs.values()):
        raise ValueError('Every final optimizer phase requires a frozen valid epoch count')
    return trial,epochs


def neural_final_worker(spec,output):
    """One fresh fixed-epoch fit followed by label-free calibration inference."""
    import numpy as np
    import torch
    from e2_development_data import load_development
    from e2_reduced_fit import fit_reduced_job
    from e2_published_pilot_training import predict_pilot
    from e2_reduced_final_analysis import residual_calibrator
    name,seed=spec['model'],spec['seed']
    if name not in NEURAL or seed not in (*FINAL_SEEDS,*PILOT_SEEDS):
        raise ValueError('Unregistered final/pilot model or seed')
    if seed in PILOT_SEEDS and name not in ('iTransformer','GRU32'):
        raise ValueError('Only the two registered pilot representatives are admitted')
    trial,epochs=validate_selection(name,spec['selection'])
    panel,data_receipt=load_development(spec['data_root'],'2024-12-30')
    train=final_training_split(panel);calib=_calibration_split(panel,train.scaler)
    output=Path(output);fitdir=output/'fit'
    result=fit_reduced_job(name,trial,seed,train,None,fitdir,fixed_epochs=epochs,
        source_roots=spec.get('source_roots'),device=spec.get('device','cuda:0'),
        source_identity=spec['source_identity'],runtime_identity=spec['runtime_identity'],
        fit_wall_seconds=spec['fit_wall_seconds'])
    checkpoint=fitdir/result['selected_checkpoint']['path']
    if file_sha(checkpoint)!=result['selected_checkpoint']['sha256']:
        raise ValueError('Final selected weights changed before calibration')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=_trial_builder(name,trial,seed,spec.get('source_roots'))
    model.load_state_dict(saved['state_dict']);model.eval()
    if saved['scaler']!=train.scaler.record():
        raise ValueError('Final saved scaler does not match its training-only identity')
    prediction=predict_pilot(model,calib.x,batch_size=1,device=spec.get('device','cuda:0')).numpy()*train.scaler.target_scale
    scaler_path=output/'scaler.json';write_json(scaler_path,train.scaler.record())
    calibrator=residual_calibrator(prediction,calib.y_original,dates=CALIBRATION_DATES,predictor_sha256=file_sha(checkpoint))
    write_json(output/'calibrator.json',calibrator)
    write_json(output/'calibration_predictions.json',{'model':name,'seed':seed,'dates':CALIBRATION_DATES,
        'prediction_original':prediction.tolist(),'target_original':list(calib.y_original),
        'checkpoint_sha256':file_sha(checkpoint),'same_predictor_as_final':True,'used_for_HPO':False})
    write_json(output/'development_admission.json',data_receipt)
    return {'version':VERSION,'status':'completed','phase':'pilot' if seed in PILOT_SEEDS else 'final-fit',
        'model':name,'seed':seed,'checkpoint':_relative(checkpoint,output),'checkpoint_sha256':file_sha(checkpoint),
        'scaler':'scaler.json','scaler_sha256':file_sha(scaler_path),
        'calibrator':'calibrator.json','calibrator_sha256':file_sha(output/'calibrator.json'),
        'selection_sha256':digest(spec['selection']),'training_label_end':'2024-10-31',
        'calibration_dates':CALIBRATION_DATES,'no_refit_after_calibration':True,
        'training_resources':{k:result[k] for k in ('optimizer_steps','phase_records','architecture')},
        'artifacts_sha256':_artifact_hashes(output)}


def _history(panel):
    import pandas as pd
    return pd.Series(panel.targets,index=panel.calendar.tz_localize(None))


def _naive_predictions(history,dates,lag):
    from e2_temporal_controls import seasonal_naive
    return [float(seasonal_naive(history,d,lag).value) for d in dates]


def _arima_walk(model,history,end,output):
    """Persist every causal forecast before revealing its same-day target."""
    import pandas as pd
    from e2_temporal_controls import day
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    values={};current=model.current+pd.Timedelta(days=1)
    end=day(end)
    while current<=end:
        if current>=model.origin:
            forecast=model.forecast(current)
            value=float(forecast.value)
            # Exclusive dated receipt prevents silent state replay after a crash.
            write_json(output/(current.strftime('%Y-%m-%d')+'.json'),
                {'date':current.strftime('%Y-%m-%d'),'prediction':value,'raw_prediction':float(forecast.raw_value),
                 'coefficients_refitted':False})
            values[current.strftime('%Y-%m-%d')]=value
        model.observe(current,history.get(current,float('nan')))
        current+=pd.Timedelta(days=1)
    return values


def classical_worker(spec,output):
    """One classical CV opportunity or one final fit; all time is budgeted."""
    import numpy as np
    import pandas as pd
    from e2_development_data import load_development
    from e2_temporal_training import prepare_b
    from e2_temporal_controls import FrozenARIMA
    from e2_reduced_final_analysis import residual_calibrator
    name=spec['model'];parameters=spec['parameters'];mode=spec['mode'];output=Path(output)
    if name not in CLASSICAL or mode not in ('classical-cv','classical-final'):
        raise ValueError('Unknown classical execution request')
    panel,data_receipt=load_development(spec['data_root'],'2024-12-30');history=_history(panel)
    if mode=='classical-cv':
        if spec['fold_origin'] not in FOLDS:
            raise ValueError('Unregistered classical CV origin')
        prepared=prepare_b(panel,spec['fold_origin'],spec['fold_origin'])
        dates=[d.strftime('%Y-%m-%d') for d in prepared.inner_validation.dates]
        truth=np.asarray(prepared.inner_validation.y_original)
        origin=dates[0]
    else:
        dates=CALIBRATION_DATES;truth=panel.targets_at(dates);origin=dates[0]
    if name=='ARIMA':
        order=tuple(parameters['order'])
        if order not in tuple((p,d,q) for p in range(3) for d in range(2) for q in range(3)):
            raise ValueError('Unregistered ARIMA order')
        try:
            model=FrozenARIMA.fit(history,origin,order,source_commit=spec['source_identity']['source_commit'])
        except Exception as error:
            receipt=getattr(error,'receipt',{})
            if mode!='classical-cv' or receipt.get('converged') is not False:
                raise
            write_json(output/'candidate_failure.json',{'candidate_status':'failed','reason':str(error),'receipt':receipt,'replacement_performed':False})
            return {'version':VERSION,'status':'completed','candidate_status':'failed','phase':mode,
                'model':name,'seed':'deterministic','parameters':parameters,'validation_count':len(truth),
                'prediction_dates':dates,'reason':str(error),'rmse_original':None,
                'artifacts_sha256':_artifact_hashes(output)}
        parameter_fingerprint=digest({'order':list(order),'parameters':model.receipt['parameters'],'scale':model.scale})
        forecasts=_arima_walk(model,history,dates[-1],output/'causal_predictions')
        prediction=np.asarray([forecasts[d] for d in dates])
        if digest({'order':list(order),'parameters':model.receipt['parameters'],'scale':model.scale})!=parameter_fingerprint:
            raise ValueError('ARIMA coefficients changed during causal calibration')
        with (output/'predictor.pkl').open('xb') as stream:
            pickle.dump(model,stream,protocol=5);stream.flush();os.fsync(stream.fileno())
        checkpoint='predictor.pkl';scaler={'target_scale':model.scale,'fit_cutoff':model.receipt['fit_cutoff']}
        write_json(output/'classical_fit.json',model.receipt)
    else:
        lag=1 if name=='Persistence' else int(parameters['period'])
        if name=='SeasonalNaive' and lag not in (7,14,28):
            raise ValueError('Unregistered seasonal period')
        prediction=np.asarray(_naive_predictions(history,dates,lag));checkpoint='predictor.json'
        write_json(output/checkpoint,{'model':name,'lag':lag,'parameters_fixed':True})
        scaler={'target_scale':1.,'scaling':'none','fit_cutoff':'2024-10-31' if mode=='classical-final' else None}
    write_json(output/'scaler.json',scaler)
    write_json(output/'predictions.json',{'dates':dates,'prediction_original':prediction.tolist(),'target_original':truth.tolist()})
    result={'version':VERSION,'status':'completed','phase':mode,'model':name,'seed':'deterministic',
        'parameters':parameters,'checkpoint':checkpoint,'checkpoint_sha256':file_sha(output/checkpoint),
        'scaler':'scaler.json','scaler_sha256':file_sha(output/'scaler.json'),
        'rmse_original':float(np.sqrt(np.mean((prediction-truth)**2))),'validation_count':len(truth),
        'prediction_dates':dates,'source_identity':spec['source_identity']}
    if mode=='classical-final':
        calibrator=residual_calibrator(prediction,truth,dates=dates,predictor_sha256=file_sha(output/checkpoint))
        write_json(output/'calibrator.json',calibrator)
        result.update(calibrator='calibrator.json',calibrator_sha256=file_sha(output/'calibrator.json'),
            training_label_end='2024-10-31',no_refit_after_calibration=True)
    result['artifacts_sha256']=_artifact_hashes(output)
    return result


def worker(spec_path):
    spec=read_json(spec_path);expected=spec.pop('worker_spec_sha256')
    if digest(spec)!=expected:
        raise ValueError('Worker specification checksum mismatch')
    output=Path(spec['output']);output.mkdir(parents=True,exist_ok=True)
    if (output/'result.json').exists():
        raise RuntimeError('An existing worker result belongs to the parent supervisor, not a new worker')
    if spec.get('fit_wall_seconds') != 600 or spec.get('device') != 'cuda:0':
        raise ValueError('Keep the prospectively registered600-second CUDA worker policy')
    source=spec.get('source_identity',{})
    if not re.fullmatch('[0-9a-f]{40}',source.get('source_commit','')) or not isinstance(source.get('files_sha256'),dict) or not source['files_sha256']:
        raise ValueError('Actual source commit and byte-bound source files are required')
    actual_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if actual_commit!=source['source_commit']:
        raise ValueError('Final worker checkout differs from the registered execution commit')
    for relative,checksum in source['files_sha256'].items():
        path=ROOT/relative
        if not path.resolve().is_relative_to(ROOT.resolve()) or path.is_symlink() or file_sha(path)!=checksum:
            raise ValueError('Final worker executable bytes differ from registration')
    from run_e2_reduced_selection import worker_preflight_policy
    actual=worker_preflight_policy()['runtime']
    for key in ('python','torch','torch_cuda','device_name','total_memory_bytes','cpu_threads','interop_threads','deterministic','tf32','cudnn_tf32'):
        if actual.get(key)!=spec['runtime_identity'].get(key):
            raise ValueError('Actual final worker policy/runtime changed: '+key)
    if spec.get('kind')=='neural-final':
        result=neural_final_worker(spec,output)
    elif spec.get('kind')=='classical':
        result=classical_worker(spec,output)
    elif spec.get('kind')=='final-score':
        final_output=Path(spec['final_output'])
        if file_sha(final_output/'final_freeze.json')!=spec['final_freeze_sha256']:
            raise ValueError('The complete predictor freeze changed before scoring')
        result=score_final(spec['project_root'],spec['data_root'],final_output,
            daily_csv=spec.get('daily_csv'),raw_root=spec.get('raw_root'),
            allow_download=spec.get('allow_download',False),device=spec['device'],
            source_roots=spec.get('source_roots'),scoring_output=output/'scientific_outputs')
        result={**result,'phase':'score','model':'all16','seed':'aggregate',
            'A2':'scientific_outputs/benchmark_A2.csv','artifacts_sha256':_artifact_hashes(output)}
    else:
        raise ValueError('Unknown reduced final worker kind')
    result['worker_spec_sha256']=expected
    result['attempt_id']=digest({'version':VERSION,'worker_spec_sha256':expected})
    write_json(output/'result.json',result)
    return result


def _dispatch(spec,directory,budget,*,persist_result=None,python=None,supervisor=None):
    from run_e2_reduced_selection import supervise_command
    supervisor=supervisor or supervise_command
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    spec={**spec,'output':str(directory.resolve()),'fit_wall_seconds':budget.stage_wall_seconds}
    spec_sha=digest(spec);spec['worker_spec_sha256']=spec_sha
    specpath=directory/'worker_spec.json';write_json(specpath,spec)
    phases=3 if spec.get('model')=='LSTM-autoencoder con predictor' else 1
    command=[str(python or sys.executable),'-u',str(Path(__file__).resolve()),'--worker-spec',str(specpath)]
    receipt=supervisor(command,budget,digest({'version':VERSION,'spec_sha256':spec_sha}),directory,spec_sha,phases=phases)
    if persist_result is not None:
        persist_result(directory,receipt)
    return receipt['result']


def _base_spec(data_root,source_identity,runtime_identity,source_roots,device):
    if isinstance(source_roots,(str,Path)):source_roots=read_json(source_roots)
    return {'version':VERSION,'data_root':str(Path(data_root).resolve()),'source_identity':source_identity,
        'runtime_identity':runtime_identity,'source_roots':source_roots,'device':device}


def run_final_fits(data_root,selections,output,budget,*,source_identity,runtime_identity,
                   source_roots=None,device='cuda:0',models=None,pilot=True,final=True,
                   persist_result=None,python=None,supervisor=None):
    """Incremental model-by-model final fits. Global scoring remains sealed."""
    requested=tuple(models or NEURAL)
    if any(name not in NEURAL for name in requested):raise ValueError('Unknown neural system')
    results=[];pending=[];base=_base_spec(data_root,source_identity,runtime_identity,source_roots,device)
    for name in requested:
        selection=selections.get(name)
        if selection is None:
            pending.append({'model':name,'state':'HPO_not_closed'});continue
        validate_selection(name,selection)
        schedule=[]
        if pilot and name in ('iTransformer','GRU32'):
            schedule.extend(('pilot',seed) for seed in PILOT_SEEDS)
        if final:schedule.extend(('final-fit',seed) for seed in FINAL_SEEDS)
        for phase,seed in schedule:
            directory=Path(output)/phase/slug(name)/str(seed)
            spec={**base,'kind':'neural-final','model':name,'seed':seed,'selection':selection}
            results.append(_dispatch(spec,directory,budget,persist_result=persist_result,python=python,supervisor=supervisor))
    return {'version':VERSION,'completed_predictors':len(results),'results':results,'pending':pending,
        'all_sixteen_finalized':False,'final_market_values_read':False}


def run_classical(data_root,output,budget,*,source_identity,runtime_identity,device='cuda:0',
                   persist_result=None,python=None,supervisor=None):
    """Same three CV folds for seasonal periods and all18 ARIMA orders."""
    import math
    base=_base_spec(data_root,source_identity,runtime_identity,None,device)
    results=[]
    for name in CLASSICAL:
        if name=='Persistence':chosen={'lag':1};ledger=[]
        else:
            configs=([{'period':period} for period in (7,14,28)] if name=='SeasonalNaive'
                else [{'order':[p,d,q]} for p in range(3) for d in range(2) for q in range(3)])
            candidates=[];ledger=[]
            for config in configs:
                scores=[]
                for fold in FOLDS:
                    directory=Path(output)/'classical-cv'/slug(name)/digest(config)[:16]/fold
                    spec={**base,'kind':'classical','mode':'classical-cv','model':name,'parameters':config,'fold_origin':fold}
                    row=_dispatch(spec,directory,budget,persist_result=persist_result,python=python,supervisor=supervisor)
                    if row['validation_count']!=83:raise ValueError('Classical common CV support changed')
                    status=row.get('candidate_status','completed')
                    if status=='completed':scores.append(row['rmse_original'])
                    ledger.append({'parameters':config,'fold':fold,'candidate_status':status,'reason':row.get('reason'),
                        'result_sha256':file_sha(directory/'result.json'),'RMSE':row['rmse_original']})
                if len(scores)==3:candidates.append((math.sqrt(sum(v*v for v in scores)/3),digest(config),config))
            if not candidates:raise ValueError('No classical configuration completed all three folds; do not substitute')
            chosen=min(candidates,key=lambda item:(item[0],item[1]))[2]
        selection_path=Path(output)/'classical-selections'/(slug(name)+'.json')
        write_json(selection_path,{'model':name,'parameters':chosen,'ledger':ledger,'selection_metric':'pooled_original_unit_CV_RMSE','calibration_used_for_selection':False})
        directory=Path(output)/'classical-final'/slug(name)
        spec={**base,'kind':'classical','mode':'classical-final','model':name,'parameters':chosen}
        results.append(_dispatch(spec,directory,budget,persist_result=persist_result,python=python,supervisor=supervisor))
    return results


def close_pilot(output,*,persist_result=None):
    from e2_reduced_final_analysis import pilot_power_mde
    predictions={};truth=None
    for name in ('iTransformer','GRU32'):
        predictions[name]=[]
        for seed in PILOT_SEEDS:
            directory=Path(output)/'pilot'/slug(name)/str(seed)
            result=read_json(directory/'result.json')
            p=read_json(directory/'calibration_predictions.json')
            if p['dates']!=CALIBRATION_DATES or result.get('status')!='completed':
                raise ValueError('The full development pilot must be completed')
            predictions[name].append(p['prediction_original'])
            if truth is not None and truth!=p['target_original']:raise ValueError('Pilot targets changed between representatives')
            truth=p['target_original']
    report=pilot_power_mde(truth,predictions,dates=CALIBRATION_DATES)
    path=Path(output)/'pilot_power_MDE.json';write_json(path,report)
    if persist_result is not None:persist_result(Path(output),{'kind':'pilot_power_MDE','status':'completed','attempt_id':'pilot-power-MDE-'+file_sha(path)[:20],'artifacts_sha256':{'pilot_power_MDE.json':file_sha(path)}})
    return report


def freeze_final(output,*,persist_result=None):
    from e2_reduced_final_analysis import analysis_registration
    from e2_reduced_final_data import validate_final_freeze, PROTOCOL_SHA256
    output=Path(output);power=read_json(output/'pilot_power_MDE.json')
    if power.get('status')!='development_pilot_completed_power_reported':
        raise ValueError('Close the actual development power/MDE pilot before final scoring')
    registration=analysis_registration();write_json(output/'analysis_registration.json',registration)
    rows=[];files={'pilot_power_MDE.json':file_sha(output/'pilot_power_MDE.json'),
                  'analysis_registration.json':file_sha(output/'analysis_registration.json')}
    schedule=[(name,seed,output/'final-fit'/slug(name)/str(seed)) for name in NEURAL for seed in FINAL_SEEDS]
    schedule.extend((name,'deterministic',output/'classical-final'/slug(name)) for name in CLASSICAL)
    for name,seed,directory in schedule:
        result=read_json(directory/'result.json')
        if result.get('status')!='completed' or result.get('model')!=name or result.get('seed')!=seed or result.get('no_refit_after_calibration') is not True:
            raise ValueError('All81 exact final predictors must complete before data admission')
        calibrated=read_json(directory/result['calibrator'])
        if (calibrated.get('predictor_sha256')!=result['checkpoint_sha256']
                or calibrated.get('dates')!=CALIBRATION_DATES
                or calibrated.get('sha256')!=digest({k:v for k,v in calibrated.items() if k!='sha256'})):
            raise ValueError('Every calibrator must bind the same frozen predictor and53development dates')
        scaler=read_json(directory/result['scaler'])
        cutoff=(scaler.get('train_dates') or [scaler.get('fit_cutoff')])[-1]
        if str(cutoff)[:10]!='2024-10-31':
            raise ValueError('Every final predictor must retain the common October31training cutoff')
        row={'model':name,'seed':seed}
        for field in ('checkpoint','scaler','calibrator'):
            path=directory/result[field];checksum=result[field+'_sha256']
            if file_sha(path)!=checksum:raise ValueError('Final predictor artifact changed: '+str(path))
            relative=_relative(path,output);files[relative]=checksum
            row[field]=relative;row[field+'_sha256']=checksum
        rows.append(row)
    value={'version':VERSION,'protocol_id':PROTOCOL_ID,'protocol_sha256':PROTOCOL_SHA256,'systems':list(SYSTEMS),'final_seed_ids':list(FINAL_SEEDS),
        'models':rows,'frozen_files_sha256':files,'pilot_power_MDE_closed':True,'analysis_frozen':True,
        'analysis_registration_sha256':registration['sha256'],'training_label_end':'2024-10-31',
        'weights_scalers_and_calibrators_frozen_before_final_data':True,'known_historical_calendar':True,
        'prospective_blinding_recovered':False,'retune_from_final_scores_allowed':False}
    validate_final_freeze(value,output)
    write_json(output/'final_freeze.json',value)
    if persist_result is not None:persist_result(output,{'kind':'final_freeze','status':'completed','attempt_id':'final-freeze-'+file_sha(output/'final_freeze.json')[:20],'artifacts_sha256':{**files,'final_freeze.json':file_sha(output/'final_freeze.json')}})
    return value


def _materialize_known_daily(project_root,final_output,freeze,*,source_commit):
    """Read the pinned known E1 derivative only after the complete artifact gate."""
    import urllib.request
    from e2_reduced_final_data import validate_final_freeze,E1_DAILY,E1_DAILY_SHA256
    validate_final_freeze(freeze,final_output)
    if not re.fullmatch('[0-9a-f]{40}',source_commit):
        raise ValueError('An immutable source commit is required for the known E1 derivative')
    local=Path(project_root)/E1_DAILY
    target=Path(final_output)/'private-known-E1-source'/'daily_market_test.csv'
    if local.is_file():return local
    if target.is_file():
        if file_sha(target)!=E1_DAILY_SHA256:raise ValueError('The private known E1 derivative changed')
        return target
    url='https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/'+source_commit+'/'+E1_DAILY
    with urllib.request.urlopen(url,timeout=45) as response:
        raw=response.read(2*1024*1024+1)
    if len(raw)>2*1024*1024 or hashlib.sha256(raw).hexdigest()!=E1_DAILY_SHA256:
        raise ValueError('Known E1 daily derivative differs from its pinned original SHA256')
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return target


def score_final(project_root,data_root,output,*,daily_csv=None,raw_root=None,allow_download=False,
                 device='cuda:0',source_roots=None,persist_result=None,scoring_output=None):
    """One aggregate scoring after81artifactfreeze; never selects or refits."""
    import csv
    import numpy as np
    import pandas as pd
    import torch
    from e2_reduced_final_data import reserve_known_e1_evaluation,load_known_e1_evaluation
    from e2_reduced_final_analysis import analyze_benchmark
    from e2_temporal_training import FoldScaler
    from e2_published_pilot_training import predict_pilot
    output=Path(output);freeze=read_json(output/'final_freeze.json')
    score_dir=output/'score' if scoring_output is None else Path(scoring_output)
    if (score_dir/'result.json').exists():
        previous=read_json(score_dir/'result.json')
        for relative,checksum in previous['artifacts_sha256'].items():
            if file_sha(score_dir/relative)!=checksum:raise ValueError('Saved final scoring artifact changed')
        if persist_result is not None:persist_result(score_dir,previous)
        return previous
    if score_dir.exists() and any(score_dir.iterdir()):
        raise RuntimeError('reconciliation_required: partial final scoring is preserved, not repeated automatically')
    score_dir.mkdir(parents=True,exist_ok=True)
    write_json(score_dir/'scoring_started.json',{'freeze_sha256':digest(freeze),'analysis_registration_sha256':freeze['analysis_registration_sha256'],'retune_allowed':False})
    reservation=reserve_known_e1_evaluation(project_root)
    if daily_csv is None:
        actual_source_commit=subprocess.check_output(['git','-C',str(project_root),'rev-parse','HEAD'],text=True).strip()
        daily_csv=_materialize_known_daily(project_root,output,freeze,source_commit=actual_source_commit)
    panel,admission=load_known_e1_evaluation(project_root,reservation,freeze,artifact_root=output,
        daily_csv=daily_csv,raw_root=raw_root,allow_download=allow_download,output_dir=output/'private-known-E1-data',development_root=data_root)
    dates=pd.date_range('2025-01-01','2026-08-31',tz='UTC');date_strings=dates.strftime('%Y-%m-%d').tolist()
    mask=panel.support.loc[dates,'common'].to_numpy(dtype=bool);targets=panel.targets_at(dates)
    prediction={name:[] for name in SYSTEMS};calibrators={name:[] for name in SYSTEMS}
    history=_history(panel)
    for row in freeze['models']:
        name=row['model'];checkpoint=output/row['checkpoint'];calibrator=read_json(output/row['calibrator'])
        if name in NEURAL:
            result_path=checkpoint.parent.parent/'result.json';result=read_json(result_path)
            spec=read_json(checkpoint.parent.parent/'worker_spec.json');trial=spec['selection']['winner_trial']
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            record=saved['scaler'];scaler=FoldScaler(tuple(record['mean']),tuple(record['std']),record['target_scale'],
                tuple(pd.Timestamp(d,tz='UTC') for d in record['train_dates']),record['training_sha256'])
            model=_trial_builder(name,trial,row['seed'],source_roots);model.load_state_dict(saved['state_dict']);model.eval()
            values=np.full(608,np.nan)
            x=torch.as_tensor(scaler.transform(panel.raw_windows(dates[mask])),dtype=torch.float32)
            values[mask]=predict_pilot(model,x,batch_size=1,device=device).numpy()*scaler.target_scale
            del model
        elif name=='ARIMA':
            with checkpoint.open('rb') as stream:model=pickle.load(stream)
            before=list(model.receipt['parameters'])
            forecasts=_arima_walk(model,history,dates[-1],score_dir/'ARIMA_causal_predictions')
            values=np.asarray([forecasts[d] for d in date_strings])
            if before!=model.receipt['parameters']:raise ValueError('ARIMA coefficients changed during scoring')
        else:
            config=read_json(checkpoint);values=np.full(608,np.nan)
            values[mask]=_naive_predictions(history,dates[mask].strftime('%Y-%m-%d').tolist(),config['lag'])
        prediction[name].append(values.tolist());calibrators[name].append(calibrator)
        write_json(score_dir/('predictions_'+slug(name)+'_'+str(row['seed'])+'.json'),
            {'model':name,'seed':row['seed'],'dates':date_strings,'prediction_original':[None if not np.isfinite(v) else float(v) for v in values],
             'checkpoint_sha256':row['checkpoint_sha256'],'mask_sha256':digest(mask.tolist())})
    analysis=analyze_benchmark(targets,prediction,dates=date_strings,mask=mask,calibrators=calibrators)
    write_json(score_dir/'benchmark_analysis.json',analysis)
    write_json(score_dir/'data_admission.json',admission)
    with (score_dir/'benchmark_A2.csv').open('x',newline='',encoding='utf-8') as stream:
        keys=['model','RMSE','RMSE_seed_SD','MAE','sMAPE','R2_mean','coverage90','mean_interval_width90','R','n_common_dates']
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        for row in analysis['table_A2']:writer.writerow({k:row[k] for k in keys})
    result={'version':VERSION,'status':'completed','systems':16,'actual_new_E2_final_metrics':True,
        'freeze_sha256':digest(freeze),'known_inherited_calendar':True,'retuning_performed':False,
        'attempt_id':'final-score-'+digest(freeze)[:20],'A2':'benchmark_A2.csv','artifacts_sha256':_artifact_hashes(score_dir)}
    write_json(score_dir/'result.json',result)
    if persist_result is not None:persist_result(score_dir,result)
    return result


def run_final_scoring(project_root,data_root,output,budget,*,source_identity,runtime_identity,
                      source_roots=None,device='cuda:0',daily_csv=None,raw_root=None,allow_download=False,
                      persist_result=None,python=None,supervisor=None):
    """Budgeted final inference/analysis worker; keeps raw data outside backup artifacts."""
    final_output=Path(output)
    spec={**_base_spec(data_root,source_identity,runtime_identity,source_roots,device),
        'kind':'final-score','model':'all16','project_root':str(Path(project_root).resolve()),
        'final_output':str(final_output.resolve()),'final_freeze_sha256':file_sha(final_output/'final_freeze.json'),
        'daily_csv':None if daily_csv is None else str(Path(daily_csv).resolve()),
        'raw_root':None if raw_root is None else str(Path(raw_root).resolve()),'allow_download':allow_download}
    return _dispatch(spec,final_output/'score-worker',budget,persist_result=persist_result,python=python,supervisor=supervisor)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker-spec',required=True)
    args=parser.parse_args(argv);result=worker(args.worker_spec)
    print(json.dumps({'status':result['status'],'phase':result['phase'],'model':result['model'],'seed':result['seed']},ensure_ascii=False),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
