"""Run the registered first DEVELOPMENT benchmark, without opening test data.

Training chronology and statistical scope are specified before fitting in
experiments/dl_benchmark_20261104/protocol_initial.json. Search, pilot and the
fresh-seed development benchmark are separate, resumable phases. A run consists
of an inner epoch-selection fit and a fresh full-2023 fixed-epoch refit.
"""
from __future__ import annotations

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
from pathlib import Path
import platform
import random
import subprocess
import time
import traceback

import numpy as np
import pandas as pd
from scipy.stats import norm
import torch
from torch import nn

from dl_data import load_development, split_training, fit_preprocessing, transform_windows, mase_scale
from dl_models import build_model, count_parameters, architecture_spec

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/dl_benchmark_20261104'
INITIAL = ROOT / 'experiments/dl_benchmark_20261104/protocol_initial.json'
FAMILIES = ('MLP', 'LSTM', 'CNN1D')
_DATA = None
_PROTOCOL = None
_CONTEXT = None


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def event(kind, **details):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'events.jsonl').open('a') as stream:
        stream.write(json.dumps(clean({'at_utc': stamp(), 'event': kind, **details}), ensure_ascii=False)+'\n')


def runtime():
    versions = {}
    for name in ['torch', 'numpy', 'pandas', 'scipy', 'statsmodels', 'matplotlib']:
        versions[name] = importlib.metadata.version(name)
    cpu_max = Path('/sys/fs/cgroup/cpu.max')
    memory_max = Path('/sys/fs/cgroup/memory.max')
    return {'at_utc': stamp(), 'python': platform.python_version(), 'platform': platform.platform(),
            'versions': versions, 'device': 'cpu', 'cuda_available': torch.cuda.is_available(),
            'cpu_count': os.cpu_count(), 'cpu_affinity_count': len(os.sched_getaffinity(0)),
            'cgroup_cpu_max': cpu_max.read_text().strip() if cpu_max.exists() else None,
            'cgroup_memory_max': memory_max.read_text().strip() if memory_max.exists() else None,
            'torch_threads_per_worker': 1, 'torch_interop_threads': 1,
            'deterministic_algorithms': True, 'mkldnn_enabled': torch.backends.mkldnn.enabled,
            'timing_scope': 'wall time; search concurrency3, pilot and development concurrency1; no GPU'}


def register_context():
    protocol = json.loads(INITIAL.read_text())
    if protocol['data']['test_files_opened_allowed'] != 0:
        raise ValueError('This runner admits development data only')
    source_paths = ['scripts/train_dl_benchmark.py', 'scripts/dl_data.py', 'scripts/dl_models.py']
    code_hashes = {name: digest(ROOT/name) for name in source_paths}
    for name, expected in protocol['source_files'].items():
        if digest(ROOT/name) != expected:
            raise ValueError(f'Registered source changed: {name}')
    observed_runtime = runtime()
    runtime_signature = {key: observed_runtime[key] for key in ['python','versions','device','mkldnn_enabled','cpu_affinity_count','cgroup_cpu_max']}
    signature = hashlib.sha256(json.dumps({'protocol': digest(INITIAL), 'code': code_hashes, 'runtime':runtime_signature}, sort_keys=True).encode()).hexdigest()
    manifest = {'at_utc': stamp(), 'protocol_path': str(INITIAL.relative_to(ROOT)),
                'protocol_sha256': digest(INITIAL), 'training_code_sha256': code_hashes,
                'code_signature': signature, 'source_commit': subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'runtime': observed_runtime, 'project_test_files_opened': 0}
    path = OUT/'code_manifest_before_search.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['code_signature'] != signature:
            raise RuntimeError('Training code changed after registration; preserve results and create an explicit experiment version')
        manifest = previous
    else:
        write_json(path, manifest)
        event('registered_training_code_before_first_project_fit', code_signature=signature)
    write_json(OUT/'runtime.json', manifest['runtime'])
    (OUT/'protocol_initial.json').write_bytes(INITIAL.read_bytes())
    return protocol, manifest


def worker_init(protocol, context):
    global _DATA, _PROTOCOL, _CONTEXT
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    _PROTOCOL, _CONTEXT = protocol, context
    _DATA = load_development(ROOT)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def metrics(y, pred, denominator):
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    if len(y) == 0 or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError('All planned predictions must be finite and cover the common cohort')
    error = y-pred
    mse = float(np.mean(error**2))
    positive = y > 0
    total = float(np.sum((y-y.mean())**2))
    return {'rmse': float(np.sqrt(mse)), 'mae': float(np.mean(np.abs(error))), 'mse': mse,
            'r2': 1-float(np.sum(error**2))/total if total > 0 else float('nan'),
            'mape': float(100*np.mean(np.abs(error[positive]/y[positive]))) if positive.any() else float('nan'),
            'mape_coverage': float(positive.mean()),
            'mase': float(np.mean(np.abs(error)/denominator)) if np.isfinite(denominator) and denominator > 0 else float('nan'),
            'mase_coverage': 1.0 if np.isfinite(denominator) and denominator > 0 else 0.0}


def optimization_fit(job, x, y, xv=None, yv=None, epochs=None, stage='inner'):
    protocol = _PROTOCOL['training']
    stage_seed = job['seed']+(10000000 if stage == 'refit' else 0)
    start = time.perf_counter()
    seed_all(stage_seed)
    model = build_model(job['model'], job['history_days'], job['width'], protocol['dropout'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=job['learning_rate'], weight_decay=protocol['weight_decay'])
    loss_fn = nn.MSELoss()
    tx, ty = torch.from_numpy(x), torch.from_numpy(y)
    vx = torch.from_numpy(xv) if xv is not None else None
    vy = torch.from_numpy(yv) if yv is not None else None
    batch = protocol['batch_size']
    shuffle = torch.Generator().manual_seed(stage_seed+17)
    best, best_epoch, stale, rows = float('inf'), 0, 0, []
    maximum = int(epochs if epochs is not None else protocol['max_epochs'])
    for epoch in range(1, maximum+1):
        model.train()
        permutation = torch.randperm(len(tx), generator=shuffle)
        losses, seen = 0.0, 0
        for first in range(0, len(tx), batch):
            indices = permutation[first:first+batch]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(tx[indices])
            loss = loss_fn(prediction, ty[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), protocol['gradient_clip_norm'], error_if_nonfinite=True)
            optimizer.step()
            losses += float(loss.detach())*len(indices)
            seen += len(indices)
        model.eval()
        with torch.inference_mode():
            train_loss = float(loss_fn(model(tx), ty))
            validation_loss = float(loss_fn(model(vx), vy)) if vx is not None else float('nan')
        if not np.isfinite(train_loss) or (vx is not None and not np.isfinite(validation_loss)):
            raise FloatingPointError('Nonfinite epoch loss')
        if vx is not None:
            if validation_loss < best-protocol['min_delta']:
                best, best_epoch, stale = validation_loss, epoch, 0
            else:
                stale += 1
        rows.append({'epoch': epoch, 'stage': stage, 'optimization_mse_scaled': losses/seen,
                     'train_mse_scaled_eval': train_loss, 'validation_mse_scaled_eval': validation_loss,
                     'best_epoch_so_far': best_epoch if vx is not None else epoch,
                     'learning_rate': job['learning_rate']})
        if vx is not None and stale >= protocol['patience']:
            break
    seconds = time.perf_counter()-start
    if vx is not None and best_epoch < 1:
        raise RuntimeError('No admissible inner epoch')
    return model, (best_epoch if vx is not None else maximum), rows, seconds


def infer(model, x):
    model.eval()
    tx = torch.from_numpy(x)
    def forward():
        chunks = [model(tx[i:i+256]).detach().cpu().numpy() for i in range(0, len(tx), 256)]
        return np.concatenate(chunks)
    with torch.inference_mode():
        for _ in range(5):
            forward()
        durations, pred = [], None
        for _ in range(20):
            t = time.perf_counter()
            pred = forward()
            durations.append(time.perf_counter()-t)
    return pred, float(np.mean(durations)), float(np.std(durations, ddof=1))


def run_job(job):
    try:
        return _run_job(job)
    except Exception:
        raise RuntimeError(json.dumps(job)+'\n'+traceback.format_exc())


def _run_job(job):
    data, protocol = _DATA, _PROTOCOL
    prefix = _CONTEXT['code_signature'][:10]
    run_id = f"{job['phase']}_{job['model']}_D{job['history_days']}_{job['config_id']}_s{job['seed']}_{prefix}"
    directory = OUT/'runs'/run_id
    complete = directory/'summary.json'
    if complete.exists():
        cached = json.loads(complete.read_text())
        if cached['code_signature'] != _CONTEXT['code_signature']:
            raise RuntimeError('Cached run has incompatible code')
        for name, expected in cached['artifact_sha256'].items():
            if digest(directory/name) != expected:
                raise RuntimeError('Cached run artifact changed')
        return cached
    if directory.exists() and any(directory.iterdir()):
        directory.rename(directory.with_name(directory.name+'_incomplete_'+str(time.time_ns())))
    directory.mkdir(parents=True, exist_ok=True)
    history = job['history_days']
    split = split_training(data, history)
    train, itrain, ival = [split[k] for k in ['train_indices','inner_train_indices','inner_validation_indices']]
    eval_indices = split['selection_indices' if job['phase'] == 'search' else 'calibration_indices']
    inner_preproc = fit_preprocessing(data, history, itrain)
    full_preproc = fit_preprocessing(data, history, train)
    xi, yi = transform_windows(data, history, itrain, inner_preproc)
    xv, yv = transform_windows(data, history, ival, inner_preproc)
    _, best_epoch, inner_curve, inner_seconds = optimization_fit(job, xi, yi, xv, yv, stage='inner')
    xt, yt = transform_windows(data, history, train, full_preproc)
    model, refit_epochs, refit_curve, refit_seconds = optimization_fit(job, xt, yt, epochs=best_epoch, stage='refit')
    xe, _ = transform_windows(data, history, eval_indices, full_preproc)
    pred_scaled, inference_seconds, inference_sd = infer(model, xe)
    predictions = pred_scaled.astype(np.float64)*full_preproc['y_scale']
    labels = data['targets'][eval_indices]
    persistence = data['persistence'][eval_indices]
    scale = mase_scale(data, train)
    values = metrics(labels, predictions, scale)
    count = count_parameters(model)
    record = {**job, 'run_id': run_id, 'started_code_signature': _CONTEXT['code_signature'],
              'train_samples': len(train), 'inner_train_samples': len(itrain),
              'inner_validation_samples': len(ival), 'phase_samples': len(eval_indices),
              'best_epoch': best_epoch, 'refit_epochs': refit_epochs,
              'parameters': count['trainable'], 'non_trainable_parameters': count['non_trainable'],
              'search_fit_seconds': inner_seconds, 'refit_seconds': refit_seconds,
              'training_seconds': inner_seconds+refit_seconds, 'inference_seconds': inference_seconds,
              'inference_seconds_sd': inference_sd, 'inference_seconds_per_sample': inference_seconds/len(labels),
              'mase_train_scale': scale, **values}
    prediction_rows = pd.DataFrame({'phase': job['phase'], 'run_id': run_id, 'model': job['model'],
        'history_days': history, 'config_id': job['config_id'], 'seed': job['seed'],
        'outcome_day': data['calendar'][eval_indices].strftime('%Y-%m-%d'),
        'prediction_origin_utc': [date.isoformat() for date in data['calendar'][eval_indices]],
        'target_available_at_utc': [(date+pd.Timedelta(days=1)).isoformat() for date in data['calendar'][eval_indices]],
        'input_first_day': data['calendar'][eval_indices-history].strftime('%Y-%m-%d'),
        'input_last_day': data['calendar'][eval_indices-1].strftime('%Y-%m-%d'),
        'y_true': labels, 'y_pred': predictions, 'persistence_pred': persistence})
    prediction_rows.to_csv(directory/'predictions.csv', index=False, lineterminator='\n')
    for curve, name, yscale in [(inner_curve,'history_inner.csv',inner_preproc['y_scale']),
                                (refit_curve,'history_refit.csv',full_preproc['y_scale'])]:
        frame = pd.DataFrame(curve)
        frame['train_rmse_original'] = np.sqrt(frame['train_mse_scaled_eval'])*yscale
        frame['validation_rmse_original'] = np.sqrt(frame['validation_mse_scaled_eval'])*yscale
        frame.to_csv(directory/name, index=False, lineterminator='\n')
    write_json(directory/'preprocessing.json', {'inner':inner_preproc,'refit':full_preproc})
    write_json(directory/'splits.json', {**split['split_metadata'],
        'train_outcome_indices':train.tolist(), 'inner_train_outcome_indices':itrain.tolist(),
        'inner_validation_outcome_indices':ival.tolist(), 'evaluation_outcome_indices':eval_indices.tolist(),
        'evaluation_phase':job['phase'], 'evaluation_first_day':str(data['calendar'][eval_indices.min()]),
        'evaluation_last_day':str(data['calendar'][eval_indices.max()])})
    write_json(directory/'configuration.json', {'run':job,'training':protocol['training'],
        'architecture':architecture_spec(model),'parameter_counts':count,
        'inner_initialization_seed':job['seed'],'refit_initialization_seed':job['seed']+10000000,
        'source_hashes':data['source_hashes'],'code_signature':_CONTEXT['code_signature'],
        'test_files_opened':0})
    if job['phase'] != 'search':
        torch.save({'state_dict':model.state_dict(),'family':job['model'], 'history_days':history,
                    'width':job['width'],'dropout':protocol['training']['dropout']}, directory/'model.pt')
    artifacts = [p for p in directory.iterdir() if p.is_file() and p.name != 'summary.json']
    summary = {'completed_at_utc':stamp(),'code_signature':_CONTEXT['code_signature'],
               'metrics':record,'artifact_sha256':{p.name:digest(p) for p in artifacts}}
    write_json(complete, summary)
    return clean(summary)


def execute(jobs, protocol, context, workers):
    collected = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                             initializer=worker_init, initargs=(protocol,context)) as pool:
        pending = {pool.submit(run_job, job):job for job in jobs}
        for index, future in enumerate(as_completed(pending),1):
            job = pending[future]
            try:
                result = future.result()
            except Exception:
                event('run_failure',job=job,error=traceback.format_exc())
                raise
            collected.append(result)
            if index == 1 or index % 5 == 0 or index == len(jobs):
                m = result['metrics']
                print(json.dumps({'completed':index,'total':len(jobs),'phase':job['phase'],
                      'last_model':job['model'],'last_D':job['history_days'],
                      'last_rmse':m['rmse'],'epoch':m['best_epoch']},ensure_ascii=False),flush=True)
    return sorted(collected,key=lambda r:r['metrics']['run_id'])


def combine(phase, summaries):
    records = [s['metrics'] for s in summaries]
    frame = pd.DataFrame(records)
    for name, names in [('runs.csv',None),('predictions/'+phase+'.csv',[])]:
        path = OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
        if name == 'runs.csv':
            if path.exists():
                prior = pd.read_csv(path)
                prior = prior.loc[prior.phase != phase]
                frame_all = pd.concat([prior,frame],ignore_index=True)
            else:
                frame_all = frame
            frame_all.sort_values(['phase','model','history_days','config_id','seed']).to_csv(path,index=False,lineterminator='\n')
        else:
            pd.concat([pd.read_csv(OUT/'runs'/r['run_id']/'predictions.csv') for r in records],ignore_index=True).sort_values(
                ['model','history_days','config_id','seed','outcome_day']).to_csv(path,index=False,lineterminator='\n')
    return frame


def configurations(protocol):
    return [{'config_id':f'w{width}_lr{int(lr*1000000)}','width':width,'learning_rate':lr}
            for width in protocol['search']['widths'] for lr in protocol['search']['learning_rates']]


def choose_search(frame, protocol, context):
    required=len(protocol['search']['seeds'])
    grouping=['model','history_days','config_id','width','learning_rate']
    summary=frame.groupby(grouping,as_index=False).agg(selection_rmse=('rmse','mean'),
          selection_rmse_sd=('rmse','std'),n_seeds=('seed','nunique'),parameters=('parameters','first'))
    if len(frame)!=protocol['search']['expected_runs'] or not (summary.n_seeds==required).all():
        raise RuntimeError('Incomplete common search budget')
    best=summary.sort_values(['selection_rmse','parameters','config_id']).groupby(['model','history_days'],as_index=False).first()
    joint=best.groupby('history_days',as_index=False).agg(joint_selection_rmse=('selection_rmse','mean'),families=('model','nunique'))
    assert (joint.families==3).all()
    chosen_D=int(joint.sort_values(['joint_selection_rmse','history_days']).iloc[0].history_days)
    chosen=best.loc[best.history_days==chosen_D].sort_values(['selection_rmse','parameters','model'])
    selected={'selected_at_utc':stamp(),'scope':'development selection, never finaltest ranking',
              'protocol_sha256':context['protocol_sha256'],'code_signature':context['code_signature'],
              'selected_common_history_days':chosen_D,'model_configurations':{row['model']:row for row in chosen.to_dict('records')},
              'operational_baseline_family_from_search':chosen.iloc[0]['model'],
              'history_selection_rule':protocol['search']['global_D_rule'],
              'test_files_opened':0}
    selected_path=OUT/'selected_configuration.json'
    if selected_path.exists():
        previous=json.loads(selected_path.read_text())
        for key in ['protocol_sha256','code_signature','selected_common_history_days','model_configurations','operational_baseline_family_from_search']:
            if previous[key]!=clean(selected[key]):
                raise RuntimeError('Search selection changed; create an explicit protocol version')
        selected=previous
    else:
        write_json(selected_path,selected)
    summary.to_csv(OUT/'search_configurations.csv',index=False,lineterminator='\n')
    joint.to_csv(OUT/'history_selection.csv',index=False,lineterminator='\n')
    event('search_selection_closed',selected=selected)
    print(json.dumps({'selected_D':chosen_D,'baseline_from_search':selected['operational_baseline_family_from_search'],
                      'configurations':selected['model_configurations']},ensure_ascii=False),flush=True)
    return selected


def make_pilot_plan(frame,protocol,selected,context):
    pred=pd.read_csv(OUT/'predictions/pilot.csv')
    first=pred[['outcome_day','y_true','persistence_pred']].drop_duplicates('outcome_day')
    b=float(np.sqrt(np.mean((first.y_true-first.persistence_pred)**2)))
    delta=.05*b
    settings=protocol['power_and_repetitions']
    z=float(norm.ppf(1-settings['planning_alpha']/2)+norm.ppf(settings['power_target']))
    rows=[]
    for family in FAMILIES:
        values=frame.loc[frame.model==family].sort_values('seed').rmse.to_numpy(float)
        if len(values)!=5:raise ValueError('Exactly five independent pilot seeds required')
        gain=b-values;sd=float(np.std(gain,ddof=1))
        required=int(np.ceil(z*z*sd*sd/(delta*delta))) if delta>0 and sd>0 else None
        rows.append({'model':family,'R_pilot':5,'mean_rmse':float(values.mean()),'sd_rmse':float(values.std(ddof=1)),
                     'coefficient_of_variation':float(values.std(ddof=1)/values.mean()) if values.mean()>0 else None,
                     'paired_gain_sd':sd,'required_R_normal_approximation':required})
    if any(r['required_R_normal_approximation'] is None for r in rows):
        requirement=None;number=settings['maximum_benchmark_seeds']
    else:
        requirement=max(r['required_R_normal_approximation'] for r in rows)
        number=min(settings['maximum_benchmark_seeds'],max(settings['minimum_benchmark_seeds'],requirement))
    for row in rows:
        row['MDE_zero_at_chosen_R']=z*row['paired_gain_sd']/np.sqrt(number)
        row['planning_power_at_delta_normal_approximation']=(float(norm.cdf(delta*np.sqrt(number)/row['paired_gain_sd']-
            norm.ppf(1-settings['planning_alpha']/2))+norm.cdf(-delta*np.sqrt(number)/row['paired_gain_sd']-
            norm.ppf(1-settings['planning_alpha']/2))) if row['paired_gain_sd']>0 and delta>0 else None)
    result={'recorded_at_utc':stamp(),'pilot_seeds':protocol['pilot']['seeds'],'period':'2024-07-01..2024-12-31',
            'persistence_RMSE':b,'delta_plan_absolute':delta,'delta_plan_reference':'5percent of calibration persistenceRMSE to detect an effect vszero, not to demonstrate exceeding5percent',
            'alpha_planning':settings['planning_alpha'],'power_target':settings['power_target'],
            'max_required_R':requirement,'chosen_benchmark_R':number,
            'registered_cap':settings['maximum_benchmark_seeds'],'cap_limits_power':requirement is None or requirement>number,
            'models':rows,'configurations_not_reselected_after_pilot':True,
            'pilot_is_initial_variance_estimate_not_guaranteed_final_power':True,'test_files_opened':0}
    prior_pilot_path=OUT/'pilot_summary.json'
    prior_final_path=OUT/'protocol_final.json'
    if prior_pilot_path.exists() and prior_final_path.exists():
        previous=json.loads(prior_pilot_path.read_text())
        final=json.loads(prior_final_path.read_text())
        if previous['chosen_benchmark_R']!=number or final['selected_configuration_sha256']!=digest(OUT/'selected_configuration.json'):
            raise RuntimeError('Pilot sizing changed; preserve original freeze and create a new version')
        event('reused_original_pilot_sizing_and_freeze',R=number)
        return previous,final
    write_json(prior_pilot_path,result)
    final={'frozen_at_utc':stamp(),'experiment_role':'first development benchmark; holdoutremainsreserved',
           'protocol_initial_sha256':context['protocol_sha256'],'code_signature':context['code_signature'],
           'selected_configuration_sha256':digest(OUT/'selected_configuration.json'),
           'pilot_summary_sha256':digest(OUT/'pilot_summary.json'),
           'selected_common_history_days':selected['selected_common_history_days'],
           'model_configurations':selected['model_configurations'],
           'operational_baseline_family_from_search':selected['operational_baseline_family_from_search'],
           'benchmark_R':number,'benchmark_seeds':settings['benchmark_seeds_in_order'][:number],
           'test_evaluation_authorized_by_this_protocol':False,'test_files_opened':0}
    write_json(OUT/'protocol_final.json',final)
    event('development_benchmark_protocol_frozen_before_fresh_seed_runs',R=number,power_cap=result['cap_limits_power'])
    print(json.dumps({'pilot_completed':True,'required_R':requirement,'chosen_R':number,
                      'cap_limits_power':result['cap_limits_power'],'delta_plan':delta}),flush=True)
    return result,final


def selected_jobs(phase, selected, seeds):
    jobs=[]
    for model in FAMILIES:
        cfg=selected['model_configurations'][model]
        for seed in seeds:
            jobs.append({'phase':phase,'model':model,'history_days':int(selected['selected_common_history_days']),
                         'config_id':cfg['config_id'],'width':int(cfg['width']),
                         'learning_rate':float(cfg['learning_rate']),'seed':int(seed)})
    return jobs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['all','search','pilot','development_benchmark','validate'],default='all')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    protocol,context=register_context()
    data=load_development(ROOT)
    audit={}
    for D in protocol['data']['histories']:
        split=split_training(data,D)
        audit[str(D)]={**split['split_metadata'],'n_train':len(split['train_indices']),
                       'n_inner_train':len(split['inner_train_indices']),'n_inner_validation':len(split['inner_validation_indices']),
                       'n_selection':len(split['selection_indices']),'n_calibration':len(split['calibration_indices']),
                       'mase_train_scale':mase_scale(data,split['train_indices'])}
    write_json(OUT/'data_audit.json',{'source_hashes':data['source_hashes'],'calendar_rows':len(data['calendar']),
              'calendar_first':str(data['calendar'][0]),'calendar_last':str(data['calendar'][-1]),
              'configurations':audit,'raw_test_files_opened':0})
    if args.phase=='validate':
        print(json.dumps({'registered':True,'data_audit':audit},ensure_ascii=False),flush=True);return
    if args.phase in ['all','search']:
        jobs=[{'phase':'search','model':model,'history_days':D,**cfg,'seed':seed}
              for model in FAMILIES for D in protocol['search']['histories']
              for cfg in configurations(protocol) for seed in protocol['search']['seeds']]
        event('search_started',planned_runs=len(jobs))
        selected=choose_search(combine('search',execute(jobs,protocol,context,3)),protocol,context)
    else:
        selected=json.loads((OUT/'selected_configuration.json').read_text())
    if args.phase in ['all','pilot']:
        event('pilot_started',seeds=protocol['pilot']['seeds'])
        pilot=combine('pilot',execute(selected_jobs('pilot',selected,protocol['pilot']['seeds']),protocol,context,1))
        _,final=make_pilot_plan(pilot,protocol,selected,context)
    else:
        final=json.loads((OUT/'protocol_final.json').read_text()) if (OUT/'protocol_final.json').exists() else None
    if args.phase in ['all','development_benchmark']:
        if final is None:raise RuntimeError('Pilot sizing and frozen development protocol required')
        event('development_benchmark_started',seeds=final['benchmark_seeds'])
        frame=combine('development_benchmark',execute(selected_jobs('development_benchmark',selected,final['benchmark_seeds']),protocol,context,1))
        event('development_benchmark_completed',runs=len(frame),test_files_opened=0)
    runs=pd.read_csv(OUT/'runs.csv')
    summary={'completed_at_utc':stamp(),'actual_run_counts':runs.groupby('phase').size().to_dict(),
             'completed_project_fit_count':int(2*len(runs)),'model_families':list(FAMILIES),
             'scope':'development only; no heldouttest or finaltest baseline',
             'results_dir':str(OUT.relative_to(ROOT)),'test_files_opened':0}
    write_json(OUT/'execution_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
