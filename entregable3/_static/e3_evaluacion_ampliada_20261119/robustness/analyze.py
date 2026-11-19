"""Analyze actual common-evaluation forecasts; never train or obtain data.

Intervals use clean calibration residuals only. Temporal resampling synchronizes
dates across the whole family and preserves every fitted seed trajectory. The
reported estimand is mean individual RMSE, never RMSE of an ensemble.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
LEGACY = ROOT/'experiments/e3_estadistica_posterior_20261117/analysis.py'
_spec = importlib.util.spec_from_file_location('e3_extended_legacy_statistics',LEGACY)
statistics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(statistics)
CANDIDATE = statistics.ORIGINAL
CLASSICAL = tuple(statistics.CLASSICAL)
ANALYTIC = ('floor_only','floor_plus_mean_excess')
FAMILY = (CANDIDATE,*statistics.NEURAL,*CLASSICAL,*ANALYTIC)
SCENARIOS = ('unperturbed','ohlcv_observation_noise_1',
             'ohlcv_observation_noise_2','ohlcv_observation_noise_3')
REQUIRED = {'dataset','split','model','seed','date','target','prediction'}
VERSION = 'e3-extended-robustness-v1'


def require(value,message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path,raw):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    require(not any(p.is_symlink() for p in (path,*path.parents)),'Output symlinks are not allowed')
    if path.exists():
        require(path.read_bytes()==raw,'Existing result differs; preserve it and use a new output directory')
    else:
        # Publish only complete bytes, without replacing an existing result.
        # An interrupted staging file is not an analysis product or commit marker.
        temporary=None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.partial-'+path.name+'.',delete=False) as stream:
                temporary=Path(stream.name);stream.write(raw);stream.flush();os.fsync(stream.fileno())
            try:os.link(temporary,path)
            except FileExistsError:
                require(path.read_bytes()==raw,'Existing result differs; preserve it and use a new output directory')
        finally:
            if temporary is not None:temporary.unlink(missing_ok=True)


def csv_bytes(rows):
    if not rows:return b''
    stream=io.StringIO(newline='');fields=list(dict.fromkeys(k for row in rows for k in row))
    writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
    for row in rows:
        writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v
                         for k,v in row.items()})
    return stream.getvalue().encode()


def finite(value,name):
    result=float(value);require(math.isfinite(result),name+' must be finite')
    return result


def boolean(value):
    return value is True or str(value).strip().lower() in {'true','1'}


def seed_id(value):
    if str(value) in {'-1','deterministic'}:return '-1'
    number=int(value);require(number>=0,'Neural seed IDs must be nonnegative integers')
    return str(number)


def file_sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def prediction_rows(path):
    """Stream CSV shards independently; the manifest binds every payload hash."""
    path=Path(path)
    require(path.is_file() and not path.is_symlink(),'A regular prediction panel or manifest is required')
    if path.suffix.lower()!='.json':
        proof={'path':str(path),'sha256':file_sha(path),'format':'single_csv','files':1}
        def rows():
            with path.open(encoding='utf-8',newline='') as stream:
                yield from csv.DictReader(stream)
            require(file_sha(path)==proof['sha256'],'Prediction CSV changed while being analyzed')
        return rows(),proof,None
    raw=path.read_bytes();manifest=json.loads(raw)
    require(manifest.get('kind')=='e3_extended_panel_manifest' and isinstance(manifest.get('files'),list)
            and manifest['files'],'A complete sharded panel manifest is required')
    records=manifest['files'];seen=set()
    for record in records:
        name=record['path'];relative=Path(name)
        require(name and not relative.is_absolute() and '..' not in relative.parts and '\\' not in name
                and relative.as_posix()==name and name not in seen and relative.suffix=='.csv',
                'Unsafe or duplicate prediction shard path')
        seen.add(name)
        target=(path.parent/relative).resolve()
        require(target.is_relative_to(path.parent.resolve()) and target.is_file()
                and not (path.parent/relative).is_symlink(),'Prediction shard escapes its manifest directory')
        require(type(record['rows'])is int and record['rows']>0
                and type(record['bytes'])is int and record['bytes']>0,'Invalid shard size or row count')
    proof={'path':str(path),'sha256':digest(raw),'format':'sharded_csv_manifest','files':len(records),
           'declared_rows':sum(r['rows'] for r in records),
           'all_shard_hashes_verified':True,'all_shard_row_counts_verified':True}
    def rows():
        for record in records:
            target=path.parent/record['path']
            require(target.stat().st_size==record['bytes'] and file_sha(target)==record['sha256'],
                    'Prediction shard bytes differ from the manifest: '+record['path'])
            count=0
            with target.open(encoding='utf-8',newline='') as stream:
                reader=csv.DictReader(stream)
                require(REQUIRED<=set(reader.fieldnames or []),'Prediction shard schema is incomplete')
                for row in reader:
                    count+=1;yield row
            require(count==record['rows'] and target.stat().st_size==record['bytes']
                    and file_sha(target)==record['sha256'],'Prediction shard changed or its row count differs')
    return rows(),proof,manifest


def spec(protocol):
    require(isinstance(protocol,dict),'Protocol must be a JSON object')
    synthetic=protocol.get('mode')=='synthetic_contract'
    family=list(protocol.get('family',FAMILY))
    candidate=protocol.get('candidate',CANDIDATE)
    deterministic=list(protocol.get('deterministic_models',[m for m in (*CLASSICAL,*ANALYTIC) if m in family]))
    history_models=list(protocol.get('history_models',[m for m in CLASSICAL if m in family]))
    require(len(family)==len(set(family)) and candidate in family,'Unique complete family and candidate required')
    require(set(deterministic)<=set(family) and set(history_models)<=set(deterministic),'Unknown deterministic/history comparator')
    datasets=protocol.get('datasets',{})
    dataset_ids=list(datasets) if isinstance(datasets,dict) else list(datasets)
    require(dataset_ids and len(dataset_ids)==len(set(dataset_ids)),'Unique datasets required')
    seeds=[seed_id(s) for s in protocol.get('seeds',[])]
    require(seeds and '-1' not in seeds and len(seeds)==len(set(seeds)),'Unique neural seeds required')
    if not synthetic:
        require(set(family)==set(FAMILY) and len(family)==19,'The real audit requires all 17 historical systems and both analytic controls')
        require(set(dataset_ids)=={'D00','D01','D02','D03'},'D00 and all three additional datasets are required')
        require(set(deterministic)==set((*CLASSICAL,*ANALYTIC)) and set(history_models)==set(CLASSICAL),'Deterministic comparator roster changed')
        require(len(seeds)>=10,'The external audit requires at least ten neural seeds')
    cfg=protocol.get('analysis',{})
    blocks=list(cfg.get('block_lengths',[7,14,28]))
    require(blocks==[7,14,28] or synthetic,'Real primary/sensitivity block lengths are fixed at 7,14,28 observations')
    require(all(type(v)is int and v>0 for v in blocks),'Positive integer block lengths required')
    reps=cfg.get('bootstrap_repetitions',9999)
    require(type(reps)is int and reps>=99,'At least 99 declared bootstrap repetitions required')
    alpha=finite(protocol.get('interval_alpha',cfg.get('interval_alpha',.1)),'Interval alpha')
    require(alpha==.1 or synthetic,'The real audit uses nominal 90 percent intervals for every model')
    confidence=finite(cfg.get('confidence_level',.95),'Confidence level')
    require(0<alpha<1 and 0<confidence<1,'Invalid probability level')
    scenarios=list(cfg.get('scenarios',protocol.get('scenarios',SCENARIOS)))
    require(scenarios and scenarios[0]=='unperturbed' and len(scenarios)==len(set(scenarios)),
            'Explicit scenario family must start with unperturbed')
    require(scenarios==list(SCENARIOS) or synthetic,'The real audit requires all three fixed noise levels')
    splits=list(cfg.get('splits',['fit','validation','calibration','test']))
    require(splits==['fit','validation','calibration','test'],'The four chronological splits are required')
    return {'synthetic':synthetic,'family':family,'candidate':candidate,'deterministic':deterministic,'history_models':history_models,
            'datasets':dataset_ids,'seeds':seeds,'alpha':alpha,'confidence':confidence,
            'blocks':blocks,'B':reps,'seed':int(cfg.get('bootstrap_seed',2026111902)),
            'scenarios':scenarios,'splits':splits}


def read_panel(path,protocol):
    """Fail closed on missing/extra rows, target changes or hidden seed selection."""
    cfg=spec(protocol);rows,input_proof,manifest=prediction_rows(path)
    if manifest is not None:
        require(manifest.get('family')==cfg['family'] and [seed_id(s) for s in manifest.get('seeds',[])]==cfg['seeds'],
                'Prediction manifest family or seeds differ from the protocol')
    groups={};raw_groups={};truth={};dates_by_split={};invariant=set()
    row_count=0
    for row in rows:
        row_count+=1
        require(REQUIRED<=set(row),'Prediction CSV schema is incomplete')
        if not cfg['synthetic']:
            require(row.get('raw_prediction') not in (None,'') and row.get('known_floor') not in (None,''),
                    'A real projected audit requires raw forecasts and the observed floor')
        dataset,split,model=row['dataset'],row['split'],row['model']
        scenario=row.get('scenario') or 'unperturbed'
        require(dataset in cfg['datasets'] and split in cfg['splits'] and model in cfg['family'],
                'Unexpected dataset, split or model in prediction panel')
        require(scenario in cfg['scenarios'] and (scenario=='unperturbed' or split=='test'),
                'Noise scenarios belong only to the held-out test split')
        seed=seed_id(row['seed']);expected=['-1'] if model in cfg['deterministic'] else cfg['seeds']
        require(seed in expected,'Unexpected or missing deterministic/neuronal seed contract')
        label=row['date'];require(date.fromisoformat(label).isoformat()==label,'Dates must be canonical ISO labels')
        y,p=finite(row['target'],'Target'),finite(row['prediction'],'Prediction')
        require(y>=0 and p>=0,'Sigma targets and predictions must be nonnegative')
        key=(dataset,split,scenario,model,seed)
        group=groups.setdefault(key,{})
        require(label not in group,'Duplicate model/seed/date row')
        group[label]=(y,p)
        raw_p=finite(row.get('raw_prediction',p),'Raw forecast')
        raw_groups.setdefault(key,{})[label]=raw_p
        if row.get('known_floor') not in (None,''):
            floor=finite(row['known_floor'],'Known floor')
            require(floor>=0 and abs(p-max(raw_p,floor))<=1e-10*max(1.,abs(p)),
                    'Primary prediction differs from the common declared projection')
        target_key=(dataset,split,label)
        if target_key in truth:require(truth[target_key]==y,'Targets differ between models, seeds or scenarios')
        truth[target_key]=y
        dates_by_split.setdefault((dataset,split),set()).add(label)
        if scenario!='unperturbed' and model in cfg['history_models']:
            require(boolean(row.get('source_raw_forecast_invariant',
                                    row.get('classical_raw_forecast_invariant_by_input_contract',False))),
                    'Classical history controls must explicitly declare raw-forecast D7-noise invariance')
            invariant.add((dataset,scenario,model,seed))
    panels={}
    for dataset in cfg['datasets']:
        previous=None;all_dates=set()
        for split in cfg['splits']:
            dates=sorted(dates_by_split.get((dataset,split),set()))
            require(dates,'A complete chronological split is missing')
            if previous is not None:require(previous<dates[0],'Split chronology overlaps or is reversed')
            previous=dates[-1];require(not all_dates.intersection(dates),'Split target IDs overlap');all_dates.update(dates)
            if split=='test':require(len(dates)>=2*max(cfg['blocks']),'Test calendar is too short for declared block sensitivity')
            scenarios=cfg['scenarios'] if split=='test' else ['unperturbed']
            y=np.asarray([truth[(dataset,split,d)] for d in dates],dtype=np.float64)
            for scenario in scenarios:
                for model in cfg['family']:
                    seeds=['-1'] if model in cfg['deterministic'] else cfg['seeds']
                    values=[];raw_values=[]
                    for seed in seeds:
                        group=groups.get((dataset,split,scenario,model,seed),{})
                        require(set(group)==set(dates),'Incomplete model/seed/scenario dates: '+repr((dataset,split,scenario,model,seed)))
                        values.append([group[d][1] for d in dates])
                        raw_values.append([raw_groups[(dataset,split,scenario,model,seed)][d] for d in dates])
                    panels[(dataset,split,scenario,model)]={'dates':dates,'y':y,
                        'p':np.asarray(values,dtype=np.float64),'raw_p':np.asarray(raw_values,dtype=np.float64),'seeds':seeds}
        for scenario in cfg['scenarios'][1:]:
            for model in cfg['history_models']:
                require(np.array_equal(panels[(dataset,'test',scenario,model)]['raw_p'],
                                       panels[(dataset,'test','unperturbed',model)]['raw_p']),
                        'A declared invariant classical raw forecast changed under D7-only noise')
    expected=sum(len(cfg['seeds']) if m not in cfg['deterministic'] else 1 for m in cfg['family'])
    expected*=sum(sum(len(panels[(d,s,'unperturbed',cfg['candidate'])]['dates'])
                      *(len(cfg['scenarios']) if s=='test' else 1) for s in cfg['splits']) for d in cfg['datasets'])
    require(row_count==expected,'Prediction panel has an unexpected number of rows')
    return cfg,panels,{**input_proof,'rows':row_count,
                       'targets_equal_across_models_seeds_scenarios':True,
                       'declared_classical_invariant_groups':len(invariant),
                       'target_support_purge_verified_here':False,
                       'chronology_scope':'Target dates checked here; raw input/label support purge is checked by the fitting executor.'}


def metrics(y,p):
    y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
    e=p-y;den=np.abs(y)+np.abs(p)
    smape=np.divide(2*np.abs(e),den,out=np.zeros_like(e),where=den>0)
    total=float(np.sum((y-y.mean())**2))
    return {'n':len(y),'RMSE':float(np.sqrt(np.mean(e**2))),
            'MAE':float(np.mean(np.abs(e))),'sMAPE_percent':100*float(smape.mean()),
            'R2':None if total==0 else 1-float(np.sum(e**2))/total,
            'R2_status':'undefined_constant_target' if total==0 else 'defined'}


def quantile(calibration_targets,calibration_predictions,alpha):
    y,p=np.asarray(calibration_targets),np.asarray(calibration_predictions)
    require(y.ndim==p.ndim==1 and y.shape==p.shape and y.size>0,'Aligned calibration arrays required')
    require(np.isfinite(y).all() and np.isfinite(p).all(),'Finite calibration residuals required')
    rank=math.ceil((len(y)+1)*(1-alpha))
    require(rank<=len(y),'Calibration split too small for the exact finite-sample order statistic')
    return float(np.sort(np.abs(y-p))[rank-1]),rank


def interval_arrays(y,p,q,alpha):
    lower=np.maximum(0.,p-q);upper=p+q
    covered=((y>=lower)&(y<=upper)).astype(float)
    widths=upper-lower
    score=widths+(2/alpha)*np.maximum(lower-y,0.)+(2/alpha)*np.maximum(y-upper,0.)
    return covered,widths,score


def basic_interval(observed,draws,confidence=.95,family=1):
    tail=(1-confidence)/(2*family);noise=np.asarray(draws)-observed
    return [float(observed-np.quantile(noise,1-tail)),float(observed-np.quantile(noise,tail))]


def chronological_subgroups(dates):
    groups={}
    for i,label in enumerate(dates):
        parsed=date.fromisoformat(label);key=f'{parsed.year}Q{(parsed.month-1)//3+1}'
        groups.setdefault(key,[]).append(i)
    return groups


def bindings(protocol_path,panel_proof,manifest):
    if manifest is None:
        return {'status':'not_supplied','prospective_freeze_verified':False,
                'reason':'No execution manifest was supplied; outputs are an explicitly unbound reanalysis.'}
    data=json.loads(Path(manifest).read_bytes()) if not isinstance(manifest,dict) else manifest
    protocol_raw=Path(protocol_path).read_bytes()
    require(data.get('protocol_sha256')==digest(protocol_raw),'Execution manifest protocol hash mismatch')
    require(data.get('panel_sha256')==panel_proof['sha256'],'Execution manifest panel hash mismatch')
    protocol=json.loads(protocol_raw)
    if protocol.get('source_commit') is not None:
        require(data.get('source_commit')==protocol['source_commit'],'Execution source commit differs from protocol')
    return {'status':'hash_bindings_checked','protocol_sha256':digest(protocol_raw),
            'source_commit':data.get('source_commit'),'protocol_tag':data.get('protocol_tag'),
            'prospective_freeze_verified':False,
            'reason':'Content bindings are checked; a JSON declaration is not an independent timestamp or an R3 verdict.'}


def analyze(panels,cfg):
    per_seed=[];aggregate=[];calibrators=[];intervals=[];subgroups=[];effects=[]
    bootstrap_metrics=[];sensitivities=[]
    family=(len(cfg['family'])-1)*len(cfg['datasets'])
    for dataset_number,dataset in enumerate(cfg['datasets']):
        q_by_model={}
        for model in cfg['family']:
            cal=panels[(dataset,'calibration','unperturbed',model)]
            values=[]
            for k,seed in enumerate(cal['seeds']):
                q,rank=quantile(cal['y'],cal['p'][k],cfg['alpha']);values.append(q)
                calibrators.append({'dataset':dataset,'model':model,'seed':seed,'q':q,'rank':rank,
                    'n_calibration':len(cal['y']),'alpha':cfg['alpha'],
                    'calibration_start':cal['dates'][0],'calibration_end':cal['dates'][-1],
                    'fit_only_on_clean_calibration':True,'recalibrated_for_noise_scenarios':False})
            q_by_model[model]=np.asarray(values)[:,None]
        for split in cfg['splits']:
            scenarios=cfg['scenarios'] if split=='test' else ['unperturbed']
            for scenario in scenarios:
                for model in cfg['family']:
                    panel=panels[(dataset,split,scenario,model)]
                    rows=[]
                    for k,seed in enumerate(panel['seeds']):
                        raw_scores=metrics(panel['y'],panel['raw_p'][k])
                        row={'dataset':dataset,'split':split,'scenario':scenario,'model':model,'seed':seed,
                             **metrics(panel['y'],panel['p'][k]),
                             'raw_RMSE_secondary':raw_scores['RMSE'],'raw_MAE_secondary':raw_scores['MAE']}
                        per_seed.append(row);rows.append(row)
                    aggregate.append({'dataset':dataset,'split':split,'scenario':scenario,'model':model,
                        'seed_count':len(rows),'RMSE_mean_individual':float(np.mean([r['RMSE'] for r in rows])),
                        'RMSE_seed_sample_SD':float(np.std([r['RMSE'] for r in rows],ddof=1)) if len(rows)>1 else None,
                        'MAE_mean_individual':float(np.mean([r['MAE'] for r in rows])),
                        'raw_RMSE_mean_individual_secondary':float(np.mean([r['raw_RMSE_secondary'] for r in rows])),
                        'raw_MAE_mean_individual_secondary':float(np.mean([r['raw_MAE_secondary'] for r in rows])),
                        'sMAPE_percent_mean_individual':float(np.mean([r['sMAPE_percent'] for r in rows])),
                        'R2_mean_individual':None if rows[0]['R2'] is None else float(np.mean([r['R2'] for r in rows])),
                        'dates':len(panel['dates']),'ensembling_performed':False,
                        'classical_history_stressed':False if model in cfg['history_models'] and scenario!='unperturbed' else None})
                    if split!='test':continue
                    covered,width,score=interval_arrays(panel['y'][None,:],panel['p'],q_by_model[model],cfg['alpha'])
                    for k,seed in enumerate(panel['seeds']):
                        intervals.append({'dataset':dataset,'scenario':scenario,'model':model,'seed':seed,
                            'n':covered.shape[1],'nominal_coverage':1-cfg['alpha'],
                            'coverage':float(covered[k].mean()),'mean_width':float(width[k].mean()),
                            'mean_interval_score':float(score[k].mean()),'calibrator_clean_and_frozen':True,
                            'prospective_or_group_coverage_guaranteed':False})
                        for name,idx in chronological_subgroups(panel['dates']).items():
                            subgroups.append({'dataset':dataset,'scenario':scenario,'model':model,'seed':seed,
                                'subgroup':name,'definition':'calendar quarter fixed by dates; not realized outcome',
                                'n':len(idx),'coverage':float(covered[k,idx].mean()),
                                'mean_width':float(width[k,idx].mean()),
                                'mean_interval_score':float(score[k,idx].mean()),
                                'small_subgroup':len(idx)<20,'guarantee_claimed':False})
        squared={};absolute={}
        for scenario in cfg['scenarios']:
            for model in cfg['family']:
                p=panels[(dataset,'test',scenario,model)]
                e=p['p']-p['y'][None,:];key=scenario+'::'+model
                squared[key]=e**2;absolute[key]=np.abs(e)
        for block in cfg['blocks']:
            print(f'Robustness {dataset}: synchronized temporal blocks {block}, draws {cfg["B"]}',flush=True)
            draws,mae_draws=statistics.temporal_draws(squared,absolute,repetitions=cfg['B'],block=block,
                seed=cfg['seed']+1009*dataset_number,chunk_size=64)
            for scenario in cfg['scenarios']:
                candidate=panels[(dataset,'test',scenario,cfg['candidate'])]
                ckey=scenario+'::'+cfg['candidate']
                c_rmse=statistics.mean_individual_rmse(squared[ckey])
                for model in cfg['family']:
                    key=scenario+'::'+model;point=statistics.mean_individual_rmse(squared[key])
                    bootstrap_metrics.append({'dataset':dataset,'scenario':scenario,'model':model,
                        'block_observations':block,'RMSE_mean_individual':point,
                        'RMSE_CI':basic_interval(point,draws[key],cfg['confidence']),
                        'confidence_level':cfg['confidence'],'scope':'Temporal uncertainty conditional on fitted seed trajectories',
                        'seed_uncertainty_resampled':False,'joint_iid_days_seeds_assumed':False})
                    if model==cfg['candidate']:continue
                    gain=point-c_rmse;gain_draws=draws[key]-draws[ckey]
                    rival_seed_rmse=np.sqrt(squared[key].mean(1));candidate_seed_rmse=np.sqrt(squared[ckey].mean(1))
                    seed_gains=rival_seed_rmse-candidate_seed_rmse
                    sd=float(np.std(seed_gains,ddof=1)) if len(seed_gains)>1 else 0.
                    contrast={'dataset':dataset,'scenario':scenario,'rival':model,'candidate':cfg['candidate'],
                        'block_observations':block,'RMSE_gain':gain,
                        'relative_RMSE_reduction_percent':None if point==0 else 100*gain/point,
                        'gain_CI_nominal':basic_interval(gain,gain_draws,cfg['confidence']),
                        'paired_seed_gain_mean':float(seed_gains.mean()),
                        'paired_Cohen_dz':None if sd==0 else float(seed_gains.mean()/sd),
                        'standardized_effect_scope':'Paired initialization variation on this fixed calendar; not independent datasets',
                        'family_size':family*(len(cfg['scenarios'])-1 if scenario!='unperturbed' else 1),
                        'formal_R3_decision':False,'p_value':None}
                    contrast['gain_CI_bonferroni']=basic_interval(gain,gain_draws,cfg['confidence'],contrast['family_size'])
                    contrast['expected_bootstrap_draws_in_each_simultaneous_tail']=cfg['B']*(1-cfg['confidence'])/(2*contrast['family_size'])
                    contrast['simultaneous_tail_below_20_draws']=contrast['expected_bootstrap_draws_in_each_simultaneous_tail']<20
                    effects.append(contrast)
                    if scenario!='unperturbed':sensitivities.append({**contrast,
                        'interpretation':'D7 measurement-corruption stress with common floor from corrupted observations; classical raw history remains clean and its projected output may change'})
    ranks=[]
    for scenario in cfg['scenarios']:
        dataset_ranks={}
        for dataset in cfg['datasets']:
            selected={r['model']:r['RMSE_mean_individual'] for r in aggregate
                      if r['dataset']==dataset and r['split']=='test' and r['scenario']==scenario}
            values=statistics.stats.rankdata([selected[m] for m in cfg['family']],method='average')
            dataset_ranks[dataset]=dict(zip(cfg['family'],values))
            ranks.extend({'scope':'within_dataset','dataset':dataset,'scenario':scenario,'model':model,
                          'rank_RMSE':float(dataset_ranks[dataset][model]),
                          'RMSE_mean_individual':selected[model],'formal_between_dataset_p_value':None}
                         for model in cfg['family'])
        for name,ids in [('all_contexts',cfg['datasets']),
                         ('additional_tasks',[d for d in cfg['datasets'] if d!='D00'])]:
            if not ids:continue
            ranks.extend({'scope':name,'dataset':','.join(ids),'scenario':scenario,'model':model,
                          'rank_RMSE':float(np.mean([dataset_ranks[d][model] for d in ids])),
                          'RMSE_mean_individual':None,'formal_between_dataset_p_value':None,
                          'independent_markets_assumed':False,'pooled_original_unit_RMSE_computed':False}
                         for model in cfg['family'])
    return {'per_seed_metrics':per_seed,'model_metrics':aggregate,'calibrators':calibrators,
            'interval_metrics':intervals,'quarter_metrics':subgroups,'calendar_CIs':bootstrap_metrics,
            'paired_effects':effects,'noise_contrasts':sensitivities,'cross_dataset_ranks':ranks}


def audit_resources(path,cfg):
    if path is None:return {'status':'not_executed','FLOPs':None,'energy_joules':None,'rows':[]}
    rows=list(csv.DictReader(io.StringIO(Path(path).read_text())))
    require(rows,'Empty resource receipt')
    for row in rows:
        require({'dataset','model','seed','phase','seconds'}<=set(row),'Resource schema missing')
        require(finite(row['seconds'],'Measured seconds')>=0,'Negative resource measurement')
        if row.get('peak_cuda_bytes') not in (None,''):
            require(finite(row['peak_cuda_bytes'],'CUDA memory')>=0,'Negative memory')
    observed={(r['dataset'],r['model'],seed_id(r['seed'])) for r in rows if r['phase'] not in {'inference','prediction'}}
    expected={(d,m,s) for d in cfg['datasets'] for m in cfg['family']
              for s in (['-1'] if m in cfg['deterministic'] else cfg['seeds'])}
    missing=sorted(expected-observed)
    return {'status':'all_model_measurement_rows_present' if not missing else 'partial_model_measurements',
            'rows':rows,'missing_fit_groups':missing,'FLOPs':None,'energy_joules':None,
            'all_hardware_comparable_verified':False,'energy_or_FLOPs_inferred_from_parameter_count':False,
            'missing_fields_are_not_zero':True}


def learning_curves(path,index_path,panels,cfg):
    """Admit curves only when each smaller sample actually has its own fit receipt."""
    if path is None or index_path is None:
        return {'status':'not_executed_or_unbound','rows':[],
                'reason':'Actual resized fits, predictions and bound fit receipts are required'},[]
    path,index_path=Path(path),Path(index_path)
    rows=list(csv.DictReader(io.StringIO(path.read_text())));index=json.loads(index_path.read_bytes())
    require(rows and REQUIRED|{'train_fraction'}<=set(rows[0]),'Learning-curve prediction schema missing')
    entries=index['fits'] if isinstance(index,dict) else index
    require(isinstance(entries,list),'Learning-curve fit index must be a list')
    records={}
    for entry in entries:
        key=(entry['dataset'],entry['model'],seed_id(entry['seed']),float(entry['train_fraction']))
        require(key not in records,'Duplicate resized-fit receipt')
        records[key]=entry
    comparator=cfg.get('learning_comparator','TSMixer')
    models=[cfg['candidate'],comparator]
    require(comparator in cfg['family'] and comparator!=cfg['candidate'],'Prespecified learning comparator is missing')
    seeds=cfg['seeds'][:3];fractions=(.25,.5,1.)
    expected={(d,m,s,f) for d in cfg['datasets'] for m in models for s in seeds for f in fractions}
    require(set(records)==expected,'All predeclared dataset/model/seed/fraction fit receipts are required')
    groups={};fit_ids=set()
    for key,entry in records.items():
        fit_id=entry['fit_id'];require(fit_id not in fit_ids,'A full fit cannot be relabelled for several sample sizes');fit_ids.add(fit_id)
        fraction=key[-1];n=int(entry['training_rows']);full=int(entry['full_training_rows'])
        require(full>0 and n>0 and abs(n-full*fraction)<=1,'Recorded resized-fit count differs from its fraction')
        require(int(entry['scaler_fit_rows'])==n,'A resized fit reused a full-data scaler')
        require(int(entry['optimizer_steps'])>0,'A fitted neural learning curve needs actual optimizer steps')
        require(boolean(entry['actual_new_fit'])==(fraction<1)
                and boolean(entry['reused_base_fit'])==(fraction==1),
                'Learning-curve actual-fit/reuse declaration differs from its fraction')
        receipt=(index_path.parent/entry['fit_receipt_path']).resolve()
        require(receipt.is_relative_to(index_path.parent.resolve()) and receipt.is_file()
                and not receipt.is_symlink(),'Learning-curve receipt escapes the index directory')
        require(digest(receipt.read_bytes())==entry['fit_receipt_sha256'],'Resized-fit receipt digest differs')
    for row in rows:
        key=(row['dataset'],row['model'],seed_id(row['seed']),float(row['train_fraction']))
        require(key in expected and row['split']=='test' and (row.get('scenario') or 'unperturbed')=='unperturbed',
                'Unexpected learning-curve model, fraction or split')
        values=groups.setdefault(key,{})
        require(row['date'] not in values,'Duplicate learning-curve date')
        y,p=finite(row['target'],'Curve target'),finite(row['prediction'],'Curve prediction')
        require(y>=0 and p>=0,'Learning-curve sigma values must be nonnegative')
        values[row['date']]=(y,p)
    metric_rows=[]
    for key in sorted(expected):
        dataset,model,seed,fraction=key;base=panels[(dataset,'test','unperturbed',model)]
        group=groups.get(key,{})
        require(set(group)==set(base['dates']),'A resized fit changed the test calendar')
        values=np.asarray([group[d] for d in base['dates']])
        require(np.array_equal(values[:,0],base['y']),'A resized fit changed the test targets')
        metric_rows.append({'dataset':dataset,'model':model,'seed':seed,'train_fraction':fraction,
            'training_rows':records[key]['training_rows'],**metrics(base['y'],values[:,1]),
            'prefix_calendar_span_confounded':True,'comparator_selection':'prespecified_before_new_test_results',
            'actual_new_fit':fraction<1,'base_fit_reused':fraction==1})
    return {'status':'actual_resized_fit_receipts_and_predictions_bound','fit_groups':len(records),
            'new_fit_groups':sum(k[-1]<1 for k in records),'base_reused_groups':sum(k[-1]==1 for k in records),
            'predictions_sha256':digest(path.read_bytes()),'index_sha256':digest(index_path.read_bytes()),
            'prefix_calendar_span_confounded':True,'isolated_random_sample_size_effect_claimed':False,
            'rows':metric_rows},metric_rows


def hpo_sensitivity(path,panels,cfg,protocol):
    """Describe all fixed HPO trials after the validation selection is closed."""
    if path is None:return {'status':'not_executed','rows':[]},[]
    path=Path(path);rows=list(csv.DictReader(io.StringIO(path.read_text())))
    require(rows and REQUIRED|{'hpo_trial','learning_rate','selected_by_validation','validation_RMSE'}<=set(rows[0]),
            'HPO sensitivity schema missing')
    hpo=protocol.get('hpo',{})
    rates=list(hpo.get('learning_rates',[.0003,.001,.003]))
    require(len(rates)==3 and len(set(rates))==3,'Three fixed HPO learning rates required')
    hpo_seed=seed_id(hpo.get('seed',protocol.get('hpo_seed',20261117)))
    models=[m for m in cfg['family'] if m not in cfg['deterministic']]
    expected={(d,m,t) for d in cfg['datasets'] for m in models for t in range(3)}
    groups={};metadata={}
    for row in rows:
        trial=int(row['hpo_trial']);key=(row['dataset'],row['model'],trial)
        require(key in expected and row['split']=='test' and (row.get('scenario') or 'unperturbed')=='unperturbed'
                and seed_id(row['seed'])==hpo_seed,'Unexpected HPO sensitivity group or seed')
        require(finite(row['learning_rate'],'HPO learning rate')==rates[trial],
                'HPO learning rate differs from the prespecified matrix')
        score=finite(row['validation_RMSE'],'HPO validation RMSE');require(score>=0,'Negative validation error')
        meta=(score,boolean(row['selected_by_validation']))
        require(key not in metadata or metadata[key]==meta,'HPO selection metadata changes within a trajectory')
        metadata[key]=meta;group=groups.setdefault(key,{})
        require(row['date'] not in group,'Duplicate HPO sensitivity date')
        y,p=finite(row['target'],'HPO test target'),finite(row['prediction'],'HPO test prediction')
        require(y>=0 and p>=0,'HPO sigma values must be nonnegative');group[row['date']]=(y,p)
    require(set(groups)==expected,'An HPO trial, model or dataset was omitted')
    metrics_rows=[]
    for dataset in cfg['datasets']:
        for model in models:
            scores=[metadata[(dataset,model,t)][0] for t in range(3)]
            chosen=min(range(3),key=lambda t:(scores[t],t))
            require([metadata[(dataset,model,t)][1] for t in range(3)]==[t==chosen for t in range(3)],
                    'HPO sensitivity selection is inconsistent with the fixed validation argmin')
            base=panels[(dataset,'test','unperturbed',model)]
            for trial in range(3):
                group=groups[(dataset,model,trial)]
                require(set(group)==set(base['dates']),'An HPO trial changed the common test calendar')
                values=np.asarray([group[d] for d in base['dates']])
                require(np.array_equal(values[:,0],base['y']),'An HPO trial changed the common test targets')
                metrics_rows.append({'dataset':dataset,'model':model,'seed':hpo_seed,'hpo_trial':trial,
                    'learning_rate':rates[trial],'validation_RMSE':scores[trial],
                    'selected_by_validation':trial==chosen,**metrics(base['y'],values[:,1]),
                    'test_based_reselection_performed':False,'new_fits_in_this_analysis':0,
                    'scope':'One fixed HPO seed, secondary sensitivity; not ten-seed primary-model performance'})
    return {'status':'all_fixed_trials_described_with_validation_argmin_checked','rows':metrics_rows,
            'panel_sha256':file_sha(path),'hpo_seed':hpo_seed,'groups':len(groups),
            'test_based_reselection_performed':False,'selection_timing_independently_attested_here':False,
            'selection_timing_source':'Executor stores its closed validation selection before test forwards'},metrics_rows


def run(panel_csv,protocol_json,output,*,run_manifest=None,resources_csv=None,
        learning_csv=None,learning_index=None,hpo_sensitivity_csv=None):
    protocol_path=Path(protocol_json);protocol=json.loads(protocol_path.read_bytes())
    cfg,panels,input_proof=read_panel(panel_csv,protocol)
    cfg['learning_comparator']=protocol.get('learning_curve_comparator','TSMixer')
    binding=bindings(protocol_path,input_proof,run_manifest)
    auxiliary={name:None if value is None else digest(canonical(value) if isinstance(value,dict)
               else Path(value).read_bytes()) for name,value in
               [('run_manifest',run_manifest),('resources',resources_csv),
                ('learning_panel',learning_csv),('learning_index',learning_index),
                ('hpo_sensitivity',hpo_sensitivity_csv)]}
    output=Path(output)
    if (output/'analysis_receipt.json').is_file():
        previous=json.loads((output/'analysis_receipt.json').read_bytes())
        require(previous['protocol_sha256']==digest(protocol_path.read_bytes())
                and previous['input_panel']['sha256']==input_proof['sha256']
                and previous['analysis_code_sha256']==digest(Path(__file__).read_bytes())
                and previous['reused_statistics_sha256']==digest(LEGACY.read_bytes())
                and previous.get('auxiliary_inputs')==auxiliary,
                'Existing analysis binds different inputs or code; preserve and choose a new output directory')
        require((output/'output_manifest.json').is_file(),'Partial output manifest missing; preserve and choose a new analysis directory')
        manifest_raw=(output/'output_manifest.json').read_bytes()
        require(previous.get('output_manifest_sha256')==digest(manifest_raw),
                'Existing analysis output manifest differs from its final receipt')
        manifest=json.loads(manifest_raw)
        for name,record in manifest['files'].items():
            path=output/name
            require(path.is_file() and path.stat().st_size==record['bytes']
                    and digest(path.read_bytes())==record['sha256'],'Existing analysis output failed its manifest')
        return previous
    tables=analyze(panels,cfg)
    product_names=[]
    for name,rows in tables.items():
        name=name+'.csv';write(output/name,csv_bytes(rows));product_names.append(name)
    resources=audit_resources(resources_csv,cfg)
    if resources['rows']:
        write(output/'resources_supplied.csv',csv_bytes(resources.pop('rows')))
        product_names.append('resources_supplied.csv')
    else:resources.pop('rows')
    curves,curve_rows=learning_curves(learning_csv,learning_index,panels,cfg)
    if curve_rows:
        write(output/'sample_size_metrics.csv',csv_bytes(curve_rows));product_names.append('sample_size_metrics.csv')
    curves.pop('rows',None)
    hpo,hpo_rows=hpo_sensitivity(hpo_sensitivity_csv,panels,cfg,protocol)
    if hpo_rows:
        write(output/'hpo_sensitivity_metrics.csv',csv_bytes(hpo_rows));product_names.append('hpo_sensitivity_metrics.csv')
    hpo.pop('rows',None)
    remaining=[]
    if curves.get('status')!='actual_resized_fit_receipts_and_predictions_bound':
        remaining.append('Actual resized-fit sample-size curves for candidate and prespecified TSMixer comparator')
    if hpo.get('status')!='all_fixed_trials_described_with_validation_argmin_checked':
        remaining.append('Hyperparameter sensitivity across all prespecified validation trials')
    remaining.extend(['Complete comparable resource/energy/FLOPs measurements',
        'Classical raw full-history perturbation differs from D7 corruption and is not claimed here',
        'Temporal test adequacy, prospective power and a full R3 decision remain separate'])
    receipt={'schema_version':1,'kind':VERSION,'completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_commit':protocol.get('source_commit'),'protocol_sha256':digest(protocol_path.read_bytes()),
        'input_panel':input_proof,'bindings':binding,'configuration':cfg,
        'auxiliary_inputs':auxiliary,
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'device':'CPU'},
        'reused_statistics_sha256':digest(LEGACY.read_bytes()),
        'analysis_code_sha256':digest(Path(__file__).read_bytes()),
        'new_model_fits':0,'new_model_inference':0,'calibration_order_statistics_computed':len(tables['calibrators']),
        'calibration_targets_used_for_fit':True,'test_targets_used_for_calibration_or_selection':False,
        'CRPS':None,'PIT':None,'distributional_score_reason':'Point forecasts and a single interval do not define a predictive distribution',
        'sample_size_curves':{**curves,'relabelled_full_fit_curves_accepted':False},
        'hyperparameter_sensitivity':hpo,
        'resources':resources,'R1_accredited':False,'R3_accredited':False,
        'robustness_component_complete':False,
        'remaining_evidence':remaining,
        'limitations':['D00 is a known historical BTC calendar, not a new blind test',
            'D02/D03 share crypto exposure; D03 overlaps BTC underlying and some legacy dates',
            'Time-block CIs condition on fitted seed trajectories; seed SD is a different uncertainty axis',
            'Bootstrap simultaneous interval tails can be sparsely sampled; the report exposes their counts',
            'Calibration order-statistic intervals are nominal; serial dependence and shift prevent a universal coverage guarantee',
            'The floor computed from corrupted inputs need not lower-bound the clean true sigma target; exceedances are not invalid data']}
    # The manifest is deterministic across retries. The receipt is the final
    # commit marker, so a stop after the manifest safely recomputes/reuses tables
    # instead of treating an incomplete directory as completed.
    manifest={name:{'sha256':digest((output/name).read_bytes()),'bytes':(output/name).stat().st_size}
              for name in sorted(product_names)}
    manifest_raw=canonical({'protocol_sha256':receipt['protocol_sha256'],
        'input_panel_sha256':input_proof['sha256'],'auxiliary_inputs':auxiliary,
        'analysis_code_sha256':receipt['analysis_code_sha256'],
        'reused_statistics_sha256':receipt['reused_statistics_sha256'],'files':manifest})
    write(output/'output_manifest.json',manifest_raw)
    receipt['output_manifest_sha256']=digest(manifest_raw)
    write(output/'analysis_receipt.json',canonical(receipt))
    print(json.dumps({'status':'analyzed_actual_saved_panel','datasets':len(cfg['datasets']),
        'models':len(cfg['family']),'scenarios':len(cfg['scenarios']),'new_model_fits':0,'R3_accredited':False}),flush=True)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run-manifest',type=Path)
    parser.add_argument('--resources',type=Path)
    parser.add_argument('--learning-panel',type=Path)
    parser.add_argument('--learning-index',type=Path)
    parser.add_argument('--hpo-sensitivity',type=Path)
    args=parser.parse_args();run(args.panel,args.protocol,args.output,
        run_manifest=args.run_manifest,resources_csv=args.resources,
        learning_csv=args.learning_panel,learning_index=args.learning_index,
        hpo_sensitivity_csv=args.hpo_sensitivity)
