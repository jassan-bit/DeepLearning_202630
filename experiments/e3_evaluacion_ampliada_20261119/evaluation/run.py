"""Real common-fitting and held-out evaluation audit, with no R3 certification.

This new audit preserves all old forecasts. It trains the 17 historical systems
under one registered fitting policy and includes two analytical floor controls.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

data_api=load(HERE/'data.py','e3_extended_data_api')
classical=load(HERE/'classical.py','e3_extended_classical_api')
fit_api=load(HERE/'fitting.py','e3_extended_fitting_api')
common=fit_api.common
CORE=(*common.NEURAL,*common.CLASSICAL,common.LEGACY)
ANALYTIC=('floor_only','floor_plus_mean_excess')
FAMILY=(*CORE,*ANALYTIC)
DATASETS=('D00','D01','D02','D03')
PANEL_FIELDS=('dataset','split','model','seed','date','target','raw_prediction','prediction','known_floor',
              'scenario','classical_raw_forecast_invariant_by_input_contract')
CURVE_FIELDS=(*PANEL_FIELDS,'train_fraction')
HPO_FIELDS=(*PANEL_FIELDS,'hpo_trial','learning_rate','selected_by_validation','validation_RMSE')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def save(path,value):common.write_json(path,value)
def slug(name):return re.sub('[^a-zA-Z0-9_-]+','_',name)

def protocol(path,freeze_path,source):
    p=json.loads(Path(path).read_bytes());f=json.loads(Path(freeze_path).read_bytes())
    if (p.get('schema_version')!=1 or p.get('kind')!='e3_extended_evaluation_protocol'
            or p.get('family')!=list(FAMILY) or p.get('candidate')!=common.LEGACY
            or p.get('source_commit')!=source):raise ValueError('Protocol family, candidate or code identity differs')
    seeds=p.get('seeds',[])
    if len(seeds)<10 or len(set(seeds))!=len(seeds) or any(type(s)is not int or s<0 for s in seeds):
        raise ValueError('At least ten distinct explicit final neural seeds required')
    if type(p.get('hpo_seed'))is not int or p['hpo_seed'] in seeds:raise ValueError('Separate fixed HPO seed required')
    if set(p.get('datasets',{}))!=set(DATASETS):raise ValueError('All four dataset digests must be fixed')
    if any(re.fullmatch('[0-9a-f]{64}',v or '') is None for v in p['datasets'].values()):raise ValueError('Exact data digests required')
    if p.get('split',{}).get('ratios')!=[.55,.15,.1] or p.get('split',{}).get('purge')!=7:
        raise ValueError('Prespecified split geometry differs')
    t=p.get('training',{})
    if t.get('lr_grid')!=[.0003,.001,.003] or not all(t.get(k,0)>0 for k in ('max_epochs','patience','batch_size','gradient_clip','per_fit_seconds')):
        raise ValueError('Common frozen search grid and positive fitting caps required')
    if p.get('evidence_mode') not in ('prespecified_exploratory_audit','prespecified_exploratory'):
        raise ValueError('This executor supports only the explicitly exploratory audit')
    if (f.get('verified') is not True or f.get('protocol_sha256')!=sha(path)
            or f.get('source_commit')!=source or not f.get('annotated_tag')
            or re.fullmatch('[0-9a-f]{40}',f.get('tag_object_sha','')) is None):
        raise ValueError('Verified annotated-tag receipt with exact code/protocol binding required')
    if f.get('datasets') is not None and f['datasets']!=p['datasets']:raise ValueError('Freeze receipt data binding differs')
    return p,f

def _job(output,dataset,name,kind,seed,request,data,config,device,rows,roots):
    directory=output/'jobs'/dataset/(slug(name)+'__'+kind+'__'+str(seed));directory.mkdir(parents=True,exist_ok=True)
    identity={'source_commit':request['source_commit'],'audit_request_sha256':digest(request),
              'dataset':dataset,'data_sha256':data['raw_sha256'],'kind':kind}
    result_path=directory/'result.json'
    if result_path.exists():
        result=json.loads(result_path.read_bytes())
        if (result.get('audit_request_sha256')!=digest(request)
                or result.get('dataset')!=dataset or result.get('model')!=name
                or result.get('seed')!=seed or result.get('kind')!=kind
                or result.get('status')!='completed'
                or result.get('fit_metadata_sha256')!=sha(directory/'fit_metadata.json')
                or result.get('selected_model_sha256')!=sha(directory/'selected_model.pt')):
            raise ValueError('Completed job identity or frozen bytes differ from this audit')
        model,meta=fit_api.load_selected(name,directory/'selected_model.pt',data,device,rows,roots)
        if meta.get('source_identity') is not None and meta['source_identity']!=identity:
            raise ValueError('Completed job source/data identity differs')
        return model,meta,directory,result
    try:
        model,meta=fit_api.fit(name,seed,data,directory,config,device,rows,roots,identity)
        result={'status':'completed','dataset':dataset,'model':name,'seed':seed,'kind':kind,
                'audit_request_sha256':digest(request),'fit_metadata_sha256':sha(directory/'fit_metadata.json'),
                'validation_projected_RMSE':meta['validation_projected_RMSE'],
                'optimizer_steps':meta['optimizer_steps'],'training_rows':len(data['train_X']),
                'scaler_fit_rows':len(data['scaler']['fit_positions']),
                'selected_model_sha256':meta['selected_model_sha256']}
        save(result_path,result)
        print('JOB_COMPLETED '+directory.relative_to(output).as_posix(),flush=True)
        return model,meta,directory,result
    except Exception as error:
        failure={'status':'failed','dataset':dataset,'model':name,'seed':seed,'kind':kind,
                 'error':type(error).__name__+': '+str(error),'unix_time':time.time()}
        save(directory/'failure.json',failure)
        with (directory/'failures.log').open('a') as stream:stream.write(canonical(failure)+'\n')
        print('JOB_FAILED '+directory.relative_to(output).as_posix()+': '+str(error),flush=True)
        raise

def _profile(data,perturb):
    bars={}
    for i in data['positions']['fit']:
        for day,raw in zip(data['window_dates'][i],data['raw_X'][i]):
            day=str(day)
            if day in bars and not np.array_equal(bars[day],raw):raise ValueError('Overlapping fit bars disagree')
            bars[day]=raw.copy()
    days=sorted(bars)
    units={'D00':'BTC','D01':'shares','D02':'ETH','D03':'contracts'}[data['dataset']]
    return perturb.fit_noise_profile(np.asarray([bars[d] for d in days]),days,data['dataset'],
        partition='fit',fit_end_declared=days[-1],volume_units=units)

def _panel_rows(data,name,seed,raw_predict,perturb,profile,*,noise=True):
    result=[];fit_positions,excluded=classical.fit_diagnostic_mask(data)
    for split in ('fit','validation','calibration','test'):
        positions=fit_positions if split=='fit' else data['positions'][split]
        scenarios=[('unperturbed',None)]+[(f'ohlcv_observation_noise_{s}',s) for s in (1,2,3)] if split=='test' and noise else [('unperturbed',None)]
        for scenario,severity in scenarios:
            raw=data['raw_X'][positions].copy()
            if severity is not None:
                raw=np.stack([perturb.perturb_window(w,data['window_dates'][i],profile,severity=severity,seed=2026111901)[0]
                              for i,w in zip(positions,raw)])
            raw_prediction,point,floor=raw_predict(raw,positions)
            if any(np.asarray(x).shape!=(len(positions),) for x in (raw_prediction,point,floor)):
                raise ValueError('Forecast panel shape differs')
            for i,r,p,f in zip(positions,raw_prediction,point,floor):
                result.append({'dataset':data['dataset'],'split':split,'model':name,'seed':seed,
                    'date':str(data['dates'][i]),'target':float(data['all_y'][i]),'raw_prediction':float(r),
                    'prediction':float(p),'known_floor':float(f),'scenario':scenario,
                    'classical_raw_forecast_invariant_by_input_contract':name in common.CLASSICAL})
    return result,excluded

def _write_csv(path,fields,rows):
    temporary=Path(str(path)+'.tmp')
    with temporary.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    temporary.replace(path)

def _curve_data(data,fraction):
    positions={k:v.copy() for k,v in data['positions'].items()};positions['fit']=positions['fit'][:int(len(positions['fit'])*fraction)]
    return data_api.package_panel(data['raw_X'],data['all_y'],data['all_A'],data['all_U'],data['dates'],positions,
        dataset=data['dataset'],raw_sha=data['raw_sha256'],window_dates=data['window_dates'],
        history=data['history'],geometry=data['geometry'])

def _resources_csv(path,resources):
    rows=[]
    for value in resources:
        basic={key:value.get(key) for key in ('dataset','model','seed','kind','parameters')}
        phases=value.get('phase_resources')
        if phases:
            for name,phase in phases.items():
                rows.append({**basic,'phase':value['kind']+':'+name,
                    'seconds':phase['measured_completed_epoch_seconds'],
                    'optimizer_steps':phase['optimizer_steps'],
                    'peak_cuda_bytes':value.get('peak_GPU_allocated_bytes'),
                    'allocated_seconds':phase['budget_seconds'],
                    'deadline_overrun_seconds':phase['deadline_overrun_seconds'],
                    'stop_reason':phase['stop_reason'],
                    'unclosed_crash_compute_unknown':phase['active_compute_lost_in_an_unclosed_crash_segment_is_unknown']})
        else:
            rows.append({**basic,'phase':value['kind'],'seconds':value.get('training_seconds',0.),
                         'optimizer_steps':0,'peak_cuda_bytes':None,'allocated_seconds':value.get('allocated_seconds'),
                         'deadline_overrun_seconds':value.get('deadline_overrun_seconds'),
                         'stop_reason':value.get('stop_reason','fixed_analytical_or_temporal_control'),
                         'unclosed_crash_compute_unknown':False})
    fields=('dataset','model','seed','kind','phase','parameters','seconds','optimizer_steps','peak_cuda_bytes',
            'allocated_seconds','deadline_overrun_seconds','stop_reason','unclosed_crash_compute_unknown')
    _write_csv(path,fields,rows)

def _model_panel(output,dataset,name,rows,inventory):
    path=output/'panels'/dataset/(slug(name)+'.csv');path.parent.mkdir(parents=True,exist_ok=True)
    _write_csv(path,PANEL_FIELDS,rows)
    relative=path.relative_to(output).as_posix()
    inventory[relative]={'path':relative,'sha256':sha(path),'bytes':path.stat().st_size,'rows':len(rows),
                         'dataset':dataset,'model':name,'observed_seeds':sorted({r['seed'] for r in rows})}

def _panel_manifest(output,request,inventory):
    manifest={'kind':'e3_extended_panel_manifest','schema_version':1,
              'source_commit':request['source_commit'],'protocol_sha256':request['protocol_sha256'],
              'family':list(FAMILY),'seeds':request['protocol']['seeds'],
              'files':[inventory[k] for k in sorted(inventory)],
              'total_rows':sum(v['rows'] for v in inventory.values()),
              'single_flat_CSV_materialized':False}
    save(output/'panel_manifest.json',manifest)
    return manifest

def run(output,source_commit,admission_path,protocol_path,freeze_receipt_path,npz_path,*,
        btc_history_paths=(),device='cuda:0',package_path=None,allow_cpu_for_synthetic_tests=False):
    import torch
    if re.fullmatch('[0-9a-f]{40}',source_commit or '') is None:raise ValueError('Full code commit required')
    if (device!='cuda:0' or not torch.cuda.is_available()) and not allow_cpu_for_synthetic_tests:
        raise ValueError('Verified Colab CUDA runtime required for real new fits')
    p,freeze=protocol(protocol_path,freeze_receipt_path,source_commit)
    compute=None
    if not allow_cpu_for_synthetic_tests:
        controller_protocol=load(HERE.parent/'orchestration/protocol.py','e3_extended_controller_protocol')
        controller_protocol.verify_receipt(protocol_path,freeze_receipt_path,source_commit)
        compute=controller_protocol.verify_compute_profile(p)
    output=Path(output).absolute()
    if any(x.is_symlink() for x in (output,*output.parents)):raise ValueError('Audit output symlink refused')
    output.mkdir(parents=True,exist_ok=True)
    request={'kind':'e3_extended_fitting_audit','source_commit':source_commit,'protocol_sha256':sha(protocol_path),
             'protocol':p,'freeze_receipt_sha256':sha(freeze_receipt_path),'device':device,
             'compute_profile':compute,'synthetic_test_only':allow_cpu_for_synthetic_tests,'family':list(FAMILY)}
    if (output/'request.json').exists():
        if json.loads((output/'request.json').read_bytes())!=request:raise ValueError('Audit request changed; existing run preserved')
    else:save(output/'request.json',request)
    package,inventory,rows,roots=common._package(output,package_path)
    sys.path.insert(0,str(package/'code/scripts'));save(output/'source_inventory.json',inventory)
    perturb=load(HERE.parent/'robustness/perturb.py','e3_extended_fixed_perturb')
    panel_files={};curve_records=[];curve_jobs=[];hpo_records=[];resources=[];failures=[];geometries={};selections={}
    training=p['training'];base_config={k:v for k,v in training.items() if k!='lr_grid'}
    base_config.setdefault('weight_decay',0.)
    for dataset in DATASETS:
        try:
            data=(data_api.btc(npz_path,p,evaluation=True,history_paths=btc_history_paths) if dataset=='D00'
                  else data_api.additional(admission_path,dataset,p,evaluation=True))
            if allow_cpu_for_synthetic_tests and len(data['train_X'])>=300:
                raise ValueError('Synthetic CPU escape hatch refuses production-size panels')
            profile=_profile(data,perturb);save(output/(dataset+'_noise_profile.json'),profile)
            fitpos,excluded=classical.fit_diagnostic_mask(data)
            geometries[dataset]={'raw_sha256':data['raw_sha256'],'clock':data['geometry']['clock'],
                'fit_training_count':len(data['train_X']),'fit_diagnostic_count':len(fitpos),
                'fit_diagnostic_exclusions':excluded,'scaler_fit_positions':data['scaler']['fit_positions'],
                'splits':{split:[str(data['dates'][i]) for i in (fitpos if split=='fit' else pos)] for split,pos in data['positions'].items()},
                'noise_profile_sha256':profile['profile_sha256'],'noise_source':'FIT input bars only'}
            selections[dataset]={}
            for name in (*common.NEURAL,common.LEGACY):
                model_records=[]
                try:
                    trials=[]
                    for index,lr in enumerate(training['lr_grid']):
                        config={**base_config,'learning_rate':lr}
                        model,meta,job,result=_job(output,dataset,name,f'hpo_lr{index}',p['hpo_seed'],request,data,config,device,rows,roots)
                        trials.append((meta['validation_projected_RMSE'],index,lr));resources.append({'dataset':dataset,'model':name,'seed':p['hpo_seed'],'kind':f'hpo_lr{index}',**meta});del model
                    winner=min(trials,key=lambda x:(x[0],x[1]));selections[dataset][name]={'learning_rate':winner[2],'trial_index':winner[1],'all_trial_validation_RMSE':[x[0] for x in trials]}
                    # The validation winner is durably fixed before any held-out
                    # test forecast from an HPO trial is calculated or scored.
                    save(output/'HPOselection.json',selections)
                    pos=data['positions']['test']
                    for score,index,lr in trials:
                        trial_job=output/'jobs'/dataset/(slug(name)+f'__hpo_lr{index}__'+str(p['hpo_seed']))
                        trial_model,_=fit_api.load_selected(name,trial_job/'selected_model.pt',data,device,rows,roots)
                        raw,point,floor=fit_api.predict(trial_model,name,data['raw_X'][pos],data['scaler'],device)
                        for i,rp,pred,fl in zip(pos,raw,point,floor):
                            hpo_records.append({'dataset':dataset,'split':'test','model':name,'seed':p['hpo_seed'],
                                'date':str(data['dates'][i]),'target':float(data['all_y'][i]),'raw_prediction':float(rp),
                                'prediction':float(pred),'known_floor':float(fl),'scenario':'unperturbed',
                                'classical_raw_forecast_invariant_by_input_contract':False,
                                'hpo_trial':index,'learning_rate':lr,'selected_by_validation':index==winner[1],
                                'validation_RMSE':score})
                        del trial_model
                    _write_csv(output/'HPO_sensitivity.csv',HPO_FIELDS,hpo_records)
                    for seed in p['seeds']:
                        config={**base_config,'learning_rate':winner[2]}
                        model,meta,job,result=_job(output,dataset,name,'final',seed,request,data,config,device,rows,roots)
                        resources.append({'dataset':dataset,'model':name,'seed':seed,'kind':'final',**meta})
                        predict=lambda raw,positions:fit_api.predict(model,name,raw,data['scaler'],device)
                        panel,_=_panel_rows(data,name,seed,predict,perturb,profile);model_records.extend(panel)
                        _model_panel(output,dataset,name,model_records,panel_files)
                        _panel_manifest(output,request,panel_files)
                        if name in (common.LEGACY,'TSMixer') and seed in p['seeds'][:3]:
                            for fraction in (.25,.5,1.):
                                if fraction==1.:
                                    cdata=data;cmodel=model;cmeta=meta;cjob=job;cresult=result
                                else:
                                    cdata=_curve_data(data,fraction)
                                    cmodel,cmeta,cjob,cresult=_job(output,dataset,name,'curve_'+str(int(100*fraction)),seed,request,cdata,config,device,rows,roots)
                                    resources.append({'dataset':dataset,'model':name,'seed':seed,'kind':'curve_'+str(fraction),**cmeta})
                                pos=cdata['positions']['test'];r,point,floor=fit_api.predict(cmodel,name,cdata['raw_X'][pos],cdata['scaler'],device)
                                for i,rp,pred,fl in zip(pos,r,point,floor):
                                    curve_records.append({'dataset':dataset,'split':'test','model':name,'seed':seed,'date':str(cdata['dates'][i]),'target':float(cdata['all_y'][i]),'raw_prediction':float(rp),'prediction':float(pred),'known_floor':float(fl),'scenario':'unperturbed','classical_raw_forecast_invariant_by_input_contract':False,'train_fraction':fraction})
                                curve_jobs.append({'dataset':dataset,'model':name,'seed':seed,'train_fraction':fraction,
                                    'training_rows':len(cdata['train_X']),'full_training_rows':len(data['train_X']),
                                    'optimizer_steps':cmeta['optimizer_steps'],'actual_new_fit':fraction<1.,'reused_base_fit':fraction==1.,
                                    'fit_receipt_path':str((cjob/'result.json').relative_to(output)),
                                    'fit_receipt_sha256':sha(cjob/'result.json'),'scaler_fit_rows':len(cdata['scaler']['fit_positions']),
                                    'fit_id':str(cjob.relative_to(output)),'prefix_calendar_span_confounded':True})
                                del cmodel
                        del model
                except Exception as error:failures.append({'dataset':dataset,'model':name,'error':type(error).__name__+': '+str(error)})
            for name in (*common.CLASSICAL,*ANALYTIC):
                try:
                    job=output/'jobs'/dataset/(slug(name)+'__deterministic__-1');job.mkdir(parents=True,exist_ok=True)
                    if name in common.CLASSICAL:
                        values,meta=classical.predictions(name,data,job,source_commit=source_commit,
                            request_sha256=digest(request),per_fit_seconds=training['per_fit_seconds'])
                        def predict(raw,positions):
                            _,A=fit_api.transform(raw,data['scaler']);floor=np.sqrt(A)
                            result=np.asarray([values[str(data['dates'][i])] for i in positions]);return result,np.maximum(result,floor),floor
                    else:
                        meta={'real_data_fits':0,'optimizer_steps':0,
                              'parameters':int(name=='floor_plus_mean_excess'),'training_seconds':0.,
                              'training_statistic_estimated':name=='floor_plus_mean_excess',
                              'statistic':'mean_fit_sigma_minus_fit_floor' if name=='floor_plus_mean_excess' else None}
                        def predict(raw,positions):
                            _,A=fit_api.transform(raw,data['scaler']);floor=np.sqrt(A)
                            result=floor.copy()+(data['scaler']['d_fit'] if name=='floor_plus_mean_excess' else 0.)
                            return result,np.maximum(result,floor),floor
                    panel,_=_panel_rows(data,name,-1,predict,perturb,profile)
                    _model_panel(output,dataset,name,panel,panel_files)
                    _panel_manifest(output,request,panel_files)
                    result={'status':'completed','model':name,'seed':-1,'audit_request_sha256':digest(request),**meta};save(job/'result.json',result)
                    resources.append({'dataset':dataset,'model':name,'seed':-1,'kind':'deterministic',**meta})
                    print('JOB_COMPLETED '+job.relative_to(output).as_posix(),flush=True)
                except Exception as error:failures.append({'dataset':dataset,'model':name,'error':type(error).__name__+': '+str(error)})
        except Exception as error:failures.append({'dataset':dataset,'error':type(error).__name__+': '+str(error)})
        panel_manifest=_panel_manifest(output,request,panel_files)
        _write_csv(output/'sample_curve_predictions.csv',CURVE_FIELDS,curve_records)
        save(output/'sample_curve_jobs.json',curve_jobs);save(output/'resources.json',resources)
        _resources_csv(output/'resources.csv',resources)
        save(output/'dataset_geometry.json',geometries);save(output/'HPOselection.json',selections);save(output/'failures.json',failures)
    manifest={'kind':'e3_extended_audit_run_manifest','source_commit':source_commit,'protocol_sha256':sha(protocol_path),
        'panel_sha256':sha(output/'panel_manifest.json'),'panel_manifest':'panel_manifest.json',
        'datasets':{k:{'raw_sha256':v} for k,v in p['datasets'].items()},
        'protocol_tag':freeze['annotated_tag'],'freeze_receipt_sha256':sha(freeze_receipt_path),
        'family':list(FAMILY),'seeds':p['seeds'],'geometry':geometries,'evidence_mode':p['evidence_mode'],
        'test_targets_used_for_selection':False,'additional_datasets_used_for_mechanism_redesign':False,
        'primary_common_projection':'max(raw_prediction,sqrt(A_from_observed_input))',
        'sample_curve_fit_policy':'prefix fraction with own fit-only scaler; full-fit HPO choice fixed; same validation; fraction1 reuses base fit',
        'HPO_sensitivity_csv':'HPO_sensitivity.csv','HPO_sensitivity_sha256':sha(output/'HPO_sensitivity.csv') if (output/'HPO_sensitivity.csv').exists() else None,
        'HPO_selection_sha256':sha(output/'HPOselection.json'),
        'R1_accredited':False,'R3_accredited':False,'prospective_power_certified':False}
    save(output/'run_manifest.json',manifest)
    attempts=list((output/'jobs').rglob('fit_request.json'));arima=list((output/'jobs').rglob('arima_fit_started.json'))
    receipt={'kind':'e3_extended_evaluation_receipt','status':'completed' if not failures else 'partial_failed',
        'source_commit':source_commit,'protocol_sha256':sha(protocol_path),'dataset_count':len(geometries),
        'additional_datasets_evaluated':sum(k in geometries and not any(f.get('dataset')==k for f in failures) for k in DATASETS[1:]),
        'new_real_data_fit_attempts':0 if allow_cpu_for_synthetic_tests else len(attempts)+len(arima),
        'new_real_data_fits':0 if allow_cpu_for_synthetic_tests else len(attempts)+len(arima),
        'neural_fit_jobs_attempted':len(attempts),'ARIMA_fit_jobs_attempted':len(arima),
        'completed_job_results':len(list((output/'jobs').rglob('result.json'))),
        'panel_rows':panel_manifest['total_rows'],'panel_files':len(panel_manifest['files']),
        'sample_curve_rows':len(curve_records),'failures':failures,
        'R1_accredited':False,'R3_accredited':False,'confirmatory_superiority_claimed':False,
        'scope':'Prespecified common-fitting exploratory audit. Actual held-out results and robustness evidence; no automatic R1/R3 accreditation.'}
    save(output/'evaluation_receipt.json',receipt)
    return receipt

def main():
    a=argparse.ArgumentParser(description=__doc__)
    for name in ('output','source-commit','admission','protocol','freeze-receipt','npz'):a.add_argument('--'+name,required=True)
    a.add_argument('--btc-history',nargs=2,required=True);a.add_argument('--device',default='cuda:0');a.add_argument('--package-path')
    x=a.parse_args();r=run(x.output,x.source_commit,x.admission,x.protocol,x.freeze_receipt,x.npz,
                         btc_history_paths=x.btc_history,device=x.device,package_path=x.package_path)
    print(canonical(r),flush=True)
    if r['status']!='completed':raise SystemExit(1)

if __name__=='__main__':main()
