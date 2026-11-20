"""Predecision-gated zero-shot transfer diagnostic; no fitting or calibration.

The existing pilot's full atom_crps remains the research candidate. All frozen
controls, calendars and failed trajectories are retained. AI assistance declared.
"""
from __future__ import annotations
import argparse,csv,json,re,signal,time,resource
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from replay_common import sha,preprocess
from frozen_models import initialize,load_model,NEURAL,SEEDS


def infer_then_score(model,name,X,A,target_returns,mean6,torch):
    predictions=[];atoms=[];probabilities=[]
    with torch.no_grad():
        for start in range(0,len(X),128):
            x=torch.from_numpy(X[start:start+128]);a=torch.from_numpy(A[start:start+128])
            if name=='additive_positive_head':prediction=model(x,a)
            else:
                outputs=model.forward_components(x,a);prediction=outputs['prediction']
                atoms.append(outputs['atoms']);probabilities.append(outputs['probabilities'])
            predictions.append(prediction.cpu().numpy())
        prediction=np.concatenate(predictions)
        # The future target magnitude is calculated after every forward of this trajectory.
        U=np.abs(target_returns-mean6)
        if not atoms:return prediction,None,U
        from src.e3_overlap_model_v3 import discrete_crps
        scores=discrete_crps(torch.cat(atoms),torch.cat(probabilities),torch.from_numpy(U))
        return prediction,scores.cpu().numpy(),U


def main():
    p=argparse.ArgumentParser();p.add_argument('--admission',required=True);p.add_argument('--decision',required=True)
    p.add_argument('--decision-sha256',required=True);p.add_argument('--registered-commit',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();root=Path(__file__).resolve().parent
    if not re.fullmatch('[0-9a-f]{40}',args.registered_commit):raise ValueError('Registered immutable decision commit required')
    decision_path=Path(args.decision)
    if sha(decision_path)!=args.decision_sha256:raise ValueError('Pre-result decision hash differs')
    decision=json.loads(decision_path.read_text())
    gates=decision['decision']
    if gates['diagnostic_inference_only_authorized'] is not True or any(gates[k] for k in ('confirmation_training_go','new_model_training_allowed','new_HPO_allowed','new_scaler_fitting_allowed','new_calibrator_fitting_allowed','R3_claim_allowed','automatic_variant_promotion_allowed')):raise ValueError('Explicit diagnostic-only decision required')
    if decision['frozen_pilot']['neural_recipes']!=list(NEURAL) or decision['frozen_pilot']['final_seeds']!=list(SEEDS):raise ValueError('Registered recipe or seed list differs')
    code_bindings={name:sha(root/name) for name in ('diagnose_transfer.py','frozen_models.py','replay_common.py')}
    domain_meta=json.loads((root/'reference/additional_domain_admission_metadata.json').read_text())
    for dataset,entry in domain_meta['datasets'].items():
        registered=decision['target_panels']['datasets'][dataset]
        if (registered['normalized_CSV_sha256']!=entry['sha256'] or registered['window_start']+7!=entry['raw_target_start_index'] or registered['window_stop_exclusive']+7!=entry['raw_target_stop_exclusive']):raise ValueError('Registered domain geometry differs')
    admission_path=Path(args.admission)
    if sha(admission_path)!=domain_meta['admission_sha256']:raise ValueError('Original admission receipt differs')
    receipt=json.loads(admission_path.read_text())
    if receipt['datasets_admitted']!=3:raise ValueError('Three physical admissions required')
    calendars=json.loads((root/'reference/additional_clean_test_metadata.json').read_text())
    torch,plugin,original,snapshot=initialize(root)
    if any(original['scaler'][k]!=v for k,v in decision['frozen_pilot']['source_BTC_scaler'].items()):raise ValueError('Registered BTC scaler differs')
    for filename,key in [('plan.json','plan_sha256'),('admission.json','admission_sha256'),('closed_selection.json','closed_selection_sha256'),('development_verdict.json','verdict_sha256')]:
        if sha(root/'reference'/filename)!=decision['frozen_pilot'][key]:raise ValueError('Registered pilot evidence differs: '+filename)
    for binding in decision['frozen_pilot']['checkpoint_bindings']:
        key=binding['pilot_job_receipt_path'].removesuffix('job.json')+'supervised/checkpoint.pt'
        if snapshot['files'][key]['sha256']!=binding['checkpoint_sha256']:raise ValueError('Registered checkpoint differs')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('New output directory required; prior results cannot be overwritten')
    results=[];failures=[];payload_hashes={};started=time.monotonic();started_at=datetime.now(timezone.utc).isoformat()
    before={'status':'gates_passed_before_first_new_forward','started_at_utc':started_at,'registered_decision_commit':args.registered_commit,'decision_sha256':args.decision_sha256,'inference_code_sha256':code_bindings,'new_model_fits':0}
    (out/'before_first_forward.json').write_text(json.dumps(before,indent=2)+'\n')
    def timeout(*unused):raise TimeoutError('Registered per-trajectory wall budget exceeded')
    signal.signal(signal.SIGALRM,timeout)
    # No additional-domain target or model forward is evaluated before all gates above.
    for dataset in ('D01','D02','D03'):
        entry=domain_meta['datasets'][dataset];path=admission_path.parent/entry['path']
        if sha(path)!=entry['sha256']:raise ValueError('Admitted domain bytes differ')
        rows=list(csv.DictReader(path.open(newline='')))
        dates=[r['date'] for r in rows];raw=np.asarray([[float(r[k]) for k in ('open','high','low','close','volume')] for r in rows])
        begin,end=entry['raw_target_start_index'],entry['raw_target_stop_exclusive']
        expected=next(x for x in calendars if x['dataset']==dataset)['clean_test_dates']
        if dates[begin:end]!=expected:raise ValueError('Original clean-test calendar differs')
        targets=np.arange(begin,end);windows=np.stack([raw[t-7:t] for t in targets])
        X,A=preprocess(windows,original['scaler'])
        logclose=np.log(raw[:,3]);returns=np.diff(logclose)
        y=np.asarray([np.std(returns[t-7:t],ddof=0) for t in targets])
        mean6=np.diff(np.log(windows[:,:,3]),axis=1).mean(1)
        floor=np.sqrt(A)
        trajectories=[]
        for name in NEURAL:
            for seed in SEEDS:
                model,meta=load_model(root,name,seed,torch,plugin,original,snapshot)
                job_started=time.monotonic()
                try:
                    if time.monotonic()-started>decision['execution_budget']['aggregate_inference_wall_limit_seconds']:raise TimeoutError('Registered aggregate wall budget exceeded')
                    signal.setitimer(signal.ITIMER_REAL,decision['execution_budget']['per_model_seed_dataset_wall_limit_seconds'])
                    prediction,crps,U=infer_then_score(model,name,X,A,returns[targets-1],mean6,torch)
                    if not np.allclose(y*y,A+(6/49)*U*U,rtol=1e-11,atol=1e-15):raise ValueError('Target overlap identity failed')
                    if prediction.shape!=y.shape or not np.isfinite(prediction).all():raise ValueError('Nonfinite or misaligned complete trajectory')
                    if np.any(prediction+1e-12<floor):raise ValueError('Structural floor violated')
                    if crps is not None and not np.isfinite(crps).all():raise ValueError('Nonfinite CRPS')
                    trajectories.append((name,seed,prediction,crps,meta))
                except Exception as error:
                    failure={'dataset':dataset,'name':name,'seed':seed,'calendar_rows':len(y),'status':'failed_retained_entire_trajectory','error_type':type(error).__name__,'message':str(error),'checkpoint':meta}
                    failures.append(failure);results.append(failure)
                finally:
                    signal.setitimer(signal.ITIMER_REAL,0)
                    meta['inference_wall_seconds']=time.monotonic()-job_started
        for name,prediction in [('floor_only',floor),('floor_plus_mean_excess',floor+original['scaler']['d_fit'])]:
            trajectories.append((name,None,prediction,None,{'fit_domain':'original BTC training only','d_fit':original['scaler']['d_fit'] if name.endswith('mean_excess') else None}))
        for name,seed,prediction,crps,meta in trajectories:
            filename=f'{dataset}__{name}__{seed if seed is not None else "analytic"}.csv';file=out/filename
            with file.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['dataset','name','seed','date','target_sigma7','prediction_sigma7','floor_sigma7','CRPS_U']);w.writeheader()
                for i,date in enumerate(expected):w.writerow({'dataset':dataset,'name':name,'seed':seed,'date':date,'target_sigma7':repr(float(y[i])),'prediction_sigma7':repr(float(prediction[i])),'floor_sigma7':repr(float(floor[i])),'CRPS_U':'' if crps is None else repr(float(crps[i]))})
            payload_hashes[filename]=sha(file)
            results.append({'dataset':dataset,'name':name,'seed':seed,'status':'finite_complete','rows':len(y),'first_date':expected[0],'last_date':expected[-1],'RMSE':float(np.sqrt(np.mean((prediction-y)**2))),'MAE':float(np.mean(np.abs(prediction-y))),'mean_CRPS_U':None if crps is None else float(crps.mean()),'floor_violations':int(np.sum(prediction+1e-12<floor)),'source':meta})
    summary=[]
    for dataset in ('D01','D02','D03'):
        for name in (*NEURAL,'floor_only','floor_plus_mean_excess'):
            entries=[r for r in results if r['dataset']==dataset and r['name']==name];ok=[r for r in entries if r['status']=='finite_complete'];expected=3 if name in NEURAL else 1
            complete=len(ok)==expected
            summary.append({'dataset':dataset,'name':name,'expected_trajectories':expected,'finite_trajectories':len(ok),'complete':complete,'primary_mean_individual_seed_RMSE':float(np.mean([x['RMSE'] for x in ok])) if complete else None,'mean_individual_seed_MAE':float(np.mean([x['MAE'] for x in ok])) if complete else None,'mean_CRPS_U':float(np.mean([x['mean_CRPS_U'] for x in ok])) if complete and all(x['mean_CRPS_U'] is not None for x in ok) else None})
    record={'status':'diagnostic_completed_with_retained_failures' if failures else 'diagnostic_completed','AI_assistance_declared':True,'registered_decision_commit':args.registered_commit,'decision_sha256':args.decision_sha256,'inference_code_sha256':code_bindings,'started_at_utc':started_at,'finished_at_utc':datetime.now(timezone.utc).isoformat(),'wall_seconds':time.monotonic()-started,'data_download_time':'excluded; all input payloads were recovered and hash checked before first forward','process_peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'RSS_scope':'whole Python process including Torch and all loaded data; not incremental model memory','scientific_scope':'zero-shot domain transfer using original BTC preprocessing and frozen pilot checkpoints; not a fit on those domains','candidate':'atom_crps','all_seven_frozen_pilot_recipes_included':True,'expected_trajectories':51,'finite_trajectories':len(results)-len(failures),'failed_trajectories':len(failures),'forecast_rows_finite':sum(r.get('rows',0) for r in results),'summary':summary,'trajectories':results,'failure_policy':'retain entire calendar and failed trajectory; no replacement or row deletion','units_warning':'AAPL raw shares/session and ETH or contracts/UTC day retain original units; frozen BTC log1p volume scaler; clocks and market calendars are dependent and previously exposed','projection_or_refitting':False,'new_model_fits':0,'new_calibrations':0,'HPO_trials':0,'R3_accredited':False,'confirmatory_claim':False,'p_values_computed':False,'pilot_heuristic_unchanged_and_failed':True,'winning_ablation_promoted':False,'payload_sha256':payload_hashes}
    (out/'diagnostic_receipt.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    (out/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    print(json.dumps({k:record[k] for k in ('status','expected_trajectories','finite_trajectories','failed_trajectories','forecast_rows_finite','new_model_fits','R3_accredited')}))

if __name__=='__main__':main()
