"""Registered exploratory frozen-input dependency diagnostic; no model fits.

The two donor interventions preserve seven-day paths within each group and use
only training donors. They break cross-group relations and window continuity:
their effects diagnose reliance under these interventions, not causal shortcuts.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'book_e2/_static/e2_closure_followup_20261116/shortcut'
OLD = ROOT/'book_e2/_static/e2_scientific_closure_20261116/cpu_test_iut'
DATA = ROOT/'book_e2/_static/e2_scientific_closure_20261116/data_audit/current_cohort_arrays.npz'
VERSION = 'e2-training-donor-input-dependency-v1'
MODELS = ('FCN-1D','ResNet-1D','Inception-1D','LSTM-autoencoder con predictor','TimesNet','TSMixer','TiDE','TimeMixer','PatchTST','iTransformer','MLP16','LSTM32','GRU32')
SEEDS = tuple(range(271801,271807))
GROUPS = {'volume': [4], 'OHLC': [0,1,2,3]}


def require(ok,message):
    if not ok: raise ValueError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def csv_write(path,rows):
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def sources():
    files={'input_arrays':DATA,'matched_CPU_plan':OLD/'analysis_plan.json','matched_CPU_report':OLD/'inference/inference_report.json',
           'matched_CPU_arrays':OLD/'inference/CPU_test_arrays.npz','matched_CPU_bindings':OLD/'inference/predictor_bindings.json',
           'CPU_adapter':ROOT/'scripts/e2_cpu_test_iut.py','calendar_bootstrap':ROOT/'scripts/e2_effects_rq2_closure.py'}
    return {k:{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for k,p in files.items()}


def prepare(folder):
    require(not folder.exists(),'Use a new immutable plan directory')
    records=sources()
    report=json.loads((OLD/'inference/inference_report.json').read_text())
    require(report['predictors_completed']==78 and report['states_unchanged'],'Prior CPU inference incomplete')
    folder.mkdir(parents=True)
    plan={'version':VERSION,'registered_at_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(__file__),'sources':records,
          'models':list(MODELS),'seeds':list(SEEDS),'runtime_to_match':report['runtime'],'new_fits':0,
          'conditions':['original','volume_training_donor','OHLC_training_donor'],'groups':GROUPS,
          'input_rule':{'donor_split':'training655 only','donor_rng':'numpy.default_rng','donor_seed':202611166401,
                        'sampling':'608 independent uniform training-window indices with replacement, identical donors across models/seeds and both group interventions',
                        'replacement':'copy complete7day normalizedFP32 channelgroup from donor, leave other group unchanged',
                        'original_check':'regenerate all78x608 originals under matched CPU runtime and require exact elementwise equality with prior CPU arrays; any mismatch stops study',
                        'scaling':'reuse immutable train-only scaler and native model input transformations'},
          'statistics':{'draws':3000,'block_days':14,'rng_seed':202611166402,
                        'unit':'same noncircular moving date blocks for originals and both interventions; all fixed6seeds retained',
                        'metrics':['mean6seed RMSE changed minus original, in sigma7 units','mean absolute prediction change across fixed6seeds and dates'],
                        'intervals':'95percent marginal percentile intervals, conditional on one fixed donor realization, six fitted trajectories, inherited test period',
                        'selection_or_multiple_testing_decision':'none; exploratory descriptive estimates and marginal intervals, no simultaneous family claims'},
          'limits':['Volume distinguished train/test periods in an earlier adversarial diagnostic; this does not establish that volume is spurious or leaking.',
                    'Donor inputs preserve each group seven-day path but can lie off joint OHLC-volume support and violate consistency between overlapping windows.',
                    'Effects identify model dependence under the specified donor intervention, not causal economic effects, expected live missing-feature performance, or benefit of retraining without the feature.',
                    'One donor realization is fixed before inference; confidence intervals do not include donor-assignment uncertainty.',
                    'No model, hyperparameter, scaler, calibrator or deployment choice is selected using these results.',
                    'Known inherited test608, not a fresh blind holdout. Original A2/RQ1 and CPU benchmark remain unchanged.']}
    write(folder/'analysis_plan.json',plan)
    return {'plan':str(folder/'analysis_plan.json'),'plan_sha256':sha(folder/'analysis_plan.json'),'new_model_fits':0}


def read_plan(folder,expected):
    p=folder/'analysis_plan.json';require(sha(p)==expected,'Plan checksum differs')
    plan=json.loads(p.read_text());require(plan['script_sha256']==sha(__file__),'Script changed after plan freeze')
    require(plan['sources']==sources(),'Sources changed after plan freeze')
    return plan


def infer(folder,expected):
    plan=read_plan(folder,expected);out=folder/'inference';require(not out.exists(),'Preserve old inference outputs');out.mkdir()
    started=time.monotonic()
    write(out/'run_admission.json',{'plan_sha256':expected,'started_at_utc':datetime.now(timezone.utc).isoformat(),'new_fits':0})
    try:
        import e2_cpu_test_iut as cpu
        oldplan=json.loads((OLD/'analysis_plan.json').read_text())
        loader=cpu.archived_loader(oldplan)
        base=json.loads(Path(oldplan['sources']['base_spec']['path']).read_text());loader.verify_inputs(base)
        torch,_,train,calibration,_,runtime=loader.runtime_and_data(base)
        require(runtime==plan['runtime_to_match'],'CPU runtime changed from reusable reference')
        from e2_staged_training import state_sha256
        with np.load(DATA,allow_pickle=False) as f: arrays={k:f[k].copy() for k in f.files}
        with np.load(OLD/'inference/CPU_test_arrays.npz',allow_pickle=False) as f: old={k:f[k].copy() for k in f.files}
        require(np.array_equal(train.x.numpy(),arrays['training_X']),'Train-only transformed donor inputs differ')
        require(arrays['evaluation_X'].dtype==np.float32 and arrays['evaluation_X'].shape==(608,7,5),'Test input shape/dtype changed')
        require(np.array_equal(old['y_true'],arrays['evaluation_y_original']) and np.array_equal(old['dates'],arrays['evaluation_dates']),'Target/calendar changed')
        donor=np.random.default_rng(plan['input_rule']['donor_seed']).integers(0,655,size=608)
        conditions={'original':arrays['evaluation_X'].copy()}
        for name,cols in GROUPS.items():
            x=arrays['evaluation_X'].copy();x[:,:,cols]=arrays['training_X'][donor][:,:,cols]
            require(np.array_equal(x[:,:,cols],arrays['training_X'][donor][:,:,cols]),'Donor path differs')
            other=[k for k in range(5) if k not in cols]
            require(np.array_equal(x[:,:,other],arrays['evaluation_X'][:,:,other]),'Nonintervened channels changed')
            conditions[name+'_training_donor']=x
        np.savez_compressed(out/'intervention_inputs.npz',donor_indices=donor,donor_dates=arrays['training_dates'][donor],dates=arrays['evaluation_dates'],**conditions)
        tensors={k:torch.from_numpy(v) for k,v in conditions.items()}
        roots={m:str(Path(base['authors_root'])/m) for m in loader.AUTHOR_MODELS}
        oldbindings={(r['model'],r['seed']):r for r in json.loads((OLD/'inference/predictor_bindings.json').read_text())}
        predictions=np.empty((3,13,6,608),dtype=np.float64);bindings=[]
        with (out/'progress.jsonl').open('x',encoding='utf-8') as log:
            for row in base['neural_rows']:
                model,before=loader.load_model(row,train,roots,torch);identity=loader.common_identity(row);key=row['model'],row['seed'];oldbinding=oldbindings[key]
                require(all(identity[k]==oldbinding[k] for k in ['checkpoint_sha256','scaler_sha256','config_sha256']),'Predictor identity changed')
                require(before==oldbinding['state_before_sha256']==oldbinding['state_after_sha256'],'Predictor state changed')
                m=MODELS.index(row['model']);s=SEEDS.index(row['seed'])
                for c,(name,x) in enumerate(tensors.items()):
                    values=np.asarray(loader.predict(model,x,train.scaler.target_scale,torch,np))
                    if c==0:require(np.array_equal(values,old['predictions'][m,s]),'Original CPU predictions not exactly reproduced: '+str(key))
                    require(state_sha256(model)==before,'Frozen state mutated during donor intervention')
                    predictions[c,m,s]=values
                bindings.append({**identity,'state_before_sha256':before,'state_after_sha256':state_sha256(model),'original_CPU_max_absolute_difference':0.0})
                log.write(json.dumps({'model':row['model'],'seed':row['seed'],'conditions':3,'states_unchanged':True})+'\n');log.flush();del model
        require(len(bindings)==78 and np.isfinite(predictions).all(),'Incomplete donor inference')
        np.savez_compressed(out/'predictions.npz',predictions=predictions,y_true=arrays['evaluation_y_original'],dates=arrays['evaluation_dates'],models=np.array(MODELS),seeds=np.array(SEEDS),conditions=np.array(list(conditions)))
        write(out/'predictor_bindings.json',bindings)
        report={'version':VERSION,'plan_sha256':expected,'status':'completed','predictors':78,'dates':608,'conditions':3,'forwards':142272,'new_fits':0,
                'original_CPU_max_absolute_difference':0.0,'states_unchanged':True,'scaler_unchanged':True,'runtime':runtime,'elapsed_seconds':time.monotonic()-started,
                'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in out.iterdir() if p.is_file()}}
        write(out/'inference_report.json',report)
        return {k:v for k,v in report.items() if k not in ['files','runtime']}
    except Exception as e:
        write(out/'failure_report.json',{'error':str(e),'traceback':traceback.format_exc(),'partial_outputs_preserved':True,'new_fits':0});raise


def statistics(folder,expected):
    plan=read_plan(folder,expected);inf=folder/'inference';report=json.loads((inf/'inference_report.json').read_text())
    require(report['status']=='completed' and report['plan_sha256']==expected and report['states_unchanged'],'Inference incomplete')
    for name,rec in report['files'].items():require(sha(inf/name)==rec['sha256'],'Inference artifact changed')
    out=folder/'statistics';require(not out.exists(),'Preserve previous statistics');out.mkdir()
    with np.load(inf/'predictions.npz',allow_pickle=False) as f:pred=f['predictions'].copy();y=f['y_true'].copy()
    require(pred.shape==(3,13,6,608),'Incomplete intervention predictions')
    from e2_effects_rq2_closure import moving_block_counts
    loss=(pred-y)**2;rmse=np.sqrt(loss.mean(axis=3)).mean(axis=2)
    response=np.abs(pred[1:]-pred[0]).mean(axis=2)
    risks=[];changes=[]
    d=plan['statistics']
    for counts in moving_block_counts(608,d['block_days'],d['draws'],d['rng_seed']):
        q=np.sqrt(np.einsum('bt,cmst->bcms',counts,loss,optimize=True)/608).mean(axis=3)
        risks.append(q-q[:,0,None,:])
        changes.append(np.einsum('bt,cmt->bcm',counts,response,optimize=True)/608)
    bootrisk=np.concatenate(risks);bootchange=np.concatenate(changes)
    rows=[]
    for c,name in enumerate(['volume_training_donor','OHLC_training_donor'],1):
        for m,model in enumerate(MODELS):
            ci=np.quantile(bootrisk[:,c,m],[.025,.975]);pi=np.quantile(bootchange[:,c-1,m],[.025,.975])
            rows.append({'model':model,'condition':name,'original_CPU_RMSE_mean6':float(rmse[0,m]),'intervention_CPU_RMSE_mean6':float(rmse[c,m]),
                         'RMSE_change':float(rmse[c,m]-rmse[0,m]),'RMSE_change_CI95_low':float(ci[0]),'RMSE_change_CI95_high':float(ci[1]),
                         'mean_absolute_prediction_change':float(response[c-1,m].mean()),'prediction_change_CI95_low':float(pi[0]),'prediction_change_CI95_high':float(pi[1]),
                         'seeds':6,'dates':608,'interval_scope':'marginal exploratory conditional on one donor realization'})
    csv_write(out/'input_dependency_by_model.csv',rows)
    summary={'version':VERSION,'plan_sha256':expected,'new_fits':0,'groups':2,'systems':13,'seed_trajectories':78,'conditions':3,'donor_realizations':1,
             'original_CPU_max_absolute_difference':report['original_CPU_max_absolute_difference'],'states_unchanged':True,'A2_or_RQ1_changed':False,
             'shortcuts_or_leakage_proven':False,'diagnostic_executed':True,'limits':plan['limits']}
    write(out/'summary.json',summary)
    write(out/'execution_receipt.json',{'script_sha256':sha(__file__),'plan_sha256':expected,'inference_report_sha256':sha(inf/'inference_report.json'),'completed_at_utc':datetime.now(timezone.utc).isoformat(),'bootstrap_draws':d['draws'],'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in out.iterdir() if p.is_file()}})
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=['prepare','infer','statistics']);p.add_argument('--folder',type=Path,default=BASE);p.add_argument('--plan-sha256');args=p.parse_args()
    if args.operation!='prepare' and not args.plan_sha256:p.error('--plan-sha256 required')
    result=prepare(args.folder) if args.operation=='prepare' else (infer(args.folder,args.plan_sha256) if args.operation=='infer' else statistics(args.folder,args.plan_sha256))
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))

if __name__=='__main__': main()
