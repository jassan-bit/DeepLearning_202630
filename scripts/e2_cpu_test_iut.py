"""Frozen CPU test inference and a new complete CPU-only secondary IUT.

Uses the exact archived eager loader from the earlier CPU diagnostic. Original
GPU parity gates and plan714 remain unexecuted/failed, and A2 is not replaced.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "book_e2/_static/e2_scientific_closure_20261116/cpu_test_iut"
OLD = ROOT / "book_e2/_static/e2_corrections_20261115/cpu_diagnostic"
DATA = ROOT / "book_e2/_static/e2_scientific_closure_20261116/data_audit"
NPZ_SHA = "186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683"
LOADER_SHA = "c0a03171aa63ea7a10f11a1bc9da14e8788fb27938039a6d4cfb37d1d4f2f8c7"
DELTA = .00025772842664981106
VERSION = "e2-cpu-test-iut-v1"
MODELS = ("FCN-1D", "ResNet-1D", "Inception-1D", "LSTM-autoencoder con predictor",
          "TimesNet", "TSMixer", "TiDE", "TimeMixer", "PatchTST", "iTransformer", "MLP16", "LSTM32", "GRU32")
SEEDS = tuple(range(271801,271807))


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def digest(value): return hashlib.sha256(canonical(value)).hexdigest()


def fp32_tensor_sha(array):
    array=np.ascontiguousarray(array)
    require(array.dtype==np.float32,"Normalized input must remain FP32")
    return digest({"dtype":"torch.float32","shape":list(array.shape),"bytes_sha256":hashlib.sha256(array.tobytes()).hexdigest()})


def scalar_sha(array): return hashlib.sha256(np.asarray(array,dtype='<f8').tobytes()).hexdigest()


def write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as f: json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def write_csv(path,rows):
    rows=list(rows)
    with Path(path).open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def checked_report(folder,name):
    report=json.loads((folder/name).read_text())
    require(report['states_unchanged'] and report['original_parity_gate_passed'] is False,"Wrong preserved CPU experiment")
    for relative,record in report['all_output_files'].items():
        require(Path(relative).name==relative and sha(folder/relative)==record['sha256'],"Earlier CPU artifact changed")
    return report


def reuse_admission(arrays,support):
    require(fp32_tensor_sha(arrays['training_X'])==support['train_x_fp32_sha256'],"Training CPU input bytes differ")
    require(fp32_tensor_sha(arrays['calibration_X'])==support['calibration_x_fp32_sha256'],"Calibration CPU input bytes differ")
    require(scalar_sha(arrays['training_y_original'])==support['train_y_original_float64_sha256'],"Training targets differ")
    require(scalar_sha(arrays['calibration_y_original'])==support['calibration_y_original_float64_sha256'],"Calibration targets differ")
    require(arrays['training_dates'].tolist()==support['train_dates'] and arrays['calibration_dates'].tolist()==support['calibration_dates'],"Development calendars differ")
    scaler=support['scaler_record']
    require(float(arrays['training_target_scale'])==scaler['target_scale'] and
            np.array_equal(arrays['training_channel_mean'],scaler['mean']) and
            np.array_equal(arrays['training_channel_std'],scaler['std']),"Training scaler differs")
    require(arrays['evaluation_X'].shape==(608,7,5) and arrays['evaluation_X'].dtype==np.float32
            and np.isfinite(arrays['evaluation_X']).all(),"Invalid admitted CPU test inputs")
    require(arrays['evaluation_y_original'].shape==(608,) and np.isfinite(arrays['evaluation_y_original']).all(),"Invalid test targets")
    return {"training_X_sha256":support['train_x_fp32_sha256'],"calibration_X_sha256":support['calibration_x_fp32_sha256'],
            "test_X_sha256":fp32_tensor_sha(arrays['evaluation_X']),"test_targets_sha256":scalar_sha(arrays['evaluation_y_original']),
            "scaler_record_sha256":digest(scaler),"previous_train655_cal53_reuse_admitted":True}


def sources():
    oldplan=json.loads((OLD/'plan_v3.json').read_text())
    files={"cohorts":DATA/'current_cohort_arrays.npz',"cohort_manifest":DATA/'cohort_array_manifest.json',
           "audit":DATA/'audit_result.json',"old_CPU_plan":OLD/'plan_v3.json',
           "loader":Path(oldplan['parent_runner']['path']),"base_spec":Path(oldplan['base_spec']['path']),
           "old_predictions_report":OLD/'predictions_complete/inference_report.json',
           "old_support":OLD/'predictions_complete/transforms_and_support.json',
           "old_bindings":OLD/'predictions_complete/predictor_bindings.json',
           "old_risks":OLD/'predictions_complete/risks_by_predictor.csv',
           "old_CPU_calibration":OLD/'predictions_complete/predictions_calibration.csv',
           "latency_report":OLD/'latency/latency_report.json',"raw_CPU_latency":OLD/'latency/raw_cpu_latency.csv',
           "kernel":ROOT/'scripts/e2_corrections_inference_statistics.py',
           "calendar_kernel":ROOT/'scripts/e2_effects_rq2_closure.py'}
    require(sha(files['cohorts'])==NPZ_SHA and sha(files['loader'])==LOADER_SHA,"Pinned data/loader changed")
    return {k:{"path":str(p.resolve()),"sha256":sha(p),"bytes":p.stat().st_size} for k,p in files.items()}


def prepare(folder):
    require(not folder.exists(),"Use a fresh plan directory")
    records=sources()
    previous=checked_report(OLD/'predictions_complete','inference_report.json')
    timing=checked_report(OLD/'latency','latency_report.json')
    support=json.loads((OLD/'predictions_complete/transforms_and_support.json').read_text())
    with np.load(DATA/'current_cohort_arrays.npz',allow_pickle=False) as arrays:
        admitted=reuse_admission(arrays,support)
    oldplan=json.loads((OLD/'plan_v3.json').read_text())
    base=json.loads(Path(oldplan['base_spec']['path']).read_text())
    require(tuple(base['model_order'])==MODELS and tuple(base['seeds'])==SEEDS,"Wrong model/seed domain")
    require(previous['predictors_completed']==78 and previous['calibration_comparisons_completed']==78
            and previous['checkpoint_index_sha256']==timing['checkpoint_index_sha256']==base['checkpoint_index_sha256'],"Prior CPU checkpoint domains differ")
    require(previous['runtime']==timing['runtime'],"Earlier development and timing runtime policies differ")
    folder.mkdir(parents=True)
    plan={"version":VERSION,"registered_at_utc":datetime.now(timezone.utc).isoformat(),"script_sha256":sha(__file__),"sources":records,
          "scope":"New posterior exploratory frozen CPU test608 and complete CPU IUT; after previous CPU cost results and necessary-component deduction, before new test forwards/gain calculation",
          "reuse_admission":admitted,"models":list(MODELS),"seeds":list(SEEDS),"delta":DELTA,
          "loader_sha256":LOADER_SHA,"inference_policy":{"device":"cpu","torch":"2.8.0+cpu","dtype":"float32","batch_size":1,"mkldnn_enabled":False,"threads":2,"interop_threads":1,"eval":True,"inference_mode":True},
          "reuse_policy":"Only exact arrays/scalers/checkpoints/states/loader and identical recorded runtime permit reuse of prior CPU train655/cal53; mismatch blocks this reuse plan, not relaxed or filled with GPU data",
          "runtime_to_match":previous['runtime'],"old_CPU_plan_sha256":sha(OLD/'plan_v3.json'),
          "statistics":{"candidates":list(MODELS[:10]),"reference":"GRU32","gain":"mean individual-seed CPU test608 RMSE reference minus candidate",
                        "gain_H0":"gain<=delta","gain_H1":"gain>delta","gain_draws":20000,"gain_block_days":14,"gain_seed":202611163501,
                        "gain_resampling":"Noncircular moving blocks over the608dated observations; common dates across models and fixed six trajectories; RMSE per seed then mean",
                        "cost_H0":"mean CPU latency candidate-minus-GRU>=0","cost_H1":"mean CPU latency candidate-minus-GRU<0",
                        "cost_draws":20000,"cost_seed":202611153101,"cost_blocks":100,
                        "cost_resampling":"Resample whole100paired global CPU measurement blocks, retaining6seeds and allcandidate/reference pairs; existing raw CPU timings with identical loaded weights/backend/inputs policy",
                        "p_formula":"null_star=boundary+(stat_star-stat_observed); one-sided tail with+1 MonteCarlo correction",
                        "IUT":"max(valid p_gain_CPU,valid p_cost_CPU); then Holm10 at0.05","CI":"95percent marginal percentile per component; not simultaneous"},
          "preserved":{"old714_executed":False,"original_GPU_parity_gate_passed":False,"A2_or_RQ1_changed":False,"new_fits":0,"new_HPO":False},
          "limits":["New CPU-only IUT, not retrospective execution of the original failed GPU-parity714plan.","Known inherited chronology; no new blind holdout or prospective confirmation.","Previous cost p-values already known; this is exploratory and methods fixed before CPU test projections.","Calendar bootstrap assumes suitable weak dependence and block adequacy; six fixed trajectories do not represent multiple markets.","Timing blocks require stable/exchangeable session measurements; cost refers to this resident CPU forward policy, not A100 or complete deployment.","Calibration remains calibration, not independent HPO validation; no equivalence or adequacy is inferred from non-rejection."]}
    write_json(folder/'analysis_plan.json',plan)
    return {"plan":str(folder/'analysis_plan.json'),"sha256":sha(folder/'analysis_plan.json'),"reuse_admitted":admitted}


def read_plan(folder,expected):
    path=folder/'analysis_plan.json';require(sha(path)==expected,"Exact approved plan checksum required")
    plan=json.loads(path.read_text());require(plan['script_sha256']==sha(__file__) and plan['sources']==sources(),"Registered code/input changed")
    return plan


def archived_loader(plan):
    path=Path(plan['sources']['loader']['path'])
    spec=importlib.util.spec_from_file_location('_exact_previous_eager_loader',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def cpu_mean_seed_rmse(predictions,y):
    predictions=np.asarray(predictions,dtype=float);y=np.asarray(y,dtype=float)
    require(predictions.ndim==3 and predictions.shape[1:]==(6,len(y)) and np.isfinite(predictions).all() and np.isfinite(y).all(),"Complete six-seed CPU prediction tensors required")
    return np.sqrt(((predictions-y)**2).mean(axis=2)).mean(axis=1)


def infer(folder,expected):
    plan=read_plan(folder,expected);out=folder/'inference';require(not out.exists(),"Preserve all earlier attempts")
    out.mkdir();write_json(out/'run_admission.json',{"plan_sha256":expected,"started_at_utc":datetime.now(timezone.utc).isoformat(),"new_fits_authorized":False})
    try:
        loader=archived_loader(plan);base=json.loads(Path(plan['sources']['base_spec']['path']).read_text());loader.verify_inputs(base)
        torch,_,train,calibration,_,runtime=loader.runtime_and_data(base)
        require(runtime==plan['runtime_to_match'],"CPU runtime differs; cannot reuse earlier train/cal/timing campaign")
        with np.load(DATA/'current_cohort_arrays.npz',allow_pickle=False) as source: arrays={k:source[k].copy() for k in source.files}
        support=json.loads((OLD/'predictions_complete/transforms_and_support.json').read_text());reuse_admission(arrays,support)
        require(loader.tensor_sha(train.x)==plan['reuse_admission']['training_X_sha256'] and loader.tensor_sha(calibration.x)==plan['reuse_admission']['calibration_X_sha256'],"Archived runtime constructs different development inputs")
        transformed=train.scaler.transform(arrays['evaluation_raw']).astype(np.float32)
        require(np.array_equal(transformed,arrays['evaluation_X']),"Test inputs do not derive from the identical frozen scaler")
        old_bindings={(r['model'],r['seed']):r for r in json.loads((OLD/'predictions_complete/predictor_bindings.json').read_text())}
        roots={m:str(Path(base['authors_root'])/m) for m in loader.AUTHOR_MODELS}
        predictions=np.empty((13,6,608),dtype=np.float64);bindings=[]
        stream=(out/'CPU_test_predictions_47424.csv').open('x',newline='',encoding='utf-8')
        writer=csv.DictWriter(stream,fieldnames=['model','seed','date','y_true','prediction','checkpoint_sha256','scaler_sha256','config_sha256']);writer.writeheader()
        log=(out/'progress.jsonl').open('x',encoding='utf-8')
        try:
            from e2_staged_training import state_sha256
            x=torch.from_numpy(arrays['evaluation_X'])
            for row in base['neural_rows']:
                model,before=loader.load_model(row,train,roots,torch);key=row['model'],row['seed'];old=old_bindings[key]
                identity=loader.common_identity(row)
                require(all(old[k]==identity[k] for k in ('checkpoint_sha256','scaler_sha256','config_sha256')) and before==old['state_before_sha256']==old['state_after_sha256'],"Predictor differs from reused CPU train/cal/timing state")
                values=loader.predict(model,x,train.scaler.target_scale,torch,np)
                after=state_sha256(model);require(after==before,"Frozen CPU state mutated")
                predictions[MODELS.index(row['model']),SEEDS.index(row['seed'])]=values
                for i,day in enumerate(arrays['evaluation_dates']): writer.writerow({**identity,'date':str(day),'y_true':float(arrays['evaluation_y_original'][i]),'prediction':values[i]})
                stream.flush();binding={**identity,'state_before_sha256':before,'state_after_sha256':after};bindings.append(binding)
                log.write(json.dumps({**binding,'stage':'CPU_test_predictor_completed'})+'\n');log.flush();del model
        finally:stream.close();log.close()
        require(len(bindings)==78 and np.isfinite(predictions).all(),"Incomplete CPU test registry")
        np.savez_compressed(out/'CPU_test_arrays.npz',predictions=predictions,y_true=arrays['evaluation_y_original'],dates=arrays['evaluation_dates'],models=np.array(MODELS),seeds=np.array(SEEDS))
        write_json(out/'predictor_bindings.json',bindings)
        result={"version":VERSION,"status":"completed_CPU_test78","plan_sha256":expected,"runtime":runtime,"predictors_completed":78,"dates":608,
                "train_calibration_reused_exactly":True,"reused_development_report_sha256":plan['sources']['old_predictions_report']['sha256'],"states_unchanged":True,
                "new_model_fits":0,"old714_executed":False,"original_GPU_parity_gate_passed":False,"A2_or_RQ1_changed":False,
                "files":{p.name:{"sha256":sha(p),"bytes":p.stat().st_size} for p in out.iterdir() if p.is_file()}}
        write_json(out/'inference_report.json',result);return {k:v for k,v in result.items() if k!='files'}
    except Exception as error:
        write_json(out/'failure_report.json',{"version":VERSION,"plan_sha256":expected,"error_type":type(error).__name__,"error":str(error),"traceback":traceback.format_exc(),"partial_outputs_preserved":True,"new_fits":0});raise


def stats_run(folder,expected):
    plan=read_plan(folder,expected);inf=folder/'inference';report=json.loads((inf/'inference_report.json').read_text())
    require(report['status']=='completed_CPU_test78' and report['plan_sha256']==expected and report['states_unchanged'],"CPU test campaign incomplete")
    for name,record in report['files'].items():require(sha(inf/name)==record['sha256'],"CPU test output changed")
    out=folder/'statistics';require(not out.exists(),"Preserve previous statistics");out.mkdir()
    from e2_effects_rq2_closure import moving_block_counts
    from e2_corrections_inference_statistics import latency_matrices,centered_summary,holm
    with np.load(inf/'CPU_test_arrays.npz',allow_pickle=False) as data:pred=data['predictions'].copy();y=data['y_true'].copy()
    require(pred.shape==(13,6,608) and y.shape==(608,),"Wrong CPU test arrays")
    selected=pred[[*range(10),MODELS.index('GRU32')]];loss=(selected-y)**2
    risks=cpu_mean_seed_rmse(selected,y);gains=risks[-1]-risks[:-1]
    design=plan['statistics'];gain_samples=np.concatenate([
        (lambda q:q[:,-1,None]-q[:,:-1])(np.sqrt(np.einsum('bt,mst->bms',counts,loss,optimize=True)/608).mean(axis=2))
        for counts in moving_block_counts(608,design['gain_block_days'],design['gain_draws'],design['gain_seed'])])
    raw=list(csv.DictReader((OLD/'latency/raw_cpu_latency.csv').open()));a,b=latency_matrices(raw)
    difference=a-b;cost=difference.mean(axis=0);rng=np.random.default_rng(design['cost_seed'])
    cost_samples=difference[rng.integers(0,100,size=(design['cost_draws'],100))].mean(axis=1)
    rows=[]
    for i,model in enumerate(MODELS[:10]):
        gain= centered_summary(gains[i],gain_samples[:,i],DELTA)
        timing= centered_summary(cost[i],cost_samples[:,i],0,lower=True)
        rows.append({'candidate':model,'reference':'GRU32','delta_RMSE':DELTA,
                     **{'gain_CPU_'+k:v for k,v in gain.items()},**{'cost_CPU_'+k:v for k,v in timing.items()},
                     'candidate_CPU_RMSE_mean':float(risks[i]),'reference_CPU_RMSE_mean':float(risks[-1]),
                     'p_IUT_CPU':max(gain['p_raw'],timing['p_raw']),'scope':'new complete CPU-only secondary IUT; not original714'})
    for row,adjusted in zip(rows,holm([r['p_IUT_CPU'] for r in rows])):row.update(p_IUT_CPU_Holm10=adjusted,joint_CPU_improvement_established=adjusted<=.05)
    write_csv(out/'CPU_full_IUT_Holm10.csv',rows)
    write_csv(out/'CPU_test_RMSE_by_seed_78.csv',[{'model':model,'seed':seed,'CPU_test_RMSE':float(np.sqrt(np.mean((pred[i,j]-y)**2)))} for i,model in enumerate(MODELS) for j,seed in enumerate(SEEDS)])
    summary={'version':VERSION,'full_CPU_IUT_executed':True,'candidates':10,'joint_CPU_improvements_Holm10':sum(r['joint_CPU_improvement_established'] for r in rows),
             'CPU_test_predictors':78,'test_dates':608,'old714_executed':False,'original_GPU_parity_gate_passed':False,'A2_or_RQ1_changed':False,'new_fits':0,'limits':plan['limits']}
    write_json(out/'summary.json',summary)
    write_json(out/'execution_receipt.json',{'version':VERSION,'completed_at_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':expected,'script_sha256':sha(__file__),'CPU_test_report_sha256':sha(inf/'inference_report.json'),'sources':sources(),'new_fits':0})
    write_json(out/'manifest.json',{'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in out.iterdir() if p.is_file()}})
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=BASE);parser.add_argument('--plan-sha256')
    action=parser.add_mutually_exclusive_group(required=True);action.add_argument('--prepare',action='store_true');action.add_argument('--infer',action='store_true');action.add_argument('--statistics',action='store_true')
    args=parser.parse_args(argv)
    if args.prepare:result=prepare(args.output)
    else:
        require(args.plan_sha256 is not None,'An approved plan checksum is required')
        result=infer(args.output,args.plan_sha256) if args.infer else stats_run(args.output,args.plan_sha256)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
