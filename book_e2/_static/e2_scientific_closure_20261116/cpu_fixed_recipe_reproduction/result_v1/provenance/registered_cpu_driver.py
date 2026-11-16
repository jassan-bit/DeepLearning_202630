"""New true-target CPU study: fresh78+3 fits, calibration, freeze, raw608 and A2.

No original CUDA/bitwise reproduction is claimed. No test-only fit flag is used.
The unchanged phase_cpu optimizer and original builders implement actual fits.
--run is blocked until independent QA and the13-model timing admission pass.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import os
from datetime import datetime, timezone
import hashlib
import importlib.util
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'book_e2/_static/e2_scientific_closure_20261116/cpu_fixed_recipe_reproduction'
BASE_DRIVER=ROOT/'scripts/e2_reproduce_from_raw.py'
ENGINE=ROOT/'scripts/e2_corrections_controls.py'
ENGINE_SHA='1d5606526f676e9aef0d83ebb95150ba8ee117d3ac3715de22c81e5a2b63129c'
CONTROLS=ROOT/'book_e2/_static/e2_scientific_closure_20261116/controls'
VERSION='e2-new-true-target-CPU-fixed-recipes-v1'
PHASE='new_true_target_CPU_fixed_recipe_reproduction'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def atomic(path,value):
    temp=path.with_suffix(path.suffix+'.pending')
    with temp.open('w') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    temp.replace(path)

def helper():
    spec=importlib.util.spec_from_file_location('_fixed_cuda_metadata_helpers',BASE_DRIVER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def prepare(args):
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Use a fresh plan directory')
    args.output.mkdir(parents=True,exist_ok=True)
    base=helper();frozen=base.load_plan(args.cuda_plan,args.source_root)
    if sha(ENGINE)!=ENGINE_SHA:raise ValueError('Frozen phase_cpu engine changed')
    plan={'version':VERSION,'registered_at_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(__file__),
          'scope':'New posterior true-target CPU scientific replication, not original CUDA parity/bitwise reproduction',
          'frozen_source_root':str(args.source_root),'base_driver':{'path':str(BASE_DRIVER),'sha256':sha(BASE_DRIVER)},'fixed_recipe_plan':{'path':str(args.cuda_plan),'sha256':sha(args.cuda_plan)},
          'CPU_engine':{'path':str(ENGINE),'sha256':ENGINE_SHA,'callable':'phase_cpu'},
          'accounting_policy':'Reserve and charge every worker separately; sum worker wallseconds including failures. Parent orchestration walltime is not charged again. Global cap is cumulative worker seconds, never parallel elapsed.',
          'budget_source_sha256':sha(ROOT/'scripts/e2_corrections_budget.py'),
          'accounting_root':str(ROOT),'active_ledger_path':str(base.budget_module().LEDGER),'historical_budget_path':str(base.budget_module().BASE),
          'parameter_counts':{r['model']:r['architecture']['actual_parameter_count'] for r in json.loads((args.source_root/'book_e2/_static/e2_flops_final_audit_20261114/frozen_architecture_inputs.json').read_text())},
          'runtime_policy':{'torch':'2.8.0+cpu','numpy':'2.3.5','pandas':'2.2.3','scipy':'1.17.0','statsmodels':'0.14.5','device':'cpu','dtype':'float32','threads':2,'interop_threads':2,'deterministic_algorithms':True,'float32_matmul_precision':'highest','mkldnn_enabled':True,'score_batch_size':1,'parallel_fit_processes':2,'worker_cpu_sets':cpu_sets()},
          'optimization':'Unchanged phase_cpu AdamW(.9,.999,eps1e-8), original decay groups,clip1,seeded batches,cosine/warmup horizon200*batches; exact frozen phase epochs; no early stopping',
          'data_policy':'Only655train true scaled targets used for supervised fitting; SAE auxiliary X targets unchanged.53calibration dates excluded from optimizer.20raw archives numerically admitted only after81weights/scalers/calibrators freeze.',
          'timing_admission_policy':{'required_models':list(base.NEURAL),'control_seed':271801,'replication_seeds':list(base.SEEDS),'safety_factor':1.5,'fixed_shared_overhead_seconds':1200,'formula':'1.5*6*sum13verified_negative_elapsed_seconds+1200','interpretation':'Conservative operational projection from one fixed-seed control per architecture; not a confidence interval or runtime guarantee','require_no_other_open_reservations':True,'resume_shared_allocation':'1200/5 per unfinished shared stage:3classics+score+refine; operational projection only'},
          'QA_required_before_fitting':True,'full_original_campaign_passed':False,'HPO_repeated':False,'NAS':'not_applicable_defined_bodies',
          'new_true_target_fits':78,'classical_reconstructions':3,'old_weights_or_predictions_used':False,
          'remaining_separate_requirements':['original HPO reproduction','new fresh-model latency/accuracy-cost IUT if requested','all model-dependent robustness/explanations/ablations']}
    write(args.output/'analysis_plan.json',plan)
    return args.output/'analysis_plan.json'


def load(args):
    plan=json.loads(args.plan.read_text())
    if plan['version']!=VERSION or plan['script_sha256']!=sha(__file__) or plan['base_driver']['sha256']!=sha(BASE_DRIVER) or plan['CPU_engine']['sha256']!=sha(ENGINE):raise ValueError('CPU adapter/engine/helper changed from registered plan')
    if plan['budget_source_sha256']!=sha(ROOT/'scripts/e2_corrections_budget.py'):raise ValueError('Active budget code changed; register a new plan rather than reset')
    source=Path(plan['fixed_recipe_plan']['path'])
    if sha(source)!=plan['fixed_recipe_plan']['sha256']:raise ValueError('Frozen recipes changed')
    if str(args.source_root)!=plan['frozen_source_root']:raise ValueError('Use the dedicated registered source root')
    base=helper();frozen=base.load_plan(source,args.source_root)
    if str(base.budget_module().LEDGER)!=plan['active_ledger_path'] or str(base.budget_module().BASE)!=plan['historical_budget_path'] or plan['accounting_root']!=str(ROOT):raise ValueError('CPU source snapshot must never supply a cloned budget')
    return plan,base,frozen


def timing_projection(results,models,factor=1.5,overhead=1200):
    if set(results)!=set(models):raise ValueError('All13architecture timing receipts required; no model can be imputed from FCN')
    times=[float(results[m]) for m in models]
    if not all(math.isfinite(v) and v>0 for v in times):raise ValueError('Finite positive elapsed time per architecture required')
    return factor*6*sum(times)+overhead


def admit(args,plan,base,frozen):
    records={};times={};spec=CONTROLS/'specification.json';spec_sha=sha(spec)
    for i,name in enumerate(base.NEURAL,1):
        path=CONTROLS/('negative_'+str(i).zfill(2)+'.json')
        if not path.is_file():raise ValueError('Wait for the complete13-model timing family: '+name)
        value=json.loads(path.read_text())
        if value.get('status')!='completed' or value['model']!=name or value['seed']!=271801 or value['spec_sha256']!=spec_sha:raise ValueError('Timing result identity differs: '+name)
        for kind in ['checkpoint','prediction']:
            if sha(Path(value[kind+'_path']))!=value[kind+'_sha256']:raise ValueError('Completed control artifact changed: '+name)
        chosen=next(j for j in frozen['jobs'] if j['model']==name)
        if value['fixed_recipe']['trial']!=chosen['selection']['winner_trial'] or value['fixed_recipe']['fixed_epochs']!=chosen['selection']['fixed_phase_epochs']:raise ValueError('Timing proxy uses a different recipe: '+name)
        times[name]=float(value['elapsed_seconds']);records[name]={'path':str(path),'sha256':sha(path),'elapsed_seconds':times[name]}
    policy=plan['timing_admission_policy'];projected=timing_projection(times,base.NEURAL,policy['safety_factor'],policy['fixed_shared_overhead_seconds'])
    budget=base.budget_module()
    with budget.locked_ledger(budget.LEDGER) as (rows,_):state=budget.summarize(rows)
    admitted=not state['open_reservations'] and projected<=state['unreserved_seconds'] and projected<=args.cap_seconds
    value={'version':VERSION,'plan_sha256':sha(args.plan),'timings':records,'projected_wall_seconds':projected,'requested_global_cap_seconds':args.cap_seconds,'budget_snapshot':state,'timing_admission_passed':admitted,'new_fits':0,'interpretation':policy['interpretation']}
    write(args.output/'timing_admission.json',value);return value


def configure_cpu(plan):
    import torch
    p=plan['runtime_policy']
    for name in ['numpy','pandas','scipy','statsmodels']:
        if importlib.metadata.version(name)!=p[name]:raise ValueError('NewCPU study dependency differs: '+name)
    if torch.__version__!=p['torch']:raise ValueError('Exact newCPU study runtime required')
    torch.set_num_threads(p['threads']);torch.set_num_interop_threads(p['interop_threads']);torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision(p['float32_matmul_precision']);torch.backends.mkldnn.enabled=p['mkldnn_enabled']
    return torch


def cpu_predict(model,X,*,batch_size=1,device='cpu',deterministic=True):
    import torch
    if device!='cpu' or batch_size!=1 or X.dtype!=torch.float32 or X.ndim!=3 or tuple(X.shape[1:])!=(7,5):raise ValueError('NewCPU inference requires FP32batch1D7x5')
    model.to('cpu').eval();values=[]
    with torch.inference_mode():
        for row in X:
            item=model(row.unsqueeze(0).to('cpu')).reshape(-1)
            if item.shape!=(1,) or not bool(torch.isfinite(item).all()) or item.item()<0:raise ValueError('Invalid newCPU scalar prediction')
            values.append(item.detach().cpu())
    return torch.cat(values)


def lease(args,base):
    budget=base.budget_module()
    with budget.locked_ledger(budget.LEDGER) as (rows,_):state=budget.summarize(rows)
    row=state['open_reservations'].get(args.reservation_id)
    if not row or row['phase']!=PHASE+':'+args.job:raise ValueError('An active parent CPU replication reservation is mandatory')
    identity=json.loads((args.output/'run_identity.json').read_text())
    if identity['plan_sha256']!=sha(args.plan):raise ValueError('Parent CPU study plan differs')


def fit_cpu(args,plan,base,frozen,job):
    torch=configure_cpu(plan);final=base.module(args.source_root,'e2_reduced_final');historical=base.module(args.source_root,'e2_staged_training')
    destination=args.output/job['directory'];destination.mkdir(parents=True,exist_ok=False)
    roots={name:str(args.author_root/name) for name in {r['model'] for r in frozen['author_files']}}
    identity={'source_commit':base.ENGINE,'driver_sha256':sha(__file__),'CPU_study_plan_sha256':sha(args.plan)}
    spec={'data_root':str(args.source_root),'model':job['model'],'seed':job['seed'],'device':'cpu','source_roots':roots,'source_identity':identity,'CPU_study':True}
    started=time.monotonic()
    if job['kind']=='classical':
        spec.update(mode='classical-final',parameters=job['parameters']);write(destination/'worker_spec.json',spec)
        result=final.classical_worker(spec,destination)
    else:
        import e2_corrections_controls as engine
        from e2_reduced_fit import _builder
        from e2_development_data import load_development
        from e2_reduced_final_analysis import residual_calibrator
        panel,admission=load_development(args.source_root,'2024-12-30');train=final.final_training_split(panel);cal=final._calibration_split(panel,train.scaler)
        if len(train.x)!=655 or len(cal.x)!=53:raise ValueError('True-target train/cal support changed')
        selection=job['selection'];trial,epochs=final.validate_selection(job['model'],selection)
        spec.update(selection=selection);write(destination/'worker_spec.json',spec)
        x,y=train.x.clone(),train.y.clone();seed=job['seed'];historical._seed(seed)
        config=historical.TrainConfig(learning_rate=trial['learning_rate'],weight_decay=trial['weight_decay'],batch_size=trial['batch_size'],max_epochs=200,patience=20,seed=seed,warmup_fraction=trial.get('warmup_fraction',0))
        phases={}
        def phase(name,model,xx,yy,scale):
            tick=time.monotonic();value=engine.phase_cpu(model,xx,yy,config,epochs[name],scale);value['wall_seconds']=time.monotonic()-tick
            write(destination/(name+'_phase.json'),value);phases[name]=value
        if job['model']==base.NEURAL[3]:
            from e2_recipe_models import SAEPretraining
            from e2_published_sae import _annotate,_verify_source_and_dropout
            _verify_source_and_dropout(trial['dropout']);manager=SAEPretraining(x,train.signature,accept_auxiliary_scale=True)
            x1,y1=manager.phase1_data();phase('ae1',manager.ae1,x1,y1,1.)
            x2,y2=manager.prepare_phase2();before=historical.state_sha256(manager.ae1.encoder)
            if not torch.equal(y1,y2):raise ValueError('SAE auxiliary targets differ')
            phase('ae2',manager.ae2,x2,y2,1.)
            if historical.state_sha256(manager.ae1.encoder)!=before:raise ValueError('Frozen SAE first encoder changed')
            historical._seed(seed);model=_annotate(manager.transfer_predictor(),dropout=trial['dropout'],seed=seed)
        else:model=_builder(job['model'],trial,seed,roots)
        if sum(p.numel() for p in model.parameters())!=plan['parameter_counts'][job['model']]:raise ValueError('Frozen predictor parameter count changed')
        phase('supervised',model,x,y,train.scaler.target_scale)
        checkpoint=destination/'fit/selected_model.pt';checkpoint.parent.mkdir()
        torch.save({'state_dict':historical.clone_state(model),'scaler':train.scaler.record(),'model':job['model'],'seed':seed,'CPU_study_plan_sha256':sha(args.plan),'fresh_true_target_CPU_fit':True},checkpoint)
        prediction=cpu_predict(model,cal.x).numpy()*train.scaler.target_scale
        write(destination/'scaler.json',train.scaler.record());write(destination/'calibrator.json',residual_calibrator(prediction,cal.y_original,dates=final.CALIBRATION_DATES,predictor_sha256=sha(checkpoint)))
        write(destination/'calibration_predictions.json',{'model':job['model'],'seed':seed,'dates':final.CALIBRATION_DATES,'prediction_original':prediction.tolist(),'target_original':list(cal.y_original),'checkpoint_sha256':sha(checkpoint),'new_true_target_CPU_fit':True})
        write(destination/'development_admission.json',admission)
        result={'status':'completed','model':job['model'],'seed':seed,'checkpoint':'fit/selected_model.pt','checkpoint_sha256':sha(checkpoint),'scaler':'scaler.json','scaler_sha256':sha(destination/'scaler.json'),'calibrator':'calibrator.json','calibrator_sha256':sha(destination/'calibrator.json'),'training_label_end':'2024-10-31','no_refit_after_calibration':True,'phase_records':phases,'true_training_y_scaled_sha256':engine.array_hash(train.y.numpy()),'label_permutation_used':False,'architecture':model.architecture_metadata,'artifacts_sha256':final._artifact_hashes(destination)}
    result.update(new_CPU_fit_scope=True,original_CUDA_reproduction_claimed=False,CPU_study_plan_sha256=sha(args.plan),new_job_wall_seconds=time.monotonic()-started)
    write(destination/'result.json',result)


def score_cpu(args,plan,base,frozen):
    configure_cpu(plan);final=base.module(args.source_root,'e2_reduced_final')
    import e2_published_pilot_training as production
    shutil.copyfile(args.source_root/frozen['pilot_path'],args.output/'pilot_power_MDE.json')
    freeze=final.freeze_final(args.output)
    write(args.output/'CPU_freeze_attestation.json',{'new_CPU_plan_sha256':sha(args.plan),'physical_freeze_sha256':sha(args.output/'final_freeze.json'),'predictors':81,'original_A100_or_bitwise_parity_claimed':False,'numeric_test_admission_within_this_pipeline_after_freeze':True,'historical_test_exposure':True})
    original=production.predict_pilot;production.predict_pilot=cpu_predict
    try:
        final.score_final(args.source_root,args.source_root,args.output,daily_csv=args.output/'raw_derivative_from_original_loader.csv',raw_root=args.raw_root,allow_download=False,device='cpu',source_roots={name:str(args.author_root/name) for name in {r['model'] for r in frozen['author_files']}})
    finally:production.predict_pilot=original


def cpu_sets():
    available=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else list(range(os.cpu_count() or 1))
    chosen=[];seen=set()
    for cpu in available:
        try:
            folder=Path('/sys/devices/system/cpu')/('cpu'+str(cpu))/'topology'
            key=((folder/'physical_package_id').read_text().strip(),(folder/'core_id').read_text().strip())
        except OSError:key=(str(cpu),'unknown')
        if key not in seen:chosen.append(cpu);seen.add(key)
        if len(chosen)==4:break
    if len(chosen)<4:raise ValueError('Two disjoint two-CPU worker affinities require four available cores')
    return [chosen[:2],chosen[2:4]]


def local_inputs(args,frozen):
    errors=[]
    for row in frozen['author_files']:
        path=args.author_root/row['model']/row['path']
        if not path.is_file() or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:errors.append('author:'+row['model']+'/'+row['path'])
    for row in frozen['raw_archives']:
        path=args.raw_root/row['relative_path']
        if not path.is_file() or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:errors.append('raw:'+row['archive'])
    if errors:raise ValueError('Required real local inputs missing/changed: '+str(errors))


def reserved_worker(args,plan,base,job,slot,key,cap):
    budget=base.budget_module();started=time.monotonic();code=1;error=None
    log=args.output/'logs'/(job['id']+'.log')
    command=[str(args.python),str(Path(__file__).resolve()),'--worker','--plan',str(args.plan),'--job',job['id'],'--source-root',str(args.source_root),'--author-root',str(args.author_root),'--raw-root',str(args.raw_root),'--output',str(args.output),'--reservation-id',key,'--cpu-set',','.join(map(str,plan['runtime_policy']['worker_cpu_sets'][slot]))]
    env=dict(os.environ)
    for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[variable]='2'
    try:
        log.parent.mkdir(exist_ok=True)
        with log.open('x') as stream:code=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=max(1,cap-2),env=env).returncode
    except subprocess.TimeoutExpired:code=124;error='Worker timed out; partial artifacts retained'
    except Exception as exc:error=str(exc)
    elapsed=time.monotonic()-started
    budget.finish(key,elapsed,code)
    result={'CPU_study_plan_sha256':sha(args.plan),'id':job['id'],'status':'completed' if code==0 else 'failed','charged_worker_wall_seconds':elapsed,'returncode':code,'reservation_id':key,'reservation_cap_seconds':cap,'cpu_affinity':plan['runtime_policy']['worker_cpu_sets'][slot],'command':command,'log_sha256':sha(log) if log.exists() else None,'error':error,'parent_wall_seconds_added':False}
    write(args.output/'accounting'/(job['id']+'.json'),result)
    return result


def remaining_budget_gate(global_cap,previous_charged,free_balance,remaining_projection):
    remaining=global_cap-previous_charged
    if remaining<0 or remaining>free_balance or remaining_projection>remaining:raise ValueError('Remaining timing projection or unchanged group cap no longer fits the free authorized balance')
    return remaining


def run(args,plan,base,frozen):
    proof=json.loads(args.admission.read_text());qa=json.loads(args.qa.read_text())
    if proof['plan_sha256']!=sha(args.plan) or proof.get('timing_admission_passed') is not True or qa.get('plan_sha256')!=sha(args.plan) or qa.get('approved_for_new_CPU_replication') is not True:raise ValueError('Timing and independent QA approval for this exact CPU plan required')
    for row in proof['timings'].values():
        if sha(Path(row['path']))!=row['sha256']:raise ValueError('Timing evidence changed')
    local_inputs(args,frozen)
    budget=base.budget_module()
    with budget.locked_ledger(budget.LEDGER) as (rows,_):state=budget.summarize(rows)
    if state['open_reservations']:raise ValueError('Settle existing reservations before CPU study execution/resume')
    if args.resume:
        prior_identity=json.loads((args.output/'run_identity.json').read_text())
        if prior_identity['plan_sha256']!=sha(args.plan) or prior_identity['global_cap_sum_worker_seconds']!=args.cap_seconds:raise ValueError('Resume plan/global group cap differs')
    elif args.output.exists() and any(args.output.iterdir()):raise ValueError('Fresh output required')
    args.output.mkdir(parents=True,exist_ok=True)
    if not args.resume:write(args.output/'run_identity.json',{'plan_sha256':sha(args.plan),'CPU_study':True,'original_CUDA_parity':False,'old_weights_or_predictions_used':False,'global_cap_sum_worker_seconds':args.cap_seconds,'accounting':'sum of individually reserved worker wallseconds; parent not added'})
    status={'version':VERSION,'status':'running','jobs':[],'original_CUDA_parity':False,'full_original_campaign_passed':False,'HPO_repeated':False,'global_cap_sum_worker_seconds':args.cap_seconds}
    prior_receipts=[json.loads(p.read_text()) for p in sorted((args.output/'accounting').glob('*.json'))]
    with budget.locked_ledger(budget.LEDGER) as (ledger_rows,_):
        finishes={r['reservation_id']:r for r in ledger_rows if r['event']=='finish'}
    for old in prior_receipts:
        matched=finishes.get(old['reservation_id'])
        if old['CPU_study_plan_sha256']!=sha(args.plan) or old['returncode'] or not matched or matched['elapsed_seconds']!=old['charged_worker_wall_seconds'] or matched['returncode']!=old['returncode']:raise ValueError('Prior worker receipt/actual ledger charge differs or requires reconciliation')
    charged=sum(r['charged_worker_wall_seconds'] for r in prior_receipts);pending=[]
    for job in frozen['jobs']:
        directory=args.output/job['directory'];receipt=args.output/'accounting'/(job['id']+'.json')
        if directory.exists() or receipt.exists():
            if not base.completed_job(directory) or not receipt.is_file():raise ValueError('Partial/unaccounted job requires reconciliation, not automatic retry')
            saved=json.loads((directory/'result.json').read_text())
            if saved['CPU_study_plan_sha256']!=sha(args.plan):raise ValueError('Resume would reuse a different CPU study')
            old=json.loads(receipt.read_text())
            if old['returncode'] or old['CPU_study_plan_sha256']!=sha(args.plan):raise ValueError('Previous CPU attempt requires reconciliation')
            status['jobs'].append(old)
        else:pending.append(job)
    completed_ids={r['id'] for r in prior_receipts}
    shared_remaining=sum(j['kind']=='classical' for j in pending)+sum(name not in completed_ids for name in ['CPU-freeze-raw-score','CPU-primary20000'])
    remaining_projection=1.5*sum(proof['timings'][j['model']]['elapsed_seconds'] for j in pending if j['kind']=='neural')+1200.*shared_remaining/5
    group_remaining=remaining_budget_gate(args.cap_seconds,charged,state['unreserved_seconds'],remaining_projection)
    status['prior_worker_seconds']=charged;status['remaining_group_cap_seconds']=group_remaining;status['remaining_projected_seconds']=remaining_projection
    failed=False
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            active={};slots=[0,1]
            while pending or active:
                while pending and slots and not failed:
                    job=pending[0]
                    cap=300. if job['kind']=='classical' else math.ceil(1.5*proof['timings'][job['model']]['elapsed_seconds']+10.)
                    held=sum(row[2] for row in active.values())
                    if charged+held+cap>args.cap_seconds:raise ValueError('Insufficient cumulative worker cap for the next complete reserved job')
                    key=budget.reserve(PHASE+':'+job['id'],cap);slot=slots.pop(0);pending.pop(0)
                    future=pool.submit(reserved_worker,args,plan,base,job,slot,key,cap);active[future]=(job,slot,cap)
                if not active:break
                done,_=wait(active,return_when=FIRST_COMPLETED)
                for future in done:
                    job,slot,cap=active.pop(future);slots.append(slot);receipt=future.result();charged+=receipt['charged_worker_wall_seconds'];status['jobs'].append(receipt)
                    failed=failed or receipt['returncode']!=0;status['charged_worker_seconds']=charged;atomic(args.output/'status.json',status)
                if failed and not active:break
        if failed:raise ValueError('CPU fit worker failed; retain partial results and all actual charges')
        for job in [{'id':'CPU-freeze-raw-score','kind':'score'},{'id':'CPU-primary20000','kind':'refine'}]:
            previous=args.output/'accounting'/(job['id']+'.json')
            if previous.exists():
                saved=json.loads(previous.read_text())
                if saved['returncode'] or saved['CPU_study_plan_sha256']!=sha(args.plan):raise ValueError('Previous analysis needs reconciliation')
                if job['kind']=='score':
                    evidence=json.loads((args.output/'score/result.json').read_text())
                    for relative,expected in evidence['artifacts_sha256'].items():
                        if sha(args.output/'score'/relative)!=expected:raise ValueError('Completed score changed')
                else:
                    evidence=json.loads((args.output/'score/primary20000_completion.json').read_text())
                    if evidence['output_sha256']!=sha(args.output/'score/benchmark_analysis_20000.json'):raise ValueError('Completed primary analysis changed')
                status['jobs'].append(saved);continue
            cap=min(600.,args.cap_seconds-charged)
            if cap<30:raise ValueError('Insufficient cumulative cap for analysis')
            key=budget.reserve(PHASE+':'+job['id'],cap);receipt=reserved_worker(args,plan,base,job,0,key,cap);charged+=receipt['charged_worker_wall_seconds'];status['jobs'].append(receipt)
            if receipt['returncode']:raise ValueError('CPU scoring/analysis failed')
        status['status']='completed_new_CPU_fixed_recipe_fit_cal_freeze_raw_test_A2_scope';code=0
    except Exception as error:status.update(status='partial_stopped',error=str(error));code=2
    with budget.locked_ledger(budget.LEDGER) as (rows,_):status['budget']=budget.summarize(rows)
    receipts=[json.loads(p.read_text()) for p in sorted((args.output/'accounting').glob('*.json'))]
    if any(r['CPU_study_plan_sha256']!=sha(args.plan) for r in receipts):raise ValueError('Mixed study accounting receipts')
    status['charged_worker_seconds']=sum(r['charged_worker_wall_seconds'] for r in receipts);status['jobs']=receipts;status['parent_wall_seconds_added']=False;atomic(args.output/'status.json',status)
    return code


def main():
    p=argparse.ArgumentParser(description=__doc__);m=p.add_mutually_exclusive_group(required=True)
    for name in ['prepare','admit','run','worker']:m.add_argument('--'+name,action='store_true')
    p.add_argument('--plan',type=Path,default=BASE/'plan_v1/analysis_plan.json');p.add_argument('--cuda-plan',type=Path,default=ROOT/'book_e2/_static/e2_scientific_closure_20261116/reproduction/v2/plan.json')
    p.add_argument('--source-root',type=Path,default=ROOT);p.add_argument('--author-root',type=Path,default=Path('/workspace/work/e2architecture_sources'));p.add_argument('--raw-root',type=Path,default=ROOT/'work/e2-scientific-market/raw')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--python',type=Path,default=Path(sys.executable));p.add_argument('--admission',type=Path);p.add_argument('--qa',type=Path);p.add_argument('--cap-seconds',type=float,default=10000)
    p.add_argument('--resume',action='store_true');p.add_argument('--job');p.add_argument('--cpu-set');p.add_argument('--reservation-id',help=argparse.SUPPRESS);args=p.parse_args()
    for name in ['plan','cuda_plan','source_root','author_root','raw_root','output','python'] :setattr(args,name,getattr(args,name).absolute())
    if not math.isfinite(args.cap_seconds) or args.cap_seconds<=0:p.error('Finite positive global cap required')
    if args.prepare:args.plan=prepare(args);print(json.dumps({'plan':str(args.plan),'fits':0}));return 0
    plan,base,frozen=load(args)
    if args.admit:
        args.output.mkdir(parents=True,exist_ok=False);value=admit(args,plan,base,frozen);print(json.dumps({'admitted':value['timing_admission_passed'],'projected_seconds':value['projected_wall_seconds']}));return 0 if value['timing_admission_passed'] else 2
    if args.worker:
        lease(args,base)
        if not args.cpu_set:raise ValueError('Explicit disjoint CPU affinity required')
        if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{int(x) for x in args.cpu_set.split(',')})
        if args.job=='CPU-freeze-raw-score':score_cpu(args,plan,base,frozen)
        elif args.job=='CPU-primary20000':
            configure_cpu(plan);base.refine(args,frozen)
        else:fit_cpu(args,plan,base,frozen,next(j for j in frozen['jobs'] if j['id']==args.job))
        return 0
    return run(args,plan,base,frozen)

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as error:print('NewCPU replication refused: '+str(error),file=sys.stderr);sys.exit(1)
