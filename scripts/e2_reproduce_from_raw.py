"""Real fixed-recipe E2 reproduction: fit78+3 -> calibrate -> freeze -> raw -> score.

Default modes only prepare/check. --run explicitly invokes production CUDA fits.
No HPO/NAS, cached predictions, CPU test fitting, downloads or historical writes.
The original HPO and later model-dependent studies remain separate requirements.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'book_e2/_static/e2_scientific_closure_20261116/reproduction'
NEURAL=('FCN-1D','ResNet-1D','Inception-1D','LSTM-autoencoder con predictor','TimesNet','TSMixer','TiDE','TimeMixer','PatchTST','iTransformer','MLP16','LSTM32','GRU32')
SEEDS=tuple(range(271801,271807))
CLASSICAL={'Persistence':{},'SeasonalNaive':{'period':7},'ARIMA':{'order':[2,0,1]}}
ENGINE='1e8745497b484518f766b6da5c19be8362d9add7'
VERSION='e2-real-fixed-recipe-from-raw-v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def atomic(path,value):
    temporary=path.with_suffix(path.suffix+'.pending')
    with temporary.open('w') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    temporary.replace(path)


def record(root,path):
    return {'path':path.relative_to(root).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)}


def verify(root,row):
    relative=Path(row['path']);path=root/relative
    if relative.is_absolute() or '..' in relative.parts or not path.resolve().is_relative_to(root.resolve()) or path.is_symlink() or not path.is_file():raise ValueError('Missing/unsafe bound input: '+str(relative))
    if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('Bound bytes changed: '+str(relative))
    return path


def slug(name):return hashlib.sha256(name.encode()).hexdigest()[:16]


def module(root,name):
    sys.path.insert(0,str(root/'scripts'))
    item=__import__(name)
    if Path(item.__file__).resolve()!=root/'scripts'/(name+'.py'):raise ValueError('Scientific module shadowed: '+name)
    return item


def prepare(args):
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Prepare requires fresh output')
    args.output.mkdir(parents=True,exist_ok=True)
    portable=runpy.run_path(str(args.source_root/'scripts/e2_standalone_code.py'))
    for relative,expected in portable['CONSTRUCTOR_SHA256'].items():
        if sha(args.source_root/relative)!=expected:raise ValueError('Original constructor changed: '+relative)
    selected={}
    folder=args.source_root/'book_e2/_static/e2_corrections_20261115/hpo/sources/selections'
    for path in folder.glob('*.json'):
        value=json.loads(path.read_text())
        if value.get('model_id') in NEURAL:selected[value['model_id']]=(path,value)
    if set(selected)!=set(NEURAL):raise ValueError('Thirteen actual selection records required')
    files=[p for p in (args.source_root/'scripts').glob('*.py') if p.name not in {'e2_reproduce_from_raw.py','e2_hpo_cost_reconciliation.py','e2_corrections_budget.py','reproduce_e2_campaign.py'}]
    # Bind the complete scientific helper tree; no Git is required at execution.
    inputs=[record(args.source_root,p) for p in sorted(files)]
    for relative in [*portable['METADATA_SHA256'],portable['FROZEN_INPUTS'],portable['REGISTER'],
                     'experiments/dl_r2_cv_20261104/protocol.json','results/eda_development/daily_market_development.csv',
                     'results/eda_development/daily_volatility_development.csv','data/manifests/test_raw_reserve_20250101_20260831.json',
                     'results/filipe_multihistory_20261104/test/raw_test_audit.json','book_e2/_static/e2_benchmark_final_20261113/pilot_power_MDE.json']:
        inputs.append(record(args.source_root,args.source_root/relative))
    jobs=[]
    for name in NEURAL:
        path,selection=selected[name];inputs.append(record(args.source_root,path))
        if not selection.get('all_eighteen_slots_accounted_for') or not selection.get('run_artifact_bytes_verified') or not selection.get('source_current_recipe_verified'):raise ValueError('Selection audit incomplete: '+name)
        for seed in SEEDS:jobs.append({'id':slug(name)+'-'+str(seed),'kind':'neural','model':name,'seed':seed,'selection':selection,'directory':'final-fit/'+slug(name)+'/'+str(seed),'phases':3 if name==NEURAL[3] else 1})
    for name,parameters in CLASSICAL.items():jobs.append({'id':slug(name)+'-deterministic','kind':'classical','model':name,'seed':'deterministic','parameters':parameters,'directory':'classical-final/'+slug(name),'phases':1})
    design=json.loads((args.source_root/portable['DESIGN']).read_text())
    authors=[r for r in design['author_sources'] if r['model'] in portable['AUTHOR_MODELS']]
    reserve=json.loads((args.source_root/'data/manifests/test_raw_reserve_20250101_20260831.json').read_text())
    plan={'version':VERSION,'created_at_utc':datetime.now(timezone.utc).isoformat(),'driver_sha256':sha(Path(__file__)),'scientific_base_commit':ENGINE,
          'bound_sources':inputs,'author_files':authors,'raw_archives':reserve['archives'],'jobs':jobs,'neural_models':list(NEURAL),'seeds':list(SEEDS),
          'fit_phase_wall_seconds':1800,'global_wall_cap_seconds':args.cap_seconds,'draws_primary':20000,
          'development_counts':{'train':655,'calibration':53,'test_after_freeze':608},
          'pilot_path':'book_e2/_static/e2_benchmark_final_20261113/pilot_power_MDE.json',
          'HPO':{'repeated':False,'scope':'Reuse exact registered winning recipes and phase epochs. Reconciliation of historical costs is metadata, not a new HPO fit.',
                 'original_selection_reproduction_pending':True,'optional_real_backend':'e2_reduced_fit.fit_reduced_job(train, validation, fixed_epochs=None)',
                 'original_search_jobs':234,'new_search_jobs_scheduled':0,'NAS':'not_applicable_defined_bodies'},
          'ordered_stages':['source/runtime/data/budget gates','78 fixed-epoch neural fits and53-date calibration','3 fixed-configuration classical fits and calibration','81-artifact freeze using original validated pilot','admit and decode20pinned raw test archives','batch1 causal predictions608dates','A2 and primary inference on NEW predictions','new-primary refinement20000draws'],
          'excluded_pending':['full original HPO reproduction','real permutation/adversarial controls under their separate plan','all model-dependent robustness/explanations and18ablations'],
          'full_original_campaign_passed':False}
    write(args.output/'plan.json',plan)
    return args.output/'plan.json'


def load_plan(path,root):
    plan=json.loads(path.read_text())
    if plan['version']!=VERSION or plan['driver_sha256']!=sha(Path(__file__)):raise ValueError('Driver differs from registered execution plan')
    for row in plan['bound_sources']:verify(root,row)
    expected={(m,s) for m in NEURAL for s in SEEDS}|{(m,'deterministic') for m in CLASSICAL}
    if len(plan['jobs'])!=81 or {(r['model'],r['seed']) for r in plan['jobs']}!=expected:raise ValueError('Exact78+3schedule required')
    selections={}
    for row in plan['bound_sources']:
        if '/hpo/sources/selections/' in row['path']:
            selected=json.loads((root/row['path']).read_text());selections[selected['model_id']]=selected
    if set(selections)!=set(NEURAL):raise ValueError('Byte-bound13selection records required')
    for job in plan['jobs']:
        if job['model'] in NEURAL:
            if job['kind']!='neural' or job['selection']!=selections[job['model']] or job['directory']!='final-fit/'+slug(job['model'])+'/'+str(job['seed']):raise ValueError('Frozen neural selection or destination changed')
        elif job['kind']!='classical' or job['parameters']!=CLASSICAL[job['model']] or job['directory']!='classical-final/'+slug(job['model']):raise ValueError('Frozen classical configuration or destination changed')
    reserve=json.loads((root/'data/manifests/test_raw_reserve_20250101_20260831.json').read_text())
    if plan['raw_archives']!=reserve['archives'] or plan['fit_phase_wall_seconds']!=1800 or plan['draws_primary']!=20000:raise ValueError('Frozen data/worker/analysis policy changed')
    return plan


def budget_module():
    path=ROOT/'scripts/e2_corrections_budget.py'
    spec=importlib.util.spec_from_file_location('e2_fresh_fit_budget',path);item=importlib.util.module_from_spec(spec);spec.loader.exec_module(item)
    if not item.LEDGER.is_file():raise ValueError('Existing additive ledger required; never reset')
    return item


def readiness(args,plan):
    checks=[]
    def add(name,passed,detail):checks.append({'gate':name,'passed':bool(passed),'detail':detail})
    try:load_plan(args.plan,args.source_root);add('source_and_recipe_bytes',True,'All registered scientific helpers and13selections verified')
    except (OSError,ValueError) as e:add('source_and_recipe_bytes',False,str(e))
    missing=[]
    for row in plan['author_files']:
        path=args.author_root/row['model']/row['path']
        if not path.is_file() or path.is_symlink() or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:missing.append(row['model']+'/'+row['path'])
    add('author_sources_76',len(plan['author_files'])==76 and not missing,{'missing_or_changed':missing})
    bad=[]
    for row in plan['raw_archives']:
        path=args.raw_root/row['relative_path']
        if not path.is_file() or path.is_symlink() or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:bad.append(row['archive'])
    add('twenty_test_archives_bytes_only',len(plan['raw_archives'])==20 and not bad,{'missing_or_changed':bad,'numerical_test_payload_decoded':False})
    probe_data = "import sys,json;from pathlib import Path;sys.path.insert(0,str(Path(sys.argv[1])/'scripts'));import e2_reduced_final as f;from e2_development_data import load_development;r=Path(sys.argv[1]);p=json.loads(Path(sys.argv[2]).read_text());panel,_=load_development(r,'2024-12-30');train=f.final_training_split(panel);cal=f._calibration_split(panel,train.scaler);[f.validate_selection(j['model'],j['selection']) for j in p['jobs'] if j['kind']=='neural'];pilot=json.loads((r/p['pilot_path']).read_text());print(json.dumps({'train':len(train.x),'calibration':len(cal.x),'pilot_status':pilot.get('status')}))"
    lease=None;checker=None;probe_started=None;probe_code=1
    try:
        checker=budget_module();lease=checker.reserve('real_from_raw_readiness_data_check',60);probe_started=time.monotonic()
        proc=subprocess.run([str(args.python),'-I','-c',probe_data,str(args.source_root),str(args.plan)],capture_output=True,text=True,timeout=60)
        probe_code=proc.returncode
        if proc.returncode:raise ValueError(proc.stderr[-1500:])
        data=json.loads(proc.stdout)
        add('real_development_train_cal',data['train']==655 and data['calibration']==53,data)
        add('original_pilot_evidence',data['pilot_status']=='development_pilot_completed_power_reported','Historical byte-bound pilot reused; no pilot refit')
    except Exception as e:add('real_development_and_pilot',False,str(e))
    finally:
        if lease:checker.finish(lease,time.monotonic()-probe_started,probe_code)
    probe='import torch,json,sys;print(json.dumps({"python":sys.version.split()[0],"torch":torch.__version__,"cuda":torch.version.cuda,"available":torch.cuda.is_available(),"device_name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"total_memory_bytes":torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None}))'
    try:
        proc=subprocess.run([str(args.python),'-I','-c',probe],capture_output=True,text=True,timeout=40);runtime=json.loads(proc.stdout) if proc.returncode==0 else {'error':proc.stderr[-1000:]}
        add('production_CUDA_runtime',runtime.get('available') and runtime.get('torch','').split('+')[0]=='2.8.0',runtime)
    except Exception as e:runtime={'error':str(e)};add('production_CUDA_runtime',False,runtime)
    try:
        ledger=budget_module()
        with ledger.locked_ledger(ledger.LEDGER) as (rows,_):state=ledger.summarize(rows)
        add('existing_authorized_additive_budget',args.cap_seconds<=state['unreserved_seconds'] and not state['open_reservations'],state)
    except Exception as e:state=None;add('existing_authorized_additive_budget',False,str(e))
    return {'version':VERSION,'mode':'readiness_only','plan_sha256':sha(args.plan),'checks':checks,'ready_to_run_fixed_recipe_chain':all(r['passed'] for r in checks),'runtime':runtime,'budget':state,
            'new_model_fits':0,'test_values_decoded':False,'full_original_campaign_passed':False,'HPO_repeated':False,'NAS_required':False}


def configure_cuda():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    if not torch.cuda.is_available():raise ValueError('Real production CUDA required; no CPU test flag is admitted')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return torch


def execute_job(args,plan,job):
    torch=configure_cuda();final=module(args.source_root,'e2_reduced_final')
    destination=args.output/job['directory'];destination.mkdir(parents=True,exist_ok=False)
    roots={name:str(args.author_root/name) for name in {r['model'] for r in plan['author_files']}}
    props=torch.cuda.get_device_properties(0)
    runtime={'device':'cuda:0','device_name':props.name,'total_memory_bytes':props.total_memory,'torch':torch.__version__,'torch_cuda':torch.version.cuda,'python':sys.version.split()[0],'cpu_threads':2,'interop_threads':1,'deterministic':True,'tf32':False,'cudnn_tf32':False}
    spec={'data_root':str(args.source_root),'model':job['model'],'seed':job['seed'],'device':'cuda:0','source_roots':roots,'fit_wall_seconds':plan['fit_phase_wall_seconds'],
          'source_identity':{'source_commit':ENGINE,'files_sha256':{r['path']:r['sha256'] for r in plan['bound_sources']},'driver_sha256':sha(Path(__file__))},'runtime_identity':runtime}
    if job['kind']=='neural':spec.update(kind='neural-final',selection=job['selection'])
    else:spec.update(kind='classical',mode='classical-final',parameters=job['parameters'])
    write(destination/'worker_spec.json',spec)
    torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    result=final.neural_final_worker(spec,destination) if job['kind']=='neural' else final.classical_worker(spec,destination)
    result['new_reproduction_wall_seconds']=time.monotonic()-started;result['new_isolated_GPU_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    write(destination/'result.json',result)
    return result


def require_worker_lease(args):
    ledger=budget_module()
    with ledger.locked_ledger(ledger.LEDGER) as (rows,_):state=ledger.summarize(rows)
    active=state['open_reservations'].get(args.reservation_id)
    if not active or active['phase']!='fixed_recipe_real_from_raw_78plus3':raise ValueError('Worker requires the active parent campaign reservation; direct unbudgeted fitting is forbidden')
    identity=json.loads((args.output/'run_identity.json').read_text())
    if identity['plan_sha256']!=sha(args.plan) or identity['driver_sha256']!=sha(Path(__file__)):raise ValueError('Worker source/plan differs from parent run')


def completed_job(directory):
    path=directory/'result.json'
    if not path.is_file():return False
    value=json.loads(path.read_text())
    if value.get('status')!='completed':raise ValueError('An earlier job did not complete; reconcile, never silently retry')
    for relative,expected in value['artifacts_sha256'].items():
        if sha(directory/relative)!=expected:raise ValueError('Existing completed artifact changed')
    for key in ['checkpoint','scaler','calibrator']:
        if sha(directory/value[key])!=value[key+'_sha256']:raise ValueError('Existing predictor binding changed')
    return True


def score(args,plan):
    configure_cuda();final=module(args.source_root,'e2_reduced_final')
    shutil.copyfile(args.source_root/plan['pilot_path'],args.output/'pilot_power_MDE.json')
    final.freeze_final(args.output)
    # Explicit absent derivative routes the original loader to the verified raw
    # root. It prevents score_final's implicit GitHub derivative fallback.
    daily=args.output/'raw_derivative_supplied_by_original_loader.csv'
    roots={name:str(args.author_root/name) for name in {r['model'] for r in plan['author_files']}}
    result=final.score_final(args.source_root,args.source_root,args.output,daily_csv=daily,raw_root=args.raw_root,allow_download=False,source_roots=roots)
    return result


def refine(args,plan):
    module(args.source_root,'e2_reduced_final')
    import numpy as np
    from e2_reduced_final_analysis import analyze_benchmark
    directory=args.output/'score';admission=json.loads((directory/'data_admission.json').read_text())
    dates=next(json.loads(p.read_text())['dates'] for p in directory.glob('predictions_*.json'))
    # Recreate targets from admitted RAW-derived daily closes, never old forecasts.
    final=module(args.source_root,'e2_reduced_final');loader=module(args.source_root,'e2_reduced_final_data')
    freeze=json.loads((args.output/'final_freeze.json').read_text());reservation=loader.reserve_known_e1_evaluation(args.source_root)
    panel,_=loader.load_known_e1_evaluation(args.source_root,reservation,freeze,artifact_root=args.output,daily_csv=args.output/'absent_derivative.csv',raw_root=args.raw_root,allow_download=False,output_dir=args.output/'private-known-E1-data',development_root=args.source_root)
    prediction={m:[] for m in (*NEURAL,*CLASSICAL)};calibrators={m:[] for m in prediction}
    for row in freeze['models']:
        value=json.loads((directory/('predictions_'+slug(row['model'])+'_'+str(row['seed'])+'.json')).read_text())
        prediction[row['model']].append(value['prediction_original']);calibrators[row['model']].append(json.loads((args.output/row['calibrator']).read_text()))
    result=analyze_benchmark(panel.targets_at(dates),prediction,dates=dates,mask=[d in set(admission['common_eligible_dates']) for d in dates],calibrators=calibrators,draws=plan['draws_primary'])
    write(directory/'benchmark_analysis_20000.json',result)
    write(directory/'primary20000_completion.json',{'output_sha256':sha(directory/'benchmark_analysis_20000.json'),'source_plan_sha256':sha(args.plan),'draws':20000,'new_forecasts_used':True})


def run_chain(args,plan):
    if args.resume:
        prior=json.loads((args.output/'run_identity.json').read_text())
        if prior['plan_sha256']!=sha(args.plan) or prior['driver_sha256']!=sha(Path(__file__)):raise ValueError('Resume plan/source differs')
    elif args.output.exists() and any(args.output.iterdir()):raise ValueError('Run requires fresh output; use explicit verified resume for completed jobs only')
    args.output.mkdir(parents=True,exist_ok=True)
    ready=readiness(args,plan)
    atomic(args.output/'readiness.json',ready)
    if not ready['ready_to_run_fixed_recipe_chain']:
        atomic(args.output/'status.json',{'status':'blocked_before_any_fit','readiness':ready,'full_original_campaign_passed':False});return 2
    if not args.resume:write(args.output/'run_identity.json',{'plan_sha256':sha(args.plan),'driver_sha256':sha(Path(__file__)),'frozen_recipe_refits_only':True})
    ledger=budget_module();key=ledger.reserve('fixed_recipe_real_from_raw_78plus3',args.cap_seconds);started=time.monotonic();code=1
    state={'version':VERSION,'status':'running','jobs':[],'full_original_campaign_passed':False,'HPO_repeated':False,'original_selection_reproduction_pending':True}
    try:
        schedule=plan['jobs']+[{'id':'freeze-admit-score','kind':'score','phases':1},{'id':'primary20000','kind':'refine','phases':1}]
        for job in schedule:
            remaining=args.cap_seconds-(time.monotonic()-started)
            if remaining<=0:raise TimeoutError('Authorized global wall cap exhausted')
            if job['kind'] in {'neural','classical'}:
                directory=args.output/job['directory']
                if directory.exists():
                    if completed_job(directory):state['jobs'].append({'id':job['id'],'status':'verified_completed_before_resume'});continue
                    raise ValueError('Partial job preserved; explicit reconciliation needed, never optimizer restart: '+job['id'])
            elif job['kind']=='score' and (args.output/'score/result.json').is_file():
                saved=json.loads((args.output/'score/result.json').read_text())
                for relative,expected in saved['artifacts_sha256'].items():
                    if sha(args.output/'score'/relative)!=expected:raise ValueError('Completed scoring artifact changed')
                module(args.source_root,'e2_reduced_final_data').validate_final_freeze(json.loads((args.output/'final_freeze.json').read_text()),args.output)
                state['jobs'].append({'id':job['id'],'status':'verified_completed_before_resume'});continue
            elif job['kind']=='refine' and (args.output/'score/primary20000_completion.json').is_file():
                saved=json.loads((args.output/'score/primary20000_completion.json').read_text())
                if saved['source_plan_sha256']!=sha(args.plan) or saved['output_sha256']!=sha(args.output/'score/benchmark_analysis_20000.json'):raise ValueError('Completed primary refinement changed')
                state['jobs'].append({'id':job['id'],'status':'verified_completed_before_resume'});continue
            command=[str(args.python),str(Path(__file__).resolve()),'--worker','--plan',str(args.plan),'--job',job['id'],'--source-root',str(args.source_root),'--author-root',str(args.author_root),'--raw-root',str(args.raw_root),'--output',str(args.output),'--reservation-id',key]
            log=args.output/'logs'/(job['id']+'.log');log.parent.mkdir(exist_ok=True);tick=time.monotonic()
            with log.open('x') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=min(remaining,job['phases']*1800+120))
            state['jobs'].append({'id':job['id'],'status':'completed' if result.returncode==0 else 'failed','wall_seconds':time.monotonic()-tick,'log_sha256':sha(log)})
            atomic(args.output/'status.json',state)
            if result.returncode:raise RuntimeError('Real scientific worker failed: '+job['id'])
        state['status']='completed_fixed_recipe_fit_cal_raw_test_A2_primary_scope';state['new_fit_predictors']=81;code=0
    except Exception as error:state.update(status='partial_stopped',error=str(error));code=2
    finally:
        ledger.finish(key,time.monotonic()-started,code)
        with ledger.locked_ledger(ledger.LEDGER) as (rows,_):state['additive_budget']=ledger.summarize(rows)
        state['integral_original_HPO_controls_robustness_reproduced']=False;atomic(args.output/'status.json',state)
    print(json.dumps({'status':state['status'],'full_original_campaign_passed':False,'output':str(args.output)}));return code


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    for mode in ['prepare-plan','readiness','run','worker']:modes.add_argument('--'+mode,action='store_true')
    parser.add_argument('--plan',type=Path,default=BASE/'v2/plan.json');parser.add_argument('--source-root',type=Path,default=ROOT)
    parser.add_argument('--author-root',type=Path,default=Path('/workspace/work/e2architecture_sources'))
    parser.add_argument('--raw-root',type=Path,default=ROOT/'work/e2-scientific-market/raw')
    parser.add_argument('--python',type=Path,default=Path(sys.executable));parser.add_argument('--output',type=Path,default=ROOT/'work'/('e2-real-reproduction-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))
    parser.add_argument('--cap-seconds',type=float,default=12900);parser.add_argument('--resume',action='store_true');parser.add_argument('--job');parser.add_argument('--reservation-id',help=argparse.SUPPRESS)
    args=parser.parse_args()
    for name in ['plan','source_root','author_root','raw_root','python','output']:setattr(args,name,getattr(args,name).absolute())
    if not 0<args.cap_seconds<=12900:parser.error('Positive cap <=12900s required; actual available balance is checked without reset')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    if args.prepare_plan:args.plan=prepare(args)
    plan=load_plan(args.plan,args.source_root)
    if args.worker:
        require_worker_lease(args)
        if args.job=='freeze-admit-score':score(args,plan)
        elif args.job=='primary20000':refine(args,plan)
        else:
            job=next(r for r in plan['jobs'] if r['id']==args.job);execute_job(args,plan,job)
        return 0
    if args.run:return run_chain(args,plan)
    if not args.prepare_plan:
        if args.output.exists() and any(args.output.iterdir()):raise ValueError('Readiness output must be fresh')
        args.output.mkdir(parents=True,exist_ok=True)
    value=readiness(args,plan);write(args.output/'readiness.json',value)
    print(json.dumps({'plan':str(args.plan),'ready':value['ready_to_run_fixed_recipe_chain'],'failed_gates':[r['gate'] for r in value['checks'] if not r['passed']],'new_fits':0}));return 0 if value['ready_to_run_fixed_recipe_chain'] else 2

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as error:print('Reproduction refused: '+str(error),file=sys.stderr);sys.exit(1)
