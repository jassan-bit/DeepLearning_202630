"""Prepared two-job historical training reproduction, never confirmation.

Default: inspect byte identities and report gates, without NumPy/Torch imports.
Execution requires an external person's declared scope/tolerances/resources,
read-only copies of inputs and a fresh C output ID. Original masters untouched.
Prepared with disclosed AI assistance; identity declarations are not signatures.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path,PurePosixPath
import platform
import re
import signal
import subprocess
import sys
import time

HOME=Path(__file__).resolve().parent
SOURCE_COMMIT='8f65eac34aaf49293924ae8328c235cedbdd2866'
MANIFEST_SHA='043f2270a5d1bc147b8ac87e6b58570ab20a3759e58eacb6665ee56c43e2d17b'
PAIR=('atom_crps','additive_positive_head')
OLD_IDS=('20261119T031104515520Z','20261120T031424176339Z')

def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for raw in iter(lambda:f.read(1024*1024),b''):h.update(raw)
    return h.hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def safe(name):
    p=PurePosixPath(name)
    require(name and not p.is_absolute() and '..' not in p.parts and '\\' not in name and p.as_posix()==name,'Unsafe relative path')
    return p
def no_symlinks(path):
    require(not any(p.is_symlink() for p in (path,*path.parents)),'Symlink path forbidden')
def atomic_json(path,value):
    no_symlinks(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.partial')
    with temp.open('w') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def manifest_at(input_root):
    p=input_root/'SOURCE_MANIFEST.json';no_symlinks(p)
    require(sha(p)==MANIFEST_SHA,'Prepared source manifest changed')
    manifest=json.loads(p.read_bytes())
    require(manifest['source_commit']==SOURCE_COMMIT,'Wrong historical source')
    require([r['name'] for r in manifest['fixed_jobs']]==list(PAIR),'Two fixed jobs changed')
    require(all(r['seed']==20261118 and r['learning_rate']==0.003 and r['phase_budget_seconds']==120.0 for r in manifest['fixed_jobs']),'Historical seed/config/budget changed')
    return manifest

def inspect(case):
    no_symlinks(case);input_root=case/'input';manifest=manifest_at(input_root);files=[]
    for row in manifest['source_files']:
        p=input_root/safe(row['path']);no_symlinks(p)
        require(p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'Historical source/reference changed:'+row['path'])
        require(p.stat().st_mode&0o222==0,'Historical source/reference is writable:'+row['path'])
        files.append(row['path'])
    npz=input_root/'data/cohort.npz';no_symlinks(npz);ready=npz.is_file()
    if ready:
        require(npz.stat().st_size==manifest['input']['bytes'] and sha(npz)==manifest['input']['sha256'],'NPZ input differs')
        require(npz.stat().st_mode&0o222==0,'Input NPZ is writable')
    return manifest,{'source_and_reference_files_verified':len(files),'sources_read_only':True,
                     'input_present_and_hash_verified':ready,'npz_arrays_opened':0,
                     'original_source_commit':SOURCE_COMMIT,'fixed_jobs':manifest['fixed_jobs']}

def request_issues(request,manifest):
    missing=[]
    def need(ok,text):
        if not ok:missing.append(text)
    need(request.get('schema_version')==1,'schema_version=1')
    need(request.get('purpose')=='historical_training_reproduction_subset_not_confirmation','historical reproduction purpose')
    need(request.get('allow_training') is True,'external allow_training=true')
    need(request.get('scope')=='two_original_pilot_final_jobs_smallest_registered_seed','fixed two-job scope')
    run_id=request.get('new_run_id');need(isinstance(run_id,str) and re.fullmatch(r'C-[A-Za-z0-9][A-Za-z0-9_-]{5,90}',run_id or '') is not None and all(x not in (run_id or '') for x in OLD_IDS),'fresh C run ID')
    person=request.get('operator',{})
    need(isinstance(person.get('name'),str) and len(person['name'].strip())>=3,'external operator name declaration')
    need(isinstance(person.get('relationship_to_design'),str) and len(person['relationship_to_design'].strip())>=3,'relationship declaration')
    need(person.get('independent_of_design') is True,'operator independence declaration')
    stamp=request.get('recorded_at_utc')
    try:
        dt=datetime.fromisoformat(stamp.replace('Z','+00:00'));need(dt.tzinfo is not None and dt<=datetime.now(timezone.utc),'pre-execution timezone-aware timestamp')
    except (ValueError,TypeError,AttributeError):missing.append('pre-execution timestamp')
    resource=request.get('resources',{})
    need(resource.get('max_nominal_fit_seconds')==240.0,'two120s nominal fit opportunities')
    need(type(resource.get('max_wall_seconds')) is int and 240<=resource.get('max_wall_seconds',0)<=3600,'wall ceiling240..3600 seconds')
    need(resource.get('device')=='cuda:0' and resource.get('cpu_threads')==2,'declared CUDA0/two CPU threads')
    need(request.get('cohort_policy')=='same_as_original','original GPU/software cohort policy')
    need(request.get('expected_cohort')==manifest['original_cohort'],'original cohort declaration')
    tolerance=request.get('tolerances',{}).get('prediction_sigma7',{})
    vals=[tolerance.get('abs_tol'),tolerance.get('rel_tol')]
    need(all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in vals) and any(type(x) in (int,float) and x>0 for x in vals),'finite nonnegative predeclared C tolerances')
    need(isinstance(tolerance.get('rationale'),str) and len(tolerance['rationale'].strip())>=12,'C tolerance rationale')
    acknowledgements=request.get('acknowledgements',{})
    for key in ['exposed_development_calendar','not_confirmation','original_inputs_read_only','no_historical_master_writes','historical_network_durability_wait_not_replayed','partial_C_not_full_historical_reproduction']:
        need(acknowledgements.get(key) is True,'acknowledgement:'+key)
    need(request.get('signature_claimed') is False and request.get('human_execution_verified_by_software') is False,'no fabricated signature/human verification')
    return missing

def load_request(case,path):
    path=path or case/'C_REQUEST.template.json';no_symlinks(path)
    return json.loads(path.read_bytes()),path

def execute_worker(case,request_path,out,prepared_sha):
    manifest,inspection=inspect(case);request,_=load_request(case,request_path)
    require(not request_issues(request,manifest),'External request incomplete')
    no_symlinks(out)
    require(out.absolute()==case/'output'/request['new_run_id'],'Worker output must be the new isolated case output')
    require(sha(request_path)==prepared_sha,'Request changed after pre-execution record')
    marker=json.loads((out/'pre_execution.json').read_bytes())
    require(marker['request_sha256']==prepared_sha and marker['driver_sha256']==sha(__file__),'Execution marker identity changed')
    require(not (out/'jobs').exists(),'Fresh two-job directory required; no historical resume allowed')
    require(inspection['input_present_and_hash_verified'],'Pinned input missing')
    require(platform.python_version()==manifest['original_cohort']['python'],'Python differs from original cohort; no training started')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';sys.dont_write_bytecode=True
    source_root=case/'input/frozen_sources';sys.path.insert(0,str(source_root))
    # All source/data/operator guards precede scientific-runtime imports.
    import torch
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    engine_path=source_root/'experiments/e3_v3_development_pilot_20261120/pilot_engine.py'
    spec=importlib.util.spec_from_file_location('C_frozen_pilot_engine',engine_path);engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
    require(engine.cohort('cuda:0')==manifest['original_cohort'],'GPU/software cohort differs; no training started')
    plugin=source_root/'experiments/e3_v3_development_pilot_20261120/plugin_control.py'
    original_plan=json.loads((case/'input/references/plan.json').read_bytes())
    original_selection=json.loads((case/'input/references/closed_selection.json').read_bytes())
    require(engine.source_identity(source_root,plugin,SOURCE_COMMIT)==original_plan['identity'],'Historical source identity differs')
    data=engine.load_data(source_root,case/'input/data/cohort.npz',manifest['input']['sha256']);split=engine.split_development(data)
    plan={'version':'historical-C-subset-v1','master_run_id':request['new_run_id'],
          'scope':'two_job_historical_training_reproduction_not_confirmation',
          'original_plan_sha256':engine.digest(original_plan),'identity':original_plan['identity'],
          'config':original_plan['config'],'budget_seconds_per_fit':120.0,
          'admission_sha256':engine.digest(data['admission']),'cohort':manifest['original_cohort'],
          'request_sha256':prepared_sha,'remote_durability_wait_replayed':False}
    atomic_json(out/'subset_plan.json',plan)
    phase=engine.module_from_path('C_frozen_phase',source_root/engine.SOURCE_FILES[-1])
    fitted={};fit_records=[]
    for fixed in manifest['fixed_jobs']:
        require(original_selection['selected_learning_rates'][fixed['name']]==fixed['learning_rate'],'Frozen selected rate changed')
        model,record=engine.fit_job(fixed['name'],fixed['seed'],fixed['learning_rate'],'C_historical_subset',
              data,split,out,phase,plugin,plan,'cuda:0',durability_dir=None)
        fitted[fixed['name']]=model.to('cpu');fit_records.append(record)
    # Score the55 historical assessment dates only after both fresh fits finish.
    with (case/'input/references/assessment_predictions.csv').open() as f:
        legacy={(r['name'],int(r['seed']),r['date']):r for r in csv.DictReader(f) if r['name'] in PAIR}
    tolerance=request['tolerances']['prediction_sigma7'];results=[];per_model=[]
    for fixed in manifest['fixed_jobs']:
        name=fixed['name'];prediction,_=engine.evaluate_model(fitted[name].to('cuda:0'),name,data,split['assessment'],'cuda:0')
        diffs=[];violations=0
        for day,target,pred in zip(data['val_dates'][split['assessment']],data['val_y_original'][split['assessment']],prediction):
            old=legacy[(name,fixed['seed'],str(day))];require(float(old['target'])==float(target),'Reference target mismatch')
            ref=float(old['prediction']);observed=float(pred);require(math.isfinite(observed),'Nonfinite reproduced prediction')
            error=abs(observed-ref);limit=tolerance['abs_tol']+tolerance['rel_tol']*abs(ref);ok=error<=limit
            diffs.append(error);violations+=not ok
            results.append({'name':name,'seed':fixed['seed'],'date':str(day),'target_sigma7':float(target),
                  'historical_prediction_sigma7':ref,'reproduced_prediction_sigma7':observed,
                  'absolute_difference_sigma7':error,'declared_tolerance_sigma7':limit,'within_declared_tolerance':ok})
        per_model.append({'name':name,'rows':len(diffs),'max_absolute_difference_sigma7':max(diffs),'violations':violations})
        fitted[name].to('cpu')
    require(len(results)==110,'Two55-date comparisons required')
    with (out/'C_subset_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    inspect(case)  # Original source/data bytes must remain unchanged/read-only.
    report={'status':'C_partial_two_fresh_fits_executed','new_run_id':request['new_run_id'],
       'source_commit':SOURCE_COMMIT,'request_sha256':prepared_sha,'new_fit_count':2,
       'fit_records':fit_records,'comparisons':per_model,'comparison_rows':110,
       'all_within_operator_predeclared_tolerance':all(r['violations']==0 for r in per_model),
       'operator_declared':request['operator'],'operator_identity_independently_verified':False,
       'human_signature_claimed':False,'full_796_or_30_reproduction':False,'R3_accredited':False,
       'scientific_NO_GO_reopened':False,'historical_network_durability_wait_replayed':False,
       'limitation':'Fresh wall-clock opportunities do not reproduce historical network/interrupt latency or guarantee byte-identical training.',
       'finished_at_utc':now()}
    atomic_json(out/'C_subset_execution.json',report);print(json.dumps(report,ensure_ascii=False))
    return 0

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case-root',type=Path,required=True)
    parser.add_argument('--request',type=Path);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--worker-output',type=Path,help=argparse.SUPPRESS);parser.add_argument('--prepared-request-sha',help=argparse.SUPPRESS)
    args=parser.parse_args();case=args.case_root.absolute();manifest,inspection=inspect(case)
    request,path=load_request(case,args.request);issues=request_issues(request,manifest)
    if not inspection['input_present_and_hash_verified']:issues.append('pinned cohort NPZ acquired under its original source conditions')
    report={'status':'prepared_C_subset_dry_run_only','inspection':inspection,'execution_gates_missing':issues,
            'training_executed':False,'scientific_runtime_imported':False,'R3_accredited':False,
            'runtime_GPU_cohort_probed':False,'execution_readiness_not_certified_by_dry_run':True,
            'human_independent_execution_observed':False,'driver_sha256':sha(__file__)}
    if args.worker_output:
        require(not issues,'External execution request incomplete')
        return execute_worker(case,path,args.worker_output,args.prepared_request_sha)
    if not args.execute:print(json.dumps(report,ensure_ascii=False,indent=2));return 0
    require(not issues,'Execution blocked before runtime import: '+'; '.join(issues))
    require(platform.python_version()==manifest['original_cohort']['python'],'Python differs from original cohort; runtime not imported')
    output=case/'output'/request['new_run_id'];no_symlinks(output)
    require(not output.exists(),'Fresh isolated output required; never resume original or existing C files')
    require(not output.is_relative_to(case/'input'),'Output cannot be an original input')
    output.mkdir(parents=True);prepared_sha=sha(path)
    atomic_json(output/'pre_execution.json',{'request_sha256':prepared_sha,'driver_sha256':sha(__file__),
                'prepared_before_first_fit_at_utc':now(),'scope':request['scope'],'operator_declared':request['operator']})
    with (output/'external_request.json').open('xb') as copied:
        copied.write(path.read_bytes());copied.flush();os.fsync(copied.fileno())
    (output/'external_request.json').chmod(0o444)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    command=[sys.executable,str(Path(__file__).resolve()),'--case-root',str(case),'--request',str(output/'external_request.json'),
             '--worker-output',str(output),'--prepared-request-sha',prepared_sha]
    with (output/'worker.log').open('wb') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        timed_out=False
        try:code=process.wait(timeout=request['resources']['max_wall_seconds'])
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(process.pid,signal.SIGTERM)
            try:code=process.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);code=process.wait()
    receipt={'status':'external_C_worker_finished' if code==0 else 'external_C_worker_failed_or_timed_out',
       'worker_exit_code':code,'timed_out':timed_out,'request_sha256':prepared_sha,
       'execution_report':str(output/'C_subset_execution.json') if (output/'C_subset_execution.json').exists() else None,
       'partial_job_artifacts_retained':str(output/'jobs') if (output/'jobs').exists() else None,
       'full_796_or_30_reproduction':False,'R3_accredited':False,'operator_identity_independently_verified':False,
       'historical_masters_written':False,'finished_at_utc':now()}
    atomic_json(output/'driver_receipt.json',receipt);print(json.dumps(receipt,ensure_ascii=False));return code or (2 if timed_out else 0)

if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,OSError,KeyError,TypeError) as error:
        print(json.dumps({'status':'C_execution_blocked_or_failed','error':type(error).__name__+':'+str(error),
                          'full_C_claimed':False,'R3_accredited':False}),file=sys.stderr);raise SystemExit(2)
