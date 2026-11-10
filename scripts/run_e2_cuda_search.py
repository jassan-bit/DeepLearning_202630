"""Budgeted registered CUDA selection; public GitHub recovery between jobs.

Plan-only uses the standard library and never authenticates, loads market
values, imports Torch, or creates a campaign. Execution is explicitly opt-in.
This selects fourteen neural systems, not the eighteen-system final benchmark.
"""
from __future__ import annotations
import argparse,json,math,os,re,subprocess,sys,tempfile,time,traceback
from pathlib import Path
import e2_cuda_search_plan as admission

ROOT=Path(__file__).resolve().parents[1]
VERSION='e2-cuda-search-runner-v1'
RESOURCE_KEYS=('model_parameters','optimizer_steps','optimizer_steps_known_lower_bound',
    'optimization_stages','elapsed_seconds','optimizer_steps_unknown',
    'partial_optimizer_update_may_have_occurred','error_code','phase_completed_count',
    'supervisor_elapsed_seconds','supervisor_rss_bytes')
PHASE_KEYS=('epochs_completed','best_epoch','optimizer_steps','optimizer_steps_known_lower_bound',
    'elapsed_seconds','fixed_epochs','selected_state_sha256','final_state_sha256')
ERROR_CODES={'timeout','memory','resource_limit','nonfinite','geometry','missing_gradient',
    'checkpoint','checkpoint_persistence','validation_inference','state_mutation',
    'external_timeout','external_memory','worker_exception','worker_result_invalid','runtime_error'}

def numeric(value,*,integer=False):
    if value is None:return None
    if isinstance(value,bool) or type(value) not in (float,int) or not math.isfinite(value) or value<0:
        raise ValueError('invalid_public_resource_number')
    if integer and type(value) is not int:raise ValueError('invalid_public_resource_count')
    return value

def identity(job):
    return {key:job[key] for key in ('model','origin','candidate_id','payload_sha256','seed','stage','prepared_signature')}

def retain_private_diagnostics(directory,job_id):
    """Failed-worker logs remain local; they are never campaign mirror inputs."""
    target=Path(tempfile.gettempdir())/'e2_cuda_search_private_diagnostics'
    target.mkdir(exist_ok=True)
    for name in ('stdout.log','stderr.log'):
        source=Path(directory)/name
        if not source.is_file():continue
        destination=target/(job_id+'-'+name)
        try:
            with source.open('rb') as inp,destination.open('xb') as out:
                out.write(inp.read(1024*1024))
        except FileExistsError:pass

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.tmp-',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:
        Path(tmp).unlink(missing_ok=True)

def public_trial(run):
    """Positive scientific-field export; exception text and paths never escape."""
    raw=run.resource_use
    resource={}
    for key in RESOURCE_KEYS:
        if key not in raw:continue
        if key in ('optimizer_steps_unknown','partial_optimizer_update_may_have_occurred'):
            if type(raw[key]) is not bool:raise ValueError('invalid_public_resource_boolean')
            resource[key]=raw[key]
        elif key=='error_code':resource[key]=raw[key] if raw[key] in ERROR_CODES else 'runtime_error'
        else:resource[key]=numeric(raw[key],integer=key not in ('elapsed_seconds','supervisor_elapsed_seconds'))
    if 'identity' in raw:
        fields={'model','origin','candidate_id','payload_sha256','seed','stage','prepared_signature'}
        item=raw['identity']
        if not isinstance(item,dict) or set(item)!=fields:raise ValueError('invalid_public_job_identity')
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}',item['origin']):raise ValueError('invalid_public_origin')
        if item['stage'] not in ('hpo','nas1','nas2'):raise ValueError('invalid_public_stage')
        numeric(item['seed'],integer=True)
        for key in ('payload_sha256','prepared_signature'):
            if not isinstance(item[key],str) or not re.fullmatch('[0-9a-f]{64}',item[key]):raise ValueError('invalid_public_identity_hash')
        for key in ('model','candidate_id'):
            if not isinstance(item[key],str) or not item[key] or '/' in item[key] or '\\' in item[key]:raise ValueError('invalid_public_identity_string')
        resource['identity']=dict(item)
    steps=resource.get('optimizer_steps')
    resource.setdefault('optimizer_steps_known_lower_bound',steps if type(steps) is int else 0)
    resource['optimizer_steps_unknown']=steps is None
    resource['checkpoint_policy']='selection_metadata_only'
    resource['phase_records']={}
    for name,phase in raw.get('phase_records',{}).items():
        if name in ('ae1','ae2','supervised'):
            item={}
            for key in PHASE_KEYS:
                if key not in phase:continue
                value=phase[key]
                if key.endswith('_sha256'):
                    if value is not None and (not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value)):
                        raise ValueError('invalid_public_phase_hash')
                    item[key]=value
                else:item[key]=numeric(value,integer=key!='elapsed_seconds')
            resource['phase_records'][name]=item
    metrics=dict(raw.get('cuda_metrics',{}))
    for public,internal in [('max_memory_allocated_bytes','max_cuda_memory_allocated_bytes'),
                            ('max_memory_reserved_bytes','max_cuda_memory_reserved_bytes')]:
        if internal in raw:metrics[public]=raw[internal]
    resource['cuda_metrics']={key:numeric(metrics[key],integer=True) for key in ('max_memory_allocated_bytes',
        'max_memory_reserved_bytes','effective_limit_bytes') if key in metrics}
    code=resource.get('error_code')
    if code is not None and code not in ERROR_CODES:resource['error_code']='runtime_error'
    reason=None if run.status=='completed' else (resource.get('error_code') or 'runtime_error')
    if type(run.fatal) is not bool:raise ValueError('invalid_trial_fatal_flag')
    if not isinstance(run.run_id,str) or not re.fullmatch(r'(?:(?:worker|external)-)?[0-9a-f]{64}',run.run_id):
        raise ValueError('invalid_public_run_id')
    if run.phase_epochs is not None:
        if not isinstance(run.phase_epochs,dict) or not run.phase_epochs:raise ValueError('invalid_phase_epochs')
        for key,value in run.phase_epochs.items():
            if key not in ('ae1','ae2','supervised') or numeric(value,integer=True) is None or value<1:raise ValueError('invalid_phase_epochs')
    result={'status':run.status,'run_id':run.run_id,'resource_use':resource,
        'predictions_original':None if run.predictions_original is None else [float(x) for x in run.predictions_original],
        'prediction_dates':None if run.prediction_dates is None else [str(x.isoformat()) for x in run.prediction_dates],
        'phase_epochs':run.phase_epochs,'reason':reason,'fatal':bool(run.fatal)}
    admission.canonical(result)
    if result['status'] not in ('completed','failed'):raise ValueError('invalid_trial_status')
    if result['status']=='completed':
        if (not result['predictions_original'] or result['phase_epochs'] is None
                or result['prediction_dates'] is None
                or len(result['predictions_original'])!=len(result['prediction_dates'])
                or any(not math.isfinite(x) for x in result['predictions_original'])):
            raise ValueError('invalid_validation_predictions')
    return result

def restore_trial(value):
    from e2_temporal_training import TrialRun
    import pandas as pd
    if set(value)!={'status','run_id','resource_use','predictions_original','prediction_dates','phase_epochs','reason','fatal'}:
        raise ValueError('invalid_public_trial_fields')
    dates=None if value['prediction_dates'] is None else tuple(pd.Timestamp(x) for x in value['prediction_dates'])
    trial=TrialRun(value['status'],value['run_id'],value['resource_use'],
        value['predictions_original'],dates,value['phase_epochs'],value['reason'],value['fatal'])
    if public_trial(trial)!=value:raise ValueError('noncanonical_public_trial')
    return trial

def job_payload(plan,prepared,model,candidate,seed):
    return {'model':model,'origin':str(prepared.origin.date()),'candidate_id':candidate.candidate_id,
        'payload_sha256':candidate.payload_sha256,'candidate':candidate.payload,'seed':seed,
        'stage':candidate.payload['stage'],'prepared_signature':prepared.signature,
        'training_signature':prepared.inner_train.signature,'validation_signature':prepared.inner_validation.signature,
        'source_commit':plan['source_commit'],'protocol_sha256':plan['protocol_sha256'],
        'hardware_sha256':admission.digest(plan['hardware_binding'])}

def prepare_panel(data_root):
    """Only the pinned development prefix; final calendar rows contain NaNs."""
    import numpy as np
    import pandas as pd
    from e2_development_data import load_development
    from e2_temporal_training import DatedPanel
    original,receipt=load_development(data_root,'2024-12-30')
    calendar=pd.date_range(original.calendar[0],pd.Timestamp('2025-01-01',tz='UTC'),freq='D')
    n=len(original.calendar);features=np.full((len(calendar),5),np.nan)
    targets=np.full(len(calendar),np.nan);quality=np.zeros(len(calendar),dtype=bool)
    features[:n],targets[:n],quality[:n]=original.features,original.targets,original.quality
    return DatedPanel(calendar,features,targets,quality),receipt

def verify_source(source_commit):
    head=subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
    if head!=source_commit:raise ValueError('source_checkout_does_not_match_pin')
    changed=subprocess.run(['git','-C',str(ROOT),'diff','--name-only','HEAD'],check=True,capture_output=True,text=True).stdout.splitlines()
    if any(p!='book_e2/_static/search_protocol_20261109/protocol.json' for p in changed):
        raise ValueError('frozen_source_has_changes')
    extra=subprocess.run(['git','-C',str(ROOT),'ls-files','--others','--exclude-standard','scripts'],check=True,capture_output=True,text=True).stdout.splitlines()
    if any(p.endswith('.py') for p in extra):raise ValueError('unregistered_python_modules')

def cuda_hardware():
    if sys.version_info[:3]!=(3,12,12):raise ValueError('managed_python_3_12_12_required')
    from importlib.metadata import version as installed_version
    for line in (ROOT/'requirements-e2-training-gpu.lock.txt').read_text().splitlines():
        line=line.strip()
        if not line or line.startswith(('#','--')):continue
        name,separator,expected=line.partition('==')
        if separator and installed_version(name)!=expected:
            raise ValueError('locked_dependency_version_required')
    from run_e2_gpu_pilot import require_cuda_environment,_public_hardware
    hw=_public_hardware(require_cuda_environment('cuda:0'))
    if hw['torch']!='2.8.0+cu126' or hw['torch_cuda']!='12.6':raise ValueError('locked_cuda_environment_required')
    return hw

def run_worker(spec_path):
    """One isolated attempt. GitHub credentials are absent from this process."""
    spec=json.loads(Path(spec_path).read_text());plan=spec['plan'];job=spec['job']
    output=Path(spec['output'])
    from e2_temporal_training import CandidateSpec,TrialRun,prepare_b
    trial=None
    try:
        verify_source(plan['source_commit'])
        if admission.source_hashes()!=plan['source_sha256']:raise ValueError('worker_source_hash_changed')
        hw=cuda_hardware()
        if hw!=plan['hardware_binding']:raise ValueError('worker_hardware_binding_changed')
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        torch.cuda.set_per_process_memory_fraction(hw['effective_memory_limit_bytes']/hw['total_memory_bytes'],0)
        torch.cuda.reset_peak_memory_stats(0)
        panel,_=prepare_panel(spec['data_root'])
        prepared=prepare_b(panel,job['origin'],job['origin'])
        if prepared.signature!=job['prepared_signature']:raise ValueError('prepared_split_changed')
        from e2_cuda_bindings import CudaNeuralTrainingAdapter
        roots={'iTransformer':spec['source_root_itransformer']} if spec.get('source_root_itransformer') else None
        adapter=CudaNeuralTrainingAdapter(job['model'],device='cuda:0',source_roots=roots,
            full_epochs=200,short_epochs=50,patience=20,wall_limit_seconds=7200,
            short_wall_limit_seconds=1800,memory_limit_bytes=12*1024**3,
            gpu_memory_limit_bytes=hw['effective_memory_limit_bytes'],deterministic=True)
        candidate=CandidateSpec(job['candidate_id'],job['candidate'])
        if candidate.payload_sha256!=job['payload_sha256']:raise ValueError('candidate_hash_changed')
        trial=adapter.fit_candidate(candidate,job['seed'],prepared.inner_train.clone(),prepared.inner_validation.clone())
        trial.resource_use['identity']=identity(job)
        payload=public_trial(trial)
        payload['resource_use']['cuda_metrics']['effective_limit_bytes']=hw['effective_memory_limit_bytes']
    except Exception:
        traceback.print_exc(file=sys.stderr)  # Isolated private stderr, never exported.
        known=None if trial is None else trial.resource_use.get('optimizer_steps')
        lower=0 if trial is None else trial.resource_use.get('optimizer_steps_known_lower_bound',known if type(known) is int else 0)
        failure=TrialRun.failed('worker_exception',{'error_code':'worker_exception',
            'optimizer_steps':known,'optimizer_steps_known_lower_bound':lower,
            'partial_optimizer_update_may_have_occurred':known is None,'identity':identity(job)},'worker-'+spec['job_id'],fatal=True)
        payload=public_trial(failure)
    write_json(output/'worker_profile.json',{'job_id':spec['job_id'],'plan_sha256':admission.digest(plan),'trial_run':payload})
    return 0

def isolated_trial(plan,job,*,data_root,source_root_itransformer,deadline):
    from profile_e2_training import run_isolated_profile
    from e2_temporal_training import TrialRun
    phases=3 if job['model']=='LSTM-autoencoder con predictor' else 1
    cap=1800 if job['stage']=='nas1' else 7200
    job_id=admission.digest({'plan_sha256':admission.digest(plan),'job':job})
    with tempfile.TemporaryDirectory(prefix='e2-cuda-search-worker-') as temporary:
        root=Path(temporary);output=root/'attempt'
        spec={'plan':plan,'job':job,'job_id':job_id,'output':str(output),
            'data_root':str(Path(data_root).resolve()),'source_root_itransformer':source_root_itransformer}
        write_json(root/'spec.json',spec)
        environment=dict(os.environ)
        for key in ('GITHUB_TOKEN','GH_TOKEN'):environment.pop(key,None)
        environment.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        environment.pop('PYTHONPATH',None)
        remaining=deadline-time.monotonic()
        if remaining<=0:raise RuntimeError('admitted_job_has_no_remaining_time')
        result=run_isolated_profile([sys.executable,str(Path(__file__).resolve()),'--worker-spec',str(root/'spec.json')],output,
            timeout_seconds=min(phases*cap+120,remaining),memory_limit_bytes=12*1024**3,cwd=ROOT,environment=environment)
        worker=result.get('worker_profile')
        if (result['status']=='success' and isinstance(worker,dict) and worker.get('job_id')==job_id
                and worker.get('plan_sha256')==admission.digest(plan)):
            trial=restore_trial(worker['trial_run'])
            if trial.resource_use.get('identity')!=identity(job):raise ValueError('worker_identity_changed')
            trial.resource_use.update(supervisor_elapsed_seconds=result['elapsed_wall_seconds'],
                supervisor_rss_bytes=result['sampled_peak_process_group_rss_bytes'])
            if trial.status=='failed':retain_private_diagnostics(output,job_id)
            return public_trial(trial)
        retain_private_diagnostics(output,job_id)
        code={'timeout':'external_timeout','memory_limit_observed':'external_memory'}.get(result['status'],'worker_result_invalid')
        lower=0
        if isinstance(worker,dict) and worker.get('job_id')==job_id and worker.get('plan_sha256')==admission.digest(plan):
            try:lower=restore_trial(worker['trial_run']).resource_use.get('optimizer_steps_known_lower_bound',0)
            except (ValueError,KeyError,TypeError):pass
        trial=TrialRun.failed(code,{'error_code':code,'optimizer_steps':None,'optimizer_steps_known_lower_bound':lower,
            'partial_optimizer_update_may_have_occurred':True,'supervisor_elapsed_seconds':result['elapsed_wall_seconds'],
            'supervisor_rss_bytes':result['sampled_peak_process_group_rss_bytes'],'identity':identity(job)},'external-'+job_id,
            fatal=code=='worker_result_invalid')
        return public_trial(trial)

def run_selection(plan,session,panel,*,worker,mirror=None,progress=None):
    """Recompute deterministic selection with cached typed outcomes on resume."""
    import e2_training_models as registry
    from e2_training_bindings import run_neural_selection
    from e2_temporal_training import prepare_b
    selections=[] if progress is None else progress
    def sync():
        if mirror is not None and session.pending_record_paths:
            paths=session.pending_record_paths
            mirror.push(session.directory,paths=paths,message='Confirm scientific search state')
            session.mark_records_synced(paths)
    for origin in plan['origins']:
        prepared=prepare_b(panel,origin,origin)
        for model in plan['models']:
            class DurableAdapter:
                wall=7200;short_wall=1800;memory=12*1024**3
                full_epochs=200;short_epochs=50;patience=20
                def fit_candidate(self,candidate,seed,train,validation):
                    if train.signature!=prepared.inner_train.signature or validation.signature!=prepared.inner_validation.signature:
                        raise ValueError('selection_callback_split_changed')
                    job=job_payload(plan,prepared,model,candidate,seed)
                    identity=admission.digest({'plan_sha256':admission.digest(plan),'job':job})
                    phases=3 if model=='LSTM-autoencoder con predictor' else 1
                    seconds=(1800 if job['stage']=='nas1' else 7200)*phases+300
                    cached=session.get_result(identity,job) is not None
                    if mirror is not None and not cached:
                        mirror.check_capacity(required_requests=32)
                    def call(deadline):
                        sync()  # Remote STARTED acknowledgement precedes fitting.
                        return worker(job,deadline)
                    value=session.run_job(identity,job,call,reservation_seconds=seconds)
                    sync()  # Completed local result is remote before the next fit.
                    if not cached:
                        print(json.dumps({'model':model,'origin':job['origin'],'stage':job['stage'],
                            'candidate_id':job['candidate_id'],'seed':seed,'status':value['status'],
                            'known_optimizer_steps':value['resource_use'].get('optimizer_steps_known_lower_bound',0)},ensure_ascii=False),flush=True)
                    return restore_trial(value)
            protocols=registry.protocol()
            architectures=admission.planner.enumerate_architectures(protocols,model) if model in registry.MODEL_IDS else []
            selected,_=run_neural_selection(prepared,model,DurableAdapter(),hpo_trials=registry.registered_trials(model),
                architectures=architectures,hpo_seeds=plan['seed_pools']['hpo'],phase1_seeds=plan['seed_pools']['nas1'],
                phase2_seeds=plan['seed_pools']['nas2'],k=40,delta=protocols['contract']['delta'])
            selections.append({'model':model,'origin':origin,'winner_id':selected.winner.candidate_id,
                'winner':selected.winner.payload,'fixed_epochs':selected.fixed_epochs,'selection_signature':selected.signature,
                'attempts':selected.attempts,'selection_rule':selected.selection_rule})
    return selections

def report(plan,session,status,selections,repository,branch,*,pause_reason=None,retry_after_seconds=None):
    completed=failed=known=unknown=0
    for identity in session.completed_job_ids:
        value=session.get_result(identity)
        completed+=1;failed+=value['status']=='failed'
        resource=value['resource_use']
        known+=resource.get('optimizer_steps_known_lower_bound',0)
        unknown+=resource.get('optimizer_steps') is None
    return {'version':VERSION,'plan_sha256':admission.digest(plan),'status':status,'scope':plan['scope'],
        'completed_jobs':completed,'failed_jobs':failed,'known_optimizer_steps':known,
        'unknown_optimizer_step_jobs':unknown,'completed_selections':selections,'coverage':plan['coverage'],
        'budget':session.summary(),'scientific_HPO_or_NAS_executed':completed>0,
        'final_market_values_read':False,'final_R':None,'full_campaign_wall_hours':None,
        'remote_repository':repository,'remote_branch':branch,'checkpoint_policy':plan['checkpoint_policy'],
        'pause_reason':pause_reason,'retry_after_seconds':retry_after_seconds}

def execute(args):
    from e2_search_session import SearchSession,SessionPaused,UnresolvedAttempt
    from e2_github_campaign import GitHubCampaignMirror,CampaignSyncError
    from e2_temporal_training import SelectionAborted
    if args.github_repository!=admission.REPOSITORY:raise ValueError('registered_repository_required')
    if not os.environ.get('GITHUB_TOKEN'):raise ValueError('github_secret_required')
    verify_source(args.source_commit)
    registered=json.loads(admission.REGISTRATION.read_text())
    if (registered.get('status')!='registered_before_scientific_cuda_fits'
            or not registered.get('implementation_sha256')):
        raise ValueError('published_scientific_admission_required')
    hardware=cuda_hardware()
    plan=admission.build_plan(source_commit=args.source_commit,models=args.models,origins=args.origins,
        session_hours=args.session_hours,campaign_hours=args.campaign_hours,storage_gib=args.storage_gib,
        device=args.device,hardware=hardware,expected_registration_sha256=args.expected_registration_sha256,
        github_branch=args.github_branch,github_prefix=args.github_prefix)
    directory=Path(args.output)
    mirror=GitHubCampaignMirror(args.github_repository,args.github_branch,args.github_prefix,
        max_snapshot_bytes=min(int(args.storage_gib*1024**3),2*1024**3))
    mirror.restore(directory)
    session=None;status='not_started';selections=[];exit_code=0
    pause_reason=None;retry_after_seconds=None
    try:
        session=SearchSession(directory,plan,session_budget_seconds=args.session_hours*3600,
            campaign_budget_seconds=args.campaign_hours*3600,max_storage_bytes=int(args.storage_gib*1024**3))
        if args.extend_campaign_hours is not None:
            target=admission.positive(args.extend_campaign_hours,'extension_hours')*3600
            if target>session.campaign_budget_seconds:
                session.extend_campaign_budget(target,reason='operator_requested_budget_extension')
            elif target<session.campaign_budget_seconds:
                raise ValueError('cannot_reduce_an_existing_budget_extension')
        paths=session.pending_record_paths
        mirror.push(directory,paths=paths,message='Seal CUDA selection plan before market fits')
        session.mark_records_synced(paths)
        panel,_=prepare_panel(args.data_root)
        status='running'
        def worker(job,deadline):
            return isolated_trial(plan,job,data_root=args.data_root,source_root_itransformer=args.source_root_itransformer,deadline=deadline)
        run_selection(plan,session,panel,worker=worker,mirror=mirror,progress=selections)
        status='completed_selected_registered_domains'
    except SessionPaused as error:
        status='paused_'+error.reason;pause_reason=error.reason
    except UnresolvedAttempt:
        status='reconciliation_required';exit_code=3
    except CampaignSyncError as error:
        status='paused_github_sync';exit_code=4
        pause_reason=error.code;retry_after_seconds=getattr(error,'retry_after_seconds',None)
    except SelectionAborted:
        status='blocked_selection_failure';exit_code=5
    except KeyboardInterrupt:
        status='interrupted_between_jobs';pause_reason='operator_interruption';exit_code=130
    except (ValueError,RuntimeError,OSError):
        status='blocked_runtime_failure';pause_reason='runtime_failure';exit_code=2
    finally:
        if session is not None:
            session.close()
            state=report(plan,session,status,selections,args.github_repository,args.github_branch)
            state.update(pause_reason=pause_reason,retry_after_seconds=retry_after_seconds)
            write_json(directory/'status.json',state)
            try:
                mirror.push(directory,paths=(*session.pending_record_paths,'status.json'),message='Save campaign pause or completion')
            except CampaignSyncError as error:
                status='paused_github_sync';exit_code=4
                state.update(status=status,pause_reason=pause_reason or error.code,
                    retry_after_seconds=retry_after_seconds if retry_after_seconds is not None else getattr(error,'retry_after_seconds',None))
                write_json(directory/'status.json',state)
            print(json.dumps({'status':status,'completed_jobs':state['completed_jobs'],
                'remote_repository':args.github_repository,'remote_branch':args.github_branch,
                'pause_reason':state['pause_reason'],'retry_after_seconds':state['retry_after_seconds'],
                'final_market_values_read':False,'final_R':None},ensure_ascii=False))
        else:
            print(json.dumps({'status':status,'completed_jobs':None,'remote_repository':args.github_repository,
                'remote_branch':args.github_branch,'final_market_values_read':False,'final_R':None}))
    return exit_code

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan-only',action='store_true');mode.add_argument('--execute',action='store_true')
    mode.add_argument('--worker-spec',type=Path)
    parser.add_argument('--source-commit');parser.add_argument('--expected-registration-sha256')
    parser.add_argument('--output',type=Path);parser.add_argument('--data-root',type=Path)
    parser.add_argument('--source-root-itransformer');parser.add_argument('--models',nargs='+')
    parser.add_argument('--origins',nargs='+');parser.add_argument('--cohort',choices=['large'],default='large')
    parser.add_argument('--session-hours',type=float,default=8);parser.add_argument('--campaign-hours',type=float,default=24)
    parser.add_argument('--storage-gib',type=float,default=10);parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--extend-campaign-hours',type=float)
    parser.add_argument('--github-repository',default=admission.REPOSITORY)
    parser.add_argument('--github-branch',default='campaigns/e2-cuda-selection-v1')
    parser.add_argument('--github-prefix',default='campaigns/e2-cuda-selection-v1')
    args=parser.parse_args(argv)
    if args.worker_spec:return run_worker(args.worker_spec)
    if args.plan_only:
        value=admission.build_plan(source_commit=args.source_commit,models=args.models,origins=args.origins,
            session_hours=args.session_hours,campaign_hours=args.campaign_hours,storage_gib=args.storage_gib,
            device=args.device,expected_registration_sha256=args.expected_registration_sha256,
            github_branch=args.github_branch,github_prefix=args.github_prefix)
        print(json.dumps(admission.plan_summary(value),ensure_ascii=False,indent=2));return 0
    if args.output is None or args.data_root is None:parser.error('--output and --data-root are required for execution')
    return execute(args)

if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RuntimeError,OSError) as error:
        code=str(error) if re.fullmatch(r'[a-z][a-z0-9_]{1,95}',str(error)) else 'runtime_preflight_failed'
        print(json.dumps({'status':'blocked_preflight','error_code':code,
            'detail':'Review the pinned plan, source, credentials and local campaign state. No automatic fit retry.'}),file=sys.stderr)
        raise SystemExit(2)
