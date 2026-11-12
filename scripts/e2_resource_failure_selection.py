"""Account closed HPO wall-limit failures and continue planned opportunities.

Only an E worker's fully evidenced synchronized wall-limit failure is admitted.
It remains failed, retains its cost and every original artifact, and is never
retried. Unknown failures and incomplete accounting still stop admission.
"""
from __future__ import annotations
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

import e2_known_setup_failure_selection as setup
original=setup.original
WALL_MESSAGE='Pilot wall limit exceeded at a synchronized boundary'


def _closed_phase_receipt(directory, request, resource, artifacts):
    """Verify the terminal journal without loading a weight into a model."""
    from e2_amended_pilot_artifacts import PilotArtifactStore
    failed=[]
    for path in sorted((directory/'phases').glob('*/manifest.json')):
        phase=path.parent.name
        identity=json.loads((path.parent/'identity.json').read_text())
        bound=identity.get('identity',{})
        if (bound!={'request_sha256':original.digest(request),'seed':request['seed'],'phase':phase}
                or original.digest(bound)!=identity.get('identity_sha256')
                or identity.get('attempt_id')!=phase):
            raise ValueError('Resource failure phase/request identity changed')
        saved=json.loads(path.read_text())
        if saved.get('state')=='completed':
            from e2_reduced_fit import verify_phase_archive
            verify_phase_archive(path.parent)
            continue
        reader=object.__new__(PilotArtifactStore)
        reader.attempt_id=phase;reader.identity_sha256=identity['identity_sha256']
        state=reader._empty_state()
        raw=(path.parent/'events.jsonl').read_bytes()
        if not raw or not raw.endswith(b'\n'):
            raise ValueError('Resource failure phase journal is incomplete')
        for line in raw.splitlines():
            event=json.loads(line); checksum=event.pop('sha256')
            if (original.digest(event)!=checksum or event.get('sequence')!=state['event_count']
                    or event.get('previous_sha256')!=state['last_event_sha256']
                    or event.get('identity_sha256')!=reader.identity_sha256):
                raise ValueError('Resource failure phase journal changed')
            event['sha256']=checksum;state=reader._advance(state,event)
        prepared=json.loads((path.parent/'prepared.json').read_text())
        if (state!=saved or state.get('state')!='failed'
                or state.get('terminal_metadata')!=resource
                or state.get('optimizer_step_open') is not False
                or state.get('optimizer_steps_known_lower_bound')!=resource['optimizer_steps']
                or original.digest(prepared)!=state.get('prepared_metadata_sha256')
                or prepared.get('config')!=request['config']
                or prepared.get('source')!=request['sources']
                or prepared.get('seed')!=request['seed']):
            raise ValueError('Resource failure terminal phase binding changed')
        for checkpoint in state.get('checkpoint_history',[]):
            relative='phases/'+phase+'/'+checkpoint['path']
            if artifacts.get(relative)!=checkpoint['sha256']:
                raise ValueError('Resource failure checkpoint inventory changed')
        for label,checkpoint in resource.get('checkpoint_receipts',{}).items():
            if state.get('checkpoints',{}).get(label)!=checkpoint:
                raise ValueError('Resource failure checkpoint receipt changed')
        failed.append(phase)
    if len(failed)!=1:
        raise ValueError('Resource failure needs exactly one fully closed failed phase')
    return failed[0]


def resource_failure_row(job, directory, budget, source, validation_dates):
    """Return an explicit failed row only for a proven terminal wall timeout."""
    directory=Path(directory)
    if job.get('stage')!='hpo' or job.get('seed')!=original.HPO_SEED or len(validation_dates)!=83:
        raise ValueError('Resource-failure continuation requires the original HPO calendar')
    if any((directory/name).exists() for name in ('result.json','job_result.json')):
        raise ValueError('Failed opportunity cannot also contain a completed result')
    failure=json.loads((directory/'failure_result.json').read_text())
    worker=json.loads((directory/'worker_failure.json').read_text())
    spec=json.loads((directory/'worker_spec.json').read_text())
    request=json.loads((directory/'request.json').read_text())
    spec_sha=original.digest(spec)
    if (failure.get('status')!='failed' or failure.get('attempt_id')!=job['logical_job_id']
            or failure.get('job')!=job or failure.get('automatic_retry') is not False
            or failure.get('worker_spec_sha256')!=spec_sha
            or spec.get('job')!=job or spec.get('source_sha256')!=source
            or spec.get('version')!=original.VERSION
            or spec.get('fit_wall_seconds') not in (600,1800)
            or worker.get('status')!='failed' or worker.get('error_type')!='TrainingFailure'
            or worker.get('reason')!=WALL_MESSAGE or worker.get('automatic_retry') is not False
            or request.get('name')!=job['model_id'] or request.get('trial')!=job['trial']
            or request.get('seed')!=job['seed'] or request.get('sources')!=source):
        raise ValueError('Unknown or differently bound worker failure requires inspection')
    resource=worker.get('resource_receipt',{})
    config=resource.get('config',{})
    steps=resource.get('optimizer_steps')
    elapsed=resource.get('elapsed_seconds')
    if (resource.get('version')!='e2-published-pilot-training-v1'
            or resource.get('status')!='failed' or resource.get('error_type')!='TimeoutError'
            or resource.get('error_code')!='resource_limit' or resource.get('message')!=WALL_MESSAGE
            or resource.get('partial_optimizer_update_may_have_occurred') is not False
            or resource.get('retry_performed') is not False
            or resource.get('resume_exact_supported') is not False
            or resource.get('requires_reconciliation') is not True
            or resource.get('device')!='cuda:0' or resource.get('seed')!=job['seed']
            or config!=request.get('config') or config.get('seed')!=job['seed']
            or config.get('wall_limit_seconds')!=spec['fit_wall_seconds']
            or config.get('max_epochs')!=200 or config.get('patience')!=20
            or any(config.get(k)!=job['trial'][k]
                   for k in ('learning_rate','weight_decay','batch_size'))
            or type(steps) is not int or steps<0 or resource.get('completed_optimizer_steps')!=steps
            or resource.get('optimizer_update_occurred') is not (steps>0)
            or resource.get('epochs_completed')!=len(resource.get('logs',[]))
            or type(elapsed) not in (int,float) or not math.isfinite(elapsed)
            or resource.get('known_cost_seconds')!=elapsed
            or elapsed<=config['wall_limit_seconds']):
        raise ValueError('Resource failure is not a closed synchronized wall-limit outcome')
    artifacts=failure.get('artifacts_sha256',{})
    if not isinstance(artifacts,dict) or not {'request.json','worker.log','worker_failure.json','worker_spec.json'}<=artifacts.keys():
        raise ValueError('Resource failure artifact inventory is incomplete')
    for relative,expected in artifacts.items():
        pure=PurePosixPath(relative)
        target=directory.joinpath(*pure.parts)
        if (pure.is_absolute() or '..' in pure.parts or str(pure)!=relative
                or not re.fullmatch(r'[0-9a-f]{64}',str(expected))
                or any(p.is_symlink() for p in (target,*target.parents) if p.is_relative_to(directory))
                or not target.is_file() or original.file_sha(target)!=expected):
            raise ValueError('Resource failure artifact changed')
    allowed=set(artifacts)|{'failure_result.json','selection_row.json','github_backup_receipt.json'}
    if any(p.is_symlink() or (p.is_file() and p.name!='.writer.lock' and str(p.relative_to(directory)) not in allowed)
           for p in directory.rglob('*')):
        raise ValueError('Unexpected output in the closed resource failure')
    backup_path=directory/'github_backup_receipt.json'
    if backup_path.exists():
        # The controller's preceding restore verifies the immutable manifest
        # and its complete asset hashes. This only permits its matching local
        # transport receipt; the failed scientific receipt stays unchanged.
        backup=json.loads(backup_path.read_text())
        if (backup.get('status')!='backed_up' or backup.get('attempt_id')!=job['logical_job_id']
                or backup.get('repository')!='jassan-bit/DeepLearning_202630'
                or backup.get('branch')!='results/e2-reduced-delivery-v1'
                or backup.get('manifest_path')!='runs/'+job['logical_job_id']+'/manifest.json'
                or backup.get('artifact_count')!=len(artifacts)
                or backup.get('retry_scientific_fit') is not False
                or not re.fullmatch(r'[0-9a-f]{64}',str(backup.get('manifest_sha256','')))):
            raise ValueError('Resource failure transport receipt identity changed')
    phase=_closed_phase_receipt(directory,request,resource,artifacts)
    charges=[r for r in setup._budget_events(budget)
             if r.get('kind')=='charged' and r.get('attempt_id')==job['logical_job_id']]
    if len(charges)!=1:
        raise ValueError('Resource failure needs exactly one canonical cost receipt')
    charge=charges[0]
    checksum=charge.get('sha256')
    if (original.digest({k:v for k,v in charge.items() if k!='sha256'})!=checksum
            or charge.get('status')!='failed' or charge.get('reservation_overrun') is not False
            or charge.get('resource_use',{}).get('worker_spec_sha256')!=spec_sha
            or charge.get('resource_use',{}).get('failure')!='worker_exit_2'
            or charge.get('resource_use',{}).get('automatic_retry') is not False
            or type(charge.get('elapsed_seconds')) not in (int,float)
            or not math.isfinite(charge['elapsed_seconds']) or charge['elapsed_seconds']<elapsed):
        raise ValueError('Resource failure canonical charge binding changed')
    supervision=failure.get('supervision',{})
    if (supervision.get('worker_spec_sha256')!=spec_sha
            or supervision.get('failure')!='worker_exit_2'
            or supervision.get('automatic_retry') is not False
            or supervision.get('budget_receipt')!=charge):
        raise ValueError('Resource failure receipt differs from the canonical charge')
    row={k:job[k] for k in ('model_id','configuration_id','fold_origin','seed','parent_trial_id','configuration_sha256')}
    row.update(status='failed',rmse_original=None,best_epoch=None,fatal=False,systematic_failure=False,
        validation_count=83,predictions_produced=0,validation_count_scope='admitted_calendar_not_predictions',
        reason='Synchronized per-phase wall limit reached; original opportunity closed without retry',
        failure_code='resource_limit',failed_phase=phase,
        run_receipt_sha256=original.file_sha(directory/'failure_result.json'),run_receipt_file='failure_result.json',
        budget_receipt_sha256=checksum,replacement_performed=False,automatic_retry=False,
        resource_use={'cache_reuse':False,'optimizer_steps':steps,
            'optimizer_steps_scope':'exact_completed_steps_in_preserved_terminal_failure_receipt',
            'original_supervision_preserved':True,'new_training_performed':False})
    return row,failure


def run_selection(args, budget, cache_report, reconciliation_root, *, persist_result=None):
    """Run admitted jobs sequentially; persist/backup before admitting another fit."""
    from run_e2_cuda_search import prepare_panel
    from e2_temporal_training import prepare_b
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    original.verify_reduced_registration(original.ROOT,require_git=True)
    runtime_profile=json.loads(Path(args.runtime_profile).read_text())
    if runtime_profile.get('fit_execution_policy_verified') is not True:
        raise ValueError('Actual fitting-policy attestation is required before cache or fits')
    domains=original.current_domains()
    jobs=original.job_plan(getattr(args,'models',None),domains=domains)
    panel,_=prepare_panel(args.data_root)
    prepared={f:prepare_b(panel,f,f) for f in original.FOLDS}
    approved={(r['model_id'],r['origin'],r['configuration_id']):r for r in cache_report.get('fits',[]) if r.get('reuse_approved')}
    source=original.source_identity()
    ledger=[]
    for job in jobs:
        directory=output/'jobs'/job['logical_job_id']
        row_path=directory/'selection_row.json'
        if job['logical_job_id'] == setup.KNOWN_JOB:
            row, result = setup.known_failure_row(job, directory, budget, source,
                prepared[job['fold_origin']].inner_validation.dates)
            if row_path.exists() and json.loads(row_path.read_text()) != row:
                raise ValueError('Saved known failure selection row changed')
            original.write_json(row_path, row)
        elif (directory/'failure_result.json').exists():
            row, result = resource_failure_row(job, directory, budget, source,
                prepared[job['fold_origin']].inner_validation.dates)
            if row_path.exists() and json.loads(row_path.read_text()) != row:
                raise ValueError('Saved resource failure selection row changed')
            original.write_json(row_path,row)
        elif row_path.exists():
            row=json.loads(row_path.read_text())
            result=json.loads((directory/'result.json').read_text())
            if row['run_receipt_sha256']!=original.file_sha(directory/'result.json'):
                raise ValueError('Saved selection row/result binding changed')
            for relative,expected in result['artifacts_sha256'].items():
                if original.file_sha(directory/relative)!=expected:raise ValueError('Saved selection artifact changed')
            if result.get('cache_reuse') and (job['model_id'],job['fold_origin'],job['configuration_id']) not in approved:
                raise ValueError('Saved cache row is no longer admitted by the current runtime audit')
            if not result.get('cache_reuse') and json.loads((directory/'worker_spec.json').read_text())['source_sha256']!=source:
                raise ValueError('Saved HPO worker closure changed; reconcile before combining recipes')
            if not result.get('cache_reuse') and budget.completed_receipt(job['logical_job_id'],
                expected_resource_use={'result_sha256':original.file_sha(directory/'result.json'),
                    'worker_spec_sha256':result['worker_spec_sha256']}) is None:
                raise RuntimeError('Saved selection lacks its matching budget completion')
        else:
            match=approved.get((job['model_id'],job['fold_origin'],job['configuration_id']))
            if match is not None:
                result=original._cached_result(job,match,reconciliation_root,directory,prepared[job['fold_origin']],runtime_profile)
            else:
                runtime=runtime_profile['runtime']
                spec={'version':original.VERSION,'job':job,'output':str(directory),'data_root':str(args.data_root),
                    'source_roots':str(args.source_roots),'source_sha256':source,'runtime_identity':runtime,
                    'fit_wall_seconds':budget.stage_wall_seconds}
                original.write_json(directory/'worker_spec.json',spec)
                print(json.dumps({'event':'fit_admission_requested','model':job['model_id'],'origin':job['fold_origin'],
                    'configuration_id':job['configuration_id'],'stage':'reduced_hpo','seed':job['seed']}),flush=True)
                try:
                    run=original.supervise_command([getattr(args,'python',sys.executable),'-u',Path(original.__file__),
                        '--worker-spec',directory/'worker_spec.json'],budget,job['logical_job_id'],directory,original.digest(spec),
                        phases=3 if job['model_id']=='LSTM-autoencoder con predictor' else 1,cwd=original.ROOT)
                except original.ReducedWorkerFailure as error:
                    failed={'version':original.VERSION,'status':'failed','attempt_id':job['logical_job_id'],'job':job,
                        'worker_spec_sha256':original.digest(spec),'supervision':error.receipt,'automatic_retry':False,
                        'artifacts_sha256':{str(p.relative_to(directory)):original.file_sha(p) for p in directory.rglob('*')
                            if p.is_file() and p.name not in ('failure_result.json','.writer.lock')}}
                    original.write_json(directory/'failure_result.json',failed)
                    try:
                        row,result=resource_failure_row(job,directory,budget,source,
                            prepared[job['fold_origin']].inner_validation.dates)
                    except Exception:
                        # Preserve unknown failures before stopping; classifying
                        # a failure never authorizes a second optimizer attempt.
                        if persist_result is not None:
                            persist_result(directory,failed)
                        raise
                    original.write_json(row_path,row)
                    if persist_result is not None:
                        persist_result(directory,result)
                    ledger.append(row)
                    print(json.dumps({'event':'resource_limited_fit_accounted','model':job['model_id'],
                        'origin':job['fold_origin'],'configuration_id':job['configuration_id'],
                        'status':'failed','automatic_retry':False,'next_planned_opportunity':True,
                        'remaining_seconds':budget.remaining_seconds}),flush=True)
                    continue
                result=run['result']
            row=original._row(job,result,original.file_sha(directory/'result.json'))
            original.write_json(row_path,row)
        if persist_result is not None:
            persist_result(directory,result)
        ledger.append(row)
        print(json.dumps({'model':job['model_id'],'origin':job['fold_origin'],'configuration_id':job['configuration_id'],
            'status':row['status'],'cache_reuse':result.get('cache_reuse',False),'remaining_seconds':budget.remaining_seconds}),flush=True)
    selections={}
    for name in dict.fromkeys(j['model_id'] for j in jobs):
        rows=[r for r in ledger if r['model_id']==name]
        selected=original._tide_selection(rows,domains[name]) if name=='TiDE' else original.select_reduced_hpo(rows,name,domains[name])
        slot=next(s for s in domains[name] if s['configuration_id']==selected['winner_configuration_id'])
        selected.update(version=original.VERSION,winner_trial=slot['trial'],run_artifact_bytes_verified=True,
            selection_ledger_sha256=original.digest(rows),source_current_recipe_verified=True,criteria_only=False,
            new_training_performed=any(r['status']=='completed' and not r['resource_use'].get('cache_reuse') for r in rows))
        original.write_json(output/'selections'/(original.slug(name)+'.json'),selected)
        selections[name]=selected
    receipt={'version':original.VERSION,'status':'completed' if set(selections)==set(original.NEURAL) else 'partial_models_completed',
        'selections':selections,'required_models':list(original.NEURAL),'completed_models':list(selections),
        'TiDE_architecture_blocked':'TiDE' not in selections,'final_benchmark_complete':False,
        'final_market_values_read':False,'selection_ledger_sha256':original.digest(ledger),'budget':budget.report()}
    original.write_json(output/'selection_complete.json',receipt)
    return receipt


