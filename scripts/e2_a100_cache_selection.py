"""Continue unchanged E fits on A100 while auditing historical T4 cache honestly.

A historical compatibility profile is reconstructed only from previously
completed, cost-bound T4 work and a preserved cache-policy attestation. It is
never presented as the observed A100 runtime. The sole lost opportunity is
accounted by a separately registered recovery helper and is never retried.
"""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import re
import sys

import e2_resource_failure_selection as resource
setup=resource.setup
original=resource.original
ENGINE_COMMIT='1e8745497b484518f766b6da5c19be8362d9add7'
LOST_JOB='f086a89ea56abb834542d12ccee473a6386d12b9de8f456043d78010ada01ede'
TRANSITION_VERSION='e2-a100-historical-cache-transition-v1'


def _read(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('A regular bound migration artifact is required')
    return json.loads(path.read_bytes())


def _strict_current(profile):
    runtime=profile.get('runtime',{})
    if (profile.get('fit_execution_policy_verified') is not True
            or 'A100' not in str(runtime.get('device_name',''))
            or runtime.get('device')!='cuda:0'
            or runtime.get('python')!='3.12.12'
            or runtime.get('torch')!='2.8.0+cu126'
            or runtime.get('torch_cuda')!='12.6'
            or runtime.get('cpu_threads')!=2 or runtime.get('interop_threads')!=1
            or runtime.get('dtype')!='float32' or runtime.get('amp') is not False
            or runtime.get('deterministic') is not True
            or runtime.get('tf32') is not False or runtime.get('cudnn_tf32') is not False):
        raise ValueError('A truthful pinned strict A100 profile is required for new work')
    return runtime


def _historical_profile(output,budget,plans):
    """Corroborate historical facts and policy without claiming a fresh T4 probe."""
    import audit_e2_reduced_cache as audit
    output=Path(output);source=original.source_identity();candidate=None
    for path in sorted((output/'jobs').glob('*/worker_spec.json')):
        result_path=path.parent/'result.json'
        if not result_path.is_file():continue
        result=_read(result_path)
        if result.get('status')!='completed' or result.get('cache_reuse'):continue
        spec=_read(path);runtime=spec.get('runtime_identity',{})
        if runtime.get('device_name')!='Tesla T4':continue
        if (spec.get('source_sha256')!=source or result.get('worker_spec_sha256')!=original.digest(spec)
                or spec.get('job')!=result.get('job')):
            raise ValueError('Historical completed T4 source/spec binding differs')
        charge=budget.completed_receipt(spec['job']['logical_job_id'],expected_resource_use={
            'worker_spec_sha256':original.digest(spec),'result_sha256':original.file_sha(result_path)})
        if charge is None:raise ValueError('Historical T4 work lacks its completed cost')
        if any(any(runtime.get(k)!=plan.get('hardware_binding',{}).get(k)
                   for k in audit.RUNTIME_FACT_KEYS) for plan in plans):
            raise ValueError('Historical T4 facts differ from the original plans')
        if any(plan.get('execution')!=plans[0].get('execution') for plan in plans):
            raise ValueError('Historical execution policies differ')
        candidate=(runtime,{'worker_spec_path':str(path.relative_to(output)),
            'worker_spec_file_sha256':original.file_sha(path),
            'result_path':str(result_path.relative_to(output)),
            'result_file_sha256':original.file_sha(result_path),
            'budget_charge_sha256':charge['sha256']})
        break
    if candidate is None:raise ValueError('No previously completed byte-bound T4 fit is available')
    attestation=None
    for path in sorted((output/'jobs').glob('*/cache_admission.json')):
        result_path=path.parent/'result.json'
        if not result_path.is_file():continue
        result=_read(result_path);admission=_read(path)
        checks=admission.get('checks',{})
        if (result.get('status')=='completed' and result.get('cache_reuse') is True
                and result.get('artifacts_sha256',{}).get('cache_admission.json')==original.file_sha(path)
                and admission.get('reuse_approved') is True
                and admission.get('runtime_facts_match') is True
                and admission.get('fit_execution_policy_attested') is True
                and admission.get('fit_execution_policy_match') is True
                and set(checks)=={'candidate_payload','recipe_identity','complete_validation_calendar',
                    'known_completed_updates','prepared_inputs','job_payload','worker_runtime'}
                and all(v is True for v in checks.values())):
            attestation={'cache_admission_path':str(path.relative_to(output)),
                'cache_admission_file_sha256':original.file_sha(path),
                'cached_result_path':str(result_path.relative_to(output)),
                'cached_result_file_sha256':original.file_sha(result_path)}
            break
    if attestation is None:raise ValueError('A preserved historical fitting-policy cache attestation is required')
    runtime,proof=candidate
    profile={'version':'e2-historical-t4-compatibility-profile-v1',
        'runtime':deepcopy(runtime),'fit_execution_policy_verified':True,
        'fit_execution_policy':deepcopy(plans[0]['execution']),
        'attestation_scope':'Previously executed T4 policy corroborated by preserved results; not an observation of the current GPU',
        'current_gpu_is_not_claimed_to_be_T4':True,
        'historical_runtime_source':proof,'historical_policy_attestation':attestation}
    return profile


def prepare_historical_cache_audit(args,budget,*,registration_commit,transition_directory=None):
    """Audit old results against old facts and keep the actual A100 profile intact."""
    import audit_e2_reduced_cache as audit
    if not re.fullmatch(r'[0-9a-f]{40}',str(registration_commit)):
        raise ValueError('An immutable hardware-transition registration is required')
    output=Path(args.output);actual_path=Path(args.runtime_profile)
    actual=_read(actual_path);_strict_current(actual)
    transition=output/'hardware_transition' if transition_directory is None else Path(transition_directory)
    if not transition.resolve().is_relative_to(output.resolve()):
        raise ValueError('Migration metadata must remain in the control output')
    receipt_path=transition/'transition.json'
    if not receipt_path.exists():
        for parent in ('pilot','final-fit'):
            directory=output/'final'/parent
            if directory.exists() and any(p.is_file() for p in directory.rglob('*')):
                raise ValueError('Initial migration requires final/pilot fitting not to have started')
    reconciliation=_read(Path(args.reconciliation_root)/'budget_evidence.json')
    plans=[]
    for branch in (audit.LEGACY_BRANCH,audit.OPTIMIZED_BRANCH):
        head=reconciliation['historical_heads'][branch]
        directory=Path(args.reconciliation_root)/'snapshots'/branch.rsplit('/',1)[-1]/head
        plans.append(_read(directory/'campaign.json')['plan'])
    historical=_historical_profile(output,budget,plans)
    if actual.get('fit_execution_policy')!=historical['fit_execution_policy']:
        raise ValueError('The new GPU must retain the historical fitting policy')
    historical_path=transition/'historical_t4_profile.json'
    audit_path=transition/'historical_cache_audit.json'
    original.write_json(historical_path,historical)
    domains=_read(original.ROOT/original.CONFIGURATIONS_PATH)
    context=audit.build_development_context(original.ROOT)
    report=audit.audit(args.reconciliation_root,original.ROOT,domains,
        prepared_context=context,runtime_profile=historical)
    original.write_json(audit_path,report)
    receipt={'version':TRANSITION_VERSION,'engine_commit':ENGINE_COMMIT,
        'registration_commit':registration_commit,
        'historical_runtime_profile_sha256':original.file_sha(historical_path),
        'current_runtime_profile_sha256':original.file_sha(actual_path),
        'cache_audit_sha256':original.file_sha(audit_path),
        'historical_runtime_source':historical['historical_runtime_source'],
        'historical_policy_attestation':historical['historical_policy_attestation'],
        'historical_heads':reconciliation['historical_heads'],
        'historical_cache_reuse_scope':'Validation metadata from actual T4 fits; no historical weights or final scores',
        'new_work_runtime':'Actual A100 profile only',
        'final_and_pilot_hardware_policy':'All newly admitted final fits/pilots on the A100 runtime',
        'historical_scores_modified':False,'optimizer_retry_performed':False,
        'budget_reset':False,'authorized_total_hours':72,'final_market_values_read':False}
    receipt['sha256']=original.digest(receipt)
    original.write_json(receipt_path,receipt)
    args.historical_runtime_profile=historical_path;args.cache_audit=audit_path
    args.hardware_transition_receipt=receipt_path
    args.hardware_transition_registration_commit=registration_commit
    return {'historical_runtime_profile':historical_path,'cache_audit':audit_path,
        'hardware_transition_receipt':receipt_path,'historical_cache_reuse_approved':report['reuse_approved']}


def validate_hardware_transition(args,cache_report):
    actual=_read(args.runtime_profile);_strict_current(actual)
    historical=_read(args.historical_runtime_profile);receipt=_read(args.hardware_transition_receipt)
    if (receipt.get('version')!=TRANSITION_VERSION or receipt.get('engine_commit')!=ENGINE_COMMIT
            or receipt.get('sha256')!=original.digest({k:v for k,v in receipt.items() if k!='sha256'})
            or receipt.get('registration_commit')!=args.hardware_transition_registration_commit
            or receipt.get('current_runtime_profile_sha256')!=original.file_sha(args.runtime_profile)
            or receipt.get('historical_runtime_profile_sha256')!=original.file_sha(args.historical_runtime_profile)
            or receipt.get('cache_audit_sha256')!=original.file_sha(args.cache_audit)
            or _read(args.cache_audit)!=cache_report
            or historical.get('runtime',{}).get('device_name')!='Tesla T4'
            or historical.get('fit_execution_policy_verified') is not True
            or historical.get('fit_execution_policy')!=actual.get('fit_execution_policy')
            or receipt.get('historical_scores_modified') is not False
            or receipt.get('optimizer_retry_performed') is not False
            or receipt.get('budget_reset') is not False or receipt.get('authorized_total_hours')!=72):
        raise ValueError('Historical-cache/current-hardware transition binding changed')
    for group,keys in [('historical_runtime_source',(('worker_spec_path','worker_spec_file_sha256'),('result_path','result_file_sha256'))),
                       ('historical_policy_attestation',(('cache_admission_path','cache_admission_file_sha256'),('cached_result_path','cached_result_file_sha256')))]:
        proof=receipt.get(group,{})
        if proof!=historical.get(group):raise ValueError('Historical runtime provenance differs')
        for path_key,hash_key in keys:
            relative=proof.get(path_key)
            if not isinstance(relative,str):raise ValueError('Historical proof path is required')
            path=Path(args.output)/relative
            if not path.resolve().is_relative_to(Path(args.output).resolve()) or original.file_sha(path)!=proof.get(hash_key):
                raise ValueError('Historical runtime provenance artifact changed')
    old_spec=_read(Path(args.output)/receipt['historical_runtime_source']['worker_spec_path'])
    if old_spec.get('runtime_identity')!=historical.get('runtime'):
        raise ValueError('Historical T4 profile is not its original worker identity')
    return actual,historical


def run_selection(args, budget, cache_report, reconciliation_root, *, persist_result=None):
    """Run admitted jobs sequentially; persist/backup before admitting another fit."""
    from run_e2_cuda_search import prepare_panel
    from e2_temporal_training import prepare_b
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    original.verify_reduced_registration(original.ROOT,require_git=True)
    runtime_profile, historical_runtime_profile = validate_hardware_transition(args, cache_report)
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
        if job['logical_job_id'] == LOST_JOB:
            from e2_expired_lease_recovery import verify_recovery_disposition
            row,result=verify_recovery_disposition(job,directory,budget,source,
                prepared[job['fold_origin']].inner_validation.dates)
            if row_path.exists() and json.loads(row_path.read_text()) != row:
                raise ValueError('The registered unknown-outcome row changed')
            original.write_json(row_path,row)
        elif job['logical_job_id'] == setup.KNOWN_JOB:
            row, result = setup.known_failure_row(job, directory, budget, source,
                prepared[job['fold_origin']].inner_validation.dates)
            if row_path.exists() and json.loads(row_path.read_text()) != row:
                raise ValueError('Saved known failure selection row changed')
            original.write_json(row_path, row)
        elif (directory/'failure_result.json').exists():
            row, result = resource.resource_failure_row(job, directory, budget, source,
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
                result=original._cached_result(job,match,reconciliation_root,directory,prepared[job['fold_origin']],historical_runtime_profile)
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
                        row,result=resource.resource_failure_row(job,directory,budget,source,
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
        if persist_result is not None and job['logical_job_id'] != LOST_JOB:
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


