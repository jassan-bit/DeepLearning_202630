"""Execute the registered six-slot E2 HPO, with durable per-fit GPU workers.

No final608-date scoring or silent retry is implemented. TiDE requires its new
published-body extension. Legacy predictions are raw HPO metadata and are used
only after the full audit approves their source/data/runtime/policy identity.
"""
from __future__ import annotations
import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

from e2_reduced_protocol import (ROOT, NEURAL, FOLDS, HPO_SEED, PARENT_INDICES, SLOT_IDS,
    CONFIGURATIONS_PATH, digest, canonical, select_reduced_hpo, verify_reduced_registration)
from e2_reduced_fit import file_sha, write_json, fit_reduced_job
from e2_reduced_supervision import supervise_command, ReducedWorkerFailure
VERSION = 'e2-reduced-selection-executor-v1'


def source_identity():
    # Hash only the transitive local import closure. A later final-stage module
    # cannot retroactively change a saved HPO worker's scientific identity.
    scripts = ROOT / 'scripts'
    pending = ['run_e2_reduced_selection']
    seen = set()
    while pending:
        name = pending.pop()
        path = scripts / (name + '.py')
        if name in seen or not path.is_file():
            continue
        seen.add(name)
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [item.name.split('.')[0] for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = ([node.module.split('.')[0]] if node.module else []) + [item.name for item in node.names]
            pending.extend(n for n in names if (scripts / (n + '.py')).is_file())
    return {'scripts/' + n + '.py': file_sha(scripts / (n + '.py')) for n in sorted(seen)}


def slug(name):
    return re.sub('[^A-Za-z0-9]+', '-', name).strip('-')


def worker_preflight_policy():
    """Apply and inspect the actual fitting context, without opening market data.

    This certifies flags of the pinned policy implementation, not acceptance of
    all model kernels. Each subsequent fit records its own applied runtime.
    """
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    import e2_cuda_training as cuda
    from check_e2_reduced_environment import inspect_runtime
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    device = cuda.validate_cuda_device('cuda:0')
    with cuda._execution_policy(device, True) as policy:
        actual = inspect_runtime(torch)
        if (not actual['production_policy_compatible'] or actual['cpu_threads'] != 2
                or actual['interop_threads'] != 1 or not actual['deterministic']
                or actual['tf32'] or actual['cudnn_tf32']):
            raise ValueError('Actual strict fitting policy did not match the registered runtime')
        actual['flags_scope'] = 'Observed inside the pinned strict fitting-policy context; no model fit executed'
        registered = {'amp': False, 'cpu_threads': 2, 'deterministic': True, 'device': 'cuda:0',
            'dtype': 'float32', 'gpu_memory_cap_bytes': 12*1024**3, 'gpu_memory_fraction': .8,
            'interop_threads': 1, 'tf32': False}
        return {'version': VERSION, 'runtime': actual, 'fit_execution_policy_verified': True,
            'fit_execution_policy': registered, 'actual_applied_context': policy,
            'training_policy_sources': {p: file_sha(ROOT / p) for p in
                ('scripts/e2_cuda_training.py', 'scripts/e2_published_pilot_training.py',
                 'scripts/e2_reduced_fit.py', 'scripts/run_e2_reduced_selection.py')},
            'all_model_kernels_validated': False, 'market_training_started': False,
            'scope': 'Applied pinned fitting policy inspected in its real CUDA context; no model fit/kernel acceptance claim'}


def current_domains():
    domains = json.loads((ROOT / CONFIGURATIONS_PATH).read_text())
    # The frozen TiDE rows remain provenance. A separately registered extension
    # may supply a new published recipe and its six newly bound optimizer rows.
    extension = ROOT / 'experiments/e2_tide_published_r0_20261112/configurations.json'
    if extension.is_file():
        rows = json.loads(extension.read_text())
        domains['TiDE'] = rows.get('TiDE', rows.get('models', {}).get('TiDE', rows)) if isinstance(rows, dict) else rows
    return domains


def job_plan(models=None, *, domains=None):
    domains = current_domains() if domains is None else domains
    models = list(NEURAL if models is None else models)
    if not models or len(set(models)) != len(models) or set(models) - set(NEURAL):
        raise ValueError('Choose only distinct mandatory neural models')
    jobs = []
    for name in models:
        for fold in FOLDS:
            for slot in domains[name]:
                if not slot.get('execution_allowed'):
                    continue
                if (slot.get('configuration_id') not in SLOT_IDS or slot.get('optimization_seed') != HPO_SEED
                        or slot.get('configuration_sha256') != digest({k:v for k,v in slot.items() if k!='configuration_sha256'})):
                    raise ValueError('Reduced optimizer-domain identity changed')
                job = {'version': VERSION, 'stage': 'hpo', 'model_id': name, 'fold_origin': fold,
                    'configuration_id': slot['configuration_id'], 'configuration_sha256': slot['configuration_sha256'],
                    'parent_trial_id': slot['parent_trial_id'], 'trial': slot['trial'], 'seed': HPO_SEED}
                job['logical_job_id'] = digest(job)
                jobs.append(job)
    return jobs


def _validate_spec(spec):
    if spec.get('version') != VERSION or spec.get('job') not in job_plan([spec['job']['model_id']]):
        raise ValueError('Worker lies outside the bound reduced HPO domain')
    if spec.get('source_sha256') != source_identity():
        raise ValueError('Reduced worker executable source identity changed')
    if spec.get('fit_wall_seconds') != 600:
        raise ValueError('Use the registered600-second per-phase migration')
    verify_reduced_registration(ROOT, require_git=True)


def run_worker(spec):
    _validate_spec(spec)
    import torch
    from run_e2_cuda_search import prepare_panel
    from e2_temporal_training import prepare_b
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    output = Path(spec['output'])
    job = spec['job']
    panel, data = prepare_panel(spec['data_root'])
    prepared = prepare_b(panel, job['fold_origin'], job['fold_origin'])
    valid = prepared.inner_validation
    if len(valid.dates) != 83:
        raise ValueError('Reduced HPO requires all83 common fold dates')
    roots = json.loads(Path(spec['source_roots']).read_text())
    print(json.dumps({'event':'new_fit_started','model':job['model_id'],'origin':job['fold_origin'],
        'configuration_id':job['configuration_id'],'seed':job['seed'],'stage':'reduced_hpo'}),flush=True)
    fit = fit_reduced_job(job['model_id'], job['trial'], job['seed'], prepared.inner_train, valid, output,
        source_roots=roots, source_identity=spec['source_sha256'], runtime_identity=spec['runtime_identity'],
        fit_wall_seconds=spec['fit_wall_seconds'])
    result = {**fit, 'version': VERSION, 'job': job, 'attempt_id': job['logical_job_id'],
        'worker_spec_sha256': digest(spec), 'prepared_signature': prepared.signature,
        'training_signature': prepared.inner_train.signature, 'validation_signature': valid.signature,
        'data_receipt_sha256': digest(data), 'best_epoch': fit['phase_best_epochs']['supervised']}
    result['artifacts_sha256'] = {str(p.relative_to(output)): file_sha(p) for p in output.rglob('*')
        if p.is_file() and p.name not in ('result.json', 'worker_spec.json', 'worker.log', '.writer.lock')}
    write_json(output / 'result.json', result)
    return result


def _row(job, result, receipt_sha):
    return {'model_id': job['model_id'], 'configuration_id': job['configuration_id'],
        'fold_origin': job['fold_origin'], 'seed': job['seed'], 'parent_trial_id': job['parent_trial_id'],
        'configuration_sha256': job['configuration_sha256'], 'validation_count': 83,
        'status': 'completed', 'rmse_original': result['rmse_original'],
        'best_epoch': result['phase_best_epochs']['supervised'], 'phase_best_epochs': result['phase_best_epochs'],
        'run_receipt_sha256': receipt_sha, 'fatal': False, 'systematic_failure': False,
        'resource_use': {'optimizer_steps': result.get('optimizer_steps'), 'cache_reuse': result.get('cache_reuse', False)}}


def _cached_result(job, audited, reconciliation_root, output, prepared, runtime_profile):
    from audit_e2_reduced_cache import inspect_record
    branch, head = audited['source_branch'], audited['source_head']
    base = Path(reconciliation_root) / 'snapshots' / branch.rsplit('/',1)[-1] / head
    path = base / 'results' / Path(audited['source_path']).name
    if file_sha(path) != audited['file_sha256']:
        raise ValueError('Audited cache source bytes changed')
    record = json.loads(path.read_text())
    plan = json.loads((base/'campaign.json').read_text())['plan']
    slot = next(s for s in current_domains()[job['model_id']] if s['configuration_id']==job['configuration_id'])
    checked = inspect_record(record, plan, model=job['model_id'], anchor=job['fold_origin'], slot=slot,
        context={'prepared_signature':prepared.signature,'training_signature':prepared.inner_train.signature,
                 'validation_signature':prepared.inner_validation.signature}, runtime_profile=runtime_profile)
    if not checked['reuse_approved'] or checked['result_sha256'] != audited['result_sha256']:
        raise ValueError('Actual cache admission failed source/data/runtime/policy checks')
    values = record['outcome']['predictions_original']
    truth = prepared.inner_validation.y_original
    score = math.sqrt(sum((p-y)**2 for p,y in zip(values,truth))/len(truth))
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'cached_source_record.json',record)
    write_json(output/'cache_admission.json',checked)
    result = {'version':VERSION,'attempt_id':job['logical_job_id'],'status':'completed','job':job,
        'cache_reuse':True,'training_performed':False,'weights_reused':False,'weights_reusable':False,
        'phase_best_epochs':{'supervised':checked['best_epoch']}, 'optimizer_steps':0,
        'rmse_original':score,'prediction_dates':record['outcome']['prediction_dates'],
        'predictions_original':values,'validation_count':83,
        'artifacts_sha256':{'cached_source_record.json':file_sha(output/'cached_source_record.json'),
                           'cache_admission.json':file_sha(output/'cache_admission.json')}}
    write_json(output/'result.json',result)
    return result


def _tide_selection(rows, domain):
    from e2_published_tide import select_reduced_tide
    return select_reduced_tide(rows, domain)


def run_selection(args, budget, cache_report, reconciliation_root, *, persist_result=None):
    """Run admitted jobs sequentially; persist/backup before admitting another fit."""
    from run_e2_cuda_search import prepare_panel
    from e2_temporal_training import prepare_b
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    verify_reduced_registration(ROOT,require_git=True)
    runtime_profile=json.loads(Path(args.runtime_profile).read_text())
    if runtime_profile.get('fit_execution_policy_verified') is not True:
        raise ValueError('Actual fitting-policy attestation is required before cache or fits')
    domains=current_domains()
    jobs=job_plan(getattr(args,'models',None),domains=domains)
    panel,_=prepare_panel(args.data_root)
    prepared={f:prepare_b(panel,f,f) for f in FOLDS}
    approved={(r['model_id'],r['origin'],r['configuration_id']):r for r in cache_report.get('fits',[]) if r.get('reuse_approved')}
    source=source_identity()
    ledger=[]
    for job in jobs:
        directory=output/'jobs'/job['logical_job_id']
        row_path=directory/'selection_row.json'
        if row_path.exists():
            row=json.loads(row_path.read_text())
            result=json.loads((directory/'result.json').read_text())
            if row['run_receipt_sha256']!=file_sha(directory/'result.json'):
                raise ValueError('Saved selection row/result binding changed')
            for relative,expected in result['artifacts_sha256'].items():
                if file_sha(directory/relative)!=expected:raise ValueError('Saved selection artifact changed')
            if result.get('cache_reuse') and (job['model_id'],job['fold_origin'],job['configuration_id']) not in approved:
                raise ValueError('Saved cache row is no longer admitted by the current runtime audit')
            if not result.get('cache_reuse') and json.loads((directory/'worker_spec.json').read_text())['source_sha256']!=source:
                raise ValueError('Saved HPO worker closure changed; reconcile before combining recipes')
            if not result.get('cache_reuse') and budget.completed_receipt(job['logical_job_id'],
                expected_resource_use={'result_sha256':file_sha(directory/'result.json'),
                    'worker_spec_sha256':result['worker_spec_sha256']}) is None:
                raise RuntimeError('Saved selection lacks its matching budget completion')
        else:
            match=approved.get((job['model_id'],job['fold_origin'],job['configuration_id']))
            if match is not None:
                result=_cached_result(job,match,reconciliation_root,directory,prepared[job['fold_origin']],runtime_profile)
            else:
                runtime=runtime_profile['runtime']
                spec={'version':VERSION,'job':job,'output':str(directory),'data_root':str(args.data_root),
                    'source_roots':str(args.source_roots),'source_sha256':source,'runtime_identity':runtime,
                    'fit_wall_seconds':budget.stage_wall_seconds}
                write_json(directory/'worker_spec.json',spec)
                print(json.dumps({'event':'fit_admission_requested','model':job['model_id'],'origin':job['fold_origin'],
                    'configuration_id':job['configuration_id'],'stage':'reduced_hpo','seed':job['seed']}),flush=True)
                try:
                    run=supervise_command([getattr(args,'python',sys.executable),'-u',Path(__file__),
                        '--worker-spec',directory/'worker_spec.json'],budget,job['logical_job_id'],directory,digest(spec),
                        phases=3 if job['model_id']=='LSTM-autoencoder con predictor' else 1,cwd=ROOT)
                except ReducedWorkerFailure as error:
                    failed={'version':VERSION,'status':'failed','attempt_id':job['logical_job_id'],'job':job,
                        'worker_spec_sha256':digest(spec),'supervision':error.receipt,'automatic_retry':False,
                        'artifacts_sha256':{str(p.relative_to(directory)):file_sha(p) for p in directory.rglob('*')
                            if p.is_file() and p.name not in ('failure_result.json','.writer.lock')}}
                    write_json(directory/'failure_result.json',failed)
                    if persist_result is not None:
                        persist_result(directory,failed)
                    raise
                result=run['result']
            row=_row(job,result,file_sha(directory/'result.json'))
            write_json(row_path,row)
        if persist_result is not None:
            persist_result(directory,result)
        ledger.append(row)
        print(json.dumps({'model':job['model_id'],'origin':job['fold_origin'],'configuration_id':job['configuration_id'],
            'status':'completed','cache_reuse':result.get('cache_reuse',False),'remaining_seconds':budget.remaining_seconds}),flush=True)
    selections={}
    for name in dict.fromkeys(j['model_id'] for j in jobs):
        rows=[r for r in ledger if r['model_id']==name]
        selected=_tide_selection(rows,domains[name]) if name=='TiDE' else select_reduced_hpo(rows,name,domains[name])
        slot=next(s for s in domains[name] if s['configuration_id']==selected['winner_configuration_id'])
        selected.update(version=VERSION,winner_trial=slot['trial'],run_artifact_bytes_verified=True,
            selection_ledger_sha256=digest(rows),source_current_recipe_verified=True,criteria_only=False,
            new_training_performed=any(not r['resource_use'].get('cache_reuse') for r in rows))
        write_json(output/'selections'/(slug(name)+'.json'),selected)
        selections[name]=selected
    receipt={'version':VERSION,'status':'completed' if set(selections)==set(NEURAL) else 'partial_models_completed',
        'selections':selections,'required_models':list(NEURAL),'completed_models':list(selections),
        'TiDE_architecture_blocked':'TiDE' not in selections,'final_benchmark_complete':False,
        'final_market_values_read':False,'selection_ledger_sha256':digest(ledger),'budget':budget.report()}
    write_json(output/'selection_complete.json',receipt)
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-spec',type=Path)
    parser.add_argument('--write-runtime-profile',type=Path)
    parser.add_argument('--execute',action='store_true')
    for name in ('output','data-root','source-roots','runtime-profile','budget-evidence','migration','cache-audit','reconciliation-root'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--migration-registration-commit')
    parser.add_argument('--inherited-ledger',type=Path,action='append',default=[])
    parser.add_argument('--models',nargs='+')
    args=parser.parse_args(argv)
    if args.worker_spec:
        try:
            run_worker(json.loads(args.worker_spec.read_text()))
            return 0
        except Exception as error:
            failure={'status':'failed','error_type':type(error).__name__,'reason':str(error),
                'resource_receipt':getattr(error,'receipt',{}),'automatic_retry':False}
            write_json(args.worker_spec.parent/'worker_failure.json',failure)
            print(json.dumps(failure),flush=True)
            return 2
    if args.write_runtime_profile:
        write_json(args.write_runtime_profile,worker_preflight_policy())
        return 0
    if not args.execute:
        print(json.dumps({'version':VERSION,'planned_admitted_jobs':len(job_plan(args.models)),
            'training_started':False,'final_market_values_read':False}))
        return 0
    required=('output','data_root','source_roots','runtime_profile','budget_evidence','migration','cache_audit','reconciliation_root','migration_registration_commit')
    if any(getattr(args,k,None) is None for k in required):
        parser.error('Execution requires every bound data/source/runtime/budget/cache path and registration commit')
    from e2_reduced_budget import ReducedBudgetLedger
    from e2_reduced_github_backup import GitHubBackup
    client=GitHubBackup()
    client.preflight()
    with ReducedBudgetLedger(args.output/'shared_budget',json.loads(args.budget_evidence.read_text()),
        migration=json.loads(args.migration.read_text()),migration_registration_commit=args.migration_registration_commit,
        inherited_ledgers=args.inherited_ledger) as budget:
        run_selection(args,budget,json.loads(args.cache_audit.read_text()),args.reconciliation_root,
            persist_result=client.persist_result)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
