"""Continue frozen E selection while retaining one already charged setup failure.

The failed opportunity is never retried or replaced. The frozen selector still
requires all eighteen rows, and excludes C01 when any of its folds failed.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import re
import sys

import run_e2_reduced_selection as original

KNOWN_JOB = '10098dda29216054585354395d2d8ef4b81c44ad31361b4c11cabf09ee998c41'
KNOWN_SPEC = '4a65560c0028dd67f792e1642135dd7276c0cf84c93f3911af97c2b5d2b700aa'
KNOWN_COST = '12dce4b730af1b18c1fee7ff948a6c93144ff29455b4cdd9178549d0b4023947'
KNOWN_FAILURE_FILE = '6f6b5a692ff123b0654694d4d761a6624da75b1ce6641dd154da9dd1dd15ddb5'
MISSING_PATH = ('/content/entregable2_gpu_workspace/'
    'reduced-source-1e8745497b484518f766b6da5c19be8362d9add7/'
    'book_e2/_static/current_implementation_20261108/source_design.json')
KNOWN_REASON = "[Errno 2] No such file or directory: '" + MISSING_PATH + "'"


def _verify_backup_receipt(directory, failure):
    """Admit only the transport receipt for these exact four failed artifacts."""
    path = directory / 'github_backup_receipt.json'
    if not path.exists():
        return
    backup = json.loads(path.read_text())
    prefix = 'runs/' + KNOWN_JOB + '/'
    artifacts = []
    for relative, expected in sorted(failure['artifacts_sha256'].items()):
        raw = (directory / relative).read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        artifacts.append({'path':relative, 'storage':'git_blob',
            'remote_path':prefix+'artifacts/'+relative, 'sha256':expected,
            'bytes':len(raw), 'git_blob_sha':blob})
    manifest = {'version':'e2-reduced-github-backup-v1','attempt_id':KNOWN_JOB,
        'fit_status':'failed','receipt_sha256':original.digest(failure),
        'artifacts':artifacts,'retry_scientific_fit':False,'raw_market_data_included':False}
    manifest_sha = hashlib.sha256(original.canonical(manifest)+b'\n').hexdigest()
    if (backup.get('version') != 'e2-reduced-github-backup-v1'
            or backup.get('status') != 'backed_up' or backup.get('attempt_id') != KNOWN_JOB
            or backup.get('repository') != 'jassan-bit/DeepLearning_202630'
            or backup.get('branch') != 'results/e2-reduced-delivery-v1'
            or backup.get('release_tag') != 'e2-reduced-results-20261112-v1'
            or backup.get('manifest_path') != prefix+'manifest.json'
            or backup.get('manifest_sha256') != manifest_sha
            or backup.get('artifact_count') != 4
            or backup.get('retry_scientific_fit') is not False
            or type(backup.get('existing_identical_result')) is not bool
            or not re.fullmatch(r'[0-9a-f]{40}', str(backup.get('remote_commit','')))):
        raise ValueError('Known failure GitHub transport receipt changed')


def _budget_events(budget):
    """Read the already validated canonical ledger through the transport wrapper."""
    seen = set()
    while id(budget) not in seen:
        seen.add(id(budget))
        if hasattr(budget, '_events'):
            return budget._events
        if hasattr(budget, '_budget'):
            budget = budget._budget
        elif hasattr(budget, '_inner'):
            budget = budget._inner
        else:
            break
    raise ValueError('Canonical failed budget receipt is unavailable')


def known_failure_row(job, directory, budget, source, validation_dates):
    """Verify the sole known failure; preserve its original receipt and cost."""
    directory = Path(directory)
    if (job['logical_job_id'] != KNOWN_JOB or job['model_id'] != 'TimesNet'
            or job['fold_origin'] != '2024-01-01' or job['configuration_id'] != 'C01'
            or job['seed'] != 271201 or len(validation_dates) != 83):
        raise ValueError('Failure continuation is restricted to the registered setup attempt')
    failure = json.loads((directory / 'failure_result.json').read_text())
    worker = json.loads((directory / 'worker_failure.json').read_text())
    spec = json.loads((directory / 'worker_spec.json').read_text())
    if (failure.get('status') != 'failed' or failure.get('attempt_id') != KNOWN_JOB
            or failure.get('job') != job or failure.get('automatic_retry') is not False
            or failure.get('worker_spec_sha256') != KNOWN_SPEC
            or original.digest(spec) != KNOWN_SPEC or spec.get('job') != job
            or spec.get('source_sha256') != source
            or worker.get('error_type') != 'FileNotFoundError'
            or worker.get('reason') != KNOWN_REASON or worker.get('status') != 'failed'
            or worker.get('automatic_retry') is not False
            or worker.get('resource_receipt') != {}
            or original.file_sha(directory / 'worker_failure.json') != KNOWN_FAILURE_FILE):
        raise ValueError('Known setup failure identity changed')
    artifacts = failure.get('artifacts_sha256', {})
    if set(artifacts) != {'request.json', 'worker.log', 'worker_failure.json', 'worker_spec.json'}:
        raise ValueError('Known setup failure has unexpected optimizer or phase artifacts')
    for relative, expected in artifacts.items():
        if original.file_sha(directory / relative) != expected:
            raise ValueError('Preserved failure artifact changed')
    _verify_backup_receipt(directory, failure)
    allowed = set(artifacts) | {'failure_result.json', 'selection_row.json',
        'github_backup_receipt.json', '.writer.lock'}
    if any(p.is_file() and str(p.relative_to(directory)) not in allowed
           for p in directory.rglob('*')):
        raise ValueError('Unexpected output prevents the zero-phase setup classification')
    charges = [r for r in _budget_events(budget)
               if r.get('kind') == 'charged' and r.get('attempt_id') == KNOWN_JOB]
    if len(charges) != 1:
        raise ValueError('Known failed attempt needs exactly one canonical cost receipt')
    charge = charges[0]
    if (charge.get('sha256') != KNOWN_COST
            or original.digest({k:v for k,v in charge.items() if k != 'sha256'}) != KNOWN_COST
            or charge.get('status') != 'failed' or charge.get('reservation_overrun') is not False
            or charge.get('resource_use', {}).get('worker_spec_sha256') != KNOWN_SPEC
            or charge.get('resource_use', {}).get('failure') != 'worker_exit_2'
            or charge.get('resource_use', {}).get('automatic_retry') is not False):
        raise ValueError('Known failure cost binding changed')
    supervision = failure.get('supervision', {})
    if (supervision.get('worker_spec_sha256') != KNOWN_SPEC
            or supervision.get('failure') != 'worker_exit_2'
            or supervision.get('automatic_retry') is not False
            or supervision.get('budget_receipt') != charge):
        raise ValueError('Failure receipt differs from the preserved canonical charge')
    row = {k:job[k] for k in ('model_id','configuration_id','fold_origin','seed',
                             'parent_trial_id','configuration_sha256')}
    row.update(status='failed', rmse_original=None, best_epoch=None, fatal=False,
        systematic_failure=False, validation_count=83, predictions_produced=0,
        validation_count_scope='admitted_calendar_not_predictions',
        reason='Known source-design metadata setup failure; original opportunity closed without retry',
        run_receipt_sha256=original.file_sha(directory / 'failure_result.json'),
        run_receipt_file='failure_result.json', budget_receipt_sha256=KNOWN_COST,
        replacement_performed=False, automatic_retry=False,
        resource_use={'cache_reuse':False, 'optimizer_steps':0,
            'optimizer_steps_scope':'specific_pre_model_setup_failure_verified',
            'original_supervision_preserved':True, 'new_training_performed':False})
    return row, failure



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
        if job['logical_job_id'] == KNOWN_JOB:
            row, result = known_failure_row(job, directory, budget, source,
                prepared[job['fold_origin']].inner_validation.dates)
            if row_path.exists() and json.loads(row_path.read_text()) != row:
                raise ValueError('Saved known failure selection row changed')
            original.write_json(row_path, row)
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
                    if persist_result is not None:
                        persist_result(directory,failed)
                    raise
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


