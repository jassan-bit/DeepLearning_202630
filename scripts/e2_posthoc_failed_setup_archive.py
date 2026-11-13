"""Preserve one charged failed clone; no optimizer, deletion or budget reset."""
from __future__ import annotations
import fcntl, hashlib, json, math, os, stat
from pathlib import Path

VERSION='e2-posthoc-failed-clone-archive-v1'
PIN='ae1a2d4f96807a20ea92c313297a5c09512a4817'
HELPER='scripts/e2_posthoc_failed_setup_archive.py'
WEIGHTS={'.pt','.pth','.ckpt','.pkl','.safetensors','.npz','.bin'}

def sha(raw):return hashlib.sha256(raw).hexdigest()
def regular(path):
    path=Path(path).absolute()
    if any(p.is_symlink() for p in (path,*path.parents)):raise ValueError('Symlink in failed setup archive')
    return path

def fingerprint(directory):
    directory=regular(directory)
    if not directory.is_dir():raise ValueError('Failed setup must be a directory')
    files=[];directories=[]
    for path in sorted(directory.rglob('*')):
        regular(path);kind=path.lstat().st_mode;relative=path.relative_to(directory).as_posix()
        if stat.S_ISDIR(kind):directories.append(relative);continue
        if not stat.S_ISREG(kind):raise ValueError('Only regular files/directories may be preserved')
        if path.suffix.lower() in WEIGHTS:raise ValueError('Weights in failed setup; archive prohibited')
        if len(files)>=10000:raise ValueError('Failed setup exceeds10000files')
        raw=path.read_bytes();files.append({'path':relative,'bytes':len(raw),'sha256':sha(raw)})
    return {'files':files,'directories':directories,'file_count':len(files),'total_bytes':sum(x['bytes'] for x in files)}

def verify_boundary(old,root,budget,remote):
    if budget.persistence is not remote:raise RuntimeError('Use the held remotely persistent ledger')
    for lock in (budget.lock,budget.original_lock):
        if lock is None or lock.closed:raise RuntimeError('Both original ledger locks must remain held')
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if Path(budget.root).resolve()!=(root/'budget').resolve():raise ValueError('Wrong shared ledger root')
    raw=Path(budget.path).read_bytes()
    if not raw or not raw.endswith(b'\n'):raise ValueError('Complete existing journal required')
    rows=[json.loads(x) for x in raw.splitlines()]
    if rows!=budget.events:raise ValueError('Live journal and held ledger differ')
    previous=old.ZERO
    for i,row in enumerate(rows):
        old.checked(row)
        if row.get('sequence')!=i or row.get('previous_sha256')!=previous:raise ValueError('Shared ledger chain differs')
        previous=row['sha256']
        if (row['kind']=='reserved' and (row.get('metadata',{}).get('kind') in {'reproduction','ablation'}
                or row.get('job_id','').startswith(('posthoc-reproduction-','posthoc-ablation-')))):
            raise RuntimeError('A model fit was admitted; failed-clone archive prohibited')
    if list((root/'reproduction/jobs').glob('*/worker_spec.json')):raise RuntimeError('An optimizer worker spec exists')
    if (root/'reproduction/jobs').exists() and any((root/'reproduction/jobs').iterdir()):raise RuntimeError('Optimizer job files exist')
    if budget.active:
        if len(budget.active)!=1:raise RuntimeError('Unexpected active work')
        job=next(iter(budget.active));reservation=next((r for r in reversed(rows) if r['kind']=='reserved' and r.get('job_id')==job),{})
        metadata=reservation.get('metadata',{})
        if metadata.get('scope')!='failed_clean_setup_archive' or metadata.get('training_performed') is not False:
            raise RuntimeError('Only caller administrative archive lease may be active')
    attempts=[r for r in rows if r['kind']=='reserved' and r.get('job_id','').startswith('clean-setup-')]
    if len(attempts)!=1:raise RuntimeError('Exactly one prior clean setup attempt is required for unambiguous clone proof')
    reserved=attempts[0]
    if reserved.get('metadata',{}).get('scope')!='source_checkout_and_actual_new_venv_install':raise ValueError('Prior clean setup scope differs')
    charges=[r for r in rows if r['kind']=='charged' and r.get('job_id')==reserved['job_id']]
    if len(charges)!=1 or charges[0].get('status')!='failed':raise RuntimeError('Original clean setup must already be charged FAILED')
    failed=charges[0]
    if (failed.get('reservation_seconds')!=reserved['seconds'] or not math.isfinite(failed.get('elapsed_seconds',-1)) or failed['elapsed_seconds']<0):
        raise ValueError('Failed setup charge is not bound to its reservation')
    return raw,reserved,failed

def verify_diagnostics(root):
    path=regular(root/'robustness/result.json');result=json.loads(path.read_bytes())
    if result.get('status')!='completed':raise RuntimeError('Frozen diagnostics aggregate is incomplete')
    predictors=sorted((root/'robustness/predictors').glob('*/result.json'))
    if len(predictors)!=79:raise RuntimeError('All79completed frozen predictors are required')
    identities={}
    for item in [path,*predictors]:
        regular(item);value=json.loads(item.read_bytes())
        if value.get('status')!='completed':raise RuntimeError('Frozen predictor receipt is incomplete')
        if result.get('run_id') and value.get('run_id') and value['run_id']!=result['run_id']:raise ValueError('Frozen diagnostics run identities differ')
        for relative,expected in value.get('artifacts_sha256',{}).items():
            artifact=regular(item.parent/relative)
            if not artifact.resolve().is_relative_to(item.parent.resolve()) or sha(artifact.read_bytes())!=expected:raise ValueError('Frozen diagnostics artifact differs')
        identities[item.relative_to(root).as_posix()]=sha(item.read_bytes())
    return {'aggregate_run_id':result.get('run_id'),'completed_predictors':79,'receipts_sha256':identities}

def verify_clone_only(old,clean):
    if not (clean/'setup_started.json').is_file() or not (clean/'checkout.log').is_file():raise RuntimeError('Known clone failure needs setup_started.json and checkout.log')
    for forbidden in ('clean_runtime.json','environment_install.log','env_install.log','environment.lock.txt','venv'):
        if (clean/forbidden).exists():raise RuntimeError('Environment setup progressed beyond clone-only boundary')
    started=json.loads((clean/'setup_started.json').read_bytes())
    if (started.get('version')!=old.VERSION or started.get('scientific_commit')!=old.ENGINE_COMMIT or started.get('no_system_site_packages') is not True):
        raise ValueError('Original clean setup marker differs')

def archive_failed_setup(old,destination,budget,remote,adapter_registration):
    """Caller owns a240s administrative lease and all accounting/persistence."""
    root=regular(destination)
    if adapter_registration.get('original_configuration_commit')!=PIN:raise ValueError('Keep original ae1 scientific configuration')
    helper_sha=sha(Path(__file__).read_bytes())
    if adapter_registration.get('files_sha256',{}).get(HELPER)!=helper_sha:raise ValueError('Executing archive helper differs from adapter registration')
    ledger_before,reserved,failed=verify_boundary(old,root,budget,remote);diagnostic_proof=verify_diagnostics(root)
    clean=regular(root/'reproduction/clean');archive=regular(root/'transport_amendments/failed_clean_setup'/failed['sha256'])
    preserved=archive/'preserved_clean';inventory_path=archive/'inventory.json';reused=inventory_path.exists()
    if reused:
        inventory=json.loads(inventory_path.read_bytes())
        if clean.exists():raise RuntimeError('A new clean setup exists; never archive it as the old failure')
        if (inventory.get('failed_charge_sha256')!=failed['sha256'] or inventory.get('original_clean_path')!=str(clean)
                or inventory.get('adapter_registration_sha256')!=old.digest(adapter_registration)):raise ValueError('Existing archive proof differs')
        actual=fingerprint(preserved)
        if actual!=inventory.get('tree') or old.digest(actual)!=inventory.get('tree_sha256'):raise ValueError('Preserved failed setup bytes differ')
        verify_clone_only(old,preserved)
    else:
        if archive.exists():raise RuntimeError('Archive exists without its bound inventory')
        verify_clone_only(old,clean);tree=fingerprint(clean)
        inventory={'version':VERSION,'failed_charge_sha256':failed['sha256'],'original_clean_path':str(clean),'preserved_clean_path':str(preserved),
            'tree':tree,'tree_sha256':old.digest(tree),'adapter_registration_sha256':old.digest(adapter_registration),'helper_source_sha256':helper_sha}
        old.immutable_json(inventory_path,inventory);os.rename(clean,preserved)
        if fingerprint(preserved)!=tree:raise ValueError('Renamed failed setup bytes differ')
    for name in ('checkout.log','setup_started.json'):
        payload=(preserved/name).read_bytes();target=archive/name
        if target.exists() and target.read_bytes()!=payload:raise ValueError('Copied failure metadata differs')
        if not target.exists():
            with target.open('xb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
    amendment={'version':VERSION,'status':'completed','original_configuration_commit':PIN,'failed_setup_reservation':reserved,'failed_setup_charge':failed,
        'ledger_prefix_sha256_before_archive':sha(ledger_before),'original_clean_path':str(clean),'preserved_clean_path':str(preserved),
        'inventory_sha256':sha(inventory_path.read_bytes()),'tree_sha256':inventory['tree_sha256'],'diagnostics':diagnostic_proof,
        'adapter_registration_sha256':old.digest(adapter_registration),'helper_source_sha256':helper_sha,'old_failed_charge_preserved':True,
        'budget_reset':False,'optimizer_fit_performed_or_retried':False,'preserved_tree_uploaded':False,
        'reason':'Preserve charged failed local Git clone before a separately bounded environment-only retry'}
    amendment_path=archive/'amendment.json'
    if amendment_path.exists():
        prior=json.loads(amendment_path.read_bytes())
        if any(prior.get(k)!=v for k,v in amendment.items() if k!='ledger_prefix_sha256_before_archive'):raise ValueError('Existing failed-clone amendment differs')
        amendment=prior
    else:old.immutable_json(amendment_path,amendment)
    names=('inventory.json','checkout.log','setup_started.json','amendment.json')
    receipt={'version':VERSION,'status':'completed','attempt_id':'failed-clean-archive-'+failed['sha256'][:32],
        'scientific_result':False,'artifacts_sha256':{name:sha((archive/name).read_bytes()) for name in names}}
    old.backup_completed(archive,receipt,remote)
    if Path(budget.path).read_bytes()!=ledger_before:raise RuntimeError('Archive helper unexpectedly mutated the held budget journal')
    return {'status':'failed_clone_preserved','archive':str(archive),'reused_verified_archive':reused,'failed_charge_sha256':failed['sha256'],
            'inventory_sha256':sha(inventory_path.read_bytes()),'optimizer_fit_performed_or_retried':False,'budget_reset':False}
