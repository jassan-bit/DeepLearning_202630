"""Reconcile one proven-unlaunched, unpublished local reservation; never fit.

This is an incident-specific cache reconciliation. The authoritative remote
journal, its costs, all completed results and the original 80-hour ledger are
unchanged. The discarded local cache tail and unused specification are durably
archived before the local cache can be brought back to that exact remote prefix.
"""
from __future__ import annotations
from contextlib import contextmanager
import fcntl,hashlib,json,os,tempfile,time
from pathlib import Path

STATE_ID='41a76ef9fd9077fcc6419c697d7ccc16adb2f9b43049313337b216149dffc99c'
REMOTE_SHA='9f13c23a70f81cf24999238a007a3a2d9b7938a218d560a35f44fc99a350aa0d'
LOCAL_SHA='b7cd892fa564676129afbeb27b863ff51037eb35bc7c2aa08e5527fca7dcdf61'
TAIL_SHA='ba3d6f9fefa87456a29e44cc27a8acb7f567740f1fcc5cace8e44c3196b67b35'
JOB_ID='posthoc-reproduction-9dd9435d53b813c54771cc58'
CONTROLLER_SHA='5d8ce059f96f86a35b7e2d2530e3fb776827240fcb29df920b4d05374f139fe9'
CONFIGURATION='ae1a2d4f96807a20ea92c313297a5c09512a4817'
REMOTE_ROWS=252
COMPLETED_FITS=72
FAILED_TRANSPORT_CONSERVATIVE_SECONDS=300.0

def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def regular(path):
    path=Path(path).absolute()
    if any(p.is_symlink() for p in (path,*path.parents)):raise ValueError('Reconciliation requires regular paths')
    return path

def _once(path,raw):
    path=regular(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes()!=raw:raise ValueError('Archived evidence differs; preserved')
    else:
        with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())

def _rows(old,raw):
    if not isinstance(raw,bytes) or not raw.endswith(b'\n'):raise ValueError('Complete byte-exact journal required')
    result=[];active={};previous=old.ZERO
    for i,line in enumerate(raw.splitlines()):
        row=json.loads(line);old.checked(row)
        if row.get('sequence')!=i or row.get('previous_sha256')!=previous or row.get('version')!=old.VERSION:
            raise ValueError('Journal sequence, version or hash chain differs')
        previous=row['sha256'];kind=row['kind']
        if kind=='reserved':
            if active or row['job_id'] in active:raise ValueError('Overlapping reservations are not reconciled')
            active[row['job_id']]=row['seconds']
        elif kind=='charged':
            if row['job_id'] not in active:raise ValueError('Unbound charge in journal')
            active.pop(row['job_id'])
        elif kind not in ('overhead','interrupted'):raise ValueError('Unknown journal event')
        result.append(row)
    return result,active

def validate_local_only(old,output,original_output,remote_ledger,expected_state):
    output=regular(output);original_output=regular(original_output)
    if expected_state.get('state_id')!=STATE_ID:raise ValueError('Only the audited incident state can be reconciled')
    journal=regular(output/'budget/posthoc_budget.jsonl');local=journal.read_bytes()
    if sha(remote_ledger)!=REMOTE_SHA or sha(local)!=LOCAL_SHA:raise ValueError('Audited local or remote journal differs')
    remote_rows,remote_active=_rows(old,remote_ledger)
    if remote_active:raise ValueError('A remote unresolved lease can never be credited or retried')
    local_rows,local_active=_rows(old,local)
    if (len(remote_rows)!=REMOTE_ROWS or len(local_rows)!=REMOTE_ROWS+1
            or not local.startswith(remote_ledger) or len(local[len(remote_ledger):].splitlines())!=1):
        raise ValueError('Local journal must be exactly the audited remote prefix plus one reservation')
    tail=local_rows[-1]
    if (tail.get('sha256')!=TAIL_SHA or tail.get('kind')!='reserved' or tail.get('job_id')!=JOB_ID
            or tail.get('seconds')!=1860.0 or local_active!={JOB_ID:1860.0}
            or tail.get('metadata',{}).get('model')!='iTransformer'
            or tail['metadata'].get('seed')!=271801 or tail['metadata'].get('kind')!='reproduction'):
        raise ValueError('Only the exact unlaunched iTransformer admission is eligible')
    state_files=expected_state.get('files',{})
    if state_files.get('budget/posthoc_budget.jsonl',{}).get('sha256')!=REMOTE_SHA:
        raise ValueError('Remote state does not bind the audited journal')
    results=[p for p in state_files if p.startswith('reproduction/jobs/') and p.endswith('/result.json')]
    if len(results)!=COMPLETED_FITS:raise ValueError('Preserve all audited completed result bindings')
    directory=regular(output/'reproduction/jobs'/JOB_ID)
    if not directory.is_dir() or set(p.name for p in directory.iterdir())!={'worker_spec.json'}:
        raise ValueError('Any worker log, result, phase or other execution evidence prohibits this reconciliation')
    path=regular(directory/'worker_spec.json')
    if not path.is_file():raise ValueError('Exact unused worker specification required')
    spec_raw=path.read_bytes();spec=json.loads(spec_raw)
    expected={'job_id':JOB_ID,'kind':'reproduction','model':'iTransformer','seed':271801,'condition':'full',
              'configuration_commit':CONFIGURATION,'controller_source_sha256':CONTROLLER_SHA}
    if any(spec.get(k)!=v for k,v in expected.items()) or regular(spec.get('original_output',''))!=original_output:
        raise ValueError('Unused specification does not match the audited first launch')
    return {'journal':journal,'local_raw':local,'remote_raw':remote_ledger,'directory':directory,'spec_raw':spec_raw,
            'spec_sha256':sha(spec_raw),'reservation':tail}

@contextmanager
def _locks(output,original_output):
    streams=[]
    try:
        for path in (regular(output/'budget/.posthoc-budget.lock'),regular(original_output/'shared_budget/.pilot-budget.lock')):
            stream=path.open('a+b');streams.append(stream)
            try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except OSError:raise RuntimeError('An existing controller holds the campaign lock') from None
        yield
    finally:
        for stream in reversed(streams):
            fcntl.flock(stream.fileno(),fcntl.LOCK_UN);stream.close()

def _require_remote(check_remote,expected_state):
    actual=check_remote()
    if canonical(actual)!=canonical(expected_state):raise RuntimeError('Remote campaign advanced; preserve local admission for reconciliation')

def _atomic_restore(path,raw):
    fd,name=tempfile.mkstemp(prefix='.prelaunch-prefix-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        os.replace(name,path)
        dfd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(dfd)
        finally:os.close(dfd)
    finally:Path(name).unlink(missing_ok=True)

def reconcile_local_only(old,output,original_output,archive_root,remote_ledger,expected_state,
                         backup_archive,check_remote,no_active_processes,*,clock=time.monotonic):
    """Archive first, then reconcile only a verified unlaunched local cache tail.

    ``backup_archive(folder, receipt)`` must durably back up every claimed byte
    to GitHub and return its verified transport receipt. ``check_remote`` reads
    the current authoritative state; ``no_active_processes`` performs a fresh
    local process check and returns True only when no controller/worker runs.
    The caller subsequently records the returned conservative prior-failure
    charge plus measured recovery overhead using the unchanged budget ledger.
    """
    started=clock();output=regular(output);original_output=regular(original_output);archive_root=regular(archive_root)
    if archive_root.resolve().is_relative_to(output.resolve()) or archive_root.resolve().is_relative_to(original_output.resolve()):
        raise ValueError('Evidence archive must be outside both scientific campaign outputs')
    if no_active_processes() is not True:raise RuntimeError('Existing worker/controller activity prohibits reconciliation')
    with _locks(output,original_output):
        old.validate_original_anchor(original_output)
        _require_remote(check_remote,expected_state)
        plan=validate_local_only(old,output,original_output,remote_ledger,expected_state)
        evidence={'version':'e2-unpublished-prelaunch-tail-reconciliation-v1','status':'proven_unlaunched_local_admission',
            'remote_state_id':STATE_ID,'remote_journal_sha256':REMOTE_SHA,'local_journal_sha256':LOCAL_SHA,
            'unpublished_reservation_sha256':TAIL_SHA,'job_id':JOB_ID,'worker_spec_sha256':plan['spec_sha256'],
            'remote_records_preserved':REMOTE_ROWS,'completed_fits_preserved':COMPLETED_FITS,
            'remote_active_reservations':{},'local_worker_directory_members':['worker_spec.json'],
            'no_active_local_controller_or_worker':True,'scientific_fit_started':False,'scientific_fit_repeated':False,
            'source_failure_boundary':'budget.reserve failed before entering the worker launch try block',
            'original_controller_sha256':CONTROLLER_SHA,'original_configuration_commit':CONFIGURATION,
            'preserve_original_80_hour_authority':True,'new_budget_credit':False,
            'prior_failed_transport_conservative_seconds':FAILED_TRANSPORT_CONSERVATIVE_SECONDS,
            'prior_failed_transport_charge_scope':'Conservative 300s for at most two 20s-timeout publication attempts; not a measured fit duration'}
        identity=sha(canonical(evidence));folder=archive_root/('prelaunch-'+identity[:24])
        artifacts={'local_journal_253.jsonl':plan['local_raw'],'remote_journal_252.jsonl':remote_ledger,
                   'unused_worker_spec.json':plan['spec_raw'],'reconciliation_evidence.json':canonical(evidence)+b'\n'}
        for name,raw in artifacts.items():_once(folder/name,raw)
        receipt={'status':'completed','attempt_id':'prelaunch-unpublished-'+identity[:24],
                 'scientific_result':False,'scientific_fit_started':False,'budget_reset':False,
                 'artifacts_sha256':{name:sha(raw) for name,raw in artifacts.items()}}
        backed=backup_archive(folder,receipt)
        if (not isinstance(backed,dict) or backed.get('status')!='backed_up'
                or not backed.get('remote_commit') or not backed.get('manifest_path')
                or backed.get('attempt_id')!=receipt['attempt_id']):
            raise RuntimeError('Verified durable GitHub archive required before local reconciliation')
        # An archive-only branch commit is allowed; a changed control state is not.
        _require_remote(check_remote,expected_state)
        if no_active_processes() is not True:raise RuntimeError('Worker activity appeared; preserve the unmodified local admission')
        again=validate_local_only(old,output,original_output,remote_ledger,expected_state)
        if again['spec_raw']!=plan['spec_raw']:raise ValueError('Unused specification changed during archival')
        old.validate_original_anchor(original_output)
        quarantined=folder/'unlaunched_job'
        if quarantined.exists():raise ValueError('Quarantine destination already exists; preserve both copies')
        os.rename(plan['directory'],quarantined)
        try:_atomic_restore(plan['journal'],remote_ledger)
        except BaseException:
            if plan['journal'].read_bytes()==plan['local_raw'] and not plan['directory'].exists():
                os.rename(quarantined,plan['directory'])
            raise
        _once(folder/'local_reconciliation_applied.json',canonical({'status':'local_cache_reconciled',
              'remote_journal_sha256':REMOTE_SHA,'preserved_unused_directory':str(quarantined),
              'archive_manifest_path':backed['manifest_path'],'archive_remote_commit':backed['remote_commit'],
              'scientific_fit_repeated':False,'budget_reset':False})+b'\n')
        return {'status':'unpublished_local_prelaunch_tail_archived_and_reconciled','archive_directory':str(folder),
                'archive_backup':backed,'remote_state_id':STATE_ID,'remote_journal_sha256':REMOTE_SHA,
                'completed_fits_preserved':COMPLETED_FITS,'scientific_fit_repeated':False,'budget_reset':False,
                'prior_failed_transport_conservative_seconds':FAILED_TRANSPORT_CONSERVATIVE_SECONDS,
                'recovery_wall_seconds':max(0.,clock()-started)}
