"""Conservatively close the one expired, lost-runtime reservation without retry.

This registered recovery does not invent an elapsed measurement, optimizer
count, worker result, weights or prediction. It consumes the complete existing
allocation, leaves the outcome unknown, and preserves every earlier backup.
Only a compare-and-swap successor of the exact declared control state may close
the reservation. A later worker result never becomes a replacement attempt.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile

from e2_reduced_github_backup import BackupPause, canonical, digest_bytes, digest_file, git_blob_sha
from e2_reduced_github_state import (GitHubState, MAX_METADATA_BYTES, _excluded,
    _relative, _no_active_lease)
from reconcile_e2_github_completed import _rows, CONTROLLERS

VERSION = 'e2-expired-lease-disposition-v1'
POLICY_VERSION = 'e2-expired-lease-policy-v1'
REGISTRATION = 'experiments/e2_a100_recovery_20261113/registration.json'
SOURCE = 'scripts/e2_expired_lease_recovery.py'
LEDGER = 'shared_budget/pilot_budget.jsonl'
KNOWN_STATE = '32a0d3eba1cf95ba73c6bf99816425e4ed34183f888f618f01f9a362c9c24a9e'
KNOWN_ATTEMPT = 'f086a89ea56abb834542d12ccee473a6386d12b9de8f456043d78010ada01ede'
KNOWN_SPEC = '12cb4f0dfbbfbeb145f11288e50c011c0552c3e42b1304dfd9ba6f3cca7e8425'
KNOWN_TIP = 'ff4407e6835ac9e8e6c0f48d4597ecd5fe63ac818849305a70b6914dada7ec20'
FUNDING = 'f540227c23c572b9ca7fb36920588727b029484a'
FUNDING_POLICY = 'a1cf19c7a4c88f525afa5d7369546b4e31a898447cee56830ec520bd542e350f'
RESOURCE = 'b352ce7f27cd7b55e4141fbfb1548f506ed80987'
RESOURCE_POLICY = '55bdc97156f3ea8524f74e2e077f66ace6ced49a79cc51fc07d2d4972b2a5d23'
ACCOUNTING = 'conservative_reserved_allocation_not_measured_elapsed'


def policy_template():
    p = {'version': POLICY_VERSION, 'expected_state_id': KNOWN_STATE,
        'attempt_id': KNOWN_ATTEMPT, 'worker_spec_sha256': KNOWN_SPEC,
        'reservation_event_sha256': KNOWN_TIP, 'expected_ledger_events': 235,
        'reserved_at_utc': '2026-11-13T02:01:04.368851+00:00',
        'reservation_seconds': 1860.0, 'expiry_margin_seconds': 300,
        'expected_backup_count': 97, 'effective_total_hours': 72,
        'funding_registration_commit': FUNDING, 'funding_policy_sha256': FUNDING_POLICY,
        'resource_registration_commit': RESOURCE, 'resource_policy_sha256': RESOURCE_POLICY,
        'accounting_basis': ACCOUNTING, 'scientific_retry_allowed': False,
        'outcome_remains_unknown': True, 'budget_reset': False}
    p['sha256'] = digest_bytes(canonical(p))
    return p


def _checked(value):
    if (not isinstance(value, dict)
            or value.get('sha256') != digest_bytes(canonical({k:v for k,v in value.items() if k!='sha256'}))):
        raise BackupPause('expired_recovery_checksum_changed')
    return value


def verify_policy(policy):
    _checked(policy)
    if policy != policy_template():
        raise BackupPause('only_the_declared_expired_opportunity_may_be_closed')
    return policy


def verify_registration(policy, commit, root):
    verify_policy(policy)
    if not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit):
        raise BackupPause('published_expired_recovery_registration_required')
    def git(*args):
        return subprocess.check_output(['git','-C',str(root),*args], stderr=subprocess.DEVNULL)
    try:
        raw = git('show', commit+':'+REGISTRATION)
        source = git('show', commit+':'+SOURCE)
        stamp = git('show','-s','--format=%cI',commit).decode().strip()
        record = json.loads(raw)
        at = datetime.fromisoformat(stamp.replace('Z','+00:00'))
    except (OSError, ValueError, subprocess.CalledProcessError):
        raise BackupPause('actual_expired_recovery_git_object_required') from None
    expected = record.get('files_sha256',{}).get(SOURCE)
    if (record.get('protocol_id')!='e2-reduced-delivery-v1'
            or record.get('expired_lease_policy')!=policy
            or expected!=hashlib.sha256(source).hexdigest()
            or expected!=digest_file(__file__) or at.tzinfo is None
            or at>datetime.now(timezone.utc)):
        raise BackupPause('expired_recovery_source_or_publication_binding_changed')
    return {'registration_commit':commit, 'registration_path':REGISTRATION,
        'registration_file_sha256':digest_bytes(raw), 'source_sha256':expected,
        'policy_sha256':policy['sha256'], 'registered_at_utc':at.isoformat()}


def _idle(proc_root):
    names=set(CONTROLLERS)|{'e2_resource_worker.py'}
    for path in Path(proc_root).glob('[0-9]*/cmdline'):
        if int(path.parent.name)==os.getpid(): continue
        try: args=path.read_bytes().decode().split('\0')
        except (OSError, UnicodeError): continue
        if any(a and not any(c.isspace() for c in a) and Path(a).name in names for a in args):
            raise BackupPause('existing_controller_prevents_expired_recovery')


@contextmanager
def _lock(output):
    root=Path(output)/'shared_budget';root.mkdir(parents=True,exist_ok=True)
    path=root/'.pilot-budget.lock'
    if any(p.is_symlink() for p in (path,*path.parents)):
        raise BackupPause('symlinked_expired_recovery_lock')
    fd=os.open(path,os.O_RDWR|os.O_CREAT|getattr(os,'O_NOFOLLOW',0),0o600)
    with os.fdopen(fd,'r+b') as stream:
        try: fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError: raise BackupPause('existing_budget_writer_prevents_expired_recovery') from None
        try: yield
        finally: fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def _metadata(state, remote, head):
    """Verify immutable archive objects; restore metadata without opening a lease."""
    descriptors=remote.get('metadata_deltas',[])
    if not isinstance(descriptors,list) or not 1<=len(descriptors)<=4096:
        raise BackupPause('invalid_expired_control_archive_index')
    def fetch(d):
        _relative(d['path'])
        raw=state._content_bytes(d['path'],head)
        if (raw is None or len(raw)!=d['bytes'] or digest_bytes(raw)!=d['sha256']
                or git_blob_sha(raw)!=d['git_blob_sha']):
            raise BackupPause('expired_control_archive_changed')
        return raw
    # No writes or mutations occur in these parallel immutable reads.
    with ThreadPoolExecutor(max_workers=4) as pool:
        archives=list(pool.map(fetch,descriptors))
    wanted=remote['files'];found={};total=0
    for raw in reversed(archives):
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            seen=set();archive_total=0
            for item in archive.infolist():
                name=_relative(item.filename).as_posix()
                if name in seen or _excluded(name) or item.is_dir():
                    raise BackupPause('unsafe_expired_control_archive_member')
                seen.add(name);archive_total+=item.file_size
                if archive_total>MAX_METADATA_BYTES:raise BackupPause('expired_control_archive_too_large')
                if name not in wanted or name in found:continue
                data=archive.read(item);identity=wanted[name]
                # Older versions in earlier archives cannot shadow current data.
                if len(data)!=identity['bytes'] or digest_bytes(data)!=identity['sha256']:continue
                found[name]=data;total+=len(data)
                if total>MAX_METADATA_BYTES:raise BackupPause('expired_control_metadata_too_large')
    if set(found)!=set(wanted):raise BackupPause('expired_control_current_metadata_missing')
    return found


def _utc(value):
    result=datetime.fromisoformat(value.replace('Z','+00:00'))
    if result.tzinfo is None:raise BackupPause('timezone_required_for_expired_recovery')
    return result.astimezone(timezone.utc)


def _validate_open(remote, metadata, policy, now):
    if (remote['state_id']!=policy['expected_state_id']
            or len(remote.get('result_backups',{}))!=policy['expected_backup_count']
            or 'jobs/'+KNOWN_ATTEMPT in remote.get('result_backups',{})):
        raise BackupPause('expired_recovery_expected_control_changed')
    rows,active=_rows(metadata[LEDGER])
    if (len(rows)!=policy['expected_ledger_events'] or rows[-1]['sha256']!=KNOWN_TIP
            or set(active)!={KNOWN_ATTEMPT}):
        raise BackupPause('expired_recovery_reservation_or_ledger_changed')
    reservation=active[KNOWN_ATTEMPT]
    if (reservation.get('sha256')!=policy['reservation_event_sha256']
            or reservation.get('utc')!=policy['reserved_at_utc']
            or reservation.get('reservation_seconds')!=policy['reservation_seconds']
            or reservation.get('effective_total_hours')!=72
            or reservation.get('funding_registration_commit')!=FUNDING
            or reservation.get('funding_policy_sha256')!=FUNDING_POLICY
            or reservation.get('metadata',{}).get('resource_registration_commit')!=RESOURCE
            or reservation.get('metadata',{}).get('resource_policy_sha256')!=RESOURCE_POLICY
            or reservation.get('metadata',{}).get('worker_spec_sha256')!=KNOWN_SPEC):
        raise BackupPause('expired_recovery_immutable_reservation_binding_changed')
    if (now-_utc(reservation['utc'])).total_seconds() < reservation['reservation_seconds']+policy['expiry_margin_seconds']:
        raise BackupPause('remote_reservation_not_yet_expired')
    spec_path='jobs/'+KNOWN_ATTEMPT+'/worker_spec.json'
    spec=json.loads(metadata[spec_path]);job=spec.get('job',{})
    if (digest_bytes(canonical(spec))!=KNOWN_SPEC
            or job.get('logical_job_id')!=KNOWN_ATTEMPT
            or digest_bytes(canonical({k:v for k,v in job.items() if k!='logical_job_id'}))!=KNOWN_ATTEMPT
            or job.get('model_id')!='TSMixer' or job.get('configuration_id')!='C02'
            or job.get('fold_origin')!='2024-07-01' or job.get('seed')!=271201
            or spec.get('fit_wall_seconds')!=1800):
        raise BackupPause('expired_recovery_original_worker_spec_changed')
    return rows,reservation,spec


def _charge(rows, reservation, policy, proof, now):
    # This is a conservative accounting allocation, deliberately NOT a duration
    # measured after VM loss. It grants no credit relative to the open lease.
    row={'version':'e2-amended-pilot-budget-v1','sequence':len(rows),'kind':'charged',
        'reconciliation_sha256':reservation['reconciliation_sha256'],
        'previous_sha256':rows[-1]['sha256'],'utc':now.isoformat(),
        'attempt_id':KNOWN_ATTEMPT,'reservation_seconds':policy['reservation_seconds'],
        'elapsed_seconds':policy['reservation_seconds'],'reservation_overrun':False,
        'status':'failed','funding_policy_sha256':FUNDING_POLICY,
        'funding_registration_commit':FUNDING,'effective_total_hours':72,
        'resource_use':{'worker_spec_sha256':KNOWN_SPEC,'failure':'expired_runtime_outcome_unknown',
            'accounting_basis':ACCOUNTING,'elapsed_seconds_measured':False,
            'measured_elapsed_seconds':None,'optimizer_steps':None,
            'optimizer_steps_known_lower_bound':0,'partial_optimizer_update_may_have_occurred':True,
            'outcome':'unknown','automatic_retry':False,
            'recovery_registration_commit':proof['registration_commit'],
            'recovery_policy_sha256':policy['sha256']}}
    row['sha256']=digest_bytes(canonical(row));return row


def _selection(job, disposition, raw):
    row={k:job[k] for k in ('model_id','configuration_id','fold_origin','seed',
        'parent_trial_id','configuration_sha256')}
    row.update(status='failed',rmse_original=None,best_epoch=None,fatal=False,systematic_failure=False,
        validation_count=83,predictions_produced=0,validation_count_scope='admitted_calendar_not_predictions',
        reason='Expired lost-runtime opportunity; outcome unknown, allocation charged conservatively; no retry',
        failure_code='expired_runtime_outcome_unknown',run_receipt_file='recovery_disposition.json',
        run_receipt_sha256=digest_bytes(raw),budget_receipt_sha256=disposition['budget_charge']['sha256'],
        replacement_performed=False,automatic_retry=False,
        resource_use={'cache_reuse':False,'optimizer_steps':None,'optimizer_steps_known_lower_bound':0,
            'partial_optimizer_update_may_have_occurred':True,'outcome_unknown':True,
            'accounting_basis':ACCOUNTING,'new_training_performed':False})
    return row


def verify_recovery_disposition(job,directory,budget,source,validation_dates):
    """Selector admission for this explicitly unknown failed opportunity only."""
    directory=Path(directory);raw=(directory/'recovery_disposition.json').read_bytes()
    report=_checked(json.loads(raw));policy=verify_policy(report.get('recovery_policy'))
    spec=json.loads((directory/'worker_spec.json').read_bytes())
    if (report.get('version')!=VERSION or report.get('status')!='unknown_outcome_not_retried'
            or report.get('job')!=job or job.get('logical_job_id')!=KNOWN_ATTEMPT
            or spec.get('job')!=job or spec.get('source_sha256')!=source
            or digest_bytes(canonical(spec))!=KNOWN_SPEC or len(validation_dates)!=83
            or report.get('original_remote_state_id')!=KNOWN_STATE
            or report.get('original_reservation_sha256')!=KNOWN_TIP
            or report.get('recovery_policy_sha256')!=policy['sha256']
            or report.get('measured_elapsed_seconds') is not None
            or report.get('optimizer_steps') is not None
            or report.get('optimizer_steps_known_lower_bound')!=0
            or report.get('partial_optimizer_update_may_have_occurred') is not True
            or report.get('scientific_fit_repeated') is not False):
        raise BackupPause('unknown_disposition_identity_or_claim_changed')
    if any((directory/n).exists() for n in ('result.json','failure_result.json','worker_failure.json','request.json')):
        raise BackupPause('scientific_output_conflicts_with_unknown_disposition')
    allowed={'worker_spec.json','recovery_disposition.json','selection_row.json'}
    if any(p.is_file() and p.name!='.writer.lock' and p.relative_to(directory).as_posix() not in allowed
            for p in directory.rglob('*')):
        raise BackupPause('unexpected_artifact_in_unknown_disposition')
    import e2_known_setup_failure_selection as setup
    charges=[r for r in setup._budget_events(budget) if r.get('kind')=='charged' and r.get('attempt_id')==KNOWN_ATTEMPT]
    if len(charges)!=1 or charges[0]!=report.get('budget_charge'):
        raise BackupPause('unknown_disposition_cost_receipt_changed')
    charge=charges[0];_checked(charge);usage=charge.get('resource_use',{})
    proof=report.get('recovery_registration_proof',{})
    if (charge.get('status')!='failed' or charge.get('elapsed_seconds')!=1860
            or charge.get('reservation_seconds')!=1860 or charge.get('reservation_overrun') is not False
            or usage.get('outcome')!='unknown' or usage.get('worker_spec_sha256')!=KNOWN_SPEC
            or usage.get('accounting_basis')!=ACCOUNTING or usage.get('elapsed_seconds_measured') is not False
            or usage.get('optimizer_steps') is not None or usage.get('automatic_retry') is not False
            or proof.get('policy_sha256')!=policy['sha256']
            or proof.get('source_sha256')!=digest_file(__file__)
            or proof.get('registration_path')!=REGISTRATION
            or proof.get('registration_commit')!=usage.get('recovery_registration_commit')
            or not re.fullmatch('[0-9a-f]{40}',str(proof.get('registration_commit','')))):
        raise BackupPause('unknown_disposition_accounting_basis_changed')
    row=_selection(job,report,raw)
    saved=directory/'selection_row.json'
    if saved.exists() and json.loads(saved.read_bytes())!=row:
        raise BackupPause('unknown_disposition_selection_row_changed')
    return row,report


def _write_new(path,raw):
    path=Path(path)
    if path.is_symlink():raise BackupPause('symlinked_expired_recovery_path')
    if path.exists():
        if path.read_bytes()!=raw:raise BackupPause('existing_expired_recovery_bytes_differ')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())


class _TerminalGuardClient:
    """Recheck late results on EVERY compare-and-swap publication attempt."""
    def __init__(self,client):self._client=client
    def __getattr__(self,name):return getattr(self._client,name)
    def _branch_head(self):
        head=self._client._branch_head()
        if self._client._content('runs/'+KNOWN_ATTEMPT+'/manifest.json',head) is not None:
            raise BackupPause('late_terminal_artifacts_preserved_without_unknown_closure')
        return head


def reconcile_expired_lease(output,client,*,recovery_policy,registration_commit,
        registration_root,proc_root=Path('/proc'),now=None):
    """Close exactly one registered expired allocation; never train or repeat it."""
    policy=verify_policy(recovery_policy)
    proof=verify_registration(policy,registration_commit,registration_root)
    now=datetime.now(timezone.utc) if now is None else now
    if now.tzinfo is None:raise BackupPause('timezone_required_for_expired_recovery')
    _idle(proc_root);state=GitHubState(output,client)
    head=client._branch_head();remote=state._current(head)
    if remote is None:raise BackupPause('prior_control_required_for_expired_recovery')
    if remote.get('output_path')!=str(state.output):raise BackupPause('expired_recovery_output_changed')
    disposition='jobs/'+KNOWN_ATTEMPT+'/recovery_disposition.json'
    if remote['state_id']!=KNOWN_STATE:
        # The normal restore path validates all newly published data. An
        # existing recovery successor needs no second charge or modification.
        if disposition in remote.get('files',{}):
            return {'status':'expired_disposition_already_recorded','reconciled':False,
                'attempt_id':KNOWN_ATTEMPT,'remote_state_id':remote['state_id']}
        raise BackupPause('different_remote_state_requires_inspection_before_recovery')
    if client._content('runs/'+KNOWN_ATTEMPT+'/manifest.json',head) is not None:
        raise BackupPause('terminal_artifacts_exist_do_not_mark_the_attempt_unknown')
    metadata=_metadata(state,remote,head)
    rows,reservation,spec=_validate_open(remote,metadata,policy,now)
    jobdir=state.output/'jobs'/KNOWN_ATTEMPT
    if jobdir.exists():
        for p in jobdir.rglob('*'):
            if (p.is_file() and p.name not in ('recovery_disposition.json','selection_row.json')
                    and p.relative_to(state.output).as_posix() not in remote['files']):
                raise BackupPause('local_scientific_artifacts_prevent_unknown_recovery')
    with _lock(state.output):
        _idle(proc_root)
        pending=jobdir/'recovery_disposition.json'
        if pending.exists():
            old=_checked(json.loads(pending.read_bytes()))
            if (old.get('recovery_registration_proof')!=proof
                    or old.get('original_remote_state_id')!=KNOWN_STATE
                    or old.get('recovery_policy')!=policy):
                raise BackupPause('different_local_expired_recovery_is_preserved')
            now=_utc(old['observed_at_utc'])
            _validate_open(remote,metadata,policy,now)
        charge=_charge(rows,reservation,policy,proof,now)
        report={'version':VERSION,'status':'unknown_outcome_not_retried','job':spec['job'],
            'attempt_id':KNOWN_ATTEMPT,'original_remote_state_id':KNOWN_STATE,
            'original_reservation_sha256':KNOWN_TIP,'worker_spec_sha256':KNOWN_SPEC,
            'recovery_policy':policy,'recovery_policy_sha256':policy['sha256'],
            'recovery_registration_proof':proof,'budget_charge':charge,
            'measured_elapsed_seconds':None,'optimizer_steps':None,
            'optimizer_steps_known_lower_bound':0,'partial_optimizer_update_may_have_occurred':True,
            'scientific_fit_repeated':False,'budget_credit_granted':False,
            'accounting_basis':ACCOUNTING,'observed_at_utc':now.isoformat()}
        report['sha256']=digest_bytes(canonical(report));raw=canonical(report)+b'\n'
        row_raw=canonical(_selection(spec['job'],report,raw))+b'\n'
        extras={disposition:raw,
            'reconciliation/expired-'+KNOWN_ATTEMPT+'.json':raw,
            'jobs/'+KNOWN_ATTEMPT+'/selection_row.json':row_raw}
        intended_ledger=metadata[LEDGER]+canonical(charge)+b'\n'
        local,local_bytes=state._scan()
        for name,record in local.items():
            if name==LEDGER:
                if local_bytes[name] not in (metadata[LEDGER],intended_ledger):
                    raise BackupPause('local_expired_ledger_has_unrelated_or_double_charge')
            elif name in extras:
                if local_bytes[name]!=extras[name]:
                    raise BackupPause('local_expired_recovery_transaction_changed')
            elif record!=remote['files'].get(name):
                raise BackupPause('local_control_diverges_from_expired_remote_state')
        for name,payload in metadata.items():
            if name!=LEDGER:_write_new(state.output/name,payload)
        # A write-ahead disposition makes a crash/network retry deterministic:
        # the same timestamp and charge are reused, never appended twice.
        for name,payload in extras.items():_write_new(state.output/name,payload)
        path=state.output/LEDGER
        if not path.exists():_write_new(path,metadata[LEDGER])
        # Recheck the original attempt before persisting a successor. A late
        # immutable result is preserved for a different reconciliation path.
        current_head=client._branch_head();current=state._current(current_head)
        if current['state_id']!=KNOWN_STATE:raise BackupPause('remote_lease_advanced_during_recovery')
        if client._content('runs/'+KNOWN_ATTEMPT+'/manifest.json',current_head) is not None:
            raise BackupPause('late_terminal_artifacts_preserved_without_unknown_closure')
        if path.read_bytes() not in (metadata[LEDGER],intended_ledger):
            raise BackupPause('local_ledger_changed_before_expired_closure')
        # Local bytes are a disposable cache; the authoritative commit is the
        # expected-state CAS below. Nothing is admitted until it succeeds.
        if path.read_bytes()==metadata[LEDGER]:
            with path.open('ab') as stream:
                stream.write(canonical(charge)+b'\n');stream.flush();os.fsync(stream.fileno())
        _no_active_lease(state.output)
        state._state=remote;state._files=remote['files'];state._archives=remote['metadata_deltas']
        state._backups=dict(remote['result_backups']);state._restored=True
        state.client=_TerminalGuardClient(client)
        result=state.snapshot('registered_expired_unknown_allocation_closed')
        _idle(proc_root)
    return {'status':'expired_unknown_allocation_conservatively_closed','reconciled':True,
        'attempt_id':KNOWN_ATTEMPT,'outcome':'unknown','charged_allocation_seconds':1860,
        'measured_elapsed_seconds':None,'optimizer_steps':None,'scientific_fit_repeated':False,
        'previous_backup_count':len(remote['result_backups']),**result}
