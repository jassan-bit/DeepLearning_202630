"""Reconcile an already completed local worker after a GitHub upload failure.

No optimizer, clock grant, retry, charge fabrication or worker launch occurs.
The remote lease is released only after its exact existing successful charge,
specification, result and artifacts have been proved and backed up.
"""
from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path
import zipfile

from e2_reduced_github_backup import BackupPause, canonical, digest_bytes, digest_file, git_blob_sha
from e2_reduced_github_state import GitHubState

LEDGER = 'shared_budget/pilot_budget.jsonl'
CONTROLLERS = {'run_e2_cuda_search.py','run_e2_optimized_selection.py',
    'run_e2_optimized_selection_v2.py','run_e2_amended_pilots.py',
    'run_e2_reduced_selection.py','run_e2_reduced_delivery.py',
    'run_e2_reduced_github_delivery.py','run_e2_reduced_github72_delivery.py'}


def _idle(proc_root):
    for path in Path(proc_root).glob('[0-9]*/cmdline'):
        if int(path.parent.name) == os.getpid():
            continue
        try:
            arguments = path.read_bytes().decode().split('\0')
        except (OSError, UnicodeError):
            continue
        if any(word and not any(c.isspace() for c in word)
               and Path(word).name in CONTROLLERS for word in arguments):
            raise BackupPause('stop_existing_controller_before_completed_reconciliation')


def _rows(raw):
    if not raw or not raw.endswith(b'\n'):
        raise BackupPause('complete_existing_budget_journal_required')
    rows=[];previous='0'*64;active={};attempts=set();anchor=None
    for sequence,line in enumerate(raw.splitlines()):
        value=json.loads(line)
        body={k:v for k,v in value.items() if k!='sha256'}
        if (value.get('sequence')!=sequence or value.get('previous_sha256')!=previous
                or value.get('sha256')!=digest_bytes(canonical(body))
                or value.get('version')!='e2-amended-pilot-budget-v1'):
            raise BackupPause('completed_reconciliation_budget_chain_changed')
        if anchor is None:anchor=value.get('reconciliation_sha256')
        if not anchor or value.get('reconciliation_sha256')!=anchor:
            raise BackupPause('completed_reconciliation_budget_anchor_changed')
        previous=value['sha256'];kind=value.get('kind')
        if kind=='reserved':
            identity=value.get('attempt_id');amount=value.get('reservation_seconds')
            if (not identity or identity in attempts or type(amount)not in(int,float)
                    or not math.isfinite(amount) or amount<=0):
                raise BackupPause('invalid_existing_reservation')
            attempts.add(identity);active[identity]=value
        elif kind=='charged':
            identity=value.get('attempt_id');amount=value.get('elapsed_seconds')
            if (identity not in active or type(amount)not in(int,float)
                    or not math.isfinite(amount) or amount<0
                    or value.get('reservation_seconds')!=active[identity]['reservation_seconds']):
                raise BackupPause('invalid_existing_completed_charge')
            if value.get('status')not in('completed','failed','paused_clean'):
                raise BackupPause('unknown_existing_charge_requires_manual_reconciliation')
            active.pop(identity)
        elif kind=='overhead':
            amount=value.get('elapsed_seconds')
            if type(amount)not in(int,float) or not math.isfinite(amount) or amount<0:
                raise BackupPause('invalid_existing_overhead')
        elif kind!='progress':
            raise BackupPause('unknown_existing_budget_record')
        rows.append(value)
    return rows,active


def _remote_ledger(state,remote,head):
    expected=remote['files'].get(LEDGER)
    if not expected:raise BackupPause('remote_ledger_identity_missing')
    for descriptor in reversed(remote.get('metadata_deltas',[])):
        raw=state._content_bytes(descriptor['path'],head)
        if (raw is None or len(raw)!=descriptor['bytes']
                or digest_bytes(raw)!=descriptor['sha256']
                or git_blob_sha(raw)!=descriptor['git_blob_sha']):
            raise BackupPause('remote_control_delta_changed')
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            candidates=[x for x in archive.infolist() if x.filename==LEDGER]
            if len(candidates)>1:raise BackupPause('duplicate_remote_ledger_member')
            if candidates:
                value=archive.read(candidates[0])
                if len(value)==expected['bytes'] and digest_bytes(value)==expected['sha256']:
                    return value
    raise BackupPause('matching_remote_budget_prefix_not_found')


def reconcile_completed_local(output,client,*,proc_root=Path('/proc')):
    """Upload a proven closed local fit before normal remote-state restoration.

    A missing control, or control with no open lease, is left to the ordinary
    restore path. Any unknown/failed/divergent continuation is preserved and
    rejected. Only one already successful local charge can close one lease.
    """
    _idle(proc_root)
    state=GitHubState(output,client);head=client._branch_head();remote=state._current(head)
    if remote is None:return {'status':'no_remote_control_to_reconcile','reconciled':False}
    if remote.get('output_path')!=str(state.output):raise BackupPause('reconciliation_output_path_changed')
    remote_raw=_remote_ledger(state,remote,head);old,active=_rows(remote_raw)
    if not active:return {'status':'remote_has_no_open_lease','reconciled':False}
    if len(active)!=1:raise BackupPause('multiple_open_leases_require_manual_reconciliation')
    identity,reservation=next(iter(active.items()))
    local_raw=(state.output/LEDGER).read_bytes()
    if not local_raw.startswith(remote_raw):raise BackupPause('local_budget_is_not_exact_remote_prefix_extension')
    rows,local_active=_rows(local_raw)
    if local_active:raise BackupPause('local_worker_cost_is_still_unknown')
    suffix=rows[len(old):]
    if (not suffix or suffix[0].get('kind')!='charged'
            or any(x.get('kind')!='overhead' for x in suffix[1:])):
        raise BackupPause('only_existing_completed_charge_and_overhead_can_reconcile')
    charged=suffix[0];usage=charged.get('resource_use',{})
    if (charged.get('attempt_id')!=identity or charged.get('status')!='completed'
            or charged.get('reservation_overrun')is not False
            or usage.get('automatic_retry')is not False
            or usage.get('worker_spec_sha256')!=reservation.get('metadata',{}).get('worker_spec_sha256')):
        raise BackupPause('successful_matching_local_charge_required')
    candidates=[]
    for relative,record in remote['files'].items():
        if relative==LEDGER:continue
        path=state.output/relative
        if (not path.resolve().is_relative_to(state.output.resolve()) or path.is_symlink()
                or not path.is_file() or path.stat().st_size!=record['bytes']
                or digest_file(path)!=record['sha256']):
            raise BackupPause('preexisting_control_metadata_changed')
        if path.name in('worker_spec.json','worker-spec.json'):
            spec=json.loads(path.read_text())
            if digest_bytes(canonical(spec))==usage.get('worker_spec_sha256'):
                candidates.append((relative,path,spec))
    if len(candidates)!=1:raise BackupPause('unique_remote_bound_worker_spec_required')
    relative,spec_path,spec=candidates[0];directory=spec_path.parent
    if spec.get('output') and Path(spec['output']).absolute()!=directory:
        raise BackupPause('worker_spec_output_changed')
    from e2_reduced_supervision import verified_worker_result
    result=verified_worker_result(directory,usage['worker_spec_sha256'])
    if (result is None or digest_file(directory/'result.json')!=usage.get('result_sha256')
            or result.get('optimizer_steps')!=usage.get('optimizer_steps')
            or type(result.get('optimizer_steps'))is not int or result['optimizer_steps']<=0):
        raise BackupPause('completed_result_or_optimizer_cost_binding_changed')
    report={'version':'e2-completed-upload-reconciliation-v1','status':'closed_local_fit_verified',
        'attempt_id':identity,'remote_state_id':remote['state_id'],
        'worker_spec_sha256':usage['worker_spec_sha256'],'result_sha256':usage['result_sha256'],
        'existing_charge_sha256':charged['sha256'],'existing_elapsed_seconds':charged['elapsed_seconds'],
        'local_ledger_tip_sha256':rows[-1]['sha256'],'optimizer_steps':result['optimizer_steps'],
        'scientific_fit_repeated':False,'budget_credit_granted':False}
    recovery='reconciliation/completed-'+identity+'.json';payload=canonical(report)+b'\n'
    local_files,_=state._scan();pending_root=directory.relative_to(state.output).as_posix()+'/'
    for path in set(local_files)-set(remote['files']):
        if not path.startswith(pending_root) and path!=recovery:
            raise BackupPause('unrelated_local_changes_require_manual_reconciliation')
    target=state.output/recovery;target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if target.read_bytes()!=payload:raise BackupPause('existing_reconciliation_receipt_changed')
    else:
        with target.open('xb')as stream:
            stream.write(payload);stream.flush();os.fsync(stream.fileno())
    state._state=remote;state._files=remote['files'];state._archives=remote['metadata_deltas']
    state._backups=dict(remote.get('result_backups',{}));state._restored=True
    state._pending={identity:dict(usage)}
    # The existing client uploads/verifies all tensors/metadata first. The state
    # class then performs its expected-state CAS; a competing controller wins
    # safely by making our snapshot fail instead of clearing its lease.
    backed=state.persist_result(directory,result)
    _idle(proc_root)
    return {**report,'status':'completed_upload_reconciled','reconciled':True,'backup':backed}
