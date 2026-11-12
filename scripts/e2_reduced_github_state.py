"""GitHub-authoritative control state for the immutable reduced E2 engines.

Local /content is a disposable cache. A remote compare-and-swap snapshot is
required after reserving a lease and before returning control to a GPU worker.
Completed leases are published only after their result artifacts are backed up.
An interrupted remote lease blocks execution rather than creating fresh credit.
"""
from __future__ import annotations
import base64
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

from e2_reduced_github_backup import BackupPause, canonical, digest_bytes, digest_file, git_blob_sha

VERSION = 'e2-github-primary-state-v1'
PREFIX = 'control/github-primary-v1'
POINTER = PREFIX + '/head.json'
SCIENTIFIC_COMMIT = '1e8745497b484518f766b6da5c19be8362d9add7'
TENSORS = {'.pt','.pth','.ckpt','.pkl','.npz','.safetensors','.bin'}
MAX_METADATA_BYTES = 256 * 1024**2


def _relative(value):
    p=PurePosixPath(value)
    if not isinstance(value,str) or not value or p.is_absolute() or '..' in p.parts or str(p)!=value:
        raise BackupPause('invalid_control_relative_path')
    return p


def _excluded(relative):
    parts=PurePosixPath(relative).parts
    return (not parts or parts[0]=='assets' or any(x.startswith('.') or x.lower().startswith('private') for x in parts)
        or PurePosixPath(relative).suffix.lower() in TENSORS)


def _decode(content):
    return base64.b64decode(content,validate=False)


def _checked_state(raw):
    value=json.loads(raw)
    identity=value.get('state_id')
    body={k:v for k,v in value.items() if k!='state_id'}
    if (value.get('version')!=VERSION or value.get('scientific_commit')!=SCIENTIFIC_COMMIT
            or identity!=digest_bytes(canonical(body))):
        raise BackupPause('control_state_identity_or_hash_changed')
    return value


def _no_active_lease(root):
    path=Path(root)/'shared_budget/pilot_budget.jsonl'
    if not path.exists():return
    raw=path.read_bytes()
    if raw and not raw.endswith(b'\n'):raise BackupPause('partial_control_budget_tail')
    active={};previous='0'*64
    for sequence,line in enumerate(raw.splitlines()):
        row=json.loads(line);body={k:v for k,v in row.items() if k!='sha256'}
        if (row.get('sequence')!=sequence or row.get('previous_sha256')!=previous
                or row.get('sha256')!=digest_bytes(canonical(body))):
            raise BackupPause('remote_budget_chain_changed')
        previous=row['sha256'];kind=row['kind']
        if kind=='reserved':active[row['attempt_id']]=row['reservation_seconds']
        elif kind=='charged':
            if row.get('status') not in ('completed','failed','paused_clean'):
                raise BackupPause('unknown_remote_attempt_requires_reconciliation')
            active.pop(row['attempt_id'],None)
    if active:raise BackupPause('unclosed_remote_lease_preserved_reconciliation_required')


class GitHubState:
    def __init__(self,output,backup_client):
        self.output=Path(output).absolute();self.client=backup_client
        self._state=None;self._restored=False;self._pending={}
        self._backups={};self._archives=[];self._files={}

    def _content_bytes(self,path,head):
        item=self.client._content(path,head)
        if item is None:return None
        content=item.get('content')
        if not content or item.get('encoding')=='none':
            content=self.client._request('GET','/git/blobs/'+item['sha'])['content']
        raw=_decode(content)
        if git_blob_sha(raw)!=item['sha']:raise BackupPause('control_git_blob_hash_changed')
        return raw

    def _current(self,head):
        raw=self._content_bytes(POINTER,head)
        return None if raw is None else _checked_state(raw)

    def _scan(self):
        files={};data={};total=0
        if self.output.is_symlink():raise BackupPause('symlinked_control_root')
        for path in sorted(self.output.rglob('*')) if self.output.exists() else []:
            relative=path.relative_to(self.output).as_posix()
            if _excluded(relative):continue
            if path.is_symlink():raise BackupPause('symlink_in_control_state')
            if not path.is_file():continue
            raw=path.read_bytes();total+=len(raw)
            if total>MAX_METADATA_BYTES:raise BackupPause('control_metadata_size_limit')
            token=getattr(self.client,'_token',None)
            if token and token.encode() in raw:raise BackupPause('credential_in_control_state')
            # Raw market caches are never backed up, regardless of filename.
            if path.suffix.lower() in {'.csv','.tsv'}:
                header=raw.splitlines()[0].decode('utf-8-sig').lower() if raw else ''
                if sum(x in header.split(',') for x in ('open','high','low','close','volume'))>=3:
                    raise BackupPause('raw_market_control_file_refused')
            files[relative]={'sha256':digest_bytes(raw),'bytes':len(raw)};data[relative]=raw
        return files,data

    def _publish(self,state,delta):
        expected=None if self._state is None else self._state['state_id']
        raw=canonical(state)+b'\n'
        for _ in range(self.client.max_conflicts):
            head=self.client._branch_head();current=self._current(head)
            actual=None if current is None else current['state_id']
            if actual!=expected:raise BackupPause('another_session_advanced_control_state')
            tree=self.client._request('GET','/git/commits/'+head)['tree']['sha'];entries=[]
            items={POINTER:raw,PREFIX+'/states/'+state['state_id']+'.json':raw}
            if delta is not None:items[delta[0]]=delta[1]
            for path,payload in sorted(items.items()):
                blob=self.client._request('POST','/git/blobs',{'encoding':'base64','content':base64.b64encode(payload).decode()})
                if blob.get('sha')!=git_blob_sha(payload):raise BackupPause('control_uploaded_blob_hash_changed')
                entries.append({'path':path,'mode':'100644','type':'blob','sha':blob['sha']})
            newtree=self.client._request('POST','/git/trees',{'base_tree':tree,'tree':entries})['sha']
            commit=self.client._request('POST','/git/commits',{'message':'Preserve E2 GitHub-primary control state','tree':newtree,'parents':[head]})['sha']
            try:
                from e2_reduced_github_backup import BRANCH
                self.client._request('PATCH','/git/refs/heads/'+BRANCH,{'sha':commit,'force':False})
                self._state=state;return commit
            except BackupPause as error:
                if error.http_status not in (409,422):raise
                # Retry only if control itself is unchanged. Artifact-only
                # branch additions are safe to merge; another lease is not.
        raise BackupPause('control_compare_and_swap_conflict_limit')

    def snapshot(self,reason,*,_completed_upload=False):
        if not self._restored:raise BackupPause('restore_control_before_snapshot')
        if self._pending and not _completed_upload:
            raise BackupPause('completed_result_requires_backup_before_control_release')
        files,data=self._scan();changed={p:b for p,b in data.items() if files[p]!=self._files.get(p)}
        archives=list(self._archives);delta=None
        if changed:
            stream=io.BytesIO()
            with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
                for path,payload in sorted(changed.items()):
                    info=zipfile.ZipInfo(path,date_time=(2026,11,12,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
                    info.external_attr=0o644<<16;archive.writestr(info,payload)
            zipped=stream.getvalue();sha=digest_bytes(zipped);path=PREFIX+'/deltas/'+sha+'.zip'
            archives.append({'path':path,'sha256':sha,'bytes':len(zipped),'git_blob_sha':git_blob_sha(zipped)})
            delta=(path,zipped)
        body={'version':VERSION,'scientific_commit':SCIENTIFIC_COMMIT,'output_path':str(self.output),
            'previous_state_id':None if self._state is None else self._state['state_id'],
            'reason':str(reason),'files':files,'metadata_deltas':archives,'result_backups':self._backups,
            'persistent_destination':'GitHub','Google_Drive_used':False}
        state={**body,'state_id':digest_bytes(canonical(body))}
        commit=self._publish(state,delta)
        self._files=files;self._archives=archives
        return {'state_id':state['state_id'],'remote_commit':commit,'metadata_files':len(files)}

    def restore(self):
        head=self.client._branch_head();state=self._current(head)
        local,_=self._scan()
        if state is None:
            if self.client._content('runs',head) is not None:
                raise BackupPause('existing_results_without_control_require_cost_reconciliation')
            if local:raise BackupPause('local_unregistered_control_requires_reconciliation')
            self.output.mkdir(parents=True,exist_ok=True);self._restored=True
            return {'status':'new_github_primary_campaign','restored':False}
        if state.get('output_path')!=str(self.output):raise BackupPause('control_output_path_changed')
        for path,record in local.items():
            if record!=state['files'].get(path):raise BackupPause('local_control_diverges_preserved_without_overwrite')
        self.output.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='e2-control-restore-') as temporary:
            staged=Path(temporary)
            for descriptor in state['metadata_deltas']:
                _relative(descriptor['path']);raw=self._content_bytes(descriptor['path'],head)
                if (raw is None or len(raw)!=descriptor['bytes'] or digest_bytes(raw)!=descriptor['sha256']
                        or git_blob_sha(raw)!=descriptor['git_blob_sha']):raise BackupPause('control_delta_changed')
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    seen=set();total=0
                    for item in archive.infolist():
                        relative=_relative(item.filename).as_posix()
                        if relative in seen or _excluded(relative):raise BackupPause('unsafe_control_archive_member')
                        seen.add(relative);total+=item.file_size
                        if total>MAX_METADATA_BYTES or item.is_dir():raise BackupPause('control_delta_size_or_type')
                        payload=archive.read(item);target=staged/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
            for relative,record in state['files'].items():
                _relative(relative);source=staged/relative
                if not source.is_file() or source.stat().st_size!=record['bytes'] or digest_file(source)!=record['sha256']:
                    raise BackupPause('restored_control_metadata_hash_changed')
                destination=self.output/relative
                if not destination.exists():
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    with destination.open('xb') as output:output.write(source.read_bytes())
        # This guard is deliberately before any artifact downloads, ledger
        # construction or worker admission. Unknown work never grants credit.
        _no_active_lease(self.output)
        from e2_reduced_github_restore import restore_results
        self._backups=state.get('result_backups',{})
        restored=restore_results(self.output,self.client,self._backups,head)
        self._state=state;self._files=state['files'];self._archives=state['metadata_deltas'];self._restored=True
        return {'status':'github_control_restored','restored':True,'state_id':state['state_id'],**restored}

    def persist_result(self,directory,receipt):
        relative=Path(directory).absolute().relative_to(self.output).as_posix()
        effective=receipt.get('result',receipt)
        matched=[]
        for identity,binding in self._pending.items():
            result_path=Path(directory)/'result.json'
            if (effective.get('worker_spec_sha256')==binding.get('worker_spec_sha256')
                    and result_path.is_file() and digest_file(result_path)==binding.get('result_sha256')):
                matched.append(identity)
        if self._pending and set(matched)!=set(self._pending):
            raise BackupPause('uploaded_result_does_not_close_the_pending_cost_binding')
        backup=self.client.persist_result(directory,receipt)
        self._backups[relative]=backup
        result=self.snapshot('completed_result_uploaded',_completed_upload=True)
        for identity in matched:self._pending.pop(identity)
        return {**backup,'control_state':result}

    @contextmanager
    def budget_factory(self,reconciliation,*,migration,migration_registration_commit,inherited_ledgers=(),budget_class=None):
        if not self._restored:raise BackupPause('restore_control_before_budget')
        _no_active_lease(self.output)
        if budget_class is None:
            from e2_reduced_budget import ReducedBudgetLedger
            budget_class=ReducedBudgetLedger
        underlying=budget_class(self.output/'shared_budget',reconciliation,migration=migration,
            migration_registration_commit=migration_registration_commit,inherited_ledgers=inherited_ledgers)
        facade=_RemoteBudget(underlying,self)
        try:
            self.snapshot('shared_budget_opened')
            yield facade
        finally:
            underlying.close()
            if not self._pending:
                self.snapshot('shared_budget_closed')


class _RemoteBudget:
    def __init__(self,budget,state):self._budget=budget;self._state=state
    def __getattr__(self,name):return getattr(self._budget,name)
    def reserve(self,*args,**kwargs):
        result=self._budget.reserve(*args,**kwargs)
        self._state.snapshot('lease_reserved_before_worker')
        return result
    def charge(self,attempt_id,*args,**kwargs):
        result=self._budget.charge(attempt_id,*args,**kwargs)
        if kwargs.get('status')=='completed':self._state._pending[attempt_id]=dict(kwargs.get('resource_use') or {})
        else:self._state.snapshot('worker_noncompleted_charge')
        return result
    def close(self):
        self._budget.close()
        if not self._state._pending:self._state.snapshot('shared_budget_closed')
