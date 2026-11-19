"""Durable, content-verified audit deltas; no training and no writes to main.

Git commits form a real parent chain, so old checkpoint blobs remain reachable.
Only the new part subtree is traversed recursively; source trees stay shallow.
All checkpoints are handled as bytes, never unpickled during persistence.
"""
from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


ROOT = Path(__file__).resolve().parents[3]
REPOSITORY = 'jassan-bit/DeepLearning_202630'
BRANCH = 'e3-extended-audit-'
PREFIX = 'runs/e3_extended_audit/'
STAMP = r'[0-9]{8}T[0-9]{12}Z'
MAX_FILE = 48 * 1024**2
MAX_PART = 192 * 1024**2
MAX_PAYLOAD = MAX_PART - 2 * 1024**2
MAX_FILES = 900


def _load_store():
    path = ROOT/'experiments/e3_cierre_integral_20261118/orchestration/github_store.py'
    spec = importlib.util.spec_from_file_location('e3_audit_snapshot_validation', path)
    store = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(store)
    store.PROFILES['audit'] = (BRANCH, PREFIX)
    # Scientific figures are produced by our pinned plotting code. Extend only
    # this independently loaded validator; the historical helper stays intact.
    store.ALLOWED = set(store.ALLOWED) | {'.svg'}
    return store


STORE = _load_store()


def _fail(api, message):
    raise api.StorageError(message)


def _check(api, condition, message):
    if not condition:
        _fail(api, message)


def _identity(api, source, protocol, master=None):
    _check(api, api.REPOSITORY == REPOSITORY, 'continuation_repository_mismatch')
    _check(api, re.fullmatch(r'[0-9a-f]{40}', source or '') is not None,
           'continuation_source_sha_required')
    _check(api, re.fullmatch(r'[0-9a-f]{64}', protocol or '') is not None,
           'continuation_protocol_sha_required')
    if master is not None:
        _check(api, re.fullmatch(STAMP, master) is not None, 'continuation_master_id_required')


def _meta(raw, api):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
            'git_blob_sha': api._git_sha(raw)}


def _regular_destination(path, api):
    path = Path(path).absolute()
    _check(api, not any(p.is_symlink() for p in (path, *path.parents)),
           'continuation_symlink_refused')
    return path


def _atomic(path, raw, api):
    path = _regular_destination(path, api)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.continuation-', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _state_path(output, statefile, api):
    output = _regular_destination(output, api)
    return _regular_destination(statefile or output.parent/(output.name+'.continuation_state.json'), api)


def _read_state(path, api):
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_bytes())
    except (ValueError, UnicodeError, OSError):
        _fail(api, 'continuation_state_invalid')
    _check(api, isinstance(value, dict) and value.get('kind') == 'e3_audit_continuation_state',
           'continuation_state_invalid')
    return value


def _new_state(source, protocol, master):
    return {'kind': 'e3_audit_continuation_state', 'schema_version': 1,
            'repository': REPOSITORY, 'source_commit': source,
            'protocol_sha256': protocol, 'master_run_id': master,
            'head_commit': None, 'head_part': None, 'head_receipt_sha256': None,
            'sequence': 0, 'files': {}, 'counted_jobs': [], 'pending': None}


def _compatible(state, source, protocol, master, api):
    _check(api, state.get('repository') == REPOSITORY
           and state.get('source_commit') == source
           and state.get('protocol_sha256') == protocol
           and state.get('master_run_id') == master,
           'continuation_state_identity_mismatch')


def _safe_file(name, api):
    return (isinstance(name, str) and len(name) <= 512 and api._safe_relative(name)
            and Path(name).suffix in STORE.ALLOWED)


def _scan(output, token, api, statepath):
    output = api._regular_path(output, directory=True)
    result = {}
    for path in sorted(output.rglob('*')):
        _check(api, not path.is_symlink(), 'continuation_symlink_refused')
        if not path.is_file() or path == statepath:
            continue
        name = path.relative_to(output).as_posix()
        # Atomic scratch files are never scientific output or checkpoints.
        if path.name.endswith('.tmp') or path.name.startswith('.partial-'):
            continue
        _check(api, _safe_file(name, api) and token not in name,
               'continuation_unsafe_output_path')
        raw = path.read_bytes()
        _check(api, len(raw) <= MAX_FILE and token.encode() not in raw,
               'continuation_output_size_or_secret')
        STORE._inspect_archive(raw, token, api)
        result[name] = _meta(raw, api)
    return result


def _attempted_jobs(files):
    jobs = set()
    for name in files:
        path = PurePosixPath(name)
        if path.name not in {'phase_started.json', 'arima_fit_started.json'}:
            continue
        parent = path.parent
        candidates = [p for p in (parent, *parent.parents) if str(p) != '.']
        fit = next((p for p in candidates if str(p/'fit_request.json') in files), None)
        jobs.add(str(fit or (parent if path.name == 'arima_fit_started.json' else parent.parent)))
    return sorted(jobs)


def _stamp(previous=None):
    now = datetime.now(timezone.utc)
    if previous:
        old = datetime.strptime(previous, '%Y%m%dT%H%M%S%fZ').replace(tzinfo=timezone.utc)
        now = max(now, old+timedelta(microseconds=1))
    return now.strftime('%Y%m%dT%H%M%S%fZ')


def _commit(api, token, sha):
    value = api._request('GET', '/repos/'+REPOSITORY+'/git/commits/'+sha, token)
    _check(api, value.get('sha') == sha and re.fullmatch(r'[0-9a-f]{40}', value.get('tree', {}).get('sha', '')),
           'continuation_commit_identity_mismatch')
    return value


def _tree(api, token, sha, recursive=False):
    cache=getattr(api,'_e3_continuation_tree_cache',None)
    if cache is None:
        cache={};setattr(api,'_e3_continuation_tree_cache',cache)
    key=(sha,recursive)
    if key in cache:
        return cache[key]
    result = api._request('GET', '/repos/'+REPOSITORY+'/git/trees/'+sha+('?recursive=1' if recursive else ''), token)
    _check(api, result.get('sha') == sha and result.get('truncated') is False
           and isinstance(result.get('tree'), list), 'continuation_complete_tree_required')
    rows = result['tree']
    _check(api, len({r.get('path') for r in rows}) == len(rows), 'continuation_duplicate_tree_paths')
    value={row['path']: row for row in rows}
    cache[key]=value
    return value


def _subtree(api, token, root_sha, prefix):
    sha = root_sha
    for component in PurePosixPath(prefix).parts:
        row = _tree(api, token, sha).get(component)
        _check(api, row is not None and row.get('type') == 'tree' and row.get('mode') == '040000',
               'continuation_missing_part_tree')
        sha = row['sha']
    return sha


def _preserved_except(api, token, before_sha, after_sha, components):
    """Compare sibling subtree hashes, descending only the new part's path."""
    before = _tree(api, token, before_sha) if before_sha else {}
    after = _tree(api, token, after_sha)
    component = components[0]
    _check(api, {k:v for k,v in before.items() if k != component}
           == {k:v for k,v in after.items() if k != component},
           'continuation_source_tree_preservation_failed')
    old, new = before.get(component), after.get(component)
    _check(api, new is not None and new.get('type') == 'tree' and new.get('mode') == '040000',
           'continuation_part_tree_required')
    if len(components) == 1:
        _check(api, old is None, 'continuation_part_already_exists_in_parent')
        return new['sha']
    _check(api, old is None or (old.get('type') == 'tree' and old.get('mode') == '040000'),
           'continuation_parent_tree_invalid')
    return _preserved_except(api, token, old['sha'] if old else None, new['sha'], components[1:])


def _part_rows(api, token, tree_sha, prefix):
    leaf = _subtree(api, token, tree_sha, prefix)
    rows = _tree(api, token, leaf, recursive=True)
    for path, row in rows.items():
        _check(api, api._safe_relative(path), 'continuation_unsafe_remote_path')
        _check(api, (row.get('type'),row.get('mode')) in {('blob','100644'),('tree','040000')},
               'continuation_remote_symlink_or_mode')
    return {p:r for p,r in rows.items() if r.get('type') == 'blob'}


def _contents(api, token, commit, path):
    result = api._request('GET', '/repos/'+REPOSITORY+'/contents/'+quote(path,safe='/')+'?ref='+commit, token)
    _check(api, isinstance(result, dict) and result.get('type') == 'file' and result.get('path') == path
           and result.get('encoding') == 'base64', 'continuation_contents_file_required')
    try:
        raw = base64.b64decode(result['content'], validate=False)
    except (ValueError, TypeError, KeyError):
        _fail(api, 'continuation_contents_invalid_base64')
    _check(api, len(raw) == result.get('size') and api._git_sha(raw) == result.get('sha'),
           'continuation_contents_integrity_mismatch')
    return raw


def _json(raw, api):
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError):
        _fail(api, 'continuation_invalid_remote_json')
    _check(api, isinstance(value, dict), 'continuation_invalid_remote_json')
    return value


def _read_part(api, token, sha, part=None):
    commit = _commit(api, token, sha)
    _check(api, part is not None and re.fullmatch(STAMP, part), 'continuation_part_id_required')
    prefix = PREFIX+part
    receipt_raw = _contents(api, token, sha, prefix+'/stage_receipt.json')
    _check(api,token.encode() not in receipt_raw,'continuation_remote_metadata_secret')
    receipt = _json(receipt_raw, api)
    _check(api, receipt.get('kind') == 'e3_closure_stage_receipt' and receipt.get('profile') == 'audit'
           and receipt.get('repository') == REPOSITORY
           and receipt.get('part_id') == part and receipt.get('R1_accredited') is False
           and receipt.get('R3_accredited') is False, 'continuation_part_scope_mismatch')
    _identity(api, receipt.get('source_commit'), receipt.get('protocol_sha256'), receipt.get('master_run_id'))
    _check(api, type(receipt.get('sequence')) is int and receipt['sequence'] >= 1
           and isinstance(receipt.get('files'), dict)
           and type(receipt.get('new_real_data_fits')) is int and receipt['new_real_data_fits'] >= 0,
           'continuation_part_manifest_invalid')
    for name, meta in receipt['files'].items():
        _check(api, _safe_file(name, api) and token not in name and isinstance(meta,dict)
               and type(meta.get('bytes')) is int and 0 <= meta['bytes'] <= MAX_FILE
               and re.fullmatch(r'[0-9a-f]{64}', meta.get('sha256','')) is not None
               and re.fullmatch(r'[0-9a-f]{40}', meta.get('git_blob_sha','')) is not None,
               'continuation_part_file_manifest_invalid')
    new_jobs=receipt.get('new_attempted_jobs')
    _check(api,len(receipt['files']) <= MAX_FILES
           and sum(meta['bytes'] for meta in receipt['files'].values()) <= MAX_PAYLOAD
           and isinstance(new_jobs,list)
           and all(isinstance(job,str) and api._safe_relative(job) and token not in job for job in new_jobs)
           and len(set(new_jobs)) == len(new_jobs)
           and receipt['new_real_data_fits'] == len(new_jobs),
           'continuation_part_counter_or_budget_invalid')
    status=receipt.get('controller_status',{})
    _check(api,isinstance(status,dict) and (not status or (
        status.get('stage_status') in {'running','completed','failed','interrupted'}
        and type(status.get('analysis_complete')) is bool)),
        'continuation_controller_status_invalid')
    previous = receipt.get('previous_part')
    if previous is None:
        _check(api, receipt['sequence'] == 1, 'continuation_root_sequence_invalid')
        parent = receipt['source_commit']
    else:
        _check(api, isinstance(previous,dict)
               and re.fullmatch(r'[0-9a-f]{40}',previous.get('commit_sha','')) is not None
               and re.fullmatch(STAMP,previous.get('part_id','')) is not None
               and re.fullmatch(r'[0-9a-f]{64}',previous.get('receipt_sha256','')) is not None,
               'continuation_previous_part_invalid')
        parent = previous['commit_sha']
    _check(api, [p.get('sha') for p in commit.get('parents',[])] == [parent],
           'continuation_commit_parent_mismatch')
    return {'commit_sha':sha,'part_id':part,'prefix':prefix,'receipt':receipt,
            'receipt_raw':receipt_raw,'receipt_sha256':hashlib.sha256(receipt_raw).hexdigest(),
            'tree_sha':commit['tree']['sha']}


def _chain(api, token, head, cache):
    result, seen = [], set()
    current = head
    while current:
        _check(api, current['commit_sha'] not in seen, 'continuation_chain_cycle')
        seen.add(current['commit_sha']);result.append(current)
        previous = current['receipt'].get('previous_part')
        if previous is None:
            break
        older = cache.get(previous['commit_sha']) or _read_part(api,token,previous['commit_sha'],previous['part_id'])
        cache[older['commit_sha']] = older
        a,b = current['receipt'],older['receipt']
        _check(api, older['receipt_sha256'] == previous['receipt_sha256']
               and older['part_id'] == previous['part_id']
               and a['sequence'] == b['sequence']+1
               and older['part_id'] < current['part_id']
               and all(a[k] == b[k] for k in ('source_commit','protocol_sha256','master_run_id')),
               'continuation_chain_binding_mismatch')
        current = older
    return list(reversed(result))


def _discover_parts(token, api, source=None):
    token = api._token(token)
    _check(api, api.REPOSITORY == REPOSITORY, 'continuation_repository_mismatch')
    refs = api._request('GET','/repos/'+REPOSITORY+'/git/matching-refs/heads/'+BRANCH,token)
    _check(api,isinstance(refs,list),'continuation_refs_invalid')
    parts = {}
    for ref in refs:
        match = re.fullmatch(r'refs/heads/'+BRANCH+'('+STAMP+')',ref.get('ref',''))
        _check(api,match is not None and ref.get('object',{}).get('type') == 'commit'
               and re.fullmatch(r'[0-9a-f]{40}',ref.get('object',{}).get('sha','')) is not None,
               'continuation_unexpected_results_ref')
        part = _read_part(api,token,ref['object']['sha'],match.group(1))
        if source is None or part['receipt']['source_commit'] == source:
            parts[part['commit_sha']] = part
    groups = {}
    for part in parts.values():
        r = part['receipt'];key=(r['master_run_id'],r['source_commit'],r['protocol_sha256'])
        groups.setdefault(key,[]).append(part)
    _check(api,len({key[0] for key in groups}) == len(groups),
           'continuation_master_has_multiple_identities')
    result = []
    for (master,source_sha,protocol),values in groups.items():
        predecessors = {p['receipt']['previous_part']['commit_sha'] for p in values if p['receipt'].get('previous_part')}
        heads = [p for p in values if p['commit_sha'] not in predecessors]
        _check(api,len(heads) == 1,'continuation_ambiguous_chain_heads')
        chain = _chain(api,token,heads[0],parts)
        result.append({'master_run_id':master,'source_commit':source_sha,'protocol_sha256':protocol,
                       'head_commit':heads[0]['commit_sha'],'head_part':heads[0]['part_id'],
                       'sequence':heads[0]['receipt']['sequence'],'parts':len(chain),
                       'stage_status':heads[0]['receipt'].get('controller_status',{}).get('stage_status','unknown'),
                       'analysis_complete':heads[0]['receipt'].get('controller_status',{}).get('analysis_complete') is True,
                       'run_url':'https://github.com/'+REPOSITORY+'/tree/'+BRANCH+heads[0]['part_id']+'/'+heads[0]['prefix'],
                       '_chain':chain})
    return sorted(result,key=lambda r:r['head_part'],reverse=True)


def discover_runs(token, api, source=None):
    """List logical runs; ambiguous forks fail instead of silently choosing one."""
    return [{k:v for k,v in row.items() if k != '_chain'}
            for row in _discover_parts(token,api,source)]


def _validate_part_files(part, api, token):
    previous=part['receipt'].get('previous_part')
    parent=_commit(api,token,previous['commit_sha'] if previous else part['receipt']['source_commit'])
    _preserved_except(api,token,parent['tree']['sha'],part['tree_sha'],
                     list(PurePosixPath(part['prefix']).parts))
    rows = _part_rows(api,token,part['tree_sha'],part['prefix'])
    manifest_raw = _contents(api,token,part['commit_sha'],part['prefix']+'/storage_manifest.json')
    _check(api,token.encode() not in manifest_raw,'continuation_remote_metadata_secret')
    manifest = _json(manifest_raw,api)
    r=part['receipt']
    _check(api,manifest.get('kind') == 'e3_closure_storage_manifest'
           and manifest.get('source_commit') == r['source_commit']
           and manifest.get('run_id') == part['part_id'] and manifest.get('profile') == 'audit',
           'continuation_storage_manifest_scope_mismatch')
    expected={'payload/'+name:meta for name,meta in r['files'].items()}
    expected['stage_receipt.json']=_meta(part['receipt_raw'],api)
    _check(api,manifest.get('files') == expected,'continuation_storage_manifest_binding_mismatch')
    expected['storage_manifest.json']=_meta(manifest_raw,api)
    _check(api,set(rows) == set(expected)
           and all(rows[name]['sha'] == meta['git_blob_sha'] for name,meta in expected.items()),
           'continuation_remote_part_files_mismatch')
    return rows


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,request,file,code,message,headers,newurl):
        return None


def _blob(api, token, sha, size):
    if hasattr(api,'raw_blob'):
        raw=api.raw_blob(sha,size,token)
    else:
        request=Request('https://api.github.com/repos/'+REPOSITORY+'/git/blobs/'+sha,
                        headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github.raw+json',
                                 'X-GitHub-Api-Version':'2022-11-28','User-Agent':'E3-Audit-Continuation'})
        try:
            with build_opener(_NoRedirect()).open(request,timeout=90) as response:
                raw=response.read(size+1)
        except Exception:
            _fail(api,'continuation_blob_download_failed_retry')
    _check(api,len(raw) == size and api._git_sha(raw) == sha,'continuation_blob_integrity_mismatch')
    return raw


def _state_from_chain(chain):
    first=chain[0]['receipt'];state=_new_state(first['source_commit'],first['protocol_sha256'],first['master_run_id'])
    for part in chain:
        r=part['receipt'];state['files'].update(r['files'])
        state['counted_jobs']=sorted(set(state['counted_jobs'])|set(r.get('new_attempted_jobs',[])))
        state.update(head_commit=part['commit_sha'],head_part=part['part_id'],
                     head_receipt_sha256=part['receipt_sha256'],sequence=r['sequence'])
    return state


def _parse_url(url,api):
    parts=urlsplit(url)
    _check(api,parts.scheme == 'https' and parts.netloc == 'github.com'
           and not parts.query and not parts.fragment,'continuation_results_url_invalid')
    pattern='/'+REPOSITORY+'/tree/('+BRANCH+'('+STAMP+r')|[0-9a-f]{40})/'+PREFIX+'('+STAMP+')'
    match=re.fullmatch(pattern,parts.path)
    _check(api,match is not None,'continuation_results_url_invalid')
    _check(api,match.group(2) is None or match.group(2) == match.group(3),
           'continuation_results_url_invalid')
    return match.group(3), (match.group(1) if match.group(2) is None else None)


def restore(output, token, source, protocol_sha256, api, *, master_run_id=None,
            run_url=None, statefile=None):
    """Verify complete immutable chain, stage bytes, then restore without clobber."""
    token=api._token(token);_identity(api,source,protocol_sha256,master_run_id)
    output=_regular_destination(output,api);statepath=_state_path(output,statefile,api)
    rows=_discover_parts(token,api,source)
    part_id,url_commit=_parse_url(run_url,api) if run_url else (None,None)
    selected=[r for r in rows if r['protocol_sha256'] == protocol_sha256
              and (master_run_id is None or r['master_run_id'] == master_run_id)
              and (part_id is None or any(p['part_id'] == part_id
                   and (url_commit is None or p['commit_sha'] == url_commit) for p in r['_chain']))]
    _check(api,len(selected) == 1,'continuation_run_selection_missing_or_ambiguous')
    row=selected[0];chain=row['_chain'];state=_state_from_chain(chain)
    local=_read_state(statepath,api)
    if local:
        _compatible(local,source,protocol_sha256,row['master_run_id'],api)
    locations={}
    for part in chain:
        _validate_part_files(part,api,token)
        for name,meta in part['receipt']['files'].items():locations[name]=(part,meta)
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='e3-audit-restore-',dir=output.parent) as temp:
        staged=Path(temp)
        for name,(part,meta) in locations.items():
            target=_regular_destination(output/name,api)
            _check(api,not target.exists() or target.is_file(),'continuation_local_file_type_conflict')
            if target.is_file():
                local_raw=target.read_bytes();actual=_meta(local_raw,api)
                if actual == meta:
                    _check(api,token.encode() not in local_raw,'continuation_file_hash_or_secret_mismatch')
                    STORE._inspect_archive(local_raw,token,api)
                    continue
                _check(api,local is not None and local['files'].get(name) == actual,
                       'continuation_local_edits_conflict')
            raw=_blob(api,token,meta['git_blob_sha'],meta['bytes'])
            _check(api,hashlib.sha256(raw).hexdigest() == meta['sha256'] and token.encode() not in raw,
                   'continuation_file_hash_or_secret_mismatch')
            STORE._inspect_archive(raw,token,api)
            path=staged/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        # All hashes and all local conflicts have been checked before writes.
        for path in sorted(staged.rglob('*')):
            if path.is_file():_atomic(output/path.relative_to(staged),path.read_bytes(),api)
    _atomic(statepath,api._canonical(state),api)
    return {k:v for k,v in row.items() if k != '_chain'}|{
        'restored_files':len(locations),'output':str(output),'statefile':str(statepath),
        'all_files_verified':True,'main_updated':False}


def _upload_part(snapshot, state, api, token):
    r=_json((snapshot/'stage_receipt.json').read_bytes(),api);part=r['part_id']
    files=STORE.snapshot(snapshot,token,state['source_commit'],part,api,'audit')
    _check(api,sum(map(len,files.values())) <= MAX_PART,'continuation_part_budget_exceeded')
    parent=state['head_commit'] or state['source_commit'];base=_commit(api,token,parent)
    prefix=PREFIX+part;ref_path='/repos/'+REPOSITORY+'/git/ref/heads/'+BRANCH+part
    ref=api._request('GET',ref_path,token,allow_missing=True)
    expected={name:api._git_sha(raw) for name,raw in files.items()}
    if ref is None:
        entries=[]
        for name,raw in sorted(files.items()):
            blob=api._request('POST','/repos/'+REPOSITORY+'/git/blobs',token,
                              {'encoding':'base64','content':base64.b64encode(raw).decode()})
            _check(api,blob.get('sha') == expected[name],'continuation_uploaded_blob_mismatch')
            entries.append({'path':prefix+'/'+name,'mode':'100644','type':'blob','sha':blob['sha']})
        tree=api._request('POST','/repos/'+REPOSITORY+'/git/trees',token,
                          {'base_tree':base['tree']['sha'],'tree':entries})
        leaf=_preserved_except(api,token,base['tree']['sha'],tree['sha'],list(PurePosixPath(prefix).parts))
        rows=_tree(api,token,leaf,recursive=True)
        blobs={p:v for p,v in rows.items() if v.get('type') == 'blob'}
        _check(api,set(blobs) == set(expected) and all(blobs[p].get('sha') == sha
               and blobs[p].get('mode') == '100644' for p,sha in expected.items()),
               'continuation_uploaded_part_tree_mismatch')
        commit=api._request('POST','/repos/'+REPOSITORY+'/git/commits',token,
                            {'message':'Guardar delta auditoría E3 '+part,'tree':tree['sha'],'parents':[parent]})
        try:
            api._request('POST','/repos/'+REPOSITORY+'/git/refs',token,
                         {'ref':'refs/heads/'+BRANCH+part,'sha':commit['sha']})
        except api.StorageError:
            recovered=api._request('GET',ref_path,token,allow_missing=True)
            if recovered is None:raise
        ref=api._request('GET',ref_path,token)
    _check(api,ref.get('ref') == 'refs/heads/'+BRANCH+part and ref.get('object',{}).get('type') == 'commit',
           'continuation_saved_ref_invalid')
    saved=_read_part(api,token,ref['object']['sha'],part)
    _check(api,saved['receipt_raw'] == files['stage_receipt.json'],'continuation_existing_part_conflict')
    _validate_part_files(saved,api,token)
    saved['run_url']='https://github.com/'+REPOSITORY+'/tree/'+BRANCH+part+'/'+prefix
    return saved


def _finish_pending(state,statepath,api,token):
    pending=state['pending'];snapshot=api._regular_path(pending['snapshot'],directory=True)
    part=_upload_part(snapshot,state,api,token);r=part['receipt']
    state['files'].update(r['files'])
    state['counted_jobs']=sorted(set(state['counted_jobs'])|set(r['new_attempted_jobs']))
    state.update(head_commit=part['commit_sha'],head_part=part['part_id'],
                 head_receipt_sha256=part['receipt_sha256'],sequence=r['sequence'],pending=None)
    _atomic(statepath,api._canonical(state),api)
    return {k:part[k] for k in ('commit_sha','part_id','run_url')}|{
        'sequence':r['sequence'],'changed_files':len(r['files']),
        'new_real_data_fits':r['new_real_data_fits']}


def save_changed(output, token, source, master_run_id, protocol_sha256, api,
                 statefile=None):
    """Save immutable deltas; an acknowledged part is never uploaded twice."""
    token=api._token(token);_identity(api,source,protocol_sha256,master_run_id)
    output=api._regular_path(output,directory=True);statepath=_state_path(output,statefile,api)
    state=_read_state(statepath,api)
    if state is None:
        existing=[r for r in _discover_parts(token,api)
                  if r['master_run_id'] == master_run_id]
        _check(api,len(existing) <= 1,'continuation_ambiguous_existing_run')
        if existing:
            _check(api,existing[0]['source_commit'] == source and existing[0]['protocol_sha256'] == protocol_sha256,
                   'continuation_state_identity_mismatch')
            for part in existing[0]['_chain']:_validate_part_files(part,api,token)
        state=_state_from_chain(existing[0]['_chain']) if existing else _new_state(source,protocol_sha256,master_run_id)
        _atomic(statepath,api._canonical(state),api)
    _compatible(state,source,protocol_sha256,master_run_id,api)
    api.validate_credentials(token)
    saved=[]
    if state.get('pending'):
        saved.append(_finish_pending(state,statepath,api,token))
    current=_scan(output,token,api,statepath)
    _check(api,set(state['files']).issubset(current),'continuation_previously_saved_file_removed')
    changed=[name for name,meta in current.items() if state['files'].get(name) != meta]
    groups=[];group=[];total=0
    for name in changed:
        size=current[name]['bytes']
        if group and (len(group) >= MAX_FILES or total+size > MAX_PAYLOAD):
            groups.append(group);group=[];total=0
        group.append(name);total+=size
    if group:groups.append(group)
    for group in groups:
        part_id=_stamp(state['head_part']);snapshot=Path(tempfile.mkdtemp(
            prefix='e3-audit-part-'+part_id+'-',dir=output.parent))
        metadata={name:current[name] for name in group}
        for name in group:
            raw=(output/name).read_bytes()
            _check(api,_meta(raw,api) == current[name],'continuation_output_changed_during_snapshot')
            target=snapshot/'payload'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        attempted=set(_attempted_jobs(current))-set(state['counted_jobs'])
        included=set(_attempted_jobs({name:current[name] for name in group}))
        new_jobs=sorted(attempted & included)
        previous=None if state['head_commit'] is None else {
            'commit_sha':state['head_commit'],'part_id':state['head_part'],
            'receipt_sha256':state['head_receipt_sha256']}
        receipt={'kind':'e3_closure_stage_receipt','profile':'audit','schema_version':1,
                 'repository':REPOSITORY,'source_commit':source,'protocol_sha256':protocol_sha256,
                 'master_run_id':master_run_id,'part_id':part_id,'sequence':state['sequence']+1,
                 'previous_part':previous,'files':metadata,'new_attempted_jobs':new_jobs,
                 'new_real_data_fits':len(new_jobs),'counter_scope':'New attempted model jobs first recorded by durable start markers in this delta; not epochs or completed fits.',
                 'R1_accredited':False,'R3_accredited':False,'main_updated':False}
        status_path=output/'controller_status.json'
        if status_path.is_file():
            status=_json(status_path.read_bytes(),api)
            _check(api,status.get('stage_status') in {'running','completed','failed','interrupted'}
                   and type(status.get('analysis_complete')) is bool,
                   'continuation_controller_status_invalid')
            receipt['controller_status']={k:status[k] for k in ('stage_status','analysis_complete')}
        (snapshot/'stage_receipt.json').write_bytes(api._canonical(receipt))
        # Persist the pending immutable bytes before any remote object creation.
        state['pending']={'part_id':part_id,'snapshot':str(snapshot)}
        _atomic(statepath,api._canonical(state),api)
        saved.append(_finish_pending(state,statepath,api,token))
    return {'master_run_id':master_run_id,'source_commit':source,'protocol_sha256':protocol_sha256,
            'head_commit':state['head_commit'],'head_part':state['head_part'],'saved_parts':saved,
            'changed_files':sum(p['changed_files'] for p in saved),'statefile':str(statepath),
            'all_files_verified':True,'main_updated':False}
