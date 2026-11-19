"""Exact, retryable snapshots in a new results branch; never mutate main."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import base64
import hashlib
import io
import json
import re
import stat
import zipfile

PROFILES = {'cpu': ('e3-closure-cpu-', 'runs/e3_closure_cpu/'),
            'pilot': ('e3-closure-pilot-', 'runs/e3_closure_pilot/')}
ALLOWED = {'.json','.csv','.txt','.md','.log','.zip','.CHECKSUM','.png','.xml','.pt','.npz'}
MAX_FILE = 48 * 1024 * 1024
MAX_TOTAL = 256 * 1024 * 1024


def _inspect_archive(raw, token, api):
    """Inspect ZIP/NPZ/Torch members as bytes, without unpickling any content."""
    stream = io.BytesIO(raw)
    if not zipfile.is_zipfile(stream):
        return
    try:
        with zipfile.ZipFile(stream) as archive:
            members = archive.infolist()
            if len(members) > 2048 or sum(m.file_size for m in members) > MAX_TOTAL:
                raise api.StorageError('archive_expansion_budget_exceeded')
            seen = set()
            for member in members:
                name = member.filename.removesuffix('/') if member.is_dir() else member.filename
                if (not api._safe_relative(name) or name in seen or token in name
                        or stat.S_ISLNK(member.external_attr >> 16)
                        or member.file_size > MAX_FILE or member.flag_bits & 1):
                    raise api.StorageError('unsafe_archive_member')
                seen.add(name)
                if member.is_dir():
                    continue
                payload = archive.read(member)
                if token.encode() in payload:
                    raise api.StorageError('archive_secret_refused')
                # Nested archives are outside these stage contracts. Refuse
                # them rather than making a secret scan arbitrarily recursive.
                if zipfile.is_zipfile(io.BytesIO(payload)):
                    raise api.StorageError('nested_archive_refused')
    except api.StorageError:
        raise
    except (zipfile.BadZipFile, RuntimeError, OSError, NotImplementedError):
        raise api.StorageError('invalid_evidence_archive') from None


def snapshot(output, token, source, run_id, api, profile):
    if profile not in PROFILES or re.fullmatch(r'[0-9a-f]{40}', source) is None or re.fullmatch(r'[0-9]{8}T[0-9]{12}Z',run_id) is None:
        raise api.StorageError('invalid_profile_source_or_run')
    output = api._regular_path(Path(output),directory=True)
    files = {}
    for p in sorted(output.rglob('*')):
        if p.is_symlink():
            raise api.StorageError('snapshot_symlink_refused')
        if not p.is_file():
            continue
        name=p.relative_to(output).as_posix()
        if not api._safe_relative(name) or token in name or p.suffix not in ALLOWED or name=='storage_manifest.json':
            raise api.StorageError('unexpected_evidence_path')
        raw=p.read_bytes()
        if len(raw)>MAX_FILE or token.encode() in raw:
            raise api.StorageError('evidence_size_or_secret_refused')
        _inspect_archive(raw, token, api)
        files[name]=raw
    if len(files)>1024 or sum(map(len,files.values()))>MAX_TOTAL:
        raise api.StorageError('snapshot_budget_exceeded')
    try:
        receipt=json.loads(files['stage_receipt.json'])
    except (KeyError,ValueError,UnicodeError):
        raise api.StorageError('invalid_or_missing_stage_receipt') from None
    if not isinstance(receipt,dict):
        raise api.StorageError('invalid_or_missing_stage_receipt')
    if receipt.get('kind')!='e3_closure_stage_receipt' or receipt.get('source_commit')!=source or receipt.get('profile')!=profile:
        raise api.StorageError('stage_scope_mismatch')
    if receipt.get('R1_accredited') is not False or receipt.get('R3_accredited') is not False:
        raise api.StorageError('stage_cannot_certify_originality_or_superiority')
    fits=receipt.get('new_real_data_fits')
    if type(fits) is not int or fits<0 or (profile=='cpu' and fits!=0):
        raise api.StorageError('fit_counter_scope_mismatch')
    manifest={'kind':'e3_closure_storage_manifest','source_commit':source,'run_id':run_id,'profile':profile,
              'files':{n:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'git_blob_sha':api._git_sha(b)} for n,b in files.items()}}
    files['storage_manifest.json']=api._canonical(manifest)
    return files


def save(output, token, source, run_id, api, *, profile='cpu', workers=4):
    token=api._token(token)
    files=snapshot(output,token,source,run_id,api,profile)
    api.validate_credentials(token)
    root='/repos/'+api.REPOSITORY
    branch=PROFILES[profile][0]+run_id
    prefix=PROFILES[profile][1]+run_id
    base=api._request('GET',root+'/git/commits/'+source,token)
    if base.get('sha')!=source:
        raise api.StorageError('source_commit_identity_mismatch')
    base_tree=base['tree']['sha']
    old=api._tree_map(api._request('GET',root+'/git/trees/'+base_tree+'?recursive=1',token))
    if any(p.startswith(prefix+'/') for p in old):
        raise api.StorageConflict('run_already_present_in_source')
    expected={prefix+'/'+n:api._git_sha(b) for n,b in files.items()}
    ref_path=root+'/git/ref/heads/'+branch

    def verify(ref):
        if ref.get('ref')!='refs/heads/'+branch or ref.get('object',{}).get('type')!='commit':
            raise api.StorageConflict('unexpected_results_ref')
        sha=ref['object']['sha']
        commit=api._request('GET',root+'/git/commits/'+sha,token)
        if commit.get('sha')!=sha or [p.get('sha') for p in commit.get('parents',[])]!=[source]:
            raise api.StorageConflict('results_have_different_source')
        api._verify_tree(api._request('GET',root+'/git/trees/'+commit['tree']['sha']+'?recursive=1',token),old,expected,prefix)
        return sha

    existing=api._request('GET',ref_path,token,allow_missing=True)
    if existing is None:
        def upload(item):
            name,raw=item
            result=api._request('POST',root+'/git/blobs',token,
                                {'encoding':'base64','content':base64.b64encode(raw).decode()})
            if result.get('sha')!=api._git_sha(raw):
                raise api.StorageError('uploaded_blob_integrity_mismatch')
            return {'path':prefix+'/'+name,'mode':'100644','type':'blob','sha':result['sha']}
        # Content-addressed blobs are independent; refs are created only after all checks.
        with ThreadPoolExecutor(max_workers=max(1,min(4,workers))) as pool:
            entries=list(pool.map(upload,sorted(files.items())))
        tree=api._request('POST',root+'/git/trees',token,{'base_tree':base_tree,'tree':entries})
        api._verify_tree(api._request('GET',root+'/git/trees/'+tree['sha']+'?recursive=1',token),old,expected,prefix)
        commit=api._request('POST',root+'/git/commits',token,
                            {'message':'Guardar etapa E3 '+profile+' '+run_id,'tree':tree['sha'],'parents':[source]})
        verify({'ref':'refs/heads/'+branch,'object':{'type':'commit','sha':commit['sha']}})
        try:
            api._request('POST',root+'/git/refs',token,{'ref':'refs/heads/'+branch,'sha':commit['sha']})
        except api.StorageError:
            recovered=api._request('GET',ref_path,token,allow_missing=True)
            if recovered is None:
                raise
            verify(recovered)
    sha=verify(api._request('GET',ref_path,token))
    return {'saved':True,'commit_sha':sha,'main_updated':False,'persisted_files':len(files),
            'run_url':'https://github.com/'+api.REPOSITORY+'/tree/'+branch+'/'+prefix,
            'all_files_verified':True}
