"""Restore selected authoritative metadata without replaying all older deltas.

This module only reads GitHub and restores byte-verified local files. It never
publishes a control pointer, starts a controller or changes budget accounting.
"""
from __future__ import annotations
import base64, hashlib, io, json, os, re, stat, tempfile, time, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

VERSION='e2-fresh-selected-metadata-restore-v1'
ENGINE_COMMIT='1e8745497b484518f766b6da5c19be8362d9add7'
STATE_VERSIONS={'e2-github-primary-state-v1','e2-guide-closure-posthoc-state-v1'}
MAX_METADATA_BYTES=256*1024**2
HEX64=re.compile('[0-9a-f]{64}\\Z')
HEX40=re.compile('[0-9a-f]{40}\\Z')
TENSORS={'.pt','.pth','.ckpt','.pkl','.npz','.safetensors','.bin'}

def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def blob_sha(raw):return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()

def relative(value):
    if not isinstance(value,str) or not value or '\\' in value:raise ValueError('Invalid metadata relative path')
    p=PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or str(p)!=value or value=='.':raise ValueError('Unsafe metadata relative path')
    return p

def regular(value):
    path=Path(value).absolute()
    if any(p.is_symlink() for p in (path,*path.parents)):raise ValueError('Symlink metadata path prohibited')
    return path

def target(root,value):return regular(root.joinpath(*relative(value).parts))

def descriptor(value,*,git=False):
    if not isinstance(value,dict) or not isinstance(value.get('sha256'),str) or not HEX64.fullmatch(value['sha256']):
        raise ValueError('Invalid metadata SHA256 descriptor')
    if type(value.get('bytes')) is not int or not 0<=value['bytes']<=MAX_METADATA_BYTES:
        raise ValueError('Invalid metadata byte count')
    if git:
        relative(value.get('path'))
        if not isinstance(value.get('git_blob_sha'),str) or not HEX40.fullmatch(value['git_blob_sha']):
            raise ValueError('Invalid metadata Git blob descriptor')

def check_deadline(deadline):
    if deadline is not None and time.monotonic()>=deadline:raise TimeoutError('Metadata restoration deadline reached')

def existing(path,expected):
    path=regular(path)
    if not path.exists():return False
    if not path.is_file() or path.stat().st_size!=expected['bytes'] or sha(path.read_bytes())!=expected['sha256']:
        raise ValueError('Existing local metadata differs; preserved: '+str(path))
    return True

def write_exact(path,payload,expected):
    path=regular(path)
    if len(payload)!=expected['bytes'] or sha(payload)!=expected['sha256']:raise ValueError('Metadata payload identity differs')
    if existing(path,expected):return False
    path.parent.mkdir(parents=True,exist_ok=True);regular(path)
    fd,temp=tempfile.mkstemp(prefix='.e2-fresh-metadata-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(payload);stream.flush();os.fsync(stream.fileno())
        # A link is exclusive: a concurrently-created target is never replaced.
        try:os.link(temp,path)
        except FileExistsError:existing(path,expected);return False
    finally:Path(temp).unlink(missing_ok=True)
    return True

def content_bytes(client,path,head):
    item=client._content(path,head)
    if not isinstance(item,dict) or not isinstance(item.get('sha'),str) or not HEX40.fullmatch(item['sha']):
        raise ValueError('Remote metadata blob is absent or invalid')
    content=item.get('content')
    if not content or item.get('encoding')=='none':
        fetched=client._request('GET','/git/blobs/'+item['sha'])
        if fetched.get('sha',item['sha'])!=item['sha']:raise ValueError('Git blob fallback identity differs')
        content=fetched.get('content')
    if not isinstance(content,str):raise ValueError('Git blob base64 payload is absent')
    try:raw=base64.b64decode(''.join(content.split()),validate=True)
    except (ValueError,TypeError) as exc:raise ValueError('Invalid remote metadata base64') from exc
    if blob_sha(raw)!=item['sha']:raise ValueError('Remote Git blob hash differs')
    return raw

def fetch_delta(client,item,head,cache,deadline):
    check_deadline(deadline);descriptor(item,git=True)
    cache_path=None if cache is None else regular(cache/(item['sha256']+'.zip'))
    if cache_path is not None and existing(cache_path,item):
        raw=cache_path.read_bytes();downloaded=False
    else:
        raw=content_bytes(client,item['path'],head);downloaded=True
    if len(raw)!=item['bytes'] or sha(raw)!=item['sha256'] or blob_sha(raw)!=item['git_blob_sha']:
        raise ValueError('Control delta identity differs')
    if cache_path is not None and downloaded:write_exact(cache_path,raw,item)
    check_deadline(deadline)
    return raw,downloaded

def validate_zip(archive):
    entries=archive.infolist();seen=set();total=0
    if len(entries)>100000:raise ValueError('Metadata archive has too many members')
    for entry in entries:
        name=str(relative(entry.filename))
        if name in seen:raise ValueError('Duplicate control delta member')
        seen.add(name);total+=entry.file_size
        mode=entry.external_attr>>16
        if entry.is_dir() or stat.S_ISLNK(mode) or entry.flag_bits&1:
            raise ValueError('Unsafe control delta member type')
        if stat.S_IFMT(mode) not in (0,stat.S_IFREG):raise ValueError('Nonregular control delta member')
        if entry.file_size<0 or total>MAX_METADATA_BYTES:raise ValueError('Control delta expands beyond metadata limit')
    return entries

def restore_metadata(root,client,state,head,paths=None,cache_root=None,parallel=6,deadline=None):
    """Restore exact current metadata, scanning newest deltas first in batches.

    `paths=None` means all current control files. A requested subset only reads
    older batches while some requested current version remains unavailable.
    The caller owns any resource lease; this routine performs no accounting.
    """
    root=regular(root)
    if (not isinstance(state,dict) or state.get('version') not in STATE_VERSIONS
            or state.get('scientific_commit')!=ENGINE_COMMIT
            or state.get('state_id')!=sha(canonical({k:v for k,v in state.items() if k!='state_id'}))):
        raise ValueError('Authoritative control state identity differs')
    if state.get('output_path')!=str(root):raise ValueError('Keep the original absolute control output path')
    if not isinstance(head,str) or not HEX40.fullmatch(head):raise ValueError('Immutable Git head required')
    if type(parallel)is not int or not 1<=parallel<=12:raise ValueError('Parallel reads must be bounded to1..12')
    inventory=state.get('files');deltas=state.get('metadata_deltas')
    if not isinstance(inventory,dict) or not isinstance(deltas,list):raise ValueError('Control file/delta inventory missing')
    for name,item in inventory.items():
        p=relative(name);descriptor(item)
        if p.suffix.lower() in TENSORS or any(x.startswith('.') or x.lower().startswith('private') for x in p.parts):
            raise ValueError('Nonmetadata path in control inventory')
    for item in deltas:descriptor(item,git=True)
    requested=set(inventory) if paths is None else set(paths)
    for name in requested:relative(name)
    if not requested.issubset(inventory):raise ValueError('Requested metadata absent from authoritative state')
    cache=None if cache_root is None else regular(cache_root)
    pending={name for name in requested if not existing(target(root,name),inventory[name])}
    reused=len(requested)-len(pending);restored=downloaded=cached=examined=0
    newest=list(reversed(deltas));cursor=0
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        while pending and cursor<len(newest):
            check_deadline(deadline);batch=newest[cursor:cursor+parallel];cursor+=len(batch)
            futures=[pool.submit(fetch_delta,client,item,head,cache,deadline) for item in batch]
            # Process newest-to-oldest even when HTTP completes out of order.
            for future in futures:
                raw,was_downloaded=future.result();downloaded+=int(was_downloaded);cached+=int(not was_downloaded);examined+=1
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    entries=validate_zip(archive)
                    for entry in entries:
                        name=entry.filename
                        if name not in pending:continue
                        expected=inventory[name]
                        if entry.file_size!=expected['bytes']:continue
                        payload=archive.read(entry)
                        if sha(payload)!=expected['sha256']:continue
                        restored+=int(write_exact(target(root,name),payload,expected));pending.remove(name)
                check_deadline(deadline)
    if pending:raise ValueError('Current metadata versions were not found in control deltas: '+', '.join(sorted(pending)[:4]))
    return {'version':VERSION,'state_id':state['state_id'],'source_head':head,
            'requested_files':len(requested),'files_restored':restored,'existing_identical_files':reused,
            'deltas_examined':examined,'deltas_downloaded':downloaded,'cached_deltas_reused':cached,
            'complete_for_requested_paths':True,'training_performed':False,'remote_writes':False,'budget_reset':False}
