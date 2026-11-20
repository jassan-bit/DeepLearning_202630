"""Hydrate pinned historical bytes, then build a scoped source/evidence capsule.

Public GET only. Never import model code, unpickle, infer, train, or publish.
The full historical snapshot is ephemeral; only public/ is an artifact output.
Prepared with declared AI assistance. No blanket license is assigned.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import io
import json
import os
import platform
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import tempfile
import threading
import time
import zlib
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

REPO='jassan-bit/DeepLearning_202630'
SOURCE='8d03a73d529eb3f9b2e89992b4488cfeb88a99e6'
SOURCE_TREE='ec193d80793dcf839c4a9d7b74f1f6266c803982'
INDEX_SHA='c2f231d140ab95513be7c0b7462202fe23b624b94de2225f8532986b8d25f272'
ARCHIVE_NAME='e3-source-evidence-8d03a73-v0.3.0-REVIEW.tar.gz'
ZIP_SHA='aa59b73311624f2a92b4497f98bfe182e3c2c320c0e0b092fefee5686bce8349'
PUBLIC_HOSTS={'raw.githubusercontent.com','codeload.github.com','github.com'}
SECRET=re.compile(rb'(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
APPROVED_ONNX={
 '881dd637837265829aa96900d7a0c864dcdab113a57a91485feb20fd2a443abc',
 '10cedceea5da544268ecda882d9ee3d9f4abc153448a599a3a9947b7cf74d255',
 'ee942ca947998a3c5c76fa21eabe401ce60989558faf82608e8107a830227033',
 '1409d0cf84aac43720b4c6978f9a1bb3b5f6c9832518acab0cba279ba8a88bd2',
 '6928a973a6639513bc94cde15996c0b06bb5615e3c0b1a77d16d47e9c7342f3b',
 'c74530282e04ac06e84932f6a65b3fd5a040829b71396d4324252972b84c5d50',
 '8a0f3829388ef20d489581ebbbeba64f9192abf05e674de09b4273b90553afcd',
 'ac110c3cc1c0d6830f03919dce1723e97a0f2ce94bea8aa03161dacd5c1382bf',
 '1cbab4bdf3040a981936220d2899e848f2319a4671dff61f371f25f7d1d8f18c',
 '8c93d7789414f9db51fd0ce25595bbcdb5948c5379e22b2bc00b5e7d518217e4',
 'f36ecb0397bd63cd86062e527e8769ae922ece64fe5660d0eaa57f3e65046861',
 'c8ca0dcdc942fe0ef59c2d79d8ffae688bc37c28e72de789f9dd4f438b5b3428',
 '0ed6c1348d5bc1ea3312aaf00d59f8651cb4949212eeb38c42a732c8f33480d3',
 '80df90b1cb38fbde7f47c8d4ca43f04a5df0c6936a740eda1e5ec7a87c8df1d1',
 '01e2283793b019d90f77489bb8cbfb82845638772532c0f00302e98a708a050b'}

def require(ok,message):
    if not ok:raise ValueError(message)

def now():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def digest(raw):return hashlib.sha256(raw).hexdigest()
def git_digest(raw):return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
def safe(name):
    require(isinstance(name,str) and name and '\\' not in name,'Unsafe path')
    p=PurePosixPath(name)
    require(not p.is_absolute() and all(x not in ('','.','..') for x in p.parts)
            and p.as_posix()==name,'Unsafe or noncanonical path')
    return p

def atomic(path,raw):
    require(not path.is_symlink(),'Symlink destination')
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial')
    require(not tmp.is_symlink(),'Symlink partial file')
    with tmp.open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

class Budget:
    def __init__(self,maximum):self.maximum=maximum;self.used=0;self.lock=threading.Lock()
    def add(self,n):
        with self.lock:
            self.used+=n;require(self.used<=self.maximum,'Network byte ceiling exceeded')

class Redirects(HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        allowed(newurl);return super().redirect_request(req,fp,code,msg,headers,newurl)

def allowed(url):
    p=urlsplit(url)
    require(p.scheme=='https' and p.hostname in PUBLIC_HOSTS and not p.username
            and not p.password and p.port in (None,443),'Unapproved public URL')
    return url

def open_public(url):
    return build_opener(Redirects()).open(Request(allowed(url),headers={'User-Agent':'E3-integral-read-only-archive/0.3.0'}),timeout=60)

def download(row,budget):
    url='https://raw.githubusercontent.com/'+REPO+'/'+row['storage_commit']+'/'+quote(row['storage_path'],safe='/')
    for attempt in range(4):
        try:
            with open_public(url) as response:
                require(response.status==200,'Public response not HTTP200')
                raw=response.read(row['bytes']+1);budget.add(len(raw))
            check(raw,row);return raw
        except HTTPError as e:
            if e.code not in (429,500,502,503,504) or attempt==3:raise
        except (URLError,TimeoutError):
            if attempt==3:raise
        time.sleep(min(30,3*(2**attempt)))

def check(raw,row):
    require(len(raw)==row['bytes'] and digest(raw)==row['sha256'],'Pinned size/SHA256 mismatch')
    require(not row.get('git_blob_sha') or git_digest(raw)==row['git_blob_sha'],'Pinned Git blob mismatch')

def read_index(home):
    manifest_raw=(home/'index_manifest.json').read_bytes();manifest=json.loads(manifest_raw)
    require(digest(manifest_raw)==INDEX_SHA,'Index manifest is not the pinned8d03a73 version')
    require(manifest['repository']==REPO and manifest['audit_part_count']==193,'Wrong index identity')
    require(manifest['audit_terminal_commit']=='16a4504f6fb9208c7d6f89b65017194fa20470da','Wrong terminal master')
    require(manifest['V3_cumulative_snapshot_commit']=='8eaa581bc34b615a8507c9e384a3e4be29084720','Wrong V3 snapshot')
    rows=[];seen=set()
    for chunk in manifest['chunks']:
        raw=(home/safe(chunk['path'])).read_bytes()
        require(len(raw)==chunk['bytes'] and digest(raw)==chunk['sha256'],'Index chunk changed')
        q=[json.loads(line) for line in raw.splitlines()]
        require(len(q)==chunk['entries'],'Index chunk count changed')
        for r in q:
            safe(r['logical_path']);safe(r['storage_path'])
            require(r['group'] in ('audit796','v3pilot30') and re.fullmatch('[0-9a-f]{40}',r['storage_commit'])
                    and re.fullmatch('[0-9a-f]{64}',r['sha256']) and type(r['bytes']) is int and r['bytes']>=0,'Invalid locator')
            k=(r['group'],r['logical_path']);require(k not in seen,'Duplicate evidence identity');seen.add(k)
        rows.extend(q)
    require(len(rows)==8841,'Wrong indexed logical count')
    for group,count,total in [('audit796',8582,4268147881),('v3pilot30',259,2359027)]:
        q=[r for r in rows if r['group']==group]
        require(len(q)==count and sum(r['bytes'] for r in q)==total,'Wrong indexed group totals')
    part=manifest['parts_catalogue'];raw=(home/safe(part['path'])).read_bytes()
    require(len(raw)==part['bytes'] and digest(raw)==part['sha256'],'Part catalogue changed')
    parts=list(csv.DictReader(io.StringIO(raw.decode())))
    require([int(r['sequence']) for r in parts]==list(range(1,194)),'193-part sequence incomplete')
    for p in parts:
        for field,name in [('receipt','stage_receipt.json'),('storage','storage_manifest.json')]:
            rows.append({'group':'audit193_metadata','logical_path':f"parts/{int(p['sequence']):03d}/{name}",
                         'storage_commit':p['commit'],'storage_path':p[field+'_path'],
                         'bytes':int(p[field+'_bytes']),'sha256':p[field+'_sha256'],
                         'git_blob_sha':p[field+'_git_blob_sha']})
    return manifest,rows,digest(manifest_raw)

def category(path):
    s=path.lower();suffix=PurePosixPath(path).suffix.lower()
    if suffix in ('.pt','.pth','.onnx'):return 'weights_checkpoints'
    if '/datasets/' in s or s.startswith(('inputs/','data/')) or suffix in ('.npz','.npy'):return 'data_inputs'
    if suffix in ('.png','.svg'):return 'figures'
    if s.startswith('analysis/') or any(x in s for x in ['summary','scores','metrics','sensitivity']):return 'tables'
    if '/panels/' in '/'+s or 'prediction' in s:return 'predictions'
    if any(x in s for x in ['request','protocol','plan','selection','config']):return 'configuration'
    return 'receipts_logs_metadata'

def evidence_permission(row):
    p=row['logical_path'];c=category(p);ext=PurePosixPath(p).suffix.lower()
    if row['group']=='audit193_metadata':return True,'pinned_public_stage_metadata'
    if c=='data_inputs' or p.startswith('admission/') or p=='dataset_geometry.json':return False,'raw_or_input_related_rights_not_extended'
    if c=='weights_checkpoints':return False,'historical_checkpoint_redistribution_not_cleared_by_selected_document_license'
    if ext in ('.zip','.gz','.tar'):return False,'nested_archive_content_not_cleared'
    if c=='predictions':return False,'full_prediction_payload_retained_in_ephemeral_snapshot_only'
    if ext in ('.json','.csv','.log','.txt','.md','.png','.svg'):return True,'project_generated_evidence_preserved_without_new_blanket_license'
    return False,'unreviewed_file_type'

def source_permission(name,raw):
    p=safe(name);low=name.lower();suffix=p.suffix.lower()
    if SECRET.search(raw):return False,'possible_secret_pattern_requires_review',raw
    if any(x in p.parts for x in ('.git','node_modules','__pycache__','_build','author_sources','published_author_sources')):
        return False,'generated_or_external_dependency_tree',raw
    if suffix=='.onnx':return (digest(raw) in APPROVED_ONNX),'approved_ONNX_hash' if digest(raw) in APPROVED_ONNX else 'ONNX_hash_not_in_approved_set',raw
    if suffix in ('.pt','.pth','.npz','.npy','.parquet','.feather','.xlsx','.xls','.pkl','.pickle'):
        return False,'raw_input_or_checkpoint_rights_not_extended',raw
    if suffix=='.json':
        try:
            obj=json.loads(raw)
            def has_windows(value):
                if isinstance(value,dict):
                    if any(key.lower() in ('windows','raw_windows','raw_ohlcv','ohlcv','raw_window') and isinstance(val,list) and val and isinstance(val[0],(list,int,float)) for key,val in value.items()):return True
                    return any(has_windows(val) for val in value.values())
                if isinstance(value,list):return any(has_windows(val) for val in value if isinstance(val,(list,dict)))
                return False
            if has_windows(obj):return False,'JSON_market_input_arrays_require_separate_rights',raw
        except (ValueError,RecursionError):return False,'JSON_parse_or_depth_requires_review',raw
    if suffix in ('.zip','.gz','.tar','.tgz','.7z','.pdf'):
        return digest(raw)==ZIP_SHA,'exact_owner_approved_documentary_ZIP' if digest(raw)==ZIP_SHA else 'nested_archive_or_full_document_not_cleared',raw
    if any(x in p.parts for x in ('data','datasets','raw','raw_data','inputs','input','inputs_private','windows','cohorts')):
        return False,'dataset_tree_requires_separate_source_rights',raw
    if suffix=='.ipynb':
        try:
            notebook=json.loads(raw)
            for cell in notebook.get('cells',[]):
                cell.pop('outputs',None);cell.pop('execution_count',None);cell.pop('attachments',None)
            return True,'notebook_source_preserved_outputs_removed_for_data_scope',json.dumps(notebook,ensure_ascii=False,sort_keys=True,indent=2).encode()+b'\n'
        except (ValueError,TypeError):return False,'notebook_parse_failed',raw
    if suffix=='.csv':
        try:
            header=next(csv.reader(io.StringIO(raw.decode('utf-8'))),[])
            if {'open','high','low','close'}.issubset({x.strip().lower() for x in header}):return False,'raw_OHLCV_CSV_requires_separate_rights',raw
        except (ValueError,UnicodeDecodeError,csv.Error):return False,'CSV_parse_requires_review',raw
        ok=bool(re.search('metrics|summary|scores|contrast|effect|calibrat|resource|inventory|selection|cost|rank|adequacy|discrep|matrix|power|coverage|sensitivity|table|seed|budget|pareto',p.name.lower()))
        return ok,'generated_aggregate_table' if ok else 'CSV_content_outside_selected_aggregate_scope',raw
    if suffix in ('.py','.md','.rst','.yml','.yaml','.toml','.lock','.txt','.cff','.bib','.ini','.cfg','.html','.css','.js','.json','.sh','.dockerignore','.gitignore') or p.name.startswith(('Dockerfile','LICENSE','NOTICE','requirements')):
        return True,'repository_source_documentation_or_configuration_original_terms_retained',raw
    return False,'binary_or_unreviewed_resource_excluded',raw

class CountReader:
    def __init__(self,source,budget):self.source=source;self.budget=budget
    def read(self,n=-1):
        raw=self.source.read(n);self.budget.add(len(raw));return raw

def write_json(path,value):atomic(path,(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode())

def build_capsule(payload,archive):
    files=[]
    for p in sorted(payload.rglob('*')):
        if p.is_file() and p.name!='CAPSULE_MANIFEST.json':
            require(not p.is_symlink(),'Capsule symlink');raw=p.read_bytes()
            files.append({'path':p.relative_to(payload).as_posix(),'bytes':len(raw),'sha256':digest(raw)})
    manifest={'schema_version':1,'capsule_version':'0.3.0-REVIEW','source_commit':SOURCE,'files':files,
              'scope':'source/evidence capsule with explicit exclusions; not integral scientific master',
              'Zenodo_publication_performed':False,'global_MIT_or_CC_license_assigned':False}
    write_json(payload/'CAPSULE_MANIFEST.json',manifest)
    with archive.open('wb') as out,gzip.GzipFile(fileobj=out,mode='wb',filename='',mtime=0,compresslevel=6) as gz:
        with tarfile.open(fileobj=gz,mode='w|',format=tarfile.PAX_FORMAT) as tar:
            for p in sorted(payload.rglob('*')):
                if not p.is_file():continue
                info=tarfile.TarInfo(p.relative_to(payload).as_posix());info.size=p.stat().st_size
                info.uid=info.gid=0;info.uname=info.gname='';info.mode=0o644;info.mtime=0
                with p.open('rb') as stream:tar.addfile(info,stream)
    return manifest

def verify_extraction(archive,manifest,work):
    expected={r['path']:r for r in manifest['files']};seen=set();extracted=0
    temp=Path(tempfile.mkdtemp(prefix='extract-check-',dir=work))
    try:
        with tarfile.open(archive,'r|gz') as tar:
            for item in tar:
                safe(item.name);require(item.isfile() and item.name not in seen,'Unsafe/duplicate capsule member');seen.add(item.name)
                require(item.size<=200_000_000,'Single extraction member exceeds cap')
                dst=temp/'one_member';h=hashlib.sha256();n=0
                with dst.open('wb') as target:
                    stream=tar.extractfile(item)
                    while True:
                        chunk=stream.read(1024*1024)
                        if not chunk:break
                        target.write(chunk);h.update(chunk);n+=len(chunk)
                if item.name=='CAPSULE_MANIFEST.json':require(json.loads(dst.read_bytes())==manifest,'Capsule manifest differs')
                else:
                    e=expected[item.name];require(n==e['bytes'] and h.hexdigest()==e['sha256'],'Extracted member hash differs')
                dst.unlink();extracted+=n
        require(seen==set(expected)|{'CAPSULE_MANIFEST.json'},'Missing extracted member')
    finally:shutil.rmtree(temp)
    return {'members_verified':len(seen),'extracted_bytes_hashed':extracted,'all_extracted_hashes_verified':True,
            'temporary_extraction_removed_after_verification':True,'models_executed':0}

def run(args):
    out=args.output.absolute();out.mkdir(parents=True,exist_ok=True);public=out/'public';public.mkdir(exist_ok=True)
    report={'started_at_utc':now(),'source_commit':SOURCE,'launcher_commit':os.environ.get('GITHUB_SHA'),
            'engine_sha256':digest(Path(__file__).read_bytes()),
            'runtime':{'python':sys.version,'platform':platform.platform(),'zlib_compile':zlib.ZLIB_VERSION,
                       'zlib_runtime':zlib.ZLIB_RUNTIME_VERSION,'archive_compression':'gzip level6, mtime0, blank original filename'},
            'mode':'hydrate_and_archive' if args.hydrate else 'plan_only','status':'not_complete',
            'new_model_fits':0,'new_model_predictions':0,'GPU_used':False,'R3_accredited':False,
            'human_independent_reproduction_claimed':False,'Zenodo_writes':0,'deposited_archive_replaced':False,
            'existing_published_DOI':'10.5281/zenodo.24356217','global_license_assigned':False}
    manifest,rows,index_sha=read_index(args.index_root)
    report['index_manifest_sha256']=index_sha
    report['groups_planned']={g:{'files':sum(r['group']==g for r in rows),'bytes':sum(r['bytes'] for r in rows if r['group']==g)} for g in ['audit796','v3pilot30','audit193_metadata']}
    if not args.hydrate:
        report.update(status='metadata_plan_verified_no_payload_hydration',finished_at_utc=now());write_json(public/'execution_receipt.json',report);print(json.dumps(report));return 0
    require(args.max_network_bytes>=sum(r['bytes'] for r in rows),'Explicit network budget insufficient')
    require(1<=args.workers<=8 and args.max_artifact_bytes<=200_000_000,'Unsafe worker/artifact bound')
    require(shutil.disk_usage(out).free>=6_000_000_000,'At least6GB free required for ephemeral snapshot')
    budget=Budget(args.max_network_bytes);cas=out/'ephemeral-cas';snapshot=out/'ephemeral-snapshot';payload=out/'capsule-payload'
    cas.mkdir(exist_ok=True);payload.mkdir(exist_ok=True)
    unique={r['sha256']:r for r in rows};progress={'verified':0,'downloaded':0,'reused':0};lock=threading.Lock()
    def hydrate(row):
        target=cas/row['sha256'][:2]/row['sha256'];reused=target.exists()
        try:
            if reused:check(target.read_bytes(),row)
            else:atomic(target,download(row,budget))
        except Exception as error:
            raise RuntimeError('Pinned evidence hydration failed: group='+row['group']+
                ' logical_path='+row['logical_path']+' storage_commit='+row['storage_commit']+
                ' SHA256='+row['sha256']+' cause='+type(error).__name__+':'+str(error)) from error
        with lock:
            progress['verified']+=1;progress['reused' if reused else 'downloaded']+=1
            if progress['verified']%250==0:print(json.dumps({'event':'ephemeral_evidence_hydration',**progress,'unique_total':len(unique),'network_bytes':budget.used}),flush=True)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(hydrate,unique.values()))
        inventory=[];scope_counts={};safe_bytes=0
        for row in rows:
            source=cas/row['sha256'][:2]/row['sha256'];logical=snapshot/row['group']/safe(row['logical_path']);logical.parent.mkdir(parents=True,exist_ok=True)
            if logical.exists():check(logical.read_bytes(),row)
            else:os.link(source,logical)
            keep,reason=evidence_permission(row)
            if keep and source.suffix not in ('.png','.svg') and SECRET.search(source.read_bytes()):keep=False;reason='possible_secret_pattern_requires_review'
            record={**row,'category':category(row['logical_path']),
                  'downloaded_SHA256_and_Git_blob_verified':True,'included_in_capsule':keep,'scope_reason':reason}
            inventory.append(record);key=row['group']+':'+record['category'];stats=scope_counts.setdefault(key,{'files':0,'bytes':0,'included_files':0,'included_bytes':0})
            stats['files']+=1;stats['bytes']+=row['bytes']
            if keep:
                stats['included_files']+=1;stats['included_bytes']+=row['bytes'];dest=payload/'historical_evidence'/row['group']/safe(row['logical_path']);dest.parent.mkdir(parents=True,exist_ok=True);os.link(source,dest);safe_bytes+=row['bytes']
        report['historical_snapshot']={'all_8841_indexed_payload_paths_materialized_and_verified':True,'all_193_receipt_storage_pairs_verified':True,
             'logical_groups':report['groups_planned'],'unique_objects_verified':len(unique),'ephemeral_only':True,
             'full_snapshot_uploaded_as_artifact':False,'training_reproduction':False}
        repo_inventory=[];repo_bytes=0;source_raw='https://codeload.github.com/'+REPO+'/tar.gz/'+SOURCE
        with open_public(source_raw) as response:
            with tarfile.open(fileobj=CountReader(response,budget),mode='r|gz') as tar:
                for member in tar:
                    if member.isdir():continue
                    parts=PurePosixPath(member.name).parts
                    require(len(parts)>1,'Unexpected source archive root');name='/'.join(parts[1:]);safe(name)
                    if not member.isfile():repo_inventory.append({'path':name,'included':False,'reason':'nonregular_source_entry','bytes':member.size});continue
                    repo_bytes+=member.size;require(repo_bytes<=args.max_source_bytes,'Source uncompressed byte ceiling exceeded')
                    raw=tar.extractfile(member).read(member.size+1);require(len(raw)==member.size,'Truncated source member')
                    keep,reason,stored=source_permission(name,raw)
                    row={'path':name,'bytes':len(raw),'sha256':digest(raw),'included':keep,'reason':reason,
                         'source_commit':SOURCE,'source_tree':SOURCE_TREE,'source_url':'https://raw.githubusercontent.com/'+REPO+'/'+SOURCE+'/'+quote(name,safe='/')}
                    if keep:
                        dest=payload/'repository'/safe(name);atomic(dest,stored);safe_bytes+=len(stored)
                        row.update(archived_bytes=len(stored),archived_sha256=digest(stored))
                        require(safe_bytes<=500_000_000,'Allowed source/evidence payload exceeds500MB uncompressed review bound')
                    repo_inventory.append(row)
        toolroot=Path(__file__).resolve().parent
        for p in sorted(toolroot.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix in ('.pyc','.tar','.gz','.zip'):continue
            rel=p.relative_to(toolroot)
            if rel.parts[0]!='human' and rel.name not in ('integral_capsule.py','test_integral_capsule.py','ARCHIVE_DESIGN.md','ZENODO_NEW_VERSION_PLAN.md'):continue
            if any(x in rel.parts for x in ('outputs','receipts','ephemeral-cas','ephemeral-snapshot','capsule-payload')):continue
            if p.stat().st_size>5_000_000:continue
            atomic(payload/'tools'/rel,p.read_bytes())
        write_json(public/'historical_inventory.json',inventory);write_json(public/'repository_inventory.json',repo_inventory)
        write_json(public/'category_coverage.json',scope_counts)
        write_json(payload/'HISTORICAL_INVENTORY.json',inventory);write_json(payload/'REPOSITORY_INVENTORY.json',repo_inventory)
        write_json(payload/'SOURCE_AND_RIGHTS_SCOPE.json',{'base_repository_commit':SOURCE,'launcher_commit':os.environ.get('GITHUB_SHA'),
             'license_policy':'Preserve original notices; no blanket MIT/CC BY grant. The owner decision in the previous documentary ZIP applies only to its selected own members.',
             'excluded':'Raw/unreviewed data, historical checkpoints, nested archives, full prediction panels and external dependency trees are enumerated individually.',
             'approved_ONNX_hashes':sorted(APPROVED_ONNX),'public_codeload_bytes_consumed_and_hashed':repo_bytes,
             'ephemeral_snapshot_is_not_a_published_deposit':True,'human_reproduction_executed':False,'Zenodo_new_version_published':False})
        atomic(payload/'README.md',('''# Source and evidence capsule0.3.0-REVIEW\n\nThis capsule preserves permitted source categories from commit8d03a73 and generated historical evidence, with every omission enumerated. It is not the integral scientific master and is not the newer book edition. The complete indexed master/pilot bytes were hydrated and hashed separately in an ephemeral snapshot, never uploaded as raw artifact. Metadata/locator inventory is not counted as deposited scientific payload.\n\nOriginal component terms remain; no global MIT/CC BY license is invented. Raw inputs, unreviewed checkpoints and full prediction panels remain excluded. Exact sources and hashes permit controlled acquisition under their own terms. The existing Zenodo0.2.0 ZIP is not changed. This0.3.0 review candidate has not been deposited in Zenodo. No training, inference, or human reproduction was performed. See inventories and execution receipt for actual counts.\n''').encode())
        archive=public/ARCHIVE_NAME;capsule_manifest=build_capsule(payload,archive)
        require(archive.stat().st_size<=args.max_artifact_bytes,'Compressed archive exceeds explicit artifact ceiling')
        extraction=verify_extraction(archive,capsule_manifest,out)
        write_json(public/'capsule_manifest.json',capsule_manifest)
        public_total=sum(p.stat().st_size for p in public.iterdir() if p.is_file())
        require(public_total<=200_000_000,'Public artifact total exceeds200MB bound')
        counts={'source_files_observed':len(repo_inventory),'source_files_included':sum(r['included'] for r in repo_inventory),
                'source_uncompressed_bytes_hashed':repo_bytes,'source_bytes_included':sum(r.get('archived_bytes',0) for r in repo_inventory),
                'source_files_excluded':sum(not r['included'] for r in repo_inventory),
                'historical_files_in_capsule':sum(r['included_in_capsule'] for r in inventory),
                'historical_bytes_in_capsule':sum(r['bytes'] for r in inventory if r['included_in_capsule']),
                'approved_ONNX_unique_hashes_included':len({r['sha256'] for r in repo_inventory if r['included'] and r['reason']=='approved_ONNX_hash'})}
        report.update(status='ephemeral_full_snapshot_verified_and_scoped_capsule_built',counts=counts,
             source_codeload_complete=True,network_bytes=budget.used,hydration=progress,public_artifact_bytes_before_receipt=public_total,
             capsule={'file':ARCHIVE_NAME,'bytes':archive.stat().st_size,'sha256':digest(archive.read_bytes()),
                      'members':len(capsule_manifest['files'])+1,'deterministic_header_policy':'sorted paths, zero mtime/uid/gid, stable gzip header'},
             extraction=extraction,integral_repository_or_scientific_master_deposited=False,metadata_and_rights_review_pending=True)
    except Exception as error:
        report.update(status='incomplete_no_full_snapshot_or_archive_claim',error={'type':type(error).__name__,'message':str(error)[:1000]},network_bytes=budget.used,hydration=progress)
        for p in public.glob('*.tar.gz'):p.unlink()
    report['finished_at_utc']=now();write_json(public/'execution_receipt.json',report)
    print('E3_INTEGRAL_ARCHIVE_RECEIPT_BEGIN');print(json.dumps(report,ensure_ascii=False,indent=2));print('E3_INTEGRAL_ARCHIVE_RECEIPT_END',flush=True)
    return 0 if report['status']=='ephemeral_full_snapshot_verified_and_scoped_capsule_built' else 2

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--index-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--hydrate',action='store_true');parser.add_argument('--max-network-bytes',type=int,default=7_000_000_000)
    parser.add_argument('--max-source-bytes',type=int,default=3_000_000_000);parser.add_argument('--max-artifact-bytes',type=int,default=190_000_000)
    parser.add_argument('--workers',type=int,default=6);raise SystemExit(run(parser.parse_args()))
