"""Build the frozen E3 documentary ZIP; no download, training or inference.

manifest.json lists package-relative payload paths. The ZIP contains exactly
those files plus manifest.json. Its external descriptor is never a ZIP member.
"""
from __future__ import annotations
import argparse,hashlib,json,os,re,shutil,stat,tempfile,zipfile
from pathlib import Path,PurePosixPath
ARCHIVE='release/e3-completion-documents-20261120-DRAFT.zip'
DESCRIPTOR='archive_descriptor.json'
SCOPE='Documentación, predicciones, código y localizadores incluidos en el manifiesto; excluye nuevos binarios ONNX, pesos y el maestro experimental de aproximadamente 4 GB.'

def digest(data):return hashlib.sha256(data).hexdigest()
def safe_name(value):
 if not isinstance(value,str) or not value or '\\' in value or '\x00' in value:raise ValueError('Unsafe manifest path')
 p=PurePosixPath(value)
 if p.is_absolute() or p.as_posix()!=value or any(v in ('','.', '..') or ':' in v for v in p.parts):raise ValueError('Unsafe manifest path: '+repr(value))
 return value

def checked_root(value):
 p=Path(value).absolute()
 for q in (p,*p.parents):
  if q.is_symlink():raise ValueError('Symlink root/ancestor rejected: '+str(q))
 if not p.is_dir():raise ValueError('Missing package root: '+str(p))
 return p

def file_path(root,name):
 safe_name(name);p=root.joinpath(*PurePosixPath(name).parts)
 for q in (p,*p.parents):
  if q==root:break
  if q.is_symlink():raise ValueError('Symlink rejected: '+str(q))
 if not p.is_file():raise ValueError('Missing regular payload file: '+str(p))
 if not stat.S_ISREG(p.stat().st_mode):raise ValueError('Nonregular payload file: '+str(p))
 return p

def check_file(root,row):
 name=safe_name(row['path']);n=row['bytes'];h=row['sha256']
 if type(n) is not int or n<0 or not isinstance(h,str) or not re.fullmatch('[0-9a-f]{64}',h):raise ValueError('Invalid bytes/SHA256 in manifest: '+name)
 p=file_path(root,name);data=p.read_bytes()
 if len(data)!=n or digest(data)!=h:raise ValueError('Payload hash/size mismatch: '+str(p))
 return data

def load_payload(root):
 manifest=file_path(root,'manifest.json').read_bytes();value=json.loads(manifest)
 if not isinstance(value,dict) or not isinstance(value.get('files'),list):raise ValueError('manifest.json must contain files[]')
 rows=[];seen=set()
 for row in value['files']:
  if not isinstance(row,dict):raise ValueError('Invalid manifest entry')
  name=safe_name(row.get('path'))
  if name in ('manifest.json',DESCRIPTOR,'verification.json') or name.lower().endswith('.zip'):raise ValueError('Excluded cyclic/generated member: '+name)
  if name.casefold() in seen:raise ValueError('Duplicate/case-colliding member: '+name)
  seen.add(name.casefold());check_file(root,row);rows.append({'path':name,'bytes':row['bytes'],'sha256':row['sha256']})
 rows.append({'path':'manifest.json','bytes':len(manifest),'sha256':digest(manifest)})
 return manifest,sorted(rows,key=lambda row:row['path'])

def write_atomic(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.is_symlink() or any(p.is_symlink() for p in path.parents):raise ValueError('Symlink destination rejected: '+str(path))
 fd,tmp=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent)
 try:
  with os.fdopen(fd,'wb') as stream:stream.write(data)
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)

def assemble(source,static,write_descriptor=False):
 source=checked_root(source);static=checked_root(static)
 if source==static or source in static.parents or static in source.parents:raise ValueError('Source and static roots must be separate')
 manifest,rows=load_payload(source);mirror,mirror_rows=load_payload(static)
 if manifest!=mirror or rows!=mirror_rows:raise ValueError('Source/static manifest or payload differs')
 # Build from the checked manifest only: generated ZIPs/descriptors are excluded.
 fd,tmp=tempfile.mkstemp(prefix='e3-documents-',suffix='.zip');os.close(fd)
 try:
  with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9,allowZip64=True) as archive:
   for row in rows:
    data=manifest if row['path']=='manifest.json' else check_file(source,row)
    if len(data)!=row['bytes'] or digest(data)!=row['sha256']:raise ValueError('Manifest changed while assembling')
    info=zipfile.ZipInfo(row['path'],date_time=(1980,1,1,0,0,0));info.create_system=3;info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=(stat.S_IFREG|0o644)<<16;info.internal_attr=0;info.extra=b'';info.comment=b''
    archive.writestr(info,data,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
  data=Path(tmp).read_bytes()
  descriptor={'schema_version':1,'archive':ARCHIVE,'bytes':len(data),'sha256':digest(data),'members':rows,'compression':'ZIP_DEFLATED level9','timestamp':'1980-01-01T00:00:00','scope':SCOPE,'descriptor_in_archive':False,'new_training_runs':0,'new_inference_runs':0}
  if write_descriptor:
   encoded=(json.dumps(descriptor,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()
   write_atomic(source/DESCRIPTOR,encoded);write_atomic(static/DESCRIPTOR,encoded)
  else:
   for root in (source,static):
    existing=json.loads(file_path(root,DESCRIPTOR).read_bytes())
    if existing!=descriptor:raise ValueError('Frozen archive descriptor does not match; do not update it implicitly: '+str(root/DESCRIPTOR))
  for root in (source,static):write_atomic(root/ARCHIVE,data)
  return descriptor
 finally:Path(tmp).unlink(missing_ok=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);p.add_argument('--static',required=True,type=Path);p.add_argument('--write-descriptor',action='store_true',help='Explicit freeze step: write external descriptors after payload manifest is final');a=p.parse_args();result=assemble(a.source,a.static,a.write_descriptor);print(json.dumps({'status':'verified','archive':result['archive'],'sha256':result['sha256'],'bytes':result['bytes'],'members':len(result['members']),'scope':result['scope']},ensure_ascii=False))
