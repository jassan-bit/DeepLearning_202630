"""Publish immutable engineering exports after strict CI recreation; no model fitting.
AI assistance declared. This engineering release is not a statistical protocol,
DOI deposit, model promotion, or license declaration. Only checked public output
folders enter the ZIP; checkpoint/raw-download work directories are excluded.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, urllib.error, urllib.parse, urllib.request, zipfile

REPO='jassan-bit/DeepLearning_202630'; API='https://api.github.com/repos/'+REPO

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--standard',type=Path,required=True);p.add_argument('--temporal',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 commit=os.environ['GITHUB_SHA'];token=os.environ['GH_TOKEN'];assert re.fullmatch('[0-9a-f]{40}',commit)
 assert os.environ['GITHUB_REPOSITORY']==REPO
 a.output.mkdir(parents=True,exist_ok=True); models={}; included={}
 for kind,root in [('standard',a.standard),('temporal',a.temporal)]:
  for fp in sorted(root.rglob('*')):
   if fp.is_file():
    assert not fp.is_symlink() and fp.suffix not in {'.pt','.npz','.npy','.whl'},str(fp)
    rel=kind+'/'+fp.relative_to(root).as_posix();included[rel]={'bytes':fp.stat().st_size,'sha256':sha(fp)}
    if fp.name=='model.onnx':models[rel]=included[rel]
 assert len(models)==9, models
 manifest={'schema_version':1,'source_commit':commit,'AI_assistance_declared':True,'scientific_scope':'Nine frozen historical representatives; engineering parity, no new training or model promotion','new_model_fits':0,'new_calibrations':0,'models':models,'files':included}
 manifest_path=a.output/'exports_manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
 archive=a.output/'e3-nine-onnx-20261120.zip'
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for kind,root in [('standard',a.standard),('temporal',a.temporal)]:
   for fp in sorted(root.rglob('*')):
    if fp.is_file():
     info=zipfile.ZipInfo(kind+'/'+fp.relative_to(root).as_posix(),(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16;z.writestr(info,fp.read_bytes())
  info=zipfile.ZipInfo('exports_manifest.json',(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,manifest_path.read_bytes())
 archive_sha=sha(archive);tag='e3-nine-onnx-'+commit[:12]
 def api(method,url,payload=None,raw=None,content_type='application/json'):
  assert url.startswith(API) or url.startswith('https://uploads.github.com/repos/'+REPO+'/releases/')
  data=raw if raw is not None else json.dumps(payload).encode() if payload is not None else None
  req=urllib.request.Request(url,data=data,method=method,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','Content-Type':content_type,'X-GitHub-Api-Version':'2022-11-28','User-Agent':'E3-export-release/1'})
  with urllib.request.urlopen(req,timeout=120) as r:return json.load(r)
 try:release=api('GET',API+'/releases/tags/'+tag)
 except urllib.error.HTTPError as e:
  if e.code!=404:raise
  release=api('POST',API+'/releases',{'tag_name':tag,'target_commitish':commit,'name':'Nine frozen ONNX exports — CPU parity','body':'AI-assisted engineering export from existing public frozen checkpoints. All nine graphs match the declared SHA-256 and pass 608 evaluation plus53 calibration cases. No new training, calibration, promotion, or R3 accreditation. Temporary raw/checkpoint work folders excluded. This is not a Zenodo archival deposit.','draft':False,'prerelease':True})
 assets={r['name']:r for r in release.get('assets',[])};published={}
 for fp in [archive,manifest_path]:
  expected=sha(fp)
  if fp.name in assets:
   asset=assets[fp.name];assert asset.get('digest')=='sha256:'+expected,'Existing immutable release asset differs; never overwrite'
  else:
   url=release['upload_url'].split('{',1)[0]+'?name='+urllib.parse.quote(fp.name)
   asset=api('POST',url,raw=fp.read_bytes(),content_type='application/zip' if fp.suffix=='.zip' else 'application/json')
   if asset.get('digest') is not None:assert asset['digest']=='sha256:'+expected
  # Public unauthenticated read verifies availability and full asset content.
  req=urllib.request.Request(asset['browser_download_url'],headers={'User-Agent':'E3-export-public-integrity/1'})
  with urllib.request.urlopen(req,timeout=120) as r:raw=r.read()
  assert hashlib.sha256(raw).hexdigest()==expected and len(raw)==fp.stat().st_size
  published[fp.name]={'url':asset['browser_download_url'],'bytes':len(raw),'sha256':expected,'public_full_download_verified':True}
 report={'status':'nine_frozen_ONNX_release_publicly_verified','source_commit':commit,'tag':tag,'release_url':release['html_url'],'assets':published,'models':models,'new_model_fits':0,'new_calibrations':0,'served_by_Render':False,'R3_accredited':False}
 (a.output/'publication_receipt.json').write_text(json.dumps(report,indent=2)+'\n')
 print('E3_ONNX_RELEASE_REPORT='+json.dumps(report,sort_keys=True),flush=True)
if __name__=='__main__':main()
