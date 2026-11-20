"""Recreate five E2 ONNX graphs from immutable public inputs; never fit models.

Only --output is publishable. --workdir retains downloaded OHLCV, author trees,
original checkpoints, intermediate graphs and logs and must never be uploaded.
"""
from __future__ import annotations
import argparse,concurrent.futures,hashlib,json,os,shutil,subprocess,sys,urllib.request,zipfile
from pathlib import Path
SCRIPT_ROOT=Path(__file__).resolve().parent
NAMES=('TimesNet','TiDE','TimeMixer','PatchTST','iTransformer')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def fetch(url,path,digest,size=None):
 if path.exists():
  assert sha(path)==digest,'Cached source changed: '+str(path)
  if size is not None:assert path.stat().st_size==size
  return {'path':str(path),'sha256':digest,'status':'verified_cache'}
 path.parent.mkdir(parents=True,exist_ok=True)
 req=urllib.request.Request(url,headers={'User-Agent':'E3-inference-export-integrity/1.0'})
 with urllib.request.urlopen(req,timeout=60) as response:data=response.read(20_000_000)
 assert hashlib.sha256(data).hexdigest()==digest,'Download SHA mismatch: '+url
 if size is not None:assert len(data)==size,'Download size mismatch: '+url
 temp=path.with_name(path.name+'.download-part');temp.write_bytes(data);temp.replace(path)
 return {'path':str(path),'sha256':digest,'status':'downloaded_verified'}
def run(args):
 work=args.workdir.resolve();output=args.output.resolve()
 assert work!=output and work not in output.parents and output not in work.parents,'Keep private workdir and public output as separate directories'
 work.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
 manifest=json.load(open(SCRIPT_ROOT/'source_input_manifest.json'))
 repo=work/'recipe';authors=work/'author_sources';metadata=work/'metadata';selection=work/'selection';exports=work/'export';inputs=exports/'inputs_private'
 base='https://raw.githubusercontent.com/'+manifest['repository']+'/'+manifest['recipe_source_commit']+'/'
 jobs=[(base+rel,repo/rel,digest,None) for rel,digest in manifest['source_code_hashes'].items()]
 jobs += [(r['url'],authors/r['model']/r['path'],r['sha256'],r['bytes']) for r in manifest['author_files']]
 aux=manifest['auxiliary_inputs']
 jobs += [(v['url'],work/(k+('.zip' if k=='metadata_zip' else '.npz' if k=='cohort' else '.json')),v['sha256'],v['bytes']) for k,v in aux.items()]
 jobs += [(r['selection_url'],selection/(r['model']+'.json'),r['selection_sha256'],None) for r in manifest['checkpoints']]
 jobs += [(r['url'],inputs/(r['model']+'.pt'),r['sha256'],r['bytes']) for r in manifest['checkpoints']]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  receipts=list(pool.map(lambda row:fetch(*row),jobs))
 with zipfile.ZipFile(work/'metadata_zip.zip') as archive:
  for row in manifest['checkpoints']:
   for kind in ('scaler','calibrator'):
    relative=row[kind+'_relative_path'];data=archive.read(relative)
    assert hashlib.sha256(data).hexdigest()==row[kind+'_sha256']
    target=metadata/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
 write(work/'download_receipts.json',receipts)
 env=dict(os.environ,E3_EXPORT_ROOT=str(exports),E3_RECIPE_REPO=str(repo),E3_AUTHOR_SOURCES=str(authors),E3_METADATA_ROOT=str(metadata),E3_SELECTION_ROOT=str(selection),E3_COHORT_NPZ=str(work/'cohort.npz'),E3_FROZEN_PREDICTORS=str(work/'predictions.json'))
 records=[]
 for name in NAMES:
  if args.models and name not in args.models:continue
  logfile=work/(name+'.log')
  with logfile.open('w') as log:
   subprocess.run([sys.executable,str(SCRIPT_ROOT/'export_temporal.py'),'--model',name],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  print(name+': checkpoint+recipe+ONNX SHA and 608+53 parity verified',flush=True)
  if name=='TimesNet':
   for test in ('test_timesnet_edges.py','test_timesnet_topk.py'):
    with (work/(test+'.log')).open('w') as log:subprocess.run([sys.executable,str(SCRIPT_ROOT/test)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  slug=name.lower();source=exports/'payload'/slug
  row=next(r for r in manifest['checkpoints'] if r['model']==name)
  assert sha(source/'model.onnx')==row['expected_onnx_sha256']
  report=json.load(open(source/'roundtrip.json'));assert report['within_tolerance'] and report['stress_within_tolerance'] and report['new_training_runs']==0
  shutil.copytree(source,output/slug,dirs_exist_ok=True)
  records.append({'model':name,'directory':slug,'checkpoint_sha256':row['sha256'],'onnx_sha256':row['expected_onnx_sha256'],'onnx_bytes':row['expected_onnx_bytes'],'roundtrip_sha256':sha(output/slug/'roundtrip.json'),'metadata_sha256':sha(output/slug/'metadata.json'),'cases':661,'absolute_tolerance_sigma7':1e-6,'new_training_runs':0,'new_calibration_fits':0})
 write(output/'temporal_export_manifest.json',{'schema_version':1,'scope':'Artifact recreation and CPU inference parity only; no model fitting, model promotion or R1/R3 accreditation','models':records,'recipe_source_commit':manifest['recipe_source_commit'],'source_input_manifest_sha256':sha(SCRIPT_ROOT/'source_input_manifest.json'),'new_training_runs':0,'new_calibration_fits':0,'private_downloads_included':False,'AI_assistance_declared':True})
 print(json.dumps({'status':'passed','models':len(records),'output':str(output),'new_training_runs':0}),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--workdir',required=True,type=Path);parser.add_argument('--output',required=True,type=Path);parser.add_argument('--models',nargs='*',choices=NAMES);run(parser.parse_args())
