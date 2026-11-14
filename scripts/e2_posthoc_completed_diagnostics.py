"""Verify already-completed frozen diagnostics without reserving or computing."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path,PurePosixPath

ENGINE='1e8745497b484518f766b6da5c19be8362d9add7'
CONFIGURATION='ae1a2d4f96807a20ea92c313297a5c09512a4817'
RUN_ID='posthoc-8295dc64b34339856f7f3dfd'
VERSION='e2-frozen-posthoc-inference-v1'
ROOT_RESULT_SHA='67c558e85ea82df9194a4a199fed7013e6fd57e82bcf512f437fabd4b9391810'
FREEZE_SHA='c0e9e2589dca9fa03b341bb72f87d8b541d33b4721715622daeae4438dbd60b4'
SOURCES={
 'e2_frozen_posthoc_gpu.py':'c4918747c58270db4c3a1e8b5021ba0f88de3817a4e1910e582fee1160436545',
 'e2_posthoc_fit_reproduction.py':'5d8ce059f96f86a35b7e2d2530e3fb776827240fcb29df920b4d05374f139fe9',
 'export_e2_f2a_cached_data.py':'6370b8e4ef1fbf7519322107de5f26eb659ecf0dfe59dd6ed7df477a2da586ef'}

def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def regular(path):
 path=Path(path).absolute()
 if any(p.is_symlink()for p in (path,*path.parents)):raise ValueError('Completed diagnostic paths must be regular')
 return path

def read(root,relative):
 name=PurePosixPath(relative)
 if name.is_absolute()or'..'in name.parts or str(name)!=relative or relative=='.':raise ValueError('Unsafe completed diagnostic artifact path')
 path=regular(Path(root)/relative)
 if not path.is_file():raise ValueError('Missing completed diagnostic artifact: '+relative)
 return path.read_bytes()

def _receipt(output,folder,state,claims):
 relative=folder+'/result.json';raw=read(output,relative);value=json.loads(raw)
 binding=state['files'].get(relative,{})
 if binding!={'bytes':len(raw),'sha256':sha(raw)}:raise ValueError('Remote control does not bind completed diagnostic receipt: '+folder)
 if (value.get('status')!='completed'or value.get('run_id')!=RUN_ID or value.get('version')!=VERSION
     or value.get('training_performed')is not False):raise ValueError('Diagnostic identity/status does not prove completion')
 if set(value.get('artifacts_sha256',{}))!=set(claims):raise ValueError('Completed diagnostic artifact set differs')
 for path,expected in value['artifacts_sha256'].items():
  if sha(read(Path(output)/folder,path))!=expected:raise ValueError('Completed diagnostic artifact checksum differs')
 if folder not in state.get('result_backups',{}):raise ValueError('Completed diagnostic backup is absent from authoritative control')
 return value,raw

def verify_completed(output_root,extension_root,remote_state,*,original_output=None):
 """Return a read-only admission for skipping a fully completed inference stage.

 No imports of Torch/NumPy, predictions, bootstrap, fitting, installation,
 network requests, ledger changes or new files are performed.
 """
 output=regular(output_root);extension=regular(extension_root)
 original=regular(original_output or output.parent/'github-primary-output-v1')
 state=remote_state
 if (not isinstance(state,dict)or state.get('version')!='e2-guide-closure-posthoc-state-v1'
     or state.get('scientific_commit')!=ENGINE or state.get('output_path')!=str(output)
     or state.get('state_id')!=sha(canonical({k:v for k,v in state.items()if k!='state_id'}))):
  raise ValueError('Authoritative posthoc control identity differs')
 root_receipt,raw=_receipt(output,'robustness',state,('registration.json','summary.json','robustness.csv'))
 if sha(raw)!=ROOT_RESULT_SHA or root_receipt.get('attempt_id')!=RUN_ID:
  raise ValueError('Original completed diagnostic aggregate receipt differs')
 config=json.loads(read(output,'robustness/registration.json'))
 if ('posthoc-'+sha(canonical(config))[:24]!=RUN_ID or config.get('version')!=VERSION
     or config.get('configuration_commit')!=CONFIGURATION or config.get('engine_commit')!=ENGINE
     or config.get('freeze_sha256')!=FREEZE_SHA or config.get('diagnostic_files_sha256')!=SOURCES):
  raise ValueError('Original frozen diagnostic configuration differs')
 for name,expected in SOURCES.items():
  if sha(read(extension,'scripts/'+name))!=expected:raise ValueError('Original diagnostic source changed')
 freeze_raw=read(original,'final/final_freeze.json')
 if sha(freeze_raw)!=FREEZE_SHA:raise ValueError('Original frozen catalogue changed')
 rows=[r for r in json.loads(freeze_raw)['models']if r['seed']!='deterministic'or r['model']=='ARIMA']
 identities={(r['model'],str(r['seed']))for r in rows}
 if len(rows)!=79 or len(identities)!=79 or sum(r['seed']!='deterministic'for r in rows)!=78:
  raise ValueError('The complete 79 frozen predictors are required')
 expected_folders=set()
 for row in rows:
  key=hashlib.sha256((row['model']+str(row['seed'])).encode()).hexdigest()[:24]
  folder='robustness/predictors/'+key;expected_folders.add(folder)
  receipt,_=_receipt(output,folder,state,('started.json','diagnostics.json'))
  if receipt.get('attempt_id')!=RUN_ID+'-'+key:raise ValueError('Predictor receipt identity differs')
  diagnostic=json.loads(read(output,folder+'/diagnostics.json'))
  baseline=diagnostic.get('baseline_predictions',[])
  if (diagnostic.get('model')!=row['model']or diagnostic.get('seed')!=row['seed']
      or diagnostic.get('run_id')!=RUN_ID or diagnostic.get('checkpoint_sha256')!=row['checkpoint_sha256']
      or diagnostic.get('baseline_projection_passed')is not True or diagnostic.get('training_performed')is not False
      or diagnostic.get('n_dates')!=608 or len(baseline)!=608
      or not all(type(v)in(int,float)and math.isfinite(v)for v in baseline)):
   raise ValueError('Completed predictor does not match its frozen projection')
 actual_folders={p[:-len('/result.json')]for p in state['files']if p.startswith('robustness/predictors/')and p.endswith('/result.json')}
 if actual_folders!=expected_folders:raise ValueError('Remote predictor completion catalogue differs')
 summary=json.loads(read(output,'robustness/summary.json'))
 if (summary.get('run_id')!=RUN_ID or summary.get('version')!=VERSION
     or summary.get('predictor_records')!=79 or summary.get('models_neural')!=13
     or summary.get('neural_predictors')!=78 or summary.get('all_baseline_projections_passed')is not True
     or len(summary.get('profiles',[]))!=78 or len(summary.get('robustness',[]))!=468
     or {x.get('model')for x in summary.get('interpretability',[])}!={'LSTM-autoencoder con predictor','GRU32','ARIMA'}):
  raise ValueError('Completed aggregate has an incomplete diagnostic population')
 return {'status':'verified_completed_diagnostics_reused','run_id':RUN_ID,'completed_predictors':79,
         'neural_inference_profiles':78,'robustness_rows':468,'explained_systems':3,
         'aggregate_result_sha256':ROOT_RESULT_SHA,'verified_predictor_artifact_count':158,
         'model_inference_performed':False,'scientific_fit_performed':False,'ledger_modified':False}
