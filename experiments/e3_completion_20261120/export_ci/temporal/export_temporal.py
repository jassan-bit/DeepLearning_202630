"""Export unchanged E2 temporal checkpoints; AI assistance declared.

No fitting, recipe selection, recalibration or GPU execution. Periodic export
adapters are recorded separately and checked against the unmodified reference.
"""
# MIT License
# 
# Copyright (c) 2021 THUML @ Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import annotations
import argparse,csv,hashlib,itertools,json,math,os,platform,resource,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import onnx
import onnxruntime as ort
SCRIPT_ROOT=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('E3_EXPORT_ROOT',str(SCRIPT_ROOT)))
REPO=Path(os.environ.get('E3_RECIPE_REPO','/workspace/e2_seed_extension/bundle/D02/code'))
AUTHORS=Path(os.environ.get('E3_AUTHOR_SOURCES','/workspace/e2_seed_extension/bundle/D02/authors'))
NPZ=Path(os.environ.get('E3_COHORT_NPZ','/workspace/e3_remediation_calibration_inputs/cohort.npz'))
PREDS=Path(os.environ.get('E3_FROZEN_PREDICTORS','/workspace/e3_completion_delivery/export_standard/inputs_private/combined_predictors_133.json'))
META=Path(os.environ.get('E3_METADATA_ROOT','/workspace/work/e3_ab/metadata/freeze'))
SELECTION=Path(os.environ.get('E3_SELECTION_ROOT','/workspace/work/e3_ab/repo/experiments/e3_bloques_ab_20261115/inputs/selection_receipts'))
TOLERANCE=1e-6
EXPECTED={'npz':'186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683','predictions':'df5206e531ad9c4e5718fb8f17d86390bb1267bcd1d724c2333cacd6c9903655'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canon(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def transform(raw,s):
 raw=np.asarray(raw,dtype=np.float64);v=raw.copy();v[:,:,:4]=np.log(raw[:,:,:4])-np.log(raw[:,-1:,3:4]);v[:,:,4]=np.log1p(raw[:,:,4]);return ((v-s['mean'])/s['std']).astype(np.float32)
def weight_digest(model):
 h=hashlib.sha256()
 for n,t in model.state_dict().items():h.update(n.encode());h.update(t.detach().cpu().numpy().tobytes())
 return h.hexdigest()

def build(name,cp):
 sys.path.insert(0,str(REPO/'scripts'))
 trial=cp['architecture']['training_hpo_record'];roots={n:str(AUTHORS/n) for n in ['TimesNet','TimeMixer','TiDE','PatchTST','iTransformer']}
 if name in ['TimesNet','TimeMixer']:
  from e2_published_periodic import build_periodic
  model=build_periodic(name,trial['dropout'],roots)
 elif name=='TiDE':
  from e2_published_tide import PublishedTiDED7
  model=PublishedTiDED7(source_roots=roots)
 elif name=='PatchTST':
  from e2_published_patchtst import build_patchtst
  model=build_patchtst(dropout=trial['dropout'],source_roots=roots)
 elif name=='iTransformer':
  # Optional author import dependency, never used as an alternative model.
  sys.path.append(os.environ.get('E3_IMPORT_DEPENDENCIES','/workspace/e2_seed_extension_cpu_preflight/runtime/venv-e2-extension/lib/python3.12/site-packages'))
  from e2_final_models_attention import AttentionVolatilityAdapter
  model=AttentionVolatilityAdapter(name,{'parameters':{'dropout':min(trial['dropout'],.3)}},roots)
  for layer in model.modules():
   if isinstance(layer,nn.Dropout):layer.p=trial['dropout']
 else:raise ValueError(name)
 model.load_state_dict(cp['state_dict'],strict=True);model.eval();return model

def radix2_real_fft8(x):
 """Non-DC rFFT8 coefficients through real butterfly additions/multiplications.

 This factorization avoids approximate sin/cos sums for exact constant or
 symmetric sequences. Sign and coefficient choices match rFFT's negative phase.
 """
 a0=x[:,0]+x[:,4];a1=x[:,0]-x[:,4];a2=x[:,2]+x[:,6];a3=x[:,2]-x[:,6]
 b0=x[:,1]+x[:,5];b1=x[:,1]-x[:,5];b2=x[:,3]+x[:,7];b3=x[:,3]-x[:,7]
 c1=(b1-b3)*math.sqrt(.5);c2=(b1+b3)*math.sqrt(.5)
 re=torch.stack([a1+c1,a0-a2,a1-c1,(a0+a2)-(b0+b2)],dim=1)
 im=torch.stack([-a3-c2,-(b0-b2),a3-c2,torch.zeros_like(a0)],dim=1)
 return re,im

def frozen_topk4_rank_table():
 """Pin CPU Torch2.8 tie order for all75 weak orderings of four amplitudes.

 No measured data or threshold epsilon enters this finite control-flow table.
 Exact comparisons compute the order type at runtime; every strict ordering
 and every equality pattern is included. This is an export semantics adapter.
 """
 table=torch.full((256,3),-1,dtype=torch.long);seen={}
 for values in itertools.product(range(4),repeat=4):
  vector=torch.tensor(values,dtype=torch.float32)
  ranks=(vector[:,None]>vector[None,:]).sum(1)
  key=int((ranks*torch.tensor([1,4,16,64])).sum())
  answer=vector.topk(3).indices
  if key in seen:assert torch.equal(seen[key],answer)
  seen[key]=answer;table[key]=answer
 assert len(seen)==75
 return table

class FinitePeriodExportBlock(nn.Module):
 """Length8 real FFT butterflies, finite branches, dynamic exact-rank selection.

 No period is frozen by tracing. The tie adapter preserves pinned Torch CPU
 topk4/k3 order instead of relying on ONNX TopK's different tie convention.
 """
 def __init__(self,source):
  super().__init__();self.conv=source.conv;self.total=source.seq_len+source.pred_len;self.k=source.k
  assert self.total==8 and self.k==3
  self.register_buffer('topk_rank_table',frozen_topk4_rank_table())
  self.register_buffer('rank_key_weights',torch.tensor([1,4,16,64],dtype=torch.long))
  self.periods=[8,4,2,2]
 def spectral(self,x):
  real,imag=radix2_real_fft8(x)
  amplitude=torch.sqrt(real.square()+imag.square())
  return amplitude.mean(-1)
 def indices(self,frequency):
  ranks=(frequency[:,None]>frequency[None,:]).sum(1)
  key=(ranks*self.rank_key_weights).sum()
  return self.topk_rank_table[key]
 def forward(self,x):
  frequency=self.spectral(x)
  # batch1 is a declared external contract, matching original eval semantics.
  index=self.indices(frequency.mean(0))
  candidates=[]
  for period in self.periods:
   grid=x.reshape(1,self.total//period,period,32).permute(0,3,1,2).contiguous()
   values=self.conv(grid).permute(0,2,3,1).reshape(1,self.total,32)
   candidates.append(values)
  branches=torch.index_select(torch.stack(candidates,-1),-1,index)
  weights=torch.index_select(frequency,1,index).softmax(1).unsqueeze(1).unsqueeze(1)
  return (branches*weights).sum(-1)+x

class FixedPatchExport(nn.Module):
 """Exact author end replication and P3/S2 Unfold expressed as fixed slices."""
 def __init__(self,source):
  super().__init__();self.original=source
 def forward(self,x):
  source=self.original
  normalized=source.backbone.revin_layer(x,"norm").transpose(1,2)
  padded=torch.cat([normalized,normalized[:,:,-1:],normalized[:,:,-1:]],dim=-1)
  patches=torch.stack([padded[:,:,start:start+3] for start in (0,2,4,6)],dim=-2).permute(0,1,3,2)
  encoded=source.backbone.backbone(patches)
  channels=source.backbone.head(encoded).squeeze(-1)
  return F.softplus(source.readout(channels)).squeeze(-1)

def adapt_timesnet(model):
 import copy
 exported=copy.deepcopy(model)
 exported.backbone.model=nn.ModuleList(FinitePeriodExportBlock(x) for x in exported.backbone.model)
 exported.eval();return exported

def run(name):
 started=time.monotonic();torch.set_num_threads(1);torch.set_num_interop_threads(1)
 assert sha(NPZ)==EXPECTED['npz'];assert sha(PREDS)==EXPECTED['predictions']
 manifest=json.load(open(SCRIPT_ROOT/'source_input_manifest.json'))
 for relative,digest in manifest['source_code_hashes'].items():assert sha(REPO/relative)==digest,'Pinned recipe source changed: '+relative
 asset=next(a for a in manifest['checkpoints'] if a['model']==name);checkpoint=ROOT/'inputs_private'/(name+'.pt');assert sha(checkpoint)==asset['sha256'] and checkpoint.stat().st_size==asset['bytes']
 cp=torch.load(checkpoint,map_location='cpu',weights_only=True);assert cp['architecture']['initialization_seed']==271801
 selection_path=SELECTION/(name+'.json');selection=json.load(open(selection_path));assert sha(selection_path)==asset['selection_sha256'];assert selection['winner_trial']==cp['architecture']['training_hpo_record'];assert selection['model_id']==name
 scaler_path=META/str(Path(asset['path']).parent.parent/'scaler.json');cal_path=scaler_path.with_name('calibrator.json')
 assert sha(scaler_path)==asset['scaler_sha256'] and sha(cal_path)==asset['calibrator_sha256']
 scaler=json.load(open(scaler_path));assert scaler==cp['scaler'];cal=json.load(open(cal_path));assert cal['sha256']==canon({k:v for k,v in cal.items() if k!='sha256'})
 allpred=json.load(open(PREDS));ref=next(v for v in allpred['predictors'] if v['model']==name and str(v['seed'])=='271801');assert ref['checkpoint_sha256']==asset['sha256'];assert cal['q']==ref['calibrator_q'];assert cal['predictor_sha256']==asset['sha256'];assert cal['calibration_predictions_sha256']==canon(ref['calibration_predictions']);assert cal['calibration_targets_sha256']==canon(ref['calibration_targets'])
 with np.load(NPZ,allow_pickle=False) as z:
  raw=np.asarray(z['evaluation_raw'],dtype=np.float64);y=np.asarray(z['evaluation_y_original'],dtype=np.float64);dates=[str(v) for v in z['evaluation_dates']];calraw=np.asarray(z['calibration_raw'],dtype=np.float64);caly=np.asarray(z['calibration_y_original'],dtype=np.float64);caldates=[str(v) for v in z['calibration_dates']];fitraw=np.asarray(z['training_raw'],dtype=np.float64);fitdates=[str(v) for v in z['training_dates']]
 assert raw.shape==(608,7,5) and dates==allpred['dates'] and np.array_equal(y,allpred['y_true']);assert caldates==cal['dates'] and len(caldates)==53 and np.array_equal(caly,ref['calibration_targets']);assert fitdates==scaler['train_dates']
 x,cx,fx=[transform(v,scaler) for v in [raw,calraw,fitraw]]
 model=build(name,cp);before=weight_digest(model);nparams=sum(v.numel() for v in model.parameters());assert nparams==cp['architecture']['actual_parameter_count']
 exported=adapt_timesnet(model) if name=='TimesNet' else FixedPatchExport(model) if name=='PatchTST' else model
 exported.eval()
 output=ROOT/'payload'/name.lower();output.mkdir(parents=True,exist_ok=True)
 (output/'AUTHOR_LICENSE.txt').write_bytes((AUTHORS/name/'LICENSE').read_bytes())
 outfile=output/'model.onnx'
 torch.onnx.export(exported,(torch.from_numpy(x[:1]),),str(outfile),input_names=['windows'],output_names=['normalized_prediction'],opset_version=17,dynamo=False)
 onnx.checker.check_model(onnx.load(str(outfile)))
 assert sha(outfile)==asset['expected_onnx_sha256'] and outfile.stat().st_size==asset['expected_onnx_bytes'],'Export bytes differ from recorded artifact: keep separately and diagnose'
 opts=ort.SessionOptions();opts.intra_op_num_threads=opts.inter_op_num_threads=1;opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
 session=ort.InferenceSession(str(outfile),sess_options=opts,providers=['CPUExecutionProvider'])
 def mem(m,v):
  with torch.inference_mode():return np.asarray([float(m(torch.from_numpy(a[None]))[0])*scaler['target_scale'] for a in v])
 def onx(v):return np.asarray([float(session.run(None,{'windows':a[None]})[0].reshape(-1)[0])*scaler['target_scale'] for a in v])
 cpu=mem(model,x);pred=onx(x);ccpu=mem(model,cx);cpred=onx(cx);frozen=np.asarray(ref['prediction_original']);fcal=np.asarray(ref['calibration_predictions']);errors={'maximum_absolute_error_vs_CPU_memory':float(abs(pred-cpu).max()),'maximum_absolute_error_vs_frozen_saved_predictions':float(abs(pred-frozen).max()),'maximum_calibration_absolute_error_vs_CPU_memory':float(abs(cpred-ccpu).max()),'maximum_calibration_absolute_error_vs_frozen_predictions':float(abs(cpred-fcal).max())}
 stress=[]
 z=np.zeros((7,5),np.float32);one=np.ones((7,5),np.float32);alt=np.tile(np.asarray([1,-1,1,-1,1,-1,1],np.float32)[:,None],(1,5));sym=np.tile(np.asarray([1,2,3,4,3,2,1],np.float32)[:,None],(1,5))
 stresscases=[('zeros',z),('ones',one),('constant_negative',-one),('alternating',alt),('symmetric',sym)]
 for tag,v in stresscases:
  a=mem(model,v[None]);b=onx(v[None]);stress.append({'case':tag,'max_absolute_error_original_sigma7':float(abs(a-b).max()),'within_tolerance':bool(abs(a-b).max()<=TOLERANCE)})
 errorspass=all(e<=TOLERANCE for e in errors.values());stresspass=all(v['within_tolerance'] for v in stress)
 assert weight_digest(model)==before
 # Provisional timings may contend with other work. Root reruns common ORT-only benchmark.
 forward=[];e2e=[]
 for i in range(110):
  t=time.perf_counter();session.run(None,{'windows':x[:1]});dt=(time.perf_counter()-t)*1000
  t=time.perf_counter();v=transform(raw[:1],scaler);session.run(None,{'windows':v});et=(time.perf_counter()-t)*1000
  if i>=10:forward.append(dt);e2e.append(et)
 report={'model':name,'seed':271801,'status':'passed' if errorspass and stresspass else 'parity_failed','evaluation_dates':608,'date_start':dates[0],'date_end':dates[-1],'calibration_dates':53,'declared_absolute_tolerance_original_sigma7':TOLERANCE,**errors,'RMSE_CPU_memory':float(np.sqrt(np.mean((cpu-y)**2))),'RMSE_ONNX':float(np.sqrt(np.mean((pred-y)**2))),'RMSE_frozen_saved_predictor':float(np.sqrt(np.mean((frozen-y)**2))),'historical_608_coverage_percent_for_this_seed':float(100*np.mean((y>=np.maximum(0,pred-cal['q']))&(y<=pred+cal['q']))),'within_tolerance':errorspass,'stress_cases':stress,'stress_within_tolerance':stresspass,'reference_checkpoint_sha256':asset['sha256'],'checkpoint_unchanged':sha(checkpoint)==asset['sha256'],'reference_state_tensors_unchanged':weight_digest(model)==before,'registered_parameters':nparams,'gradient_enabled_parameters':sum(v.numel() for v in model.parameters() if v.requires_grad),'onnx_bytes':outfile.stat().st_size,'model_sha256':sha(outfile),'runtime_peak_RSS_process_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'RSS_scope':'process peak including Torch, checkpoint, ORT, arrays and export; not incremental ORT memory','provisional_concurrent_forward_median_ms':float(np.median(forward)),'provisional_concurrent_forward_p95_ms':float(np.quantile(forward,.95)),'provisional_concurrent_end_to_end_median_ms':float(np.median(e2e)),'provisional_concurrent_end_to_end_p95_ms':float(np.quantile(e2e,.95)),'timing_scope':'CPU one intra/inter-op thread, batch1,10warmups100calls; provisional contention, excludes explanations','calibration_q_reused_unchanged':cal['q'],'new_training_runs':0,'new_calibration_fits':0,'GPU_used':False,'live_registry_updated':False,'elapsed_seconds':time.monotonic()-started,'reference_eval_batch_policy':'each window independently; frozen TimesNet adapter already enforces this in eval','adapter':('real FFT8 butterfly, all four period branches, dynamic rank/gather with pinned CPU TopK tie order; input periods are not frozen' if name=='TimesNet' else 'P3/S2 fixed slices and explicit end replication exactly replace Unfold; remaining model unchanged' if name=='PatchTST' else 'unchanged model forward; standard ONNX export')}
 metadata={'schema_version':1,'model':name,'seed':271801,'representative_seed_rule':'smallest frozen seed ID; not chosen by outcome','format':'ONNX','opset':17,'model_sha256':sha(outfile),'input_shape':[1,7,5],'input_dtype':'float32','uses_observable_floor':False,'output_name':'normalized_prediction','output_scale':scaler['target_scale'],'output_unit':'original population sigma7 ddof0, daily not annualized','horizon_days':1,'preprocessing':{'mean':scaler['mean'],'std':scaler['std'],'fit_count':655,'fit_end':fitdates[-1],'target_scale':scaler['target_scale']},'calibration_q':cal['q'],'calibration_valid_after':caldates[-1],'interval_nominal_coverage':.9,'calibrator_source':{'file_sha256':sha(cal_path),'content_sha256':cal['sha256'],'q':cal['q'],'n':53,'date_start':caldates[0],'date_end':caldates[-1],'checkpoint_sha256':cal['predictor_sha256']},'interval_limitations':'53 serially dependent historical dates, no general temporal/subgroup coverage guarantee; no refit','ood':{'method':'observed standardized fit ranges only','minimum_standardized':fx.min(0).astype(float).tolist(),'maximum_standardized':fx.max(0).astype(float).tolist(),'not_reliability_guarantee':True},'scientific_source':{'type':'existing_E2_frozen_final_checkpoint','checkpoint_sha256':asset['sha256'],'checkpoint_url':asset['url'],'recipe_source_commit':'bbd2604a9b78bab7fc8289ad07aae0a6117e884e','recipe_source_verification':'Eight recipe script hashes and source manifest match immutable commit; author hashes checked by constructor','registered_trial_id':cp['architecture']['registered_trial_id'],'request_sha256':cp['request_sha256']},'data_source':{'npz_sha256':EXPECTED['npz'],'predictions_sha256':EXPECTED['predictions']},'source_architecture':cp['architecture'],'roundtrip':report,'libraries':{'python':platform.python_version(),'torch_export_only':torch.__version__,'onnx_export_only':onnx.__version__,'onnxruntime':ort.__version__,'numpy':np.__version__},'AI_assistance_declared':True,'serving_enabled':False}
 write(output/'roundtrip.json',report);write(output/'metadata.json',metadata);write(output/'provenance.json',{'model':name,'input_hashes':{'checkpoint':sha(checkpoint),'cohort':sha(NPZ),'predictions':sha(PREDS),'scaler':sha(scaler_path),'calibrator':sha(cal_path),'selection':sha(selection_path)},'output_hashes':{'onnx':sha(outfile),'metadata':sha(output/'metadata.json')},'recipe_source_commit':manifest['recipe_source_commit'],'source_code_hashes':manifest['source_code_hashes'],'export_script_sha256':sha(Path(__file__)),'input_manifest_sha256':sha(SCRIPT_ROOT/'source_input_manifest.json'),'no_training_no_recalibration':True,'source_roles':'Pinned author files are verified locally by constructors; do not copy third-party bodies into this packet.'})
 write(ROOT/'registry'/ (name+'.json'),{'model':name,'onnx_path':str(outfile),'metadata_path':str(output/'metadata.json'),'input_name':'windows','output_name':'normalized_prediction','input_dtype':'float32','input_shape':[1,7,5],'output_scale':scaler['target_scale'],'input_sample_path':str(ROOT/'inputs_private'/(name+'_sample.npy')),'benchmark_status':'requires root common isolated measurement','parity_passed':errorspass and stresspass})
 np.save(ROOT/'inputs_private'/(name+'_sample.npy'),x[:1],allow_pickle=False)
 print(json.dumps(report),flush=True)
 return report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=['TimesNet','TiDE','TimeMixer','PatchTST','iTransformer']);args=p.parse_args()
 try:run(args.model)
 except Exception as e:
  write(ROOT/'failures'/(args.model+'.json'),{'model':args.model,'status':'export_attempt_failed','exception_type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'checkpoint_recovered_and_verified':True,'new_training_runs':0,'new_calibration_fits':0})
  raise
