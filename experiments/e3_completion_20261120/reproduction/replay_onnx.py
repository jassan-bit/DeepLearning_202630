"""CPU-only frozen V3 ONNX replay. AI assistance declared; no model fitting."""
from __future__ import annotations
import argparse
import csv
import json
import os
import platform
import resource
import statistics
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from replay_common import ABS_TOL_SIGMA7, development_inputs, preprocess, session_options, sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--npz',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parent
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    profile=root/'artifacts/atom_crps_seed20261118'
    meta=json.loads((profile/'metadata.json').read_text())
    model=profile/'model.onnx'
    if sha(model)!=meta['onnx_sha256']:raise ValueError('ONNX SHA256 differs')
    manifest=root/'MANIFEST.sha256.json'
    if manifest.exists():
        for path,expected in json.loads(manifest.read_text())['files'].items():
            if sha(root/path)!=expected:raise ValueError('Input manifest differs: '+path)
    raw,X,A,rows,admission=development_inputs(root,args.npz)
    session=ort.InferenceSession(str(model),sess_options=session_options(ort),providers=['CPUExecutionProvider'])
    def forward(i):return session.run(['sigma7'],{'values':X[i:i+1],'A':A[i:i+1]})[0][0]
    predictions=np.asarray([forward(i) for i in range(len(rows))],dtype=np.float64)
    saved=np.asarray([float(r['prediction']) for r in rows])
    errors=np.abs(predictions-saved)
    if not np.isfinite(predictions).all() or not np.all(errors<=ABS_TOL_SIGMA7):raise ValueError('Frozen prediction tolerance failed')
    floor=np.sqrt(A)
    if np.any(predictions<floor):raise ValueError('Structural floor violated')
    # Fixed workload; deterministic cyclic development windows, no best case selection.
    for i in range(10):forward(i%len(rows))
    forward_ms=[];end_to_end_ms=[]
    for i in range(100):
        index=i%len(rows);start=time.perf_counter_ns();forward(index)
        forward_ms.append((time.perf_counter_ns()-start)/1e6)
    for i in range(10):
        index=i%len(rows);xx,aa=preprocess(raw[index:index+1],admission['scaler'])
        session.run(['sigma7'],{'values':xx,'A':aa})
    for i in range(100):
        index=i%len(rows);start=time.perf_counter_ns()
        xx,aa=preprocess(raw[index:index+1],admission['scaler'])
        session.run(['sigma7'],{'values':xx,'A':aa})
        end_to_end_ms.append((time.perf_counter_ns()-start)/1e6)
    def timing(values):return {'calls':len(values),'median_ms':float(np.median(values)),'p95_ms':float(np.quantile(values,.95)),'mean_ms':float(np.mean(values))}
    with (out/'predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['date','saved_GPU_sigma7','replayed_ORT_sigma7','absolute_error_sigma7']);writer.writeheader()
        writer.writerows({'date':row['date'],'saved_GPU_sigma7':repr(float(y)),'replayed_ORT_sigma7':repr(float(p)),'absolute_error_sigma7':repr(float(e))} for row,y,p,e in zip(rows,saved,predictions,errors))
    receipt={'status':'passed','AI_assistance_declared':True,'level':'B: one frozen V3 checkpoint, original preprocessing and 55 development predictions',
      'scope_excludes':['training replay','all796 additive fits','human independent reproduction','calibration','new domain evaluation','serving acceptance'],
      'profile':meta['profile_id'],'onnx_sha256':sha(model),'checkpoint_sha256':meta['source_checkpoint_sha256'],
      'npz_sha256':sha(args.npz),'NPZ_members_opened':['training_raw','training_dates'],
      'comparison':{'rows':55,'first_date':rows[0]['date'],'last_date':rows[-1]['date'],'units':'daily unannualized population sigma7, ddof0','absolute_tolerance':ABS_TOL_SIGMA7,'max_absolute_error':float(errors.max()),'floor_violations':int(np.sum(predictions<floor))},
      'benchmark':{'backend':'ONNX Runtime CPUExecutionProvider','batch':1,'intra_op_threads':1,'inter_op_threads':1,'warmup_calls_each':10,'timed_calls_each':100,'forward':timing(forward_ms),'end_to_end_raw_validation_preprocessing_forward':timing(end_to_end_ms),'process_peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'process_peak_RSS_scope':'standalone runtime including Python, NumPy, ORT, loaded compressed cohort; no Torch or exporter','shared_host_load':'uncontrolled; not a comparable isolated hardware cost study'},
      'software':{'python':platform.python_version(),'numpy':np.__version__,'onnxruntime':ort.__version__,'platform':platform.platform()},
      'new_model_fits':0,'optimizer_steps':0,'new_calibrations':0,'torch_imported':False,'human_reproduction_observed':False}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main()
