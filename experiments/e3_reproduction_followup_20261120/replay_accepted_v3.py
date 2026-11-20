"""Replay the public accepted V3 ONNX on its 55 existing dates. No new science.

AI assistance declared. Preprocessing is the original replay_common.preprocess
from commit7db8040619f602d7c52ae0c8bf54f7b6f5b5110e, reproduced here with
input-manifest verification and per-field discrepancies. No Torch/exporter.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time
import numpy as np
import onnxruntime as ort
from hydrate_accepted import verify

TOL=1e-6


def preprocess(raw,scaler):
    raw=np.asarray(raw,dtype=np.float64)
    if raw.ndim!=3 or raw.shape[1:]!=(7,5) or not np.isfinite(raw).all():
        raise ValueError('Finite N x7 x5 OHLCV required')
    if not (raw[...,:4]>0).all() or not (raw[...,4]>=0).all():
        raise ValueError('Positive prices and nonnegative volumes required')
    if not ((raw[...,1]>=np.maximum(raw[...,0],raw[...,3])).all()
            and (raw[...,2]<=np.minimum(raw[...,0],raw[...,3])).all()
            and (raw[...,1]>=raw[...,2]).all()):raise ValueError('Invalid OHLC geometry')
    returns=np.diff(np.log(raw[...,3]),axis=1)
    A=((returns-returns.mean(axis=1)[:,None])**2).sum(axis=1)/7.0
    relative=np.empty_like(raw)
    relative[...,:4]=np.log(raw[...,:4])-np.log(raw[:,-1,3])[:,None,None]
    relative[...,4]=np.log1p(raw[...,4])
    X=((relative-np.asarray(scaler['mean']))/np.asarray(scaler['std'])).astype(np.float32)
    if not np.isfinite(X).all() or not np.isfinite(A).all():raise ValueError('Invalid preprocessing')
    return np.ascontiguousarray(X),np.ascontiguousarray(A,dtype=np.float64)


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();root=Path(__file__).resolve().parent;inputs=Path(args.inputs);out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();started=datetime.now(timezone.utc).isoformat()
    manifest=json.loads((root/'accepted_input_manifest.json').read_text())
    bindings=verify(inputs,manifest)
    metadata=json.loads((inputs/'metadata.json').read_text())
    if metadata['calibration']['status']!='not_calibrated' or metadata['default_serving_enabled']:
        raise ValueError('Experimental uncalibrated profile required')
    with (inputs/'roundtrip_predictions.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    expected_dates=metadata['dates']
    if [r['date'] for r in rows]!=expected_dates or len(set(expected_dates))!=55:
        raise ValueError('Reference dates/order differ')
    with np.load(inputs/'cohort.npz',allow_pickle=False) as data:
        raw=data['training_raw'].copy();dates=data['training_dates'].astype(str).copy()
    if len(set(dates.tolist()))!=len(dates):raise ValueError('Duplicate source dates')
    lookup={date:i for i,date in enumerate(dates)}
    if not all(date in lookup for date in expected_dates):raise ValueError('Missing reference date')
    raw=raw[[lookup[date] for date in expected_dates]];X,A=preprocess(raw,metadata['scaler'])
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    sess=ort.InferenceSession(str(inputs/'model.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
    contract=[{'name':v.name,'type':v.type,'shape':v.shape} for v in sess.get_inputs()]
    if contract!=[{'name':'values','type':'tensor(float)','shape':[1,7,5]},
                  {'name':'A','type':'tensor(double)','shape':[1]}]:raise ValueError('ONNX contract differs')
    if sess.get_outputs()[0].name!='sigma7' or sess.get_outputs()[0].type!='tensor(double)':
        raise ValueError('Output contract differs')
    preds=np.asarray([float(sess.run(['sigma7'],{'values':X[i:i+1],'A':A[i:i+1]})[0][0]) for i in range(55)])
    gpu=np.asarray([float(r['frozen_GPU']) for r in rows]);accepted=np.asarray([float(r['exported_ORT']) for r in rows])
    gpu_errors=np.abs(preds-gpu);accepted_errors=np.abs(preds-accepted);floor=np.sqrt(A)
    if not np.isfinite(preds).all() or np.any(gpu_errors>TOL) or np.any(accepted_errors>TOL) or np.any(preds<floor):
        raise ValueError('Accepted replay discrepancy or structural floor failure')
    fields=['date','frozen_GPU_sigma7','accepted_ORT_sigma7','current_ORT_sigma7','absolute_error_GPU_sigma7','absolute_error_accepted_ORT_sigma7','known_A_sigma7_squared','structural_floor_sigma7']
    with (out/'discrepancies.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i,row in enumerate(rows):
            values=[row['date']]+[repr(float(value)) for value in (gpu[i],accepted[i],preds[i],gpu_errors[i],accepted_errors[i],A[i],floor[i])]
            w.writerow(dict(zip(fields,values)))
    receipt={'status':'passed','started_at_UTC':started,'finished_at_UTC':datetime.now(timezone.utc).isoformat(),'AI_assistance_declared':True,'level':'B','execution_kind':'automated accepted-binary replay in separate process/container',
      'inputs':bindings,'profile_id':metadata['profile_id'],'seed':metadata['seed'],'source_state':metadata['source_state'],
      'comparison':{'rows':55,'date_order_exact':True,'input_contract':contract,'units':'daily unannualized population sigma7 ddof0',
         'absolute_tolerance_sigma7':TOL,'max_absolute_error_vs_frozen_GPU_sigma7':float(gpu_errors.max()),
         'max_absolute_error_vs_accepted_ORT_sigma7':float(accepted_errors.max()),'finite_outputs':bool(np.isfinite(preds).all()),'structural_floor_violations':int(np.sum(preds<floor)),
         'discrepancy_file':'discrepancies.csv'},
      'opened_NPZ_members':['training_raw','training_dates'],
      'software':{'python':platform.python_version(),'numpy':np.__version__,'onnxruntime':ort.__version__,'platform':platform.platform()},
      'resources':{'elapsed_seconds':time.perf_counter()-start,'process_peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'scope':'whole process; shared host, no cross-model performance claim','threads':1,'provider':'CPUExecutionProvider'},
      'new_model_fits':0,'optimizer_steps':0,'reexports':0,'new_calibrations':0,'new_scientific_dates':0,
      'torch_imported':'torch' in sys.modules,'human_independent_reproduction_observed':False,
      'level_C_training_reproduced':False,'R3_accredited':False,'default_serving_enabled':False,
      'excludes':['training reconstruction','independent human validation','Render acceptance','prospective confirmation','new market evaluation']}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
