"""Comparable ORT-only CPU measurements; no training, calibration, or HTTP.

AI-assisted technical audit. Each model runs in a fresh process on one logical
CPU, one ORT session/thread, ten warmups and one hundred single-window calls.
RSS is total process high-water mark, never incremental model memory. Includes
neither Dash/browser transport nor occlusion explanations. Filesystem caches are
not flushed; session load times are observations, not cold-start estimates.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, platform, resource, subprocess, sys, time
from pathlib import Path

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def child(row, raw, cpu):
    os.sched_setaffinity(0,{cpu})
    import numpy as np
    import onnxruntime as ort
    raw=np.asarray(raw,dtype=np.float64)
    assert raw.shape==(7,5) and np.isfinite(raw).all()
    assert sha(row['model_path'])==row['model_sha256']
    assert sha(row['metadata_path'])==row['metadata_sha256']
    rel=raw.copy(); rel[:,:4]=np.log(raw[:,:4])-np.log(raw[-1,3]);rel[:,4]=np.log1p(raw[:,4])
    scaler=row['scaler']; x=((rel-np.asarray(scaler['mean']))/np.asarray(scaler['std'])).astype('float32')[None]
    r=np.diff(np.log(raw[:,3])); A=np.asarray([np.sum((r-r.mean())**2)/7.0],dtype='float64')
    opt=ort.SessionOptions();opt.intra_op_num_threads=1;opt.inter_op_num_threads=1
    opt.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    baseline=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    start=time.perf_counter();sess=ort.InferenceSession(row['model_path'],opt,providers=['CPUExecutionProvider']); load=time.perf_counter()-start
    feeds={row['input_name']:x}
    if row.get('floor_input_name'):feeds[row['floor_input_name']]=A
    names=[i.name for i in sess.get_inputs()]; assert set(feeds)==set(names),(row['model'],names)
    def run():return sess.run(None,feeds)
    first_start=time.perf_counter();pred=run();first=time.perf_counter()-first_start
    for _ in range(10):run()
    values=[]
    for _ in range(100):
        start=time.perf_counter_ns();run();values.append((time.perf_counter_ns()-start)/1e6)
    p=float(pred[0].reshape(-1)[0])*row['output_scale']; assert np.isfinite(p) and p>=0
    return dict(model=row['model'],profile=row['profile'],model_sha256=row['model_sha256'],model_bytes=Path(row['model_path']).stat().st_size,metadata_sha256=row['metadata_sha256'],cpu=cpu,providers=sess.get_providers(),ort_threads=1,batch_size=1,warmups=10,timed_calls=100,session_load_ms=load*1000,first_call_ms=first*1000,median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),min_ms=min(values),max_ms=max(values),calls_ms=values,process_baseline_RSS_bytes=baseline,process_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,prediction_sigma7=p,software={'python':platform.python_version(),'numpy':np.__version__,'onnxruntime':ort.__version__},new_model_fits=0,new_calibrations=0)

def main():
    p=argparse.ArgumentParser();p.add_argument('--registry',type=Path,required=True);p.add_argument('--input-json',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--child',type=int);p.add_argument('--cpu',type=int);a=p.parse_args()
    registry=json.loads(a.registry.read_text());inp=json.loads(a.input_json.read_text());rows=registry['models']
    if a.child is not None:
        print(json.dumps(child(rows[a.child],inp['raw_window'],a.cpu),allow_nan=False));return
    a.output.mkdir(parents=True,exist_ok=True);cpu=min(os.sched_getaffinity(0)); results=[]
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    for i,row in enumerate(rows):
        cmd=[sys.executable,str(Path(__file__).resolve()),'--registry',str(a.registry.resolve()),'--input-json',str(a.input_json.resolve()),'--output',str(a.output.resolve()),'--child',str(i),'--cpu',str(cpu)]
        proc=subprocess.run(cmd,capture_output=True,text=True,env=env,timeout=90)
        (a.output/(str(i).zfill(2)+'_stderr.txt')).write_text(proc.stderr)
        if proc.returncode:raise RuntimeError(row['model']+': '+proc.stderr[-1800:])
        result=json.loads(proc.stdout);results.append(result);dump(a.output/(str(i).zfill(2)+'_measurement.json'),result)
        print(json.dumps({k:result[k] for k in ['model','median_ms','p95_ms','process_peak_RSS_bytes']}),flush=True)
    report={'kind':'sequential_ORТ_CPU_only_comparison','AI_assistance_declared':True,'started_after_export_completion':True,'registry_sha256':sha(a.registry),'input_json_sha256':sha(a.input_json),'input_source':inp['provenance'],'cpu_logical_id':cpu,'cpu_model':next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),None),'software':{'python':platform.python_version(),'platform':platform.platform()},'models':results,'scope':'one identical raw seven-bar window, model-specific frozen scalers, process-per-model; no Torch imported','memory_scope':'total ORT-only process high-water RSS; baseline separately reported, not an incremental memory estimate','excluded':'HTTP/browser/Dash, explanations, concurrent sessions, GPU training, FLOPs, energy','filesystem_cache':'not flushed; load observations are not guaranteed cold loads','new_model_fits':0,'new_calibrations':0,'scientific_confirmation':False}
    dump(a.output/'receipt.json',report)
    fields=['model','profile','model_bytes','session_load_ms','first_call_ms','median_ms','p95_ms','process_baseline_RSS_bytes','process_peak_RSS_bytes','prediction_sigma7']
    with (a.output/'costs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in results)
if __name__=='__main__':main()
