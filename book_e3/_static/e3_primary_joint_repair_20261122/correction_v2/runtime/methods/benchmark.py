"""Timing only on synthetic fixtures; never selects a statistical outcome."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import primary_rmse as method


def benchmark(n=768,B=8191,reference_reps=49999,lags=48,block=32):
    rng=np.random.default_rng(202611220987654)
    models,seeds=21,10
    z=np.column_stack([np.ones(n),.6+.2*np.sin(np.arange(n)/17)])
    one=np.exp(rng.normal(0,.25,(models,seeds,n))+.15*np.sin(np.arange(n)/11)[None,None,:])
    base=np.exp(rng.normal(0,.25,(3,1,n)))
    shared=np.concatenate([base[:2],np.repeat(base[2:],19,axis=0)],axis=0)
    shared=np.repeat(shared,10,axis=1)
    risk=np.array([[.3,1.7],[.01,4.],[.7,2.3]])[:,:,None]
    hetero=np.exp(rng.normal(0,.25,(3,2,n)))*risk
    hetero=np.concatenate([hetero[:2],np.repeat(hetero[2:],19,axis=0)],axis=0)
    hetero=np.repeat(hetero,5,axis=1)
    measurements=[]
    for label,losses in [('full_distinct',one),('exact_rival_duplicates_seed_R1',shared),('exact_rival_duplicates_two_seed_groups_R2',hetero)]:
        cpu=time.process_time();wall=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
        output=method.analyze_losses(losses,list(method.EXPECTED_SYSTEMS),bootstrap_reps=B,block_length=block,
            hac_lags=lags,instruments=z,reference_reps=reference_reps,reference_seed=2026112202,seed=202611220987655)
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        measurements.append(dict(fixture=label,wall_seconds=time.monotonic()-wall,parent_cpu_seconds=time.process_time()-cpu,
            child_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
            backend=output['bootstrap_backend'],reduction=output['exact_bootstrap_reduction']))
        # No p-value, rejection count, interval or calibration criterion is inspected.
        del output
    return dict(kind='SOFTWARE_THROUGHPUT_BENCHMARK_NOT_CALIBRATION',created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        dimensions=[models,seeds,n],B=B,block_length=block,hac_lags=lags,reference_reps=reference_reps,
        measurements=measurements,benchmark_seed=202611220987654,
        outputs_inspected='timing, input identity, backend and exact compression metadata only',
        methods_sha256=hashlib.sha256(Path(method.__file__).read_bytes()).hexdigest(),
        kernel_sha256=hashlib.sha256(Path(method.__file__).with_name('primary_bootstrap.cpp').read_bytes()).hexdigest(),
        new_market_fits=0,new_formal_calibration_replicates=0)


if __name__=='__main__':
    report=benchmark()
    output=Path(__file__).with_name('THROUGHPUT_BENCHMARK_EXACT_COMPRESSION.json')
    if output.exists():raise RuntimeError('Preserve the prior timing attempt and use a new output version')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
