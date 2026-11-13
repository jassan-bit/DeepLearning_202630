"""Reconstruct only the original, SHA-bound daily cache after a Colab loss.

No fitting, model inference, scoring, new first-access marker, or new original
admission receipt is produced. The caller owns the bounded restoration lease.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
from urllib.request import urlopen

ENGINE_COMMIT='1e8745497b484518f766b6da5c19be8362d9add7'
FREEZE_SHA='c0e9e2589dca9fa03b341bb72f87d8b541d33b4721715622daeae4438dbd60b4'
LOADER_SHA='693842a92386cd3c49ed3e10a9296428e900010df7ae7db373a80d625e851867'
ADMISSION_SHA='1d430f16a4d363055800fb4ec62cc4a7bc9062a131e3a792324b8440e5cebd9c'
CACHE_SHA='9563ef557e180f34ee37dce3b1480c38aaefd7063df21c031644973310e28deb'
PANDAS_VERSION='2.2.3'
MONTHS=tuple([f'2025-{m:02d}' for m in range(1,13)]+[f'2026-{m:02d}' for m in range(1,9)])
VERSION='e2-original-market-cache-restoration-v1'

def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def regular(path):
    path=Path(path).absolute()
    if any(p.is_symlink() for p in (path,*path.parents)):raise ValueError('Restoration paths must not use symlinks')
    return path
def pinned(path,expected):
    path=regular(path)
    if not path.is_file():raise ValueError('Missing original restoration input: '+str(path))
    raw=path.read_bytes()
    if sha(raw)!=expected:raise ValueError('Original restoration input SHA256 differs: '+str(path))
    return raw

def load_original(source):
    sys.path.insert(0,str(source/'scripts'))
    def load(filename,name):
        path=source/'scripts'/filename
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        return module
    loader=load('e2_reduced_final_data.py','_e2_fresh_cache_original_loader')
    aggregate=load('load_filipe_reserved_test.py','_e2_fresh_cache_original_aggregator')
    return loader,aggregate.aggregate_month_payload

def gate(source_root,original_output):
    source,output=regular(source_root),regular(original_output)
    head=subprocess.check_output(['git','--no-optional-locks','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if head!=ENGINE_COMMIT:raise ValueError('Restore the original scientific source commit')
    dirty=subprocess.run(['git','--no-optional-locks','-C',str(source),'diff','--no-ext-diff','--quiet',ENGINE_COMMIT,'--','scripts'],
                         stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if dirty.returncode:raise ValueError('Original scientific source scripts changed')
    pinned(source/'scripts/e2_reduced_final_data.py',LOADER_SHA)
    freeze_raw=pinned(output/'final/final_freeze.json',FREEZE_SHA)
    admission_raw=pinned(output/'final/score-worker-source-recovery-v1/scientific_outputs/data_admission.json',ADMISSION_SHA)
    freeze,admission=json.loads(freeze_raw),json.loads(admission_raw)
    loader,aggregate=load_original(source)
    checked=loader.validate_final_freeze(freeze,output/'final')
    reservation=loader.reserve_known_e1_evaluation(source)
    if (admission.get('freeze_sha256')!=checked['freeze_sha256'] or admission.get('reservation_sha256')!=reservation['sha256']
            or admission.get('daily_market_sha256')!=CACHE_SHA):
        raise ValueError('Original cache admission does not match the frozen predictors and archive reservation')
    records=reservation.get('archives',[])
    if len(records)!=20 or [r.get('archive') for r in records]!=['BTCUSDT-1m-'+m+'.zip' for m in MONTHS]:
        raise ValueError('Exactly the original twenty monthly archives in chronological order are required')
    return source,output,loader,aggregate,records,admission_raw,reservation

def atomic_exact(path,raw,expected):
    path=regular(path)
    if sha(raw)!=expected:raise ValueError('Reconstructed cache differs from the original SHA256')
    if path.exists():
        if not path.is_file() or path.read_bytes()!=raw:raise ValueError('Existing restoration target differs; preserve it for review')
        return False
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.original-cache-restore-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        # Link is exclusive: another writer can never be overwritten.
        try:os.link(name,path)
        except FileExistsError:
            if path.read_bytes()!=raw:raise ValueError('Restoration target changed concurrently')
    finally:Path(name).unlink(missing_ok=True)
    return True

def rebuild_original_cache(source_root,original_output,parallel=4,opener=None):
    if type(parallel) is not int or not 1<=parallel<=4:raise ValueError('Use one to four bounded archive download threads')
    source,output,loader,aggregate,records,admission_raw,reservation=gate(source_root,original_output)
    cache=regular(output/'final/private-known-E1-data/known_E1_daily.csv')
    receipt=regular(cache.parent/'known_E1_data_receipt.json')
    if receipt.exists() and receipt.read_bytes()!=admission_raw:raise ValueError('Existing original cache receipt differs')
    reused=cache.exists()
    if reused:pinned(cache,CACHE_SHA)
    downloads=[]
    if not reused:
        import pandas as pd
        if pd.__version__!=PANDAS_VERSION:raise ValueError('Exact original Pandas '+PANDAS_VERSION+' is required for cache bytes')
        raw_root=regular(output.parent/'private_raw_E1');open_archive=urlopen if opener is None else opener
        def fetch(row):
            relative=PurePosixPath(row['relative_path'])
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe reserved archive path')
            target=regular(raw_root/str(relative));had=target.exists()
            raw=loader._download_archive(row,target,open_archive)
            if len(raw)!=row['bytes'] or sha(raw)!=row['sha256']:raise ValueError('Reserved archive bytes changed')
            return {'archive':row['archive'],'sha256':row['sha256'],'bytes':len(raw),'existing_raw_reused':had}
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures={pool.submit(fetch,r):r for r in records}
            for future in as_completed(futures):
                downloads.append(future.result())
        frames=[]
        for month,row in zip(MONTHS,records):
            raw=pinned(raw_root/row['relative_path'],row['sha256'])
            frame,_=aggregate(raw,month,expected_sha256=row['sha256']);frames.append(frame)
        evaluation=pd.concat(frames)
        evaluation.index=pd.DatetimeIndex(pd.to_datetime(evaluation.index,utc=True))
        expected=pd.date_range('2025-01-01','2026-08-31',tz='UTC')
        if not evaluation.index.equals(expected):raise ValueError('Reconstructed original daily calendar differs')
        evaluation.index.name='market_day_utc'
        # This is exactly the original loader's to_csv policy. Validate bytes
        # before any final target is created, never normalize or force its hash.
        stream=io.StringIO(newline='');evaluation.to_csv(stream)
        raw=stream.getvalue().encode()
        atomic_exact(cache,raw,CACHE_SHA)
    atomic_exact(receipt,admission_raw,ADMISSION_SHA)
    evidence={'version':VERSION,'status':'original_daily_cache_restored','observed_at_utc':datetime.now(timezone.utc).isoformat(),
        'scientific_commit':ENGINE_COMMIT,'freeze_file_sha256':FREEZE_SHA,'original_admission_sha256':ADMISSION_SHA,
        'cache_sha256':CACHE_SHA,'reservation_sha256':reservation['sha256'],'cache_path':str(cache),
        'original_receipt_path':str(receipt),'existing_daily_cache_reused':reused,
        'archives':sorted(downloads,key=lambda r:r['archive']),'parallel_download_limit':parallel,
        'original_admission_bytes_preserved':True,'new_first_access_marker_created':False,
        'market_payload_bytes_opened':True,'numerical_market_values_parsed':not reused,'new_blind_test_claimed':False,
        'scientific_fit_performed':False,'model_inference_performed':False,'benchmark_recomputed':False,'budget_reset':False}
    proof_root=regular(output.parent/'fresh-session-cache-restoration-v1');proof_root.mkdir(parents=True,exist_ok=True)
    encoded=canonical(evidence)+b'\n';proof=proof_root/(sha(encoded)+'.json');atomic_exact(proof,encoded,sha(encoded))
    return {**evidence,'restoration_receipt_path':str(proof)}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--original-output',type=Path,required=True)
    parser.add_argument('--parallel',type=int,default=4)
    args=parser.parse_args()
    print(json.dumps(rebuild_original_cache(args.source_root,args.original_output,parallel=args.parallel),ensure_ascii=False,indent=2))
