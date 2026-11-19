"""Acquire fixed legacy BTC daily history, preserving calendar gaps.

This auxiliary D00 history is not an additional dataset admission. It can
reconstruct known historical sigma7 labels, but never trains a model or
authorizes a predictor to consume observations at or after its forecast date.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import urllib.request

SOURCE_COMMIT = '18429f618a4540f2973dc397e6fbcb8808f2b300'
NPZ_SHA256 = '186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683'
NPZ_COUNTS = {'training':655,'calibration':53,'evaluation':608}
FIELDS = ('open','high','low','close','volume')
SOURCES = (
    {'file':'daily_market_development.csv','path':'results/eda_development/daily_market_development.csv',
     'sha256':'bf6bb5160af924aef6a15c3684aeae932df3017e8639888bb7e91ce787900171',
     'bytes':60427,'rows':731,'start':'2023-01-01','end':'2024-12-31',
     'header':('market_day_utc',*FIELDS,'trades')},
    {'file':'daily_market_evaluation.csv',
     'path':'book_e2/_static/e2_scientific_closure_20261116/data_admission/known_E1_daily.csv',
     'sha256':'9563ef557e180f34ee37dce3b1480c38aaefd7063df21c031644973310e28deb',
     'bytes':45135,'rows':608,'start':'2025-01-01','end':'2026-08-31',
     'header':('market_day_utc',*FIELDS)},
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _days(start, end):
    start,end=date.fromisoformat(start),date.fromisoformat(end)
    return [(start+timedelta(days=i)).isoformat() for i in range((end-start).days+1)]


def _directory(path):
    path=Path(path).absolute()
    if any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('History source/output cannot traverse a symlink')
    path.mkdir(parents=True,exist_ok=True)
    return path


def acquire(directory, *, opener=urllib.request.urlopen):
    """Return both source paths and metadata; exact cached bytes are reusable."""
    directory=_directory(directory);paths=[];metadata=[]
    for source in SOURCES:
        path=directory/source['file']
        url=('https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/'
             +SOURCE_COMMIT+'/'+source['path'])
        cached=path.is_file()
        if path.is_symlink():raise ValueError('History cache symlink refused')
        if cached:
            raw=path.read_bytes()
        else:
            with opener(url,timeout=45) as response:raw=response.read(131073)
        if len(raw)!=source['bytes'] or sha(raw)!=source['sha256']:
            raise ValueError('Pinned D00 daily source differs: '+source['file'])
        if not cached:
            with path.open('xb') as stream:stream.write(raw)
        paths.append(path)
        metadata.append({'file':source['file'],'bytes':len(raw),'sha256':sha(raw),
                         'url':url,'cached':cached,'source_commit':SOURCE_COMMIT})
    return tuple(paths),metadata


def parse_daily(raw, source):
    reader=csv.DictReader(io.StringIO(raw.decode('utf-8-sig'),newline=''))
    if tuple(reader.fieldnames or ())!=source['header']:
        raise ValueError('D00 daily source header differs')
    records=[]
    for row in reader:
        stamp=row['market_day_utc']
        day=date.fromisoformat(stamp[:10]).isoformat()
        if stamp != day+' 00:00:00+00:00':raise ValueError('D00 day is not canonical UTC midnight')
        values=[row[name].strip() for name in FIELDS]
        if not any(values):
            if any(row[name].strip() for name in source['header'][1:]):
                raise ValueError('D00 partially missing OHLCV row')
            record={'date':day,'values':None,'missing':True}
        else:
            if not all(values):raise ValueError('D00 partially missing OHLCV row')
            numbers=tuple(float(value) for value in values)
            op,high,low,close,volume=numbers
            if (not all(math.isfinite(value) for value in numbers)
                    or min(op,high,low,close)<=0 or volume<0
                    or low>min(op,close) or high<max(op,close) or high<low):
                raise ValueError('D00 daily OHLCV geometry or finiteness invalid')
            record={'date':day,'values':numbers,'missing':False}
        records.append(record)
    if (len(records)!=source['rows']
            or [r['date'] for r in records]!=_days(source['start'],source['end'])):
        raise ValueError('D00 daily calendar is incomplete, duplicated or reordered')
    return records


def history(paths):
    if len(paths)!=2:raise ValueError('Both development and evaluation daily sources are required')
    records=[]
    for source,path in zip(SOURCES,paths):
        path=Path(path)
        if path.is_symlink() or not path.is_file():raise ValueError('Expected regular D00 source')
        raw=path.read_bytes()
        if len(raw)!=source['bytes'] or sha(raw)!=source['sha256']:
            raise ValueError('Pinned history size or hash mismatch')
        records.extend(parse_daily(raw,source))
    if [r['date'] for r in records]!=_days('2023-01-01','2026-08-31'):
        raise ValueError('The combined BTC daily calendar has a gap or overlap')
    return records


def sigma7(records):
    """Known sigma for each daily label; missing/insufficient support stays None."""
    result={}
    for i,row in enumerate(records):
        support=records[i-7:i+1] if i>=7 else []
        value=None
        if len(support)==8 and all(r['values'] is not None for r in support):
            expected=_days(support[0]['date'],support[-1]['date'])
            if [r['date'] for r in support]!=expected or len(expected)!=8:
                raise ValueError('Sigma support must be eight consecutive calendar days')
            closes=[r['values'][3] for r in support]
            returns=[math.log(closes[k])-math.log(closes[k-1]) for k in range(1,8)]
            mean=math.fsum(returns)/7
            value=math.sqrt(math.fsum((v-mean)**2 for v in returns)/7)
        result[row['date']]=value
    return result


def verify_npz(records, npz_path):
    """Compare known legacy labels/windows, not any additional dataset labels."""
    import numpy as np
    raw=Path(npz_path).read_bytes()
    if sha(raw)!=NPZ_SHA256:raise ValueError('Known cohort NPZ hash differs')
    positions={r['date']:i for i,r in enumerate(records)};labels=sigma7(records);reports={}
    with np.load(io.BytesIO(raw),allow_pickle=False) as source:
        for split,count in NPZ_COUNTS.items():
            dates=[str(x) for x in source[split+'_dates']]
            y=np.asarray(source[split+'_y_original'],dtype=np.float64)
            windows=np.asarray(source[split+'_raw'],dtype=np.float64)
            if len(dates)!=count or y.shape!=(count,) or windows.shape!=(count,7,5):
                raise ValueError('Known cohort geometry differs')
            reconstructed=[];reconstructed_windows=[]
            for day in dates:
                if day not in positions or labels[day] is None:
                    raise ValueError('A recorded cohort target has incomplete original daily support')
                i=positions[day]
                reconstructed.append(labels[day])
                reconstructed_windows.append([r['values'] for r in records[i-7:i]])
            error=float(np.max(np.abs(np.asarray(reconstructed)-y)))
            actual_windows=np.asarray(reconstructed_windows)
            window_error=float(np.max(np.abs(actual_windows-windows)))
            prices_equal=np.array_equal(actual_windows[:,:,:4],windows[:,:,:4])
            volume_error=np.abs(actual_windows[:,:,4]-windows[:,:,4])
            volume_ulp=float(np.max(volume_error/np.spacing(np.maximum(np.abs(windows[:,:,4]),1.))))
            # These fixed CSV and NPZ bytes are independently hash-bound. Their
            # evaluation-volume decimal parse differs by at most one float64
            # ULP; all four price fields agree exactly. Preserve/report that
            # conversion effect instead of changing a source or rounding it.
            if (not math.isfinite(error) or error>1e-12 or not prices_equal
                    or not math.isfinite(volume_ulp) or volume_ulp>1.):
                raise ValueError('D00 daily history fails known cohort label/input parity')
            reports[split]={'dates':count,'first':dates[0],'last':dates[-1],
                            'maximum_absolute_sigma7_error':error,
                            'maximum_absolute_input_error':window_error,
                            'raw_price_fields_bit_identical':prices_equal,
                            'maximum_volume_float64_ULP_difference':volume_ulp,
                            'all_OHLCV_values_bit_identical':window_error==0.,
                            'volume_used_by_classical_target_history':False}
    return {'npz_sha256':NPZ_SHA256,'partitions':reports,
            'known_targets_compared':sum(NPZ_COUNTS.values()),
            'additional_dataset_targets_opened':False,'model_fits':0,'model_predictions':0}


def run(output, *, npz=None):
    folder=_directory(output);paths,metadata=acquire(folder);records=history(paths);labels=sigma7(records)
    report={'kind':'e3_auxiliary_known_BTC_daily_history',
            'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
            'source_commit':SOURCE_COMMIT,'files':metadata,
            'calendar_rows':len(records),'first':records[0]['date'],'last':records[-1]['date'],
            'missing_OHLCV_dates':[r['date'] for r in records if r['missing']],
            'rows_filled':0,'rows_removed':0,
            'sigma7_available_dates':sum(v is not None for v in labels.values()),
            'sigma7_unavailable_dates':[d for d,v in labels.items() if v is None],
            'additional_dataset_admission_count_contribution':0,
            'known_historical_sigma7_reconstruction':True,
            'additional_dataset_targets_opened':False,
            'model_fits':0,'model_predictions':0,'R1_accredited':False,'R3_accredited':False,
            'scope':'Auxiliary D00 history. A causal predictor must restrict access to days before each forecast target.',
            'rights':'Original E1/E2 provenance and rights retained; no retroactive application of new Vision terms or new licence claim.',
            'AI_assistance_declared':True,'implementation_sha256':sha(Path(__file__).read_bytes())}
    if npz is not None:report['cohort_parity']=verify_npz(records,npz)
    (folder/'d00_history_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return tuple(paths),report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--npz',type=Path)
    args=parser.parse_args()
    _,report=run(args.output,npz=args.npz)
    print(json.dumps({'calendar_rows':report['calendar_rows'],
                      'missing_dates':report['missing_OHLCV_dates'],
                      'cohort_parity':report.get('cohort_parity'),'model_fits':0},ensure_ascii=False))
