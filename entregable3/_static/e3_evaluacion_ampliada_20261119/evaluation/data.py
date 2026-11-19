"""Admitted chronological panels; fitting statistics never use reserved rows."""
from __future__ import annotations
import csv
from datetime import date,timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
NPZ_SHA='186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683'
FIELDS=('date','open','high','low','close','volume')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def array_sha(x):
    x=np.ascontiguousarray(x)
    return hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes()).hexdigest()

def old_geometry():
    p=ROOT/'experiments/e3_cierre_integral_20261118/campaign/data.py'
    s=importlib.util.spec_from_file_location('e3_extended_geometry',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def _relative(raw,scaler):
    result=np.empty_like(raw,dtype=np.float64)
    result[:,:,:4]=np.log(raw[:,:,:4])-np.log(raw[:,-1,3])[:,None,None]
    result[:,:,4]=np.log1p(raw[:,:,4])
    result=(result-np.asarray(scaler['mean']))/np.asarray(scaler['std'])
    if not np.isfinite(result).all():raise ValueError('Nonfinite transformed windows')
    return result.astype(np.float32)

def package_panel(raw_X,y,A,U,dates,positions,*,dataset,raw_sha,window_dates,history,geometry):
    g=old_geometry();base={'raw_X':np.asarray(raw_X),'y':np.asarray(y),'A':np.asarray(A),'U':np.asarray(U)}
    X,scaler=g.training_scaler(base,positions['fit'])
    floor=np.sqrt(base['A']);scaler['d_fit']=float(np.mean(base['y'][positions['fit']]-floor[positions['fit']]))
    result={'dataset':dataset,'raw_sha256':raw_sha,'scaler':scaler,'positions':positions,
            'raw_X':base['raw_X'],'all_y':base['y'],'all_A':base['A'],'all_U':base['U'],
            'dates':np.asarray(dates,dtype='U10'),'window_dates':np.asarray(window_dates,dtype='U10'),
            'history':history,'geometry':geometry}
    for label,indexes in positions.items():
        prefix='train' if label=='fit' else ('val' if label=='validation' else label)
        for key,values in [('X',X),('raw_X',base['raw_X']),('A',base['A']),('U',base['U']),('y_original',base['y']),('dates',result['dates']),('window_dates',result['window_dates']),('floor',floor)]:
            result[prefix+'_'+key]=values[indexes]
    result['admission']={'arrays_sha256':{'train_X':array_sha(result['train_X'])},
                         'n_fit':len(positions['fit']),'n_validation':len(positions['validation']),
                         'scaler_fit_only':True,'history_clock':geometry['clock']}
    return result

def read_admitted(admission_path,dataset,expected):
    path=Path(admission_path);receipt=json.loads(path.read_bytes())
    if receipt.get('datasets_admitted')!=3:raise ValueError('All three physical admissions must pass')
    rows=[r for r in receipt['datasets'] if r['id']==dataset]
    if len(rows)!=1 or rows[0].get('admitted') is not True:raise ValueError('Unadmitted dataset')
    row=rows[0];p=(path.parent/row['path']).resolve()
    if not p.is_relative_to(path.parent.resolve()) or p.is_symlink():raise ValueError('Unsafe admitted path')
    if row['sha256']!=expected or sha(p)!=expected:raise ValueError('Dataset digest differs from frozen protocol')
    records=list(csv.DictReader(p.open(newline='')))
    if not records or tuple(records[0])!=FIELDS:raise ValueError('Normalized CSV schema differs')
    dates=[r['date'] for r in records]
    if dates!=sorted(set(dates)):raise ValueError('Dataset chronology changed')
    raw=np.asarray([[float(r[k]) for k in FIELDS[1:]] for r in records])
    return raw,dates,row

def additional(admission_path,dataset,protocol,*,evaluation=True):
    raw,dates,row=read_admitted(admission_path,dataset,protocol['datasets'][dataset])
    split=old_geometry().propose_split_rows(len(raw),tuple(protocol['split']['ratios']),protocol['split']['purge'])
    positions={k:np.arange(v['window_start'],v['window_stop_exclusive']) for k,v in split['parts'].items()}
    if not evaluation:
        last=int(positions['validation'][-1]);raw=raw[:last+8];dates=dates[:last+8]
        positions={k:v for k,v in positions.items() if k in ('fit','validation')}
    panel=old_geometry().windows(raw)
    ydates=np.asarray(dates[7:],dtype='U10')
    history={str(d):float(y) for d,y in zip(ydates,panel['y'])}
    window_dates=np.stack([dates[t-7:t] for t in range(7,len(dates))])
    return package_panel(panel['raw_X'],panel['y'],panel['A'],panel['U'],ydates,positions,
        dataset=dataset,raw_sha=row['sha256'],window_dates=window_dates,history=history,
        geometry={**split,'clock':'session' if dataset=='D01' else 'UTC_daily','units':row.get('volume_units','native'),
                  'reserved_target_values_computed':evaluation,'D01_RAW_corporate_actions_retained':dataset=='D01'})

def btc_history(paths,expected_hashes):
    if len(paths)!=2 or sorted(sha(p) for p in paths)!=sorted(expected_hashes):
        raise ValueError('Both pinned daily BTC source files are required for classical causal history')
    closes={}
    for path in paths:
        for r in csv.DictReader(Path(path).open(newline='')):
            day=(r.get('market_day_utc') or r.get('date') or '')[:10]
            date.fromisoformat(day)
            value=float(r['close'])
            if not np.isfinite(value) or value<=0 or ('complete' in r and float(r['complete'])!=1):continue
            if day in closes and closes[day]!=value:raise ValueError('BTC daily sources disagree')
            closes[day]=value
    history={}
    for day in sorted(closes):
        end=date.fromisoformat(day);support=[(end-timedelta(days=j)).isoformat() for j in range(7,-1,-1)]
        if all(d in closes for d in support):history[day]=float(np.std(np.diff(np.log([closes[d] for d in support])),ddof=0))
    return history

def btc(npz_path,protocol,*,evaluation=True,history_paths=()):
    if sha(npz_path)!=NPZ_SHA or protocol['datasets']['D00']!=NPZ_SHA:raise ValueError('Pinned BTC NPZ digest differs')
    labels=['training']+(['calibration','evaluation'] if evaluation else [])
    with np.load(npz_path,allow_pickle=False) as source:
        arrays={label:{k:np.asarray(source[label+'_'+k]).copy() for k in ('raw','y_original','dates')} for label in labels}
    raw=np.concatenate([arrays[k]['raw'] for k in labels]);y=np.concatenate([arrays[k]['y_original'] for k in labels]);dates=np.concatenate([arrays[k]['dates'] for k in labels]).astype('U10')
    if dates.tolist()!=sorted(set(dates.tolist())):raise ValueError('BTC chronology differs')
    known=np.diff(np.log(raw[:,:,3]),axis=1);mean=known.mean(1);A=((known-mean[:,None])**2).sum(1)/7
    margin=y*y-A
    if (margin < -1e-12).any():raise ValueError('BTC target violates observed variance support')
    U=np.sqrt(np.maximum(margin,0)/(6/49))
    positions={'fit':np.flatnonzero(dates<='2024-06-30'),
               'validation':np.flatnonzero((dates>='2024-07-08')&(dates<='2024-10-31'))}
    if evaluation:
        positions.update(calibration=np.flatnonzero((dates>='2024-11-08')&(dates<='2024-12-30')),
                         test=np.flatnonzero(dates>='2025-01-01'))
        if [len(positions[k]) for k in positions]!=[532,116,53,608]:raise ValueError('BTC frozen cohorts differ')
        history=btc_history(history_paths,protocol.get('auxiliary_sources',[]))
        for day,value in zip(dates,y):
            if day in history and abs(history[day]-value)>1e-12:raise ValueError('BTC reconstructed history differs from supplied targets')
            history[str(day)]=float(value)
    else:history={str(d):float(v) for d,v in zip(dates,y)}
    wd=np.asarray([[(date.fromisoformat(str(d))-timedelta(days=k)).isoformat() for k in range(7,0,-1)] for d in dates])
    return package_panel(raw,y,A,U,dates,positions,dataset='D00',raw_sha=NPZ_SHA,
        window_dates=wd,history=history,geometry={'clock':'UTC_daily','units':'BTC',
            'reserved_target_values_computed':evaluation,'calendar_already_exposed':True})
