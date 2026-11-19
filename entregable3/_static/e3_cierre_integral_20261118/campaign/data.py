"""Common chronological geometry; reserved additional data stay sealed by default."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def propose_split_rows(raw_rows, ratios=(.55, .15, .10), purge=7):
    if type(raw_rows) is not int or raw_rows < 120 or purge != 7:
        raise ValueError('At least 120 chronological bars and a seven-bar support purge are required')
    if len(ratios)!=3 or any(not np.isfinite(r) or r<=0 for r in ratios) or sum(ratios)>=1:
        raise ValueError('Three positive fit/validation/calibration ratios summing below one required')
    n = raw_rows - 7
    sizes = [int(n * x) for x in ratios]
    sizes.append(n - sum(sizes) - 3 * purge)
    if min(sizes) < 10:
        raise ValueError('Proposed partitions are too small')
    parts, start = {}, 0
    for label, size in zip(('fit','validation','calibration','test'), sizes):
        parts[label] = {'window_start':start,'window_stop_exclusive':start+size,'count':size}
        start += size + purge
    return {'raw_bars':raw_rows,'window_targets':n,'purged_window_targets':3*purge,
            'purge_bars':purge,'parts':parts,'target_values_computed':False,
            'scope':'Count-only proposal; no freeze, power or model evaluation.'}


def windows(raw):
    """Caller-authorized in-memory OHLCV only; target requires eight closes."""
    raw=np.asarray(raw,dtype=np.float64)
    if raw.ndim!=2 or raw.shape[1]!=5 or len(raw)<8 or not np.isfinite(raw).all():
        raise ValueError('Expected finite N×5 OHLCV')
    if (raw[:,:4]<=0).any() or (raw[:,4]<0).any():raise ValueError('Invalid prices or volume')
    if (raw[:,1]<np.maximum(raw[:,0],raw[:,3])).any() or (raw[:,2]>np.minimum(raw[:,0],raw[:,3])).any():
        raise ValueError('Incoherent OHLC')
    X=np.stack([raw[t-7:t] for t in range(7,len(raw))])
    closes=np.log(raw[:,3]);returns=np.diff(closes)
    Y=np.array([np.std(returns[t-7:t],ddof=0) for t in range(7,len(raw))])
    known=np.diff(np.log(X[:,:,3]),axis=1);mean=known.mean(axis=1)
    A=((known-mean[:,None])**2).sum(axis=1)/7
    U=np.abs(returns[6:]-mean)
    if not np.allclose(Y**2,A+(6/49)*U**2,rtol=1e-11,atol=1e-15):
        raise ValueError('Variance identity does not match the supplied target')
    return {'raw_X':X,'y':Y,'A':A,'U':U}


def training_scaler(panel, fit_positions):
    raw=panel['raw_X'];fit=np.asarray(fit_positions,dtype=int)
    if fit.ndim!=1 or len(fit)==0 or len(np.unique(fit))!=len(fit) or (fit<0).any() or (fit>=len(raw)).any():
        raise ValueError('Nonempty distinct in-bounds training positions required')
    relative=np.empty_like(raw)
    relative[:,:,:4]=np.log(raw[:,:,:4])-np.log(raw[:,-1,3])[:,None,None]
    relative[:,:,4]=np.log1p(raw[:,:,4])
    mean=relative[fit].mean(axis=(0,1));sd=relative[fit].std(axis=(0,1),ddof=0)
    sd=np.where(sd==0,1.,sd)
    y_scale=float(panel['y'][fit].mean());u_scale=float(panel['U'][fit].mean())
    if y_scale<=0 or u_scale<=0:raise ValueError('Training target scales must be positive')
    return ((relative-mean)/sd).astype(np.float32),{
        'mean':mean.tolist(),'std':sd.tolist(),'y_scale':y_scale,'u_scale':u_scale,
        'relative_window_anchor':'last_known_close','fit_positions':fit.tolist(),
        'fitted_on_training_only':True,'volume_transform':'log1p; source units retained per dataset'}


def build_reserved_dataset(admission_path,dataset_id,*,frozen_protocol):
    """Not called by the development pilot. No implicit access to confirmation."""
    if not isinstance(frozen_protocol,dict) or frozen_protocol.get('stage')!='confirmatory_frozen':
        raise ValueError('Reserved datasets cannot be built before a final protocol freeze')
    if frozen_protocol.get('power_dimensioned') is not True or not frozen_protocol.get('annotated_protocol_tag'):
        raise ValueError('Final prospective power and annotated protocol tag are required')
    path=Path(admission_path);receipt=json.loads(path.read_text())
    row=next(x for x in receipt['datasets'] if x['id']==dataset_id)
    if row.get('admitted') is not True or receipt['datasets_admitted']!=3:
        raise ValueError('All three physical data admissions are required')
    source=(path.parent/row['path']).resolve()
    if not source.is_relative_to(path.parent.resolve()):raise ValueError('Escaping CSV path')
    raw=source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('Admitted bytes changed')
    records=list(csv.DictReader(raw.decode('utf-8').splitlines()))
    if list(records[0])!=['date','open','high','low','close','volume']:raise ValueError('CSV schema differs')
    result=windows([[float(r[k]) for k in ('open','high','low','close','volume')] for r in records])
    result['dates']=np.asarray([r['date'] for r in records[7:]])
    result['admission']=row
    return result
