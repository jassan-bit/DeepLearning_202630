"""Observation-clock controls with explicit NaN calendar states and durable forecasts."""
from __future__ import annotations
import csv
from datetime import date,timedelta
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import time
import zipfile
import numpy as np

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)

def _axis(data):
    history=data['history'];days=sorted(history)
    if data['geometry']['clock']=='session':return days
    start,end=date.fromisoformat(days[0]),date.fromisoformat(days[-1])
    return [(start+timedelta(days=k)).isoformat() for k in range((end-start).days+1)]

def fit_diagnostic_mask(data,burn_in=7):
    axis=_axis(data);lookup={d:i for i,d in enumerate(axis)};selected=[];excluded=[]
    for n,pos in enumerate(data['positions']['fit']):
        day=str(data['dates'][pos]);i=lookup[day];reasons=[]
        if n<burn_in:reasons.append('predeclared_first_seven_fit_targets')
        for lag in (1,7):
            if i<lag or not np.isfinite(data['history'].get(axis[i-lag],np.nan)):
                reasons.append('unavailable_prior_sigma_lag_'+str(lag))
        if reasons:excluded.append({'date':day,'reasons':reasons})
        else:selected.append(int(pos))
    if not selected:raise ValueError('No common fit diagnostic dates remain')
    return np.asarray(selected,dtype=int),excluded

def _load_fit(path,identity):
    with zipfile.ZipFile(path) as z:
        if z.namelist()!=['identity.json','fit.pkl']:raise ValueError('Unexpected ARIMA fit members')
        if json.loads(z.read('identity.json'))!=identity:raise ValueError('ARIMA saved fit identity differs')
        return pickle.loads(z.read('fit.pkl'))

def _save_fit(path,result,identity):
    output=io.BytesIO()
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in [('identity.json',canonical(identity).encode()),('fit.pkl',pickle.dumps(result,protocol=5))]:
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,raw)
    temporary=Path(str(path)+'.tmp');temporary.write_bytes(output.getvalue());temporary.replace(path)

def predictions(name,data,directory,*,source_commit,request_sha256,order=(1,0,1),
                expected_statsmodels='0.14.5',per_fit_seconds=120.):
    """Return one raw forecast per available requested date, one fit for ARIMA.

    Fit diagnostics for ARIMA are conditional in-sample fitted values. All
    post-fit forecasts are written before incorporating that day's observation.
    Missing daily labels advance Kalman time with NaN, never synthetic truth.
    """
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    axis=_axis(data);lookup={d:i for i,d in enumerate(axis)};history=data['history']
    fit_end=str(data['train_dates'][-1]);cut=lookup[fit_end]
    required={str(d) for split in ('val','calibration','test') for d in data.get(split+'_dates',[])}
    diagnostic,_=fit_diagnostic_mask(data)
    fit_dates={str(data['dates'][i]) for i in diagnostic}
    output={};resource={'model':name,'real_data_fits':0,'seed':-1,'clock':data['geometry']['clock'],
                        'deterministic_single_trajectory':True,'raw_forecast_noise_invariant':True}
    if name in ('Persistence','SeasonalNaive'):
        lag=1 if name=='Persistence' else 7
        for day in sorted(fit_dates|required):
            i=lookup[day]
            raw=history.get(axis[i-lag],np.nan) if i>=lag else np.nan
            if not np.isfinite(raw):raise ValueError('Unavailable predeclared '+name+' lag for '+day)
            output[day]=float(raw)
        resource.update(lag_observations=lag,training_seconds=0.,parameters=0)
        return output,resource
    if name!='ARIMA':raise ValueError('Unknown temporal control')
    if type(per_fit_seconds) not in (int,float) or not math.isfinite(per_fit_seconds) or per_fit_seconds<=0:
        raise ValueError('Positive common ARIMA fitting opportunity budget required')
    from importlib.metadata import version
    if version('statsmodels')!=expected_statsmodels:raise ValueError('Exact statsmodels profile required')
    from statsmodels.tsa.arima.model import ARIMA
    identity={'source_commit':source_commit,'request_sha256':request_sha256,'order':list(order),
              'fit_end':fit_end,'clock':data['geometry']['clock'],'statsmodels':expected_statsmodels,
              'per_fit_seconds':float(per_fit_seconds)}
    fit_path=directory/'arima_fit.zip';marker=directory/'arima_fit_started.json';t0=time.perf_counter()
    if fit_path.exists():model=_load_fit(fit_path,identity)
    else:
        if marker.exists():raise RuntimeError('Interrupted ARIMA fit has no durable result; explicit reconciliation required')
        marker.write_text(canonical(identity)+'\n')
        endog=np.asarray([history.get(d,np.nan) for d in axis[:cut+1]],dtype=float)
        if np.isfinite(endog).sum()<90:raise ValueError('At least 90 genuine ARIMA fit labels required')
        scale=float(np.nanmean(endog))
        if scale<=0 or not np.isfinite(scale):raise ValueError('Invalid fit-only ARIMA scale')
        def budget_check(_parameters):
            if time.perf_counter()-t0>=per_fit_seconds:
                raise RuntimeError('ARIMA fitting budget expired at an optimizer iteration boundary; no replacement forecast')
        budget_check(None)
        fit=ARIMA(endog/scale,order=tuple(order),trend='c' if order[1]==0 else 'n',missing='none',
                  enforce_stationarity=True,enforce_invertibility=True).fit(
                      method_kwargs={'maxiter':200,'disp':False,'callback':budget_check})
        if not bool(fit.mle_retvals.get('converged',False)) or not np.isfinite(fit.params).all():
            raise RuntimeError('ARIMA did not converge; no replacement or row deletion')
        # Statsmodels records the operational callback in mle_settings. It is
        # irrelevant to forecasting and cannot be serialized as a local
        # function. Remove that setting only; fitted parameters/Kalman state
        # and convergence information remain untouched.
        for fitted in (fit,getattr(fit,'_results',None)):
            settings=getattr(fitted,'mle_settings',None)
            if isinstance(settings,dict):settings.pop('callback',None)
        elapsed=time.perf_counter()-t0
        model={'result':fit,'scale':scale,'training_seconds':elapsed,
               'allocated_seconds':float(per_fit_seconds),
               'deadline_overrun_seconds':max(0.,elapsed-per_fit_seconds),
               'fit_labels':len(endog),'fit_observed_labels':int(np.isfinite(endog).sum()),
               'missing_positions':np.flatnonzero(~np.isfinite(endog)).tolist()}
        _save_fit(fit_path,model,identity)
    fit=model['result'];scale=model['scale']
    fitted=np.asarray(fit.fittedvalues,dtype=float)*scale
    for day in fit_dates:output[day]=float(fitted[lookup[day]])
    log=directory/'causal_forecasts.log'
    existing={}
    if log.exists():
        for line in log.read_text().splitlines():
            row=json.loads(line)
            if row['date'] in existing:raise ValueError('Duplicate durable ARIMA forecast')
            existing[row['date']]=row
    import os
    with log.open('a') as stream:
        for day in axis[cut+1:]:
            prediction=float(np.asarray(fit.forecast(1))[0])*scale
            if not np.isfinite(prediction):raise ValueError('ARIMA forecast is nonfinite')
            row={'date':day,'raw_prediction':prediction,'written_before_observation':True}
            if day in existing:
                if existing[day]!=row:raise ValueError('Replayed causal forecast differs from durable bytes')
            else:
                stream.write(canonical(row)+'\n');stream.flush();os.fsync(stream.fileno())
            if day in required:output[day]=row['raw_prediction']
            observed=history.get(day,np.nan)
            fit=fit.extend(np.asarray([observed/scale]))
    if set(output)!=(fit_dates|required):raise ValueError('Temporal-control output dates incomplete')
    resource.update(real_data_fits=1,training_seconds=model['training_seconds'],order=list(order),
                    allocated_seconds=model['allocated_seconds'],
                    deadline_overrun_seconds=model['deadline_overrun_seconds'],
                    stop_reason='fixed_ARIMA_converged_with_iteration_boundary_budget',
                    hard_process_timeout_enforced=False,
                    parameters=len(model['result'].params),
                    fit_labels=model['fit_labels'],fit_observed_labels=model['fit_observed_labels'],
                    missing_positions=model['missing_positions'],parameters_frozen_during_filtering=True,
                    fit_diagnostics='conditional_in_sample_fitted_values',
                    fit_archive_sha256=hashlib.sha256(fit_path.read_bytes()).hexdigest())
    return output,resource
