"""Registered E2 numerical analysis, without fitting or acquiring market data.

Consumes closed private panels. The 20261106 completion amendment purges P1 to
8 Nov--30 Dec 2024 (53 dates); planning resamples to the inherited 60-date final
calendar and explicitly assumes transfer. None of the original 599a29f files are
changed. Primary hypotheses, estimand, margins and 30/30/199 families are inherited.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from hashlib import sha256
import json
import math
from pathlib import Path
import warnings

import numpy as np
from scipy import stats

from analyze_e2_benchmark import (adjust_pvalues, block_indices, clean, dm_hln,
    ljung_box_diagnostics, mean_paired_seed_ranks, paired_seed_diagnostics,
    regression_metrics, validate_dense_panel)

TOP10 = ['TSMixer','TiDE','TimeMixer','TimesNet','PatchTST','Informer','Autoformer',
         'iTransformer','Crossformer','S-Mamba']
ANCHORS = ['GRU/D7-c03','LSTM/D7-c02','TCN/D7-c02']
MODELS = TOP10+['GRU-E2','LSTM-E2','TCN-E2']+ANCHORS+['Ridge','Persistence','SeasonalNaive','ARIMA']
DETERMINISTIC = ['Ridge','Persistence','SeasonalNaive','ARIMA']
REFERENCES = ['GRU/D7-c03','GRU-E2']
DELTA = .00025772842664981106
THRESHOLD = .024247163097617024
P1_MODELS = ['TimeMixer','iTransformer','PatchTST','GRU/D7-c03','GRU-E2']
P1_SEEDS = list(range(660201,660211))
FINAL_SEEDS = list(range(670101,670301))
R_GRID = [5,6,8,10,15,20,30,40,60,80,100,150,200]
ROOT = Path(__file__).resolve().parents[1]


def _require(condition, message):
    if not condition: raise ValueError(message)


def _rows_hash(rows):
    digest=sha256()
    for row in rows:digest.update((json.dumps(clean(row),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode())
    return digest.hexdigest()


def _calendar(start, end):
    a,b=date.fromisoformat(start),date.fromisoformat(end)
    return [(a+timedelta(days=i)).isoformat() for i in range((b-a).days+1)]


def _boolean(value):
    if value in (True,1,'1','True','true'):return True
    if value in (False,0,'0','False','false'):return False
    raise ValueError('An explicit eligibility/regime boolean is required')


def _stamp(value):
    moment=datetime.fromisoformat(str(value).replace('Z','+00:00'))
    _require(moment.tzinfo is not None and moment.utcoffset()==timedelta(0),'Receipt timestamps require UTC')
    return moment


def _settings(prereg):
    _require(prereg['inherited_contract']['absolute_delta']==DELTA,'Inherited delta changed')
    contract=prereg['inherited_contract'];_require(contract['history_days']==7 and contract['horizon']==1 and contract['primary_metric']=='mean_seed_RMSE_original_scale' and contract['baseline']==REFERENCES[0],'Inherited task/estimand/baseline changed')
    _require([f['id'] for f in prereg['families']]==['historical_anchor','fair_retuned'],'Primary family IDs changed')
    for family,reference in zip(prereg['families'],REFERENCES):_require(all(h['reference']==reference and h['candidate'] in TOP10 and h['question'] in ['RQ1','RQ2','RQ3'] for h in family['hypotheses']),'Registered comparison reference/candidate changed')
    _require(prereg['temporal_inference']['primary_block_days']==7 and prereg['temporal_inference']['sensitivity_blocks']==[14,28],'Temporal bootstrap blocks changed')
    _require(prereg['alternatives_for_planning']['RQ1_gain']==2*DELTA and prereg['alternatives_for_planning']['RQ2_erosion']==2*DELTA and prereg['alternatives_for_planning']['RQ3_latency_relative_reduction']==.2,'Planning alternatives changed')
    _require([f['hypothesis_count'] for f in prereg['families']]==[30,30], 'Exactly two30-hypothesis families required')
    _require([f['alpha'] for f in prereg['families']]==[.02,.02], 'Family alpha allocation changed')
    _require(prereg['diagnostic_family']['hypothesis_count']==199 and prereg['diagnostic_family']['alpha']==.01,'Diagnostic family changed')
    _require(prereg['temporal_inference']['bootstrap_repetitions']==99999,'Final bootstrap99999 required')
    _require(prereg['power']['candidate_R']==R_GRID and prereg['power']['mc_repetitions']==10000,'Registered planning grid/draws changed')
    _require(prereg['power']['target']==.8 and prereg['generalization_gap']['delta_g']==DELTA,'Planning/overfit margin changed')
    return prereg


def _panel(rows, *, kind, fixture=False):
    rows=list(rows);_require(rows,'Empty prediction panel')
    if kind=='P1':
        models=P1_MODELS;seeds=P1_SEEDS;calendar=_calendar('2024-11-08','2024-12-30');cutoff='2024-10-31';stage='E2_P1_dimensioning';det=[]
    elif kind=='final':
        models=MODELS;calendar=_calendar('2026-09-01','2026-10-30');cutoff='2024-12-24';stage='E2_final_primary';det=DETERMINISTIC
        actual=sorted({int(r['seed']) for r in rows if r['model'] not in det})
        _require(len(actual) in R_GRID and actual==FINAL_SEEDS[:len(actual)],'Final seed registry is not the registered R prefix')
        seeds=actual
    else:raise ValueError('Unknown panel kind')
    converted=[];regimes={};closures=[];scorings=[];mase={};run_cohorts={}
    for row in rows:
        r=dict(row);origin=str(r.get('outcome_day',r.get('origin_utc','')))[:10]
        _require(r.get('source_stage')==('synthetic_fixture' if fixture else stage),'Panel stage differs from required closed experiment')
        for field in ['fit_outcome_max','scaler_fit_outcome_max','configuration_selection_outcome_max']:
            _require(date.fromisoformat(str(r[field])[:10])<=date.fromisoformat(cutoff),'Fit/scaler/configuration exceeds registered cutoff')
        for field in ['checkpoint_sha256','scaler_sha256','config_sha256']:
            checksum=r[field];_require(isinstance(checksum,str) and len(checksum)==64 and set(checksum)<=set('0123456789abcdef'),'Invalid closed identity hash')
        _require(r.get('run_id'),'A closed run ID is required')
        closures.append(_stamp(r['checkpoint_closed_at_utc']));scorings.append(_stamp(r['scoring_started_at_utc']))
        known=float(r['persistence_pred']);high=_boolean(r['high_volatility_known'])
        _require(math.isfinite(known) and high==(known>THRESHOLD),'Regime must use known lagged sigma, not the target')
        _require(origin not in regimes or regimes[origin]==high,'Regime differs across models/seeds')
        regimes[origin]=high;scale=float(r['mase_scale']);_require(math.isfinite(scale) and scale>0,'Train-only MASE scale must be positive')
        key=(r['model'],str(r['seed']));identity=tuple(r[k] for k in ['run_id','checkpoint_sha256','scaler_sha256','config_sha256'])
        _require(key not in run_cohorts or run_cohorts[key]==identity,'Frozen run identity switched during evaluation')
        run_cohorts[key]=identity
        _require(key not in mase or math.isclose(mase[key],scale,rel_tol=1e-12),'Train-only MASE scale switched within a frozen run')
        mase[key]=scale
        r.update(origin_utc=origin,horizon=1,eligible=r.get('eligible',True));converted.append(r)
    _require(max(closures)<min(scorings),'All checkpoints must close before the first phase scoring')
    panel=validate_dense_panel(converted,models,seeds,calendar,deterministic_models=det)
    panel['high']=np.array([regimes[d] for d in calendar],bool);panel['mase_scale']=mase;panel['rows']=converted
    known_by_day={str(r['origin_utc'])[:10]:float(r['persistence_pred']) for r in converted}
    for i,d in enumerate(calendar[1:],1):
        if panel['eligible'][i-1] and panel['eligible'][i]:_require(math.isclose(known_by_day[d],float(panel['targets'][i-1]),rel_tol=1e-12,abs_tol=1e-15),'Regime covariate must equal previous-day sigma when that label is present')
    panel['stage']=stage;panel['calendar_dates']=calendar
    return panel


def _costs(rows, models, *, phase, checkpoint_hashes=None):
    rows=list(rows);out={};protocol={}
    for r in rows:
        _require(r['model'] in models,'Unexpected timing model')
        _require(r['phase']==phase,'Timing phase mismatch')
        _require(int(r['warmup_calls'])==256 and int(r['calls_per_block'])==256,'Timing warmup/calls differ')
        _require(len(str(r['source_checkpoint_sha256']))==64,'Timing needs checkpoint identity')
        if checkpoint_hashes is not None:_require(r['source_checkpoint_sha256'] in checkpoint_hashes[r['model']],'Timing checkpoint differs from evaluated model')
        key=(r['model'],int(r['block']));_require(key not in out,'Duplicate paired timing block')
        latency=float(r['latency_ms']);_require(math.isfinite(latency) and latency>0,'Positive finite latency required')
        out[key]=latency
    _require(len(out)==len(models)*61 and all((m,b) in out for m in models for b in range(61)),'Exactly61 interleaved paired timing blocks per model required')
    return np.array([[out[m,b] for b in range(61)] for m in models])


def _mean_seed_rmse(squared, counts, mask=None):
    weights=np.asarray(counts,dtype=float)
    if mask is not None:weights=weights*np.asarray(mask,dtype=float)
    denominator=weights.sum(axis=-1)
    # squared M×S×T, weights B×T -> B×M×S. No pooled-seed MSE substitution.
    result=np.einsum('mst,bt->bms',squared,weights,optimize=True)
    with np.errstate(invalid='ignore',divide='ignore'):
        result=np.sqrt(result/denominator[:,None,None])
    result[denominator==0]=np.nan
    return result


def _counts(n, draws, block, rng, length=None):
    length=n if length is None else length
    starts=rng.integers(0,n,(draws,math.ceil(length/block)))
    idx=((starts[:,:,None]+np.arange(block))%n).reshape(draws,-1)[:,:length]
    counts=np.zeros((draws,n),float)
    np.add.at(counts,(np.arange(draws)[:,None],idx),1)
    return counts


def _timing_bootstrap(timings,draws,rng):
    # Paired iid timing blocks, separate from circular predictive-date blocks.
    counts=_counts(61,draws,1,rng)
    return (np.asarray(timings)@counts.T/61).T


def _boot_scores(panel, *, repetitions, block, seed):
    sq=(panel['predictions']-panel['targets'][None,None,:])**2
    sq[:,:,~panel['eligible']]=0
    rng=np.random.default_rng(seed);m=len(panel['models']);rmse=np.empty((repetitions,m));high=np.empty_like(rmse);rest=np.empty_like(rmse)
    for start in range(0,repetitions,256):
        stop=min(repetitions,start+256);counts=_counts(sq.shape[-1],stop-start,block,rng);counts*=panel['eligible'][None,:]
        rmse[start:stop]=_mean_seed_rmse(sq,counts).mean(axis=-1)
        high[start:stop]=_mean_seed_rmse(sq,counts,panel['high']).mean(axis=-1)
        rest[start:stop]=_mean_seed_rmse(sq,counts,~panel['high']).mean(axis=-1)
    return rmse,high,rest


def volatility_scores(target,prediction):
    y=np.asarray(target,float);p=np.asarray(prediction,float)
    _require(y.shape==p.shape and y.size>0 and np.isfinite(y).all() and np.isfinite(p).all() and np.min(y)>=0 and np.min(p)>=0,'Finite nonnegative volatility arrays required')
    raw=p**2;floor=1e-12;variance=np.maximum(raw,floor)
    return {'QLIKE_log_variance_plus_ratio':float((np.log(variance)+y**2/variance).mean()),'MSE_realized_variance':float(((raw-y**2)**2).mean()),'QLIKE_variance_floor':floor,'QLIKE_forecast_floor_fraction':float((raw<floor).mean()),'zero_target_fraction':float((y==0).mean()),'VaR_coverage_tests':'not_applicable: no predictive return distribution/VaR is declared'}


def _secondary_bootstrap(panel, *, repetitions,block,seed):
    y=panel['targets'];pred=panel['predictions'];err=pred-y[None,None,:];M,S,T=pred.shape;mask=panel['eligible'];sq=err**2
    scales=np.array([[panel['mase_scale'][m,'deterministic' if m in DETERMINISTIC else str(seed_id)] for seed_id in panel['seeds']] for m in panel['models']])
    variance=np.maximum(pred**2,1e-12)
    losses={'MAE':np.abs(err),'MSE':sq,'MASE':np.abs(err)/scales[:,:,None],
            'QLIKE':np.log(variance)+y[None,None,:]**2/variance,
            'MSE_variance':(pred**2-y[None,None,:]**2)**2}
    positive=(y>0)&mask
    with np.errstate(invalid='ignore',divide='ignore'):
        losses['MAPE_percent']=np.where(positive[None,None,:],np.abs(err)/y[None,None,:]*100,0.)
        denominator=np.abs(pred)+np.abs(y)[None,None,:];smask=(denominator>0)&mask[None,None,:]
        losses['SMAPE_percent']=np.where(smask,2*np.abs(err)/denominator*100,0.)
    for values in losses.values():values[:,:,~mask]=0
    storage={key:np.empty((repetitions,M)) for key in losses};storage['R2']=np.empty((repetitions,M))
    rng=np.random.default_rng(seed)
    for start in range(0,repetitions,256):
        stop=min(repetitions,start+256);counts=_counts(T,stop-start,block,rng)*mask[None,:];n=counts.sum(axis=1)
        for key,values in losses.items():
            valid=positive[None,None,:] if key=='MAPE_percent' else smask if key=='SMAPE_percent' else mask[None,None,:]
            valid=np.broadcast_to(valid,values.shape)
            if np.array_equal(valid,np.broadcast_to(valid[:,0:1,:],values.shape)):
                num=counts@values.mean(axis=1).T;den=counts@valid[:,0,:].astype(float).T
                with np.errstate(invalid='ignore',divide='ignore'):result=num/den
            else:
                num=np.einsum('mst,bt->bms',values,counts,optimize=True);den=np.einsum('mst,bt->bms',valid.astype(float),counts,optimize=True)
                with np.errstate(invalid='ignore',divide='ignore'):result=(num/den).mean(axis=-1)
            storage[key][start:stop]=result
        with np.errstate(invalid='ignore',divide='ignore'):
            centered_y=np.where(mask,y-y[mask][0],0.);center_mean=counts@centered_y/n;target_var=(counts@(centered_y**2))/n-center_mean**2
            storage['R2'][start:stop]=1-storage['MSE'][start:stop]/target_var[:,None]
        storage['R2'][start:stop][target_var<=0]=np.nan
    return {model:{key:{'ci95_nominal':np.quantile(array[:,i][np.isfinite(array[:,i])],[.025,.975]).tolist() if np.isfinite(array[:,i]).sum()>=.95*repetitions else None,'estimable_draws':int(np.isfinite(array[:,i]).sum())} for key,array in storage.items()} for i,model in enumerate(panel['models'])}


def _one_sided(observed, boot, boundary, *, side='greater', alpha=.02/30):
    x=np.asarray(boot,float);valid=np.isfinite(x);n=len(x)
    if not np.isfinite(observed) or valid.sum()<.95*n:
        return {'effect':None if not np.isfinite(observed) else float(observed),'p':None,'ci95_nominal':None,'simultaneous_bound':None,'estimable_draws':int(valid.sum()),'requested_draws':n,'status':'not_estimable'}
    values=x[valid];noise=values-observed
    if side=='greater':
        extreme=np.count_nonzero(noise>=observed-boundary);bound=observed-np.quantile(noise,1-alpha)
    elif side=='less':
        extreme=np.count_nonzero(noise<=observed-boundary);bound=observed-np.quantile(noise,alpha)
    else:raise ValueError('Unknown alternative')
    # Unestimable bootstrap draws conservatively count as exceedances.
    p=(1+extreme+n-valid.sum())/(n+1)
    return {'effect':float(observed),'p':float(p),'ci95_nominal':np.quantile(values,[.025,.975]).tolist(),'simultaneous_bound':float(bound),'estimable_draws':int(valid.sum()),'requested_draws':n,'status':'estimable','boundary':boundary,'alternative':side,'bound_alpha':alpha}


def _point(panel):
    err=panel['predictions']-panel['targets'][None,None,:];mask=panel['eligible'];sq=err**2
    mean=lambda sel:np.sqrt(sq[:,:,sel].mean(axis=-1)).mean(axis=-1) if sel.any() else np.full(sq.shape[0],np.nan)
    return mean(mask),mean(mask&panel['high']),mean(mask&~panel['high']),sq


def _holm_rows(rows, *, alpha, family_size):
    adjusted=adjust_pvalues([r.get('p') for r in rows],method='Holm',family_size=family_size)
    for row,p in zip(rows,adjusted):
        row.update(p_Holm=p,family_alpha=alpha,family_size=family_size,
                   reject=False if p is None else bool(p<alpha))
    return rows


def _families(panel, bootstrap, timings, prereg, *, repetitions, seed):
    observed,obs_high,obs_rest,_=_point(panel);br,bh,bl=bootstrap;models=panel['models'];rng=np.random.default_rng(seed+1)
    bt=_timing_bootstrap(timings,repetitions,rng);means=timings.mean(axis=-1)
    allrows=[]
    for fam,ref in zip(prereg['families'],REFERENCES):
        bi=models.index(ref);records=[]
        for model in TOP10:
            mi=models.index(model);gain=observed[bi]-observed[mi];gainb=br[:,bi]-br[:,mi]
            erosion=(obs_rest[bi]-obs_rest[mi])-(obs_high[bi]-obs_high[mi]);eb=(bl[:,bi]-bl[:,mi])-(bh[:,bi]-bh[:,mi])
            cost=means[mi]-means[bi];cb=bt[:,mi]-bt[:,bi]
            g=_one_sided(gain,gainb,DELTA,alpha=fam['alpha']/30);d=_one_sided(erosion,eb,DELTA,alpha=fam['alpha']/30)
            c=_one_sided(cost,cb,0,side='less',alpha=fam['alpha']/30)
            for rq,result in [('RQ1',g),('RQ2',d)]:records.append({'id':f"{fam['id']}:{rq}:{model}",'family':fam['id'],'reference':ref,'model':model,'RQ':rq,**result})
            p=None if g['p'] is None or c['p'] is None else max(g['p'],c['p'])
            records.append({'id':f"{fam['id']}:RQ3:{model}",'family':fam['id'],'reference':ref,'model':model,'RQ':'RQ3','p':p,'gain_effect':g['effect'],'cost_ms_effect':c['effect'],'gain_p':g['p'],'cost_p':c['p'],'gain_simultaneous_lower':g['simultaneous_bound'],'cost_simultaneous_upper':c['simultaneous_bound'],'method':'intersection_union_max','status':'not_estimable' if p is None else 'estimable'})
        _require(len(records)==fam['hypothesis_count']==30,'Family registry mismatch')
        _require({r['id'] for r in records}=={r['id'] for r in fam['hypotheses']},'Hypothesis IDs differ from preregistration')
        allrows.extend(_holm_rows(records,alpha=fam['alpha'],family_size=30))
    return allrows


def _diagnostics(panel,prereg):
    from statsmodels.stats.diagnostic import het_arch
    from statsmodels.tsa.stattools import bds
    import statsmodels.api as sm
    y=panel['targets'];models=panel['models'];n=len(y);mask=panel['eligible'];out=[]
    sq=(panel['predictions']-y[None,None,:])**2
    # Gaps cannot be compressed into daily residual diagnostics.
    dense=bool(mask.all())
    for item in prereg['diagnostic_family']['hypotheses']:
        row=dict(item);row.update(p=None,statistic=None,status='not_estimable',reason=None)
        try:
            _require(dense,'Residual diagnostics require complete daily cohort; missing dates cannot be compressed')
            if item['id'].startswith('diagnostic:DM_HLN:'):
                a=models.index(item['reference']);b=models.index(item['candidate'])
                result=dm_hln(sq[a].mean(axis=0),sq[b].mean(axis=0),panel['origins'],horizon=1,hac_bandwidth=7)
                row.update(p=result['p_two_sided'],statistic=result['statistic'],mean_loss_difference=result['mean_loss_difference'],reason=result.get('reason'))
            else:
                i=models.index(item['model']);pred=panel['predictions'][i,0];e=y-pred;test=item['test'];_require(np.std(e)>0,'Residual variance zero')
                if test.startswith('LB_'):
                    lag=int(test.rsplit('lag',1)[1]);kind='squared_residuals' if test.startswith('LB_squared') else 'residuals'
                    values=ljung_box_diagnostics(e,panel['origins'],lags=[lag],model_df=0)
                    result=next(x for x in values if x['series']==kind);row.update(p=result['p_unadjusted'],statistic=result['Q'])
                elif test.startswith('ARCH_'):
                    lm,p,_,_=het_arch(e,nlags=7,ddof=0);row.update(p=float(p),statistic=float(lm))
                elif test.startswith('BDS_'):
                    dim=int(test.split('dim')[1][0]);z,p=bds(e,max_dim=dim,epsilon=1.5*np.std(e,ddof=1));row.update(p=float(np.atleast_1d(p)[-1]),statistic=float(np.atleast_1d(z)[-1]))
                elif test=='Jarque_Bera':
                    result=stats.jarque_bera(e);row.update(p=float(result.pvalue),statistic=float(result.statistic))
                elif test.startswith('Mincer_'):
                    X=np.column_stack([np.ones(n),pred]);_require(np.linalg.matrix_rank(X)==2,'Constant forecast prevents intercept/slope identification')
                    fit=sm.OLS(y,X).fit(cov_type='HAC',cov_kwds={'maxlags':7,'use_correction':True});d=fit.params-np.array([0.,1.]);cov=fit.cov_params();_require(np.linalg.matrix_rank(cov)==2,'Singular HAC covariance')
                    w=float(d@np.linalg.solve(cov,d));row.update(p=float(stats.chi2.sf(w,2)),statistic=w,intercept=float(fit.params[0]),slope=float(fit.params[1]))
                else:raise ValueError('Unknown registered diagnostic')
                row['designated_seed']='deterministic' if item['model'] in DETERMINISTIC else panel['seeds'][0]
            if row['p'] is not None and math.isfinite(row['p']) and row['statistic'] is not None and math.isfinite(row['statistic']):row['status']='estimable'
            else:row.update(p=None,status='not_estimable',reason=row.get('reason') or 'Nonfinite diagnostic')
        except (ValueError,ZeroDivisionError,np.linalg.LinAlgError,FloatingPointError) as error:
            row.update(p=None,status='not_estimable',reason=str(error))
        out.append(row)
    _require(len(out)==199,'Exactly199 diagnostics required')
    return _holm_rows(out,alpha=.01,family_size=199)


def model_confidence_set(losses,models, *, repetitions=9999,block=7,seed=202611067,alpha=.1):
    """Sequential Tmax MCS on per-date mean individual seed squared loss.

    A separately registered exploratory family; not covered by primary alpha .05.
    Circular-block consistency requires suitable weak dependence/stability.
    """
    x=np.asarray(losses,float);_require(x.ndim==2 and x.shape[0]==len(models) and np.isfinite(x).all(),'MCS needs finite M×T losses')
    _require(repetitions>=999 and 0<alpha<1,'Invalid MCS settings')
    counts=_counts(x.shape[1],repetitions,block,np.random.default_rng(seed));boot=(x@counts.T/counts.sum(axis=1)).T
    active=list(range(len(models)));steps=[];excluded={};running_p=0.
    while len(active)>1:
        observed=x[active].mean(axis=1);d=observed-observed.mean();bd=boot[:,active]-boot[:,active].mean(axis=1)[:,None]
        centered=bd-d[None,:];sd=centered.std(axis=0,ddof=1)
        if (sd<=0).any():break
        statistic=float(np.max(d/sd));null=np.max(centered/sd[None,:],axis=1)
        p=float((1+np.count_nonzero(null>=statistic))/(repetitions+1));running_p=max(running_p,p)
        worst=int(np.argmax(d/sd));steps.append({'active':[models[i] for i in active],'Tmax':statistic,'p':p,'p_stepwise':running_p})
        if running_p>=alpha:break
        removed=active.pop(worst);excluded[models[removed]]=running_p
    return {'level':1-alpha,'included_models':[models[i] for i in active],'excluded_pvalues':excluded,'steps':steps,'loss':'per-date mean individual squared-seed error','block_days':block,'draws':repetitions,'separate_exploratory_family':True,'does_not_prove_equivalence_or_margin_gain':True}


def wilson_lower(successes,total,z=1.6448536269514722):
    _require(total>0 and 0<=successes<=total,'Invalid Monte Carlo count')
    p=successes/total;den=1+z*z/total
    return (p+z*z/(2*total)-z*math.sqrt(p*(1-p)/total+z*z/(4*total*total)))/den


def _tail_p(null_noise,alternative_noise,shift, *, side='greater'):
    n=len(null_noise);valid=np.isfinite(null_noise);reference=np.sort(np.asarray(null_noise)[valid]);observed=np.asarray(alternative_noise)+shift
    out=np.ones(len(observed));ok=np.isfinite(observed)
    if len(reference)<2 or np.std(reference)<=np.finfo(float).eps:return out
    if side=='greater':count=len(reference)-np.searchsorted(reference,observed[ok],side='left')
    else:count=np.searchsorted(reference,observed[ok],side='right')
    out[ok]=(1+count+n-valid.sum())/(n+1)
    return out


def _planning_MDE(null_noise,alternative_noise, *, cost_pass=None,alpha=.02/30,target=.8):
    n=len(null_noise);a=np.asarray(null_noise,float);v=np.asarray(alternative_noise,float);valid=np.isfinite(a);reference=np.sort(a[valid]);invalid=n-int(valid.sum())
    allowed=math.ceil(alpha*(n+1)-1-invalid)-1
    if allowed<0 or len(reference)<2 or np.std(reference)<=np.finfo(float).eps:return None
    # Upper-tail finite-MC threshold, including conservative invalid-null draws.
    null_threshold=np.nextafter(reference[len(reference)-allowed-1],np.inf)
    feasible=np.isfinite(v)
    if cost_pass is not None:feasible&=cost_pass
    thresholds=np.full(n,np.inf);thresholds[feasible]=null_threshold-v[feasible]
    required=next((k for k in range(int(target*n),n+1) if wilson_lower(k,n)>=target),n)
    value=float(np.sort(thresholds)[required-1])
    return max(float(np.spacing(DELTA)),value) if math.isfinite(value) else None


def _power_seed_scores(panel,draws,rng):
    sq=(panel['predictions']-panel['targets'][None,None,:])**2;sq[:,:,~panel['eligible']]=0
    counts=_counts(sq.shape[-1],draws,7,rng,length=60);counts*=panel['eligible'][None,:]
    return np.stack([_mean_seed_rmse(sq,counts),_mean_seed_rmse(sq,counts,panel['high']),_mean_seed_rmse(sq,counts,~panel['high'])],axis=-1)


def _contrast_noise(seed_scores,weights,models):
    # B×M×S×3 metrics, B×S seed weights; output B×2reference×3measured×2gain/erosion.
    scores=np.einsum('bmsk,bs->bmk',seed_scores,weights,optimize=True);out=np.empty((len(weights),2,3,2))
    for ref_i,reference in enumerate(REFERENCES):
        b=models.index(reference)
        for i,model in enumerate(P1_MODELS[:3]):
            m=models.index(model);out[:,ref_i,i,0]=scores[:,b,0]-scores[:,m,0]
            out[:,ref_i,i,1]=(scores[:,b,2]-scores[:,m,2])-(scores[:,b,1]-scores[:,m,1])
    return out


def _proxy_expand(noise,panel):
    measured=P1_MODELS[:3];out=np.empty((noise.shape[0],2,10,2));proxies={}
    point,high,rest,sq=_point(panel)
    masks=[panel['eligible'],panel['eligible']&panel['high'],panel['eligible']&~panel['high']]
    byseed=np.stack([np.sqrt(sq[:,:,mask].mean(axis=-1)) if mask.any() else np.full(sq.shape[:2],np.nan) for mask in masks],axis=-1)
    for b,reference in enumerate(REFERENCES):
        ri=panel['models'].index(reference)
        for q in range(2):
            sds=[]
            for model in measured:
                mi=panel['models'].index(model)
                d=byseed[ri,:,0]-byseed[mi,:,0] if q==0 else (byseed[ri,:,2]-byseed[mi,:,2])-(byseed[ri,:,1]-byseed[mi,:,1])
                sd=float(np.std(d,ddof=1));upper=sd*math.sqrt(9/stats.chi2.ppf(.025,9));sds.append((upper,sd))
            worst=int(np.argmax([x[0] for x in sds]));upper,sd=sds[worst]
            factor=1.5*upper/sd if sd>0 else None
            for i,model in enumerate(TOP10):
                if model in measured:out[:,b,i,q]=noise[:,b,measured.index(model),q]
                else:
                    out[:,b,i,q]=noise[:,b,worst,q]*factor if factor is not None else np.nan
                    proxies[f'{reference}:{model}:{q}']={'source_model':measured[worst],'training_SD_upper95':upper,'inflation':1.5,'noise_scaling':factor,'empirical_power_certified':False,'shared_proxy_noise_correlation_assumed':True}
    return out,proxies


def _matrix_holm(p):
    order=np.argsort(p,axis=1,kind='stable');ranked=np.take_along_axis(p,order,axis=1);adj=np.minimum(1,np.maximum.accumulate(ranked*np.arange(p.shape[1],0,-1),axis=1));out=np.empty_like(adj);np.put_along_axis(out,order,adj,axis=1);return out


def _power_cells(null,alternative,cost_null,cost_alt,cost_reference_means, *, draws):
    # The same centered null noise is never substituted for observed improvement.
    rows=[];rejections=[]
    for b,reference in enumerate(REFERENCES):
        pg=np.empty((draws,10));pe=np.empty_like(pg);pc=np.empty_like(pg)
        for m in range(10):
            pg[:,m]=_tail_p(null[:,b,m,0],alternative[:,b,m,0],DELTA)
            pe[:,m]=_tail_p(null[:,b,m,1],alternative[:,b,m,1],DELTA)
            pc[:,m]=_tail_p(cost_null[:,b,m],cost_alt[:,b,m],-.2*cost_reference_means[b],side='less')
        p=np.stack([pg,pe,np.maximum(pg,pc)],axis=-1).reshape(draws,30);adjusted=_matrix_holm(p);reject=adjusted<.02;rejections.append(reject)
        for i,model in enumerate(TOP10):
            for q,rq in enumerate(['RQ1','RQ2','RQ3']):
                count=int(reject[:,i*3+q].sum())
                qindex=1 if rq=='RQ2' else 0
                nf=int(np.isfinite(null[:,b,i,qindex]).sum());af=int(np.isfinite(alternative[:,b,i,qindex]).sum())
                if rq=='RQ3':nf=min(nf,int(np.isfinite(cost_null[:,b,i]).sum()));af=min(af,int(np.isfinite(cost_alt[:,b,i]).sum()))
                rows.append({'reference':reference,'model':model,'RQ':rq,'power':count/draws,'Wilson_lower95':wilson_lower(count,draws),'MC_successes':count,'MC_draws':draws,'finite_null_draws':nf,'finite_alternative_draws':af,'estimable':nf>=.95*draws and af>=.95*draws and bool(np.std(null[:,b,i,qindex][np.isfinite(null[:,b,i,qindex])])>np.finfo(float).eps) and (rq!='RQ3' or bool(np.std(cost_null[:,b,i][np.isfinite(cost_null[:,b,i])])>np.finfo(float).eps)),'unestimable_draws_count_as_nonrejections':True})
    return rows,rejections


def power_selection(pilot_panel,cost_panel,prereg):
    """Prospective P1 planning using independent null/alternative MC samples.

    Inestimable draws count as non-rejections. Source/calendar/P1-to-final transfer
    assumptions are reported; simulated capability is never called achieved power.
    """
    _settings(prereg);cost_panel=list(cost_panel);panel=_panel(pilot_panel,kind='P1');timing=_costs(cost_panel,P1_MODELS,phase='P1',checkpoint_hashes={m:{r['checkpoint_sha256'] for r in panel['rows'] if r['model']==m} for m in P1_MODELS})
    _require(panel['n_eligible_origins']>=40,'P1 needs enough common dates; sparse strata remain non-estimable')
    draws=10000;rng=np.random.default_rng(prereg['power']['mc_seed']);S=10
    temporal=[_power_seed_scores(panel,draws,rng) for _ in range(2)]
    sq=(panel['predictions']-panel['targets'][None,None,:])**2
    masks=[panel['eligible'],panel['eligible']&panel['high'],panel['eligible']&~panel['high']]
    fixed=np.stack([np.sqrt(sq[:,:,mask].mean(axis=-1)) if mask.any() else np.full(sq.shape[:2],np.nan) for mask in masks],axis=-1)
    fixed=np.broadcast_to(fixed,(draws,*fixed.shape))
    uniform=np.full((draws,S),1/S)
    cost_noise=[];cost_proxy={};refmeans=[float(timing[P1_MODELS.index(m)].mean()) for m in REFERENCES]
    for _ in range(2):
        sample=_timing_bootstrap(timing,draws,rng);sample-=timing.mean(axis=-1)[None,:]
        values=np.empty((draws,2,10))
        for b,reference in enumerate(REFERENCES):
            ri=P1_MODELS.index(reference);measured=np.stack([sample[:,P1_MODELS.index(m)]-sample[:,ri] for m in P1_MODELS[:3]],axis=-1)
            sds=measured.std(axis=0,ddof=1);worst=int(np.argmax(sds));upper=float(sds[worst]*math.sqrt(60/stats.chi2.ppf(.025,60)));factor=1.5*upper/sds[worst] if sds[worst]>0 else None
            for i,m in enumerate(TOP10):
                if m in P1_MODELS[:3]:values[:,b,i]=measured[:,P1_MODELS[:3].index(m)]
                else:values[:,b,i]=measured[:,worst]*factor if factor else np.nan
            cost_proxy[reference]={'source':P1_MODELS[worst],'SD_upper95':upper,'inflation':1.5,'unmeasured_cost_noise_proxy':True}
        cost_noise.append(values)
    allrows=[];proxy_registry={};selected=None;selected_cells=None;last_noise={}
    for R in R_GRID:
        weights=[rng.multinomial(R,np.full(S,1/S),draws)/R for _ in range(2)]
        source_noises={}
        for source,score_sets,weight_sets in [('training', [fixed,fixed],weights),('calendar',temporal,[uniform,uniform]),('joint',temporal,weights)]:
            pair=[]
            for scores,w in zip(score_sets,weight_sets):
                raw=_contrast_noise(scores,w,P1_MODELS);center=np.nanmean(raw,axis=0);noise=raw-center[None,...];expanded,proxies=_proxy_expand(noise,panel);pair.append(expanded);proxy_registry.update(proxies)
            source_noises[source]=pair
        current=[]
        for source,(null,alt) in source_noises.items():
            cells,_=_power_cells(null,alt,cost_noise[0],cost_noise[1],refmeans,draws=draws)
            for cell in cells:cell.update(R=R,source=source)
            current.extend(cells);last_noise[source]=(null,alt)
        allrows.extend(current)
        if selected is None and all(r['Wilson_lower95']>=.8 for r in current):selected=R;selected_cells=current;break
    if selected is None:selected=200;selected_cells=current
    # Conservative Bonferroni MDE beyond the practical-null boundary at selected R.
    mde=[]
    for source,(null,alt) in last_noise.items():
        for b,ref in enumerate(REFERENCES):
            for i,model in enumerate(TOP10):
                for q,rq in enumerate(['RQ1','RQ2']):
                    a=null[:,b,i,q];v=alt[:,b,i,q];excess=_planning_MDE(a,v)
                    mde.append({'source':source,'reference':ref,'model':model,'RQ':rq,'R':selected,'excess_beyond_null':excess,'true_alternative_effect':None if excess is None else DELTA+excess,'method':'conservative marginal Bonferroni bootstrap MDE with Wilson planning precision and invalid draws counted as non-rejections; not full-family joint80% proof'})
                cn=cost_noise[0][:,b,i];ca=cost_noise[1][:,b,i];pc=_tail_p(cn,ca,-.2*refmeans[b],side='less');cost_pass=pc<.02/30
                a=null[:,b,i,0];v=alt[:,b,i,0];mde3=_planning_MDE(a,v,cost_pass=cost_pass)
                mde.append({'source':source,'reference':ref,'model':model,'RQ':'RQ3','R':selected,'excess_beyond_null':mde3,'true_alternative_effect':None if mde3 is None else DELTA+mde3,'fixed_cost_reduction':.2,'cost_component_rejection_probability_Bonferroni':float(cost_pass.mean()),'status':'estimable_conservative_MDE' if mde3 is not None else 'not_estimable_or_cost_limits_joint_power','method':'conservative Bonferroni gain-and-cost IUT; null MC distribution held fixed'})
    result={'selected_R':selected,'planning_status':'target_met_in_simulation' if all(r['Wilson_lower95']>=.8 for r in selected_cells) else ('unknown_capability_at_R200' if any(not r['estimable'] for r in selected_cells) else 'target_not_met_at_R200'),'target_met_in_planning_simulation':all(r['Wilson_lower95']>=.8 for r in selected_cells),'target':.8,'selection_rule':'minimum R with one-sided95%Wilson lower>=.80 for every hypothesis and training/calendar/joint source; elseR200 with unmet/unknown capability','candidate_R':R_GRID,'MC_null_draws':draws,'MC_alternative_draws':draws,'null_and_alternative_samples_independent':True,'planned_gain_and_erosion':2*DELTA,'planned_excess_beyond_null':DELTA,'planned_relative_latency_reduction':.2,'power_cells':allrows,'selected_R_cells':selected_cells,'MDE_conservative_cells':mde,'proxy_registry':proxy_registry,'timing_proxy_registry':cost_proxy,'empirical_model_power_available_only_for':P1_MODELS[:3],'final_post_HPO_power_is_planned_not_observed':True,'P1_calendar_amendment':{'start':'2024-11-08','end':'2024-12-30','source_dates':53,'fit_and_selection_max':'2024-10-31','final_calendar_positions_simulated':60,'circular_resampling_does_not_create_new_dates':True,'source_599_calendar_preserved_as_historical_record':True},'assumptions':['Empirical P1 bootstrap approximates future training/temporal error distributions','P1 fit throughOct31 transfers to final fit throughDec24','One known Bitcoin chronology; seeds never create independent market dates','Seven unpiloted Top10 recipes use worst representative upper95%SD times1.5; correlation follows that proxy, not measured architecture-specific capability','Latency blocks are independent of predictive-calendar resampling; temporal and training source checks plus conservative joint-source condition','Simulation Wilson precision does not eliminate uncertainty in the small P1 empirical distribution'],'fits_performed':0,'market_data_acquired':False,'P1_canonical_panel_sha256':_rows_hash(panel['rows']),'timing_canonical_panel_sha256':_rows_hash(cost_panel)}
    return clean(result)


def _overfit(rows,prereg=None, *, repetitions=4999,expected_seeds=None,source_stage='E2_final_inner_diagnostics',formal=False,validation_calendar=None,cutoff=None,expected_checkpoints=None,block=7):
    """Same frozen predictor; dense paired calendars and optional A53 families."""
    rows=list(rows);_require(rows,'Inner train/validation predictions required')
    groups={};date_sets={'train':set(),'validation':set()};common_truth={}
    for r in rows:
        _require(r['source_stage']==source_stage,'Wrong train/validation diagnostic stage')
        _require(len(str(r['checkpoint_sha256']))==64,'Training inference needs checkpoint identity')
        if cutoff is not None:
            for field in ['fit_outcome_max','scaler_fit_outcome_max','configuration_selection_outcome_max']:_require(date.fromisoformat(str(r[field])[:10])<=date.fromisoformat(cutoff),'Train/validation fit/scaler/selection exceeds cutoff')
        _require(r['model'] in MODELS and r['split'] in ['train','validation'],'Unknown inner group')
        key=(r['model'],str(r['seed']));g=groups.setdefault(key,{'train':{},'validation':{},'hashes':set(),'identities':set()})
        d=str(r.get('origin',r.get('outcome_day')))[:10];_require(d not in g[r['split']],'Duplicate inner date')
        g[r['split']][d]=r;g['hashes'].add(r['checkpoint_sha256']);date_sets[r['split']].add(d)
        if formal:
            for field in ['scaler_sha256','config_sha256']:_require(len(str(r[field]))==64 and set(str(r[field]))<=set('0123456789abcdef'),'A53 needs frozen scaler/config hashes')
            g['identities'].add(tuple(r[f] for f in ['checkpoint_sha256','scaler_sha256','config_sha256']))
    if expected_seeds is not None:
        expected={(m,'deterministic') for m in DETERMINISTIC}|{(m,str(seed)) for m in MODELS if m not in DETERMINISTIC for seed in expected_seeds}
        _require(set(groups)==expected,'Inner diagnostics need the same full final model/seed registry')
    _require(all(date_sets.values()),'Both inner partitions required')
    calendars={split:_calendar(min(ds),max(ds)) for split,ds in date_sets.items()}
    if validation_calendar is not None:_require(calendars['validation']==validation_calendar,'Validation must preserve the full registered calendar')
    _require(date.fromisoformat(calendars['train'][-1])+timedelta(days=8)<=date.fromisoformat(calendars['validation'][0]),'Train/validation purge missing')
    if cutoff is not None:_require(calendars['train'][-1]<=cutoff,'Training inference contains post-cutoff outcomes')
    if formal:
        _require(repetitions==19999,'Formal A53 overfit requires19999 bootstrap draws')
        closed=[_stamp(r['checkpoint_closed_at_utc']) for r in rows if r['split']=='validation'];scored=[_stamp(r['scoring_started_at_utc']) for r in rows if r['split']=='validation'];_require(max(closed)<min(scored),'All A53 predictors close before first validation scoring')
    common_masks={};model_errors={m:{'train':[],'validation':[]} for m in MODELS};output=[]
    for (model,seed),g in groups.items():
        _require(len(g['hashes'])==1,'Train/validation must use identical frozen checkpoint')
        if expected_checkpoints is not None:_require(g['hashes']=={expected_checkpoints[model,seed]},'Training inference is not from the evaluated refit checkpoint')
        if formal:_require(len(g['identities'])==1,'A53 predictor/scaler/config changed between training and validation')
        values={}
        for split in ['train','validation']:
            errors=[];mask=[]
            for d in calendars[split]:
                row=g[split].get(d);eligible=row is not None and _boolean(row.get('eligible',True))
                if eligible:
                    y,pred=float(row['y_true']),float(row['y_pred']);_require(np.isfinite([y,pred]).all(),'Nonfinite inner predictions');truthkey=(split,d)
                    _require(truthkey not in common_truth or math.isclose(common_truth[truthkey],y,rel_tol=1e-12,abs_tol=1e-15),'Train/validation ground truth differs across models/seeds');common_truth[truthkey]=y;errors.append(pred-y)
                else:
                    _require(row is None or row.get('y_pred') in ('',None),'Ineligible inner rows must retain blank predictions');errors.append(0.)
                mask.append(eligible)
            mask=np.array(mask,bool);_require(mask.sum()>=15,'At least15 inner eligible origins required')
            if split in common_masks:_require(np.array_equal(mask,common_masks[split]),'Inner diagnostics require common masks, including explicit calendar gaps')
            else:common_masks[split]=mask
            values[split]=np.asarray(errors);model_errors[model][split].append(values[split]);values[split+'_RMSE']=float(np.sqrt((values[split][mask]**2).mean()))
        t,v=values['train_RMSE'],values['validation_RMSE'];output.append({'model':model,'seed':seed,'train_RMSE_inference':t,'validation_RMSE_inference':v,'gap':v-t,'delta_g':DELTA,'beyond_margin_descriptive':v-t>DELTA,'ratio':None if t==0 else v/t})
    _require(set(r['model'] for r in output)==set(MODELS),'Inner diagnostics must cover all20 models')
    _require(block in [7,14,28],'Overfit calendar block must follow the registered primary/sensitivity set')
    rng=np.random.default_rng(202611069);counts={split:_counts(len(calendars[split]),repetitions,block,rng)*common_masks[split][None,:] for split in calendars};violation=[];tolerance=[];summaries=[]
    for model in MODELS:
        boots={}
        for split in calendars:
            sq=np.asarray(model_errors[model][split])**2;weights=counts[split];den=weights.sum(axis=1)
            with np.errstate(invalid='ignore',divide='ignore'):boots[split]=np.sqrt((sq@weights.T)/den[None,:]).mean(axis=0)
        point=float(np.mean([r['gap'] for r in output if r['model']==model]));gap_boot=boots['validation']-boots['train'];finite=gap_boot[np.isfinite(gap_boot)]
        ensemble_RMSE={split:float(np.sqrt(np.mean(np.mean(np.asarray(model_errors[model][split]),axis=0)[common_masks[split]]**2))) for split in calendars}
        summaries.append({'model':model,'ensemble_training_RMSE_descriptive':ensemble_RMSE['train'],'ensemble_validation_RMSE_descriptive':ensemble_RMSE['validation'],'ensemble_gap_descriptive_not_delta_g_test':ensemble_RMSE['validation']-ensemble_RMSE['train'],'mean_seed_train_validation_gap':point,'ci95_temporal_nominal':np.quantile(finite,[.025,.975]).tolist() if len(finite)>=.95*repetitions else None,'delta_g':DELTA,'scope':'descriptive; period/regime/data-selection differences can cause a gap without model overfitting'})
        if formal:
            v=_one_sided(point,gap_boot,DELTA,alpha=.005/20);t=_one_sided(point,gap_boot,DELTA,side='less',alpha=.005/20)
            violation.append({'model':model,'family':'A53_gap_violation20',**v});tolerance.append({'model':model,'family':'A53_gap_tolerance20',**t})
    if formal:
        violation=_holm_rows(violation,alpha=.005,family_size=20);tolerance=_holm_rows(tolerance,alpha=.005,family_size=20)
        for v,t,summary in zip(violation,tolerance,summaries):
            bad=v['reject'] and v['simultaneous_bound']>DELTA;acceptable=t['reject'] and t['simultaneous_bound']<DELTA
            _require(not(bad and acceptable),'Conflicting complementary gap decisions')
            summary.update(decision='beyond_tolerance' if bad else 'within_operational_tolerance' if acceptable else 'inconclusive',violation_supported=bad,tolerance_supported=acceptable)
    return {'per_seed':output,'model_summaries':summaries,'violation_family20':violation,'tolerance_family20':tolerance,'scope':'A53 separate exploratory complementary20+20 subfamilies alpha.005each' if formal else 'descriptive gap only; no absence-of-overfitting conclusion','bootstrap_draws':repetitions,'block_days':block,'primary_decision':block==7,'train_and_validation_calendars_resampled_separately_without_compressing_gaps':True,'train_calendar_dates':len(calendars['train']),'validation_calendar_dates':len(calendars['validation']),'formal_hypotheses':40 if formal else 0}


def analyze_A53_overfit(train_valid_panel):
    """Fixed A53 predictor, disjoint purged validation, five members or one deterministic.

    Violation and tolerance are complementary hypotheses with separate20-member
    Holm subfamilies .005 each. Non-rejection is explicitly inconclusive.
    """
    rows=list(train_valid_panel);Aseeds=[str(s) for s in range(680101,680106)]
    result=_overfit(rows,repetitions=19999,expected_seeds=Aseeds,source_stage='E2_A53_train_validation',formal=True,validation_calendar=_calendar('2024-11-08','2024-12-30'),cutoff='2024-10-31')
    result['calendar_block_sensitivities']=[_overfit(rows,repetitions=19999,expected_seeds=Aseeds,source_stage='E2_A53_train_validation',formal=True,validation_calendar=_calendar('2024-11-08','2024-12-30'),cutoff='2024-10-31',block=block) for block in [14,28]]
    result['sensitivity_only_not_additional_rejection_opportunity']=True
    result.update(estimand='mean of five per-member RMSE gaps; deterministic references once',not_ensemble_RMSE_of_calibration_mean_prediction=True,source_stage='E2_A53_train_validation',branch='secondary_frozen_calibration_A53',does_not_establish_absence_of_overfit_for_primary_Dec24_predictor=True,primary_global_alpha_05_not_extended=True,fits_performed=0)
    return clean(result)



def efficiency_equivalence(panel,bootstrap,timings,selection, *, seed=202611070):
    """TOST versus development winner, and cost per improvement versus devbaseline.

    Every non-reference model remains in the19-comparison exploratory family;
    post-test latency ordering does not change its size. No gain/zero denominator
    becomes an artificial positive efficiency.
    """
    _require(selection and selection.get('frozen_before_test') is True,'Frozen development selection required for efficiency')
    _require(selection.get('source')=='complete20model_common_development' and selection.get('outcome_max')<='2024-10-31','Efficiency winner must come from complete common development cohort')
    best=selection['best_model'];baseline=selection['best_baseline'];_require(best in MODELS and baseline in MODELS[10:],'Invalid development efficiency references')
    point,_,_,_= _point(panel);br=bootstrap[0];B=len(br);bi=MODELS.index(best);basei=MODELS.index(baseline)
    means=timings.mean(axis=1);rng=np.random.default_rng(seed);bt=_timing_bootstrap(timings,B,rng);eq=[];eta=[]
    for i,model in enumerate(MODELS):
        if i!=bi:
            gain=point[bi]-point[i];draw=br[:,bi]-br[:,i];lo=_one_sided(gain,draw,-DELTA,alpha=.01/19);hi=_one_sided(gain,draw,DELTA,side='less',alpha=.01/19)
            equiv_p=None if lo['p'] is None or hi['p'] is None else max(lo['p'],hi['p'])
            cost=_one_sided(float(means[i]-means[bi]),bt[:,i]-bt[:,bi],0,side='less',alpha=.01/19)
            p=None if equiv_p is None or cost['p'] is None else max(equiv_p,cost['p'])
            eq.append({'model':model,'reference_development_winner':best,'p':p,'effect':float(gain),'margin':DELTA,'p_lower':lo['p'],'p_upper':hi['p'],'p_TOST':equiv_p,'p_cost':cost['p'],'cost_simultaneous_upper':cost['simultaneous_bound'],'ci95_nominal':lo['ci95_nominal'],'cheaper_latency_descriptive':bool(means[i]<means[bi]),'method':'IUT of bootstrap TOST two practical boundaries and lower latency; all19 retained'})
        if model in TOP10:
            error_gain=point[basei]-point[i];cost_increment=means[i]-means[basei];numerator=br[:,basei]-br[:,i];denominator=bt[:,i]-bt[:,basei];nonzero=denominator!=0
            signstable=bool(nonzero.all() and (np.all(denominator>0) or np.all(denominator<0)))
            estimate=float(error_gain/cost_increment) if cost_increment!=0 else None
            interval=np.quantile(numerator[nonzero]/denominator[nonzero],[.025,.975]).tolist() if signstable and np.isfinite(numerator).all() else None
            eta.append({'model':model,'reference_development_best_baseline':baseline,'score_orientation':'negative RMSE','RMSE_improvement':float(error_gain),'latency_increment_ms':float(cost_increment),'eta_score_improvement_per_latency_ms':estimate,'ci95_nominal':interval,'ratio_estimable_with_sign_stable_denominator':signstable,'beneficial_predictive_improvement_observed':bool(error_gain>0),'interpretation':'Negative/zero improvement or sign-changing cost denominator is not a claim of beneficial efficiency'})
    return {'equivalence19':_holm_rows(eq,alpha=.01,family_size=19),'cost_per_improvement':eta,'scope':'separate exploratory equivalence family; not included in primaryglobal.05','development_selection':selection,'a_nonsignificant_difference_does_not_prove_equivalence':True}


def variance_partition(panel):
    """Balanced descriptive sum-of-squares decomposition, excluding fake seed replicas.

    Calendar/seed/model interactions remain in the remainder. This is explicitly
    not a fitted Gaussian mixed-effects model or independence claim about dates.
    """
    indices=[i for i,m in enumerate(panel['models']) if m not in DETERMINISTIC]
    losses=(panel['predictions'][indices][:,:,panel['eligible']]-panel['targets'][panel['eligible']][None,None,:])**2
    M,S,T=losses.shape;grand=losses.mean();total=float(((losses-grand)**2).sum())
    pieces={'model_fixed_main_effect':float(S*T*((losses.mean(axis=(1,2))-grand)**2).sum()),
            'training_seed_main_effect':float(M*T*((losses.mean(axis=(0,2))-grand)**2).sum()),
            'calendar_date_main_effect':float(M*S*((losses.mean(axis=(0,1))-grand)**2).sum())}
    pieces['interactions_and_remainder']=max(0.,total-sum(pieces.values()))
    return {'sum_of_squares':pieces,'fractions':{k:None if total==0 else v/total for k,v in pieces.items()},
            'model_count':M,'paired_seeds':S,'eligible_dates':T,'excluded_deterministic_references':DETERMINISTIC,
            'scale':'individual squared prediction error','scope':'descriptive orthogonal main-effectSSpartition; interactions included in remainder; not estimated population random-effect variances',
            'mixed_effects_model_not_falsely_claimed':True,'serial_dependence_handled_by_separate_temporal_inference':True}


def probability_of_superiority(panel, *, repetitions=9999):
    sq=(panel['predictions']-panel['targets'][None,None,:])**2;loss=sq.mean(axis=1);mask=panel['eligible']
    weights=_counts(len(mask),repetitions,7,np.random.default_rng(202611071))*mask[None,:];den=weights.sum(axis=1)
    rows=[]
    for reference in REFERENCES:
        b=panel['models'].index(reference)
        for model in TOP10:
            i=panel['models'].index(model);indicator=(loss[i]<loss[b]).astype(float)+.5*(loss[i]==loss[b]);point=float(indicator[mask].mean())
            with np.errstate(invalid='ignore',divide='ignore'):boot=(weights@indicator)/den
            valid=boot[np.isfinite(boot)];rows.append({'reference':reference,'model':model,'probability_smaller_mean_individual_squared_loss_on_date':point,'ci95_temporal_nominal':np.quantile(valid,[.025,.975]).tolist() if len(valid)>=.95*repetitions else None,'ties_weight':.5,'no_new_hypothesis_decision':True})
    return rows


def _hac_error(series,bandwidth=7):
    x=np.asarray(series,float);_require(np.isfinite(x).all(),'Nonfinite influence series');center=x-x.mean();n=len(x);v=float(center@center/n)
    for lag in range(1,min(bandwidth,n-1)+1):v+=2*(1-lag/(bandwidth+1))*float(center[lag:]@center[:-lag]/n)
    return math.sqrt(v/n) if v>0 else None


def alternative_HAC_sensitivity(panel,timings,primary_rows,prereg):
    """Delta-method/HAC alternative to the registered primary block bootstrap.

    Exact mean-seed-RMSE derivative; not DM on MSE. No OR-of-tests decision.
    """
    point,hi,rest,sq=_point(panel);T=len(panel['eligible']);_require(panel['eligible'].all(),'HAC influence diagnostics require dense eligible calendar')
    def influences(mask):
        n=int(mask.sum())
        if not n:return np.full((len(MODELS),T),np.nan)
        mse=sq[:,:,mask].mean(axis=-1);rmse=np.sqrt(mse)
        with np.errstate(invalid='ignore',divide='ignore'):psi=((sq-mse[:,:,None])/(2*rmse[:,:,None])).mean(axis=1)
        return psi*mask[None,:]*T/n
    allpsi=influences(panel['eligible']);highpsi=influences(panel['high']);restpsi=influences(~panel['high']);results=[]
    primary={r['id']:r for r in primary_rows if r['block_days']==7}
    for family,reference in zip(prereg['families'],REFERENCES):
        bi=MODELS.index(reference);records=[]
        for model in TOP10:
            mi=MODELS.index(model);g=point[bi]-point[mi];d=(rest[bi]-rest[mi])-(hi[bi]-hi[mi]);c=float((timings[mi]-timings[bi]).mean())
            def pvalue(effect,influence):
                try:se=_hac_error(influence)
                except ValueError:return None,None
                return (None,None) if se is None else (float(stats.norm.sf((effect-DELTA)/se)),se)
            pg,se_g=pvalue(g,allpsi[bi]-allpsi[mi]);pd,se_d=pvalue(d,(restpsi[bi]-restpsi[mi])-(highpsi[bi]-highpsi[mi]));cs=(timings[mi]-timings[bi]).std(ddof=1)/math.sqrt(61);pc=float(stats.t.cdf(c/cs,60)) if cs>0 else None
            for rq,pv,se in [('RQ1',pg,se_g),('RQ2',pd,se_d),('RQ3',None if pg is None or pc is None else max(pg,pc),None)]:records.append({'id':f"{family['id']}:{rq}:{model}",'model':model,'reference':reference,'RQ':rq,'p':pv,'HAC_standard_error':se})
        adjusted=_holm_rows(records,alpha=.02,family_size=30)
        for row in adjusted:
            row['primary_bootstrap_reject']=primary[row['id']]['reject'];row['decision_same_as_primary']=row['reject']==row['primary_bootstrap_reject'];row['sensitivity_only_not_additional_primary_rejection_opportunity']=True
        results.extend(adjusted)
    return results



def full_cost_summary(rows,timings,point,bootstrap):
    rows=list(rows);keys=['parameters','FLOPs_per_window','peak_memory_bytes','training_seconds','hpo_seconds','throughput_samples_sec'];summaries=[]
    for i,model in enumerate(MODELS):
        selected=[r for r in rows if r['model']==model];record={'model':model,'latency_ms_mean':float(timings[i].mean()),'latency_ms_SD_blocks':float(timings[i].std(ddof=1))}
        for key in keys:
            values=[float(r[key]) for r in selected if r.get(key) is not None]
            _require(all(math.isfinite(v) and v>=0 for v in values),'Cost metadata must be nonnegative finite or missing')
            _require(not values or all(math.isclose(v,values[0],rel_tol=1e-10) for v in values),'Per-model resource metadata changed between timing blocks')
            record[key]=values[0] if len(values)==len(selected) else None
        record['timing_batch_size']=1;record['hpo_seconds_kind']=selected[0].get('hpo_seconds_kind','not_reported');record['memory_scope']='reported process peak; not pure model tensor bytes'
        summaries.append(record)
    axes=['latency_ms_mean','parameters','FLOPs_per_window','peak_memory_bytes','training_seconds','hpo_seconds'];pareto=[]
    B=min(9999,len(bootstrap));rng=np.random.default_rng(202611072);bt=_timing_bootstrap(timings,B,rng)
    for axis in axes:
        valid=[i for i,r in enumerate(summaries) if r[axis] is not None]
        for i,model in enumerate(MODELS):
            if i not in valid:
                pareto.append({'model':model,'axis':axis,'nondominated':None,'stability':None,'status':'missing_cost_not_zero'});continue
            cost=summaries[i][axis];dom=any(point[j]<=point[i] and summaries[j][axis]<=cost and (point[j]<point[i] or summaries[j][axis]<cost) for j in valid if j!=i)
            dominated_draws=np.zeros(B,bool)
            for j in valid:
                if j==i:continue
                c_i=bt[:,i] if axis=='latency_ms_mean' else cost;c_j=bt[:,j] if axis=='latency_ms_mean' else summaries[j][axis]
                dominated_draws|=(bootstrap[:B,j]<=bootstrap[:B,i])&(c_j<=c_i)&((bootstrap[:B,j]<bootstrap[:B,i])|(c_j<c_i))
            pareto.append({'model':model,'axis':axis,'nondominated':not dom,'stability':float((~dominated_draws).mean()),'stability_draws':B,'scope':'descriptive among models with measured comparable cost; fixed parameter/FLOP counts have no invented timing uncertainty','models_with_comparable_cost':len(valid),'not_population_superiority':True})
    return summaries,pareto


def analyze_final(panel,train_valid_panel,cost_panel,prereg,output=None, *, fixture=False,repetitions=None,efficiency_selection=None,primary_training_panel=None):
    """Analyze complete closed20-model final panels, exporting aggregates only.

    primary_training_panel is inference from the actual final refit checkpoint.
    Legacy train_valid_panel, when supplied, is an appendix, never a proxy for
    primary train/test gaps. Formal A53 diagnostics use analyze_A53_overfit.
    """
    _settings(prereg);cost_panel=list(cost_panel);train_valid_panel=list(train_valid_panel);legacy_inner=None;p=_panel(panel,kind='final',fixture=fixture);timings=_costs(cost_panel,MODELS,phase='primary_final',checkpoint_hashes={m:{r['checkpoint_sha256'] for r in p['rows'] if r['model']==m} for m in MODELS})
    B=99999 if repetitions is None else int(repetitions)
    _require(B==99999 or (fixture and B>=999),'Production final analysis requires99999 registered resamples')
    point,hi,rest,sq=_point(p);allfamilies=[];allaggregates=[];bootprimary=None
    for block in [7,14,28]:
        boot=_boot_scores(p,repetitions=B,block=block,seed=prereg['temporal_inference']['bootstrap_seed']);secondary=_secondary_bootstrap(p,repetitions=B,block=block,seed=prereg['temporal_inference']['bootstrap_seed']);families=_families(p,boot,timings,prereg,repetitions=B,seed=prereg['temporal_inference']['bootstrap_seed'])
        for row in families:row.update(block_days=block,primary_decision=block==7)
        allfamilies.extend(families)
        for i,m in enumerate(MODELS):
            vals=boot[0][:,i];good=vals[np.isfinite(vals)]
            allaggregates.append({'model':m,'block_days':block,'mean_seed_RMSE':float(point[i]),'ci95_temporal_nominal':np.quantile(good,[.025,.975]).tolist() if len(good)>=.95*B else None,'high_RMSE':float(hi[i]),'rest_RMSE':float(rest[i]),'high_minus_rest':float(hi[i]-rest[i]),'secondary_metric_temporal_CIs':secondary[m]})
        if block==7:bootprimary=boot
    metrics=[];paired=[]
    for i,model in enumerate(MODELS):
        seeds=['deterministic'] if model in DETERMINISTIC else p['seeds']
        for j,seed in enumerate(seeds):
            y=p['targets'][p['eligible']];prediction=p['predictions'][i,j,p['eligible']]
            metrics.append({'model':model,'seed':seed,**regression_metrics(y,prediction,mase_scale=p['mase_scale'][model,str(seed)]),**volatility_scores(y,prediction)})
            if np.ptp(y)==0:metrics[-1]['R2']=None
    for record in allaggregates:
        values=[r for r in metrics if r['model']==record['model']]
        record['randomized_training_runs']=0 if record['model'] in DETERMINISTIC else len(values)
        for key in ['RMSE','MAE','MSE','R2','MAPE_percent','MAPE_coverage','SMAPE_percent','SMAPE_coverage','MASE','QLIKE_log_variance_plus_ratio','MSE_realized_variance']:
            finite=[r[key] for r in values if r.get(key) is not None and math.isfinite(r[key])]
            record['mean_'+key]=float(np.mean(finite)) if len(finite)==len(values) else None
            record['SD_seed_'+key]=float(np.std(finite,ddof=1)) if len(finite)>1 and len(finite)==len(values) else None
    for reference in REFERENCES:
        b=MODELS.index(reference)
        for model in TOP10:
            i=MODELS.index(model);d=np.sqrt(sq[b][:,p['eligible']].mean(axis=-1))-np.sqrt(sq[i][:,p['eligible']].mean(axis=-1))
            paired.append({'reference':reference,'model':model,**paired_seed_diagnostics(d,unit_scope='training_seeds_conditional_fixed_calendar',repetitions=4999,seed=202611068)})
    diagnostics=_diagnostics(p,prereg)
    if primary_training_panel is not None:
        trainrows=list(primary_training_panel);combined=[];checkpoints={(r['model'],str(r['seed'])):r['checkpoint_sha256'] for r in p['rows']}
        for raw in trainrows:
            _require(raw['source_stage']=='E2_primary_train_inference','Primary training inference has wrong stage')
            r=dict(raw);_require(str(r.get('outcome_day',r.get('origin')))[:10]<='2024-12-24','Primary training inference includes future labels');r.update(split='train',source_stage='E2_primary_train_test_descriptive');combined.append(r)
        for raw in p['rows']:
            r=dict(raw);r.update(split='validation',source_stage='E2_primary_train_test_descriptive');combined.append(r)
        overfit=_overfit(combined,expected_seeds=p['seeds'],source_stage='E2_primary_train_test_descriptive',validation_calendar=_calendar('2026-09-01','2026-10-30'),expected_checkpoints=checkpoints)
    elif train_valid_panel:
        legacy_inner=_overfit(train_valid_panel,prereg,expected_seeds=p['seeds'])
        overfit={'scope':'primary train/test gap is not estimated from early-stopping inner weights','formal_hypotheses':0}
    else:overfit={'scope':'primary train/test descriptive gap requires frozen refit training inference','formal_hypotheses':0}

    decomposition=variance_partition(p);superiority=probability_of_superiority(p);alternative=alternative_HAC_sensitivity(p,timings,allfamilies,prereg) if p['eligible'].all() else {'status':'not_estimable_non_dense_calendar'}
    meanloss=sq.mean(axis=1);mcs=model_confidence_set(meanloss,MODELS,repetitions=9999,block=7) if p['eligible'].all() else {'status':'not_estimable','reason':'Calendar contains missing evaluation dates; no compression allowed'}
    efficiency=efficiency_equivalence(p,bootprimary,timings,efficiency_selection) if efficiency_selection is not None else {'status':'pending_frozen_development_selection'}
    rank=mean_paired_seed_ranks(p);costs,pareto=full_cost_summary(cost_panel,timings,point,bootprimary[0])
    result={'scope':'synthetic_fixture_not_market_result' if fixture else 'registered_exploratory_known_test_E2_final_analysis','primary_families':allfamilies,'diagnostics_199':diagnostics,'aggregate_metrics':allaggregates,'seed_metrics':metrics,'mean_paired_seed_ranks':rank,'paired_training_seed_diagnostics_exploratory':paired,'primary_train_test_gap_descriptive':overfit,'legacy_inner_gap_appendix_descriptive':legacy_inner,'descriptive_variance_partition':decomposition,'probability_of_superiority_descriptive':superiority,'HAC_alternative_sensitivity':alternative,'timing_summary':costs,'cost_Pareto_axes':pareto,'MCS_exploratory':mcs,'efficiency_equivalence_exploratory':efficiency,'global_independent_dataset_tests':{'Friedman_CD_status':'not_applicable_as_independent_dataset_test: oneBTCseries and overlapping training histories; seeds are not datasets','RM_ANOVA_status':'not_asserted without valid independent units/sphericity','Levene_status':'unpaired-group variance test not a required equal-marginal-variance assumption for paired differences; variances/paired-differenceShapiro reported'},'separate_branch_receipts_required':['Predictive PIT/coverage/calibration and their separate exploratory family are handled by uncertainty module','Predictability reference and gamma handled by dedicated registered module','Robustness and explanation/ablation require their closed branch inputs; missing comparable cost axes remain explicitly unknown'],'counts':{'models':20,'registered_R_per_stochastic_model':len(p['seeds']),'stochastic_models':16,'deterministic_references_stored_once':4,'calendar_dates':60,'eligible_dates':int(p['eligible'].sum()),'hypotheses_each_reference_family':30,'diagnostic_hypotheses':199,'bootstrap_draws':B},'primary_bootstrap_scope':'Calendar uncertainty conditional on frozen fitted trajectories; between-seed variability reported separately, not HPO-selection or market-population uncertainty','inference_limit':'Historical already observed test, singleBTCchronology, approximate block bootstrap/HAC/asymptotic residual tests; no confirmatory blinding restored','fits_performed':0,'market_data_acquired':False}
    result=clean(result)
    if output is not None:
        out=Path(output).resolve();_require(not out.is_relative_to(ROOT),'Actual analysis output must remain private outside repository');_require(not out.exists(),'Use a fresh private aggregate directory')
        out.mkdir(parents=True);payload=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n';(out/'final_statistics.json').write_text(payload)
        receipt={'analyzer_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),'primary_hypothesis_count':60,'diagnostic_count':199,'primary_B':B,'raw_predictions_or_checkpoints_exported':False,'fits_performed':0,'source_stage':p['stage'],'fixture':fixture,'input_canonical_panel_sha256':_rows_hash(p['rows']),'inner_canonical_panel_sha256':_rows_hash(train_valid_panel),'cost_canonical_panel_sha256':_rows_hash(cost_panel)}
        (out/'analysis_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return result


def compare_matched_ablation_errors(panel, *, repetitions=9999,fixture=False):
    """Practical two-sided component effects on the purged53-date development branch.

    Three variants versus their independently refitted full control. Positive
    ablated-minus-full RMSE means the removed component helped; negative means
    the lesion improved performance. Separate exploratory Holm3 alpha .05.
    """
    rows=list(panel);_require(rows,'Ablation predictions required');names=sorted({str(r.get('variant',r['model'])) for r in rows})
    controls=[n for n in names if n in ['control','full']]
    if not controls:
        controls=sorted({str(r.get('variant',r['model'])) for r in rows if r.get('comparison_role')=='control'})
    _require(len(names)==4 and len(controls)==1,'Exactly one full control and three registered ablations required')
    control=controls[0];models=[control]+[n for n in names if n!=control];seeds=sorted({int(r['seed']) for r in rows})
    _require(len(seeds) in R_GRID and seeds==FINAL_SEEDS[:len(seeds)],'Ablations require the identical registered final R seed prefix')
    _require(repetitions==9999 or (fixture and repetitions>=999),'Ablation production requires9999 resamples')
    calendar=_calendar('2024-11-08','2024-12-30');converted=[];closures=[];scorings=[];identity={};schedules=set();schedule_times=set();fit_starts=[];global_closures=set()
    for raw in rows:
        r=dict(raw);r['model']=str(r.get('variant',r['model']));r['origin_utc']=str(r.get('outcome_day',r.get('origin_utc')))[:10];r['horizon']=1;r['eligible']=r.get('eligible',True)
        _require(r['source_stage']==('synthetic_fixture_ablation' if fixture else 'E2_ablation_control_2024'),'Wrong ablation source stage')
        for field in ['fit_outcome_max','scaler_fit_outcome_max','configuration_selection_outcome_max']:_require(date.fromisoformat(str(r[field])[:10])<=date(2024,10,31),'Ablation development predictor has future training/selection data')
        for field in ['checkpoint_sha256','scaler_sha256','config_sha256']:_require(len(str(r[field]))==64 and set(str(r[field]))<=set('0123456789abcdef'),'Missing ablation frozen identity')
        key=(r['model'],str(r['seed']));checksum=tuple(r[f] for f in ['checkpoint_sha256','scaler_sha256','config_sha256'])
        _require(key not in identity or identity[key]==checksum,'Ablation predictor changed during evaluation');identity[key]=checksum
        closed=_stamp(r['checkpoint_closed_at_utc']);scored=_stamp(r['scoring_started_at_utc'])
        _require(closed<scored,'Each ablation checkpoint must close before its own development score')
        if not fixture:
            schedule_hash=str(r['ablation_schedule_sha256']);_require(len(schedule_hash)==64 and set(schedule_hash)<=set('0123456789abcdef'),'Ablation rows require the sealed complete schedule identity')
            schedules.add(schedule_hash);schedule_times.add(_stamp(r['ablation_schedule_closed_at_utc']));fit_started=_stamp(r['ablation_fit_started_at_utc']);fit_starts.append(fit_started)
            global_closures.add(_stamp(r['all_primary_A_weights_closed_at_utc']));_require(fit_started<=closed,'Ablation fit cannot start after checkpoint closure')
        closures.append(closed);scorings.append(scored);converted.append(r)
    if not fixture:
        _require(len(schedules)==len(schedule_times)==len(global_closures)==1,'Ablation schedule or primary/A closure changed during the fixed branch')
        _require(next(iter(schedule_times))<min(fit_starts),'Complete four-variant paired-seed schedule must close before the first ablation fit')
        _require(next(iter(global_closures))<min(scorings),'All primary and calibration predictors must close before first ablation development score')
    p=validate_dense_panel(converted,models,seeds,calendar);sq=(p['predictions']-p['targets'][None,None,:])**2;mask=p['eligible'];observed=np.sqrt(sq[:,:,mask].mean(axis=-1));point=observed.mean(axis=-1);sq[:,:,~mask]=0;results=[]
    for block in [7,14,28]:
        rng=np.random.default_rng(202611073);boot=np.empty((repetitions,4))
        for start in range(0,repetitions,256):
            stop=min(repetitions,start+256);counts=_counts(53,stop-start,block,rng)*mask[None,:];boot[start:stop]=_mean_seed_rmse(sq,counts).mean(axis=-1)
        family=[]
        for i,model in enumerate(models[1:],1):
            effect=float(point[i]-point[0]);draw=boot[:,i]-boot[:,0];positive=_one_sided(effect,draw,DELTA,alpha=.05/6);negative=_one_sided(effect,draw,-DELTA,side='less',alpha=.05/6)
            pvalue=None if positive['p'] is None or negative['p'] is None else min(1.,2*min(positive['p'],negative['p']))
            differences=observed[i]-observed[0];paired=paired_seed_diagnostics(differences,unit_scope='training_seeds_conditional_fixed_calendar',repetitions=4999,seed=202611074)
            family.append({'variant':model,'control':control,'effect_RMSE_ablated_minus_full':effect,'ci95_nominal':positive['ci95_nominal'],'family_lower_bound':positive['simultaneous_bound'],'family_upper_bound':negative['simultaneous_bound'],'p':pvalue,'p_greater_delta':positive['p'],'p_less_negative_delta':negative['p'],'delta':DELTA,'direction_descriptive':'removed_component_helped' if effect>0 else 'lesion_improved_error','paired_Cohen_dz_conditional_training':paired['paired_Cohen_dz'],'paired_Cohen_dz_ci95':paired['paired_Cohen_dz_ci95'],'block_days':block,'primary_decision':block==7,'n_unique_origins':int(mask.sum()),'randomized_seeds':len(seeds),'scope':'separate exploratory practical-ablation family; not jointprimary.05'})
        results.extend(_holm_rows(family,alpha=.05,family_size=3))
    return clean({'comparisons':results,'source_calendar':{'start':'2024-11-08','end':'2024-12-30','dates':53,'fit_selection_cutoff':'2024-10-31'},'hypotheses':3,'family_alpha':.05,'bootstrap_draws':repetitions,'primary_block_days':7,'sensitivity_blocks':[14,28],'null':'absolute mean-seed RMSE difference <= inheriteddelta','practical_two_sided_p':'min(1,2*min(one-sided upper practical-boundary p, one-sided lower practical-boundary p))','development_streaming_allowed_after_global_primary_A_closure_and_presealed_complete_schedule':True,'ablation_schedule_sha256':next(iter(schedules)) if schedules else None,'not_new_test_or_primary20model_comparison':True,'power_for_this53date_branch_not_certified_by_final60planning':True,'fits_performed_by_analyzer':0})
