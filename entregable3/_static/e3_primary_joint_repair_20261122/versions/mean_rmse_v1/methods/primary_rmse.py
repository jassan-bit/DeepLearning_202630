"""Paired temporal inference for mean_s sqrt(mean_t squared_error_s,t).

New version; the earlier pooled-MSE battery remains immutable. The three
procedures are a smooth-functional scalar delta/HAC-HLN test, exact-functional
stationary bootstrap-t, and nuisance-adjusted conditional functional GMM.
They condition on the fitted seed panel; seeds*time are never iid units.

HAC/fixed-b reference arithmetic is adapted from the project's frozen
engine/statistics.py (c54a2fd, SHA a14e20b7...). All nonlinearity, nuisance
correction, bootstrap kernel and decision wrapper here are versioned additions.
AI assistance: implementation and mathematical/software review. No market I/O,
forecast model fitting, or final-data access occurs in this module.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Sequence

import numpy as np

PROTOCOL_VERSION = 'temporal-fixed-seeds-mean-rmse-v1'
METHODS = ('delta_hac_hln','stationary_bootstrap_t','conditional_functional_gmm')
EXPECTED_SYSTEMS = ('input_only_mse','FCN-1D','ResNet-1D','Inception-1D','LSTM-autoencoder con predictor',
    'TimesNet','TSMixer','TiDE','TimeMixer','PatchTST','iTransformer','MLP16','LSTM32','GRU32',
    'Persistence','SeasonalNaive','ARIMA','additive_positive_head','floor_only','floor_plus_mean_excess','global_mse')


def holm(p_values):
    p=np.asarray(p_values,dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Finite [0,1] p-values required')
    order=np.argsort(p,kind='stable');result=np.empty_like(p)
    result[order]=np.minimum(1.,np.maximum.accumulate(p[order]*np.arange(len(p),0,-1)))
    return result.tolist()


def _windows_last(values,lags):
    a=np.asarray(values,dtype=float)
    a=np.ascontiguousarray(a-a.mean(axis=-1,keepdims=True))
    padded=np.pad(a,((0,0),(0,0),(lags,lags)))
    cumulative=np.concatenate([np.zeros((*a.shape[:2],1)),np.cumsum(padded,axis=-1)],axis=-1)
    return cumulative[...,lags+1:]-cumulative[...,:-lags-1]


def hac_covariance(values,lags):
    a=np.asarray(values,dtype=float)
    if a.ndim==1:a=a[:,None]
    if a.ndim!=2 or not np.isfinite(a).all() or not 0<=lags<len(a):
        raise ValueError('Finite [time,moments] array and valid lag required')
    windows=_windows_last(a.T[None],lags)[0].T
    return windows.T@windows/(len(a)*(lags+1))


@lru_cache(maxsize=24)
def fixed_b_reference(n,lags,q,horizon,reps,seed):
    """Frozen Gaussian FCLT reference, not an exact finite-sample market law."""
    rng=np.random.default_rng(np.random.SeedSequence([seed,n,lags,q,horizon,reps]))
    values=np.empty(reps)
    hln=math.sqrt((n+1-2*horizon+horizon*(horizon-1)/n)/n)
    for start in range(0,reps,128):
        count=min(128,reps-start)
        draw=rng.standard_normal((count,q,n))
        mean=draw.mean(axis=-1)
        windows=_windows_last(draw,lags)
        cov=np.einsum('bqt,brt->bqr',windows,windows)/(n*(lags+1))
        if q==1:values[start:start+count]=np.abs(mean[:,0]/np.sqrt(cov[:,0,0]/n)*hln)
        else:values[start:start+count]=n*np.einsum('bi,bi->b',mean,np.linalg.solve(cov,mean[...,None])[...,0])
    values.sort();values.flags.writeable=False
    return values,hashlib.sha256(values.tobytes()).hexdigest()


def mc_p(statistic,reference,name):
    count=int(len(reference)-np.searchsorted(reference,abs(statistic),side='left'))
    return dict(raw_p=(count+1)/(len(reference)+1),mc_tail_count=count,mc_reps=len(reference),
                mc_p_rule='absolute_tail_plus_one',reference=name)


def stationary_indices(n,reps,block_length,rng):
    if n<2 or reps<1 or not 1<=block_length<=n:
        raise ValueError('Invalid stationary-bootstrap dimensions')
    out=np.empty((reps,n),dtype=np.int32);out[:,0]=rng.integers(n,size=reps)
    for t in range(1,n):
        out[:,t]=np.where(rng.random(reps)<1/block_length,rng.integers(n,size=reps),(out[:,t-1]+1)%n)
    return out


@lru_cache(maxsize=8)
def _cached_stationary_indices(n,reps,block_length,seed):
    """Exact memoization: same arguments produce the same readonly index bytes.

    This changes no draw or result compared with the uncached PCG generator.
    A validation protocol may predeclare fixed inner seeds; then its Monte
    Carlo assesses data variation conditional on that deterministic algorithm.
    """
    indices=stationary_indices(n,reps,block_length,np.random.default_rng(seed))
    identity=hashlib.sha256(indices.tobytes()).hexdigest()
    indices.flags.writeable=False
    return indices,identity


def functional_and_influence(losses):
    """Exact empirical theta and centered delta influence [model,time]."""
    a=np.asarray(losses,dtype=float)
    if a.ndim!=3 or not np.isfinite(a).all() or np.any(a<0):
        raise ValueError('Finite nonnegative losses[models,seeds,time] required')
    mu=a.mean(axis=-1)
    if np.any(mu<=0):raise ValueError('Strictly positive seed risks required for smooth-functional inference')
    rho=np.sqrt(mu);theta=rho.mean(axis=1)
    psi=((a-mu[...,None])/(2*rho[...,None])).mean(axis=1)
    return theta,psi,mu,rho


def conditional_moments(losses,instruments,model,practical_factor=1.,_components=None):
    """Estimate g=E[z(a h_model-h_candidate)] and its full nuisance IF.

    h_j=theta_j+psi_j has E[h_j]=theta_j. Mu is estimated from the same
    temporal sample. The D_j*(loss-mu) term is essential for nonconstant z.
    The vector null is stronger than equality of the scalar primary metric.
    """
    theta,psi,mu,rho=functional_and_influence(losses) if _components is None else _components
    z=np.asarray(instruments,dtype=float)
    h=theta[:,None]+psi
    score=practical_factor*h[model]-h[0]
    g=(z*score[:,None]).mean(axis=0)
    seeds=losses.shape[1]
    derivatives=[];corrections=[]
    for j in (0,model):
        centered=losses[j]-mu[j,:,None]
        # Algebraically -(E[z*(loss-mu)])/(4 R mu^(3/2)), scaled to avoid
        # needlessly cubing very small market risk units.
        derivative=-(z.T@(centered/mu[j,:,None]).T)/(4*len(z)*seeds*rho[j][None,:])
        derivatives.append(derivative)
        corrections.append((derivative@centered).T)
    influence=z*score[:,None]-g[None,:]+practical_factor*corrections[1]-corrections[0]
    return dict(moment_mean=g,influence=influence,score=score,
                candidate_risk_derivative=derivatives[0],rival_risk_derivative=derivatives[1],
                raw_moment_influence_without_nuisance=z*score[:,None]-g[None,:])


@lru_cache(maxsize=1)
def _native_kernel():
    source=Path(__file__).with_name('primary_bootstrap.cpp')
    compiler=shutil.which('c++')
    if compiler is None or not source.is_file():return None
    flags=b'-O3 -std=c++11 -shared -fPIC'
    identity=hashlib.sha256(source.read_bytes()+flags).hexdigest()
    folder=Path(tempfile.mkdtemp(prefix='primary-rmse-bootstrap-'))
    artifact=folder/('kernel-'+identity+'.so')
    try:
        subprocess.run([compiler,'-O3','-std=c++11','-shared','-fPIC',str(source),'-o',str(artifact)],
                       check=True,capture_output=True,timeout=60)
        library=ctypes.CDLL(str(artifact));function=library.bootstrap_primary_rmse
        f64=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        i32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
        function.argtypes=[f64,i32,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,
                          ctypes.c_double,f64,f64,f64,f64,f64]
        function.restype=None
        return function,identity
    except (OSError,subprocess.SubprocessError):return None


def exact_bootstrap_reduction(losses):
    """Compress only byte-verified equal traces; retain all statistical slots.

    Joint seed columns are reduced only if EVERY equivalence class has the
    SAME multiplicity. Otherwise every original seed is retained; no weights
    or approximations are introduced. Candidate stays a separate channel,
    even when a rival is identical, so its degenerate contrasts remain explicit.
    """
    values=np.ascontiguousarray(losses,dtype=float)
    models,seeds,n=values.shape
    seed_groups=[];seed_hashes={}
    for r in range(seeds):
        trace=np.ascontiguousarray(values[:,r,:]);key=hashlib.sha256(trace.tobytes()).digest()
        found=None
        for group in seed_hashes.get(key,[]):
            if np.array_equal(trace,values[:,seed_groups[group][0],:]):found=group;break
        if found is None:
            found=len(seed_groups);seed_groups.append([]);seed_hashes.setdefault(key,[]).append(found)
        seed_groups[found].append(r)
    equal_counts=len({len(group) for group in seed_groups})==1
    retained_seeds=[group[0] for group in seed_groups] if equal_counts else list(range(seeds))
    reduced=values[:,retained_seeds,:]
    representatives=[0];inverse=[0];hashes={}
    for j in range(1,models):
        key=hashlib.sha256(np.ascontiguousarray(reduced[j]).tobytes()).digest();match=None
        for position in hashes.get(key,[]):
            if np.array_equal(reduced[j],reduced[representatives[position]]):match=position;break
        if match is None:
            match=len(representatives);representatives.append(j);hashes.setdefault(key,[]).append(match)
        inverse.append(match)
    metadata=dict(original_shape=list(values.shape),computed_shape=[len(representatives),len(retained_seeds),n],
        retained_seed_indices=retained_seeds,joint_seed_class_multiplicities=[len(group) for group in seed_groups],
        equal_multiplicity_seed_reduction=equal_counts and len(retained_seeds)<seeds,
        model_representatives=representatives,model_expansion_indices=inverse,
        exact_equality_verified=True,statistical_family_slots_removed=0)
    return np.ascontiguousarray(reduced[representatives]),np.asarray(inverse,dtype=int),metadata


def bootstrap_functional(losses,reps,block_length,lags,margin,seed,backend='auto',indices=None,compress_exact=True):
    """Recompute seed roots and influence/HAC in EVERY common-time resample."""
    original=np.ascontiguousarray(losses,dtype=float)
    if compress_exact:
        a,inverse,reduction=exact_bootstrap_reduction(original)
    else:
        a=original;inverse=np.arange(len(original));reduction=dict(original_shape=list(a.shape),computed_shape=list(a.shape),
            exact_equality_verified=True,statistical_family_slots_removed=0,compression_disabled=True)
    observed,_,_,_=functional_and_influence(a)
    models,seeds,n=a.shape
    if backend not in ('auto','numpy','native'):raise ValueError('Unknown bootstrap backend')
    native=_native_kernel() if backend!='numpy' else None
    if backend=='native' and native is None:raise RuntimeError('C++ kernel unavailable')
    if indices is None:
        idx,index_identity=_cached_stationary_indices(n,reps,block_length,int(seed))
    else:
        idx=np.ascontiguousarray(indices,dtype=np.int32);index_identity=hashlib.sha256(idx.tobytes()).hexdigest()
    if idx.shape!=(reps,n) or np.any((idx<0)|(idx>=n)):raise ValueError('Invalid fixed bootstrap indices')
    theta=np.empty((reps,models));td,tw,tl=(np.empty((reps,models-1)) for _ in range(3))
    factor=1-margin
    if native is not None:
        native[0](a,idx,reps,models,seeds,n,lags,factor,np.ascontiguousarray(observed),theta,td,tw,tl)
        return dict(theta=theta[:,inverse],epa_t=td[:,inverse[1:]-1],practical_t=tw[:,inverse[1:]-1],
                    log_ratio_t=tl[:,inverse[1:]-1],backend='native',kernel_sha256=native[1],exact_reduction=reduction,
                    bootstrap_indices_sha256=index_identity)
    chunk=max(1,min(16,(48*1024**2)//(models*seeds*n*8*4)))
    def studentize(num,var,scale):
        valid=var>np.finfo(float).eps*np.maximum(scale,np.finfo(float).tiny)*64
        result=np.full_like(num,np.inf);np.divide(num,np.sqrt(np.maximum(var,0)/n),out=result,where=valid)
        return result
    for start in range(0,reps,chunk):
        stop=min(reps,start+chunk)
        sample=a[:,:,idx[start:stop]].transpose(2,0,1,3)
        mu=sample.mean(axis=-1);rho=np.sqrt(mu)
        good=np.all(mu>0,axis=(1,2))
        means=rho.mean(axis=2);theta[start:stop]=means
        with np.errstate(divide='ignore',invalid='ignore'):
            psi=((sample-mu[...,None])/(2*rho[...,None])).mean(axis=2)
            windows=_windows_last(psi,lags)
            variance=np.sum(windows**2,axis=-1)/(n*(lags+1))
            cross=np.sum(windows*windows[:,:1,:],axis=-1)/(n*(lags+1))
            vc,vj,cv=variance[:,:1],variance[:,1:],cross[:,1:]
            td[start:stop]=studentize(means[:,1:]-means[:,:1]-(observed[1:]-observed[0]),vj+vc-2*cv,vj+vc)
            tw[start:stop]=studentize(factor*means[:,1:]-means[:,:1]-(factor*observed[1:]-observed[0]),factor**2*vj+vc-2*factor*cv,factor**2*vj+vc)
            varlog=vc/means[:,:1]**2+vj/means[:,1:]**2-2*cv/(means[:,:1]*means[:,1:])
            scalelog=vc/means[:,:1]**2+vj/means[:,1:]**2
            tl[start:stop]=studentize(np.log(means[:,:1]/means[:,1:])-np.log(observed[0]/observed[1:]),varlog,scalelog)
        td[start:stop][~good]=np.inf;tw[start:stop][~good]=np.inf;tl[start:stop][~good]=np.inf
    return dict(theta=theta[:,inverse],epa_t=td[:,inverse[1:]-1],practical_t=tw[:,inverse[1:]-1],
                log_ratio_t=tl[:,inverse[1:]-1],backend='numpy',kernel_sha256=None,exact_reduction=reduction,
                bootstrap_indices_sha256=index_identity)


def _unavailable(reason,point):
    return dict(status=reason,statistic=None,raw_p=1.,direction='candidate_better' if point>0 else 'candidate_worse' if point<0 else 'equal')


def _scalar_test(point,influence,lags,horizon,reference,bootstrap=False):
    scale=float(np.max(np.abs(influence)))
    normalized=influence/scale if scale else influence
    variance=float(hac_covariance(normalized,lags)[0,0])
    if variance<=np.finfo(float).eps*float(np.mean(normalized**2))*64 or scale==0:
        return _unavailable('degenerate_functional_long_run_variance',point)
    n=len(influence);se=scale*math.sqrt(variance/n)
    hln=1. if bootstrap else math.sqrt((n+1-2*horizon+horizon*(horizon-1)/n)/n)
    statistic=point/se*hln
    return dict(status='ok',statistic=statistic,standard_error=se,hln_factor=hln,
        direction='candidate_better' if point>0 else 'candidate_worse' if point<0 else 'equal',
        **mc_p(statistic,reference,'exact_functional_stationary_bootstrap_t' if bootstrap else 'smooth_RMSE_delta_hac_hln_fixed_b_scalar'))


def _conditional_test(losses,z,model,factor,lags,reference,components=None):
    if z is None:return _unavailable('predictable_nonconstant_instruments_not_supplied',0.)
    values=conditional_moments(losses,z,model,factor,components)
    g=values['moment_mean'];influence=values['influence']
    scale=float(np.max(np.abs(influence)))
    cov=hac_covariance(influence/scale if scale else influence,lags)
    eigen=np.linalg.eigvalsh(cov);largest=max(float(eigen.max()),np.finfo(float).tiny)
    if scale==0 or np.count_nonzero(eigen>largest*1e-10)<len(g):
        return _unavailable('singular_conditional_functional_covariance',float(g[0]))
    normalized=g/scale
    statistic=max(0.,float(len(z)*normalized@np.linalg.solve(cov,normalized)))
    return dict(status='ok',statistic=statistic,df=len(g),moment_means=g.tolist(),
        nuisance_adjustment='full_delta_GMM_risk_mean_influence',
        covariance_condition_number=float(eigen[-1]/eigen[0]),
        interpretation='Vector conditional-functional null is stronger than scalar primary equality; rejection alone is not directional superiority.',
        direction='candidate_better' if g[0]>0 else 'candidate_worse' if g[0]<0 else 'equal',
        **mc_p(statistic,reference,'smooth_functional_nuisance_adjusted_fixed_b_vector_wald'))


def _pivotal_ci(point,se,distribution,level):
    if not np.isfinite(distribution).all() or not np.isfinite(se):return None
    lo,hi=np.quantile(distribution,[(1-level)/2,(1+level)/2])
    return [float(point-hi*se),float(point-lo*se)]


def analyze(predictions,y,systems,**kwargs):
    values=np.asarray(predictions,dtype=float);target=np.asarray(y,dtype=float)
    if values.ndim!=3 or target.shape!=(values.shape[-1],) or not np.isfinite(values).all() or not np.isfinite(target).all():
        raise ValueError('Aligned finite predictions[models,seeds,time] and y[time] required')
    with np.errstate(over='raise'):
        losses=(values-target[None,None,:])**2
    return analyze_losses(losses,systems,**kwargs)


def analyze_losses(losses,systems,*,bootstrap_reps,block_length,hac_lags,instruments,
                   margin=.02,seed=2026112201,horizon=1,reference_reps=49999,
                   reference_seed=2026112202,bootstrap_backend='auto',alpha=.025):
    original=np.asarray(losses,dtype=float)
    if original.ndim!=3 or original.shape[0]<2 or original.shape[1]<1 or original.shape[2]<12 or not np.isfinite(original).all() or np.any(original<0):
        raise ValueError('Finite nonnegative losses[>=2systems,>=1fixedseeds,>=12dates] required')
    names=list(systems);models,seeds,n=original.shape
    if len(names)!=models or len(set(names))!=models or not all(isinstance(x,str) and x for x in names):
        raise ValueError('Unique ordered system names matching losses required')
    if not isinstance(bootstrap_reps,(int,np.integer)) or bootstrap_reps<99 or reference_reps<999:
        raise ValueError('B>=99 and fixed-b reference>=999 required; formal validation separately enforces family resolution')
    if not 1<=hac_lags<n//3 or not 2<=block_length<=n/3 or not 0<=margin<1 or not 0<alpha<=.05 or not 1<=horizon<n:
        raise ValueError('Invalid temporal settings, margin or alpha')
    z=None
    if instruments is not None:
        z=np.asarray(instruments,dtype=float)
        if z.ndim!=2 or z.shape[0]!=n or not 2<=z.shape[1]<n/5 or not np.isfinite(z).all() or not np.allclose(z[:,0],1.,rtol=0,atol=1e-12) or np.linalg.matrix_rank(z)!=z.shape[1]:
            raise ValueError('Full-rank predictable instruments with constant and nonconstant column required')
    unit=math.sqrt(float(original.max())) if original.max()>0 else 1.
    a=np.ascontiguousarray(original/(unit*unit))
    mu=a.mean(axis=-1);rho=np.sqrt(mu);theta=rho.mean(axis=1)
    regular=bool(np.all(mu>0))
    metrics={name:dict(mean_seed_rmse=float(theta[j]*unit),seed_rmse=(rho[j]*unit).tolist(),
        std_seed_rmse=float(np.std(rho[j],ddof=1)*unit) if seeds>1 else 0.,
        pooled_rmse_diagnostic_only=float(np.sqrt(mu[j].mean())*unit),
        mse_diagnostic_only=float(mu[j].mean()*unit*unit),
        unique_loss_seed_traces=int(len(np.unique(original[j],axis=0)))) for j,name in enumerate(names)}
    boot=None;psi=None;scalar_ref=vector_ref=None;scalar_sha=vector_sha=None;components=None
    if regular:
        components=functional_and_influence(a);theta,psi,_,_=components
        boot=bootstrap_functional(a,int(bootstrap_reps),block_length,hac_lags,margin,seed,bootstrap_backend)
        scalar_ref,scalar_sha=fixed_b_reference(n,hac_lags,1,horizon,int(reference_reps),reference_seed)
        if z is not None:vector_ref,vector_sha=fixed_b_reference(n,hac_lags,z.shape[1],horizon,int(reference_reps),reference_seed)
    comparisons=[];slots=[];factor=1-margin;gmm_cache={}
    model_representatives=list(range(models))
    if regular:
        for j,position in enumerate(boot['exact_reduction'].get('model_expansion_indices',range(models))):
            model_representatives[j]=boot['exact_reduction'].get('model_representatives',list(range(models)))[position]
    for j in range(1,models):
        point=float(theta[j]-theta[0]);practical=float(factor*theta[j]-theta[0])
        methods={}
        for method in METHODS:
            methods[method]={}
            for null,point_value,fac in (('epa',point,1.),('practical',practical,factor)):
                if not regular:test=_unavailable('nonpositive_seed_risk_nonsmooth_functional',point_value)
                elif method=='delta_hac_hln':test=_scalar_test(point_value,fac*psi[j]-psi[0],hac_lags,horizon,scalar_ref)
                elif method=='stationary_bootstrap_t':
                    distribution=boot['epa_t' if null=='epa' else 'practical_t'][:,j-1]
                    test=_scalar_test(point_value,fac*psi[j]-psi[0],hac_lags,horizon,np.sort(np.abs(distribution)),True)
                    test['invalid_bootstrap_studentizers']=int(np.count_nonzero(~np.isfinite(distribution)))
                else:
                    key=(model_representatives[j],fac)
                    if key not in gmm_cache:gmm_cache[key]=_conditional_test(a,z,key[0],fac,hac_lags,vector_ref,components)
                    test=dict(gmm_cache[key])
                methods[method][null]=test;slots.append(test)
        ci_diff=ci99_diff=ci_gain=ci99_gain=None;effect=None
        if regular:
            influence=psi[j]-psi[0]
            se=math.sqrt(max(0.,float(hac_covariance(influence,hac_lags)[0,0]))/n)
            sd=float(np.std(influence,ddof=1))
            effect=point/sd if sd>np.finfo(float).eps*max(float(np.max(np.abs(influence))),np.finfo(float).tiny)*64 else None
            ci_diff=_pivotal_ci(point,se,boot['epa_t'][:,j-1],.95)
            ci99_diff=_pivotal_ci(point,se,boot['epa_t'][:,j-1],.99)
            if ci_diff is not None:ci_diff=[x*unit for x in ci_diff]
            if ci99_diff is not None:ci99_diff=[x*unit for x in ci99_diff]
            log_point=math.log(theta[0]/theta[j])
            log_influence=psi[0]/theta[0]-psi[j]/theta[j]
            log_se=math.sqrt(max(0.,float(hac_covariance(log_influence,hac_lags)[0,0]))/n)
            for confidence in (95,99):
                log_ci=_pivotal_ci(log_point,log_se,boot['log_ratio_t'][:,j-1],confidence/100)
                converted=[1-math.exp(log_ci[1]),1-math.exp(log_ci[0])] if log_ci is not None and max(log_ci)<700 else None
                if confidence==95:ci_gain=converted
                else:ci99_gain=converted
        comparisons.append(dict(rival=names[j],mean_rmse_difference=point*unit,practical_mean=practical*unit,
            relative_rmse_gain=float(1-theta[0]/theta[j]) if theta[j]>0 else None,
            historical_mean_seed_rmse_favorable=bool(point>0),temporal_effect_cohen_d=effect,
            effect_definition='mean_RMSE_difference / SD(temporal_delta_pseudo_loss_difference); descriptive smooth-functional effect, not seed-population Cohen d',
            ci95_rmse_difference=ci_diff,ci99_rmse_difference=ci99_diff,
            ci95_relative_rmse_gain=ci_gain,ci99_relative_rmse_gain=ci99_gain,
            methods=methods,interval_method='exact_functional_stationary_bootstrap_t_reestimated_roots_and_HAC'))
    for test,adjusted in zip(slots,holm([x['raw_p'] for x in slots])):test['adjusted_p']=adjusted
    for c in comparisons:
        c['provisional_statistical_gate']=bool(c['mean_rmse_difference']>0 and c['practical_mean']>0
            and c['temporal_effect_cohen_d'] is not None and c['temporal_effect_cohen_d']>=.2
            and all(t['status']=='ok' and t['adjusted_p']<alpha for m in c['methods'].values() for t in m.values())
            and c['ci95_rmse_difference'] is not None and c['ci95_rmse_difference'][0]>0
            and c['ci99_rmse_difference'] is not None and c['ci99_rmse_difference'][0]>0
            and c['ci95_relative_rmse_gain'] is not None and c['ci95_relative_rmse_gain'][0]>margin
            and c['ci99_relative_rmse_gain'] is not None and c['ci99_relative_rmse_gain'][0]>margin)
    return dict(protocol_version=PROTOCOL_VERSION,estimand='theta_j = mean_fixed_seed sqrt(E_time squared_error_j,s,t); plug-in mean individual-seed RMSE',
        candidate=names[0],systems=names,n_dates=n,n_fixed_seeds=seeds,metrics=metrics,comparisons=comparisons,
        family=dict(tests=len(slots),rivals=models-1,methods=list(METHODS),nulls=['epa','practical'],alpha=float(alpha),
            adjustment='Holm across all rivals, three methods and both nulls within this panel',
            minimum_bootstrap_p=1/(bootstrap_reps+1),minimum_reference_p=1/(reference_reps+1)),
        settings=dict(bootstrap_reps=int(bootstrap_reps),block_length=float(block_length),hac_lags=int(hac_lags),
            horizon=int(horizon),margin_relative_mean_seed_RMSE=float(margin),seed=int(seed),alpha=float(alpha),
            reference_reps=int(reference_reps),reference_seed=int(reference_seed),
            scalar_reference_sha256=scalar_sha,vector_reference_sha256=vector_sha,gate_interval_confidence=.99),
        bootstrap_backend=None if boot is None else boot['backend'],
        bootstrap_kernel_sha256=None if boot is None else boot['kernel_sha256'],
        exact_bootstrap_reduction=None if boot is None else boot['exact_reduction'],
        bootstrap_indices_sha256=None if boot is None else boot['bootstrap_indices_sha256'],
        all_rivals_statistical_gate=all(c['provisional_statistical_gate'] for c in comparisons),
        R3_accredited=False,regular_positive_seed_risks=regular,
        limitations=['Temporal inference conditions on this fitted seed panel; it does not integrate unseen training randomness.',
            'Smoothness requires strictly positive risks; zero risks are conservatively unavailable, never epsilon-clamped.',
            'HAC/fixed-b references are FCLT approximations; finite validity is assessed only in the separately frozen Monte Carlo scenarios.',
            'Conditional functional GMM has a stronger null than scalar primary equality; it is an additional conjunction, not a directional replacement.',
            'Predictable instruments, weak dependence/stationarity and representative dates remain assumptions.',
            'CI95/99 are marginal temporal intervals, not simultaneous family confidence bands.',
            'These new procedures preserve the primary functional but are explicit smooth-functional extensions; names do not confer academic acceptance.'])


def statistical_rule(results,margin=.02,*,require_full_family=True):
    """Unchanged main-all + two-of-three-additional policy; no all-three guard."""
    if len(results)!=4:raise ValueError('The complete four-panel rule is required')
    rivals=[c['rival'] for c in results[0]['comparisons']]
    alpha=float(results[0]['family']['alpha'])
    if require_full_family and (results[0]['systems']!=list(EXPECTED_SYSTEMS) or rivals!=list(EXPECTED_SYSTEMS[1:]) or alpha!=.025 or margin!=.02):
        raise ValueError('Formal decision requires the frozen 21-system family, alpha .025 and margin .02')
    keys=('bootstrap_reps','block_length','hac_lags','horizon','margin_relative_mean_seed_RMSE','alpha','reference_reps','reference_seed')
    expected_settings={key:results[0]['settings'][key] for key in keys}
    for result in results:
        if result.get('protocol_version')!=PROTOCOL_VERSION or result['family']['methods']!=list(METHODS) or result['family']['nulls']!=['epa','practical']:
            raise ValueError('Primary-RMSE protocol/method version differs; pooled-MSE results cannot be reused')
        if {key:result['settings'][key] for key in keys}!=expected_settings or result['settings']['margin_relative_mean_seed_RMSE']!=margin or result['family']['alpha']!=result['settings']['alpha']:
            raise ValueError('Panel statistical settings or margin differ')
        if require_full_family and (result['systems']!=list(EXPECTED_SYSTEMS) or result['candidate']!=EXPECTED_SYSTEMS[0] or
                                   result['n_dates']!=results[0]['n_dates'] or result['n_fixed_seeds']!=results[0]['n_fixed_seeds'] or result['family']['tests']!=120):
            raise ValueError('Full-panel system identity, temporal size or fixed seed count differs')
        if any(c['interval_method']!='exact_functional_stationary_bootstrap_t_reestimated_roots_and_HAC' for c in result['comparisons']):
            raise ValueError('Confidence intervals are from another functional or method version')
        if any(set(c['methods'])!=set(METHODS) or any(set(tests)!={'epa','practical'} for tests in c['methods'].values()) for c in result['comparisons']):
            raise ValueError('Every contrast requires all three procedures and both nulls')
    if any(float(r['family']['alpha'])!=alpha for r in results):raise ValueError('Common frozen alpha required')
    passing=np.zeros((4,len(rivals)),dtype=bool);adverse=[];negative=[];method_order=[]
    for pi,result in enumerate(results):
        if [c['rival'] for c in result['comparisons']]!=rivals:raise ValueError('Common ordered complete rival family required')
        for ri,c in enumerate(result['comparisons']):
            passing[pi,ri]=bool(c['mean_rmse_difference']>0 and c['practical_mean']>0
                and c['temporal_effect_cohen_d'] is not None and c['temporal_effect_cohen_d']>=.2
                and all(c['methods'][m][null]['status']=='ok' and c['methods'][m][null]['adjusted_p']<alpha for m in METHODS for null in ('epa','practical'))
                and c['ci95_rmse_difference'] is not None and c['ci95_rmse_difference'][0]>0
                and c['ci99_rmse_difference'] is not None and c['ci99_rmse_difference'][0]>0
                and c['ci95_relative_rmse_gain'] is not None and c['ci95_relative_rmse_gain'][0]>margin
                and c['ci99_relative_rmse_gain'] is not None and c['ci99_relative_rmse_gain'][0]>margin)
            if pi:
                for m in METHODS:
                    t=c['methods'][m]['epa'];adverse.append(t['raw_p'] if t['status']=='ok' else 1.)
                    negative.append(c['mean_rmse_difference']<0);method_order.append(m)
    adjusted=holm(adverse)
    inferior=any(neg and p<alpha and m in ('delta_hac_hln','stationary_bootstrap_t') for neg,p,m in zip(negative,adjusted,method_order))
    additional=bool(np.all(passing[1:].sum(axis=0)>=2))
    return dict(contrast_pass=passing.tolist(),main_all=bool(passing[0].all()),
        additional_statistical_two_of_three=additional,additional_significant_inferiority=bool(inferior),
        additional_adverse_Holm_slots=len(adverse),additional_adverse_adjusted_p=adjusted,
        joint_statistical_rule_pass=bool(passing[0].all() and additional and not inferior),
        all_contrasts_pass=bool(passing.all()),full_family_enforced=bool(require_full_family),R3_accredited=False)
