"""Experimental crossed-seed/time RMSE inference and a global MCS complement.

Prepared with AI assistance. These are proposed extensions, not three accredited
pairwise R3 tests. No training, prediction, network or automatic promotion occurs.
Input squared errors must already be aligned on the same seed/date/model grid.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np
from scipy import stats

P1 = 'P1_CROSSED_STATIONARY_BOOTSTRAP_RMSE'
P2 = 'P2_CROSSED_DELTA_HAC_HLN_PROPOSED_EXTENSION'
P3 = 'P3_FUNCTIONAL_MCS_GLOBAL_PROPOSED_EXTENSION'


@dataclass(frozen=True)
class Settings:
    bootstrap_replicates: int = 2047
    stationary_mean_block: float = 12.0
    hac_lag: int = 8
    alpha: float = .05
    mcs_alpha: float = .10
    minimum_seeds: int = 10
    minimum_dates: int = 32


def require(ok, message):
    if not ok:
        raise ValueError(message)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def holm(pvalues, total_slots=None):
    """Holm with optional explicit missing slots reserved conservatively at p=1."""
    p = np.asarray(pvalues, dtype=float)
    require(p.ndim == 1 and p.size and np.isfinite(p).all()
            and ((p >= 0) & (p <= 1)).all(), 'Invalid p-value family')
    total = p.size if total_slots is None else total_slots
    require(isinstance(total, (int, np.integer)) and total >= p.size,
            'Family reservation cannot remove tests')
    padded = np.r_[p, np.ones(total - p.size)]
    order = np.argsort(padded, kind='stable')
    adjusted = np.minimum(1., np.maximum.accumulate((total - np.arange(total)) * padded[order]))
    result = np.empty(total)
    result[order] = adjusted
    return result[:p.size]


def bartlett_lrv(values, lag):
    """Diagonal Bartlett HAC LRV; denominator T for every covariance lag."""
    x = np.asarray(values, dtype=float)
    require(x.ndim == 2 and x.shape[0] > lag >= 0 and np.isfinite(x).all(),
            'HAC needs finite T x features and lag<T')
    x = x - x.mean(axis=0)
    T = x.shape[0]
    result = np.sum(x*x, axis=0) / T
    for k in range(1, lag + 1):
        result += 2 * (1 - k/(lag+1)) * np.sum(x[k:]*x[:-k], axis=0) / T
    tolerance = 1e-12 * np.maximum(1., np.sum(x*x, axis=0)/T)
    require((result >= -tolerance).all(), 'Unexpected negative Bartlett LRV')
    return np.maximum(result, 0.)


def validate_grid(squared_errors, names, candidate, margins, settings):
    q = np.asarray(squared_errors, dtype=np.float64)
    require(q.ndim == 3, 'Expected squared errors [seed,date,system]')
    S,T,M = q.shape
    require(type(settings.minimum_seeds) is int and settings.minimum_seeds >= 10
            and type(settings.minimum_dates) is int and settings.minimum_dates >= 32,
            'Configured inference minimum cannot relax R10/T32')
    require(S >= settings.minimum_seeds and T >= settings.minimum_dates and 2 <= M <= 64,
            'Insufficient seed/date/system grid')
    require(q.size <= 50_000_000 and np.isfinite(q).all() and (q >= 0).all(),
            'Squared errors must be finite, nonnegative and bounded in size')
    require(len(names) == M and len(set(names)) == M
            and all(isinstance(n,str) and n for n in names), 'Unique model names required')
    require(isinstance(candidate, (int,np.integer)) and 0 <= candidate < M, 'Bad candidate index')
    require(type(settings.bootstrap_replicates) is int and settings.bootstrap_replicates >= 31
            and 1 <= settings.stationary_mean_block <= T
            and type(settings.hac_lag) is int and 0 <= settings.hac_lag < T and 0 < settings.alpha < .5
            and 0 < settings.mcs_alpha < .5, 'Invalid fixed settings')
    rivals = np.array([i for i in range(M) if i != candidate],dtype=int)
    margins = np.broadcast_to(np.asarray(margins,dtype=float), (M-1,)).copy()
    require(np.isfinite(margins).all() and (margins >= 0).all(), 'Margins must be finite/nonnegative')
    mu = q.mean(axis=1)
    floor = 64*np.finfo(float).eps*max(1.,float(mu.max()))
    require(np.isfinite(mu).all() and (mu > floor).all(),
            'Near-zero MSE: RMSE derivative/inference unsupported')
    return q,rivals,margins,mu


def stationary_indices(B,T,mean_block,rng):
    indices = np.empty((B,T), dtype=np.int64)
    indices[:,0] = rng.integers(T,size=B)
    for t in range(1,T):
        restart = rng.random(B) < 1/mean_block
        indices[:,t] = np.where(restart,rng.integers(T,size=B),(indices[:,t-1]+1)%T)
    return indices


def crossed_bootstrap_metrics(q,settings,rng):
    """Same temporal indices and matched seed indices for all systems.

    This product/pigeonhole resampling is asymptotic for a nondegenerate crossed
    process; finite-S interaction overcount is a documented limitation, not IID
    resampling of S*T cells. Every draw recomputes sqrt of each seed's mean MSE.
    """
    S,T,M = q.shape
    B = settings.bootstrap_replicates
    dates = stationary_indices(B,T,settings.stationary_mean_block,rng)
    date_counts = np.zeros((B,T))
    np.add.at(date_counts,(np.arange(B)[:,None],dates),1.)
    seeds = rng.integers(S,size=(B,S))
    seed_counts = np.zeros((B,S))
    np.add.at(seed_counts,(np.arange(B)[:,None],seeds),1.)
    date_mean = (date_counts @ np.transpose(q,(1,0,2)).reshape(T,S*M) / T).reshape(B,S,M)
    require(np.isfinite(date_mean).all() and (date_mean >= 0).all(), 'Invalid bootstrap mean')
    return np.einsum('bs,bsm->bm',seed_counts,np.sqrt(date_mean),optimize=True)/S


def crossed_delta(q,mu,rivals,candidate,theta,margins,settings):
    S,T,M = q.shape
    root = np.sqrt(mu)
    seed_d = root[:,rivals]-root[:,[candidate]]
    psi = ((q[:,:,rivals]-mu[:,None,rivals])/(2*root[:,None,rivals])
           -(q[:,:,[candidate]]-mu[:,None,[candidate]])/(2*root[:,None,[candidate]]))
    temporal = psi.mean(axis=0)
    seed_variance = np.var(seed_d,axis=0,ddof=1)/S
    time_variance = bartlett_lrv(temporal,settings.hac_lag)/T
    residual = psi-temporal[None,:,:]
    interaction_lrv = bartlett_lrv(np.transpose(residual,(1,0,2)).reshape(T,-1),settings.hac_lag)
    interaction_variance = interaction_lrv.reshape(S,len(rivals)).mean(axis=0)/((S-1)*T)
    corrected_diagnostic = seed_variance+time_variance-interaction_variance
    # Under the declared additive crossed-effects linearization the addition
    # counts nonnegative idiosyncratic interaction twice. We retain that upper
    # variance, rather than selecting/clipping a corrected estimate for power.
    conservative_variance = seed_variance+time_variance
    hln = np.sqrt((T-1)/T)  # h=1 only; not a theorem for the crossed functional.
    se = np.sqrt(conservative_variance)/hln
    good = se > np.finfo(float).tiny
    statistic = np.zeros_like(theta)
    statistic[good] = (theta[good]-margins[good])/se[good]
    p = np.ones_like(theta)
    p[good] = stats.t.sf(statistic[good],df=S-1)
    critical = stats.t.ppf(1-settings.alpha/2,df=S-1)
    critical_one = stats.t.ppf(1-settings.alpha,df=S-1)
    return dict(p=p,se=se,statistic=statistic,lower=theta-critical*se,
                upper=theta+critical*se,lower_one=theta-critical_one*se,
                good=good,seed_variance=seed_variance,time_variance=time_variance,
                interaction_variance_diagnostic=interaction_variance,
                interaction_corrected_variance_diagnostic=corrected_diagnostic,
                variance_used=conservative_variance,HLN_h1_factor=float(hln),reference_df=S-1)


def functional_mcs(metrics,bootstrap,names,alpha):
    """Global equal-predictive-ability MCS extension for the full functional.

    Exclusion p-values concern the global elimination sequence. They are NEVER
    pairwise p(theta_rival - theta_candidate <= practical_margin).
    """
    B,M = bootstrap.shape
    observed = metrics[:,None]-metrics[None,:]
    raw_diff = bootstrap[:,:,None]-bootstrap[:,None,:]
    scales = np.std(raw_diff,axis=0,ddof=1)
    numeric_floor=64*np.finfo(float).eps*max(1.,float(np.max(np.abs(metrics))))
    valid = scales > numeric_floor
    unsupported = (~valid) & (np.abs(observed)>numeric_floor)
    np.fill_diagonal(unsupported,False)
    if unsupported.any():
        return dict(status='degenerate_nonzero_pair_unsupported',retained=list(names),
                    eliminated=[],initial_global_p=None,third_directional_p_provided=False)
    obs_t = np.divide(observed,scales,out=np.zeros_like(observed),where=valid)
    boot_t = np.divide(raw_diff-observed[None,:,:],scales[None,:,:],
                       out=np.zeros_like(raw_diff),where=valid[None,:,:])
    remaining = list(range(M));eliminated=[];previous=0.;initial=None;steps=[]
    while len(remaining)>1:
        ij = np.ix_(remaining,remaining)
        statistic = float(np.max(np.abs(obs_t[ij])))
        boot_stat = np.max(np.abs(boot_t[:,remaining][:,:,remaining]),axis=(1,2))
        p = float((1+np.count_nonzero(boot_stat>=statistic))/(B+1))
        if initial is None:initial=p
        monotone=max(previous,p)
        steps.append({'members':[names[i] for i in remaining],'global_p':p,'monotone_p':monotone})
        if monotone >= alpha:break
        worst_local=int(np.argmax(np.max(obs_t[ij],axis=1)))
        worst=remaining.pop(worst_local)
        eliminated.append({'model':names[worst],'model_exclusion_p':monotone})
        previous=monotone
    return dict(status='global_functional_extension_computed',confidence=1-alpha,
                retained=[names[i] for i in remaining],eliminated=eliminated,
                initial_global_p=initial,steps=steps,
                hypothesis='global equal predictive ability of retained full-functional risks',
                third_directional_p_provided=False)


def run_battery(squared_errors,names,candidate,margins,rng,settings=Settings()):
    q,rivals,margins,mu=validate_grid(squared_errors,names,candidate,margins,settings)
    metrics=np.sqrt(mu).mean(axis=0)
    theta=metrics[rivals]-metrics[candidate]
    bootstrap=crossed_bootstrap_metrics(q,settings,rng)
    boot_theta=bootstrap[:,rivals]-bootstrap[:,[candidate]]
    roots=boot_theta-theta[None,:]
    p1=(1+np.count_nonzero(roots >= (theta-margins)[None,:],axis=0))/(settings.bootstrap_replicates+1)
    numeric_floor=64*np.finfo(float).eps*max(1.,float(np.max(np.abs(metrics))))
    boot_good=np.std(roots,axis=0,ddof=1)>numeric_floor
    p1[~boot_good]=1.
    p1_lower=theta-np.quantile(roots,1-settings.alpha/2,axis=0)
    p1_upper=theta-np.quantile(roots,settings.alpha/2,axis=0)
    p1_one=theta-np.quantile(roots,1-settings.alpha,axis=0)
    delta=crossed_delta(q,mu,rivals,candidate,theta,margins,settings)
    raw=np.column_stack([p1,delta['p']])
    adjusted38=holm(raw.ravel()).reshape(raw.shape)
    reserved=holm(raw.ravel(),total_slots=3*len(rivals)).reshape(raw.shape)
    records=[]
    for k,model in enumerate(rivals):
        row={'rival':names[model],'theta_hat':float(theta[k]),'margin':float(margins[k])}
        row['methods']={P1:{'p_raw':float(p1[k]),'p_Holm_two_method_family':float(adjusted38[k,0]),
                             'p_Holm_reserved_third_slot':float(reserved[k,0]),
                             'CI95':[float(p1_lower[k]),float(p1_upper[k])],
                             'lower95_one_sided':float(p1_one[k]),'nondegenerate':bool(boot_good[k])},
                        P2:{'p_raw':float(delta['p'][k]),'p_Holm_two_method_family':float(adjusted38[k,1]),
                             'p_Holm_reserved_third_slot':float(reserved[k,1]),
                             'CI95':[float(delta['lower'][k]),float(delta['upper'][k])],
                             'lower95_one_sided':float(delta['lower_one'][k]),
                             'nondegenerate':bool(delta['good'][k])}}
        row['variance_components']={key:float(delta[key][k]) for key in
             ('seed_variance','time_variance','interaction_variance_diagnostic',
              'interaction_corrected_variance_diagnostic','variance_used')}
        records.append(row)
    return {'status':'proposed_two_pairwise_extensions_and_global_complement_computed',
            'shape':list(q.shape),'candidate':names[candidate],
            'metric':'mean of individual-seed RMSE; not pooled or ensemble RMSE',
            'pairwise':records,'global_MCS':functional_mcs(metrics,bootstrap,names,settings.mcs_alpha),
            'directional_pairwise_methods':2,'global_complement_methods':1,
            'three_tests_per_contrast_gate':False,'final_method_admissibility_approved':False,
            'complete_three_method_power_computed':False,'confirmation_enabled':False,
            'R3_accredited':False,'HLN_h1_factor':delta['HLN_h1_factor'],
            'reference_df':delta['reference_df'],
            'bootstrap_minimum_p':1/(settings.bootstrap_replicates+1),
            'new_market_fits':0,'new_market_predictions':0,'AI_assistance_declared':True}
