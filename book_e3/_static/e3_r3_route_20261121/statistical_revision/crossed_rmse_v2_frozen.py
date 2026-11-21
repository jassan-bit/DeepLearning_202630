"""Development-only studentized crossed-RMSE inference, prepared with AI.

The estimand remains E_seed[sqrt(E_time squared error)]. This is a proposed
crossed-functional MCS extension, not classical MCS validity by nomenclature,
not a third primary pairwise procedure, and not an authorization to confirm.
"""
from dataclasses import dataclass
import numpy as np
from scipy import stats

VERSION = 'crossed-functional-bootstrap-t-v2'


@dataclass(frozen=True)
class Settings:
    B: int = 999
    mean_block: float = 12.
    hac_lag: int = 8
    alpha: float = .05
    mcs_alpha: float = .10
    batch: int = 16


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(q, names, candidate, settings):
    q = np.asarray(q, dtype=np.float64)
    require(q.ndim == 3 and q.shape[0] >= 10 and q.shape[1] >= 32
            and 2 <= q.shape[2] <= 32, 'Need aligned R>=10,T>=32,2<=M<=32')
    require(q.size <= 5_000_000 and np.isfinite(q).all() and (q >= 0).all(),
            'Finite nonnegative squared errors required')
    require(len(names) == q.shape[2] and len(set(names)) == len(names)
            and all(isinstance(x, str) and x for x in names), 'Unique model names required')
    require(isinstance(candidate, (int, np.integer)) and 0 <= candidate < len(names),
            'Invalid candidate')
    require(isinstance(settings.B, int) and settings.B >= 63
            and isinstance(settings.hac_lag, int) and 0 <= settings.hac_lag < q.shape[1]
            and 1 <= settings.mean_block <= q.shape[1]
            and isinstance(settings.batch, int) and 1 <= settings.batch <= 64
            and 0 < settings.alpha < .5 and 0 < settings.mcs_alpha < .5,
            'Invalid settings')
    mu = q.mean(axis=1)
    scale = float(mu.max())
    require(scale > 0 and (mu > 64*np.finfo(float).eps*scale).all(),
            'Zero or relatively near-zero MSE unsupported')
    return q


def stationary_indices(B, T, mean_block, rng):
    out = np.empty((B, T), dtype=np.int64)
    out[:, 0] = rng.integers(T, size=B)
    for t in range(1, T):
        restart = rng.random(B) < 1/mean_block
        out[:, t] = np.where(restart, rng.integers(T, size=B), (out[:, t-1]+1) % T)
    return out


def functional_and_covariance(q, lag):
    """Return risks and full seed+time HAC covariance for B x R x T x M.

    This plug-in covariance counts nonnegative interaction twice only under
    its additive crossed-effects linearization. It is NOT generally an upper
    bound under finite bandwidth, nonlinear roots or arbitrary dependence.
    """
    x = np.asarray(q, dtype=np.float64)
    require(x.ndim == 4 and x.shape[1] >= 2 and 0 <= lag < x.shape[2],
            'Expected B x R x T x M')
    B, R, T, M = x.shape
    mu = x.mean(axis=2)
    require(np.isfinite(mu).all() and (mu > 0).all(), 'Bootstrap zero-MSE sample unsupported')
    roots = np.sqrt(mu)
    risks = roots.mean(axis=1)
    centered = roots-risks[:, None, :]
    seed = np.matmul(centered.transpose(0, 2, 1), centered)/((R-1)*R)
    temporal = (x/(2*roots[:, :, None, :])-roots[:, :, None, :]/2).mean(axis=1)
    temporal -= temporal.mean(axis=1, keepdims=True)
    time = np.matmul(temporal.transpose(0, 2, 1), temporal)/T
    for k in range(1, lag+1):
        cov = np.matmul(temporal[:, k:].transpose(0, 2, 1), temporal[:, :-k])/T
        time += (1-k/(lag+1))*(cov+cov.transpose(0, 2, 1))
    time /= T
    covariance = seed+time
    covariance = (covariance+covariance.transpose(0, 2, 1))/2
    require(np.isfinite(risks).all() and np.isfinite(covariance).all(), 'Nonfinite risks/covariance')
    return risks, covariance, seed, time


def pair_scales(cov):
    d = np.diagonal(cov, axis1=-2, axis2=-1)
    variance = d[..., :, None]+d[..., None, :]-2*cov
    tolerance = 256*np.finfo(float).eps*np.max(np.abs(d), axis=-1)[..., None, None]
    require((variance >= -tolerance).all(), 'Negative pair variance beyond rounding')
    return np.sqrt(np.maximum(variance, 0.))


def mcs_from_pivots(risks, scales, pivots, names, alpha):
    """Global range elimination; exclusion never becomes a pairwise p-value."""
    M = len(risks)
    diffs = risks[:, None]-risks[None, :]
    magnitude = max(float(np.max(np.abs(risks))), np.finfo(float).tiny)
    floor = 64*np.finfo(float).eps*magnitude
    good = scales > floor
    unsupported = (~good) & (np.abs(diffs) > floor)
    np.fill_diagonal(unsupported, False)
    if unsupported.any() or not np.isfinite(pivots).all():
        return {'status':'unsupported_nonzero_or_nonfinite_pair', 'retained':list(names),
                'eliminated':[], 'initial_global_p':None, 'steps':[],
                'third_pairwise_p_provided':False}
    statistic = np.divide(diffs, scales, out=np.zeros_like(diffs), where=good)
    remaining = list(range(M)); steps=[]; eliminated=[]; previous=0.; initial=None
    while len(remaining) > 1:
        current = statistic[np.ix_(remaining, remaining)]
        observed = float(np.max(np.abs(current)))
        boot = np.max(np.abs(pivots[:, remaining][:, :, remaining]), axis=(1, 2))
        p = float((1+np.count_nonzero(boot >= observed))/(len(pivots)+1))
        if initial is None:
            initial = p
        monotone = max(previous, p)
        steps.append({'members':[names[i] for i in remaining], 'global_p':p,
                      'monotone_p':monotone})
        if monotone >= alpha:
            break
        # Same range-elimination rule as v1: do not change the selection rule.
        worst = remaining.pop(int(np.argmax(np.max(current, axis=1))))
        eliminated.append({'model':names[worst], 'exclusion_p':monotone})
        previous = monotone
    return {'status':'proposed_crossed_functional_MCS_computed', 'retained':[names[i] for i in remaining],
            'eliminated':eliminated, 'initial_global_p':initial, 'steps':steps,
            'third_pairwise_p_provided':False}


def holm(pvalues):
    p=np.asarray(pvalues, dtype=float)
    require(p.ndim == 1 and p.size and np.isfinite(p).all()
            and ((0 <= p) & (p <= 1)).all(), 'Invalid p family')
    order=np.argsort(p, kind='stable');out=np.empty_like(p)
    out[order]=np.minimum(1., np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    return out


def run(q, names, candidate, rng, settings=Settings(), legacy_mcs=None):
    q=validate(q, names, candidate, settings)
    R,T,M=q.shape
    risks,cov,seed,time=functional_and_covariance(q[None], settings.hac_lag)
    risks=risks[0]; scales=pair_scales(cov)[0]
    observed=risks[:, None]-risks[None, :]
    dates=stationary_indices(settings.B,T,settings.mean_block,rng)
    seeds=rng.integers(R,size=(settings.B,R))
    boot_risks=np.empty((settings.B,M));boot_scales=np.empty((settings.B,M,M))
    for start in range(0,settings.B,settings.batch):
        end=min(start+settings.batch,settings.B)
        draw=q[seeds[start:end,:,None],dates[start:end,None,:],:]
        rr,cc,_,_=functional_and_covariance(draw,settings.hac_lag)
        boot_risks[start:end]=rr;boot_scales[start:end]=pair_scales(cc)
    roots=(boot_risks[:,:,None]-boot_risks[:,None,:])-observed[None]
    magnitude=max(float(np.max(np.abs(risks))),np.finfo(float).tiny)
    floor=64*np.finfo(float).eps*magnitude
    obs_good=scales>floor;boot_good=boot_scales>floor
    unsupported_boot=(~boot_good)&(np.abs(roots)>floor)
    # Zero-variance nonzero bootstrap roots are not silently assigned t=0.
    # Conservatively retain every model if any such pair is encountered.
    pivots=np.divide(roots,boot_scales,out=np.zeros_like(roots),where=boot_good)
    fixed=np.divide(roots,scales[None],out=np.zeros_like(roots),where=obs_good[None])
    mcs=mcs_from_pivots(risks,scales,pivots,names,settings.mcs_alpha)
    if unsupported_boot.any():
        mcs={'status':'unsupported_bootstrap_studentizer', 'retained':list(names),
             'eliminated':[], 'initial_global_p':None, 'steps':[],
             'third_pairwise_p_provided':False}
    fixed_mcs=mcs_from_pivots(risks,scales,fixed,names,settings.mcs_alpha)
    rivals=np.array([i for i in range(M) if i != candidate])
    theta=risks[rivals]-risks[candidate]
    se=scales[rivals,candidate]
    centered=roots[:,rivals,candidate]
    p_basic=(1+np.count_nonzero(np.abs(centered)>=np.abs(theta)[None],axis=0))/(settings.B+1)
    hln=np.sqrt((T-1)/T)  # h=1 factor retained from v1; crossed validity is proposed.
    p2se=se/hln
    z=np.divide(theta,p2se,out=np.zeros_like(theta),where=se>floor)
    p_delta=2*stats.t.sf(np.abs(z),df=R-1)
    p_basic[se<=floor]=1.;p_delta[se<=floor]=1.
    # Finite-sample reference remains proposed; no classical DM theorem claimed.
    raw=np.column_stack([p_basic,p_delta]);adjusted=holm(raw.ravel()).reshape(raw.shape)
    critical=stats.t.ppf(1-settings.alpha/2,df=R-1)
    p1lo=theta-np.quantile(centered,1-settings.alpha/2,axis=0)
    p1hi=theta-np.quantile(centered,settings.alpha/2,axis=0)
    pair=[]
    for k,i in enumerate(rivals):
        supported=bool(se[k]>floor)
        pair.append({'rival':names[i], 'theta_hat':float(theta[k]), 'SE':float(se[k]),
                     'nondegenerate':supported,
                     'status':'proposed_inference_computed' if supported else 'degenerate_contrast_no_interval',
                     'P1_equal_two_sided_p':float(p_basic[k]), 'P2_equal_two_sided_p':float(p_delta[k]),
                     'P1_Holm':float(adjusted[k,0]), 'P2_Holm':float(adjusted[k,1]),
                     'P1_CI95':[float(p1lo[k]),float(p1hi[k])] if supported else None,
                     'P2_SE':float(p2se[k]),
                     'P2_CI95':[float(theta[k]-critical*p2se[k]),float(theta[k]+critical*p2se[k])] if supported else None,
                     'seed_variance':float(pair_scales(seed)[0,i,candidate]**2),
                     'time_variance':float(pair_scales(time)[0,i,candidate]**2)})
    out={'version':VERSION, 'shape':list(q.shape), 'risks':risks.tolist(), 'pairwise':pair,
            'MCS_bootstrap_t':mcs,'MCS_fixed_studentizer_development_control':fixed_mcs,
            'bootstrap_studentizers_recomputed':settings.B,
            'unsupported_bootstrap_pairs':int(unsupported_boot.sum()),
            'P2_HLN_h1_factor':float(hln),'P2_reference_df':R-1,
            'pairwise_methods':2,'third_primary_method_accredited':False,
            'confirmed_valid_for_market':False,'R3_accredited':False,
            'new_market_fits':0,'new_market_predictions':0,'AI_assistance_declared':True}
    if legacy_mcs is not None:
        out['MCS_legacy_frozen_control_same_draws']=legacy_mcs(risks,boot_risks,names,settings.mcs_alpha)
    return out
