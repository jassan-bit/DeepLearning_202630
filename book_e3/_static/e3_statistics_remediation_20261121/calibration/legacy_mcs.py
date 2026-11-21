"""Verbatim functions from preserved temporal_battery.py; original source remains unchanged.
AI-assisted extraction; reference hashes are in LEGACY_REFERENCE.json.
"""
import numpy as np

def require(ok, message):
    if not ok: raise ValueError(message)


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

