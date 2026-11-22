"""Truth witnesses and inference-independent composition of the frozen rule."""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.stats import beta


def joint_rule(contrasts, adverse=False):
    """Principal all rivals and EACH rival in at least two additional panels."""
    flags=np.asarray(contrasts,dtype=bool)
    if flags.ndim!=2 or flags.shape[0]!=4 or flags.shape[1]<1:
        raise ValueError('Exactly four panels and a nonempty common rival family required')
    return bool(flags[0].all() and np.all(flags[1:].sum(axis=0)>=2) and not adverse)


def witness(truth):
    """Choose by population truth, never by observed p or decisions.

    A false principal assertion gives one panel/rival witness. Otherwise a
    rival with fewer than two truly positive additional panels gives two
    fixed additional null witnesses. Each contrast has its own EPA/practical
    nulls; the practical scalar null suffices when practical superiority fails.
    """
    truth=np.asarray(truth,dtype=bool)
    joint_rule(truth)
    bad=np.flatnonzero(~truth[0])
    if len(bad):
        return [(0,int(bad[0]))]
    bad=np.flatnonzero(truth[1:].sum(axis=0)<2)
    if not len(bad):
        return []
    rival=int(bad[0])
    panels=(np.flatnonzero(~truth[1:,rival])+1)[:2]
    return [(int(p),rival) for p in panels]


def composition_bound(panel_fwer=(.025,.025,.025,.025)):
    """Conditional theorem, not finite-sample validation of marginal p values."""
    q=np.asarray(panel_fwer,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or (q<0).any() or (q>1).any():
        raise ValueError('Four valid panel family-error bounds required')
    bound=max(float(q[0]),max(float(q[a]+q[b]) for a,b in itertools.combinations((1,2,3),2)))
    return min(1.,bound)


def verify_witnesses():
    """Exhaust all one-rival truth/decision combinations; lifting is by witness."""
    checked=0
    for population in itertools.product((False,True),repeat=4):
        truth=np.asarray(population)[:,None]
        if joint_rule(truth):
            continue
        chosen=witness(truth)
        for decision in itertools.product((False,True),repeat=4):
            flags=np.asarray(decision)[:,None]
            if not joint_rule(flags):
                continue
            if not any(flags[p,r] for p,r in chosen):
                raise AssertionError('The fixed null witness was not rejected')
            if not all(not truth[p,r] for p,r in chosen):
                raise AssertionError('Witness is not null under population truth')
            checked+=1
    return {'checked_false_truth_accepted_decision_pairs':checked,
            'no_independence_assumption':True,
            'formal_bound_if_panel_strong_FWER_is_0.025':composition_bound(),
            'does_not_derive_panel_FWER_from_old_CP_bounds':True,
            'scope':'direction/practical population assertions of the statistical rule; not all scientific R3 gates'}


def cp_interval(count,n,error=.05):
    if not 0<=count<=n or n<1 or not 0<error<1:
        raise ValueError('Invalid binomial count or confidence error')
    return [0. if count==0 else float(beta.ppf(error/2,count,n-count+1)),
            1. if count==n else float(beta.ppf(1-error/2,count+1,n-count))]


def cp_upper(count,n,error):
    if not 0<=count<=n or n<1 or not 0<error<1:
        raise ValueError('Invalid binomial count or confidence error')
    return 1. if count==n else float(beta.ppf(1-error,count+1,n-count))


def cp_lower(count,n,error):
    if not 0<=count<=n or n<1 or not 0<error<1:
        raise ValueError('Invalid binomial count or confidence error')
    return 0. if count==0 else float(beta.ppf(error,count,n-count+1))


def binomial_capacity(n=512,scenarios=9,total_claims=107):
    """Calculate MC precision before any formal replicate is observed."""
    a=.05/total_claims
    allowed=[k for k in range(n+1) if cp_lower(n-k,n,a)>=.95]
    return dict(replicates=n,simultaneous_total_claims=total_claims,simultaneous_error_each=a,
        upper_zero_global_errors=cp_upper(0,n,a),
        lower_perfect_strong_joint_power=cp_lower(n,n,a),
        lower_perfect_80_contrast_power=cp_lower(n,n,a),
        coverage99_two_interval_simultaneous_error=a,
        coverage99_lower_if_zero_failures=cp_lower(n,n,a),
        CI99_maximum_failures_to_certify_coverage_floor95=max(allowed),
        CI95_coverage_diagnostic_only=True,
        no_market_power_certificate=True)
