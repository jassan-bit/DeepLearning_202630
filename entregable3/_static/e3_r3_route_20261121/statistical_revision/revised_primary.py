"""New development P1 bootstrap-t; frozen P2/MCS controls, not accreditation.

Target: E_seed sqrt(E_time q), with joint seed/calendar sampling and aligned
models. Fixed R12, T128, lag8 and block12 have no general finite-sample theorem.
"""
import hashlib
import math
import numpy as np
from scipy import stats
from crossed_rmse_v2_frozen import (
    Settings, validate, stationary_indices, functional_and_covariance,
    pair_scales, mcs_from_pivots, holm, require,
)

VERSION = 'new-primary-pairwise-bootstrap-t-v1-development'


def absolute_tail(observed, pivots, alpha):
    """Plus-one bilateral p and closed symmetric CI critical, ties count >=.

    The discrete rank uses the B+1 denominator, no interpolated quantile.
    With rejection p<alpha, the rank is floor((1-alpha)(B+1))+1. At a boundary
    atom p=alpha the closed CI retains the null. No random tie breaking.
    """
    pivots = np.asarray(pivots, dtype=float)
    observed = np.asarray(observed, dtype=float)
    require(pivots.ndim == 2 and pivots.shape[1:] == observed.shape
            and np.isfinite(pivots).all() and np.isfinite(observed).all()
            and 0 < alpha < 1, 'Finite B x contrasts pivots required')
    B = len(pivots)
    require(B > 0, 'Nonempty bootstrap required')
    magnitude = np.abs(pivots)
    p = (1 + np.count_nonzero(magnitude >= np.abs(observed)[None], axis=0))/(B+1)
    rank = math.floor((1-alpha)*(B+1))+1
    critical = (np.full(observed.shape, np.inf) if rank > B
                else np.sort(magnitude, axis=0)[rank-1])
    return p, critical, rank


def from_indices(q, names, candidate, seeds, dates, settings=Settings()):
    """Deterministic core: all models share each supplied crossed draw.

    Public validation uses run(); this entry permits guards without any RNG.
    No bootstrap draws or rivals are removed after observing degeneracy.
    """
    q = validate(q, names, candidate, settings)
    R, T, M = q.shape
    seeds = np.asarray(seeds)
    dates = np.asarray(dates)
    require(seeds.shape == (settings.B, R) and dates.shape == (settings.B, T)
            and np.issubdtype(seeds.dtype, np.integer)
            and np.issubdtype(dates.dtype, np.integer)
            and ((0 <= seeds) & (seeds < R)).all()
            and ((0 <= dates) & (dates < T)).all(), 'Invalid frozen draw indices')
    risks, cov, seed, time = functional_and_covariance(q[None], settings.hac_lag)
    risks = risks[0]
    scales = pair_scales(cov)[0]
    observed = risks[:, None] - risks[None, :]
    boot_risks = np.empty((settings.B, M))
    boot_scales = np.empty((settings.B, M, M))
    for start in range(0, settings.B, settings.batch):
        end = min(start+settings.batch, settings.B)
        draw = q[seeds[start:end, :, None], dates[start:end, None, :], :]
        rr, cc, _, _ = functional_and_covariance(draw, settings.hac_lag)
        boot_risks[start:end] = rr
        boot_scales[start:end] = pair_scales(cc)
    roots = (boot_risks[:, :, None]-boot_risks[:, None, :])-observed[None]
    magnitude = max(float(np.max(np.abs(risks))), np.finfo(float).tiny)
    floor = 64*np.finfo(float).eps*magnitude
    obs_good = scales > floor
    boot_good = boot_scales > floor
    unsupported_boot = (~boot_good) & (np.abs(roots) > floor)
    pivots = np.divide(roots, boot_scales, out=np.zeros_like(roots), where=boot_good)
    mcs = mcs_from_pivots(risks, scales, pivots, names, settings.mcs_alpha)
    if unsupported_boot.any():
        mcs = {'status': 'unsupported_bootstrap_studentizer', 'retained': list(names),
               'eliminated': [], 'initial_global_p': None, 'steps': [],
               'third_pairwise_p_provided': False}
    rivals = np.array([i for i in range(M) if i != candidate])
    theta = risks[rivals]-risks[candidate]
    se = scales[rivals, candidate]
    centered = roots[:, rivals, candidate]
    studentized = pivots[:, rivals, candidate]
    t_observed = np.divide(theta, se, out=np.zeros_like(theta), where=se > floor)
    p_t, critical_t, rank_t = absolute_tail(t_observed, studentized, settings.alpha)
    p_basic = (1+np.count_nonzero(np.abs(centered) >= np.abs(theta)[None], axis=0))/(settings.B+1)
    p1_supported = (se > floor) & (~unsupported_boot[:, rivals, candidate].any(axis=0))
    p_t[~p1_supported] = 1.
    # P2 expression is copied unchanged from the frozen v2 primary control.
    hln = np.sqrt((T-1)/T)
    p2se = se/hln
    z = np.divide(theta, p2se, out=np.zeros_like(theta), where=se > floor)
    p_delta = 2*stats.t.sf(np.abs(z), df=R-1)
    p_basic[se <= floor] = 1.
    p_delta[se <= floor] = 1.
    raw = np.column_stack([p_t, p_delta])
    adjusted = holm(raw.ravel()).reshape(raw.shape)
    control_adjusted = holm(np.column_stack([p_basic, p_delta]).ravel()).reshape(raw.shape)
    critical = stats.t.ppf(1-settings.alpha/2, df=R-1)
    basic_lo = theta-np.quantile(centered, 1-settings.alpha/2, axis=0)
    basic_hi = theta-np.quantile(centered, settings.alpha/2, axis=0)
    seed_scales = pair_scales(seed)[0]
    time_scales = pair_scales(time)[0]
    pair = []
    for k, i in enumerate(rivals):
        supported = bool(se[k] > floor)
        t_supported = bool(p1_supported[k] and np.isfinite(critical_t[k]))
        status = ('proposed_inference_computed' if t_supported else
                  ('unsupported_bootstrap_studentizer' if supported else
                   'degenerate_observed_contrast_no_interval'))
        pair.append({'rival': names[i], 'theta_hat': float(theta[k]), 'SE': float(se[k]),
                     'nondegenerate': supported, 'P1_supported': t_supported,
                     'P2_supported': supported, 'status': status,
                     'P1_equal_two_sided_p': float(p_t[k]),
                     'P2_equal_two_sided_p': float(p_delta[k]),
                     'P1_Holm': float(adjusted[k, 0]), 'P2_Holm': float(adjusted[k, 1]),
                     'P1_CI95': [float(theta[k]-critical_t[k]*se[k]),
                                 float(theta[k]+critical_t[k]*se[k])] if t_supported else None,
                     'P2_SE': float(p2se[k]),
                     'P2_CI95': [float(theta[k]-critical*p2se[k]),
                                 float(theta[k]+critical*p2se[k])] if supported else None,
                     'P1_basic_frozen_control_p': float(p_basic[k]),
                     'P1_basic_frozen_control_Holm38': float(control_adjusted[k, 0]),
                     'P2_frozen_control_Holm38': float(control_adjusted[k, 1]),
                     'P1_basic_frozen_control_CI95': [float(basic_lo[k]), float(basic_hi[k])] if supported else None,
                     'P1_bootstrap_unsupported_draws': int(unsupported_boot[:, i, candidate].sum()),
                     'seed_variance': float(seed_scales[i, candidate]**2),
                     'time_variance': float(time_scales[i, candidate]**2)})
    return {'version': VERSION, 'shape': list(q.shape), 'risks': risks.tolist(),
            'pairwise': pair, 'MCS_bootstrap_t_frozen_control': mcs,
            'bootstrap_studentizers_recomputed': settings.B,
            'unsupported_bootstrap_pairs': int(unsupported_boot.sum()),
            'P1_CI_absolute_pivot_order_rank': rank_t,
            'P1_minimum_empirical_p': 1/(settings.B+1),
            'draw_indices_SHA256': hashlib.sha256(seeds.tobytes()+dates.tobytes()).hexdigest(),
            'P2_HLN_h1_factor': float(hln), 'P2_reference_df': R-1,
            'pairwise_methods': 2, 'third_primary_method_accredited': False,
            'confirmed_valid_for_market': False, 'R3_accredited': False,
            'new_market_fits': 0, 'new_market_predictions': 0,
            'AI_assistance_declared': True}


def run(q, names, candidate, rng, settings=Settings()):
    q = validate(q, names, candidate, settings)
    R, T, _ = q.shape
    dates = stationary_indices(settings.B, T, settings.mean_block, rng)
    seeds = rng.integers(R, size=(settings.B, R))
    return from_indices(q, names, candidate, seeds, dates, settings)
