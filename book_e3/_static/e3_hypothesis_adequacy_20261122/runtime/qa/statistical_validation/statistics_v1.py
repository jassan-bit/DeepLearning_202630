"""Prospective, fixed-seed temporal inference; never treats R*T as iid.

The first system is the candidate. See docs/TEMPORAL_BATTERY.md for the
estimand change, hypotheses, conditional GW interpretation and limitations.
Only numpy/scipy are required. This module never fits a forecasting model.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy import stats

PROTOCOL_VERSION = "temporal-fixed-seeds-mse-v1"
METHODS = ("dm_hln_hac", "stationary_bootstrap", "gw_conditional")


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm adjusted p-values in input order, including ties and p=1."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be a finite vector in [0,1]")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted.tolist()


def hac_covariance(values: np.ndarray, lags: int) -> np.ndarray:
    """Bartlett/Newey-West long-run covariance, with divisor T at every lag."""
    a = np.asarray(values, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2 or not np.isfinite(a).all() or not 0 <= lags < len(a):
        raise ValueError("invalid finite time-by-moment array or HAC lag count")
    centered = a - a.mean(axis=0)
    cov = centered.T @ centered / len(a)
    for lag in range(1, lags + 1):
        gamma = centered[lag:].T @ centered[:-lag] / len(a)
        cov += (1 - lag / (lags + 1)) * (gamma + gamma.T)
    return (cov + cov.T) / 2


def stationary_indices(n: int, reps: int, block_length: float, rng: np.random.Generator) -> np.ndarray:
    """Circular stationary bootstrap with geometric lengths and uniform restarts.

    One index array is shared across systems/seeds: pairing and cross-rival
    dependence are preserved. Circular wrapping is part of the bootstrap only.
    """
    if n < 2 or reps < 1 or not 1 <= block_length <= n:
        raise ValueError("invalid n, reps or mean block length")
    idx = np.empty((reps, n), dtype=np.int32)
    idx[:, 0] = rng.integers(0, n, size=reps)
    for t in range(1, n):
        restart = rng.random(reps) < 1 / block_length
        idx[:, t] = np.where(restart, rng.integers(0, n, size=reps), (idx[:, t - 1] + 1) % n)
    return idx


def _bootstrap_means(time_losses: np.ndarray, reps: int, length: float, seed: int) -> np.ndarray:
    n = time_losses.shape[1]
    rng = np.random.default_rng(seed)
    # Bound temporary index/count arrays to approximately 64 MiB, independently
    # of B. Output stores B*systems means, never B*T*systems predictions.
    chunk_size = max(1, min(2048, (64*1024**2)//(24*n)))
    means = np.empty((reps, time_losses.shape[0]))
    for start in range(0, reps, chunk_size):
        count = min(chunk_size, reps-start)
        idx = stationary_indices(n, count, length, rng)
        counts = np.bincount((idx.astype(np.int64) + np.arange(count)[:, None] * n).ravel(), minlength=count*n)
        means[start:start+count] = counts.reshape(count, n) @ time_losses.T / n
    return means


def _direction(value: float) -> str:
    return "candidate_better" if value > 0 else "candidate_worse" if value < 0 else "equal"


def _unavailable(reason: str, mean: float) -> dict[str, Any]:
    return {"statistic": None, "raw_p": 1., "direction": _direction(mean), "status": reason}


def _dm(values: np.ndarray, lags: int, horizon: int) -> dict[str, Any]:
    n, mean = len(values), float(values.mean())
    lrv = float(hac_covariance(values, lags)[0, 0])
    if lrv <= np.finfo(float).eps * max(float(np.mean(values**2)), 1.):
        result = _unavailable("degenerate_long_run_variance", mean)
        result["long_run_variance"] = lrv
        return result
    factor_squared = (n + 1 - 2*horizon + horizon*(horizon-1)/n) / n
    if factor_squared <= 0:
        return _unavailable("insufficient_horizon_sample", mean)
    statistic = mean / math.sqrt(lrv/n) * math.sqrt(factor_squared)
    return {"statistic": float(statistic), "raw_p": float(2*stats.t.sf(abs(statistic), n-1)),
            "direction": _direction(mean), "status": "ok", "long_run_variance": lrv,
            "hln_factor": math.sqrt(factor_squared), "reference": "student_t", "df": n-1}


def _bootstrap_test(values: np.ndarray, boot: np.ndarray) -> dict[str, Any]:
    mean = float(values.mean())
    centered = boot - mean
    if float(np.ptp(boot)) <= np.finfo(float).eps * max(abs(mean), 1.) * 32:
        return _unavailable("degenerate_bootstrap_distribution", mean)
    # Equal-tail centered bootstrap test of E[D]=0; +1 prevents zero Monte Carlo p.
    right = (1 + np.count_nonzero(centered >= mean)) / (len(boot) + 1)
    left = (1 + np.count_nonzero(centered <= mean)) / (len(boot) + 1)
    return {"statistic": mean, "raw_p": float(min(1., 2*min(left, right))),
            "direction": _direction(mean), "status": "ok", "reference": "centered_stationary_bootstrap"}


def _gw(values: np.ndarray, instruments: np.ndarray | None, lags: int) -> dict[str, Any]:
    mean = float(values.mean())
    if instruments is None:
        return _unavailable("predictable_instruments_not_supplied", mean)
    moments = values[:, None] * instruments
    cov = hac_covariance(moments, lags)
    eigenvalues = np.linalg.eigvalsh(cov)
    q = instruments.shape[1]
    scale = max(float(np.max(eigenvalues)), np.finfo(float).tiny)
    rank = int(np.count_nonzero(eigenvalues > scale*1e-10))
    if rank < q or float(np.max(eigenvalues)) <= np.finfo(float).eps:
        result = _unavailable("singular_conditional_moment_covariance", mean)
        result.update({"moment_rank": rank, "df": q})
        return result
    avg = moments.mean(axis=0)
    statistic = float(len(values) * avg @ np.linalg.solve(cov, avg))
    return {"statistic": max(0., statistic), "raw_p": float(stats.chi2.sf(max(0., statistic), q)),
            "direction": _direction(mean), "status": "ok", "reference": "chi_square",
            "df": q, "moment_rank": rank, "moment_means": avg.tolist(),
            "covariance_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
            "interpretation": "conditional_moment_rejection_is_not_directional_superiority"}


def _basic_ci(estimate: float, bootstrap: np.ndarray) -> list[float]:
    lo, hi = np.quantile(bootstrap - estimate, [.025, .975])
    return [float(estimate-hi), float(estimate-lo)]


def analyze(
    predictions: np.ndarray,
    y: np.ndarray,
    systems: Sequence[str],
    *,
    bootstrap_reps: int,
    block_length: float,
    hac_lags: int,
    instruments: np.ndarray | None,
    margin: float = .02,
    seed: int = 20261121,
    horizon: int = 1,
) -> dict[str, Any]:
    """Analyze aligned [systems, fixed_seeds, chronological_dates] predictions.

    The first named system is the candidate; all remaining systems are rivals.
    ``instruments`` must have shape [T,K>=2], start with a constant 1, and
    contain only information available at the corresponding forecast origin.
    Caller is responsible for proving availability, especially for overlapping
    labels. None returns GW unavailable, never a disguised constant-only test.
    HAC lags and bootstrap block length are fixed by development planning.
    The caller must enforce a complete 21-system family at final evaluation.
    """
    p, target = np.asarray(predictions, dtype=float), np.asarray(y, dtype=float)
    if p.ndim != 3 or p.shape[0] < 2 or p.shape[1] < 1 or p.shape[2] < 12:
        raise ValueError("predictions must have shape [>=2 systems, >=1 seeds, >=12 dates]")
    names = list(systems)
    if len(names) != p.shape[0] or len(set(names)) != len(names) or not all(isinstance(s, str) and s for s in names):
        raise ValueError("systems must be unique nonempty names matching axis 0")
    if target.shape != (p.shape[2],) or not np.isfinite(p).all() or not np.isfinite(target).all():
        raise ValueError("target must be aligned [T], all predictions and labels finite")
    n = p.shape[2]
    if not isinstance(bootstrap_reps, (int, np.integer)) or bootstrap_reps < 99:
        raise ValueError("bootstrap_reps must be an integer >=99")
    if not isinstance(hac_lags, (int, np.integer)) or not 1 <= hac_lags < n // 3:
        raise ValueError("HAC lags must be positive and < T/3; lag 0 is not adequate for overlapping labels")
    if not isinstance(horizon, (int, np.integer)) or not 1 <= horizon < n:
        raise ValueError("horizon must be a positive integer smaller than T")
    if not 2 <= block_length <= n/3 or not 0 <= margin < 1:
        raise ValueError("block length must be in [2,T/3], and margin in [0,1)")
    z = None
    if instruments is not None:
        z = np.asarray(instruments, dtype=float)
        if z.ndim != 2 or z.shape[0] != n or z.shape[1] < 2 or not np.isfinite(z).all():
            raise ValueError("instruments must be a finite [T,K>=2] predictable matrix")
        if not np.allclose(z[:, 0], 1., rtol=0, atol=1e-12) or np.linalg.matrix_rank(z) != z.shape[1]:
            raise ValueError("instruments require first column 1 plus nonconstant, full-rank predictable covariates")
        if z.shape[1] >= n/5:
            raise ValueError("too many conditional moments for the sample")
        # Deterministic scale normalization improves conditioning without changing the moment null.
        z = z.copy()
        z[:, 1:] = (z[:, 1:] - z[:, 1:].mean(axis=0)) / z[:, 1:].std(axis=0)
    losses = (p - target[None, None, :])**2
    if not np.isfinite(losses).all():
        raise ValueError("squared losses overflowed")
    # Averaging fixed seeds BEFORE resampling preserves all seed/date dependence.
    time_losses = losses.mean(axis=1)
    means = time_losses.mean(axis=1)
    boot = _bootstrap_means(time_losses, int(bootstrap_reps), block_length, seed)
    metrics = {}
    for j, name in enumerate(names):
        seed_rmse = np.sqrt(losses[j].mean(axis=1))
        metrics[name] = {"mse": float(means[j]), "rmse_pooled": float(np.sqrt(means[j])),
                         "mean_seed_rmse": float(seed_rmse.mean()),
                         "std_seed_rmse": float(seed_rmse.std(ddof=1)) if len(seed_rmse)>1 else 0.,
                         "seed_rmse": seed_rmse.tolist(),
                         "unique_prediction_seed_traces": int(len(np.unique(p[j], axis=0)))}
    comparisons, family_slots = [], []
    for j in range(1, len(names)):
        d = time_losses[j] - time_losses[0]
        w = (1-margin)**2 * time_losses[j] - time_losses[0]
        boot_d = boot[:, j] - boot[:, 0]
        boot_w = (1-margin)**2 * boot[:, j] - boot[:, 0]
        gain = 1 - math.sqrt(means[0]/means[j]) if means[j] > 0 else None
        boot_gain = 1 - np.sqrt(boot[:, 0]/boot[:, j]) if np.all(boot[:, j] > 0) else None
        sd = float(np.std(d, ddof=1))
        effect = float(d.mean()/sd) if sd > np.finfo(float).eps else None
        methods = {}
        for method in METHODS:
            methods[method] = {}
            for label, values, b in (("epa", d, boot_d), ("practical", w, boot_w)):
                test = _dm(values, hac_lags, horizon) if method == "dm_hln_hac" else _bootstrap_test(values, b) if method == "stationary_bootstrap" else _gw(values, z, hac_lags)
                methods[method][label] = test
                family_slots.append(test)
        ci_d = _basic_ci(float(d.mean()), boot_d)
        # A paired temporal standardized effect uses sample SD as a descriptive
        # scale; uncertainty is reported for its numerator and RMSE gain.
        comparison = {"rival": names[j], "mean_mse_difference": float(d.mean()),
                      "practical_mean": float(w.mean()), "relative_rmse_gain": gain,
                      "historical_mean_seed_rmse_favorable": bool(metrics[names[0]]["mean_seed_rmse"] < metrics[names[j]]["mean_seed_rmse"]),
                      "temporal_effect_cohen_d": effect,
                      "ci95_mse_difference": ci_d,
                      "ci95_relative_rmse_gain": [float(x) for x in np.quantile(boot_gain, [.025, .975])] if boot_gain is not None else None,
                      "methods": methods,
                      "diagnostics": {"lag1_loss_difference_correlation": float(np.corrcoef(d[1:], d[:-1])[0,1]) if np.std(d[1:])>1e-14 and np.std(d[:-1])>1e-14 else None,
                                      "first_half_mean": float(d[:n//2].mean()), "second_half_mean": float(d[n//2:].mean()),
                                      "stationarity_verified": False}}
        comparisons.append(comparison)
    adjusted = holm([test["raw_p"] for test in family_slots])
    for test, adj in zip(family_slots, adjusted):
        test["adjusted_p"] = adj
    for c in comparisons:
        tests = [test for m in c["methods"].values() for test in m.values()]
        c["least_favorable_adjusted_p"] = max(test["adjusted_p"] for test in tests)
        c["significant_inferiority"] = any(m["epa"]["adjusted_p"] < .05 and c["mean_mse_difference"] < 0 for m in c["methods"].values())
        c["provisional_statistical_gate"] = bool(
            c["mean_mse_difference"] > 0 and c["practical_mean"] > 0
            and c["historical_mean_seed_rmse_favorable"]
            and all(test["status"] == "ok" and test["adjusted_p"] < .05 for test in tests)
            and c["temporal_effect_cohen_d"] is not None and c["temporal_effect_cohen_d"] >= .2
            and c["ci95_mse_difference"][0] > 0)
    family_n = len(family_slots)
    return {"protocol_version": PROTOCOL_VERSION,
            "estimand": "time expectation of fixed-seed mean squared loss; sqrt of aggregate MSE is a monotone reporting transform",
            "candidate": names[0], "systems": names, "n_dates": n, "n_fixed_seeds": p.shape[1],
            "metrics": metrics, "comparisons": comparisons,
            "family": {"tests": family_n, "rivals": len(names)-1, "methods": list(METHODS), "nulls": ["epa", "practical"],
                       "adjustment": "Holm across all methods, rivals and both nulls in this panel", "alpha": .05,
                       "bootstrap_minimum_two_sided_p": 2/(bootstrap_reps+1),
                       "bootstrap_resolution_sufficient_at_first_holm_step": bool(2/(bootstrap_reps+1) <= .05/family_n)},
            "settings": {"bootstrap_reps": int(bootstrap_reps), "block_length": float(block_length), "hac_lags": int(hac_lags),
                         "horizon": int(horizon), "margin_relative_rmse": float(margin), "seed": int(seed),
                         "instruments_columns": int(z.shape[1]) if z is not None else 0},
            "all_rivals_statistical_gate": all(c["provisional_statistical_gate"] for c in comparisons),
            "r3_accredited": False,
            "limitations": ["Prospective estimand differs from historical mean of seed RMSE; compatibility with the E1 primary endpoint must be resolved explicitly.",
                            "Temporal inference is conditional on these fitted models and fixed seeds; it does not integrate training/seed uncertainty.",
                            "Stationary weakly dependent loss/moment sequences and predictable instruments are assumptions, not established by this call.",
                            "HAC lags/block length must be fixed on development and cover overlapping label support; h=1 does not imply independence.",
                            "Confidence intervals are marginal temporal bootstrap intervals, not simultaneous family intervals.",
                            "GW tests conditional moments and is not itself a directional test or a duplicate of DM.",
                            "No market fitting, panel adequacy, cross-panel consistency or complete R3 robustness is established here."]}
