"""Prospective, fixed-seed temporal inference; never treats R*T as iid.

The first system is the candidate. See docs/TEMPORAL_BATTERY.md for the
estimand change, hypotheses, conditional GW interpretation and limitations.
Only numpy/scipy are required. This module never fits a forecasting model.
"""
from __future__ import annotations

import math
import hashlib
from functools import lru_cache
from pathlib import Path
import ctypes
import shutil
import subprocess
import tempfile
from typing import Any, Sequence

import numpy as np

PROTOCOL_VERSION = "temporal-fixed-seeds-mse-v4"
METHODS = ("dm_hln_hac", "stationary_bootstrap", "gw_conditional")


@lru_cache(maxsize=1)
def _native_kernel():
    """Optional local C++ accelerator; numpy remains the portable reference.

    Only our bundled, reviewable arithmetic kernel is compiled. The artifact
    is cached by source hash in a private temporary directory; no downloads.
    """
    source = Path(__file__).resolve().parents[1]/"qa/statistical_validation/v3/bootstrap_kernel.cpp"
    compiler = shutil.which("c++")
    if compiler is None or not source.is_file():
        return None
    digest = hashlib.sha256(source.read_bytes()+b"-O3 -std=c++11 -shared -fPIC").hexdigest()
    folder = Path(tempfile.mkdtemp(prefix="temporal-hac-"))
    artifact = folder/f"kernel-{digest}.so"
    try:
        subprocess.run([compiler, "-O3", "-std=c++11", "-shared", "-fPIC", str(source), "-o", str(artifact)],
                       check=True, capture_output=True, timeout=60)
        library = ctypes.CDLL(str(artifact))
        function = library.bootstrap_hac
        f64 = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        function.argtypes = [f64, i32, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             ctypes.c_double, f64, f64, f64, f64, f64]
        function.restype = None
        return function
    except (OSError, subprocess.SubprocessError):
        return None


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
    windows = _hac_windows(a[None, :, :], lags)[0]
    return windows.T @ windows / (len(a)*(lags+1))


def _hac_windows(values: np.ndarray, lags: int) -> np.ndarray:
    """Centered zero-padded moving sums [batch,T+lags,channels].

    Sum of outer products / (T*(lags+1)) equals Bartlett HAC exactly:
    each lag-k pair occurs lags+1-k times. O(T), without a lag loop.
    """
    return _hac_windows_last(values.transpose(0, 2, 1), lags).transpose(0, 2, 1)


def _hac_windows_last(values: np.ndarray, lags: int) -> np.ndarray:
    """Same moving sums with contiguous time axis, avoiding advanced-index copies."""
    centered = np.ascontiguousarray(values-values.mean(axis=-1, keepdims=True))
    padded = np.pad(centered, ((0, 0), (0, 0), (lags, lags)))
    cs = np.concatenate([np.zeros((*values.shape[:2], 1)), np.cumsum(padded, axis=-1)], axis=-1)
    return cs[..., lags+1:] - cs[..., :-lags-1]


@lru_cache(maxsize=24)
def _fixed_b_reference(n: int, lags: int, q: int, horizon: int, reps: int, seed: int) -> tuple[np.ndarray, str]:
    """Gaussian finite-grid approximation to Kiefer--Vogelsang fixed-b pivot.

    q=1: absolute DM-HLN statistic. q>=2: vector mean Wald, not q copies
    of the scalar test. This is an FCLT approximation, not an exact temporal
    finite-sample calibration. The reference never depends on observed data.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, n, lags, q, horizon, reps]))
    draws = np.empty(reps)
    hln = math.sqrt((n+1-2*horizon+horizon*(horizon-1)/n)/n)
    for start in range(0, reps, 128):
        count = min(128, reps-start)
        values = rng.standard_normal((count, n, q))
        avg = values.mean(axis=1)
        windows = _hac_windows(values, lags)
        cov = np.einsum('bti,btj->bij', windows, windows)/(n*(lags+1))
        if q == 1:
            draws[start:start+count] = np.abs(avg[:, 0]/np.sqrt(cov[:, 0, 0]/n)*hln)
        else:
            draws[start:start+count] = n*np.einsum('bi,bi->b', avg, np.linalg.solve(cov, avg[..., None])[..., 0])
    draws.sort()
    draws.flags.writeable = False
    return draws, hashlib.sha256(draws.tobytes()).hexdigest()


def _mc_p(statistic: float, reference: np.ndarray, reference_name: str) -> dict[str, Any]:
    count = int(len(reference)-np.searchsorted(reference, abs(statistic), side='left'))
    return {"raw_p": (count+1)/(len(reference)+1), "mc_tail_count": count, "mc_reps": len(reference),
            "mc_p_rule": "absolute_tail_plus_one", "reference": reference_name}


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


def _bootstrap_studentized(time_losses: np.ndarray, reps: int, length: float, lags: int,
                          margin: float, seed: int, backend: str = "auto") -> dict[str, np.ndarray]:
    """Joint stationary bootstrap with a fresh HAC variance in EVERY replicate.

    Linear loss contrasts and the log ratio share the same resampled bivariate
    loss covariance. Memory is bounded by small [batch,T,systems] tensors.
    """
    n = time_losses.shape[1]
    rng = np.random.default_rng(seed)
    channels = time_losses.shape[0]
    chunk_size = max(1, min(64, (48*1024**2)//(40*(n+lags)*channels)))
    means = np.empty((reps, time_losses.shape[0]))
    td, tw, tl = (np.empty((reps, channels-1)) for _ in range(3))
    obs = time_losses.mean(axis=1)
    obs_d, obs_w = obs[1:]-obs[0], (1-margin)**2*obs[1:]-obs[0]
    log_obs = np.log(obs[0]/obs[1:]) if np.all(obs > 0) else np.full(channels-1, np.nan)
    a = (1-margin)**2
    if backend not in ("auto", "numpy", "native"):
        raise ValueError("bootstrap backend must be auto, numpy or native")
    native = _native_kernel() if backend != "numpy" else None
    if backend == "native" and native is None:
        raise RuntimeError("native backend requested but compiler/bundled kernel unavailable")
    # Generate indices in larger bounded slabs: Python's T-step loop runs once
    # per slab, rather than once for each small arithmetic/memory batch.
    index_chunk = max(chunk_size, min(reps, (64*1024**2)//(4*n)))
    index_chunk = max(chunk_size, (index_chunk//chunk_size)*chunk_size)
    cached_indices, cache_start = None, 0
    def studentize(numerator, variance, scale):
        valid = variance > np.finfo(float).eps*np.maximum(scale, np.finfo(float).tiny)*64
        out = np.full_like(numerator, np.inf)
        np.divide(numerator, np.sqrt(np.maximum(variance, 0)/n), out=out, where=valid)
        return out
    for start in range(0, reps, chunk_size):
        count = min(chunk_size, reps-start)
        if cached_indices is None or start >= cache_start+len(cached_indices):
            cache_start = start
            cached_indices = stationary_indices(n, min(index_chunk, reps-start), length, rng)
        idx = cached_indices[start-cache_start:start-cache_start+count]
        if native is not None:
            native(np.ascontiguousarray(time_losses), idx, count, channels, n, lags, a, np.ascontiguousarray(obs),
                   means[start:start+count], td[start:start+count], tw[start:start+count], tl[start:start+count])
            continue
        sample = time_losses[:, idx].transpose(1, 0, 2)
        avg = sample.mean(axis=-1)
        means[start:start+count] = avg
        windows = _hac_windows_last(sample, lags)
        variances = np.sum(windows**2, axis=-1)/(n*(lags+1))
        cross = np.sum(windows*windows[:, :1, :], axis=-1)/(n*(lags+1))
        vc, vj, cv = variances[:, :1], variances[:, 1:], cross[:, 1:]
        td[start:start+count] = studentize(avg[:, 1:]-avg[:, :1]-obs_d, vj+vc-2*cv, vj+vc)
        tw[start:start+count] = studentize(a*avg[:, 1:]-avg[:, :1]-obs_w, a*a*vj+vc-2*a*cv, a*a*vj+vc)
        if np.all(avg > 0) and np.isfinite(log_obs).all():
            log_diff = np.log(avg[:, :1]/avg[:, 1:])-log_obs
            log_var = vc/avg[:, :1]**2+vj/avg[:, 1:]**2-2*cv/(avg[:, :1]*avg[:, 1:])
            log_scale = vc/avg[:, :1]**2+vj/avg[:, 1:]**2
            tl[start:start+count] = studentize(log_diff, log_var, log_scale)
        else:
            tl[start:start+count] = np.inf
    return {"means": means, "epa_t": td, "practical_t": tw, "log_ratio_t": tl,
            "backend": "native" if native is not None else "numpy"}


def _direction(value: float) -> str:
    return "candidate_better" if value > 0 else "candidate_worse" if value < 0 else "equal"


def _unavailable(reason: str, mean: float) -> dict[str, Any]:
    return {"statistic": None, "raw_p": 1., "direction": _direction(mean), "status": reason}


def _dm(values: np.ndarray, lags: int, horizon: int, reference: np.ndarray) -> dict[str, Any]:
    n, mean = len(values), float(values.mean())
    scale = float(np.max(np.abs(values)))
    normalized = values/scale if scale else values
    normalized_lrv = float(hac_covariance(normalized, lags)[0, 0])
    lrv = normalized_lrv*scale**2
    if normalized_lrv <= np.finfo(float).eps * float(np.mean(normalized**2)):
        result = _unavailable("degenerate_long_run_variance", mean)
        result["long_run_variance"] = lrv
        return result
    factor_squared = (n + 1 - 2*horizon + horizon*(horizon-1)/n) / n
    if factor_squared <= 0:
        return _unavailable("insufficient_horizon_sample", mean)
    statistic = float(normalized.mean()) / math.sqrt(normalized_lrv/n) * math.sqrt(factor_squared)
    return {"statistic": float(statistic), **_mc_p(statistic, reference, "fixed_b_gaussian_scalar_hln"),
            "direction": _direction(mean), "status": "ok", "long_run_variance": lrv,
            "hln_factor": math.sqrt(factor_squared), "b": (lags+1)/n}


def _bootstrap_test(values: np.ndarray, t_boot: np.ndarray, lags: int) -> dict[str, Any]:
    mean = float(values.mean())
    scale = float(np.max(np.abs(values)))
    normalized = values/scale if scale else values
    lrv = float(hac_covariance(normalized, lags)[0, 0])
    if lrv <= np.finfo(float).eps*float(np.mean(normalized**2)):
        return _unavailable("degenerate_bootstrap_distribution", mean)
    statistic = float(normalized.mean()/np.sqrt(lrv/len(values)))
    # Failed replicate variances count as infinite |t|, conservatively preserving B.
    return {"statistic": statistic, **_mc_p(statistic, np.sort(np.abs(t_boot)), "stationary_bootstrap_studentized_hac"),
            "direction": _direction(mean), "status": "ok", "invalid_bootstrap_variances": int(np.count_nonzero(~np.isfinite(t_boot)))}


def _gw(values: np.ndarray, instruments: np.ndarray | None, lags: int, reference: np.ndarray | None) -> dict[str, Any]:
    mean = float(values.mean())
    if instruments is None:
        return _unavailable("predictable_instruments_not_supplied", mean)
    value_scale = float(np.max(np.abs(values)))
    if value_scale == 0:
        return _unavailable("singular_conditional_moment_covariance", mean)
    moments = (values/value_scale)[:, None] * instruments
    cov = hac_covariance(moments, lags)
    eigenvalues = np.linalg.eigvalsh(cov)
    q = instruments.shape[1]
    scale = max(float(np.max(eigenvalues)), np.finfo(float).tiny)
    rank = int(np.count_nonzero(eigenvalues > scale*1e-10))
    if rank < q or float(np.max(eigenvalues)) <= np.finfo(float).eps * float(np.mean(moments**2)):
        result = _unavailable("singular_conditional_moment_covariance", mean)
        result.update({"moment_rank": rank, "df": q})
        return result
    avg = moments.mean(axis=0)
    statistic = float(len(values) * avg @ np.linalg.solve(cov, avg))
    return {"statistic": max(0., statistic), **_mc_p(statistic, reference, "fixed_b_gaussian_vector_wald"),
            "direction": _direction(mean), "status": "ok", "b": (lags+1)/len(values),
            "df": q, "moment_rank": rank, "moment_means": avg.tolist(),
            "covariance_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
            "interpretation": "conditional_moment_rejection_is_not_directional_superiority"}


def _pivotal_ci(estimate: float, se: float, t_boot: np.ndarray, level: float) -> list[float] | None:
    if not np.isfinite(t_boot).all() or not math.isfinite(estimate):
        return None
    lo, hi = np.quantile(t_boot, [(1-level)/2, (1+level)/2])
    return [float(estimate-hi*se), float(estimate-lo*se)]


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
    reference_reps: int = 49999,
    reference_seed: int = 2026112103,
    bootstrap_backend: str = "auto",
    alpha: float = .025,
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
    if not isinstance(alpha, (float, int)) or not 0 < alpha <= .05:
        raise ValueError("Operational alpha must be in (0,.05]; V4 frozen default is .025")
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
    if not isinstance(reference_reps, (int, np.integer)) or reference_reps < 999:
        raise ValueError("fixed-b reference_reps must be an integer >=999")
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
    boot = _bootstrap_studentized(time_losses, int(bootstrap_reps), block_length, hac_lags, margin, seed, bootstrap_backend)
    dm_reference, dm_reference_sha = _fixed_b_reference(n, hac_lags, 1, horizon, int(reference_reps), int(reference_seed))
    gw_reference, gw_reference_sha = _fixed_b_reference(n, hac_lags, z.shape[1], horizon, int(reference_reps), int(reference_seed)) if z is not None else (None, None)
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
        boot_d_t, boot_w_t = boot["epa_t"][:, j-1], boot["practical_t"][:, j-1]
        gain = 1 - math.sqrt(means[0]/means[j]) if means[j] > 0 else None
        sd = float(np.std(d, ddof=1))
        dscale = float(np.max(np.abs(d)))
        effect = float(d.mean()/sd) if sd > np.finfo(float).eps*dscale*32 else None
        methods = {}
        for method in METHODS:
            methods[method] = {}
            for label, values, b in (("epa", d, boot_d_t), ("practical", w, boot_w_t)):
                test = _dm(values, hac_lags, horizon, dm_reference) if method == "dm_hln_hac" else _bootstrap_test(values, b, hac_lags) if method == "stationary_bootstrap" else _gw(values, z, hac_lags, gw_reference)
                methods[method][label] = test
                family_slots.append(test)
        dse = math.sqrt(max(0., float(hac_covariance(d, hac_lags)[0, 0]))/n)
        ci_d = _pivotal_ci(float(d.mean()), dse, boot_d_t, .95)
        ci99_d = _pivotal_ci(float(d.mean()), dse, boot_d_t, .99)
        gain_cis = {95: None, 99: None}
        if means[0] > 0 and means[j] > 0:
            log_point = math.log(means[0]/means[j])
            log_score = time_losses[0]/means[0]-time_losses[j]/means[j]
            log_se = math.sqrt(max(0., float(hac_covariance(log_score, hac_lags)[0, 0]))/n)
            for confidence in (95, 99):
                log_ci = _pivotal_ci(log_point, log_se, boot["log_ratio_t"][:, j-1], confidence/100)
                if log_ci is not None and max(log_ci) < 1400:
                    gain_cis[confidence] = [float(1-math.exp(log_ci[1]/2)), float(1-math.exp(log_ci[0]/2))]
        # A paired temporal standardized effect uses sample SD as a descriptive
        # scale; uncertainty is reported for its numerator and RMSE gain.
        comparison = {"rival": names[j], "mean_mse_difference": float(d.mean()),
                      "practical_mean": float(w.mean()), "relative_rmse_gain": gain,
                      "historical_mean_seed_rmse_favorable": bool(metrics[names[0]]["mean_seed_rmse"] < metrics[names[j]]["mean_seed_rmse"]),
                      "temporal_effect_cohen_d": effect,
                      "ci95_mse_difference": ci_d,
                      "ci95_relative_rmse_gain": gain_cis[95],
                      "ci99_mse_difference": ci99_d,
                      "ci99_relative_rmse_gain": gain_cis[99],
                      "interval_method": "stationary_bootstrap_t_with_replicate_HAC; log_MSE_ratio_delta_studentization_for_gain",
                      "methods": methods,
                      "diagnostics": {"lag1_loss_difference_correlation": float(np.corrcoef(d[1:], d[:-1])[0,1]) if np.std(d[1:])>np.finfo(float).eps*dscale*32 and np.std(d[:-1])>np.finfo(float).eps*dscale*32 else None,
                                      "first_half_mean": float(d[:n//2].mean()), "second_half_mean": float(d[n//2:].mean()),
                                      "stationarity_verified": False}}
        comparisons.append(comparison)
    adjusted = holm([test["raw_p"] for test in family_slots])
    for test, adj in zip(family_slots, adjusted):
        test["adjusted_p"] = adj
    for c in comparisons:
        tests = [test for m in c["methods"].values() for test in m.values()]
        c["least_favorable_adjusted_p"] = max(test["adjusted_p"] for test in tests)
        c["significant_inferiority"] = any(c["methods"][m]["epa"]["adjusted_p"] < alpha and c["mean_mse_difference"] < 0 for m in ("dm_hln_hac", "stationary_bootstrap"))
        c["gw_conditional_rejection_with_negative_sample_mean"] = bool(c["methods"]["gw_conditional"]["epa"]["adjusted_p"] < alpha and c["mean_mse_difference"] < 0)
        c["provisional_statistical_gate"] = bool(
            c["mean_mse_difference"] > 0 and c["practical_mean"] > 0
            and c["historical_mean_seed_rmse_favorable"]
            and all(test["status"] == "ok" and test["adjusted_p"] < alpha for test in tests)
            and c["temporal_effect_cohen_d"] is not None and c["temporal_effect_cohen_d"] >= .2
            and c["ci95_mse_difference"] is not None and c["ci95_mse_difference"][0] > 0
            and c["ci99_mse_difference"] is not None and c["ci99_mse_difference"][0] > 0
            and c["ci99_relative_rmse_gain"] is not None and c["ci99_relative_rmse_gain"][0] > margin)
    family_n = len(family_slots)
    return {"protocol_version": PROTOCOL_VERSION,
            "estimand": "time expectation of fixed-seed mean squared loss; sqrt of aggregate MSE is a monotone reporting transform",
            "candidate": names[0], "systems": names, "n_dates": n, "n_fixed_seeds": p.shape[1],
            "metrics": metrics, "comparisons": comparisons,
            "family": {"tests": family_n, "rivals": len(names)-1, "methods": list(METHODS), "nulls": ["epa", "practical"],
                       "adjustment": "Holm across all methods, rivals and both nulls in this panel", "alpha": float(alpha),
                       "bootstrap_minimum_two_sided_p": 1/(bootstrap_reps+1),
                       "bootstrap_resolution_sufficient_at_first_holm_step": bool(1/(bootstrap_reps+1) <= alpha/family_n),
                       "reference_minimum_p": 1/(reference_reps+1)},
            "settings": {"bootstrap_reps": int(bootstrap_reps), "block_length": float(block_length), "hac_lags": int(hac_lags),
                         "horizon": int(horizon), "margin_relative_rmse": float(margin), "seed": int(seed),
                         "instruments_columns": int(z.shape[1]) if z is not None else 0,
                         "reference_reps": int(reference_reps), "reference_seed": int(reference_seed),
                         "dm_reference_sha256": dm_reference_sha, "gw_reference_sha256": gw_reference_sha,
                         "b": (hac_lags+1)/n, "gate_interval_confidence": .99, "alpha": float(alpha)},
            "bootstrap_backend": boot["backend"],
            "all_rivals_statistical_gate": all(c["provisional_statistical_gate"] for c in comparisons),
            "r3_accredited": False,
            "limitations": ["Prospective estimand differs from historical mean of seed RMSE; compatibility with the E1 primary endpoint must be resolved explicitly.",
                            "Temporal inference is conditional on these fitted models and fixed seeds; it does not integrate training/seed uncertainty.",
                            "Stationary weakly dependent loss/moment sequences and predictable instruments are assumptions, not established by this call.",
                            "HAC lags/block length must be fixed on development and cover overlapping label support; h=1 does not imply independence.",
                            "Confidence intervals are marginal temporal bootstrap intervals, not simultaneous family intervals.",
                            "Fixed-b Gaussian scalar/vector references approximate an FCLT pivot, not an exact finite-sample market law; Monte Carlo tail uncertainty is reported for all three methods.",
                            "GW tests conditional moments and is not itself a directional test or a duplicate of DM.",
                            "No market fitting, panel adequacy, cross-panel consistency or complete R3 robustness is established here."]}
