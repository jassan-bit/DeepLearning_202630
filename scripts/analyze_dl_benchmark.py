"""Analyze stored, paired DEVELOPMENT predictions from the three DL models.

This script never loads market archives, trains networks or evaluates the reserved
2025--2026 test. Its intervals and tests describe development data and explicitly
distinguish training-seed variability from temporal evaluation uncertainty.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-e1-dl-analysis")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/cache-e1-dl-analysis")
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.special import gammaln, log_ndtr, logsumexp
import statsmodels
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, bds


MODELS = ("MLP", "LSTM", "CNN1D")
METHODS = MODELS + ("Persistencia",)
PHASE = "development_benchmark"
DESIGNATED_SEED = 1001
CREDIT = "Datos: Binance Vision / BTCUSDT | Procesamiento: Jassan Arteta | CC BY-NC-SA 4.0"
SCOPE = ("Exploratory development evidence on 2024 calibration dates; not reserved-test "
         "evidence or confirmation of generalization. Seed intervals condition on this calendar.")


def clean_json(value):
    """Write undefined numerical quantities as JSON null, never NaN/Infinity."""
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def p_fields(log_p: float) -> dict:
    if not np.isfinite(log_p):
        if log_p == -np.inf:
            return {"p_value": 0.0, "log_p_value": -np.inf,
                    "p_value_display": "<1e-300", "p_underflow": True}
        return {"p_value": np.nan, "log_p_value": np.nan,
                "p_value_display": "undefined", "p_underflow": False}
    log_p = min(0.0, float(log_p))
    p = float(np.exp(log_p))
    return {"p_value": p, "log_p_value": log_p,
            "p_value_display": "<1e-300" if log_p < -300 * np.log(10) else f"{p:.12g}",
            "p_underflow": p == 0.0}


def add_holm(rows: list[dict], family: str) -> list[dict]:
    """Retain planned family size, including hypotheses that are undefined.

    Undefined hypotheses receive p=1 internally so their absence cannot shrink
    the correction, then their adjusted output is restored to undefined.
    """
    if not rows:
        return rows
    log_p = np.array([r.get("log_p_value", np.nan) for r in rows], dtype=float)
    defined = ~np.isnan(log_p)
    working = np.where(defined, np.minimum(log_p, 0.0), 0.0)
    order = np.argsort(working, kind="stable")
    adjusted_ordered = np.minimum(0.0, np.maximum.accumulate(
        working[order] + np.log(len(rows) - np.arange(len(rows)))))
    adjusted = np.empty(len(rows))
    adjusted[order] = adjusted_ordered
    for i, row in enumerate(rows):
        row["family"] = family
        row["family_size_planned"] = len(rows)
        values = p_fields(adjusted[i] if defined[i] else np.nan)
        row["p_holm"] = values["p_value"]
        row["log_p_holm"] = values["log_p_value"]
        row["p_holm_display"] = values["p_value_display"]
    return rows


def paired_statistics(x, y, *, null_difference=0.0, alternative="two-sided") -> dict:
    """Paired t summary with unshifted effect and a separately specified null."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Paired inputs must be finite one-dimensional arrays of equal length")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(alternative)
    d = x - y
    n = len(d)
    mean = float(d.mean()) if n else np.nan
    # A repeated decimal can have a tiny nonzero np.std from mean rounding.
    sd = (0.0 if np.ptp(d) == 0 else float(d.std(ddof=1))) if n >= 2 else np.nan
    row = {"n_pairs": n, "mean_difference": mean, "sd_difference": sd,
           "null_difference": float(null_difference), "alternative": alternative,
           "df": n - 1 if n >= 2 else np.nan, "t_statistic": np.nan,
           "ci95_lower": np.nan, "ci95_upper": np.nan, "paired_cohen_d_z": np.nan,
           "standardized_distance_from_null": np.nan, "shapiro_W": np.nan,
           "shapiro_p": np.nan, "ci_method": "marginal Student t, conditional on fixed evaluation calendar",
           "reference": "Student t on independent training-randomization pairs, conditions required",
           "scope": SCOPE, "status": "undefined_insufficient_pairs"}
    row.update(p_fields(np.nan))
    if n < 2:
        return row
    if sd == 0 or not np.isfinite(sd):
        row["status"] = "undefined_zero_pairwise_standard_deviation"
        return row
    se = sd / np.sqrt(n)
    t_stat = (mean - null_difference) / se
    critical = stats.t.ppf(.975, n - 1)
    row.update({"status": "calculated_under_assumptions", "t_statistic": float(t_stat),
                "ci95_lower": float(mean - critical * se),
                "ci95_upper": float(mean + critical * se),
                "paired_cohen_d_z": mean / sd,
                "standardized_distance_from_null": (mean - null_difference) / sd})
    if alternative == "two-sided":
        log_p = np.log(2) + stats.t.logsf(abs(t_stat), n - 1)
    elif alternative == "greater":
        log_p = stats.t.logsf(t_stat, n - 1)
    else:
        log_p = stats.t.logcdf(t_stat, n - 1)
    row.update(p_fields(float(log_p)))
    if n >= 3:
        with warnings.catch_warnings(record=True):
            result = stats.shapiro(d)
        row.update(shapiro_W=float(result.statistic), shapiro_p=float(result.pvalue))
    return row


def longest_contiguous_indices(dates) -> np.ndarray:
    """Choose the longest daily segment by availability, earliest on equal size."""
    dates = pd.DatetimeIndex(pd.to_datetime(dates, utc=True))
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("Dates must be sorted and unique")
    if not len(dates):
        return np.array([], dtype=int)
    # Pandas3 may infer microsecond resolution; Timedelta.value is nanoseconds.
    cuts = np.flatnonzero(np.diff(dates.as_unit("ns").asi8) != pd.Timedelta(days=1).value) + 1
    segments = np.split(np.arange(len(dates)), cuts)
    return max(segments, key=len)


def dm_hln(differences, *, lag=14, horizon=1) -> dict:
    """Approximate DM/HLN using a Bartlett long-run variance estimate."""
    d = np.asarray(differences, dtype=float)
    if d.ndim != 1 or not np.isfinite(d).all():
        raise ValueError("Loss differences must be a finite vector")
    n = len(d)
    row = {"n_dates": n, "horizon": horizon, "HAC_lag": lag,
           "kernel": "Bartlett", "mean_squared_loss_difference": float(d.mean()) if n else np.nan,
           "long_run_variance": np.nan, "DM": np.nan, "DM_HLN": np.nan,
           "reference": "Student t(T-1), approximate; HAC/HLN conditions not guaranteed by the factor",
           "estimand": "temporal mean of seed-averaged squared-loss differences; not mean-seed RMSE",
           "scope": SCOPE, "status": "undefined_insufficient_dates"}
    row.update(p_fields(np.nan))
    if n <= max(lag + 1, horizon) or lag < 0 or horizon < 1:
        return row
    if np.ptp(d) == 0:
        row["status"] = "undefined_constant_loss_differences"
        row["long_run_variance"] = 0.0
        return row
    centered = d - d.mean()
    omega = float(centered @ centered / n)
    for k in range(1, lag + 1):
        gamma = float(centered[k:] @ centered[:-k] / n)
        omega += 2 * (1 - k / (lag + 1)) * gamma
    row["long_run_variance"] = omega
    factor_squared = (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
    if omega <= 0 or not np.isfinite(omega) or factor_squared <= 0:
        row["status"] = "undefined_nonpositive_variance_or_HLN_factor"
        return row
    statistic = float(np.sqrt(n) * d.mean() / np.sqrt(omega))
    corrected = statistic * np.sqrt(factor_squared)
    row.update(status="calculated_approximate", DM=statistic, DM_HLN=float(corrected))
    row.update(p_fields(float(np.log(2) + stats.t.logsf(abs(corrected), n - 1))))
    return row


def block_indices(n: int, length: int, replicates: int, rng) -> np.ndarray:
    if n < 1 or length < 1 or length > n or replicates < 1:
        raise ValueError("A nonempty series and block length in [1,n] are required")
    starts = rng.integers(n, size=(replicates, int(np.ceil(n / length))))
    return ((starts[..., None] + np.arange(length)) % n).reshape(replicates, -1)[:, :n]


def mean_seed_rmse(squared_errors: np.ndarray) -> np.ndarray:
    """Input method by seed by date; average roots, never ensemble predictions."""
    values = np.asarray(squared_errors, dtype=float)
    if values.ndim != 3 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Expected finite nonnegative method×seed×date squared errors")
    return np.sqrt(values.mean(axis=2)).mean(axis=1)


def bootstrap_scores(squared_errors, *, length=14, replicates=2000, seed=20261104) -> np.ndarray:
    losses = np.asarray(squared_errors, dtype=float)
    mean_seed_rmse(losses)  # Validate before allocating.
    rng = np.random.default_rng(seed)
    indices = block_indices(losses.shape[2], length, replicates, rng)
    scores = np.empty((replicates, losses.shape[0]))
    for start in range(0, replicates, 32):
        selected = losses[:, :, indices[start:start + 32]]
        scores[start:start + 32] = np.sqrt(selected.mean(axis=-1)).mean(axis=1).T
    return scores


def chi_square_logsf(q: float, degrees: int) -> float:
    """Stable integer-df reference tail, including values that underflow in sf."""
    if q == 0:
        return 0.0
    if q < 0 or degrees < 1:
        raise ValueError("Nonnegative statistic and positive integer df required")
    x = q / 2
    if degrees % 2 == 0:
        k = np.arange(degrees // 2, dtype=float)
        return float(-x + logsumexp(k * np.log(x) - gammaln(k + 1)))
    answer = float(np.log(2) + log_ndtr(-np.sqrt(q)))
    a = .5
    for _ in range((degrees - 1) // 2):
        answer = float(np.logaddexp(answer, a * np.log(x) - x - gammaln(a + 1)))
        a += 1
    return min(answer, 0.0)


@dataclass
class PredictionSet:
    dates: pd.DatetimeIndex
    seeds: np.ndarray
    truth: np.ndarray
    predictions: np.ndarray  # method by seed by date; persistence repeated as deterministic values.
    history_days: int
    source: Path
    run_ids: dict = field(default_factory=dict)
    config_ids: dict = field(default_factory=dict)

    @property
    def squared_errors(self):
        return (self.predictions - self.truth[None, None, :]) ** 2


def load_predictions(path: Path) -> PredictionSet:
    frame = pd.read_csv(path)
    required = {"phase", "run_id", "model", "history_days", "config_id", "seed",
                "outcome_day", "y_true", "y_pred", "persistence_pred"}
    if not required.issubset(frame):
        raise ValueError(f"Missing prediction columns: {sorted(required - set(frame))}")
    if set(frame.phase) != {PHASE} or set(frame.model) != set(MODELS):
        raise ValueError("Only the three-model development benchmark phase is admitted")
    frame["outcome_day"] = pd.to_datetime(frame.outcome_day, utc=True)
    if not frame.outcome_day.dt.year.eq(2024).all():
        raise ValueError("Analysis refuses non-development years; reserved test is not admitted")
    if frame.duplicated(["model", "seed", "outcome_day"]).any():
        raise ValueError("Duplicate method/seed/date predictions")
    if frame.history_days.nunique() != 1:
        raise ValueError("The comparison requires the same selected history across models")
    dates = pd.DatetimeIndex(sorted(frame.outcome_day.unique()))
    seeds = np.sort(frame.seed.unique().astype(int))
    truth = None
    persistence = None
    arrays, run_ids, config_ids = [], {}, {}
    for model in MODELS:
        subset = frame[frame.model == model]
        if set(subset.seed) != set(seeds) or subset.config_id.nunique() != 1:
            raise ValueError("Model seed/configuration coverage is not paired")
        config_ids[model] = str(subset.config_id.iloc[0])
        model_rows = []
        for seed in seeds:
            rows = subset[subset.seed == seed].sort_values("outcome_day")
            if not pd.DatetimeIndex(rows.outcome_day).equals(dates) or rows.run_id.nunique() != 1:
                raise ValueError("Every method/seed must have the identical date cohort and one run ID")
            run_ids[(model, int(seed))] = str(rows.run_id.iloc[0])
            values = rows[["y_true", "y_pred", "persistence_pred"]].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite observations or predictions cannot be silently omitted")
            if truth is None:
                truth, persistence = values[:, 0].copy(), values[:, 2].copy()
            if not np.allclose(values[:, 0], truth, rtol=0, atol=1e-12):
                raise ValueError("Targets differ across paired predictions")
            if not np.allclose(values[:, 2], persistence, rtol=0, atol=1e-12):
                raise ValueError("Persistence differs across paired predictions")
            model_rows.append(values[:, 1])
        arrays.append(model_rows)
    arrays.append(np.broadcast_to(persistence, (len(seeds), len(dates))).copy())
    return PredictionSet(dates, seeds, truth, np.asarray(arrays), int(frame.history_days.iloc[0]), path,
                         run_ids, config_ids)


def summary_row(values, **labels) -> dict:
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    n = len(valid)
    mean = float(valid.mean()) if n else np.nan
    sd = (0.0 if np.ptp(valid) == 0 else float(valid.std(ddof=1))) if n >= 2 else np.nan
    half = stats.t.ppf(.975, n - 1) * sd / np.sqrt(n) if n >= 2 and sd > 0 else np.nan
    return dict(labels, n_total=len(values), n_defined=n, mean=mean, sample_sd=sd,
                ci95_lower=mean-half, ci95_upper=mean+half,
                ci_method="marginal Student t across training seeds, conditional on fixed development dates",
                interval_scope="training randomization only", scope=SCOPE,
                ci_status="calculated_under_assumptions" if np.isfinite(half) else "undefined_insufficient_or_zero_seed_variance")


def write_frame(rows, path: Path):
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


def build_metric_summary(data: PredictionSet, runs: pd.DataFrame) -> list[dict]:
    rows = []
    for i, model in enumerate(MODELS):
        records = runs[runs.model == model].set_index("seed").loc[data.seeds]
        if not records.history_days.eq(data.history_days).all():
            raise ValueError(f"Run history disagrees with predictions for {model}")
        if not records.config_id.astype(str).eq(data.config_ids[model]).all():
            raise ValueError(f"Run configuration disagrees with predictions for {model}")
        if any(str(records.loc[seed, "run_id"]) != data.run_ids[(model, int(seed))] for seed in data.seeds):
            raise ValueError(f"Run identifiers disagree with predictions for {model}")
        computed = np.sqrt(data.squared_errors[i].mean(axis=1))
        if not np.allclose(computed, records.rmse.to_numpy(float), rtol=1e-8, atol=1e-12):
            raise ValueError(f"Stored run RMSE does not match predictions for {model}")
        for metric in ["rmse", "mae", "mse", "r2", "mape", "mase",
                       "parameters", "training_seconds", "inference_seconds", "mape_coverage", "mase_coverage"]:
            if metric in records:
                row = summary_row(records[metric], model=model, metric=metric,
                                  history_days=data.history_days, n_dates=len(data.dates))
                if metric == "parameters":
                    row.update(ci95_lower=np.nan, ci95_upper=np.nan,
                               ci_method="implementation count; no confidence interval", ci_status="not_applicable_count")
                elif metric.endswith("_coverage"):
                    row.update(ci95_lower=np.nan, ci95_upper=np.nan,
                               ci_method="cohort coverage; no training-seed interval", ci_status="not_applicable_coverage")
                rows.append(row)
    e = data.truth-data.predictions[-1, 0]
    mse = float(np.mean(e**2))
    denominator = float(np.sum((data.truth-data.truth.mean())**2))
    positive = data.truth > 0
    persistence_values = {"rmse": np.sqrt(mse), "mse": mse, "mae": np.mean(np.abs(e)),
                          "r2": 1-np.sum(e**2)/denominator if denominator > 0 else np.nan,
                          "mape": 100*np.mean(np.abs(e[positive]/data.truth[positive])) if positive.any() else np.nan,
                          "mase": np.nan, "parameters": 0.0,
                          "mape_coverage": float(positive.mean()), "mase_coverage": 0.0}
    if "mase_train_scale" in runs:
        scales = runs.mase_train_scale.to_numpy(float)
        if np.isfinite(scales).all() and np.allclose(scales, scales[0], rtol=0, atol=1e-15) and scales[0] > 0:
            persistence_values["mase"] = persistence_values["mae"]/scales[0]
            persistence_values["mase_coverage"] = 1.0
    for metric, value in persistence_values.items():
        rows.append({"model": "Persistencia", "metric": metric, "history_days": data.history_days,
                     "n_dates": len(data.dates), "n_total": 1, "n_defined": int(np.isfinite(value)),
                     "mean": float(value), "sample_sd": np.nan, "ci95_lower": np.nan,
                     "ci95_upper": np.nan, "ci_method": "one deterministic forecast; not R independent executions",
                     "interval_scope": "no training-seed interval", "scope": SCOPE})
    return rows


def paired_test_table(data: PredictionSet) -> list[dict]:
    rmse = np.sqrt(data.squared_errors.mean(axis=2))
    rows = []
    for a, b in combinations(range(3), 2):
        rows.append(dict(comparison=f"{MODELS[a]} - {MODELS[b]}", model_a=MODELS[a], model_b=MODELS[b],
                         estimand="difference in mean seed RMSE", improvement_sign="negative favors A",
                         **paired_statistics(rmse[a], rmse[b])))
    for a, model in enumerate(MODELS):
        rows.append(dict(comparison=f"{model} - Persistencia", model_a=model, model_b="Persistencia",
                         estimand="difference in mean seed RMSE", improvement_sign="negative favors model",
                         **paired_statistics(rmse[a], rmse[3])))
    for a, model in enumerate(MODELS):
        if (rmse[3] <= 0).any():
            result = paired_statistics(np.zeros(len(data.seeds)), np.zeros(len(data.seeds)),
                                       null_difference=.05, alternative="greater")
            result["status"] = "undefined_zero_persistence_RMSE"
        else:
            result = paired_statistics(1-rmse[a]/rmse[3], np.zeros(len(data.seeds)),
                                       null_difference=.05, alternative="greater")
        rows.append(dict(comparison=f"{model} relative gain > 5%", model_a=model, model_b="Persistencia",
                         estimand="mean seed relative RMSE gain above 0.05", improvement_sign="positive favors model",
                         **result))
    return add_holm(rows, "nine_predeclared_development_mean_seed_hypotheses")


def temporal_tables(data: PredictionSet, segment: np.ndarray):
    mean_losses = data.squared_errors.mean(axis=1)[:, segment]
    rows = []
    for lag in [14, 7, 28]:
        group = []
        for a, b in combinations(range(4), 2):
            row = dm_hln(mean_losses[a]-mean_losses[b], lag=lag, horizon=1)
            row.update(model_a=METHODS[a], model_b=METHODS[b],
                       first_day=str(data.dates[segment[0]].date()), last_day=str(data.dates[segment[-1]].date()),
                       lag_role="primary" if lag == 14 else "predeclared sensitivity; not separate confirmation")
            group.append(row)
        if lag == 14:
            add_holm(group, "six_temporal_loss_comparisons_primary_HAC14")
        rows.extend(group)
    return rows


def diagnostic_tables(data: PredictionSet, segment: np.ndarray):
    if DESIGNATED_SEED not in data.seeds:
        raise ValueError("The predeclared diagnostic seed1001 is missing; no favorable replacement is permitted")
    seed_index = int(np.flatnonzero(data.seeds == DESIGNATED_SEED)[0])
    rows, acf_rows = [], []
    for i, model in enumerate(MODELS):
        residual = (data.truth-data.predictions[i, seed_index])[segment]
        sd = (0.0 if np.ptp(residual) == 0 else float(residual.std(ddof=1))) if len(residual) >= 2 else np.nan
        valid = len(residual) > 28 and np.isfinite(sd) and sd > 0
        if valid:
            values = acf(residual, nlags=28, fft=False)
            for lag, value in enumerate(values):
                acf_rows.append({"model": model, "seed": DESIGNATED_SEED, "lag": lag,
                                 "acf": float(value), "n_dates": len(residual), "scope": SCOPE})
            lb = acorr_ljungbox(residual, lags=[7, 14, 28], model_df=0, return_df=True)
        for lag in [7, 14, 28]:
            row = {"model": model, "seed": DESIGNATED_SEED, "test": "Ljung-Box", "lag": lag,
                   "model_df": 0, "reference_df": lag, "n_dates": len(residual), "statistic": np.nan,
                   "status": "undefined_short_or_constant_residuals", "scope": SCOPE,
                   "reference": "asymptotic chi-square under conditions; no ARMA fitted to these residuals, no NN weight subtraction"}
            row.update(p_fields(np.nan))
            if valid:
                statistic = float(lb.loc[lag, "lb_stat"])
                row.update(statistic=statistic, status="calculated_approximate")
                row.update(p_fields(chi_square_logsf(statistic, lag)))
            rows.append(row)
        bds_statistics = None
        caught = ""
        if valid:
            try:
                with warnings.catch_warnings(record=True) as messages:
                    bds_statistics, _ = bds(residual, max_dim=3, epsilon=1.5*sd)
                caught = "; ".join(str(message.message) for message in messages)
                bds_statistics = np.atleast_1d(bds_statistics)
            except (ValueError, ZeroDivisionError, FloatingPointError) as exc:
                caught = f"{type(exc).__name__}: {exc}"
        for dim in [2, 3]:
            row = {"model": model, "seed": DESIGNATED_SEED, "test": "BDS", "dimension": dim,
                   "epsilon": 1.5*sd if np.isfinite(sd) else np.nan, "epsilon_rule": "1.5 sample residual SD (ddof1)",
                   "n_dates": len(residual), "embedding_vectors": max(0, len(residual)-dim+1),
                   "statistic": np.nan, "status": "undefined_short_constant_or_invalid", "warning": caught,
                   "reference": "asymptotic standard normal under i.i.d. null and its conditions; rejection not exclusive nonlinearity",
                   "scope": SCOPE}
            row.update(p_fields(np.nan))
            if bds_statistics is not None and np.isfinite(bds_statistics[dim-2]):
                z = float(bds_statistics[dim-2])
                row.update(statistic=z, status="calculated_approximate")
                row.update(p_fields(float(np.log(2)+stats.norm.logsf(abs(z)))))
            rows.append(row)
    add_holm(rows, "fifteen_predeclared_residual_diagnostics_designated_seed1001")
    return rows, acf_rows


def bootstrap_tables(data: PredictionSet, segment, replicates):
    losses = data.squared_errors[:, :, segment]
    observed = mean_seed_rmse(losses)
    rows, rankings = [], []
    for length in [14, 7, 28]:
        draws = bootstrap_scores(losses, length=length, replicates=replicates, seed=20261104)
        for a, b in combinations(range(4), 2):
            difference = draws[:, a]-draws[:, b]
            valid = draws[:, b] > 0
            relative = 1-draws[valid, a]/draws[valid, b]
            diff_ci = np.quantile(difference, [.025, .975])
            rel_ci = np.quantile(relative, [.025, .975]) if valid.all() else [np.nan, np.nan]
            rows.append({"model_a": METHODS[a], "model_b": METHODS[b], "block_length": length,
                         "length_role": "primary" if length == 14 else "sensitivity",
                         "replicates": replicates, "rng_seed": 20261104, "n_dates": len(segment),
                         "rmse_difference": observed[a]-observed[b], "ci95_difference_lower": diff_ci[0],
                         "ci95_difference_upper": diff_ci[1],
                         "relative_gain_fraction": 1-observed[a]/observed[b] if observed[b] > 0 else np.nan,
                         "ci95_relative_lower": rel_ci[0], "ci95_relative_upper": rel_ci[1],
                         "relative_replicates_defined": int(valid.sum()),
                         "ci_method": "paired circular moving-block percentile bootstrap of dates, marginal95%",
                         "functional": "mean across seed-specific RMSE; persistence denominator recomputed jointly",
                         "interval_scope": "temporal evaluation conditional on stored trained models; no retraining/seed resampling",
                         "scope": SCOPE})
        for include_persistence in [False, True]:
            k = 4 if include_persistence else 3
            minimum = draws[:, :k].min(axis=1, keepdims=True)
            winner = draws[:, :k] == minimum
            frequencies = (winner/winner.sum(axis=1, keepdims=True)).mean(axis=0)
            for i in range(k):
                rankings.append({"model": METHODS[i], "block_length": length,
                                 "comparison_set": "three_DL_and_persistence" if include_persistence else "three_DL",
                                 "bootstrap_lowest_error_frequency": float(frequencies[i]),
                                 "ties": "equal share among exact tied minima", "replicates": replicates,
                                 "meaning": "resampling frequency, not posterior probability or proof of population ranking",
                                 "scope": SCOPE})
    return rows, rankings


def save_figure(fig, path):
    fig.text(.5, .008, CREDIT, ha="center", fontsize=7)
    fig.tight_layout(rect=(0, .035, 1, 1))
    fig.savefig(path, dpi=160, metadata={"Description": SCOPE})
    plt.close(fig)


def plots(data: PredictionSet, metrics, acf_rows, directory: Path):
    directory.mkdir(exist_ok=True)
    selected_seed = int(np.flatnonzero(data.seeds == DESIGNATED_SEED)[0])
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.5), sharex=True)
    for i, model in enumerate(MODELS):
        axes[i].plot(data.dates, data.truth, color="black", linewidth=1.3, label="Volatilidad observada")
        axes[i].plot(data.dates, data.predictions[i, selected_seed], linewidth=1.1, label=f"{model}, semilla1001")
        axes[i].plot(data.dates, data.predictions[-1, 0], color="gray", alpha=.6, linewidth=.8, label="Persistencia")
        axes[i].set_ylabel("σ7 diaria");axes[i].grid(alpha=.2);axes[i].legend(loc="upper right", fontsize=8)
    axes[0].set_title("Predicciones de desarrollo: corrida predefinida, sin evaluación del test")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    save_figure(fig, directory/"predictions_designated_seed.png")
    entries = {row["model"]: row for row in metrics if row["metric"] == "rmse"}
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    means = [entries[m]["mean"] for m in METHODS]
    deviation = [entries[m]["sample_sd"] if m != "Persistencia" else 0 for m in METHODS]
    ax.bar(METHODS, means, yerr=deviation, capsize=5, color=["#3465a4", "#3a9679", "#bc6c25", "#888888"])
    ax.set_ylabel("RMSE en escala original");ax.set_title("Desarrollo: media ± DE entre semillas\nPersistencia es un único método determinista")
    ax.grid(axis="y", alpha=.2)
    save_figure(fig, directory/"rmse_seed_variability.png")
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for i, model in enumerate(MODELS):
        rows = [row for row in acf_rows if row["model"] == model and row["lag"] > 0]
        if rows:
            axes[i].stem([r["lag"] for r in rows], [r["acf"] for r in rows], basefmt=" ")
        axes[i].axhline(0, color="black", linewidth=.7);axes[i].set_ylabel(model);axes[i].grid(alpha=.2)
    axes[0].set_title("ACF residual de desarrollo, semilla1001; diagnóstico condicionado")
    axes[-1].set_xlabel("Rezago en días")
    save_figure(fig, directory/"residual_acf_designated_seed.png")


def learning_curves(runs: pd.DataFrame, result_dir: Path, directory: Path) -> list[dict]:
    """Plot real inner-fit logs only; never mix refit training with inner validation."""
    histories, report = {}, []
    required = {"epoch", "train_mse_scaled_eval", "validation_mse_scaled_eval",
                "train_rmse_original", "validation_rmse_original"}
    for model in MODELS:
        selected = runs[(runs.model == model) & (runs.seed == DESIGNATED_SEED)]
        if len(selected) != 1:
            raise ValueError(f"Expected one designated-seed run for learning curves: {model}")
        run = selected.iloc[0]
        path = result_dir/"runs"/str(run.run_id)/"history_inner.csv"
        item = {"model": model, "seed": DESIGNATED_SEED, "run_id": str(run.run_id),
                "source": str(path), "scope": "inner chronological stopping data; not benchmark or test curves"}
        if not path.exists():
            item["status"] = "not_provided"
        else:
            history = pd.read_csv(path).sort_values("epoch")
            if not required.issubset(history) or history.empty or history.epoch.duplicated().any():
                raise ValueError(f"Invalid inner-history schema in {path}")
            if not np.isfinite(history[list(required)].to_numpy(dtype=float)).all():
                raise ValueError(f"Inner-history plot cannot silently omit nonfinite metrics in {path}")
            best = int(run.best_epoch)
            if best not in set(history.epoch):
                raise ValueError(f"Selected epoch is absent from its real inner history: {path}")
            histories[model] = (history, best)
            item.update(status="plotted", epochs_logged=len(history), best_epoch=best,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        report.append(item)
    if not histories:
        return report
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), squeeze=False)
    for i, model in enumerate(MODELS):
        if model not in histories:
            axes[i, 0].text(.5, .5, "Historial no proporcionado", ha="center")
            axes[i, 1].text(.5, .5, "Historial no proporcionado", ha="center")
            continue
        history, best = histories[model]
        for j, (train, validation, ylabel) in enumerate([
                ("train_rmse_original", "validation_rmse_original", "RMSE en escala original"),
                ("train_mse_scaled_eval", "validation_mse_scaled_eval", "MSE con escala interna de Y")]):
            ax = axes[i, j]
            ax.plot(history.epoch, history[train], label="Entrenamiento interno")
            ax.plot(history.epoch, history[validation], label="Validación interna")
            ax.axvline(best, color="black", linestyle="--", linewidth=.8,
                       label=f"Época seleccionada: {best}")
            ax.set_title(model);ax.set_ylabel(ylabel);ax.set_xlabel("Época")
            ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle("Curvas reales de parada interna, semilla1001\nConjuntos y preprocesamiento comparables dentro de cada ajuste", y=.998)
    save_figure(fig, directory/"learning_curves_designated_seed.png")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    result_dir = args.results_dir.resolve()
    output = result_dir/"analysis"
    output.mkdir(parents=True, exist_ok=True)
    source = result_dir/"predictions/development_benchmark.csv"
    data = load_predictions(source)
    expected_dates = pd.date_range("2024-07-01", "2024-12-31", freq="D", tz="UTC")
    if not data.dates.as_unit("ns").equals(expected_dates.as_unit("ns")):
        raise ValueError("The registered development benchmark requires the184 common dates July1--December31,2024")
    runs_path = result_dir/"runs.csv"
    runs = pd.read_csv(runs_path)
    runs = runs[runs.phase == PHASE].copy()
    if runs.duplicated(["model", "seed"]).any() or len(runs) != 3*len(data.seeds):
        raise ValueError("The run table must contain one paired record per model/seed")
    segment = longest_contiguous_indices(data.dates)
    if len(segment) <= 28:
        raise ValueError("The common contiguous development segment is too short for the registered diagnostics")
    metrics = build_metric_summary(data, runs)
    paired = paired_test_table(data)
    temporal = temporal_tables(data, segment)
    diagnostics, acf_rows = diagnostic_tables(data, segment)
    bootstrap, ranking = bootstrap_tables(data, segment, args.bootstrap_replicates)
    for rows, name in [(metrics, "metrics_summary.csv"), (paired, "paired_tests.csv"),
                       (temporal, "temporal_dm.csv"), (diagnostics, "diagnostics.csv"),
                       (acf_rows, "acf.csv"), (bootstrap, "block_bootstrap.csv"),
                       (ranking, "ranking_bootstrap.csv")]:
        write_frame(rows, output/name)
    values = [np.sqrt(data.squared_errors[i].mean(axis=1)) for i in range(3)]
    friedman = {"status": "undefined", "scope": SCOPE,
                "unit": "training seeds as paired blocks, conditional on fixed development calendar",
                "reference": "asymptotic chi-square; not independent markets", "n_blocks": len(data.seeds)}
    with warnings.catch_warnings(record=True):
        result = stats.friedmanchisquare(*values)
    if np.isfinite(result.statistic) and np.isfinite(result.pvalue):
        friedman.update(status="calculated_approximate", statistic=float(result.statistic), p_value=float(result.pvalue))
    plots(data, metrics, acf_rows, output/"figures")
    curves = learning_curves(runs, result_dir, output/"figures")
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "phase": PHASE,
              "scope": SCOPE, "reserved_test_evaluated": False, "raw_market_files_opened": 0,
              "sources": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                          for p in [source, runs_path]],
              "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                           "statsmodels": statsmodels.__version__, "matplotlib": matplotlib.__version__},
              "n_seeds": len(data.seeds), "seeds": data.seeds, "history_days": data.history_days,
              "n_dates": len(data.dates), "first_day": str(data.dates[0].date()), "last_day": str(data.dates[-1].date()),
              "cohort_dates_sha256": hashlib.sha256(data.dates.as_unit("ns").asi8.astype("<i8").tobytes()).hexdigest(),
              "cohort_hash_encoding": "sorted UTC timestamps, nanoseconds since Unix epoch, little-endian int64",
              "common_contiguous_segment": {"n_dates": len(segment), "first_day": str(data.dates[segment[0]].date()),
                                             "last_day": str(data.dates[segment[-1]].date()),
                                             "selection_rule": "longest daily segment by availability, earliest on ties"},
              "metrics": metrics, "paired_tests": paired, "friedman": friedman,
              "diagnostic_seed": DESIGNATED_SEED, "diagnostic_family_size": 15,
              "learning_curves": curves,
              "paired_family_size": 9, "temporal_primary_family_size": 6,
              "bootstrap": {"replicates": args.bootstrap_replicates, "primary_block_length": 14,
                            "sensitivity_lengths": [7, 28], "rng_seed": 20261104,
                            "interval_scope": "stored-model temporal evaluation uncertainty; no refitting or seed resampling",
                            "p_values_from_bootstrap": False},
              "mde_and_power": "Refer to the separately registered pilot/dimensioning report; not recomputed from observed benchmark effects.",
              "limitations": ["Development calendar has been exposed to EDA and pilot size calibration; no held-out confirmation.",
                              "Seed-level t intervals/tests require their conditional design assumptions; Shapiro is diagnostic, not a test selector.",
                              "Marginal95% intervals are not simultaneous merely because p-values have Holm correction.",
                              "HAC/HLN and residual diagnostic references are approximate; nonstationarity and overlapping labels limit interpretation.",
                              "Bootstrap ranking frequencies are not posterior probabilities or evidence of universal architecture superiority."]}
    (output/"summary.json").write_text(json.dumps(clean_json(report), ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    (output/"README.md").write_text(
        "# Análisis del benchmark de desarrollo\n\n"+SCOPE+"\n\n"
        "Las tablas separan incertidumbre entre semillas, diferencias temporales de pérdidas, bootstrap de fechas y diagnósticos residuales. "
        "Los intervalos95% son marginales. Los tres grupos de contrastes temporales/diagnósticos no constituyen confirmaciones independientes. "
        "La persistencia es determinista y no representa R entrenamientos. El test de 2025–2026 no se lee ni se evalúa aquí.\n")
    print(json.dumps({"analysis_dir": str(output), "n_seeds": len(data.seeds), "n_dates": len(data.dates),
                      "paired_tests": len(paired), "temporal_tests": len(temporal), "diagnostics": len(diagnostics),
                      "reserved_test_evaluated": False}))


if __name__ == "__main__":
    main()
