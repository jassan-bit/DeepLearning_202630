"""Post-inspection statistical supplement on saved, frozen-model test forecasts.

Only aggregate statistics and figures are written. This program does not train,
run model forward passes, open raw market ZIPs or change the selected baseline.
The protocol is registered after the test was inspected, before these additional
calculations; the results must never be presented as prospective confirmation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", "/tmp/e1-final-inference-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/e1-final-inference-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import special, stats

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/e1_audit_improvements_20261104/protocol.json"
OUTPUT = ROOT / "results/e1_audit_improvements_20261104/inference"
FORECASTS = ROOT / "results/filipe_multihistory_20261104/test/predictions.csv"
FREEZE = ROOT / "experiments/filipe_multihistory_20261104/freeze.json"
MODELS = ("MLP", "LSTM", "CNN1D")
METHODS = (*MODELS, "Persistencia")
SEEDS = tuple(range(4101, 4111))
PAIRS = tuple(combinations(range(4), 2))
SCOPE = ("Post-hoc supplement registered after inspection of the final test. "
         "Primary three D7 configurations were chosen using development CV; "
         "no test-driven reselection, training or model inference is performed.")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                                   allow_nan=False)+"\n")


def probability(log_probability, adjusted=False):
    """Keep logarithmic tails when their binary64 probability underflows."""
    prefix = "p_holm" if adjusted else "p_value"
    log_key = "log_p_holm" if adjusted else "log_p_value"
    if np.isnan(log_probability):
        return {prefix: np.nan, log_key: np.nan, prefix+"_display": "undefined",
                prefix+"_underflow": False}
    log_probability = min(0., float(log_probability))
    value = float(np.exp(log_probability))
    underflow = bool(value == 0. and np.isfinite(log_probability))
    display = f"exp({log_probability:.9g})" if underflow else f"{value:.12g}"
    return {prefix: value, log_key: log_probability, prefix+"_display": display,
            prefix+"_underflow": underflow}


def holm(rows, family):
    """Undefined hypotheses stay in the planned family but remain undefined."""
    count = len(rows)
    values = np.asarray([row.get("log_p_value", np.nan) for row in rows], dtype=float)
    order = np.argsort(np.where(np.isnan(values), 0., values), kind="stable")
    maximum = -np.inf
    adjusted = np.full(count, np.nan)
    for rank, index in enumerate(order):
        candidate = (values[index] if not np.isnan(values[index]) else 0.) + math.log(count-rank)
        maximum = max(maximum, min(0., candidate))
        if not np.isnan(values[index]):
            adjusted[index] = maximum
    for index, row in enumerate(rows):
        row.update(family=family, family_size_planned=count,
                   **probability(adjusted[index], adjusted=True))
    return rows


@dataclass
class Cube:
    dates: pd.DatetimeIndex
    target: np.ndarray
    predictions: np.ndarray  # method, training seed, date; P broadcast only algebraically

    @property
    def errors(self):
        return self.predictions-self.target[None, None, :]

    def seed_scores(self, metric="rmse"):
        if metric == "rmse":
            return np.sqrt(np.mean(self.errors**2, axis=2))
        if metric == "mae":
            return np.mean(np.abs(self.errors), axis=2)
        raise ValueError("Only predeclared RMSE/MAE estimands are admitted")


def cube_from_frame(frame, expected_dates=None, seeds=SEEDS):
    """Admit only common causal D7/h1 paths; tests may supply a shorter calendar."""
    dates = (pd.date_range("2025-01-01", "2026-08-31", tz="UTC")
             if expected_dates is None else pd.DatetimeIndex(expected_dates))
    if dates.tz is None or len(dates) < 4 or not dates.is_unique:
        raise ValueError("A nonempty unique UTC evaluation calendar is required")
    expected = dates.strftime("%Y-%m-%d").tolist()
    needed = {"model", "seed", "history_days", "variant", "outcome_day", "fold_id",
              "y_true", "y_pred", "persistence_pred"}
    if not needed <= set(frame):
        raise ValueError("Stored test forecasts lack required columns")
    frame = frame.loc[frame.history_days.eq(7)].copy()
    if (set(frame.model) != set(MODELS) or set(frame.seed) != set(seeds)
            or set(frame.variant) != {"relative_window"}
            or len(frame) != len(MODELS)*len(seeds)*len(dates)
            or frame.duplicated(["model", "seed", "outcome_day"]).any()):
        raise ValueError("Exactly the complete selected D7/model/seed grid is required")
    predictions = np.empty((4, len(seeds), len(dates)))
    target = reference = None
    for i, model in enumerate(MODELS):
        for j, seed in enumerate(seeds):
            group = frame.loc[frame.model.eq(model) & frame.seed.eq(seed)].sort_values("outcome_day")
            if (group.outcome_day.tolist() != expected
                    or group.fold_id.tolist() != [date[:7] for date in expected]):
                raise ValueError("Every path must cover the same declared dates/months")
            if not np.isfinite(group[["y_true", "y_pred", "persistence_pred"]]).all().all():
                raise ValueError("Predictions and labels must be finite")
            if (group[["y_true", "y_pred", "persistence_pred"]] < 0).any().any():
                raise ValueError("Volatility labels and forecasts must be nonnegative")
            if "horizon" in group and not group.horizon.eq(1).all():
                raise ValueError("Only next-day forecasts were predeclared")
            for field, wanted in [("input_first_day", dates-pd.Timedelta(days=7)),
                                  ("input_last_day", dates-pd.Timedelta(days=1)),
                                  ("prediction_origin_utc", dates),
                                  ("target_available_at_utc", dates+pd.Timedelta(days=1))]:
                if field in group and not np.array_equal(pd.to_datetime(group[field], utc=True), wanted):
                    raise ValueError("Forecast information clock is not causal: "+field)
            y, p = group.y_true.to_numpy(), group.persistence_pred.to_numpy()
            if target is None:
                target, reference = y, p
            elif not (np.allclose(target, y, rtol=0, atol=1e-14)
                      and np.allclose(reference, p, rtol=0, atol=1e-14)):
                raise ValueError("All methods must share the same labels and reference")
            predictions[i, j] = group.y_pred.to_numpy()
    predictions[3] = reference
    return Cube(dates, target, predictions)


def paired_result(first, second):
    difference = np.asarray(first, dtype=float)-np.asarray(second, dtype=float)
    n = len(difference)
    if n < 3 or not np.isfinite(difference).all():
        raise ValueError("At least three finite paired scores are required")
    mean = float(difference.mean())
    sd = 0. if np.ptp(difference) == 0 else float(difference.std(ddof=1))
    defined = sd > 0
    statistic = mean/(sd/np.sqrt(n)) if defined else np.nan
    critical = float(stats.t.ppf(.975, n-1))
    half = critical*sd/np.sqrt(n) if defined else np.nan
    shapiro = stats.shapiro(difference) if defined else None
    row = {"n_pairs": n, "mean_difference": mean, "sd_difference": sd,
           "ci95_lower": mean-half, "ci95_upper": mean+half,
           "t_statistic": statistic, "df": n-1,
           "d_z": mean/sd if defined else np.nan,
           "shapiro_statistic": float(shapiro.statistic) if defined else np.nan,
           "shapiro_p_value": float(shapiro.pvalue) if defined else np.nan,
           "normality_flag_p_lt_0_05": bool(shapiro.pvalue < .05) if defined else False,
           "t_reference_warning": ("Shapiro flags a departure; t reference and t CI require caution; no test selected by significance"
                                   if defined and shapiro.pvalue < .05 else
                                   "Ten pairs and non-rejection do not establish normality; calendar-conditional approximate reference"),
           "status": "calculated_under_assumptions" if defined else "undefined_zero_paired_dispersion"}
    row.update(probability(np.log(2)+stats.t.logsf(abs(statistic), n-1) if defined else np.nan))
    return row


def bootstrap_dz(difference, indices):
    """Percentile CI across paired training-seed randomizations, not dates."""
    values = np.asarray(difference)[indices]
    sd = values.std(axis=1, ddof=1)
    defined = np.isfinite(sd) & (sd > 0)
    effect = values[defined].mean(axis=1)/sd[defined]
    complete = defined.mean() >= .9
    lower, upper = np.quantile(effect, [.025, .975]) if complete else (np.nan, np.nan)
    return {"d_z_ci95_lower": float(lower), "d_z_ci95_upper": float(upper),
            "d_z_bootstrap_defined": int(defined.sum()), "d_z_bootstrap_replicates": len(indices),
            "d_z_ci_method": "percentile paired training-seed bootstrap; conditional on fixed dates",
            "d_z_ci_status": "calculated" if complete else "undefined_insufficient_nondegenerate_resamples"}


def paired_tests(cube, replicates=2000, seed=20261105):
    scores = cube.seed_scores()
    indices = np.random.default_rng(seed).integers(0, scores.shape[1], (replicates, scores.shape[1]))
    rows = []
    for a, b in PAIRS:
        row = paired_result(scores[a], scores[b])
        row.update(model_a=METHODS[a], model_b=METHODS[b], metric="rmse",
                   comparison=f"{METHODS[a]} − {METHODS[b]}",
                   estimand="mean paired training-seed RMSE difference; negative favors first",
                   reference_stochastic=b != 3,
                   inference_unit="initialization-seed pair conditional on the same608dates",
                   scope=SCOPE, **bootstrap_dz(scores[a]-scores[b], indices))
        rows.append(row)
    return holm(rows, "six_primary_final_D7_training_seed_RMSE_comparisons")


def wilcoxon_result(first, second):
    """Exact signed-rank sensitivity only with nonzero, untied differences."""
    difference = np.asarray(first, dtype=float)-np.asarray(second, dtype=float)
    zeros = int(np.sum(difference == 0))
    tied = len(np.unique(np.abs(difference[difference != 0]))) < len(difference)-zeros
    row = {"n_pairs": len(difference), "zero_differences": zeros,
           "tied_absolute_differences": bool(tied),
           "assumptions": "Independent training randomizations conditional on calendar; symmetric paired-difference location model, not mean-error hypothesis"}
    if zeros or tied:
        return {**row, "statistic": np.nan,
                "status": "undefined_standard_exact_with_zeros_or_absolute_ties",
                "method": "Standard exact signed-rank reference not used when unsupported",
                **probability(np.nan)}
    result = stats.wilcoxon(difference, zero_method="wilcox", alternative="two-sided", method="exact")
    return {**row, "statistic": float(result.statistic), "status": "calculated_exact_under_symmetry",
            "method": "two-sided exact Wilcoxon signed-rank; no zeros or absolute ties",
            "minimum_possible_two_sided_p": float(2/2**len(difference)),
            **probability(float(np.log(result.pvalue)))}


def wilcoxon_tests(cube):
    scores = cube.seed_scores()
    rows = []
    for a, b in PAIRS:
        rows.append({"model_a": METHODS[a], "model_b": METHODS[b],
                     "comparison": f"{METHODS[a]} − {METHODS[b]}",
                     "metric": "rmse", "scope": SCOPE,
                     "interpretation": "Predeclared location sensitivity, not an alternative chosen because it rejects or independent confirmation",
                     **wilcoxon_result(scores[a], scores[b])})
    return holm(rows, "six_predeclared_final_D7_signed_rank_sensitivity_comparisons")


def dm_hln(difference, lag=14, horizon=1):
    difference = np.asarray(difference, dtype=float)
    n = len(difference)
    if not np.isfinite(difference).all() or n <= lag+1 or horizon < 1:
        raise ValueError("Finite sufficiently long temporal losses required")
    centered = difference-difference.mean()
    gamma0 = float(centered@centered/n)
    lrv = gamma0
    for k in range(1, lag+1):
        lrv += 2*(1-k/(lag+1))*float(centered[k:]@centered[:-k]/n)
    factor = (n+1-2*horizon+horizon*(horizon-1)/n)/n
    defined = bool(np.ptp(difference) > 0 and lrv > 0 and factor > 0)
    dm = float(difference.mean()/np.sqrt(lrv/n)) if defined else np.nan
    statistic = float(dm*np.sqrt(factor)) if defined else np.nan
    return {"n_dates": n, "HAC_lag": lag, "forecast_horizon": horizon,
            "mean_loss_difference": float(difference.mean()), "long_run_variance": lrv,
            "DM": dm, "HLN_factor": float(np.sqrt(factor)) if factor > 0 else np.nan,
            "DM_HLN": statistic, "reference_df": n-1,
            "status": "calculated_approximate" if defined else "undefined_nonpositive_HAC_or_constant_losses",
            **probability(np.log(2)+stats.t.logsf(abs(statistic), n-1) if defined else np.nan)}


def temporal_tests(cube, lengths=(14, 7, 28)):
    losses = np.mean(cube.errors**2, axis=1)
    rows = []
    for length in lengths:
        batch = []
        for a, b in PAIRS:
            row = dm_hln(losses[a]-losses[b], length)
            row.update(model_a=METHODS[a], model_b=METHODS[b],
                       comparison=f"{METHODS[a]} − {METHODS[b]}", primary=length == 14,
                       loss="date-wise mean squared error across training seeds; not ensemble error",
                       scope=SCOPE)
            batch.append(row)
        if length == 14:
            holm(batch, "six_primary_final_D7_temporal_squared_loss_comparisons")
        else:
            for row in batch:
                row.update(family="HAC_sensitivity_no_reselection", family_size_planned=6,
                           **probability(np.nan, adjusted=True))
        rows += batch
    return rows


def chi2_logsf(statistic, df):
    value = float(stats.chi2.logsf(statistic, df))
    if np.isfinite(value) or statistic == 0:
        return value
    if not np.isfinite(statistic) or statistic < 0 or df <= 0:
        return np.nan
    # Stable upper-incomplete-gamma expansion in the extreme tail. In this branch
    # x is sufficiently large for scipy's survival probability to underflow.
    a, x = df/2., statistic/2.
    term = total = 1.
    previous = np.inf
    for k in range(1, 1000):
        term *= (a-k)/x
        if abs(term) > previous:
            break
        total += term
        if abs(term) <= abs(total)*1e-15:
            break
        previous = abs(term)
    return float(-x+(a-1)*np.log(x)-special.gammaln(a)+np.log(total))


def ljung_box_result(residual, lag):
    residual = np.asarray(residual, dtype=float)
    n = len(residual)
    if not np.isfinite(residual).all() or n <= lag:
        raise ValueError("Finite residuals longer than the requested lag required")
    centered = residual-residual.mean()
    denominator = float(centered@centered)
    if denominator <= 0 or np.ptp(residual) == 0:
        return {"statistic": np.nan, "n_dates": n, "lag": lag, "model_df": 0,
                "reference_df": lag, "status": "undefined_constant_residual", **probability(np.nan)}
    correlations = np.asarray([centered[k:]@centered[:-k]/denominator for k in range(1, lag+1)])
    statistic = float(n*(n+2)*np.sum(correlations**2/(n-np.arange(1, lag+1))))
    return {"statistic": statistic, "n_dates": n, "lag": lag, "model_df": 0,
            "reference_df": lag, "status": "calculated_approximate",
            **probability(chi2_logsf(statistic, lag))}


def residual_tests(cube, lags=(7, 14, 28), seed=4101):
    index = SEEDS.index(seed)
    rows = []
    for i, model in enumerate(MODELS):
        residual = cube.target-cube.predictions[i, index]
        for lag in lags:
            rows.append({"model": model, "seed": seed, **ljung_box_result(residual, lag),
                         "scope": SCOPE,
                         "limitation": "Overlapping sigma7 labels and nonstationarity affect asymptotic calibration; model_df0 is not NN parameter count."})
    return holm(rows, "nine_designated_seed_final_D7_LjungBox_diagnostics")


def circular_indices(n, length, replicates, rng):
    starts = rng.integers(0, n, size=(replicates, math.ceil(n/length)))
    return ((starts[:, :, None]+np.arange(length)) % n).reshape(replicates, -1)[:, :n]


def bootstrap_date_scores(cube, length, replicates, seed):
    n = len(cube.dates)
    indices = circular_indices(n, length, replicates, np.random.default_rng(seed+length))
    counts = np.zeros((replicates, n), dtype=float)
    for row, sequence in enumerate(indices):
        counts[row] = np.bincount(sequence, minlength=n)/n
    errors = cube.errors
    squared = (counts @ (errors**2).reshape(-1, n).T).reshape(replicates, 4, errors.shape[1])
    absolute = (counts @ np.abs(errors).reshape(-1, n).T).reshape(replicates, 4, errors.shape[1])
    return {"rmse": np.sqrt(np.maximum(squared, 0.)).mean(axis=2), "mae": absolute.mean(axis=2)}


def date_bootstrap(cube, lengths=(14, 7, 28), replicates=2000, seed=20261104):
    rows = []
    for length in lengths:
        samples = bootstrap_date_scores(cube, length, replicates, seed)
        for metric in ("rmse", "mae"):
            observed = cube.seed_scores(metric).mean(axis=1)
            for a, b in PAIRS:
                difference = samples[metric][:, a]-samples[metric][:, b]
                lower, upper = np.quantile(difference, [.025, .975])
                reference = samples[metric][:, b]
                valid = reference > 0
                gains = 100*(reference[valid]-samples[metric][valid, a])/reference[valid]
                gl, gu = np.quantile(gains, [.025, .975]) if len(gains) else (np.nan, np.nan)
                rows.append({"model_a": METHODS[a], "model_b": METHODS[b], "metric": metric,
                    "comparison": f"{METHODS[a]} − {METHODS[b]}", "block_length": length,
                    "primary": length == 14, "bootstrap_replicates": replicates,
                    "estimate_difference": float(observed[a]-observed[b]),
                    "ci95_lower": float(lower), "ci95_upper": float(upper),
                    "relative_gain_pct": float(100*(observed[b]-observed[a])/observed[b]) if observed[b] > 0 else np.nan,
                    "ci95_relative_gain_lower": float(gl), "ci95_relative_gain_upper": float(gu),
                    "relative_defined_replicates": int(valid.sum()),
                    "ci_method": "marginal percentile paired circular DATE bootstrap; no seed resampling or refitting",
                    "scope": SCOPE})
    return rows


def sensitivity(cube, family_size=6):
    scores = cube.seed_scores()
    alpha = .05/family_size
    multiplier = stats.norm.ppf(1-alpha/2)+stats.norm.ppf(.8)
    rows = []
    for a, b in PAIRS:
        difference = scores[a]-scores[b]
        sd = 0. if np.ptp(difference) == 0 else float(difference.std(ddof=1))
        rows.append({"model_a": METHODS[a], "model_b": METHODS[b], "n_seed_pairs": len(difference),
            "sd_paired_seed_RMSE": sd, "planning_alpha": alpha, "planning_family_size": family_size,
            "hypothetical_power_target": .8,
            "normal_approximation_MDE0": float(multiplier*sd/np.sqrt(len(difference))) if sd > 0 else np.nan,
            "status": "retrospective_conditional_design_sensitivity" if sd > 0 else "undefined_zero_dispersion",
            "scope": "Dispersion-based MDE against zero, after observing test; not a prior pilot, achieved power, or power to prove an advanced improvement above a practical boundary."})
    return rows


def relative_differences(cube, best="MLP", worst="LSTM"):
    """References fixed by CV primary-metric ordering, never by test ordering."""
    rows = []
    for metric in ("rmse", "mae"):
        scores = cube.seed_scores(metric).mean(axis=1)
        for role, reference in (("CV_best_neural_RMSE", best), ("CV_worst_neural_RMSE", worst),
                                ("deterministic_context", "Persistencia")):
            index = METHODS.index(reference)
            for i, model in enumerate(METHODS):
                rows.append({"model": model, "reference": reference, "reference_role": role,
                    "metric": metric, "mean_model_score": float(scores[i]),
                    "mean_reference_score": float(scores[index]),
                    "absolute_model_minus_reference": float(scores[i]-scores[index]),
                    "relative_error_reduction_pct": float(100*(scores[index]-scores[i])/scores[index]) if scores[index] > 0 else np.nan,
                    "definition": "100*(mean_reference_score-mean_model_score)/mean_reference_score; positive favors model",
                    "scope": "Descriptive test differences using CV-fixed references, no baseline reselection."})
    return rows


def settings(plan):
    cfg = plan.get("final_inference", {})
    # A broader protocol may store predeclared subplans under a plans object.
    if not cfg and isinstance(plan.get("plans"), dict):
        cfg = plan["plans"].get("final_inference", {})
    return {"replicates": int(cfg.get("bootstrap_replicates", 2000)),
            "lengths": tuple(cfg.get("bootstrap_lengths", [14, 7, 28])),
            "seed": int(cfg.get("bootstrap_seed", 20261104)),
            "effect_seed": int(cfg.get("effect_bootstrap_seed", 20261105)),
            "dm_lengths": (int(cfg.get("dm_main_lag", 14)), *cfg.get("dm_sensitivity_lags", [7, 28])),
            "lb_lags": tuple(cfg.get("lb_lags", [7, 14, 28])),
            "designated_seed": int(cfg.get("designated_seed", 4101)),
            "pair_family_size": int(cfg.get("pair_family_size", 6))}


def validate_plan(path=PROTOCOL):
    path = Path(path)
    plan = json.loads(path.read_text())
    cfg = settings(plan)
    if (cfg["pair_family_size"] != 6 or cfg["replicates"] != 2000
            or cfg["lengths"] != (14, 7, 28) or cfg["dm_lengths"] != (14, 7, 28)
            or cfg["lb_lags"] != (7, 14, 28) or cfg["designated_seed"] != 4101):
        raise ValueError("Protocol does not match the fixed final-supplement family and estimands")
    registered = plan.get("source_sha256", plan.get("sources", {}))
    if not registered:
        raise ValueError("The post-hoc plan must register its saved input hashes before calculation")
    for name, expected in registered.items():
        if sha(ROOT / name) != expected:
            raise ValueError("Registered source changed: "+name)
    freeze = json.loads(FREEZE.read_text())
    if (freeze["selection"]["baseline"]["model"] != "MLP"
            or freeze["selection"]["baseline"]["history_days"] != 7
            or any(freeze["selection"]["selected_by_model"][m]["history_days"] != 7 for m in MODELS)):
        raise ValueError("The primary configurations must retain their pre-test CV selection")
    # Registration time must precede this new analysis, not the original test.
    timestamp = plan.get("registered_at_utc", plan.get("created_at_utc"))
    if not timestamp or pd.Timestamp(timestamp) > pd.Timestamp.now(tz="UTC"):
        raise ValueError("A valid pre-calculation registration timestamp is required")
    return plan, cfg


def plots(cube, paired, bootstrap, output):
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2))
    scores = cube.seed_scores()
    for axis, (a, b) in zip(axes.ravel(), PAIRS):
        difference = scores[a]-scores[b]
        if np.ptp(difference) > 0:
            stats.probplot(difference, dist="norm", plot=axis)
        else:
            axis.text(.5, .5, "Dispersión nula", ha="center", transform=axis.transAxes)
        axis.set_title(f"{METHODS[a]} − {METHODS[b]}", fontsize=10)
        axis.set_xlabel("Cuantil normal teórico", fontsize=8)
        axis.set_ylabel("Δ RMSE entre semillas", fontsize=8)
        axis.tick_params(labelsize=8)
    fig.suptitle("Q–Q de diferencias pareadas · D7 · test final · análisis posterior", fontsize=12)
    fig.tight_layout(rect=(0, .02, 1, .95))
    fig.savefig(output / "paired_qq.png", dpi=170)
    plt.close(fig)
    data = [row for row in bootstrap if row["metric"] == "rmse" and row["primary"]]
    fig, axis = plt.subplots(figsize=(10, 4.6))
    for i, row in enumerate(data):
        axis.plot([row["ci95_lower"], row["ci95_upper"]], [i, i], color="#31688e", lw=3)
        axis.plot(row["estimate_difference"], i, "o", color="#202020")
    axis.set_yticks(range(len(data)), [row["comparison"] for row in data])
    axis.axvline(0, color="#555555", ls="--", lw=1)
    axis.set_xlabel("Δ RMSE media por semilla; negativo favorece al primer método")
    axis.set_title("IC marginales de fechas · bloques circulares de 14 días · post-hoc")
    axis.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "date_bootstrap_intervals.png", dpi=170)
    plt.close(fig)


def analyze(protocol_path=PROTOCOL, output=OUTPUT):
    plan, cfg = validate_plan(protocol_path)
    output = Path(output)
    if (output / "summary.json").exists():
        old = json.loads((output / "summary.json").read_text())
        if old["protocol_sha256"] != sha(protocol_path) or old["analysis_code_sha256"] != sha(__file__):
            raise ValueError("Preserve a completed supplement when code/protocol changes")
        for name, expected in old["output_sha256"].items():
            if sha(output / name) != expected:
                raise ValueError("Completed aggregate output changed: "+name)
        return old
    output.parent.mkdir(parents=True, exist_ok=True)
    registration_path = output.parent / "inference_registration.json"
    registration = {"registered_before_supplemental_calculation_at_utc": datetime.now(timezone.utc).isoformat(),
                    "protocol_sha256": sha(protocol_path), "analysis_code_sha256": sha(__file__),
                    "timing": "After test inspection; before this additional calculation. Not pre-test registration.",
                    "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
                    "new_project_fits": 0, "checkpoint_forward_calls": 0, "raw_market_archives_opened": 0}
    if registration_path.exists():
        previous = json.loads(registration_path.read_text())
        if (previous["protocol_sha256"] != registration["protocol_sha256"]
                or previous["analysis_code_sha256"] != registration["analysis_code_sha256"]):
            raise ValueError("Previously registered supplement code/protocol changed")
    else:
        write_json(registration_path, registration)
    cube = cube_from_frame(pd.read_csv(FORECASTS))
    paired = paired_tests(cube, cfg["replicates"], cfg["effect_seed"])
    wilcoxon = wilcoxon_tests(cube)
    dm = temporal_tests(cube, cfg["dm_lengths"])
    lb = residual_tests(cube, cfg["lb_lags"], cfg["designated_seed"])
    bootstrap = date_bootstrap(cube, cfg["lengths"], cfg["replicates"], cfg["seed"])
    tables = {"paired_comparisons.csv": paired, "wilcoxon_sensitivity.csv": wilcoxon,
              "dm_tests.csv": dm, "ljung_box.csv": lb,
              "date_bootstrap.csv": bootstrap, "design_sensitivity.csv": sensitivity(cube),
              "relative_differences.csv": relative_differences(cube)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".final_inference_", dir=output.parent) as temporary:
        stage = Path(temporary)
        for name, rows in tables.items():
            pd.DataFrame(rows).to_csv(stage / name, index=False, lineterminator="\n")
        plots(cube, paired, bootstrap, stage)
        (stage / "README.md").write_text("""# Inferencia complementaria del test final: D7

El plan se registró **después de observar los resultados del test**, antes de estos cálculos. Es un suplemento post-hoc/exploratorio, no una confirmación prospectiva. Se reutilizan las predicciones de MLP/LSTM/CNN1D con D7 y diez semillas, seleccionadas previamente por CV, y la persistencia determinista. No se cambian el baseline MLP/D7, los modelos, el periodo ni sus pesos.

`paired_comparisons.csv` contiene seis comparaciones bilaterales de RMSE por semillas, Holm-6, IC t marginales, d_z e IC percentile de d_z por remuestreo pareado de semillas. Las seis diferencias están condicionadas al mismo calendario. La persistencia se resta como puntuación fija; no representa diez entrenamientos independientes. Shapiro y Q–Q no certifican normalidad con diez pares.

`wilcoxon_sensitivity.csv` conserva seis sensibilidades de localización signed-rank con Holm-6 separado. Solo usa la distribución exacta cuando las diferencias son distintas de cero y sus valores absolutos no tienen empates; de otro modo conserva estado no definido. Requiere simetría para interpretar un parámetro de localización y no contrasta la misma media que t. No se elige la prueba que produzca rechazo; ninguna de estas referencias constituye confirmación independiente de las demás.

`dm_tests.csv` compara pérdidas cuadráticas por fecha, promediadas sobre semillas; usa Bartlett/HAC14, corrección HLN de h=1 y Holm-6 para los seis pares principales. HAC7/28 son sensibilidades descriptivas y no se selecciona la variante favorable. La pérdida media por fecha no es el error de una predicción ensemble ni la RMSE media por semilla.

`ljung_box.csv` diagnostica los residuos de la semilla designada4101 de las tres redes, lags7/14/28, model_df0 y Holm-9. No se resta el número de parámetros de la red como grados de libertad ARMA. Las etiquetas sigma7 solapadas, la deriva y la aproximación asintótica limitan la calibración.

`date_bootstrap.csv` conserva 2.000 muestras de bloques circulares pareados de fechas, L14 principal y L7/28 de sensibilidad, para ΔRMSE/ΔMAE y reducciones relativas. Recalcula primero cada métrica por semilla y después su media. No remuestrea semillas, reajusta modelos ni identifica incertidumbre entre mercados. Los intervalos de esta tabla son marginales, no simultáneos por llevar p-valores Holm en otra tabla.

`relative_differences.csv` fija MLP como mejor base y LSTM como peor base según la métrica primaria de **CV**, sin elegir referentes por el test; también informa persistencia. Sus reducciones son100×(error medio de referencia−error medio del modelo)/error medio de referencia. `design_sensitivity.csv` calcula una MDE frente a cero desde la dispersión observada, α de planificación0,05/6 y potencia hipotética0,80. No es piloto prospectivo, potencia lograda ni dimensión para demostrar una mejora superior a un margen práctico.

Todos los archivos son agregados. No se exportan predicciones individuales, precios diarios, residuos por fecha, pesos ni paquetes completos. Rechazar autocorrelación/iid y demostrar precisión comparativa son preguntas diferentes. No se ejecuta TOST/ROPE ni se concluye equivalencia por ausencia de significancia.
""")
        report = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "scope": SCOPE,
            "registered_at_utc": plan.get("registered_at_utc", plan.get("created_at_utc")),
            "protocol_sha256": sha(protocol_path), "analysis_code_sha256": sha(__file__),
            "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (FORECASTS, FREEZE)},
            "configuration": "three pre-test CV-selected D7 primary models plus deterministic persistence",
            "baseline_selection_retained": "MLP/D7", "n_dates": len(cube.dates), "n_training_seeds": 10,
            "paired_family_size": 6, "DM_primary_family_size": 6, "LjungBox_family_size": 9,
            "bootstrap_replicates": cfg["replicates"], "bootstrap_lengths": list(cfg["lengths"]),
            "effect_bootstrap_seed": cfg["effect_seed"],
            "row_counts": {name: len(rows) for name, rows in tables.items()},
            "paired_rejections_Holm05": sum(row["p_holm"] < .05 for row in paired),
            "paired_Shapiro_flags_05": sum(row["normality_flag_p_lt_0_05"] for row in paired),
            "Wilcoxon_sensitivity_rejections_Holm05": sum(row["p_holm"] < .05 for row in wilcoxon),
            "DM_primary_rejections_Holm05": sum(row["primary"] and row["p_holm"] < .05 for row in dm),
            "LjungBox_rejections_Holm05": sum(row["p_holm"] < .05 for row in lb),
            "new_project_fits": 0, "checkpoint_forward_calls": 0, "raw_market_archives_opened": 0,
            "prospective_confirmation": False,
            "output_sha256": {path.name: sha(path) for path in sorted(stage.iterdir()) if path.is_file()}}
        write_json(stage / "summary.json", report)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("An incomplete output directory needs an explicit audit before replacement")
        output.mkdir(parents=True, exist_ok=True)
        for path in stage.iterdir():
            path.replace(output / path.name)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("validate", "run"), default="validate")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.phase == "validate":
        _, cfg = validate_plan(args.protocol)
        print(json.dumps({"status": "posthoc_plan_validated", "config": cfg, "project_calculations_executed": False}))
        return
    report = analyze(args.protocol, args.output)
    print(json.dumps({key: report[key] for key in ("scope", "n_dates", "row_counts", "paired_rejections_Holm05", "DM_primary_rejections_Holm05", "LjungBox_rejections_Holm05")}))


if __name__ == "__main__":
    main()
