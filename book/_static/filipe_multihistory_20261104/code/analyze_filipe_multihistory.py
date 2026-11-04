"""Report the four input histories from stored Filipe-CV/frozen-test forecasts.

This program never fits a model or opens raw market archives.  CV uses the
original primary D7 trajectories and the newly recorded D14/D21/D28 paths.
Test months are reporting blocks of frozen models, not additional refits.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/filipe-multihistory-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/filipe-multihistory-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import bds

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/filipe_multihistory_20261104"
HISTORIES = (7, 14, 21, 28)
MODELS = ("MLP", "LSTM", "CNN1D")
SEEDS = tuple(range(4101, 4111))
METRICS = ("mape", "mae", "rmse", "mse", "r2", "mase")
BDS_LIMIT = ("Approximate BDS iid diagnostic on short monthly blocks; overlapping "
             "volatility labels and estimated networks can affect its reference distribution. "
             "A p-value average is descriptive, never a combined hypothesis test.")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n")


def calendar_for(phase):
    if phase == "cv":
        return pd.date_range("2024-01-01", "2024-12-31", tz="UTC")
    if phase == "test":
        return pd.date_range("2025-01-01", "2026-08-31", tz="UTC")
    raise ValueError("Unknown reporting phase")


def validate_predictions(frame, phase, expected_days=None, histories=HISTORIES,
                         seeds=SEEDS):
    """Reject an incomplete grid, look-ahead clocks and unequal comparisons."""
    frame = frame.copy()
    required = {"history_days", "model", "seed", "fold_id", "outcome_day", "y_true",
                "y_pred", "persistence_pred", "mase_train_scale"}
    if not required <= set(frame):
        raise ValueError("Missing required prediction columns: " + str(required-set(frame)))
    if "variant" in frame and set(frame.variant) != {"relative_window"}:
        raise ValueError("Only the predeclared primary relative_window representation is admitted")
    if "horizon" in frame and not frame.horizon.eq(1).all():
        raise ValueError("Only the next-day h1 contract is admitted")
    frame["outcome_day"] = pd.to_datetime(frame.outcome_day, utc=True).dt.strftime("%Y-%m-%d")
    frame["history_days"] = frame.history_days.astype(int)
    frame["seed"] = frame.seed.astype(int)
    if (set(frame.history_days) != set(histories) or set(frame.model) != set(MODELS)
            or set(frame.seed) != set(seeds)
            or frame.duplicated(["history_days", "model", "seed", "outcome_day"]).any()):
        raise ValueError("A unique complete history/model/seed grid is required")
    full_days = calendar_for(phase).strftime("%Y-%m-%d").tolist()
    expected_days = full_days if expected_days is None else list(expected_days)
    if (not expected_days or sorted(set(expected_days)) != expected_days
            or not set(expected_days) <= set(full_days)):
        raise ValueError("Common outcome days must be unique, ordered and within the phase calendar")
    if len(frame) != len(histories)*len(MODELS)*len(seeds)*len(expected_days):
        raise ValueError("A complete common-calendar forecast grid is required")
    if not np.isfinite(frame[["y_true", "y_pred", "persistence_pred", "mase_train_scale"]]).all().all():
        raise ValueError("Predictions, labels and historical MASE denominators must be finite")
    if (frame.y_true < 0).any() or (frame.y_pred < 0).any() or (frame.mase_train_scale <= 0).any():
        raise ValueError("Volatility labels/forecasts must be nonnegative and historical scales positive")
    targets = references = None
    for (history, model, seed), group in frame.groupby(["history_days", "model", "seed"]):
        group = group.sort_values("outcome_day")
        if group.outcome_day.tolist() != expected_days:
            raise ValueError("Every path must cover exactly the common declared dates")
        if group.fold_id.tolist() != group.outcome_day.str[:7].tolist():
            raise ValueError("Monthly fold/block identifiers must match outcome dates")
        days = pd.DatetimeIndex(pd.to_datetime(group.outcome_day, utc=True))
        for field, expected in [("input_first_day", days-pd.Timedelta(days=int(history))),
                                ("input_last_day", days-pd.Timedelta(days=1)),
                                ("prediction_origin_utc", days),
                                ("target_available_at_utc", days+pd.Timedelta(days=1))]:
            if field in group and not np.array_equal(pd.to_datetime(group[field], utc=True), expected):
                raise ValueError("Forecast information clock is inconsistent: " + field)
        if targets is None:
            targets, references = group.y_true.to_numpy(), group.persistence_pred.to_numpy()
        elif (not np.allclose(targets, group.y_true, rtol=1e-12, atol=1e-14)
              or not np.allclose(references, group.persistence_pred, rtol=1e-12, atol=1e-14)):
            raise ValueError("All methods and histories must share labels and persistence")
    # MASE's denominator can differ between histories, but never between seeds/models
    # evaluated with the same admitted training calendar.
    for _, group in frame.groupby(["history_days", "outcome_day"]):
        if not np.allclose(group.mase_train_scale, group.mase_train_scale.iloc[0], rtol=1e-12):
            raise ValueError("Historical MASE scales differ within an admitted cohort")
    return frame.sort_values(["history_days", "model", "seed", "outcome_day"]).reset_index(drop=True)


def load_predictions(phase, root=ROOT, output=OUT):
    root, output = Path(root), Path(output)
    sources = []
    if phase == "cv":
        old_path = root / "results/dl_r2_cv_20261104/predictions.csv"
        new_path = output / "predictions.csv"
        old = pd.read_csv(old_path)
        old = old.loc[old.variant.eq("relative_window") & old.history_days.eq(7)]
        old["horizon"] = 1  # Original stored D7 forecasts use this implicit contract.
        new = pd.read_csv(new_path)
        if "horizon" not in new:
            new["horizon"] = 1
        if 7 in set(new.history_days):
            raise ValueError("D7 must retain the original recorded trajectories, without duplicate refitting")
        frame = pd.concat([old, new], ignore_index=True)
        sources = [old_path, new_path]
        days = None
    else:
        path = output / "test/predictions.csv"
        cohort_path = output / "test/cohort_manifest.json"
        cohort = json.loads(cohort_path.read_text())
        days = cohort["common_outcome_days_utc"]
        if cohort.get("calendar_days") != 608 or cohort.get("n_common_outcomes") != len(days):
            raise ValueError("Test cohort manifest has inconsistent calendar/coverage")
        frame, sources = pd.read_csv(path), [path, cohort_path]
    return validate_predictions(frame, phase, days), sources


def metric_values(group):
    target, prediction = group.y_true.to_numpy(float), group.y_pred.to_numpy(float)
    error = target-prediction
    mse = float(np.mean(error**2))
    positive = target > 0
    denominator = group.mase_train_scale.to_numpy(float)
    allowed = np.isfinite(denominator) & (denominator > 0)
    total = float(np.sum((target-target.mean())**2))
    return {"n_dates": len(group), "rmse": math.sqrt(mse), "mse": mse,
            "mae": float(np.mean(np.abs(error))),
            "r2": 1-float(np.sum(error**2))/total if total > 0 else np.nan,
            "mape": float(100*np.mean(np.abs(error[positive]/target[positive]))) if positive.any() else np.nan,
            "mape_coverage": float(positive.mean()),
            "mase": float(np.mean(np.abs(error[allowed])/denominator[allowed])) if allowed.any() else np.nan,
            "mase_coverage": float(allowed.mean())}


def metric_tables(frame):
    """Return annual/path and monthly metrics; persistence occurs once, not ten times."""
    rows, folds = [], []
    for keys, group in frame.groupby(["history_days", "model", "seed"]):
        identity = dict(zip(["history_days", "model", "seed"], keys))
        rows.append({**identity, "stochastic_training": True, **metric_values(group)})
        for month, part in group.groupby("fold_id"):
            folds.append({**identity, "fold_id": month, "stochastic_training": True,
                          **metric_values(part)})
    for history, group in frame.groupby("history_days"):
        reference = group.sort_values(["outcome_day", "model", "seed"]).drop_duplicates("outcome_day").copy()
        reference["y_pred"] = reference.persistence_pred
        identity = {"history_days": history, "model": "Persistencia", "seed": np.nan,
                    "stochastic_training": False}
        rows.append({**identity, **metric_values(reference)})
        for month, part in reference.groupby("fold_id"):
            folds.append({**identity, "fold_id": month, **metric_values(part)})
    return pd.DataFrame(rows), pd.DataFrame(folds)


def metric_summary(frame, by_fold=False):
    keys = ["history_days", "model"] + (["fold_id"] if by_fold else [])
    rows = []
    for values, group in frame.groupby(keys):
        identity = dict(zip(keys, values))
        stochastic = bool(group.stochastic_training.iloc[0])
        for metric in METRICS:
            all_values = group[metric].to_numpy(float)
            values = all_values[np.isfinite(all_values)]
            count = len(values)
            mean = float(values.mean()) if count else np.nan
            sd = float(values.std(ddof=1)) if count > 1 and stochastic else np.nan
            width = float(stats.t.ppf(.975, count-1)*sd/np.sqrt(count)) if count > 1 and stochastic else np.nan
            rows.append({**identity, "metric": metric, "n_total": len(group), "n_defined": count,
                         "n_dates": int(group.n_dates.iloc[0]), "mean": mean, "sample_sd": sd,
                         "ci95_lower": mean-width, "ci95_upper": mean+width,
                         "stochastic_training": stochastic,
                         "ci_method": "marginal Student t across initialization seeds, conditional on fixed dates"
                         if stochastic else "not applicable: deterministic reference, no seed replication",
                         "scope": "Training-seed dispersion; not uncertainty across independent markets or folds."})
    return pd.DataFrame(rows)


def descriptive_fold_spread(folds):
    rows = []
    for (history, model), group in folds.groupby(["history_days", "model"]):
        for metric in METRICS:
            monthly = group.groupby("fold_id")[metric].mean()
            rows.append({"history_days": history, "model": model, "metric": metric,
                         "n_monthly_blocks": len(monthly), "mean_monthly_metric": monthly.mean(),
                         "sd_between_monthly_metrics": monthly.std(ddof=1),
                         "scope": "Unweighted descriptive spread of monthly means; blocks are not independent replicates; not the pooled-year metric."})
    return pd.DataFrame(rows)


def probability_fields(log_probability):
    if not np.isfinite(log_probability):
        return {"p_value": np.nan, "log_p_value": np.nan, "p_value_display": "undefined",
                "p_underflow": False}
    log_probability = min(0., float(log_probability))
    p = float(np.exp(log_probability))
    return {"p_value": p, "log_p_value": log_probability,
            "p_value_display": f"{p:.12g}" if p > 0 else f"exp({log_probability:.12g})",
            "p_underflow": p == 0.}


def adjust_holm(rows, family):
    """Use log-p values; retain undefined planned hypotheses in family size."""
    count = len(rows)
    if not count:
        return rows
    values = np.asarray([row.get("log_p_value", np.nan) for row in rows], dtype=float)
    sortable = np.where(np.isfinite(values), np.minimum(values, 0.), 0.)
    order = np.argsort(sortable, kind="stable")
    adjusted = np.empty(count)
    cumulative = -np.inf
    for rank, index in enumerate(order):
        cumulative = max(cumulative, sortable[index]+math.log(count-rank))
        adjusted[index] = min(0., cumulative)
    for i, row in enumerate(rows):
        fields = probability_fields(adjusted[i] if np.isfinite(values[i]) else np.nan)
        row.update({"family": family, "family_size_planned": count,
                    "p_holm": fields["p_value"], "log_p_holm": fields["log_p_value"],
                    "p_holm_display": fields["p_value_display"],
                    "p_holm_underflow": fields["p_underflow"]})
    return rows


def bds_for_residuals(residuals, dimensions=(2, 3), epsilon_multiplier=1.5):
    """Keep every predeclared dimension, including singular/too-short cases."""
    residuals = np.asarray(residuals, dtype=float)
    sample_sd = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else np.nan
    epsilon = epsilon_multiplier*sample_sd
    rows = []
    warning_text = ""
    if (not np.isfinite(residuals).all() or not np.isfinite(epsilon)
            or epsilon <= 0 or len(residuals) <= max(dimensions)+2):
        values = None
    else:
        try:
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter("always")
                statistic, _ = bds(residuals, max_dim=max(dimensions), epsilon=epsilon)
            warning_text = "; ".join(str(item.message) for item in observed)
            values = dict(zip(range(2, max(dimensions)+1), np.atleast_1d(statistic)))
        except (ValueError, ZeroDivisionError, FloatingPointError, IndexError) as exc:
            values, warning_text = None, f"{type(exc).__name__}: {exc}"
    for dimension in dimensions:
        statistic = float(values[dimension]) if values is not None else np.nan
        finite = np.isfinite(statistic)
        log_p = math.log(2.)+float(stats.norm.logsf(abs(statistic))) if finite else np.nan
        rows.append({"dimension": dimension, "n_dates": len(residuals), "epsilon": epsilon,
                     "epsilon_sample_sd_multiplier": epsilon_multiplier, "residual_sample_sd": sample_sd,
                     "statistic": statistic if finite else np.nan,
                     "status": "calculated_approximate_short_block" if finite else "undefined_singular_or_insufficient_sample",
                     "warning": warning_text, "interpretation_limit": BDS_LIMIT,
                     **probability_fields(log_p)})
    return rows


def bds_tables(frame, phase):
    rows = []
    months = sorted(set(calendar_for(phase).strftime("%Y-%m")))
    for (history, model, seed), trajectory in frame.groupby(["history_days", "model", "seed"]):
        for month in months:
            group = trajectory.loc[trajectory.fold_id.eq(month)].sort_values("outcome_day")
            dates = pd.DatetimeIndex(pd.to_datetime(group.outcome_day, utc=True))
            contiguous = len(dates) < 2 or np.all((dates[1:]-dates[:-1]) == pd.Timedelta(days=1))
            residuals = group.y_true.to_numpy()-group.y_pred.to_numpy()
            computed = bds_for_residuals(residuals if contiguous else np.full(len(residuals), np.nan))
            for row in computed:
                if not contiguous:
                    row["status"] = "undefined_noncontiguous_calendar"
                    row["warning"] = "Internal missing calendar days; residuals are not compressed into false adjacent observations."
                rows.append({"phase": phase, "history_days": history, "model": model, "seed": seed,
                             "fold_id": month, "horizon": 1, **row})
    adjust_holm(rows, f"all_predeclared_{phase}_monthly_BDS_dimensions_histories_models_seeds")
    raw = pd.DataFrame(rows)
    summaries = []
    for keys, group in raw.groupby(["history_days", "model", "dimension", "fold_id"]):
        fields = dict(zip(["history_days", "model", "dimension", "fold_id"], keys))
        values = group.p_value.to_numpy(float)
        finite = values[np.isfinite(values)]
        summaries.append({**fields, "n_planned_seeds": len(group), "n_defined": len(finite),
                          "mean_p_value_descriptive": float(np.mean(finite)) if len(finite) else np.nan,
                          "sd_p_value_descriptive": float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan,
                          "minimum_p_value": float(np.min(finite)) if len(finite) else np.nan,
                          "maximum_p_value": float(np.max(finite)) if len(finite) else np.nan,
                          "scope": "Descriptive average across seeds; not a combined test and not a decision rule."})
    seed_means = []
    for keys, group in raw.groupby(["history_days", "model", "seed", "dimension"]):
        fields = dict(zip(["history_days", "model", "seed", "dimension"], keys))
        finite = group.p_value.dropna()
        seed_means.append({**fields, "n_planned_months": len(group), "n_defined": len(finite),
                           "mean_p_value_descriptive": finite.mean(), "sd_p_value_descriptive": finite.std(ddof=1),
                           "scope": "Descriptive mean across monthly blocks; never a combined p-value."})
    return raw, pd.DataFrame(summaries), pd.DataFrame(seed_means)


def bds_descriptive_summary(raw):
    rows = []
    for keys, group in raw.groupby(["history_days", "model", "dimension"]):
        fields = dict(zip(["history_days", "model", "dimension"], keys))
        values = group.p_value.dropna()
        rows.append({**fields, "n_planned": len(group), "n_defined": len(values),
                     "n_monthly_blocks": group.fold_id.nunique(), "n_seeds": group.seed.nunique(),
                     "mean_p_value_descriptive": values.mean(), "sd_p_value_descriptive": values.std(ddof=1),
                     "minimum_p_value": values.min(), "maximum_p_value": values.max(),
                     "n_raw_below_005_descriptive": int((values < .05).sum()),
                     "n_Holm_below_005": int((group.p_holm < .05).sum()),
                     "scope": "Descriptive mean over monthly-block/seed p-values; neither combined evidence nor an independent-replicate proportion.",
                     "interpretation_limit": BDS_LIMIT})
    return pd.DataFrame(rows)


def select_from_cv(summary):
    values = summary.loc[summary.metric.eq("rmse") & summary.model.isin(MODELS)].copy()
    values["parameters"] = [80*int(history)+33 if model == "MLP" else 5025 if model == "LSTM" else 1057
                            for history, model in zip(values.history_days, values.model)]
    def winner(part):
        minimum = float(part["mean"].min())
        tied = part.loc[part["mean"] <= minimum+1e-12]
        return tied.sort_values(["history_days", "parameters", "model"], kind="stable").iloc[0]
    selected = {model: winner(values.loc[values.model.eq(model)])[["history_days", "mean", "sample_sd", "parameters"]].to_dict()
                for model in MODELS}
    best = winner(values)
    return {"criterion": "Minimum mean seed RMSE on the full2024 forward validation; absolute tie tolerance1e-12, then smaller input history, fewer parameters, and lexicographic model name.",
            "tie_absolute_tolerance": 1e-12,
            "selected_history_by_model": selected,
            "point_best_neural_candidate": {"model": best.model, "history_days": int(best.history_days),
                                            "mean_seed_rmse": best["mean"]},
            "scope": "Development selection only, using previously exposed2024; does not establish a significant winner or use reserved test.",
            "all_twelve_settings_reported_on_test": True,
            "test_selection_allowed": False}


def table_text(summary, phase):
    title = "Validación forward2024" if phase == "cv" else "Test reservado2025–agosto2026"
    text = [f"# {title}: cuatro longitudes de entrada", "",
            "Objetivo fijo σ7, horizonte de un día y representación relativa; D cambia la historia de entrada.", "",
            "Media ± DE entre diez semillas; Persistencia es determinista y no tiene diez entrenamientos.", "",
            "| Entrada (días) | Modelo | MAPE (%) | MAE | RMSE | MSE | R² | MASE |",
            "|---:|---|---:|---:|---:|---:|---:|---:|"]
    indexed = summary.set_index(["history_days", "model", "metric"])
    for history in HISTORIES:
        for model in (*MODELS, "Persistencia"):
            cells = []
            for metric in METRICS:
                row = indexed.loc[(history, model, metric)]
                mean, sd = row["mean"], row.sample_sd
                precision = ".5g" if metric in ("mse", "mae", "rmse") else ".5f"
                formatted = format(mean, precision) if np.isfinite(mean) else "No definido"
                if np.isfinite(sd):
                    formatted += " ± " + format(sd, precision)
                cells.append(formatted)
            text.append(f"| {history} | {model} | " + " | ".join(cells) + " |")
    text.extend(["", "Los IC95 marginales y el detalle por semilla y bloque mensual están en los CSV. "
                 "La DE entre semillas no representa la incertidumbre entre periodos independientes.", "",
                 "BDS se calcula por bloque mensual, modelo, D y semilla, con dimensiones2/3 y radio1,5DE muestral. "
                 "Los bloques cortos limitan su aproximación asintótica. El promedio de p-valores es exclusivamente descriptivo.", ""])
    return "\n".join(text)


def plot_scores(summary, phase, destination):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = {"MLP": "#31688e", "LSTM": "#35b779", "CNN1D": "#8856a7"}
    x = np.arange(len(HISTORIES))
    for ax, metric, label in zip(axes, ["rmse", "mape"], ["RMSE (escala original de σ7)", "MAPE (%)"]):
        for ordinal, model in enumerate(MODELS):
            part = summary.loc[summary.metric.eq(metric) & summary.model.eq(model)].set_index("history_days").loc[list(HISTORIES)]
            ax.bar(x+(ordinal-1)*.23, part["mean"], .22, yerr=part.sample_sd,
                   color=colors[model], capsize=3, label=model)
        reference = summary.loc[summary.metric.eq(metric) & summary.model.eq("Persistencia")].set_index("history_days").loc[list(HISTORIES)]
        ax.plot(x, reference["mean"], "o--", color="#222222", label="Persistencia")
        ax.set_xticks(x, [str(h) for h in HISTORIES])
        ax.set_xlabel("Días de entrada (D)")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=8)
    label = "CV de Filipe · desarrollo2024" if phase == "cv" else "Test2025–agosto2026 · modelos congelados"
    fig.suptitle(label + " | objetivo σ7 · h=1 día")
    fig.text(.5, .015, "Barras: media ± DE entre diez semillas; referencia determinista. Datos: Binance Vision/BTCUSDT.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def analyze(phase, output=OUT, root=ROOT, run_bds=True):
    output, root = Path(output), Path(root)
    frame, sources = load_predictions(phase, root, output)
    destination = output / ("analysis_cv" if phase == "cv" else "test/analysis")
    destination.mkdir(parents=True, exist_ok=True)
    annual, monthly = metric_tables(frame)
    summary, monthly_summary = metric_summary(annual), metric_summary(monthly, by_fold=True)
    tables = {"run_metrics.csv": annual, "fold_metrics.csv": monthly,
              "metrics_summary.csv": summary, "fold_metrics_summary.csv": monthly_summary,
              "between_fold_spread_descriptive.csv": descriptive_fold_spread(monthly)}
    bds_summary = None
    if run_bds:
        raw, fold_summary, seed_means = bds_tables(frame, phase)
        tables.update({"bds_by_fold.csv": raw, "bds_by_fold_summary.csv": fold_summary,
                       "bds_mean_across_folds_descriptive.csv": seed_means,
                       "bds_pvalues_descriptive_summary.csv": bds_descriptive_summary(raw)})
        bds_summary = {"planned_hypotheses": len(raw), "defined": int(raw.p_value.notna().sum()),
                       "dimensions": [2, 3], "epsilon_sample_sd_multiplier": 1.5,
                       "family": raw.family.iloc[0], "raw_p_value_averages": "descriptive only, never combined tests",
                       "sample_size_min": int(raw.n_dates.min()), "sample_size_max": int(raw.n_dates.max()),
                       "limitation": BDS_LIMIT}
    for name, table in tables.items():
        table.to_csv(destination / name, index=False, lineterminator="\n")
    (destination / "tables.md").write_text(table_text(summary, phase))
    plot_scores(summary, phase, destination / "metrics_by_input_history.png")
    if phase == "cv":
        write_json(destination / "cv_selection.json", select_from_cv(summary))
    report = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "phase": phase,
              "histories": list(HISTORIES), "models": list(MODELS), "seeds": list(SEEDS),
              "target": "sigma7 population SD of daily log returns, unannualized", "horizon_days": 1,
              "preprocessing": "relative_window", "outcome_days": int(frame.outcome_day.nunique()),
              "calendar_days": len(calendar_for(phase)), "coverage": float(frame.outcome_day.nunique()/len(calendar_for(phase))),
              "neural_forecasts": len(frame), "monthly_reporting_blocks": int(frame.fold_id.nunique()),
              "monthly_blocks_are": "actual forward-validation refits" if phase == "cv" else "reporting periods of frozen final models; no test refitting",
              "bds": bds_summary,
              "source_sha256": {str(path.relative_to(root)): sha(path) for path in sources},
              "analysis_code_sha256": sha(Path(__file__)),
              "output_sha256": {path.name: sha(path) for path in sorted(destination.iterdir()) if path.is_file() and path.name != "summary.json"},
              "new_model_fits": 0,
              "scope": "Descriptive, calendar-conditional multi-history analysis; test never selects settings; no independent-market or guaranteed-power claim."}
    write_json(destination / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["cv", "test"], required=True)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--skip-bds", action="store_true", help="Preview metrics only; does not complete BDS reporting")
    args = parser.parse_args()
    result = analyze(args.phase, args.output, run_bds=not args.skip_bds)
    print(json.dumps({key: result[key] for key in ["phase", "outcome_days", "neural_forecasts", "monthly_reporting_blocks", "bds"]}))


if __name__ == "__main__":
    main()
