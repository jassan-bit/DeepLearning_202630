"""Post-evaluation descriptive supplement from already saved forecasts.

No model, raw archive, optimizer, torch module or inference routine is imported.
Public outputs contain period/bin aggregates, never daily forecast vectors.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/e1-audit-descriptive-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/e1-audit-descriptive-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/e1_audit_improvements_20261104/descriptive"
PROTOCOL = ROOT / "experiments/e1_audit_improvements_20261104/protocol.json"
HISTORIES = (7, 14, 21, 28)
MODELS = ("MLP", "LSTM", "CNN1D")
SEEDS = tuple(range(4101, 4111))
METRICS = ("rmse", "mae", "mse", "mape", "r2", "mase")
MIN_PUBLIC_N = 10
SOURCES = (
    "results/dl_r2_cv_20261104/predictions.csv",
    "results/filipe_multihistory_20261104/predictions.csv",
    "results/filipe_multihistory_20261104/test/predictions.csv",
    "results/eda_development/daily_market_development.csv",
    "results/eda_development/daily_volatility_development.csv",
    "results/filipe_multihistory_20261104/test/daily_market_test.csv",
    "results/filipe_multihistory_20261104/test/daily_volatility_test.csv",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def metrics(group):
    y = group.y_true.to_numpy(float)
    pred = group.y_pred.to_numpy(float)
    if not len(y) or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Finite, nonempty saved forecasts required")
    error = y-pred
    squared = error**2
    positive = y > 0
    scale = group.mase_train_scale.to_numpy(float)
    valid_scale = np.isfinite(scale) & (scale > 0)
    total = float(np.sum((y-y.mean())**2))
    return {"n_dates": len(y), "rmse": float(np.sqrt(squared.mean())),
            "mae": float(np.abs(error).mean()), "mse": float(squared.mean()),
            "r2": 1-float(squared.sum())/total if total > 0 else np.nan,
            "mape": float(100*np.mean(np.abs(error[positive]/y[positive]))) if positive.any() else np.nan,
            "mape_coverage": float(positive.mean()),
            "mase": float(np.mean(np.abs(error[valid_scale])/scale[valid_scale])) if valid_scale.any() else np.nan,
            "mase_coverage": float(valid_scale.mean()), "mean_error": float(error.mean()),
            "error_variance": float(np.var(error, ddof=0)),
            "target_mean": float(y.mean()), "mean_prediction": float(pred.mean())}


def add_reference(frame):
    """One deterministic reference trajectory per D, never ten pseudo-seeds."""
    extra = []
    for history, group in frame.groupby("history_days", sort=True):
        part = group.sort_values(["outcome_day", "model", "seed"]).drop_duplicates("outcome_day").copy()
        part["y_pred"] = part.persistence_pred
        part["model"] = "Persistencia"
        part["seed"] = 0
        part["stochastic_training"] = False
        extra.append(part)
    result = frame.copy()
    result["stochastic_training"] = True
    return pd.concat([result, *extra], ignore_index=True)


def load_forecasts(root=ROOT):
    previous = pd.read_csv(root/SOURCES[0])
    previous = previous.loc[previous.variant.eq("relative_window") & previous.history_days.eq(7)].copy()
    current = pd.read_csv(root/SOURCES[1])
    test = pd.read_csv(root/SOURCES[2])
    cv = pd.concat([previous, current], ignore_index=True)
    outputs = {}
    for phase, frame, first, last in [("cv", cv, "2024-01-01", "2024-12-31"),
                                       ("test", test, "2025-01-01", "2026-08-31")]:
        expected_dates = pd.date_range(first, last).strftime("%Y-%m-%d").tolist()
        if (set(frame.history_days) != set(HISTORIES) or set(frame.model) != set(MODELS)
                or set(frame.seed) != set(SEEDS)
                or frame.duplicated(["history_days", "model", "seed", "outcome_day"]).any()
                or len(frame) != len(expected_dates)*120):
            raise ValueError("Incomplete registered forecast grid: "+phase)
        reference = None
        for _, group in frame.groupby(["history_days", "model", "seed"]):
            group = group.sort_values("outcome_day")
            if group.outcome_day.tolist() != expected_dates:
                raise ValueError("Forecast paths use unequal calendar dates")
            values = group[["y_true", "persistence_pred"]].to_numpy(float)
            if reference is None:
                reference = values
            elif not np.allclose(values, reference, rtol=0, atol=1e-13):
                raise ValueError("Forecast paths have unequal targets/reference")
        frame["phase"] = phase
        outputs[phase] = add_reference(frame)
    return outputs


def annual_tables(frame):
    frame = frame.copy()
    frame["period"] = np.where(frame.outcome_day.str.startswith("2025-"), "2025", "2026_Jan_Aug")
    records = []
    for keys, part in frame.groupby(["history_days", "model", "seed", "period"], sort=True):
        identity = dict(zip(["history_days", "model", "seed", "period"], keys))
        records.append({**identity, "stochastic_training": bool(part.stochastic_training.iloc[0]), **metrics(part)})
    runs = pd.DataFrame(records)
    return runs, summarize_metrics(runs, ["history_days", "model", "period"])


def summarize_metrics(records, keys):
    rows = []
    for values, group in records.groupby(keys, sort=True, dropna=False):
        if not isinstance(values, tuple): values = (values,)
        identity = dict(zip(keys, values))
        stochastic = bool(group.stochastic_training.iloc[0])
        for metric in METRICS:
            finite = group[metric].to_numpy(float)
            finite = finite[np.isfinite(finite)]
            count = len(finite)
            mean = float(finite.mean()) if count else np.nan
            sd = float(finite.std(ddof=1)) if stochastic and count > 1 else np.nan
            half = float(stats.t.ppf(.975, count-1)*sd/np.sqrt(count)) if np.isfinite(sd) else np.nan
            n = group.n_dates.to_numpy(int)
            rows.append({**identity, "metric": metric, "mean": mean, "sample_sd": sd,
                         "ci95_lower": mean-half, "ci95_upper": mean+half,
                         "n_defined": count, "n_seed_records": len(group),
                         "n_dates": int(n[0]) if len(set(n)) == 1 else np.nan,
                         "n_dates_min": int(n.min()), "n_dates_max": int(n.max()),
                         "stochastic_training": stochastic,
                         "ci_scope": "Marginal t95 across training seeds conditional on fixed dates, not market uncertainty"
                         if stochastic else "Not applicable: one deterministic reference path"})
    return pd.DataFrame(rows)


def monthly_tables(frame):
    rows = []
    for keys, part in frame.groupby(["phase", "history_days", "model", "seed", "fold_id"], sort=True):
        identity = dict(zip(["phase", "history_days", "model", "seed", "month"], keys))
        rows.append({**identity, "stochastic_training": bool(part.stochastic_training.iloc[0]), **metrics(part)})
    return pd.DataFrame(rows)


def variance_components(frame, factors, value="rmse"):
    """Orthogonal descriptive main effects on a complete balanced cell panel.

    Population variance over observed cells = sum(main-effect mean squares) +
    residual interaction variance. No random-effect, independence or causal claim.
    """
    if frame.duplicated(factors).any():
        raise ValueError("Exactly one metric per factorial cell is required")
    levels = [frame[factor].nunique() for factor in factors]
    if not len(frame) or math.prod(levels) != len(frame):
        raise ValueError("A complete balanced panel is required")
    values = frame[value].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Finite cell metrics required")
    grand = float(values.mean())
    total = float(np.mean((values-grand)**2))
    residual = values-grand
    pieces = []
    for factor in factors:
        effect = frame.groupby(factor)[value].transform("mean").to_numpy()-grand
        amount = float(np.mean(effect**2))
        residual -= effect
        pieces.append((factor, amount))
    pieces.append(("interactions_residual", float(np.mean(residual**2))))
    if not np.isclose(sum(amount for _, amount in pieces), total, rtol=1e-10, atol=1e-18):
        raise AssertionError("Balanced descriptive variance identity failed")
    return pd.DataFrame([{"factor": factor, "total_variance": total, "component_variance": amount,
                          "share": amount/total if total > 0 else np.nan,
                          "n_cells": len(frame), "status": "defined" if total > 0 else "constant_panel",
                          "interpretation": "Descriptive population variance of monthly RMSE cells; no inferential ANOVA or iid-fold assumption"}
                         for factor, amount in pieces])


def variance_tables(monthly):
    pieces = []
    neural = monthly.loc[monthly.model.isin(MODELS)]
    for (phase, history), panel in neural.groupby(["phase", "history_days"], sort=True):
        result = variance_components(panel, ["model", "seed", "month"])
        pieces.append(result.assign(phase=phase, history_days=history, panel="three_neural_models", model="all_three"))
        for model, part in panel.groupby("model", sort=True):
            result = variance_components(part, ["seed", "month"])
            pieces.append(result.assign(phase=phase, history_days=history, panel="within_model", model=model))
    return pd.concat(pieces, ignore_index=True)


def training_thresholds(root=ROOT):
    targets = pd.read_csv(root/SOURCES[4])
    market = pd.read_csv(root/SOURCES[3])
    days = pd.to_datetime(targets.market_day_utc, utc=True)
    if not np.array_equal(days, pd.to_datetime(market.market_day_utc, utc=True)):
        raise ValueError("Development labels and OHLCV use different dates")
    # Final D28 training outcomes end on Dec03; use that common admitted cohort
    # for every method's descriptive target/prediction bins.
    finite = np.isfinite(market[["open", "high", "low", "close", "volume"]]).all(axis=1)
    admitted_inputs = finite.shift(1, fill_value=False).rolling(28, min_periods=28).sum().eq(28)
    selected = (days.le(pd.Timestamp("2024-12-03", tz="UTC")) & targets.sigma_7.notna()
                & admitted_inputs)
    values = targets.loc[selected, "sigma_7"].to_numpy(float)
    boundaries = np.quantile(values, [.25, .5, .75])
    if not np.all(np.diff(boundaries) > 0):
        raise ValueError("Distinct development quantiles required")
    return {"common_training_last_outcome_day": "2024-12-03", "target": "sigma7",
            "quantile_method": "numpy linear", "training_nonmissing_targets": len(values),
            "q25": float(boundaries[0]), "q50": float(boundaries[1]), "q75": float(boundaries[2]),
            "input_eligibility": "All28previousOHLCVdays finite; same common admitted D28 final-training outcomes, no gap bridging",
            "scope": "Boundaries use pre-test development only; segmentation is retrospective diagnosis, not a new selection rule."}


def severity_label(values, bounds):
    return np.where(np.asarray(values) < bounds[0], "low", np.where(np.asarray(values) < bounds[1], "middle", "high"))


def severity_tables(frames, threshold):
    rows = []
    edges = [threshold["q25"], threshold["q75"]]
    for phase, frame in frames.items():
        frame = frame.copy()
        frame["severity"] = severity_label(frame.y_true, edges)
        for keys, part in frame.groupby(["history_days", "model", "seed", "severity"], sort=True):
            identity = dict(zip(["history_days", "model", "seed", "severity"], keys))
            record = {"phase": phase, **identity, "stochastic_training": bool(part.stochastic_training.iloc[0]), **metrics(part)}
            record["status"] = "defined" if len(part) >= MIN_PUBLIC_N else "suppressed_fewer_than_10_dates"
            if len(part) < MIN_PUBLIC_N:
                for name in (*METRICS, "mean_error", "error_variance", "target_mean", "mean_prediction"):
                    record[name] = np.nan
            rows.append(record)
    runs = pd.DataFrame(rows)
    return runs, summarize_metrics(runs, ["phase", "history_days", "model", "severity"])


def residual_bins(frame, threshold):
    rows = []
    cuts = [threshold["q25"], threshold["q50"], threshold["q75"]]
    for keys, part in frame.groupby(["history_days", "model", "seed"], sort=True):
        part = part.copy()
        part["predicted_bin"] = np.searchsorted(cuts, part.y_pred.to_numpy(float), side="right")+1
        for predicted_bin in range(1, 5):
            group = part.loc[part.predicted_bin.eq(predicted_bin)]
            identity = dict(zip(["history_days", "model", "seed"], keys))
            record = {**identity, "predicted_bin": predicted_bin, "n_dates": len(group),
                      "stochastic_training": bool(part.stochastic_training.iloc[0]),
                      "status": "defined" if len(group) >= MIN_PUBLIC_N else "suppressed_fewer_than_10_dates"}
            if len(group) >= MIN_PUBLIC_N:
                record.update(metrics(group))
            else:
                record.update({name: np.nan for name in ("rmse", "mae", "mean_error", "error_variance", "target_mean", "mean_prediction")})
            rows.append(record)
    individual = pd.DataFrame(rows)
    summaries = []
    for keys, group in individual.groupby(["history_days", "model", "predicted_bin"], sort=True):
        valid = group.loc[group.status.eq("defined")]
        identity = dict(zip(["history_days", "model", "predicted_bin"], keys))
        rec = {**identity, "n_seed_records": len(group), "n_defined_seed_records": len(valid),
               "n_dates_min": int(group.n_dates.min()), "n_dates_max": int(group.n_dates.max()),
               "scope": "Binned error moments averaged across seeds, conditional on fixed dates; no hypothesis test of heteroscedasticity"}
        for metric in ["mean_error", "error_variance", "rmse", "mae", "mean_prediction"]:
            values = valid[metric].to_numpy(float)
            rec[metric] = float(values.mean()) if len(values) else np.nan
            rec[metric+"_sample_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        summaries.append(rec)
    return individual, pd.DataFrame(summaries)


def public_monthly(monthly):
    rows = []
    for keys, group in monthly.groupby(["phase", "history_days", "model", "month"], sort=True):
        identity = dict(zip(["phase", "history_days", "model", "month"], keys))
        row = {**identity, "n_dates": int(group.n_dates.iloc[0]), "n_seed_records": len(group),
               "stochastic_training": bool(group.stochastic_training.iloc[0]),
               "target_mean": float(group.target_mean.iloc[0]),
               "prediction_mean": float(group.mean_prediction.mean()),
               "prediction_mean_sample_sd_across_seeds": float(group.mean_prediction.std(ddof=1)) if len(group) > 1 else np.nan,
               "rmse_mean": float(group.rmse.mean()), "mae_mean": float(group.mae.mean()),
               "bias_mean": float(group.mean_error.mean()), "error_variance_mean": float(group.error_variance.mean()),
               "scope": "Monthly aggregates; every point represents at least 28 dates, not an individual forecast"}
        rows.append(row)
    aggregates = pd.DataFrame(rows)
    test = aggregates.loc[aggregates.phase.eq("test")].copy()
    test["difficulty_rank"] = test.groupby(["history_days", "model"]).rmse_mean.rank(method="first", ascending=False).astype(int)
    difficult = test.loc[test.difficulty_rank.le(3)].sort_values(["history_days", "model", "difficulty_rank"])
    return aggregates, difficult


def distribution_tables(root=ROOT):
    development = pd.read_csv(root/SOURCES[3])
    evaluation = pd.read_csv(root/SOURCES[5])
    market = pd.concat([development, evaluation], ignore_index=True)
    market["market_day_utc"] = pd.to_datetime(market.market_day_utc, utc=True)
    market = market.set_index("market_day_utc").sort_index()
    target_dev = pd.read_csv(root/SOURCES[4])
    target_test = pd.read_csv(root/SOURCES[6])
    targets = pd.concat([target_dev.loc[:, ["market_day_utc", "sigma_7"]],
                         target_test.loc[:, ["market_day_utc", "sigma_7"]]], ignore_index=True)
    targets["market_day_utc"] = pd.to_datetime(targets.market_day_utc, utc=True)
    market["sigma7"] = targets.set_index("market_day_utc").sigma_7
    market["daily_log_return"] = np.log(market.close).diff()
    market["log1p_volume"] = np.log1p(market.volume)
    market["period"] = np.where(market.index.year == 2026, "2026_Jan_Aug", market.index.year.astype(str))
    features = ["open", "high", "low", "close", "volume", "daily_log_return", "log1p_volume", "sigma7"]
    fit = market.loc[:"2024-12-03", features]
    rows = []
    for period, group in market.groupby("period", sort=True):
        for feature in features:
            values = group[feature].dropna().to_numpy(float)
            historical = fit[feature].dropna().to_numpy(float)
            train_sd = float(historical.std(ddof=0))
            quantiles = np.quantile(values, [.1, .25, .5, .75, .9])
            row = {"period": period, "feature": feature, "n_calendar_days": len(group), "n_observed": len(values),
                   "coverage": float(len(values)/len(group)), "mean": float(values.mean()),
                   "sd_ddof0": float(values.std(ddof=0)),
                   "q10": float(quantiles[0]), "q25": float(quantiles[1]), "median": float(quantiles[2]),
                   "q75": float(quantiles[3]), "q90": float(quantiles[4]),
                   "standardized_mean_shift_vs_training": float((values.mean()-historical.mean())/train_sd) if train_sd > 0 else np.nan,
                   "wasserstein_over_training_sd": float(stats.wasserstein_distance(values, historical)/train_sd) if train_sd > 0 else np.nan,
                   "reference": "Observed development2023 through2024-Dec03; descriptive only, not a new p-value or tuning rule"}
            rows.append(row)
    return pd.DataFrame(rows)


def figures(destination, annual, variance, monthly, binned):
    colors = {"MLP": "#31688e", "LSTM": "#35b779", "CNN1D": "#8856a7"}
    selected = annual.loc[annual.history_days.eq(7) & annual.metric.eq("rmse")]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(2)
    for index, model in enumerate(MODELS):
        part = selected.loc[selected.model.eq(model)].set_index("period").loc[["2025", "2026_Jan_Aug"]]
        ax.bar(x+(index-1)*.2, part["mean"], .2, yerr=part.sample_sd, capsize=3, label=model, color=colors[model])
    ref = selected.loc[selected.model.eq("Persistencia")].set_index("period").loc[["2025", "2026_Jan_Aug"]]
    ax.plot(x, ref["mean"], "o--", color="#222", label="Persistencia")
    ax.set_xticks(x, ["2025 · 365 fechas", "Ene–ago 2026 · 243 fechas"])
    ax.set_ylabel("RMSE de σ7, escala original")
    ax.set_title("Evaluación final por año · D=7 · h=1 día")
    ax.legend(ncol=4, fontsize=8); ax.grid(axis="y", alpha=.2);ax.set_axisbelow(True)
    fig.text(.5, .01, "Media ± DE entre semillas; periodos diferentes, sin interpretación causal o como mercados independientes.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(destination/"annual_rmse.png", dpi=170);plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8), sharex=True)
    months = sorted(monthly.loc[monthly.phase.eq("test"), "month"].unique())
    for ax, model in zip(axes, MODELS):
        part = monthly.loc[monthly.phase.eq("test") & monthly.history_days.eq(7) & monthly.model.eq(model)].set_index("month").loc[months]
        values = part.prediction_mean.to_numpy(float);sd = part.prediction_mean_sample_sd_across_seeds.to_numpy(float)
        ax.plot(range(len(months)), part.target_mean, "o-", color="#222", label="Media mensual observada")
        ax.plot(range(len(months)), values, "o-", color=colors[model], label=model+": media de medias mensuales")
        ax.fill_between(range(len(months)), values-sd, values+sd, color=colors[model], alpha=.2, label="±DE entre semillas")
        ax.set_ylabel("σ7 media");ax.grid(alpha=.2);ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xticks(range(len(months)), months, rotation=50, ha="right")
    fig.suptitle("Valores observados y pronósticos: agregados mensuales · D=7")
    fig.text(.5,.005,"Cada punto agrupa 28–31 días. Las bandas no son IC de mercado ni representan predicciones individuales.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.02,1,.96));fig.savefig(destination/"monthly_predictions_D7.png",dpi=170);plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11,4.4))
    for model in MODELS:
        part = binned.loc[binned.history_days.eq(7)&binned.model.eq(model)]
        axes[0].plot(part.predicted_bin, part.error_variance, "o-", label=model, color=colors[model])
        axes[1].plot(part.predicted_bin, part.mean_error, "o-", label=model, color=colors[model])
    axes[0].set_ylabel("Varianza poblacional del error (media entre semillas)")
    axes[1].set_ylabel("Error medio: observado − pronosticado")
    for ax in axes:
        ax.set_xlabel("Intervalo del pronóstico, límites fijados en desarrollo");ax.set_xticks([1,2,3,4]);ax.grid(alpha=.2)
    axes[1].axhline(0,color="#222",linewidth=.8);axes[0].legend(fontsize=8)
    fig.suptitle("Diagnóstico descriptivo de dispersión y sesgo · Test · D=7")
    fig.text(.5,.01,"Celdas con menos de diez fechas por semilla suprimidas; no se efectuó prueba de heterocedasticidad.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(destination/"residual_variance_D7.png",dpi=170);plt.close(fig)

    fig, ax = plt.subplots(figsize=(9,4.4));bottom=np.zeros(4)
    palette = ["#31688e", "#35b779", "#8856a7", "#aaa"]
    labels = {"model":"Arquitectura", "seed":"Semilla", "month":"Mes", "interactions_residual":"Interacciones residuales"}
    panel = variance.loc[variance.phase.eq("test") & variance.panel.eq("three_neural_models")]
    for factor,color in zip(["model","seed","month","interactions_residual"],palette):
        part=panel.loc[panel.factor.eq(factor)].set_index("history_days").loc[list(HISTORIES)]
        height=100*part.share.to_numpy(float);ax.bar(range(4),height,bottom=bottom,label=labels[factor],color=color);bottom+=height
    ax.set_xticks(range(4),list(HISTORIES));ax.set_xlabel("Días de entrada D");ax.set_ylabel("Fracción de varianza observada del RMSE mensual (%)")
    ax.set_title("Descomposición descriptiva del panel arquitectura × semilla × mes")
    ax.legend(fontsize=8,ncol=2);ax.set_ylim(0,105)
    fig.text(.5,.01,"Factores sobre celdas balanceadas; no es ANOVA inferencial ni atribución causal.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(destination/"variance_components.png",dpi=170);plt.close(fig)


def run(root=ROOT, output=OUT, protocol=PROTOCOL):
    root, output, protocol = Path(root), Path(output), Path(protocol)
    registered = json.loads(protocol.read_text())
    for name, expected in registered["source_sha256"].items():
        if digest(root/name) != expected:
            raise ValueError("Registered source changed: "+name)
    if (registered.get("new_project_training_allowed") is not False
            or registered.get("new_checkpoint_forward_calls_allowed") is not False
            or registered.get("raw_market_archive_reads_allowed") is not False):
        raise ValueError("This supplement must remain descriptive, with no fitting or inference")
    output.mkdir(parents=True,exist_ok=True)
    manifest_path = output/"code_manifest_before_analysis.json"
    manifest = {"registered_before_descriptive_calculations_at_utc":datetime.now(timezone.utc).isoformat(),
                "protocol_sha256":digest(protocol), "analysis_code_sha256":digest(Path(__file__)),
                "source_sha256":{name:digest(root/name) for name in SOURCES},
                "new_model_fits":0,"new_checkpoint_forward_calls":0,"raw_archives_opened":0}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in ["protocol_sha256","analysis_code_sha256","source_sha256"]:
            if previous[key] != manifest[key]:
                raise ValueError("Preserve the completed descriptive analysis after code/source changes")
    else:
        write_json(manifest_path,manifest)
    frames = load_forecasts(root)
    annual_runs, annual_summary = annual_tables(frames["test"])
    monthly = pd.concat([monthly_tables(frame) for frame in frames.values()],ignore_index=True)
    variance = variance_tables(monthly)
    threshold = training_thresholds(root)
    segment_runs, segment_summary = severity_tables(frames, threshold)
    binned_runs, binned_summary = residual_bins(frames["test"],threshold)
    aggregates, difficult = public_monthly(monthly)
    distribution = distribution_tables(root)
    tables={"annual_run_metrics.csv":annual_runs,"annual_metrics_summary.csv":annual_summary,
            "variance_components.csv":variance,"severity_run_metrics.csv":segment_runs,
            "severity_metrics_summary.csv":segment_summary,"residual_bins_by_seed.csv":binned_runs,
            "residual_bins_summary.csv":binned_summary,"monthly_prediction_aggregates.csv":aggregates,
            "difficult_months.csv":difficult,"distribution_by_period.csv":distribution}
    for name,table in tables.items():table.to_csv(output/name,index=False,lineterminator="\n")
    write_json(output/"development_bin_thresholds.json",threshold)
    figures(output,annual_summary,variance,aggregates,binned_summary)
    (output/"README.md").write_text("""# Complemento descriptivo posterior a la evaluación

Se reutilizan los pronósticos guardados. No hay nuevos entrenamientos, pasadas de modelos ni lecturas de ZIP raw. Este complemento se prepara después de observar el test: no es una confirmación prospectiva ni autoriza cambiar configuraciones.

`annual_*` separa 2025 (365 fechas) de enero–agosto de 2026 (243); primero calcula las métricas por semilla y periodo, y después media/DE/IC marginal t95 entre las diez semillas. Persistencia aparece una vez por D: no tiene DE ni IC por semillas. R² se calcula con el promedio del objetivo del periodo correspondiente; no se promedian R² mensuales.

`variance_components.csv` aplica la identidad descriptiva de varianza poblacional a un panel balanceado de RMSE mensuales. Se descomponen efectos principales por arquitectura, semilla y mes; el resto son interacciones. Las fracciones no constituyen ANOVA inferencial, efectos causales o mercados independientes. Se añade el panel semilla×mes dentro de cada red. La varianza de RMSE entre celdas no es la MSE de los pronósticos.

Los límites de `severity_*` y `residual_bins_*` se calculan solo con σ7 de desarrollo hasta el 03/12/2024, último resultado admitido del entrenamiento final con D28. Son límites comunes, anteriores al test; la elegibilidad exige los 28 días OHLCV previos completos, sin cerrar huecos; los segmentos no alteran el pipeline. Bajo/medio/alto usa q25/q75 del objetivo; los cuatro intervalos del pronóstico usan q25/q50/q75. Las celdas con menos de diez fechas suprimen sus métricas; se conserva su conteo. Los momentos residuales identifican patrones de dispersión/sesgo, sin declarar una prueba formal de heterocedasticidad o una mejora futura alcanzable.

`monthly_prediction_aggregates.csv` y las figuras muestran únicamente medias mensuales y dispersión entre semillas. Cada punto resume 28–31 fechas. `difficult_months.csv` selecciona los tres meses con mayor RMSE media por D/modelo; es un relato retrospectivo de errores, no una política de selección. No se exportan vectores diarios, precios diarios, residuos individuales ni pesos.

`distribution_by_period.csv` describe objetivo y covariables de 2023,2024,2025 y enero–agosto2026. Distancia de Wasserstein/DE de entrenamiento y desplazamiento de media estandarizada son índices descriptivos, no nuevos p-valores ni evidencia causal. Los niveles de precios de periodos distintos pueden cambiar estructuralmente; observarlos tras cerrar test no permite reajustar el modelo y volver a llamar independiente a la misma evaluación.
""")
    summary={"completed_at_utc":datetime.now(timezone.utc).isoformat(),"analysis":"post_evaluation_descriptive_audit_improvements",
            "source_sha256":{**registered["source_sha256"],**manifest["source_sha256"]},"protocol_sha256":digest(protocol),"analysis_code_sha256":digest(Path(__file__)),
             "new_model_fits":0,"new_checkpoint_forward_calls":0,"raw_archives_opened":0,
             "period_days":{"2025":365,"2026_Jan_Aug":243},"training_bin_thresholds":threshold,
             "counts":{name:len(table) for name,table in tables.items()},
             "scope":"Descriptive supplement after test inspection; fixed model selection and forecasts remain unchanged. Only aggregates/bins/period summaries are outputs.",
             "output_sha256":{p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file() and p.name!="summary.json"}}
    write_json(output/"summary.json",summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("validate","run"),default="validate")
    parser.add_argument("--output",type=Path,default=OUT)
    parser.add_argument("--protocol",type=Path,default=PROTOCOL)
    args=parser.parse_args()
    if args.phase=="validate":
        print(json.dumps({"analysis_code_sha256":digest(Path(__file__)),"source_sha256":{name:digest(ROOT/name) for name in SOURCES},"new_model_fits":0,"new_checkpoint_forward_calls":0,"raw_archives_opened":0}))
        return
    result=run(output=args.output,protocol=args.protocol)
    print(json.dumps({"status":"completed","counts":result["counts"],"new_model_fits":0,"new_checkpoint_forward_calls":0}))


if __name__=="__main__":main()
