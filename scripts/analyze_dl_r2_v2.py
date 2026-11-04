"""Compare stored exploratory development variants with the unchanged V1.

Only prediction/run CSVs are read. No fitting, raw market access or reserved
test evaluation occurs. A variant must provide exactly one forecast per
model/seed/date: overlapping CV forecasts require a declared aggregation rule
upstream, never silent deduplication here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-e1-r2-v2")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/cache-e1-r2-v2")
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analyze_dl_benchmark import add_holm, clean_json, paired_statistics


MODELS = ("MLP", "LSTM", "CNN1D")
ORIGINAL = "original_static"
PERSISTENCE = "Persistencia"
SCOPE = ("Exploratory development comparison designed after observing V1. "
         "Training-seed inference is conditional on the shared historical calendar. "
         "No reserved-test confirmation or guaranteed planning power is established.")
CREDIT = "Datos: Binance Vision / BTCUSDT | Procesamiento: Jassan Arteta | CC BY-NC-SA 4.0"


def prediction_metrics(y_true, y_pred, mase_scale=None) -> dict:
    """Original-unit metrics and bias decomposition on one common cohort."""
    y, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Expected nonempty paired one-dimensional predictions")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Targets and predictions must be finite")
    error = p-y
    mse = float(np.mean(error**2))
    target_var = 0.0 if np.ptp(y) == 0 else float(np.mean((y-y.mean())**2))
    bias = float(error.mean())
    positive = y > 0
    target_sd = 0.0 if np.ptp(y) == 0 else float(y.std(ddof=0))
    pred_sd = 0.0 if np.ptp(p) == 0 else float(p.std(ddof=0))
    result = {
        "rmse": np.sqrt(mse), "mae": float(np.abs(error).mean()), "mse": mse,
        "r2": 1-mse/target_var if target_var > 0 else np.nan,
        "mape": float(100*np.abs(error[positive]/y[positive]).mean()) if positive.any() else np.nan,
        "mape_coverage": float(positive.mean()), "bias": bias,
        "squared_bias": bias*bias, "error_variance": 0.0 if np.ptp(error) == 0 else float(error.var(ddof=0)),
        "squared_bias_fraction_of_mse": bias*bias/mse if mse > 0 else np.nan,
        "prediction_mean": float(p.mean()), "target_mean": float(y.mean()),
        "prediction_sd_ddof0": pred_sd, "target_sd_ddof0": target_sd,
        "prediction_target_sd_ratio": pred_sd/target_sd if target_sd > 0 else np.nan,
        "pearson_correlation": float(np.corrcoef(p, y)[0, 1]) if min(pred_sd, target_sd) > 0 else np.nan,
        "mase": np.nan, "mase_coverage": 0.0,
    }
    if mase_scale is not None:
        scale = np.broadcast_to(np.asarray(mase_scale, dtype=float), y.shape)
        usable = np.isfinite(scale) & (scale > 0)
        result["mase"] = float(np.mean(np.abs(error[usable])/scale[usable])) if usable.any() else np.nan
        result["mase_coverage"] = float(usable.mean())
    return result


def mean_interval(values) -> dict:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    n = len(finite)
    mean = float(finite.mean()) if n else np.nan
    sd = (0.0 if np.ptp(finite) == 0 else float(finite.std(ddof=1))) if n > 1 else np.nan
    row = {"n_total": len(values), "n_defined": n, "mean": mean, "sample_sd": sd,
           "ci95_lower": np.nan, "ci95_upper": np.nan,
           "ci_method": "marginal Student t across seeds conditional on fixed dates",
           "ci_status": "undefined_insufficient_or_zero_variance"}
    if n > 1 and sd > 0:
        half = float(stats.t.ppf(.975, n-1)*sd/np.sqrt(n))
        row.update(ci95_lower=mean-half, ci95_upper=mean+half,
                   ci_status="calculated_under_assumptions")
    return row


def validate_prediction_frame(frame, *, variants, seeds, dates, require_monthly_folds=False) -> pd.DataFrame:
    required = {"variant", "model", "seed", "history_days", "outcome_day", "y_true", "y_pred", "persistence_pred"}
    if not required.issubset(frame):
        raise ValueError(f"Missing prediction columns: {sorted(required-set(frame.columns))}")
    frame = frame.copy()
    parsed = pd.DatetimeIndex(pd.to_datetime(frame.outcome_day, utc=True))
    if not (parsed.year == 2024).all():
        raise ValueError("Reserved test or non2024 evaluation dates are forbidden")
    frame["outcome_day"] = parsed.strftime("%Y-%m-%d")
    if require_monthly_folds:
        if "fold_id" not in frame or not np.array_equal(frame.fold_id.astype(str).to_numpy(), parsed.strftime("%Y-%m").to_numpy()):
            raise ValueError("Every forecast must retain its matching actual monthly CV fold_id")
    if set(frame.variant) != set(variants) or set(frame.model) != set(MODELS):
        raise ValueError("Variants and models must match the complete declared design")
    groups = set(frame[["variant", "model"]].itertuples(index=False, name=None))
    if groups != {(variant, model) for variant in variants for model in MODELS}:
        raise ValueError("Missing complete variant/model group from the declared design")
    if frame.duplicated(["variant", "model", "seed", "outcome_day"]).any():
        raise ValueError("Duplicate/overlapping CV forecasts require an upstream declared policy")
    if not frame.history_days.eq(7).all():
        raise ValueError("This comparison requires the shared D7 input contract")
    if "input_last_day" in frame and not (pd.to_datetime(frame.input_last_day, utc=True) < parsed).all():
        raise ValueError("Covariate windows must end before their result day")
    if {"prediction_origin_utc", "target_available_at_utc"}.issubset(frame):
        if not (pd.to_datetime(frame.prediction_origin_utc, utc=True) < pd.to_datetime(frame.target_available_at_utc, utc=True)).all():
            raise ValueError("Targets must not be available at the declared forecast origin")
    numbers = frame[["y_true", "y_pred", "persistence_pred"]].to_numpy(dtype=float)
    if not np.isfinite(numbers).all() or (numbers < 0).any():
        raise ValueError("Forecasts and targets must be finite and nonnegative")
    reference = None
    expected_dates = list(dates)
    for (variant, model), group in frame.groupby(["variant", "model"]):
        if set(group.seed) != set(seeds):
            raise ValueError(f"Incomplete declared seeds for {variant}/{model}")
        for seed, run in group.groupby("seed"):
            run = run.sort_values("outcome_day")
            if run.outcome_day.tolist() != expected_dates:
                raise ValueError(f"Incomplete common dates for {variant}/{model}/{seed}")
            values = run[["y_true", "persistence_pred"]].to_numpy(dtype=float)
            if reference is None:
                reference = values
            elif not np.allclose(values, reference, rtol=1e-12, atol=1e-14):
                raise ValueError("Targets and persistence must be identical on all paired dates")
    return frame.sort_values(["variant", "model", "seed", "outcome_day"]).reset_index(drop=True)


def recompute_runs(predictions, runs, *, validate_stored=True) -> pd.DataFrame:
    keys = ["variant", "model", "seed"]
    if runs.duplicated(keys).any():
        raise ValueError("Duplicate run summaries")
    expected = set(map(tuple, predictions[keys].drop_duplicates().to_numpy()))
    actual = set(map(tuple, runs[keys].to_numpy()))
    if expected != actual:
        raise ValueError("Run summaries do not match all paired prediction groups")
    lookup = runs.set_index(keys)
    results = []
    for key, group in predictions.groupby(keys, sort=True):
        stored = lookup.loc[key]
        group = group.sort_values("outcome_day")
        scale = group.mase_train_scale.to_numpy() if "mase_train_scale" in group else stored.get("mase_train_scale", None)
        values = prediction_metrics(group.y_true, group.y_pred, scale)
        for metric in ["rmse", "mae", "mse", "r2", "mape"]:
            if validate_stored and (metric not in stored or not np.isclose(values[metric], stored[metric], rtol=1e-8, atol=1e-12, equal_nan=True)):
                raise ValueError(f"Stored {metric} disagrees with predictions: {key}")
        if validate_stored and scale is not None and "mase" in stored and not np.isclose(values["mase"], stored.mase, rtol=1e-8, atol=1e-12, equal_nan=True):
            raise ValueError(f"Stored MASE disagrees with calendar-specific scales: {key}")
        if scale is None:
            values["mase"] = stored.get("mase", np.nan)
            values["mase_coverage"] = stored.get("mase_coverage", np.nan)
        results.append({**dict(zip(keys, key)), "n_dates": len(group), **values,
                        **{name: stored.get(name, np.nan) if validate_stored or name == "parameters" else np.nan for name in ["parameters", "training_seconds", "inference_seconds"]}})
    return pd.DataFrame(results)


def compare_variants(run_metrics, variants) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        subset = run_metrics.loc[run_metrics.model.eq(model)]
        for old, new in combinations(variants, 2):
            a = subset.loc[subset.variant.eq(old)].set_index("seed").sort_index()
            b = subset.loc[subset.variant.eq(new)].set_index("seed").sort_index()
            if not a.index.equals(b.index):
                raise ValueError("Comparisons require exactly matched seed identities")
            row = paired_statistics(b.rmse.to_numpy(), a.rmse.to_numpy())
            r2 = mean_interval(b.r2.to_numpy()-a.r2.to_numpy())
            row.update(model=model, old_variant=old, new_variant=new,
                       estimand="mean paired-seed RMSE difference new minus old; negative improves",
                       scope=SCOPE, r2_mean_difference=r2["mean"],
                       r2_ci95_lower=r2["ci95_lower"], r2_ci95_upper=r2["ci95_upper"],
                       r2_interval_scope="secondary descriptive marginal interval; no separate R2 significance test")
            rows.append(row)
    return pd.DataFrame(add_holm(rows, "all_declared_within_model_variant_pairs_mean_seed_RMSE"))


def summarize_metrics(run_metrics, persistence_metrics, variants) -> pd.DataFrame:
    metrics = [column for column in run_metrics if column not in ["variant", "model", "seed", "n_dates"]]
    rows = []
    for variant in variants:
        for model in MODELS:
            group = run_metrics.loc[run_metrics.variant.eq(variant) & run_metrics.model.eq(model)]
            for metric in metrics:
                values = mean_interval(group[metric].to_numpy())
                if metric in ["parameters", "mape_coverage", "mase_coverage"]:
                    values.update(ci95_lower=np.nan, ci95_upper=np.nan, ci_method="no interval for implementation count or cohort coverage", ci_status="not_applicable")
                rows.append({"variant": variant, "model": model, "metric": metric,
                             "n_dates": int(group.n_dates.iloc[0]), **values, "scope": SCOPE,
                             "cost_scope": "CV variants concurrent; V1 serial. Common-H2 subset has no allocated timing; cross-version speed is not controlled"})
    for metric, value in persistence_metrics.items():
        rows.append({"variant": "deterministic_context", "model": PERSISTENCE, "metric": metric,
                     "n_dates": int(run_metrics.n_dates.iloc[0]), "n_total": 1, "n_defined": int(np.isfinite(value)),
                     "mean": value, "sample_sd": np.nan, "ci95_lower": np.nan, "ci95_upper": np.nan,
                     "ci_method": "one deterministic forecast; no training-seed interval", "ci_status": "not_applicable",
                     "scope": SCOPE, "cost_scope": "not a learned model",
                     "mase_scope": "V1 fixed2023 denominator only; monthly CV scales define a separate MASE policy"})
    return pd.DataFrame(rows)


def _bar_plot(summary, variants, metric, target, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(MODELS))
    width = .8/len(variants)
    colors = ["#777777", "#2a7f9e", "#bc6a22", "#5f8d4e"]
    labels = {ORIGINAL: "V1: estático2023", "absolute_log": "CV: log precios absolutos",
              "relative_window": "CV: precios relativos por ventana"}
    for i, variant in enumerate(variants):
        values = summary.loc[summary.variant.eq(variant) & summary.metric.eq(metric)].set_index("model").loc[list(MODELS)]
        count = int(values.n_defined.iloc[0])
        ax.bar(x+(i-(len(variants)-1)/2)*width, values["mean"], width, yerr=values.sample_sd,
               capsize=4, color=colors[i % len(colors)], label=f"{labels.get(variant, variant)}, n={count}")
    context = summary.loc[summary.model.eq(PERSISTENCE) & summary.metric.eq(metric), "mean"]
    if len(context):
        value = context.iloc[0]
        ax.axhline(value, ls="--", color="#38824e", label=f"Persistencia: {value:.4f}")
    ax.axhline(0, color="black", lw=.7)
    ax.set(xticks=x, xticklabels=MODELS,
           ylabel=("R²" if metric == "r2" else "Predicción − objetivo")+", media ± DE entre semillas",
           title=title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.2)
    fig.text(.5, .01, CREDIT, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(target, dpi=150)
    plt.close(fig)


def plot_results(predictions, old_predictions, summary, common_summary, variants, out, designated_seed):
    figures = out/"figures"
    figures.mkdir(exist_ok=True)
    _bar_plot(summary, variants, "r2", figures/"r2_cv_full2024.png",
              "CV mensual real: las mismas 366 fechas de desarrollo de 2024")
    _bar_plot(common_summary, [ORIGINAL, *variants], "r2", figures/"r2_common_h2_context.png",
              "Julio–diciembre2024: V1 como contexto, con distintas semillas y procedimiento")
    _bar_plot(summary, variants, "bias", figures/"bias_cv_full2024.png",
              "Sesgo de nivel de los pronósticos de CV mensual de 2024")
    labels = {"absolute_log": "CV: log precios absolutos", "relative_window": "CV: precios relativos"}
    colors = ["#2a7f9e", "#bc6a22", "#5f8d4e"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, model in zip(axes, MODELS):
        sample = predictions.loc[predictions.model.eq(model) & predictions.seed.eq(designated_seed)]
        reference = sample.loc[sample.variant.eq(variants[0])].sort_values("outcome_day")
        dates = pd.to_datetime(reference.outcome_day)
        ax.plot(dates, reference.y_true, color="black", lw=1.3, label="Objetivo")
        ax.plot(dates, reference.persistence_pred, color="#38824e", ls="--", lw=1, label="Persistencia")
        for i, variant in enumerate(variants):
            run = sample.loc[sample.variant.eq(variant)].sort_values("outcome_day")
            ax.plot(pd.to_datetime(run.outcome_day), run.y_pred, color=colors[i % len(colors)], lw=1,
                    label=labels.get(variant, variant))
        previous = old_predictions.loc[old_predictions.model.eq(model) & old_predictions.seed.eq(1001)].sort_values("outcome_day")
        ax.plot(pd.to_datetime(previous.outcome_day), previous.y_pred, color="#888888", lw=.8,
                alpha=.7, label="V1 semilla1001: contexto distinto")
        ax.set(title=model, ylabel="σ7 no anualizada")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8, ncol=3)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle(f"Predicciones reales CV, semilla designada{designated_seed}; V1 no es un par de esta semilla")
    fig.text(.5, .01, CREDIT, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .96))
    fig.savefig(figures/"predictions_designated_seed.png", dpi=150)
    plt.close(fig)


def per_fold_metrics(predictions) -> pd.DataFrame:
    rows = []
    for key, group in predictions.groupby(["variant", "model", "seed", "fold_id"], sort=True):
        scale = group.mase_train_scale.to_numpy() if "mase_train_scale" in group else None
        rows.append({**dict(zip(["variant", "model", "seed", "fold_id"], key)),
                     "n_dates": len(group), **prediction_metrics(group.y_true, group.y_pred, scale)})
    return pd.DataFrame(rows)


def analyze(results_dir, v1_results_dir, *, variants, seeds, start="2024-01-01", end="2024-12-31"):
    results_dir, v1_results_dir = Path(results_dir), Path(v1_results_dir)
    if ORIGINAL in variants or len(set(variants)) != len(variants) or len(variants) < 2:
        raise ValueError("Declare the complete distinct CV variant names without original_static")
    seeds = sorted(set(seeds))
    if len(seeds) < 2:
        raise ValueError("At least two declared paired CV seeds are required")
    dates = pd.date_range(start, end, tz="UTC").strftime("%Y-%m-%d").tolist()
    if dates != pd.date_range("2024-01-01", "2024-12-31", tz="UTC").strftime("%Y-%m-%d").tolist():
        raise ValueError("This registered monthly CV analysis requires all366 dates of2024")
    sources = [results_dir/"predictions.csv", results_dir/"runs.csv",
               v1_results_dir/"predictions/development_benchmark.csv", v1_results_dir/"runs.csv"]
    new, new_runs, old, old_runs = [pd.read_csv(path) for path in sources]
    predictions = validate_prediction_frame(new, variants=variants, seeds=seeds, dates=dates,
                                             require_monthly_folds=True)
    if set(predictions.fold_id) != set(pd.period_range("2024-01", "2024-12", freq="M").astype(str)):
        raise ValueError("Exactly12 actual monthly CV fold identifiers are required")
    if "mase_train_scale" not in predictions or not np.isfinite(predictions.mase_train_scale).all() or not predictions.mase_train_scale.gt(0).all():
        raise ValueError("Calendar-specific positive train-only MASE scales must be recorded")
    run_metrics = recompute_runs(predictions, new_runs)
    old = old.loc[old.phase.eq("development_benchmark")].copy()
    old_runs = old_runs.loc[old_runs.phase.eq("development_benchmark")].copy()
    old["variant"] = old_runs["variant"] = ORIGINAL
    old_dates = pd.date_range("2024-07-01", "2024-12-31", tz="UTC").strftime("%Y-%m-%d").tolist()
    old = validate_prediction_frame(old, variants=[ORIGINAL], seeds=list(range(1001, 1031)), dates=old_dates)
    old_metrics = recompute_runs(old, old_runs)
    common = predictions.loc[predictions.outcome_day.ge("2024-07-01")].copy()
    common_metrics = recompute_runs(common, new_runs, validate_stored=False)
    # All comparisons below pair the two CV procedures by the same new seeds.
    # V1 seed1001..1030 versus CV4101..4110 is descriptive context only.
    comparisons = compare_variants(run_metrics, variants)
    reference = predictions.loc[predictions.variant.eq(variants[0]) & predictions.model.eq("MLP") & predictions.seed.eq(seeds[0])].sort_values("outcome_day")
    reference_old = old.loc[old.model.eq("MLP") & old.seed.eq(1001)].sort_values("outcome_day")
    paired_labels = reference.loc[reference.outcome_day.ge("2024-07-01"), ["y_true", "persistence_pred"]].to_numpy()
    if not np.allclose(paired_labels, reference_old[["y_true", "persistence_pred"]].to_numpy(), rtol=1e-12, atol=1e-14):
        raise ValueError("V1 descriptive context must use identical H2 labels and persistence")
    # Deterministic persistence has one path, and its normalization is policy-specific.
    persistence = prediction_metrics(reference.y_true, reference.persistence_pred, reference.mase_train_scale)
    persistence["parameters"] = 0
    fixed_scale = old_runs.loc[old_runs.model.eq("MLP") & old_runs.seed.eq(1001), "mase_train_scale"].iloc[0]
    persistence_common = prediction_metrics(reference_old.y_true, reference_old.persistence_pred, fixed_scale)
    persistence_common["parameters"] = 0
    summary = summarize_metrics(run_metrics, persistence, variants)
    summary.loc[summary.model.eq(PERSISTENCE), "mase_scope"] = "CV monthly training scales of the first declared variant; policy-specific values stored separately"
    combined_common_metrics = pd.concat([old_metrics, common_metrics], ignore_index=True)
    common_summary = summarize_metrics(combined_common_metrics, persistence_common, [ORIGINAL, *variants])
    folds = per_fold_metrics(predictions)
    out = results_dir/"analysis"
    out.mkdir(parents=True, exist_ok=True)
    run_metrics.to_csv(out/"run_metrics_recomputed.csv", index=False)
    summary.to_csv(out/"metrics_summary.csv", index=False)
    common_summary.to_csv(out/"metrics_common_h2.csv", index=False)
    comparisons.to_csv(out/"comparisons.csv", index=False)
    folds.to_csv(out/"fold_metrics.csv", index=False)
    decomposition_columns = ["variant", "model", "seed", "n_dates", "mse", "bias", "squared_bias", "error_variance", "squared_bias_fraction_of_mse", "prediction_mean", "target_mean", "prediction_sd_ddof0", "target_sd_ddof0", "prediction_target_sd_ratio", "pearson_correlation"]
    run_metrics[decomposition_columns].to_csv(out/"bias_decomposition.csv", index=False)
    combined_common_metrics[decomposition_columns].to_csv(out/"bias_decomposition_common_h2.csv", index=False)
    scale_rows = []
    for variant in variants:
        sample = predictions.loc[predictions.variant.eq(variant) & predictions.model.eq("MLP") & predictions.seed.eq(seeds[0])].sort_values("outcome_day")
        values = prediction_metrics(sample.y_true, sample.persistence_pred, sample.mase_train_scale)
        scale_rows.append({"cohort": "full2024", "scale_policy_variant": variant, "forecast": PERSISTENCE,
                           "independent_forecast_repetitions": 1, **values})
    scale_rows.append({"cohort": "common_H2", "scale_policy_variant": ORIGINAL,
                       "forecast": PERSISTENCE, "independent_forecast_repetitions": 1, **persistence_common})
    pd.DataFrame(scale_rows).to_csv(out/"persistence_by_scale_policy.csv", index=False)
    plot_results(predictions, old, summary, common_summary, variants, out, seeds[0])
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "scope": SCOPE,
        "CV_variants": list(variants), "descriptive_context_variant": ORIGINAL,
        "models": list(MODELS), "CV_seeds": seeds, "CV_n_seeds": len(seeds),
        "V1_context_seeds": list(range(1001, 1031)), "V1_context_n_seeds": 30,
        "CV_first_day": dates[0], "CV_last_day": dates[-1], "CV_n_dates": len(dates),
        "common_H2_first_day": old_dates[0], "common_H2_last_day": old_dates[-1], "common_H2_n_dates": len(old_dates),
        "CV_monthly_folds": 12, "CV_prediction_rows": len(predictions), "CV_run_rows": len(run_metrics),
        "CV_fold_metric_rows": len(folds), "paired_RMSE_family_size": len(comparisons),
        "multiplicity": "Holm over all declared within-model CV-variant pairs; V1 excluded",
        "V1_comparison": "descriptive only on the common184dates; seeds and fitting policy differ; no paired V1 p-values",
        "R2": "Recalculated as1-MSE/population variance on the shared cohort; undefined for zero-variance or singleton fold",
        "R2_inference": "secondary descriptive paired intervals only; no separately selected R2 significance tests",
        "interval_scope": "marginal Student t across training seeds on fixed dates; not temporal confidence intervals",
        "persistence": "one deterministic forecast; MASE is specific to the training-scale policy",
        "timing": "V1 serial versus CV concurrent workers; cross-version speed comparisons are not controlled",
        "reserved_test_evaluated": False, "raw_market_files_opened": 0, "new_model_fits_in_analysis": 0,
        "V1_overwritten": False, "operational_reference_reselected": False, "planning_power_0_8_established": False,
        "sources": [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sources],
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inherited_statistical_helpers_sha256": hashlib.sha256((Path(__file__).parent/"analyze_dl_benchmark.py").read_bytes()).hexdigest(),
        "limitations": ["CV changes were motivated by observed V1 failure and exposed development data; no independent confirmation.",
                        "Ten seeds are a resource-bounded diagnostic, not a pilot dimensioning result or evidence of0.8power.",
                        "Training seeds do not create independent market trajectories; fold/date dependence remains.",
                        "Fold identifiers verify prediction membership; actual model fitting is evidenced separately by runner artifacts.",
                        "Bias decomposition is descriptive and does not isolate a unique causal explanation."]}
    (out/"summary.json").write_text(json.dumps(clean_json(report), ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    (out/"README.md").write_text("# Comparación exploratoria de R² con CV mensual\n\n"+SCOPE+"\n\nLas métricas anuales usan las mismas366fechas de2024 para los dos procedimientos CV. El resumen H2 se limita a184fechas comunes y mantiene V1 como contexto con30semillas distintas frentea10deCV; no hay contrastes pareados contraV1. Los tres contrastes primarios RMSE comparan los dos procedimientos CV con las mismas semillas y ajusteHolm3. Los IC95% son marginales, condicionados al calendario; no representan incertidumbre temporal. Las métricas por fold conservan29/30/31fechas segúnel mes y R² indefinido si nohay varianza o solohayun caso. Persistencia es un único pronóstico determinista; MASE depende de la escala histórica registrada por fecha. No se entrena ni se consulta test; V1 no se modifica. Los tiempos seriales V1 no son comparables con la ejecución concurrente CV.\n")
    print(json.dumps({"analysis": str(out), "CV_variants": list(variants), "CV_seeds": len(seeds), "CV_dates": len(dates), "paired_tests": len(comparisons), "reserved_test_evaluated": False}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--results-dir", type=Path, default=root/"results/dl_r2_cv_20261104")
    parser.add_argument("--v1-results-dir", type=Path, default=root/"results/dl_benchmark_20261104")
    parser.add_argument("--variants", nargs="+", default=["absolute_log", "relative_window"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(4101, 4111)))
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args()
    analyze(args.results_dir, args.v1_results_dir, variants=args.variants, seeds=args.seeds, start=args.start, end=args.end)


if __name__ == "__main__":
    main()
