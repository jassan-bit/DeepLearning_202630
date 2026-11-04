"""Export the Filipe-CV presentation without changing recorded experiments.

The public view contains the two actual monthly-CV representations only. It
reads stored predictions and summaries, never fits models or opens market data.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("MPLCONFIGDIR", "/tmp/filipe-book-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/filipe-book-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/dl_r2_cv_20261104"
OUTPUT = ROOT / "book/_static/dl_r2_cv_20261104/analysis"
MODELS = ("MLP", "LSTM", "CNN1D")
VARIANTS = ("absolute_log", "relative_window")
LABELS = {"absolute_log": "CV: precios absolutos", "relative_window": "CV: precios relativos"}
SCOPE = ("CV mensual exploratoria de desarrollo en 2024. Inferencia entre semillas "
         "condicionada al calendario compartido; sin confirmación en test reservado.")
CREDIT = "Datos: Binance Vision / BTCUSDT | Procesamiento: Jassan Arteta | CC BY-NC-SA 4.0"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_scope(frame):
    frame = frame.copy()
    if "scope" in frame:
        frame["scope"] = SCOPE
    if "cost_scope" in frame:
        frame["cost_scope"] = ("CV con tres trabajadores concurrentes; sin asignar "
                               "tiempo independiente al subconjunto julio-diciembre")
        frame.loc[frame.model.eq("Persistencia"), "cost_scope"] = "Referencia determinista sin ajuste"
    if "mase_scope" in frame:
        frame["mase_scope"] = "Escala de entrenamiento mensual registrada por fecha"
    return frame


def prediction_metrics(y, prediction, scale):
    y, prediction, scale = [np.asarray(v, dtype=float) for v in (y, prediction, scale)]
    if not (np.isfinite(y).all() and np.isfinite(prediction).all() and np.isfinite(scale).all()
            and np.all(scale > 0)):
        raise ValueError("Finite predictions and positive monthly MASE scales required")
    error = prediction-y
    mse, bias = float(np.mean(error**2)), float(np.mean(error))
    target_sd, pred_sd = float(y.std(ddof=0)), float(prediction.std(ddof=0))
    return {
        "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(error))), "mse": mse,
        "r2": 1-mse/float(np.var(y, ddof=0)),
        "mape": float(100*np.mean(np.abs(error/y))), "mape_coverage": 1.0,
        "mase": float(np.mean(np.abs(error)/scale)), "mase_coverage": 1.0,
        "bias": bias, "squared_bias": bias*bias, "error_variance": float(np.var(error, ddof=0)),
        "squared_bias_fraction_of_mse": bias*bias/mse,
        "prediction_mean": float(np.mean(prediction)), "target_mean": float(np.mean(y)),
        "prediction_sd_ddof0": pred_sd, "target_sd_ddof0": target_sd,
        "prediction_target_sd_ratio": pred_sd/target_sd,
        "pearson_correlation": float(np.corrcoef(prediction, y)[0, 1]), "parameters": 0,
    }


def half_year_summary(predictions, runs, annual):
    """Recompute the semestral view from the current CV predictions alone."""
    subset = predictions.loc[predictions.outcome_day.ge("2024-07-01")]
    row_metrics = []
    run_lookup = runs.set_index(["variant", "model", "seed"])
    for key, group in subset.groupby(["variant", "model", "seed"], sort=True):
        if len(group) != 184:
            raise ValueError("Each semestral trajectory must contain184 dates")
        values = prediction_metrics(group.y_true, group.y_pred, group.mase_train_scale)
        values["parameters"] = int(run_lookup.loc[key, "parameters"])
        values["training_seconds"] = values["inference_seconds"] = np.nan
        row_metrics.append({"variant": key[0], "model": key[1], "seed": key[2], **values})
    run_metrics = pd.DataFrame(row_metrics)
    rows = []
    for variant in VARIANTS:
        for model in MODELS:
            group = run_metrics.loc[run_metrics.variant.eq(variant) & run_metrics.model.eq(model)]
            if len(group) != 10:
                raise ValueError("Ten paired training seeds required")
            for metric in annual.metric.unique():
                if metric not in group:
                    continue
                finite = group[metric].to_numpy(dtype=float)
                finite = finite[np.isfinite(finite)]
                count = len(finite)
                mean = float(np.mean(finite)) if count else np.nan
                sd = (0.0 if np.ptp(finite) == 0 else float(np.std(finite, ddof=1))) if count > 1 else np.nan
                lo = hi = np.nan
                status = "undefined_insufficient_or_zero_variance"
                method = "IC t marginal entre semillas, condicionado al calendario"
                if count > 1 and sd > 0:
                    half = stats.t.ppf(.975, count-1)*sd/np.sqrt(count)
                    lo, hi, status = mean-half, mean+half, "calculated_under_assumptions"
                if metric in ["parameters", "mape_coverage", "mase_coverage"]:
                    lo = hi = np.nan
                    method, status = "Sin intervalo para conteo o cobertura", "not_applicable"
                rows.append({"variant": variant, "model": model, "metric": metric,
                    "n_dates": 184, "n_total": 10, "n_defined": count, "mean": mean,
                    "sample_sd": sd, "ci95_lower": lo, "ci95_upper": hi,
                    "ci_method": method, "ci_status": status})
    reference = subset.loc[subset.variant.eq("absolute_log") & subset.model.eq("MLP")
        & subset.seed.eq(4101)].sort_values("outcome_day")
    persistence = prediction_metrics(reference.y_true, reference.persistence_pred, reference.mase_train_scale)
    for metric, value in persistence.items():
        rows.append({"variant": "deterministic_context", "model": "Persistencia", "metric": metric,
            "n_dates": 184, "n_total": 1, "n_defined": int(np.isfinite(value)), "mean": value,
            "sample_sd": np.nan, "ci95_lower": np.nan, "ci95_upper": np.nan,
            "ci_method": "Referencia determinista; sin intervalo entre semillas", "ci_status": "not_applicable"})
    result = pd.DataFrame(rows)
    result["scope"] = SCOPE
    result["cost_scope"] = "CV concurrente; el subconjunto semestral no tiene tiempo asignado"
    result["mase_scope"] = "Escala de entrenamiento mensual registrada por fecha"
    return result, persistence


def bar_plot(frame, metric, target, title):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    x = np.arange(len(MODELS))
    for i, (variant, color) in enumerate(zip(VARIANTS, ["#397ba4", "#bc6a22"])):
        values = frame.loc[frame.variant.eq(variant) & frame.metric.eq(metric)].set_index("model").loc[list(MODELS)]
        ax.bar(x+(i-.5)*.35, values["mean"], width=.35, yerr=values.sample_sd,
               capsize=4, color=color, label=LABELS[variant]+", n=10")
    reference = frame.loc[frame.model.eq("Persistencia") & frame.metric.eq(metric), "mean"].iloc[0]
    ax.axhline(reference, color="#36824b", linestyle="--", label=f"Persistencia: {reference:.4f}")
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_xticks(x, MODELS)
    ax.set_ylabel(("R²" if metric == "r2" else "Sesgo")+", media ± DE entre semillas")
    ax.set_title(title)
    ax.grid(axis="y", alpha=.2)
    ax.legend(fontsize=9)
    fig.text(.5, .018, CREDIT, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, 1))
    fig.savefig(target, dpi=150)
    plt.close(fig)


def plot_predictions(predictions, target):
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for model, ax in zip(MODELS, axes):
        reference = predictions.loc[predictions.variant.eq(VARIANTS[0])
            & predictions.model.eq(model) & predictions.seed.eq(4101)].sort_values("outcome_day")
        days = pd.to_datetime(reference.outcome_day)
        ax.plot(days, reference.y_true, color="black", linewidth=1.4, label="Objetivo observado")
        ax.plot(days, reference.persistence_pred, color="#36824b", linestyle="--", alpha=.7,
                linewidth=1.0, label="Persistencia")
        for variant, color in zip(VARIANTS, ["#397ba4", "#bc6a22"]):
            group = predictions.loc[predictions.variant.eq(variant)
                & predictions.model.eq(model) & predictions.seed.eq(4101)].sort_values("outcome_day")
            ax.plot(pd.to_datetime(group.outcome_day), group.y_pred, color=color,
                    linewidth=1.0, alpha=.9, label=LABELS[variant])
        ax.set_title(model)
        ax.set_ylabel("Volatilidad σ7")
        ax.grid(alpha=.2)
        ax.legend(ncol=4, fontsize=8, loc="upper left")
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    axes[-1].set_xlabel("Día de resultado UTC")
    fig.suptitle("Predicciones de CV mensual: semilla designada 4101, año 2024")
    fig.text(.5, .012, CREDIT, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, .97))
    fig.savefig(target, dpi=150)
    plt.close(fig)


def main():
    inputs = [SOURCE/"predictions.csv", SOURCE/"runs.csv",
              SOURCE/"analysis/metrics_summary.csv", SOURCE/"analysis/comparisons.csv"]
    initial_hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    predictions = pd.read_csv(inputs[0])
    runs = pd.read_csv(inputs[1])
    if (set(predictions.variant) != set(VARIANTS) or set(predictions.model) != set(MODELS)
            or sorted(predictions.seed.unique()) != list(range(4101, 4111))
            or len(predictions) != 21960):
        raise ValueError("The complete recorded CV design is required")
    if predictions.duplicated(["variant", "model", "seed", "outcome_day"]).any():
        raise ValueError("Each CV trajectory must predict each outcome once")
    for _, group in predictions.groupby(["variant", "model", "seed"]):
        if len(group) != 366 or group.outcome_day.min() != "2024-01-01" or group.outcome_day.max() != "2024-12-31":
            raise ValueError("Complete2024 CV trajectories required")
    verified_metrics = []
    run_lookup = runs.set_index(["variant", "model", "seed"])
    for key, group in predictions.groupby(["variant", "model", "seed"], sort=True):
        values = prediction_metrics(group.y_true, group.y_pred, group.mase_train_scale)
        recorded = run_lookup.loc[key]
        for metric in ["rmse", "mae", "mse", "r2", "mape", "mase"]:
            np.testing.assert_allclose(values[metric], recorded[metric], rtol=1e-10, atol=1e-13)
        verified_metrics.append({"variant": key[0], "model": key[1], **values})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figures = OUTPUT/"figures"
    figures.mkdir(exist_ok=True)
    original_summary = pd.read_csv(inputs[2])
    annual = clean_scope(original_summary)
    if not set(annual.variant).issubset({*VARIANTS, "deterministic_context"}):
        raise ValueError("Annual summary includes an unrelated procedure")
    # All annual numerical values are preserved; only presentation scopes change.
    numeric = original_summary.select_dtypes(include="number").columns
    np.testing.assert_allclose(annual[numeric], original_summary[numeric], equal_nan=True, rtol=0, atol=0)
    checked = pd.DataFrame(verified_metrics)
    for (variant, model), group in checked.groupby(["variant", "model"]):
        for metric in ["rmse", "mae", "mse", "r2", "mape", "mase"]:
            recorded = annual.loc[annual.variant.eq(variant) & annual.model.eq(model)
                                  & annual.metric.eq(metric), "mean"].iloc[0]
            np.testing.assert_allclose(group[metric].mean(), recorded, rtol=1e-10, atol=1e-13)
    annual.to_csv(OUTPUT/"metrics_summary.csv", index=False)
    common, persistence = half_year_summary(predictions, runs, annual)
    common.to_csv(OUTPUT/"metrics_common_h2_cv_only.csv", index=False)
    np.testing.assert_allclose(persistence["r2"], .6699952794658537, rtol=1e-12)
    comparisons = pd.read_csv(inputs[3])
    clean_comparisons = clean_scope(comparisons)
    numeric = comparisons.select_dtypes(include="number").columns
    np.testing.assert_allclose(clean_comparisons[numeric], comparisons[numeric], equal_nan=True, rtol=0, atol=0)
    if len(clean_comparisons) != 3 or not clean_comparisons.n_pairs.eq(10).all():
        raise ValueError("Exactly three paired preprocessing comparisons required")
    clean_comparisons.to_csv(OUTPUT/"comparisons.csv", index=False)
    for name in ["fold_metrics.csv", "run_metrics_recomputed.csv", "bias_decomposition.csv"]:
        frame = pd.read_csv(SOURCE/"analysis"/name)
        if "variant" in frame and not set(frame.variant).issubset(set(VARIANTS)):
            raise ValueError("Unrelated procedure in "+name)
        clean_scope(frame).to_csv(OUTPUT/name, index=False)
    bar_plot(annual, "r2", figures/"r2_cv_full2024.png", "CV de Filipe: R² en las 366 fechas de 2024")
    bar_plot(common, "r2", figures/"r2_common_h2_cv_only.png", "CV de Filipe: R² en julio–diciembre de 2024")
    bar_plot(annual, "bias", figures/"bias_cv_full2024.png", "CV de Filipe: sesgo de las predicciones en 2024")
    plot_predictions(predictions, figures/"predictions_designated_seed.png")
    shutil.copyfile(SOURCE/"analysis/figures/cv_folds_actual.png", figures/"cv_folds_actual.png")
    report = {
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "Vista editorial derivada de resultados registrados de CV de Filipe",
        "scope": SCOPE, "variants": list(VARIANTS), "models": list(MODELS),
        "seeds": list(range(4101, 4111)), "monthly_folds": 12, "annual_dates": 366,
        "second_semester_dates": 184, "stored_prediction_rows": len(predictions),
        "annual_numeric_values_preserved": True,
        "controlled_comparison_numeric_values_preserved": True,
        "annual_trajectory_metrics_recomputed_and_verified": len(verified_metrics),
        "persistence_H2_mase": persistence["mase"],
        "mase_policy": "Denominador del entrenamiento mensual de cada fecha; mismo criterio para redes y persistencia",
        "inference": "Tres contrastes RMSE por semillas emparejadas, Holm-3; IC marginales condicionados al calendario",
        "temporal_tests_executed": False, "CV_specific_pilot_completed": False,
        "planning_power_0_8_established": False, "reserved_test_evaluated": False,
        "new_fits_in_export": 0, "raw_market_files_opened_in_export": 0,
        "limitations": ["Configuraciones seleccionadas usando desarrollo ya expuesto; CV exploratoria, sin selección anidada independiente.",
                        "Diez semillas no constituyen un dimensionamiento de potencia ni mercados independientes.",
                        "No se atribuyen a estas predicciones pruebas temporales o de residuos que no se han ejecutado."],
        "sources_sha256": initial_hashes, "exporter_sha256": sha(Path(__file__)),
        "outputs_sha256": {str(p.relative_to(OUTPUT)): sha(p)
            for p in [OUTPUT/name for name in ["metrics_summary.csv", "metrics_common_h2_cv_only.csv",
                       "comparisons.csv", "fold_metrics.csv", "run_metrics_recomputed.csv", "bias_decomposition.csv"]]
            + [figures/name for name in ["r2_cv_full2024.png", "r2_common_h2_cv_only.png", "bias_cv_full2024.png",
                                         "predictions_designated_seed.png", "cv_folds_actual.png"]]},
    }
    (OUTPUT/"summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    (OUTPUT/"README.md").write_text(
        "# Resultados de CV mensual de Filipe\n\n"+SCOPE+"\n\n"
        "Las dos representaciones se ajustan con los mismos doce cortes mensuales y diez semillas. "
        "La tabla anual reúne366fechas; la tabla semestral184fechas. Los R² se calculan agrupando "
        "las predicciones de cada trayectoria, no promediando los R² de meses aislados. "
        "Persistencia es una única trayectoria determinista; MASE usa la escala mensual registrada "
        "en cada fecha. Los tres contrastes/Holm corresponden exclusivamente a la diferencia entre "
        "los dos preprocesamientos dentro de cada red. No se ejecutaron aquí pruebas temporales, "
        "diagnósticos residuales, ajustes nuevos ni evaluación del test.\n\n"
        "Esta carpeta es una vista editorial derivada: no modifica los archivos registrados de entrenamiento.\n")
    allowed = {"metrics_summary.csv", "metrics_common_h2_cv_only.csv", "comparisons.csv",
               "fold_metrics.csv", "run_metrics_recomputed.csv", "bias_decomposition.csv", "summary.json", "README.md",
               *{"figures/"+v for v in ["r2_cv_full2024.png", "r2_common_h2_cv_only.png", "bias_cv_full2024.png",
                                          "predictions_designated_seed.png", "cv_folds_actual.png"]}}
    for path in OUTPUT.rglob("*"):
        if path.is_file() and str(path.relative_to(OUTPUT)) not in allowed:
            path.unlink()
    for path in OUTPUT.rglob("*"):
        if path.is_file() and path.suffix in [".csv", ".json", ".md"]:
            content = path.read_text()
            if "original_static" in content or "V1" in content:
                raise ValueError("Unexpected retired presentation in "+str(path))
    if initial_hashes != {str(p.relative_to(ROOT)): sha(p) for p in inputs}:
        raise ValueError("Recorded source files must not be changed by export")
    print(json.dumps({"export": str(OUTPUT.relative_to(ROOT)), "new_fits": 0,
                      "CV_only": True, "persistence_H2_mase": persistence["mase"]}))


if __name__ == "__main__":
    main()
