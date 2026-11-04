"""Exploratory statistics on stored, current Filipe-CV predictions only.

No optimizer, market archive or reserved-test data is read. The supplementary
plan must be recorded before this script is run on the project predictions.
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
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/filipe-statistics-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/filipe-statistics-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import statsmodels
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, bds

# Pure numerical utilities only: no prior experiment loader, table or figure.
from analyze_dl_benchmark import (
    add_holm, block_indices, bootstrap_scores, chi_square_logsf,
    clean_json, dm_hln, mean_seed_rmse, p_fields, paired_statistics,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/dl_r2_cv_20261104"
OUTPUT = ROOT / "results/filipe_statistics_20261104"
PROTOCOL = ROOT / "experiments/filipe_statistics_20261104/protocol.json"
MODELS = ("MLP", "LSTM", "CNN1D")
METHODS = (*MODELS, "Persistencia")
SEEDS = tuple(range(4101, 4111))
DESIGNATED_SEED = 4101
SCOPE = ("Complemento exploratorio de CV mensual de Filipe sobre desarrollo2024 "
         "ya examinado. Inferencia por semillas condicionada al calendario; "
         "sin confirmación independiente ni evaluación del test reservado.")
CREDIT = "Datos: Binance Vision / BTCUSDT | Procesamiento: Jassan Arteta | CC BY-NC-SA 4.0"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class Cube:
    dates: pd.DatetimeIndex
    target: np.ndarray
    predictions: np.ndarray  # method, training seed, date; persistence broadcast
    run_metrics: pd.DataFrame

    @property
    def squared_errors(self):
        return (self.predictions-self.target[None, None, :])**2

    @property
    def seed_rmse(self):
        return np.sqrt(self.squared_errors.mean(axis=2))


def load_cube(predictions_path, runs_path):
    frame, runs = pd.read_csv(predictions_path), pd.read_csv(runs_path)
    required = {"variant", "model", "seed", "fold_id", "outcome_day",
                "y_true", "y_pred", "persistence_pred"}
    if not required <= set(frame):
        raise ValueError("Missing required CV prediction columns")
    dates = pd.date_range("2024-01-01", "2024-12-31", tz="UTC")
    date_text = dates.strftime("%Y-%m-%d").tolist()
    if (set(frame.variant) != {"absolute_log", "relative_window"}
            or set(frame.model) != set(MODELS) or set(frame.seed) != set(SEEDS)
            or len(frame) != 2*3*10*366
            or frame.duplicated(["variant", "model", "seed", "outcome_day"]).any()):
        raise ValueError("Complete two-variant, three-model, ten-seed CV grid required")
    if not np.isfinite(frame[["y_true", "y_pred", "persistence_pred"]].to_numpy()).all():
        raise ValueError("All stored predictions and labels must be finite")
    predictions = np.empty((4, 10, 366), dtype=float)
    target = reference = None
    for variant in ["absolute_log", "relative_window"]:
        for i, model in enumerate(MODELS):
            for j, seed in enumerate(SEEDS):
                group = frame.loc[(frame.variant == variant) & (frame.model == model)
                                  & (frame.seed == seed)].sort_values("outcome_day")
                if group.outcome_day.tolist() != date_text:
                    raise ValueError("Each CV path must cover all366dates, including29February")
                if not np.array_equal(group.fold_id.to_numpy(), group.outcome_day.str[:7].to_numpy()):
                    raise ValueError("Prediction dates must match twelve monthly fold identifiers")
                if "history_days" in group and not group.history_days.eq(7).all():
                    raise ValueError("Only the common D7 contract is admitted")
                outcome = pd.to_datetime(group.outcome_day, utc=True)
                for field, expected in [("prediction_origin_utc", outcome),
                    ("input_last_day", outcome-pd.Timedelta(days=1)),
                    ("target_available_at_utc", outcome+pd.Timedelta(days=1))]:
                    if field in group and not np.array_equal(pd.to_datetime(group[field], utc=True).to_numpy(), expected.to_numpy()):
                        raise ValueError("Stored forecast information clock is inconsistent")
                y, persistence = group.y_true.to_numpy(), group.persistence_pred.to_numpy()
                if target is None:
                    target, reference = y, persistence
                elif not (np.array_equal(target, y) and np.array_equal(reference, persistence)):
                    raise ValueError("All paths must share identical labels and persistence")
                if variant == "relative_window":
                    predictions[i, j] = group.y_pred.to_numpy()
                    row = runs.loc[(runs.variant == variant) & (runs.model == model) & (runs.seed == seed)]
                    if len(row) != 1:
                        raise ValueError("Exactly one run record required per relative-CV path")
                    error = predictions[i, j]-target
                    calculated = {"mse": np.mean(error**2), "rmse": np.sqrt(np.mean(error**2)),
                                  "mae": np.mean(np.abs(error)), "r2": 1-np.mean(error**2)/np.var(target)}
                    for metric, value in calculated.items():
                        if metric not in row or not np.isclose(row.iloc[0][metric], value, rtol=1e-10, atol=1e-13):
                            raise ValueError("Run metrics do not match stored predictions: "+metric)
    predictions[3] = reference
    if np.any(predictions < 0) or np.any(target <= 0) or np.var(target) <= 0:
        raise ValueError("Expected nonnegative forecasts and positive varying volatility labels")
    return Cube(dates, target, predictions, runs)


def seed_tests(cube, controls):
    scores = cube.seed_rmse
    rows = []
    for a, b in combinations(range(4), 2):
        row = paired_statistics(scores[a], scores[b])
        row.update(comparison=f"{METHODS[a]} − {METHODS[b]}", model_a=METHODS[a], model_b=METHODS[b],
                   estimand="difference in mean training-seed RMSE; negative favors first method", scope=SCOPE)
        rows.append(row)
    baseline = float(scores[3, 0])
    for i, model in enumerate(MODELS):
        gain = 1-scores[i]/baseline if baseline > 0 else np.full(10, np.nan)
        if np.isfinite(gain).all():
            row = paired_statistics(gain, np.zeros(10), null_difference=.05, alternative="greater")
        else:
            row = {"n_pairs": 10, "status": "undefined_zero_reference_RMSE", **p_fields(np.nan)}
        row.update(comparison=f"{model}: mejora relativa >5%", model_a=model, model_b="Persistencia",
                   estimand="mean training-seed relative RMSE gain above0.05; positive favors model", scope=SCOPE)
        rows.append(row)
    add_holm(rows, "nine_supplementary_seed_hypotheses")
    if len(controls) != 3 or set(controls.model) != set(MODELS):
        raise ValueError("All three previously declared CV representation controls are required")
    global_rows = []
    for row in rows:
        global_rows.append({**row, "component_family": "supplementary_nine", "p_holm_component": row["p_holm"]})
    for row in controls.to_dict("records"):
        global_rows.append({**row, "comparison": f"{row['model']}: relativa − absoluta",
            "component_family": "representation_controls_three", "p_holm_component": row["p_holm"], "scope": SCOPE})
    add_holm(global_rows, "global_twelve_training_seed_hypotheses")
    for row, global_row in zip(rows, global_rows):
        row["p_holm9"] = row["p_holm"]
        row["p_holm12"] = global_row["p_holm"]
        row["p_holm12_display"] = global_row["p_holm_display"]
    ranks = stats.rankdata(scores[:3], axis=0)
    with warnings.catch_warnings(record=True):
        q = stats.friedmanchisquare(*scores[:3])
    friedman = {"status": "calculated_approximate" if np.isfinite(q.statistic) else "undefined",
        "n_seed_blocks": 10, "models": list(MODELS), "statistic": float(q.statistic),
        "p_value": float(q.pvalue), "reference": "asymptotic chi-square2; exploratory, not paired-test gate",
        "mean_ranks": dict(zip(MODELS, ranks.mean(axis=1).tolist())),
        "scope": SCOPE, "independent_market_count": 1}
    return rows, global_rows, friedman


def sensitivity_and_superiority(cube, alpha_family=12):
    scores = cube.seed_rmse
    alpha = .05/alpha_family
    quantile_sum = stats.norm.ppf(1-alpha/2)+stats.norm.ppf(.8)
    sensitivity, superiority = [], []
    for a, b in combinations(range(4), 2):
        difference = scores[a]-scores[b]
        sd = 0. if np.ptp(difference) == 0 else float(difference.std(ddof=1))
        delta = .05*float(scores[b, 0]) if b == 3 else np.nan
        defined = sd > 0 and np.isfinite(sd)
        required = math.ceil((quantile_sum*sd/delta)**2) if defined and np.isfinite(delta) and delta > 0 else np.nan
        sensitivity.append({"comparison": f"{METHODS[a]} − {METHODS[b]}", "model_a": METHODS[a], "model_b": METHODS[b],
            "n_seed_pairs": 10, "sd_seed_RMSE_difference": sd, "planning_alpha": alpha,
            "planning_family_size": alpha_family, "hypothetical_power_target": .8,
            "normal_approximation_MDE0": float(quantile_sum*sd/np.sqrt(10)) if defined else np.nan,
            "planning_delta_vs_zero": delta, "normal_approximation_R_for_delta": required,
            "status": "retrospective_dispersion_calibration" if defined else "undefined_zero_dispersion",
            "scope": "Sensibilidad retrospectiva condicionada al calendario, no piloto previo ni certificado de potencia; contraste frente a cero, no superación del5%."})
        wins, ties = int(np.sum(difference < 0)), int(np.sum(difference == 0))
        if ties == 0:
            lower = 0. if wins == 0 else float(stats.beta.ppf(.025, wins, 11-wins))
            upper = 1. if wins == 10 else float(stats.beta.ppf(.975, wins+1, 10-wins))
            ci_status = "Clopper-Pearson95, conditional independent training-seed pairs, no ties"
        else:
            lower = upper = np.nan
            ci_status = "No binomial interval assigned to half-weighted ties"
        superiority.append({"comparison": f"{METHODS[a]} frente a {METHODS[b]}", "model_a": METHODS[a], "model_b": METHODS[b],
            "n_seed_pairs": 10, "strict_wins_a": wins, "ties": ties, "strict_losses_a": 10-wins-ties,
            "probability_superiority_half_ties": (wins+.5*ties)/10,
            "ci95_lower": lower, "ci95_upper": upper, "ci_status": ci_status,
            "scope": "Probabilidad descriptiva entre aleatorizaciones de entrenamiento, condicionada al calendario; no probabilidad universal de ganar en otro mercado."})
    return sensitivity, superiority


def temporal_tests(cube, lags=(14, 7, 28)):
    losses = cube.squared_errors.mean(axis=1)
    rows = []
    for lag in lags:
        batch = []
        for a, b in combinations(range(4), 2):
            row = dm_hln(losses[a]-losses[b], lag=lag, horizon=1)
            row.update(comparison=f"{METHODS[a]} − {METHODS[b]}", model_a=METHODS[a], model_b=METHODS[b],
                       primary=lag == 14, scope=SCOPE)
            batch.append(row)
        if lag == 14:
            add_holm(batch, "six_primary_temporal_squared_loss_comparisons")
        else:
            for row in batch:
                row.update(family="sensitivity_no_primary_reselection", family_size_planned=6,
                           p_holm=np.nan, log_p_holm=np.nan, p_holm_display="not_adjusted_sensitivity")
        rows.extend(batch)
    return rows


def residual_diagnostics(cube, lb_lags=(7, 14, 28), dimensions=(2, 3)):
    rows, acf_rows, descriptive = [], [], []
    seed_index = SEEDS.index(DESIGNATED_SEED)
    for i, model in enumerate(MODELS):
        error = cube.target-cube.predictions[i, seed_index]
        varying = np.ptp(error) > 0
        common = {"model": model, "seed": DESIGNATED_SEED, "n_dates": len(error), "scope": SCOPE}
        if varying:
            correlation = acf(error, nlags=28, fft=False)
            lb = acorr_ljungbox(error, lags=list(lb_lags), model_df=0, return_df=True)
            radius = 1.5*float(error.std(ddof=1))
            with warnings.catch_warnings(record=True):
                bds_stat, _ = bds(error, max_dim=max(dimensions), epsilon=radius)
            bds_stat = np.atleast_1d(bds_stat)
        else:
            correlation, lb, radius = np.full(29, np.nan), None, np.nan
            bds_stat = np.full(max(dimensions)-1, np.nan)
        for lag, value in enumerate(correlation):
            acf_rows.append({**common, "lag": lag, "acf": float(value)})
        for lag in lb_lags:
            q = float(lb.loc[lag, "lb_stat"]) if varying else np.nan
            row = {**common, "test": "Ljung-Box", "lag_or_dimension": lag, "statistic": q,
                "model_df": 0, "reference_degrees_freedom": lag,
                "status": "calculated_approximate" if np.isfinite(q) else "undefined_constant_residual",
                "interpretation": "Autocorrelación conjunta de residuos; referencia aproximada ante etiquetas solapadas y cambios mensuales de red."}
            row.update(p_fields(chi_square_logsf(q, lag) if np.isfinite(q) else np.nan))
            rows.append(row)
        for dim in dimensions:
            z = float(bds_stat[dim-2])
            row = {**common, "test": "BDS", "lag_or_dimension": dim, "statistic": z,
                "epsilon": radius, "epsilon_sd_multiplier": 1.5,
                "status": "calculated_approximate" if np.isfinite(z) else "undefined_BDS_statistic",
                "interpretation": "Contrasta i.i.d.; rechazo no demuestra exclusivamente no linealidad aprovechable."}
            row.update(p_fields(float(np.log(2)+stats.norm.logsf(abs(z))) if np.isfinite(z) else np.nan))
            rows.append(row)
        pred = cube.predictions[i, seed_index]
        descriptive.append({**common, "residual_mean_y_minus_pred": float(error.mean()),
            "residual_sd_ddof1": float(error.std(ddof=1)), "residual_mae": float(np.abs(error).mean()),
            "residual_rmse": float(np.sqrt(np.mean(error**2))), "prediction_sd": float(pred.std(ddof=0)),
            "target_sd": float(cube.target.std(ddof=0)),
            "first_half_residual_mean": float(error[:182].mean()) if len(error) else np.nan,
            "second_half_residual_mean": float(error[182:].mean()) if len(error) > 182 else np.nan})
    add_holm(rows, "fifteen_designated_seed_residual_diagnostics")
    return rows, acf_rows, descriptive


def temporal_bootstrap(cube, lengths=(14, 7, 28), replicates=2000, seed=20261104):
    squared = cube.squared_errors
    observed = mean_seed_rmse(squared)
    intervals, rankings = [], []
    for length in lengths:
        scores = bootstrap_scores(squared, length=length, replicates=replicates, seed=seed)
        for a, b in combinations(range(4), 2):
            difference = scores[:, a]-scores[:, b]
            denominator = scores[:, b]
            valid = denominator > 0
            relative = 1-scores[valid, a]/denominator[valid]
            ci = np.quantile(difference, [.025, .975])
            relative_ci = np.quantile(relative, [.025, .975]) if valid.all() else [np.nan, np.nan]
            intervals.append({"model_a": METHODS[a], "model_b": METHODS[b], "block_length": length,
                "primary": length == 14, "replicates": replicates, "rng_seed": seed,
                "observed_difference": float(observed[a]-observed[b]),
                "ci95_difference_lower": float(ci[0]), "ci95_difference_upper": float(ci[1]),
                "observed_relative_gain": float(1-observed[a]/observed[b]) if observed[b] > 0 else np.nan,
                "ci95_relative_lower": float(relative_ci[0]), "ci95_relative_upper": float(relative_ci[1]),
                "defined_relative_replicates": int(valid.sum()),
                "interval_method": "paired circular date-block percentile, marginal95%",
                "scope": "Incertidumbre temporal condicionada a modelos guardados; sin remuestrear semillas ni reentrenar, sin confirmación externa."})
        for count in [3, 4]:
            winners = scores[:, :count] == scores[:, :count].min(axis=1, keepdims=True)
            frequencies = (winners/winners.sum(axis=1, keepdims=True)).mean(axis=0)
            for i, model in enumerate(METHODS[:count]):
                rankings.append({"comparison_set": "three_DL" if count == 3 else "three_DL_and_persistence",
                    "model": model, "block_length": length, "replicates": replicates,
                    "bootstrap_lowest_error_frequency": float(frequencies[i]),
                    "scope": "Frecuencia de remuestreo condicionada a modelos guardados; no posterior ni probabilidad universal."})
    return intervals, rankings


def save_figure(fig, path):
    fig.text(.5, .012, CREDIT, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_plots(cube, acf_rows, bootstrap, output):
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(acf_rows)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for model, axis in zip(MODELS, axes):
        group = frame.loc[frame.model == model]
        axis.stem(group.lag, group.acf, basefmt="k-")
        axis.set_title(model); axis.set_ylabel("ACF residual")
    axes[-1].set_xlabel("Rezago diario")
    fig.suptitle("CV de Filipe: ACF de residuos, representación relativa, semilla4101")
    save_figure(fig, output / "residual_acf_seed4101.png")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for i, (model, axis) in enumerate(zip(MODELS, axes)):
        axis.hist(cube.target-cube.predictions[i, 0], bins=25, color="#397ba4", alpha=.8)
        axis.axvline(0, color="black", linewidth=1); axis.set_title(model)
        axis.set_xlabel("Residuo: observado − predicho"); axis.set_ylabel("Frecuencia")
    fig.suptitle("Distribución residual de CV, semilla4101; diagnóstico descriptivo")
    save_figure(fig, output / "residual_distribution_seed4101.png")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for (a, b), axis in zip(combinations(range(3), 2), axes):
        stats.probplot(cube.seed_rmse[a]-cube.seed_rmse[b], dist="norm", plot=axis)
        axis.set_title(f"{METHODS[a]} − {METHODS[b]}"); axis.set_xlabel("Cuantiles normales")
        axis.set_ylabel("Diferencia RMSE entre semillas")
    fig.suptitle("Q-Q de diez diferencias pareadas; no prueba independencia de mercados")
    save_figure(fig, output / "paired_seed_qq.png")
    frame = pd.DataFrame(bootstrap)
    group = frame.loc[frame.block_length == 14]
    fig, axis = plt.subplots(figsize=(10, 5))
    for j, row in enumerate(group.itertuples()):
        axis.plot([row.ci95_difference_lower, row.ci95_difference_upper], [j, j], color="#397ba4", linewidth=3)
        axis.scatter([row.observed_difference], [j], color="#bc6a22", zorder=3)
    axis.axvline(0, color="black", linestyle="--")
    axis.set_yticks(range(len(group)), [f"{r.model_a} − {r.model_b}" for r in group.itertuples()])
    axis.set_xlabel("Diferencia de RMSE media; IC temporal marginal95%")
    fig.suptitle("Bootstrap circular pareado de fechas: bloque14, 2.000 réplicas")
    save_figure(fig, output / "temporal_bootstrap_intervals.png")


def plot_epoch_curves(runs, result_dir, output, fold="2024-12"):
    fig, axes = plt.subplots(3, 1, figsize=(11, 9))
    metadata = []
    for model, axis in zip(MODELS, axes):
        row = runs.loc[(runs.variant == "relative_window") & (runs.model == model)
                       & (runs.seed == DESIGNATED_SEED)]
        if len(row) != 1:
            raise ValueError("Representative epoch curve requires one designated trajectory")
        directory = result_dir / "runs" / row.iloc[0].run_id / fold
        history, configuration = directory / "history_inner.csv", directory / "configuration.json"
        if not history.exists() or not configuration.exists():
            plt.close(fig)
            return {"status": "missing_archived_run_files", "instruction": "Restore run_artifacts.tar.gz to plot real epoch curves"}
        frame = pd.read_csv(history)
        config = json.loads(configuration.read_text())
        epoch = int(config["best_epoch"])
        axis.plot(frame.epoch, frame.train_rmse_original, label="Subentrenamiento")
        axis.plot(frame.epoch, frame.validation_rmse_original, label="Validación interna")
        axis.axvline(epoch, color="black", linestyle="--", label=f"Época elegida: {epoch}")
        axis.set_title(model); axis.set_xlabel("Época"); axis.set_ylabel("RMSE original"); axis.legend()
        metadata.append({"model": model, "seed": DESIGNATED_SEED, "fold": fold,
            "epochs_logged": len(frame), "best_epoch": epoch,
            "history_path": str(history), "history_sha256": sha(history),
            "configuration_path": str(configuration), "configuration_sha256": sha(configuration)})
    fig.suptitle("Curvas reales de parada interna: diciembre2024, semilla4101")
    save_figure(fig, output / "internal_epoch_curves_december_seed4101.png")
    return {"status": "plotted", "scope": "Curvas por época; no curvas de aprendizaje frente a tamaño muestral ni curvas de test", "sources": metadata}


def verify_protocol(protocol_path):
    protocol = json.loads(Path(protocol_path).read_text())
    if protocol.get("test_evaluation_allowed") is not False:
        raise ValueError("The supplementary plan must prohibit reserved-test evaluation")
    if "source_sha256" not in protocol:
        raise ValueError("Protocol must register the input digests before calculation")
    required = {str((RESULTS / name).relative_to(ROOT)) for name in
                ["predictions.csv", "runs.csv", "analysis/comparisons.csv"]}
    if not required <= set(protocol["source_sha256"]):
        raise ValueError("The plan must register all three current-CV numerical inputs")
    for name, expected in protocol["source_sha256"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != expected:
            raise ValueError("Registered supplementary input changed: "+name)
    planned = protocol.get("planned", {})
    expected = {"paired_family": 9, "global_family": 12, "dm_primary_lag": 14,
        "dm_sensitivity_lags": [7, 28], "diagnostic_seed": 4101, "diagnostic_fold": "2024-12",
        "lb_lags": [7, 14, 28], "bds_dimensions": [2, 3], "bds_epsilon_sd": 1.5,
        "bootstrap_replicates": 2000, "bootstrap_lengths": [14, 7, 28], "bootstrap_seed": 20261104,
        "forecast_horizon": 1, "lb_model_df": 0, "design_sensitivity_R": 10,
        "design_sensitivity_alpha": .05/12, "design_sensitivity_power_target": .8,
        "design_sensitivity_reference_margin": .05}
    for name, value in expected.items():
        if planned.get(name) != value:
            raise ValueError("Supplementary configuration must match its registered fixed design: "+name)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    protocol = verify_protocol(args.protocol)
    cube = load_cube(RESULTS / "predictions.csv", RESULTS / "runs.csv")
    controls_path = RESULTS / "analysis/comparisons.csv"
    controls = pd.read_csv(controls_path)
    paired, global_rows, friedman = seed_tests(cube, controls)
    sensitivity, superiority = sensitivity_and_superiority(cube)
    temporal = temporal_tests(cube)
    diagnostics, acf_rows, residual = residual_diagnostics(cube)
    bootstrap, ranking = temporal_bootstrap(cube)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    tables = {"paired_tests.csv": paired, "global_seed_tests.csv": global_rows,
        "design_sensitivity.csv": sensitivity, "probability_superiority.csv": superiority,
        "temporal_dm.csv": temporal, "diagnostics.csv": diagnostics, "acf.csv": acf_rows,
        "residual_summary.csv": residual, "block_bootstrap.csv": bootstrap,
        "ranking_bootstrap.csv": ranking}
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(output / name, index=False)
    make_plots(cube, acf_rows, bootstrap, output / "figures")
    curves = plot_epoch_curves(cube.run_metrics, RESULTS, output / "figures")
    helper = Path(__file__).with_name("analyze_dl_benchmark.py")
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "scope": SCOPE,
        "protocol": {"path": str(args.protocol), "sha256": sha(args.protocol), "record": protocol},
        "analysis_script_sha256": sha(__file__), "numeric_helper_source": str(helper),
        "numeric_helper_sha256": sha(helper),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "statsmodels": statsmodels.__version__},
        "primary_representation": "relative_window", "methods": list(METHODS), "seeds": list(SEEDS),
        "n_dates": 366, "date_start": "2024-01-01", "date_end": "2024-12-31",
        "paired_supplement_family_size": 9, "global_training_seed_family_size": 12,
        "prior_representation_controls": 3, "temporal_primary_family_size": 6,
        "residual_family_size": 15, "designated_diagnostic_seed": 4101,
        "bootstrap": {"replicates": 2000, "primary_block_length": 14, "sensitivity_lengths": [7, 28],
            "seed": 20261104, "scope": "paired dates, stored models, no seed resampling or refitting"},
        "friedman": friedman, "epoch_curves": curves,
        "design_sensitivity": "Retrospective dispersion calibration, not a prior pilot or proof of0.8power; no observed-effect power.",
        "sources": [{"path": str(path), "sha256": sha(path)} for path in
                    [RESULTS / "predictions.csv", RESULTS / "runs.csv", controls_path]],
        "new_model_fits": 0, "raw_market_files_opened": 0, "reserved_test_evaluated": False,
        "limitations": ["All2024was already exposed to development and selection; this is exploratory.",
            "Seed intervals condition on one shared calendar; neither ten seeds nor twelve monthly folds create new markets.",
            "Global Holm12 covers declared seed comparisons, not all temporal/diagnostic research as one family.",
            "DM/HAC/HLN, BDS and Ljung-Box references are approximate under regime changes, overlapping targets and monthly model changes.",
            "BDS rejects i.i.d., not exclusively exploitable nonlinearity; residual structure does not quantify future achievable gain.",
            "Date bootstrap assumes an appropriate dependence/local stability representation and conditions on stored model paths.",
            "Marginal95% intervals are not simultaneous merely because p-values were multiplicity adjusted.",
            "No equivalence or advanced-model margin has been defined; no TOST or ROPE is fabricated."]}
    (output / "README.md").write_text("# Complemento estadístico de la CV de Filipe\n\n"+SCOPE+"\n\n"
        "La familia global de doce contrastes por semillas reúne los nueve contrastes del complemento y los tres controles de representación. "
        "Las familias DM6 y diagnósticos15 responden preguntas distintas; no son confirmaciones independientes. Los IC son marginales. "
        "El bootstrap remuestrea fechas emparejadas con modelos guardados, sin reentrenar ni remuestrear semillas. "
        "La sensibilidad del diseño usa dispersión retrospectiva y no certifica potencia. Las curvas graficadas son de épocas internas, no de tamaño muestral ni de test.\n")
    report["outputs"] = {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file() and p.name != "summary.json"}
    (output / "summary.json").write_text(json.dumps(clean_json(report), ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": "completed", "output": str(output), "paired_family": 9,
        "global_family": 12, "DM_rows": len(temporal), "diagnostic_rows": len(diagnostics),
        "bootstrap_rows": len(bootstrap), "model_fits": 0, "reserved_test_evaluated": False}))


if __name__ == "__main__":
    main()
