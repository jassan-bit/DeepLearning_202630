"""Plan and analyze a new registered, conditional-seed D7 comparison.

The 20 new precision-pilot seeds alone determine R. The old five seeds are only
descriptive sensitivity; all25 remain excluded from definitive inference.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from scipy import stats

from analyze_filipe_pilot import (
    PlanningLimitError, digest, minimum_detectable_difference,
    minimum_repetitions, normal_approximation_R, normal_difference_power,
    upper_difference_sd, write_json,
)
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
STUDY = "e1_filipe_confirmatory_d7_20261105"
PLAN = ROOT / "experiments" / STUDY / "protocol.json"
OUT = ROOT / "results" / STUDY
OLD = ROOT / "results/e1_filipe_pilot_20261105/runs.csv"


def stamp():
    return datetime.now(timezone.utc).isoformat()


def validate_plan(plan):
    models = list(plan["models"])
    pairs = plan["primary_pairs"]
    expected = {(7, *sorted(pair)) for pair in itertools.combinations(models, 2)}
    observed = [(p["history_days"], *sorted([p["model_a"], p["model_b"]])) for p in pairs]
    d = plan["dimensioning"]
    if (plan["study_id"] != STUDY or plan["histories"] != [7]
            or set(models) != {"MLP", "LSTM", "CNN1D"} or len(pairs) != 3
            or len(set(observed)) != 3 or set(observed) != expected
            or plan["variance_extension_seeds"] != list(range(6301, 6341, 2))
            or plan["original_pilot_seeds"] != list(range(6201, 6206))
            or d.get("n_pilot", d.get("variance_extension_n")) != 20
            or d["primary_family_size"] != 3
            or not np.isclose(d["alpha_each"], d["alpha_global"]/3)
            or d["execution_floor"] != 30 or d["power_target"] != .8
            or d["sd_upper_confidence"] != .95
            or d.get("sd_estimation", d.get("sd_source")) not in
               {"new20_only; pooled25 sensitivity, never choosesR", "20 new extension seeds only"}
            or plan.get("definitive_seed_step") != 2
            or plan["analysis"]["bootstrap_replicates"] != 4999
            or plan["analysis"]["bootstrap_seed"] != 20261105
            or not 0 < d["delta_absolute"] < np.inf):
        raise ValueError("Wrong registered D7 scope, precision seeds, family or variance rule")


def validate_frame(frame, seeds, models, *, allow_other_histories=False):
    required = {"history_days", "model", "seed", "rmse", "n_dates"}
    if not required.issubset(frame.columns):
        raise ValueError("Missing aggregate trajectory fields")
    if allow_other_histories:
        frame = frame.loc[frame.history_days.eq(7)].copy()
    keys = list(frame[["model", "seed"]].itertuples(index=False, name=None))
    expected = set(itertools.product(models, seeds))
    if (len(keys) != len(expected) or len(set(keys)) != len(keys) or set(keys) != expected
            or not frame.history_days.eq(7).all() or not frame.n_dates.eq(366).all()
            or not np.isfinite(frame.rmse).all() or (frame.rmse < 0).any()):
        raise ValueError("Wrong, duplicate or incomplete registered seed trajectories")
    if "variant" in frame and not frame.variant.eq("relative_window").all():
        raise ValueError("Wrong registered representation")
    return frame


def difference(frame, pair, seeds):
    cell = frame.pivot(index="seed", columns="model", values="rmse")
    return (cell.loc[seeds, pair["model_a"]]-cell.loc[seeds, pair["model_b"]]).to_numpy()


def _planned_R(delta, sd, settings):
    if not np.isfinite(sd) or sd <= 0:
        return None
    try:
        return minimum_repetitions(delta, sd, settings["alpha_each"], settings["power_target"],
                                   settings["maximum_planning_R"], settings["execution_floor"])
    except PlanningLimitError:
        return None


def dimension_from_metrics(oldpilotframe, extensionframe, plan):
    """Pure calculation: no files, fits or use of observed means for power."""
    validate_plan(plan)
    models, d = list(plan["models"]), plan["dimensioning"]
    old_seeds, new_seeds = plan["original_pilot_seeds"], plan["variance_extension_seeds"]
    old = validate_frame(oldpilotframe, old_seeds, models, allow_other_histories=True)
    new = validate_frame(extensionframe, new_seeds, models)
    pooled = pd.concat([old, new], ignore_index=True)
    rows = []
    for pair in plan["primary_pairs"]:
        before, fresh, combined = [difference(frame, pair, seeds) for frame, seeds in
            [(old, old_seeds), (new, new_seeds), (pooled, old_seeds+new_seeds)]]
        sd_old, sd_new, sd_pool = [float(values.std(ddof=1)) for values in [before, fresh, combined]]
        upper_new = upper_difference_sd(sd_new, 20, d["sd_upper_confidence"]) if sd_new > 0 else None
        upper_pool = upper_difference_sd(sd_pool, 25, d["sd_upper_confidence"]) if sd_pool > 0 else None
        point_r = _planned_R(d["delta_absolute"], sd_new, d)
        upper_r = _planned_R(d["delta_absolute"], upper_new, d) if upper_new is not None else None
        pooled_r = _planned_R(d["delta_absolute"], upper_pool, d) if upper_pool is not None else None
        rows.append({
            **pair, "n_primary_variance_seeds": 20, "n_old_seeds_sensitivity": 5, "n_pooled_sensitivity": 25,
            "SD_original5_descriptive": sd_old, "SD_new20_primary": sd_new,
            "SD_pooled25_sensitivity": sd_pool, "SD_upper95_new20_primary": upper_new,
            "SD_upper95_pooled25_sensitivity": upper_pool, "delta_absolute": d["delta_absolute"],
            "standardized_planning_delta_primary": d["delta_absolute"]/sd_new if sd_new > 0 else None,
            "planning_alpha_each": d["alpha_each"], "marginal_power_target": d["power_target"],
            "R_point_new20": point_r, "R_upper_new20_primary": upper_r,
            "R_normal_approximation_new20": normal_approximation_R(d["delta_absolute"], sd_new, d["alpha_each"], d["power_target"], d["execution_floor"]) if sd_new > 0 else None,
            "R_upper_pooled25_sensitivity_not_used": pooled_r,
            "status": "conditional_normal_planning" if upper_r is not None else "zero_SD_or_planning_limit_no_sufficient_R",
        })
    common_r = max(r["R_upper_new20_primary"] for r in rows) if all(r["R_upper_new20_primary"] is not None for r in rows) else None
    point_r = max(r["R_point_new20"] for r in rows) if all(r["R_point_new20"] is not None for r in rows) else None
    for row in rows:
        sd = row["SD_new20_primary"]
        upper = row["SD_upper95_new20_primary"]
        row["common_R"] = common_r
        row["MDE_at_primary_pilot20_point_SD"] = minimum_detectable_difference(20, sd, d["alpha_each"], d["power_target"]) if sd > 0 else None
        row["MDE_at_common_R_upper_SD"] = minimum_detectable_difference(common_r, upper, d["alpha_each"], d["power_target"]) if common_r is not None and upper is not None else None
        row["planning_power_at_common_R_upper_SD"] = normal_difference_power(common_r, d["delta_absolute"], upper, d["alpha_each"]) if common_r is not None and upper is not None else None
    budget = int(plan.get("maximum_definitive_seeds", 300))
    status = ("blocked_unidentifiable_dimensioning" if common_r is None else
              "blocked_budget_and_plan_complete" if common_r > budget else "design_locked_ready_for_new_seed_runs")
    summary = {
        "stage_status": status, "common_R": common_r, "common_point_R_new20": point_r,
        "primary_variance_seed_count": 20, "pooled_sensitivity_seed_count": 25,
        "old_pilot_in_primary_variance": False, "pooled_variance_chooses_R": False,
        "execution_floor": d["execution_floor"], "maximum_planning_R": d["maximum_planning_R"],
        "workflow_budget_seed_limit": budget, "budget_limit_caps_required_R": False,
        "delta_absolute": d["delta_absolute"], "alpha_global": d["alpha_global"],
        "alpha_each": d["alpha_each"], "marginal_power_target": d["power_target"],
        "SD_bound_is_marginal_normal_reference": True, "joint_or_achieved_power_claim": False,
        "all25_pilot_seeds_excluded_from_definitive": True,
        "definitive_seed_first_last": [plan["definitive_seed_start"], plan["definitive_seed_start"]+plan["definitive_seed_step"]*(common_r-1)] if common_r is not None else None,
        "definitive_seed_step": plan["definitive_seed_step"],
        "definitive_trajectories": 3*common_r if common_r is not None else None,
        "definitive_fits": 72*common_r if common_r is not None else None,
        "future_advanced_boundary_test": False, "new_independent_market_evaluation": False,
        "scope": "RNG confirmation conditional on known2024 calendar; no repair of the old benchmark",
    }
    return pd.DataFrame(rows), summary


def verify_dimension_lock(oldpilotframe, extensionframe, plan, lock):
    """Pure, read-only recomputation for the runner before definitive launch."""
    _, summary = dimension_from_metrics(oldpilotframe, extensionframe, plan)
    for key, expected in summary.items():
        if lock.get(key) != expected:
            raise ValueError("Design lock differs from registered dimensioning: "+key)
    return summary


def holm_adjust(pvalues):
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or len(values) != 3 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Exactly three finite valid primary p-values are required")
    order = np.argsort(values)
    result = np.empty(3)
    result[order] = np.minimum(1., np.maximum.accumulate(values[order]*np.arange(3, 0, -1)))
    return result


def _bootstrap_pairs(scores, indices, B, seed):
    rng = np.random.default_rng(seed)
    means, effects = np.empty((3, B)), np.full((3, B), np.nan)
    for start in range(0, B, 128):
        count = min(128, B-start)
        draws = scores[rng.integers(0, len(scores), size=(count, len(scores)))]
        for j, (a, b) in enumerate(indices):
            values = draws[:, :, a]-draws[:, :, b]
            mean, sd = values.mean(axis=1), values.std(axis=1, ddof=1)
            means[j, start:start+count] = mean
            np.divide(mean, sd, out=effects[j, start:start+count], where=sd > 0)
    return means, effects


def final_analysis(frame, plan, lock):
    """Return metric/pair tables and JSON summary from aggregate definitive scores."""
    validate_plan(plan)
    n = lock.get("common_R")
    if not isinstance(n, int) or not plan["dimensioning"]["execution_floor"] <= n <= plan["dimensioning"]["maximum_planning_R"]:
        raise ValueError("A fixed valid design must precede definitive inference")
    models, d = list(plan["models"]), plan["dimensioning"]
    seeds = [plan["definitive_seed_start"]+plan["definitive_seed_step"]*i for i in range(n)]
    frame = validate_frame(frame, seeds, models)
    pivot = frame.pivot(index="seed", columns="model", values="rmse").loc[seeds, models]
    scores = pivot.to_numpy()
    metrics = []
    for j, model in enumerate(models):
        values = scores[:, j]
        mean, sd = float(values.mean()), float(values.std(ddof=1))
        half = stats.t.ppf(.975, n-1)*sd/np.sqrt(n)
        metrics.append({"history_days": 7, "model": model, "n_seeds": n, "n_dates_per_trajectory": 366,
                        "rmse_mean": mean, "rmse_sample_sd": sd,
                        "rmse_CV_percent": 100*sd/mean if mean > 0 else None,
                        "rmse_ci95_lower": mean-half, "rmse_ci95_upper": mean+half,
                        "scope": "Fresh training seeds conditional on fixed2024; not market uncertainty"})
    pairs = plan["primary_pairs"]
    indices = [(models.index(p["model_a"]), models.index(p["model_b"])) for p in pairs]
    B = plan["analysis"]["bootstrap_replicates"]
    boot_mean, boot_dz = _bootstrap_pairs(scores, indices, B, plan["analysis"]["bootstrap_seed"])
    rows, pvalues = [], []
    for j, (pair, (a, b)) in enumerate(zip(pairs, indices)):
        diff = scores[:, a]-scores[:, b]
        mean, sd = float(diff.mean()), float(diff.std(ddof=1))
        se = sd/np.sqrt(n)
        critical95 = stats.t.ppf(.975, n-1)
        critical_family = stats.t.ppf(1-d["alpha_each"]/2, n-1)
        lo, hi = mean-critical_family*se, mean+critical_family*se
        if sd > 0:
            statistic = mean/se
            logp = float(np.log(2.)+stats.t.logsf(abs(statistic), n-1))
            p = float(np.exp(logp)) if np.isfinite(logp) else 0.
            shapiro = stats.shapiro(diff)
            W, normal_p = float(shapiro.statistic), float(shapiro.pvalue)
            status = "paired_t_primary_predeclared_conditional_seed_mean"
        else:
            statistic = logp = p = W = normal_p = None
            status = "degenerate_zero_SD_no_paired_t_claim"
        pvalues.append(p)
        dz_values = boot_dz[j, np.isfinite(boot_dz[j])]
        practical = ("A_lower_error_beyond_delta_by_family_CI" if hi < -d["delta_absolute"] else
                     "B_lower_error_beyond_delta_by_family_CI" if lo > d["delta_absolute"] else
                     "practical_advantage_beyond_delta_not_demonstrated")
        rows.append({**pair, "n_pairs": n, "mean_difference_A_minus_B": mean, "paired_SD": sd,
            "mean_difference_ci95_lower": mean-critical95*se, "mean_difference_ci95_upper": mean+critical95*se,
            "family_CI_level": 1-d["alpha_each"], "mean_difference_family_ci_lower": lo,
            "mean_difference_family_ci_upper": hi, "t_statistic": statistic, "df": n-1,
            "p_value": p, "log_p_value": logp, "p_underflow": bool(p == 0.) if p is not None else False,
            "paired_Cohen_d_z": mean/sd if sd > 0 else None,
            "mean_difference_bootstrap95_lower_sensitivity": float(np.quantile(boot_mean[j], .025)),
            "mean_difference_bootstrap95_upper_sensitivity": float(np.quantile(boot_mean[j], .975)),
            "d_z_bootstrap95_lower_sensitivity": float(np.quantile(dz_values, .025)) if len(dz_values) else None,
            "d_z_bootstrap95_upper_sensitivity": float(np.quantile(dz_values, .975)) if len(dz_values) else None,
            "bootstrap_replicates": B, "bootstrap_defined_effect_replicates": len(dz_values),
            "shapiro_W": W, "shapiro_p_value": normal_p,
            "normality_warning_p_lt_005": bool(normal_p < .05) if normal_p is not None else None,
            "delta_absolute": d["delta_absolute"], "practical_interpretation": practical,
            "zero_null_and_practical_boundary_are_distinct": True, "status": status})
    adjusted = holm_adjust([p if p is not None else 1. for p in pvalues])
    for row, p, correction in zip(rows, pvalues, adjusted):
        row["p_holm3"] = float(correction) if p is not None else None
        row["reject_zero_null_holm3"] = bool(correction < d["alpha_global"]) if p is not None else None
    metric_frame, pair_frame = pd.DataFrame(metrics), pd.DataFrame(rows)
    summary = {
        "status": "definitive_conditional_seed_analysis_complete", "R_fixed_before_definitive": n,
        "fresh_definitive_trajectories": len(frame), "primary_comparisons": 3,
        "primary_test": "Two-sided paired t on annual RMSE seed differences; Holm3",
        "bootstrap_replicates_sensitivity": B, "bootstrap_seed": plan["analysis"]["bootstrap_seed"],
        "family_CI_level": 1-d["alpha_each"], "alpha_global": d["alpha_global"],
        "family_CI_individual_adjusted_level": 1-d["alpha_each"],
        "family_CI_simultaneous_minimum_coverage": 1-d["alpha_global"],
        "delta_absolute": d["delta_absolute"], "planning_power_target_not_achieved_power": d["power_target"],
        "reject_zero_null_pairs": int(pair_frame.reject_zero_null_holm3.eq(True).sum()),
        "normality_warning_pairs": int(pair_frame.normality_warning_p_lt_005.eq(True).sum()),
        "normality_diagnostic_changes_primary_test": False,
        "lowest_mean_model_descriptive": metrics[int(np.argmin([r["rmse_mean"] for r in metrics]))]["model"],
        "pilot_seeds_in_definitive": False, "test2025_2026_read": False,
        "new_market_generalization_claim": False, "historical_inference_relabelled_confirmatory": False,
        "advanced_gain_boundary_tested": False, "achieved_or_joint_power_claim": False,
        "scope": "Prospectively registered fresh-RNG comparisons conditional on familiar2024 calendar and frozen D7 recipes",
    }
    return metric_frame, pair_frame, summary


def _read_context(phase):
    plan = json.loads(PLAN.read_text())
    validate_plan(plan)
    for key, expected in plan.get("code_sha256", {}).items():
        if digest(ROOT/key) != expected:
            raise ValueError("Registered code changed: "+key)
    for key, expected in plan.get("source_sha256", {}).items():
        if digest(ROOT/key) != expected:
            raise ValueError("Registered source changed: "+key)
    directory = OUT/phase
    execution = json.loads((directory/"execution_summary.json").read_text())
    if execution.get("protocol_sha256", execution.get("registered_protocol_sha256")) != digest(PLAN):
        raise ValueError("Execution does not match the sealed protocol")
    frame = pd.read_csv(directory/"runs.csv")
    return plan, frame, execution


def _metric_plot(path, metrics, plan):
    fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    axis.errorbar(metrics.model, metrics.rmse_mean, yerr=(metrics.rmse_ci95_upper-metrics.rmse_ci95_lower)/2,
                  fmt="o", capsize=6, color="#1764a0")
    reference = plan["dimensioning"].get("reference_persistence_rmse", plan["dimensioning"]["delta_absolute"]/.05)
    axis.axhline(reference, color="black", linestyle="--", label="Persistencia histórica descriptiva")
    axis.set_ylabel("RMSE anual · media e IC marginal 95 % entre semillas")
    axis.set_title("Definitiva D7 · aleatoriedad de entrenamiento · calendario 2024")
    axis.legend(fontsize=8)
    axis.grid(alpha=.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def all_metric_summary(frame):
    """Long secondary table; R2 has no coefficient-of-variation interpretation."""
    rows = []
    for model in ["MLP", "LSTM", "CNN1D"]:
        group = frame.loc[frame.model.eq(model)]
        for metric in ["rmse", "mse", "mae", "r2", "mape", "mase"]:
            if metric not in group:
                continue
            values = group.loc[np.isfinite(group[metric]), metric].to_numpy()
            count = len(values)
            mean = float(values.mean()) if count else None
            sd = float(values.std(ddof=1)) if count >= 2 else None
            half = stats.t.ppf(.975, count-1)*sd/np.sqrt(count) if count >= 2 else None
            rows.append({"history_days": 7, "model": model, "metric": metric,
                         "n_seeds": len(group), "n_defined": count,
                         "n_dates_per_trajectory": 366, "mean": mean, "sample_sd": sd,
                         "CV_percent": 100*sd/mean if metric != "r2" and mean is not None and mean > 0 and sd is not None else None,
                         "ci95_lower": mean-half if half is not None else None,
                         "ci95_upper": mean+half if half is not None else None,
                         "scope": "Marginal initialization uncertainty on known2024; no temporal/generalization CI"})
    return pd.DataFrame(rows)


def _qq_plot(path, frame, plan, seeds, title):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, pair in zip(axes, plan["primary_pairs"]):
        stats.probplot(difference(frame, pair, seeds), dist="norm", plot=axis)
        axis.set_title(pair["model_a"]+" − "+pair["model_b"], fontsize=10)
        axis.set_xlabel("Cuantiles normales teóricos")
        axis.set_ylabel("Diferencia de RMSE")
        axis.grid(alpha=.25)
    fig.suptitle(title)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def lock_design():
    if (OUT/"locked_design.json").exists() or (OUT/"planning").exists():
        raise FileExistsError("Preserve the completed design lock; use --phase check")
    plan, frame, _ = _read_context("variance_extension")
    table, summary = dimension_from_metrics(pd.read_csv(OLD), frame, plan)
    stage = Path(tempfile.mkdtemp(prefix=".design_stage_", dir=OUT))
    table.to_csv(stage/"paired_dimensioning.csv", index=False, lineterminator="\n")
    all_metric_summary(frame).to_csv(stage/"pilot_metrics_summary.csv", index=False, lineterminator="\n")
    write_json(stage/"variance_summary.json", summary)
    _qq_plot(stage/"qq_new20.png", frame, plan, plan["variance_extension_seeds"],
             "Precisión de varianza: 20 semillas nuevas · diagnóstico, sin contrastar medias")
    (stage/"README.md").write_text(
        "# Dimensionamiento bloqueado antes de la fase definitiva\n\n"
        f"R común requerido: {summary['common_R']}. Se usa exclusivamente la DE de las 20 semillas nuevas y su límite superior normal unilateral marginal del 95 %. "
        "Las cinco semillas anteriores y la sensibilidad de 25 combinadas no eligen R; las 25 quedan fuera de la definitiva. "
        "El mínimo de 30 es una regla de diseño, no una garantía de normalidad. Se planifica detectar δ frente a cero, con potencia marginal 0,80 y α=0,05/3; no una mejora avanzada superior al 5 %. "
        "R nunca se recorta al presupuesto. Si excede el límite operativo, se informa y se bloquea el lanzamiento. "
        "Este registro no convierte el calendario conocido en una prueba de mercado nueva ni reescribe el benchmark anterior.\n")
    hashes = {p.name: digest(p) for p in stage.iterdir() if p.is_file()}
    stage.rename(OUT/"planning")
    record = {**summary, "locked_at_utc": stamp(), "registered_protocol_sha256": digest(PLAN),
              "analysis_code_sha256": digest(Path(__file__)), "original_pilot_runs_sha256": digest(OLD),
              "variance_extension_runs_sha256": digest(OUT/"variance_extension/runs.csv"),
              "variance_extension_summary_sha256": digest(OUT/"variance_extension/execution_summary.json"),
              "input_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                  [PLAN, OLD, OUT/"variance_extension/runs.csv", OUT/"variance_extension/execution_summary.json"]},
              "design_output_sha256": hashes}
    write_json(OUT/"locked_design.json", record)
    return record


def analyze_definitive():
    destination = OUT/"analysis"
    if destination.exists():
        raise FileExistsError("Preserve definitive outputs; use --phase check")
    plan, frame, execution = _read_context("definitive")
    lock = json.loads((OUT/"locked_design.json").read_text())
    if lock["registered_protocol_sha256"] != digest(PLAN):
        raise ValueError("Lock protocol changed")
    if datetime.fromisoformat(lock["locked_at_utc"]) >= datetime.fromisoformat(execution["first_new_trajectory_started_at_utc"]):
        raise ValueError("Definitive runs must start strictly after the design lock")
    for key, expected in lock["input_sha256"].items():
        if digest(ROOT/key) != expected:
            raise ValueError("Changed design input before definitive analysis: "+key)
    metrics, pairs, summary = final_analysis(frame, plan, lock)
    stage = Path(tempfile.mkdtemp(prefix=".analysis_stage_", dir=OUT))
    metrics.to_csv(stage/"rmse_summary.csv", index=False, lineterminator="\n")
    all_metric_summary(frame).to_csv(stage/"metrics_summary.csv", index=False, lineterminator="\n")
    pairs.to_csv(stage/"paired_comparisons.csv", index=False, lineterminator="\n")
    _metric_plot(stage/"metrics.png", metrics, plan)
    seeds = [plan["definitive_seed_start"]+plan["definitive_seed_step"]*i for i in range(lock["common_R"])]
    _qq_plot(stage/"qq_differences.png", frame, plan, seeds, "Definitiva D7 · diagnósticos predeclarados")
    (stage/"README.md").write_text(
        "# Inferencia prospectiva condicionada a la aleatoriedad de entrenamiento\n\n"
        f"Se analizaron {lock['common_R']} semillas nuevas por receta D7 sobre las mismas 366 fechas de 2024. "
        "El número quedó bloqueado después del piloto de precisión y antes del primer ajuste definitivo. "
        "Las comparaciones primarias son tres t pareadas bilaterales de RMSE anual, con Holm-3. Se entregan IC marginales del 95 % e intervalos Bonferroni individuales del 98,333 %, cuya cobertura conjunta es al menos del 95 % para las tres diferencias, además de tamaños de efecto y sensibilidad bootstrap pareada de semillas (4.999 remuestreos). "
        "Shapiro y Q–Q no seleccionan otra prueba. Las sensibilidades no son réplicas independientes. "
        "Rechazar igualdad frente a cero no prueba una mejora mayor que δ: se interpreta por separado la posición del IC familiar respecto de ±δ. "
        "No se ensayó un modelo avanzado ni se probó su frontera del 5 %. La persistencia es descriptiva; no pertenece a los tres contrastes. "
        "La confirmación se refiere a nuevas inicializaciones bajo recetas y calendario conocidos, no a mercado desconocido. El test histórico y su inferencia posterior conservan su alcance original.\n")
    summary.update({"analyzed_at_utc": stamp(), "registered_protocol_sha256": digest(PLAN),
                    "locked_design_sha256": digest(OUT/"locked_design.json"),
                    "analysis_code_sha256": digest(Path(__file__)),
                    "input_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                        [PLAN, OUT/"locked_design.json", OUT/"definitive/runs.csv", OUT/"definitive/execution_summary.json"]},
                    "output_sha256": {p.name: digest(p) for p in stage.iterdir() if p.is_file()}})
    write_json(stage/"summary.json", summary)
    stage.rename(destination)
    return summary


def check():
    lock = json.loads((OUT/"locked_design.json").read_text())
    if lock["analysis_code_sha256"] != digest(Path(__file__)):
        raise ValueError("Analyzer changed after design lock")
    for key, expected in lock["input_sha256"].items():
        if digest(ROOT/key) != expected:
            raise ValueError("Changed design input: "+key)
    for key, expected in lock["design_output_sha256"].items():
        if digest(OUT/"planning"/key) != expected:
            raise ValueError("Changed design output: "+key)
    final_path = OUT/"analysis/summary.json"
    count = len(lock["design_output_sha256"])
    if final_path.exists():
        final = json.loads(final_path.read_text())
        for key, expected in final["input_sha256"].items():
            if digest(ROOT/key) != expected:
                raise ValueError("Changed final input: "+key)
        for key, expected in final["output_sha256"].items():
            if digest(OUT/"analysis"/key) != expected:
                raise ValueError("Changed final output: "+key)
        count += len(final["output_sha256"])
    # Recompute aggregate calculations; do not train or read individual outputs.
    plan, extension, _ = _read_context("variance_extension")
    recomputed_table, recomputed_summary = dimension_from_metrics(pd.read_csv(OLD), extension, plan)
    saved = pd.read_csv(OUT/"planning/paired_dimensioning.csv")
    pd.testing.assert_frame_equal(saved, recomputed_table, check_dtype=False, rtol=1e-12, atol=1e-14)
    pd.testing.assert_frame_equal(pd.read_csv(OUT/"planning/pilot_metrics_summary.csv"),
                                  all_metric_summary(extension), check_dtype=False,
                                  rtol=1e-12, atol=1e-14)
    for key, expected in recomputed_summary.items():
        if lock[key] != expected:
            raise ValueError("Changed recomputed design summary: "+key)
    if final_path.exists():
        _, frame, _ = _read_context("definitive")
        metrics, pairs, summary = final_analysis(frame, plan, lock)
        for name, expected in [("rmse_summary.csv", metrics), ("metrics_summary.csv", all_metric_summary(frame)),
                               ("paired_comparisons.csv", pairs)]:
            pd.testing.assert_frame_equal(pd.read_csv(OUT/"analysis"/name), expected,
                                          check_dtype=False, rtol=1e-12, atol=1e-14)
        for key, expected in summary.items():
            if final[key] != expected:
                raise ValueError("Changed recomputed final summary: "+key)
    return {"status": "PASS", "verified_outputs": count, "new_model_fits": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["lock", "analyze", "check"], required=True)
    args = parser.parse_args()
    result = {"lock": lock_design, "analyze": analyze_definitive, "check": check}[args.phase]()
    print(json.dumps({k: result[k] for k in ["status", "stage_status", "common_R", "R_fixed_before_definitive"] if k in result}))


if __name__ == "__main__":
    main()
