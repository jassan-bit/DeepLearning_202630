"""Dimension a future conditional-seed benchmark from the registered Filipe pilot.

Reads only the new pilot's aggregate run metrics and signed registration. It does
not load checkpoints, individual predictions, market archives or the old test.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "e1_filipe_pilot_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "experiments/e1_filipe_pilot_20261105/protocol.json"
OUT = ROOT / "results/e1_filipe_pilot_20261105"


class PlanningLimitError(ValueError):
    """The registered planning bound cannot attain the requested power."""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_real(value, *, positive=False):
    return bool(np.isfinite(value) and (value > 0 if positive else True))


def normal_difference_power(n, difference, sd, alpha):
    """Two-sided paired-t power, assuming a true normal difference SD."""
    if (not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 2
            or not _valid_real(difference) or difference < 0
            or not _valid_real(sd, positive=True) or not 0 < alpha < 1):
        raise ValueError("Need integer n>=2, nonnegative finite difference, positive SD and valid alpha")
    critical = stats.t.ppf(1-alpha/2, n-1)
    noncentrality = difference/sd*np.sqrt(n)
    # Evaluate both tails using positive noncentralities to avoid unstable CDFs.
    power = float(stats.nct.sf(critical, n-1, noncentrality)
                  + stats.nct.sf(critical, n-1, -noncentrality))
    if not np.isfinite(power):
        raise ValueError("Nonfinite planning probability; cannot certify this scenario")
    return float(np.clip(power, 0., 1.))


def minimum_repetitions(difference, sd, alpha, target=.8, maximum=10000, floor=5):
    if (not _valid_real(difference, positive=True)
            or not _valid_real(sd, positive=True) or not 0 < alpha < target < 1
            or not isinstance(maximum, int) or not isinstance(floor, int)
            or not 2 <= floor <= maximum):
        raise ValueError("Invalid effect, SD, error rates, floor or planning bound")
    if normal_difference_power(maximum, difference, sd, alpha) < target:
        raise PlanningLimitError(f"Required repetitions exceed registered maximum {maximum}")
    low, high = floor, maximum
    while low < high:
        middle = (low+high)//2
        if normal_difference_power(middle, difference, sd, alpha) >= target:
            high = middle
        else:
            low = middle+1
    return low


def minimum_detectable_difference(n, sd, alpha, target=.8):
    if not _valid_real(sd, positive=True) or not 0 < alpha < target < 1:
        raise ValueError("Need a positive SD and alpha < target <1")
    # Validate n even when the bracketing path would otherwise terminate early.
    normal_difference_power(n, 0., sd, alpha)
    high = 1.
    while normal_difference_power(n, high, 1., alpha) < target:
        high *= 2
        if high > 1e6:
            raise PlanningLimitError("MDE standardization exceeds the numerical bound")
    standardized = optimize.brentq(
        lambda effect: normal_difference_power(n, effect, 1., alpha)-target,
        0., high, xtol=1e-12)
    return float(standardized*sd)


def upper_difference_sd(sample_sd, n, confidence=.95):
    """One-sided marginal SD upper confidence bound under normal differences."""
    if (not _valid_real(sample_sd, positive=True) or not isinstance(n, int)
            or n < 2 or not 0 < confidence < 1):
        raise ValueError("Need a positive sample SD, integer n>=2 and valid confidence")
    return float(sample_sd*np.sqrt((n-1)/stats.chi2.ppf(1-confidence, n-1)))


def normal_approximation_R(delta, sd, alpha, target, floor=5):
    if not _valid_real(delta, positive=True) or not _valid_real(sd, positive=True):
        raise ValueError("Need positive finite delta and SD")
    if not 0 < alpha < target < 1 or floor < 2:
        raise ValueError("Invalid error rates or execution floor")
    estimate = (stats.norm.ppf(1-alpha/2)+stats.norm.ppf(target))**2*sd**2/delta**2
    if not np.isfinite(estimate):
        raise ValueError("Nonfinite normal approximation")
    return max(floor, int(np.ceil(estimate)))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                               allow_nan=False)+"\n")


def validate_inputs(plan_path=PLAN, output=OUT):
    plan = json.loads(plan_path.read_text())
    execution_path = output / "execution_summary.json"
    execution = json.loads(execution_path.read_text())
    if execution.get("protocol_sha256") != digest(plan_path):
        raise ValueError("Pilot execution does not match the registered protocol hash")
    code = execution.get("code_sha256", {})
    for key in ["scripts/analyze_filipe_pilot.py", "analyze_filipe_pilot.py"]:
        if key in code and code[key] != digest(Path(__file__)):
            raise ValueError("Analyzer changed after the registered pilot execution")
    histories, seeds = plan["histories"], plan["seeds"]
    models = list(plan["models"])
    if (histories != [7, 14, 21, 28] or len(seeds) != 5 or len(set(seeds)) != 5
            or set(models) != {"MLP", "LSTM", "CNN1D"}):
        raise ValueError("Expected the registered four-history, three-model, five-seed pilot")
    expected_pairs = {(d, *sorted(pair)) for d in histories
                      for pair in itertools.combinations(models, 2)}
    pairs = plan["primary_pairs"]
    actual_pairs = [(p["history_days"], *sorted([p["model_a"], p["model_b"]])) for p in pairs]
    settings = plan["dimensioning"]
    if (len(pairs) != 12 or len(set(actual_pairs)) != 12
            or set(actual_pairs) != expected_pairs
            or settings["primary_family_size"] != len(pairs)
            or not np.isclose(settings["alpha_each"], settings["alpha_global"]/len(pairs))
            or not np.isclose(settings["delta_absolute"], .05*settings["reference_persistence_rmse"],
                              rtol=1e-12, atol=0.)):
        raise ValueError("Registered contrast family, multiplicity or fixed margin is inconsistent")
    runs_path = output / "runs.csv"
    runs = pd.read_csv(runs_path)
    required = {"model", "history_days", "seed", "rmse", "n_dates"}
    if not required.issubset(runs.columns):
        raise ValueError(f"Missing pilot columns: {required-set(runs.columns)}")
    expected_keys = set(itertools.product(histories, models, seeds))
    actual_keys = list(runs[["history_days", "model", "seed"]].itertuples(index=False, name=None))
    if (len(runs) != 60 or len(set(actual_keys)) != 60 or set(actual_keys) != expected_keys
            or not runs.n_dates.eq(366).all() or not np.isfinite(runs.rmse).all()
            or (runs.rmse < 0).any()):
        raise ValueError("Pilot trajectories must cover every registered cell and all366 dates exactly once")
    if "variant" in runs and not runs.variant.eq("relative_window").all():
        raise ValueError("Only the registered relative-window representation is admissible")
    return plan, runs, execution, {
        str(plan_path.relative_to(ROOT)): digest(plan_path),
        str(runs_path.relative_to(ROOT)): digest(runs_path),
        str(execution_path.relative_to(ROOT)): digest(execution_path),
    }


def calculate(plan, runs):
    settings = plan["dimensioning"]
    alpha, target, delta = (settings[k] for k in ["alpha_each", "power_target", "delta_absolute"])
    floor, bound = settings["execution_floor"], settings["maximum_planning_R"]
    metrics = []
    for history in plan["histories"]:
        for model in plan["models"]:
            values = runs.loc[(runs.history_days == history) & (runs.model == model)].sort_values("seed").rmse.to_numpy()
            mean, sd = float(values.mean()), float(values.std(ddof=1))
            half = stats.t.ppf(.975, len(values)-1)*sd/np.sqrt(len(values))
            metrics.append({"history_days": history, "model": model, "n_seeds": len(values),
                "n_dates_per_trajectory": 366, "rmse_mean": mean, "rmse_sample_sd": sd,
                "rmse_coefficient_variation_percent": 100*sd/mean if mean > 0 else None,
                "rmse_ci95_lower": mean-half, "rmse_ci95_upper": mean+half,
                "ci_scope": "Marginal Student-t across five pilot initializations, conditional on the fixed2024 calendar"})
    rows, differences = [], []
    for pair in plan["primary_pairs"]:
        history, a, b = pair["history_days"], pair["model_a"], pair["model_b"]
        cell = runs.loc[runs.history_days == history].pivot(index="seed", columns="model", values="rmse")
        d = (cell.loc[plan["seeds"], a]-cell.loc[plan["seeds"], b]).to_numpy()
        sd = float(d.std(ddof=1))
        row = {"history_days": history, "model_a": a, "model_b": b, "n_pilot_pairs": len(d),
            "mean_difference_descriptive": float(d.mean()), "paired_difference_sample_sd": sd,
            "delta_absolute": delta, "delta_reference": "5percent of the fixed historical2024 persistence RMSE; adopted for this new pilot",
            "standardized_planning_delta_over_sd": None, "alpha_global": settings["alpha_global"],
            "primary_family_size": settings["primary_family_size"], "planning_alpha_each": alpha,
            "marginal_power_target": target, "shapiro_W": None, "shapiro_p": None,
            "normality_warning_p_lt_005": None,
            "normality_limit": "Five differences cannot establish normality; diagnostics do not select a favorable test or effect",
            "R_normal_approximation_ceiling": None, "R_exact_noncentral_t_point_SD": None,
            "paired_SD_one_sided95_upper_normal_assumption": None, "R_exact_noncentral_t_upper_SD": None,
            "MDE_at_pilot_R5_point_SD": None, "MDE_at_pilot_R5_upper_SD": None,
            "status": "valid_under_stated_normal_difference_planning_assumptions", "planning_limit": ""}
        if sd <= 0:
            row["status"] = "degenerate_zero_pilot_SD_cannot_dimension"
        else:
            shapiro = stats.shapiro(d)
            upper = upper_difference_sd(sd, len(d), settings["sd_upper_confidence"])
            row.update(standardized_planning_delta_over_sd=delta/sd,
                       shapiro_W=float(shapiro.statistic), shapiro_p=float(shapiro.pvalue),
                       normality_warning_p_lt_005=bool(shapiro.pvalue < .05),
                       R_normal_approximation_ceiling=normal_approximation_R(delta, sd, alpha, target, floor),
                       paired_SD_one_sided95_upper_normal_assumption=upper,
                       MDE_at_pilot_R5_point_SD=minimum_detectable_difference(len(d), sd, alpha, target),
                       MDE_at_pilot_R5_upper_SD=minimum_detectable_difference(len(d), upper, alpha, target))
            for key, dispersion in [("R_exact_noncentral_t_point_SD", sd), ("R_exact_noncentral_t_upper_SD", upper)]:
                try:
                    row[key] = minimum_repetitions(delta, dispersion, alpha, target, bound, floor)
                except PlanningLimitError as error:
                    row["status"] = "exceeds_registered_planning_bound_no_capped_R_claim"
                    row["planning_limit"] += str(error)+"; "
        rows.append(row)
        differences.append((pair, d))
    point_values, upper_values = [[r[key] for r in rows] for key in
        ["R_exact_noncentral_t_point_SD", "R_exact_noncentral_t_upper_SD"]]
    point_R = max(point_values) if all(v is not None for v in point_values) else None
    upper_R = max(upper_values) if all(v is not None for v in upper_values) else None
    for row in rows:
        row["common_point_SD_R"] = point_R
        row["common_recommended_upper_SD_R"] = upper_R
        for suffix, n in [("common_point_R", point_R), ("common_recommended_R", upper_R)]:
            for label, sd in [("point_SD", row["paired_difference_sample_sd"]),
                              ("upper_SD", row["paired_SD_one_sided95_upper_normal_assumption"])]:
                row[f"MDE_at_{suffix}_{label}"] = (minimum_detectable_difference(n, sd, alpha, target)
                    if n is not None and sd is not None and sd > 0 else None)
        row["planned_power_at_common_recommended_R_upper_SD"] = (
            normal_difference_power(upper_R, delta, row["paired_SD_one_sided95_upper_normal_assumption"], alpha)
            if upper_R is not None and row["paired_SD_one_sided95_upper_normal_assumption"] is not None else None)
        row["power_scope"] = "Hypothetical marginal power at a fixed delta under the planning model; not achieved or joint power"
    return pd.DataFrame(metrics), pd.DataFrame(rows), differences, point_R, upper_R


def plot_outputs(directory, metrics, differences, plan):
    fig, axes = plt.subplots(4, 3, figsize=(14, 13), constrained_layout=True)
    for axis, (pair, d) in zip(axes.flat, differences):
        stats.probplot(d, dist="norm", plot=axis)
        axis.set_title(f"D{pair['history_days']} · {pair['model_a']} − {pair['model_b']}", fontsize=10)
        axis.set_xlabel("Cuantiles normales teóricos")
        axis.set_ylabel("Diferencia de RMSE")
        axis.grid(alpha=.25)
    fig.suptitle("Q–Q de cinco diferencias piloto por contraste\nDiagnóstico limitado; no certifica normalidad", fontsize=14)
    fig.savefig(directory / "qq_difference_panels.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    colors = {"MLP": "#1764a0", "LSTM": "#bd5e00", "CNN1D": "#007d62"}
    for model in plan["models"]:
        part = metrics.loc[metrics.model.eq(model)].sort_values("history_days")
        axes[0].errorbar(part.history_days, part.rmse_mean, yerr=part.rmse_sample_sd,
                         marker="o", capsize=3, label=model, color=colors[model])
        axes[1].plot(part.history_days, part.rmse_sample_sd, marker="o", label=model, color=colors[model])
        axes[2].plot(part.history_days, part.rmse_coefficient_variation_percent, marker="o", label=model, color=colors[model])
    axes[0].axhline(plan["dimensioning"]["reference_persistence_rmse"], color="black", linestyle="--", label="Persistencia histórica")
    for axis, title in zip(axes, ["RMSE: media ± DE entre cinco semillas", "DE muestral de RMSE", "Coeficiente de variación (%)"]):
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Historia de entrada D (días)")
        axis.set_xticks(plan["histories"])
        axis.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Piloto registrado del procedimiento forward de Filipe · validación 2024", fontsize=13)
    fig.savefig(directory / "pilot_metrics.png", dpi=150)
    plt.close(fig)


def analyze(plan_path=PLAN, output=OUT):
    destination = output / "analysis"
    if destination.exists():
        raise FileExistsError("Preserve the completed pilot analysis; use --check to verify it")
    plan, runs, execution, inputs = validate_inputs(plan_path, output)
    metrics, pairs, differences, point_R, upper_R = calculate(plan, runs)
    stage = Path(tempfile.mkdtemp(prefix=".analysis_staging_", dir=output))
    metrics.to_csv(stage / "pilot_metrics.csv", index=False, lineterminator="\n")
    pairs.to_csv(stage / "paired_dimensioning.csv", index=False, lineterminator="\n")
    plot_outputs(stage, metrics, differences, plan)
    rank_floor = max(6, int(np.ceil(1-np.log2(plan["dimensioning"]["alpha_each"]))))
    point_label = str(point_R) if point_R is not None else "no identificable con este piloto"
    upper_label = str(upper_R) if upper_R is not None else "no identificable dentro del límite registrado"
    note = f"""# Piloto registrado y dimensionamiento de nuevas ejecuciones

Se ejecutaron cinco semillas nuevas para MLP, LSTM y CNN1D en D=7/14/21/28: 60 trayectorias anuales sobre las 366 fechas de validación 2024. Cada trayectoria agrupa los doce folds mensuales. Las fechas, las arquitecturas y el calendario ya eran conocidos; el registro es anterior a estos nuevos ajustes, no a la investigación histórica.

`pilot_metrics.csv` informa media, DE muestral, CV porcentual e IC t marginal del 95 % entre cinco inicializaciones. `paired_dimensioning.csv` contiene los 12 pares neuronales definidos antes del piloto: tres por historia. La aleatoriedad es de entrenamiento condicionada a un calendario; las fechas y los meses no son réplicas independientes.

El margen absoluto fijado para este nuevo piloto es δ={plan['dimensioning']['delta_absolute']:.17g}, el 5 % del RMSE histórico de persistencia. Se detecta una diferencia de tamaño δ frente a cero mediante una referencia t bilateral de diferencias normales. Esto no prueba una mejora superior al 5 % ni es el criterio futuro de un modelo avanzado frente a dos referencias.

La planificación usa α global=0,05 y α por contraste=0,05/12, conservadora para una familia futura de Holm-12. La potencia objetivo es 0,80 por contraste, no la probabilidad conjunta de detectar todos los pares. El efecto medio observado solo se describe; no se utiliza para calcular potencia observada ni elegir el tamaño de efecto de planificación.

Se informa la aproximación normal de la guía y la inversión de t no central. El máximo común con DE puntual es R={point_label}; con el límite superior unilateral marginal del 95 % de la DE bajo normalidad, R={upper_label}. El segundo es la recomendación condicional común, si no está ausente. Los límites de DE son marginales, no un intervalo simultáneo para las 12 varianzas. Un resultado que exceda 10000 se declara sin recortarlo como tamaño suficiente. La MDE se informa para las cinco semillas y para los tamaños comunes calculados.

Cinco diferencias no acreditan normalidad. Shapiro–Wilk y los gráficos Q–Q son diagnósticos de baja resolución, no una garantía ni un mecanismo para escoger una prueba favorable. La incertidumbre de la dispersión piloto limita estas recomendaciones. Para Wilcoxon bilateral exacto, {rank_floor} pares sin ceros ni empates solo permiten aritméticamente alcanzar el primer umbral de Holm-12; no garantizan potencia.

El R propuesto corresponde a semillas definitivas nuevas, excluyendo las cinco del piloto. Este análisis no inicia esas ejecuciones ni consulta el test 2025–agosto de 2026. Tampoco reasigna septiembre–octubre de 2026, crea una evaluación independiente de mercado o subsana retroactivamente el piloto ausente del benchmark anterior. La evaluación temporal independiente futura permanece por definir.
"""
    (stage / "README.md").write_text(note)
    report = {
        "status": "pilot_analyzed_conditional_planning" if upper_R is not None else "pilot_analyzed_dimensioning_flagged_no_complete_R",
        "analyzed_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": digest(plan_path),
        "analysis_code_sha256": digest(Path(__file__)), "input_sha256": inputs,
        "pilot_trajectories": len(runs), "histories": plan["histories"], "models": list(plan["models"]),
        "pilot_seeds": plan["seeds"], "pilot_seeds_per_cell": 5, "primary_paired_contrasts": len(pairs),
        "delta_absolute": plan["dimensioning"]["delta_absolute"], "alpha_global": plan["dimensioning"]["alpha_global"],
        "planning_alpha_each": plan["dimensioning"]["alpha_each"], "marginal_power_target": plan["dimensioning"]["power_target"],
        "common_point_SD_R": point_R, "common_recommended_upper_SD_R": upper_R,
        "normality_warning_pairs_p_lt_005": int(pairs.normality_warning_p_lt_005.eq(True).sum()),
        "normality_not_certified_with_five_pairs": True,
        "recommendation": plan["dimensioning"]["recommendation"], "wilcoxon_minimum_discrete_resolution_pairs": rank_floor,
        "SD_bounds_are_simultaneous": False, "achieved_power_claim": False, "joint_power_claim": False,
        "historical_study_repaired": False, "independent_market_evaluation_performed": False,
        "pilot_seeds_in_future_R": False, "definitive_training_started": False,
        "models_fitted_by_this_analyzer": 0, "individual_predictions_read": 0, "old_test_accesses": 0,
        "limits": ["Known2024calendar; conditional training randomness only", "Normal-difference power and SD bounds are assumptions, not certified by five pairs",
                   "Fixed operational delta, not observed-effect power or the advanced gain boundary",
                   "New independent market evaluation and definitive fresh-seed runs are not performed here"],
        "output_sha256": {path.name: digest(path) for path in stage.iterdir() if path.is_file()},
    }
    write_json(stage / "summary.json", report)
    stage.rename(destination)
    return report


def check(plan_path=PLAN, output=OUT):
    _, _, _, inputs = validate_inputs(plan_path, output)
    directory = output / "analysis"
    report = json.loads((directory / "summary.json").read_text())
    if report["input_sha256"] != inputs or report["analysis_code_sha256"] != digest(Path(__file__)):
        raise ValueError("Analysis inputs or code changed after the saved pilot analysis")
    for name, expected in report["output_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError("Changed pilot analysis output: "+name)
    return {"status": "PASS", "verified_outputs": len(report["output_sha256"]), "new_model_fits": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["analyze", "check"], default="analyze")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = check() if args.check or args.phase == "check" else analyze()
    print(json.dumps(clean({key: report[key] for key in ["status", "common_point_SD_R", "common_recommended_upper_SD_R"] if key in report}), allow_nan=False))


if __name__ == "__main__":
    main()
