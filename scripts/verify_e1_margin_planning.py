"""Independently audit the hypothetical future-margin planning tables.

No project planning/training/analysis helper is imported. Only frozen selection,
protocol, aggregate planning tables and their hashes are read. No market data,
model weights, raw archives or prediction vectors are opened.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PLAN_DIR = ROOT / "results/e1_audit_improvements_20261104/planning"
PROTOCOL = ROOT / "experiments/e1_audit_improvements_20261104/protocol.json"
FREEZE = ROOT / "experiments/filipe_multihistory_20261104/freeze.json"
SOURCE_CODE = ROOT / "scripts/plan_e1_advanced_margin.py"
REPORT = ROOT / "references/audits/revision_e1_margin_planning_20261104.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def power_vector(repetitions, difference, difference_sd, alpha):
    """Two-tailed t power using survival functions in both symmetric tails."""
    n = np.asarray(repetitions, dtype=float)
    df = n - 1
    critical = stats.t.isf(alpha / 2, df)
    noncentrality = difference * np.sqrt(n) / difference_sd
    # Reflection: P(T_ncp < -critical) = SF(critical; df, -ncp).
    return (stats.nct.sf(critical, df, noncentrality)
            + stats.nct.sf(critical, df, -noncentrality))


def independent_minimum_n(difference, dispersion, alpha, target):
    """Enumerate every integer n, independently of the planner's binary search."""
    approximate = ((stats.norm.isf(alpha / 2) + stats.norm.ppf(target))
                   * dispersion / difference) ** 2
    upper = max(32, int(np.ceil(approximate)) * 2 + 32)
    while True:
        candidates = np.arange(2, upper + 1)
        values = power_vector(candidates, difference, dispersion, alpha)
        if not np.isfinite(values).all():
            raise AssertionError("Non-finite independent power calculation")
        eligible = np.flatnonzero(values >= target)
        if len(eligible):
            first = int(eligible[0])
            return int(candidates[first]), float(values[first]), float(values[first-1]) if first else None
        upper *= 2
        if upper > 2_000_000:
            raise AssertionError("No minimum n in the independently searched range")


def independent_mde(dispersion, alpha, target):
    """Bisection in original RMSE units, distinct from planner root finding."""
    low, high = 0.0, dispersion
    while float(power_vector(10, high, dispersion, alpha)) < target:
        high *= 2
    for _ in range(100):
        middle = (low + high) / 2
        if float(power_vector(10, middle, dispersion, alpha)) < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def audit():
    protocol = read_json(PROTOCOL)
    freeze = read_json(FREEZE)
    summary = read_json(PLAN_DIR / "summary.json")
    registration = read_json(PLAN_DIR / "registration.json")
    settings = protocol["advanced_planning"]
    rows = read_csv(PLAN_DIR / "advanced_margin_scenarios.csv")
    thresholds = read_csv(PLAN_DIR / "operational_threshold_scenarios.csv")
    baseline = freeze["selection"]["baseline"]
    assert baseline["model"] == "MLP" and baseline["history_days"] == 7
    assert baseline == summary["baseline_anchor"]
    mean, sd = float(baseline["cv_rmse_mean"]), float(baseline["cv_rmse_sample_sd"])
    persistence = float(freeze["selection"]["persistence_cv_rmse"])
    alpha, target = float(settings["planning_alpha"]), float(settings["power_target"])
    assert alpha == .025 and target == .8 and settings["planned_primary_future_comparisons"] == 2
    assert alpha == .05 / settings["planned_primary_future_comparisons"]
    assert mean > 0 and sd > 0 and persistence > 0
    assert summary["scenarios"] == len(rows) == 36 and len(thresholds) == 4
    assert summary["applied_margin_closed"] is False
    for record in (summary, registration):
        assert record["new_model_fits"] == record["new_checkpoint_forward_calls"] == record["raw_archives_opened"] == 0
        assert record["protocol_sha256"] == digest(PROTOCOL)
        assert record["source_freeze_sha256"] == digest(FREEZE)
        assert record["code_sha256"] == digest(SOURCE_CODE)
    assert datetime.fromisoformat(protocol["registered_at_utc"]) < datetime.fromisoformat(registration["registered_before_calculation_at_utc"])
    assert datetime.fromisoformat(registration["registered_before_calculation_at_utc"]) < datetime.fromisoformat(summary["completed_at_utc"])
    assert "Scenario" in summary["scope"] and "unvalidated" in summary["scope"]

    expected_grid = set(itertools.product(settings["relative_margin_scenarios"],
                                         settings["advanced_sd_to_baseline_sd"],
                                         settings["paired_score_correlation_scenarios"]))
    actual_grid = {(float(row["relative_reduction"]), float(row["advanced_sd_to_baseline_sd"]),
                    float(row["assumed_paired_correlation"])) for row in rows}
    assert actual_grid == expected_grid and len(actual_grid) == len(rows)
    maximum_difference = 0.0

    def near(actual, expected, label):
        nonlocal maximum_difference
        actual, expected = float(actual), float(expected)
        assert np.isfinite([actual, expected]).all(), label
        maximum_difference = max(maximum_difference, abs(actual - expected))
        assert np.isclose(actual, expected, rtol=2e-10, atol=1e-13), (label, actual, expected)

    required_n, mdes = [], []
    for row in rows:
        fraction = float(row["relative_reduction"])
        advanced_sd = sd * float(row["advanced_sd_to_baseline_sd"])
        correlation = float(row["assumed_paired_correlation"])
        # General paired variance, evaluated separately from the planner helper.
        variance = advanced_sd * advanced_sd + sd * sd - 2 * correlation * advanced_sd * sd
        dispersion, difference = float(np.sqrt(variance)), fraction * mean
        near(row["baseline_seed_sd"], sd, "Baseline dispersion")
        near(row["assumed_difference_sd"], dispersion, "Paired difference SD")
        near(row["delta_rmse"], difference, "Alternative effect")
        near(row["planned_alternative_difference"], difference, "Alternative definition")
        near(row["null_difference"], 0, "Zero null")
        near(row["planning_alpha_two_sided"], alpha, "Planning alpha")
        near(row["power_target"], target, "Power target")
        approximate = int(np.ceil(((stats.norm.isf(alpha/2) + stats.norm.ppf(target))
                                   * dispersion / difference) ** 2))
        assert int(row["R_normal_approximation_ceiling"]) == max(2, approximate)
        minimum, power_at_minimum, previous_power = independent_minimum_n(difference, dispersion, alpha, target)
        assert int(row["R_noncentral_t_normal_difference"]) == minimum
        assert power_at_minimum >= target and (previous_power is None or previous_power < target)
        near(row["scenario_power_at_minimum_R"], power_at_minimum, "Power at independent minimum")
        mde = independent_mde(dispersion, alpha, target)
        near(row["MDE_zero_at_R10_under_scenario"], mde, "Independent R10 MDE")
        near(power_vector(10, mde, dispersion, alpha), target, "Power at independent R10 MDE")
        assert "Hypothetical" in row["scope"] and "not achieved power" in row["scope"]
        required_n.append(minimum)
        mdes.append(mde)
    assert {float(row["relative_reduction"]) for row in thresholds} == set(settings["relative_margin_scenarios"])
    for row in thresholds:
        fraction = float(row["relative_reduction"])
        near(row["delta_rmse_vs_neural_CV"], fraction * mean, "Operational delta")
        near(row["neural_CV_anchor_rmse"], mean, "CV-only neural anchor")
        near(row["neural_target_rmse"], (1 - fraction) * mean, "Neural target")
        near(row["persistence_CV_reference_rmse"], persistence, "CV-only persistence anchor")
        near(row["persistence_target_at_same_fraction"], (1 - fraction) * persistence, "Contextual persistence target")
        assert "not a validated application margin" in row["status"]

    for name, expected in summary["output_sha256"].items():
        assert digest(PLAN_DIR / name) == expected, name
    readme = (PLAN_DIR / "README.md").read_text()
    for fragment in ("no una tolerancia financiera validada", "no equivale a demostrar", "margen aplicado permanece pendiente"):
        assert fragment in readme, fragment
    source = SOURCE_CODE.read_text()
    assert 'FREEZE = ROOT / "experiments/filipe_multihistory_20261104/freeze.json"' in source
    assert "test/predictions" not in source and "torch" not in source
    source_paths = [PROTOCOL, FREEZE, SOURCE_CODE, Path(__file__), PLAN_DIR/"summary.json",
                    PLAN_DIR/"registration.json", PLAN_DIR/"README.md",
                    PLAN_DIR/"advanced_margin_scenarios.csv", PLAN_DIR/"operational_threshold_scenarios.csv"]
    return {
        "status": "PASS", "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "verifier": "scripts/verify_e1_margin_planning.py",
        "independence": "No project planning/math/training/analysis helper imported; minimum R enumerated directly; MDE independently bisected in original units.",
        "counts": {"variance_scenarios": 36, "normal_R_scenarios": 36,
                   "noncentral_t_minimum_R_scenarios": 36, "R10_MDE_scenarios": 36,
                   "operational_threshold_scenarios": 4, "signed_planning_outputs_verified": len(summary["output_sha256"])},
        "planning_model": {"alpha_two_sided": alpha, "target_power": target,
                           "normal_paired_differences_assumed": True,
                           "two_tail_formula": "SF(tcrit;df,ncp)+SF(tcrit;df,-ncp)",
                           "alternative": "difference delta versus a zero null, not proving improvement exceeds delta"},
        "baseline_anchor": baseline, "persistence_cv_rmse": persistence,
        "contextual_neural_reduction_to_match_persistence_cv": 1-persistence/mean,
        "all_illustrative_neural_targets_above_persistence_cv": all(
            float(row["neural_target_rmse"]) > persistence for row in thresholds),
        "required_R_nct_range": [min(required_n), max(required_n)],
        "MDE_R10_range": [min(mdes), max(mdes)],
        "maximum_numeric_absolute_difference": maximum_difference,
        "sources_sha256": {path.relative_to(ROOT).as_posix(): digest(path) for path in source_paths},
        "new_raw_archive_reads": 0, "new_project_model_fits": 0, "new_checkpoint_forward_calls": 0,
        "scope_limits": [
            "Future-model dispersion and correlation are hypothetical, not measured from an advanced model.",
            "Normal paired differences and independent seed pairs are planning assumptions conditional on the historical calendar.",
            "These tables do not measure achieved power and do not replace a prospective pilot.",
            "A finite-sample noncentral-t calculation is exact only within its assumed normal-difference model.",
            "The applied utility margin remains open until the user/application supplies a defensible decision/cost context.",
            "The baseline comes from frozen CV2024 selection; no test-driven baseline selection or new model scoring occurred.",
        ],
        "applied_margin_closed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    result = audit()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    text = ["# Auditoría independiente de escenarios de margen y potencia", "", "Estado: **PASS**.", "",
            result["independence"], "",
            "Se verificaron 36 escenarios de varianza pareada, R por aproximación normal, mínimo entero R de t no central y MDE para R=10; además de cuatro umbrales operativos. Ambos extremos de cada mínimo R se comprobaron.", "",
            "La referencia sigue siendo MLP/D7 elegida con CV2024. No se seleccionó el baseline con test. α=0,025 bilateral y potencia objetivo 0,80 son supuestos de planificación para dos comparaciones futuras.", "",
            f"R mínimo condicionado a los escenarios: {result['required_R_nct_range'][0]}–{result['required_R_nct_range'][1]}. Estos extremos no son una recomendación única de tamaño muestral.", "",
            "Los resultados no acreditan potencia alcanzada, un piloto prospectivo ni un margen financiero validado. El margen aplicado sigue pendiente de contexto de decisión/costo. Detectar delta frente a cero no demuestra exceder una frontera delta.", "",
            "Los cuatro objetivos neuronales ilustrativos conservan mayor RMSE que la referencia CV de persistencia. El salto necesario para igualar esa referencia es contexto descriptivo, no un margen aplicado o una garantía de utilidad.", "",
            "No se leyeron archivos raw, pesos o vectores de predicciones, ni se entrenó o puntuó un modelo.", "",
            "## Límites", ""]
    text += ["- " + entry for entry in result["scope_limits"]]
    text += ["", "## Fuentes y hashes", ""]
    text += [f"- `{name}`: `{checksum}`." for name, checksum in result["sources_sha256"].items()]
    args.report.with_suffix(".md").write_text("\n".join(text) + "\n")
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "R_nct_range": result["required_R_nct_range"],
                      "maximum_numeric_absolute_difference": result["maximum_numeric_absolute_difference"]}))


if __name__ == "__main__":
    main()
