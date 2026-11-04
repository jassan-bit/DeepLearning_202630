"""Independent audit of the future academic boundary and joint-power scenarios.

Reads only policy, frozen CV selection, aggregate planning outputs and timestamps.
No project planning helpers, model weights, predictions or market data are used.
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
BASE = ROOT / "results/e1_audit_improvements_20261104/planning"
POLICY = ROOT / "experiments/e1_audit_improvements_20261104/academic_margin_policy.json"
GLOBAL_PROTOCOL = POLICY.with_name("protocol.json")
FREEZE = ROOT / "experiments/filipe_multihistory_20261104/freeze.json"
CODE = ROOT / "scripts/plan_e1_academic_benchmark.py"
REPORT = ROOT / "references/audits/revision_e1_academic_policy_20261104.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def power_for_n(n, gap, dispersion, alpha):
    """Upper-tail normal-difference t power; no subtraction from a CDF."""
    n = np.asarray(n, dtype=float)
    df = n-1
    critical = stats.t.isf(alpha, df)
    ncp = np.sqrt(n)*gap/dispersion
    return stats.nct.sf(critical, df, ncp)


def independent_minimum_r(gap, dispersion, alpha, target):
    """Exhaustive integer scan, independent of the planner's binary search."""
    assert gap > 0 and dispersion > 0
    approximation = ((stats.norm.isf(alpha)+stats.norm.ppf(target))*dispersion/gap)**2
    high = max(32, int(np.ceil(approximation))*2+32)
    while high <= 2_000_000:
        ns = np.arange(2, high+1)
        powers = power_for_n(ns, gap, dispersion, alpha)
        assert np.isfinite(powers).all()
        success = np.flatnonzero(powers >= target)
        if len(success):
            index = int(success[0])
            return int(ns[index]), float(powers[index]), float(powers[index-1]) if index else None
        high *= 2
    raise AssertionError("Independent integer scan exceeded planning bound")


def audit():
    policy = read_json(POLICY)
    freeze = read_json(FREEZE)
    summary_path = BASE / "academic_planning_summary.json"
    summary = read_json(summary_path)
    original_planning_summary = read_json(BASE / "summary.json")
    raw_marker = read_json(ROOT / "results/filipe_multihistory_20261104/test/raw_access_started.json")
    test_completion = read_json(ROOT / "results/filipe_multihistory_20261104/test/execution_summary.json")
    with (BASE / "academic_joint_power_scenarios.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    baseline = freeze["selection"]["baseline"]
    assert baseline["model"] == "MLP" and baseline["history_days"] == 7
    baseline_mean, baseline_sd = float(baseline["cv_rmse_mean"]), float(baseline["cv_rmse_sample_sd"])
    persistence = float(freeze["selection"]["persistence_cv_rmse"])
    q = float(policy["minimum_relative_improvement"])
    alpha, target = float(policy["planning_alpha_each"]), float(policy["planning_power_each"])
    assert q == .05 and alpha == .025 and target == .9
    assert policy["planning_alpha_global"] == 2*alpha == .05
    assert policy["joint_power_lower_bound_under_assumptions"] == .8
    assert np.isclose(1-2*(1-target), .8)
    assert policy["execution_floor"] == 5
    assert policy["status"] == "academic_operational_criterion_for_future_comparisons"
    assert policy["financial_utility_claim"] is False
    assert policy["new_evaluation_period_assigned"] is False
    assert policy["new_models_trained"] == policy["new_forecasts"] == 0
    assert summary["academic_context_confirmed"] is True
    assert summary["financial_utility_validated"] is False
    assert summary["current_test_pre_registration_claim"] is False
    assert summary["new_model_fits"] == summary["new_checkpoint_forward_calls"] == summary["raw_archive_reads"] == 0
    assert summary["policy_sha256"] == sha(POLICY)
    assert summary["freeze_sha256"] == sha(FREEZE)
    assert summary["code_sha256"] == sha(CODE)
    # This additional future-policy file must not rewrite the original supplement.
    assert sha(GLOBAL_PROTOCOL) == original_planning_summary["protocol_sha256"]
    assert sha(GLOBAL_PROTOCOL) == "e6319edf4c320707d51fc987b59f62967538784718ffe369d5a0e5b468576442"
    policy_time = datetime.fromisoformat(policy["recorded_at_utc"])
    assert datetime.fromisoformat(raw_marker["at_utc"]) < policy_time
    assert datetime.fromisoformat(test_completion["completed_at_utc"]) < policy_time
    assert policy_time < datetime.fromisoformat(summary["recorded_at_utc"])
    assert "after inspection" in policy["timing"] and "not a pre-test" in policy["timing"]

    keys = ("assumed_alternative_gain_vs_persistence", "advanced_SD_to_baseline_SD",
            "assumed_correlation_advanced_neural")
    expected = set(itertools.product(policy["illustrative_alternative_gain_vs_persistence"],
                                    policy["advanced_sd_to_neural_baseline_sd"],
                                    policy["correlations_with_neural_baseline"]))
    actual = {tuple(float(row[key]) for key in keys) for row in rows}
    assert expected == actual and len(rows) == len(actual) == summary["scenarios"] == 27
    maximum_error = 0.0

    def near(value, wanted, label):
        nonlocal maximum_error
        value, wanted = float(value), float(wanted)
        assert np.isfinite([value, wanted]).all(), label
        maximum_error = max(maximum_error, abs(value-wanted))
        assert np.isclose(value, wanted, rtol=2e-10, atol=1e-13), (label, value, wanted)

    all_joint_r, joint_bounds = [], []
    for row in rows:
        alternative, ratio, correlation = (float(row[key]) for key in keys)
        advanced_mean = (1-alternative)*persistence
        advanced_sd = ratio*baseline_sd
        gap_neural = (1-q)*baseline_mean-advanced_mean
        gap_persistence = (1-q)*persistence-advanced_mean
        # A common advanced model mean is used for both references.
        variance_neural = (advanced_sd*advanced_sd+(1-q)*(1-q)*baseline_sd*baseline_sd
                           -2*(1-q)*correlation*advanced_sd*baseline_sd)
        sd_neural = float(np.sqrt(variance_neural))
        # Persistence has no training-seed randomness on the fixed cohort.
        sd_persistence = advanced_sd
        near(row["boundary_relative_gain"], q, "Boundary")
        near(row["CV_neural_reference_rmse"], baseline_mean, "CV neural anchor")
        near(row["CV_persistence_reference_rmse"], persistence, "CV persistence anchor")
        near(row["assumed_advanced_mean_rmse"], advanced_mean, "One coherent advanced mean")
        near(row["mean_separation_above_neural_boundary"], gap_neural, "Neural boundary separation")
        near(row["mean_separation_above_persistence_boundary"], gap_persistence, "Persistence boundary separation")
        near(row["assumed_boundary_difference_SD_neural"], sd_neural, "Var(A-.95B) square root")
        near(row["assumed_boundary_difference_SD_persistence"], sd_persistence, "Deterministic persistence difference SD")
        near(row["one_sided_alpha_each"], alpha, "Unilateral alpha")
        near(row["power_target_each"], target, "Marginal target")
        rb, pb, pb_previous = independent_minimum_r(gap_neural, sd_neural, alpha, target)
        rp, pp, pp_previous = independent_minimum_r(gap_persistence, sd_persistence, alpha, target)
        assert rb == int(row["R_neural_theoretical"])
        assert rp == int(row["R_persistence_theoretical"])
        assert pb >= target and (pb_previous is None or pb_previous < target)
        assert pp >= target and (pp_previous is None or pp_previous < target)
        joint_r = max(5, rb, rp)
        assert joint_r == int(row["R_joint_with_guide_min5"])
        power_b = float(power_for_n(joint_r, gap_neural, sd_neural, alpha))
        power_p = float(power_for_n(joint_r, gap_persistence, sd_persistence, alpha))
        near(row["normal_assumption_power_neural_at_R"], power_b, "Neural power at joint R")
        near(row["normal_assumption_power_persistence_at_R"], power_p, "Persistence power at joint R")
        assert power_b >= target and power_p >= target
        joint_lower = max(0., power_b+power_p-1.)
        assert joint_lower >= .8-1e-12
        assert "hypothetical" in row["scope"] and "not achieved power" in row["scope"]
        all_joint_r.append(joint_r)
        joint_bounds.append(joint_lower)

    for name, checksum in summary["output_sha256"].items():
        assert sha(BASE/name) == checksum
    readme = (BASE/"academic_margin_README.md").read_text()
    for fragment in ("académico", "comparaciones futuras", "no es una tolerancia financiera",
                     "no es", "No se presume independencia", "piloto prospectivo"):
        assert fragment in readme
    assert "test/predictions" not in CODE.read_text()
    paths = [Path(__file__), POLICY, GLOBAL_PROTOCOL, FREEZE, CODE,
             BASE/"academic_joint_power_scenarios.csv", summary_path,
             BASE/"academic_margin_README.md", BASE/"summary.json",
             ROOT/"results/filipe_multihistory_20261104/test/raw_access_started.json",
             ROOT/"results/filipe_multihistory_20261104/test/execution_summary.json"]
    return {
        "status": "PASS", "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "verifier": "scripts/verify_e1_academic_policy.py",
        "independence": "No project math/planning helper imported; both component minimum integer R values independently enumerated using upper-tail noncentral-t probabilities.",
        "counts": {"joint_scenarios": 27, "component_variances": 54,
                   "independent_minimum_R_checks": 54, "power_at_joint_R_checks": 54,
                   "union_bound_checks": 27, "signed_output_files": len(summary["output_sha256"])},
        "academic_boundary": {"relative_gain_required_against_each_reference": q,
                              "references": ["Frozen CV-selected MLP/D7", "Deterministic persistence"],
                              "defined_after_current_test_for_future_evaluations": True,
                              "financial_utility_validated": False},
        "estimand": {
            "seed_score": "Pooled-date RMSE for each trained seed; then its expectation/mean across seeds conditional on the same evaluation calendar.",
            "relative_gain": "1 - E[m_A] / E[m_reference], requiring a positive reference mean",
            "paired_boundary_difference": "d_s = .95*m_reference_s - m_A_s; E[d_s] > 0 is the 5% ratio-of-means improvement condition",
            "not_claimed": "The mean of seed-wise ratios E[1-m_A_s/m_reference_s] is a different estimand for a stochastic neural reference.",
            "documentation_note": "Make the ratio-of-means definition explicit in the book or a separately dated clarification without silently changing the already hashed policy.",
        },
        "power_model": {"one_sided_alpha_each": alpha, "marginal_power_target": target,
                        "joint_power_lower_bound_under_assumptions": .8,
                        "does_not_assume_independence_between_comparisons": True,
                        "normal_seed_difference_assumption": True},
        "joint_R_range": [min(all_joint_r), max(all_joint_r)],
        "conditional_joint_power_lower_bound_range": [min(joint_bounds), max(joint_bounds)],
        "maximum_numeric_absolute_difference": maximum_error,
        "original_global_protocol_unchanged": True,
        "source_sha256": {path.relative_to(ROOT).as_posix(): sha(path) for path in paths},
        "software": {"numpy": np.__version__, "scipy": scipy.__version__},
        "raw_market_reads": 0, "new_project_training": 0, "new_checkpoint_forward_calls": 0,
        "limits": [
            "A future academic convention is now defined; this does not turn it into a margin registered before the already-inspected test.",
            "Dispersion, correlation and advanced-model alternatives remain illustrative; no advanced model was trained or measured.",
            "R and the union-bound power refer to the assumed normal paired-seed model conditional on a fixed calendar, not achieved market generalization power.",
            "The 5% boundary is an academic operational choice, not a validated economic or financial decision threshold.",
            "A prospective pilot and a new independently evaluated cohort are still required before claiming prospective design sufficiency.",
            "No new test period is assigned and the current baseline/epochs/checkpoints remain unchanged.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    result = audit()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    text = ["# Auditoría de política académica y planificación conjunta", "", "Estado: **PASS**.", "",
            result["independence"], "",
            "Se verificaron 27 escenarios, dos varianzas y dos mínimos enteros R por escenario. Se usó una única media avanzada por fila frente a ambas fronteras del 5 %; la persistencia es determinista respecto a las semillas.", "",
            "El estimando es la mejora de la razón de medias de RMSE por semilla: 1−E[m_A]/E[m_ref]. Con E[m_ref]>0, la condición E[0,95m_ref−m_A]>0 representa superar el 5 %. No es la media de las razones por semilla, que puede diferir para una referencia neuronal aleatoria.", "",
            "Cada contraste es unilateral con α=0,025 y potencia marginal objetivo0,90. El mínimo conjunto es max(5,R_neuronal,R_persistencia). La cota de unión da una potencia conjunta al menos0,80 bajo los supuestos, sin asumir independencia entre contrastes.", "",
            f"R conjunto ilustrativo: {result['joint_R_range'][0]}–{result['joint_R_range'][1]}; no es una recomendación única ni potencia alcanzada.", "",
            "La política se fechó después de observar el test actual y antes del nuevo cálculo de planificación. El protocolo global original conserva su SHA-256. MLP/D7 se toma de la selección congelada por CV, sin reselección por test.", "",
            "Conviene explicitar esta definición de razón de medias en el libro o una aclaración fechada separada, conservando los hashes de la política ya calculada.", "",
            "El 5 % queda definido como convención operativa para un benchmark académico futuro; no acredita utilidad financiera ni se presenta como margen previo al estudio terminado. El piloto prospectivo y otra evaluación independiente permanecen pendientes.", "",
            "## Límites", ""]
    text += ["- "+item for item in result["limits"]]
    text += ["", "## Fuentes y hashes", ""]
    text += [f"- `{path}`: `{checksum}`." for path, checksum in result["source_sha256"].items()]
    args.report.with_suffix(".md").write_text("\n".join(text)+"\n")
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "joint_R_range": result["joint_R_range"],
                      "maximum_numeric_absolute_difference": result["maximum_numeric_absolute_difference"]}))


if __name__ == "__main__":
    main()
