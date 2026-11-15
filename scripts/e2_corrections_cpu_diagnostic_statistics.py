"""New CPU-only gap and timing diagnostics; the original GPU gate stays failed.

Consumes the complete new CPU experiment. It never loads a model, uses GPU test
predictions, changes margins or claims reproduction of the original benchmark.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np

from e2_corrections_inference_statistics import six_seed_test, holm, latency_matrices, centered_summary
from e2_effects_rq2_closure import NEURAL, TOP10, SEEDS, DELTA, sha256, require, write_csv, write_json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "book_e2/_static/e2_corrections_20261115/cpu_diagnostic"
DEFAULT_PLAN = BASE / "statistics/statistical_plan.json"
VERSION = "e2-cpu-diagnostic-statistics-v1"


def program_hashes():
    return {name: sha256(ROOT / "scripts" / name) for name in (
        "e2_corrections_cpu_diagnostic_statistics.py", "e2_corrections_inference_statistics.py",
        "e2_effects_rq2_closure.py")}


def register_plan(path, runner_plan):
    runner = json.loads(runner_plan.read_text())
    failures = sorted((ROOT / "book_e2/_static/e2_corrections_20261115/frozen_inference").rglob("failure_report.json"))
    value = {
        "version": VERSION, "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "New exploratory CPU-only diagnosis after two GPU-parity gate failures and21partial outputs; before complete78CPU forwards and new statistical calculation",
        "inference_plan_sha256": sha256(runner_plan),
        "inference_plan_path": runner_plan.relative_to(ROOT).as_posix(),
        "program_sha256": program_hashes(),
        "previous_gate_failures": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p)} for p in failures],
        "old_plan714_executed_or_replaced": False,
        "delta_g": DELTA, "seed_ids": list(SEEDS), "models": list(NEURAL),
        "inclusion": "All78 exact frozen neural checkpoints; CPUevalFP32batch1; all1e-7projection differences reported as alerts, never exclusion",
        "delta_g_analysis": {
            "gap": "CPU calibration53 RMSE minus CPU train655 RMSE for same frozen seed",
            "H0": "mean(six gaps)>=delta_g", "H1": "mean(six gaps)<delta_g",
            "test": "Student t lower-tail, df5, independent initialization pairs and adequate small-sample normal approximation assumed",
            "family": "13 neural models, Holm13, alpha0.05",
            "CI": "marginal95percent two-sided Student t interval on mean gap across six initialization repetitions",
            "zero_variance": "not estimable; no fabricated zero p-value",
            "decision": "within conditional operational tolerance only if valid p_Holm13<=0.05; otherwise not established or not estimable",
            "scope": "optimization uncertainty conditional on CPU predictor and fixed train/calibration periods; not market-population absence of overfitting",
        },
        "cost_analysis": {
            "candidates": list(TOP10), "reference": "GRU32", "blocks": 100, "seeds_per_block": 6,
            "effect": "equal-six-seed mean forward latency candidate minus GRU32, averaged over100paired global measurement blocks",
            "bootstrap": "20000 resamples of whole100paired global blocks with replacement; retain all six seed and candidate/reference associations",
            "bootstrap_seed": 202611153101, "replicates": 20000,
            "CI": "marginal95percent percentile interval of mean latency saving/difference; reportms and raw timing source",
            "H0": "mean cost difference>=0", "H1": "mean cost difference<0",
            "p": "(1+count(0+(cost_star-cost_observed)<=cost_observed))/(20000+1)",
            "family": "10CPUcost-only comparisons, Holm10,alpha0.05; separate from delta_g and originalRQ3",
            "invalid_policy": "missing,duplicate,nonfinite or mismatched pairs block timing inference; never impute missing elapsed with0",
            "scope": "new CPUFP32batch1forward on development inputs; no originalA100timing replacement or accuracy-cost IUT",
        },
        "prohibited_analyses": ["CPUcalibration versus GPUtest consistency", "original714IUT", "newRQ3gain from GPU trajectories", "new model fits", "changes to A2,RQ1,delta,delta_g or calibrators"],
        "limits": [
            "All diagnoses are posterior exploratory evidence; neither the original1e-7gate nor original benchmark adequacy is certified.",
            "The same numerical weights/scalers do not establish equivalence of CPU/GPU predictions; every deviation is retained.",
            "Six optimization pairs and the small-sample t approximation do not model calendar, HPO-selection or market-population uncertainty.",
            "Calibration53 was already used for pilot and intervals; it is not independent HPO validation and a small gap is not causal proof of absence of overfitting.",
            "Timer blocks require sufficiently stable/exchangeable session measurements; clock repeats do not create independent markets.",
            "A cost reduction alone is not joint predictive improvement, operational utility or equivalence to GPU deployment.",
        ],
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": importlib.metadata.version("scipy"), "platform": platform.platform()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)
    return {"statistical_plan": str(path), "sha256": sha256(path),
            "inference_plan_sha256": value["inference_plan_sha256"]}


def report(directory, name, stage, plan_sha, inference_sha):
    directory = Path(directory)
    item = json.loads((directory / name).read_text())
    expected_status = ("completed_CPU_diagnostic_with_projection_alerts" if stage == "predictions"
                       else "completed_CPU_latency_diagnostic")
    require(item["stage"] == stage and item["status"] == expected_status, "Not a completed new CPU diagnosis")
    require(item["spec_sha256"] == inference_sha and item["states_unchanged"], "Wrong CPU plan or changed weights")
    require(item.get("original_parity_gate_passed") is False, "CPU supplement must not claim original GPU gate passed")
    require(item.get("statistical_plan_sha256") == plan_sha, "Missing pre-forward statistics linkage")
    for name, expected in item["all_output_files"].items():
        require(Path(name).name == name, "Unexpected output path")
        path = directory / name
        require(path.stat().st_size == expected["bytes"] and sha256(path) == expected["sha256"], "CPU output bytes changed")
    return item


def gaps_from_rows(rows, delta_g):
    indexed = {}
    for row in rows:
        key = row["model"], int(row["seed"])
        require(key not in indexed, "Duplicate CPU risk predictor")
        require(int(row["train_count"]) == 655 and int(row["calibration_count"]) == 53,
                "Wrong CPU risk supports")
        require(row["state_before_sha256"] == row["state_after_sha256"]
                and int(row["batch_size"]) == 1 and row["device"] == "cpu", "Changed CPU state/batch/backend")
        indexed[key] = row
    require(set(indexed) == {(model, seed) for model in NEURAL for seed in SEEDS}, "All78CPUrisks required; alerts cannot filter rows")
    result = []
    for model in NEURAL:
        train = np.array([float(indexed[model, seed]["train_rmse"]) for seed in SEEDS])
        calibration = np.array([float(indexed[model, seed]["calibration_rmse"]) for seed in SEEDS])
        require(np.isfinite(train).all() and np.isfinite(calibration).all(), "Nonfinite CPU risk")
        statistic = six_seed_test(calibration-train, delta_g)
        result.append({"model": model, "mean_CPU_train_RMSE": float(train.mean()),
                       "mean_CPU_calibration_RMSE": float(calibration.mean()), "delta_g": delta_g,
                       **statistic, "n_seed_pairs": 6, "df": 5,
                       "scope": "CPU-only optimization uncertainty on fixed train655/calibration53; approximate small-samplet"})
    for row, adjusted in zip(result, holm([r["p_raw"] for r in result])):
        row["p_Holm13"] = adjusted
        row["within_tolerance_conditionally"] = adjusted is not None and adjusted <= .05
        row["decision"] = ("not_estimable" if adjusted is None else "within_conditional_operational_tolerance"
                           if row["within_tolerance_conditionally"] else "not_established")
    return result


def cost_from_rows(rows, plan):
    design = plan["cost_analysis"]
    a, b = latency_matrices(rows, blocks=design["blocks"], candidates=tuple(design["candidates"]))
    differences = a-b  # Each row is the mean of all six paired seed measurements in a global block.
    observed = differences.mean(axis=0)
    rng = np.random.default_rng(design["bootstrap_seed"])
    sampled = differences[rng.integers(0, design["blocks"], size=(design["replicates"], design["blocks"]))].mean(axis=1)
    result = []
    for j, model in enumerate(design["candidates"]):
        summary = centered_summary(observed[j], sampled[:, j], 0, lower=True)
        result.append({"candidate": model, "reference": "GRU32",
                       "mean_CPU_candidate_latency_ms": float(a[:, j].mean()*1000),
                       "mean_CPU_reference_latency_ms": float(b[:, j].mean()*1000),
                       "CPU_latency_difference_ms": summary["effect"]*1000,
                       "difference_CI95_lower_ms": summary["CI95_lower"]*1000,
                       "difference_CI95_upper_ms": summary["CI95_upper"]*1000,
                       "p_cost_lower_raw": summary["p_raw"], "valid_draws": summary["valid_draws"],
                       "blocks": design["blocks"], "seeds_per_block": 6,
                       "scope": "new CPU-onlyforward cost; no predictive improvement or originalRQ3IUT"})
    for row, adjusted in zip(result, holm([r["p_cost_lower_raw"] for r in result])):
        row["p_cost_Holm10"] = adjusted
        row["CPU_cost_reduction_supported"] = adjusted is not None and adjusted <= .05
    block_rows = [{"block_id": block, "candidate": model, "reference": "GRU32",
                   "candidate_mean_six_seed_ms": float(a[block, j]*1000),
                   "reference_mean_six_seed_ms": float(b[block, j]*1000),
                   "paired_block_difference_ms": float(differences[block, j]*1000)}
                  for block in range(design["blocks"]) for j, model in enumerate(design["candidates"])]
    return result, block_rows


def execute(args):
    started = datetime.now(timezone.utc).isoformat()
    plan_sha = sha256(args.plan)
    plan = json.loads(args.plan.read_text())
    require(plan["version"] == VERSION and plan["program_sha256"] == program_hashes(), "CPU statistics program changed from registered bytes")
    require(sha256(ROOT / plan["inference_plan_path"]) == plan["inference_plan_sha256"], "CPU inference plan changed")
    inference = report(args.predictions_output, "inference_report.json", "predictions", plan_sha, plan["inference_plan_sha256"])
    timing = report(args.latency_output, "latency_report.json", "latency", plan_sha, plan["inference_plan_sha256"])
    require(inference["predictors_completed"] == inference["calibration_comparisons_completed"] == 78, "Incomplete78predictors")
    risks = list(csv.DictReader((args.predictions_output / "risks_by_predictor.csv").open()))
    agreement = list(csv.DictReader((args.predictions_output / "calibration_agreement.csv").open()))
    require({(r["model"], int(r["seed"])) for r in agreement} == {(m, s) for m in NEURAL for s in SEEDS}
            and len(agreement) == 78, "Every original-GPU projection comparison must be disclosed")
    raw = list(csv.DictReader((args.latency_output / "raw_cpu_latency.csv").open()))
    require(not args.output.exists(), "Use a fresh result directory")
    args.output.mkdir(parents=True)
    gaps = gaps_from_rows(risks, plan["delta_g"])
    costs, block_rows = cost_from_rows(raw, plan)
    write_csv(args.output / "CPU_delta_g_Holm13.csv", gaps)
    write_csv(args.output / "CPU_cost_only_Holm10.csv", costs)
    write_csv(args.output / "CPU_latency_block_means_1000.csv", block_rows)
    differences = [float(r["max_absolute_prediction_difference"]) for r in agreement]
    summary = {"version": VERSION, "scope": "new CPU-only posterior diagnosis; not original benchmark reproduction or adequacy",
               "CPU_gap_tests": 13, "CPU_within_tolerance_conditionally": sum(r["within_tolerance_conditionally"] for r in gaps),
               "CPU_gap_not_estimable": sum(r["decision"] == "not_estimable" for r in gaps),
               "CPU_cost_tests": 10, "CPU_cost_reductions_Holm10": sum(r["CPU_cost_reduction_supported"] for r in costs),
               "projection_comparisons_reported": 78, "projection_alert_threshold": 1e-7,
               "projection_alerts": sum(value > 1e-7 for value in differences),
               "maximum_CPU_original_GPU_projection_difference": max(differences),
               "all78included_regardless_projection_alert": True,
               "original_parity_gate_passed": False, "old714statistics_executed": False,
               "original_RQ3_IUT_executed": False, "mixed_CPU_GPU_period_test_executed": False,
               "new_fits": 0, "A2_or_RQ1_changed": False, "limits": plan["limits"]}
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "execution_receipt.json", {"version": VERSION, "started_at_utc": started,
               "completed_at_utc": datetime.now(timezone.utc).isoformat(), "statistical_plan_sha256": plan_sha,
               "inference_plan_sha256": plan["inference_plan_sha256"], "program_sha256": program_hashes(),
               "inference_report_sha256": sha256(args.predictions_output / "inference_report.json"),
               "latency_report_sha256": sha256(args.latency_output / "latency_report.json"),
               "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": importlib.metadata.version("scipy")}})
    (args.output / "README.md").write_text("""# Diagnóstico estadístico del suplemento CPU

Esta campaña conserva los pesos congelados y usa únicamente sus nuevos riesgos
CPU train655/calibración53 y sus nuevas latencias CPU. Las diferencias frente a
las proyecciones GPU originales se publican completas: 1e−7 es una alerta, no un
filtro ni una puerta aprobada. Los dos intentos originales fallidos se conservan.

La tolerancia δg usa seis brechas por red, t unilateral inferior con5gl y Holm13.
El IC95 % es marginal y describe optimización condicionada a estos periodos y a
la aproximación t; no demuestra ausencia de sobreajuste del benchmark original.
La calibración no se convierte en validación independiente de selección.

El coste usa100bloques completos de medición pareada, manteniendo seis semillas
por candidato/referencia. Se publican medias, diferencias, IC percentiles20.000 y
p unilaterales con Holm10 propio. Requiere estabilidad suficiente entre bloques;
no convierte repeticiones del reloj en mercados independientes. Esta comparación
CPU no sustituye los costes A100 ni acredita una mejora conjunta de error y coste.

No se ejecuta la IUT del plan714, no se mezcla calibración CPU con test GPU y no
se modifica A2, RQ1, ningún peso, calibrador o margen. No rechazo no demuestra
equivalencia. El plan estadístico y el de inferencia se vinculan antes de los
forwards de esta campaña; el registro es posterior al benchmark y a dos fallos.
""", encoding="utf-8")
    write_json(args.output / "manifest.json", {"files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
               for p in sorted(args.output.iterdir()) if p.is_file()}, "scope": "CPU-only posterior diagnostics"})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--runner-plan", type=Path, default=BASE / "plan.json")
    parser.add_argument("--register-plan", action="store_true")
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--latency-output", type=Path)
    parser.add_argument("--output", type=Path, default=BASE / "statistics/results")
    args = parser.parse_args(argv)
    if args.register_plan:
        result = register_plan(args.plan, args.runner_plan)
    else:
        require(args.predictions_output is not None and args.latency_output is not None, "Both new CPU stages required")
        result = execute(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
