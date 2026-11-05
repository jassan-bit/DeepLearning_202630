"""Register and run a new-seed, development-only pilot of the frozen Filipe recipe.

The 2024 calendar is already known. Prospective registration concerns only these
new training runs, not the old benchmark or an independent market evaluation.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import multiprocessing as mp
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

import train_filipe_multihistory as core
import train_dl_benchmark as fitting
from dl_data import load_development, SOURCE_PATHS

ROOT = Path(__file__).resolve().parents[1]
STUDY = "e1_filipe_pilot_20261105"
OUT = ROOT / "results" / STUDY
PROTOCOL = ROOT / "experiments" / STUDY / "protocol.json"
CONTEXT = OUT / "code_manifest_before_fitting.json"
HISTORIES = [7, 14, 21, 28]
SEEDS = [6201, 6202, 6203, 6204, 6205]
FOLD_SOURCES = {d: ("results/dl_r2_cv_20261104/folds.json" if d == 7 else
                  f"results/filipe_multihistory_20261104/D{d}/folds.json") for d in HISTORIES}
CODE = ["scripts/run_filipe_pilot.py", "scripts/analyze_filipe_pilot.py",
        "scripts/train_filipe_multihistory.py", "scripts/train_dl_benchmark.py",
        "scripts/dl_data.py", "scripts/dl_data_v2.py", "scripts/dl_models.py"]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    fitting.write_json(path, value)


def validate_plan(plan):
    original = read(ROOT / "experiments/filipe_multihistory_20261104/protocol.json")
    if (plan["study_id"] != STUDY or plan["histories"] != HISTORIES
            or plan["seeds"] != SEEDS or list(plan["models"]) != ["MLP", "LSTM", "CNN1D"]
            or plan["variant"] != "relative_window" or plan["workers"] != 3
            or plan["test_files_opened_allowed"] != 0 or plan["raw_archives_opened_allowed"] != 0
            or plan["models"] != original["models"] or plan["training"] != original["training"]
            or plan["target_return_days"] != 7 or plan["forecast_horizon_days"] != 1):
        raise ValueError("Pilot scope, seeds, recipe or development-only boundary changed")
    expected = [{"history_days": d, "model_a": a, "model_b": b}
                for d in HISTORIES for a, b in combinations(plan["models"], 2)]
    dim = plan["dimensioning"]
    if (plan["primary_pairs"] != expected or dim["primary_family_size"] != 12
            or dim["alpha_global"] != .05 or dim["alpha_each"] != .05/12
            or dim["power_target"] != .8 or dim["execution_floor"] != 5
            or not np.isclose(dim["delta_absolute"], .05*dim["reference_persistence_rmse"], rtol=1e-15)):
        raise ValueError("Registered contrast family, margin or power plan changed")
    previous_seeds = set()
    for name in ["results/dl_r2_cv_20261104/runs.csv", "results/filipe_multihistory_20261104/runs.csv"]:
        previous_seeds.update(pd.read_csv(ROOT / name, usecols=["seed"])["seed"].astype(int))
    if previous_seeds.intersection(plan["seeds"]):
        raise ValueError("Pilot seeds must be new, not reused historical trajectories")


def validate_inputs(plan):
    validate_plan(plan)
    for name, expected in plan["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Registered source changed: " + name)
    data = load_development(ROOT)
    manifests = {}
    for d in HISTORIES:
        source = ROOT / FOLD_SOURCES[d]
        manifest = read(source)
        if (manifest["history_days"] != d or manifest["target_returns_days"] != 7
                or manifest["forecast_days"] != 1 or manifest["monthly_fold_count"] != 12
                or manifest["library"]["version"] != "0.1.5"):
            raise ValueError("Wrong registered library, history, target or fold manifest")
        outcomes = [i for fold in manifest["folds"]
                    for i in core.validate_fold(data, d, fold).tolist()]
        if outcomes != np.flatnonzero(data["calendar"].year == 2024).tolist():
            raise ValueError("Every 2024 date must appear once and in chronological order")
        manifests[d] = manifest
    indices = np.flatnonzero(data["calendar"].year == 2024)
    persistence = float(np.sqrt(np.mean((data["targets"][indices]-data["persistence"][indices])**2)))
    if not np.isclose(persistence, plan["dimensioning"]["reference_persistence_rmse"], rtol=1e-12):
        raise ValueError("Fixed persistence anchor differs on the registered cohort")
    return manifests


def make_plan():
    old_path = ROOT / "experiments/filipe_multihistory_20261104/protocol.json"
    freeze_path = old_path.with_name("freeze.json")
    old, freeze = read(old_path), read(freeze_path)
    anchor = freeze["selection"]["persistence_cv_rmse"]
    sources = list(SOURCE_PATHS) + list(FOLD_SOURCES.values()) + [str(old_path.relative_to(ROOT)), str(freeze_path.relative_to(ROOT))]
    return {
        "study_id": STUDY, "registered_at_utc": stamp(),
        "status": "registered_before_new_pilot_runs_not_before_historical_benchmark",
        "purpose": "Pilot variance and repetition planning for the current recipe, conditional on the already exposed 2024 calendar",
        "histories": HISTORIES, "seeds": SEEDS, "models": old["models"],
        "variant": "relative_window", "workers": 3, "training": old["training"],
        "target_return_days": 7, "forecast_horizon_days": 1, "primary_metric": "Annual pooled RMSE in original unannualized sigma7 units per seed trajectory",
        "initial_training_calendar": "2023", "validation_calendar": "2024-01-01..2024-12-31",
        "monthly_folds": 12, "embargo_days": "D, both inner and outer", "HPO": "None; original family recipes remain fixed",
        "epoch_rule": "Inner early stopping followed by a fresh refit for exactly the selected inner epochs, separately in every month",
        "seed_rule": "base+100000*monthordinal; refit adds10000000; minibatch generator adds17",
        "test_files_opened_allowed": 0, "raw_archives_opened_allowed": 0,
        "prior_exposure": "2023/2024 and original 2025-Aug2026 test have already been examined; this pilot adds fresh RNG trajectories only",
        "independence": "Five training randomizations conditional on one market/calendar; folds and dates are not independent market replications",
        "primary_pairs": [{"history_days": d, "model_a": a, "model_b": b}
                          for d in HISTORIES for a, b in combinations(old["models"], 2)],
        "dimensioning": {
            "alpha_global": .05, "primary_family_size": 12, "alpha_each": .05/12,
            "correction": "Conservative Bonferroni planning for twelve two-sided paired neural comparisons; future reporting may use Holm12",
            "power_target": .8, "power_scope": "Marginal per contrast under planning assumptions, not joint or achieved power",
            "delta_absolute": .05*anchor, "reference_persistence_rmse": anchor,
            "margin_timing": "This absolute between-network margin is adopted now, anchored to the previous academic 5% persistence convention; not retroactively inherited by the old study",
            "hypothesis": "Detect a paired RMSE difference of magnitude delta against zero; not prove an improvement above a 5% boundary",
            "normal_approximation": "ceil((z(1-alpha_each/2)+z(power))^2 * sd_difference^2 / delta^2)",
            "exact_planning": "Two-sided noncentral paired t under normal independent seed differences",
            "execution_floor": 5, "sd_upper_confidence": .95, "maximum_planning_R": 10000,
            "recommendation": "common_max_upper_SD_R",
            "R_definition": "New definitive seed trajectories per model and history, excluding the five pilot seeds",
            "future_seed_rule": "If a new conditional-seed benchmark is authorized, use base seeds50000001..50000000+R; never pool pilot6201..6205 or choose seeds by outcome",
            "uncertainty": "Report point-SD R and common maximum R using a one-sided95% marginal chi-square upper SD; not a simultaneous variance guarantee",
            "normality": "Shapiro-Wilk and QQ are diagnostics only; five pairs cannot validate normality or final power",
            "if_unidentifiable": "Report limitation for zero/nonfinite dispersion or requiredR beyond search bound; do not silently cap or claim sufficiency",
            "MDE": "At pilotR5 and at the recommended commonR, using the same alpha and power assumptions",
        },
        "secondary_reference": "Persistence RMSE is deterministic and descriptive here; neural-vs-persistence tests are not part of this primary family",
        "advanced_policy_separate": "The previous >5% versus MLP/D7 AND persistence policy requires its own advanced-method pilot and independent future evaluation",
        "compute_budget": {"trajectories": 60, "monthly_segments": 720, "inner_and_refit_fits": 1440,
                           "unique_validation_forecasts": 21960, "inference_passes_per_month": 25,
                           "validation_forward_batches": 18000, "new_reserved_test_forecasts": 0},
        "publication_scope": "Aggregate metrics, plans, config metadata and hashes; individual predictions, scalers and weights retained privately",
        "source_sha256": {name: digest(ROOT / name) for name in sources},
        "code_sha256": {name: digest(ROOT / name) for name in CODE},
    }


def register():
    if PROTOCOL.exists() or CONTEXT.exists() or OUT.exists():
        raise FileExistsError("Preserve registered studies; registration requires a fresh study directory")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip():
        raise ValueError("Commit the pilot implementation before recording its source revision")
    tracked = set(subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines())
    if not set(CODE).issubset(tracked):
        raise ValueError("Every registered source file must be versioned before registration")
    plan = make_plan()
    manifests = validate_inputs(plan)
    write(PROTOCOL, plan)
    for d in HISTORIES:
        path = OUT / f"D{d}/folds.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / FOLD_SOURCES[d]).read_bytes())
    runtime = fitting.runtime()
    runtime["timing_scope"] = "Pilot worker-wall time at three concurrent CPU trajectories; not serial hardware benchmarking"
    signature = hashlib.sha256(json.dumps({"protocol": digest(PROTOCOL), "code": plan["code_sha256"],
                                         "versions": runtime["versions"]}, sort_keys=True).encode()).hexdigest()
    context = {"registered_before_first_pilot_fit_at_utc": stamp(),
               "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "protocol_sha256": digest(PROTOCOL), "code_sha256": plan["code_sha256"],
               "code_signature": signature, "runtime": runtime, "test_files_opened": 0}
    write(CONTEXT, context)
    write(OUT / "protocol.json", plan)
    write(OUT / "runtime.json", runtime)
    print(json.dumps({"registered": True, "protocol_sha256": context["protocol_sha256"],
                      "pilot_trajectories": 60, "fits_executed": 0, "test_files_opened": 0}))


def validated_registration():
    plan, context = read(PROTOCOL), read(CONTEXT)
    if digest(PROTOCOL) != context["protocol_sha256"]:
        raise ValueError("Registered pilot protocol changed")
    for name, expected in plan["code_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Registered pilot code changed: " + name)
    if fitting.runtime()["versions"] != context["runtime"]["versions"]:
        raise ValueError("Registered runtime versions changed")
    manifests = validate_inputs(plan)
    for d in HISTORIES:
        if digest(OUT / f"D{d}/folds.json") != plan["source_sha256"][FOLD_SOURCES[d]]:
            raise ValueError("Copied fold manifest changed")
    return plan, manifests, context


def worker_init(plan, manifests, context):
    # Rebinding is confined to new spawned processes; no original file changes.
    core.OUT = OUT
    core.init_worker(plan, manifests, context)


def execute(job):
    receipt = OUT / "trajectory_start_receipts" / f"D{job['history_days']}_{job['model']}_s{job['seed']}.json"
    if not receipt.exists():
        write(receipt, {"started_at_utc": stamp(), "job": job, "protocol_sha256": digest(PROTOCOL)})
    return core.run_trajectory(job)


def aggregate(plan, context):
    records, predictions, folds = [], [], []
    for path in sorted((OUT / "runs").glob("*/summary.json")):
        record = core.verify_completion(path.parent, context["code_signature"], "summary.json")
        records.append(record)
        predictions.append(pd.read_csv(path.parent / "predictions.csv"))
        folds.append(pd.read_csv(path.parent / "fold_metrics.csv"))
    frame = pd.DataFrame(records).sort_values(["history_days", "model", "seed"])
    expected = {(d, model, seed) for d in HISTORIES for model in plan["models"] for seed in SEEDS}
    actual = set(zip(frame.history_days, frame.model, frame.seed))
    if actual != expected or len(frame) != 60 or not (frame.n_dates == 366).all():
        raise ValueError("Pilot trajectories are missing, duplicated or use the wrong cohort")
    merged = pd.concat(predictions, ignore_index=True)
    if len(merged) != 21960:
        raise ValueError("Wrong number of distinct validation predictions")
    frame.to_csv(OUT / "runs.csv", index=False, lineterminator="\n")
    merged.to_csv(OUT / "predictions.csv", index=False, lineterminator="\n")
    pd.concat(folds, ignore_index=True).to_csv(OUT / "fold_metrics.csv", index=False, lineterminator="\n")
    receipts = [read(path) for path in (OUT / "trajectory_start_receipts").glob("*.json")]
    earliest = min(row["started_at_utc"] for row in receipts)
    if context["registered_before_first_pilot_fit_at_utc"] >= earliest:
        raise ValueError("Pilot registration must precede all new training starts")
    summary = {"study_id": STUDY, "completed_at_utc": stamp(),
               "first_new_trajectory_started_at_utc": earliest,
               "registration_precedes_every_new_trajectory": True,
               "protocol_sha256": context["protocol_sha256"], "code_sha256": context["code_sha256"],
               "code_signature": context["code_signature"], "trajectories": 60,
               "monthly_segments": int(frame.folds_fitted.sum()), "project_fits": int(frame.model_fits.sum()),
               "unique_validation_predictions": len(merged), "inference_forward_batches": 720*25,
               "pilot_seeds": SEEDS, "dates_per_trajectory": 366, "histories": HISTORIES,
               "test_files_opened": 0, "raw_archives_opened": 0, "new_test_forecasts": 0,
               "historical_D7_reused": False, "all_fits_are_new_pilot_runs": True,
               "scope": "New registered pilot for training randomization on known development2024; not independent temporal validation or a retrospective repair",
               "output_sha256": {name: digest(OUT / name) for name in ["runs.csv", "fold_metrics.csv", "predictions.csv"]}}
    write(OUT / "execution_summary.json", summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["register", "validate", "run"], default="validate")
    args = parser.parse_args()
    if args.phase == "register":
        register()
        return
    plan, manifests, context = validated_registration()
    if args.phase == "validate":
        print(json.dumps({"status": "validated", "histories": HISTORIES, "seeds": SEEDS,
                          "prospective_for_new_pilot_only": True, "test_files_opened": 0}))
        return
    started = OUT / "run_started.json"
    if started.exists():
        if read(started)["protocol_sha256"] != digest(PROTOCOL):
            raise ValueError("Existing start marker belongs to another protocol")
    else:
        write(started, {"started_at_utc": stamp(), "protocol_sha256": digest(PROTOCOL),
                        "phase": "new pilot on known development only"})
    jobs = [{"variant": "relative_window", "model": model, "seed": seed, "history_days": d,
             "width": config["width"], "learning_rate": config["learning_rate"]}
            for d in HISTORIES for model, config in plan["models"].items() for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=plan["workers"], mp_context=mp.get_context("spawn"),
                             initializer=worker_init, initargs=(plan, manifests, context)) as executor:
        futures = [executor.submit(execute, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            record = future.result()
            print(json.dumps({"completed": i, "total": len(jobs), "model": record["model"],
                              "D": record["history_days"], "seed": record["seed"], "RMSE": record["rmse"]}), flush=True)
    aggregate(plan, context)


if __name__ == "__main__":
    main()
