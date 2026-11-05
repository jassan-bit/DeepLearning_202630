"""Registered equal-budget tuning, prospective pilot and a new historical holdout.

This separate study never rewrites the earlier exploratory benchmark. Register
before fitting; select recipes with development; run the pilot before sizing;
freeze every final weight and scaler before the first new holdout price read.
The historical acquisition already inspected ZIP schemas/timestamps: the new
claim concerns unexamined study outcomes, never unseen calendar-future prices.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
import sys
sys.dont_write_bytecode = True

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

FAMILIES = ("GRU", "LSTM", "TCN")
HISTORY = 7
DEFAULT_OUTPUT = Path("/workspace/e1_new_study_20261105/run")
DESIGN = "experiments/gru_lstm_tcn_planned_20261105/design.json"
FOLDS = "results/dl_r2_cv_20261104/folds.json"
DEVELOPMENT = ("results/eda_development/daily_market_development.csv",
               "results/eda_development/daily_volatility_development.csv")
CODE = ("scripts/run_gru_lstm_tcn_planned.py", "scripts/planned_study_data.py",
        "scripts/planned_study_statistics.py", "scripts/dl_models_gru_lstm_tcn.py",
        "scripts/dl_models.py", "scripts/dl_data.py", "scripts/dl_data_v2.py",
        "scripts/train_dl_benchmark.py", "scripts/final_fit_filipe_multihistory.py",
        "scripts/load_filipe_reserved_test.py", "scripts/gru_lstm_tcn_diagnostics.py",
        "scripts/train_filipe_multihistory.py", "requirements-training.lock.txt", "requirements-cv.txt")
TRAINING = {"optimizer": "AdamW", "weight_decay": .0001, "batch_size": 32,
            "max_epochs": 100, "gradient_clip_norm": 1., "patience": 12,
            "min_delta": .00001, "dropout": .1}
_WORKER = None


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, document, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as stream:
        stream.write(json.dumps(clean(document), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def safe_path(base, name):
    base, name = Path(base).resolve(), Path(name)
    if name.is_absolute() or ".." in name.parts:
        raise ValueError("Signed artifact paths must remain relative")
    path = (base / name).resolve()
    if not path.is_relative_to(base):
        raise ValueError("Signed artifact path escapes its root")
    return path


def verify_mapping(base, mapping):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("A nonempty explicit artifact SHA mapping is required")
    for name, expected in mapping.items():
        path = safe_path(base, name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError("Missing or changed signed artifact: " + str(path))
    return len(mapping)


def source_files(root):
    names = (*CODE, *DEVELOPMENT, FOLDS, DESIGN)
    return {name: digest(safe_path(root, name)) for name in names}


def effective_settings(design):
    search = design["search"]
    pilot, final, reserve = design["pilot"], design["definitive"], design["reserve"]
    if (design["models"] != list(FAMILIES) or design["history_days"] != 7
            or design["target_returns_days"] != 7 or design["forecast_days"] != 1
            or search["method"] != "exhaustive_grid" or search["expected_new_fits"] != 960
            or search["folds_source"] != FOLDS):
        raise ValueError("The explicit D7 three-family 960-fit search design is required")
    result = {"histories": [7], "target_days": 7, "horizon": 1,
        "hpo": {"seeds": search["seeds"], "candidates": search["candidates"], "months": search["months"]},
        "pilot": {"seeds": pilot["seeds"], "origin": pilot["origin"],
                  "first_day": pilot["start"], "last_day": pilot["end"], "days": pilot["calendar_days"]},
        "final_fit_origin": final["origin"], "training": {**design["training"], "dropout": .1},
        "holdout_first_day": reserve["start"], "holdout_last_day": reserve["end"], "holdout_days": reserve["calendar_days"],
        "failure_policy": design["failure_policy"], "selection": search["evaluation"],
        "tie_tolerance": search["tie_tolerance"]}
    if (set(search["candidates"]) != set(FAMILIES)
            or any(len(search["candidates"][model]) != 8 for model in FAMILIES)
            or len(result["hpo"]["seeds"]) != 2 or len(result["pilot"]["seeds"]) < 5
            or search["months"] != [f"2024-{month:02d}" for month in range(1, 11)]
            or pilot["start"] != "2024-11-01" or pilot["end"] != "2024-12-30"
            or pilot["origin"] != "2024-11-01" or final["origin"] != "2025-01-01"
            or reserve["start"] != "2026-09-01" or reserve["end"] != "2026-10-30"
            or pilot["calendar_days"] != 60 or reserve["calendar_days"] != 60):
        raise ValueError("The registered equal eight-candidate/two-seed search and >=5 pilot seeds are required")
    for family, candidates in search["candidates"].items():
        widths = [16, 32] if family != "TCN" else [8, 16]
        expected = {(w, lr, d) for w in widths for lr in [.001, .003] for d in [0., .1]}
        actual = {(c["width"], c["learning_rate"], c["dropout"]) for c in candidates}
        if actual != expected or len({c["candidate_id"] for c in candidates}) != 8:
            raise ValueError("Each explicit candidate grid must cover the eight comparable declared combinations")
    return result


def register(root, out, workers=3, raw_root=None):
    root, out = Path(root).resolve(), Path(out).resolve()
    if (out / "registration.json").exists():
        return load_registration(root, out)
    if out.is_relative_to(root) or (out.exists() and any(out.iterdir())):
        raise ValueError("Use a fresh external output directory; do not overwrite existing experiments")
    if int(workers) != workers or workers < 1:
        raise ValueError("workers must be a positive integer")
    design = read_json(root / DESIGN)
    settings = effective_settings(design)
    sources = source_files(root)
    sys.path.insert(0, str(root / "scripts"))
    from dl_data import load_development
    from planned_study_data import reserve_metadata
    from train_dl_benchmark import runtime
    load_development(root)  # Already public development only; no reserve payload.
    reservation = reserve_metadata(root, raw_root=raw_root)
    folds = read_json(root / FOLDS)
    if (folds.get("history_days") != 7 or len(folds.get("folds", [])) != 12
            or folds.get("library", {}).get("version") != "0.1.5"):
        raise ValueError("The archived Filipe D7 fold contract is required")
    observed_runtime = runtime()
    observed_runtime["timing_scope"] = f"Wall time; CPU {int(workers)} workers, one intra/inter-op thread per worker; no GPU"
    try:
        source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                                text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        source_commit = None  # Restored public source ZIPs need not contain .git.
    registration = {"schema_version": 1, "registered_at_utc": stamp(), "design": design,
        "settings": settings, "sources_sha256": sources, "reservation": reservation,
        "runtime": observed_runtime,
        "source_commit": source_commit,
        "workers": int(workers), "reserve_price_payloads_read": 0,
        "source_scope": "A new study registered after prior development and historical-test exposure; not a retroactive preregistration",
        "holdout_scope": "Additional historical period withheld from this project's model evaluation; acquisition previously inspected archive schemas/timestamps"}
    signature_payload = json.dumps(clean(registration), sort_keys=True).encode()
    registration["registration_signature"] = hashlib.sha256(signature_payload).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "registration.json", registration, exclusive=True)
    write_json(out / "reservation.json", reservation, exclusive=True)
    write_json(out / "runtime.json", observed_runtime, exclusive=True)
    shutil.copyfile(root / DESIGN, out / "design_original.json")
    return registration


def load_registration(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    registration = read_json(out / "registration.json")
    claimed = registration["registration_signature"]
    unsigned = {k: v for k, v in registration.items() if k != "registration_signature"}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest() != claimed:
        raise ValueError("The original registration changed")
    verify_mapping(root, registration["sources_sha256"])
    if read_json(out / "reservation.json") != registration["reservation"]:
        raise ValueError("The reserved acquisition identities changed")
    return registration


def candidate_grid(registration):
    search = registration["settings"]["hpo"]
    return [{**candidate, "model": family, "history_days": 7}
            for family in FAMILIES for candidate in search["candidates"][family]]


def worker_init(root, out, registration):
    global _WORKER
    root, out = Path(root), Path(out)
    sys.path.insert(0, str(root / "scripts"))
    import torch
    import train_dl_benchmark as fitting
    import dl_models_gru_lstm_tcn as models
    from dl_data import load_development
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    fitting.build_model = models.build_model
    _WORKER = {"root": root, "out": out, "registration": registration,
               "data": load_development(root), "fitting": fitting, "models": models}


def infer_once(model, values):
    """One deterministic scoring pass, without warmup or repeated timing passes."""
    import torch
    model.eval()
    started = time.perf_counter()
    with torch.inference_mode():
        predictions = np.concatenate([model(torch.from_numpy(values[first:first + 256])).cpu().numpy()
                                      for first in range(0, len(values), 256)])
    return predictions, time.perf_counter() - started


def split_at(data, origin, history=7, inner_days=90):
    """Development-only selection/refit split including the Jan2025 boundary."""
    calendar = pd.DatetimeIndex(data["calendar"])
    start = pd.Timestamp(origin, tz="UTC")
    expected = pd.date_range("2023-01-01", "2024-12-31", tz="UTC")
    if not calendar.equals(expected) or history != 7 or start > calendar[-1] + pd.Timedelta(days=1):
        raise ValueError("Only the registered development calendar and D7 are allowed")
    origin_index = int((start - calendar[0]).days)
    inner_index = origin_index - inner_days
    positions = np.arange(len(calendar))
    eligible = np.asarray(data["eligible_by_history"][history], dtype=bool)
    train = np.flatnonzero(eligible & (positions <= origin_index - history - 1))
    inner = np.flatnonzero(eligible & (positions <= inner_index - history - 1))
    validation = np.flatnonzero(eligible & (positions >= inner_index) & (positions <= origin_index - history - 1))
    if any(not len(a) for a in (train, inner, validation)) or inner.max() > validation.min() - history - 1:
        raise ValueError("Invalid past-only split")
    return {"train_indices": train, "inner_train_indices": inner,
            "inner_validation_indices": validation, "fit_origin_day_utc": origin}


def cached_completion(directory, signature):
    path = directory / "completion.json"
    if not path.is_file():
        return None
    record = read_json(path)
    if record["registration_signature"] != signature:
        raise ValueError("Completed job belongs to a different registration")
    verify_mapping(directory, record["artifact_sha256"])
    return record["record"]


def prepare_directory(directory):
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError("An incomplete attempt is preserved at " + str(directory)
                           + "; review the root cause and explicitly preserve/archive it before resuming the identical registered job")
    directory.mkdir(parents=True, exist_ok=True)


def fit_block(job, split, directory, evaluation=None):
    """Inner epoch selection, then fresh refit; optional development prediction."""
    import torch
    from dl_data import mase_scale
    from dl_data_v2 import fit_relative_preprocessing, transform_relative_windows
    state = _WORKER
    signature = state["registration"]["registration_signature"]
    existing = cached_completion(directory, signature)
    if existing is not None:
        return existing
    prepare_directory(directory)
    data, fitting, models = state["data"], state["fitting"], state["models"]
    training = {**state["registration"]["settings"]["training"], "dropout": job["dropout"]}
    fitting._PROTOCOL = {"training": training}
    train, inner, validation = [np.asarray(split[key], dtype=int) for key in
                                ("train_indices", "inner_train_indices", "inner_validation_indices")]
    if (train.max() > int((pd.Timestamp(split["fit_origin_day_utc"], tz="UTC") - data["calendar"][0]).days) - 8
            or inner.max() > validation.min() - 8):
        raise ValueError("The training or inner embargo was violated")
    pre_inner = fit_relative_preprocessing(data, 7, inner)
    pre_full = fit_relative_preprocessing(data, 7, train)
    xi, yi = transform_relative_windows(data, 7, inner, pre_inner)
    xv, yv = transform_relative_windows(data, 7, validation, pre_inner)
    _, epoch, inner_curve, inner_seconds = fitting.optimization_fit(job, xi, yi, xv, yv, stage="inner")
    xt, yt = transform_relative_windows(data, 7, train, pre_full)
    model, _, full_curve, full_seconds = fitting.optimization_fit(job, xt, yt, epochs=epoch, stage="refit")
    denominator = mase_scale(data, train)
    record = {**job, "selected_epoch": epoch, "train_samples": len(train), "inner_train_samples": len(inner),
              "inner_validation_samples": len(validation), "training_seconds": inner_seconds + full_seconds,
              "parameters": models.count_parameters(model)["trainable"], "mase_scale": denominator}
    if evaluation is not None:
        evaluation = np.asarray(evaluation, dtype=int)
        if (np.intersect1d(train, evaluation).size or not data["eligible_by_history"][7][evaluation].all()
                or evaluation.min() < int((pd.Timestamp(split["fit_origin_day_utc"], tz="UTC") - data["calendar"][0]).days)):
            raise ValueError("Invalid development evaluation cohort")
        x, _ = transform_relative_windows(data, 7, evaluation, pre_full)
        scaled, seconds = infer_once(model, x)
        prediction = scaled.astype(float) * pre_full["y_scale"]
        labels = data["targets"][evaluation]
        frame = pd.DataFrame({"model": job["model"], "seed": job["trajectory_seed"],
            "fit_seed": job["seed"], "candidate_id": job["candidate_id"],
            "outcome_day": data["calendar"][evaluation].strftime("%Y-%m-%d"),
            "y_true": labels, "y_pred": prediction, "persistence_pred": data["persistence"][evaluation],
            "mase_scale": denominator})
        frame.to_csv(directory / "predictions.csv", index=False)
        record.update({"evaluation_days": len(evaluation), "inference_seconds": seconds,
                       **fitting.metrics(labels, prediction, denominator)})
    pd.DataFrame(inner_curve).to_csv(directory / "history_inner.csv", index=False)
    pd.DataFrame(full_curve).to_csv(directory / "history_refit.csv", index=False)
    write_json(directory / "preprocessing.json", {"inner": pre_inner, "refit": pre_full})
    write_json(directory / "split.json", split)
    write_json(directory / "configuration.json", {"job": job, "training": training,
        "architecture": models.architecture_spec(model), "inner_seed": job["seed"], "refit_seed": job["seed"] + 10000000})
    torch.save({"state_dict": model.state_dict(), "family": job["model"], "history_days": 7,
                "width": job["width"], "dropout": job["dropout"]}, directory / "model.pt")
    write_json(directory / "summary.json", record)
    artifacts = {p.name: digest(p) for p in directory.iterdir() if p.is_file()}
    write_json(directory / "completion.json", {"completed_at_utc": stamp(),
        "registration_signature": signature, "record": record, "artifact_sha256": artifacts})
    return record


def hpo_job(job):
    state = _WORKER
    from train_filipe_multihistory import validate_fold
    folds = read_json(state["root"] / FOLDS)["folds"][:10]
    directory = state["out"] / "hpo" / "runs" / f"{job['model']}_{job['candidate_id']}_s{job['trajectory_seed']}"
    records = []
    for ordinal, fold in enumerate(folds):
        evaluation = validate_fold(state["data"], 7, fold)
        month_start = pd.Timestamp(f"2024-{ordinal + 1:02d}-01", tz="UTC")
        expected_month = pd.date_range(month_start, month_start + pd.offsets.MonthBegin(1), inclusive="left")
        if not state["data"]["calendar"][evaluation].equals(expected_month):
            raise ValueError("The source folds must match each registered January--October month exactly")
        split = {key: fold[key] for key in ("train_indices", "inner_train_indices", "inner_validation_indices", "fit_origin_day_utc")}
        stage = {**job, "seed": job["trajectory_seed"] + ordinal * 100000}
        record = fit_block(stage, split, directory / f"month_{ordinal + 1:02d}", evaluation)
        records.append(record)
    frame = pd.concat([pd.read_csv(directory / f"month_{month:02d}" / "predictions.csv") for month in range(1, 11)], ignore_index=True)
    dates = pd.date_range("2024-01-01", "2024-10-31", tz="UTC").strftime("%Y-%m-%d").tolist()
    if frame.outcome_day.tolist() != dates or len(frame) != 305:
        raise ValueError("HPO must contain exactly all305 Jan--Oct2024 outcome days")
    result = {**job, "rmse": float(np.sqrt(np.mean((frame.y_true - frame.y_pred) ** 2))),
              "parameters": records[0]["parameters"], "evaluation_days": len(frame),
              "training_seconds": sum(row["training_seconds"] for row in records)}
    write_json(directory / "trajectory.json", result)
    return result


def selected_recipes(out, registration):
    selection = read_json(Path(out) / "hpo" / "selection.json")
    if selection["registration_signature"] != registration["registration_signature"]:
        raise ValueError("HPO selection belongs to another study")
    verify_mapping(out, selection["artifact_sha256"])
    return selection


def select_candidates(rows, candidates, tolerance=1e-12):
    frame = pd.DataFrame(rows)
    if frame.duplicated(["model", "candidate_id", "trajectory_seed"]).any():
        raise ValueError("Duplicate tuning run")
    summaries, recipes = [], {}
    for candidate in candidates:
        group = frame[(frame.model == candidate["model"]) & (frame.candidate_id == candidate["candidate_id"])]
        if len(group) != 2 or not np.isfinite(group.rmse.to_numpy()).all():
            raise ValueError("Every candidate needs its two registered finite tuning runs")
        summaries.append({**candidate, "mean_rmse": float(group.rmse.mean()),
                          "sample_sd": float(group.rmse.std(ddof=1)), "parameters": int(group.parameters.iloc[0])})
    for family in FAMILIES:
        group = [row for row in summaries if row["model"] == family]
        minimum = min(row["mean_rmse"] for row in group)
        chosen = min((row for row in group if row["mean_rmse"] <= minimum + tolerance),
                     key=lambda row: (row["parameters"], row["candidate_id"]))
        recipes[family] = chosen
    minimum = min(row["mean_rmse"] for row in recipes.values())
    baseline = min((row for row in recipes.values() if row["mean_rmse"] <= minimum + tolerance),
                   key=lambda row: (row["parameters"], row["candidate_id"], row["model"]))["model"]
    return recipes, baseline, summaries


def execute_pool(root, out, registration, jobs, function):
    rows = []
    with ProcessPoolExecutor(max_workers=registration["workers"], mp_context=mp.get_context("spawn"),
            initializer=worker_init, initargs=(str(root), str(out), registration)) as pool:
        for future in as_completed([pool.submit(function, job) for job in jobs]):
            record = future.result()
            rows.append(record)
            print(json.dumps({"completed": len(rows), "total": len(jobs), "model": record["model"],
                              "seed": record.get("trajectory_seed", record.get("seed")),
                              "candidate_id": record.get("candidate_id")}), flush=True)
    return rows


def run_hpo(root, out, registration):
    if (out / "hpo" / "selection.json").exists():
        return selected_recipes(out, registration)
    candidates = candidate_grid(registration)
    jobs = [{**candidate, "trajectory_seed": int(seed), "seed": int(seed), "phase": "hpo"}
            for candidate in candidates for seed in registration["settings"]["hpo"]["seeds"]]
    rows = execute_pool(root, out, registration, jobs, hpo_job)
    recipes, baseline, summaries = select_candidates(rows, candidates, registration["settings"]["tie_tolerance"])
    hpo = out / "hpo"
    pd.DataFrame(rows).sort_values(["model", "candidate_id", "trajectory_seed"]).to_csv(hpo / "run_metrics.csv", index=False)
    pd.DataFrame(summaries).to_csv(hpo / "candidate_summary.csv", index=False)
    hashes = {p.relative_to(out).as_posix(): digest(p) for p in hpo.rglob("*") if p.is_file()}
    selection = {"selected_at_utc": stamp(), "registration_signature": registration["registration_signature"],
        "recipes": recipes, "baseline_model": baseline, "candidate_evaluations_per_family": 8,
        "tuning_runs_per_family": 16, "monthly_segments_per_family": 160,
        "new_fits_per_family": 320, "reserve_price_payloads_read": 0, "artifact_sha256": hashes}
    write_json(hpo / "selection.json", selection, exclusive=True)
    return selection


def fixed_job(job):
    state = _WORKER
    if job["phase"] == "pilot":
        split = split_at(state["data"], "2024-11-01")
        calendar = state["data"]["calendar"]
        evaluation = np.flatnonzero((calendar >= pd.Timestamp("2024-11-01", tz="UTC")) &
                                   (calendar <= pd.Timestamp("2024-12-30", tz="UTC")))
        if len(evaluation) != 60:
            raise ValueError("The pilot must cover60 fixed-model outcome days")
    else:
        split, evaluation = split_at(state["data"], "2025-01-01"), None
    directory = state["out"] / job["phase"] / f"{job['model']}_s{job['trajectory_seed']}"
    return fit_block(job, split, directory, evaluation)


def run_pilot(root, out, registration):
    selection = selected_recipes(out, registration)
    if (out / "pilot" / "complete.json").exists():
        return validate_pilot(out, registration)
    jobs = [{**recipe, "phase": "pilot", "trajectory_seed": int(seed), "seed": int(seed)}
            for recipe in selection["recipes"].values() for seed in registration["settings"]["pilot"]["seeds"]]
    rows = execute_pool(root, out, registration, jobs, fixed_job)
    pd.DataFrame(rows).sort_values(["model", "trajectory_seed"]).to_csv(out / "pilot" / "metrics.csv", index=False)
    hashes = {p.relative_to(out).as_posix(): digest(p) for p in (out / "pilot").rglob("*") if p.is_file()}
    record = {"completed_at_utc": stamp(), "registration_signature": registration["registration_signature"],
        "hpo_selection_sha256": digest(out / "hpo" / "selection.json"), "runs": len(rows),
        "fits": 2 * len(rows), "reserve_price_payloads_read": 0, "artifact_sha256": hashes}
    write_json(out / "pilot" / "complete.json", record, exclusive=True)
    return record


def validate_pilot(out, registration):
    record = read_json(out / "pilot" / "complete.json")
    if (record["registration_signature"] != registration["registration_signature"]
            or record["hpo_selection_sha256"] != digest(out / "hpo" / "selection.json")
            or record["runs"] != len(registration["settings"]["pilot"]["seeds"]) * len(FAMILIES)):
        raise ValueError("Pilot completion does not match the originally frozen selected recipes")
    verify_mapping(out, record["artifact_sha256"])
    return record


def run_plan(root, out, registration):
    sys.path.insert(0, str(root / "scripts"))
    from planned_study_statistics import plan_sample_size
    selection = selected_recipes(out, registration)
    validate_pilot(out, registration)
    if (out / "plan.json").exists():
        return validate_plan(out, registration)
    metrics = pd.read_csv(out / "pilot" / "metrics.csv")
    result = plan_sample_size(metrics, registration["design"])
    result.update({"planned_at_utc": stamp(), "registration_signature": registration["registration_signature"],
        "pilot_complete_sha256": digest(out / "pilot" / "complete.json"),
        "hpo_selection_sha256": digest(out / "hpo" / "selection.json"),
        "baseline_model": selection["baseline_model"], "reserve_price_payloads_read": 0})
    write_json(out / "plan.json", result, exclusive=True)
    return validate_plan(out, registration)


def validate_plan(out, registration):
    from planned_study_statistics import plan_sample_size
    validate_pilot(out, registration)
    plan = read_json(out / "plan.json")
    if (plan["registration_signature"] != registration["registration_signature"]
            or plan["pilot_complete_sha256"] != digest(out / "pilot" / "complete.json")
            or plan["hpo_selection_sha256"] != digest(out / "hpo" / "selection.json")):
        raise ValueError("The dimensional plan or its signed source changed")
    if (len(plan["seeds"]) != int(plan["R"]) or len(set(plan["seeds"])) != len(plan["seeds"])
            or set(plan["seeds"]) & (set(registration["settings"]["hpo"]["seeds"]) | set(registration["settings"]["pilot"]["seeds"]))):
        raise ValueError("The definitive seeds must be fresh, complete and unique")
    recalculated = plan_sample_size(pd.read_csv(out / "pilot" / "metrics.csv"), registration["design"])
    if int(plan["R"]) != int(recalculated["R"]) or plan["seeds"] != recalculated["seeds"]:
        raise ValueError("The frozen sample size differs from the full pilot-derived calculation")
    return plan


def fit_final(root, out, registration):
    if (out / "score" / "raw_access_started.json").exists():
        raise RuntimeError("Final fitting is closed after reserve price access starts")
    selection = selected_recipes(out, registration)
    plan = validate_plan(out, registration)
    if (out / "final_models_manifest.json").exists():
        return validate_final(out, registration)
    jobs = [{**recipe, "phase": "final", "trajectory_seed": int(seed), "seed": int(seed)}
            for recipe in selection["recipes"].values() for seed in plan["seeds"]]
    rows = execute_pool(root, out, registration, jobs, fixed_job)
    pd.DataFrame(rows).sort_values(["model", "trajectory_seed"]).to_csv(out / "final" / "metrics.csv", index=False)
    hashes = {p.relative_to(out).as_posix(): digest(p) for p in (out / "final").rglob("*") if p.is_file()}
    manifest = {"all_weights_frozen_at_utc": stamp(), "registration_signature": registration["registration_signature"],
        "plan_sha256": digest(out / "plan.json"), "hpo_selection_sha256": digest(out / "hpo" / "selection.json"),
        "models": len(rows), "R": plan["R"], "jobs": rows,
        "reserve_price_payloads_read": 0, "artifact_sha256": hashes}
    write_json(out / "final_models_manifest.json", manifest, exclusive=True)
    return validate_final(out, registration)


def validate_final(out, registration):
    plan = validate_plan(out, registration)
    manifest = read_json(out / "final_models_manifest.json")
    if (manifest["registration_signature"] != registration["registration_signature"]
            or manifest["plan_sha256"] != digest(out / "plan.json")
            or manifest["hpo_selection_sha256"] != digest(out / "hpo" / "selection.json")):
        raise ValueError("Final weights do not belong to the closed plan")
    expected = {(family, int(seed)) for family in FAMILIES for seed in plan["seeds"]}
    actual = {(job["model"], int(job["trajectory_seed"])) for job in manifest["jobs"]}
    if expected != actual or len(manifest["jobs"]) != len(expected) or manifest["models"] != len(expected):
        raise ValueError("Every planned model and fresh seed must be frozen before access")
    verify_mapping(out, manifest["artifact_sha256"])
    return manifest


def score(root, out, registration, raw_root):
    manifest = validate_final(out, registration)
    if (out / "score" / "complete.json").exists():
        complete = read_json(out / "score" / "complete.json")
        verify_mapping(out, complete["artifact_sha256"])
        return complete
    sys.path.insert(0, str(root / "scripts"))
    import torch
    from planned_study_data import load_reserved_data
    from dl_models_gru_lstm_tcn import build_model
    from final_fit_filipe_multihistory import apply_frozen_test_preprocessing
    from train_dl_benchmark import metrics
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    score_dir = out / "score"
    score_dir.mkdir(parents=True, exist_ok=True)
    write_json(score_dir / "frozen_access_gate.json", {"checked_at_utc": stamp(),
        "registration_sha256": digest(out / "registration.json"),
        "plan_sha256": digest(out / "plan.json"),
        "final_models_manifest_sha256": digest(out / "final_models_manifest.json"),
        "all_final_models_verified": len(manifest["jobs"])})
    data, audit = load_reserved_data(root, raw_root, registration["reservation"], score_dir)
    indices = np.asarray(data["test_indices"], dtype=int)
    expected_dates = pd.date_range("2026-09-01", "2026-10-30", tz="UTC").strftime("%Y-%m-%d").tolist()
    if data["calendar"][indices].strftime("%Y-%m-%d").tolist() != expected_dates:
        raise ValueError("The holdout must preserve all60 predetermined dates")
    frames, rows = [], []
    for job in sorted(manifest["jobs"], key=lambda row: (row["model"], row["trajectory_seed"])):
        name = f"{job['model']}_s{job['trajectory_seed']}"
        directory = score_dir / "runs" / name
        existing = cached_completion(directory, registration["registration_signature"])
        if existing is None:
            prepare_directory(directory)
            model_dir = out / "final" / name
            checkpoint = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
            pre = read_json(model_dir / "preprocessing.json")["refit"]
            model = build_model(job["model"], 7, job["width"], job["dropout"])
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            x = apply_frozen_test_preprocessing(data, 7, indices, pre)
            scaled, seconds = infer_once(model, x)
            prediction = scaled.astype(float) * pre["y_scale"]
            frame = pd.DataFrame({"model": job["model"], "seed": job["trajectory_seed"],
                "outcome_day": expected_dates, "y_true": data["targets"][indices], "y_pred": prediction,
                "persistence_pred": data["persistence"][indices], "mase_scale": job["mase_scale"]})
            frame.to_csv(directory / "predictions.csv", index=False)
            existing = {"model": job["model"], "seed": job["trajectory_seed"], "inference_seconds": seconds,
                "n_dates": len(indices), "model_sha256": digest(model_dir / "model.pt"),
                **metrics(frame.y_true, frame.y_pred, job["mase_scale"])}
            write_json(directory / "completion.json", {"completed_at_utc": stamp(),
                "registration_signature": registration["registration_signature"], "record": existing,
                "artifact_sha256": {"predictions.csv": digest(directory / "predictions.csv")}})
        frames.append(pd.read_csv(directory / "predictions.csv"))
        rows.append(existing)
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(score_dir / "predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(score_dir / "run_metrics.csv", index=False)
    write_json(score_dir / "data_audit.json", audit)
    hashes = {p.relative_to(out).as_posix(): digest(p) for p in score_dir.rglob("*") if p.is_file()}
    complete = {"completed_at_utc": stamp(), "registration_signature": registration["registration_signature"],
        "frozen_models_manifest_sha256": digest(out / "final_models_manifest.json"),
        "trajectories": len(rows), "prediction_rows": len(predictions), "dates": len(indices),
        "scoring_passes_per_checkpoint": 1, "artifact_sha256": hashes}
    write_json(score_dir / "complete.json", complete, exclusive=True)
    return complete


def analyze(root, out, registration):
    sys.path.insert(0, str(root / "scripts"))
    from planned_study_statistics import analyze_predictions
    complete = read_json(out / "score" / "complete.json")
    verify_mapping(out, complete["artifact_sha256"])
    plan = validate_plan(out, registration)
    selection = selected_recipes(out, registration)
    design = {**registration["design"], "baseline_model": selection["baseline_model"]}
    predictions = pd.read_csv(out / "score" / "predictions.csv")
    result = analyze_predictions(predictions, design, plan)
    destination = out / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    for key, value in result.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(destination / (key + ".csv"), index=False)
        else:
            write_json(destination / (key + ".json"), value)
    hashes = {p.relative_to(out).as_posix(): digest(p) for p in destination.iterdir() if p.is_file() and p.name != "manifest.json"}
    manifest = {"analyzed_at_utc": stamp(), "registration_signature": registration["registration_signature"],
        "score_complete_sha256": digest(out / "score" / "complete.json"), "artifact_sha256": hashes}
    write_json(destination / "manifest.json", manifest)
    return manifest


def verify(root, out, registration):
    checked = []
    if (out / "hpo" / "selection.json").exists():
        selected_recipes(out, registration); checked.append("hpo")
    if (out / "pilot" / "complete.json").exists():
        validate_pilot(out, registration); checked.append("pilot")
    if (out / "plan.json").exists():
        validate_plan(out, registration); checked.append("plan")
    if (out / "final_models_manifest.json").exists():
        validate_final(out, registration); checked.append("final_weights")
    for name in ("score/complete.json", "analysis/manifest.json"):
        if (out / name).exists():
            verify_mapping(out, read_json(out / name)["artifact_sha256"]); checked.append(name)
    return {"status": "signed_artifacts_verified", "phases": checked,
            "training_performed": False, "reserve_prices_read_by_verification": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("register", "hpo", "pilot", "plan", "fit", "score", "analyze", "verify"), required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    root, out = args.project_root.resolve(), args.output_dir.resolve()
    if args.phase == "register":
        result = register(root, out, args.workers, args.raw_root)
    else:
        registration = load_registration(root, out)
        functions = {"hpo": run_hpo, "pilot": run_pilot, "plan": run_plan, "fit": fit_final,
                     "score": lambda r, o, reg: score(r, o, reg, args.raw_root),
                     "analyze": analyze, "verify": verify}
        result = functions[args.phase](root, out, registration)
    print(json.dumps(clean(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
