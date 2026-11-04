"""Frozen development refits followed by one reserved-test evaluation per run.

``fit`` uses 2023--2024 derivatives only. ``score`` opens the reserved archives
only after every final model and configuration has been signed. Monthly test
groups are reporting periods, never additional training or model-selection folds.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from dl_data import load_development, mase_scale
from dl_data_v2 import fit_relative_preprocessing, transform_relative_windows, relative_features
from dl_models import architecture_spec, build_model, count_parameters
import train_dl_benchmark as trainer
from load_filipe_reserved_test import (ROOT, OUT, FREEZE, PROTOCOL, HISTORIES,
    digest, now, write_json, event, validate_freeze, validate_final_models, load_reserved_data)

_DATA = _PLAN = _FREEZE = _FREEZE_HASH = None


def run_name(job):
    return f"D{int(job['history_days'])}_{job['model']}_s{int(job['seed'])}"


def check_jobs(jobs):
    expected = {(d, m, s) for d in HISTORIES for m in ("MLP", "LSTM", "CNN1D") for s in range(4101, 4111)}
    actual = {(int(j["history_days"]), j["model"], int(j["seed"])) for j in jobs}
    if actual != expected or len(jobs) != 120:
        raise ValueError("Freeze must contain exactly the 120 predeclared model/history/seed jobs")
    for job in jobs:
        if isinstance(job["epochs"], bool) or int(job["epochs"]) != job["epochs"] or not 1 <= job["epochs"] <= 100:
            raise ValueError("Final epochs must be the closed development-selected integer in [1,100]")
        if job["width"] < 1 or not np.isfinite(job["learning_rate"]) or job["learning_rate"] <= 0:
            raise ValueError("Invalid frozen model hyperparameters")


def final_training_indices(data, history):
    """Retain development labels before the same D-day embargo used in CV."""
    calendar = pd.DatetimeIndex(data["calendar"])
    expected = pd.date_range("2023-01-01", "2025-01-01", inclusive="left", tz="UTC")
    if not calendar.equals(expected) or history not in HISTORIES:
        raise ValueError("Only the full pre-test development calendar is admitted")
    positions = np.arange(len(calendar))
    train = np.flatnonzero(np.asarray(data["eligible_by_history"][history], dtype=bool) &
                           (positions <= len(calendar) - history - 1))
    if not len(train) or np.max(train) + history >= len(calendar):
        raise ValueError("Final training labels violate the registered embargo")
    return train


def apply_frozen_test_preprocessing(data, history, indices, preproc):
    """Apply training parameters only; this intentionally does not fit a scaler."""
    indices = np.asarray(indices, dtype=int)
    if history != int(preproc["history_days"]) or not len(indices):
        raise ValueError("Nonempty test cohort and matching frozen history are required")
    calendar = pd.DatetimeIndex(data["calendar"])
    if (calendar[indices] < pd.Timestamp("2025-01-01", tz="UTC")).any():
        raise ValueError("The score cohort must belong to the reserved test")
    if not np.asarray(data["eligible_by_history"][history], dtype=bool)[indices].all():
        raise ValueError("An ineligible outcome entered test scoring")
    values = np.asarray(data["windows_by_history"][history])[indices]
    mean = np.asarray(preproc["x_mean"], dtype=float)
    scale = np.asarray(preproc["x_scale"], dtype=float)
    if mean.shape != (5,) or scale.shape != (5,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid frozen training scales")
    x = ((relative_features(values) - mean) / scale).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite frozen-preprocessed test input")
    return x


def init_worker(plan, freeze, freeze_hash):
    global _DATA, _PLAN, _FREEZE, _FREEZE_HASH
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    _DATA, _PLAN, _FREEZE, _FREEZE_HASH = load_development(ROOT), plan, freeze, freeze_hash
    trainer._PROTOCOL = {"training": plan["training"]}


def fit_job(job):
    directory = OUT / "final_fit" / run_name(job)
    completion = directory / "completion.json"
    if completion.exists():
        saved = json.loads(completion.read_text())
        if saved["freeze_sha256"] != _FREEZE_HASH:
            raise ValueError("Cached final model belongs to a different configuration freeze")
        for name, expected in saved["artifact_sha256"].items():
            if digest(directory / name) != expected:
                raise ValueError("A cached final-fit artifact changed: " + name)
        return saved["record"]
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError("Partial final model found; retain and audit it before resuming: " + str(directory))
    directory.mkdir(parents=True, exist_ok=True)
    data, history = _DATA, int(job["history_days"])
    train = final_training_indices(data, history)
    preproc = fit_relative_preprocessing(data, history, train)
    x, y = transform_relative_windows(data, history, train, preproc)
    started = now()
    model, epochs, curve, seconds = trainer.optimization_fit(job, x, y, epochs=int(job["epochs"]), stage="refit")
    model_path = directory / "model.pt"
    torch.save({"state_dict": model.state_dict(), "family": job["model"],
        "history_days": history, "width": int(job["width"]),
        "dropout": float(_PLAN["training"]["dropout"]), "freeze_sha256": _FREEZE_HASH}, model_path)
    pd.DataFrame(curve).to_csv(directory / "history_refit.csv", index=False)
    write_json(directory / "preprocessing.json", preproc)
    denominator = mase_scale(data, train)
    record = {**job, "run_id": run_name(job), "variant": "relative_window", "phase": "final_development_fit",
        "started_at_utc": started, "completed_at_utc": now(), "training_seconds": seconds,
        "train_samples": len(train), "train_first_day_utc": data["calendar"][train.min()].strftime("%Y-%m-%d"),
        "train_last_day_utc": data["calendar"][train.max()].strftime("%Y-%m-%d"),
        "embargo_days": history, "final_refit_epochs": epochs, "initialization_seed": int(job["seed"]) + 10_000_000,
        "parameters": count_parameters(model)["trainable"], "mase_train_scale": denominator,
        "freeze_sha256": _FREEZE_HASH, "test_files_opened": 0}
    write_json(directory / "configuration.json", {**record, "training": _PLAN["training"],
        "architecture": architecture_spec(model), "train_indices": train.tolist()})
    artifacts = {p.name: digest(p) for p in directory.iterdir() if p.is_file()}
    write_json(completion, {"freeze_sha256": _FREEZE_HASH, "record": record, "artifact_sha256": artifacts})
    return record


def fit_all():
    freeze = validate_freeze()
    check_jobs(freeze["final_jobs"])
    if (OUT / "test/raw_test_audit.json").exists() or (OUT / "test/raw_access_started.json").exists():
        raise RuntimeError("Reserved test has already been opened: final refitting is closed")
    plan = json.loads(PROTOCOL.read_text())
    freeze_hash = digest(FREEZE)
    event("final_fit_started_after_configuration_freeze", freeze_sha256=freeze_hash, jobs=120)
    records = []
    with ProcessPoolExecutor(max_workers=int(plan.get("workers", 3)), mp_context=mp.get_context("spawn"),
        initializer=init_worker, initargs=(plan, freeze, freeze_hash)) as executor:
        futures = [executor.submit(fit_job, job) for job in freeze["final_jobs"]]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps({"final_models_completed": len(records), "total": 120,
                "run_id": record["run_id"], "epochs": record["final_refit_epochs"]}), flush=True)
    records.sort(key=lambda r: (r["history_days"], r["model"], r["seed"]))
    pd.DataFrame(records).to_csv(OUT / "final_fit_summary.csv", index=False)
    artifacts = {str(p.relative_to(OUT)): digest(p) for p in (OUT / "final_fit").rglob("*") if p.is_file()}
    manifest = {"completed_at_utc": now(), "freeze_sha256": freeze_hash,
        "source_commit": freeze.get("source_commit"), "jobs": records, "artifact_sha256": artifacts,
        "completed_project_fits": len(records), "test_files_opened": 0,
        "policy": "Weights and training-only scales fixed before first reserved-test access; no refits or selection on test"}
    write_json(OUT / "final_models_manifest.json", manifest)
    event("all_final_models_frozen_before_test", manifest_sha256=digest(OUT / "final_models_manifest.json"), count=len(records))
    print(json.dumps({"status": "all_final_models_frozen", "count": len(records), "test_files_opened": 0}), flush=True)


def score_all(raw_root=None):
    freeze, models = validate_final_models()
    check_jobs(freeze["final_jobs"])
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    data, cohort, audit = load_reserved_data(raw_root=raw_root)
    indices = data["test_indices"]
    if not len(indices):
        raise ValueError("No complete common test cohort is available")
    records, frames, folds = [], [], []
    for job in models["jobs"]:
        name, history = run_name(job), int(job["history_days"])
        source = OUT / "final_fit" / name
        destination = OUT / "test/runs" / name
        completion = destination / "completion.json"
        if completion.exists():
            saved = json.loads(completion.read_text())
            if saved["freeze_sha256"] != digest(FREEZE):
                raise ValueError("A scored run belongs to a different freeze")
            for artifact, expected in saved["artifact_sha256"].items():
                if digest(destination / artifact) != expected:
                    raise ValueError("A saved test result changed: " + artifact)
            frame = pd.read_csv(destination / "predictions.csv")
            record = saved["record"]
            fold_frame = pd.read_csv(destination / "fold_metrics.csv")
        else:
            if destination.exists() and any(destination.iterdir()):
                raise RuntimeError("Partial test run requires an audit before repeating a checkpoint forward pass")
            destination.mkdir(parents=True, exist_ok=True)
            preproc = json.loads((source / "preprocessing.json").read_text())
            checkpoint = torch.load(source / "model.pt", map_location="cpu", weights_only=True)
            model = build_model(job["model"], history, int(job["width"]), checkpoint["dropout"])
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.eval()
            x = apply_frozen_test_preprocessing(data, history, indices, preproc)
            started = time.perf_counter()
            with torch.inference_mode():
                scaled_prediction = model(torch.from_numpy(x)).cpu().numpy()
            seconds = time.perf_counter() - started
            prediction = scaled_prediction.astype(float) * preproc["y_scale"]
            frame = pd.DataFrame({"variant": "relative_window", "run_id": name,
                "history_days": history, "model": job["model"], "seed": job["seed"],
                "fold_id": data["calendar"][indices].strftime("%Y-%m"),
                "outcome_day": data["calendar"][indices].strftime("%Y-%m-%d"),
                "prediction_origin_utc": [d.isoformat() for d in data["calendar"][indices]],
                "target_available_at_utc": [(d + pd.Timedelta(days=1)).isoformat() for d in data["calendar"][indices]],
                "input_first_day": data["calendar"][indices - history].strftime("%Y-%m-%d"),
                "input_last_day": data["calendar"][indices - 1].strftime("%Y-%m-%d"),
                "y_true": data["targets"][indices], "y_pred": prediction,
                "persistence_pred": data["persistence"][indices], "mase_train_scale": job["mase_train_scale"]})
            frame.to_csv(destination / "predictions.csv", index=False)
            record = {"variant": "relative_window", "run_id": name, "phase": "reserved_test",
                "history_days": history, "model": job["model"], "seed": job["seed"],
                "n_dates": len(frame), "parameters": job["parameters"],
                "training_seconds": job["training_seconds"], "inference_seconds": seconds,
                "checkpoint_forward_calls": 1, **trainer.metrics(frame.y_true, frame.y_pred, job["mase_train_scale"])}
            per_fold = [{"variant": "relative_window", "history_days": history, "model": job["model"],
                "seed": job["seed"], "fold_id": month, "n_dates": len(group),
                "model_update": "none; frozen final development model", **trainer.metrics(group.y_true, group.y_pred, job["mase_train_scale"])}
                for month, group in frame.groupby("fold_id", sort=True)]
            fold_frame = pd.DataFrame(per_fold)
            fold_frame.to_csv(destination / "fold_metrics.csv", index=False)
            artifacts = {p.name: digest(p) for p in destination.iterdir() if p.is_file()}
            write_json(completion, {"completed_at_utc": now(), "freeze_sha256": digest(FREEZE),
                "model_sha256": digest(source / "model.pt"), "cohort_sha256": digest(OUT / "test/cohort_manifest.json"),
                "record": record, "artifact_sha256": artifacts})
            event("test_run_scored_once", run_id=name, checkpoint_forward_calls=1, predictions=len(frame))
        frames.append(frame); records.append(record); folds.append(fold_frame)
        print(json.dumps({"test_runs_scored": len(records), "total": 120, "run_id": name}), flush=True)
    pd.concat(frames, ignore_index=True).to_csv(OUT / "test/predictions.csv", index=False)
    pd.DataFrame(records).to_csv(OUT / "test/runs.csv", index=False)
    pd.concat(folds, ignore_index=True).to_csv(OUT / "test/fold_metrics.csv", index=False)
    write_json(OUT / "test/execution_summary.json", {"completed_at_utc": now(),
        "freeze_sha256": digest(FREEZE), "final_models_manifest_sha256": digest(OUT / "final_models_manifest.json"),
        "model_runs": len(records), "histories": list(HISTORIES), "seeds": 10, "horizon_days": 1,
        "common_prediction_dates": len(indices), "predictions": sum(len(f) for f in frames),
        "calendar_days": 608, "monthly_reporting_blocks": 20, "test_period_model_refits": 0,
        "actual_checkpoint_forward_calls": sum(int(r["checkpoint_forward_calls"]) for r in records),
        "raw_archives_opened": audit["archives_opened"], "test_used_for_configuration_selection": False,
        "scope": "Reserved test evaluation after development configuration closure; monthly groups are reporting periods, not additional refit folds"})
    event("reserved_test_evaluation_complete", runs=len(records), common_dates=len(indices))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["validate", "fit", "score"], default="validate")
    parser.add_argument("--raw-root", type=Path)
    args = parser.parse_args()
    if args.phase == "validate":
        freeze = validate_freeze()
        check_jobs(freeze["final_jobs"])
        print(json.dumps({"status": "closed_configuration_validated", "jobs": 120, "test_files_opened": 0}))
    elif args.phase == "fit":
        fit_all()
    else:
        score_all(args.raw_root)


if __name__ == "__main__":
    main()
