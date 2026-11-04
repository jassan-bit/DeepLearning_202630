"""Extend the registered Filipe monthly CV to D14/D21/D28, development only.

The target remains the population SD of seven daily log returns. D7 is reused
from its original signed run rather than silently retrained. Every completed
monthly fit is independently hash-checked so an interrupted trajectory resumes
without replacing any completed fit. No raw archive or reserved-test file is
read by this runner.
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
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import torch

import train_dl_benchmark as fitting
from dl_data import load_development, mase_scale
from dl_data_v2 import fit_relative_preprocessing, transform_relative_windows
from dl_models import count_parameters, architecture_spec


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/filipe_multihistory_20261104"
PROTOCOL = ROOT / "experiments/filipe_multihistory_20261104/protocol.json"
NEW_HISTORIES = (14, 21, 28)
_DATA = _PLAN = _FOLDS = _CONTEXT = None


def write(path, value):
    fitting.write_json(path, value)


def validate_fold(data, history, fold):
    """Verify chronology, inner/outer purges, eligibility and exact inputs."""
    train, inner, valid, evaluation = [np.asarray(fold[key], dtype=np.int64) for key in
        ("train_indices", "inner_train_indices", "inner_validation_indices", "test_indices")]
    origin = int(data["calendar"].get_loc(pd.Timestamp(fold["fit_origin_day_utc"], tz="UTC")))
    if (any(not len(values) for values in (train, inner, valid, evaluation))
            or max(train) > origin - history - 1
            or not np.isin(inner, train).all() or not np.isin(valid, train).all()
            or max(inner) > min(valid) - history - 1
            or min(evaluation) != origin
            or len(np.intersect1d(train, evaluation))
            or len(np.intersect1d(inner, valid))):
        raise ValueError("Invalid past-only fold: " + fold["fold_id"])
    for values in (train, inner, valid, evaluation):
        if (len(np.unique(values)) != len(values)
                or not np.all(np.diff(values) > 0)
                or not data["eligible_by_history"][history][values].all()):
            raise ValueError("Unordered, duplicate or ineligible fold outcomes")
    expected = evaluation[:, None] - np.arange(history, 0, -1)
    if not np.array_equal(expected, np.asarray(fold["test_input_indices"])):
        raise ValueError("Library validation inputs differ from model inputs")
    if (fold["outer_training_last_allowed_index"] != origin-history-1
            or len(valid) != 90-history):
        raise ValueError("Registered embargo or recent inner-calendar policy changed")
    return evaluation


def register():
    plan = json.loads(PROTOCOL.read_text())
    if (not plan["histories"] or len(set(plan["histories"])) != len(plan["histories"])
            or any(history not in NEW_HISTORIES for history in plan["histories"])
            or plan.get("variant", "relative_window") != "relative_window"):
        raise ValueError("Only registered D14/D21/D28 relative-window extensions may be fitted")
    for name, expected in plan["sources"].items():
        if fitting.digest(ROOT / name) != expected:
            raise ValueError("Registered input changed: " + name)
    data = load_development(ROOT)
    manifests = {}
    for history in plan["histories"]:
        manifest = json.loads((OUT / f"D{history}" / "folds.json").read_text())
        if (manifest["history_days"] != history or manifest["target_returns_days"] != 7
                or manifest["forecast_days"] != 1 or manifest["monthly_fold_count"] != 12
                or manifest["library"]["version"] != "0.1.5"):
            raise ValueError("Wrong history, target, horizon or library manifest")
        covered = [outcome for fold in manifest["folds"]
                   for outcome in validate_fold(data, history, fold).tolist()]
        expected = np.flatnonzero(data["calendar"].year == 2024).tolist()
        if covered != expected or len(covered) != len(set(covered)):
            raise ValueError("Exactly one daily evaluation per 2024 calendar day is required")
        manifests[history] = manifest
    hashes = {name: fitting.digest(ROOT / name) for name in plan["training_code"]}
    observed = fitting.runtime()
    observed["timing_scope"] = "Wall time under registered concurrent CPU workers; no serial latency claim"
    observed["forward_cv_library"] = next(iter(manifests.values()))["library"]
    signature = hashlib.sha256(json.dumps({"protocol": fitting.digest(PROTOCOL),
        "code": hashes, "versions": observed["versions"]}, sort_keys=True).encode()).hexdigest()
    context = {"registered_before_first_extension_fit_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "protocol_sha256": fitting.digest(PROTOCOL), "code_sha256": hashes,
        "code_signature": signature, "runtime": observed, "test_files_opened": 0}
    path = OUT / "code_manifest_before_fitting.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["code_signature"] != signature:
            raise RuntimeError("Code/protocol changed after registration; preserve this experiment")
        context = previous
    else:
        write(path, context)
    write(OUT / "runtime.json", context["runtime"])
    (OUT / "protocol.json").write_bytes(PROTOCOL.read_bytes())
    return plan, manifests, context


def init_worker(plan, manifests, context):
    global _DATA, _PLAN, _FOLDS, _CONTEXT
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    _DATA, _PLAN, _FOLDS, _CONTEXT = load_development(ROOT), plan, manifests, context
    fitting._PROTOCOL = {"training": plan["training"]}


def verify_completion(directory, signature, completion_name):
    """Return a completed record only after every registered artifact matches."""
    path = Path(directory) / completion_name
    if not path.exists():
        return None
    completion = json.loads(path.read_text())
    if completion["code_signature"] != signature:
        raise RuntimeError("Cached fit has incompatible code signature")
    for name, expected in completion["artifact_sha256"].items():
        artifact = Path(directory) / name
        if not artifact.is_file() or fitting.digest(artifact) != expected:
            raise RuntimeError("Completed fit artifact changed: " + str(artifact))
    return completion["record"]


def _archive_incomplete(directory):
    if directory.exists() and any(directory.iterdir()):
        archive = OUT / "incomplete_preserved"
        archive.mkdir(parents=True, exist_ok=True)
        directory.rename(archive / (directory.parent.name + "_" + directory.name + "_" + str(time.time_ns())))
    directory.mkdir(parents=True, exist_ok=True)


def run_month(job, fold, ordinal, directory):
    data, plan, history = _DATA, _PLAN, job["history_days"]
    cached = verify_completion(directory, _CONTEXT["code_signature"], "completion.json")
    if cached is not None:
        return cached
    _archive_incomplete(directory)
    train, inner, valid, evaluation = [np.asarray(fold[key], dtype=np.int64) for key in
        ("train_indices", "inner_train_indices", "inner_validation_indices", "test_indices")]
    stage_job = {**job, "seed": job["seed"] + ordinal * 100000}
    inner_pre = fit_relative_preprocessing(data, history, inner)
    full_pre = fit_relative_preprocessing(data, history, train)
    xi, yi = transform_relative_windows(data, history, inner, inner_pre)
    xv, yv = transform_relative_windows(data, history, valid, inner_pre)
    _, best_epoch, inner_curve, inner_seconds = fitting.optimization_fit(
        stage_job, xi, yi, xv, yv, stage="inner")
    xt, yt = transform_relative_windows(data, history, train, full_pre)
    model, _, refit_curve, refit_seconds = fitting.optimization_fit(
        stage_job, xt, yt, epochs=best_epoch, stage="refit")
    xe, _ = transform_relative_windows(data, history, evaluation, full_pre)
    scaled, infer_seconds, infer_sd = fitting.infer(model, xe)
    predictions = scaled.astype(np.float64) * full_pre["y_scale"]
    denominator = mase_scale(data, train)
    frame = pd.DataFrame({"variant": "relative_window", "run_id": directory.parent.name,
        "fold_id": fold["fold_id"], "model": job["model"], "seed": job["seed"],
        "fit_seed": stage_job["seed"], "history_days": history, "horizon": 1,
        "outcome_day": data["calendar"][evaluation].strftime("%Y-%m-%d"),
        "prediction_origin_utc": [v.isoformat() for v in data["calendar"][evaluation]],
        "target_available_at_utc": [(v+pd.Timedelta(days=1)).isoformat() for v in data["calendar"][evaluation]],
        "input_first_day": data["calendar"][evaluation-history].strftime("%Y-%m-%d"),
        "input_last_day": data["calendar"][evaluation-1].strftime("%Y-%m-%d"),
        "y_true": data["targets"][evaluation], "y_pred": predictions,
        "persistence_pred": data["persistence"][evaluation], "mase_train_scale": denominator})
    frame.to_csv(directory / "predictions.csv", index=False)
    for values, name, scale in ((inner_curve, "history_inner.csv", inner_pre["y_scale"]),
                                (refit_curve, "history_refit.csv", full_pre["y_scale"])):
        curve = pd.DataFrame(values)
        curve["train_rmse_original"] = np.sqrt(curve.train_mse_scaled_eval) * scale
        curve["validation_rmse_original"] = np.sqrt(curve.validation_mse_scaled_eval) * scale
        curve.to_csv(directory / name, index=False)
    write(directory / "preprocessing.json", {"inner": inner_pre, "refit": full_pre})
    write(directory / "splits.json", fold)
    write(directory / "configuration.json", {**stage_job, "trajectory_seed": job["seed"],
        "inner_initialization_seed": stage_job["seed"],
        "refit_initialization_seed": stage_job["seed"] + 10000000,
        "training": plan["training"], "architecture": architecture_spec(model),
        "best_epoch": best_epoch, "code_signature": _CONTEXT["code_signature"],
        "fold_manifest_sha256": fitting.digest(OUT / f"D{history}" / "folds.json"),
        "test_files_opened": 0})
    torch.save({"state_dict": model.state_dict(), "family": job["model"], "history_days": history,
                "width": job["width"], "dropout": plan["training"]["dropout"],
                "variant": "relative_window", "y_scale": full_pre["y_scale"]}, directory / "model.pt")
    record = {"variant": "relative_window", "model": job["model"], "seed": job["seed"],
        "fold_id": fold["fold_id"], "history_days": history, "horizon": 1, "best_epoch": best_epoch,
        "train_samples": len(train), "inner_train_samples": len(inner),
        "inner_validation_samples": len(valid), "evaluation_samples": len(evaluation),
        "training_seconds": inner_seconds + refit_seconds, "inference_seconds": infer_seconds,
        "inference_seconds_sd": infer_sd, "parameters": count_parameters(model)["trainable"],
        **fitting.metrics(frame.y_true, frame.y_pred, denominator)}
    write(directory / "summary.json", record)
    hashes = {p.name: fitting.digest(p) for p in directory.iterdir() if p.is_file()}
    write(directory / "completion.json", {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_signature": _CONTEXT["code_signature"], "record": record, "artifact_sha256": hashes})
    return record


def run_trajectory(job):
    history = job["history_days"]
    run_id = f"relative_window_{job['model']}_D{history}_s{job['seed']}_{_CONTEXT['code_signature'][:10]}"
    directory = OUT / "runs" / run_id
    cached = verify_completion(directory, _CONTEXT["code_signature"], "summary.json")
    if cached is not None:
        return cached
    directory.mkdir(parents=True, exist_ok=True)
    records, frames = [], []
    for ordinal, fold in enumerate(_FOLDS[history]["folds"]):
        part = directory / fold["fold_id"]
        records.append(run_month(job, fold, ordinal, part))
        frames.append(pd.read_csv(part / "predictions.csv"))
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(directory / "predictions.csv", index=False)
    pd.DataFrame(records).to_csv(directory / "fold_metrics.csv", index=False)
    values = fitting.metrics(predictions.y_true, predictions.y_pred, float("nan"))
    scales = predictions.mase_train_scale.to_numpy()
    valid = np.isfinite(scales) & (scales > 0)
    errors = np.abs(predictions.y_true.to_numpy() - predictions.y_pred.to_numpy())
    values["mase"] = float(np.mean(errors[valid] / scales[valid]))
    values["mase_coverage"] = float(valid.mean())
    record = {**job, "run_id": run_id, "phase": "2024_cv", "n_dates": len(predictions),
        "folds_fitted": len(records), "model_fits": 2*len(records),
        "parameters": records[0]["parameters"],
        "training_seconds": sum(r["training_seconds"] for r in records),
        "inference_seconds": sum(r["inference_seconds"] for r in records), **values}
    hashes = {str(p.relative_to(directory)): fitting.digest(p) for p in directory.rglob("*")
              if p.is_file() and p != directory / "summary.json"}
    write(directory / "summary.json", {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_signature": _CONTEXT["code_signature"], "record": record, "artifact_sha256": hashes})
    return record


def aggregate(plan, context):
    records, predictions, folds = [], [], []
    for path in sorted((OUT / "runs").glob("*/summary.json")):
        record = verify_completion(path.parent, context["code_signature"], "summary.json")
        records.append(record)
        predictions.append(pd.read_csv(path.parent / "predictions.csv"))
        folds.append(pd.read_csv(path.parent / "fold_metrics.csv"))
    frame = pd.DataFrame(records).sort_values(["history_days", "model", "seed"])
    frame.to_csv(OUT / "runs.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(OUT / "predictions.csv", index=False)
    pd.concat(folds, ignore_index=True).to_csv(OUT / "fold_metrics.csv", index=False)
    expected = len(plan["histories"]) * len(plan["models"]) * len(plan["seeds"])
    write(OUT / "execution_summary.json", {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "trajectories": len(frame), "planned_new_trajectories": expected,
        "all_registered_trajectories_complete": len(frame) == expected,
        "fold_model_segments": int(frame.folds_fitted.sum()),
        "completed_project_fits": int(frame.model_fits.sum()), "actual_cv_folds": 12,
        "prediction_dates_per_trajectory": 366, "seeds_per_history_model": len(plan["seeds"]),
        "histories": plan["histories"], "variants": ["relative_window"], "test_files_opened": 0,
        "D7_reuse": "Original signed relative_window results are not retrained or overwritten",
        "scope": "Development2024 forward validation, no reserved-test confirmation"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("validate", "run"), default="validate")
    parser.add_argument("--histories", nargs="+", type=int, choices=NEW_HISTORIES)
    args = parser.parse_args()
    plan, manifests, context = register()
    histories = args.histories or plan["histories"]
    if any(history not in plan["histories"] for history in histories) or len(histories) != len(set(histories)):
        raise ValueError("Requested histories must be a unique subset of the frozen protocol")
    if args.phase == "validate":
        print(json.dumps({"status": "validated", "histories": histories,
            "actual_cv_folds_per_history": 12, "evaluation_dates_per_trajectory": 366,
            "test_files_opened": 0, "code_signature": context["code_signature"]}))
        return
    jobs = [{"variant": "relative_window", "model": model, "seed": seed, "history_days": history,
             "width": config["width"], "learning_rate": config["learning_rate"]}
            for history in histories for model, config in plan["models"].items() for seed in plan["seeds"]]
    with ProcessPoolExecutor(max_workers=plan["workers"], mp_context=mp.get_context("spawn"),
            initializer=init_worker, initargs=(plan, manifests, context)) as executor:
        futures = {executor.submit(run_trajectory, job): job for job in jobs}
        for count, future in enumerate(as_completed(futures), 1):
            record = future.result()
            print(json.dumps({"completed_trajectories": count, "total": len(jobs),
                "history_days": record["history_days"], "model": record["model"],
                "seed": record["seed"], "rmse": record["rmse"], "r2": record["r2"]}), flush=True)
    aggregate(plan, context)


if __name__ == "__main__":
    main()
