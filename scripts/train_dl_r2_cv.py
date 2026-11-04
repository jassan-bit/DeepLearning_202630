"""Fit the three references on actual monthly blocks of Filipe's forward splits.

The dedicated timeseries-cv environment generates folds.json. This runner uses
those exact indices for every fit, scaler and prediction. Raw reserved test
archives are never read. Original V1 scripts/results remain unchanged.
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

import train_dl_benchmark as v1
from dl_data import load_development, fit_preprocessing, transform_windows, mase_scale
from dl_data_v2 import fit_relative_preprocessing, transform_relative_windows
from dl_models import count_parameters, architecture_spec

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/dl_r2_cv_20261104"
PROTOCOL = ROOT / "experiments/dl_r2_cv_20261104/protocol.json"
_DATA = _PLAN = _FOLDS = _CONTEXT = None


def write(path, value):
    v1.write_json(path, value)


def indices_digest(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def register():
    plan = json.loads(PROTOCOL.read_text())
    for name, expected in plan["sources"].items():
        if v1.digest(ROOT / name) != expected:
            raise ValueError("Registered input changed: " + name)
    fold_manifest = json.loads((OUT / "folds.json").read_text())
    folds = fold_manifest["folds"]
    if len(folds) != 12:
        raise ValueError("Twelve real monthly training/validation blocks required")
    data = load_development(ROOT)
    covered = []
    for f in folds:
        train, inner, valid, test = [np.asarray(f[k], dtype=int) for k in
            ["train_indices", "inner_train_indices", "inner_validation_indices", "test_indices"]]
        origin = int(data["calendar"].get_loc(pd.Timestamp(f["fit_origin_day_utc"], tz="UTC")))
        if (not len(train) or not len(inner) or not len(valid) or not len(test)
                or max(train) > origin - 8 or not np.isin(inner, train).all()
                or not np.isin(valid, train).all() or max(inner) > min(valid) - 8
                or min(test) != origin or len(np.intersect1d(train, test))):
            raise ValueError("Invalid past-only fold: " + f["fold_id"])
        for values in [train, inner, valid, test]:
            if not data["eligible_by_history"][7][values].all():
                raise ValueError("Ineligible window in fold")
        expected_inputs = test[:, None] - np.arange(7, 0, -1)
        if not np.array_equal(expected_inputs, np.asarray(f["test_input_indices"])):
            raise ValueError("Filipe validation windows differ from model inputs")
        covered.extend(test.tolist())
    expected = np.flatnonzero(data["calendar"].year == 2024)
    if sorted(covered) != expected.tolist() or len(covered) != len(set(covered)):
        raise ValueError("Exactly one prediction per calendar day of 2024 required")
    hashes = {name: v1.digest(ROOT / name) for name in plan["training_code"]}
    runtime = v1.runtime()
    runtime["timing_scope"] = "wall time under three concurrent CPU trajectory workers; not directly comparable to serial V1 cost"
    runtime["forward_cv_runtime"] = fold_manifest.get("library", {})
    signature = hashlib.sha256(json.dumps({"protocol": v1.digest(PROTOCOL),
        "code": hashes, "versions": runtime["versions"]}, sort_keys=True).encode()).hexdigest()
    context = {"registered_before_first_cv_fit_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "protocol_sha256": v1.digest(PROTOCOL), "code_sha256": hashes,
        "code_signature": signature, "runtime": runtime, "test_files_opened": 0}
    path = OUT / "code_manifest_before_fitting.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["code_signature"] != signature:
            raise RuntimeError("Preserve existing results and create a new version after code changes")
        context = previous
    else:
        write(path, context)
    write(OUT / "runtime.json", context["runtime"])
    (OUT / "protocol.json").write_bytes(PROTOCOL.read_bytes())
    return plan, folds, context


def init_worker(plan, folds, context):
    global _DATA, _PLAN, _FOLDS, _CONTEXT
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    _DATA, _PLAN, _FOLDS, _CONTEXT = load_development(ROOT), plan, folds, context
    v1._PROTOCOL = {"training": plan["training"]}


def run_trajectory(job):
    data, plan = _DATA, _PLAN
    family, variant, seed = job["model"], job["variant"], job["seed"]
    run_id = f"{variant}_{family}_s{seed}_{_CONTEXT['code_signature'][:10]}"
    directory = OUT / "runs" / run_id
    summary_path = directory / "summary.json"
    if summary_path.exists():
        cached = json.loads(summary_path.read_text())
        if cached["code_signature"] != _CONTEXT["code_signature"]:
            raise RuntimeError("Incompatible cached trajectory")
        for path, expected in cached["artifact_sha256"].items():
            if v1.digest(directory / path) != expected:
                raise RuntimeError("Cached artifact changed")
        return cached["record"]
    if directory.exists() and any(directory.iterdir()):
        directory.rename(directory.with_name(directory.name + "_incomplete_" + str(time.time_ns())))
    directory.mkdir(parents=True, exist_ok=True)
    fit_pre = fit_preprocessing if variant == "absolute_log" else fit_relative_preprocessing
    transform = transform_windows if variant == "absolute_log" else transform_relative_windows
    prediction_frames, records = [], []
    for ordinal, fold in enumerate(_FOLDS):
        part = directory / fold["fold_id"]
        part.mkdir()
        train, inner, valid, test = [np.asarray(fold[k], dtype=int) for k in
            ["train_indices", "inner_train_indices", "inner_validation_indices", "test_indices"]]
        stage_job = {**job, "seed": seed + ordinal * 100000}
        inner_pre, full_pre = fit_pre(data, 7, inner), fit_pre(data, 7, train)
        xi, yi = transform(data, 7, inner, inner_pre)
        xv, yv = transform(data, 7, valid, inner_pre)
        _, best_epoch, inner_curve, inner_seconds = v1.optimization_fit(
            stage_job, xi, yi, xv, yv, stage="inner")
        xt, yt = transform(data, 7, train, full_pre)
        model, _, refit_curve, refit_seconds = v1.optimization_fit(
            stage_job, xt, yt, epochs=best_epoch, stage="refit")
        xe, _ = transform(data, 7, test, full_pre)
        scaled, infer_seconds, infer_sd = v1.infer(model, xe)
        predictions = scaled.astype(np.float64) * full_pre["y_scale"]
        denominator = mase_scale(data, train)
        frame = pd.DataFrame({"variant": variant, "run_id": run_id, "fold_id": fold["fold_id"],
            "model": family, "seed": seed, "fit_seed": stage_job["seed"], "history_days": 7,
            "outcome_day": data["calendar"][test].strftime("%Y-%m-%d"),
            "prediction_origin_utc": [v.isoformat() for v in data["calendar"][test]],
            "target_available_at_utc": [(v+pd.Timedelta(days=1)).isoformat() for v in data["calendar"][test]],
            "input_first_day": data["calendar"][test-7].strftime("%Y-%m-%d"),
            "input_last_day": data["calendar"][test-1].strftime("%Y-%m-%d"),
            "y_true": data["targets"][test], "y_pred": predictions,
            "persistence_pred": data["persistence"][test], "mase_train_scale": denominator})
        frame.to_csv(part / "predictions.csv", index=False)
        for curve, name, scale in [(inner_curve, "history_inner.csv", inner_pre["y_scale"]),
                                    (refit_curve, "history_refit.csv", full_pre["y_scale"])]:
            curve = pd.DataFrame(curve)
            curve["train_rmse_original"] = np.sqrt(curve.train_mse_scaled_eval) * scale
            curve["validation_rmse_original"] = np.sqrt(curve.validation_mse_scaled_eval) * scale
            curve.to_csv(part / name, index=False)
        write(part / "preprocessing.json", {"inner": inner_pre, "refit": full_pre})
        write(part / "splits.json", fold)
        write(part / "configuration.json", {**stage_job, "trajectory_seed": seed,
            "inner_initialization_seed": stage_job["seed"],
            "refit_initialization_seed": stage_job["seed"] + 10000000,
            "training": plan["training"], "architecture": architecture_spec(model),
            "best_epoch": best_epoch, "code_signature": _CONTEXT["code_signature"],
            "fold_manifest_sha256": plan["sources"]["results/dl_r2_cv_20261104/folds.json"],
            "test_files_opened": 0})
        torch.save({"state_dict": model.state_dict(), "family": family, "history_days": 7,
                    "width": job["width"], "dropout": plan["training"]["dropout"],
                    "variant": variant, "y_scale": full_pre["y_scale"]}, part / "model.pt")
        record = {"variant": variant, "model": family, "seed": seed,
            "fold_id": fold["fold_id"], "history_days": 7, "best_epoch": best_epoch,
            "train_samples": len(train), "inner_train_samples": len(inner),
            "inner_validation_samples": len(valid), "evaluation_samples": len(test),
            "training_seconds": inner_seconds + refit_seconds,
            "inference_seconds": infer_seconds, "inference_seconds_sd": infer_sd,
            "parameters": count_parameters(model)["trainable"],
            **v1.metrics(frame.y_true, frame.y_pred, denominator)}
        write(part / "summary.json", record)
        prediction_frames.append(frame)
        records.append(record)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(directory / "predictions.csv", index=False)
    pd.DataFrame(records).to_csv(directory / "fold_metrics.csv", index=False)
    values = v1.metrics(predictions.y_true, predictions.y_pred, float("nan"))
    scales = predictions.mase_train_scale.to_numpy()
    allowed = np.isfinite(scales) & (scales > 0)
    values["mase"] = float(np.mean(np.abs(predictions.y_true.to_numpy()[allowed]-predictions.y_pred.to_numpy()[allowed])/scales[allowed]))
    values["mase_coverage"] = float(allowed.mean())
    record = {**job, "run_id": run_id, "phase": "2024_cv", "n_dates": len(predictions),
        "folds_fitted": len(records), "model_fits": 2*len(records),
        "parameters": records[0]["parameters"],
        "training_seconds": sum(r["training_seconds"] for r in records),
        "inference_seconds": sum(r["inference_seconds"] for r in records), **values}
    hashes = {str(p.relative_to(directory)): v1.digest(p) for p in directory.rglob("*") if p.is_file()}
    write(summary_path, {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_signature": _CONTEXT["code_signature"], "record": record, "artifact_sha256": hashes})
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["validate", "run"], default="validate")
    args = parser.parse_args()
    plan, folds, context = register()
    if args.phase == "validate":
        print(json.dumps({"status": "validated", "actual_cv_folds_planned": len(folds),
                          "n_days": 366, "test_files_opened": 0}))
        return
    jobs = [{"variant": variant, "model": model, "seed": seed, "history_days": 7,
             "width": config["width"], "learning_rate": config["learning_rate"]}
            for variant in plan["variants"] for model, config in plan["models"].items()
            for seed in plan["seeds"]]
    records = []
    with ProcessPoolExecutor(max_workers=plan["workers"], mp_context=mp.get_context("spawn"),
                             initializer=init_worker, initargs=(plan, folds, context)) as executor:
        futures = {executor.submit(run_trajectory, job): job for job in jobs}
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps({"completed_trajectories": len(records), "total": len(jobs),
                "variant": record["variant"], "model": record["model"],
                "seed": record["seed"], "rmse": record["rmse"], "r2": record["r2"]}), flush=True)
    frame = pd.DataFrame(records).sort_values(["variant", "model", "seed"])
    frame.to_csv(OUT / "runs.csv", index=False)
    frames, per_fold = [], []
    for row in frame.to_dict("records"):
        directory = OUT / "runs" / row["run_id"]
        frames.append(pd.read_csv(directory / "predictions.csv"))
        per_fold.append(pd.read_csv(directory / "fold_metrics.csv"))
    pd.concat(frames, ignore_index=True).to_csv(OUT / "predictions.csv", index=False)
    pd.concat(per_fold, ignore_index=True).to_csv(OUT / "fold_metrics.csv", index=False)
    write(OUT / "execution_summary.json", {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "trajectories": len(frame), "fold_model_segments": int(frame.folds_fitted.sum()),
        "completed_project_fits": int(frame.model_fits.sum()), "actual_cv_folds": 12,
        "prediction_dates_per_trajectory": 366, "seeds_per_variant_model": len(plan["seeds"]),
        "variants": plan["variants"], "test_files_opened": 0,
        "scope": "Exploratory forward validation on previously exposed development2024, not reserved-test confirmation"})


if __name__ == "__main__":
    main()
