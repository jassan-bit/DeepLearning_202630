"""Verify saved Filipe-CV artifacts without fitting or opening reserved raw data.

The only write is the explicitly selected JSON report (default: /tmp). Restore
the archived run directories before invoking this verifier in a fresh checkout.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from dl_data import load_development, fit_preprocessing, transform_windows, mase_scale
from dl_data_v2 import fit_relative_preprocessing, transform_relative_windows
from dl_models import build_model, count_parameters


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/dl_r2_cv_20261104"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_indices(indices) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


class Audit:
    def __init__(self):
        self.failures = []

    def require(self, condition, description):
        if not condition:
            self.failures.append(description)

    def close(self, actual, expected, description, *, rtol=1e-10, atol=1e-13):
        self.require(np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True), description)


def metrics(y, prediction, scales):
    """Independent metric formulas, including the fold-specific MASE scale."""
    y, prediction, scales = [np.asarray(v, dtype=float) for v in (y, prediction, scales)]
    errors = y - prediction
    squared = np.square(errors)
    total = float(np.square(y - y.mean()).sum())
    positive = y > 0
    valid_scale = np.isfinite(scales) & (scales > 0)
    return {
        "mse": float(squared.mean()),
        "rmse": float(np.sqrt(squared.mean())),
        "mae": float(np.abs(errors).mean()),
        "r2": 1. - float(squared.sum()) / total if total > 0 else np.nan,
        "mape": float(100 * np.abs(errors[positive] / y[positive]).mean()) if positive.any() else np.nan,
        "mape_coverage": float(positive.mean()),
        "mase": float(np.abs(errors[valid_scale] / scales[valid_scale]).mean()) if valid_scale.any() else np.nan,
        "mase_coverage": float(valid_scale.mean()),
    }


def verify() -> dict:
    audit = Audit()
    read = lambda name: json.loads((OUT / name).read_text())
    plan, manifest = read("protocol.json"), read("code_manifest_before_fitting.json")
    folds_record, execution = read("folds.json"), read("execution_summary.json")
    runtime, erratum = read("runtime.json"), read("protocol_erratum.json")
    protocol_path = ROOT / "experiments/dl_r2_cv_20261104/protocol.json"
    audit.require(digest(protocol_path) == digest(OUT / "protocol.json") == manifest["protocol_sha256"],
                  "Original and copied signed protocol digests")
    audit.require(manifest["source_commit"] == "1f0ed23aa40c8160f3a04ae66094aee037fb0128",
                  "Registered source is the pre-fit commit")
    audit.require(set(manifest["code_sha256"]) == set(plan["training_code"]), "Registered code file set")
    for name, expected in manifest["code_sha256"].items():
        audit.require(digest(ROOT / name) == expected, "Immutable training source: " + name)
    for name, expected in plan["sources"].items():
        audit.require(digest(ROOT / name) == expected, "Registered data/selection/fold source: " + name)
    audit.require(digest(ROOT / folds_record["generator"]) == folds_record["generator_sha256"],
                  "The recorded Filipe adapter generator remains unchanged")
    expected_signature = hashlib.sha256(json.dumps({
        "protocol": manifest["protocol_sha256"], "code": manifest["code_sha256"],
        "versions": runtime["versions"],
    }, sort_keys=True).encode()).hexdigest()
    audit.require(expected_signature == manifest["code_signature"], "Reconstructed training signature")
    audit.require(runtime == manifest["runtime"], "Runtime matches the pre-fit snapshot")
    audit.require(runtime["forward_cv_runtime"] == folds_record["library"], "Filipe runtime provenance")
    audit.require(folds_record["library"]["distribution"] == "timeseries-cv"
                  and folds_record["library"]["version"] == "0.1.5", "Recorded Filipe package and version")
    audit.require(folds_record["library"]["function"] == "tsxv.splitTrainVal.split_train_val_forwardChaining",
                  "Recorded native forward-chaining function")
    audit.require(runtime["device"] == "cpu" and not runtime["cuda_available"]
                  and runtime["deterministic_algorithms"] and runtime["torch_threads_per_worker"] == 1,
                  "Recorded deterministic CPU execution")
    audit.require(execution["test_files_opened"] == manifest["test_files_opened"]
                  == folds_record["test_files_opened"] == plan["test_files_opened_allowed"] == 0,
                  "All reserved-test counters are zero")
    audit.require(digest(ROOT / "data/manifests/test_raw_reserve_20250101_20260831.json")
                  == "d2eb4e84fff85ee957e6b02890cead3cec37fd593ec2a8db487dbe137d2d3a10",
                  "Reserved raw-test metadata identity remains unchanged")
    audit.require(erratum == json.loads((ROOT / "experiments/dl_r2_cv_20261104/protocol_erratum.json").read_text()),
                  "Erratum source and result copies agree")
    audit.require(erratum["protocol_sha256"] == manifest["protocol_sha256"]
                  and erratum["field"] == "training.refit"
                  and erratum["original"] == plan["training"]["refit"]
                  and erratum["changes_to_inputs_weights_optimization_or_results"] is False
                  and erratum["original_signed_protocol_preserved"] is True,
                  "Erratum corrects only the inherited explanatory refit string")

    data = load_development(ROOT)
    calendar = data["calendar"]
    audit.require(data["source_hashes"] == folds_record["source_hashes"], "Daily development source digests")
    folds = folds_record["folds"]
    audit.require([f["fold_id"] for f in folds] == [f"2024-{month:02d}" for month in range(1, 13)],
                  "Twelve ordered monthly forward blocks")
    expected_days = calendar[calendar.year == 2024].strftime("%Y-%m-%d").tolist()
    covered, expected_pre, scale_by_fold, arrays_by_fold = [], {}, {}, {}
    for fold in folds:
        fid = fold["fold_id"]
        train, inner, valid, test = [np.asarray(fold[k], dtype=np.int64) for k in
            ["train_indices", "inner_train_indices", "inner_validation_indices", "test_indices"]]
        origin = int(calendar.get_loc(pd.Timestamp(fold["fit_origin_day_utc"], tz="UTC")))
        audit.require(fold["fit_origin_index"] == origin and train.max() == origin - 8,
                      fid + ": outer embargo D=7")
        audit.require(valid.min() == origin - 90 and inner.max() == valid.min() - 8
                      and valid.max() == train.max(), fid + ": recent inner validation and inner embargo")
        audit.require(not np.intersect1d(inner, valid).size and not np.intersect1d(train, test).size
                      and np.isin(inner, train).all() and np.isin(valid, train).all(),
                      fid + ": chronological cohort separation")
        for key, values in [("train", train), ("inner_train", inner),
                            ("inner_validation", valid), ("test", test)]:
            audit.require(len(values) == len(np.unique(values)) and (values >= 0).all()
                          and (values < len(calendar)).all() and data["eligible_by_history"][7][values].all(),
                          fid + ": valid unique calendar indices " + key)
            detail = fold[key]
            audit.require(detail["count"] == len(values) and detail["indices_sha256"] == digest_indices(values)
                          and detail["first_day"] == calendar[values.min()].strftime("%Y-%m-%d")
                          and detail["last_day"] == calendar[values.max()].strftime("%Y-%m-%d"),
                          fid + ": recorded cohort metadata " + key)
        inputs = test[:, None] - np.arange(7, 0, -1)
        audit.require(np.array_equal(inputs, fold["test_input_indices"])
                      and digest_indices(inputs) == fold["test_input_indices_sha256"],
                      fid + ": exact Filipe evaluation input windows")
        dates = calendar[test]
        audit.require(test.min() == origin and (dates.year == 2024).all()
                      and (dates.month == int(fid[-2:])).all(), fid + ": monthly development cohort")
        audit.require(fold["daily_prediction_origins_utc"] == [v.isoformat() for v in dates]
                      and fold["daily_target_available_at_utc"] ==
                      [(v + pd.Timedelta(days=1)).isoformat() for v in dates],
                      fid + ": origins and outcome availability")
        covered.extend(dates.strftime("%Y-%m-%d").tolist())
        arrays_by_fold[fid] = (train, inner, valid, test)
        scale_by_fold[fid] = mase_scale(data, train)
        audit.require(np.isfinite(scale_by_fold[fid]) and scale_by_fold[fid] > 0,
                      fid + ": positive training-calendar MASE scale")
        for variant, fit in [("absolute_log", fit_preprocessing),
                             ("relative_window", fit_relative_preprocessing)]:
            expected_pre[(variant, fid)] = {"inner": fit(data, 7, inner), "refit": fit(data, 7, train)}
    audit.require(covered == expected_days and len(set(covered)) == 366, "All 366 development dates exactly once")

    runs = pd.read_csv(OUT / "runs.csv")
    top_predictions = pd.read_csv(OUT / "predictions.csv")
    top_fold_metrics = pd.read_csv(OUT / "fold_metrics.csv")
    audit.require(plan["variants"] == ["absolute_log", "relative_window"]
                  and plan["seeds"] == list(range(4101, 4111)), "Frozen variants and ten base seeds")
    expected_keys = {(v, m, s) for v in plan["variants"] for m in plan["models"] for s in plan["seeds"]}
    actual_keys = set(zip(runs.variant, runs.model, runs.seed))
    audit.require(len(runs) == 60 and runs.run_id.nunique() == 60 and actual_keys == expected_keys,
                  "Sixty unique paired-seed trajectories")
    directories = {p.name for p in (OUT / "runs").iterdir() if p.is_dir()}
    audit.require(directories == set(runs.run_id), "No missing or extra trajectory directories")
    audit.require(len(top_predictions) == 21960 and len(top_fold_metrics) == 720,
                  "Aggregate prediction and fold-metric counts")
    selected = json.loads((ROOT / "results/dl_benchmark_20261104/selected_configuration.json").read_text())
    for model, conf in plan["models"].items():
        source = selected["model_configurations"][model]
        audit.require(all(conf[k] == source[k] for k in ["width", "learning_rate", "parameters"]),
                      model + ": fixed V1 architecture and learning rate")
    registration = pd.Timestamp(manifest["registered_before_first_cv_fit_at_utc"])
    artifact_hashes = predictions_count = curve_rows = fold_configs = 0
    trajectory_predictions, all_fold_metrics, replay = [], [], []
    for row in runs.to_dict("records"):
        rid, variant, model, seed = [row[k] for k in ["run_id", "variant", "model", "seed"]]
        directory = OUT / "runs" / rid
        summary = json.loads((directory / "summary.json").read_text())
        audit.require(rid == f"{variant}_{model}_s{seed}_{manifest['code_signature'][:10]}", rid + ": id")
        audit.require(row["history_days"] == 7 and row["width"] == plan["models"][model]["width"]
                      and row["learning_rate"] == plan["models"][model]["learning_rate"]
                      and row["parameters"] == plan["models"][model]["parameters"]
                      and row["phase"] == "2024_cv", rid + ": trajectory configuration metadata")
        audit.require(set(row) == set(summary["record"]), rid + ": cached trajectory record fields")
        for key, value in row.items():
            if isinstance(value, str):
                audit.require(summary["record"][key] == value, rid + ": cached trajectory identity " + key)
            else:
                audit.close(summary["record"][key], value, rid + ": cached trajectory number " + key)
        audit.require(summary["code_signature"] == manifest["code_signature"]
                      and pd.Timestamp(summary["completed_at_utc"]) > registration,
                      rid + ": registered signature and completion chronology")
        actual_files = {str(p.relative_to(directory)) for p in directory.rglob("*")
                        if p.is_file() and p != directory / "summary.json"}
        audit.require(actual_files == set(summary["artifact_sha256"]), rid + ": complete artifact file set")
        for name, expected in summary["artifact_sha256"].items():
            audit.require(digest(directory / name) == expected, rid + ": artifact digest " + name)
            artifact_hashes += 1
        traj = pd.read_csv(directory / "predictions.csv")
        audit.require(len(traj) == 366 and traj.outcome_day.tolist() == expected_days
                      and not traj.outcome_day.duplicated().any(), rid + ": 366 chronological predictions")
        audit.require(traj.variant.eq(variant).all() and traj.model.eq(model).all()
                      and traj.seed.eq(seed).all() and traj.run_id.eq(rid).all(), rid + ": prediction identity")
        local_frames, local_metrics = [], []
        for ordinal, fold in enumerate(folds):
            fid, fold_seed = fold["fold_id"], seed + ordinal * 100000
            part = directory / fid
            config = json.loads((part / "configuration.json").read_text())
            pre = json.loads((part / "preprocessing.json").read_text())
            split = json.loads((part / "splits.json").read_text())
            fold_record = json.loads((part / "summary.json").read_text())
            audit.require(split == fold, rid + ": exact supplied Filipe fold " + fid)
            audit.require(pre == expected_pre[(variant, fid)], rid + ": training-only scalers " + fid)
            train, inner, valid, test = arrays_by_fold[fid]
            audit.require(config["variant"] == variant and config["model"] == model
                          and config["trajectory_seed"] == seed and config["seed"] == fold_seed
                          and config["inner_initialization_seed"] == fold_seed
                          and config["refit_initialization_seed"] == fold_seed + 10000000,
                          rid + ": fixed fold/trajectory/stage seeds " + fid)
            audit.require(config["width"] == plan["models"][model]["width"]
                          and config["learning_rate"] == plan["models"][model]["learning_rate"]
                          and config["history_days"] == 7 and config["training"] == plan["training"]
                          and config["code_signature"] == manifest["code_signature"]
                          and config["fold_manifest_sha256"] == digest(OUT / "folds.json")
                          and config["test_files_opened"] == 0,
                          rid + ": frozen monthly configuration " + fid)
            audit.require(config["architecture"]["parameters"]["trainable"] == plan["models"][model]["parameters"]
                          and config["architecture"]["parameters"]["non_trainable"] == 0
                          and config["architecture"]["family"] == model
                          and config["architecture"]["history_days"] == 7
                          and config["architecture"]["input_features"] == 5,
                          rid + ": parameter count " + fid)
            audit.require(fold_record["variant"] == variant and fold_record["model"] == model
                          and fold_record["seed"] == seed and fold_record["fold_id"] == fid
                          and fold_record["history_days"] == 7,
                          rid + ": monthly summary identity " + fid)
            frame = pd.read_csv(part / "predictions.csv")
            audit.require(frame.outcome_day.tolist() == calendar[test].strftime("%Y-%m-%d").tolist()
                          and frame.fit_seed.eq(fold_seed).all() and frame.fold_id.eq(fid).all(),
                          rid + ": exact monthly prediction cohort " + fid)
            audit.require(frame.input_first_day.tolist() == calendar[test - 7].strftime("%Y-%m-%d").tolist()
                          and frame.input_last_day.tolist() == calendar[test - 1].strftime("%Y-%m-%d").tolist()
                          and frame.prediction_origin_utc.tolist() == fold["daily_prediction_origins_utc"]
                          and frame.target_available_at_utc.tolist() == fold["daily_target_available_at_utc"],
                          rid + ": actual input/output chronology " + fid)
            audit.close(frame.y_true, data["targets"][test], rid + ": labels " + fid)
            audit.close(frame.persistence_pred, data["persistence"][test], rid + ": persistence " + fid)
            audit.close(frame.mase_train_scale, scale_by_fold[fid], rid + ": monthly MASE scale " + fid)
            audit.require(np.isfinite(frame.y_pred).all() and (frame.y_pred >= 0).all(),
                          rid + ": finite nonnegative predictions " + fid)
            for metric, value in metrics(frame.y_true, frame.y_pred, frame.mase_train_scale).items():
                audit.close(fold_record[metric], value, rid + ": recomputed monthly " + metric + " " + fid)
            audit.require(fold_record["train_samples"] == len(train)
                          and fold_record["inner_train_samples"] == len(inner)
                          and fold_record["inner_validation_samples"] == len(valid)
                          and fold_record["evaluation_samples"] == len(test), rid + ": cohort sizes " + fid)
            ci, cr = pd.read_csv(part / "history_inner.csv"), pd.read_csv(part / "history_refit.csv")
            best_epoch = int(config["best_epoch"])
            audit.require(best_epoch == fold_record["best_epoch"] == ci.best_epoch_so_far.iloc[-1]
                          and 1 <= best_epoch <= len(ci) <= plan["training"]["max_epochs"]
                          and len(cr) == best_epoch, rid + ": epoch selection and fresh-refit duration " + fid)
            for curve, stage, scaler in [(ci, "inner", pre["inner"]), (cr, "refit", pre["refit"])]:
                audit.require(curve.epoch.tolist() == list(range(1, len(curve) + 1))
                              and curve.stage.eq(stage).all(), rid + ": epoch chronology " + fid + " " + stage)
                audit.close(curve.train_rmse_original,
                            np.sqrt(curve.train_mse_scaled_eval) * scaler["y_scale"],
                            rid + ": original-scale train curves " + fid + " " + stage)
                audit.close(curve.validation_rmse_original,
                            np.sqrt(curve.validation_mse_scaled_eval) * scaler["y_scale"],
                            rid + ": original-scale validation curves " + fid + " " + stage)
                curve_rows += len(curve)
            audit.require(cr.validation_mse_scaled_eval.isna().all()
                          and np.isfinite(ci.validation_mse_scaled_eval).all(),
                          rid + ": validation used only by inner stopping " + fid)
            local_frames.append(frame)
            local_metrics.append(fold_record)
            predictions_count += len(frame)
            fold_configs += 1
        merged = pd.concat(local_frames, ignore_index=True)
        try:
            pd.testing.assert_frame_equal(merged, traj, check_exact=False, rtol=1e-12, atol=1e-15)
        except AssertionError:
            audit.failures.append(rid + ": trajectory predictions differ from monthly files")
        recomputed = metrics(traj.y_true, traj.y_pred, traj.mase_train_scale)
        for metric, value in recomputed.items():
            audit.close(row[metric], value, rid + ": recomputed pooled " + metric)
            audit.close(summary["record"][metric], value, rid + ": cached pooled " + metric)
        audit.require(row["n_dates"] == 366 and row["folds_fitted"] == 12 and row["model_fits"] == 24,
                      rid + ": actual monthly segments and effective fits")
        for timing in ["training_seconds", "inference_seconds"]:
            audit.close(row[timing], sum(r[timing] for r in local_metrics), rid + ": summed " + timing)
        local_metrics = pd.DataFrame(local_metrics)
        try:
            pd.testing.assert_frame_equal(local_metrics, pd.read_csv(directory / "fold_metrics.csv"),
                                          check_exact=False, rtol=1e-12, atol=1e-15)
        except AssertionError:
            audit.failures.append(rid + ": trajectory fold metrics differ from monthly summaries")
        trajectory_predictions.append(traj)
        all_fold_metrics.append(local_metrics)

    sort_predictions = ["variant", "model", "seed", "outcome_day"]
    sort_fold_metrics = ["variant", "model", "seed", "fold_id"]
    for observed, frames, keys, label in [
        (top_predictions, trajectory_predictions, sort_predictions, "Aggregate prediction CSV"),
        (top_fold_metrics, all_fold_metrics, sort_fold_metrics, "Aggregate fold-metric CSV"),
    ]:
        try:
            pd.testing.assert_frame_equal(observed.sort_values(keys).reset_index(drop=True),
                                          pd.concat(frames, ignore_index=True).sort_values(keys).reset_index(drop=True),
                                          check_exact=False, rtol=1e-12, atol=1e-15)
        except AssertionError:
            audit.failures.append(label + " differs from per-trajectory artifacts")
    audit.require(execution["trajectories"] == 60 and execution["fold_model_segments"] == fold_configs == 720
                  and execution["completed_project_fits"] == int(runs.model_fits.sum()) == 1440
                  and predictions_count == 21960 and execution["actual_cv_folds"] == 12,
                  "Execution counters agree with all completed trajectory artifacts")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    last_fold = folds[-1]
    for variant in plan["variants"]:
        transform = transform_windows if variant == "absolute_log" else transform_relative_windows
        for model in plan["models"]:
            row = runs[(runs.variant == variant) & (runs.model == model) & (runs.seed == 4101)].iloc[0]
            part = OUT / "runs" / row.run_id / last_fold["fold_id"]
            checkpoint = torch.load(part / "model.pt", map_location="cpu", weights_only=True)
            pre = expected_pre[(variant, last_fold["fold_id"])]["refit"]
            network = build_model(checkpoint["family"], checkpoint["history_days"], checkpoint["width"], checkpoint["dropout"])
            network.load_state_dict(checkpoint["state_dict"], strict=True)
            network.eval()
            x, _ = transform(data, 7, arrays_by_fold[last_fold["fold_id"]][3], pre)
            with torch.inference_mode():
                actual = network(torch.from_numpy(x)).cpu().numpy().astype(np.float64) * pre["y_scale"]
            stored = pd.read_csv(part / "predictions.csv").y_pred.to_numpy()
            maximum = float(np.max(np.abs(actual - stored)))
            audit.close(actual, stored, variant + " " + model + ": December checkpoint reproduces predictions",
                        rtol=1e-12, atol=1e-15)
            audit.require(count_parameters(network)["trainable"] == plan["models"][model]["parameters"]
                          and checkpoint["variant"] == variant and checkpoint["y_scale"] == pre["y_scale"],
                          variant + " " + model + ": checkpoint configuration and scale")
            replay.append({"variant": variant, "model": model, "trajectory_seed": 4101,
                           "fold_id": last_fold["fold_id"], "predictions_replayed": len(actual),
                           "max_absolute_difference": maximum})
    return {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not audit.failures else "FAIL",
        "failures": audit.failures,
        "scope": "Stored monthly forward CV on development2024; no new fitting or reserved raw-test reads",
        "portable_verifier": "scripts/verify_dl_r2_cv.py",
        "registered_source_commit": manifest["source_commit"],
        "code_signature": manifest["code_signature"],
        "checks": {"trajectories": len(runs), "monthly_cv_folds": len(folds),
                   "fold_model_configurations": fold_configs, "recorded_effective_model_fits": int(runs.model_fits.sum()),
                   "per_trajectory_artifact_hashes_checked": artifact_hashes,
                   "prediction_rows_checked": predictions_count, "training_curve_rows_checked": curve_rows,
                   "unique_dates_per_trajectory": 366, "base_seeds_per_variant_model": 10,
                   "paired_variants": plan["variants"], "checkpoint_replays": replay,
                   "reserved_raw_test_files_opened_during_audit": 0, "audit_model_fits": 0,
                   "recorded_test_files_opened": execution["test_files_opened"]},
        "protocol_erratum": {"field": erratum["field"], "changes_numerical_design": False,
                             "signed_protocol_preserved": True,
                             "note": "The inherited full2023 prose is superseded by the dated erratum; every saved monthly split/scaler was checked against the expanding fold cohort."},
        "limitations": [
            "All2024 is exposed development; V1 width/LR/D were chosen with January-June, so this is not nested confirmation of selection.",
            "Ten training-seed trajectories condition on one calendar; monthly folds are dependent, not extra markets or power-derived independent replications.",
            "Absolute versus relative variants isolate the representation package within the same CV, including its scaler weighting; V1 static versus this CV also differs in update/inner-split/seed design.",
            "The Filipe package provenance and fixed index manifest were checked without regenerating the timestamped fold file or refitting models.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/dl_r2_cv_verification.json"))
    args = parser.parse_args()
    try:
        report = verify()
    except Exception as error:
        report = {"audited_at_utc": datetime.now(timezone.utc).isoformat(), "status": "FAIL",
                  "failures": [f"{type(error).__name__}: {error}"],
                  "portable_verifier": "scripts/verify_dl_r2_cv.py",
                  "scope": "Verification stopped; no model fitting or reserved raw-test reads performed"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "report": str(args.output),
                      "failures": report["failures"], "checks": report.get("checks")}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
