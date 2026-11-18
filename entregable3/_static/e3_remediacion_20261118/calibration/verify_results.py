"""Verify the recorded diagnostic without executing any model forwards."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from run import (HERE, ROOT, SEEDS, SOURCE_HASHES, admit_inputs, fit_quantile,
                 interval_values, metric_summary, require, rmse, sha, write_json)


def verify(input_dir, result_dir):
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT))
    from src.e3_innovation_model import load_innovation_data

    folder = admit_inputs(input_dir, download=False)
    result_dir = Path(result_dir)
    results = json.loads((result_dir / "results.json").read_bytes())
    for filename, key in (("analysis_plan.json", "analysis_plan_sha256"),
                          ("input_manifest.json", "input_manifest_sha256"),
                          ("run.py", "script_sha256")):
        require(sha(HERE / filename) == results[key], "Result/code binding changed: " + filename)
    for name, expected in SOURCE_HASHES.items():
        require(sha(ROOT / name) == expected, "Scientific source changed")
    admission = json.loads((folder / "admission.json").read_bytes())
    train = load_innovation_data(folder / "cohort.npz", admission["npz_sha256"])
    require(train["admission"]["arrays_sha256"] == admission["arrays_sha256"],
            "Original fit/validation array reconstruction differs")
    require(train["scaler"] == admission["scaler"], "Original fit-only scaler differs")
    require(len(train["train_dates"]) == 532 and len(train["val_dates"]) == 116,
            "Original fit/selection membership differs")
    checkpoint_dtypes = {}
    for seed in SEEDS:
        state = torch.load(folder / "checkpoints" / (str(seed) + ".pt"),
                           weights_only=True, map_location="cpu")["best_model_state"]
        require(all(str(v.dtype) == "torch.float32" for k, v in state.items()
                    if k.startswith("base.encoder.") or k in {
                        "base.mechanism.query_raw", "base.mechanism.temperature_raw", "base.mechanism.P"}),
                "Frozen encoder/attention precision differs")
        require(all(str(state[k].dtype) == "torch.float64" for k in (
            "base.mechanism.readout_raw", "base.mechanism.bias", "y_scale", "u_scale")),
            "Frozen readout precision differs")
        checkpoint_dtypes[str(seed)] = "encoder/attention float32; readout/scales float64"
    with (folder / "cohort.npz").open("rb") as stream:
        with np.load(stream, allow_pickle=False) as values:
            cal_dates = values["calibration_dates"].tolist()
            cal_targets = values["calibration_y_original"].tolist()
    with (folder / "evaluation_predictions.csv").open(newline="", encoding="utf-8") as stream:
        frozen = list(csv.DictReader(stream))
    with (result_dir / "calibration_predictions.csv").open(newline="", encoding="utf-8") as stream:
        calibration = list(csv.DictReader(stream))
    with (result_dir / "per_seed_intervals.csv").open(newline="", encoding="utf-8") as stream:
        per_seed = list(csv.DictReader(stream))
    require(len(calibration) == 530 and len(per_seed) == 10, "Diagnostic output membership differs")
    for seed in SEEDS:
        cal = [row for row in calibration if int(row["seed"]) == seed]
        require([row["target_date"] for row in cal] == cal_dates, "Calibration dates differ")
        require([float(row["target_sigma7"]) for row in cal] == cal_targets, "Calibration targets differ")
        forecast = [float(row["frozen_CPU_prediction_sigma7"]) for row in cal]
        require(all(float(row["absolute_residual_sigma7"]) == abs(y - p)
                    for row, y, p in zip(cal, cal_targets, forecast, strict=True)), "Residuals differ")
        q, rank = fit_quantile(cal_targets, forecast, .1)
        recorded = next(row for row in per_seed if int(row["seed"]) == seed)
        json_record = next(row for row in results["per_seed"] if row["seed"] == seed)
        require(q == float(recorded["q_sigma7"]) == json_record["q_sigma7"] and rank == 49,
                "Calibration order statistic differs")
        saved = [row for row in frozen if int(row["seed"]) == seed]
        require(len(saved) == 608, "Evaluation membership differs")
        targets = [float(row["target_sigma7"]) for row in saved]
        predictions = [float(row["prediction_sigma7"]) for row in saved]
        metrics = metric_summary(interval_values(targets, predictions, q, .1))
        for key, value in metrics.items():
            require(float(recorded[key]) == value == json_record[key], "Interval reconstruction differs: " + key)
        require(rmse(targets, predictions) == float(recorded["unchanged_point_RMSE"]), "Point forecast changed")
    immutable = ["results.json", "per_seed_intervals.csv", "calibration_predictions.csv", "subgroup_diagnostics.csv"]
    report = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "original_inputs_verified_by_bytes_and_SHA256": 15,
        "original_fit_validation_arrays_reconstructed_by_SHA256": admission["arrays_sha256"],
        "original_scaler_equal": True,
        "frozen_checkpoint_precision": checkpoint_dtypes,
        "calibration_rows_reconstructed": 530,
        "quantiles_reconstructed_from_calibration_only": 10,
        "saved_test_point_predictions_reused": 6080,
        "interval_metrics_reconstructed": True,
        "new_model_forward_calls_in_this_verification": 0,
        "new_fit_steps_in_this_verification": 0,
        "result_sha256": {name: sha(result_dir / name) for name in immutable},
        "verifier_sha256": sha(__file__),
        "limits": "Automated integrity and numerical reconstruction; not a human independent reproduction or confirmatory R3 evidence."
    }
    write_json(result_dir / "verification.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    arguments = parser.parse_args()
    report = verify(arguments.input_dir, arguments.results)
    print(json.dumps({"status": report["status"], "model_forward_calls": 0,
                      "original_inputs_verified": report["original_inputs_verified_by_bytes_and_SHA256"],
                      "result_files_verified": len(report["result_sha256"])}, indent=2))
