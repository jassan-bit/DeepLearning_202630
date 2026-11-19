"""Export the existing seed-271801 TSMixer; never train, tune or recalibrate.

All external frozen inputs are checked before weights_only deserialization.
This writes an audit artifact, not a change to the live model registry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import onnx
import onnxruntime as ort
import torch

EXPECTED = {
    "checkpoint": "9ead27bd519975e8a62abc79c92ee326aec8bece71170828666b9964ae454be7",
    "npz": "186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683",
    "predictions": "df5206e531ad9c4e5718fb8f17d86390bb1267bcd1d724c2333cacd6c9903655",
    "protocol": "5195337af13f4193da16bf4499c8396586c621d4ab084a5b296a297f2141edeb",
    "scripts/e2_final_models_mlp.py": "beeada2f6754f0ca3ff1304ec090a7259a774cc7ea611fa912428cf9210f6c68",
    "scripts/e2_training_models.py": "46be0644e0ed9d966163b17f171092e81b6a82dd734efc7954d6104f5ee5360b",
}
TOLERANCE = 1e-6


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(path, key):
    actual = sha(path)
    if actual != EXPECTED[key]:
        raise RuntimeError(f"Frozen input mismatch: {key}")
    return actual


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def transform(raw, scaler):
    raw = np.asarray(raw, dtype=np.float64)
    relative = raw.copy()
    relative[:, :, :4] = np.log(raw[:, :, :4]) - np.log(raw[:, -1:, 3:4])
    relative[:, :, 4] = np.log1p(raw[:, :, 4])
    return ((relative - scaler["mean"]) / scaler["std"]).astype(np.float32)


def run(args):
    for key in ("checkpoint", "npz", "predictions", "protocol"):
        check(getattr(args, key), key)
    for relative in ("scripts/e2_final_models_mlp.py", "scripts/e2_training_models.py"):
        check(args.repo / relative, relative)
    sys.path.insert(0, str(args.repo / "scripts"))
    import e2_training_models as factory
    from e2_reduced_final_analysis import digest
    # The sparse working tree lacks the protocol. This immutable copy has the
    # same expected checksum, and the unmodified factory validates the HPO row.
    factory.PROTOCOL_PATH = args.protocol
    cp = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    selection = json.loads(args.selection.read_bytes())
    trial = selection["winner_trial"]
    assert selection["model_id"] == "TSMixer" and selection["winner_configuration_id"] == "C03"
    assert cp["architecture"]["training_hpo_record"] == trial
    assert cp["architecture"]["initialization_seed"] == 271801
    model = factory.build_train_model("TSMixer", trial, seed=271801).eval()
    model.load_state_dict(cp["state_dict"], strict=True)
    scaler = cp["scaler"]
    assert scaler == json.loads(args.scaler.read_bytes())
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == 35044
    prior = json.loads(args.predictions.read_bytes())
    reference = next(p for p in prior["predictors"] if p["model"] == "TSMixer" and str(p["seed"]) == "271801")
    assert reference["checkpoint_sha256"] == EXPECTED["checkpoint"]
    cal = json.loads(args.calibrator.read_bytes())
    assert cal["sha256"] == digest({k: v for k, v in cal.items() if k != "sha256"})
    assert cal["predictor_sha256"] == EXPECTED["checkpoint"]
    assert cal["q"] == reference["calibrator_q"] and cal["n"] == 53
    assert cal["dates"] == reference["calibration_dates"]
    assert cal["calibration_predictions_sha256"] == digest(reference["calibration_predictions"])
    assert cal["calibration_targets_sha256"] == digest(reference["calibration_targets"])
    with np.load(args.npz, allow_pickle=False) as z:
        raw = np.asarray(z["evaluation_raw"], dtype=np.float64)
        y = np.asarray(z["evaluation_y_original"], dtype=np.float64)
        dates = [str(x) for x in z["evaluation_dates"]]
        fit_raw = np.asarray(z["training_raw"], dtype=np.float64)
        fit_dates = [str(x) for x in z["training_dates"]]
        cal_raw = np.asarray(z["calibration_raw"], dtype=np.float64)
        cal_y = np.asarray(z["calibration_y_original"], dtype=np.float64)
        cal_dates = [str(x) for x in z["calibration_dates"]]
    assert dates == prior["dates"] and np.array_equal(y, prior["y_true"])
    assert raw.shape == (608, 7, 5)
    assert fit_dates == scaler["train_dates"] and len(fit_dates) == 655
    assert cal_dates == cal["dates"] and np.array_equal(cal_y, reference["calibration_targets"])
    x, fit_x, cal_x = [transform(value, scaler) for value in (raw, fit_raw, cal_raw)]
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "model.onnx"
    torch.onnx.export(model, (torch.from_numpy(x[:1]),), str(path),
                      input_names=["windows"], output_names=["normalized_prediction"],
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(str(path)))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    def memory(values):
        with torch.no_grad():
            return np.asarray([float(model(torch.from_numpy(v[None]))[0]) * scaler["target_scale"] for v in values])
    def exported(values):
        return np.asarray([float(session.run(None, {"windows": v[None]})[0].reshape(-1)[0]) * scaler["target_scale"] for v in values])
    cpu, pred = memory(x), exported(x)
    cal_cpu, cal_pred = memory(cal_x), exported(cal_x)
    frozen = np.asarray(reference["prediction_original"], dtype=np.float64)
    errors = {
        "maximum_absolute_error_vs_CPU_memory": float(np.max(np.abs(pred - cpu))),
        "maximum_absolute_error_vs_frozen_saved_predictions": float(np.max(np.abs(pred - frozen))),
        "maximum_calibration_absolute_error_vs_CPU_memory": float(np.max(np.abs(cal_pred - cal_cpu))),
        "maximum_calibration_absolute_error_vs_frozen_predictions": float(np.max(np.abs(cal_pred - reference["calibration_predictions"]))),
    }
    assert all(e <= TOLERANCE for e in errors.values())
    q = cal["q"]
    coverage = float(100*np.mean((y >= np.maximum(0, pred-q)) & (y <= pred+q)))
    timings = []
    for i in range(110):
        started = perf_counter()
        xi = transform(raw[:1], scaler)
        session.run(None, {"windows": xi})
        if i >= 10:
            timings.append((perf_counter()-started)*1000)
    report = {
        "model": "TSMixer", "seed": 271801, "evaluation_dates": 608,
        "date_start": dates[0], "date_end": dates[-1], "calibration_dates": 53,
        "declared_absolute_tolerance_original_sigma7": TOLERANCE, **errors,
        "RMSE_CPU_memory": float(np.sqrt(np.mean((cpu-y)**2))),
        "RMSE_ONNX": float(np.sqrt(np.mean((pred-y)**2))),
        "RMSE_frozen_saved_predictor": float(np.sqrt(np.mean((frozen-y)**2))),
        "within_tolerance": True, "cpu_median_ms": float(np.median(timings)),
        "cpu_p95_ms": float(np.quantile(timings, .95)),
        "latency_scope": "preprocessing + warm one-thread ORT CPU, batch one; 10 warmups and 100 calls; excludes explanation",
        "onnx_bytes": path.stat().st_size, "trainable_parameters": parameters,
        "historical_608_coverage_percent_for_this_seed": coverage,
        "FLOPs": None, "runtime_peak_memory": None,
        "new_training_runs": 0, "new_calibration_fits": 0, "GPU_used": False,
        "live_registry_updated": False, "calibration_q_reused_unchanged": True,
    }
    metadata = {
        "schema_version": 1, "model": "TSMixer", "seed": 271801,
        "representative_seed_rule": "smallest fixed seed ID, not selected by test performance",
        "format": "ONNX", "opset": 17, "model_sha256": sha(path),
        "input_shape": [1, 7, 5], "input_dtype": "float32", "uses_observable_floor": False,
        "floor_input_dtype": None, "output_scale": scaler["target_scale"],
        "preprocessing": {"mean": scaler["mean"], "std": scaler["std"], "fit_count": 655,
                          "fit_end": fit_dates[-1], "target_scale": scaler["target_scale"]},
        "calibration_q": q, "interval_nominal_coverage": .9,
        "historical_608_coverage_percent_for_this_seed": coverage,
        "interval_limitations": "Historical BTCUSDT calibration, 53 serially dependent days; no exact exchangeability or future/subgroup coverage guarantee.",
        "output_unit": "original population sigma7, not annualized", "horizon_days": 1,
        "raw_schema": ["date", "open", "high", "low", "close", "volume"], "max_csv_bytes": 102400,
        "ood": {"method": "per-lag/channel observed range in standardized fit windows",
                "minimum_standardized": fit_x.min(axis=0).astype(float).tolist(),
                "maximum_standardized": fit_x.max(axis=0).astype(float).tolist(),
                "fit_only": True, "not_a_calibrated_reliability_guarantee": True},
        "explanation_reference_raw": fit_raw[0].tolist(), "explanation_reference_target_date": fit_dates[0],
        "scientific_source": {"type": "existing_E2_frozen_final_checkpoint", "checkpoint_sha256": EXPECTED["checkpoint"],
                              "registered_trial_id": trial["trial_id"], "request_sha256": cp["request_sha256"],
                              "source_hashes": {k: v for k, v in EXPECTED.items() if k.startswith("scripts/")}},
        "data_source": {"npz_sha256": EXPECTED["npz"], "predictions_sha256": EXPECTED["predictions"]},
        "calibrator_source": {"file_sha256": sha(args.calibrator), "content_sha256": cal["sha256"],
                              "q": q, "rank": 49, "n": 53, "date_start": cal_dates[0], "date_end": cal_dates[-1],
                              "checkpoint_sha256": cal["predictor_sha256"]},
        "roundtrip": report,
        "libraries": {"python": platform.python_version(), "torch_export_only": str(torch.__version__),
                      "onnx_export_only": onnx.__version__, "onnxruntime": ort.__version__, "numpy": np.__version__},
    }
    write(args.output / "metadata.json", metadata)
    write(args.output / "roundtrip.json", report)
    write(args.output / "provenance.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {key: {"file": str(getattr(args, key)), "sha256": sha(getattr(args, key))}
                   for key in ("checkpoint", "npz", "predictions", "protocol", "selection", "scaler", "calibrator")},
        "source_hashes": metadata["scientific_source"]["source_hashes"],
        "output": {"model_sha256": sha(path), "metadata_sha256": sha(args.output / "metadata.json")},
        "scope": "one pre-existing representative seed; export and validation only", "no_training_no_recalibration": True})
    print(json.dumps(report))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "checkpoint", "npz", "predictions", "protocol", "selection", "scaler", "calibrator", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    run(p.parse_args())
