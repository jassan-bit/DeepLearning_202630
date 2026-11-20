"""Replay nine accepted ONNX models against 608+53 frozen cases using ORT only.

No Torch, ONNX exporter, fitting, optimization or recalibration is imported.
Downloads stay in --workdir; only the numeric/hash receipt is publishable.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import urllib.request

import numpy as np
import onnxruntime as ort

INPUTS = {
    "cohort.npz": ("https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/af3f1223ad5899ed63016d2af2d09bae8470ab00/book_e2/_static/e2_scientific_closure_20261116/data_audit/current_cohort_arrays.npz", "186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683", 460090),
    "predictions.json": ("https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/5e9023ea821bfa9fb0b92340d518413f6ab89161/runs/e2_extension_10_semillas_v1/combined_predictors.json", "df5206e531ad9c4e5718fb8f17d86390bb1267bcd1d724c2333cacd6c9903655", 3393801),
}
TOLERANCE = 1e-6


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value):
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())


def read_json(path):
    return json.loads(path.read_bytes())


def fetch(path, row):
    url, expected, size = row
    if not path.exists():
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "E3-accepted-ORT-parity/1.0"}), timeout=60) as response:
            data = response.read(10_000_000)
        require(len(data) == size and digest(data) == expected, "Public frozen input hash/size failed")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    require(path.stat().st_size == size and digest(path.read_bytes()) == expected, "Cached frozen input changed")


def transform(raw, scaler):
    values = np.asarray(raw, dtype=np.float64).copy()
    require(values.ndim == 3 and values.shape[1:] == (7, 5) and np.isfinite(values).all(), "Invalid raw cohort")
    require((values[:, :, :4] > 0).all() and (values[:, :, 4] >= 0).all(), "Invalid price/volume domain")
    values[:, :, :4] = np.log(values[:, :, :4]) - np.log(values[:, -1:, 3:4])
    values[:, :, 4] = np.log1p(values[:, :, 4])
    return np.asarray((values - scaler["mean"]) / scaler["std"], dtype=np.float32)


def run(args):
    require(np.__version__ == "2.3.5" and ort.__version__ == "1.23.2", "Use the fixed NumPy/ORT CPU profile")
    require("torch" not in sys.modules, "Torch must not be imported by this verifier")
    artifacts, work = args.artifacts.resolve(), args.workdir.resolve()
    require(artifacts != work and artifacts not in work.parents and work not in artifacts.parents,
            "Keep private downloads outside public artifact tree")
    work.mkdir(parents=True, exist_ok=True)
    for name, row in INPUTS.items():
        fetch(work / name, row)
    prior = read_json(work / "predictions.json")
    manifest = read_json(args.manifest)
    require(len(manifest["onnx_models"]) == 9, "Exactly nine accepted artifacts are required")
    with np.load(work / "cohort.npz", allow_pickle=False) as z:
        raw = np.asarray(z["evaluation_raw"], dtype=np.float64)
        y = np.asarray(z["evaluation_y_original"], dtype=np.float64)
        dates = [str(x) for x in z["evaluation_dates"]]
        cal_raw = np.asarray(z["calibration_raw"], dtype=np.float64)
        cal_y = np.asarray(z["calibration_y_original"], dtype=np.float64)
        cal_dates = [str(x) for x in z["calibration_dates"]]
    require(raw.shape == (608, 7, 5) and cal_raw.shape == (53, 7, 5), "Frozen cohort counts changed")
    require(dates == prior["dates"] and np.array_equal(y, prior["y_true"]), "Evaluation dates/targets changed")
    records = []
    for accepted in manifest["onnx_models"]:
        relative = Path(accepted["file"])
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe model path")
        model = artifacts / relative
        directory = model.parent
        require(model.stat().st_size == accepted["bytes"] and digest(model.read_bytes()) == accepted["sha256"], "Accepted ONNX hash/size changed")
        for name in ("metadata.json", "scaler.json", "calibrator.json"):
            path = directory / name
            member = relative.with_name(name).as_posix()
            expected = manifest["members"][member]
            require(path.stat().st_size == expected["bytes"] and digest(path.read_bytes()) == expected["sha256"], "Frozen metadata identity changed")
        metadata, scaler, cal = [read_json(directory / n) for n in ("metadata.json", "scaler.json", "calibrator.json")]
        refs = [r for r in prior["predictors"] if r["model"] == accepted["model"] and str(r["seed"]) == "271801"]
        require(len(refs) == 1, "Original seed-271801 predictor is not unique")
        ref = refs[0]
        require(ref["checkpoint_sha256"] == accepted["checkpoint_sha256"] == cal["predictor_sha256"], "Predictor/checkpoint/calibrator identity mismatch")
        require(cal["sha256"] == canonical_hash({k: v for k, v in cal.items() if k != "sha256"}), "Calibrator internal hash changed")
        require(cal["q"] == ref["calibrator_q"] and cal["n"] == 53 and cal_dates == cal["dates"] == ref["calibration_dates"], "Original calibrator changed")
        require(cal["calibration_predictions_sha256"] == canonical_hash(ref["calibration_predictions"]) and cal["calibration_targets_sha256"] == canonical_hash(ref["calibration_targets"]), "Calibration reference hashes changed")
        require(np.array_equal(cal_y, ref["calibration_targets"]), "Calibration targets changed")
        # Both metadata schemas are supported via the original scaler. The ONNX
        # graph emits sigma7/target_scale; inversion happens exactly once here.
        scale = scaler["target_scale"]
        require(metadata["output_scale"] == scale, "Metadata and original target scale differ")
        x, cal_x = transform(raw, scaler), transform(cal_raw, scaler)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
        require(len(session.get_inputs()) == 1 and session.get_inputs()[0].name == "windows" and session.get_inputs()[0].shape == [1, 7, 5], "Unexpected model input contract")
        require(len(session.get_outputs()) == 1 and session.get_outputs()[0].name == "normalized_prediction", "Unexpected model output contract")
        def replay(values):
            return np.asarray([float(session.run(["normalized_prediction"], {"windows": v[None]})[0].reshape(-1)[0]) * scale for v in values])
        prediction, cal_prediction = replay(x), replay(cal_x)
        frozen = np.asarray(ref["prediction_original"], dtype=np.float64)
        frozen_cal = np.asarray(ref["calibration_predictions"], dtype=np.float64)
        require(np.isfinite(prediction).all() and np.isfinite(cal_prediction).all(), "Nonfinite replay output")
        error = float(np.max(np.abs(prediction - frozen)))
        cal_error = float(np.max(np.abs(cal_prediction - frozen_cal)))
        require(error <= TOLERANCE and cal_error <= TOLERANCE, "Frozen historical parity failed: " + accepted["model"])
        q = cal["q"]
        covered = (y >= np.maximum(0, prediction - q)) & (y <= prediction + q)
        frozen_covered = (y >= np.maximum(0, frozen - q)) & (y <= frozen + q)
        records.append({"model": accepted["model"], "seed": 271801, "onnx_sha256": accepted["sha256"],
            "checkpoint_sha256": accepted["checkpoint_sha256"], "evaluation_cases": 608, "calibration_cases": 53,
            "maximum_evaluation_absolute_error_original_sigma7": error, "maximum_calibration_absolute_error_original_sigma7": cal_error,
            "tolerance_original_sigma7_fixed_before_evaluation": TOLERANCE, "within_tolerance": True,
            "output_scale_applied_once": scale, "calibration_q_unchanged": q,
            "coverage_boundary_disagreements": int(np.sum(covered != frozen_covered)),
            "new_training_runs": 0, "new_calibration_fits": 0})
        print(json.dumps({"model": accepted["model"], "status": "661_ORT_only_frozen_cases_passed", "max_error": error, "max_calibration_error": cal_error}), flush=True)
        del session
    require("torch" not in sys.modules, "Torch unexpectedly imported")
    result = {"status": "nine_accepted_ONNX_models_passed_ORT_only_real_frozen_parity", "created_utc": datetime.now(timezone.utc).isoformat(),
        "models": records, "manifest_sha256": digest(args.manifest.read_bytes()), "verifier_sha256": digest(Path(__file__).read_bytes()),
        "numpy": np.__version__, "onnxruntime": ort.__version__, "python": platform.python_version(), "platform": platform.platform(),
        "provider": "CPUExecutionProvider", "threads_intra_inter": [1, 1], "historical_replay_inference_calls": 9 * 661,
        "public_inputs": {name: {"url": row[0], "sha256": row[1], "bytes": row[2]} for name, row in INPUTS.items()},
        "Torch_imported": False, "ONNX_reexported": False, "new_training_runs": 0, "new_calibration_fits": 0, "new_forecast_dates": 0,
        "scope": "Byte-preserved accepted artifacts replayed against historical saved predictions. Does not identify the cause of a different CI re-export binary, promote variants or accredit R1/R3.", "AI_assistance_declared": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    require(not args.output.exists(), "Do not overwrite an earlier parity receipt")
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--workdir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    run(p.parse_args())
