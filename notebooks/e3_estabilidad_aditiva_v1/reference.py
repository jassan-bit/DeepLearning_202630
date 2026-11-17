"""Frozen additive-v2 inference on the known E2 historical test calendar.

No fitting, optimization, bias matching, calibration, seed search or epoch
selection is performed. Validation reproduction precedes opening test members.
The output is a descriptive comparison of differently trained systems.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

VERSION = "e3-frozen-additive-historical-evaluation-v1"
NPZ_SHA256 = "186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683"
SNAPSHOT_SHA256 = "5fe65596b99b334de197f3fc08811514c5ee6ea48b8447318057fea1b7cf9be1"
BEST_SHA256 = "65641a377ab3e6051081d8a8d15f55e7eece3765b305a0029b032418c8095916"
RESULT_COMMIT = "1ca029e869a517cb8de704e573c78ca1e0cbad3c"
E2_COMMIT = "f65b14d38e38f8d5fd80c717d71f2abd9ea708e2"
VARIANT = "additive_positive_head"
ROUNDTRIP_TOLERANCE = 1e-7
E2_HASHES = {
    "e2_partition.json": "69ac07eebcfad0e946d2edaaf433871e4045696b95035e7b5252cf3409a7e02f",
    "e2_benchmark_A2.csv": "879e2ce52709657a06bc6c009f0cd553423f2eaa0bbdf16dc7f5844dfec9fcc1",
    "e2_data_admission.json": "1d430f16a4d363055800fb4ec62cc4a7bc9062a131e3a792324b8440e5cebd9c",
}
TOP10 = (
    "FCN-1D", "ResNet-1D", "Inception-1D", "LSTM-autoencoder con predictor",
    "TimesNet", "TSMixer", "TiDE", "TimeMixer", "PatchTST", "iTransformer",
)
NEURAL_BASELINES = ("MLP16", "LSTM32", "GRU32")
DETERMINISTIC = ("Persistence", "SeasonalNaive", "ARIMA")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def checked(path, expected, expected_bytes=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Missing or symlinked input: " + str(path))
    require(digest(path) == expected, "SHA256 mismatch: " + str(path))
    if expected_bytes is not None:
        require(path.stat().st_size == expected_bytes, "Byte count mismatch: " + str(path))
    return path


def dates(values, expected_count, name):
    result = [str(v) for v in values]
    require(len(result) == expected_count and result == sorted(set(result)), name + " dates/count invalid")
    require(all(date.fromisoformat(v).isoformat() == v for v in result), name + " dates not ISO")
    return result


def transform(raw, scaler):
    """Apply saved fit statistics only; A uses six already observed returns."""
    raw = np.asarray(raw, dtype=np.float64)
    require(raw.ndim == 3 and raw.shape[1:] == (7, 5) and len(raw) > 0,
            "Expected nonempty raw N x 7 x 5")
    require(np.isfinite(raw).all() and (raw[..., :4] > 0).all()
            and (raw[..., 4] >= 0).all(), "Invalid OHLCV")
    require((raw[..., 1] >= np.maximum(raw[..., 0], raw[..., 3])).all()
            and (raw[..., 2] <= np.minimum(raw[..., 0], raw[..., 3])).all()
            and (raw[..., 1] >= raw[..., 2]).all(), "Invalid OHLC geometry")
    mean, std = np.asarray(scaler["mean"], dtype=np.float64), np.asarray(scaler["std"], dtype=np.float64)
    require(mean.shape == std.shape == (5,) and np.isfinite(mean).all()
            and np.isfinite(std).all() and (std > 0).all(), "Invalid saved scaler")
    require(scaler["ddof"] == 0 and scaler["relative_window_anchor"] == "last_known_close"
            and scaler["target_centered"] is False, "Unexpected saved preprocessing")
    for key in ("y_scale", "u_scale", "target_scale"):
        require(math.isfinite(float(scaler[key])) and float(scaler[key]) > 0, "Invalid saved scale")
    require(scaler["y_scale"] == scaler["target_scale"], "Inconsistent target scales")
    relative = np.empty_like(raw)
    relative[..., :4] = np.log(raw[..., :4]) - np.log(raw[:, -1, 3])[:, None, None]
    relative[..., 4] = np.log1p(raw[..., 4])
    x = ((relative - mean) / std).astype(np.float32)
    returns = np.diff(np.log(raw[..., 3]), axis=1)
    a = ((returns - returns.mean(axis=1)[:, None]) ** 2).sum(axis=1) / 7.0
    require(np.isfinite(x).all() and np.isfinite(a).all() and (a >= 0).all(),
            "Nonfinite transformed input")
    return x, a


def infer(model, raw, scaler, device):
    x, a = transform(raw, scaler)
    parts = []
    model.eval()
    with torch.inference_mode():
        for first in range(0, len(x), 32):
            output = model(torch.as_tensor(x[first:first + 32], device=device),
                           torch.as_tensor(a[first:first + 32], dtype=torch.float64, device=device))
            parts.append(output.cpu().numpy().astype(np.float64))
    predicted = np.concatenate(parts)
    require(predicted.shape == (len(raw),) and np.isfinite(predicted).all()
            and (predicted >= 0).all(), "Invalid inference output")
    # InnovationModel already returns original sigma7; no scale multiplication.
    require((predicted >= np.sqrt(a)).all(), "Frozen additive prediction violates its floor")
    return predicted, np.sqrt(a)


def validate_target(y, n, floor=None):
    y = np.asarray(y, dtype=np.float64)
    require(y.shape == (n,) and np.isfinite(y).all() and (y >= 0).all(), "Invalid target array")
    if floor is not None:
        require((y * y - floor * floor >= -1e-12).all(), "Targets violate centered sigma7 floor")
    return y


def metrics(y, prediction):
    errors = prediction - y
    return {"RMSE": math.sqrt(math.fsum(float(e) ** 2 for e in errors) / len(y)),
            "MAE": math.fsum(abs(float(e)) for e in errors) / len(y),
            "bias": math.fsum(float(e) for e in errors) / len(y)}


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def evaluate(source_root, inputs, output, device, *, roundtrip_only=False):
    inputs, output, source_root = Path(inputs), Path(output), Path(source_root).resolve()
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
            "Output must be absent or empty; preserve existing results")
    require(torch.__version__.split("+")[0] == "2.8.0" and np.__version__ == "2.3.5",
            "Requires PyTorch 2.8.0 and NumPy 2.3.5")
    require(device in ("cpu", "cuda"), "Unknown device")
    require(device != "cuda" or torch.cuda.is_available(), "Requested CUDA is unavailable")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if device == "cuda":
        import os
        require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8"),
                "CUBLAS_WORKSPACE_CONFIG must be set before CUDA initialization")
    checked(inputs / "snapshot_manifest.json", SNAPSHOT_SHA256)
    snapshot = read_json(inputs / "snapshot_manifest.json")
    require(snapshot["progress"]["state_status"] == "completed", "Source v2 pilot is not complete")
    for name in ("manifest.json", "data_admission.json", "additive_positive_head/best.pt",
                 "additive_positive_head/validation_predictions.csv"):
        record = snapshot["files"][name]
        checked(inputs / name, record["sha256"], record["bytes"])
    checked(inputs / "additive_positive_head/best.pt", BEST_SHA256)
    checked(inputs / "current_cohort_arrays.npz", NPZ_SHA256)
    for name, sha in E2_HASHES.items():
        checked(inputs / name, sha)

    manifest, admission = read_json(inputs / "manifest.json"), read_json(inputs / "data_admission.json")
    require(manifest["run_id"] == "6cc32b1a91224c4f9144a1c831394746", "Unexpected source run")
    require(manifest["npz_sha256"] == admission["npz_sha256"] == NPZ_SHA256, "Source data identity mismatch")
    require(hashlib.sha256(canonical(admission)).hexdigest() == manifest["data_fingerprint"],
            "Saved admission fingerprint mismatch")
    for key, value in snapshot["scientific_identity"].items():
        require(manifest[key] == value, "Snapshot identity mismatch: " + key)
    for name, sha in manifest["source_hashes"].items():
        checked(source_root / name, sha)
    require(admission["n_fit"] == 532 and admission["n_validation"] == 116,
            "Unexpected frozen E3 cohorts")
    require(admission["scaler"]["fit_dates"] == admission["fit_dates"], "Scaler fit dates changed")
    require(len(admission["fit_dates"]) == 532 and max(admission["fit_dates"]) == "2024-06-30",
            "Scaler fit identity changed")
    sys.path.insert(0, str(source_root))
    from src.e3_innovation_model import InnovationModel
    loaded_module = sys.modules["src.e3_innovation_model"]
    require(Path(loaded_module.__file__).resolve() == source_root / "src/e3_innovation_model.py",
            "A different source module was cached")
    checkpoint = torch.load(inputs / "additive_positive_head/best.pt", map_location="cpu", weights_only=True)
    for key in ("source_hashes", "plan_sha256", "data_fingerprint", "run_id", "claim_id"):
        require(checkpoint[key] == manifest[key], "Checkpoint provenance mismatch: " + key)
    require(checkpoint["version"] == "e3-innovation-development-v2"
            and checkpoint["model_name"] == VARIANT and checkpoint["epoch"] == 38
            and checkpoint["best_epoch"] == 38, "Wrong frozen checkpoint or epoch")
    require(all(isinstance(v, torch.Tensor) and bool(torch.isfinite(v).all())
                for v in checkpoint["model_state"].values()), "Invalid model state")
    scaler = admission["scaler"]
    rule = source_root / "experiments/e3_ficha_cnn_lstm_atencion_v2_20261116/verification/quadrature_k128.json"
    model = InnovationModel(VARIANT, u_scale=scaler["u_scale"], y_scale=scaler["y_scale"],
                            seed=20261116, rule_path=rule)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    require(sum(p.numel() for p in model.parameters()) == 1210, "Unexpected model capacity")
    require(float(model.y_scale) == scaler["y_scale"] and float(model.u_scale) == scaler["u_scale"],
            "Saved model/scaler scales differ")
    model = model.to(device).eval()

    # Reproduce all saved validation outputs before any evaluation member access.
    validation_rows = read_csv(inputs / "additive_positive_head/validation_predictions.csv")
    validation_dates = dates([r["date"] for r in validation_rows], 116, "Saved validation")
    require(validation_dates == admission["validation_dates"], "Saved validation IDs differ")
    with np.load(inputs / "current_cohort_arrays.npz", allow_pickle=False) as source:
        training_dates = dates(source["training_dates"], 655, "Source training")
        positions = {d: i for i, d in enumerate(training_dates)}
        indices = np.array([positions[d] for d in validation_dates], dtype=np.int64)
        validation_raw = np.asarray(source["training_raw"], dtype=np.float64)[indices].copy()
        validation_y = np.asarray(source["training_y_original"], dtype=np.float64)[indices].copy()
    expected_y = np.array([float(r["target_sigma7"]) for r in validation_rows])
    require(np.array_equal(validation_y, expected_y), "Validation targets differ")
    validation_prediction, validation_floor = infer(model, validation_raw, scaler, device)
    validate_target(validation_y, 116, validation_floor)
    expected_prediction = np.array([float(r["prediction_sigma7"]) for r in validation_rows])
    roundtrip_error = float(np.max(np.abs(validation_prediction - expected_prediction)))
    require(roundtrip_error <= ROUNDTRIP_TOLERANCE,
            f"Validation roundtrip failed ({roundtrip_error}); evaluation remains unopened")
    print(f"Validación reproducida: 116 fechas, diferencia máxima {roundtrip_error:.3g}", flush=True)
    if roundtrip_only:
        print("Comprobación de validación terminada; miembros de prueba no abiertos.", flush=True)
        return {"status": "validation_roundtrip_only", "evaluation_members_opened": False,
                "roundtrip_max_absolute_error": roundtrip_error, "examples": 116}

    partition = read_json(inputs / "e2_partition.json")["final_partition"]
    expected_dates = dates(partition["evaluation"]["sample_IDs"], 608, "E2 test")
    require(expected_dates[0] == "2025-01-01" and expected_dates[-1] == "2026-08-31",
            "Historical test calendar changed")
    e2_admission = read_json(inputs / "e2_data_admission.json")
    require(e2_admission["target"] == "population sigma7 of daily Close log returns, ddof0, unannualized; h1"
            and e2_admission["common_eligible_dates"] == expected_dates,
            "E2 target or calendar identity changed")
    require(not set(expected_dates) & (set(admission["fit_dates"]) | set(validation_dates)),
            "Historical evaluation overlaps E3 fit or stopping cohort")
    a2 = read_csv(inputs / "e2_benchmark_A2.csv")
    require(len(a2) == 16 and {r["model"] for r in a2} == set(TOP10 + NEURAL_BASELINES + DETERMINISTIC),
            "E2 comparison family changed")
    for row in a2:
        require(int(row["n_common_dates"]) == 608 and int(row["R"]) == (1 if row["model"] in DETERMINISTIC else 6),
                "E2 date/seed count mismatch")
        require(all(math.isfinite(float(row[k])) and float(row[k]) >= 0 for k in ("RMSE", "MAE")),
                "Invalid E2 saved metric")

    started = time.perf_counter()
    with np.load(inputs / "current_cohort_arrays.npz", allow_pickle=False) as source:
        test_dates = dates(source["evaluation_dates"], 608, "NPZ test")
        require(test_dates == expected_dates, "NPZ/E2 test dates differ")
        raw = np.asarray(source["evaluation_raw"], dtype=np.float64).copy()
        y = np.asarray(source["evaluation_y_original"], dtype=np.float64).copy()
    require(len(raw) == 608, "Unexpected raw evaluation row count")
    prediction, floor = infer(model, raw, scaler, device)
    y = validate_target(y, 608, floor)
    scored = metrics(y, prediction)
    elapsed = time.perf_counter() - started
    require(all(torch.equal(value.detach().cpu(), checkpoint["model_state"][key])
                for key, value in model.state_dict().items()), "Inference modified frozen model state")

    prediction_rows = [{"date": d, "target_sigma7": float(t), "prediction_sigma7": float(p),
                        "known_floor": float(f)} for d, t, p, f in zip(test_dates, y, prediction, floor)]
    comparison = []
    for row in a2:
        deterministic = row["model"] in DETERMINISTIC
        comparison.append({"model": row["model"], "source": "E2_frozen",
            "role": "Top10" if row["model"] in TOP10 else "baseline",
            "RMSE": float(row["RMSE"]), "MAE": float(row["MAE"]),
            "metric_aggregation": "single_deterministic_trajectory" if deterministic else "arithmetic_mean_of_six_seed_metrics",
            "predictors": int(row["R"]), "training_examples": None if deterministic else 655,
            "training_count_scope": "not_comparable_classical_recipe" if deterministic else "supervised_objective_windows",
            "training_end": "2024-10-31", "evaluation_examples": 608,
            "status": "historical_descriptive_not_equal_budget_comparison"})
    comparison.append({"model": "CNN-LSTM-atencion_aditiva_v2", "source": "E3_frozen",
        "role": "development_reference", "RMSE": scored["RMSE"], "MAE": scored["MAE"],
        "metric_aggregation": "one_frozen_seed_metric", "predictors": 1,
        "training_examples": 532, "training_count_scope": "supervised_objective_windows",
        "training_end": "2024-06-30", "evaluation_examples": 608,
        "status": "historical_descriptive_not_equal_budget_comparison"})
    payloads = {"predictions.csv": csv_bytes(prediction_rows), "comparison.csv": csv_bytes(comparison)}
    input_names = ["snapshot_manifest.json", "manifest.json", "data_admission.json",
                   "additive_positive_head/best.pt", "additive_positive_head/validation_predictions.csv",
                   "current_cohort_arrays.npz", *E2_HASHES]
    summary = {"version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed", "scope": "historical_descriptive", "variant": VARIANT, "seed": 20261116,
        "best_epoch_frozen_before_evaluation": 38, "checkpoint_sha256": BEST_SHA256,
        "result_source_commit": RESULT_COMMIT, "E2_source_commit": E2_COMMIT,
        "frozen_checkpoint_sha256": BEST_SHA256, "source_result_commit": RESULT_COMMIT,
        "roundtrip_max_absolute_error": roundtrip_error, "evaluation_n": 608, "no_training": True,
        "data_npz_sha256": NPZ_SHA256, "source_run_id": manifest["run_id"],
        "source_hashes": manifest["source_hashes"],
        "input_hashes": {n: digest(inputs / n) for n in input_names},
        "target": e2_admission["target"], "prediction_units": "sigma7_original",
        "validation_roundtrip": {"examples": 116, "maximum_absolute_error": roundtrip_error,
                                  "tolerance": ROUNDTRIP_TOLERANCE, "passed_before_test_members_opened": True},
        "evaluation": {"first": test_dates[0], "last": test_dates[-1], "examples": 608,
                       **scored, "below_floor_count": int((prediction < floor).sum())},
        "training": {"new_model_fits": 0, "new_scaler_fits": 0, "optimizer_steps": 0,
                     "bias_matching_calls": 0, "calibration_fits": 0,
                     "E3_examples": 532, "E2_neural_examples": 655,
                     "E2_classical_training_counts_claimed": False, "equal_training_budget_claimed": False},
        "members_read": ["training_dates", "training_raw", "training_y_original",
                         "evaluation_dates", "evaluation_raw", "evaluation_y_original"],
        "calibration_members_opened": False, "rows_dropped": 0,
        "statistical_tests_performed": False, "R1_R3_verified": False, "winner_declared": False,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "numpy": np.__version__, "device": device,
                    "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                    "cuda_build": torch.version.cuda, "cpu_threads": torch.get_num_threads(),
                    "AMP": False, "TF32": False, "inference_seconds": elapsed},
        "limits": ["Known historical test calendar; prospective blinding is not recovered.",
                   "E3 was selected from exposed development results and has one seed.",
                   "E2 neural metrics average six seed metrics; E3 uses one frozen trajectory.",
                   "Training samples, tuning histories and computation budgets are different.",
                   "This is descriptive inference, not a confirmatory superiority or novelty decision."],
        "artifact_sha256": {n: hashlib.sha256(v).hexdigest() for n, v in payloads.items()}}
    payloads["summary.json"] = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False).encode() + b"\n"
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        with (output / name).open("xb") as stream:
            stream.write(payload)
    print("Evaluación descriptiva completada: 608 fechas; pesos y escalador conservados.", flush=True)
    print(json.dumps({"RMSE": scored["RMSE"], "MAE": scored["MAE"],
                      "output": str(output)}, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--roundtrip-only", action="store_true",
                        help="Reproduce frozen validation only; do not open evaluation members or write outputs")
    args = parser.parse_args()
    evaluate(args.source_root, args.inputs, args.output, args.device, roundtrip_only=args.roundtrip_only)


if __name__ == "__main__":
    main()
