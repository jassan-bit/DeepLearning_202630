"""Measure frozen E2 predictors without fitting, selection or recalibration.

Prepare an immutable specification first, then run predictions and CPU latency
as separate, externally budgeted phases. Preparation never deserializes weights.
The latency phase requires a completed prediction/parity report from this spec.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback

VERSION = "e2-corrections-frozen-inference-v2-eager"
SCIENCE_COMMIT = "1e8745497b484518f766b6da5c19be8362d9add7"
MODELS = ("FCN-1D", "ResNet-1D", "Inception-1D", "LSTM-autoencoder con predictor",
          "TimesNet", "TSMixer", "TiDE", "TimeMixer", "PatchTST", "iTransformer",
          "MLP16", "LSTM32", "GRU32")
SEEDS = tuple(range(271801, 271807))
BASELINE = "GRU32"
TOLERANCE = 1e-7
DELTA = 0.00025772842664981106
TARGET_TOLERANCE = 1e-12
ARTIFACTS = ("checkpoint", "scaler", "calibrator", "calibration_predictions", "test_predictions")
AUTHOR_MODELS = frozenset(("TSMixer", "TiDE", "TimeMixer", "TimesNet", "PatchTST", "iTransformer"))
FROZEN = "book_e2/_static/e2_flops_final_audit_20261114/frozen_architecture_inputs.json"
FROZEN_SHA = "da004139cfc4663ce1d9a13c88a422df8865b53f6448fbdf015da0244f9f51cf"
DATA_SOURCES = {
    "experiments/dl_r2_cv_20261104/protocol.json": "e8da0f0098acd24a58664b83d0ba499b43aa9d597a36c52da6b3325c2c3c7bec",
    "results/eda_development/daily_market_development.csv": "bf6bb5160af924aef6a15c3684aeae932df3017e8639888bb7e91ce787900171",
    "results/eda_development/daily_volatility_development.csv": "20b96374a659eb87dadfa68aa3d6b6b5fb6bea5b10eebe6f2c186ad69861844a",
}
EXTRA_SOURCE_FILES = (
    "scripts/e2_standalone_code.py", "scripts/e2_development_data.py",
    "scripts/e2_reduced_final.py", "scripts/e2_temporal_training.py",
    "scripts/e2_current_models.py", "scripts/e2_real_executor.py",
    "scripts/e2_search_plan.py", "scripts/e2_nas_factory.py",
    "scripts/e2_reduced_protocol.py", "scripts/e2_staged_training.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return sha256(canonical(value)).hexdigest()


def file_sha(path):
    result = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(value) + b"\n"
    if path.exists():
        require(path.read_bytes() == data, "Immutable JSON already exists with different bytes: " + str(path))
        return
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def regular_file(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Missing regular file: " + str(path))
    return path.resolve()


def literal_constants(path):
    wanted = {"CONSTRUCTOR_SHA256", "METADATA_SHA256"}
    result = {}
    for item in ast.parse(Path(path).read_text(encoding="utf-8")).body:
        if (isinstance(item, ast.Assign) and len(item.targets) == 1
                and isinstance(item.targets[0], ast.Name) and item.targets[0].id in wanted):
            result[item.targets[0].id] = ast.literal_eval(item.value)
    require(set(result) == wanted, "Pinned standalone constructor hashes are missing")
    require(len(result["CONSTRUCTOR_SHA256"]) == 12 and len(result["METADATA_SHA256"]) == 4,
            "Pinned constructor/metadata domain changed")
    return result


def checked_index(path, frozen):
    path = regular_file(path)
    payload = read_json(path)
    require(payload.get("status") == "verified" and payload.get("scientific_commit") == SCIENCE_COMMIT,
            "Use the completed, verified recovery index for the fixed scientific source")
    rows = payload.get("rows")
    require(isinstance(rows, list), "checkpoint_index.json must contain rows")
    asset_root = path.parent.parent if path.parent.name == "metadata" else path.parent
    selected = {}
    for supplied in rows:
        if supplied.get("model") not in MODELS:
            continue
        row = dict(supplied)
        key = (row["model"], row.get("seed"))
        require(type(key[1]) is int and key[1] in SEEDS and key not in selected,
                "Duplicate or unregistered neural predictor in index: " + str(key))
        require(row.get("trial") == frozen[key[0]]["trial"], "Index trial differs from frozen recipe: " + str(key))
        expected = row.get("expectedsha")
        require(isinstance(expected, dict) and set(expected) == set(ARTIFACTS),
                "Each neural row needs the exact five-artifact expectedsha map")
        for name in ARTIFACTS:
            h = expected[name]
            require(isinstance(h, str) and len(h) == 64 and all(c in "0123456789abcdef" for c in h),
                    "Invalid artifact checksum: " + name)
            location = Path(row[name + "_path"])
            require(location.is_absolute(), "Index artifact paths must be absolute")
            require(location.resolve().is_relative_to(asset_root), "Artifact escapes the recovery directory")
            require(not location.is_symlink(), "Symlink artifacts are not admitted")
            row[name + "_path"] = str(location.resolve())
            if name + "_sha256" in row:
                require(row[name + "_sha256"] == h, "Duplicate checksum fields disagree: " + name)
        selected[key] = row
    require(set(selected) == {(m, s) for m in MODELS for s in SEEDS},
            "All thirteen recipes and six fixed seeds are required")
    return payload, [selected[m, s] for m in MODELS for s in SEEDS]


def prepare_spec(args):
    require(args.checkpoint_index and args.source_root and args.authors_root,
            "--prepare-spec requires --checkpoint-index, --source-root and --authors-root")
    source = Path(args.source_root).resolve()
    authors = Path(args.authors_root).resolve()
    require(file_sha(regular_file(source / FROZEN)) == FROZEN_SHA, "Frozen architecture JSON changed")
    entries = read_json(source / FROZEN)
    frozen = {row["model"]: row for row in entries}
    require(len(entries) == 13 and set(frozen) == set(MODELS), "Frozen architecture domain changed")
    index = regular_file(args.checkpoint_index)
    _, rows = checked_index(index, frozen)
    pinned = literal_constants(source / "scripts/e2_standalone_code.py")
    source_hashes = {**pinned["CONSTRUCTOR_SHA256"], **pinned["METADATA_SHA256"], FROZEN: FROZEN_SHA}
    for relative, expected in source_hashes.items():
        require(file_sha(regular_file(source / relative)) == expected, "Original scientific file changed: " + relative)
    for relative in EXTRA_SOURCE_FILES:
        source_hashes[relative] = file_sha(regular_file(source / relative))
    design = read_json(source / "experiments/e2_completion_20261106/design.json")
    author_rows = [r for r in design["author_sources"] if r["model"] in AUTHOR_MODELS]
    require(len(author_rows) == 76, "Expected exactly 76 pinned author files")
    for row in author_rows:
        path = regular_file(authors / row["model"] / row["path"])
        require(path.stat().st_size == row["bytes"] and file_sha(path) == row["sha256"],
                "Pinned author source changed: " + row["model"] + "/" + row["path"])
    for relative, expected in DATA_SOURCES.items():
        require(file_sha(regular_file(source / relative)) == expected, "Development source changed: " + relative)
    plan = None
    if args.statistical_plan:
        path = regular_file(args.statistical_plan)
        plan = {"path": str(path), "sha256": file_sha(path)}
    require(args.previous_attempt is not None, "The eager correction requires its preserved previous-attempt lineage")
    previous_path = regular_file(args.previous_attempt)
    previous = read_json(previous_path)
    require(previous.get("prediction_tolerance_unchanged") == TOLERANCE,
            "The fixed calibration tolerance must not change")
    for archived in previous["preserved_files"]:
        require(file_sha(regular_file(archived["path"])) == archived["sha256"], "Preserved v1 evidence changed")
    spec = {
        "version": VERSION, "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner_path": str(Path(__file__).resolve()), "runner_sha256": file_sha(__file__),
        "scientific_source_commit": SCIENCE_COMMIT, "source_root": str(source),
        "source_files_sha256": source_hashes, "authors_root": str(authors),
        "author_files": author_rows, "development_sources_sha256": DATA_SOURCES,
        "checkpoint_index": str(index), "checkpoint_index_sha256": file_sha(index),
        "checkpoint_lineage": "Recovered historical clean-reproduction checkpoints, not the serialized files in the original final_freeze. Historical producer receipts record equality to original model states and predictions; this run preserves their own checksums and does not replace the original freeze.",
        "neural_rows": rows, "model_order": list(MODELS), "seeds": list(SEEDS),
        "statistical_plan": plan,
        "compatibility_correction": {"lineage_path": str(previous_path), "lineage_sha256": file_sha(previous_path),
            "policy": "One predeclared alternative: MKLDNN disabled globally before model construction and every forward; eager PyTorch CPU recurrence. No tolerance, weight, data or scaling change.",
            "stop_rule": "If any calibration parity check fails, stop; do not try another backend or enlarge tolerance."},
        "prediction_policy": {
            "device": "cpu", "torch_version": "2.8.0+cpu", "numpy_version": "2.3.5",
            "pandas_version": "2.2.3", "dtype": "float32", "batch_size": 1,
            "cpu_threads": 2, "interop_threads": 1, "eval": True, "inference_mode": True,
            "deterministic_algorithms": True, "autocast": False,
            "mkldnn_enabled": False, "cpu_recurrent_backend": "eager_pytorch",
            "train_count": 655, "train_label_end": "2024-10-31", "calibration_count": 53,
            "calibration_dates": ["2024-11-08", "2024-12-30"],
            "calibration_prediction_absolute_tolerance": TOLERANCE,
            "calibration_target_absolute_tolerance": TARGET_TOLERANCE,
            "tolerance_reason": "FP32 CPU versus recorded A100 arithmetic; absolute sigma7 tolerance fixed before inference, much smaller than delta; not a statistical equivalence margin",
            "delta_reference": DELTA, "abort_on_any_calibration_mismatch": True,
            "original_units": "population sigma7, nonannualized", "risk": "RMSE computed in float64 from batch-one original-unit predictions",
            "state_dict_must_remain_identical": True, "test_inference": False,
        },
        "latency_policy": {
            "candidate_order": list(MODELS[:10]), "baseline": BASELINE,
            "blocks": 100, "seeds_per_block": list(SEEDS), "pairs_per_candidate": 600,
            "train_input_indices": [(i * 654) // 99 for i in range(100)],
            "input_rule": "floor(block_id*654/99), common raw development window for candidate/reference and all six seeds; frozen model-specific transforms remain legitimate",
            "warmup_calls_per_predictor": 32, "warmup_input_blocks": list(range(32)),
            "order_rule": "candidate-first iff (block_id+seed_position+candidate_position)%2==0; otherwise reference-first",
            "clock": "perf_counter_ns", "timestamp": "time_ns", "timed_scope": "CPU model forward call on an already resident batch-one FP32 tensor only",
            "excluded_scope": "loading, preprocessing, serialization, finite checks, compilation and warmup; no GPU or service latency claim",
            "estimand": "equal-seed mean of (candidate mean forward latency minus GRU32 mean forward latency), in seconds",
            "statistical_unit": "whole global timing block containing all six seed pairs; preserve blocks for subsequent resampling",
            "bootstrap_or_hypothesis_tests_in_this_runner": False,
            "hardware_scope": "new secondary CPU measurement, not a replacement for historical A100 costs",
            "mkldnn_enabled": False, "backend_scope": "The same eager CPU backend is fixed for every candidate/reference and seed.",
        },
        "prohibited_operations": ["optimizer creation", "model fitting", "checkpoint updates", "calibrator refit", "new ranking or selection", "old campaign controller", "test prediction generation"],
        "budget_policy": "Each --run invocation must be launched through the external additive corrections budget wrapper; this runner never resets or writes the campaign ledger.",
    }
    write_new_json(args.spec, spec)
    return {"status": "prepared", "spec_path": str(Path(args.spec).resolve()),
            "spec_sha256": file_sha(args.spec), "predictors": 78,
            "weights_deserialized": False, "model_forwards": 0}


def verify_inputs(spec):
    require(spec.get("version") == VERSION and spec.get("model_order") == list(MODELS)
            and spec.get("seeds") == list(SEEDS), "Unknown specification domain")
    require(file_sha(__file__) == spec["runner_sha256"], "Runner changed after specification")
    require(file_sha(regular_file(spec["checkpoint_index"])) == spec["checkpoint_index_sha256"], "Recovery index changed after specification")
    source = Path(spec["source_root"])
    for relative, expected in {**spec["source_files_sha256"], **spec["development_sources_sha256"]}.items():
        require(file_sha(regular_file(source / relative)) == expected, "Source changed after specification: " + relative)
    for row in spec["author_files"]:
        path = regular_file(Path(spec["authors_root"]) / row["model"] / row["path"])
        require(path.stat().st_size == row["bytes"] and file_sha(path) == row["sha256"], "Author file changed")
    if spec["statistical_plan"]:
        require(file_sha(regular_file(spec["statistical_plan"]["path"])) == spec["statistical_plan"]["sha256"], "Statistical plan changed")
    correction = spec["compatibility_correction"]
    require(file_sha(regular_file(correction["lineage_path"])) == correction["lineage_sha256"], "Correction lineage changed")
    for archived in read_json(correction["lineage_path"])["preserved_files"]:
        require(file_sha(regular_file(archived["path"])) == archived["sha256"], "Preserved failed attempt changed")
    require(len(spec["neural_rows"]) == 78, "Incomplete neural specification")
    for row in spec["neural_rows"]:
        for name in ARTIFACTS:
            require(file_sha(regular_file(row[name + "_path"])) == row["expectedsha"][name],
                    "Recovered artifact changed: " + str((row["model"], row["seed"], name)))


def runtime_and_data(spec):
    import numpy as np
    import pandas as pd
    import torch
    require(torch.__version__ == "2.8.0+cpu" and torch.version.cuda is None, "Use the pinned Torch 2.8.0+cpu environment")
    require(np.__version__ == "2.3.5" and pd.__version__ == "2.2.3", "Use the pinned NumPy/pandas transform environment")
    require(spec["prediction_policy"].get("mkldnn_enabled") is False
            and spec["latency_policy"].get("mkldnn_enabled") is False, "The eager backend must be fixed in both phase policies")
    torch.backends.mkldnn.enabled = False
    require(sys.prefix != sys.base_prefix, "Use an isolated Python venv")
    cfg = Path(sys.prefix) / "pyvenv.cfg"
    require(cfg.is_file() and "include-system-site-packages = false" in cfg.read_text().lower(), "System site packages are not admitted")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision("highest")
    require(torch.get_default_dtype() == torch.float32, "Default tensor dtype must be FP32")
    sys.path.insert(0, str(Path(spec["source_root"]) / "scripts"))
    from e2_development_data import load_development
    from e2_reduced_final import final_training_split, _calibration_split
    panel, admission = load_development(spec["source_root"], "2024-12-30")
    train = final_training_split(panel)
    calibration = _calibration_split(panel, train.scaler)
    require(len(train.x) == 655 and len(calibration.x) == 53, "Frozen development support changed")
    runtime = {"python": platform.python_version(), "python_executable": sys.executable,
               "venv": sys.prefix, "system_site_packages": False, "torch": torch.__version__,
               "numpy": np.__version__, "pandas": pd.__version__, "device": "cpu",
               "platform": platform.platform(), "machine": platform.machine(),
               "processor": platform.processor(), "logical_cpus": os.cpu_count(),
               "affinity_cpus": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
               "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
               "dtype": "float32", "mkldnn_enabled": torch.backends.mkldnn.enabled,
               "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
               "matmul_precision": torch.get_float32_matmul_precision(), "cuda": torch.version.cuda}
    return torch, np, train, calibration, admission, runtime


def tensor_sha(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return digest({"dtype": str(tensor.dtype), "shape": list(tensor.shape),
                   "bytes_sha256": sha256(tensor.reshape(-1).view(__import__("torch").uint8).numpy().tobytes()).hexdigest()})


def scalar_sha(values, np):
    array = np.asarray(values, dtype="<f8")
    return sha256(array.tobytes()).hexdigest()


def load_model(row, train, roots, torch):
    from e2_reduced_fit import _builder
    from e2_staged_training import state_sha256
    scaler = read_json(row["scaler_path"])
    require(scaler == train.scaler.record(), "Frozen scaler differs from reconstructed train-only scaler: " + row["model"])
    saved = torch.load(row["checkpoint_path"], map_location="cpu", weights_only=True)
    require(isinstance(saved, dict) and "state_dict" in saved and saved.get("scaler") == scaler,
            "Checkpoint/scaler binding differs")
    model = _builder(row["model"], row["trial"], row["seed"], roots).cpu().eval()
    model.load_state_dict(saved["state_dict"], strict=True)
    require(all(p.dtype == torch.float32 for p in model.parameters()), "Non-FP32 parameter")
    before = state_sha256(saved["state_dict"])
    require(before == state_sha256(model), "Loaded model differs from checkpoint state")
    model.requires_grad_(False)
    return model, before


def predict(model, x, scale, torch, np):
    values = []
    with torch.inference_mode(), torch.autocast(device_type="cpu", enabled=False):
        for i in range(len(x)):
            result = model(x[i:i + 1])
            require(tuple(result.shape) == (1,) and bool(torch.isfinite(result).all())
                    and bool((result >= 0).all()), "Invalid frozen scalar prediction")
            values.append(float(result.item()))
    # Preserve the recorded pipeline's FP32 array times target-scale semantics.
    restored = np.asarray(values, dtype=np.float32) * scale
    require(np.isfinite(restored).all(), "Nonfinite original-unit prediction")
    return [float(value) for value in restored]


def csv_open(path, fields):
    stream = Path(path).open("x", encoding="utf-8", newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    return stream, writer


def common_identity(row):
    return {"model": row["model"], "seed": row["seed"], "checkpoint_sha256": row["expectedsha"]["checkpoint"],
            "scaler_sha256": row["expectedsha"]["scaler"], "config_sha256": digest(row["trial"])}


def write_predictions(writer, row, split, prepared, values):
    identity = common_identity(row)
    for date, truth, prediction in zip(prepared.dates, prepared.y_original, values):
        error = prediction - truth
        writer.writerow({**identity, "date": date.strftime("%Y-%m-%d"), "split": split,
                         "y_true": truth, "prediction": prediction, "signed_error": error,
                         "absolute_error": abs(error), "squared_error": error * error,
                         "batch_size": 1, "device": "cpu", "eval_mode": True, "inference_mode": True})


def rmse(truth, values):
    return math.sqrt(math.fsum((p - y) ** 2 for p, y in zip(values, truth)) / len(truth))


def output_hashes(output):
    return {p.name: {"bytes": p.stat().st_size, "sha256": file_sha(p)}
            for p in sorted(Path(output).iterdir()) if p.is_file() and p.name not in ("inference_report.json", "latency_report.json", "failure_report.json")}


def progress(handle, value):
    handle.write(canonical(value).decode() + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def prediction_stage(spec, spec_sha, output):
    torch, np, train, calibration, admission, runtime = runtime_and_data(spec)
    roots = {m: str(Path(spec["authors_root"]) / m) for m in AUTHOR_MODELS}
    fields = ["model", "seed", "date", "split", "y_true", "prediction", "signed_error", "absolute_error", "squared_error",
              "batch_size", "device", "eval_mode", "inference_mode", "checkpoint_sha256", "scaler_sha256", "config_sha256"]
    train_stream, train_writer = csv_open(output / "predictions_train.csv", fields)
    cal_stream, cal_writer = csv_open(output / "predictions_calibration.csv", fields)
    agreement_stream, agreement_writer = csv_open(output / "calibration_agreement.csv", ["model", "seed", "max_absolute_prediction_difference", "absolute_tolerance", "passed", "reference_sha256", "checkpoint_sha256"])
    risk_stream, risk_writer = csv_open(output / "risks_by_predictor.csv", ["model", "seed", "train_rmse", "calibration_rmse", "calibration_minus_train_rmse", "train_count", "calibration_count", "batch_size", "device", "checkpoint_sha256", "scaler_sha256", "config_sha256", "state_before_sha256", "state_after_sha256"])
    log = (output / "progress.jsonl").open("x", encoding="utf-8")
    agreements, bindings = [], []
    try:
        for row in spec["neural_rows"]:
            model, before = load_model(row, train, roots, torch)
            saved_cal = read_json(row["calibration_predictions_path"])
            dates = [d.strftime("%Y-%m-%d") for d in calibration.dates]
            require(saved_cal.get("dates") == dates, "Saved calibration dates changed")
            reference = saved_cal.get("prediction_original")
            require(isinstance(reference, list) and len(reference) == 53 and all(math.isfinite(v) for v in reference), "Invalid saved calibration trajectory")
            require(saved_cal.get("model", row["model"]) == row["model"]
                    and saved_cal.get("seed", row["seed"]) == row["seed"], "Saved calibration predictor identity changed")
            require(isinstance(saved_cal.get("target_original"), list) and len(saved_cal["target_original"]) == 53,
                    "Saved calibration targets are required")
            if "target_original" in saved_cal:
                require(len(saved_cal["target_original"]) == 53 and max(abs(a - b) for a, b in zip(saved_cal["target_original"], calibration.y_original)) <= TARGET_TOLERANCE, "Saved calibration targets changed")
            if "checkpoint_sha256" in saved_cal:
                require(saved_cal["checkpoint_sha256"] == row["expectedsha"]["checkpoint"], "Calibration predictions bind another checkpoint")
            calibrator = read_json(row["calibrator_path"])
            require(calibrator.get("predictor_sha256") == row["expectedsha"]["checkpoint"] and calibrator.get("dates") == dates,
                    "Frozen calibrator/checkpoint binding changed")
            require(calibrator.get("sha256") == digest({k: v for k, v in calibrator.items() if k != "sha256"}), "Frozen calibrator integrity changed")
            require(calibrator.get("calibration_predictions_sha256") == digest(reference),
                    "Frozen calibrator does not bind the saved calibration trajectory")
            require(calibrator.get("calibration_targets_sha256") == digest(saved_cal["target_original"]),
                    "Frozen calibrator does not bind the saved calibration targets")
            values_cal = predict(model, calibration.x, train.scaler.target_scale, torch, np)
            difference = max(abs(a - b) for a, b in zip(values_cal, reference))
            agreement = {"model": row["model"], "seed": row["seed"], "max_absolute_prediction_difference": difference,
                         "absolute_tolerance": TOLERANCE, "passed": difference <= TOLERANCE,
                         "reference_sha256": row["expectedsha"]["calibration_predictions"], "checkpoint_sha256": row["expectedsha"]["checkpoint"]}
            agreement_writer.writerow(agreement)
            agreement_stream.flush()
            progress(log, {"stage": "calibration_parity", **agreement})
            require(agreement["passed"], "CPU calibration parity exceeds fixed tolerance: " + str((row["model"], row["seed"], difference)))
            values_train = predict(model, train.x, train.scaler.target_scale, torch, np)
            from e2_staged_training import state_sha256
            after = state_sha256(model)
            require(before == after, "Frozen model state changed during inference")
            write_predictions(train_writer, row, "train", train, values_train)
            write_predictions(cal_writer, row, "calibration", calibration, values_cal)
            train_risk, cal_risk = rmse(train.y_original, values_train), rmse(calibration.y_original, values_cal)
            risk_writer.writerow({**common_identity(row), "train_rmse": train_risk, "calibration_rmse": cal_risk,
                                  "calibration_minus_train_rmse": cal_risk - train_risk, "train_count": 655,
                                  "calibration_count": 53, "batch_size": 1, "device": "cpu",
                                  "state_before_sha256": before, "state_after_sha256": after})
            for stream in (train_stream, cal_stream, risk_stream):
                stream.flush()
            bindings.append({**common_identity(row), "state_before_sha256": before, "state_after_sha256": after,
                             "checkpoint_lineage": "historical_clean_reproduction",
                             "source_result_sha256": row.get("source_result_sha256"),
                             "source_manifest_sha256": row.get("source_manifest_sha256"),
                             "recorded_original_prediction_max_difference": row.get("recorded_original_prediction_max_difference"),
                             "train_prediction_float64_sha256": scalar_sha(values_train, np),
                             "calibration_prediction_float64_sha256": scalar_sha(values_cal, np)})
            agreements.append(agreement)
            progress(log, {"stage": "predictor_completed", **common_identity(row), "state_unchanged": True})
            del model
    finally:
        for stream in (train_stream, cal_stream, agreement_stream, risk_stream, log):
            stream.close()
    transformations = {"train_signature": train.signature, "calibration_signature": calibration.signature,
                       "scaler_record": train.scaler.record(), "scaler_canonical_sha256": digest(train.scaler.record()),
                       "train_x_fp32_sha256": tensor_sha(train.x), "calibration_x_fp32_sha256": tensor_sha(calibration.x),
                       "train_y_original_float64_sha256": scalar_sha(train.y_original, np),
                       "calibration_y_original_float64_sha256": scalar_sha(calibration.y_original, np),
                       "train_dates": [d.strftime("%Y-%m-%d") for d in train.dates],
                       "calibration_dates": [d.strftime("%Y-%m-%d") for d in calibration.dates],
                       "train_mask": [True] * 655, "calibration_mask": [True] * 53,
                       "mask_scope": "eligible dates listed explicitly; full source calendar and incomplete-day policy remain in development admission"}
    write_new_json(output / "transforms_and_support.json", transformations)
    write_new_json(output / "development_admission.json", admission)
    write_new_json(output / "predictor_bindings.json", bindings)
    report = {"version": VERSION, "stage": "predictions", "status": "passed", "spec_sha256": spec_sha,
              "checkpoint_index_sha256": spec["checkpoint_index_sha256"], "runtime": runtime,
              "predictors_completed": len(agreements), "calibration_checks_passed": len(agreements),
              "maximum_calibration_absolute_difference": max(r["max_absolute_prediction_difference"] for r in agreements),
              "calibration_absolute_tolerance": TOLERANCE, "train_prediction_rows": 78 * 655,
              "calibration_prediction_rows": 78 * 53, "batch_size": 1, "states_unchanged": True,
              "model_forward_calls": 78 * (655 + 53),
              "new_model_fits": 0, "optimizers_created": 0, "calibrators_refitted": 0,
              "test_predictions_generated": 0, "new_model_ranking": False,
              "checkpoint_lineage": spec["checkpoint_lineage"], "original_final_freeze_modified": False,
              "statistical_tests_executed": 0, "all_output_files": output_hashes(output)}
    write_new_json(output / "inference_report.json", report)
    return report


def require_prediction_report(directory, spec_sha):
    directory = Path(directory).resolve()
    report = read_json(regular_file(directory / "inference_report.json"))
    require(report.get("version") == VERSION and report.get("stage") == "predictions"
            and report.get("status") == "passed" and report.get("spec_sha256") == spec_sha
            and report.get("predictors_completed") == 78 and report.get("calibration_checks_passed") == 78,
            "Latency requires all 78 prediction/parity checks from this immutable specification")
    for relative, row in report["all_output_files"].items():
        require(Path(relative).name == relative, "Unsafe prior output path")
        path = regular_file(directory / relative)
        require(path.stat().st_size == row["bytes"] and file_sha(path) == row["sha256"], "Prior prediction output changed")
    return {"path": str(directory / "inference_report.json"), "sha256": file_sha(directory / "inference_report.json")}


def latency_stage(spec, spec_sha, output, predictions_output):
    previous = require_prediction_report(predictions_output, spec_sha)
    torch, np, train, calibration, admission, runtime = runtime_and_data(spec)
    roots = {m: str(Path(spec["authors_root"]) / m) for m in AUTHOR_MODELS}
    needed = set(MODELS[:10]) | {BASELINE}
    loaded, states, identities = {}, {}, {}
    from e2_staged_training import state_sha256
    for row in spec["neural_rows"]:
        if row["model"] in needed:
            key = row["model"], row["seed"]
            loaded[key], states[key] = load_model(row, train, roots, torch)
            identities[key] = common_identity(row)
    inputs = [train.x[i:i + 1].clone() for i in spec["latency_policy"]["train_input_indices"]]
    stream, writer = csv_open(output / "raw_cpu_latency.csv", ["candidate", "baseline", "block_id", "seed", "pair_id", "order", "position", "role", "model", "input_train_index", "input_date", "start_time_ns", "start_perf_counter_ns", "end_perf_counter_ns", "elapsed_ns", "checkpoint_sha256", "scaler_sha256", "config_sha256", "batch_size", "device"])
    log = (output / "progress.jsonl").open("x", encoding="utf-8")
    count = 0
    try:
        with torch.inference_mode(), torch.autocast(device_type="cpu", enabled=False):
            for key, model in loaded.items():
                for block in range(32):
                    result = model(inputs[block])
                    require(tuple(result.shape) == (1,) and bool(torch.isfinite(result).all()), "Invalid warmup output")
            progress(log, {"stage": "warmup_completed", "predictors": len(loaded), "calls_per_predictor": 32})
            for block in range(100):
                x = inputs[block]
                input_index = spec["latency_policy"]["train_input_indices"][block]
                for seed_position, seed in enumerate(SEEDS):
                    for candidate_position, candidate in enumerate(MODELS[:10]):
                        candidate_first = (block + seed_position + candidate_position) % 2 == 0
                        order = [("candidate", candidate), ("reference", BASELINE)]
                        if not candidate_first:
                            order.reverse()
                        pair_id = candidate + ":" + str(seed) + ":" + str(block)
                        for position, (role, name) in enumerate(order):
                            model = loaded[name, seed]
                            start_time = time.time_ns()
                            start = time.perf_counter_ns()
                            result = model(x)
                            end = time.perf_counter_ns()
                            require(tuple(result.shape) == (1,) and bool(torch.isfinite(result).all()), "Invalid timed output")
                            writer.writerow({**identities[name, seed], "candidate": candidate, "baseline": BASELINE,
                                             "block_id": block, "pair_id": pair_id, "order": "AB" if candidate_first else "BA",
                                             "position": position, "role": role, "input_train_index": input_index,
                                             "input_date": train.dates[input_index].strftime("%Y-%m-%d"),
                                             "start_time_ns": start_time, "start_perf_counter_ns": start,
                                             "end_perf_counter_ns": end, "elapsed_ns": end - start,
                                             "batch_size": 1, "device": "cpu"})
                            count += 1
                stream.flush()
                progress(log, {"stage": "timing_block_completed", "block_id": block, "raw_operations": count})
    finally:
        stream.close()
        log.close()
    require(count == 12000, "Incomplete timing domain")
    state_checks = []
    for key, model in loaded.items():
        after = state_sha256(model)
        require(after == states[key], "Model state changed during timed inference")
        state_checks.append({**identities[key], "state_before_sha256": states[key], "state_after_sha256": after})
    write_new_json(output / "latency_design.json", spec["latency_policy"])
    write_new_json(output / "predictor_bindings.json", state_checks)
    report = {"version": VERSION, "stage": "latency", "status": "passed", "spec_sha256": spec_sha,
              "checkpoint_index_sha256": spec["checkpoint_index_sha256"], "runtime": runtime,
              "prediction_parity_report": previous, "raw_timed_operations": count, "paired_observations": 6000,
              "blocks": 100, "seed_pairs_per_candidate_per_block": 6, "candidates": 10,
              "warmup_calls_per_predictor": 32, "states_unchanged": True,
              "warmup_forward_calls": 66 * 32, "total_model_forward_calls": 12000 + 66 * 32,
              "estimand": spec["latency_policy"]["estimand"], "resampling_unit": spec["latency_policy"]["statistical_unit"],
              "statistical_tests_executed": 0, "new_model_fits": 0, "optimizers_created": 0,
              "calibrators_refitted": 0, "new_model_ranking": False, "A100_cost_replaced": False,
              "checkpoint_lineage": spec["checkpoint_lineage"], "original_final_freeze_modified": False,
              "all_output_files": output_hashes(output)}
    write_new_json(output / "latency_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-spec", action="store_true")
    modes.add_argument("--run", action="store_true")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--checkpoint-index", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--authors-root", type=Path)
    parser.add_argument("--statistical-plan", type=Path)
    parser.add_argument("--previous-attempt", type=Path)
    parser.add_argument("--stage", choices=("predictions", "latency"), default="predictions")
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = None
    started = time.time_ns()
    try:
        if args.prepare_spec:
            result = prepare_spec(args)
        else:
            require(args.output is not None, "--run requires a fresh --output directory")
            require(not any((args.checkpoint_index, args.source_root, args.authors_root, args.statistical_plan, args.previous_attempt)),
                    "Execution inputs come only from the immutable specification")
            spec = read_json(regular_file(args.spec))
            spec_sha = file_sha(args.spec)
            verify_inputs(spec)
            output = args.output.resolve()
            require(not output.exists(), "Preserve previous attempts; use a new output directory")
            output.mkdir(parents=True)
            write_new_json(output / "run_admission.json", {"version": VERSION, "stage": args.stage,
                "spec_path": str(args.spec.resolve()), "spec_sha256": spec_sha,
                "started_time_ns": started, "new_model_fits_authorized": False})
            if args.stage == "predictions":
                result = prediction_stage(spec, spec_sha, output)
            else:
                require(args.predictions_output is not None, "--stage latency requires --predictions-output")
                result = latency_stage(spec, spec_sha, output, args.predictions_output)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
        return 0
    except Exception as error:
        report = {"version": VERSION, "status": "failed", "stage": args.stage,
                  "error_type": type(error).__name__, "error": str(error),
                  "started_time_ns": started, "failed_time_ns": time.time_ns(),
                  "new_model_fits": 0, "optimizers_created": 0, "calibrators_refitted": 0,
                  "partial_outputs_preserved": output is not None and output.exists(),
                  "traceback": traceback.format_exc()}
        if output is not None and output.exists():
            write_new_json(output / "failure_report.json", report)
        print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
