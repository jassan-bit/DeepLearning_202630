"""Export four existing E2 seed-271801 checkpoints; no fitting or recalibration.

CPU only, ONNX batch-one contract. All frozen inputs and source bodies are
verified before deserialization. Inputs remain outside the publishable payload.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter
import warnings

import numpy as np
import onnx
import onnxruntime as ort
import torch

SEED = 271801
TOLERANCE = 1e-6
INPUT_HASHES = {
    "npz": "186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683",
    "predictions": "df5206e531ad9c4e5718fb8f17d86390bb1267bcd1d724c2333cacd6c9903655",
    "protocol": "5195337af13f4193da16bf4499c8396586c621d4ab084a5b296a297f2141edeb",
}
SOURCE_HASHES = {
    "scripts/e2_training_models.py": "46be0644e0ed9d966163b17f171092e81b6a82dd734efc7954d6104f5ee5360b",
    "scripts/e2_recipe_models.py": "47380cc87e2d4b4221ee4fc97da3286990470b3304dd1d9ef990ace51c338b75",
    "scripts/e2_published_registry.py": "0704784aa0bb2887908d12ad12345094c8f6a707918aa695dbc11cda88092fec",
    "scripts/e2_published_sae.py": "0436d73f9724bb065dc3642f18218cb35cda5e9abfa4de5090f8c5d69eb6c1aa",
    "scripts/e2_search_plan.py": "7169d56f67a07b3ab6d55bd1730dabdec5fe4a54e1f3385ba0db9e84cc3a2314",
    "scripts/e2_reduced_fit.py": "64e83d0a2a39920ae470252b8eb81c8f78429817fed978bd22f64866fdfb8f5f",
    "scripts/e2_staged_training.py": "b66ac8d2beb3748f4934e55e99514f53c52eada4a4dbd2e4b597d080c9f4722b",
    "experiments/e2_published_architectures_20261112/architecture_policy.json": "a1c9440046af34c2b70bef9c05229dff2c6d863aa06ae191ed02c2de6d8bf0f2",
}
MODELS = {
    "FCN-1D": {"slug": "fcn_1d", "key": "759a1b80df7f02c7", "selection": "FCN-1D", "parameters": 268929,
        "checkpoint": "8d64ff057a66fd9f82b209a16bb2927aad118a9e8fee46e9baf5fc97b9f1ecc0", "bytes": 1101214},
    "ResNet-1D": {"slug": "resnet_1d", "key": "ee2248ba4eaf9fef", "selection": "ResNet-1D", "parameters": 522955,
        "checkpoint": "f3332149da587f1c9af5ce4cb302298c2a5f8361a5986062e5cfbcc7ac8ca253", "bytes": 2144018},
    "Inception-1D": {"slug": "inception_1d", "key": "8a6bc9d3fddf7474", "selection": "Inception-1D", "parameters": 490561,
        "checkpoint": "47fcee9055df2ca911cd080d871c66b3e6764b166e2b592ad6abe57e83321d97", "bytes": 2007109},
    "LSTM-autoencoder con predictor": {"slug": "lstm_autoencoder", "key": "9fc03d0694186e23", "selection": "LSTM-autoencoder-con-predictor", "parameters": 8739,
        "checkpoint": "75210b30963879444901e3e9df6764108146d235f656b2b7aa8a390d2b932ebd", "bytes": 54341},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def transform(raw, scaler):
    """Use the existing train-only scaler, without clipping or fitting it."""
    raw = np.asarray(raw, dtype=np.float64)
    require(raw.ndim == 3 and raw.shape[1:] == (7, 5), "Invalid OHLCV shape")
    require(np.isfinite(raw).all() and (raw[:, :, :4] > 0).all() and (raw[:, :, 4] >= 0).all(), "Invalid raw OHLCV")
    relative = raw.copy()
    relative[:, :, :4] = np.log(raw[:, :, :4]) - np.log(raw[:, -1:, 3:4])
    relative[:, :, 4] = np.log1p(raw[:, :, 4])
    return np.asarray((relative - scaler["mean"]) / scaler["std"], dtype=np.float32)


def _no_backward(*args, **kwargs):
    raise RuntimeError("Training/backpropagation forbidden in export-only protocol")


def run(args):
    spec = MODELS[args.model]
    for name, expected in INPUT_HASHES.items():
        require(sha(getattr(args, name)) == expected, "Frozen input mismatch: " + name)
    for relative, expected in SOURCE_HASHES.items():
        require(sha(args.repo / relative) == expected, "Source body mismatch: " + relative)
    checkpoint = args.checkpoints / "final-fit" / spec["key"] / str(SEED) / "fit/selected_model.pt"
    require(sha(checkpoint) == spec["checkpoint"] and checkpoint.stat().st_size == spec["bytes"], "Checkpoint hash/size mismatch")
    selection_path = args.selections / (spec["selection"] + ".json")
    scaler_path = args.freeze / "final-fit" / spec["key"] / str(SEED) / "scaler.json"
    calibrator_path = scaler_path.with_name("calibrator.json")
    selection = json.loads(selection_path.read_bytes())
    scaler_file = json.loads(scaler_path.read_bytes())
    cal = json.loads(calibrator_path.read_bytes())
    sys.path.insert(0, str(args.repo / "scripts"))
    from e2_reduced_fit import _builder
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.Tensor.backward = _no_backward
    torch.autograd.backward = _no_backward
    cp = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(cp["architecture"]["initialization_seed"] == SEED, "Checkpoint seed changed")
    require(selection["model_id"] == args.model, "Selection model mismatch")
    trial = selection["winner_trial"]
    require(cp["architecture"]["training_hpo_record"] == trial, "Frozen winning HPO row changed")
    model = _builder(args.model, trial, SEED, {}).eval()
    model.load_state_dict(cp["state_dict"], strict=True)
    require(not model.training and all(not m.training for m in model.modules()), "Model is not in eval mode")
    require(all(p.device.type == "cpu" for p in model.parameters()), "CPU-only export required")
    require(sum(p.numel() for p in model.parameters()) == spec["parameters"], "Parameter count changed")
    scaler = cp["scaler"]
    require(scaler == scaler_file, "Checkpoint and original scaler differ")
    prior = json.loads(args.predictions.read_bytes())
    refs = [p for p in prior["predictors"] if p["model"] == args.model and str(p["seed"]) == str(SEED)]
    require(len(refs) == 1, "Frozen predictor is not unique")
    ref = refs[0]
    require(ref["checkpoint_sha256"] == spec["checkpoint"], "Saved predictions belong to another checkpoint")
    require(cal["sha256"] == digest({k: v for k, v in cal.items() if k != "sha256"}), "Calibrator internal digest mismatch")
    require(cal["predictor_sha256"] == spec["checkpoint"] and cal["q"] == ref["calibrator_q"] and cal["n"] == 53, "Calibrator identity changed")
    require(cal["dates"] == ref["calibration_dates"], "Calibration dates changed")
    require(cal["calibration_predictions_sha256"] == digest(ref["calibration_predictions"]), "Calibration predictions hash changed")
    require(cal["calibration_targets_sha256"] == digest(ref["calibration_targets"]), "Calibration target hash changed")
    with np.load(args.npz, allow_pickle=False) as z:
        raw = np.asarray(z["evaluation_raw"], dtype=np.float64)
        y = np.asarray(z["evaluation_y_original"], dtype=np.float64)
        dates = [str(x) for x in z["evaluation_dates"]]
        fit_dates = [str(x) for x in z["training_dates"]]
        cal_raw = np.asarray(z["calibration_raw"], dtype=np.float64)
        cal_y = np.asarray(z["calibration_y_original"], dtype=np.float64)
        cal_dates = [str(x) for x in z["calibration_dates"]]
    require(raw.shape == (608, 7, 5) and cal_raw.shape == (53, 7, 5), "Unexpected cohort counts")
    require(dates == prior["dates"] and np.array_equal(y, prior["y_true"]), "Frozen evaluation dates/targets mismatch")
    require(fit_dates == scaler["train_dates"] and len(fit_dates) == 655, "Scaler is not tied to frozen 655 fit dates")
    require(cal_dates == cal["dates"] and np.array_equal(cal_y, ref["calibration_targets"]), "Frozen calibration cohort mismatch")
    x, cal_x = [transform(values, scaler) for values in (raw, cal_raw)]
    out = args.output / spec["slug"]
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / "model.onnx"
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with torch.inference_mode():
            torch.onnx.export(model, (torch.from_numpy(x[:1]),), str(onnx_path),
                input_names=["windows"], output_names=["normalized_prediction"],
                opset_version=17, dynamo=False)
    export_warnings = sorted(set(str(w.message) for w in captured))
    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    require(session.get_inputs()[0].name == "windows" and session.get_inputs()[0].shape == [1, 7, 5], "ONNX input contract changed")
    require(session.get_outputs()[0].name == "normalized_prediction", "ONNX output contract changed")
    scale = scaler["target_scale"]
    def memory(values):
        with torch.inference_mode():
            return np.asarray([float(model(torch.from_numpy(v[None])).reshape(-1)[0]) * scale for v in values])
    def exported(values):
        return np.asarray([float(session.run(None, {"windows": v[None]})[0].reshape(-1)[0]) * scale for v in values])
    cpu, pred = memory(x), exported(x)
    cal_cpu, cal_pred = memory(cal_x), exported(cal_x)
    frozen = np.asarray(ref["prediction_original"], dtype=np.float64)
    cal_frozen = np.asarray(ref["calibration_predictions"], dtype=np.float64)
    errors = {
        "maximum_absolute_error_vs_CPU_memory": float(np.max(np.abs(pred - cpu))),
        "maximum_absolute_error_vs_frozen_saved_predictions": float(np.max(np.abs(pred - frozen))),
        "maximum_CPU_absolute_error_vs_frozen_saved_predictions": float(np.max(np.abs(cpu - frozen))),
        "maximum_calibration_absolute_error_vs_CPU_memory": float(np.max(np.abs(cal_pred - cal_cpu))),
        "maximum_calibration_absolute_error_vs_frozen_predictions": float(np.max(np.abs(cal_pred - cal_frozen))),
        "maximum_calibration_CPU_absolute_error_vs_frozen_predictions": float(np.max(np.abs(cal_cpu - cal_frozen))),
    }
    require(all(value <= TOLERANCE for value in errors.values()), "Real frozen parity tolerance failed: " + json.dumps(errors))
    require(len(np.unique(pred)) > 1, "Export returned a constant across distinct real windows")
    # Evaluating all 661 distinct windows is the trace-branch check. Shape guards
    # can be constant for this fixed shape; no model input-dependent branch is
    # accepted merely because the tracing example worked.
    trace_note = "Fixed batch-one [1,7,5] contract. Trace warnings from shape/finiteness guards or constant padding are retained. Parity is checked over 608 distinct evaluation and 53 calibration windows, not just the trace example. External input validation remains required."
    q = cal["q"]
    frozen_covered = (y >= np.maximum(0, frozen - q)) & (y <= frozen + q)
    covered = (y >= np.maximum(0, pred - q)) & (y <= pred + q)
    forward_ms, e2e_ms = [], []
    for i in range(110):
        t = perf_counter(); session.run(None, {"windows": x[:1]}); forward = 1000 * (perf_counter() - t)
        t = perf_counter(); xi = transform(raw[:1], scaler); float(session.run(None, {"windows": xi})[0].reshape(-1)[0]) * scale; end_to_end = 1000 * (perf_counter() - t)
        if i >= 10:
            forward_ms.append(forward); e2e_ms.append(end_to_end)
    common = {
        "model": args.model, "seed": SEED, "evaluation_dates": 608,
        "date_start": dates[0], "date_end": dates[-1], "calibration_dates": 53,
        "declared_absolute_tolerance_original_sigma7": TOLERANCE, **errors,
        "RMSE_CPU_memory": float(np.sqrt(np.mean((cpu - y) ** 2))),
        "RMSE_ONNX": float(np.sqrt(np.mean((pred - y) ** 2))),
        "RMSE_frozen_saved_predictor": float(np.sqrt(np.mean((frozen - y) ** 2))),
        "within_tolerance": True, "onnx_checker_passed": True,
        "distinct_eval_predictions": len(np.unique(pred)), "all_real_windows_verified": 661,
        "onnx_bytes": onnx_path.stat().st_size, "onnx_sha256": sha(onnx_path),
        "checkpoint_bytes": spec["bytes"], "trainable_parameters": spec["parameters"],
        "historical_608_coverage_percent_for_this_seed": float(100 * np.mean(covered)),
        "frozen_historical_608_coverage_percent_for_this_seed": float(100 * np.mean(frozen_covered)),
        "interval_coverage_boundary_disagreements": int(np.sum(covered != frozen_covered)),
        "calibration_q_reused_unchanged": True,
        "new_training_runs": 0, "new_calibration_fits": 0, "GPU_used": False,
        "live_registry_updated": False, "export_warnings": export_warnings, "trace_validation": trace_note,
        "provisional_cost_measurement": {
            "median_forward_ms": float(np.median(forward_ms)), "p95_forward_ms": float(np.quantile(forward_ms, .95)),
            "median_end_to_end_ms": float(np.median(e2e_ms)), "p95_end_to_end_ms": float(np.quantile(e2e_ms, .95)),
            "process_peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "RSS_definition": "Whole export+Torch+ORT process peak RSS, not incremental ORT/model memory",
            "scope": "CPUExecutionProvider, one intra/inter-op thread, batch 1; 10 warmups+100 calls each; forward excludes preprocessing; end-to-end includes raw transform and inverse target scale; excludes intervals/explanations/startup.",
            "comparability": "Provisional: concurrent export/inference tasks may affect costs; a common sequential ORT-only benchmark is required for comparison.",
        },
    }
    lib = {"python": platform.python_version(), "torch_export_only": str(torch.__version__), "onnx_export_only": onnx.__version__, "onnxruntime": ort.__version__, "numpy": np.__version__, "platform": platform.platform(), "cpu_count": os.cpu_count()}
    metadata = {
        "schema_version": 1, "model": args.model, "seed": SEED,
        "representative_seed_rule": "Smallest prespecified final seed ID 271801; not selected by test performance",
        "profile": "offline_cpu_export_20261120", "format": "ONNX", "opset": 17,
        "model_sha256": sha(onnx_path), "model_bytes": onnx_path.stat().st_size,
        "input_name": "windows", "input_shape": [1, 7, 5], "input_dtype": "float32",
        "output_name": "normalized_prediction", "onnx_output_unit": "sigma7 / frozen target_scale",
        "output_scale": scale, "served_output_unit": "original population sigma7, not annualized",
        "uses_observable_floor": False, "floor_input_dtype": None,
        "raw_feature_order": ["open", "high", "low", "close", "volume"], "horizon_days": 1,
        "preprocessing": {"price_transform": "log(price_t) - log(last_observed_close) for all four price channels", "volume_transform": "log1p(volume_t)", "standardize": "(relative_features - frozen mean) / frozen std, then float32", "mean": scaler["mean"], "std": scaler["std"], "fit_count": 655, "fit_start": fit_dates[0], "fit_end": fit_dates[-1], "target_scale": scale, "clipping": False, "refitted": False},
        "calibration_q": q, "interval_nominal_coverage": .9,
        "interval_formula": "[max(0, prediction_original - q), prediction_original + q]",
        "interval_limitations": "Historical BTCUSDT calibration, 53 serially dependent days; no exact exchangeability or future/subgroup coverage guarantee.",
        "scientific_source": {"type": "existing_E2_frozen_final_checkpoint", "checkpoint_sha256": spec["checkpoint"], "checkpoint_bytes": spec["bytes"], "registered_trial_id": trial["trial_id"], "request_sha256": cp["request_sha256"], "architecture": cp["architecture"], "source_hashes": SOURCE_HASHES},
        "data_source": {"npz_sha256": INPUT_HASHES["npz"], "predictions_sha256": INPUT_HASHES["predictions"], "prediction_count_in_source": len(prior["predictors"])},
        "calibrator_source": {"file_sha256": sha(calibrator_path), "content_sha256": cal["sha256"], "q": q, "rank": 49, "n": 53, "date_start": cal_dates[0], "date_end": cal_dates[-1], "checkpoint_sha256": cal["predictor_sha256"]},
        "original_scaler_file_sha256": sha(scaler_path), "libraries": lib,
        "roundtrip_status": "passed_real_608_and_53", "new_training_runs": 0, "new_calibration_fits": 0,
        "live_registry_updated": False, "AI_assistance": "OpenAI Codex assisted export reconstruction, validation and documentation; no claim of independent human reproduction.",
    }
    write(out / "metadata.json", metadata)
    write(out / "roundtrip.json", common)
    # Preserve original byte identities as well as numeric/internal digests.
    (out / "calibrator.json").write_bytes(calibrator_path.read_bytes())
    (out / "scaler.json").write_bytes(scaler_path.read_bytes())
    (out / "selection_receipt.json").write_bytes(selection_path.read_bytes())
    asset_map = json.loads((args.checkpoints / "checkpoint_assets.json").read_bytes())
    asset = next(a for a in asset_map if a["model"] == args.model)
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "model": args.model, "seed": SEED,
        "checkpoint": asset, "final_freeze_sha256": "c0e9e2589dca9fa03b341bb72f87d8b541d33b4721715622daeae4438dbd60b4",
        "source_hashes": SOURCE_HASHES,
        "frozen_inputs": {**INPUT_HASHES, "selection_receipt": sha(selection_path), "scaler": sha(scaler_path), "calibrator_file": sha(calibrator_path), "calibrator_content": cal["sha256"]},
        "prediction_source_url": "https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/5e9023ea821bfa9fb0b92340d518413f6ab89161/runs/e2_extension_10_semillas_v1/combined_predictors.json",
        "selection_source_repository_path": "experiments/e3_bloques_ab_20261115/inputs/selection_receipts/" + spec["selection"] + ".json",
        "raw_OHLCV_published_in_this_payload": False,
        "outputs": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file() and p.name != "provenance.json"},
        "scope": "One pre-existing representative checkpoint; export and real frozen parity only. Existing 796-job audit and 30-fit V3 pilot were not changed.",
        "no_training_no_recalibration": True, "libraries": lib,
    }
    write(out / "provenance.json", provenance)
    print(json.dumps({"model": args.model, "status": "passed_real_608_and_53", "onnx_bytes": onnx_path.stat().st_size, **errors}, allow_nan=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=MODELS, required=True)
    for key in ("repo", "checkpoints", "npz", "predictions", "protocol", "selections", "freeze", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    run(p.parse_args())
