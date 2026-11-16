"""Audit current E2 neural ports on six fixed REAL development windows.

Prepare and hash the plan after unit checks, before loading weights or calling
models. Execute only through the shared scientific budget wrapper. This is a
new CPU implementation audit, not original GPU parity or an evaluation audit.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

VERSION = "e2-real-development-port-audit-v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "book_e2/_static/e2_scientific_closure_20261116/port_audit"
DEFAULT_BASE = ROOT / "book_e2/_static/e2_corrections_20261115/frozen_inference/eager_spec.json"
DEFAULT_AUDIT = ROOT / "book_e2/_static/e2_corrections_20261115/data_audit/audit_result.json"
DEFAULT_CPU = ROOT / "book_e2/_static/e2_corrections_20261115/cpu_diagnostic/predictions_complete/inference_report.json"
FP32_EPSILON = 2.0 ** -23
NUMERICAL_FACTOR = 64
ATOL = NUMERICAL_FACTOR * FP32_EPSILON
RTOL = NUMERICAL_FACTOR * FP32_EPSILON
INDICES = {"training": [0, 327, 654], "calibration": [0, 26, 52]}
HEADS = {"FCN-1D": "head", "ResNet-1D": "head", "Inception-1D": "head",
         "LSTM-autoencoder con predictor": "head", "TimesNet": "readout",
         "TSMixer": "readout", "TiDE": "readout", "TimeMixer": "readout",
         "PatchTST": "readout", "iTransformer": "head", "MLP16": "network.4",
         "LSTM32": "head", "GRU32": "head"}
CONDITIONS = ("batch_six", "batch_reverse", "isolated_reverse",
              "future_raw_and_all_targets_perturbed", "all_targets_only_perturbed")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def file_sha(path):
    h = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical(data) + b"\n")


def binding(path):
    path = Path(path).resolve()
    require(path.is_file() and not path.is_symlink(), "Missing regular input: " + str(path))
    return {"path": str(path), "sha256": file_sha(path), "bytes": path.stat().st_size}


def verify_binding(record):
    path = Path(record["path"])
    require(path.is_file() and not path.is_symlink()
            and path.stat().st_size == record["bytes"] and file_sha(path) == record["sha256"],
            "Changed frozen input: " + str(path))


def load_helper(path):
    module_spec = importlib.util.spec_from_file_location("_e2_port_audit_frozen_helpers", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def prepare(args):
    output = args.output.resolve()
    require(not output.exists() or (output.is_dir() and
            {p.name for p in output.iterdir()} <= {"unit_checks.log", "unit_checks.json"}),
            "Prepare admits only the prior unit-test receipt/log in a fresh registration directory")
    helper_path = ROOT / "scripts/e2_corrections_frozen_inference.py"
    helper = load_helper(helper_path)
    base = read_json(args.base_spec)
    helper.verify_inputs(base)
    audit = read_json(args.data_audit)
    require(audit["actual_cohorts_audited"].get("training") == 655
            and audit["actual_cohorts_audited"].get("calibration") == 53,
            "The actual data audit must identify train655/cal53")
    arrays = ROOT / audit["arrays_npz_path"]
    require(file_sha(arrays) == audit["arrays_npz_sha256"], "Audited array archive changed")
    cpu = read_json(args.previous_cpu_report)
    require(cpu["predictors_completed"] == 78 and cpu["original_parity_gate_passed"] is False,
            "Preserve the complete CPU diagnostic and its failed original parity claim")
    tests = read_json(args.unit_receipt)
    require(tests["status"] == "passed" and tests["runner_sha256"] == file_sha(__file__)
            and tests["returncode"] == 0 and tests["tests_run"] >= 7,
            "Passing unit receipt must bind this exact runner before registration")
    verify_binding(tests["test_source"])
    verify_binding(tests["log"])
    plan = {
        "version": VERSION, "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "New post hoc implementation audit of real train655/cal53; six fixed available windows per predictor; no Xtest608, general proof, original GPU parity, fitting, scoring or hypothesis tests",
        "runner": binding(__file__), "helper": binding(helper_path),
        "base_spec": binding(args.base_spec), "data_audit": binding(args.data_audit),
        "arrays_npz": binding(arrays), "array_keys": audit["array_keys"],
        "unit_receipt": binding(args.unit_receipt), "previous_cpu_report": binding(args.previous_cpu_report),
        "previous_alerts": {"projection_alert_predictors": cpu["projection_alert_predictors"],
                            "maximum_calibration_absolute_difference": cpu["maximum_calibration_absolute_difference"],
                            "absolute_tolerance": cpu["calibration_absolute_tolerance"],
                            "original_parity_gate_passed": False},
        "model_order": list(helper.MODELS), "seed_ids": list(helper.SEEDS),
        "fixed_zero_based_indices": INDICES, "conditions": list(CONDITIONS),
        "head_input_probes": HEADS,
        "probe_scope": "Tensor entering the current scalar head, after the port's own normalization, layout changes, encoder and any same-window moments; detached diagnostic hooks do not modify tensors",
        "batch_policy": "Reference: one real window per call. Compare six windows together and reversed through the unchanged current port. TimesNet's existing eval adapter invokes its forecasting body once per window; do not bypass or replace this adapter. Other ports receive six-window calls directly.",
        "numerical_policy": {"dtype": "float32", "device": "cpu", "threads": 2,
            "interop_threads": 1, "mkldnn_enabled": False, "deterministic_algorithms": True,
            "inference_mode": True, "eval": True, "autocast": False,
            "absolute_tolerance_scaled_output_and_head_features": ATOL,
            "relative_tolerance_scaled_output_and_head_features": RTOL,
            "comparison_rule": "abs(observed-reference) <= 64*eps32 + 64*eps32*abs(reference), elementwise; every value finite and every shape exact",
            "reason": "64 FP32 machine epsilons allow small CPU accumulation-order differences between singleton and six-window GEMM/encoder operations. This fixed engineering threshold is selected before forwards, is not a statistical margin, and cannot amend the original 1e-7 CPU/GPU gate.",
            "raw_relative_scaled_inputs_state_and_scaler": "exact equality or cryptographic equality; no numerical tolerance",
            "failure_rule": "Record every predictor/condition failure; continue the fixed domain without exclusions, backend changes, tolerance tuning or automatic method repair"},
        "perturbations": {"future_raw": "For each query separately, multiply all four OHLC fields by1.17 and volume by1.31 at dates >= queried outcome, preserving missing values and the seven strictly previous input bars; keep the already frozen train scaler",
            "targets": "Replace every finite development target y by1.37*y+0.0001; do not refit the scaler or pass targets into a model",
            "not_claimed": "No refit-invariance to changes inside the training cohort; no security sandbox against arbitrary source access; no attribution of general causality from six finite cases"},
        "operations": {"model_fits": 0, "optimizers": 0, "recalibration": 0,
                       "evaluation_inference": 0, "new_predictive_scores": 0,
                       "hypothesis_tests": 0, "bootstrap": 0},
        "budget": {"wrapper_required": "scripts/e2_corrections_budget.py", "cap_seconds": 600,
                   "ledger": "book_e2/_static/e2_scientific_closure_20261116/compute_ledger.jsonl"},
    }
    output.mkdir(parents=True, exist_ok=True)
    write_new_json(output / "plan.json", plan)
    return {"status": "registered_before_forwards", "plan_sha256": file_sha(output / "plan.json"),
            "model_forwards": 0, "weights_deserialized": False}


def comparison(reference, observed, np, atol=ATOL, rtol=RTOL):
    a, b = np.asarray(reference), np.asarray(observed)
    shape_ok = a.shape == b.shape and a.size > 0
    finite = shape_ok and bool(np.isfinite(a).all() and np.isfinite(b).all())
    if not finite:
        return {"passed": False, "shape_equal": a.shape == b.shape, "finite": False,
                "max_absolute_difference": None, "max_tolerance_fraction": None}
    difference = np.abs(b.astype(np.float64) - a.astype(np.float64))
    bound = atol + rtol * np.abs(a.astype(np.float64))
    return {"passed": bool((difference <= bound).all()), "shape_equal": True, "finite": True,
            "max_absolute_difference": float(difference.max()),
            "max_tolerance_fraction": float((difference / bound).max())}


def capture(model, x, head_name, torch):
    features = []

    def hook(module, inputs):
        require(len(inputs) == 1 and isinstance(inputs[0], torch.Tensor), "Unexpected scalar-head signature")
        features.append(inputs[0].detach().cpu().clone())

    handle = model.get_submodule(head_name).register_forward_pre_hook(hook)
    before = x.detach().clone()
    try:
        with torch.inference_mode(), torch.autocast(device_type="cpu", enabled=False):
            output = model(x)
        require(torch.equal(before, x), "Model modified its supplied input")
        require(tuple(output.shape) == (len(x),) and bool(torch.isfinite(output).all())
                and bool((output >= 0).all()), "Invalid nonnegative scalar output")
        require(len(features) == 1 and features[0].ndim == 2 and len(features[0]) == len(x)
                and bool(torch.isfinite(features[0]).all()), "Unexpected or nonfinite head-input probe")
        return output.detach().cpu().numpy().copy(), features[0].numpy().copy()
    finally:
        handle.remove()


def isolated_capture(model, x, head, torch, np, order=None):
    order = list(range(len(x))) if order is None else list(order)
    require(sorted(order) == list(range(len(x))), "Order must preserve every queried window exactly once")
    predictions, features = {}, {}
    for i in order:
        p, f = capture(model, x[i:i + 1].clone(), head, torch)
        predictions[i], features[i] = p[0], f[0]
    return np.asarray([predictions[i] for i in range(len(x))]), np.stack([features[i] for i in range(len(x))])


def transformed_perturbations(panel, dates, scaler, torch, np):
    from e2_temporal_training import DatedPanel
    future_inputs, target_inputs, records = [], [], []
    original_record = scaler.record()
    for date in dates:
        original_raw = panel.raw_windows((date,))
        raw = panel.features
        mask = panel.calendar >= date
        raw[mask, :4] *= 1.17
        raw[mask, 4] *= 1.31
        targets = panel.targets
        valid_targets = np.isfinite(targets)
        targets[valid_targets] = 1.37 * targets[valid_targets] + .0001
        future = DatedPanel(panel.calendar, raw, targets, panel.quality)
        labels = DatedPanel(panel.calendar, panel.features, targets, panel.quality)
        changed_raw = future.raw_windows((date,))
        target_raw = labels.raw_windows((date,))
        x = scaler.transform(original_raw).astype(np.float32)
        x_future = scaler.transform(changed_raw).astype(np.float32)
        x_labels = scaler.transform(target_raw).astype(np.float32)
        require(np.array_equal(original_raw, changed_raw) and np.array_equal(original_raw, target_raw)
                and np.array_equal(x, x_future) and np.array_equal(x, x_labels),
                "Current/future raw values or target labels reached the queried frozen input")
        require(scaler.record() == original_record, "Perturbation changed the frozen scaler")
        future_inputs.append(x_future[0])
        target_inputs.append(x_labels[0])
        records.append({"date": date.strftime("%Y-%m-%d"), "raw_rows_perturbed_at_or_after_query": int(mask.sum()),
                        "finite_labels_perturbed": int(valid_targets.sum()), "raw_window_exactly_unchanged": True,
                        "scaled_fp32_input_exactly_unchanged": True, "frozen_scaler_unchanged": True})
    return torch.from_numpy(np.stack(future_inputs)), torch.from_numpy(np.stack(target_inputs)), records


def audit_model(model, x, future_x, target_x, head, torch, np):
    from e2_staged_training import state_sha256
    before = state_sha256(model)
    original_x = x.clone()
    reference, reference_features = isolated_capture(model, x, head, torch, np)
    observations = {}
    observations["batch_six"] = capture(model, x.clone(), head, torch)
    reverse = list(reversed(range(len(x))))
    p, f = capture(model, x[reverse].clone(), head, torch)
    observations["batch_reverse"] = (p[reverse], f[reverse])
    observations["isolated_reverse"] = isolated_capture(model, x, head, torch, np, reverse)
    observations["future_raw_and_all_targets_perturbed"] = isolated_capture(model, future_x, head, torch, np)
    observations["all_targets_only_perturbed"] = isolated_capture(model, target_x, head, torch, np)
    checks = []
    for condition, (predicted, features) in observations.items():
        output_check = comparison(reference, predicted, np)
        feature_check = comparison(reference_features, features, np)
        checks.append({"condition": condition, "passed": output_check["passed"] and feature_check["passed"],
                       "scaled_output": output_check, "head_features": feature_check,
                       "observed_scaled_outputs": [float(v) for v in predicted]})
    after = state_sha256(model)
    state_ok = before == after
    input_ok = torch.equal(original_x, x) and torch.equal(x, future_x) and torch.equal(x, target_x)
    eval_ok = not any(module.training for module in model.modules())
    return {"passed": all(r["passed"] for r in checks) and state_ok and input_ok and eval_ok,
            "conditions": checks, "state_before_sha256": before, "state_after_sha256": after,
            "state_unchanged": state_ok, "inputs_exactly_unchanged": input_ok,
            "all_modules_eval": eval_ok, "head_probe_shape": list(reference_features.shape),
            "reference_scaled_outputs": [float(v) for v in reference],
            "reference_head_features_sha256": sha256(reference_features.tobytes()).hexdigest(),
            "outer_model_forward_calls": 26,
            "adapter_note": "TimesNet current eval wrapper internally calls its unchanged forecasting body separately for each window" if getattr(model, "name", None) == "TimesNet" else "unchanged current port receives requested batch"}


def execute(args):
    output = args.output.resolve()
    plan_path = output / "plan.json"
    plan_sha = file_sha(plan_path)
    require(args.plan_sha256 == plan_sha, "Supply the exact plan SHA frozen before execution")
    plan = read_json(plan_path)
    require(plan["version"] == VERSION and not (output / "audit_report.json").exists()
            and not (output / "failure_report.json").exists() and not (output / "progress.jsonl").exists(),
            "Execution requires this registered, unexecuted output")
    for name in ("runner", "helper", "base_spec", "data_audit", "arrays_npz", "unit_receipt", "previous_cpu_report"):
        verify_binding(plan[name])
    helper = load_helper(plan["helper"]["path"])
    base = read_json(plan["base_spec"]["path"])
    helper.verify_inputs(base)
    started = time.monotonic()
    results, global_records, runtime = [], [], None
    log = (output / "progress.jsonl").open("x", encoding="utf-8")
    try:
        torch, np, train, cal, admission, runtime = helper.runtime_and_data(base)
        from e2_development_data import load_development
        from e2_staged_training import state_sha256
        panel, _ = load_development(base["source_root"], "2024-12-30")
        splits = {"training": train, "calibration": cal}
        dates, identities, inputs = [], [], []
        with np.load(plan["arrays_npz"]["path"], allow_pickle=False) as arrays:
            for split, selected in INDICES.items():
                prepared = splits[split]
                keys = plan["array_keys"][split]
                require(len(prepared.x) == (655 if split == "training" else 53), "Cohort length changed")
                require(np.array_equal(prepared.x.numpy(), arrays[keys["X_scaled"]])
                        and np.array_equal(np.asarray(prepared.y_original), arrays[keys["y_original"]])
                        and np.array_equal(panel.raw_windows(prepared.dates), arrays[keys["raw"]]),
                        "Rebuilt real cohort differs from audited input archive")
                for index in selected:
                    dates.append(prepared.dates[index])
                    inputs.append(prepared.x[index].clone())
                    identities.append({"split": split, "zero_based_index": index,
                                       "date": prepared.dates[index].strftime("%Y-%m-%d")})
        x = torch.stack(inputs)
        future_x, target_x, global_records = transformed_perturbations(panel, dates, train.scaler, torch, np)
        require(torch.equal(x, future_x) and torch.equal(x, target_x), "Fixed queried inputs changed")
        scaler_before = train.scaler.record()
        roots = {name: str(Path(base["authors_root"]) / name) for name in helper.AUTHOR_MODELS}
        write_new_json(output / "real_input_controls.json", {"queries": identities, "perturbations": global_records,
            "train_signature": train.signature, "calibration_signature": cal.signature,
            "queried_fp32_tensor_sha256": helper.tensor_sha(x),
            "scaler_record": scaler_before, "scaler_canonical_sha256": helper.digest(scaler_before),
            "all_available_train_cal_arrays_exact": True, "development_admission": admission})
        for row in base["neural_rows"]:
            result = {"model": row["model"], "seed": row["seed"],
                      "checkpoint_sha256": row["expectedsha"]["checkpoint"],
                      "scaler_sha256": row["expectedsha"]["scaler"], "trial": row["trial"],
                      "head_probe": HEADS[row["model"]]}
            try:
                model, loaded_state = helper.load_model(row, train, roots, torch)
                require(state_sha256(model) == loaded_state, "Initial checkpoint state differs")
                result.update(audit_model(model, x, future_x, target_x, result["head_probe"], torch, np))
                result["checkpoint_state_matches_before"] = result["state_before_sha256"] == loaded_state
                result["scaler_unchanged"] = train.scaler.record() == scaler_before and read_json(row["scaler_path"]) == scaler_before
                result["passed"] = result["passed"] and result["checkpoint_state_matches_before"] and result["scaler_unchanged"]
                restored_reference = np.asarray(result["reference_scaled_outputs"], dtype=np.float32) * train.scaler.target_scale
                for condition in result["conditions"]:
                    restored_observed = np.asarray(condition["observed_scaled_outputs"], dtype=np.float32) * train.scaler.target_scale
                    condition["maximum_original_unit_output_difference"] = float(np.abs(restored_observed.astype(np.float64) - restored_reference.astype(np.float64)).max())
                del model
            except Exception as error:
                result.update(passed=False, failure_type=type(error).__name__, failure_message=str(error),
                              traceback=traceback.format_exc())
            results.append(result)
            log.write(canonical(result).decode() + "\n")
            log.flush()
            print(json.dumps({"model": result["model"], "seed": result["seed"], "passed": result["passed"]}), flush=True)
        write_new_json(output / "predictor_checks.json", results)
        csv_fields = ["model", "seed", "passed", "head_probe", "state_unchanged", "inputs_exactly_unchanged", "all_modules_eval", "scaler_unchanged", "condition", "condition_passed", "output_max_abs_difference_scaled", "output_max_abs_difference_original", "output_max_tolerance_fraction", "head_features_max_abs_difference", "head_features_max_tolerance_fraction"]
        with (output / "condition_checks.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=csv_fields, lineterminator="\n")
            writer.writeheader()
            for result in results:
                for condition in result.get("conditions", [{"condition": "execution_failure", "passed": False}]):
                    writer.writerow({**{key: result.get(key, "") for key in csv_fields[:8]},
                        "condition": condition["condition"], "condition_passed": condition["passed"],
                        "output_max_abs_difference_scaled": condition.get("scaled_output", {}).get("max_absolute_difference", ""),
                        "output_max_abs_difference_original": condition.get("maximum_original_unit_output_difference", ""),
                        "output_max_tolerance_fraction": condition.get("scaled_output", {}).get("max_tolerance_fraction", ""),
                        "head_features_max_abs_difference": condition.get("head_features", {}).get("max_absolute_difference", ""),
                        "head_features_max_tolerance_fraction": condition.get("head_features", {}).get("max_tolerance_fraction", "")})
        passed = sum(r["passed"] for r in results)
        require(len(results) == 78, "Incomplete predictor domain")
        report = {"version": VERSION, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                  "status": "PASS" if passed == 78 else "FAIL", "plan_sha256": plan_sha,
                  "scope": plan["scope"], "predictors_expected": 78, "predictors_completed": len(results),
                  "predictors_passed": passed, "predictors_failed": len(results) - passed,
                  "models": 13, "seeds_per_model": 6, "queries_per_predictor": 6,
                  "conditions_per_predictor": 5, "available_cohort_sizes": {"training": 655, "calibration": 53},
                  "runtime": runtime, "elapsed_seconds": time.monotonic() - started,
                  "numerical_policy": plan["numerical_policy"], "previous_alerts_preserved": plan["previous_alerts"],
                  "original_gpu_parity_passed": False, "evaluation_features_audited": False,
                  "new_model_fits": 0, "new_predictive_scores": 0, "hypothesis_tests": 0,
                  "limitations": ["Six fixed development cases per predictor, not exhaustive input-space certification", "No actual608 evaluation OHLCV available or generated", "Batch invariance is numerical at the registered tolerance, not statistical equivalence", "State checks cover parameters and registered buffers; legitimate transient normalization attributes are not claimed immutable", "No random-label fit or adversarial domain-classifier fit was performed"],
                  "outputs": {p.name: binding(p) for p in output.iterdir() if p.is_file() and p.name != "progress.jsonl"}}
        write_new_json(output / "audit_report.json", report)
        return report
    except Exception as error:
        write_new_json(output / "failure_report.json", {"version": VERSION, "status": "FAIL",
            "plan_sha256": plan_sha, "predictors_completed": len(results),
            "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
            "elapsed_seconds": time.monotonic() - started, "runtime": runtime, "outputs_preserved": True})
        raise
    finally:
        log.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-spec", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--data-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--previous-cpu-report", type=Path, default=DEFAULT_CPU)
    parser.add_argument("--unit-receipt", type=Path)
    parser.add_argument("--plan-sha256")
    args = parser.parse_args(argv)
    if args.operation == "prepare":
        require(args.unit_receipt is not None, "Prepare requires the prior passing unit receipt")
        result = prepare(args)
    else:
        result = execute(args)
    print(json.dumps({k: v for k, v in result.items() if k != "outputs"}, ensure_ascii=False, indent=2))
    return 1 if result.get("status") == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
