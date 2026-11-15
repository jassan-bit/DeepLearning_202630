"""Registered supplemental CPU diagnostics of all frozen E2 neural predictors.

This is separate from both preserved failed GPU-parity gates. The identical
1e-7 difference threshold is an alert only: no finite predictor is filtered.
No optimizer, fitting, calibration update, old IUT or new ranking is supplied.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from hashlib import sha256

VERSION = "e2-corrections-cpu-diagnostic-v1"
PARENT_SHA = "c0a03171aa63ea7a10f11a1bc9da14e8788fb27938039a6d4cfb37d1d4f2f8c7"
BASE_SPEC_SHA = "76ce5b4b84d1dd752707fd42cf53c20ffaf91f57781986eb4ccc07d74e29e515"
PARENT = None
ACTIVE_ADMISSION = None


def local_sha(path):
    result = sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(data)
    return result.hexdigest()


def load_parent(path):
    global PARENT
    path = Path(path).resolve()
    if local_sha(path) != PARENT_SHA:
        raise ValueError("The preserved eager runner source changed")
    module_spec = importlib.util.spec_from_file_location("_e2_preserved_cpu_eager_v2", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    PARENT = module
    for name in ("require", "canonical", "digest", "file_sha", "read_json", "write_new_json", "regular_file",
                 "runtime_and_data", "load_model", "csv_open", "common_identity", "predict", "rmse", "progress",
                 "output_hashes", "tensor_sha", "scalar_sha", "MODELS", "SEEDS", "BASELINE", "TOLERANCE",
                 "TARGET_TOLERANCE", "AUTHOR_MODELS", "write_predictions"):
        globals()[name] = getattr(module, name)
    return module


def prepare_plan(args):
    parent = load_parent(args.parent_runner)
    base_path = regular_file(args.base_spec)
    require(file_sha(base_path) == BASE_SPEC_SHA, "The preserved eager specification changed")
    base = read_json(base_path)
    parent.verify_inputs(base)
    stats_path = args.statistics_plan.resolve()
    previous_output = base_path.parent / "eager_predictions"
    failure = regular_file(previous_output / "failure_report.json")
    plan = {
        "version": VERSION, "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "New exploratory CPU diagnostic registered after two failed strict CPU-to-GPU parity attempts; no historical preregistration, original parity, equivalence or new test claimed",
        "runner_path": str(Path(__file__).resolve()), "runner_sha256": file_sha(__file__),
        "supersedes_registered_plan": ({"path": str(args.supersedes_plan.resolve()),
            "sha256": file_sha(regular_file(args.supersedes_plan)),
            "reason": args.supersession_reason}
            if args.supersedes_plan else None),
        "parent_runner": {"path": str(Path(args.parent_runner).resolve()), "sha256": PARENT_SHA},
        "base_spec": {"path": str(base_path), "sha256": BASE_SPEC_SHA},
        "previous_attempt": {"output": str(previous_output), "failure_report_sha256": file_sha(failure),
                             "status": "failed_original_absolute_parity_gate", "outputs_preserved": True},
        "statistics_plan_path": str(stats_path),
        "statistics_admission": "Independent statistics plan will bind this inference-plan SHA; execution admission must bind both hashes before loading Torch or weights. No circular modification of either plan.",
        "model_order": list(MODELS), "seed_ids": list(SEEDS), "predictors_required": 78,
        "checkpoint_index_sha256": base["checkpoint_index_sha256"],
        "checkpoint_lineage": base["checkpoint_lineage"],
        "prediction_policy": {**base["prediction_policy"],
            "abort_on_any_calibration_mismatch": False,
            "difference_threshold_role": "Published diagnostic alert only; retain all finite predictors and all differences, without filtering or stopping on an alert",
            "original_parity_gate_passed_claimed": False,
            "comparison_scope": "New CPU eager frozen projection versus saved GPU calibration reference; difference is numerical framework/hardware compatibility, not statistical equivalence",
            "hard_failures": ["changed source/hash/target/scaler/calibrator binding", "missing predictor", "nonfinite or invalid scalar", "model state mutation"]},
        "latency_policy": {**base["latency_policy"],
            "admission_gate": "All 78 CPU diagnostic predictors complete with immutable source/state, finite predictions and published alerts; original GPU-parity gate is not asserted",
            "projection_alert_exclusion": False,
            "statistical_scope": "Separate CPU conditional/descriptive analysis only; original accuracy-GPU/cost-CPU IUT is not executed"},
        "outputs": {"predictions_train.csv": "51090 rows, original units, model/seed/date/split/error and hashes",
                    "predictions_calibration.csv": "4134 rows, same schema and all projections",
                    "risks_by_predictor.csv": "78 rows; CPU train RMSE, calibration RMSE and their difference; seeds paired, train/cal dates not paired",
                    "calibration_agreement.csv": "78 comparisons; max difference and alert flag, never a filter",
                    "raw_cpu_latency.csv": "12000 individual operations in 100 global blocks with 6 seed pairs per each of 10 candidate/GRU comparisons"},
        "no_changes": ["weights", "source scalers", "source calibrators", "data", "backend", "1e-7 alert threshold", "primary A2", "model selection"],
        "old_statistical_plan_714f32": "Preserved as history but not an eligible IUT/gate for this new supplement",
        "operations": {"new_model_fits": 0, "optimizers_created": 0, "calibrators_refitted": 0,
                       "original_IUT_executed": False, "test_inference": False},
        "budget": "External additive corrections ledger, predictions cap3600s and latency cap900s; no reset or legacy controller"}
    write_new_json(args.plan, plan)
    return {"status": "plan_registered", "plan_path": str(args.plan.resolve()), "plan_sha256": file_sha(args.plan),
            "new_model_forwards": 0, "new_model_fits": 0}


def require_prediction_report(directory, spec_sha):
    directory = Path(directory).resolve()
    report = read_json(regular_file(directory / "inference_report.json"))
    require(report.get("version") == VERSION and report.get("stage") == "predictions"
            and report.get("status") == "completed_CPU_diagnostic_with_projection_alerts"
            and report.get("spec_sha256") == spec_sha and report.get("predictors_completed") == 78
            and report.get("calibration_comparisons_completed") == 78 and report.get("states_unchanged") is True,
            "Latency requires all 78 finite CPU diagnostics; alerts are included, not filtered")
    for relative, row in report["all_output_files"].items():
        require(Path(relative).name == relative, "Unsafe prior output path")
        path = regular_file(directory / relative)
        require(path.stat().st_size == row["bytes"] and file_sha(path) == row["sha256"], "Prior diagnostic output changed")
    return {"path": str(directory / "inference_report.json"), "sha256": file_sha(directory / "inference_report.json")}

def prediction_stage(spec, spec_sha, output):
    torch, np, train, calibration, admission, runtime = runtime_and_data(spec)
    roots = {m: str(Path(spec["authors_root"]) / m) for m in AUTHOR_MODELS}
    fields = ["model", "seed", "date", "split", "y_true", "prediction", "signed_error", "absolute_error", "squared_error",
              "batch_size", "device", "eval_mode", "inference_mode", "checkpoint_sha256", "scaler_sha256", "config_sha256"]
    train_stream, train_writer = csv_open(output / "predictions_train.csv", fields)
    cal_stream, cal_writer = csv_open(output / "predictions_calibration.csv", fields)
    agreement_stream, agreement_writer = csv_open(output / "calibration_agreement.csv", ["model", "seed", "max_absolute_prediction_difference", "absolute_tolerance", "within_alert_tolerance", "projection_alert", "reference_sha256", "checkpoint_sha256"])
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
                         "absolute_tolerance": TOLERANCE, "within_alert_tolerance": difference <= TOLERANCE, "projection_alert": difference > TOLERANCE,
                         "reference_sha256": row["expectedsha"]["calibration_predictions"], "checkpoint_sha256": row["expectedsha"]["checkpoint"]}
            agreement_writer.writerow(agreement)
            agreement_stream.flush()
            progress(log, {"stage": "calibration_projection_comparison", **agreement})
            # All finite projections are retained, including every alert.
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
    report = {"version": VERSION, "stage": "predictions", "status": "completed_CPU_diagnostic_with_projection_alerts", "spec_sha256": spec_sha,
              "checkpoint_index_sha256": spec["checkpoint_index_sha256"], "runtime": runtime,
              "predictors_completed": len(agreements), "calibration_comparisons_completed": len(agreements),
              "projection_alert_predictors": sum(r["projection_alert"] for r in agreements),
              "original_parity_gate_passed": False, "calibration_alerts_used_for_exclusion": False,
              "maximum_calibration_absolute_difference": max(r["max_absolute_prediction_difference"] for r in agreements),
              "calibration_absolute_tolerance": TOLERANCE, "train_prediction_rows": 78 * 655,
              "calibration_prediction_rows": 78 * 53, "batch_size": 1, "states_unchanged": True,
              "model_forward_calls": 78 * (655 + 53),
              "new_model_fits": 0, "optimizers_created": 0, "calibrators_refitted": 0,
              "test_predictions_generated": 0, "new_model_ranking": False,
              "checkpoint_lineage": spec["checkpoint_lineage"], "original_final_freeze_modified": False,
              "statistical_tests_executed": 0, "inference_plan_sha256": spec_sha,
              "statistical_plan_sha256": ACTIVE_ADMISSION["statistical_plan_sha256"],
              "scope": "New conditional CPU diagnostic, not original GPU parity or replacement of A2; all 78 finite frozen projections retained",
              "all_output_files": output_hashes(output)}
    write_new_json(output / "inference_report.json", report)
    return report


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
    report = {"version": VERSION, "stage": "latency", "status": "completed_CPU_latency_diagnostic", "spec_sha256": spec_sha,
              "checkpoint_index_sha256": spec["checkpoint_index_sha256"], "runtime": runtime,
              "cpu_diagnostic_prediction_report": previous, "raw_timed_operations": count, "paired_observations": 6000,
              "blocks": 100, "seed_pairs_per_candidate_per_block": 6, "candidates": 10,
              "warmup_calls_per_predictor": 32, "states_unchanged": True,
              "warmup_forward_calls": 66 * 32, "total_model_forward_calls": 12000 + 66 * 32,
              "estimand": spec["latency_policy"]["estimand"], "resampling_unit": spec["latency_policy"]["statistical_unit"],
              "statistical_tests_executed": 0, "new_model_fits": 0, "optimizers_created": 0,
              "calibrators_refitted": 0, "new_model_ranking": False, "A100_cost_replaced": False,
              "checkpoint_lineage": spec["checkpoint_lineage"], "original_final_freeze_modified": False,
              "inference_plan_sha256": spec_sha, "statistical_plan_sha256": ACTIVE_ADMISSION["statistical_plan_sha256"],
              "original_parity_gate_passed": False, "original_RQ3_IUT_executed": False,
              "scope": "New secondary CPU forward measurement on development windows; no original GPU parity or historical A100 cost replacement",
              "all_output_files": output_hashes(output)}
    write_new_json(output / "latency_report.json", report)
    return report



def main(argv=None):
    global ACTIVE_ADMISSION
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-plan", action="store_true")
    modes.add_argument("--run", action="store_true")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--parent-runner", type=Path)
    parser.add_argument("--base-spec", type=Path)
    parser.add_argument("--statistics-plan", type=Path)
    parser.add_argument("--supersedes-plan", type=Path)
    parser.add_argument("--supersession-reason", default="Implementation correction; all scientific policies unchanged")
    parser.add_argument("--statistical-plan-sha256")
    parser.add_argument("--stage", choices=("predictions", "latency"), default="predictions")
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = None
    started = time.time_ns()
    try:
        if args.prepare_plan:
            if not all((args.parent_runner, args.base_spec, args.statistics_plan)):
                raise ValueError("--prepare-plan requires --parent-runner, --base-spec and --statistics-plan")
            result = prepare_plan(args)
        else:
            with args.plan.open(encoding="utf-8") as handle:
                plan = json.load(handle)
            plan_sha = local_sha(args.plan)
            if plan.get("version") != VERSION or local_sha(__file__) != plan["runner_sha256"]:
                raise ValueError("The diagnostic runner or plan changed")
            parent = load_parent(plan["parent_runner"]["path"])
            require(local_sha(plan["base_spec"]["path"]) == plan["base_spec"]["sha256"] == BASE_SPEC_SHA,
                    "Preserved eager base specification changed")
            stats_path = regular_file(plan["statistics_plan_path"])
            require(args.statistical_plan_sha256 is not None and file_sha(stats_path) == args.statistical_plan_sha256,
                    "An exact independent statistics-plan checksum is required before inference")
            stats = read_json(stats_path)
            require(stats.get("inference_plan_sha256") == plan_sha, "Statistics plan does not bind this CPU inference plan")
            base = read_json(plan["base_spec"]["path"])
            parent.verify_inputs(base)
            require(args.output is not None, "Use a fresh --output directory")
            output = args.output.resolve()
            require(not output.exists(), "Preserve earlier attempts; use a fresh output directory")
            output.mkdir(parents=True)
            ACTIVE_ADMISSION = {"version": VERSION, "stage": args.stage,
                "inference_plan_path": str(args.plan.resolve()), "inference_plan_sha256": plan_sha,
                "statistical_plan_path": str(stats_path), "statistical_plan_sha256": args.statistical_plan_sha256,
                "parent_runner_sha256": PARENT_SHA, "base_spec_sha256": BASE_SPEC_SHA,
                "started_time_ns": started, "threshold_role": "alert_only_no_filtering",
                "original_parity_gate_passed": False, "new_model_fits_authorized": False}
            write_new_json(output / "run_admission.json", ACTIVE_ADMISSION)
            if args.stage == "predictions":
                result = prediction_stage(base, plan_sha, output)
            else:
                require(args.predictions_output is not None, "Latency needs --predictions-output")
                result = latency_stage(base, plan_sha, output, args.predictions_output)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
        return 0
    except Exception as error:
        result = {"version": VERSION, "status": "hard_failure", "stage": args.stage,
                  "error_type": type(error).__name__, "error": str(error), "started_time_ns": started,
                  "failed_time_ns": time.time_ns(), "new_model_fits": 0, "optimizers_created": 0,
                  "calibrators_refitted": 0, "original_parity_gate_passed": False,
                  "partial_outputs_preserved": output is not None and output.exists(),
                  "traceback": traceback.format_exc()}
        if output is not None and output.exists():
            write_new_json(output / "failure_report.json", result)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
