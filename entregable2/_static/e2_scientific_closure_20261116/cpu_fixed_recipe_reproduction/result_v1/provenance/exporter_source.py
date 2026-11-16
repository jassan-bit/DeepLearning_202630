"""Export a completed, registered CPU refit without fitting or rescoring models.

This program validates private physical checkpoints before omitting binary
weights. Public prediction arrays are derivatives of the saved JSON forecasts;
observed targets come from the independently audited cohort, joined by date and
the identical raw-derived daily CSV. It never imports Torch or model builders.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
SCIENCE = ROOT / "book_e2/_static/e2_scientific_closure_20261116"
STUDY = SCIENCE / "cpu_fixed_recipe_reproduction"
PLAN_SHA = "53704931391109bf2179985f5026af0316eacc892b8ebe8cfe62b87cd5aa5c7f"
PROGRAM_SHA = "0a8d5bfc58bdb8810f80493b0092fbe71f50ddb2299e89829a8dc0b9a1ea5c9a"
COHORT_SHA = "186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683"
HISTORICAL_SHA = "a3949762183627e5e7e5b39c9411c6d64214d726bfad03c83fc2ac3eeeea07e4"
COMPLETE = "completed_new_CPU_fixed_recipe_fit_cal_freeze_raw_test_A2_scope"
PHASE = "new_true_target_CPU_fixed_recipe_reproduction:"
SHARED = ("CPU-freeze-raw-score", "CPU-primary20000")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-8)


def finite(value, *, positive=False):
    x = float(value)
    require(math.isfinite(x) and (x > 0 if positive else x >= 0),
            "Invalid elapsed seconds")
    return x


def inside(root, relative):
    root = Path(root).resolve()
    rel = PurePosixPath(relative)
    require(not rel.is_absolute() and ".." not in rel.parts, "Unsafe artifact path")
    path = root.joinpath(*rel.parts)
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root),
            "Missing or escaping physical artifact: " + str(path))
    return path


def verified(root, relative, expected):
    path = inside(root, relative)
    require(sha(path) == expected, "Physical SHA256 mismatch: " + str(path))
    return path


def validate_status(status, identity, plan_sha, expected_ids):
    """First gate: no final artifacts are written for a partial or mixed run."""
    require(status.get("status") == COMPLETE, "CPU run is not complete; export refused")
    require(identity.get("plan_sha256") == plan_sha and identity.get("CPU_study") is True,
            "Run identity belongs to a different plan")
    require(identity.get("original_CUDA_parity") is False
            and identity.get("old_weights_or_predictions_used") is False,
            "Unexpected run scope")
    require(status.get("original_CUDA_parity") is False
            and status.get("full_original_campaign_passed") is False
            and status.get("HPO_repeated") is False
            and status.get("parent_wall_seconds_added") is False,
            "Unsupported completion or accounting claim")
    rows = status.get("jobs", [])
    ids = [r.get("id") for r in rows]
    require(len(ids) == len(expected_ids) == 83 and len(set(ids)) == 83
            and set(ids) == set(expected_ids), "Exactly the registered83 completed jobs are required")
    for row in rows:
        require(row.get("CPU_study_plan_sha256") == plan_sha,
                "Mixed plan in final status")
        require(row.get("status") == "completed" and row.get("returncode") == 0
                and row.get("parent_wall_seconds_added") is False,
                "Failed or unclosed worker in final status")
        finite(row["charged_worker_wall_seconds"])
    require(close(sum(float(r["charged_worker_wall_seconds"]) for r in rows),
                  status["charged_worker_seconds"]), "Status total differs from its83 workers")
    require(close(status["global_cap_sum_worker_seconds"], identity["global_cap_sum_worker_seconds"]),
            "Run cap differs from identity")
    require(status["charged_worker_seconds"] <= identity["global_cap_sum_worker_seconds"] + 1e-8,
            "Run exceeded its immutable worker cap")


def validate_ledger(raw):
    """Read-only validation, including closure; never initializes a ledger."""
    lines = raw.splitlines(keepends=True)
    require(bool(lines), "Missing ledger")
    rows, previous, active, finishes = [], None, {}, {}
    for line in lines:
        require(line.endswith(b"\n"), "Incomplete ledger record")
        row = json.loads(line)
        require(row.get("previous_record_sha256") == previous, "Broken ledger hash chain")
        previous = hashlib.sha256(line).hexdigest()
        if not rows:
            require(row.get("event") == "anchor"
                    and row.get("historical_budget_sha256") == HISTORICAL_SHA
                    and row.get("approved_total_hours") == 84,
                    "Unexpected historical84h anchor")
        else:
            key = row["reservation_id"]
            if row["event"] == "reserve":
                require(key not in active and key not in finishes, "Duplicate ledger reservation")
                finite(row["cap_seconds"], positive=True)
                active[key] = row
            elif row["event"] == "finish":
                require(key in active, "Unmatched ledger completion")
                held = active.pop(key)
                require(row["phase"] == held["phase"]
                        and close(row["cap_seconds"], held["cap_seconds"]), "Reservation binding changed")
                finite(row["elapsed_seconds"])
                finishes[key] = row
            else:
                raise ValueError("Unknown ledger event")
        rows.append(row)
    require(not active, "Ledger still has open reservations; wait for scientific jobs to finish")
    return rows, finishes


def closed_ledger(path):
    require(path.is_file() and not path.is_symlink(), "Physical active ledger required")
    with path.open("rb") as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        raw = stream.read()
        rows, finishes = validate_ledger(raw)
        fcntl.flock(stream, fcntl.LOCK_UN)
    return raw, rows, finishes


def validate_receipts(run, status, plan_sha, finishes, origin_outputs=None):
    files = sorted((run / "accounting").glob("*.json"))
    require(len(files) == 83, "Exactly83 physical accounting receipts required")
    expected = {r["id"]: r for r in status["jobs"]}
    seen, reservations = set(), set()
    rows = []
    for path in files:
        row = read(path)
        key = row["id"]
        require(path.stem == key and key in expected and key not in seen,
                "Unexpected, duplicate or missing worker receipt")
        require(row == expected[key], "Physical accounting receipt differs from final status")
        require(row["CPU_study_plan_sha256"] == plan_sha, "Mixed plan in receipt")
        reservation = row["reservation_id"]
        require(reservation not in reservations and reservation in finishes,
                "Missing or reused ledger completion")
        entry = finishes[reservation]
        require(entry["phase"] == PHASE + key and entry["returncode"] == 0
                and close(entry["elapsed_seconds"], row["charged_worker_wall_seconds"])
                and close(entry["cap_seconds"], row["reservation_cap_seconds"]),
                "Accounting receipt is not bound to its actual ledger charge")
        verified(run, "logs/" + key + ".log", row["log_sha256"])
        command = row["command"]
        origin = (origin_outputs or {}).get(key, str(run))
        for flag, value in [("--job", key), ("--output", origin),
                            ("--reservation-id", reservation)]:
            require(command.count(flag) == 1 and command[command.index(flag) + 1] == value,
                    "Worker command belongs to a different output/job")
        seen.add(key)
        reservations.add(reservation)
        rows.append(row)
    require(seen == set(expected), "Missing physical receipt")
    return rows


def array_sha(value):
    import numpy as np
    a = np.asarray(value)
    require(not a.dtype.hasobject, "Object arrays forbidden")
    a = np.ascontiguousarray(a.astype(a.dtype.newbyteorder("<"), copy=False))
    header = json.dumps({"shape": a.shape, "dtype": a.dtype.str}, sort_keys=True).encode()
    return hashlib.sha256(header + b"\0" + a.tobytes()).hexdigest()


def true_y_sha(value):
    import numpy as np
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode() + canonical(list(a.shape)) + a.tobytes()).hexdigest()


class Export:
    def __init__(self, args):
        self.args = args
        self.run = args.run.resolve()
        self.copy_jobs = {}
        self.sources = {}

    def note(self, source, role, expected=None):
        source = Path(source).absolute()
        require(source.is_file() and not source.is_symlink(), "Physical input missing: " + str(source))
        actual = sha(source)
        require(expected is None or actual == expected, "Input hash mismatch: " + str(source))
        key = str(source)
        self.sources[key] = {"path": key, "sha256": actual, "bytes": source.stat().st_size,
                             "role": role,
                             "included_in_export": self.sources.get(key, {}).get("included_in_export", False)}
        return source

    def copy(self, source, relative, role, expected=None):
        source = self.note(source, role, expected)
        require(relative not in self.copy_jobs or self.copy_jobs[relative] == source,
                "Export path collision")
        self.copy_jobs[relative] = source
        self.sources[str(source)]["included_in_export"] = True
        return source

    def validate(self):
        import numpy as np
        a, run = self.args, self.run
        require(not a.output.exists(), "Use a new output path; existing exports are immutable")
        plan = read(a.plan)
        require(sha(a.plan) == PLAN_SHA and plan["script_sha256"] == PROGRAM_SHA,
                "Only the approvedV2 CPU plan/source may be exported")
        frozen_path = Path(plan["fixed_recipe_plan"]["path"])
        frozen = read(frozen_path)
        require(sha(frozen_path) == plan["fixed_recipe_plan"]["sha256"], "Fixed recipes changed")
        jobs = frozen["jobs"]
        require(len(jobs) == 81 and sum(j["kind"] == "neural" for j in jobs) == 78
                and len({j["id"] for j in jobs}) == 81, "Invalid registered81 predictor roster")
        status, identity = read(run / "status.json"), read(run / "run_identity.json")
        validate_status(status, identity, PLAN_SHA, [j["id"] for j in jobs] + list(SHARED))
        self.recovery = None
        origin_outputs = {}
        if identity.get("recovery_plan_sha256"):
            recovery_dir = a.plan.parent.parent / "runtime_recovery_v3"
            recovery = read(recovery_dir / "recovery_plan.json")
            require(sha(recovery_dir / "recovery_plan.json") == identity["recovery_plan_sha256"]
                    and recovery["original_plan"]["sha256"] == PLAN_SHA
                    and recovery["original_driver"]["sha256"] == PROGRAM_SHA
                    and recovery["new_output"] == str(run)
                    and recovery["reused_predictors"] == 80
                    and recovery["new_neural_fits_permitted"] == 0
                    and close(recovery["new_output_worker_cap_seconds"], identity["global_cap_sum_worker_seconds"]),
                    "Recovery identity/source/cap differs")
            approval = read(recovery_dir / "QA_approved.json")
            require(approval["approved_for_ARIMA_name_compat_recovery"] is True
                    and approval["recovery_plan_sha256"] == identity["recovery_plan_sha256"]
                    and approval["original_CPU_plan_sha256"] == PLAN_SHA
                    and approval["original_CPU_driver_sha256"] == PROGRAM_SHA
                    and approval["wrapper_sha256"] == recovery["wrapper"]["sha256"]
                    and approval["compatibility_adapter_sha256"] == recovery["compatibility_adapter"]["sha256"]
                    and approval["permitted_new_neural_fits"] == 0,
                    "Missing or mismatched prior recovery QA")
            reused = read(recovery_dir / "reused80_inventory.json")
            require(sha(recovery_dir / "reused80_inventory.json") == recovery["reused80_inventory_sha256"]
                    and len(set(recovery["reused_job_ids"])) == 80
                    and set(recovery["remaining_job_ids"]) == {"6c395251bcaf9062-deterministic", *SHARED},
                    "Recovery reuse roster differs")
            for entry in reused["files"]:
                self.note(verified(run, entry["path"], entry["sha256"]), "byte-identical reusedV3 artifact", entry["sha256"])
            origin_outputs = {key: recovery["prior_output"] for key in recovery["reused_job_ids"]}
            for path in sorted(recovery_dir.rglob("*")):
                if path.is_file():
                    self.copy(path, "provenance/runtime_recovery_v3/" + path.relative_to(recovery_dir).as_posix(),
                              "registered ARIMA compatibility recovery provenance")
            for key in ["wrapper", "compatibility_adapter", "preparer"]:
                entry = recovery[key]
                self.copy(Path(entry["path"]), "provenance/" + Path(entry["path"]).name,
                          "recovery source: " + key, entry["sha256"])
            self.recovery = recovery
        # The live ledger is first opened only after the83-job completion gate.
        self.ledger_raw, ledger_rows, finishes = closed_ledger(Path(plan["active_ledger_path"]))
        receipts = validate_receipts(run, status, PLAN_SHA, finishes, origin_outputs)
        self.receipts = {r["id"]: r for r in receipts}
        ledger_path = str(Path(plan["active_ledger_path"]).absolute())
        self.sources[ledger_path] = {"path": ledger_path,
            "sha256": hashlib.sha256(self.ledger_raw).hexdigest(), "bytes": len(self.ledger_raw),
            "role": "closed read-only84h ledger snapshot", "included_in_export": True}
        self.copy(a.plan, "provenance/analysis_plan.json", "registered CPUV2 plan", PLAN_SHA)
        self.copy(frozen_path, "provenance/fixed_recipe_plan.json", "registered fixed recipes")
        self.copy(ROOT / "scripts/e2_cpu_fixed_recipe_reproduction.py",
                  "provenance/registered_cpu_driver.py", "executed registered adapter", PROGRAM_SHA)
        for key, filename in [("base_driver", "base_cuda_driver.py"), ("CPU_engine", "phase_cpu_engine.py")]:
            entry = plan[key]
            self.copy(Path(entry["path"]), "provenance/" + filename, key, entry["sha256"])
        self.copy(ROOT / "scripts/e2_corrections_budget.py", "provenance/budget_source.py",
                  "active accounting source", plan["budget_source_sha256"])
        self.copy(Path(plan["historical_budget_path"]), "provenance/historical_budget.json",
                  "closed historical84h anchor", HISTORICAL_SHA)
        for relative in ["status.json", "run_identity.json", "CPU_freeze_attestation.json",
                         "final_freeze.json", "pilot_power_MDE.json", "analysis_registration.json"]:
            self.copy(run / relative, "provenance/" + relative, "new CPU run provenance")
        for filename in ["qa_v2.json", "admission_v2/timing_admission.json"]:
            source = a.plan.parent.parent / filename
            value = read(source)
            require(value["plan_sha256"] == PLAN_SHA, "QA/admission plan differs")
            if filename.startswith("qa"):
                require(value["approved_for_new_CPU_replication"] is True
                        and value["script_sha256"] == PROGRAM_SHA, "Missing prior CPU QA")
            else:
                require(value["timing_admission_passed"] is True, "Timing admission failed")
            self.copy(source, "provenance/" + filename, "prior execution gate")
        source_root = Path(plan["frozen_source_root"])
        for receipt in receipts:
            command = receipt["command"]
            require(command[1] == str(ROOT / "scripts/e2_cpu_fixed_recipe_reproduction.py"),
                    "Unexpected worker program path")
            for flag, value in [("--plan", str(a.plan)), ("--source-root", str(source_root))]:
                require(command.count(flag) == 1 and command[command.index(flag) + 1] == value,
                        "Worker command source/plan binding differs")
        for entry in frozen["bound_sources"]:
            self.note(inside(source_root, entry["path"]), "bound frozen scientific input", entry["sha256"])
        author_roots = {r["command"][r["command"].index("--author-root") + 1] for r in receipts}
        require(len(author_roots) == 1, "Mixed author source roots")
        author_root = Path(next(iter(author_roots)))
        for entry in frozen["author_files"]:
            self.note(inside(author_root / entry["model"], entry["path"]),
                      "bound author source/license", entry["sha256"])
        raw_roots = {r["command"][r["command"].index("--raw-root") + 1] for r in receipts}
        require(len(raw_roots) == 1 and len(frozen["raw_archives"]) == 20,
                "Mixed or incomplete raw source roots")
        raw_root = Path(next(iter(raw_roots)))
        for entry in frozen["raw_archives"]:
            path = self.note(inside(raw_root, entry["relative_path"]), "admitted original raw archive, not bundled",
                             entry["sha256"])
            require(path.stat().st_size == entry["bytes"], "Raw archive size differs")
        for receipt in receipts:
            key = receipt["id"]
            self.copy(run / "accounting" / (key + ".json"), "accounting/" + key + ".json", "charged worker receipt")
            self.copy(run / "logs" / (key + ".log"), "logs/" + key + ".log", "original worker log", receipt["log_sha256"])
        cohort_dir = a.cohort_manifest.parent
        cohort = read(a.cohort_manifest)
        npz_path = self.copy(cohort_dir / cohort["file"], "inputs/current_cohort_arrays.npz",
                             "independently audited real cohort", COHORT_SHA)
        require(cohort["sha256"] == COHORT_SHA, "Cohort manifest changed")
        self.copy(a.cohort_manifest, "inputs/cohort_array_manifest.json", "audited array hashes/source bindings")
        self.copy(cohort_dir / "evaluation_aggregation_receipt.json", "inputs/evaluation_aggregation_receipt.json",
                  "independent20-archive aggregation")
        with np.load(npz_path, allow_pickle=False) as data:
            self.cohort = {key: data[key].copy() for key in data.files}
        for key, value in self.cohort.items():
            description = cohort["arrays"][key]
            require(list(value.shape) == description["shape"] and value.dtype.str == description["dtype"]
                    and array_sha(value) == description["array_sha256"], "Audited array hash differs: " + key)
        dates = self.cohort["evaluation_dates"].tolist()
        cal_dates = self.cohort["calibration_dates"].tolist()
        require(len(dates) == len(set(dates)) == 608 and dates[0] == "2025-01-01"
                and dates[-1] == "2026-08-31" and len(cal_dates) == 53, "Audited support changed")
        self.dates, self.cal_dates = dates, cal_dates
        freeze = read(run / "final_freeze.json")
        require([(r["model"], str(r["seed"])) for r in freeze["models"]]
                == [(j["model"], str(j["seed"])) for j in jobs], "Freeze81 roster/order differs")
        require(freeze["prospective_blinding_recovered"] is False
                and freeze["retune_from_final_scores_allowed"] is False
                and freeze["weights_scalers_and_calibrators_frozen_before_final_data"] is True,
                "Unexpected freeze claims")
        for relative, expected in freeze["frozen_files_sha256"].items():
            self.note(verified(run, relative, expected), "physical frozen artifact", expected)
        attestation = read(run / "CPU_freeze_attestation.json")
        require(attestation["new_CPU_plan_sha256"] == PLAN_SHA
                and attestation["physical_freeze_sha256"] == sha(run / "final_freeze.json")
                and attestation["predictors"] == 81
                and attestation["original_A100_or_bitwise_parity_claimed"] is False
                and attestation["historical_test_exposure"] is True
                and attestation["numeric_test_admission_within_this_pipeline_after_freeze"] is True,
                "CPU freeze attestation differs")
        self.predictors, self.phases, self.checkpoints = [], [], []
        predictions, calibrations = [], []
        mask = np.ones(608, dtype=bool)
        for job, frozen_row in zip(jobs, freeze["models"]):
            directory = run / job["directory"]
            result = read(directory / "result.json")
            require(result.get("status") == "completed" and result.get("model") == job["model"]
                    and result.get("seed") == job["seed"] and result.get("CPU_study_plan_sha256") == PLAN_SHA
                    and result.get("new_CPU_fit_scope") is True
                    and result.get("original_CUDA_reproduction_claimed") is False
                    and result.get("no_refit_after_calibration") is True,
                    "Wrong result identity/scope: " + job["id"])
            for rel, expected in result["artifacts_sha256"].items():
                self.note(verified(directory, rel, expected), "original fitted artifact", expected)
            metadata = [directory / "result.json", directory / "worker_spec.json"]
            for rel in result["artifacts_sha256"]:
                if Path(rel).suffix.lower() not in {".pt", ".pkl", ".pickle", ".pth"}:
                    metadata.append(directory / rel)
            for source in metadata:
                self.copy(source, "models/" + job["id"] + "/" + source.relative_to(directory).as_posix(), "fitted predictor metadata")
            for field in ["checkpoint", "scaler", "calibrator"]:
                path = verified(directory, result[field], result[field + "_sha256"])
                require(path.relative_to(run).as_posix() == frozen_row[field]
                        and result[field + "_sha256"] == frozen_row[field + "_sha256"], "Result/freeze binding changed")
            checkpoint = directory / result["checkpoint"]
            binary = checkpoint.suffix.lower() in {".pt", ".pth", ".pkl", ".pickle"}
            self.checkpoints.append({"model": job["model"], "seed": job["seed"], "job_id": job["id"],
                "source_path": str(checkpoint), "sha256": sha(checkpoint), "bytes": checkpoint.stat().st_size,
                "local_physical_sha_verified": True, "included": not binary,
                "reason": "Binary fitted weights omitted; public archive cannot independently verify omitted bytes" if binary else "Naive predictor JSON included as metadata"})
            cal_file = "calibration_predictions.json" if job["kind"] == "neural" else "predictions.json"
            cal = read(directory / cal_file)
            calibrator = read(directory / result["calibrator"])
            require(cal["dates"] == cal_dates and calibrator["dates"] == cal_dates
                    and calibrator["n"] == 53 and calibrator["predictor_sha256"] == result["checkpoint_sha256"]
                    and calibrator["sha256"] == digest({k: v for k, v in calibrator.items() if k != "sha256"})
                    and calibrator["calibration_predictions_sha256"] == digest(cal["prediction_original"])
                    and calibrator["calibration_targets_sha256"] == digest(cal["target_original"]), "Calibration binding differs")
            cal_prediction = np.asarray(cal["prediction_original"], dtype=np.float64)
            require(cal_prediction.shape == (53,) and np.isfinite(cal_prediction).all()
                    and np.array_equal(np.asarray(cal["target_original"]), self.cohort["calibration_y_original"]),
                    "Calibration support/targets differ from audited cohort")
            calibrations.append(cal_prediction)
            phase_seconds = None
            if job["kind"] == "neural":
                require(result["label_permutation_used"] is False
                        and result["true_training_y_scaled_sha256"] == true_y_sha(self.cohort["training_y_scaled"]),
                        "Result did not fit the audited true training targets")
                scaler = read(directory / result["scaler"])
                require(scaler["train_dates"] == self.cohort["training_dates"].tolist()
                        and np.array_equal(scaler["mean"], self.cohort["training_channel_mean"])
                        and np.array_equal(scaler["std"], self.cohort["training_channel_std"])
                        and close(scaler["target_scale"], self.cohort["training_target_scale"].item()), "Training scaler differs")
                epochs = job["selection"]["fixed_phase_epochs"]
                require(set(result["phase_records"]) == set(epochs), "Fixed phase recipe differs")
                phase_seconds = 0.0
                for name, phase in result["phase_records"].items():
                    require(read(directory / (name + "_phase.json")) == phase
                            and phase["fixed_epochs"] == epochs[name]
                            and len(phase["logs"]) == epochs[name], "Phase receipt differs from frozen epochs")
                    elapsed = finite(phase["wall_seconds"])
                    phase_seconds += elapsed
                    self.phases.append({"model": job["model"], "seed": job["seed"], "job_id": job["id"],
                        "phase": name, "phase_wall_seconds_nested_in_worker": elapsed,
                        "fixed_epochs": phase["fixed_epochs"], "optimizer_steps": phase["optimizer_steps"],
                        "scheduler_total_steps": phase["scheduler_total_steps"],
                        "initial_state_sha256": phase["initial_state_sha256"], "final_state_sha256": phase["final_state_sha256"]})
            prediction_path = run / "score" / ("predictions_" + hashlib.sha256(job["model"].encode()).hexdigest()[:16] + "_" + str(job["seed"]) + ".json")
            prediction = read(prediction_path)
            require(prediction["model"] == job["model"] and prediction["seed"] == job["seed"]
                    and prediction["dates"] == dates and prediction["mask_sha256"] == digest(mask.tolist())
                    and prediction["checkpoint_sha256"] == result["checkpoint_sha256"], "Saved608 forecasts are not bound to the new predictor")
            values = np.asarray(prediction["prediction_original"], dtype=np.float64)
            require(values.shape == (608,) and np.isfinite(values).all(), "Missing/nonfinite saved forecasts")
            predictions.append(values)
            self.copy(prediction_path, "predictions/" + prediction_path.name, "original saved608 forecasts")
            self.predictors.append({"model": job["model"], "seed": job["seed"], "job_id": job["id"], "kind": job["kind"],
                "charged_worker_wall_seconds": self.receipts[job["id"]]["charged_worker_wall_seconds"],
                "body_calibration_wall_seconds_nested_in_worker": finite(result["new_job_wall_seconds"]),
                "phase_wall_seconds_nested_in_worker": phase_seconds,
                "actual_predictor_parameters": plan["parameter_counts"].get(job["model"]),
                "checkpoint_sha256": result["checkpoint_sha256"], "calibrator_sha256": result["calibrator_sha256"]})
        require(len(self.phases) == 90 and sum(not r["included"] for r in self.checkpoints) == 79,
                "Expected90 phases and79 omitted binary checkpoints")
        self.arrays = {"dates": np.asarray(dates), "model": np.asarray([j["model"] for j in jobs]),
            "seed": np.asarray([str(j["seed"]) for j in jobs]), "prediction_original": np.stack(predictions),
            "observed_y_original": self.cohort["evaluation_y_original"], "mask": mask,
            "calibration_dates": np.asarray(cal_dates), "calibration_prediction_original": np.stack(calibrations),
            "calibration_y_original": self.cohort["calibration_y_original"],
            "checkpoint_sha256": np.asarray([p["checkpoint_sha256"] for p in self.predictors]),
            "calibrator_sha256": np.asarray([p["calibrator_sha256"] for p in self.predictors])}
        score = read(run / "score/result.json")
        require(score["status"] == "completed" and score["systems"] == 16
                and score["actual_new_E2_final_metrics"] is True
                and score["freeze_sha256"] == digest(freeze) and score["retuning_performed"] is False,
                "New forecast analysis did not complete")
        for relative, expected in score["artifacts_sha256"].items():
            self.note(verified(run / "score", relative, expected), "original scored artifact", expected)
        admission = read(run / "score/data_admission.json")
        aggregate = read(cohort_dir / "evaluation_aggregation_receipt.json")
        require(admission["freeze_sha256"] == digest(freeze)
                and admission["common_eligible_dates"] == dates and admission["common_eligible_count"] == 608
                and admission["blindness_recovered"] is False
                and admission["known_historical_E1_period"] is True
                and admission["previous_E1_exposure_disclosed"] is True
                and admission["new_independent_prospective_confirmation"] is False
                and admission["model_refits_or_retuning_after_access"] is False,
                "Test admission is not bound to the same608 support/freeze")
        daily = run / "private-known-E1-data/known_E1_daily.csv"
        self.copy(daily, "inputs/known_E1_daily.csv", "new pipeline20-archive daily derivative",
                  admission["daily_market_sha256"])
        require(sha(daily) == aggregate["daily_csv_sha256"],
                "Target-source join refused: daily source bytes differ from audited cohort")
        raw_expected = {r["archive"].removeprefix("BTCUSDT-1m-").removesuffix(".zip"): r["sha256"] for r in frozen["raw_archives"]}
        for quality in [admission["quality_by_archive"], aggregate["quality_by_archive"]]:
            require(len(quality) == 20 and {r["month"]: r["sha256"] for r in quality} == raw_expected,
                    "Twenty raw archive bindings differ")
        for filename in ["result.json", "benchmark_A2.csv", "benchmark_analysis.json", "data_admission.json",
                         "benchmark_analysis_20000.json", "primary20000_completion.json", "scoring_started.json"]:
            self.copy(run / "score" / filename, "analysis/" + filename, "existing new CPU forecast analysis")
        analysis = read(run / "score/benchmark_analysis.json")
        primary = read(run / "score/benchmark_analysis_20000.json")
        completion = read(run / "score/primary20000_completion.json")
        require(completion["source_plan_sha256"] == PLAN_SHA and completion["draws"] == 20000
                and completion["new_forecasts_used"] is True
                and completion["output_sha256"] == sha(run / "score/benchmark_analysis_20000.json"),
                "Missing registered20k completion")
        expected_models = set(j["model"] for j in jobs)
        for value in [analysis, primary]:
            require(len(value["table_A2"]) == 16 and {r["model"] for r in value["table_A2"]} == expected_models
                    and len(value["primary_contrasts"]) == 20 and value["n_common_dates"] == 608
                    and value["common_mask_sha256"] == digest(mask.tolist())
                    and value["analysis_registration_sha256"] == freeze["analysis_registration_sha256"], "Analysis support/registration differs")
        point_rows = {r["model"]: r for r in analysis["table_A2"]}
        for row in self.predictors:
            job = next(j for j in jobs if j["id"] == row["job_id"])
            values = point_rows[row["model"]]["RMSE_per_seed"]
            index = frozen["seeds"].index(job["seed"]) if job["kind"] == "neural" else 0
            row["RMSE_copied_from_existing_analysis"] = values[index]
        self.model_rows = []
        for name in frozen["neural_models"]:
            group = [p for p in self.predictors if p["model"] == name]
            times = [p["charged_worker_wall_seconds"] for p in group]
            phase_times = [p["phase_wall_seconds_nested_in_worker"] for p in group]
            self.model_rows.append({"model": name, "R": 6, "actual_predictor_parameters": plan["parameter_counts"][name],
                "charged_worker_wall_seconds_sum": sum(times), "charged_worker_wall_seconds_mean": statistics.mean(times),
                "charged_worker_wall_seconds_sample_SD": statistics.stdev(times),
                "phase_wall_seconds_sum_nested_in_workers": sum(phase_times),
                "phase_wall_seconds_mean_nested_in_workers": statistics.mean(phase_times),
                "phase_wall_seconds_sample_SD_nested_in_workers": statistics.stdev(phase_times),
                "RMSE_copied_from_existing_analysis": point_rows[name]["RMSE"],
                "RMSE_seed_SD_copied_from_existing_analysis": point_rows[name]["RMSE_seed_SD"]})
        recovery_dir = a.plan.parent.parent / "runtime_recovery_v2"
        recovery = read(recovery_dir / "recovery_decision.json")
        v3_failed_seconds = 0.0
        if self.recovery:
            v3_path = a.plan.parent.parent / "runtime_recovery_v3/failed_ARIMA_accounting.json"
            v3_failed = read(v3_path)
            entry = finishes.get(v3_failed["reservation_id"])
            require(v3_failed["CPU_study_plan_sha256"] == PLAN_SHA
                    and v3_failed["id"] == "6c395251bcaf9062-deterministic"
                    and v3_failed["returncode"] == 1 and entry
                    and entry["phase"] == PHASE + v3_failed["id"]
                    and entry["returncode"] == v3_failed["returncode"]
                    and close(entry["elapsed_seconds"], v3_failed["charged_worker_wall_seconds"]),
                    "PriorV3 ARIMA failure charge differs")
            self.note(verified(v3_path.parent, "failed_ARIMA.log", v3_failed["log_sha256"]),
                      "preserved failedARIMA log", v3_failed["log_sha256"])
            v3_failed_seconds = finite(v3_failed["charged_worker_wall_seconds"])
            require(close(v3_failed_seconds, self.recovery["priorV3_failed_ARIMA_worker_seconds"]),
                    "Recovery cap does not bind the actual ARIMA failure")
        require(recovery["registered_plan_sha256"] == PLAN_SHA
                and recovery["registered_program_sha256"] == PROGRAM_SHA
                and recovery["new_fits_in_failed_attempt"] == 0
                and close(recovery["remaining_retry_cap_seconds"], identity["global_cap_sum_worker_seconds"] + v3_failed_seconds),
                "Prior failed attempt/cap binding differs")
        failed_seconds = 0.0
        failed_count = 0
        for path in sorted(recovery_dir.iterdir()):
            if not path.is_file():
                continue
            self.copy(path, "provenance/runtime_recovery_v2/" + path.name, "preserved failed dependency attempt")
            if path.suffix == ".json":
                value = read(path)
                if "reservation_id" in value:
                    finished = finishes.get(value["reservation_id"])
                    require(value["CPU_study_plan_sha256"] == PLAN_SHA and value["returncode"] != 0
                            and finished and finished["phase"] == PHASE + value["id"]
                            and finished["returncode"] == value["returncode"]
                            and close(finished["elapsed_seconds"], value["charged_worker_wall_seconds"]),
                            "Prior failed charge is unverified")
                    failed_seconds += finite(value["charged_worker_wall_seconds"])
                    failed_count += 1
                    self.note(verified(recovery_dir, path.stem + ".txt", value["log_sha256"]),
                              "preserved failed worker log", value["log_sha256"])
        require(failed_count == 2 and close(failed_seconds, recovery["previous_failed_worker_seconds_charged"]),
                "Prior failed worker total differs")
        charged = sum(r["charged_worker_wall_seconds"] for r in receipts)
        require(charged + failed_seconds + v3_failed_seconds <= recovery["original_replication_cap_seconds"] + 1e-8,
                "Combined attempts exceed the original10000s cap")
        self.accounting = {"canonical_completed_worker_count": 83, "new_neural_fits_across_study": 78,
            "classical_reconstructions": 3, "shared_analysis_workers": 2,
            "canonical83_sum_worker_wall_seconds": charged,
            "reused_successful_workers_from_V3": 80 if self.recovery else 0,
            "new_workers_in_final_retry": 3 if self.recovery else 83,
            "new_neural_fits_in_final_retry": 0 if self.recovery else 78,
            "prior_failed_dependency_workers": 2, "prior_failed_dependency_seconds": failed_seconds,
            "prior_failed_ARIMA_workers": 1 if self.recovery else 0,
            "prior_failed_ARIMA_seconds": v3_failed_seconds,
            "combined_study_worker_wall_seconds": charged + failed_seconds + v3_failed_seconds,
            "original_group_cap_seconds": recovery["original_replication_cap_seconds"],
            "retry_cap_seconds": identity["global_cap_sum_worker_seconds"],
            "phase_times_nested_in_worker_charges": True, "parent_wall_seconds_added": False,
            "all83_canonical_and_prior_failed_receipts_match_closed_ledger": True,
            "closed_ledger_sha256": hashlib.sha256(self.ledger_raw).hexdigest(),
            "historical_budget_sha256": HISTORICAL_SHA, "budget_reset": False,
            "accounting_unit": "sum of individual worker elapsed seconds, including imports/failures; not GPU utilization or monetary cost",
            "ledger_snapshot_is_closed_at_export": True,
            "whole_campaign_costs_not_reestimated_or_added_to_this_study": True}
        self.descriptor = {"representation": "derivative of saved forecast JSON; exact float64 values, without model forwards or rescoring",
            "forecast_rows": 81, "evaluation_dates": 608, "calibration_dates": 53,
            "forecast_source": "81 copied predictions/*.json bound to the new frozen checkpoints",
            "target_source": "inputs/current_cohort_arrays.npz; evaluation_y_original, not the score forecast JSON",
            "target_npz_sha256": COHORT_SHA, "target_join": "exact608 date equality, all-common support, identical20 archive hashes and byte-identical raw-derived daily CSV",
            "daily_csv_sha256": admission["daily_market_sha256"]}
        self.status = status
        return self

    def emit(self):
        import numpy as np
        output = self.args.output
        output.mkdir(parents=True, exist_ok=False)
        for relative, source in sorted(self.copy_jobs.items()):
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            require(sha(destination) == self.sources[str(source)]["sha256"], "Copy changed: " + relative)
        (output / "provenance/closed_compute_ledger.jsonl").write_bytes(self.ledger_raw)
        np.savez_compressed(output / "predictions_81.npz", **self.arrays)
        self.json("prediction_array_descriptor.json", {**self.descriptor,
            "sha256": sha(output / "predictions_81.npz"),
            "arrays": {k: {"shape": list(v.shape), "dtype": v.dtype.str, "array_sha256": array_sha(v)} for k, v in self.arrays.items()}})
        worker_rows = []
        predictors = {r["job_id"]: r for r in self.predictors}
        for key, receipt in sorted(self.receipts.items()):
            worker_rows.append({"job_id": key, "role": predictors[key]["kind"] if key in predictors else "shared_score" if key == SHARED[0] else "shared_primary20000",
                "model": predictors.get(key, {}).get("model", ""), "seed": predictors.get(key, {}).get("seed", ""),
                "charged_worker_wall_seconds": receipt["charged_worker_wall_seconds"],
                "reservation_id": receipt["reservation_id"], "reservation_cap_seconds": receipt["reservation_cap_seconds"],
                "returncode": receipt["returncode"], "cpu_affinity": json.dumps(receipt["cpu_affinity"]),
                "log_sha256": receipt["log_sha256"]})
        self.csv("worker_costs_83.csv", worker_rows)
        self.csv("predictor_costs_81.csv", self.predictors)
        self.csv("neural_model_costs_13.csv", self.model_rows)
        self.csv("classical_model_costs_3.csv", [r for r in self.predictors if r["kind"] == "classical"])
        self.csv("shared_stage_costs_2.csv", [r for r in worker_rows if r["job_id"] in SHARED])
        self.csv("phase_costs_90.csv", self.phases)
        self.csv("checkpoint_inventory_81.csv", self.checkpoints)
        self.json("excluded_checkpoint_inventory.json", {"files": [r for r in self.checkpoints if not r["included"]],
            "scope": "Exporter verified all physical SHA256 values locally before omission; public readers cannot reverify omitted checkpoint bytes or rerun fitted models from this evidence-only export."})
        self.json("accounting_summary.json", self.accounting)
        self.json("source_inputs_manifest.json", {"inputs": list(self.sources.values()),
            "closed_ledger_snapshot_sha256": hashlib.sha256(self.ledger_raw).hexdigest()})
        shutil.copyfile(__file__, output / "provenance/exporter_source.py")
        self.json("export_receipt.json", {"exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_run": str(self.run), "CPU_plan_sha256": PLAN_SHA, "registered_driver_sha256": PROGRAM_SHA,
            "run_identity_sha256": sha(self.run / "run_identity.json"), "status_sha256": sha(self.run / "status.json"),
            "exporter_sha256": sha(__file__), "new_fits_or_forwards_performed_by_exporter": 0,
            "inferential_statistics_recomputed": False, "point_metrics_recomputed": False,
            "new_posthoc_CPU_fixed_recipe_scope_completed": True, "original_CUDA_parity": False,
            "full_original_campaign_passed": False, "HPO_repeated": False, "historical_test_exposure": True,
            "prospective_blinding_recovered": False, "fresh_model_latency_or_cost_IUT_executed": False})
        (output / "README.md").write_text(
            "# Réplica científica CPU con recetas fijas: evidencia final\n\n"
            "Ejecución posterior nueva: 78 ajustes supervisados reales (13 cuerpos ×6 semillas), "
            "tres reconstrucciones clásicas, calibración53, congelamiento81, admisión20ZIP "
            "y evaluación608. Las dos etapas compartidas completan83recibos. Se conserva "
            "la exposición histórica al test; no se recupera ceguera prospectiva.\n\n"
            "`analysis/` copia la nueva A2 y el análisis primario de20000remuestreos ya ejecutado. "
            "Este exportador no ajusta modelos, no hace forwards y no calcula métricas ni contrastes. "
            "`predictions/` conserva los81JSON originales. `predictions_81.npz` deriva sus valores "
            "exactos; los targets provienen del NPZ auditado, unidos por608fechas y fuente diaria "
            "idéntica. El descriptor documenta los hashes y el origen separado de los targets.\n\n"
            "El coste cobrado es la suma de83workers incluyendo importaciones;90tiempos de fase "
            "están contenidos en esos workers. No sumar fases o walltime del padre al cargo. "
            "Los fallos previos se conservan:dos de dependencia(5.254804967s) y,cuando aplica, "
            "ARIMA por el alias del recibo(3.327804276s). V4 reutiliza80workers verificados deV3 "
            "y sólo ejecuta ARIMA+score+primaria:no son83workers nuevos en el retry. Estos fallos "
            "se añaden al coste del estudio, "
            "sin reiniciar el presupuesto84h. No se redistribuyen etapas compartidas por modelo.\n\n"
            "Los79checkpoints binarios se comprobaron localmente por SHA256 antes de excluirlos; "
            "el inventario declara hashes y tamaños. El profesor no puede verificar bytes omitidos "
            "ni reconstruir esos pesos usando sólo este export. Los dos predictores ingenuos JSON "
            "sí se incluyen. La reproducción de ajustes requiere código, fuentes, datos y cómputo "
            "del paquete correspondiente.\n\n"
            "Alcance: réplica CPU con recetas históricas fijadas, sin repetirHPO; NAS no aplica "
            "a cuerpos definidos. No implica paridad A100/bitwise, reproducción del HPO original, "
            "IUT de coste/latencia para estos modelos nuevos ni cierre de toda la campaña original.\n",
            encoding="utf-8")
        files = [{"path": p.relative_to(output).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)}
                 for p in sorted(output.rglob("*")) if p.is_file()]
        self.json("manifest.json", {"version": "e2-new-CPU-fixed-recipe-export-v1",
            "CPU_plan_sha256": PLAN_SHA, "files": files,
            "binary_checkpoints_included": False, "all_export_files_hashed_except_this_manifest": True})
        return {"output": str(output), "files": len(files) + 1, "predictors": 81,
                "accounting_workers": 83, "phase_records": 90, "status": "exported_verified_new_CPU_scope"}

    def json(self, relative, value):
        path = self.args.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")

    def csv(self, relative, rows):
        with (self.args.output / relative).open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("/workspace/work/e2-true-target-cpu-reproduction-v4"))
    parser.add_argument("--plan", type=Path, default=STUDY / "plan_v2/analysis_plan.json")
    parser.add_argument("--cohort-manifest", type=Path, default=SCIENCE / "data_audit/cohort_array_manifest.json")
    parser.add_argument("--output", type=Path, default=STUDY / "result_v1")
    parser.add_argument("--check", action="store_true", help="Validate a completed run without writing output")
    args = parser.parse_args(argv)
    for name in ["run", "plan", "cohort_manifest", "output"]:
        setattr(args, name, getattr(args, name).absolute())
    exporter = Export(args).validate()
    result = {"status": "verified_completed_CPU_scope", "output_written": False} if args.check else exporter.emit()
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError) as error:
        print("CPU evidence export refused: " + str(error), file=sys.stderr)
        sys.exit(2)
