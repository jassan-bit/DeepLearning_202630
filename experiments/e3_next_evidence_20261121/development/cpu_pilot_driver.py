"""Bounded NEW CPU development pilot; default CLI cannot train implicitly.

Preparation admits only the three historical training NPZ arrays. Execution
requires a separately verified public plan/source anchor and software QA. The
old V3/GPU campaigns and their results are never resumed or overwritten here.
Prepared with declared AI assistance; this is not confirmation or human review.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid

VERSION = "e3-v4-new-CPU-development-pilot-1"
NPZ_SHA = "186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683"
NPZ_BYTES = 460090
RECIPES = ("sigma_crps_v4", "u_crps_control", "mse_only_control", "additive_positive_head")
ANALYTICAL = ("floor_only", "floor_plus_mean_excess")
ANALYTICAL_PARAMETERS = {"floor_only": 0, "floor_plus_mean_excess": 1}
SEEDS = tuple(range(20261121, 20261126))
PARAMETERS = {name: 1210 if name == "additive_positive_head" else 1262 for name in RECIPES}
CONFIG = {"learning_rate": 0.003, "weight_decay": 0.0, "batch_size": 32,
          "max_epochs": 40, "patience": 8, "gradient_clip": 1.0}
RULE = {"minimum_relative_mean_RMSE_gain_against_best_neural_control": 0.02,
        "minimum_relative_mean_RMSE_gain_against_best_all_five_controls": 0.02,
        "scope": "new development continuation only; not confirmatory delta or significance",
        "failure": "abandon current V4 recipe; never promote a control automatically"}
PHASE_SECONDS = 120.0
PROCESS_SECONDS = 150.0
TOTAL_SECONDS = 3300.0
REQUIRED_SOURCE = {
    "cpu_pilot_driver.py", "outcome_crps_v4.py",
    "pilot_sources/src/original_model.py", "pilot_sources/src/e3_development.py",
    "pilot_sources/src/e3_innovation_model.py", "pilot_sources/src/e3_overlap_model_v3.py",
    "pilot_sources/experiments/e3_v3_development_pilot_20261120/pilot_engine.py",
    "pilot_sources/experiments/e3_evaluacion_ampliada_20261119/evaluation/phase.py",
    "pilot_sources/experiments/e3_ficha_cnn_lstm_atencion_v2_20261116/verification/quadrature_k128.json",
    "pilot_sources/experiments/e3_ficha_cnn_lstm_atencion_v2_20261116/verification/quadrature_k128.json.sha256",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_pairs)


def no_symlink_path(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Symlink path/ancestor refused")
    return path


def guarded_write_path(path):
    path = no_symlink_path(path)
    if path.exists():
        require(path.is_file() and path.stat().st_nlink == 1, "Aliased/nonregular output file refused")
    return path


def guard_case_tree(case_root):
    """Resume cannot append through a tainted log/temp/checkpoint alias."""
    for path in case_root.rglob("*"):
        no_symlink_path(path)
        require(path.is_dir() or (path.is_file() and path.stat().st_nlink == 1),
                "Aliased/nonregular member in owned case")


def write_json(path, value, *, immutable=False):
    path = guarded_write_path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        require(read_json(path) == value, "Immutable record differs: " + path.name)
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".partial-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def verify_sources(root, manifest_path):
    root = no_symlink_path(root); manifest_path = no_symlink_path(manifest_path)
    manifest = read_json(manifest_path)
    records = manifest.get("files"); require(isinstance(records, list), "Source manifest files required")
    seen = set()
    for item in records:
        name = item.get("path"); path = PurePosixPath(name or "")
        require(isinstance(name, str) and name and path.as_posix() == name
                and not path.is_absolute() and "\\" not in name
                and all(p not in ("", ".", "..") for p in name.split("/")), "Unsafe source member")
        require(name not in seen, "Duplicate source member"); seen.add(name)
        target = no_symlink_path(root / name)
        require(target.is_file() and target.stat().st_size == item.get("bytes")
                and sha(target) == item.get("sha256"), "Source member differs: " + name)
    require(REQUIRED_SOURCE <= seen, "Source manifest omits an executable dependency")
    return sha(manifest_path)


def validate_npz(path):
    path = no_symlink_path(path)
    require(path.is_file() and path.stat().st_size == NPZ_BYTES and sha(path) == NPZ_SHA,
            "Pinned NPZ transport identity differs")
    return path


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "Frozen module missing")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def scientific_runtime(root):
    """Called only after source/input/anchor guards; constructs no model."""
    frozen = Path(root) / "pilot_sources"
    sys.path.insert(0, str(frozen)); sys.path.insert(1, str(root))
    import numpy as np
    import torch
    require(platform.python_version() == "3.12.14" and torch.__version__ == "2.8.0+cpu"
            and np.__version__ == "2.3.5", "New declared CPU software cohort differs")
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        require(torch.get_num_interop_threads() == 1, "Interop thread cohort differs")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from src.e3_innovation_model import load_innovation_data, InnovationModel
    from outcome_crps_v4 import OutcomeCRPSDevelopmentModel
    require(Path(sys.modules["src.e3_innovation_model"].__file__).resolve()
            == (frozen / "src/e3_innovation_model.py").resolve(), "Frozen dependency module cache differs")
    require(Path(sys.modules["src.e3_overlap_model_v3"].__file__).resolve()
            == (frozen / "src/e3_overlap_model_v3.py").resolve(), "Frozen V3 dependency module cache differs")
    old = module_from_file("e3_v4_frozen_pilot_helpers", frozen / "experiments/e3_v3_development_pilot_20261120/pilot_engine.py")
    phase = module_from_file("e3_v4_frozen_optimizer_phase", frozen / "experiments/e3_evaluacion_ampliada_20261119/evaluation/phase.py")
    cohort = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
              "platform": platform.platform(), "device": "cpu", "cpu_threads": torch.get_num_threads(),
              "interop_threads": torch.get_num_interop_threads(),
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "environment_thread_limits": {name: os.environ.get(name) for name in
                                             ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
              "energy_FLOPs_or_CPU_capacity_measured": False}
    return np, torch, load_innovation_data, InnovationModel, OutcomeCRPSDevelopmentModel, old, phase, cohort


def admit(runtime, npz):
    data = runtime[2](npz, NPZ_SHA); split = runtime[5].split_development(data)
    require(data["admission"]["keys_read"] == ["training_raw", "training_y_original", "training_dates"]
            and data["admission"]["forbidden_arrays_loaded"] is False, "Training-only loader contract changed")
    require(len(data["train_X"]) == 532 and len(data["val_X"]) == 116
            and tuple(map(len, (split["select"], split["purged"], split["assessment"]))) == (54, 7, 55),
            "Fixed exposed development geometry differs")
    admission = {**data["admission"], "scope": "previously_exposed_development_only",
                 "selection_dates": data["val_dates"][split["select"]].tolist(),
                 "purged_dates": data["val_dates"][split["purged"]].tolist(),
                 "assessment_dates": data["val_dates"][split["assessment"]].tolist(),
                 "assessment_previously_exposed": True,
                 "evaluation608_labels_deserialized": False, "calibration_arrays_deserialized": False,
                 "whole_NPZ_bytes_hashed_only_for_transport_integrity": True}
    return data, split, admission


def jobs():
    return [{"index": i, "name": name, "seed": seed,
             "job_id": "seed" + str(seed) + "/" + name,
             "parameters": PARAMETERS[name]} for i, (seed, name) in
            enumerate((seed, name) for seed in SEEDS for name in RECIPES)]


def prepare(root, manifest_path, npz, case_root):
    source_hash = verify_sources(root, manifest_path); npz = validate_npz(npz)
    case_root = no_symlink_path(case_root)
    require(case_root.name.startswith("v4-cpu-dev-") and not case_root.exists(),
            "A fresh v4-cpu-dev-* case is required; existing bytes preserved")
    runtime = scientific_runtime(root); data, split, admission = admit(runtime, npz)
    case_root.mkdir(parents=True); (case_root / "inputs").mkdir()
    target = case_root / "inputs/current_cohort_arrays.npz"
    shutil.copyfile(npz, target); target.chmod(0o444); validate_npz(target)
    plan = {"version": VERSION, "case_id": "v4-cpu-dev-" + uuid.uuid4().hex,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "base_commit": "f5c960eefedb5fc49b313dd795b1ec84e784bd78",
            "scope": "new_exposed_CPU_development_not_confirmation_or_reproduction",
            "source_manifest_sha256": source_hash, "admission_sha256": digest(admission),
            "npz_sha256": NPZ_SHA, "npz_bytes": NPZ_BYTES,
            "cohort": runtime[7], "config": CONFIG, "recipes": list(RECIPES),
            "analytical_controls": list(ANALYTICAL),
            "analytical_parameters": ANALYTICAL_PARAMETERS,
            "analytical_optimizer_trainable_parameters": {name: 0 for name in ANALYTICAL},
            "analytical_parameter_meaning": "floor-only has no fitted parameter; mean-excess uses one training-derived scalar d_fit, no optimizer",
            "seeds": list(SEEDS), "jobs": jobs(), "parameters": PARAMETERS,
            "capacity_matched": False, "atomic_controls_identical_architecture_and_initial_values": True,
            "additive_control_shares_encoder_attention_not_full_readout": True,
            "fit_rows": 532, "stop_rows": 54, "purged_rows": 7, "assessment_rows": 55,
            "assessment_predictions_expected": 1210, "neural_assessment_predictions_expected": 1100,
            "analytical_assessment_predictions_expected": 110, "assessment_trajectories_expected": 22,
            "analytical_controls_evaluated_once_not_repeated_per_seed": True,
            "analytical_d_fit_source": "training532 mean(target-sqrt(A)); no assessment fitting",
            "bounded_controls_omitted": "fixed_probs and atom_plugin are not rerun; this is not full-family planning",
            "HPO_fits": 0,
            "learning_rate_rationale": "fixed shared0.003 transferred from prior common development choice; not optimized forV4",
            "control_identity_aliases": {"u_crps_control": "original atom_crps loss and architecture",
                                         "mse_only_control": "original atom_mse_only loss and architecture",
                                         "additive_positive_head": "original additive control architecture and loss"},
            "new_designs_in_this_pilot": 1, "paired_seeds_are_not_new_designs": True,
            "budget": {"phase_seconds_per_fit": PHASE_SECONDS, "process_seconds_per_fit": PROCESS_SECONDS,
                       "total_optimizer_nominal_seconds": 2400.0, "end_to_end_max_seconds": TOTAL_SECONDS,
                       "max_epochs_per_fit": 40, "deadline_includes_restart_downtime": True},
            "assessment_policy": "all20 trained checkpoint jobs validated before any55-date assessment forward",
            "continuation_rule": RULE,
            "complete_family_planning_pilot": False, "statistical_significance_or_power_claimed": False,
            "old_GPU_weights_used_for_fair_new_comparison": False,
            "original_V3_NO_GO_preserved": True, "R1_inherited": False, "R3_accredited": False,
            "new_real_data_fits": 0, "new_assessment_predictions": 0,
            "old796_old30_and_old20400_mutated": False, "AI_assistance_declared": True}
    write_json(case_root / "admission.json", admission, immutable=True)
    write_json(case_root / "cpu_pilot_plan.json", plan, immutable=True)
    write_json(case_root / "prepared_receipt.json", {"status": "prepared_zero_fits_waiting_public_anchor_and_QA",
               "plan_sha256": digest(plan), "plan_file_sha256": sha(case_root / "cpu_pilot_plan.json"),
               "plan_sha256_definition": "SHA256 of sorted compact JSON canonical(plan)",
               "source_manifest_sha256": source_hash,
               "case_id": plan["case_id"], "new_real_data_fits": 0,
               "new_assessment_predictions": 0, "R3_accredited": False}, immutable=True)
    return plan


def check_plan(plan):
    require(plan.get("version") == VERSION and plan.get("jobs") == jobs()
            and plan.get("recipes") == list(RECIPES) and plan.get("seeds") == list(SEEDS)
            and plan.get("config") == CONFIG and plan.get("parameters") == PARAMETERS,
            "Fixed pilot specification differs")
    require(plan.get("scope") == "new_exposed_CPU_development_not_confirmation_or_reproduction"
            and plan.get("R1_inherited") is False and plan.get("R3_accredited") is False,
            "Development identity/scope differs")
    require(plan.get("continuation_rule") == RULE and plan.get("HPO_fits") == 0
            and [plan.get(k) for k in ("fit_rows", "stop_rows", "purged_rows", "assessment_rows",
                                     "assessment_predictions_expected")] == [532, 54, 7, 55, 1210]
            and plan.get("analytical_controls") == list(ANALYTICAL)
            and plan.get("analytical_parameters") == ANALYTICAL_PARAMETERS
            and plan.get("neural_assessment_predictions_expected") == 1100
            and plan.get("analytical_assessment_predictions_expected") == 110
            and plan.get("assessment_trajectories_expected") == 22
            and plan.get("analytical_controls_evaluated_once_not_repeated_per_seed") is True
            and plan.get("capacity_matched") is False
            and plan.get("old_GPU_weights_used_for_fair_new_comparison") is False
            and plan.get("complete_family_planning_pilot") is False
            and plan.get("statistical_significance_or_power_claimed") is False
            and plan.get("original_V3_NO_GO_preserved") is True,
            "Fixed treatment, continuation rule or inferential scope differs")
    require(plan.get("budget") == {"phase_seconds_per_fit": PHASE_SECONDS,
            "process_seconds_per_fit": PROCESS_SECONDS, "total_optimizer_nominal_seconds": 2400.0,
            "end_to_end_max_seconds": TOTAL_SECONDS, "max_epochs_per_fit": 40,
            "deadline_includes_restart_downtime": True}, "Fixed resource budget differs")


def verify_anchor(plan, receipt):
    require(bool(re.fullmatch("[0-9a-f]{40}", receipt.get("public_anchor_commit", "")))
            and receipt.get("plan_sha256") == digest(plan)
            and receipt.get("source_manifest_sha256") == plan["source_manifest_sha256"]
            and receipt.get("development_only_not_confirmation") is True
            and receipt.get("software_QA_passed") is True, "Public plan/source anchor and software QA required")


def load_case(root, manifest_path, case_root, anchor_path):
    case_root = no_symlink_path(case_root)
    require(case_root.is_dir() and case_root.name.startswith("v4-cpu-dev-"), "Unknown development case")
    guard_case_tree(case_root)
    plan = read_json(case_root / "cpu_pilot_plan.json"); check_plan(plan)
    verify_anchor(plan, read_json(no_symlink_path(anchor_path)))
    require(verify_sources(root, manifest_path) == plan["source_manifest_sha256"], "Case source manifest changed")
    require(digest(read_json(case_root / "admission.json")) == plan["admission_sha256"], "Case admission changed")
    validate_npz(case_root / "inputs/current_cohort_arrays.npz")
    return plan


def execution_marker(case_root, plan, *, create=False):
    path = case_root / "execution_started.json"
    if not path.exists():
        require(create, "Outer execution marker must precede a worker")
        now = time.time()
        write_json(path, {"plan_sha256": digest(plan), "started_unix": now,
                         "deadline_unix": now + TOTAL_SECONDS}, immutable=True)
    marker = read_json(path)
    require(marker.get("plan_sha256") == digest(plan)
            and type(marker.get("started_unix")) in (int, float)
            and type(marker.get("deadline_unix")) in (int, float)
            and math.isfinite(marker["started_unix"]) and math.isfinite(marker["deadline_unix"])
            and math.isclose(marker["deadline_unix"] - marker["started_unix"], TOTAL_SECONDS, abs_tol=1e-5),
            "Immutable end-to-end deadline differs")
    return marker


def remaining_seconds(marker, now=None):
    remaining = marker["deadline_unix"] - (time.time() if now is None else now)
    require(remaining > 0, "Immutable end-to-end pilot deadline expired")
    return remaining


def job_directory(case_root, job):
    return no_symlink_path(case_root / "jobs" / job["job_id"])


def completed_job(case_root, plan, job):
    directory = job_directory(case_root, job); record_path = directory / "job.json"
    if not record_path.exists():
        return None
    record = read_json(no_symlink_path(record_path)); checkpoint = no_symlink_path(directory / "supervised/checkpoint.pt")
    require(record.get("plan_sha256") == digest(plan) and record.get("job") == job
            and record.get("status") == "completed_validated_new_fit"
            and type(record.get("optimizer_steps")) is int and record["optimizer_steps"] > 0
            and checkpoint.is_file() and sha(checkpoint) == record.get("checkpoint_sha256"),
            "Completed job identity/hash differs; preserve old bytes")
    return record


def factory(runtime, root, name, seed, data):
    args = {"u_scale": data["scaler"]["u_scale"], "y_scale": data["scaler"]["y_scale"], "seed": seed}
    if name == "additive_positive_head":
        rule = Path(root) / "pilot_sources/experiments/e3_ficha_cnn_lstm_atencion_v2_20261116/verification/quadrature_k128.json"
        model = runtime[3](name, **args, rule_path=rule)
    else:
        model = runtime[4](name, **args)
    require(sum(p.numel() for p in model.parameters()) == PARAMETERS[name], "Declared model capacity differs")
    return model.to("cpu")


def worker(root, manifest_path, case_root, anchor_path, index):
    plan = load_case(root, manifest_path, case_root, anchor_path)
    require(type(index) is int and 0 <= index < 20, "Unknown worker index")
    marker = execution_marker(case_root, plan); remaining_seconds(marker)
    job = plan["jobs"][index]
    if completed_job(case_root, plan, job):
        return {"reused_completed_job": True, "new_fit_this_worker": False}
    runtime = scientific_runtime(root)
    require(runtime[7] == plan["cohort"], "Current CPU cohort differs from frozen plan")
    data, split, admission = admit(runtime, case_root / "inputs/current_cohort_arrays.npz")
    require(digest(admission) == plan["admission_sha256"], "Loaded development admission differs")
    torch, helpers, phase = runtime[1], runtime[5], runtime[6]
    helpers.seed_all(job["seed"]); setup = time.perf_counter()
    model = factory(runtime, root, job["name"], job["seed"], data)
    tensors = helpers._tensors(data, "cpu"); ix = torch.as_tensor(split["select"], device="cpu")
    x, vx = tensors["train_X"], tensors["val_X"][ix]; scale = data["scaler"]["y_scale"]
    def loss(current, indexes):
        if indexes is None:
            return ((current(vx, tensors["val_A"][ix]) - tensors["val_y_original"][ix]) / scale).square().mean()
        if job["name"] == "additive_positive_head":
            return ((current(x[indexes], tensors["train_A"][indexes]) - tensors["train_y_original"][indexes]) / scale).square().mean()
        outputs = current.forward_components(x[indexes], tensors["train_A"][indexes])
        return current.loss_components(outputs, tensors["train_y_original"][indexes], tensors["train_U"][indexes])["loss"]
    identity = {"new_CPU_plan_sha256": digest(plan), "job": job,
                "source_manifest_sha256": plan["source_manifest_sha256"],
                "admission_sha256": plan["admission_sha256"], "cohort": plan["cohort"],
                "model_configuration": model.configuration()}
    setup_seconds = time.perf_counter() - setup; directory = job_directory(case_root, job)
    remaining_seconds(marker)
    state = phase.phase(model, x, tensors["train_y_original"] / scale, vx,
                        tensors["val_y_original"][ix] / scale, directory / "supervised", CONFIG,
                        identity=identity, loss_fn=loss, budget_seconds=PHASE_SECONDS)
    require(state["completed"] is True and state["steps"] > 0 and state["best_score"] is not None,
            "No trained validated checkpoint exists")
    checkpoint = directory / "supervised/checkpoint.pt"
    record = {"status": "completed_validated_new_fit", "job": job, "plan_sha256": digest(plan),
              "checkpoint_sha256": sha(checkpoint), "optimizer_steps": state["steps"],
              "best_epoch": state["best_epoch"], "epochs": state["epoch"],
              "selection_RMSE": math.sqrt(state["best_score"]) * scale,
              "model_configuration": model.configuration(), "phase": state["resource_receipt"],
              "construction_and_tensor_setup_seconds": setup_seconds,
              "assessment_predictions_generated": 0}
    write_json(directory / "job.json", record, immutable=True)
    return record


def continuation(summary):
    require(set(summary) == set(RECIPES + ANALYTICAL), "Every declared recipe/control required for decision")
    for value in summary.values():
        require(type(value) in (float, int) and math.isfinite(value) and value >= 0, "Finite nonnegative RMSE required")
    neural = min((name for name in RECIPES if name != "sigma_crps_v4"), key=lambda name: (summary[name], name))
    control = min((name for name in RECIPES + ANALYTICAL if name != "sigma_crps_v4"), key=lambda name: (summary[name], name))
    def gain(name):
        return None if summary[name] == 0 else (summary[name] - summary["sigma_crps_v4"]) / summary[name]
    neural_gain, all_gain = gain(neural), gain(control)
    return {"best_neural_control": neural, "best_all_control": control,
            "relative_mean_RMSE_gain_vs_best_neural": neural_gain,
            "relative_mean_RMSE_gain_vs_best_all": all_gain,
            "minimum_relative_gain_fixed_before_fits": 0.02,
            "decision": "CONTINUE_DEVELOPMENT_ONLY" if neural_gain is not None and all_gain is not None and min(neural_gain, all_gain) >= 0.02 else "ABANDON_CURRENT_V4_RECIPE",
            "control_automatically_promoted": False, "confirmation_allowed": False,
            "R1_inherited": False, "R3_accredited": False}


def write_csv(path, rows):
    require(rows, "Empty output table refused")
    path = guarded_write_path(path)
    temporary = guarded_write_path(path.with_suffix(path.suffix + ".tmp"))
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def assess(root, manifest_path, case_root, anchor_path):
    plan = load_case(root, manifest_path, case_root, anchor_path)
    records = [completed_job(case_root, plan, job) for job in plan["jobs"]]
    require(all(records), "All20 validated completed jobs required before any assessment forward")
    marker = execution_marker(case_root, plan); remaining_seconds(marker)
    runtime = scientific_runtime(root); require(runtime[7] == plan["cohort"], "Assessment cohort differs")
    data, split, admission = admit(runtime, case_root / "inputs/current_cohort_arrays.npz")
    require(digest(admission) == plan["admission_sha256"], "Assessment input admission differs")
    np, torch, helpers = runtime[0], runtime[1], runtime[5]
    ix = split["assessment"]; x = torch.as_tensor(data["val_X"][ix]); A = torch.as_tensor(data["val_A"][ix])
    target = data["val_y_original"][ix]; dates = data["val_dates"][ix]
    predictions, metrics = [], []
    for job, record in zip(plan["jobs"], records):
        remaining_seconds(marker); helpers.seed_all(job["seed"])
        model = factory(runtime, root, job["name"], job["seed"], data)
        checkpoint = job_directory(case_root, job) / "supervised/checkpoint.pt"
        require(sha(checkpoint) == record["checkpoint_sha256"], "New checkpoint changed before assessment")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        require(state.get("completed") is True and state.get("steps", 0) > 0, "Untrained checkpoint refused")
        model.load_state_dict(state["best"]); model.eval()
        with torch.no_grad(): result = model(x, A).detach().cpu().numpy()
        require(result.shape == target.shape and np.isfinite(result).all(), "Finite aligned assessment outputs required")
        m = helpers.metrics(result, target)
        metrics.append({"name": job["name"], "seed": job["seed"], **m, "parameters": job["parameters"],
                        "optimizer_trainable_parameters": job["parameters"],
                        "optimizer_steps": record["optimizer_steps"], "best_epoch": record["best_epoch"], "epochs": record["epochs"]})
        predictions.extend({"name": job["name"], "seed": job["seed"], "date": str(date),
                            "target": float(y), "prediction": float(p)} for date, y, p in zip(dates, target, result))
    require(len(predictions) == 1100 and len(metrics) == 20, "Neural assessment inventory differs")
    analytic = {"floor_only": np.sqrt(data["val_A"][ix]),
                "floor_plus_mean_excess": np.sqrt(data["val_A"][ix]) + data["scaler"]["d_fit"]}
    for name in ANALYTICAL:
        result = analytic[name]; m = helpers.metrics(result, target)
        metrics.append({"name": name, "seed": None, **m, "parameters": ANALYTICAL_PARAMETERS[name],
                        "optimizer_trainable_parameters": 0,
                        "optimizer_steps": 0, "best_epoch": None, "epochs": 0})
        predictions.extend({"name": name, "seed": None, "date": str(date),
                            "target": float(y), "prediction": float(p)} for date, y, p in zip(dates, target, result))
    require(len(predictions) == 1210 and len(metrics) == 22, "Complete six-recipe assessment inventory differs")
    recipe = []
    for name in RECIPES:
        subset = [m for m in metrics if m["name"] == name]
        recipe.append({"name": name, "seeds": len(subset), "mean_RMSE": float(np.mean([m["RMSE"] for m in subset])),
                       "sample_SD_RMSE": float(np.std([m["RMSE"] for m in subset], ddof=1)),
                       "mean_MAE": float(np.mean([m["MAE"] for m in subset])), "parameters": PARAMETERS[name],
                       "optimizer_trainable_parameters": PARAMETERS[name],
                       "trajectories": 5, "stochastic": True})
    for name in ANALYTICAL:
        m = next(m for m in metrics if m["name"] == name)
        recipe.append({"name": name, "seeds": None, "mean_RMSE": m["RMSE"], "sample_SD_RMSE": None,
                       "mean_MAE": m["MAE"], "parameters": ANALYTICAL_PARAMETERS[name],
                       "optimizer_trainable_parameters": 0, "trajectories": 1, "stochastic": False})
    write_csv(case_root / "assessment_predictions.csv", predictions)
    write_csv(case_root / "per_trajectory_metrics.csv", metrics); write_csv(case_root / "recipe_summary.csv", recipe)
    decision = continuation({m["name"]: m["mean_RMSE"] for m in recipe})
    files = {name: {"bytes": (case_root / name).stat().st_size, "sha256": sha(case_root / name)} for name in
             ("assessment_predictions.csv", "per_trajectory_metrics.csv", "recipe_summary.csv")}
    report = {"status": "completed_new_exposed_CPU_development_pilot", "version": VERSION,
              "case_id": plan["case_id"], "plan_sha256": digest(plan),
              "public_anchor_commit": read_json(anchor_path)["public_anchor_commit"],
              "source_manifest_sha256": plan["source_manifest_sha256"], "cohort": plan["cohort"],
              "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "new_real_data_fits": 20, "new_assessment_predictions": 1210,
              "new_neural_assessment_predictions": 1100, "new_analytical_assessment_predictions": 110,
              "new_assessment_trajectories": 22, "analytical_controls_evaluated_once": True,
              "analytical_d_fit_from_training532_only": float(data["scaler"]["d_fit"]),
              "new_HPO_fits": 0, "new_confirmation_predictions": 0, "decision": decision,
              "original_V3_NO_GO_preserved": True, "R3_accredited": False,
              "old796_old30_old20400_mutated": False,
              "data_scope": "same already exposed historical development, not independent confirmation",
              "fit_resource_records": [{"job": r["job"], "checkpoint_sha256": r["checkpoint_sha256"],
                                        "phase": r["phase"], "construction_and_tensor_setup_seconds": r["construction_and_tensor_setup_seconds"]} for r in records],
              "elapsed_end_to_end_including_restart_downtime_seconds": time.time() - marker["started_unix"],
              "nominal_optimizer_budget_seconds": 2400, "measured_phase_wall_seconds": sum(r["phase"]["elapsed_wall_including_setup_io_and_restart_seconds"] for r in records),
              "energy_or_FLOPs_measured": False, "files": files, "AI_assistance_declared": True}
    write_json(case_root / "pilot_result.json", report, immutable=True)
    return report


def run_process(command, log, timeout):
    log = guarded_write_path(log)
    with log.open("ab") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2",
                                        "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"})
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # No additional grace interval extends the declared opportunity.
            # Frozen phases persist checkpoints by atomic replacement.
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait()
            raise RuntimeError("New CPU development worker exceeded its declared hard process limit") from None
    require(code == 0, "Worker failed; preserve logs/checkpoints before explicit recovery")


def execute(root, manifest_path, case_root, anchor_path):
    plan = load_case(root, manifest_path, case_root, anchor_path)
    final = case_root / "pilot_result.json"
    if final.exists():
        report = read_json(final)
        require(report.get("plan_sha256") == digest(plan)
                and report.get("status") == "completed_new_exposed_CPU_development_pilot"
                and report.get("new_real_data_fits") == 20 and report.get("new_assessment_predictions") == 1210
                and set(report.get("files", {})) == {"assessment_predictions.csv", "per_trajectory_metrics.csv", "recipe_summary.csv"},
                "Completed result source/inventory differs")
        require(all(completed_job(case_root, plan, j) for j in plan["jobs"]), "Completed job missing")
        for name, expected in report["files"].items():
            output = no_symlink_path(case_root / name)
            require(output.stat().st_size == expected["bytes"] and sha(output) == expected["sha256"], "Completed output hash/size differs")
        return {"status": "verified_completed_pilot_reused_without_training_or_assessment", "new_fits_this_invocation": 0,
                "new_predictions_this_invocation": 0, "original_result": report}
    marker = execution_marker(case_root, plan, create=True)
    write_json(case_root / "execution_anchor.json", read_json(anchor_path), immutable=True)
    for job in plan["jobs"]:
        if completed_job(case_root, plan, job):
            continue
        timeout = min(PROCESS_SECONDS, remaining_seconds(marker)); log = job_directory(case_root, job) / "worker.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(job["index"]),
                   "--case-root", str(case_root), "--source-manifest", str(manifest_path),
                   "--public-anchor-receipt", str(anchor_path)]
        run_process(command, log, timeout)
        require(completed_job(case_root, plan, job) is not None, "Worker exited without a validated job receipt")
        print(json.dumps({"event": "new_CPU_development_job_completed", "index": job["index"],
                          "name": job["name"], "seed": job["seed"], "completed_jobs": sum(completed_job(case_root, plan, j) is not None for j in plan["jobs"])}), flush=True)
    # Assessment is a separate hard-bounded child process, too. A stalled model
    # forward cannot silently evade the immutable total execution opportunity.
    command = [sys.executable, str(Path(__file__).resolve()), "--assess-worker",
               "--case-root", str(case_root), "--source-manifest", str(manifest_path),
               "--public-anchor-receipt", str(anchor_path)]
    run_process(command, case_root / "assessment_worker.log", remaining_seconds(marker))
    require(final.is_file(), "Assessment worker exited without final evidence")
    return read_json(final)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true"); action.add_argument("--execute", action="store_true")
    action.add_argument("--worker", type=int)
    action.add_argument("--assess-worker", action="store_true")
    parser.add_argument("--case-root", required=True, type=Path)
    parser.add_argument("--npz", type=Path)
    parser.add_argument("--source-manifest", type=Path, default=Path(__file__).with_name("pilot_source_manifest.json"))
    parser.add_argument("--public-anchor-receipt", type=Path)
    args = parser.parse_args(); root = Path(__file__).resolve().parent
    # The CLI is stdlib-only up to here, so these limits precede NumPy/Torch
    # imports and are identical in preparation, fit and assessment processes.
    os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    try:
        if args.prepare:
            require(args.npz is not None and args.public_anchor_receipt is None, "Preparation requires pinned NPZ only")
            result = prepare(root, args.source_manifest, args.npz, args.case_root)
        else:
            require(args.public_anchor_receipt is not None and args.npz is None, "Execution requires frozen case/anchor; cannot replace input")
            case = no_symlink_path(args.case_root)
            if args.worker is not None:
                result = worker(root, args.source_manifest, case, args.public_anchor_receipt, args.worker)
            elif args.assess_worker:
                result = assess(root, args.source_manifest, case, args.public_anchor_receipt)
            else:
                result = execute(root, args.source_manifest, case, args.public_anchor_receipt)
        print(json.dumps(result, indent=2, allow_nan=False)); return 0
    except Exception as error:
        # Only an already prepared owned case may receive a failure record.
        try:
            case = no_symlink_path(args.case_root)
        except ValueError:
            case = None
        if case is not None and case.is_dir() and case.name.startswith("v4-cpu-dev-") and (case / "cpu_pilot_plan.json").is_file():
            path = case / "failures.json"; entries = read_json(path) if path.exists() else []
            entries.append({"at_utc": datetime.now(timezone.utc).isoformat(), "type": type(error).__name__,
                            "message": str(error), "worker": args.worker, "old_masters_unchanged": True})
            write_json(path, entries)
        print(json.dumps({"status": "failed_preserving_evidence", "error_type": type(error).__name__,
                          "message": str(error), "R3_accredited": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
