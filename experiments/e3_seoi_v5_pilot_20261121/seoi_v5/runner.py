"""Explicit, bounded CPU development pilot; importing this module never trains.

AI assistance declared. The run and assess commands are intentionally separate.
Durable checkpoints are epoch-boundary snapshots. An interrupted epoch is rolled
back and charged its precommitted five-second reservation; an operation already
running may overrun this cooperative allowance. This is not a hard wall timeout.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import random
import shutil
import sys
import tempfile
import time
import uuid

VERSION = "e3-seoi-v5-development-pilot-1"
RECIPES = ("seoi_mse", "input_only_mse", "output_only_mse", "global_mse", "global_crps", "additive_mse")
SEEDS = tuple(range(20261127, 20261132))
ANALYTICAL = ("floor_only", "floor_plus_mean_excess")
TRAINING = {"learning_rate": .003, "weight_decay": 0.0, "batch_size": 32,
            "max_epochs": 40, "patience": 8, "gradient_clip": 1.0}
BUDGET = {"active_seconds_per_fit": 120.0, "epoch_reservation_seconds": 5.0}
CONTINUATION = {"minimum_relative_mean_RMSE_gain": .02,
                "minimum_seed_wins": 4, "maximum_MAE_ratio": 1.05}
NPZ_SHA256 = "186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683"
NPZ_BYTES = 460090
ROOT_EXPORT = ("plan.json", "prepared.json", "cohort.json", "source_identity.json", "admission.json",
               "protocol.json", "PUBLIC_SOURCE_ANCHOR.json", "failures.json", "fit_status.json", "pilot_result.json", "metrics.csv",
               "predictions.csv", "recipe_summary.csv", "resources.csv", "checkpoint_inventory.json",
               "paired_differences.csv", "selection_assessment_gaps.csv")
JOB_EXPORT = ("job.json", "curve.json", "attempts.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def safe(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Symlink path refused")
    if path.exists():
        require(path.is_file() or path.is_dir(), "Nonregular case member refused")
    return path


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(safe(path).read_text(), object_pairs_hook=unique)


def atomic_bytes(path, raw, *, immutable=False):
    path = safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        require(path.read_bytes() == raw, "Immutable record changed: " + path.name)
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".partial-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value, *, immutable=False):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode(), immutable=immutable)


def write_csv(path, rows):
    require(bool(rows), "Empty result table refused")
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    atomic_bytes(path, stream.getvalue().encode())


def verify_sources(root):
    root = safe(root)
    manifest_path = safe(root / "SOURCE_MANIFEST.json")
    manifest = read_json(manifest_path)
    records = manifest.get("files")
    require(isinstance(records, list) and bool(records), "Source manifest files required")
    seen = set()
    for record in records:
        name = record.get("path")
        require(isinstance(name, str) and name and "\\" not in name, "Unsafe source path")
        member = PurePosixPath(name)
        require(not member.is_absolute() and member.as_posix() == name
                and all(p not in ("", ".", "..") for p in name.split("/")), "Unsafe source member")
        require(name not in seen and name != "SOURCE_MANIFEST.json", "Duplicate/self source member")
        seen.add(name); path = safe(root / name)
        require(path.is_file() and path.stat().st_size == record.get("bytes")
                and sha(path) == record.get("sha256"), "Source identity differs: " + name)
    require({"seoi_v5/runner.py", "seoi_v5/model.py", "seoi_v5/data.py", "protocol.json"} <= seen,
            "Executable/protocol source missing from manifest")
    runtime_sources = {p.relative_to(root).as_posix() for p in (root / "seoi_v5").rglob("*.py")}
    require(runtime_sources <= seen, "An executable package source is not covered by the manifest")
    anchor_path = root / "PUBLIC_SOURCE_ANCHOR.json"
    return {"manifest_sha256": sha(manifest_path), "files": records,
            "source_root": str(root), "manifest_metadata": {k: v for k, v in manifest.items() if k != "files"},
            "public_source_anchor": read_json(anchor_path) if anchor_path.exists() else None,
            "public_source_anchor_scope": "bootstrap-reported source metadata; not a per-case preregistration"}


def protocol(root):
    value = read_json(Path(root) / "protocol.json")
    require(tuple(value.get("recipes", ())) == RECIPES, "Protocol recipe family differs")
    require(tuple(value.get("seeds", ())) == SEEDS, "Protocol five-seed design differs")
    for key, expected in (("training", TRAINING), ("budget", BUDGET), ("continuation", CONTINUATION)):
        require(isinstance(value.get(key), dict) and all(value[key].get(k) == v for k, v in expected.items()),
                "Protocol fixed configuration differs: " + key)
    return value


def runtime():
    """CPU software cohort, not a requirement for one particular GPU/CPU model."""
    for key, value in (("OMP_NUM_THREADS", "2"), ("MKL_NUM_THREADS", "2"), ("OPENBLAS_NUM_THREADS", "1")):
        os.environ[key] = value
    import numpy as np
    import torch
    require(sys.version_info.major == 3 and 11 <= sys.version_info.minor <= 13,
            "Python3.11-3.13 required; each prepared case binds its actual version")
    require(torch.__version__ == "2.8.0+cpu" and np.__version__ == "2.3.5",
            "Pinned Torch2.8.0+cpu / NumPy2.3.5 required")
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        require(torch.get_num_interop_threads() == 1, "Interop thread count differs")
    torch.use_deterministic_algorithms(True)
    binding = {"python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
               "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
               "device": "cpu", "torch_threads": 2, "interop_threads": 1,
               "deterministic_algorithms": True,
               "environment_threads": {key: os.environ[key] for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
    observed = {"python": platform.python_version(), "platform": platform.platform(),
                "processor": platform.processor(), "CPU_brand_or_GPU_model_required": False}
    return np, torch, {"binding": binding, "observed": observed}


def array_identity(value):
    import numpy as np
    value = np.ascontiguousarray(value)
    return {"shape": list(value.shape), "dtype": value.dtype.str,
            "sha256": hashlib.sha256(canonical({"shape": list(value.shape), "dtype": value.dtype.str}) + value.tobytes()).hexdigest()}


def data_identity(data):
    result = {}
    for split in ("train", "selection", "assessment"):
        rows = data[split]
        require(set(rows["X"]) == set(RECIPES), "Every recipe input representation required")
        result[split] = {key: array_identity(rows[key]) for key in ("A", "U", "y", "scale", "dates")}
        result[split]["X"] = {name: array_identity(rows["X"][name]) for name in RECIPES}
    require(tuple(result[s]["y"]["shape"] for s in ("train", "selection", "assessment")) == ([532], [54], [55]),
            "Fixed exposed development split geometry differs")
    return result


def jobs():
    return [{"recipe": recipe, "seed": seed, "job_id": "seed" + str(seed) + "/" + recipe}
            for seed in SEEDS for recipe in RECIPES]


def _modules(root):
    root = str(Path(root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from seoi_v5 import data, model
    require(Path(data.__file__).resolve() == Path(root) / "seoi_v5/data.py"
            and Path(model.__file__).resolve() == Path(root) / "seoi_v5/model.py", "Cached module belongs to another source tree")
    return data, model


def prepare(source_root, npz_path, case_root):
    source_root, npz_path, case_root = map(safe, (source_root, npz_path, case_root))
    sources = verify_sources(source_root); spec = protocol(source_root)
    require(npz_path.is_file() and npz_path.stat().st_size == NPZ_BYTES and sha(npz_path) == NPZ_SHA256,
            "Pinned raw archive transport identity differs")
    require(not case_root.exists(), "Fresh case directory required; existing case is preserved for run/status")
    _, _, cohort = runtime(); data_module, _ = _modules(source_root)
    loaded = data_module.load_data(npz_path)
    identities = data_identity(loaded)
    case_root.mkdir(parents=True); (case_root / "inputs").mkdir()
    raw = case_root / "inputs/cohort.npz"
    shutil.copyfile(npz_path, raw)
    require(sha(raw) == NPZ_SHA256 and raw.stat().st_size == NPZ_BYTES, "Private input copy differs")
    plan = {"version": VERSION, "case_id": "seoi-v5-dev-" + uuid.uuid4().hex, "created_at_utc": utc_now(),
            "source_manifest_sha256": sources["manifest_sha256"], "protocol_sha256": sha(source_root / "protocol.json"),
            "npz_sha256": NPZ_SHA256, "npz_bytes": NPZ_BYTES, "data_identities": identities,
            "admission_sha256": digest(loaded["admission"]), "scalers_sha256": digest(loaded["scalers"]),
            "cohort_binding": cohort["binding"], "jobs": jobs(), "recipes": list(RECIPES), "seeds": list(SEEDS),
            "training": TRAINING, "budget": BUDGET, "continuation": CONTINUATION,
            "analytical_controls": list(ANALYTICAL), "assessment_previously_exposed": True,
            "scope": "new_exposed_development_only_not_confirmation", "HPO_fits": 0,
            "planned_neural_fits": 30, "planned_assessment_predictions": 1760,
            "old796_old30_and_V4_mutated": False, "new_training_performed_by_prepare": 0,
            "R1_inherited": False, "R3_accredited": False, "AI_assistance_declared": True,
            "checkpoint_policy": "epoch-boundary; interrupted epoch rollback charged reserved cooperative allowance, never reset",
            "concurrency_policy": "one VM per Drive case; host-local advisory lock is not a distributed lock",
            "epoch_allowance_is_hard_wall_timeout": False,
            "cooperative_allowance_overrun": "a running batch/validation operation may overrun; observed overruns reported",
            "interrupted_epoch_runtime_beyond_last_durable_boundary": "not exactly observed; reserved amount charged; one-operation overrun cannot be excluded",
            "case_plan_publication": "immutable local/Drive record; not falsely represented as an independently published preregistration"}
    write_json(case_root / "source_identity.json", sources, immutable=True)
    if sources["public_source_anchor"] is not None:
        write_json(case_root / "PUBLIC_SOURCE_ANCHOR.json", sources["public_source_anchor"], immutable=True)
    write_json(case_root / "protocol.json", spec, immutable=True)
    write_json(case_root / "cohort.json", cohort, immutable=True)
    write_json(case_root / "admission.json", loaded["admission"], immutable=True)
    write_json(case_root / "plan.json", plan, immutable=True)
    receipt = {"status": "prepared_not_trained", "created_at_utc": utc_now(), "case_id": plan["case_id"],
               "plan_sha256": digest(plan), "plan_file_sha256": sha(case_root / "plan.json"),
               "source_manifest_sha256": sources["manifest_sha256"], "new_real_data_fits": 0,
               "new_model_predictions": 0, "local_plan_only": True, "R3_accredited": False}
    write_json(case_root / "prepared.json", receipt, immutable=True)
    write_json(case_root / "failures.json", [])
    return receipt


def load_case(source_root, case_root, *, load_arrays=True):
    source_root, case_root = map(safe, (source_root, case_root))
    sources = verify_sources(source_root); spec = protocol(source_root)
    plan = read_json(case_root / "plan.json"); prepared = read_json(case_root / "prepared.json")
    require(plan.get("version") == VERSION and plan.get("jobs") == jobs(), "Case model/job identity differs")
    require(prepared.get("plan_sha256") == digest(plan)
            and prepared.get("plan_file_sha256") == sha(case_root / "plan.json"), "Prepared plan changed")
    require(plan["source_manifest_sha256"] == sources["manifest_sha256"]
            and plan["protocol_sha256"] == sha(source_root / "protocol.json")
            and read_json(case_root / "protocol.json") == spec, "Case source/protocol differs")
    raw = safe(case_root / "inputs/cohort.npz")
    require(raw.is_file() and raw.stat().st_size == NPZ_BYTES and sha(raw) == plan["npz_sha256"] == NPZ_SHA256,
            "Case raw transport differs")
    if not load_arrays:
        return plan, None, None
    _, _, cohort = runtime()
    require(cohort["binding"] == plan["cohort_binding"], "Current software/thread cohort differs from prepared case")
    data_module, model_module = _modules(source_root)
    data = data_module.load_data(raw)
    require(data_identity(data) == plan["data_identities"]
            and digest(data["admission"]) == plan["admission_sha256"]
            and digest(data["scalers"]) == plan["scalers_sha256"], "Admitted arrays/scalers differ")
    return plan, data, model_module


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed % (2**32)); torch.manual_seed(seed)


def capture_rng():
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}


def restore_rng(value):
    import numpy as np
    import torch
    random.setstate(value["python"]); np.random.set_state(value["numpy"]); torch.set_rng_state(value["torch"])


def clone_weights(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def save_snapshot(directory, state):
    """Commit a new immutable payload, then atomically move a small digest pointer."""
    import torch
    directory = safe(directory); directory.mkdir(parents=True, exist_ok=True)
    filename = "snapshot_" + uuid.uuid4().hex + ".pt"
    path = safe(directory / filename)
    with tempfile.NamedTemporaryFile(dir=directory, prefix=".snapshot-", delete=False) as stream:
        temporary = Path(stream.name)
        torch.save(state, stream); stream.flush(); os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    pointer = {"file": filename, "bytes": path.stat().st_size, "sha256": sha(path),
               "request_sha256": state["request_sha256"]}
    write_json(directory / "checkpoint.json", pointer)
    return pointer


def load_snapshot(directory, request_sha256):
    """Only owned newly produced snapshots with a checked pointer are admitted."""
    import torch
    directory = safe(directory); pointer = read_json(directory / "checkpoint.json")
    name = pointer.get("file", "")
    require(isinstance(name, str) and name.startswith("snapshot_") and name.endswith(".pt")
            and PurePosixPath(name).name == name, "Unsafe checkpoint pointer")
    path = safe(directory / name)
    require(path.is_file() and path.stat().st_size == pointer.get("bytes")
            and sha(path) == pointer.get("sha256")
            and pointer.get("request_sha256") == request_sha256, "Snapshot identity/hash differs")
    state = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(state, dict) and state.get("request_sha256") == request_sha256, "Snapshot request differs")
    return state, pointer


def settle_interrupted_epoch(directory, state):
    """Debit an unclosed reservation once; never silently replay lost work free."""
    path = safe(Path(directory) / "epoch_pending.json")
    if not path.exists():
        return state, False
    marker = read_json(path)
    require(marker.get("request_sha256") == state["request_sha256"], "Interrupted epoch request differs")
    segment = marker["segment_id"]
    if segment not in state["settled_segments"]:
        require(marker["epoch"] == state["epoch"] + 1, "Interrupted epoch counter differs")
        charge = marker["reserved_seconds"]
        require(type(charge) in (int, float) and 0 < charge <= BUDGET["epoch_reservation_seconds"],
                "Interrupted epoch reservation differs")
        attempts_path = Path(directory) / "attempts.json"
        attempts = read_json(attempts_path) if attempts_path.exists() else []
        known = [a["aborted_active_seconds_observed"] for a in attempts
                 if a.get("segment_id") == segment and "aborted_active_seconds_observed" in a]
        if known:
            require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in known),
                    "Invalid observed aborted epoch cost")
            charge = max(charge, max(known))
        state["active_seconds_charged"] += charge
        state["interrupted_epochs"] += 1
        state["settled_segments"].append(segment)
        state["lost_partial_epoch_optimizer_steps"] = None
        state["lost_partial_epoch_elapsed_is_exact"] = False
        save_snapshot(directory, state)
    path.unlink()
    return state, True


def append_attempt(directory, value):
    path = Path(directory) / "attempts.json"
    rows = read_json(path) if path.exists() else []
    rows.append(value); write_json(path, rows)


def record_failure(case_root, job, error):
    path = Path(case_root) / "failures.json"
    rows = read_json(path) if path.exists() else []
    rows.append({"at_utc": utc_now(), "job": job, "error_type": type(error).__name__,
                 "message": str(error), "scope": "failed/interrupted attempt, not a new design or successful fit"})
    write_json(path, rows)


def job_request(plan, job):
    return {"version": VERSION, "plan_sha256": digest(plan), "job": job,
            "training": TRAINING, "budget": BUDGET, "cohort": plan["cohort_binding"]}


def complete_job(case_root, plan, job):
    directory = safe(Path(case_root) / "jobs" / job["job_id"])
    path = directory / "job.json"
    if not path.exists():
        return None
    result = read_json(path)
    request_hash = digest(job_request(plan, job))
    require(result.get("status") == "completed_validated_fit" and result.get("request_sha256") == request_hash,
            "Completed job record differs")
    state, pointer = load_snapshot(directory, request_hash)
    require(state.get("completed") is True and state.get("steps_committed", 0) > 0
            and state.get("best") is not None and state.get("best_score") is not None
            and result.get("checkpoint_sha256") == pointer["sha256"], "No matching validated trained checkpoint")
    require(state.get("stop_reason") in ("early_stopping", "max_epochs")
            and type(state.get("best_score")) in (int, float) and math.isfinite(state["best_score"])
            and state["best_score"] >= 0 and 1 <= state.get("epoch", 0) <= TRAINING["max_epochs"]
            and 0 <= state.get("active_seconds_charged", -1) <= BUDGET["active_seconds_per_fit"],
            "Completed checkpoint stopping/budget/score is invalid")
    require((state["stop_reason"] == "max_epochs" and state["epoch"] == TRAINING["max_epochs"])
            or (state["stop_reason"] == "early_stopping" and state.get("stale", 0) >= TRAINING["patience"]),
            "Completed checkpoint did not reach its prescribed stopping rule")
    return result


def _selection_score(model, split, recipe, y_scale):
    import torch
    model.eval()
    with torch.no_grad():
        prediction = model.forward_components(torch.as_tensor(split["X"][recipe]),
                                              torch.as_tensor(split["A"]), torch.as_tensor(split["scale"]))["prediction"]
        target = torch.as_tensor(split["y"], dtype=torch.float64)
        require(prediction.shape == target.shape and bool(torch.isfinite(prediction).all()), "Invalid selection prediction")
        return float(((prediction - target) / y_scale).square().mean())


def fit_job(case_root, plan, data, model_module, job):
    """Training implementation. Called only by the explicit run command."""
    import numpy as np
    import torch
    directory = safe(Path(case_root) / "jobs" / job["job_id"]); directory.mkdir(parents=True, exist_ok=True)
    request = job_request(plan, job); request_hash = digest(request)
    write_json(directory / "request.json", request, immutable=True)
    seed_all(job["seed"])
    scales = data["scalers"][job["recipe"]]
    model = model_module.build_model(job["recipe"], job["seed"], u_scale=scales["u_scale"], y_scale=scales["y_scale"]).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING["learning_rate"], weight_decay=TRAINING["weight_decay"])
    if (directory / "checkpoint.json").exists():
        state, _ = load_snapshot(directory, request_hash)
        state, recovered = settle_interrupted_epoch(directory, state)
        # The snapshot pointer can become durable before its companion CSV/JSON
        # receipt. Reconstruct the curve from the authoritative owned snapshot.
        write_json(directory / "curve.json", state["curve"])
        if state.get("completed") is True:
            require(state.get("best") is not None and state.get("steps_committed", 0) > 0,
                    "Completed state has no trained validated checkpoint")
            model.load_state_dict(state["current"])
            _, pointer = load_snapshot(directory, request_hash)
            record = _job_record(plan, job, state, pointer, model, scales)
            write_json(directory / "job.json", record, immutable=True)
            return record
    else:
        state = {"request_sha256": request_hash, "current": clone_weights(model), "optimizer": optimizer.state_dict(),
                 "rng": capture_rng(), "epoch": 0, "steps_committed": 0, "best": None, "best_score": None,
                 "best_epoch": None, "stale": 0, "active_seconds_charged": 0.0,
                 "checkpoint_io_seconds": 0.0, "interrupted_epochs": 0, "settled_segments": [],
                 "sampler_order": None, "next_batch": 0, "completed": False, "curve": [],
                 "lost_partial_epoch_optimizer_steps": 0, "lost_partial_epoch_elapsed_is_exact": True}
        save_snapshot(directory, state); recovered = False
    model.load_state_dict(state["current"]); optimizer.load_state_dict(state["optimizer"]); restore_rng(state["rng"])
    append_attempt(directory, {"at_utc": utc_now(), "starting_epoch": state["epoch"],
                              "active_seconds_previously_charged": state["active_seconds_charged"],
                              "interrupted_epoch_recovered": recovered})
    train, selection = data["train"], data["selection"]
    x = torch.as_tensor(train["X"][job["recipe"]]); A = torch.as_tensor(train["A"])
    y = torch.as_tensor(train["y"]); U = torch.as_tensor(train["U"]); scale = torch.as_tensor(train["scale"])
    while state["epoch"] < TRAINING["max_epochs"] and state["stale"] < TRAINING["patience"]:
        remaining = BUDGET["active_seconds_per_fit"] - state["active_seconds_charged"]
        require(remaining > 0, "Active fit budget exhausted before prescribed stopping; partial evidence retained")
        allowance = min(BUDGET["epoch_reservation_seconds"], remaining)
        marker = {"segment_id": uuid.uuid4().hex, "request_sha256": request_hash,
                  "epoch": state["epoch"] + 1, "reserved_seconds": allowance, "started_at_utc": utc_now()}
        write_json(directory / "epoch_pending.json", marker, immutable=True)
        started = time.perf_counter(); model.train(); order = torch.randperm(len(x)); losses = []; epoch_steps = 0
        try:
            for first in range(0, len(x), TRAINING["batch_size"]):
                require(time.perf_counter() - started < allowance,
                        "Cooperative epoch allowance exhausted before validation; partial evidence retained")
                indexes = order[first:first + TRAINING["batch_size"]]
                optimizer.zero_grad(set_to_none=True)
                outputs = model.forward_components(x[indexes], A[indexes], scale[indexes])
                loss = model.loss_components(outputs, y[indexes], U[indexes])["loss"]
                require(loss.ndim == 0 and bool(torch.isfinite(loss)), "Nonfinite/non-scalar training loss")
                loss.backward()
                require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()), "Nonfinite training gradient")
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING["gradient_clip"], error_if_nonfinite=True)
                optimizer.step(); epoch_steps += 1; losses.append(float(loss.detach()))
                require(time.perf_counter() - started <= allowance,
                        "Cooperative epoch allowance overrun at batch boundary; partial evidence retained")
            score = _selection_score(model, selection, job["recipe"], scales["y_scale"])
            elapsed = time.perf_counter() - started
            require(math.isfinite(score) and elapsed <= allowance,
                    "Selection/epoch operation exceeded cooperative allowance; partial evidence retained")
        except BaseException:
            elapsed = time.perf_counter() - started
            # Keep the marker. The next invocation restores the previous durable
            # state and debits its reservation exactly once. Observed aborted
            # operation cost/steps stay in a separate attempt record.
            append_attempt(directory, {"at_utc": utc_now(), "epoch": marker["epoch"], "segment_id": marker["segment_id"],
                                      "aborted_active_seconds_observed": elapsed,
                                      "optimizer_steps_in_discarded_epoch_observed": epoch_steps,
                                      "reserved_seconds": allowance, "cooperative_overrun_seconds": max(0.0, elapsed - allowance),
                                      "epoch_committed": False})
            raise
        state["epoch"] += 1; state["steps_committed"] += epoch_steps
        state["active_seconds_charged"] += elapsed
        if state["best_score"] is None or score < state["best_score"]:
            state["best_score"] = score; state["best_epoch"] = state["epoch"]
            state["best"] = clone_weights(model); state["stale"] = 0
        else:
            state["stale"] += 1
        state["curve"].append({"epoch": state["epoch"], "train_loss_mean_batches": float(np.mean(losses)),
                               "selection_mse_scaled": score, "active_seconds": elapsed,
                               "optimizer_steps_committed": state["steps_committed"]})
        state["current"] = clone_weights(model); state["optimizer"] = optimizer.state_dict(); state["rng"] = capture_rng()
        state["sampler_order"] = order; state["next_batch"] = len(order)
        state["settled_segments"].append(marker["segment_id"])
        io_started = time.perf_counter(); save_snapshot(directory, state)
        state["checkpoint_io_seconds"] += time.perf_counter() - io_started
        (directory / "epoch_pending.json").unlink()
        write_json(directory / "curve.json", state["curve"])
    require(state["best"] is not None and state["steps_committed"] > 0, "No validated trained checkpoint exists")
    require(state["active_seconds_charged"] <= BUDGET["active_seconds_per_fit"], "Active fit budget overrun; fit not accepted")
    write_json(directory / "curve.json", state["curve"])
    state["completed"] = True
    state["stop_reason"] = "early_stopping" if state["stale"] >= TRAINING["patience"] else "max_epochs"
    pointer = save_snapshot(directory, state)
    record = _job_record(plan, job, state, pointer, model, scales)
    write_json(directory / "job.json", record, immutable=True)
    return record


def _job_record(plan, job, state, pointer, model, scales):
    return {"status": "completed_validated_fit", "job": job, "request_sha256": state["request_sha256"],
            "plan_sha256": digest(plan), "checkpoint_sha256": pointer["sha256"],
            "checkpoint_file": pointer["file"], "epochs": state["epoch"], "best_epoch": state["best_epoch"],
            "selection_RMSE": math.sqrt(state["best_score"]) * scales["y_scale"],
            "optimizer_steps_committed": state["steps_committed"],
            "actual_optimizer_steps_may_include_uncommitted_crash_work": state["interrupted_epochs"] > 0,
            "active_seconds_charged": state["active_seconds_charged"],
            "checkpoint_io_seconds": state["checkpoint_io_seconds"],
            "checkpoint_IO_record_is_partial_if_an_unclosed_write_was_interrupted": True,
            "interrupted_epochs": state["interrupted_epochs"], "stop_reason": state["stop_reason"],
            "parameters": sum(p.numel() for p in model.parameters()), "device": "cpu",
            "cooperative_allowance_is_hard_wall_timeout": False, "assessment_predictions_generated": 0}


@contextmanager
def case_lock(case_root):
    """Host-local lock only. Never run two VMs against the same Drive case.

    A local lock avoids unsupported DriveFUSE flock calls. The kernel releases
    it when a process dies; it does not coordinate different Colab VMs.
    """
    import fcntl
    folder = safe(Path(tempfile.gettempdir()) / "e3-seoi-v5-host-locks")
    folder.mkdir(mode=0o700, exist_ok=True)
    name = hashlib.sha256(str(safe(case_root).resolve()).encode()).hexdigest() + ".lock"
    path = safe(folder / name)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another process owns this case; no concurrent run is permitted") from None
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run(source_root, case_root):
    plan, data, model_module = load_case(source_root, case_root)
    for job in plan["jobs"]:
        if complete_job(case_root, plan, job) is not None:
            continue
        try:
            fit_job(case_root, plan, data, model_module, job)
        except Exception as error:
            record_failure(case_root, job, error)
            print(json.dumps({"event": "fit_attempt_failed", "job": job, "error": str(error)}), flush=True)
        else:
            print(json.dumps({"event": "fit_completed", "job": job}), flush=True)
    completed = [complete_job(case_root, plan, job) for job in plan["jobs"]]
    result = {"status": "all_fits_completed" if all(completed) else "incomplete_fits_evidence_retained",
              "planned_neural_fits": 30, "completed_neural_fits": sum(r is not None for r in completed),
              "assessment_started_by_run": False, "R3_accredited": False}
    write_json(Path(case_root) / "fit_status.json", result)
    return result


def metrics(prediction, target):
    import numpy as np
    prediction, target = np.asarray(prediction, dtype=float), np.asarray(target, dtype=float)
    require(prediction.shape == target.shape and prediction.ndim == 1 and len(target) > 0
            and np.isfinite(prediction).all() and np.isfinite(target).all(), "Aligned finite metric arrays required")
    return {"RMSE": float(np.sqrt(np.mean((prediction-target)**2))), "MAE": float(np.mean(np.abs(prediction-target)))}


def continuation(summary, metrics_by_seed, *, complete_fits):
    require(set(summary) == set(RECIPES + ANALYTICAL), "All neural and analytical controls required")
    for row in summary.values():
        require(all(type(row.get(k)) in (float, int) and math.isfinite(row[k]) and row[k] >= 0 for k in ("mean_RMSE", "mean_MAE")),
                "Finite nonnegative aggregate metrics required")
    candidate = summary["seoi_mse"]
    neural = min((r for r in RECIPES if r != "seoi_mse"), key=lambda r: (summary[r]["mean_RMSE"], (RECIPES + ANALYTICAL).index(r)))
    all_control = min((r for r in RECIPES + ANALYTICAL if r != "seoi_mse"), key=lambda r: (summary[r]["mean_RMSE"], (RECIPES + ANALYTICAL).index(r)))
    baseline = summary[all_control]["mean_RMSE"]
    gain = None if baseline == 0 else (baseline-candidate["mean_RMSE"])/baseline
    require(set(metrics_by_seed) == set(RECIPES), "Every neural recipe requires the same five seeds")
    for value in metrics_by_seed.values():
        require(set(value) == set(SEEDS), "No missing/additional seed admitted")
        require(all(type(row.get("RMSE")) in (int, float) and math.isfinite(row["RMSE"]) and row["RMSE"] >= 0
                    for row in value.values()), "Finite per-seed RMSE required")
    wins = sum(metrics_by_seed["seoi_mse"][s]["RMSE"] < metrics_by_seed[neural][s]["RMSE"] for s in SEEDS)
    mae_limit = CONTINUATION["maximum_MAE_ratio"] * summary[neural]["mean_MAE"]
    passed = bool(complete_fits and gain is not None and gain >= CONTINUATION["minimum_relative_mean_RMSE_gain"]
                  and wins >= CONTINUATION["minimum_seed_wins"] and candidate["mean_MAE"] <= mae_limit)
    return {"decision": "CONTINUE_DEVELOPMENT_ONLY" if passed else "ABANDON_CURRENT_SEOI_RECIPE",
            "best_all_control_by_mean_RMSE": all_control, "best_neural_control_by_mean_RMSE": neural,
            "relative_mean_RMSE_gain_vs_best_all": gain, "seed_wins_vs_best_neural": wins,
            "seed_pairs": 5, "candidate_mean_MAE": candidate["mean_MAE"], "maximum_allowed_MAE": mae_limit,
            "thresholds_fixed_before_fits": CONTINUATION, "complete_30_fits": bool(complete_fits),
            "control_automatically_promoted": False, "confirmation_allowed": False,
            "statistical_significance_claimed": False, "R3_accredited": False}


def _cached_assessment(case_root, plan, job, record, data, model_module):
    import numpy as np
    import torch
    split = data["assessment"]; recipe, seed = job["recipe"], job["seed"]
    directory = safe(Path(case_root) / "assessment_cache"); directory.mkdir(exist_ok=True)
    path = directory / (recipe + "_" + str(seed) + ".json")
    binding = digest({"plan": digest(plan), "job": job, "checkpoint_sha256": record["checkpoint_sha256"],
                      "assessment_identity": plan["data_identities"]["assessment"]})
    if path.exists():
        cached = read_json(path)
        require(cached.get("binding") == binding and len(cached.get("prediction", [])) == len(split["y"]), "Assessment cache differs")
        result = np.asarray(cached["prediction"], dtype=float)
        metrics(result, split["y"])
        return result
    scales = data["scalers"][recipe]
    model = model_module.build_model(recipe, seed, u_scale=scales["u_scale"], y_scale=scales["y_scale"]).cpu()
    state, _ = load_snapshot(Path(case_root) / "jobs" / job["job_id"], record["request_sha256"])
    model.load_state_dict(state["best"]); model.eval()
    with torch.no_grad():
        result = model.forward_components(torch.as_tensor(split["X"][recipe]), torch.as_tensor(split["A"]),
                                          torch.as_tensor(split["scale"]))["prediction"].detach().cpu().numpy()
    metrics(result, split["y"])
    require((result + 1e-12 >= np.sqrt(split["A"])).all(), "Assessment violates physical floor")
    write_json(path, {"binding": binding, "prediction": result.tolist(), "created_at_utc": utc_now()}, immutable=True)
    return result


def assess(source_root, case_root):
    plan, _, _ = load_case(source_root, case_root, load_arrays=False)
    records = [complete_job(case_root, plan, job) for job in plan["jobs"]]
    require(all(records), "All30 validated completed fits required before assessment")
    final_path = Path(case_root) / "pilot_result.json"
    if final_path.exists():
        result = read_json(final_path)
        require(result.get("plan_sha256") == digest(plan), "Existing final assessment plan differs")
        for name, identity in result["files"].items():
            path = safe(Path(case_root) / name)
            require(path.stat().st_size == identity["bytes"] and sha(path) == identity["sha256"], "Existing final output differs")
        return result
    plan, data, model_module = load_case(source_root, case_root)
    assessment = data["assessment"]
    write_json(Path(case_root) / "assessment_started.json", {"plan_sha256": digest(plan), "all30_fits_verified": True}, immutable=True)
    prediction_rows, metric_rows, resource_rows, inventory = [], [], [], []
    for job, record in zip(plan["jobs"], records):
        prediction = _cached_assessment(case_root, plan, job, record, data, model_module)
        metric_rows.append({"recipe": job["recipe"], "seed": job["seed"], **metrics(prediction, assessment["y"])})
        prediction_rows.extend({"recipe": job["recipe"], "seed": job["seed"], "date": str(day),
                                "target": float(y), "prediction": float(p)} for day, y, p in zip(assessment["dates"], assessment["y"], prediction))
        resource_rows.append({key: record[key] for key in ("job", "parameters", "epochs", "best_epoch", "optimizer_steps_committed",
                                                          "active_seconds_charged", "checkpoint_io_seconds", "interrupted_epochs", "stop_reason")})
        inventory.append({"job": job, "checkpoint_sha256": record["checkpoint_sha256"], "checkpoint_payload_exported": False})
    analytical = {"floor_only": __import__("numpy").sqrt(assessment["A"]),
                  "floor_plus_mean_excess": __import__("numpy").sqrt(assessment["A"]) + data["d_fit"]}
    for recipe, prediction in analytical.items():
        metric_rows.append({"recipe": recipe, "seed": None, **metrics(prediction, assessment["y"])})
        prediction_rows.extend({"recipe": recipe, "seed": None, "date": str(day), "target": float(y), "prediction": float(p)}
                               for day, y, p in zip(assessment["dates"], assessment["y"], prediction))
    import numpy as np
    summary, by_seed = {}, {name: {} for name in RECIPES}
    for recipe in RECIPES + ANALYTICAL:
        rows = [r for r in metric_rows if r["recipe"] == recipe]
        summary[recipe] = {"recipe": recipe, "mean_RMSE": float(np.mean([r["RMSE"] for r in rows])),
                           "sample_SD_RMSE": float(np.std([r["RMSE"] for r in rows], ddof=1)) if len(rows) > 1 else None,
                           "mean_MAE": float(np.mean([r["MAE"] for r in rows])), "trajectories": len(rows)}
        if recipe in by_seed:
            by_seed[recipe] = {r["seed"]: r for r in rows}
    require(len(metric_rows) == 32 and len(prediction_rows) == 1760, "Assessment inventory differs")
    paired = []
    for control in tuple(r for r in RECIPES if r != "seoi_mse") + ANALYTICAL:
        for seed in SEEDS:
            candidate = by_seed["seoi_mse"][seed]
            reference = (by_seed[control][seed] if control in by_seed
                         else next(r for r in metric_rows if r["recipe"] == control))
            paired.append({"candidate": "seoi_mse", "control": control, "seed": seed,
                           "candidate_RMSE": candidate["RMSE"], "control_RMSE": reference["RMSE"],
                           "RMSE_difference_candidate_minus_control": candidate["RMSE"] - reference["RMSE"],
                           "relative_RMSE_gain": ((reference["RMSE"] - candidate["RMSE"]) / reference["RMSE"]
                                                  if reference["RMSE"] > 0 else None),
                           "candidate_MAE": candidate["MAE"], "control_MAE": reference["MAE"],
                           "control_repeated_fit": control not in ANALYTICAL})
    gaps = []
    for job, record in zip(plan["jobs"], records):
        rmse = by_seed[job["recipe"]][job["seed"]]["RMSE"]
        selected = record["selection_RMSE"]
        gaps.append({"recipe": job["recipe"], "seed": job["seed"], "best_epoch": record["best_epoch"],
                     "selection_RMSE": selected, "assessment_RMSE": rmse,
                     "assessment_minus_selection_RMSE": rmse - selected,
                     "relative_gap": (rmse - selected) / selected if selected > 0 else None,
                     "assessment_previously_exposed": True})
    write_csv(Path(case_root) / "metrics.csv", metric_rows)
    write_csv(Path(case_root) / "predictions.csv", prediction_rows)
    write_csv(Path(case_root) / "recipe_summary.csv", list(summary.values()))
    for row in resource_rows:
        row["job"] = row["job"]["job_id"]
    write_csv(Path(case_root) / "resources.csv", resource_rows)
    write_json(Path(case_root) / "checkpoint_inventory.json", inventory)
    write_csv(Path(case_root) / "paired_differences.csv", paired)
    write_csv(Path(case_root) / "selection_assessment_gaps.csv", gaps)
    files = {name: {"bytes": (Path(case_root)/name).stat().st_size, "sha256": sha(Path(case_root)/name)}
             for name in ("metrics.csv", "predictions.csv", "recipe_summary.csv", "resources.csv", "checkpoint_inventory.json",
                          "paired_differences.csv", "selection_assessment_gaps.csv")}
    result = {"status": "completed_exposed_development_pilot", "completed_at_utc": utc_now(), "plan_sha256": digest(plan),
              "completed_neural_fits": 30, "assessment_predictions_published": 1760,
              "neural_assessment_predictions_published": 1650, "analytical_predictions_published": 110,
              "actual_forward_count_if_an_unclosed_assessment_crash_occurred": "not inferred from unique cached/published rows",
              "decision": continuation(summary, by_seed, complete_fits=True), "files": files,
              "old796_old30_and_V4_mutated": False, "R3_accredited": False,
              "scope": "exposed development; no inferential significance or independent confirmation"}
    write_json(final_path, result, immutable=True)
    return result


def export_case(source_root, case_root, output):
    plan, _, _ = load_case(source_root, case_root, load_arrays=False)
    case_root, output = map(safe, (case_root, output))
    require(not output.exists() and not output.is_relative_to(case_root), "Fresh public destination outside private case required")
    selected = []
    for name in ROOT_EXPORT:
        path = case_root / name
        if path.exists():
            selected.append((name, safe(path)))
    for job in plan["jobs"]:
        for name in JOB_EXPORT:
            path = case_root / "jobs" / job["job_id"] / name
            if path.exists():
                selected.append((path.relative_to(case_root).as_posix(), safe(path)))
    require(all(p.is_file() and p.stat().st_size <= 2_000_000 for _, p in selected)
            and sum(p.stat().st_size for _, p in selected) <= 20_000_000, "Public metadata export exceeds byte bound")
    require(not any(n.endswith((".npz", ".pt", ".npy")) or n.startswith("inputs/") for n, _ in selected), "Raw/weight export refused")
    output.mkdir(parents=True)
    inventory = []
    for name, path in selected:
        target = output / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        require(sha(target) == sha(path), "Export copy differs")
        inventory.append({"path": name, "bytes": target.stat().st_size, "sha256": sha(target)})
    receipt = {"version": VERSION, "case_id": plan["case_id"], "plan_sha256": digest(plan), "files": inventory,
               "exported_at_utc": utc_now(), "raw_arrays_exported": False, "weights_optimizer_or_RNG_exported": False,
               "new_training_or_inference_by_exporter": 0, "R3_accredited": False}
    write_json(output / "EXPORT_MANIFEST.json", receipt)
    return receipt


def status(source_root, case_root):
    plan, _, _ = load_case(source_root, case_root, load_arrays=False)
    records = [complete_job(case_root, plan, job) for job in plan["jobs"]]
    failures = read_json(Path(case_root) / "failures.json")
    return {"case_id": plan["case_id"], "scope": plan["scope"], "planned_neural_fits": 30,
            "completed_neural_fits": sum(r is not None for r in records), "failed_attempt_records": len(failures),
            "assessment_complete": (Path(case_root) / "pilot_result.json").is_file(), "R3_accredited": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "assess", "export", "status"):
        command = sub.add_parser(name); command.add_argument("--case-root", type=Path, required=True)
        if name == "prepare":
            command.add_argument("--npz", type=Path, required=True)
        if name == "export":
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.source_root, args.npz, args.case_root)
    elif args.command == "run":
        with case_lock(args.case_root):
            result = run(args.source_root, args.case_root)
    elif args.command == "assess":
        with case_lock(args.case_root):
            result = assess(args.source_root, args.case_root)
    elif args.command == "export":
        result = export_case(args.source_root, args.case_root, args.output)
    else:
        result = status(args.source_root, args.case_root)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 2 if result.get("status") == "incomplete_fits_evidence_retained" else 0


if __name__ == "__main__":
    raise SystemExit(main())
