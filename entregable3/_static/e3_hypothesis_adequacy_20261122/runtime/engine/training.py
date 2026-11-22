"""Owned, epoch-boundary training snapshots for the confirmatory engine.

Only supplied fit/selection arrays are inspected. This module never opens a
panel or reserved partition. Budget limits are cooperative active-time limits,
not claims of hard wall-clock interruption.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import platform
import sys
import tempfile
import time
import uuid

import numpy as np
import torch

from . import models

VERSION = "e3-confirmatory-training-2"
DEFAULT_TRAINING = {"learning_rate": .003, "weight_decay": 0., "batch_size": 32,
                    "max_epochs": 40, "patience": 8, "gradient_clip": 1.,
                    "warmup_fraction": 0., "scheduler": "constant"}
DEFAULT_BUDGET = {"total_budget_seconds": 120., "epoch_reservation_seconds": 5.}


class TrainingIncomplete(RuntimeError):
    """No fully validated fit reached its prescribed stopping rule."""


class DurabilityError(RuntimeError):
    """The checkpoint barrier failed; no further optimizer step is permitted."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _sha(path):
    result = hashlib.sha256()
    with _safe(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _safe(path):
    path = Path(os.path.abspath(path))
    _require(not any(p.is_symlink() for p in (path, *path.parents)),
             "Job paths cannot traverse symlinks")
    return path


def _sync(directory):
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _read(path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            _require(key not in out, "Duplicate JSON key")
            out[key] = value
        return out
    return json.loads(_safe(path).read_text(), object_pairs_hook=unique)


def _write(path, value, *, immutable=False):
    path = _safe(path)
    raw = (_canonical(value) + "\n").encode()
    if immutable and path.exists():
        _require(path.read_bytes() == raw, "Immutable job identity differs; old evidence retained")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".write-", delete=False) as stream:
        temp = Path(stream.name)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temp, path)
        _sync(path.parent)
    finally:
        temp.unlink(missing_ok=True)


def _append(directory, filename, value):
    path = directory / filename
    rows = _read(path) if path.exists() else []
    rows.append(value)
    _write(path, rows)


@contextmanager
def _lock(directory):
    import fcntl
    path = _safe(directory / "job.lock")
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another process owns this job") from None
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _array_identity(array):
    array = np.asarray(array)
    if array.dtype.hasobject or array.dtype.kind in "USM":
        payload = _canonical(array.astype(str).tolist()).encode()
    else:
        payload = np.ascontiguousarray(array).tobytes()
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(payload).hexdigest()}


def data_identity(partition):
    """Hash every supplied array, including causal support/history metadata."""
    return {key: _array_identity(value) for key, value in sorted(partition.items())
            if isinstance(value, (np.ndarray, list, tuple))}


def _partition(partition, *, target=True):
    _require(isinstance(partition, dict) and "raw" in partition, "Partition requires raw arrays")
    raw = np.asarray(partition["raw"])
    _require(raw.ndim == 3 and raw.shape[1:] == (7, 5) and len(raw) > 0,
             "Expected nonempty raw[N,7,5]")
    _require(np.isfinite(raw).all(), "Nonfinite raw input")
    if target:
        y = np.asarray(partition.get("y"))
        _require(y.shape == (len(raw),) and np.isfinite(y).all(), "Expected aligned finite y[N]")
    for key in ("dates", "datesISO", "support_start", "support_end", "known_sigma_history",
                "known_floor_A", "forecast_offsets", "history_end_positions"):
        if key in partition:
            _require(len(partition[key]) == len(raw), "Unaligned partition metadata: " + key)
    return raw


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def _rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state()}


def _restore_rng(value):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])


def _weights(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _weights_hash(weights):
    result = hashlib.sha256()
    for name, value in sorted(weights.items()):
        result.update(name.encode())
        result.update(_canonical(_array_identity(value.detach().cpu().numpy())).encode())
    return result.hexdigest()


def _snapshot(directory, state):
    """Immutable payload first, atomic checked pointer second; retain old payloads."""
    started = time.perf_counter()
    filename = "snapshot_" + uuid.uuid4().hex + ".pt"
    path = _safe(directory / filename)
    with tempfile.NamedTemporaryFile(dir=directory, prefix=".snapshot-", delete=False) as stream:
        temp = Path(stream.name)
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temp, path)
        _sync(directory)
    finally:
        temp.unlink(missing_ok=True)
    pointer = {"file": filename, "bytes": path.stat().st_size, "sha256": _sha(path),
               "request_sha256": state["request_sha256"]}
    _write(directory / "checkpoint.json", pointer)
    # The duration of the last write cannot be included in its own payload.
    state["checkpoint_io_seconds_observed"] += time.perf_counter() - started
    return pointer


def _load(directory, request_sha256):
    pointer = _read(directory / "checkpoint.json")
    name = pointer.get("file", "")
    _require(isinstance(name, str) and Path(name).name == name
             and name.startswith("snapshot_") and name.endswith(".pt"), "Unsafe checkpoint pointer")
    path = _safe(directory / name)
    _require(path.is_file() and path.stat().st_size == pointer.get("bytes")
             and _sha(path) == pointer.get("sha256")
             and pointer.get("request_sha256") == request_sha256,
             "Checkpoint bytes/hash/request mismatch")
    state = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(state, dict) and state.get("request_sha256") == request_sha256,
             "Checkpoint payload identity mismatch")
    return state, pointer


def checkpoint_barrier(callback, directory, event):
    """Callback must finish remote read/hash verification before returning."""
    if callback is None:
        return
    try:
        result = callback(Path(directory), event)
        if result is False:
            raise RuntimeError("Checkpoint callback returned false")
    except Exception as error:
        _append(Path(directory), "failures.json", {
            "at_utc": _now(), "kind": "durability_failure", "event": event,
            "error_type": type(error).__name__,
            "message": "Remote checkpoint verification failed; exception text omitted to protect credentials"})
        raise DurabilityError("Checkpoint save/remote verification failed; local evidence retained") from None


def _configuration(job):
    name = job.get("model", job.get("recipe"))
    _require(isinstance(name, str), "Explicit model identity required")
    _require(type(job.get("seed")) is int, "Explicit integer seed required")
    identity = job.get("identity")
    _require(isinstance(identity, dict) and bool(identity), "Explicit job identity required")
    protocol = job.get("protocol_sha256", identity.get("protocol_sha256"))
    _require(isinstance(protocol, str) and len(protocol) == 64
             and all(c in "0123456789abcdef" for c in protocol), "Protocol SHA256 required")
    recipe = models.model_config(name)
    config = dict(DEFAULT_TRAINING)
    aliases = {"lr": "learning_rate", "wd": "weight_decay", "clip": "gradient_clip",
               "gradient_clip_norm": "gradient_clip"}
    for key, value in recipe.items():
        key = aliases.get(key, key)
        if key in config:
            config[key] = value
    if recipe.get("source_recipe", {}).get("role") == "E2_candidate":
        config["scheduler"] = "warmup_cosine"
    override = job.get("training", {})
    _require(isinstance(override, dict), "training must be an object")
    config.update({k: v for k, v in override.items() if k in config})
    budget = dict(DEFAULT_BUDGET)
    for source in (job.get("budget", {}), override):
        _require(isinstance(source, dict), "budget must be an object")
        for key in budget:
            if key in source:
                budget[key] = source[key]
    for key in ("batch_size", "max_epochs", "patience"):
        _require(type(config[key]) is int and config[key] > 0, "Invalid " + key)
    for key in ("learning_rate", "weight_decay", "gradient_clip"):
        value = config[key]
        _require(type(value) in (int, float) and math.isfinite(value)
                 and (value >= 0 if key == "weight_decay" else value > 0), "Invalid " + key)
    for key, value in budget.items():
        _require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                 "Invalid " + key)
    _require(config["warmup_fraction"] in (0, .05, .1)
             and config["scheduler"] in ("constant", "warmup_cosine"), "Invalid frozen scheduler")
    return name, config, budget, recipe


def _source_identity():
    identity = {"trainer_sha256": _sha(__file__), "models_sha256": _sha(models.__file__)}
    if hasattr(models, "verify_sources"):
        identity["frozen_models"] = models.verify_sources()
    return identity


def critical_runtime_identity():
    """Software cohort that must remain identical for fit/resume/prediction.

    The worker owns a CPU/two-thread execution profile. Python patch/platform
    are observed separately; major/minor and numeric-library versions bind the
    request, so a new Colab interpreter cannot silently resume an old cohort.
    """
    return {"python_implementation": platform.python_implementation(),
            "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
            "torch": str(torch.__version__), "numpy": str(np.__version__),
            "scipy": importlib.metadata.version("scipy"),
            "statsmodels": importlib.metadata.version("statsmodels"),
            "device": "cpu", "threads": int(torch.get_num_threads())}


def runtime_observation():
    """Informational host details, never a GPU/CPU model admission gate."""
    return {"python_full_version": platform.python_version(),
            "python_full_build": sys.version, "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(),
            "scope": "observed at initial checkpoint; not a numerical identity or elapsed-cost measurement"}


def _verify_runtime(request):
    _require(request.get("immutable_runtime") == critical_runtime_identity(),
             "Critical software runtime differs; existing evidence retained")


def _request(job, fit_data, selection_data):
    name, config, budget, recipe = _configuration(job)
    return {"version": VERSION, "job_spec": job, "training": config, "budget": budget,
            "recipe": recipe, "fit_data": data_identity(fit_data),
            "selection_data": data_identity(selection_data),
            "sources": _source_identity(),
            "immutable_runtime": critical_runtime_identity(),
            "phase_names": list(models.training_phases(name)),
            "budget_stop_policy": "nominal_budget selects best fully validated epoch per phase; require >=1 full epoch in every optimizer phase; interrupted epochs rollback and charge reservation or larger observed elapsed",
            "selection": "projected_sigma7 MSE / fit-only y_scale squared; reconstruction MSE for auxiliary phases"}


def _settle(directory, state):
    path = directory / "epoch_pending.json"
    if not path.exists():
        return False
    pending = _read(path)
    _require(pending.get("request_sha256") == state["request_sha256"], "Pending epoch identity differs")
    if pending["segment_id"] not in state["settled_segments"]:
        _require(pending["phase_index"] == state["phase_index"]
                 and pending["epoch"] == state["phase_state"]["epoch"] + 1,
                 "Pending epoch position differs")
        reserved = pending["reserved_seconds"]
        _require(type(reserved) in (int, float) and math.isfinite(reserved) and reserved > 0
                 and reserved <= state["epoch_reservation_seconds"], "Pending reservation differs")
        attempts = _read(directory / "attempts.json") if (directory / "attempts.json").exists() else []
        observed = [a["aborted_seconds_observed"] for a in attempts
                    if a.get("segment_id") == pending["segment_id"] and "aborted_seconds_observed" in a]
        charge = max([reserved] + observed)
        state["active_seconds_charged"] += charge
        state["phase_state"]["seconds_charged"] += charge
        state["interrupted_epochs"] += 1
        state["compute_uncertain"] = True
        state["uncommitted_optimizer_steps"] = None
        state["settled_segments"].append(pending["segment_id"])
        _snapshot(directory, state)
        _append(directory, "attempts.json", {"at_utc": _now(), "kind": "interrupted_epoch_settled",
                "segment_id": pending["segment_id"], "seconds_charged": charge,
                "charge_is_exact_elapsed": False, "optimizer_steps_lost": None})
    path.unlink()
    _sync(directory)
    return True


def _phase_state():
    return {"epoch": 0, "best_epoch": None, "best_score": None, "best": None,
            "stale": 0, "steps": 0, "seconds_charged": 0., "curve": []}


def _score(model, name, phase, partition, fit_state, batch_size):
    raw = partition["raw"]
    model.eval()
    total = 0.
    with torch.no_grad():
        for first in range(0, len(raw), batch_size):
            last = first + batch_size
            if phase == "supervised":
                _, projected, _ = models.predict(model, name, raw[first:last], fit_state)
                target = np.asarray(partition["y"])[first:last]
                _require(np.asarray(projected).shape == target.shape, "Selection shape mismatch")
                scale = float(fit_state.get("y_scale", 1.))
                _require(math.isfinite(scale) and scale > 0, "Invalid fit-only y scale")
                value = float(np.sum(((np.asarray(projected) - target) / scale)**2))
            else:
                loss = models.phase_loss(model, name, phase, raw[first:last],
                                         np.asarray(partition["y"])[first:last], fit_state)
                value = float(loss.detach()) * len(raw[first:last])
            total += value
    score = total / len(raw)
    _require(math.isfinite(score) and score >= 0, "Invalid selection score")
    return score


def _receipt(request, state, pointer, model):
    selected = state.get("selected")
    deterministic = state["deterministic"]
    _require(state.get("completed") is True and (deterministic or (
        selected is not None and state["steps_committed"] > 0 and state["phase_receipts"]
        and all(p["best_score"] is not None and p["steps"] > 0 for p in state["phase_receipts"]))),
        "No complete validated selected checkpoint")
    return {"status": "completed_validated_fit" if not deterministic else "completed_deterministic_system",
            "version": VERSION, "request_sha256": state["request_sha256"],
            "job_spec": request["job_spec"], "model_spec": request["recipe"],
            "sources": request["sources"], "immutable_runtime": request["immutable_runtime"],
            "runtime_observed_at_creation": state["runtime_observed_at_creation"],
            "fit_data": request["fit_data"],
            "selection_data": request["selection_data"], "preprocessor": state["fit_state"],
            "checkpoint_file": pointer["file"], "checkpoint_sha256": pointer["sha256"],
            "selected_state_sha256": _weights_hash(selected) if selected is not None else None,
            "selection_best_score": state["phase_receipts"][-1]["best_score"] if state["phase_receipts"] else None,
            "phase_receipts": [{k: v for k, v in p.items() if k != "best"} for p in state["phase_receipts"]],
            "parameters": sum(p.numel() for p in model.parameters()) if hasattr(model, "parameters") else 0,
            "optimizer_steps_committed": state["steps_committed"],
            "independent_fit": not deterministic, "deterministic_seed_repetition_is_not_independent": deterministic,
            "active_seconds_charged": state["active_seconds_charged"],
            "remaining_nominal_budget_seconds": max(0., request["budget"]["total_budget_seconds"] - state["active_seconds_charged"]),
            "cooperative_nominal_budget_overrun_seconds": max(0., state["active_seconds_charged"] - request["budget"]["total_budget_seconds"]),
            "budget_policy": "one nominal active budget shared by all phases; checked before every batch; not a hard wall deadline",
            "phase_budget_seconds": state["phase_budget_seconds"],
            "checkpoint_io_seconds_observed": state["checkpoint_io_seconds_observed"],
            "checkpoint_io_last_write_not_in_payload": True,
            "setup_seconds_observed": state["setup_seconds_observed"],
            "costs_excluded_from_active_budget": ["initial preprocessing/model setup", "checkpoint IO", "remote backup", "restart downtime"],
            "interrupted_epochs": state["interrupted_epochs"], "compute_uncertain": state["compute_uncertain"],
            "uncommitted_optimizer_steps": state["uncommitted_optimizer_steps"],
            "uncertain_cost_policy": "partial epoch rolls back; charge predeclared reservation or larger observed time once; true unobserved cost unknown",
            "stop_reason": "all_phases_completed" if not deterministic else "deterministic_no_optimizer",
            "completed_at_utc": state["completed_at_utc"]}


def _optimizer(model, name, phase, config):
    parameters = list(models.phase_parameters(model, name, phase))
    _require(bool(parameters), "Optimizer phase has no parameters")
    if config["scheduler"] == "warmup_cosine":
        # Historical E2 optimizer_groups: biases/norms are exempt from decay.
        submodel = models.phase_model(model, name, phase)
        normalization = set()
        norm_types = (torch.nn.LayerNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                      torch.nn.BatchNorm3d, torch.nn.GroupNorm, torch.nn.InstanceNorm1d,
                      torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d)
        for module in submodel.modules():
            if isinstance(module, norm_types):
                normalization.update(id(p) for p in module.parameters(recurse=False))
        decayed, exempt = [], []
        for key, p in submodel.named_parameters():
            if p.requires_grad:
                (exempt if p.ndim < 2 or key.endswith("bias") or id(p) in normalization else decayed).append(p)
        return torch.optim.AdamW([{"params": decayed, "weight_decay": config["weight_decay"]},
                                  {"params": exempt, "weight_decay": 0.}], lr=config["learning_rate"])
    return torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"])


def _learning_rate(config, step, total_steps):
    if config["scheduler"] == "constant":
        return config["learning_rate"]
    # Attributed historical e2_staged_training.learning_rate_multiplier policy.
    if step == total_steps:
        return 0.
    warmup = int(round(total_steps * config["warmup_fraction"]))
    if warmup and step < warmup:
        multiplier = (step + 1) / warmup
    else:
        progress = (step - warmup) / (total_steps - warmup)
        multiplier = .5 * (1 + math.cos(math.pi * progress))
    return config["learning_rate"] * multiplier


def fit_job(job_spec, fit_data, selection_data, job_dir, *, on_checkpoint=None,
            checkpoint_interval_seconds=60.):
    """Fit/resume one owned job; return a hash-verified selected-state receipt.

    ``on_checkpoint(directory, event)`` is a synchronous remote durability
    barrier. The orchestrator must supply its GitHub read/hash verifier for real
    campaigns. ``None`` permits isolated local software fixtures only.
    """
    _require(type(checkpoint_interval_seconds) in (int, float)
             and 0 < checkpoint_interval_seconds <= 60, "Checkpoint interval must be <=60 seconds")
    _partition(fit_data)
    _partition(selection_data)
    directory = _safe(job_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with _lock(directory):
        return _fit(job_spec, fit_data, selection_data, directory, on_checkpoint,
                    float(checkpoint_interval_seconds))


def _fit(job, fit_data, selection_data, directory, callback, interval):
    setup_started = time.perf_counter()
    torch.set_num_threads(2)
    if (directory / "request.json").exists():
        _verify_runtime(_read(directory / "request.json"))
    request = _request(job, fit_data, selection_data)
    request_hash = _digest(request)
    _write(directory / "request.json", request, immutable=True)
    name = job.get("model", job.get("recipe"))
    phases = request["phase_names"]
    config, budget = request["training"], request["budget"]
    _seed(job["seed"])
    fit_state = models.fit_preprocessor(fit_data["raw"], fit_data["y"])
    model = models.build_model(name, job["seed"], fit_state)
    if hasattr(model, "cpu"):
        model = model.cpu()
    if (directory / "checkpoint.json").exists():
        state, pointer = _load(directory, request_hash)
        _require(state["fit_state"] == fit_state, "Fit-only preprocessing differs")
        _settle(directory, state)
    else:
        _require(not (directory / "done.json").exists(), "Completion marker has no checkpoint")
        state = {"request_sha256": request_hash, "fit_state": fit_state,
                 "runtime_observed_at_creation": runtime_observation(),
                 "deterministic": not bool(phases), "phase_index": 0,
                 "phase_state": _phase_state(), "phase_receipts": [],
                 "phase_optimizers": {},
                 "current": _weights(model) if phases else None, "optimizer": None,
                 "rng": _rng(), "selected": None, "steps_committed": 0,
                 "completed": False, "active_seconds_charged": 0.,
                 "epoch_reservation_seconds": budget["epoch_reservation_seconds"] if phases else budget["total_budget_seconds"],
                 "phase_budget_seconds": budget["total_budget_seconds"] / max(1, len(phases)),
                 "settled_segments": [], "interrupted_epochs": 0,
                 "compute_uncertain": False, "uncommitted_optimizer_steps": 0,
                 "checkpoint_io_seconds_observed": 0.,
                 "setup_seconds_observed": time.perf_counter() - setup_started}
        pointer = _snapshot(directory, state)
    checkpoint_barrier(callback, directory, {"kind": "before_fit", "request_sha256": request_hash,
                                          "completed": state["completed"]})
    last_barrier = time.perf_counter()
    if state["completed"]:
        state, pointer = _load(directory, request_hash)
        receipt = _receipt(request, state, pointer, model)
        if (directory / "done.json").exists():
            _require(_read(directory / "done.json") == receipt, "Done receipt/checkpoint mismatch")
        else:
            _write(directory / "done.json", receipt, immutable=True)
        checkpoint_barrier(callback, directory, {"kind": "job_completed_verified", "request_sha256": request_hash})
        return receipt
    if (directory / "running.json").exists():
        _require(_read(directory / "running.json").get("request_sha256") == request_hash,
                 "Running marker identity differs")
    else:
        _write(directory / "running.json", {"request_sha256": request_hash,
               "started_at_utc": _now(), "policy": "nominal active budget; resume never resets charges"}, immutable=True)
    _append(directory, "attempts.json", {"at_utc": _now(), "kind": "attempt_started",
            "phase_index": state["phase_index"], "epoch": state["phase_state"]["epoch"],
            "previous_seconds_charged": state["active_seconds_charged"]})
    try:
        if not phases:
            # Deterministic adapters own their causal, fit-only calibration.
            if state.get("deterministic_calibrated"):
                model.load_state_dict(state["deterministic_state"])
            else:
                remaining = budget["total_budget_seconds"] - state["active_seconds_charged"]
                if remaining <= 0:
                    raise TrainingIncomplete("Deterministic calibration budget exhausted; never reset on recovery")
                pending = {"request_sha256": request_hash, "segment_id": uuid.uuid4().hex,
                           "phase_index": 0, "phase": "deterministic_calibration", "epoch": 1,
                           "reserved_seconds": remaining, "started_at_utc": _now()}
                _write(directory / "epoch_pending.json", pending, immutable=True)
                started = time.perf_counter()
                try:
                    fitted = models.fit_deterministic(model, name, fit_data, fit_state,
                                                      budget_seconds=remaining)
                    if fitted is not None:
                        model = fitted
                    elapsed = time.perf_counter() - started
                    if elapsed > remaining:
                        raise TrainingIncomplete("Deterministic calibration exceeded cooperative nominal budget")
                except BaseException:
                    _append(directory, "attempts.json", {"at_utc": _now(), "kind": "aborted_calibration",
                            "segment_id": pending["segment_id"], "aborted_seconds_observed": time.perf_counter() - started,
                            "reserved_seconds": remaining, "epoch_committed": False})
                    raise
                state["active_seconds_charged"] += elapsed
                state["deterministic_state"] = model.state_dict()
                state["deterministic_calibrated"] = True
                state["settled_segments"].append(pending["segment_id"])
                _snapshot(directory, state)
                (directory / "epoch_pending.json").unlink()
                _sync(directory)
        while state["phase_index"] < len(phases):
            phase = phases[state["phase_index"]]
            model.load_state_dict(state["current"])
            optimizer = _optimizer(model, name, phase, config)
            if state["optimizer"] is not None:
                optimizer.load_state_dict(state["optimizer"])
            _restore_rng(state["rng"])
            ps = state["phase_state"]
            while ps["epoch"] < config["max_epochs"] and ps["stale"] < config["patience"]:
                remaining = min(budget["total_budget_seconds"] - state["active_seconds_charged"],
                                state["phase_budget_seconds"] - ps["seconds_charged"])
                if remaining <= 0:
                    if ps["best"] is None:
                        raise TrainingIncomplete("Nominal phase/system budget exhausted before a validated full epoch")
                    ps["budget_stop"] = True
                    break
                allowance = min(budget["epoch_reservation_seconds"], remaining)
                pending = {"request_sha256": request_hash, "segment_id": uuid.uuid4().hex,
                           "phase_index": state["phase_index"], "phase": phase,
                           "epoch": ps["epoch"] + 1, "reserved_seconds": allowance, "started_at_utc": _now()}
                _write(directory / "epoch_pending.json", pending, immutable=True)
                started = time.perf_counter()
                epoch_steps = 0
                losses = []
                rates = []
                callback_seconds = 0.
                try:
                    model.eval()
                    models.phase_model(model, name, phase).train()
                    order = torch.randperm(len(fit_data["raw"]))
                    for first in range(0, len(order), config["batch_size"]):
                        if time.perf_counter() - last_barrier >= interval:
                            callback_started = time.perf_counter()
                            checkpoint_barrier(callback, directory, {"kind": "minute_checkpoint", "phase": phase,
                                              "epoch": pending["epoch"], "pending_epoch_is_not_committed": True})
                            callback_seconds += time.perf_counter() - callback_started
                            last_barrier = time.perf_counter()
                        if time.perf_counter() - started - callback_seconds >= remaining:
                            raise TrainingIncomplete("Nominal phase/system budget reached; partial epoch is rolled back")
                        idx = order[first:first + config["batch_size"]].numpy()
                        rate = _learning_rate(config, ps["steps"] + epoch_steps,
                                              config["max_epochs"] * math.ceil(len(order) / config["batch_size"]))
                        for group in optimizer.param_groups:
                            group["lr"] = rate
                        optimizer.zero_grad(set_to_none=True)
                        loss = models.phase_loss(model, name, phase, np.asarray(fit_data["raw"])[idx],
                                                 np.asarray(fit_data["y"])[idx], fit_state)
                        _require(torch.is_tensor(loss) and loss.ndim == 0 and bool(torch.isfinite(loss)),
                                 "Training loss must be a finite scalar tensor")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(models.phase_parameters(model, name, phase),
                                                      config["gradient_clip"], error_if_nonfinite=True)
                        optimizer.step()
                        epoch_steps += 1
                        rates.append(rate)
                        losses.append(float(loss.detach()))
                    score = _score(model, name, phase, selection_data, fit_state, config["batch_size"])
                    elapsed = time.perf_counter() - started - callback_seconds
                    if elapsed > remaining:
                        raise TrainingIncomplete("Epoch/selection exceeded cooperative nominal budget; partial epoch is rolled back")
                except BaseException as error:
                    _append(directory, "attempts.json", {"at_utc": _now(), "kind": "aborted_epoch",
                            "segment_id": pending["segment_id"], "phase": phase, "epoch": pending["epoch"],
                            "aborted_seconds_observed": time.perf_counter() - started - callback_seconds,
                            "optimizer_steps_observed_uncommitted": epoch_steps, "epoch_committed": False,
                            "reserved_seconds": allowance})
                    if isinstance(error, TrainingIncomplete) and ps["best"] is not None:
                        _settle(directory, state)
                        model.load_state_dict(state["current"])
                        optimizer.load_state_dict(state["optimizer"])
                        _restore_rng(state["rng"])
                        ps["budget_stop"] = True
                        break
                    raise
                ps["epoch"] += 1
                ps["steps"] += epoch_steps
                ps["seconds_charged"] += elapsed
                state["steps_committed"] += epoch_steps
                state["active_seconds_charged"] += elapsed
                if ps["best_score"] is None or score < ps["best_score"]:
                    ps.update(best_score=score, best_epoch=ps["epoch"], best=_weights(model), stale=0)
                else:
                    ps["stale"] += 1
                ps["curve"].append({"epoch": ps["epoch"], "train_loss_mean_batches": float(np.mean(losses)),
                                    "selection_score": score, "seconds": elapsed, "optimizer_steps": epoch_steps,
                                    "learning_rate_first": rates[0], "learning_rate_last": rates[-1]})
                state.update(current=_weights(model), optimizer=optimizer.state_dict(), rng=_rng())
                state["settled_segments"].append(pending["segment_id"])
                _snapshot(directory, state)
                (directory / "epoch_pending.json").unlink()
                _sync(directory)
                _write(directory / "curve.json", {"completed_phases": [{k: v for k, v in p.items() if k != "best"}
                                                       for p in state["phase_receipts"]], "current_phase": phase,
                                                    "curve": ps["curve"]})
                checkpoint_barrier(callback, directory, {"kind": "epoch_checkpoint", "phase": phase,
                                                          "epoch": ps["epoch"]})
                last_barrier = time.perf_counter()
            _require(ps["best"] is not None and ps["steps"] > 0, "No validated trained phase checkpoint")
            model.load_state_dict(ps["best"])
            ps["name"] = phase
            ps["stop_reason"] = "nominal_budget" if ps.get("budget_stop") else (
                "early_stopping" if ps["stale"] >= config["patience"] else "max_epochs")
            state["phase_receipts"].append(ps)
            state["phase_optimizers"][phase] = optimizer.state_dict()
            if phase == "supervised":
                state["selected"] = _weights(model)
            models.transition_phase(model, name, phase)
            state.update(phase_index=state["phase_index"] + 1, phase_state=_phase_state(),
                         current=_weights(model), optimizer=None, rng=_rng())
            _snapshot(directory, state)
        state.update(completed=True, completed_at_utc=_now())
        pointer = _snapshot(directory, state)
        state, pointer = _load(directory, request_hash)
        receipt = _receipt(request, state, pointer, model)
        _write(directory / "done.json", receipt, immutable=True)
        _write(directory / "resources.json", receipt)
        checkpoint_barrier(callback, directory, {"kind": "job_completed", "request_sha256": request_hash})
        return receipt
    except BaseException as error:
        _append(directory, "failures.json", {"at_utc": _now(), "kind": "fit_failure",
                "error_type": type(error).__name__, "message": str(error),
                "phase_index": state["phase_index"], "committed_steps": state["steps_committed"],
                "scope": "attempt; never a new model or successful fit"})
        export_job_resources(directory)
        raise


def export_job_resources(job_dir):
    """Export committed costs plus separate failed-attempt observations."""
    directory = _safe(job_dir)
    request = _read(directory / "request.json")
    state, pointer = _load(directory, _digest(request))
    attempts = _read(directory / "attempts.json") if (directory / "attempts.json").exists() else []
    failures = _read(directory / "failures.json") if (directory / "failures.json").exists() else []
    pending = _read(directory / "epoch_pending.json") if (directory / "epoch_pending.json").exists() else None
    result = {"version": VERSION, "request_sha256": state["request_sha256"],
              "immutable_runtime": request["immutable_runtime"],
              "runtime_observed_at_creation": state["runtime_observed_at_creation"],
              "status": "completed" if state["completed"] else "incomplete",
              "checkpoint_sha256": pointer["sha256"], "optimizer_steps_committed": state["steps_committed"],
              "active_seconds_charged": state["active_seconds_charged"],
              "remaining_nominal_budget_seconds": max(0., request["budget"]["total_budget_seconds"] - state["active_seconds_charged"]),
              "checkpoint_io_seconds_observed": state["checkpoint_io_seconds_observed"],
              "setup_seconds_observed": state["setup_seconds_observed"],
              "compute_uncertain": state["compute_uncertain"] or pending is not None,
              "interrupted_epochs_settled": state["interrupted_epochs"],
              "pending_reservation": pending, "attempts": attempts, "failures": failures,
              "unobserved_partial_compute_not_reconstructed": True,
              "active_budget_excludes": ["setup", "checkpoint_IO", "remote_backup", "downtime"],
              "budget_is_cooperative_not_hard_wall_deadline": True}
    _write(directory / "resources.json", result)
    return result


def verify_job(job_spec, job_dir):
    """Verify the owned selected checkpoint without opening any data partition."""
    directory = _safe(job_dir)
    request = _read(directory / "request.json")
    torch.set_num_threads(2)
    _verify_runtime(request)
    _require(request.get("job_spec") == job_spec, "Job identity/configuration differs")
    _require(request["sources"] == _source_identity(),
             "Source identity differs")
    state, pointer = _load(directory, _digest(request))
    name = job_spec.get("model", job_spec.get("recipe"))
    saved_rng = _rng()
    try:
        model = models.build_model(name, job_spec["seed"], state["fit_state"])
    finally:
        _restore_rng(saved_rng)
    receipt = _receipt(request, state, pointer, model)
    _require((directory / "done.json").exists() and _read(directory / "done.json") == receipt,
             "Missing/different completed receipt")
    return receipt


def predict_job(job_spec, partition, job_dir, *, batch_size=256):
    """Load best selected weights; deterministic batched raw/projected sigma7."""
    _require(type(batch_size) is int and batch_size > 0, "Positive prediction batch size required")
    raw = _partition(partition, target=False)
    directory = _safe(job_dir)
    with _lock(directory):
        receipt = verify_job(job_spec, directory)
        request = _read(directory / "request.json")
        state, _ = _load(directory, _digest(request))
        name = job_spec.get("model", job_spec.get("recipe"))
        saved_rng = _rng()
        try:
            torch.set_num_threads(2)
            model = models.build_model(name, job_spec["seed"], state["fit_state"])
            if state["deterministic"]:
                if state.get("deterministic_state") is not None:
                    model.load_state_dict(state["deterministic_state"])
            else:
                model.load_state_dict(state["selected"])
                model.eval()
            chunks = [[], [], []]
            with torch.no_grad():
                for first in range(0, len(raw), batch_size):
                    last = first + batch_size
                    row_keys = {"raw", "y", "dates", "datesISO", "support_start", "support_end",
                                "known_sigma_history", "known_floor_A", "forecast_offsets", "history_end_positions",
                                "arima_in_sample_positions"}
                    supplied = {k: np.asarray(v)[first:last] if k in row_keys
                                else v for k, v in partition.items()} if state["deterministic"] else raw[first:last]
                    output = models.predict(model, name, supplied, state["fit_state"])
                    _require(len(output) == 3, "Predict adapter must return raw, projected, floor")
                    for dest, value in zip(chunks, output):
                        value = np.asarray(value)
                        if value.ndim == 0:
                            value = np.full(len(raw[first:last]), float(value))
                        _require(value.shape == (len(raw[first:last]),) and np.isfinite(value).all(),
                                 "Invalid sigma7 prediction")
                        dest.append(value)
            return {"raw_sigma7": np.concatenate(chunks[0]), "projected_sigma7": np.concatenate(chunks[1]),
                    "floor": np.concatenate(chunks[2]), "checkpoint_sha256": receipt["checkpoint_sha256"],
                    "selected_state_sha256": receipt["selected_state_sha256"]}
        finally:
            _restore_rng(saved_rng)
