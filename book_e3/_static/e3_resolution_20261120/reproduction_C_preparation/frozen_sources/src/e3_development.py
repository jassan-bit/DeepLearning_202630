"""A small, bounded E3 development trial: full candidate versus joint M00 control.

Only the three training_* raw arrays are admitted from the known E2 NPZ. New
inner-fit statistics are estimated before using the development validation
period. This is not preregistration, confirmation, an E2 rerun, or evidence of
novelty/superiority. No calibration/evaluation array is indexed or scored here.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import time
import uuid

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.original_model import CandidateConfig, OriginalModel


VERSION = "e3-development-v1"
MODEL_NAMES = ("full", "M00")
DEFAULT_PLAN = {
    "version": VERSION,
    "scope": "development_only_not_preregistered_not_confirmation",
    "seed": 20261116,
    "models": list(MODEL_NAMES),
    "data": {"fit_end": "2024-06-30", "validation_start": "2024-07-08",
             "validation_end": "2024-10-31"},
    "training": {"optimizer": "AdamW", "learning_rate": 1e-3,
                 "weight_decay": 0.0, "batch_size": 32, "max_epochs": 40,
                 "patience": 8, "gradient_clip_norm": 1.0},
    "budget": {"total_seconds": 1200.0},
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + canonical(list(value.shape))
                          + value.tobytes()).hexdigest()


def _iso(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError("Dates must be canonical ISO YYYY-MM-DD")
    return value


def load_development_data(npz_path, expected_sha256, fit_end="2024-06-30",
                          validation_start="2024-07-08", validation_end="2024-10-31"):
    """Return inner-fit/development arrays; never index other NPZ members.

    The whole-file hash verifies transport without analysing its other arrays.
    Channel means/population SD and target mean use only dates <= fit_end.
    """
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("An explicit expected NPZ SHA-256 is required")
    actual = sha256_file(npz_path)
    if actual != expected_sha256.lower():
        raise ValueError("NPZ SHA-256 mismatch")
    fit_end, validation_start, validation_end = map(_iso, (fit_end, validation_start, validation_end))
    if date.fromisoformat(validation_start) - timedelta(days=7) <= date.fromisoformat(fit_end):
        raise ValueError("Seven-day raw supports of fit and validation would overlap")
    if validation_start > validation_end or validation_end > "2024-10-31":
        raise ValueError("Development validation must end no later than 2024-10-31")
    keys = ("training_raw", "training_y_original", "training_dates")
    try:
        with np.load(npz_path, allow_pickle=False) as source:
            raw = np.asarray(source[keys[0]], dtype=np.float64).copy()
            target = np.asarray(source[keys[1]], dtype=np.float64).copy()
            loaded_dates = np.asarray(source[keys[2]]).copy()
    except KeyError as error:
        raise ValueError("Required training-only NPZ member is missing") from error
    if (raw.ndim != 3 or raw.shape[1:] != (7, 5) or raw.shape[0] < 2
            or target.shape != (raw.shape[0],) or loaded_dates.shape != target.shape):
        raise ValueError("Expected training_raw N x 7 x 5 and N targets/dates")
    dates = np.array([_iso(str(x)) for x in loaded_dates], dtype="U10")
    if dates.tolist() != sorted(set(dates.tolist())) or dates[-1] > "2024-10-31":
        raise ValueError("Training dates must be sorted, unique and within E2 development")
    if (not np.isfinite(raw).all() or not np.isfinite(target).all()
            or (raw[:, :, :4] <= 0).any() or (raw[:, :, 4] < 0).any() or (target < 0).any()):
        raise ValueError("Invalid raw OHLCV or original volatility target")
    if ((raw[:, :, 1] < np.maximum(raw[:, :, 0], raw[:, :, 3])).any()
            or (raw[:, :, 2] > np.minimum(raw[:, :, 0], raw[:, :, 3])).any()
            or (raw[:, :, 1] < raw[:, :, 2]).any()):
        raise ValueError("Invalid OHLC geometry")
    fit = dates <= fit_end
    validation = (dates >= validation_start) & (dates <= validation_end)
    if fit.sum() < 2 or validation.sum() < 1:
        raise ValueError("At least two inner-fit and one validation examples are required")
    relative = raw.copy()
    relative[:, :, :4] = np.log(raw[:, :, :4]) - np.log(raw[:, -1, 3])[:, None, None]
    relative[:, :, 4] = np.log1p(raw[:, :, 4])
    mean = relative[fit].mean(axis=(0, 1))
    std = relative[fit].std(axis=(0, 1), ddof=0)
    std = np.where(std == 0, 1.0, std)
    target_scale = float(target[fit].mean())
    if not math.isfinite(target_scale) or target_scale <= 0:
        raise ValueError("Inner-fit target mean must be positive")
    train_x = ((relative[fit] - mean) / std).astype(np.float32)
    validation_x = ((relative[validation] - mean) / std).astype(np.float32)
    if not np.isfinite(train_x).all() or not np.isfinite(validation_x).all():
        raise ValueError("Preprocessing produced nonfinite float32 values")
    scaler = {"mean": mean.tolist(), "std": std.tolist(), "target_scale": target_scale,
              "ddof": 0, "fit_dates": dates[fit].tolist(), "relative_window_anchor": "last_known_close",
              "target_scaling": "divide_by_inner_fit_mean_without_centering"}
    admission = {
        "version": VERSION, "npz_sha256": actual, "keys_read": list(keys),
        "forbidden_arrays_loaded": False, "training_scaled_arrays_reused": False,
        "n_training_source": len(raw), "n_fit": int(fit.sum()),
        "n_validation": int(validation.sum()), "n_purged_or_unused": int((~(fit | validation)).sum()),
        "fit_dates": dates[fit].tolist(), "validation_dates": dates[validation].tolist(),
        "fit_end": fit_end, "validation_start": validation_start, "validation_end": validation_end,
        "last_fit_support_end": str(dates[fit][-1]),
        "first_validation_support_start": (date.fromisoformat(str(dates[validation][0])) - timedelta(days=7)).isoformat(),
        "raw_supports_disjoint": True, "scaler": scaler,
        "arrays_sha256": {"train_X": _array_hash(train_x), "train_y_original": _array_hash(target[fit]),
                          "validation_X": _array_hash(validation_x),
                          "validation_y_original": _array_hash(target[validation])},
        "interpretation": "validation reused for stopping; descriptive development results only",
    }
    return {"train_X": train_x, "train_y_scaled": (target[fit] / target_scale).astype(np.float32),
            "train_y_original": target[fit], "train_dates": dates[fit],
            "validation_X": validation_x,
            "validation_y_scaled": (target[validation] / target_scale).astype(np.float32),
            "validation_y_original": target[validation], "validation_dates": dates[validation],
            "scaler": scaler, "admission": admission}


class DevelopmentModel(nn.Module):
    """Full candidate or joint M00 control; immutable original_model.py is reused.

    Both constructors first create exactly the same seeded full model. M00 then
    removes penalty_raw and radial saturation, preserving encoder, q, w and tau
    initial tensors exactly. It has one fewer parameter and no bounded-head
    certificate. This pair does not isolate the two component effects.
    """
    def __init__(self, model_name="full", seed=20261116):
        super().__init__()
        if model_name not in MODEL_NAMES:
            raise ValueError("model_name must be full or M00")
        self.model_name = model_name
        self.model = OriginalModel(CandidateConfig(), seed=seed)
        if model_name == "M00":
            self.model.mechanism.use_penalty = False
            self.model.mechanism.use_saturation = False
            self.model.mechanism.register_parameter("penalty_raw", None)

    def forward(self, values):
        return self.model(values)

    def configuration(self):
        return {"model_name": self.model_name, "base": self.model.configuration(),
                "penalty_enabled": self.model.mechanism.use_penalty,
                "saturation_enabled": self.model.mechanism.use_saturation,
                "latent_bound_applicable": self.model_name == "full",
                "trainable_parameters": sum(p.numel() for p in self.parameters()),
                "comparison": "joint_two_component_development_control_not_separate_attribution"}


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode() + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def run_lock(run_dir):
    """Kernel-owned exclusive lock; a killed process releases it automatically."""
    try:
        import fcntl
    except ImportError as error:
        raise RuntimeError("This runner requires a working POSIX file lock (Colab/Linux)") from error
    stream = (Path(run_dir) / ".run.lock").open("a+")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as error:
            raise RuntimeError("Run is locked by another process; no changes made") from error
        yield
    finally:
        stream.close()


def source_fingerprints():
    root = Path(__file__).resolve().parents[1]
    return {name: sha256_file(root / name) for name in
            ["src/original_model.py", "src/e3_development.py", "scripts/e3_run_development.py"]}


def normalized_plan(plan=None):
    if plan is not None and not isinstance(plan, dict):
        raise ValueError("Plan must be a JSON object")
    result = deepcopy(DEFAULT_PLAN)
    supplied = deepcopy(plan or {})
    for section in ("data", "training", "budget"):
        if section in supplied:
            if not isinstance(supplied[section], dict):
                raise ValueError(f"Plan {section} must be an object")
            unknown = set(supplied[section]) - set(result[section])
            if unknown:
                raise ValueError(f"Unknown {section} settings: {sorted(unknown)}")
            result[section].update(supplied.pop(section))
    result.update(supplied)  # Additional documentary fields are preserved and fingerprinted.
    if type(result["seed"]) is not int or not 0 <= result["seed"] < 2 ** 32:
        raise ValueError("Seed must be a nonnegative 32-bit integer")
    if result["models"] != list(MODEL_NAMES):
        raise ValueError("The paired trial must retain full and M00 in recorded order")
    train = result["training"]
    if train["optimizer"] != "AdamW":
        raise ValueError("Only the recorded AdamW recipe is supported")
    for key in ("batch_size", "max_epochs", "patience"):
        if type(train[key]) is not int or train[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate", "gradient_clip_norm"):
        if isinstance(train[key], bool) or not math.isfinite(float(train[key])) or train[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if isinstance(train["weight_decay"], bool) or not math.isfinite(float(train["weight_decay"])) or train["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    seconds = result["budget"]["total_seconds"]
    if isinstance(seconds, bool) or not math.isfinite(float(seconds)) or not 0 < seconds <= 1200:
        raise ValueError("A positive finite wall budget no larger than 1200 seconds is required")
    if result["data"]["validation_end"] > "2024-10-31":
        raise ValueError("Only the training-era development data may be admitted")
    canonical(result)
    return result


def runtime_info(device="auto"):
    numbers = torch.__version__.split("+")[0].split(".")
    if tuple(int(x) for x in numbers[:2]) < (2, 8):
        raise RuntimeError("This development code requires PyTorch >= 2.8")
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    selected = "cuda" if device == "auto" and torch.cuda.is_available() else device
    selected = "cpu" if selected == "auto" else selected
    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    info = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "platform": platform.platform(), "device_type": selected,
            "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "cpu_threads": torch.get_num_threads(), "deterministic_algorithms": True,
            "AMP": False, "TF32": False, "cudnn_benchmark": False}
    if selected == "cuda":
        info.update({"gpu": torch.cuda.get_device_name(0),
                     "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                     "gpu_count": torch.cuda.device_count()})
    return info


def _configure_determinism(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise RuntimeError("CUDA generator topology changed; cannot restore RNG state")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["torch_cuda"]])


def _new_checkpoint(model, optimizer, manifest, seed):
    generator = torch.Generator().manual_seed(seed)
    return {"version": VERSION, "model_name": model.model_name, "model_state": deepcopy(model.state_dict()),
            "optimizer_state": deepcopy(optimizer.state_dict()), "rng": _rng_state(),
            "dataloader_generator_state": generator.get_state(), "epoch": 0,
            "active_permutation": None, "batch_cursor": 0, "epoch_loss_sum": 0.0,
            "epoch_examples": 0, "best_validation_rmse": None, "best_epoch": None,
            "no_improvement": 0, "history": [], "complete": False,
            "source_hashes": manifest["source_hashes"], "plan_sha256": manifest["plan_sha256"],
            "data_fingerprint": manifest["data_fingerprint"]}


def create_run(npz_path, expected_sha256, run_dir, *, device="auto", plan_path=None, plan=None):
    """Preflight and initialize a new directory. No optimizer step or fit occurs."""
    if plan_path is not None and plan is not None:
        raise ValueError("Provide plan_path or plan, not both")
    plan = normalized_plan(_read_json(plan_path) if plan_path is not None else plan)
    data = load_development_data(npz_path, expected_sha256, **plan["data"])
    runtime = runtime_info(device)
    sources = source_fingerprints()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"version": VERSION, "run_id": uuid.uuid4().hex, "created_at_utc": _utc(),
                "scope": "development_only_not_preregistered_not_confirmation", "npz_sha256": expected_sha256.lower(),
                "plan_sha256": hashlib.sha256(canonical(plan)).hexdigest(), "source_hashes": sources,
                "data_fingerprint": hashlib.sha256(canonical(data["admission"])).hexdigest(),
                "runtime_initial": runtime, "models": {}}
    with run_lock(run_dir):
        _configure_determinism(plan["seed"])
        for name in MODEL_NAMES:
            model = DevelopmentModel(name, plan["seed"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=plan["training"]["learning_rate"],
                                          weight_decay=plan["training"]["weight_decay"])
            directory = run_dir / name
            directory.mkdir()
            manifest["models"][name] = model.configuration()
            _save_checkpoint(directory / "last.pt", _new_checkpoint(model, optimizer, manifest, plan["seed"]))
        for name, value in [("manifest.json", manifest), ("plan.json", plan), ("runtime.json", runtime),
                            ("data_admission.json", data["admission"]),
                            ("state.json", {"status": "initialized", "models": {n: {"epoch": 0, "complete": False} for n in MODEL_NAMES}}),
                            ("ledger.json", {"version": VERSION, "limit_seconds": float(plan["budget"]["total_seconds"]),
                                              "charged_seconds": 0.0, "events": [], "active_session": None})]:
            _atomic_json(run_dir / name, value)
        _append_event(run_dir, {"event": "preflight_completed", "training_performed": False})
    return manifest


def _append_event(run_dir, event):
    with (Path(run_dir) / "history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"utc": _utc(), **event}, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load_checkpoint(path, manifest):
    # These locally produced full optimizer/RNG checkpoints contain NumPy RNG
    # state. Deserialize only after run fingerprints and checkpoint hash check.
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sha256_file(path) not in sidecar.read_text().splitlines():
        raise RuntimeError("Checkpoint byte hash mismatch")
    value = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("source_hashes", "plan_sha256", "data_fingerprint"):
        if value[key] != manifest[key]:
            raise RuntimeError("Checkpoint fingerprint mismatch")
    return value


def _save_checkpoint(path, checkpoint):
    """Commit a complete checkpoint with a crash-safe two-generation digest.

    Receipt old+new is committed before replacing the checkpoint, then compacted
    to new. Every possible interrupted boundary therefore admits exactly the
    old or new complete file, never a partially written tensor payload.
    """
    path = Path(path)
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    payload = buffer.getvalue()
    new_digest = hashlib.sha256(payload).hexdigest()
    receipt = path.with_suffix(path.suffix + ".sha256")
    accepted = ([sha256_file(path)] if path.exists() else []) + [new_digest]
    def replace_bytes(destination, value):
        temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    replace_bytes(receipt, ("\n".join(dict.fromkeys(accepted)) + "\n").encode())
    replace_bytes(path, payload)
    replace_bytes(receipt, (new_digest + "\n").encode())


def _recover_best(run_dir, name, checkpoint):
    """Regenerate best/CSV derivatives from last.pt's committed best snapshot."""
    snapshot = checkpoint.get("best_snapshot")
    if snapshot is None:
        if checkpoint["best_epoch"] is not None:
            raise RuntimeError("Best epoch has no committed recovery snapshot")
        return
    best_path = run_dir / name / "best.pt"
    rewrite = not best_path.exists()
    if not rewrite:
        try:
            manifest = {key: checkpoint[key] for key in ("source_hashes", "plan_sha256", "data_fingerprint")}
            best = _load_checkpoint(best_path, manifest)
            rewrite = best["epoch"] != checkpoint["best_epoch"]
        except (RuntimeError, OSError, KeyError):
            rewrite = True
    if rewrite:
        _save_checkpoint(best_path, snapshot["checkpoint"])
    _write_csv(run_dir / name / "validation_predictions.csv", snapshot["predictions"])


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def _write_csv(path, rows):
    if not rows:
        return
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(stream.getvalue())
    os.replace(temporary, path)


class _WallBudget:
    def __init__(self, directory, ledger, clock, device):
        self.directory, self.ledger, self.clock, self.device = directory, ledger, clock, device
        self.last = float(clock())
        if not math.isfinite(self.last):
            raise ValueError("Clock must be finite")
        self.session_id = uuid.uuid4().hex
        active = ledger.get("active_session")
        if active:
            gap = max(0.0, (datetime.now(timezone.utc) - datetime.fromisoformat(active["last_wall_utc"])).total_seconds())
            charge = min(gap, max(0.0, ledger["limit_seconds"] - ledger["charged_seconds"]))
            ledger["charged_seconds"] += charge
            ledger["events"].append({"kind": "unclosed_session_recovery", "seconds": charge,
                                      "previous_session": active["session_id"],
                                      "scope": "conservative wall-gap charge; may include downtime"})
        ledger["active_session"] = {"session_id": self.session_id, "last_wall_utc": _utc()}
        _atomic_json(directory / "ledger.json", ledger)

    def tick(self):
        _sync(self.device)
        current = float(self.clock())
        if not math.isfinite(current) or current < self.last:
            raise ValueError("Clock must be finite and monotonic")
        delta, self.last = current - self.last, current
        self.ledger["charged_seconds"] += delta
        self.ledger["active_session"]["last_wall_utc"] = _utc()
        _atomic_json(self.directory / "ledger.json", self.ledger)
        return max(0.0, self.ledger["limit_seconds"] - self.ledger["charged_seconds"])

    def close(self, status):
        self.tick()
        self.ledger["events"].append({"kind": "session_closed", "session_id": self.session_id,
                                      "status": status, "charged_total_seconds": self.ledger["charged_seconds"]})
        self.ledger["active_session"] = None
        _atomic_json(self.directory / "ledger.json", self.ledger)


def _validate_resume(run_dir, npz_path, expected_sha256, device):
    manifest, plan = _read_json(run_dir / "manifest.json"), _read_json(run_dir / "plan.json")
    if source_fingerprints() != manifest["source_hashes"]:
        raise RuntimeError("Source fingerprint changed; do not mix code versions in a run")
    if hashlib.sha256(canonical(plan)).hexdigest() != manifest["plan_sha256"]:
        raise RuntimeError("Plan fingerprint changed")
    if expected_sha256.lower() != manifest["npz_sha256"]:
        raise RuntimeError("Input fingerprint differs from the run manifest")
    data = load_development_data(npz_path, expected_sha256, **plan["data"])
    if hashlib.sha256(canonical(data["admission"])).hexdigest() != manifest["data_fingerprint"]:
        raise RuntimeError("Data/scaler fingerprint changed")
    runtime = runtime_info(device)
    old = manifest["runtime_initial"]
    if runtime["device_type"] != old["device_type"]:
        raise RuntimeError("Resume requires the original CPU/CUDA device kind")
    if runtime["torch"].split("+")[0].split(".")[:2] != old["torch"].split("+")[0].split(".")[:2]:
        raise RuntimeError("Resume requires the same PyTorch major/minor version")
    return manifest, plan, data, runtime


def _summary(run_dir, manifest, status):
    ledger = _read_json(run_dir / "ledger.json")
    models = {}
    for name in MODEL_NAMES:
        last = _load_checkpoint(run_dir / name / "last.pt", manifest)
        # last.pt is authoritative even if publishing a derived best/CSV failed.
        snapshot = last.get("best_snapshot")
        best = snapshot["checkpoint"] if snapshot else None
        published_best, pending_recovery = None, False
        if best:
            try:
                published_best = _load_checkpoint(run_dir / name / "best.pt", manifest)
                pending_recovery = (published_best["epoch"] != last["best_epoch"]
                                    or published_best["validation_rmse"] != last["best_validation_rmse"])
            except (RuntimeError, OSError, KeyError):
                pending_recovery = True
        models[name] = {"completed_epochs": last["epoch"], "complete": last["complete"],
                        "best_epoch": last["best_epoch"], "best_validation_rmse": last["best_validation_rmse"],
                        "best_validation_mae": best.get("validation_mae") if best else None,
                        "last_checkpoint_sha256": sha256_file(run_dir / name / "last.pt"),
                        "best_checkpoint_sha256": sha256_file(run_dir / name / "best.pt") if best and not pending_recovery else None,
                        "best_artifact_pending_recovery": pending_recovery,
                        "parameters": manifest["models"][name]["trainable_parameters"]}
    result = {"version": VERSION, "status": status, "models": models,
              "comparison_complete": all(row["complete"] for row in models.values()),
              "winner_declared": False, "statistical_tests_performed": False,
              "charged_seconds": ledger["charged_seconds"], "limit_seconds": ledger["limit_seconds"],
              "budget_remaining_seconds": max(0.0, ledger["limit_seconds"] - ledger["charged_seconds"]),
              "budget_overshoot_seconds": max(0.0, ledger["charged_seconds"] - ledger["limit_seconds"]),
              "scope": "one-seed development validation reused for stopping; no p-values or R3",
              "calibration_scored": False, "evaluation_scored": False,
              "E2_results_changed": False, "source_hashes": manifest["source_hashes"],
              "plan_sha256": manifest["plan_sha256"], "data_fingerprint": manifest["data_fingerprint"]}
    _atomic_json(run_dir / "summary.json", result)
    return result


def run_development(run_dir, npz_path, expected_sha256, *, resume=False, device="auto",
                    stop_after_epochs=None, clock=time.monotonic):
    """Interleave model-epochs, checkpoint each completed batch and charge all time.

    stop_after_epochs is an execution pause after that many model-epochs during
    this invocation. It never changes the recipe, fingerprints or stopping rule.
    A hard-killed session is charged conservatively on resume before any work.
    """
    if stop_after_epochs is not None and (type(stop_after_epochs) is not int or stop_after_epochs < 1):
        raise ValueError("stop_after_epochs must be a positive integer")
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError("Run directory is absent; prepare it first")
    with run_lock(run_dir):
        manifest, plan, data, runtime = _validate_resume(run_dir, npz_path, expected_sha256, device)
        previous = _read_json(run_dir / "state.json")
        if previous["status"] != "initialized" and not resume:
            raise RuntimeError("This run has already started; explicit resume is required")
        if previous["status"] == "completed":
            return _summary(run_dir, manifest, "completed")
        budget = _WallBudget(run_dir, _read_json(run_dir / "ledger.json"), clock, runtime["device_type"])
        status, completed_this_call, error = "running", 0, None
        checkpoints, models, optimizers, generators = {}, {}, {}, {}
        _append_event(run_dir, {"event": "session_started", "resume": resume, "runtime": runtime})
        _atomic_json(run_dir / "runtime.json", runtime)
        try:
            _configure_determinism(plan["seed"])
            training = plan["training"]
            for name in MODEL_NAMES:
                checkpoint = _load_checkpoint(run_dir / name / "last.pt", manifest)
                _recover_best(run_dir, name, checkpoint)
                _write_csv(run_dir / name / "curves.csv", checkpoint["history"])
                model = DevelopmentModel(name, plan["seed"]).to(runtime["device_type"])
                model.load_state_dict(checkpoint["model_state"], strict=True)
                optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                for state in optimizer.state.values():
                    for key, value in state.items():
                        if torch.is_tensor(value):
                            state[key] = value.to(runtime["device_type"])
                generator = torch.Generator()
                generator.set_state(checkpoint["dataloader_generator_state"].cpu())
                checkpoints[name], models[name], optimizers[name], generators[name] = checkpoint, model, optimizer, generator
            train_x = torch.from_numpy(data["train_X"]).to(runtime["device_type"])
            train_y = torch.from_numpy(data["train_y_scaled"]).to(runtime["device_type"])
            val_x = torch.from_numpy(data["validation_X"]).to(runtime["device_type"])
            batch = training["batch_size"]
            while not all(x["complete"] for x in checkpoints.values()):
                order = sorted(MODEL_NAMES, key=lambda name: (checkpoints[name]["epoch"], MODEL_NAMES.index(name)))
                for name in order:
                    cp, model, optimizer, generator = checkpoints[name], models[name], optimizers[name], generators[name]
                    if cp["complete"]:
                        continue
                    if budget.tick() <= 0:
                        status = "budget_exhausted"
                        break
                    _restore_rng(cp["rng"])
                    if cp["active_permutation"] is None:
                        cp["active_permutation"] = torch.randperm(len(train_x), generator=generator)
                        cp["dataloader_generator_state"] = generator.get_state()
                    while cp["batch_cursor"] < len(train_x):
                        if budget.tick() <= 0:
                            status = "budget_exhausted"
                            break
                        start = cp["batch_cursor"]
                        idx = cp["active_permutation"][start:start + batch].to(runtime["device_type"])
                        model.train()
                        optimizer.zero_grad(set_to_none=True)
                        loss = F.mse_loss(model(train_x[idx]), train_y[idx])
                        if not bool(torch.isfinite(loss)):
                            raise RuntimeError("Nonfinite training loss")
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_norm"], error_if_nonfinite=True)
                        optimizer.step()
                        cp["epoch_loss_sum"] += float(loss.detach()) * len(idx)
                        cp["epoch_examples"] += len(idx)
                        cp["batch_cursor"] += len(idx)
                        cp["model_state"], cp["optimizer_state"], cp["rng"] = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict()), _rng_state()
                        _save_checkpoint(run_dir / name / "last.pt", cp)
                        budget.tick()
                    if status == "budget_exhausted":
                        break
                    predictions = []
                    model.eval()
                    with torch.no_grad():
                        for first in range(0, len(val_x), batch):
                            if budget.tick() <= 0:
                                status = "budget_exhausted"
                                break
                            predictions.append(model(val_x[first:first + batch]).cpu().numpy())
                    if status == "budget_exhausted":
                        break
                    predicted = np.concatenate(predictions).astype(np.float64) * data["scaler"]["target_scale"]
                    residual = predicted - data["validation_y_original"]
                    rmse, mae = float(np.sqrt(np.mean(residual ** 2))), float(np.mean(np.abs(residual)))
                    if not math.isfinite(rmse) or not math.isfinite(mae):
                        raise RuntimeError("Nonfinite development validation metric")
                    cp["epoch"] += 1
                    row = {"epoch": cp["epoch"], "training_batch_MSE_scaled": cp["epoch_loss_sum"] / cp["epoch_examples"],
                           "validation_RMSE_original": rmse, "validation_MAE_original": mae}
                    cp["history"].append(row)
                    improved = cp["best_validation_rmse"] is None or rmse < cp["best_validation_rmse"]
                    cp["no_improvement"] = 0 if improved else cp["no_improvement"] + 1
                    if improved:
                        cp["best_validation_rmse"], cp["best_epoch"] = rmse, cp["epoch"]
                        cp["validation_rmse"], cp["validation_mae"] = rmse, mae
                    cp["active_permutation"], cp["batch_cursor"] = None, 0
                    cp["epoch_loss_sum"], cp["epoch_examples"] = 0.0, 0
                    cp["complete"] = cp["epoch"] >= training["max_epochs"] or cp["no_improvement"] >= training["patience"]
                    cp["rng"] = _rng_state()
                    prediction_rows = [{"date": str(day), "target_sigma7": float(target), "prediction_sigma7": float(prediction)}
                                       for day, target, prediction in zip(data["validation_dates"], data["validation_y_original"], predicted)]
                    if improved:
                        best_copy = deepcopy({key: value for key, value in cp.items() if key != "best_snapshot"})
                        cp["best_snapshot"] = {"checkpoint": best_copy, "predictions": prediction_rows}
                    _save_checkpoint(run_dir / name / "last.pt", cp)
                    if improved:
                        _save_checkpoint(run_dir / name / "best.pt", cp["best_snapshot"]["checkpoint"])
                        _write_csv(run_dir / name / "validation_predictions.csv",
                                   prediction_rows)
                    _write_csv(run_dir / name / "curves.csv", cp["history"])
                    completed_this_call += 1
                    _append_event(run_dir, {"event": "epoch_completed", "model": name, **row,
                                           "best_epoch": cp["best_epoch"], "complete": cp["complete"]})
                    print(json.dumps({"event": "epoch_completed", "model": name, **row,
                                      "remaining_wall_seconds": budget.tick()}, allow_nan=False), flush=True)
                    _atomic_json(run_dir / "state.json", {"status": "running", "models": {n: {"epoch": c["epoch"], "complete": c["complete"]} for n, c in checkpoints.items()}})
                    if stop_after_epochs is not None and completed_this_call >= stop_after_epochs:
                        status = "paused"
                        break
                if status in ("paused", "budget_exhausted"):
                    break
            if all(x["complete"] for x in checkpoints.values()):
                status = "completed"
        except BaseException as caught:
            status, error = ("interrupted" if isinstance(caught, (KeyboardInterrupt, SystemExit)) else "failed"), caught
            _append_event(run_dir, {"event": "session_error", "kind": type(caught).__name__, "message": str(caught)})
        finally:
            budget.close(status)
            _atomic_json(run_dir / "state.json", {"status": status, "models": {n: {"epoch": c["epoch"], "complete": c["complete"]} for n, c in checkpoints.items()}})
            _append_event(run_dir, {"event": "session_closed", "status": status})
        summary = _summary(run_dir, manifest, status)
        if error is not None:
            raise error
        return summary
