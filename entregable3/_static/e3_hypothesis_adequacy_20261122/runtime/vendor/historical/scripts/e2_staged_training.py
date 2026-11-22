"""Fold-local neural optimization stages; no data loader or market CLI.

The caller owns chronology, train-only transforms and fresh model construction.
This module accepts explicit artificial/admissible tensors, initializes a new
AdamW per stage, and never retries a failed stage. Runtime limits are checked at
boundaries, not enforced as a process sandbox; a profiling supervisor can impose
hard wall/RSS limits independently.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

VERSION = "e2-staged-training-v1"


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int
    seed: int
    warmup_fraction: float = 0.0
    gradient_clip: float = 1.0
    wall_limit_seconds: float | None = None
    memory_limit_bytes: int | None = None

    def __post_init__(self):
        def finite(value):
            return type(value) in (int, float) and math.isfinite(value)
        if not finite(self.learning_rate) or not 1e-5 <= self.learning_rate <= .003:
            raise ValueError("Learning rate outside registered [1e-5,.003]")
        if (not finite(self.weight_decay) or not (self.weight_decay == 0
                or 1e-6 <= self.weight_decay <= .01)):
            raise ValueError("Weight decay outside registered mixture")
        if type(self.batch_size) is not int or self.batch_size not in (16, 32, 64):
            raise ValueError("Batch size must be16,32 or64")
        if type(self.max_epochs) is not int or not 1 <= self.max_epochs <= 200:
            raise ValueError("Epoch cap must be within1..200")
        if type(self.patience) is not int or self.patience < 1:
            raise ValueError("Positive patience required")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Nonnegative integer seed required")
        if not finite(self.warmup_fraction) or self.warmup_fraction not in (0, .05, .1):
            raise ValueError("Warmup must be0,.05 or.1")
        if not finite(self.gradient_clip) or self.gradient_clip != 1:
            raise ValueError("Registered global gradient clipping norm is1")
        if self.wall_limit_seconds is not None and (not finite(self.wall_limit_seconds)
                                                      or self.wall_limit_seconds <= 0):
            raise ValueError("Wall limit must be finite and positive")
        if self.memory_limit_bytes is not None and (type(self.memory_limit_bytes) is not int
                                                       or self.memory_limit_bytes < 1):
            raise ValueError("Memory limit must be a positive byte count")


@dataclass(frozen=True)
class StageResult:
    state_dict: dict[str, Tensor]
    best_epoch: int
    epochs_completed: int
    optimizer_steps: int
    elapsed_seconds: float
    logs: tuple[dict[str, Any], ...]
    initial_state_sha256: str
    selected_state_sha256: str
    optimizer_initial_state_entries: int
    stop_reason: str
    selection_metric: str
    fixed_epochs: int | None
    scheduler_total_steps: int
    max_observed_rss_bytes: int
    seed: int

    def to_dict(self):
        # Tensor checkpoints remain explicit and are never silently JSON-encoded.
        return {"version": VERSION, **{key: value for key, value in self.__dict__.items()
                                      if key != "state_dict"}}


@dataclass(frozen=True)
class SAEResult:
    model: nn.Module
    phases: dict[str, StageResult]
    auxiliary_scale: float
    fold_id: str

    def to_dict(self):
        return {"fold_id": self.fold_id, "auxiliary_scale": self.auxiliary_scale,
                "phases": {key: value.to_dict() for key, value in self.phases.items()},
                "optimizer_steps": sum(value.optimizer_steps for value in self.phases.values()),
                "decoder_parameters_in_predictor": any("decoder" in key for key in self.model.state_dict())}


class TrainingFailure(RuntimeError):
    """Terminal failed-stage record. The caller records it and does not retry."""
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


class TrainingGeometryError(RuntimeError):
    error_code = "geometry"


class TrainingNumericalError(RuntimeError):
    error_code = "nonfinite"


class TrainingMissingGradientError(RuntimeError):
    error_code = "missing_gradient"


class TrainingCheckpointError(RuntimeError):
    error_code = "checkpoint"


def _failure_code(error):
    if isinstance(error, (TimeoutError, MemoryError, torch.OutOfMemoryError)):
        return "resource_limit"
    return getattr(error, "error_code", "runtime_error")


def state_sha256(state_or_model):
    state = state_or_model.state_dict() if isinstance(state_or_model, nn.Module) else state_or_model
    digest = sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def clone_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def batch_indices(size, batch_size, *, generator):
    """Shuffle every row once; redistribute a singleton tail without dropping it."""
    if type(size) is not int or size < 2 or type(batch_size) is not int or batch_size < 2:
        raise ValueError("Training requires at least two examples and batch capacity2")
    order = torch.randperm(size, generator=generator)
    chunks = list(order.split(batch_size))
    if len(chunks[-1]) == 1:
        if len(chunks) < 2:
            raise ValueError("Singleton training set is not admissible")
        previous, tail = chunks[-2], chunks[-1]
        chunks[-2:] = [previous[:-1], torch.cat((previous[-1:], tail))]
    if any(len(chunk) < 2 or len(chunk) > batch_size for chunk in chunks):
        raise RuntimeError("Batch redistribution violated cardinality")
    return tuple(chunks)


def _seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rss_bytes():
    try:
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


def _pairs(x, y, *, supervised, training):
    if (not isinstance(x, Tensor) or x.dtype != torch.float32 or x.ndim != 3
            or x.shape[1] != 7 or x.shape[2] < 1 or len(x) < (2 if training else 1)):
        raise ValueError("Expected FP32 nonempty seven-step training/validation tensors")
    if not isinstance(y, Tensor) or y.dtype != torch.float32 or y.ndim < 1 or len(y) != len(x):
        raise ValueError("Targets must be FP32 and align with input examples")
    if supervised:
        if x.shape[2] != 5 or y.ndim != 1 or bool((y < 0).any()):
            raise ValueError("Supervised D7 inputs need5 channels and nonnegative scalar targets")
    elif y.ndim != 3 or tuple(y.shape[1:]) != (7, 5):
        raise ValueError("Auxiliary reconstruction targets must retain seven by five coordinates")
    if x.device != y.device or not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
        raise ValueError("Inputs and targets must share a device and contain only finite values")
    return x.detach().clone(), y.detach().clone()


def optimizer_groups(model, weight_decay):
    """Decay matrix weights, excluding biases and normalization parameters."""
    decayed, exempt = [], []
    normalization = set()
    for _, module in model.named_modules():
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
                               nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm1d,
                               nn.InstanceNorm2d, nn.InstanceNorm3d)):
            normalization.update(id(parameter) for parameter in module.parameters(recurse=False))
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith("bias") or id(parameter) in normalization:
            exempt.append(parameter)
        else:
            decayed.append(parameter)
    if not decayed and not exempt:
        raise ValueError("A stage needs trainable parameters")
    return [{"params": decayed, "weight_decay": weight_decay},
            {"params": exempt, "weight_decay": 0.0}]


def learning_rate_multiplier(step, total_steps, warmup_fraction):
    if (type(step) is not int or type(total_steps) is not int or total_steps < 1
            or not 0 <= step <= total_steps):
        raise ValueError("Step must lie inside the declared scheduler horizon")
    if type(warmup_fraction) not in (int, float) or warmup_fraction not in (0, .05, .1):
        raise ValueError("Warmup must match the registered domain")
    # Standard after-optimizer scheduling: the LR used by the last admitted
    # update is positive; zero is the boundary immediately after the budget.
    # In particular, a one-batch, one-epoch profile still changes weights.
    if step == total_steps:
        return 0.0
    warmup = int(round(total_steps * warmup_fraction))
    if warmup and step < warmup:
        return (step + 1) / warmup
    remaining = total_steps - warmup
    progress = (step - warmup) / remaining
    return .5 * (1 + math.cos(math.pi * progress))


def _mse(model, x, y, batch_size):
    model.eval()
    total, elements = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            prediction = model(x[start:start+batch_size])
            truth = y[start:start+batch_size]
            if prediction.shape != truth.shape:
                raise TrainingGeometryError("Prediction geometry failed during evaluation")
            if not bool(torch.isfinite(prediction).all()):
                raise TrainingNumericalError("Nonfinite prediction during evaluation")
            total += float(F.mse_loss(prediction, truth, reduction="sum"))
            elements += truth.numel()
    result = total / elements
    if not math.isfinite(result):
        raise TrainingNumericalError("Nonfinite evaluation loss")
    return result


def _fit(model, x_train, y_train, x_val, y_val, target_scale, config, *,
         fixed_epochs, supervised):
    if not isinstance(model, nn.Module) or not isinstance(config, TrainConfig):
        raise TypeError("Explicit module and validated TrainConfig required")
    start = time.perf_counter()
    if type(target_scale) not in (float, int) or not math.isfinite(target_scale) or target_scale <= 0:
        raise ValueError("Training-only target scale must be positive and finite")
    if fixed_epochs is not None:
        if type(fixed_epochs) is not int or not 1 <= fixed_epochs <= config.max_epochs:
            raise ValueError("Fixed refit epochs must lie within the registered epoch cap")
        if x_val is not None or y_val is not None:
            raise ValueError("Fixed refit accepts no validation inputs or labels")
    elif x_val is None or y_val is None:
        raise ValueError("Selection stage requires an explicit validation pair")
    x_train, y_train = _pairs(x_train, y_train, supervised=supervised, training=True)
    if fixed_epochs is None:
        x_val, y_val = _pairs(x_val, y_val, supervised=supervised, training=False)
        if x_val.device != x_train.device:
            raise ValueError("Train and validation devices differ")
    parameters = list(model.parameters())
    if any(parameter.dtype != torch.float32 or parameter.device != x_train.device for parameter in parameters):
        raise ValueError("Model and inputs must use the same device and FP32 parameters")
    _seed(config.seed)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    # Number of batches is independent of the permutation; a singleton tail is
    # redistributed, never an additional dropped or undersized batch.
    batches_per_epoch = len(batch_indices(len(x_train), config.batch_size,
                                          generator=torch.Generator().manual_seed(config.seed)))
    scheduler_steps = config.max_epochs * batches_per_epoch
    initial_hash = state_sha256(model)
    logs, steps, best_epoch, best_metric, best_state = [], 0, 0, math.inf, None
    max_rss = 0
    optimizer = None

    def limits():
        nonlocal max_rss
        rss = _rss_bytes()
        max_rss = max(max_rss, rss)
        if config.wall_limit_seconds is not None and time.perf_counter() - start > config.wall_limit_seconds:
            raise TimeoutError("Stage wall limit exceeded at a checked boundary")
        if config.memory_limit_bytes is not None and rss > config.memory_limit_bytes:
            raise MemoryError("Stage process RSS limit exceeded at a checked boundary")

    try:
        limits()
        optimizer = torch.optim.AdamW(optimizer_groups(model, config.weight_decay),
                                     lr=config.learning_rate, betas=(.9, .999), eps=1e-8)
        initial_entries = len(optimizer.state)
        epoch_count = fixed_epochs if fixed_epochs is not None else config.max_epochs
        stale, stop_reason = 0, "fixed_epochs" if fixed_epochs is not None else "epoch_cap"
        for epoch in range(1, epoch_count + 1):
            limits()
            model.train()
            batch_sizes, rate_first, rate_last = [], None, None
            for indices in batch_indices(len(x_train), config.batch_size, generator=generator):
                limits()
                indices = indices.to(x_train.device)
                rate = config.learning_rate * learning_rate_multiplier(steps, scheduler_steps, config.warmup_fraction)
                for group in optimizer.param_groups:
                    group["lr"] = rate
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x_train[indices])
                truth = y_train[indices]
                if prediction.shape != truth.shape:
                    raise TrainingGeometryError("Prediction geometry failed during training")
                if not bool(torch.isfinite(prediction).all()):
                    raise TrainingNumericalError("Nonfinite prediction during training")
                loss = F.mse_loss(prediction, truth)
                if not bool(torch.isfinite(loss)):
                    raise TrainingNumericalError("Nonfinite objective")
                loss.backward()
                trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
                if any(parameter.grad is None for parameter in trainable):
                    raise TrainingMissingGradientError("A trainable parameter has no gradient")
                if any(not bool(torch.isfinite(parameter.grad).all()) for parameter in trainable):
                    raise TrainingNumericalError("Nonfinite parameter gradient")
                # Inspect the returned aggregate norm explicitly so numerical
                # overflow is classified without swallowing unrelated runtime
                # failures from the clipping backend. The finite-success math
                # is identical to error_if_nonfinite=True.
                gradient_norm = nn.utils.clip_grad_norm_(trainable, config.gradient_clip,
                                                        error_if_nonfinite=False)
                if not bool(torch.isfinite(gradient_norm)):
                    raise TrainingNumericalError("Nonfinite aggregate gradient norm")
                optimizer.step()
                steps += 1
                if any(not bool(torch.isfinite(parameter).all()) for parameter in trainable):
                    raise TrainingNumericalError("Nonfinite updated parameter")
                batch_sizes.append(len(indices))
                rate_first = rate if rate_first is None else rate_first
                rate_last = rate
                limits()
            train_mse = _mse(model, x_train, y_train, config.batch_size)
            score = None
            if fixed_epochs is None:
                val_mse = _mse(model, x_val, y_val, config.batch_size)
                score = math.sqrt(val_mse) * target_scale if supervised else val_mse
                if not math.isfinite(score):
                    raise TrainingNumericalError("Nonfinite original-unit selection score")
                if score < best_metric:  # Strict: first checkpoint wins exact ties.
                    best_metric, best_epoch, stale = score, epoch, 0
                    best_state = clone_state(model)
                else:
                    stale += 1
            else:
                best_epoch = epoch
            row = {"epoch": epoch, "train_mse_scaled": train_mse,
                   "train_rmse_original": math.sqrt(train_mse)*target_scale if supervised else None,
                   "validation_score": score, "batch_sizes": batch_sizes,
                   "examples_seen": sum(batch_sizes), "learning_rate_first": rate_first,
                   "learning_rate_last": rate_last, "learning_rate_after_epoch":
                       config.learning_rate * learning_rate_multiplier(steps, scheduler_steps, config.warmup_fraction),
                   "optimizer_steps_total": steps}
            logs.append(row)
            limits()
            if fixed_epochs is None and stale >= config.patience:
                stop_reason = "patience"
                break
        if fixed_epochs is not None:
            best_state = clone_state(model)
        if best_state is None:
            raise TrainingCheckpointError("No admissible checkpoint completed")
        model.load_state_dict(best_state)
        model.eval()
        selected_hash = state_sha256(best_state)
        limits()
        return StageResult(state_dict=best_state, best_epoch=best_epoch,
                           epochs_completed=len(logs), optimizer_steps=steps,
                           elapsed_seconds=time.perf_counter()-start, logs=tuple(logs),
                           initial_state_sha256=initial_hash, selected_state_sha256=selected_hash,
                           optimizer_initial_state_entries=initial_entries, stop_reason=stop_reason,
                           selection_metric="rmse_original" if supervised else "reconstruction_mse",
                           fixed_epochs=fixed_epochs, scheduler_total_steps=scheduler_steps,
                           max_observed_rss_bytes=max_rss, seed=config.seed)
    except Exception as error:
        receipt = {"version": VERSION, "status": "failed", "error_type": type(error).__name__,
                   "error_code": _failure_code(error),
                   "message": str(error), "optimizer_steps": steps, "epochs_completed": len(logs),
                   "elapsed_seconds": time.perf_counter()-start, "initial_state_sha256": initial_hash,
                   "seed": config.seed, "config": asdict(config), "retry_performed": False,
                   "logs": logs, "max_observed_rss_bytes": max_rss}
        raise TrainingFailure(str(error), receipt) from error
    finally:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)


def fit_supervised(model, x_train, y_train_scaled, x_val, y_val_scaled, target_scale,
                   config, *, fixed_epochs=None):
    return _fit(model, x_train, y_train_scaled, x_val, y_val_scaled, target_scale,
                config, fixed_epochs=fixed_epochs, supervised=True)


def fit_reconstruction(model, x_train, reconstruction_train, x_val, reconstruction_val,
                       config, *, fixed_epochs=None):
    return _fit(model, x_train, reconstruction_train, x_val, reconstruction_val, 1.0,
                config, fixed_epochs=fixed_epochs, supervised=False)


def fit_sae(nas_config, x_train, y_train_scaled, x_val, y_val_scaled, target_scale,
            config, *, fold_id, fixed_epochs=None):
    """Two auxiliary fits and a transferred supervised fit, freshly instantiated.

    fixed_epochs is either None for inner selection or exactly the three
    previously selected counts {ae1,ae2,supervised}. No validation pair is
    accepted in fixed mode. The same replicate seed is reset for every stage;
    batch shuffling uses a new independent local generator for each call.
    """
    try:
        from .e2_nas_autoencoder import build_auxiliary_stages
    except ImportError:
        from e2_nas_autoencoder import build_auxiliary_stages
    if not isinstance(config, TrainConfig):
        raise TypeError("Explicit TrainConfig required")
    if type(target_scale) not in (int, float) or not math.isfinite(target_scale) or target_scale <= 0:
        raise ValueError("Training-only target scale must be positive and finite")
    if fixed_epochs is not None:
        if not isinstance(fixed_epochs, dict) or set(fixed_epochs) != {"ae1", "ae2", "supervised"}:
            raise ValueError("Fixed SAE refit needs all three registered stage epoch counts")
        if x_val is not None or y_val_scaled is not None:
            raise ValueError("Fixed SAE refit accepts no validation inputs or labels")
        for count in fixed_epochs.values():
            if type(count) is not int or not 1 <= count <= config.max_epochs:
                raise ValueError("SAE stage epoch count exceeds configured horizon")
    x_train, y_train_scaled = _pairs(x_train, y_train_scaled, supervised=True, training=True)
    if fixed_epochs is None:
        x_val, y_val_scaled = _pairs(x_val, y_val_scaled, supervised=True, training=False)
    _seed(config.seed)
    manager = build_auxiliary_stages(nas_config, x_train, fold_id=fold_id)
    phases = {}
    ax, ay = manager.phase1_data()
    validation_target = manager.reconstruction_target(x_val) if fixed_epochs is None else None
    def stage(name, function, *args, **kwargs):
        try:
            result = function(*args, **kwargs)
        except TrainingFailure as error:
            receipt = {**error.receipt, "sae_phase": name, "fold_id": fold_id,
                       "completed_phases": {key: value.to_dict() for key, value in phases.items()}}
            raise TrainingFailure(str(error), receipt) from error
        phases[name] = result
        return result
    stage("ae1", fit_reconstruction, manager.ae1, ax, ay, x_val, validation_target, config,
          fixed_epochs=fixed_epochs["ae1"] if fixed_epochs else None)
    bx, by = manager.prepare_phase2()
    val_encoded = manager.phase2_inputs(x_val) if fixed_epochs is None else None
    stage("ae2", fit_reconstruction, manager.ae2, bx, by, val_encoded, validation_target, config,
          fixed_epochs=fixed_epochs["ae2"] if fixed_epochs else None)
    # Head initialization must not depend on how many dropout calls happened
    # during the auxiliary selection stages.
    _seed(config.seed)
    predictor = manager.transfer_predictor()
    stage("supervised", fit_supervised, predictor, x_train, y_train_scaled,
          x_val, y_val_scaled, target_scale, config,
          fixed_epochs=fixed_epochs["supervised"] if fixed_epochs else None)
    return SAEResult(predictor, phases, float(manager.scale), fold_id)
