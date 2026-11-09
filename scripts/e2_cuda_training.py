"""Explicit CUDA backend for the frozen E2 FP32 training semantics.

The historical CPU modules remain unchanged. This module moves admitted
tensors and fresh models to an explicitly selected CUDA device. It preserves
MSE, AdamW groups, cosine horizon, singleton redistribution, clipping, strict
early stopping and all three SAE stages. CUDA is required unless a caller
explicitly enables the CPU-only testing route; production CLIs must not enable
that route. No market loader, HPO selection or automatic retry is provided.

Resource checks occur at synchronized boundaries. They are observations, not
hard allocation barriers. CUDA peaks are measured since the caller's last
reset, so a caller can retain construction and earlier SAE-stage peaks.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
import os
import time

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from . import e2_staged_training as cpu
except ImportError:
    import e2_staged_training as cpu

VERSION = "e2-explicit-cuda-training-v1"
TrainConfig = cpu.TrainConfig
TrainingFailure = cpu.TrainingFailure
SAEResult = cpu.SAEResult


class CudaUnavailableError(RuntimeError):
    error_code = "cuda_unavailable"


def validate_cuda_device(device="cuda:0"):
    """Resolve one real CUDA device, with no CPU or accelerator substitution."""
    if not isinstance(device, (str, torch.device)):
        raise TypeError("Pass an explicit CUDA device string or torch.device")
    try:
        selected = torch.device(device)
    except (RuntimeError, ValueError) as error:
        raise ValueError("A valid explicit CUDA device is required") from error
    if selected.type != "cuda":
        raise ValueError("The CUDA backend requires a CUDA device")
    if not torch.cuda.is_available():
        raise CudaUnavailableError("CUDA is not available; no CPU fallback was performed")
    index = 0 if selected.index is None else selected.index
    if not 0 <= index < torch.cuda.device_count():
        raise CudaUnavailableError("The requested CUDA device is not present")
    return torch.device("cuda", index)


def _resolve_device(device, allow_cpu_for_tests):
    if type(allow_cpu_for_tests) is not bool:
        raise TypeError("The CPU testing flag must be explicit boolean")
    try:
        selected = torch.device(device)
    except (TypeError, RuntimeError, ValueError) as error:
        raise ValueError("A valid explicit device is required") from error
    if selected.type == "cpu" and allow_cpu_for_tests:
        if selected.index not in (None, 0):
            raise ValueError("Only the ordinary CPU device is admitted for tests")
        return torch.device("cpu")
    return validate_cuda_device(device)


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate_gpu_limit(value):
    if value is not None and (type(value) is not int or value < 1):
        raise ValueError("CUDA allocated-memory threshold must be a positive byte count")


def _validate_model(model):
    if not isinstance(model, nn.Module):
        raise TypeError("An explicit fresh model is required")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("All model parameters must remain FP32")


@contextmanager
def _execution_policy(device, deterministic):
    """Set strict FP32 CUDA policy for this sequential process, then restore it."""
    if type(deterministic) is not bool:
        raise TypeError("Deterministic execution must be an explicit boolean")
    if device.type == "cpu":
        # This route exists only for parity tests. Keep the CPU algorithm flags
        # exactly as the historical trainer sees them.
        with torch.autocast(device_type="cpu", enabled=False):
            yield _runtime(device, deterministic)
        return
    if deterministic and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise ValueError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization")
    old = {"deterministic": torch.are_deterministic_algorithms_enabled(),
           "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
           "precision": torch.get_float32_matmul_precision(),
           "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
           "cudnn_tf32": torch.backends.cudnn.allow_tf32,
           "cudnn_benchmark": torch.backends.cudnn.benchmark,
           "cudnn_deterministic": torch.backends.cudnn.deterministic}
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=False)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
        with torch.cuda.device(device), torch.autocast(device_type="cuda", enabled=False):
            yield _runtime(device, deterministic)
    finally:
        torch.use_deterministic_algorithms(old["deterministic"], warn_only=old["warn_only"])
        # Setting allow_tf32 can itself change the precision selector in
        # PyTorch (medium/True becomes high/True). Restore that flag first,
        # then restore the exact selector as the last precision operation.
        torch.backends.cuda.matmul.allow_tf32 = old["matmul_tf32"]
        torch.backends.cudnn.allow_tf32 = old["cudnn_tf32"]
        torch.backends.cudnn.benchmark = old["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = old["cudnn_deterministic"]
        torch.set_float32_matmul_precision(old["precision"])


def _runtime(device, deterministic):
    return {"backend_version": VERSION, "device": str(device),
            "test_only_cpu_execution": device.type == "cpu", "torch": str(torch.__version__),
            "precision": "float32", "amp_enabled": False,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_tf32_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_tf32_enabled": bool(torch.backends.cudnn.allow_tf32),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "deterministic_requested": deterministic,
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "cuda_build": torch.version.cuda,
            "cuda_peak_scope": "since_callers_last_reset" if device.type == "cuda" else "not_applicable",
            "same_seed_cross_device_bitwise_equivalence_claimed": False}


@dataclass(frozen=True)
class CudaStageResult(cpu.StageResult):
    device: str
    runtime: dict
    max_cuda_memory_allocated_bytes: int
    max_cuda_memory_reserved_bytes: int
    gpu_memory_limit_bytes: int | None
    cuda_synchronized_timing: bool

    def to_dict(self):
        return {"version": VERSION, **{key: value for key, value in self.__dict__.items()
                                      if key != "state_dict"}}


def _mse(model, x, y, batch_size):
    # The historical evaluator supports CUDA tensors already. Each conversion
    # to float synchronizes its reduction; the enclosing checks synchronize
    # the whole selected device before timing and resource observations.
    return cpu._mse(model, x, y, batch_size)


def _fit(model, x_train, y_train, x_val, y_val, target_scale, config, *,
         fixed_epochs, supervised, device, allow_cpu_for_tests,
         gpu_memory_limit_bytes, deterministic):
    selected = _resolve_device(device, allow_cpu_for_tests)
    _validate_gpu_limit(gpu_memory_limit_bytes)
    _validate_model(model)
    if not isinstance(config, TrainConfig):
        raise TypeError("A validated historical TrainConfig is required")
    if type(target_scale) not in (float, int) or not math.isfinite(target_scale) or target_scale <= 0:
        raise ValueError("Training-only target scale must be positive and finite")
    if fixed_epochs is not None:
        if type(fixed_epochs) is not int or not 1 <= fixed_epochs <= config.max_epochs:
            raise ValueError("Fixed refit epochs must lie within the registered epoch cap")
        if x_val is not None or y_val is not None:
            raise ValueError("Fixed refit accepts no validation inputs or labels")
    elif x_val is None or y_val is None:
        raise ValueError("Selection stage requires an explicit validation pair")
    x_train, y_train = cpu._pairs(x_train, y_train, supervised=supervised, training=True)
    if fixed_epochs is None:
        x_val, y_val = cpu._pairs(x_val, y_val, supervised=supervised, training=False)
        if x_val.device != x_train.device:
            raise ValueError("Train and validation devices differ")
    with _execution_policy(selected, deterministic) as runtime:
        _synchronize(selected)
        start = time.perf_counter()
        initial_hash = cpu.state_sha256(model)
        logs, steps, best_epoch, best_metric, best_state = [], 0, 0, math.inf, None
        max_rss = max_allocated = max_reserved = 0
        optimizer = None
        step_in_progress = False

        def limits():
            nonlocal max_rss, max_allocated, max_reserved
            _synchronize(selected)
            max_rss = max(max_rss, cpu._rss_bytes())
            if selected.type == "cuda":
                max_allocated = max(max_allocated, int(torch.cuda.max_memory_allocated(selected)))
                max_reserved = max(max_reserved, int(torch.cuda.max_memory_reserved(selected)))
            if config.wall_limit_seconds is not None and time.perf_counter()-start > config.wall_limit_seconds:
                raise TimeoutError("Stage wall limit exceeded at a synchronized checked boundary")
            if config.memory_limit_bytes is not None and max_rss > config.memory_limit_bytes:
                raise MemoryError("Stage process RSS limit exceeded at a checked boundary")
            if gpu_memory_limit_bytes is not None and max_allocated > gpu_memory_limit_bytes:
                raise MemoryError("CUDA allocated-memory threshold exceeded at a checked boundary")

        try:
            model.to(selected)
            x_train, y_train = x_train.to(selected), y_train.to(selected)
            if fixed_epochs is None:
                x_val, y_val = x_val.to(selected), y_val.to(selected)
            cpu._seed(config.seed)
            generator = torch.Generator(device="cpu").manual_seed(config.seed)
            batches = len(cpu.batch_indices(len(x_train), config.batch_size,
                         generator=torch.Generator().manual_seed(config.seed)))
            scheduler_steps = config.max_epochs*batches
            limits()
            optimizer = torch.optim.AdamW(cpu.optimizer_groups(model, config.weight_decay),
                                          lr=config.learning_rate, betas=(.9, .999), eps=1e-8)
            initial_entries = len(optimizer.state)
            epoch_count = fixed_epochs if fixed_epochs is not None else config.max_epochs
            stale, stop_reason = 0, "fixed_epochs" if fixed_epochs is not None else "epoch_cap"
            for epoch in range(1, epoch_count+1):
                limits()
                model.train()
                batch_sizes, rate_first, rate_last = [], None, None
                for indices in cpu.batch_indices(len(x_train), config.batch_size, generator=generator):
                    limits()
                    indices = indices.to(selected)
                    rate = config.learning_rate*cpu.learning_rate_multiplier(
                        steps, scheduler_steps, config.warmup_fraction)
                    for group in optimizer.param_groups:
                        group["lr"] = rate
                    optimizer.zero_grad(set_to_none=True)
                    prediction, truth = model(x_train[indices]), y_train[indices]
                    if prediction.shape != truth.shape:
                        raise cpu.TrainingGeometryError("Prediction geometry failed during training")
                    if not bool(torch.isfinite(prediction).all()):
                        raise cpu.TrainingNumericalError("Nonfinite prediction during training")
                    loss = F.mse_loss(prediction, truth)
                    if not bool(torch.isfinite(loss)):
                        raise cpu.TrainingNumericalError("Nonfinite objective")
                    loss.backward()
                    trainable = [p for p in model.parameters() if p.requires_grad]
                    if any(p.grad is None for p in trainable):
                        raise cpu.TrainingMissingGradientError("A trainable parameter has no gradient")
                    if any(not bool(torch.isfinite(p.grad).all()) for p in trainable):
                        raise cpu.TrainingNumericalError("Nonfinite parameter gradient")
                    norm = nn.utils.clip_grad_norm_(trainable, config.gradient_clip, error_if_nonfinite=False)
                    if not bool(torch.isfinite(norm)):
                        raise cpu.TrainingNumericalError("Nonfinite aggregate gradient norm")
                    step_in_progress = True
                    optimizer.step()
                    _synchronize(selected)
                    steps += 1
                    step_in_progress = False
                    if any(not bool(torch.isfinite(p).all()) for p in trainable):
                        raise cpu.TrainingNumericalError("Nonfinite updated parameter")
                    batch_sizes.append(len(indices))
                    rate_first = rate if rate_first is None else rate_first
                    rate_last = rate
                    limits()
                train_mse = _mse(model, x_train, y_train, config.batch_size)
                score = None
                if fixed_epochs is None:
                    val_mse = _mse(model, x_val, y_val, config.batch_size)
                    score = math.sqrt(val_mse)*target_scale if supervised else val_mse
                    if not math.isfinite(score):
                        raise cpu.TrainingNumericalError("Nonfinite original-unit selection score")
                    if score < best_metric:
                        best_metric, best_epoch, stale = score, epoch, 0
                        best_state = cpu.clone_state(model)
                    else:
                        stale += 1
                else:
                    best_epoch = epoch
                logs.append({"epoch": epoch, "train_mse_scaled": train_mse,
                             "train_rmse_original": math.sqrt(train_mse)*target_scale if supervised else None,
                             "validation_score": score, "batch_sizes": batch_sizes,
                             "examples_seen": sum(batch_sizes), "learning_rate_first": rate_first,
                             "learning_rate_last": rate_last, "learning_rate_after_epoch":
                                 config.learning_rate*cpu.learning_rate_multiplier(
                                     steps, scheduler_steps, config.warmup_fraction),
                             "optimizer_steps_total": steps})
                limits()
                if fixed_epochs is None and stale >= config.patience:
                    stop_reason = "patience"
                    break
            if fixed_epochs is not None:
                best_state = cpu.clone_state(model)
            if best_state is None:
                raise cpu.TrainingCheckpointError("No admissible checkpoint completed")
            model.load_state_dict(best_state)
            model.eval()
            selected_hash = cpu.state_sha256(best_state)
            limits()
            return CudaStageResult(
                state_dict=best_state, best_epoch=best_epoch, epochs_completed=len(logs),
                optimizer_steps=steps, elapsed_seconds=time.perf_counter()-start, logs=tuple(logs),
                initial_state_sha256=initial_hash, selected_state_sha256=selected_hash,
                optimizer_initial_state_entries=initial_entries, stop_reason=stop_reason,
                selection_metric="rmse_original" if supervised else "reconstruction_mse",
                fixed_epochs=fixed_epochs, scheduler_total_steps=scheduler_steps,
                max_observed_rss_bytes=max_rss, seed=config.seed, device=str(selected),
                runtime=dict(runtime), max_cuda_memory_allocated_bytes=max_allocated,
                max_cuda_memory_reserved_bytes=max_reserved, gpu_memory_limit_bytes=gpu_memory_limit_bytes,
                cuda_synchronized_timing=selected.type == "cuda")
        except Exception as error:
            synchronization_error = None
            try:
                _synchronize(selected)
                if selected.type == "cuda":
                    max_allocated = max(max_allocated, int(torch.cuda.max_memory_allocated(selected)))
                    max_reserved = max(max_reserved, int(torch.cuda.max_memory_reserved(selected)))
            except Exception as sync_error:
                synchronization_error = type(sync_error).__name__
            receipt = {"version": VERSION, "status": "failed", "error_type": type(error).__name__,
                       "error_code": cpu._failure_code(error), "message": str(error),
                       "optimizer_steps": steps, "completed_optimizer_steps": steps,
                       "partial_optimizer_update_may_have_occurred": step_in_progress,
                       "epochs_completed": len(logs), "elapsed_seconds": time.perf_counter()-start,
                       "initial_state_sha256": initial_hash, "seed": config.seed, "config": asdict(config),
                       "retry_performed": False, "logs": logs, "max_observed_rss_bytes": max_rss,
                       "device": str(selected), "runtime": dict(runtime),
                       "max_cuda_memory_allocated_bytes": max_allocated,
                       "max_cuda_memory_reserved_bytes": max_reserved,
                       "gpu_memory_limit_bytes": gpu_memory_limit_bytes,
                       "cuda_synchronization_error_type": synchronization_error}
            raise TrainingFailure(str(error), receipt) from error
        finally:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)


def fit_supervised(model, x_train, y_train_scaled, x_val, y_val_scaled, target_scale,
                   config, *, device="cuda:0", fixed_epochs=None, allow_cpu_for_tests=False,
                   gpu_memory_limit_bytes=None, deterministic=True):
    return _fit(model, x_train, y_train_scaled, x_val, y_val_scaled, target_scale, config,
                fixed_epochs=fixed_epochs, supervised=True, device=device,
                allow_cpu_for_tests=allow_cpu_for_tests, gpu_memory_limit_bytes=gpu_memory_limit_bytes,
                deterministic=deterministic)


def fit_reconstruction(model, x_train, reconstruction_train, x_val, reconstruction_val,
                       config, *, device="cuda:0", fixed_epochs=None, allow_cpu_for_tests=False,
                       gpu_memory_limit_bytes=None, deterministic=True):
    return _fit(model, x_train, reconstruction_train, x_val, reconstruction_val, 1.0, config,
                fixed_epochs=fixed_epochs, supervised=False, device=device,
                allow_cpu_for_tests=allow_cpu_for_tests, gpu_memory_limit_bytes=gpu_memory_limit_bytes,
                deterministic=deterministic)


def fit_sae(nas_config, x_train, y_train_scaled, x_val, y_val_scaled, target_scale,
            config, *, fold_id, device="cuda:0", fixed_epochs=None, allow_cpu_for_tests=False,
            gpu_memory_limit_bytes=None, deterministic=True):
    """Fresh three-stage SAE with train-only reconstruction scale and transfer."""
    try:
        from .e2_nas_autoencoder import build_auxiliary_stages
    except ImportError:
        from e2_nas_autoencoder import build_auxiliary_stages
    selected = _resolve_device(device, allow_cpu_for_tests)
    _validate_gpu_limit(gpu_memory_limit_bytes)
    if not isinstance(config, TrainConfig):
        raise TypeError("A validated historical TrainConfig is required")
    if not isinstance(fold_id, str) or not fold_id.strip():
        raise ValueError("A nonempty training-fold identifier is required")
    if type(target_scale) not in (int, float) or not math.isfinite(target_scale) or target_scale <= 0:
        raise ValueError("Training-only target scale must be positive and finite")
    if fixed_epochs is not None:
        if not isinstance(fixed_epochs, dict) or set(fixed_epochs) != {"ae1", "ae2", "supervised"}:
            raise ValueError("Fixed SAE refit needs all three registered epoch counts")
        if x_val is not None or y_val_scaled is not None:
            raise ValueError("Fixed SAE refit accepts no validation inputs or labels")
        if any(type(count) is not int or not 1 <= count <= config.max_epochs
               for count in fixed_epochs.values()):
            raise ValueError("SAE stage epoch count exceeds configured horizon")
    x_train, y_train_scaled = cpu._pairs(x_train, y_train_scaled, supervised=True, training=True)
    if fixed_epochs is None:
        x_val, y_val_scaled = cpu._pairs(x_val, y_val_scaled, supervised=True, training=False)
    common = {"device": selected, "allow_cpu_for_tests": allow_cpu_for_tests,
              "gpu_memory_limit_bytes": gpu_memory_limit_bytes, "deterministic": deterministic}
    with _execution_policy(selected, deterministic) as runtime:
        started = time.perf_counter()
        phases = {}
        boundary = "construction"

        def stage(name, function, *args):
            try:
                result = function(*args, fixed_epochs=fixed_epochs[name] if fixed_epochs else None, **common)
            except TrainingFailure as error:
                known = error.receipt.get("completed_optimizer_steps", error.receipt.get("optimizer_steps"))
                receipt = {**error.receipt, "sae_phase": name, "fold_id": fold_id,
                           "completed_phases": {key: value.to_dict() for key, value in phases.items()},
                           "sae_elapsed_seconds": time.perf_counter()-started,
                           "known_optimizer_steps_all_sae_phases":
                               sum(value.optimizer_steps for value in phases.values())+known
                               if type(known) is int else None}
                raise TrainingFailure(str(error), receipt) from error
            phases[name] = result
            return result

        try:
            cpu._seed(config.seed)
            x_train, y_train_scaled = x_train.to(selected), y_train_scaled.to(selected)
            if fixed_epochs is None:
                x_val, y_val_scaled = x_val.to(selected), y_val_scaled.to(selected)
            manager = build_auxiliary_stages(nas_config, x_train, fold_id=fold_id)
            ax, ay = manager.phase1_data()
            validation_target = manager.reconstruction_target(x_val) if fixed_epochs is None else None
            boundary = "ae1"
            stage("ae1", fit_reconstruction, manager.ae1, ax, ay, x_val, validation_target, config)
            boundary = "phase2_preparation"
            bx, by = manager.prepare_phase2()
            val_encoded = manager.phase2_inputs(x_val) if fixed_epochs is None else None
            boundary = "ae2"
            stage("ae2", fit_reconstruction, manager.ae2, bx, by, val_encoded, validation_target, config)
            cpu._seed(config.seed)
            boundary = "predictor_transfer"
            predictor = manager.transfer_predictor()
            boundary = "supervised"
            stage("supervised", fit_supervised, predictor, x_train, y_train_scaled,
                  x_val, y_val_scaled, target_scale, config)
            return SAEResult(predictor, phases, float(manager.scale), fold_id)
        except TrainingFailure:
            raise
        except Exception as error:
            # Construction/encoding/transfer has no optimizer of its own.
            # Preserve the completed earlier stages if this boundary OOMs.
            receipt = {"version": VERSION, "status": "failed", "error_type": type(error).__name__,
                       "error_code": cpu._failure_code(error), "message": str(error),
                       "optimizer_steps": 0, "completed_optimizer_steps": 0,
                       "partial_optimizer_update_may_have_occurred": False,
                       "known_optimizer_steps_all_sae_phases":
                           sum(value.optimizer_steps for value in phases.values()),
                       "epochs_completed": 0, "elapsed_seconds": time.perf_counter()-started,
                       "sae_elapsed_seconds": time.perf_counter()-started, "seed": config.seed,
                       "config": asdict(config), "retry_performed": False,
                       "sae_phase": boundary, "fold_id": fold_id,
                       "completed_phases": {key: value.to_dict() for key, value in phases.items()},
                       "device": str(selected), "runtime": dict(runtime)}
            raise TrainingFailure(str(error), receipt) from error


def predict_scaled(model, x, *, device="cuda:0", batch_size=32,
                   allow_cpu_for_tests=False, deterministic=True):
    """FP32 batched inference on the declared device; return detached CPU values."""
    selected = _resolve_device(device, allow_cpu_for_tests)
    _validate_model(model)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("A positive inference batch size is required")
    if (not isinstance(x, Tensor) or x.dtype != torch.float32 or x.ndim != 3
            or x.shape[1:] != (7, 5) or len(x) < 1 or not bool(torch.isfinite(x).all())):
        raise ValueError("Inference requires finite FP32 [N,7,5] windows")
    was_training = model.training
    with _execution_policy(selected, deterministic):
        try:
            model.to(selected)
            model.eval()
            result = []
            with torch.no_grad():
                for start in range(0, len(x), batch_size):
                    values = model(x[start:start+batch_size].to(selected))
                    if values.shape != (min(batch_size, len(x)-start),):
                        raise cpu.TrainingGeometryError("Inference prediction geometry changed")
                    if not bool(torch.isfinite(values).all()):
                        raise cpu.TrainingNumericalError("Nonfinite inference prediction")
                    result.append(values.detach().cpu().clone())
            _synchronize(selected)
            return torch.cat(result)
        finally:
            model.train(was_training)
