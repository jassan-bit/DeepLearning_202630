"""Durable, prospective FP32 pilot training for supplied published modules.

There is no data loader, model selector, retry or checkpoint-resume path here.
The caller seals chronology, architecture, scaler and budget before supplying
tensors and an artifact store. Production requires CUDA; CPU execution is an
explicit synthetic-test opt-in. Checkpoints are audit artifacts, not a claim
that an interrupted optimizer can be resumed exactly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

try:
    from . import e2_staged_training as historical
    from . import e2_cuda_training as cuda
except ImportError:
    import e2_staged_training as historical
    import e2_cuda_training as cuda

VERSION = "e2-published-pilot-training-v1"
TrainConfig = historical.TrainConfig
TrainingFailure = historical.TrainingFailure


def _cpu_tree(value):
    """Clone tensor leaves including non-contiguous author parameter layouts."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


@dataclass(frozen=True)
class PilotFitOutcome:
    state_dict: dict
    best_epoch: int
    epochs_completed: int
    optimizer_steps: int
    logs: tuple
    receipt: dict

    def to_dict(self):
        return dict(self.receipt)


def _store_ready(store, metadata):
    # Artifact identity/scientific metadata may already have been prepared by
    # the runner. begin_training must persist an exclusive training admission.
    if not callable(getattr(store, "begin_training", None)):
        raise TypeError("An explicit durable PilotArtifactStore is required")
    for name in ("event", "checkpoint", "finish"):
        if not callable(getattr(store, name, None)):
            raise TypeError("PilotArtifactStore is missing " + name)
    store.begin_training(metadata)


def fit_pilot(model, X_train, y_train, config, store, X_val=None, y_val=None,
              fixed_epochs=None, device="cuda:0", allow_cpu_for_tests=False, *,
              target_scale=1.0, gpu_memory_limit_bytes=None, deterministic=True,
              step_events=False):
    """Fit a fresh supplied module with the frozen AdamW optimization math.

    Validation chooses strict first-best RMSE in original units. A fixed-epoch
    refit refuses validation tensors and retains the full configured scheduler
    horizon. Data scaling/inversion belongs to the caller. The store receives
    initial weights before updates, per-epoch receipts/curves and
    matched latest/best model+AdamW+RNG snapshots. Any failure is terminal for
    the attempt: this function never loads an existing checkpoint or retries.
    The default durable boundary is an epoch, avoiding per-update fsync costs
    on network storage. A hard interruption leaves an open attempt to reconcile;
    its durable step count remains a lower bound. Explicit ``step_events=True``
    enables finer diagnostic receipts without changing optimization math.
    """
    try:
        selected = cuda._resolve_device(device, allow_cpu_for_tests)
    except cuda.CudaUnavailableError as error:
        admission = {"trainer_version": VERSION, "config": asdict(config),
            "fixed_epochs": fixed_epochs, "device": str(device),
            "resume_exact_supported": False, "retry_performed": False}
        _store_ready(store, admission)
        receipt = {"version": VERSION, "status": "failed", "error_code": "cuda_unavailable",
            "error_type": type(error).__name__, "message": str(error),
            "optimizer_steps": 0, "completed_optimizer_steps": 0,
            "epochs_completed": 0, "elapsed_seconds": 0., "known_cost_seconds": 0.,
            "partial_optimizer_update_may_have_occurred": False,
            "retry_performed": False, "resume_exact_supported": False}
        store.finish("failed", receipt)
        raise TrainingFailure(str(error), receipt) from error
    cuda._validate_gpu_limit(gpu_memory_limit_bytes)
    cuda._validate_model(model)
    if type(step_events) is not bool:
        raise TypeError("The optional per-step event flag must be explicit boolean")
    if not isinstance(config, TrainConfig):
        raise TypeError("A validated historical TrainConfig is required")
    if type(target_scale) not in (int, float) or not math.isfinite(target_scale) or target_scale <= 0:
        raise ValueError("Training-only target scale must be finite and positive")
    if fixed_epochs is not None:
        if type(fixed_epochs) is not int or not 1 <= fixed_epochs <= config.max_epochs:
            raise ValueError("Fixed refit epochs must lie within the registered cap")
        if X_val is not None or y_val is not None:
            raise ValueError("Fixed refit accepts no validation inputs or labels")
    elif X_val is None or y_val is None:
        raise ValueError("Selection requires an explicit validation pair")
    X_train, y_train = historical._pairs(X_train, y_train, supervised=True, training=True)
    if fixed_epochs is None:
        X_val, y_val = historical._pairs(X_val, y_val, supervised=True, training=False)
        if X_val.device != X_train.device:
            raise ValueError("Training and validation input devices differ")

    # Device/policy validation precedes durable admission and RNG mutation.
    with cuda._execution_policy(selected, deterministic) as runtime:
        _store_ready(store, {"trainer_version": VERSION, "config": asdict(config),
            "fixed_epochs": fixed_epochs, "target_scale": target_scale,
            "device": str(selected), "test_only_cpu_execution": selected.type == "cpu",
            "durable_event_cadence": "optimizer_step" if step_events else "epoch",
            "resume_exact_supported": False, "retry_performed": False})
        return _fit(model, X_train, y_train, X_val, y_val, config, store,
                    fixed_epochs=fixed_epochs, selected=selected,
                    target_scale=target_scale, gpu_memory_limit_bytes=gpu_memory_limit_bytes,
                    runtime=runtime, step_events=step_events)


def _fit(model, x_train, y_train, x_val, y_val, config, store, *,
         fixed_epochs, selected, target_scale, gpu_memory_limit_bytes, runtime, step_events):
    cuda._synchronize(selected)
    start = time.perf_counter()
    logs, steps, best_epoch, best_metric, best_state = [], 0, 0, math.inf, None
    max_rss = max_allocated = max_reserved = 0
    optimizer = None
    initial_hash = historical.state_sha256(model)
    generator = None
    checkpoint_receipts = {}
    step_in_progress = False
    optimizer_update_may_have_occurred = False

    def resources():
        nonlocal max_rss, max_allocated, max_reserved
        cuda._synchronize(selected)
        max_rss = max(max_rss, historical._rss_bytes())
        if selected.type == "cuda":
            max_allocated = max(max_allocated, int(torch.cuda.max_memory_allocated(selected)))
            max_reserved = max(max_reserved, int(torch.cuda.max_memory_reserved(selected)))
        elapsed = time.perf_counter() - start
        return {"elapsed_seconds": elapsed, "known_cost_seconds": elapsed,
                "max_observed_rss_bytes": max_rss,
                "max_cuda_memory_allocated_bytes": max_allocated,
                "max_cuda_memory_reserved_bytes": max_reserved,
                "optimizer_steps": steps, "epochs_completed": len(logs)}

    def limits():
        record = resources()
        if config.wall_limit_seconds is not None and record["elapsed_seconds"] > config.wall_limit_seconds:
            raise TimeoutError("Pilot wall limit exceeded at a synchronized boundary")
        if config.memory_limit_bytes is not None and max_rss > config.memory_limit_bytes:
            raise MemoryError("Pilot process RSS limit exceeded at a checked boundary")
        if gpu_memory_limit_bytes is not None and max_allocated > gpu_memory_limit_bytes:
            raise MemoryError("CUDA allocated-memory threshold exceeded at a checked boundary")
        return record

    def snapshot(epoch, scheduler_steps, stale):
        return {"version": VERSION, "state_dict": historical.clone_state(model),
                "model_state_sha256": historical.state_sha256(model),
                "optimizer_state_dict": _cpu_tree(optimizer.state_dict()),
                "rng": {"torch_cpu": torch.get_rng_state().clone(),
                        "torch_cuda": ([state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
                                       if selected.type == "cuda" else []),
                        "python": random.getstate(),
                        "batch_generator": generator.get_state().clone()},
                "epoch": epoch, "optimizer_steps": steps,
                "scheduler_total_steps": scheduler_steps,
                "best_epoch": best_epoch, "best_metric": None if math.isinf(best_metric) else best_metric,
                "stale_epochs": stale, "config": asdict(config),
                "fixed_epochs": fixed_epochs, "target_scale": target_scale,
                "runtime": dict(runtime), "resources": resources(),
                "durable_event_cadence": "optimizer_step" if step_events else "epoch",
                "resume_exact_supported": False}

    try:
        model.to(selected)
        x_train, y_train = x_train.to(selected), y_train.to(selected)
        if fixed_epochs is None:
            x_val, y_val = x_val.to(selected), y_val.to(selected)
        historical._seed(config.seed)
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        batches = len(historical.batch_indices(len(x_train), config.batch_size,
                      generator=torch.Generator().manual_seed(config.seed)))
        scheduler_steps = config.max_epochs * batches
        optimizer = torch.optim.AdamW(historical.optimizer_groups(model, config.weight_decay),
                                     lr=config.learning_rate, betas=(.9, .999), eps=1e-8)
        initial_entries = len(optimizer.state)
        checkpoint_receipts["initial"] = store.checkpoint(snapshot(0, scheduler_steps, 0), label="initial")
        store.event("initial_weights_saved", {"initial_state_sha256": initial_hash,
            "optimizer_initial_state_entries": initial_entries, **resources()})
        limits()
        epoch_count = fixed_epochs if fixed_epochs is not None else config.max_epochs
        stale, stop_reason = 0, "fixed_epochs" if fixed_epochs is not None else "epoch_cap"
        for epoch in range(1, epoch_count + 1):
            limits()
            store.event("epoch_started", {"epoch": epoch, **resources()})
            model.train()
            batch_sizes, rate_first, rate_last = [], None, None
            for indices in historical.batch_indices(len(x_train), config.batch_size, generator=generator):
                limits()
                indices = indices.to(selected)
                rate = config.learning_rate * historical.learning_rate_multiplier(
                    steps, scheduler_steps, config.warmup_fraction)
                for group in optimizer.param_groups:
                    group["lr"] = rate
                optimizer.zero_grad(set_to_none=True)
                prediction, truth = model(x_train[indices]), y_train[indices]
                if prediction.shape != truth.shape:
                    raise historical.TrainingGeometryError("Prediction geometry failed during training")
                if not bool(torch.isfinite(prediction).all()):
                    raise historical.TrainingNumericalError("Nonfinite prediction during training")
                loss = F.mse_loss(prediction, truth)
                if not bool(torch.isfinite(loss)):
                    raise historical.TrainingNumericalError("Nonfinite objective")
                loss.backward()
                trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
                if any(parameter.grad is None for parameter in trainable):
                    raise historical.TrainingMissingGradientError("A trainable parameter has no gradient")
                if any(not bool(torch.isfinite(parameter.grad).all()) for parameter in trainable):
                    raise historical.TrainingNumericalError("Nonfinite parameter gradient")
                norm = nn.utils.clip_grad_norm_(trainable, config.gradient_clip, error_if_nonfinite=False)
                if not bool(torch.isfinite(norm)):
                    raise historical.TrainingNumericalError("Nonfinite aggregate gradient norm")
                if step_events:
                    store.event("optimizer_step_started", {"epoch": epoch,
                        "next_optimizer_step": steps + 1, **resources()})
                step_in_progress = True
                optimizer.step()
                cuda._synchronize(selected)
                steps += 1
                step_in_progress = False
                optimizer_update_may_have_occurred = True
                if step_events:
                    store.event("optimizer_step_completed", {"epoch": epoch, **resources()})
                if any(not bool(torch.isfinite(parameter).all()) for parameter in trainable):
                    raise historical.TrainingNumericalError("Nonfinite updated parameter")
                batch_sizes.append(len(indices))
                rate_first = rate if rate_first is None else rate_first
                rate_last = rate
                limits()
            train_mse = historical._mse(model, x_train, y_train, config.batch_size)
            score, improved = None, False
            if fixed_epochs is None:
                score = math.sqrt(historical._mse(model, x_val, y_val, config.batch_size)) * target_scale
                if not math.isfinite(score):
                    raise historical.TrainingNumericalError("Nonfinite selection score")
                if score < best_metric:
                    best_metric, best_epoch, stale = score, epoch, 0
                    best_state, improved = historical.clone_state(model), True
                else:
                    stale += 1
            else:
                best_epoch = epoch
                improved = True
            row = {"epoch": epoch, "train_mse_scaled": train_mse,
                   "train_rmse_original": math.sqrt(train_mse) * target_scale,
                   "validation_score": score, "batch_sizes": batch_sizes,
                   "examples_seen": sum(batch_sizes), "learning_rate_first": rate_first,
                   "learning_rate_last": rate_last, "learning_rate_after_epoch":
                       config.learning_rate * historical.learning_rate_multiplier(
                           steps, scheduler_steps, config.warmup_fraction),
                   "optimizer_steps_total": steps}
            logs.append(row)
            payload = snapshot(epoch, scheduler_steps, stale)
            checkpoint_receipts["latest"] = store.checkpoint(payload, label="latest")
            if improved:
                checkpoint_receipts["best"] = store.checkpoint(payload, label="best")
            epoch_resources = resources()
            store.event("epoch_completed", {**row, "resources": epoch_resources,
                "optimizer_steps": steps, "known_cost_seconds": epoch_resources["known_cost_seconds"],
                "latest_checkpoint": checkpoint_receipts["latest"],
                "best_checkpoint": checkpoint_receipts.get("best")})
            limits()
            if fixed_epochs is None and stale >= config.patience:
                stop_reason = "patience"
                break
        if fixed_epochs is not None:
            best_state = historical.clone_state(model)
        if best_state is None:
            raise historical.TrainingCheckpointError("No admissible checkpoint completed")
        model.load_state_dict(best_state)
        model.eval()
        limits()
        receipt = {"version": VERSION, "status": "completed", **resources(),
            "best_epoch": best_epoch, "initial_state_sha256": initial_hash,
            "selected_state_sha256": historical.state_sha256(best_state),
            "optimizer_initial_state_entries": initial_entries, "stop_reason": stop_reason,
            "selection_metric": "rmse_original", "fixed_epochs": fixed_epochs,
            "scheduler_total_steps": scheduler_steps, "seed": config.seed,
            "device": str(selected), "runtime": dict(runtime), "logs": logs,
            "gpu_memory_limit_bytes": gpu_memory_limit_bytes,
            "cuda_synchronized_timing": selected.type == "cuda",
            "checkpoint_receipts": checkpoint_receipts,
            "durable_event_cadence": "optimizer_step" if step_events else "epoch",
            "resume_exact_supported": False, "retry_performed": False}
        # The runner still owns prediction persistence and scoring; only it
        # can close the scientific attempt after those artifacts are durable.
        store.event("training_finished", receipt)
        return PilotFitOutcome(best_state, best_epoch, len(logs), steps, tuple(logs), receipt)
    except BaseException as error:
        try:
            usage = resources()
        except Exception:
            usage = {"elapsed_seconds": time.perf_counter() - start,
                     "optimizer_steps": steps, "epochs_completed": len(logs)}
        status = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
        receipt = {"version": VERSION, "status": status, **usage,
            "error_type": type(error).__name__, "error_code": historical._failure_code(error),
            "message": str(error), "completed_optimizer_steps": steps,
            "partial_optimizer_update_may_have_occurred": step_in_progress,
            "optimizer_update_occurred": optimizer_update_may_have_occurred,
            "initial_state_sha256": initial_hash, "seed": config.seed,
            "config": asdict(config), "device": str(selected), "logs": logs,
            "checkpoint_receipts": checkpoint_receipts,
            "durable_event_cadence": "optimizer_step" if step_events else "epoch",
            "resume_exact_supported": False, "retry_performed": False,
            "requires_reconciliation": True}
        try:
            store.finish(status, receipt)
        except Exception as persistence_error:
            receipt["failure_receipt_persistence_error"] = type(persistence_error).__name__
        raise TrainingFailure(str(error), receipt) from error
    finally:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)


def predict_pilot(model, X, *, batch_size=32, device="cuda:0", allow_cpu_for_tests=False,
                  deterministic=True):
    """Return scaled finite scalar predictions without accepting labels."""
    selected = cuda._resolve_device(device, allow_cpu_for_tests)
    cuda._validate_model(model)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("A positive inference batch size is required")
    if (not isinstance(X, torch.Tensor) or X.dtype != torch.float32 or X.ndim != 3
            or tuple(X.shape[1:]) != (7, 5) or len(X) < 1 or not bool(torch.isfinite(X).all())):
        raise ValueError("Inference requires finite FP32 D7 input tensors")
    with cuda._execution_policy(selected, deterministic), torch.no_grad():
        model.to(selected).eval()
        pieces = []
        for start in range(0, len(X), batch_size):
            values = model(X[start:start + batch_size].to(selected))
            if tuple(values.shape) != (min(batch_size, len(X) - start),):
                raise historical.TrainingGeometryError("Inference geometry is not one scalar per row")
            if not bool(torch.isfinite(values).all()):
                raise historical.TrainingNumericalError("Nonfinite inference prediction")
            pieces.append(values.detach().cpu().clone())
        return torch.cat(pieces)
