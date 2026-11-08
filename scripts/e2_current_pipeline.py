"""Bounded, synthetic-only plumbing for the provisional current E2 catalogue.

This module never loads market files. Its sole executable experiment is a small
mechanical check, not a benchmark, selection, or claim about generalization.
The unchanged E1 relative-window *function* is reused without calling its loader.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from dl_data_v2 import relative_features

VERSION = "e2-current-synthetic-pipeline-v1"
GENERATOR_VERSION = "synthetic-ohlcv-v1"
HISTORY = 7
FEATURES = ("open", "high", "low", "close", "volume")
MAX_DAYS = 128
MAX_TRAIN = 64
MAX_VALIDATION = 8
MAX_STEPS = 3
MAX_PRETRAIN_STEPS = 2
MAX_PARAMETERS = 2_000_000
LSTM_AE_NAME = "LSTM-autoencoder con predictor"


def _integer(value: Any, name: str, lower: int, upper: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not lower <= int(value) <= upper:
        raise ValueError(f"{name} must lie in [{lower}, {upper}]")
    return int(value)


def array_sha256(values: Any) -> str:
    array = np.asarray(values)
    if array.dtype.hasobject:
        raise ValueError("Object arrays cannot have a portable numeric fingerprint")
    # Hash shape and dtype as well as bytes, so shape-changing views differ.
    array = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
    header = json.dumps({"shape": array.shape, "dtype": array.dtype.str}, sort_keys=True).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def _indices(values: Any, n: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or not len(array) or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be nonempty one-dimensional integer indices")
    if np.any(array < HISTORY) or np.any(array >= n):
        raise ValueError(f"{name} requires complete preceding windows inside the calendar")
    array = array.astype(np.int64, copy=False)
    if len(array) > 1 and np.any(np.diff(array) <= 0):
        raise ValueError(f"{name} must be unique and chronologically ordered")
    return array


def _calendar(value: Any) -> pd.DatetimeIndex:
    calendar = pd.DatetimeIndex(value)
    if (len(calendar) < HISTORY + 1 or calendar.hasnans or not calendar.is_unique
            or not calendar.is_monotonic_increasing or calendar.tz is None
            or str(calendar.tz) != "UTC" or not calendar.equals(calendar.normalize())):
        raise ValueError("Calendar must contain unique ordered UTC midnight dates")
    if not np.all(np.diff(calendar.asi8) == pd.Timedelta(days=1).value):
        raise ValueError("Calendar must be daily; missing dates must not be compressed")
    return calendar


def make_synthetic_data(*, seed: int = 20261108, days: int = 96) -> dict:
    """Make OHLCV and seven-return population-SD targets entirely in memory.

    Seven synthetic warm-up returns precede the retained calendar. No input
    windows use those warm-up days. The labels for admitted windows have the
    same seven-log-return definition as E1, including their outcome day's close.
    """
    seed = _integer(seed, "seed", 0, 2**32 - 1)
    days = _integer(days, "days", 40, MAX_DAYS)
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0002, 0.012, size=days + HISTORY)
    padded_close = 100. * np.exp(np.cumsum(log_returns))
    close = padded_close[HISTORY:]
    open_ = close * np.exp(rng.normal(0., 0.002, size=days))
    high = np.maximum(open_, close) * (1. + rng.uniform(.001, .01, size=days))
    low = np.minimum(open_, close) * (1. - rng.uniform(.001, .01, size=days))
    volume = rng.lognormal(8., .3, size=days)
    features = np.column_stack([open_, high, low, close, volume])
    targets = np.array([log_returns[i + 1:i + HISTORY + 1].std(ddof=0)
                        for i in range(days)], dtype=np.float64)
    calendar = pd.date_range("2001-01-01", periods=days, tz="UTC")
    outcomes = np.arange(HISTORY, days, dtype=np.int64)
    input_days = outcomes[:, None] - np.arange(HISTORY, 0, -1)
    data = {
        "synthetic_only": True,
        "generator": GENERATOR_VERSION,
        "seed": seed,
        "calendar": calendar,
        "features": features,
        "targets": targets,
        "outcome_indices": outcomes,
        "outcome_days_utc": calendar[outcomes],
        "window_day_indices": input_days,
        "window_days_utc_ns": calendar.asi8[input_days],
        "windows": features[input_days].copy(),
    }
    validate_synthetic_data(data)
    return data


def validate_synthetic_data(data: dict) -> pd.DatetimeIndex:
    """Check every supplied value and its index/date correspondence.

    The provenance marker is an assertion for the array API, not a detector of
    market data. The CLI therefore exposes no data-path or external-array input.
    """
    if data.get("synthetic_only") is not True or data.get("generator") != GENERATOR_VERSION:
        raise ValueError("Only the explicit in-memory synthetic fixture is admitted")
    calendar = _calendar(data["calendar"])
    if len(calendar) > MAX_DAYS:
        raise ValueError("Synthetic data exceeds the smoke-test day cap")
    features = np.asarray(data["features"], dtype=np.float64)
    targets = np.asarray(data["targets"], dtype=np.float64)
    if features.shape != (len(calendar), 5) or targets.shape != (len(calendar),):
        raise ValueError("Expected calendar-aligned five-channel OHLCV and scalar targets")
    if (not np.isfinite(features).all() or not np.isfinite(targets).all()
            or np.any(features[:, :4] <= 0) or np.any(features[:, 4] < 0)
            or np.any(targets < 0)):
        raise ValueError("Nonfinite or invalid raw features/targets are forbidden")
    if (np.any(features[:, 1] < np.maximum(features[:, 0], features[:, 3]))
            or np.any(features[:, 2] > np.minimum(features[:, 0], features[:, 3]))):
        raise ValueError("OHLC high/low bounds are invalid")
    outcomes = _indices(data["outcome_indices"], len(calendar), "outcomes")
    expected_outcomes = np.arange(HISTORY, len(calendar), dtype=np.int64)
    if not np.array_equal(outcomes, expected_outcomes):
        raise ValueError("Outcome positions must retain the complete daily calendar")
    if not pd.DatetimeIndex(data["outcome_days_utc"]).equals(calendar[outcomes]):
        raise ValueError("Outcome dates do not match their intended calendar indices")
    expected_days = outcomes[:, None] - np.arange(HISTORY, 0, -1)
    supplied_days = np.asarray(data["window_day_indices"])
    if supplied_days.dtype.kind not in "iu" or not np.array_equal(supplied_days, expected_days):
        raise ValueError("Window indices must be outcome-7 through outcome-1, oldest first")
    supplied_dates = np.asarray(data["window_days_utc_ns"])
    if supplied_dates.dtype.kind not in "iu" or not np.array_equal(supplied_dates, calendar.asi8[expected_days]):
        raise ValueError("Window dates differ from their intended UTC calendar days")
    windows = np.asarray(data["windows"], dtype=np.float64)
    if windows.shape != (len(outcomes), HISTORY, 5) or not np.isfinite(windows).all():
        raise ValueError("Windows must be finite N x 7 x 5 arrays")
    if not np.array_equal(windows, features[expected_days]):
        raise ValueError("Windows are misdated or differ from their recorded daily features")
    return calendar


def validate_fold(data: dict, fold: dict) -> tuple[np.ndarray, np.ndarray]:
    calendar = validate_synthetic_data(data)
    origin = _integer(fold["origin_index"], "origin_index", HISTORY + 1, len(calendar) - 1)
    origin_day = pd.Timestamp(fold["origin_day_utc"])
    if (origin_day.tzinfo is None or str(origin_day.tzinfo) != "UTC"
            or origin_day != calendar[origin]):
        raise ValueError("Origin timestamp must match its explicit UTC calendar index")
    train = _indices(fold["train_indices"], len(calendar), "train")
    validation = _indices(fold["validation_indices"], len(calendar), "validation")
    if len(train) > MAX_TRAIN or len(validation) > MAX_VALIDATION:
        raise ValueError("Fold exceeds bounded synthetic cohort sizes")
    if np.intersect1d(train, validation).size:
        raise ValueError("Train and validation outcomes overlap")
    if np.any(train > origin - HISTORY - 1):
        raise ValueError("Training labels must be no later than origin-8")
    if np.any(validation < origin):
        raise ValueError("Validation cannot precede the declared prediction origin")
    if np.max(train) >= np.min(validation) - HISTORY:
        raise ValueError("Temporal guard forbids shared train-label/validation-window days")
    for name, indices in [("train", train), ("validation", validation)]:
        if not pd.DatetimeIndex(fold[f"{name}_days_utc"]).equals(calendar[indices]):
            raise ValueError(f"{name} dates do not match the selected outcomes")
    return train, validation


def make_synthetic_fold(data: dict, *, origin_index: int = 72,
                        train_count: int = 32, validation_count: int = 8) -> dict:
    """Build one admitted train cohort and a later synthetic-only check cohort."""
    calendar = validate_synthetic_data(data)
    origin = _integer(origin_index, "origin_index", 16, len(calendar) - 1)
    train_count = _integer(train_count, "train_count", 2, MAX_TRAIN)
    validation_count = _integer(validation_count, "validation_count", 1, MAX_VALIDATION)
    train_last = origin - HISTORY - 1
    train = np.arange(train_last - train_count + 1, train_last + 1, dtype=np.int64)
    validation = np.arange(origin, origin + validation_count, dtype=np.int64)
    # Validate positions before using them as potentially negative NumPy indices.
    train = _indices(train, len(calendar), "train")
    validation = _indices(validation, len(calendar), "validation")
    fold = {
        "origin_index": origin,
        "origin_day_utc": calendar[origin],
        "train_indices": train,
        "validation_indices": validation,
        "train_days_utc": calendar[train],
        "validation_days_utc": calendar[validation],
    }
    validate_fold(data, fold)
    return fold


def _train_signature(data: dict, fold: dict) -> dict:
    train, _ = validate_fold(data, fold)
    return {
        "train_indices_sha256": array_sha256(train),
        "train_windows_sha256": array_sha256(np.asarray(data["windows"])[train - HISTORY]),
        "train_targets_sha256": array_sha256(np.asarray(data["targets"])[train]),
        "origin_day_utc": pd.Timestamp(fold["origin_day_utc"]).isoformat(),
    }


def fit_preprocessing(data: dict, fold: dict) -> dict:
    """Fit all feature statistics and y scaling on admitted train arrays only."""
    train, _ = validate_fold(data, fold)
    raw = np.asarray(data["windows"])[train - HISTORY]
    transformed = relative_features(raw).reshape(-1, 5)
    mean, std = transformed.mean(axis=0), transformed.std(axis=0, ddof=0)
    scale = np.where(std == 0., 1., std)
    y_scale = float(np.asarray(data["targets"])[train].mean())
    if not np.isfinite(y_scale) or y_scale <= 0:
        raise ValueError("A finite strictly positive train-target mean is required")
    preproc = {
        "version": VERSION,
        "history_days": HISTORY,
        "feature_order": list(FEATURES),
        "representation": "E1 relative_window via dl_data_v2.relative_features",
        "x_mean": mean.tolist(),
        "x_scale": scale.tolist(),
        "x_std_ddof0": std.tolist(),
        "x_fit_window_positions": int(len(train) * HISTORY),
        "x_fit_unit": "all_positions_in_admitted_training_windows",
        "y_scale": y_scale,
        "target_transform": "divide_by_positive_training_target_mean_without_centering",
        "target_centered": False,
        **_train_signature(data, fold),
    }
    json.dumps(preproc, allow_nan=False)
    return preproc


def transform_fold(data: dict, fold: dict, preproc: dict, cohort: str) -> tuple[np.ndarray, np.ndarray]:
    """Apply the frozen, fingerprinted training transform to one declared cohort."""
    train, validation = validate_fold(data, fold)
    if cohort not in {"train", "validation"}:
        raise ValueError("Only declared train or validation cohorts are admitted")
    if any(preproc.get(key) != value for key, value in _train_signature(data, fold).items()):
        raise ValueError("Preprocessing provenance does not match the admitted train cohort")
    if (preproc.get("history_days") != HISTORY or preproc.get("target_centered") is not False
            or preproc.get("feature_order") != list(FEATURES)):
        raise ValueError("Preprocessing contract differs from D=7 OHLCV, noncentered y")
    mean = np.asarray(preproc["x_mean"], dtype=np.float64)
    scale = np.asarray(preproc["x_scale"], dtype=np.float64)
    y_scale = float(preproc["y_scale"])
    if (mean.shape != (5,) or scale.shape != (5,) or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)
            or not np.isfinite(y_scale) or y_scale <= 0):
        raise ValueError("Frozen preprocessing must have finite positive scales")
    selected = train if cohort == "train" else validation
    with np.errstate(over="ignore", invalid="ignore"):
        x = ((relative_features(np.asarray(data["windows"])[selected - HISTORY]) - mean) / scale).astype(np.float32)
        y = (np.asarray(data["targets"])[selected] / y_scale).astype(np.float32)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Transformed features and targets must remain finite float32 values")
    return x, y


@dataclass(frozen=True)
class SmokeConfig:
    """Low immutable ceilings prevent this check from becoming a training run."""
    seed: int = 20261108
    days: int = 96
    origin_index: int = 72
    train_count: int = 32
    validation_count: int = 8
    batch_size: int = 8
    supervised_steps: int = 2
    pretrain_steps: int = 1
    learning_rate: float = .001

    def validate(self) -> None:
        _integer(self.seed, "seed", 0, 2**32 - 1)
        _integer(self.days, "days", 40, MAX_DAYS)
        _integer(self.origin_index, "origin_index", 16, self.days - 1)
        _integer(self.train_count, "train_count", 2, MAX_TRAIN)
        _integer(self.validation_count, "validation_count", 1, MAX_VALIDATION)
        _integer(self.batch_size, "batch_size", 4, min(self.train_count, 16))
        _integer(self.supervised_steps, "supervised_steps", 1, MAX_STEPS)
        _integer(self.pretrain_steps, "pretrain_steps", 1, MAX_PRETRAIN_STEPS)
        if (isinstance(self.learning_rate, bool) or not np.isfinite(self.learning_rate)
                or not 0 < self.learning_rate <= .01):
            raise ValueError("learning_rate must be finite and in (0, .01]")


def _state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode() + b"\0")
        digest.update(array_sha256(value.detach().cpu().numpy()).encode())
    return digest.hexdigest()


def _predictions(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    predicted = model(x)
    if not isinstance(predicted, torch.Tensor) or predicted.shape != (len(x),):
        raise ValueError("Every adapter must return exactly one scalar per sample as Tensor[N]")
    if not torch.isfinite(predicted).all():
        raise ValueError("Model produced nonfinite regression predictions")
    return predicted


def predict_original_units(model: nn.Module, x: np.ndarray, preproc: dict) -> np.ndarray:
    """Return predictions in target units; do not center or clip regression output."""
    array = np.asarray(x, dtype=np.float32)
    y_scale = float(preproc["y_scale"])
    if (array.ndim != 3 or array.shape[1:] != (HISTORY, 5) or len(array) < 1
            or not np.isfinite(array).all() or not np.isfinite(y_scale) or y_scale <= 0):
        raise ValueError("Prediction requires finite N x 7 x 5 features and positive y_scale")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            result = _predictions(model, torch.from_numpy(array)).cpu().numpy().astype(np.float64) * y_scale
    finally:
        model.train(was_training)
    if not np.isfinite(result).all():
        raise ValueError("Inverse-scaled predictions must remain finite")
    return result


def _optimize(model: nn.Module, x: torch.Tensor, y: torch.Tensor, *,
              steps: int, config: SmokeConfig, reconstruct: bool) -> tuple[list[float], dict]:
    # A separate optimizer for each stage is deliberate and is recorded below.
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    model.train()
    losses = []
    allowed_unused = ()
    if getattr(model, "architecture_metadata", {}).get("model") == LSTM_AE_NAME:
        allowed_unused = (("core.head.",) if reconstruct else
                          ("core.decoder.", "core.reconstruction."))
    gradient_audit = {}
    for step in range(steps):
        selected = (torch.arange(config.batch_size) + step * config.batch_size) % len(x)
        xb, yb = x[selected], y[selected]
        optimizer.zero_grad(set_to_none=True)
        if reconstruct:
            predicted = model.reconstruct(xb)
            if predicted.shape != xb.shape or not torch.isfinite(predicted).all():
                raise ValueError("Autoencoder must reconstruct a finite full D=7 x 5 sequence")
            loss = nn.functional.mse_loss(predicted, xb)
        else:
            loss = nn.functional.mse_loss(_predictions(model, xb), yb)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite synthetic optimization loss")
        loss.backward()
        trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        missing = [name for name, p in trainable if p.grad is None]
        unexpected = [name for name in missing if not name.startswith(allowed_unused)]
        gradients = [p.grad for _, p in trainable if p.grad is not None]
        if unexpected or not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise ValueError("Missing or nonfinite training gradients: " + ", ".join(unexpected))
        gradient_audit = {"finite": True, "unexpected_missing": [],
                          "expected_unused_parameter_names": missing,
                          "parameter_tensors_with_gradient": len(gradients)}
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1., error_if_nonfinite=True)
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("Nonfinite model parameters after optimization")
        losses.append(float(loss.detach()))
    return losses, gradient_audit


def train_synthetic_model(name: str, data: dict, fold: dict, preproc: dict,
                          config: SmokeConfig, *, recipe: dict | None = None) -> tuple[nn.Module, dict]:
    """Exercise genuine forward/backward steps without model selection or refit."""
    from e2_current_models import build_model

    config.validate()
    train, validation = validate_fold(data, fold)
    if (data["seed"] != config.seed or len(data["calendar"]) != config.days or len(train) != config.train_count
            or len(validation) != config.validation_count or fold["origin_index"] != config.origin_index):
        raise ValueError("Synthetic data/fold differs from the declared smoke configuration")
    x_train, y_train = transform_fold(data, fold, preproc, "train")
    x_validation, _ = transform_fold(data, fold, preproc, "validation")
    torch.manual_seed(config.seed)
    selected_recipe = dict(recipe or {})
    if set(selected_recipe) - {"preset", "source_roots"}:
        raise ValueError("Synthetic smoke dimensions are fixed; only preset/source_roots may be supplied")
    if selected_recipe.get("preset", "compact_synthetic") != "compact_synthetic":
        raise ValueError("The executable synthetic smoke only admits its compact preset")
    selected_recipe["preset"] = "compact_synthetic"
    model = build_model(name, selected_recipe).cpu()
    parameter_count = sum(p.numel() for p in model.parameters())
    if not 0 < parameter_count <= MAX_PARAMETERS:
        raise ValueError("Model is empty or exceeds the synthetic parameter cap")
    initial_sha = _state_sha256(model)
    x, y = torch.from_numpy(x_train), torch.from_numpy(y_train)
    pretrain_losses, pretrain_gradients = [], None
    if name == LSTM_AE_NAME:
        if not callable(getattr(model, "reconstruct", None)):
            raise ValueError("LSTM-AE requires actual sequence reconstruction pretraining")
        pretrain_losses, pretrain_gradients = _optimize(model, x, y, steps=config.pretrain_steps,
                                    config=config, reconstruct=True)
    post_pretrain_sha = _state_sha256(model)
    losses, supervised_gradients = _optimize(model, x, y, steps=config.supervised_steps,
                       config=config, reconstruct=False)
    predictions = predict_original_units(model, x_validation, preproc)
    targets = np.asarray(data["targets"])[validation]
    mse = float(np.mean((predictions - targets) ** 2))
    if not np.isfinite(mse):
        raise ValueError("Synthetic original-unit MSE must be finite")
    record = {
        "model": name,
        "parameter_count": int(parameter_count),
        "trainable_parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "recipe": {key: value for key, value in getattr(model, "recipe", selected_recipe).items()
                   if key != "source_roots"},
        "architecture": getattr(model, "architecture_metadata", {}),
        "initial_state_sha256": initial_sha,
        "post_pretrain_state_sha256": post_pretrain_sha,
        "final_state_sha256": _state_sha256(model),
        "pretrain_steps": len(pretrain_losses),
        "supervised_steps": len(losses),
        "pretrain_reconstruction_mse": pretrain_losses,
        "supervised_scaled_target_mse": losses,
        "pretrain_gradient_audit": pretrain_gradients,
        "supervised_gradient_audit": supervised_gradients,
        "synthetic_validation_mse_original_units": mse,
        "synthetic_predictions_original_units": predictions.tolist(),
        "all_losses_and_predictions_finite": True,
        "ae_protocol": ({
            "status": "provisional_synthetic_only_not_paper_hyperparameter_replication",
            "pretrain": "reconstruct full standardized 7x5 training windows using encoder and decoder",
            "supervised": "fine-tune encoder and scalar regression head on scaled train targets",
            "decoder": "used for reconstruction, not for supervised scalar prediction",
            "optimizer": "fresh Adam state for each stage; learning rate shared",
            "validation_used_for_optimization_or_selection": False,
        } if name == LSTM_AE_NAME else None),
    }
    json.dumps(record, allow_nan=False)
    return model, record


def run_synthetic_smoke(*, model_names: list[str] | None = None,
                        config: SmokeConfig | None = None, recipe: dict | None = None) -> dict:
    """Run one small synthetic fold and return a self-contained audit receipt."""
    from e2_current_models import MODEL_NAMES

    config = config or SmokeConfig()
    config.validate()
    names = list(MODEL_NAMES) if model_names is None else list(model_names)
    if not names or len(names) > len(MODEL_NAMES) or len(set(names)) != len(names):
        raise ValueError("Choose a nonempty unique subset of the current model catalogue")
    if any(name not in MODEL_NAMES for name in names):
        raise ValueError("Unknown model in synthetic catalogue")
    data = make_synthetic_data(seed=config.seed, days=config.days)
    fold = make_synthetic_fold(data, origin_index=config.origin_index,
                               train_count=config.train_count, validation_count=config.validation_count)
    preproc = fit_preprocessing(data, fold)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        records = [train_synthetic_model(name, data, fold, preproc, config, recipe=recipe)[1]
                   for name in names]
    finally:
        torch.set_num_threads(previous_threads)
    calendar = data["calendar"]
    train, validation = validate_fold(data, fold)
    root = Path(__file__).resolve().parent
    sources = ["e2_current_pipeline.py", "e2_current_models.py", "smoke_e2_current_catalogue.py",
               "dl_data_v2.py", "dl_data.py", "e2_proposed_models.py",
               "e2_final_models_mlp.py", "e2_final_models_attention.py"]
    source_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}
    dependency_versions = {}
    for package in ["einops", "reformer-pytorch"]:
        try:
            dependency_versions[package] = package_version(package)
        except PackageNotFoundError:
            dependency_versions[package] = None
    receipt = {
        "receipt_version": VERSION,
        "status": "synthetic_mechanism_smoke_only",
        "synthetic_only": True,
        "requested_models": names,
        "executed_model_count": len(records),
        "skipped_model_count": 0,
        "skipped_models": [],
        "market_data_read": False,
        "heldout_market_data_read": False,
        "registered_experiment_run": False,
        "full_current_benchmark_executed": False,
        "generalization_or_superiority_demonstrated": False,
        "claim_limit": "Finite synthetic forward/backward mechanics only; no market performance, model ranking or generalization claim.",
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "numpy": np.__version__, "pandas": pd.__version__, "device": "cpu", "torch_threads": 1,
                    "optional_dependencies": dependency_versions},
        "config": asdict(config),
        "source_sha256": source_hashes,
        "synthetic_fixture": {"generator": GENERATOR_VERSION, "calendar_start_utc": calendar[0].isoformat(),
            "calendar_end_utc": calendar[-1].isoformat(), "calendar_days": len(calendar),
            "calendar_sha256": array_sha256(calendar.asi8),
            "features_sha256": array_sha256(data["features"]), "targets_sha256": array_sha256(data["targets"]),
            "windows_sha256": array_sha256(data["windows"]),
            "window_dates_sha256": array_sha256(data["window_days_utc_ns"])},
        "fold": {"origin_index": int(fold["origin_index"]), "origin_day_utc": fold["origin_day_utc"].isoformat(),
            "train_indices": train.tolist(), "validation_indices": validation.tolist(),
            "train_days_utc": [day.isoformat() for day in calendar[train]],
            "validation_days_utc": [day.isoformat() for day in calendar[validation]],
            "last_allowed_train_label_utc": calendar[fold["origin_index"] - 8].isoformat(),
            "training_labels_at_or_before_origin_minus_8": True,
            "history_days": HISTORY, "features": list(FEATURES),
            "calendar_is_daily_without_compressed_gaps": True,
            "validation_used_for_optimization_or_selection": False},
        "preprocessing": preproc,
        "training_contract": {"objective": "MSE on training-mean-scaled, noncentered targets",
            "prediction_units": "original target units", "optimizer": "Adam", "gradient_clip_norm": 1.,
            "batches": "deterministic chronological cyclic training-only batches",
            "early_stopping": False, "hyperparameter_search": False, "checkpoint_or_registered_output_written": False,
            "provisional_recipe": True},
        "models": records,
    }
    json.dumps(receipt, allow_nan=False)
    return receipt
