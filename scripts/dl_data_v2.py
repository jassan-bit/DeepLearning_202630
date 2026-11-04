"""Causal relative-window preprocessing for the registered R2 ablation.

The V1 data loader and model code remain unchanged. These functions consume its
2023--2024 daily derivatives only; they do not read files, open test data, train
models, add target lags or change the five-channel OHLCV input contract.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from dl_data import CALENDAR, FEATURES, HISTORIES


def _digest(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def _calendar(data: dict) -> pd.DatetimeIndex:
    calendar = pd.DatetimeIndex(data["calendar"])
    if (not len(calendar) or calendar.hasnans or not calendar.is_unique
            or not calendar.is_monotonic_increasing or calendar.tz is None
            or str(calendar.tz) != "UTC"):
        raise ValueError("A nonempty, ordered, unique UTC calendar is required")
    if not np.isin(calendar.year, [2023, 2024]).all():
        raise ValueError("Only the 2023--2024 development calendar is admitted")
    if len(calendar) > 1 and not np.asarray(
        calendar[1:] - calendar[:-1] == pd.Timedelta(days=1)
    ).all():
        raise ValueError("Calendar positions must be daily; do not compress missing days")
    return calendar


def _outcomes(data: dict, history: int, indices: Any, *, unique: bool) -> tuple[pd.DatetimeIndex, np.ndarray]:
    calendar = _calendar(data)
    if (isinstance(history, bool) or not isinstance(history, (int, np.integer))
            or history < 1):
        raise ValueError("History must be a positive integer")
    values = np.asarray(indices)
    if values.ndim != 1 or values.dtype.kind not in "iu" or not len(values):
        raise ValueError("Nonempty one-dimensional integer outcome indices are required")
    values = values.astype(np.int64, copy=False)
    if np.any(values < history) or np.any(values >= len(calendar)):
        raise ValueError("Outcome indices require a complete preceding history")
    if unique and len(np.unique(values)) != len(values):
        raise ValueError("Training outcomes must not be duplicated")
    eligible = np.asarray(data["eligible_by_history"][history], dtype=bool)
    windows = np.asarray(data["windows_by_history"][history])
    if eligible.shape != (len(calendar),) or windows.shape != (len(calendar), history, 5):
        raise ValueError("Window/eligibility shapes do not match the calendar and history")
    if not eligible[values].all():
        raise ValueError("Ineligible outcome supplied; missing-day windows are excluded")
    return calendar, values


def relative_features(raw: np.ndarray) -> np.ndarray:
    """Return log-price ratios to each window's last observed close and log1p V.

    Input and output have shape (N, D, 5), oldest day first. The denominator is
    close at outcome day minus one, already known at the prediction origin.
    Log differences avoid overflow of a direct ratio for extreme price scales.
    """
    values = np.asarray(raw, dtype=np.float64)
    if (values.ndim != 3 or values.shape[-1] != 5
            or values.shape[0] < 1 or values.shape[1] < 1):
        raise ValueError("Expected nonempty (batch, history, five OHLCV channels)")
    if (not np.isfinite(values).all() or np.any(values[..., :4] <= 0)
            or np.any(values[..., 4] < 0)):
        raise ValueError("Relative preprocessing requires finite positive OHLC and nonnegative volume")
    result = np.empty_like(values)
    log_prices = np.log(values[..., :4])
    result[..., :4] = log_prices - log_prices[:, -1:, 3:4]
    result[..., 4] = np.log1p(values[..., 4])
    return result


def fit_relative_preprocessing(data: dict, history: int, train_indices: Any) -> dict:
    """Fit five channel scales using only positions in admitted training windows.

    A calendar day has a different price ratio under different window anchors.
    Accordingly this ablation fits over N*D training-window positions, rather
    than pretending the representation is pointwise and deduplicating days.
    """
    calendar, selected = _outcomes(data, history, train_indices, unique=True)
    windows = np.asarray(data["windows_by_history"][history])[selected]
    transformed = relative_features(windows)
    pooled = transformed.reshape(-1, len(FEATURES))
    mean, std = pooled.mean(axis=0), pooled.std(axis=0, ddof=0)
    scale = np.where(std == 0., 1., std)
    targets = np.asarray(data["targets"], dtype=np.float64)
    if targets.shape != (len(calendar),):
        raise ValueError("Target/calendar lengths differ")
    labels = targets[selected]
    if not np.isfinite(labels).all() or np.any(labels < 0):
        raise ValueError("Training targets must be finite and nonnegative")
    y_scale = float(labels.mean())
    if not np.isfinite(y_scale) or y_scale <= 0:
        raise ValueError("A strictly positive training-target mean is required")
    input_days = np.unique((selected[:, None] - np.arange(history, 0, -1)).ravel())
    anchors = selected - 1
    return {
        "history_days": int(history),
        "feature_order": list(FEATURES),
        "feature_transforms": ["log_price_relative_to_window_last_close"] * 4 + ["log1p"],
        "anchor": "Each outcome's last observed close at outcome_day-1, never outcome_day",
        "x_mean": mean.tolist(),
        "x_scale": scale.tolist(),
        "x_std_ddof0": std.tolist(),
        "zero_variance_feature_indices": np.flatnonzero(std == 0.).tolist(),
        "x_fit_unit": "all_positions_of_supplied_training_windows_with_window_specific_anchors",
        "fit_window_positions": int(len(selected) * history),
        "y_scale": y_scale,
        "target_transform": "divide_by_positive_training_target_mean_without_centering",
        "target_centered": False,
        "fit_outcome_count": int(len(selected)),
        "fit_outcome_indices_sha256": _digest(selected),
        "first_fit_outcome_day_utc": calendar[selected.min()].strftime("%Y-%m-%d"),
        "last_fit_outcome_day_utc": calendar[selected.max()].strftime("%Y-%m-%d"),
        "input_day_indices": input_days.tolist(),
        "input_day_indices_sha256": _digest(input_days),
        "input_day_count": int(len(input_days)),
        "first_input_day_utc": calendar[input_days.min()].strftime("%Y-%m-%d"),
        "last_input_day_utc": calendar[input_days.max()].strftime("%Y-%m-%d"),
        "anchor_day_indices_sha256": _digest(anchors),
        "first_anchor_day_utc": calendar[anchors.min()].strftime("%Y-%m-%d"),
        "last_anchor_day_utc": calendar[anchors.max()].strftime("%Y-%m-%d"),
    }


def transform_relative_windows(data: dict, history: int, indices: Any,
                               preproc: dict) -> tuple[np.ndarray, np.ndarray]:
    """Apply frozen training scales and return float32 X and positive-scaled Y."""
    calendar, selected = _outcomes(data, history, indices, unique=False)
    if preproc.get("history_days") != history:
        raise ValueError("Preprocessing history does not match the requested windows")
    mean = np.asarray(preproc["x_mean"], dtype=np.float64)
    scale = np.asarray(preproc["x_scale"], dtype=np.float64)
    y_scale = float(preproc["y_scale"])
    if (mean.shape != (5,) or scale.shape != (5,) or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)
            or not np.isfinite(y_scale) or y_scale <= 0):
        raise ValueError("Invalid preprocessing parameters")
    raw = np.asarray(data["windows_by_history"][history])[selected]
    x = ((relative_features(raw) - mean) / scale).astype(np.float32)
    targets = np.asarray(data["targets"], dtype=np.float64)
    if targets.shape != (len(calendar),):
        raise ValueError("Target/calendar lengths differ")
    y = (targets[selected] / y_scale).astype(np.float32)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError("Transformed values must be finite, with nonnegative targets")
    return x, y


def split_for_origin(data: dict, history: int, origin: Any, inner_days: int = 90) -> dict:
    """Expanding past-only training and recent inner validation at a 2024 origin.

    For the static preprocessing-only ablation use V1 ``split_training`` instead;
    changing its internal dates would confound that comparison.
    """
    calendar = _calendar(data)
    if not calendar.equals(CALENDAR) or history not in HISTORIES or isinstance(history, bool):
        raise ValueError("The registered full development calendar/history is required")
    if isinstance(inner_days, bool) or not isinstance(inner_days, (int, np.integer)) or inner_days < 1:
        raise ValueError("Inner validation days must be a positive integer")
    start = pd.Timestamp(origin)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    if pd.isna(start) or start.year != 2024 or start != start.normalize() or start not in calendar:
        raise ValueError("Origin must be a UTC calendar-day start in development year 2024")
    origin_index = int(calendar.get_loc(start))
    inner_start = start - pd.Timedelta(days=int(inner_days))
    if inner_start not in calendar:
        raise ValueError("Inner validation starts outside the admitted calendar")
    inner_index = int(calendar.get_loc(inner_start))
    train_cutoff = origin_index - int(history) - 1
    inner_cutoff = inner_index - int(history) - 1
    positions = np.arange(len(calendar))
    eligible = np.asarray(data["eligible_by_history"][history], dtype=bool)
    if eligible.shape != (len(calendar),):
        raise ValueError("Eligibility mask has the wrong shape")
    train = np.flatnonzero(eligible & (positions <= train_cutoff))
    inner_train = np.flatnonzero(eligible & (positions <= inner_cutoff))
    inner_validation = np.flatnonzero(
        eligible & (positions >= inner_index) & (positions <= train_cutoff)
    )
    if not len(train) or not len(inner_train) or not len(inner_validation):
        raise ValueError("Past-only full, inner-training and inner-validation cohorts must be nonempty")
    for indices in [train, inner_train, inner_validation]:
        _outcomes(data, history, indices, unique=True)
    metadata = {
        "history_days": int(history),
        "fit_policy": "expanding_history_with_recent_inner_validation",
        "origin_day_utc": start.strftime("%Y-%m-%d"),
        "embargo_days": int(history),
        "inner_validation_calendar_days": int(inner_days),
        "outer_training_last_allowed_day_utc": calendar[train_cutoff].strftime("%Y-%m-%d"),
        "inner_training_last_allowed_day_utc": calendar[inner_cutoff].strftime("%Y-%m-%d"),
        "inner_validation_start_day_utc": inner_start.strftime("%Y-%m-%d"),
    }
    for name, indices in [("train", train), ("inner_train", inner_train),
                           ("inner_validation", inner_validation)]:
        metadata[f"{name}_indices_sha256"] = _digest(indices)
        metadata[f"n_{name}"] = int(len(indices))
        metadata[f"{name}_first_day_utc"] = calendar[indices.min()].strftime("%Y-%m-%d")
        metadata[f"{name}_last_day_utc"] = calendar[indices.max()].strftime("%Y-%m-%d")
    return {
        "train_indices": train,
        "inner_train_indices": inner_train,
        "inner_validation_indices": inner_validation,
        "train_cutoff_index": train_cutoff,
        "inner_train_cutoff_index": inner_cutoff,
        "inner_validation_start_index": inner_index,
        "embargo_days": int(history),
        "split_metadata": metadata,
    }
