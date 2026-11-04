"""Development-only data and causal preprocessing for the three DL references.

Only the two pre-existing daily CSVs for 2023--2024 are read. Calendar positions
are retained, including the incomplete day, so windows never bridge missing days.
This module does not import a training framework or open raw/test archives.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FEATURES = ("open", "high", "low", "close", "volume")
HISTORIES = (7, 14, 21, 28)
CALENDAR = pd.date_range("2023-01-01", "2025-01-01", inclusive="left", tz="UTC")
SOURCE_PATHS = (
    "results/eda_development/daily_market_development.csv",
    "results/eda_development/daily_volatility_development.csv",
)
EXPECTED_TRAIN_COUNTS = {7: 343, 14: 322, 21: 301, 28: 280}


def _digest_indices(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def _indices(indices: Any, length: int, *, unique: bool = True) -> np.ndarray:
    values = np.asarray(indices)
    if values.ndim != 1:
        raise ValueError("Outcome indices must be one-dimensional")
    if values.size == 0:
        return np.empty(0, dtype=np.int64)
    if values.dtype.kind not in "iu" or values.dtype.kind == "b":
        raise ValueError("Outcome indices must be integers")
    values = values.astype(np.int64, copy=False)
    if np.any(values < 0) or np.any(values >= length):
        raise ValueError("Outcome index outside the development calendar")
    if unique and len(np.unique(values)) != len(values):
        raise ValueError("Training outcomes must not be duplicated")
    return values


def _calendar(data: dict) -> pd.DatetimeIndex:
    calendar = pd.DatetimeIndex(data["calendar"])
    if calendar.hasnans or not calendar.is_unique or not calendar.is_monotonic_increasing:
        raise ValueError("A unique, ordered calendar is required")
    if calendar.tz is None or str(calendar.tz) != "UTC":
        raise ValueError("The calendar must use UTC")
    return calendar


def make_windows(features: np.ndarray, history: int) -> np.ndarray:
    """Return N x D x 5 windows ending strictly before each outcome day."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(FEATURES):
        raise ValueError("Expected the five OHLCV columns")
    if not isinstance(history, (int, np.integer)) or history < 1:
        raise ValueError("History must be a positive integer")
    result = np.full((len(values), int(history), len(FEATURES)), np.nan)
    for outcome in range(int(history), len(values)):
        result[outcome] = values[outcome - history:outcome]
    return result


def load_development(root: Path) -> dict:
    """Load and validate only the established daily development derivatives."""
    root = Path(root)
    market_path, target_path = [root / name for name in SOURCE_PATHS]
    market = pd.read_csv(market_path, usecols=["market_day_utc", *FEATURES])
    targets_frame = pd.read_csv(target_path, usecols=["market_day_utc", "sigma_7"])
    for frame, path in [(market, market_path), (targets_frame, target_path)]:
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("market_day_utc"), utc=True))
        if not frame.index.equals(CALENDAR):
            raise ValueError(f"{path.name}: expected exactly the 2023--2024 UTC calendar")
    features = market.loc[:, list(FEATURES)].to_numpy(dtype=np.float64)
    targets = targets_frame["sigma_7"].to_numpy(dtype=np.float64)
    if np.isinf(features).any() or np.isinf(targets).any():
        raise ValueError("Infinite values are not admissible development observations")
    complete = np.isfinite(features).all(axis=1)
    if CALENDAR[~complete].strftime("%Y-%m-%d").tolist() != ["2023-03-24"]:
        raise ValueError("The established incomplete-day mask has changed")
    observed = features[complete]
    if (np.any(observed[:, :4] <= 0) or np.any(observed[:, 4] < 0)
            or np.any(observed[:, 1] < np.maximum(observed[:, 0], observed[:, 3]))
            or np.any(observed[:, 2] > np.minimum(observed[:, 0], observed[:, 3]))
            or np.any(observed[:, 1] < observed[:, 2])):
        raise ValueError("Invalid OHLCV observation in the development derivatives")
    # Independently check the recorded label's alignment, population convention,
    # and missing-day propagation. This does not read another data source.
    returns = pd.Series(np.log(features[:, 3]), index=CALENDAR).diff()
    expected_target = returns.rolling(7, min_periods=7).std(ddof=0).to_numpy()
    if not np.array_equal(np.isfinite(targets), np.isfinite(expected_target)):
        raise ValueError("sigma_7 missingness does not match the full-calendar target")
    if not np.allclose(targets, expected_target, rtol=1e-9, atol=1e-13, equal_nan=True):
        raise ValueError("sigma_7 differs from the population SD of seven daily log returns")
    windows = {history: make_windows(features, history) for history in HISTORIES}
    eligible = {
        history: np.isfinite(value).all(axis=(1, 2)) & np.isfinite(targets)
        for history, value in windows.items()
    }
    return {
        "calendar": CALENDAR.copy(),
        "features": features,
        "targets": targets,
        "windows_by_history": windows,
        "eligible_by_history": eligible,
        "persistence": np.concatenate(([np.nan], targets[:-1])),
        "source_hashes": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in SOURCE_PATHS
        },
    }


def split_training(data: dict, history: int) -> dict:
    """Static-2023 fit, past-only inner validation, and two 2024 cohorts."""
    calendar = _calendar(data)
    if not calendar.equals(CALENDAR) or history not in HISTORIES:
        raise ValueError("The registered development calendar/history is required")
    eligible = np.asarray(data["eligible_by_history"][history], dtype=bool)
    if eligible.shape != (len(calendar),):
        raise ValueError("Eligibility mask has the wrong shape")
    positions = np.arange(len(calendar))
    first_outer = int(calendar.get_loc(pd.Timestamp("2024-01-01", tz="UTC")))
    first_inner = int(calendar.get_loc(pd.Timestamp("2023-10-01", tz="UTC")))
    train_cutoff = first_outer - history - 1
    inner_cutoff = first_inner - history - 1
    train = np.flatnonzero(eligible & (calendar.year == 2023) & (positions <= train_cutoff))
    inner_train = np.flatnonzero(eligible & (calendar.year == 2023) & (positions <= inner_cutoff))
    inner_validation = np.flatnonzero(
        eligible & (calendar.year == 2023) & (positions >= first_inner) & (positions <= train_cutoff)
    )
    selection = np.flatnonzero(eligible & (calendar.year == 2024) & (calendar.month <= 6))
    calibration = np.flatnonzero(eligible & (calendar.year == 2024) & (calendar.month >= 7))
    validation = np.flatnonzero(eligible & (calendar.year == 2024))
    if len(train) != EXPECTED_TRAIN_COUNTS[history]:
        raise ValueError("Registered training eligibility/count changed")
    if len(selection) != 182 or len(calibration) != 184 or len(validation) != 366:
        raise ValueError("Expected every registered 2024 outcome to be eligible")
    if not len(inner_train) or not len(inner_validation):
        raise ValueError("The internal past-only split must be nonempty")
    return {
        "train_indices": train,
        "inner_train_indices": inner_train,
        "inner_validation_indices": inner_validation,
        "selection_indices": selection,
        "calibration_indices": calibration,
        "validation_indices": validation,
        "train_cutoff_index": train_cutoff,
        "inner_train_cutoff_index": inner_cutoff,
        "inner_validation_start_index": first_inner,
        "embargo_days": int(history),
        "split_metadata": {
            "history_days": int(history),
            "fit_policy": "static_2023_no_daily_refit",
            "outer_training_last_allowed_day_utc": calendar[train_cutoff].strftime("%Y-%m-%d"),
            "inner_training_last_allowed_day_utc": calendar[inner_cutoff].strftime("%Y-%m-%d"),
            "inner_validation_start_day_utc": "2023-10-01",
            "selection_period": ["2024-01-01", "2024-06-30"],
            "calibration_period": ["2024-07-01", "2024-12-31"],
            "train_indices_sha256": _digest_indices(train),
            "inner_train_indices_sha256": _digest_indices(inner_train),
            "inner_validation_indices_sha256": _digest_indices(inner_validation),
        },
    }


def _pointwise_log(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values[..., :4] <= 0) or np.any(values[..., 4] < 0):
        raise ValueError("Log preprocessing requires finite positive OHLC and nonnegative volume")
    transformed = np.empty_like(values)
    transformed[..., :4] = np.log(values[..., :4])
    transformed[..., 4] = np.log1p(values[..., 4])
    return transformed


def fit_preprocessing(data: dict, history: int, indices: Any) -> dict:
    """Fit channel scaling on unique input days of the supplied training cohort."""
    calendar = _calendar(data)
    selected = _indices(indices, len(calendar))
    if not len(selected) or np.any(selected < history):
        raise ValueError("Nonempty eligible training outcomes with complete histories are required")
    eligible = np.asarray(data["eligible_by_history"][history], dtype=bool)
    if not eligible[selected].all():
        raise ValueError("Ineligible outcome supplied for preprocessing fit")
    input_days = np.unique((selected[:, None] - np.arange(history, 0, -1)).ravel())
    transformed = _pointwise_log(np.asarray(data["features"])[input_days])
    mean = transformed.mean(axis=0)
    std = transformed.std(axis=0, ddof=0)
    scale = np.where(std == 0.0, 1.0, std)
    labels = np.asarray(data["targets"], dtype=np.float64)[selected]
    if not np.isfinite(labels).all() or np.any(labels < 0):
        raise ValueError("Training targets must be finite nonnegative volatilities")
    y_scale = float(labels.mean())
    if not np.isfinite(y_scale) or y_scale <= 0:
        raise ValueError("A strictly positive training-target mean is required for target scaling")
    return {
        "history_days": int(history),
        "feature_order": list(FEATURES),
        "feature_transforms": ["log", "log", "log", "log", "log1p"],
        "x_mean": mean.tolist(),
        "x_scale": scale.tolist(),
        "x_std_ddof0": std.tolist(),
        "zero_variance_feature_indices": np.flatnonzero(std == 0.0).tolist(),
        "y_scale": y_scale,
        "target_transform": "divide_by_positive_training_target_mean_without_centering",
        "target_centered": False,
        "fit_outcome_count": int(len(selected)),
        "fit_outcome_indices_sha256": _digest_indices(selected),
        "first_fit_outcome_day_utc": calendar[selected.min()].strftime("%Y-%m-%d"),
        "last_fit_outcome_day_utc": calendar[selected.max()].strftime("%Y-%m-%d"),
        "input_day_indices": input_days.tolist(),
        "input_day_indices_sha256": _digest_indices(input_days),
        "input_day_count": int(len(input_days)),
        "first_input_day_utc": calendar[input_days.min()].strftime("%Y-%m-%d"),
        "last_input_day_utc": calendar[input_days.max()].strftime("%Y-%m-%d"),
        "x_fit_unit": "unique_calendar_input_days_covered_by_training_windows",
    }


def transform_windows(data: dict, history: int, indices: Any, preproc: dict) -> tuple[np.ndarray, np.ndarray]:
    """Apply training-only parameters; return float32 sequences and positive-scaled Y."""
    calendar = _calendar(data)
    selected = _indices(indices, len(calendar), unique=False)
    if preproc.get("history_days", history) != history:
        raise ValueError("Preprocessing history does not match the requested windows")
    if not np.asarray(data["eligible_by_history"][history], dtype=bool)[selected].all():
        raise ValueError("Ineligible outcome supplied for transformation")
    mean = np.asarray(preproc["x_mean"], dtype=np.float64)
    scale = np.asarray(preproc["x_scale"], dtype=np.float64)
    y_scale = float(preproc["y_scale"])
    if (mean.shape != (5,) or scale.shape != (5,) or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)
            or not np.isfinite(y_scale) or y_scale <= 0):
        raise ValueError("Invalid preprocessing parameters")
    raw = np.asarray(data["windows_by_history"][history])[selected]
    x = ((_pointwise_log(raw) - mean) / scale).astype(np.float32)
    y = (np.asarray(data["targets"], dtype=np.float64)[selected] / y_scale).astype(np.float32)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError("Transformed values must be finite, with nonnegative targets")
    return x, y


def mase_scale(data: dict, train_indices: Any) -> float:
    """Mean absolute lag-one-calendar-day differences within the training cohort."""
    calendar = _calendar(data)
    indices = _indices(train_indices, len(calendar))
    targets = np.asarray(data["targets"], dtype=np.float64)
    if targets.shape != (len(calendar),):
        raise ValueError("Target/calendar lengths differ")
    included = np.zeros(len(calendar), dtype=bool)
    included[indices] = True
    consecutive_days = np.asarray((calendar[1:] - calendar[:-1]) == pd.Timedelta(days=1))
    valid = included[1:] & included[:-1] & consecutive_days
    valid &= np.isfinite(targets[1:]) & np.isfinite(targets[:-1])
    if not valid.any():
        return float("nan")
    result = float(np.abs(targets[1:] - targets[:-1])[valid].mean())
    return result if np.isfinite(result) and result > 0 else float("nan")


training_mase_scale = mase_scale
