"""ETTh1 data boundary for the external TimeMixer paper replication.

This is an independent, minimal implementation of the published ETT-hour
window/split conventions. It does not import Torch, fit models, inspect scores,
or modify the inherited Bitcoin dataset. The public source CSV is read for
integrity; every fitted transformation uses only rows [0, 8640).
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import io
from pathlib import Path
import operator

import numpy as np
import pandas as pd


CHANNELS = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")
TRAIN_END = 12 * 30 * 24
VALIDATION_END = (12 + 4) * 30 * 24
TEST_END = (12 + 8) * 30 * 24
SPLITS = {"train": (0, TRAIN_END), "val": (TRAIN_END, VALIDATION_END),
          "test": (VALIDATION_END, TEST_END)}


def _positive_integer(value, name, allow_zero=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error
    if result < (0 if allow_zero else 1):
        raise ValueError(f"{name} is outside its allowed range")
    return result


def _matrix(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(CHANNELS):
        raise ValueError("ETTh1 values must have exactly seven channels")
    return values


def _dates(dates, length=None):
    result = pd.DatetimeIndex(dates)
    if length is not None and len(result) != length:
        raise ValueError("Dates and values have different lengths")
    if result.hasnans or not result.is_unique or not result.is_monotonic_increasing:
        raise ValueError("ETTh1 dates must be finite, unique and chronological")
    if result.tz is not None:
        raise ValueError("The original ETTh1 timestamps are timezone-naive")
    # pandas 3 may infer microsecond resolution; asi8 uses the index's own unit.
    nanoseconds = result.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    if len(result) > 1 and not np.all(np.diff(nanoseconds) == 3_600_000_000_000):
        raise ValueError("ETTh1 dates must form a regular hourly calendar")
    return result


def load_etth1(csv_path, expected_sha256=None):
    """Read and validate the official source without deriving evaluation scores.

    Hash verification is performed before CSV parsing. The documented original
    snapshot contains 17,420 rows; extra tail rows are preserved in the returned
    source, while the original 12/4/4-month protocol uses only the first 14,400.
    The returned numeric array is read-only.
    """
    payload = Path(csv_path).read_bytes()
    actual = sha256(payload).hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError("ETTh1 source SHA-256 differs from the registered file")
    frame = pd.read_csv(io.BytesIO(payload))
    if tuple(frame.columns) != ("date", *CHANNELS):
        raise ValueError("ETTh1 source columns differ from the original seven-channel order")
    if len(frame) != 17_420:
        raise ValueError("The registered ETTh1 snapshot must contain 17,420 rows")
    dates = _dates(pd.to_datetime(frame["date"], errors="raise"), len(frame))
    values = _matrix(frame.loc[:, list(CHANNELS)].to_numpy(dtype=np.float64, copy=True))
    if not np.isfinite(values).all():
        raise ValueError("ETTh1 source contains missing or nonfinite numeric values")
    values.setflags(write=False)
    return {"values": values, "dates": dates, "columns": list(CHANNELS),
            "source_sha": actual, "source_sha256": actual, "rows": len(frame),
            "source_timezone": "not specified; original timezone-naive hourly timestamps"}


def fit_scaler(values, train_end=TRAIN_END):
    """Fit StandardScaler semantics on the original training prefix only.

    Population variance (ddof=0) and unit scale for constant features match the
    operational convention. Fitting does not inspect rows after ``train_end``.
    JSON-compatible lists permit a scaler receipt without pickle or sklearn.
    """
    values = _matrix(values)
    train_end = _positive_integer(train_end, "train_end")
    if train_end > len(values):
        raise ValueError("Training prefix exceeds the available source")
    training = values[:train_end]
    if not np.isfinite(training).all():
        raise ValueError("The scaler training prefix contains nonfinite values")
    mean = np.mean(training, axis=0, dtype=np.float64)
    variance = np.var(training, axis=0, ddof=0, dtype=np.float64)
    scale = np.sqrt(variance)
    # StandardScaler also treats numerically constant features as constants.
    epsilon = np.finfo(np.float64).eps
    upper_bound = train_end * epsilon * variance + (train_end * mean * epsilon) ** 2
    scale[variance <= upper_bound] = 1.0
    return {"mean": mean.tolist(), "scale": scale.tolist(),
            "variance": variance.tolist(), "n_samples": train_end,
            "train_end_exclusive": train_end, "ddof": 0}


def calendar_features(dates):
    """Hourly timeF marks, ordered as the original ``utils/timefeatures.py``.

    These are known calendar attributes in the external ETTh1 experiment. Their
    presence does not authorize calendar covariates for the Bitcoin D7 contract.
    """
    dates = _dates(dates)
    result = np.column_stack((dates.hour.to_numpy(dtype=np.float64) / 23.0 - 0.5,
                              dates.day_of_week.to_numpy(dtype=np.float64) / 6.0 - 0.5,
                              (dates.day.to_numpy(dtype=np.float64) - 1.0) / 30.0 - 0.5,
                              (dates.day_of_year.to_numpy(dtype=np.float64) - 1.0) / 365.0 - 0.5))
    return result.astype(np.float32)


@dataclass(frozen=True)
class WindowDataset:
    """Read-only supervised windows using the original ETT-hour boundaries."""

    values: np.ndarray
    marks: np.ndarray
    source_indices: np.ndarray
    forecast_origins: np.ndarray
    dates: pd.DatetimeIndex
    split: str
    seq_len: int
    pred_len: int
    label_len: int
    border_start: int
    border_end: int
    scaler_mean: tuple
    scaler_scale: tuple
    scaler_n_samples: int

    def __len__(self):
        return len(self.forecast_origins)

    @property
    def positions(self):
        return self.forecast_origins

    @property
    def scaler(self):
        return {"mean": list(self.scaler_mean), "scale": list(self.scaler_scale),
                "n_samples": self.scaler_n_samples,
                "train_end_exclusive": self.scaler_n_samples, "ddof": 0}

    def __getitem__(self, index):
        if isinstance(index, (bool, np.bool_)):
            raise IndexError("A window index must be an integer")
        try:
            index = operator.index(index)
        except TypeError as error:
            raise IndexError("A window index must be an integer") from error
        if not 0 <= index < len(self):
            raise IndexError("Window index is outside this split")
        input_end = index + self.seq_len
        target_start = input_end - self.label_len
        target_end = input_end + self.pred_len
        # Copies permit safe Torch collation while preserving immutable storage.
        return (self.values[index:input_end].copy(),
                self.values[target_start:target_end].copy(),
                self.marks[index:input_end].copy(),
                self.marks[target_start:target_end].copy())


def make_split(values, dates, split, seq_len=96, pred_len=96, label_len=0, scaler=None):
    """Build just the requested segment; no targets from other splits are built.

    For L96/H96 the lengths are train=8,449 and val=test=2,785. Validation and
    test reuse only L rows of past context across their nominal boundaries.
    Batching and drop_last are runner policies rather than hidden data filtering.
    """
    values = _matrix(values)
    dates = _dates(dates, len(values))
    if split not in SPLITS:
        raise ValueError("split must be train, val or test")
    seq_len = _positive_integer(seq_len, "seq_len")
    pred_len = _positive_integer(pred_len, "pred_len")
    label_len = _positive_integer(label_len, "label_len", allow_zero=True)
    if label_len > seq_len:
        raise ValueError("label_len cannot exceed the input history")
    if len(values) < TEST_END:
        raise ValueError("The source does not cover the original ETT-hour split")
    if seq_len >= TRAIN_END:
        raise ValueError("Input history exceeds the original training segment")
    scaler = fit_scaler(values) if scaler is None else scaler
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    scale = np.asarray(scaler["scale"], dtype=np.float64)
    sample_count = scaler.get("n_samples", scaler.get("train_end_exclusive"))
    if (mean.shape != (len(CHANNELS),) or scale.shape != mean.shape or
            not np.isfinite(mean).all() or not np.isfinite(scale).all() or
            np.any(scale <= 0) or sample_count != TRAIN_END or scaler.get("ddof", 0) != 0):
        raise ValueError("A finite training-only seven-channel scaler is required")
    nominal_start, end = SPLITS[split]
    start = nominal_start - (seq_len if split != "train" else 0)
    segment = values[start:end]
    if not np.isfinite(segment).all():
        raise ValueError("The requested split contains nonfinite values")
    length = end - start - seq_len - pred_len + 1
    if length < 1:
        raise ValueError("Requested history/horizon leaves no complete windows")
    scaled = np.ascontiguousarray((segment - mean) / scale, dtype=np.float32)
    marks = np.ascontiguousarray(calendar_features(dates[start:end]))
    source_indices = np.arange(start, end, dtype=np.int64)
    origins = np.arange(start + seq_len, start + seq_len + length, dtype=np.int64)
    for array in (scaled, marks, source_indices, origins):
        array.setflags(write=False)
    return WindowDataset(scaled, marks, source_indices, origins, dates[start:end],
                         split, seq_len, pred_len, label_len, start, end,
                         tuple(mean), tuple(scale), int(sample_count))
