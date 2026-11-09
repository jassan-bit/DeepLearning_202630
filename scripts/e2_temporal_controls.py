"""Individual temporal-control APIs for E2 §21, with no data loader or search.

Only in-memory daily series are accepted. No CV/selection, market execution,
metrics, durable forecast persistence or production checkpoint format is supplied.
Caller must persist each returned prediction before revealing its observation.
ARIMA requires the explicitly pinned optional statsmodels version below; missing
or mismatched dependencies fail, never trigger installation or another recipe.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import platform
import time
import warnings

import numpy as np
import pandas as pd

PERIODS = (7, 14, 28)
ORDERS = tuple((p, d, q) for p in range(3) for d in range(2) for q in range(3))
STATSMODELS_VERSION = "0.14.5"
DAY = pd.Timedelta(days=1)


class ControlFailure(RuntimeError):
    """Explicit failure with an audit receipt; never a zero-valued forecast."""

    def __init__(self, message, receipt=None):
        super().__init__(message)
        self.receipt = dict(receipt or {}, status="failed", error=message)


def day(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Date cannot be missing")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    if stamp != stamp.normalize():
        raise ValueError("Dates must be daily UTC midnight")
    return stamp


def daily(series):
    """Insert absent calendar dates as NaN, never interpolate or compress."""
    if not isinstance(series, pd.Series) or series.empty:
        raise ValueError("A nonempty dated Series is required")
    index = pd.DatetimeIndex([day(x) for x in series.index])
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Dates must be unique and ordered")
    result = pd.Series(series.to_numpy(dtype=float), index=index)
    return result.reindex(pd.date_range(index[0], index[-1], freq="D"))


def labels(series):
    result = daily(series)
    return result.where(np.isfinite(result) & (result >= 0))


def sigma7(closes, complete):
    """Population std of seven log returns; all eight closes need quality."""
    prices = daily(closes)
    if not isinstance(complete, pd.Series):
        raise ValueError("Explicit daily quality flags are required")
    quality = daily(complete).reindex(prices.index).eq(1)
    valid = quality & np.isfinite(prices) & (prices > 0)
    clean = prices.where(valid)
    returns = np.log(clean).diff()
    return returns.rolling(7, min_periods=7).std(ddof=0).where(
        valid.rolling(8, min_periods=8).sum().eq(8))


def seasonal_naive(history, target, period):
    if type(period) is not int or period not in (1, *PERIODS):
        raise ValueError("Period must be persistence 1 or seasonal 7, 14, 28")
    source = day(target) - period * DAY
    value = labels(history).get(source, np.nan)
    if not np.isfinite(value):
        raise ControlFailure(f"Unavailable lag on {source.date()}")
    return Prediction(day(target).isoformat(), float(value), float(value), False)


def common_mask(history, neural_eligible, targets):
    """Availability only: truth, neural mask, persistence, all seasonal lags.

    Returns one row per planned date, including exclusions and reasons. ARIMA
    forecast failures must stop the candidate, not remove rows from this mask.
    """
    y = labels(history)
    target_days = [day(t) for t in targets]
    if len(set(target_days)) != len(target_days):
        raise ValueError("Duplicate target dates")
    eligible = daily(neural_eligible)
    rows = []
    for t in target_days:
        reasons = []
        if eligible.get(t, np.nan) != 1:
            reasons.append("neural_support")
        for lag in (0, 1, *PERIODS):
            if not np.isfinite(y.get(t - lag * DAY, np.nan)):
                reasons.append("truth" if lag == 0 else f"lag_{lag}")
        rows.append({"date": t, "eligible": not reasons, "reasons": tuple(reasons)})
    return pd.DataFrame(rows, columns=["date", "eligible", "reasons"])


@dataclass(frozen=True)
class Prediction:
    date: str
    raw: float
    value: float
    truncated: bool


def output(target, raw):
    raw = float(raw)
    if not np.isfinite(raw):
        raise ControlFailure("Nonfinite prediction; no fallback permitted")
    return Prediction(day(target).isoformat(), raw, max(0.0, raw), raw < 0)


def _installed_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _backend(expected):
    try:
        actual = version("statsmodels")
    except PackageNotFoundError as exc:
        raise ControlFailure("Optional statsmodels dependency is not installed") from exc
    if actual != expected:
        raise ControlFailure(f"statsmodels version {actual}; required {expected}")
    from statsmodels.tsa.arima.model import ARIMA
    return ARIMA


class FrozenARIMA:
    """One fit through origin−8, then daily filtering with frozen parameters.

    Fit may receive later data, but slices it away before scaling/estimation.
    forecast(t) requires the state exactly at t−1. observe(t, y) is sequential;
    on/after origin, a forecast must have been emitted first. Missing y advances
    the calendar with NaN. Backend/output failures terminally stop this instance;
    argument-validation errors leave it usable. No method refits, retries or selects.
    """

    @classmethod
    def fit(cls, history, origin, order, *, source_commit, expected_version=STATSMODELS_VERSION):
        if (not isinstance(order, tuple) or len(order) != 3
                or any(type(x) is not int for x in order) or order not in ORDERS):
            raise ValueError("Order must be one of the 18 declared specifications")
        if not source_commit or not expected_version:
            raise ValueError("Source revision and exact dependency version required")
        origin = day(origin)
        cutoff = origin - 8 * DAY
        y = labels(history).loc[:cutoff]
        receipt = {"status": "started", "source_commit": source_commit,
                   "order": list(order), "origin": origin.isoformat(),
                   "fit_cutoff": cutoff.isoformat(), "expected_statsmodels": expected_version,
                   "python": platform.python_version(), "numpy": np.__version__,
                   "pandas": pd.__version__, "scipy": _installed_version("scipy"), "retries": 0,
                   "filter_method": "extend, fixed parameters", "warnings": []}
        start = time.perf_counter()
        try:
            if y.empty:
                raise ControlFailure("No training history")
            y = y.reindex(pd.date_range(y.index[0], cutoff, freq="D"))
            receipt.update(history_start=y.index[0].isoformat(), calendar_count=len(y),
                           valid_count=int(y.notna().sum()),
                           mask_sha256=sha256(y.notna().to_numpy().tobytes()).hexdigest(),
                           history_sha256=sha256(y.to_csv().encode()).hexdigest())
            if y.notna().sum() < 90:
                raise ControlFailure("At least 90 valid training labels required")
            scale = float(y.mean())
            if not np.isfinite(scale) or scale <= 0:
                raise ControlFailure("Training mean must be positive")
            receipt["scale"] = scale
            backend = _backend(expected_version)
            options = {"method": "lbfgs", "maxiter": 200, "pgtol": 1e-5,
                       "factr": 1e7, "disp": False}
            receipt.update(statsmodels=expected_version, optimizer=options.copy(),
                           initialization="version-pinned statsmodels defaults; no restarts",
                           trend="c" if order[1] == 0 else "n",
                           seasonal_order=[0, 0, 0, 0], enforce_stationarity=True,
                           enforce_invertibility=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    model = backend(y / scale, order=order, trend=receipt["trend"],
                                    seasonal_order=(0, 0, 0, 0),
                                    enforce_stationarity=True, enforce_invertibility=True)
                    result = model.fit(method="statespace", method_kwargs=options)
                finally:
                    receipt["warnings"] = [str(w.message) for w in caught]
            receipt.update(converged=bool(result.mle_retvals.get("converged", False)),
                           parameters=np.asarray(result.params).tolist(),
                           log_likelihood=float(result.llf))
            if (not receipt["converged"] or not np.isfinite(result.llf)
                    or not np.isfinite(result.params).all()):
                raise ControlFailure("Nonconvergence or nonfinite fitted model")
            receipt["state_at_cutoff"] = np.asarray(result.filtered_state[:, -1]).tolist()
            receipt["state_cov_at_cutoff"] = np.asarray(result.filtered_state_cov[:, :, -1]).tolist()
            receipt.update(status="fitted", duration_seconds=time.perf_counter() - start)
            obj = cls()
            obj._result, obj.scale, obj.origin, obj.current = result, scale, origin, cutoff
            obj.receipt, obj.predictions, obj.observations = receipt, [], []
            obj._pending = None
            obj._failure = None
            return obj
        except Exception as exc:
            receipt["duration_seconds"] = time.perf_counter() - start
            raise ControlFailure(str(exc), receipt) from exc

    def _require_active(self):
        if self._failure is not None:
            raise ControlFailure("Candidate stopped after failure: " + self._failure, self.receipt)

    def _stop(self, exc, stage, date):
        self._failure = str(exc)
        self.receipt.update(status="failed", error=self._failure,
                            failure_stage=stage, failure_date=date.isoformat())
        return ControlFailure(self._failure, self.receipt)

    def forecast(self, target):
        self._require_active()
        target = day(target)
        if target < self.origin or target != self.current + DAY or self._pending is not None:
            raise ValueError("Forecast requires next date, on/after origin, with no pending forecast")
        try:
            prediction = output(target, np.asarray(self._result.forecast(1)).item() * self.scale)
        except Exception as exc:
            raise self._stop(exc, "forecast", target) from exc
        self.predictions.append(asdict(prediction))
        self._pending = target
        return prediction

    def observe(self, date, value):
        self._require_active()
        date = day(date)
        if date != self.current + DAY:
            raise ValueError("Observations must advance exactly one calendar day")
        if date >= self.origin and self._pending != date:
            raise ValueError("Emit and persist forecast before revealing its observation")
        value = float(value) if value is not None else np.nan
        value = value if np.isfinite(value) and value >= 0 else np.nan
        observation = pd.Series([value / self.scale], index=pd.date_range(date, periods=1, freq="D"))
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    result = self._result.extend(observation)
                finally:
                    self.receipt["warnings"].extend(str(w.message) for w in caught)
            if not np.array_equal(np.asarray(result.params), np.asarray(self._result.params)):
                raise ControlFailure("Filtering changed parameters")
        except Exception as exc:
            raise self._stop(exc, "observe", date) from exc
        self._result, self.current, self._pending = result, date, None
        self.observations.append({"date": date.isoformat(), "value": value if np.isfinite(value) else None})

    @property
    def truncation_rate(self):
        return (sum(p["truncated"] for p in self.predictions) / len(self.predictions)
                if self.predictions else None)
