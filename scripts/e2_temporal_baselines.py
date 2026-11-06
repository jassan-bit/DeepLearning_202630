"""Causal h=1 references for E2 F2a; importing/running this module reads no data.

Inputs are already validated daily sigma_7 labels keyed by UTC ``date``. A
missing calendar day remains a missing observation. No market-file loader,
hyperparameter selector, scoring step or automatic refit is provided here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta
import json
import math
from typing import Mapping
import warnings

DAY = timedelta(days=1)
LabelHistory = Mapping[date, float | None]


def _day(value: date) -> date:
    if not isinstance(value, date) or isinstance(value, datetime):
        raise TypeError("Use calendar date keys in UTC, not timestamps or row indices")
    return value


def _label(history: LabelHistory, day: date) -> float | None:
    """Only touch the requested calendar label; absent/NaN labels stay missing."""
    value = history.get(day)
    if value is None:
        return None
    value = float(value)
    if math.isnan(value):
        return None
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid sigma_7 label at {day.isoformat()}")
    return value


def _calendar(start: date, end: date) -> tuple[date, ...]:
    _day(start)
    _day(end)
    if end < start:
        raise ValueError("Calendar end precedes start")
    return tuple(start + i * DAY for i in range((end - start).days + 1))


@dataclass(frozen=True)
class Forecast:
    model: str
    origin: date
    value: float | None
    eligible: bool
    information_end: date
    effective_close_start: date
    effective_close_end: date
    history_span_days: int
    reason: str | None = None
    raw_value: float | None = None
    postprocessing: str = "none"


def benchmark_nonnegative(forecast: Forecast) -> Forecast:
    """Apply the preregistered point link, retaining the raw forecast for audit.

    This deterministic max(0, raw) rule is fixed before benchmark results. It
    never repairs NaN/Inf, creates an eligible observation or deletes a date.
    ARIMA's accessor remains raw; the benchmark must explicitly use this wrapper
    for ARIMA/Ridge, consistently with architecture_decisions.json and HPO.
    """
    for value in (forecast.value, forecast.raw_value):
        if value is not None and not math.isfinite(float(value)):
            raise ValueError("Nonfinite forecast is a recorded failure, not clipped to zero")
    raw = forecast.raw_value if forecast.raw_value is not None else forecast.value
    if not forecast.eligible:
        return replace(forecast, value=None, raw_value=raw,
                       postprocessing="max(0,raw_forecast)")
    if forecast.value is None or raw is None:
        raise ValueError("An eligible forecast requires a finite value")
    return replace(forecast, value=max(0.0, float(raw)), raw_value=float(raw),
                   postprocessing="max(0,raw_forecast)")


def naive_forecast(history: LabelHistory, origin: date, lag: int = 1) -> Forecast:
    """Predict Y_t by Y_(t-lag); lag is in calendar days, never retained rows.

    lag=1 uses closes [t-8,t-1]. lag=7 uses closes [t-14,t-7]: eight
    effective closes, with a 14-day reach relative to origin t. No intermediate
    unavailable label is interpolated. No covariates or outcome Y_t are read.
    """
    _day(origin)
    if type(lag) is not int or lag not in (1, 7):
        raise ValueError("Registered references use calendar lag 1 or 7 only")
    source_day = origin - lag * DAY
    value = _label(history, source_day)
    return Forecast("naive" if lag == 1 else "seasonal_naive_7", origin,
                    value, value is not None, source_day, source_day - 7 * DAY,
                    source_day, lag + 7,
                    None if value is not None else "required_past_label_missing")


@dataclass(frozen=True)
class FitWindow:
    training_start: date
    fit_cutoff: date
    evaluation_start: date
    embargo_days: int = 7

    def __post_init__(self):
        for day in (self.training_start, self.fit_cutoff, self.evaluation_start):
            _day(day)
        if self.training_start > self.fit_cutoff:
            raise ValueError("Empty ARIMA training calendar")
        if self.embargo_days != 7:
            raise ValueError("The inherited D7 training embargo must remain seven days")
        if self.fit_cutoff > self.evaluation_start - 8 * DAY:
            raise ValueError("Fit labels must end at least eight days before first origin")


@dataclass(frozen=True)
class ArimaConfig:
    order: tuple[int, int, int] = (1, 0, 0)
    maxiter: int = 100
    minimum_observed_training_labels: int = 32

    def __post_init__(self):
        if (len(self.order) != 3 or any(type(n) is not int or n < 0 for n in self.order)
                or self.order[1] > 1):
            raise ValueError("ARIMA order must be nonnegative integers, with d in {0,1}")
        if type(self.maxiter) is not int or self.maxiter < 1:
            raise ValueError("Positive maxiter required")
        if (type(self.minimum_observed_training_labels) is not int
                or self.minimum_observed_training_labels < 8):
            raise ValueError("At least eight observed training labels required")

    @property
    def trend(self) -> str:
        return "c" if self.order[1] == 0 else "n"


def local_arima_grid() -> tuple[ArimaConfig, ...]:
    """Twelve declared local candidates; no winner or equal budget is implied.

    The common HPO protocol hpo.json assigns all twelve with a common resource
    cap before real selection. This function itself never selects or fits.
    """
    return tuple(ArimaConfig((p, d, q))
                 for p in (0, 1, 2) for d in (0, 1) for q in (0, 1))


class FixedParameterARIMA:
    """Fit once per registered outer origin; update past states without refit.

    Parameters/scaling see labels no later than ``fit_cutoff``. At forecast t,
    only labels up to t-1 enter a Kalman state update, with parameters fixed.
    Missing observations are represented by NaN at their true calendar dates;
    they are not removed or materialized as observed imputed labels. Calls must
    have strictly increasing origins. Initialize a new object for each fold.
    """

    def __init__(self, history: LabelHistory, window: FitWindow,
                 config: ArimaConfig = ArimaConfig()):
        import numpy as np
        from statsmodels.tsa.arima.model import ARIMA

        self.window = window
        self.config = config
        self.training_dates = _calendar(window.training_start, window.fit_cutoff)
        values = [_label(history, day) for day in self.training_dates]
        observed = [value for value in values if value is not None]
        if len(observed) < config.minimum_observed_training_labels:
            raise ValueError("Insufficient observed labels in registered training window")
        # Same training-only target mean convention; zero-series scale is exactly 1.
        self.target_scale = float(np.mean(observed)) or 1.0
        scaled = np.array([np.nan if y is None else y / self.target_scale for y in values])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = ARIMA(scaled, order=config.order, trend=config.trend,
                           missing="none", enforce_stationarity=True,
                           enforce_invertibility=True).fit(
                               method_kwargs={"maxiter": config.maxiter})
        self.fit_warnings = [str(item.message) for item in caught]
        if not bool(result.mle_retvals.get("converged", False)):
            raise RuntimeError("ARIMA did not converge; record failure, do not select/test-tune")
        if not np.isfinite(result.params).all():
            raise RuntimeError("ARIMA returned nonfinite fitted parameters")
        self._result = result
        self._latest_state_day = window.fit_cutoff
        self._last_origin: date | None = None
        self.fit_count = 1
        self.observed_training_labels = len(observed)
        self.missing_training_labels = len(values) - len(observed)
        self.state_updates = 0

    def predict(self, history: LabelHistory, origin: date) -> Forecast:
        import numpy as np

        _day(origin)
        if origin < self.window.evaluation_start:
            raise ValueError("Origin precedes the registered evaluation period")
        if self._last_origin is not None and origin <= self._last_origin:
            raise ValueError("Forecast origins must increase; no hindsight or repeated scoring")
        information_end = origin - DAY
        dates = _calendar(self._latest_state_day + DAY, information_end)
        values = [_label(history, day) for day in dates]
        updates = np.array([np.nan if y is None else y / self.target_scale for y in values])
        self._result = self._result.append(updates, refit=False)
        self.state_updates += len(updates)
        self._latest_state_day = information_end
        self._last_origin = origin
        prediction = float(np.asarray(self._result.forecast(steps=1))[0] * self.target_scale)
        if not math.isfinite(prediction):
            raise RuntimeError("ARIMA forecast is nonfinite")
        # A missing latest realized target excludes that forecast from the common
        # observed-information panel; a Kalman prediction is not an observed label.
        eligible = _label(history, information_end) is not None
        return Forecast("ARIMA", origin, prediction if eligible else None, eligible,
                        information_end, self.window.training_start - 7 * DAY,
                        information_end, (origin - self.window.training_start).days + 7,
                        None if eligible else "latest_past_label_missing",
                        raw_value=prediction)

    def receipt(self) -> dict:
        """Configuration/provenance only: no individual labels or predictions."""
        return {"family": "ARIMA", "order": list(self.config.order),
                "trend": self.config.trend, "maxiter": self.config.maxiter,
                "fit_count": self.fit_count,
                "training_start": self.window.training_start.isoformat(),
                "fit_cutoff": self.window.fit_cutoff.isoformat(),
                "evaluation_start": self.window.evaluation_start.isoformat(),
                "fit_embargo_days": self.window.embargo_days,
                "observed_training_labels": self.observed_training_labels,
                "missing_training_labels": self.missing_training_labels,
                "training_calendar_days": len(self.training_dates),
                "target_scaling": "positive training-only mean; all-zero scale=1",
                "state_updates_calendar_days": self.state_updates,
                "refit_after_cutoff": False, "uses_future_covariates": False,
                "missing_policy": "preserve calendar as NaN in state-space model",
                "raw_output_clipping": False,
                "required_benchmark_postprocessing": "benchmark_nonnegative: max(0,raw_forecast)",
                "postprocessing_protocol": "experiments/e2_remediation_20261106/architecture_decisions.json",
                "warnings": self.fit_warnings}


def describe() -> dict:
    return {"route": "F2a time series", "horizon_days": 1,
            "module_execution_reads_market_data": False,
            "default_execution_fits_models": False,
            "references": ["naive_lag1", "seasonal_naive_lag7", "ARIMA"],
            "ARIMA_local_candidates": [asdict(config) for config in local_arima_grid()],
            "common_HPO_budget_registered_by_this_module": False,
            "common_HPO_protocol": "experiments/e2_remediation_20261106/hpo.json",
            "benchmark_postprocessing": "benchmark_nonnegative: max(0,raw_forecast); retain raw",
            "real_Bitcoin_fits_executed_by_this_module": 0}


if __name__ == "__main__":
    print(json.dumps(describe(), indent=2))
