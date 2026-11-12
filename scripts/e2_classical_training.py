"""Classical inner selection and fresh refits for dated E2 training.

Only supplied in-memory panels are used. ARIMA uses the existing version-pinned
FrozenARIMA implementation, with forecasts emitted before observations. This
module does not open market archives or install dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import hashlib
import json
import time

import numpy as np
import pandas as pd

try:
    from . import e2_real_executor as dated
    from . import e2_temporal_controls as controls
except ImportError:
    import e2_real_executor as dated
    import e2_temporal_controls as controls

VERSION = "e2-classical-training-v1"
SYSTEM_NAMES = ("Ridge", "ARIMA", "SeasonalNaive", "Persistence")
RIDGE_ALPHAS = tuple(float(10.0 ** (-6 + 9*i/17)) for i in range(18))


class ClassicalSelectionFailure(RuntimeError):
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = deepcopy(receipt)


@dataclass(frozen=True)
class ClassicalSelection:
    ridge_alpha: float | None
    arima_order: tuple[int, int, int]
    seasonal_period: int
    ledger: tuple[dict, ...]
    selected_scores: dict
    prepared_signature: str
    arima_domain: tuple[tuple[int, int, int], ...]
    _seal: str = field(init=False, repr=False)

    def _identity(self):
        return {"ridge_alpha": self.ridge_alpha, "arima_order": self.arima_order,
                "seasonal_period": self.seasonal_period, "ledger": self.ledger,
                "selected_scores": self.selected_scores,
                "prepared_signature": self.prepared_signature,
                "arima_domain": self.arima_domain}

    def __post_init__(self):
        object.__setattr__(self, "ledger", tuple(deepcopy(self.ledger)))
        object.__setattr__(self, "selected_scores", deepcopy(self.selected_scores))
        object.__setattr__(self, "_seal", dated.canonical_hash(self._identity()))

    def validate(self):
        if dated.canonical_hash(self._identity()) != self._seal:
            raise ValueError("Classical selection receipt changed after sealing")
        expected = set(SYSTEM_NAMES) - ({"Ridge"} if self.ridge_alpha is None else set())
        if (set(self.selected_scores) != expected
                or {row.get("system") for row in self.ledger} != expected):
            raise ValueError("Selection ledger and scores must agree with the declared Ridge inclusion")

    def record(self):
        self.validate()
        result = {"version": VERSION, "status": "selected_on_inner_validation",
                "ridge_alpha": self.ridge_alpha, "arima_order": list(self.arima_order),
                "seasonal_period": self.seasonal_period,
                "selected_scores": deepcopy(self.selected_scores),
                "prepared_signature": self.prepared_signature,
                "arima_domain": [list(order) for order in self.arima_domain],
                "full_18_order_domain": self.arima_domain == controls.ORDERS,
                "ledger": deepcopy(list(self.ledger))}
        if self.ridge_alpha is None:
            result.update(include_ridge=False, excluded_systems=["Ridge"])
        return result


@dataclass(frozen=True)
class ClassicalRefits:
    systems: tuple
    receipts: dict
    # Live fitted objects are exposed for explicit caller-owned checkpointing.
    # ARIMA receipt/state values alone are an audit record, not a claim that
    # statsmodels can be restored exactly without its versioned fitted result.
    fitted_models: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RidgeFit:
    coefficients: np.ndarray
    intercept: float
    alpha: float

    def predict_scaled(self, x):
        values = _numpy(x).reshape(len(x), -1)
        result = values @ self.coefficients + self.intercept
        if not np.isfinite(result).all():
            raise RuntimeError("Nonfinite Ridge prediction; no fallback")
        return result

    def record(self):
        return {"alpha": self.alpha, "intercept": self.intercept,
                "intercept_penalized": False,
                "coefficient_sha256": hashlib.sha256(self.coefficients.tobytes()).hexdigest(),
                "coefficient_count": len(self.coefficients)}


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _json_safe(value):
    """Keep failed fit diagnostics JSON-safe without dropping their records."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def fit_ridge(split, alpha):
    """Augmented least squares with an unpenalized intercept; no inverse/extra scaler."""
    split.validate()
    if type(alpha) not in (float, int) or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Positive finite Ridge alpha required")
    x = _numpy(split.x).reshape(len(split.dates), -1)
    y = _numpy(split.y)
    if not np.isfinite(x).all() or y.shape != (len(x),) or not np.isfinite(y).all():
        raise ValueError("Ridge inputs must be finite aligned training tensors")
    design = np.column_stack((x, np.ones(len(x))))
    penalty = np.eye(design.shape[1]) * np.sqrt(float(alpha))
    penalty[-1, -1] = 0
    augmented_x = np.concatenate((design, penalty), axis=0)
    augmented_y = np.concatenate((y, np.zeros(design.shape[1])))
    solution = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]
    if not np.isfinite(solution).all():
        raise RuntimeError("Nonfinite Ridge fit")
    coefficients = np.array(solution[:-1], copy=True)
    coefficients.setflags(write=False)
    return RidgeFit(coefficients, float(solution[-1]), float(alpha))


def _history(panel):
    calendar = dated.dated_index(panel.calendar, consecutive=True)
    support = panel.support
    values = np.asarray(panel.targets, dtype=np.float64).copy()
    if values.shape != (len(calendar),):
        raise ValueError("Panel calendar/target alignment differs")
    valid = support.loc[calendar, "target_sigma7_quality"].to_numpy(dtype=bool)
    values[~valid] = np.nan
    return pd.Series(values, index=calendar)


def _score(truth, prediction):
    truth, prediction = _numpy(truth), _numpy(prediction)
    if (truth.ndim != 1 or prediction.shape != truth.shape or not len(truth)
            or not np.isfinite(truth).all() or not np.isfinite(prediction).all()):
        raise ValueError("Complete finite common validation predictions required")
    error = prediction - truth
    mse = float(np.mean(error * error))
    if not np.isfinite(mse):
        raise ValueError("Validation error overflow")
    return float(np.sqrt(mse))


def _validation_days(prepared):
    wanted = tuple(prepared.inner_validation.dates)
    from_mask = tuple(day for day, ok in zip(prepared.inner_plan.evaluation_dates,
                                            prepared.inner_plan.eligible) if ok)
    if wanted != from_mask:
        raise ValueError("Prepared common validation mask changed")
    return wanted


def _observe_value(history, target):
    value = float(history.get(target, np.nan))
    return value if np.isfinite(value) and value >= 0 else None


def _arima_validation(panel, prepared, order, source_commit, prediction_sink=None):
    """Inner fit and sequential filtering, preserving gaps and coefficients."""
    plan = prepared.inner_plan
    history = _history(panel)
    model = controls.FrozenARIMA.fit(history.loc[:plan.cutoff], plan.origin, order,
                                    source_commit=source_commit)
    parameter_fingerprint = dated.canonical_hash(
        {"parameters": np.asarray(model._result.params).tolist(), "scale": model.scale})
    for target in plan.bridge_dates:
        model.observe(target, _observe_value(history, target))
    predictions, events = {}, []
    for target, eligible in zip(plan.evaluation_dates, plan.eligible):
        forecast = model.forecast(target)
        event = {"system": "ARIMA", "order": list(order), "date": target.isoformat(),
                 "eligible": eligible, "raw": forecast.raw, "value": forecast.value,
                 "truncated": forecast.truncated}
        # A supplied durable sink must complete before observation. The default
        # ledger is in memory and is not described as a durable journal.
        if prediction_sink is not None:
            prediction_sink(deepcopy(event))
        events.append(event)
        if eligible:
            predictions[target] = forecast.value
        model.observe(target, _observe_value(history, target))
        now = dated.canonical_hash(
            {"parameters": np.asarray(model._result.params).tolist(), "scale": model.scale})
        if now != parameter_fingerprint:
            raise RuntimeError("ARIMA filtering changed frozen coefficients/scale")
    dates = _validation_days(prepared)
    values = np.asarray([predictions[d] for d in dates], dtype=float)
    return values, {"fit_receipt": deepcopy(model.receipt),
                    "bridge_days": len(plan.bridge_dates),
                    "forecast_events": events,
                    "prediction_sink_supplied": prediction_sink is not None,
                    "coefficient_fingerprint": parameter_fingerprint,
                    "truncation_rate": model.truncation_rate}


def _naive_validation(panel, prepared, period, prediction_sink=None):
    plan = prepared.inner_plan
    full_history = _history(panel)
    past = full_history.loc[:plan.origin - dated.DAY].copy()
    wanted, predictions, events = set(_validation_days(prepared)), {}, []
    for target, eligible in zip(plan.evaluation_dates, plan.eligible):
        raw = past.get(target-period*dated.DAY, np.nan)
        supported = bool(np.isfinite(raw))
        event = {"system": "Persistence" if period == 1 else "SeasonalNaive",
                 "period": period, "date": target.isoformat(), "eligible": eligible,
                 "raw": float(raw) if supported else None,
                 "value": float(raw) if supported else None,
                 "reason": None if supported else "required_calendar_lag_unavailable"}
        if prediction_sink is not None:
            prediction_sink(deepcopy(event))
        events.append(event)
        if target in wanted:
            if not supported:
                raise RuntimeError("Naive missing lag on a frozen eligible validation date")
            predictions[target] = float(raw)
        observed = _observe_value(full_history, target)
        past.loc[target] = np.nan if observed is None else observed
    return np.asarray([predictions[d] for d in _validation_days(prepared)]), events


def select_classical(panel, prepared, *, source_commit, orders=controls.ORDERS,
                     prediction_sink=None, include_ridge=True):
    """Select classical systems without touching outer labels for selection.

    The historical default includes Ridge. With include_ridge=False, neither
    selection nor the subsequent refit executes Ridge; ridge_alpha=None seals
    that exclusion. ARIMA and both naive comparators remain mandatory.
    Default ARIMA exhausts all18 orders. A declared subset is supported only as
    an explicitly recorded bounded fixture; no failed order is replaced.
    """
    if type(include_ridge) is not bool:
        raise ValueError("include_ridge must be an explicit boolean")
    prepared.validate()
    orders = tuple(orders)
    if (not source_commit or not orders or len(set(orders)) != len(orders)
            or any(order not in controls.ORDERS for order in orders)):
        raise ValueError("Unique declared ARIMA orders and source commit required")
    dates = _validation_days(prepared)
    truth = prepared.inner_validation.y_original
    ledger, winners, scores = [], {}, {}
    ridge_successes, arima_successes, seasonal_successes = [], [], []

    for i, alpha in enumerate(RIDGE_ALPHAS if include_ridge else ()):
        start = time.perf_counter()
        row = {"system": "Ridge", "candidate_id": f"ridge-{i:02d}",
               "alpha": alpha, "fit_count": 1, "status": "started"}
        try:
            model = fit_ridge(prepared.inner_train, alpha)
            raw = (model.predict_scaled(prepared.inner_validation.x)
                   * prepared.inner_train.scaler.target_scale)
            value = np.maximum(0.0, raw)
            score = _score(truth, value)
            row.update(status="completed", rmse=score, fit_receipt=model.record(),
                       dates=dated.date_strings(dates),
                       predictions_original=value.tolist(), raw_predictions_original=raw.tolist(),
                       truncation_count=int((raw < 0).sum()))
            ridge_successes.append((score, i, alpha))
        except Exception as error:
            row.update(status="failed", rmse=None, reason=str(error),
                       error_type=type(error).__name__)
        row["elapsed_seconds"] = time.perf_counter()-start
        ledger.append(row)

    for i, order in enumerate(orders):
        start = time.perf_counter()
        row = {"system": "ARIMA", "candidate_id": "arima-"+"-".join(map(str, order)),
               "order": list(order), "fit_count": 1, "status": "started"}
        try:
            value, details = _arima_validation(panel, prepared, order, source_commit,
                                               prediction_sink)
            score = _score(truth, value)
            row.update(status="completed", rmse=score, dates=dated.date_strings(dates),
                       predictions_original=value.tolist(), **details)
            arima_successes.append((score, i, order))
        except Exception as error:
            row.update(status="failed", rmse=None, reason=str(error),
                       error_type=type(error).__name__,
                       failure_receipt=_json_safe(getattr(error, "receipt", None)))
        row["elapsed_seconds"] = time.perf_counter()-start
        ledger.append(row)

    for period in controls.PERIODS:
        start = time.perf_counter()
        row = {"system": "SeasonalNaive", "candidate_id": f"seasonal-{period}",
               "period": period, "fit_count": 0, "status": "started"}
        try:
            value, events = _naive_validation(panel, prepared, period, prediction_sink)
            score = _score(truth, value)
            row.update(status="completed", rmse=score, dates=dated.date_strings(dates),
                       predictions_original=value.tolist(), forecast_events=events)
            seasonal_successes.append((score, period))
        except Exception as error:
            row.update(status="failed", rmse=None, reason=str(error),
                       error_type=type(error).__name__)
        row["elapsed_seconds"] = time.perf_counter()-start
        ledger.append(row)

    persistence = {"system": "Persistence", "candidate_id": "persistence-1",
                   "period": 1, "fit_count": 0, "status": "started"}
    try:
        value, events = _naive_validation(panel, prepared, 1, prediction_sink)
        persistence.update(status="completed", rmse=_score(truth, value),
                           dates=dated.date_strings(dates), predictions_original=value.tolist(),
                           forecast_events=events)
    except Exception as error:
        persistence.update(status="failed", rmse=None, reason=str(error),
                           error_type=type(error).__name__)
    ledger.append(persistence)
    if ((include_ridge and not ridge_successes) or not arima_successes
            or not seasonal_successes or persistence["status"] != "completed"):
        raise ClassicalSelectionFailure(
            "At least one classical family has no valid candidate; no fallback or date deletion",
            {"version": VERSION, "status": "failed", "prepared_signature": prepared.signature,
             "ledger": ledger, "retry_or_fallback": False})
    alpha = None
    selected_scores = {}
    if include_ridge:
        ridge_score, _, alpha = min(ridge_successes)
        selected_scores["Ridge"] = ridge_score
    arima_score, _, order = min(arima_successes)
    seasonal_score, period = min(seasonal_successes)
    selected_scores.update(ARIMA=arima_score, SeasonalNaive=seasonal_score,
                           Persistence=persistence["rmse"])
    return ClassicalSelection(alpha, order, period, tuple(deepcopy(ledger)),
        selected_scores, prepared.signature, orders)


def _ridge_system(model, scaler):
    def fingerprint():
        return {"ridge": model.record(), "scaler": scaler.record()}

    def predict(unused_target, raw_window):
        raw = np.asarray(raw_window, dtype=np.float64)
        if raw.shape != (7, 5):
            raise ValueError("One raw D7 window required")
        x = scaler.transform(raw[None])
        return float(model.predict_scaled(x)[0] * scaler.target_scale)

    return dated.FrozenSystem("Ridge", "window", predict=predict,
                              fingerprint=fingerprint, clip_nonnegative=True)


def refit_classical(panel, prepared, selection, *, source_commit):
    """Fresh fits at outer cutoff, retaining the selection's Ridge inclusion.

    ridge_alpha=None forbids a Ridge refit. Bridge filtering is delegated to
    dated replay; the historical four-system selection remains unchanged.
    """
    prepared.validate()
    if (not isinstance(selection, ClassicalSelection)
            or selection.prepared_signature != prepared.signature
            or (selection.ridge_alpha is not None and selection.ridge_alpha not in RIDGE_ALPHAS)
            or selection.arima_order not in selection.arima_domain
            or selection.seasonal_period not in controls.PERIODS or not source_commit):
        raise ValueError("Selection identity/parameters differ from prepared B")
    selection.validate()
    include_ridge = selection.ridge_alpha is not None
    if include_ridge:
        ridge_start = time.perf_counter()
        ridge = fit_ridge(prepared.refit_train, selection.ridge_alpha)
        ridge_receipt = {**ridge.record(), "fit_count": 1, "fresh_fit": True,
                         "coefficients": ridge.coefficients.tolist(),
                         "cutoff": prepared.outer_plan.cutoff.isoformat(),
                         "scaler": prepared.refit_train.scaler.record(),
                         "elapsed_seconds": time.perf_counter()-ridge_start}
    arima = controls.FrozenARIMA.fit(_history(panel).loc[:prepared.outer_plan.cutoff],
            prepared.outer_plan.origin, selection.arima_order, source_commit=source_commit)
    systems = ((_ridge_system(ridge, prepared.refit_train.scaler),) if include_ridge else ()) + (
               dated.frozen_arima_system(arima, prepared.outer_plan),
               dated.FrozenSystem("SeasonalNaive", "naive", lag=selection.seasonal_period),
               dated.FrozenSystem("Persistence", "naive", lag=1))
    for system in systems:
        system.validate()
    receipts = {"Ridge": ridge_receipt} if include_ridge else {}
    receipts.update({"ARIMA": {**deepcopy(arima.receipt), "fit_count": 1, "fresh_fit": True,
                          "bridge_not_filtered_yet": True},
                "SeasonalNaive": {"period": selection.seasonal_period, "fit_count": 0},
                "Persistence": {"period": 1, "fit_count": 0},
                "selection_ledger": selection.record(), "new_parameter_fits": 2 if include_ridge else 1})
    fitted_models = {"Ridge": ridge, "ARIMA": arima} if include_ridge else {"ARIMA": arima}
    return ClassicalRefits(systems, receipts, fitted_models)
