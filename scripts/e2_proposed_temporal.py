"""Prototype chronology guards; no market reads, training, or test scoring."""
from datetime import date, datetime, timedelta
from dataclasses import dataclass
import math


def day(value):
    if not isinstance(value, date) or isinstance(value, datetime):
        raise TypeError('Explicit UTC calendar dates required')
    return value


def har_features(history, origin, windows=(1, 7, 22)):
    """Provisional calendar-day HAR means of sigma7 through origin-1.

    sigma7(t-22) itself needs closes t-29..t-22: extended 29-day reach.
    Missing days fail eligibility; never compress, interpolate or read origin.
    These 1/7/22 windows are a prototype choice, NOT a registered BTC recipe.
    """
    day(origin)
    if not windows or any(type(w) is not int or w < 1 for w in windows):
        raise ValueError('Positive calendar windows required')
    values = []
    for lag in range(1, max(windows) + 1):
        value = history.get(origin - timedelta(days=lag))
        if value is None:
            raise ValueError('Missing calendar label in HAR support')
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError('Invalid sigma7 label')
        values.append(value)
    return tuple(sum(values[:w]) / w for w in windows)


def validate_training_scope(origins, training_origins):
    """Use before scaling, AE pretraining and supervised fitting in EACH fold.

    Caller must supply the actual fold's allowed origins, not full development.
    Guard cannot establish the truth of provenance supplied by a caller.
    """
    observed = tuple(day(x) for x in origins)
    allowed = set(day(x) for x in training_origins)
    if not observed or len(set(observed)) != len(observed):
        raise ValueError('Empty or duplicate fitting origins')
    if not set(observed) <= allowed:
        raise ValueError('Fit/pretraining leaks outside this training fold')


@dataclass(frozen=True)
class OOFResidual:
    origin: date
    fit_label_end: date
    target_available_on: date
    prediction: float
    truth: float


def residual_training_targets(records, training_origins, correction_fit_end):
    """Reject in-sample base predictions and unavailable residual labels.

    Conservative provisional seven-calendar-day purge matches inherited embargo.
    Historical target is sigma7 at origin (not forward seven-day volatility).
    Future protocol must confirm fold windows and embargo semantics explicitly.
    """
    day(correction_fit_end)
    records = tuple(records)  # Iterators must survive both scope and per-record checks.
    validate_training_scope([r.origin for r in records], training_origins)
    residuals = []
    for r in records:
        day(r.fit_label_end); day(r.target_available_on)
        if r.fit_label_end >= r.origin - timedelta(days=7):
            raise ValueError('Base fit violates seven-day temporal purge')
        if r.target_available_on < r.origin or r.target_available_on > correction_fit_end:
            raise ValueError('Residual target unavailable at correction fitting')
        if not all(math.isfinite(x) for x in (r.prediction, r.truth)) or r.truth < 0:
            raise ValueError('Invalid residual record')
        residuals.append(r.truth - r.prediction)
    return tuple(residuals)
