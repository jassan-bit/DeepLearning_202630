"""Causal input-corruption stress for frozen OHLCV predictors.

One date-keyed log-price observation error multiplies all OHLC values of a bar;
volume has a separate date-keyed error. Overlapping windows see the same corrupt
bar. This is a measurement sensitivity experiment, not an economic simulator,
an iid return-innovation model, a relabeling operation or a calibration fit.
"""
from __future__ import annotations

from datetime import date
import hashlib
import json
import math

import numpy as np


FAMILY = "ohlcv_observation_noise"
SEVERITIES = {1: .25, 2: .50, 3: 1.0}
NORMAL_CLIP = 3.0
VERSION = "e3-ohlcv-observation-stress-v1"


def _canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _dates(labels, count):
    labels = [str(x) for x in labels]
    if len(labels) != count or labels != sorted(set(labels)):
        raise ValueError("Dates must be unique, ordered and aligned with OHLCV")
    if any(date.fromisoformat(x).isoformat() != x for x in labels):
        raise ValueError("Canonical ISO session/day labels are required")
    return labels


def _raw(values, count=None):
    result = np.asarray(values, dtype=np.float64)
    if (result.ndim != 2 or result.shape[1] != 5
            or (count is not None and len(result) != count)
            or not np.isfinite(result).all()):
        raise ValueError("Finite aligned N by 5 OHLCV required")
    if (result[:, :4] <= 0).any() or (result[:, 4] < 0).any():
        raise ValueError("Positive prices and nonnegative native volume required")
    if ((result[:, 1] < np.max(result[:, [0, 2, 3]], axis=1)).any()
            or (result[:, 2] > np.min(result[:, [0, 1, 3]], axis=1)).any()):
        raise ValueError("OHLC order is incoherent")
    return result


def _scale(values):
    """MAD scale, falling back to population SD, never test-derived floors."""
    values = np.asarray(values, dtype=np.float64)
    mad = float(1.4826 * np.median(np.abs(values - np.median(values))))
    if mad > 0:
        return mad, "1.4826_times_MAD"
    return float(np.std(values, ddof=0)), "population_SD_when_MAD_zero"


def _profile_hash(profile):
    return _sha(_canonical({k: v for k, v in profile.items() if k != "profile_sha256"}))


def fit_noise_profile(raw_fit_Nx5, dates, dataset_id, *, partition="fit",
                      fit_end_declared=None, expected_fit_dates_sha256=None,
                      volume_units="native"):
    """Estimate scale ONLY from the explicitly supplied fit bars.

The caller must select fit rows. The partition marker/boundary are audited
assertions, not independent proof of data provenance. An expected date digest
can be supplied from the frozen split manifest to bind that selection.
"""
    raw = _raw(raw_fit_Nx5)
    if len(raw) < 8 or partition != "fit":
        raise ValueError("At least eight FIT bars are required; cal/test fitting is refused")
    labels = _dates(dates, len(raw))
    if not isinstance(dataset_id, str) or not dataset_id or len(dataset_id) > 128:
        raise ValueError("A fixed dataset identifier is required")
    if volume_units not in {"native", "shares", "ETH", "BTC", "contracts"}:
        raise ValueError("Declare native volume units without conversion")
    if volume_units == "contracts" and not np.equal(raw[:, 4], np.rint(raw[:, 4])).all():
        raise ValueError("Contract counts must be integral before perturbation")
    boundary = labels[-1] if fit_end_declared is None else str(fit_end_declared)
    if date.fromisoformat(boundary).isoformat() != boundary or labels[-1] > boundary:
        raise ValueError("Fit rows extend beyond the declared training boundary")
    dates_sha = _sha(_canonical(labels))
    if expected_fit_dates_sha256 is not None and dates_sha != expected_fit_dates_sha256:
        raise ValueError("Fit dates differ from the frozen training digest")
    price_scale, price_method = _scale(np.diff(np.log(raw[:, 3])))
    volume_scale, volume_method = _scale(np.log1p(raw[:, 4]))
    profile = {
        "version": VERSION, "family": FAMILY, "dataset": dataset_id,
        "partition": "fit", "fit_rows": len(raw), "fit_first_date": labels[0],
        "fit_last_date": labels[-1], "fit_end_declared": boundary,
        "fit_dates_sha256": dates_sha,
        "fit_ohlcv_float64_le_sha256": _sha(np.ascontiguousarray(raw, dtype="<f8").tobytes()),
        "price_scale": price_scale, "price_scale_method": price_method,
        "price_scale_input": "consecutive_Close_log_returns_in_supplied_fit_bars",
        "volume_scale": volume_scale, "volume_scale_method": volume_method,
        "volume_scale_input": "log1p_native_volume_in_supplied_fit_bars",
        "volume_units": volume_units,
        "severity_factors": {str(k): v for k, v in SEVERITIES.items()},
        "standard_normal_truncation": NORMAL_CLIP,
        "selection_assertion": "Caller supplies FIT bars; helper does not access a dataset or targets",
        "fit_date_digest_bound_to_external_manifest": expected_fit_dates_sha256 is not None,
        "independent_provenance_verified": False,
        "scope": "Fixed-target input observation corruption, not an economic generating law",
    }
    profile["profile_sha256"] = _profile_hash(profile)
    return profile


def validate_profile(profile):
    if (not isinstance(profile, dict) or profile.get("version") != VERSION
            or profile.get("family") != FAMILY or profile.get("partition") != "fit"
            or profile.get("profile_sha256") != _profile_hash(profile)):
        raise ValueError("Noise profile identity, fit partition or integrity differs")
    if (profile.get("severity_factors") != {str(k): v for k, v in SEVERITIES.items()}
            or profile.get("standard_normal_truncation") != NORMAL_CLIP
            or type(profile.get("fit_rows")) is not int or profile["fit_rows"] < 8):
        raise ValueError("The prespecified stress matrix changed")
    for key in ("price_scale", "volume_scale"):
        if type(profile.get(key)) not in (int, float) or not math.isfinite(profile[key]) or profile[key] < 0:
            raise ValueError("Finite nonnegative FIT-only noise scales required")
    if (profile.get("fit_first_date", "") > profile.get("fit_last_date", "")
            or profile.get("fit_last_date", "") > profile.get("fit_end_declared", "")):
        raise ValueError("Noise-profile training chronology differs")
    return profile


def _normal(profile, seed, label, component):
    # Include profile and dataset to avoid accidental cross-dataset reuse.
    key = [VERSION, profile["profile_sha256"], profile["dataset"], seed, label, component]
    digest = hashlib.sha256(_canonical(key)).digest()
    u1 = (int.from_bytes(digest[:8], "big") + .5) / 2**64
    u2 = (int.from_bytes(digest[8:16], "big") + .5) / 2**64
    value = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    return max(-NORMAL_CLIP, min(NORMAL_CLIP, value))


def perturb_window(raw7x5, dates7, profile, *, family=FAMILY, severity=1, seed=2026111901):
    """Return (corrupted seven-bar copy, provenance); no target is accepted.

The random key does not include model, model seed, window start or severity.
Consequently every model and every overlapping window receives the same bar
error, scaled monotonically by the three prespecified severity factors.
"""
    validate_profile(profile)
    raw = _raw(raw7x5, count=7)
    labels = _dates(dates7, 7)
    if family != FAMILY or type(severity) is not int or severity not in SEVERITIES:
        raise ValueError("Use the prespecified observation-noise family and severity 1, 2 or 3")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Use an explicit nonnegative integer perturbation seed")
    if labels[-1] < profile["fit_end_declared"]:
        raise ValueError("Noise scale was fitted beyond the requested inference origin")
    factor = SEVERITIES[severity]
    price_errors = np.asarray([_normal(profile, seed, d, "price") for d in labels]) * factor * profile["price_scale"]
    volume_errors = np.asarray([_normal(profile, seed, d, "volume") for d in labels]) * factor * profile["volume_scale"]
    result = raw.copy()
    try:
        with np.errstate(over="raise", invalid="raise"):
            result[:, :4] *= np.exp(price_errors)[:, None]
            result[:, 4] *= np.exp(volume_errors)
    except FloatingPointError:
        raise ValueError("Stress overflows numerical range; no substituted bars were emitted") from None
    if profile["volume_units"] == "contracts":
        result[:, 4] = np.rint(result[:, 4])
    _raw(result, count=7)
    provenance = {
        "family": FAMILY, "severity": severity, "factor": factor, "seed": seed,
        "dataset": profile["dataset"], "profile_sha256": profile["profile_sha256"],
        "origin_date": labels[-1], "window_dates": labels,
        "input_float64_le_sha256": _sha(np.ascontiguousarray(raw, dtype="<f8").tobytes()),
        "perturbed_float64_le_sha256": _sha(np.ascontiguousarray(result, dtype="<f8").tobytes()),
        "log_price_errors": price_errors.tolist(), "log_volume_errors": volume_errors.tolist(),
        "same_error_for_all_OHLC_per_bar": True, "volume_zero_is_preserved": True,
        "contract_volume_rounded_to_nearest_integer": profile["volume_units"] == "contracts",
        "key_excludes_model_and_window_start": True,
        "targets_changed": False, "models_fitted": False,
        "interpretation": "Sensitivity to plausible-order-preserving input corruption; not causal market intervention",
    }
    return result, provenance
