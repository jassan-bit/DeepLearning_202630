"""Training-era-only admission and causal SE-OI features. Never forecast or fit.

Only training_raw/training_y_original/training_dates are deserialized. Hashing
the complete NPZ checks transport, and does not load its other arrays.
"""
from __future__ import annotations
from datetime import date, timedelta
import hashlib
from pathlib import Path
import numpy as np

NPZ_SHA256 = "186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683"
NPZ_BYTES = 460090
NPZ_URL = "https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/8f65eac34aaf49293924ae8328c235cedbdd2866/book_e2/_static/e2_scientific_closure_20261116/data_audit/current_cohort_arrays.npz"
EPSILON = 1e-6
C = 6.0 / 49.0
RECIPES = ("seoi_mse", "input_only_mse", "output_only_mse", "global_mse", "global_crps", "additive_mse")
LOCAL_INPUT = ("seoi_mse", "input_only_mse")
ALLOWED_KEYS = ("training_raw", "training_y_original", "training_dates")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_raw(raw):
    raw = np.asarray(raw, dtype=np.float64)
    require(raw.ndim == 3 and raw.shape[1:] == (7, 5) and len(raw) > 0, "Expected Bx7x5 raw OHLCV")
    require(np.isfinite(raw).all() and (raw[..., :4] > 0).all() and (raw[..., 4] >= 0).all(), "Invalid raw OHLCV")
    require((raw[..., 1] >= np.maximum(raw[..., 0], raw[..., 3])).all() and
            (raw[..., 2] <= np.minimum(raw[..., 0], raw[..., 3])).all(), "Invalid high/low geometry")
    return raw


def known_geometry(raw):
    raw = validate_raw(raw)
    log_prices = np.log(raw[..., :4])
    returns = np.diff(log_prices[..., 3], axis=1)
    mean6 = returns.mean(1)
    A = ((returns - mean6[:, None])**2).sum(1) / 7.0
    rms = np.sqrt((returns**2).mean(1))
    relative = log_prices - log_prices[:, -1, 3][:, None, None]
    return A, rms, relative, returns


def raw_features(raw, reference_rms, *, local):
    raw = validate_raw(raw)
    require(np.isfinite(reference_rms) and reference_rms > 0, "Positive fit reference RMS required")
    A, rms, relative, _ = known_geometry(raw)
    scale = np.maximum(rms, EPSILON * reference_rms)
    divisor = scale if local else np.full_like(scale, reference_rms)
    features = np.empty_like(raw, dtype=np.float64)
    features[..., :4] = relative / divisor[:, None, None]
    features[..., 4] = np.log1p(raw[..., 4])
    require(np.isfinite(features).all(), "Nonfinite causal features")
    return features, A, scale / reference_rms, rms <= EPSILON * reference_rms


def fit_preprocessor(raw_fit):
    _, _, _, returns = known_geometry(raw_fit)
    reference_rms = float(np.sqrt(np.mean(returns**2)))
    require(np.isfinite(reference_rms) and reference_rms > 0, "Degenerate fit reference; no physical epsilon invented")
    record = {"reference_rms": reference_rms, "epsilon_dimensionless": EPSILON, "coordinates": {}}
    for name, local in (("local", True), ("global", False)):
        features, _, _, _ = raw_features(raw_fit, reference_rms, local=local)
        mean, std = features.mean((0, 1)), features.std((0, 1), ddof=0)
        std = np.where(std == 0, 1.0, std)
        record["coordinates"][name] = {"mean": mean.tolist(), "std": std.tolist()}
    return record


def transform(raw, preprocessor, recipe):
    require(recipe in RECIPES, "Unknown recipe")
    local = recipe in LOCAL_INPUT
    coordinates = preprocessor["coordinates"]["local" if local else "global"]
    features, A, scale, binding = raw_features(raw, preprocessor["reference_rms"], local=local)
    X = ((features - np.asarray(coordinates["mean"])) / np.asarray(coordinates["std"])).astype(np.float32)
    require(np.isfinite(X).all(), "Nonfinite standardized causal features")
    return X, A, scale, binding


def prepare_arrays(raw, y, dates, *, enforce_fixed_geometry=True):
    raw = validate_raw(raw)
    y, dates = np.asarray(y, dtype=np.float64), np.asarray(dates).astype(str)
    require(y.shape == dates.shape == (len(raw),) and np.isfinite(y).all() and (y >= 0).all(), "Invalid aligned labels")
    require(dates.tolist() == sorted(set(dates.tolist())), "Dates must be sorted and unique")
    for value in dates:
        require(date.fromisoformat(str(value)).isoformat() == value, "Noncanonical ISO date")
    require(dates[-1] <= "2024-10-31", "Only exposed training-era data admitted")
    fit = np.flatnonzero(dates <= "2024-06-30")
    val = np.flatnonzero((dates >= "2024-07-08") & (dates <= "2024-10-31"))
    require(len(fit) >= 2 and len(val) >= 62, "Insufficient fixed development partitions")
    if enforce_fixed_geometry:
        require(len(fit) == 532 and len(val) == 116, "Expected fit532 / validation116")
    selection, purge, assessment = val[:54], val[54:61], val[61:]
    require(date.fromisoformat(str(dates[selection[0]])) - timedelta(days=7) > date.fromisoformat(str(dates[fit[-1]])), "Fit/selection raw support overlaps")
    require(date.fromisoformat(str(dates[assessment[0]])) - timedelta(days=7) > date.fromisoformat(str(dates[selection[-1]])), "Selection/assessment raw support overlaps")
    A, _, _, _ = known_geometry(raw)
    margin = y**2 - A
    require((margin >= -1e-12).all(), "Target below algebraic floor; no rows dropped")
    U = np.sqrt(np.maximum(margin, 0) / C)
    processor = fit_preprocessor(raw[fit])
    u_scale, y_scale = float(U[fit].mean()), float(y[fit].mean())
    require(u_scale > 0 and y_scale > 0, "Positive fitting label scales required")
    d_fit = float((y[fit] - np.sqrt(A[fit])).mean())
    scalers = {recipe: {"u_scale": u_scale, "y_scale": y_scale, "d_fit": d_fit} for recipe in RECIPES}
    result = {"scalers": scalers, "d_fit": d_fit, "preprocessor": processor}
    binding_counts = {}
    for name, indices in (("train", fit), ("selection", selection), ("assessment", assessment)):
        X = {}
        for recipe in RECIPES:
            X[recipe], _, scales, binding = transform(raw[indices], processor, recipe)
        result[name] = {"X": X, "A": A[indices].copy(), "U": U[indices].copy(),
                        "y": y[indices].copy(), "scale": scales, "dates": dates[indices].copy()}
        binding_counts[name] = int(binding.sum())
    result["admission"] = {"scope": "new_development_on_previously_exposed_calendar",
        "keys_read": list(ALLOWED_KEYS), "forbidden_arrays_loaded": False,
        "npz_sha256": NPZ_SHA256, "fit_rows": len(fit), "selection_rows": len(selection),
        "purged_rows": len(purge), "assessment_rows": len(assessment),
        "fit_dates": dates[fit].tolist(), "selection_dates": dates[selection].tolist(),
        "assessment_dates": dates[assessment].tolist(), "purged_dates": dates[purge].tolist(),
        "source_rows": len(raw),
        "fit_selection_purge_rows": int(((dates > "2024-06-30") & (dates < "2024-07-08")).sum()),
        "selection_assessment_purge_rows": len(purge),
        "total_excluded_rows": len(raw) - len(fit) - len(selection) - len(assessment),
        "assessment_previously_exposed": True, "fit_only_statistics": True,
        "d_fit": d_fit, "u_scale": u_scale, "y_scale": y_scale,
        "preprocessor": processor, "scale_floor_binding_counts": binding_counts,
        "calibration_and_evaluation_arrays_read": False, "confirmation_claimed": False}
    return result


def load_data(npz_path):
    p = Path(npz_path)
    require(not p.is_symlink() and p.is_file(), "Regular NPZ file required")
    require(p.stat().st_size == NPZ_BYTES and sha256(p) == NPZ_SHA256, "NPZ bytes/hash mismatch")
    with np.load(p, allow_pickle=False) as archive:
        raw = np.asarray(archive[ALLOWED_KEYS[0]], dtype=np.float64).copy()
        y = np.asarray(archive[ALLOWED_KEYS[1]], dtype=np.float64).copy()
        dates = np.asarray(archive[ALLOWED_KEYS[2]]).copy()
    return prepare_arrays(raw, y, dates)
