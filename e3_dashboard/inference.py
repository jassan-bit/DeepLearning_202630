"""CPU serving only: strict CSV admission and verified ONNX artifacts.

This module never imports the training pipeline, Torch, or user serialized code.
"""
from __future__ import annotations

import csv
from datetime import date, datetime, timedelta, timezone
import gc
import hashlib
import io
import json
import math
from pathlib import Path
from threading import RLock
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/e3_dashboard_v1"
EXTENSION_ROOT = ROOT / "artifacts/e3_serving_extensions_v2"
CALIBRATION_ROOT = ROOT / "artifacts/e3_serving_calibration_v2"
MAX_UPLOAD_BYTES = 102400
FEATURES = ("open", "high", "low", "close", "volume")
_LOCK = RLock()
_SESSION = None
_SESSION_ID = None


class InputError(ValueError):
    """An actionable, safe validation message for the interface."""


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _registry():
    result, seen = [], set()
    for root in (ARTIFACT_ROOT, EXTENSION_ROOT):
        path = root / "manifest.json"
        if not path.is_file():
            continue
        try:
            document = json.loads(path.read_bytes())
            if document.get("schema_version") != 1:
                raise ValueError()
            for row in document["models"]:
                slug = row["slug"]
                if (row["id"] in seen or not isinstance(slug, str)
                        or Path(slug).name != slug or slug in ("", ".", "..")
                        or (root / slug).is_symlink()):
                    raise ValueError()
                seen.add(row["id"])
                result.append({**row, "_artifact_root": root})
        except (OSError, KeyError, TypeError, ValueError):
            raise RuntimeError("El catálogo de inferencia no es compatible.") from None
    return result


def _metadata(model_id):
    entry = next((x for x in _registry() if x["id"] == model_id), None)
    if entry is None:
        raise InputError("Este modelo está disponible como resultado histórico; su inferencia en vivo no está habilitada.")
    folder = entry["_artifact_root"] / entry["slug"]
    raw = (folder / "metadata.json").read_bytes()
    if _sha(raw) != entry["metadata_sha256"]:
        raise RuntimeError("No se pudo verificar el artefacto de inferencia.")
    return _with_serving_calibration(json.loads(raw), _sha(raw)), folder


def _with_serving_calibration(meta, base_metadata_sha):
    """Adopt a versioned posterior q only for its exact frozen predictor.

    Historical model metadata and diagnostic files remain unchanged. An absent
    adoption profile leaves the point-only capability explicit; corruption of a
    declared profile fails before its interval can be used.
    """
    manifest_path = CALIBRATION_ROOT / "manifest.json"
    if not manifest_path.is_file():
        return meta
    try:
        manifest = json.loads(manifest_path.read_bytes())
        if manifest.get("schema_version") != 1:
            raise ValueError()
        rows = manifest["models"]
        if len({r["model_id"] for r in rows}) != len(rows):
            raise ValueError()
        entry = next((r for r in rows if r["model_id"] == meta["model"]), None)
        if entry is None:
            return meta
        name = entry["file"]
        if Path(name).name != name or not name.endswith(".json"):
            raise ValueError()
        path = CALIBRATION_ROOT / name
        if path.is_symlink():
            raise ValueError()
        raw = path.read_bytes()
        if len(raw) != entry["bytes"] or _sha(raw) != entry["sha256"]:
            raise ValueError()
        profile = json.loads(raw)
        if (profile["schema_version"] != 1 or profile["model_id"] != meta["model"]
                or profile["seed"] != meta["seed"]
                or profile["base_metadata_sha256"] != base_metadata_sha
                or profile["model_sha256"] != meta["model_sha256"]
                or profile["checkpoint_sha256"] != meta["scientific_source"]["checkpoint_sha256"]
                or profile["output_scale"] != meta["output_scale"]
                or profile["output_unit"] != meta["output_unit"]):
            raise ValueError()
        # Reconstruct the saved finite-sample order statistic, without fitting a
        # new q or opening new data. This catches unit, seed and rank mistakes.
        residuals = profile["absolute_calibration_residuals_sigma7"]
        dates = profile["calibration_dates"]
        alpha, q = profile["alpha"], profile["q_sigma7"]
        if (len(residuals) != profile["calibration_n"] or len(dates) != len(residuals)
                or dates != sorted(set(dates)) or len(dates) < 1
                or not all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in residuals)
                or not math.isfinite(alpha) or not 0 < alpha < 1
                or not math.isfinite(q) or q < 0):
            raise ValueError()
        rank = math.ceil((len(residuals) + 1) * (1-alpha))
        if (rank != profile["order_rank"] or rank > len(residuals)
                or sorted(residuals)[rank-1] != q
                or dates[0] != profile["calibration_first_target"]
                or dates[-1] != profile["calibration_last_target"]
                or dates[-1] != profile["applicable_after_target_date"]
                or profile["nominal_coverage"] != 1-alpha
                or profile["refit_model"] is not False or profile["refit_quantile"] is not False):
            raise ValueError()
        for value in dates:
            if date.fromisoformat(value).isoformat() != value:
                raise ValueError()
        if dates[0] <= profile["chronology"]["validation_last_target"]:
            raise ValueError()
    except (OSError, KeyError, TypeError, ValueError, IndexError):
        raise RuntimeError("No se pudo verificar el perfil de calibración del predictor.") from None
    result = dict(meta)
    result.update(calibration_q=q, interval_nominal_coverage=1-alpha,
                  historical_608_coverage_percent_for_this_seed=profile["historical_evaluation"]["coverage_percent"],
                  serving_calibration=profile)
    return result


def interval_metadata(model_id):
    """Public, session-free description of the matching interval capability."""
    meta, _ = _metadata(model_id)
    return {"q": meta["calibration_q"], "nominal_coverage": meta["interval_nominal_coverage"],
            "profile": meta.get("serving_calibration")}


def available_models():
    """Return serving capabilities; reading this list does not load sessions."""
    result = []
    for entry in _registry():
        meta, _ = _metadata(entry["id"])
        result.append({"id": entry["id"], "label": entry["id"], "seed": meta["seed"],
                       "has_interval": meta["calibration_q"] is not None,
                       "explanation_method": "occlusion_raw_window",
                       "latency_measurement_profile": ("legacy_four_cpu_20261117" if entry["_artifact_root"] == ARTIFACT_ROOT
                                                       else "tsmixer_cpu_20261118"),
                       "roundtrip": meta["roundtrip"]})
    return result


def validate_csv(csv_text, target_date, *, today=None):
    """Admit exactly seven completed daily bars before the forecast origin."""
    if not isinstance(csv_text, str):
        raise InputError("Carga un archivo CSV de texto UTF-8.")
    try:
        raw_bytes = csv_text.encode("utf-8")
    except UnicodeError:
        raise InputError("El archivo debe usar texto UTF-8.") from None
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise InputError("El CSV supera el límite de 100 KiB.")
    if "\x00" in csv_text:
        raise InputError("El CSV contiene caracteres no admitidos.")
    try:
        origin = date.fromisoformat(str(target_date))
        if origin.isoformat() != str(target_date):
            raise ValueError()
    except (TypeError, ValueError):
        raise InputError("Selecciona una fecha objetivo con formato AAAA-MM-DD.") from None
    if origin > (today or datetime.now(timezone.utc).date()):
        raise InputError("La fecha objetivo no puede estar en el futuro UTC; usa únicamente barras diarias completas.")
    reader = csv.DictReader(io.StringIO(csv_text.lstrip("\ufeff"), newline=""))
    if reader.fieldnames != ["date", *FEATURES]:
        raise InputError("Las columnas deben ser: date,open,high,low,close,volume, en ese orden.")
    try:
        rows = list(reader)
    except csv.Error:
        raise InputError("No se pudo leer el formato del CSV.") from None
    if len(rows) != 7:
        raise InputError("El CSV debe contener exactamente siete filas de datos.")
    values, dates = [], []
    for i, row in enumerate(rows):
        if set(row) != {"date", *FEATURES} or any(row[k] is None for k in ("date", *FEATURES)):
            raise InputError(f"La fila {i + 1} no tiene las seis columnas requeridas.")
        try:
            observed = date.fromisoformat(row["date"])
            if observed.isoformat() != row["date"]:
                raise ValueError()
            item = [float(row[k]) for k in FEATURES]
        except (TypeError, ValueError):
            raise InputError(f"Revisa la fecha y los valores numéricos de la fila {i + 1}.") from None
        if not all(math.isfinite(v) for v in item):
            raise InputError(f"La fila {i + 1} contiene un valor no finito.")
        opn, high, low, close, volume = item
        if min(opn, high, low, close) <= 0 or volume < 0:
            raise InputError(f"La fila {i + 1} requiere precios positivos y volumen no negativo.")
        if not (low <= min(opn, close) <= max(opn, close) <= high):
            raise InputError(f"La fila {i + 1} no cumple low ≤ open/close ≤ high.")
        expected = origin - timedelta(days=7 - i)
        if observed != expected:
            raise InputError("Las siete fechas deben estar ordenadas, sin huecos ni duplicados, desde t−7 hasta t−1.")
        dates.append(observed.isoformat())
        values.append(item)
    return np.asarray(values, dtype=np.float64), dates


def transform_window(raw, meta):
    """Apply saved fit-only statistics and compute the observable variance floor."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape != (7, 5) or not np.isfinite(raw).all():
        raise InputError("La ventana de entrada no tiene la forma o los valores requeridos.")
    relative = raw.copy()
    relative[:, :4] = np.log(raw[:, :4]) - math.log(float(raw[-1, 3]))
    relative[:, 4] = np.log1p(raw[:, 4])
    scaler = meta["preprocessing"]
    normalized = (relative - np.asarray(scaler["mean"])) / np.asarray(scaler["std"])
    returns = np.diff(np.log(raw[:, 3]))
    floor = float(np.sum((returns - returns.mean()) ** 2) / 7.0)
    return normalized.astype(np.float32)[None], np.asarray([floor], dtype=np.float64)


def clear_cache():
    global _SESSION, _SESSION_ID
    with _LOCK:
        _SESSION = None
        _SESSION_ID = None
        gc.collect()


def _session(model_id, meta, folder):
    global _SESSION, _SESSION_ID
    cache_key = (model_id, meta["model_sha256"])
    if _SESSION_ID == cache_key and _SESSION is not None:
        return _SESSION
    raw = (folder / "model.onnx").read_bytes()
    if _sha(raw) != meta["model_sha256"]:
        raise RuntimeError("No se pudo verificar el artefacto de inferencia.")
    # Clear before construction: one resident session after loading; serialized access.
    _SESSION = None
    _SESSION_ID = None
    gc.collect()
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_cpu_mem_arena = True
    _SESSION = ort.InferenceSession(raw, sess_options=options, providers=["CPUExecutionProvider"])
    _SESSION_ID = cache_key
    return _SESSION


def _run(raw, meta, session):
    x, a = transform_window(raw, meta)
    feeds = {"windows": x}
    if meta["uses_observable_floor"]:
        feeds["variance_floor"] = a
    result = float(session.run(None, feeds)[0].reshape(-1)[0]) * meta["output_scale"]
    if not math.isfinite(result) or result < 0:
        raise RuntimeError("No se obtuvo un pronóstico numérico válido para esta entrada.")
    return result


def predict_csv(model_id, csv_text, target_date):
    """Predict one admitted case, with no logging, fitting or recalibration."""
    raw, _ = validate_csv(csv_text, target_date)
    meta, folder = _metadata(model_id)
    with _LOCK:
        started = perf_counter()
        session = _session(model_id, meta, folder)
        prediction = _run(raw, meta, session)
        latency_ms = (perf_counter() - started) * 1000
        x, _ = transform_window(raw, meta)
        lower = np.asarray(meta["ood"]["minimum_standardized"])
        upper = np.asarray(meta["ood"]["maximum_standardized"])
        outside = (x[0] < lower) | (x[0] > upper)
        from e3_dashboard.explain import occlusion_explanation
        explanation = occlusion_explanation(raw, meta["explanation_reference_raw"],
                                            lambda item: _run(item, meta, session), prediction)
    q = meta["calibration_q"]
    profile = meta.get("serving_calibration")
    scope = (profile["coverage_statement"] if profile else
             "Calibrador histórico del mismo predictor; sin garantía bajo cambio temporal.")
    if profile and str(target_date) <= profile["applicable_after_target_date"]:
        q = None
        scope = ("El calibrador posterior usa datos hasta " + profile["calibration_last_target"]
                 + "; no se aplica a esa fecha objetivo anterior o coincidente.")
    if q is None and not profile:
        scope = "Este predictor no tiene un intervalo calibrado habilitado."
    return {"model_id": model_id, "seed": meta["seed"], "target_date": str(target_date),
            "prediction": prediction,
            "interval": None if q is None else [max(0.0, prediction - q), prediction + q],
            "nominal_coverage": None if q is None else meta["interval_nominal_coverage"],
            "coverage_scope": scope,
            "calibration_profile": None if profile is None else profile["profile"],
            "historical_coverage_percent_for_this_seed": None if profile is None else profile["historical_evaluation"]["coverage_percent"],
            "latency_ms": latency_ms,
            "latency_scope": "carga si procede, preprocesamiento e inferencia; excluye explicación",
            "ood": bool(outside.any()), "ood_fraction": float(outside.mean()),
            "ood_message": "La entrada excede rangos observados en ajuste; el pronóstico es una extrapolación." if outside.any() else "No excede los rangos observados en ajuste; esto no certifica fiabilidad.",
            "explanation": explanation, "explanation_status": "sensitivity_not_causal"}
