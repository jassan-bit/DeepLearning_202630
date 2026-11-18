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
    path = ARTIFACT_ROOT / "manifest.json"
    if not path.is_file():
        return []
    document = json.loads(path.read_bytes())
    if document.get("schema_version") != 1:
        raise RuntimeError("El catálogo de inferencia no es compatible.")
    return document["models"]


def _metadata(model_id):
    entry = next((x for x in _registry() if x["id"] == model_id), None)
    if entry is None:
        raise InputError("Este modelo está disponible como resultado histórico; su inferencia en vivo no está habilitada.")
    folder = ARTIFACT_ROOT / entry["slug"]
    raw = (folder / "metadata.json").read_bytes()
    if _sha(raw) != entry["metadata_sha256"]:
        raise RuntimeError("No se pudo verificar el artefacto de inferencia.")
    return json.loads(raw), folder


def available_models():
    """Return serving capabilities; reading this list does not load sessions."""
    result = []
    for entry in _registry():
        meta, _ = _metadata(entry["id"])
        result.append({"id": entry["id"], "label": entry["id"], "seed": meta["seed"],
                       "has_interval": meta["calibration_q"] is not None,
                       "explanation_method": "occlusion_raw_window",
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
    return {"model_id": model_id, "seed": meta["seed"], "target_date": str(target_date),
            "prediction": prediction,
            "interval": None if q is None else [max(0.0, prediction - q), prediction + q],
            "nominal_coverage": None if q is None else 0.9,
            "coverage_scope": "Calibrador histórico del mismo predictor; sin garantía bajo cambio temporal." if q is not None else "E3 no tiene un intervalo calibrado.",
            "latency_ms": latency_ms,
            "latency_scope": "carga si procede, preprocesamiento e inferencia; excluye explicación",
            "ood": bool(outside.any()), "ood_fraction": float(outside.mean()),
            "ood_message": "La entrada excede rangos observados en ajuste; el pronóstico es una extrapolación." if outside.any() else "No excede los rangos observados en ajuste; esto no certifica fiabilidad.",
            "explanation": explanation, "explanation_status": "sensitivity_not_causal"}
