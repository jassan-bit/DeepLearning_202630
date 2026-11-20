"""Hash-bound adapter for fourteen accepted historical ONNX representatives.

The frozen five-predictor UI is unchanged. This supplementary profile does not
load V3, refit a scaler/quantile, import Torch or serialize uploaded windows.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path
from threading import RLock
from time import perf_counter

import numpy as np

from e3_dashboard import inference as legacy
from e3_dashboard.explain import occlusion_explanation

ROOT = Path(__file__).resolve().parents[1]
PROFILE = Path(__file__).with_name("profile.json")
LOCK = RLock()
SESSION = None
SESSION_KEY = None


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def profile():
    document = json.loads(PROFILE.read_bytes())
    if (document["schema_version"] != 1 or len(document["models"]) != 14
            or len({r["id"] for r in document["models"]}) != 14
            or document["V3_enabled"] is not False):
        raise RuntimeError("Perfil histórico incompatible.")
    return document


def _file(relative, expected):
    name = Path(relative)
    if name.is_absolute() or ".." in name.parts:
        raise RuntimeError("Ruta de artefacto incompatible.")
    path = ROOT / name
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Falta un artefacto del perfil histórico.")
    raw = path.read_bytes()
    if sha(raw) != expected:
        raise RuntimeError("La huella del artefacto histórico no coincide.")
    return path, raw


def metadata(model_id):
    row = next((r for r in profile()["models"] if r["id"] == model_id), None)
    if row is None:
        raise legacy.InputError("Elige uno de los catorce predictores históricos; V3 no está habilitada.")
    path, raw = _file(row["metadata_path"], row["metadata_sha256"])
    meta = json.loads(raw)
    if meta["model"] != model_id or meta["model_sha256"] != row["model_sha256"]:
        raise RuntimeError("Identidad del predictor incompatible.")
    if row["legacy"]:
        adopted, _ = legacy._metadata(model_id)
        # The unchanged E3 posterior calibration is validated by its original
        # implementation against this exact model and metadata fingerprint.
        meta = adopted
    else:
        # Reuse fit-only E2 range/reference, never target data. These nine models
        # share the exact 655-window fit scaler with the frozen TSMixer profile.
        _, reference_raw = _file(profile()["E2_reference_metadata_path"],
                                 profile()["E2_reference_metadata_sha256"])
        reference = json.loads(reference_raw)
        for key in ("mean", "std", "fit_count", "fit_end", "target_scale"):
            if meta["preprocessing"][key] != reference["preprocessing"][key]:
                raise RuntimeError("La referencia de ajuste no coincide con el escalador.")
        meta = dict(meta)
        if "ood" in meta:
            for bound in ("minimum_standardized", "maximum_standardized"):
                if meta["ood"][bound] != reference["ood"][bound]:
                    raise RuntimeError("Los rangos de ajuste no coinciden.")
        meta["ood"] = reference["ood"]
        meta["explanation_reference_raw"] = reference["explanation_reference_raw"]
        calibrator = meta["calibrator_source"]
        if (calibrator["q"] != meta["calibration_q"] or calibrator["n"] != 53
                or calibrator["date_end"] != row["calibration_valid_after"]
                or calibrator["checkpoint_sha256"] != meta["scientific_source"]["checkpoint_sha256"]):
            raise RuntimeError("El calibrador histórico no corresponde al predictor.")
    return row, meta


def catalogue():
    return [{"id": row["id"], "seed": metadata(row["id"])[1]["seed"],
             "model_sha256": row["model_sha256"],
             "metadata_sha256": row["metadata_sha256"],
             "calibration_valid_after": row["calibration_valid_after"],
             "output_unit": "population sigma7 ddof0, daily, not annualized"}
            for row in profile()["models"]]


def _session(row):
    global SESSION, SESSION_KEY
    key = (row["id"], row["model_sha256"])
    if SESSION is not None and SESSION_KEY == key:
        return SESSION
    _, raw = _file(row["model_path"], row["model_sha256"])
    SESSION, SESSION_KEY = None, None
    gc.collect()
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(raw, sess_options=options, providers=["CPUExecutionProvider"])
    expected_inputs = [("windows", "tensor(float)", [1, 7, 5])]
    if row["uses_observable_floor"]:
        expected_inputs.append(("variance_floor", "tensor(double)", [1]))
    observed = [(i.name, i.type, i.shape) for i in session.get_inputs()]
    expected_outputs = ([("prediction", "tensor(double)", [1])]
                        if row["uses_observable_floor"] else
                        [("normalized_prediction", "tensor(float)", [1])])
    observed_outputs = [(i.name, i.type, i.shape) for i in session.get_outputs()]
    if observed != expected_inputs or observed_outputs != expected_outputs:
        raise RuntimeError("El contrato ONNX no corresponde al perfil.")
    SESSION, SESSION_KEY = session, key
    return SESSION


def _run(raw, meta, row, session):
    x, floor = legacy.transform_window(raw, meta)
    feeds = {"windows": x}
    if row["uses_observable_floor"]:
        feeds["variance_floor"] = floor
    value = float(session.run(None, feeds)[0].reshape(-1)[0]) * meta["output_scale"]
    if not math.isfinite(value) or value < 0:
        raise RuntimeError("El predictor no devolvió una volatilidad válida.")
    return value


def predict(model_id, csv_text, target_date):
    if not isinstance(model_id, str) or not isinstance(target_date, str):
        raise legacy.InputError("Elige un predictor y una fecha objetivo AAAA-MM-DD como texto.")
    raw, _ = legacy.validate_csv(csv_text, target_date)
    row, meta = metadata(model_id)
    with LOCK:
        started = perf_counter()
        session = _session(row)
        value = _run(raw, meta, row, session)
        point_ms = (perf_counter() - started) * 1000
        x, _ = legacy.transform_window(raw, meta)
        outside = ((x[0] < np.asarray(meta["ood"]["minimum_standardized"]))
                   | (x[0] > np.asarray(meta["ood"]["maximum_standardized"])))
        explanation = occlusion_explanation(raw, meta["explanation_reference_raw"],
                        lambda window: _run(window, meta, row, session), value)
    q = meta["calibration_q"] if str(target_date) > row["calibration_valid_after"] else None
    return {"model_id": model_id, "profile": profile()["profile"], "seed": meta["seed"],
            "target_date": str(target_date), "prediction": value, "unit": "sigma7",
            "interval": None if q is None else [max(0.0, value - q), value + q],
            "calibration_q": q, "nominal_coverage": None if q is None else meta["interval_nominal_coverage"],
            "calibration_kind": ("posterior_E3_frozen_profile" if row["uses_observable_floor"]
                                 else "E2_frozen_original"),
            "calibration_profile": meta.get("serving_calibration", {}).get("profile"),
            "calibration_valid_after": row["calibration_valid_after"],
            "coverage_scope": ("Calibrador histórico BTC del mismo predictor; 53 fechas dependientes, sin garantía futura o por subgrupo."
                               if q is not None else "No se aplica un calibrador posterior a la fecha objetivo."),
            "model_sha256": row["model_sha256"], "metadata_sha256": row["metadata_sha256"],
            "profile_sha256": sha(PROFILE.read_bytes()),
            "ood": bool(outside.any()), "ood_fraction": float(outside.mean()),
            "ood_scope": "Rangos observados durante el ajuste; estar dentro no certifica fiabilidad.",
            "explanation": explanation, "explanation_status": "sensitivity_not_causal",
            "latency_ms": point_ms, "latency_scope": "Carga si procede, transformación e inferencia; explicación medida aparte.",
            "new_training_runs": 0, "new_calibration_fits": 0}
