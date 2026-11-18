"""Read-only, integrity-checked dashboard data; no network or training imports."""
from __future__ import annotations

from copy import deepcopy
import csv
from datetime import date, timedelta
import hashlib
import io
import json
import math
from pathlib import Path

DEFAULT_BUNDLE_DIR = Path(__file__).resolve().parents[1] / "data/e3_dashboard_v1"
PARTITIONS = ("training", "calibration", "evaluation")
E3_MODEL = "CNN-LSTM-atencion-aditiva E3"
CLASSICAL = ("Persistence", "SeasonalNaive", "ARIMA")
MODEL_IDS = (E3_MODEL, "FCN-1D", "ResNet-1D", "Inception-1D", "LSTM-autoencoder con predictor",
             "TimesNet", "TSMixer", "TiDE", "TimeMixer", "PatchTST", "iTransformer",
             "MLP16", "LSTM32", "GRU32", *CLASSICAL)
OBSERVATION_NUMBERS = ("target", "last_open", "last_high", "last_low", "last_close",
                       "last_volume", "known_return6", "known_floor")


class DataIntegrityError(ValueError):
    """A bundled file or its temporal contract differs from the reviewed data."""


def _require(condition, message):
    if not condition:
        raise DataIntegrityError(message)


def _json(raw):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            _require(key not in obj, "Campo JSON duplicado en los datos publicados.")
            obj[key] = value
        return obj
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(DataIntegrityError("Valor no finito en los datos.")))


def _checked_files(folder):
    folder = Path(folder)
    _require(folder.is_dir() and not folder.is_symlink(), "No se encontró el paquete de datos verificado.")
    manifest = _json((folder / "manifest.json").read_bytes())
    _require(manifest.get("schema_version") == 1, "Versión de datos no admitida.")
    expected = {"catalogue.json", "observations.csv", "predictions.json", "examples.json", "metadata.json"}
    _require(set(manifest.get("files", {})) == expected, "Inventario de datos incompleto.")
    raw = {}
    for name, record in manifest["files"].items():
        path = folder / name
        _require(path.is_file() and not path.is_symlink(), "Falta un archivo de datos verificado.")
        value = path.read_bytes()
        _require(len(value) == record["bytes"] and hashlib.sha256(value).hexdigest() == record["sha256"],
                 "Los datos publicados no coinciden con su huella de integridad.")
        raw[name] = value
    return manifest, raw


def load_bundle(bundle_dir=None):
    """Load verified local assets. Callers receive fresh values they may filter."""
    manifest, raw = _checked_files(bundle_dir or DEFAULT_BUNDLE_DIR)
    records = list(csv.DictReader(io.StringIO(raw["observations.csv"].decode())))
    observations = []
    seen = set()
    for row in records:
        _require(row["partition"] in PARTITIONS, "Partición desconocida en el paquete.")
        target_date, feature_date = date.fromisoformat(row["target_date"]), date.fromisoformat(row["feature_date"])
        _require((target_date - feature_date).days == 1 and row["target_date"] not in seen,
                 "Las fechas del objetivo y de sus entradas no respetan el contrato temporal.")
        seen.add(row["target_date"])
        parsed = {"target_date": row["target_date"], "feature_date": row["feature_date"], "partition": row["partition"]}
        for key in OBSERVATION_NUMBERS:
            value = float(row[key])
            _require(math.isfinite(value), "Observación no finita.")
            parsed[key] = value
        _require(parsed["target"] >= 0 and parsed["known_floor"] >= 0 and
                 parsed["target"] + 1e-12 >= parsed["known_floor"], "El objetivo no respeta su piso observable.")
        _require(all(parsed[key] > 0 for key in ("last_open", "last_high", "last_low", "last_close")) and
                 parsed["last_volume"] >= 0 and
                 parsed["last_low"] <= min(parsed["last_open"], parsed["last_close"]) <=
                 max(parsed["last_open"], parsed["last_close"]) <= parsed["last_high"],
                 "La observación OHLCV publicada es inválida.")
        observations.append(parsed)
    catalogue_rows = _json(raw["catalogue.json"])
    predictions = _json(raw["predictions.json"])
    examples = _json(raw["examples.json"])
    metadata = _json(raw["metadata.json"])
    ids = [row["model"] for row in catalogue_rows]
    _require(len(ids) == 17 and set(ids) == set(MODEL_IDS), "El catálogo debe conservar los 17 sistemas.")
    _require(len(observations) == 1316 and
             {key: sum(r["partition"] == key for r in observations) for key in PARTITIONS} ==
             {"training": 655, "calibration": 53, "evaluation": 608}, "La cohorte publicada está incompleta.")
    _require(len(predictions["dates"]) == len(predictions["targets"]) == 608 and
             set(predictions["models"]) == set(ids), "Las trayectorias no cubren la familia completa.")
    evaluation = [row for row in observations if row["partition"] == "evaluation"]
    _require([row["target_date"] for row in evaluation] == predictions["dates"] and
             [row["target"] for row in evaluation] == predictions["targets"], "Las trayectorias y la cohorte difieren.")
    for model_id, model in predictions["models"].items():
        _require(len(model["predictions"]) == 608 and
                 all(type(v) in (float, int) and math.isfinite(v) for v in model["predictions"]),
                 "Una trayectoria está incompleta o contiene valores inválidos.")
        _require(model["seed"] == ("deterministic" if model_id in CLASSICAL else 271801),
                 "La curva usa una semilla distinta de la representante fijada.")
        _require(model["q"] is None if model_id == E3_MODEL else
                 type(model["q"]) in (float, int) and math.isfinite(model["q"]) and model["q"] >= 0,
                 "El estado de calibración del predictor difiere del registrado.")
    _require(len(examples) == 3 and {r["id"] for r in examples} == {"evaluation-0", "evaluation-303", "evaluation-607"},
             "El paquete de ejemplos está incompleto.")
    for example in examples:
        text = example["csv_text"]
        _require(hashlib.sha256(text.encode()).hexdigest() == example["csv_sha256"], "La huella del ejemplo CSV no coincide.")
        rows = list(csv.DictReader(io.StringIO(text)))
        target_day = date.fromisoformat(example["target_date"])
        expected_days = [(target_day - timedelta(days=7-i)).isoformat() for i in range(7)]
        _require(len(rows) == 7 and [r["date"] for r in rows] == expected_days,
                 "El ejemplo incluye una fecha futura o una ventana temporal incompleta.")
    return {"catalogue": catalogue_rows, "observations": observations, "predictions": predictions,
            "examples": examples, "metadata": metadata, "mcs": metadata["mcs"],
            "contrasts": metadata["contrasts"], "manifest": manifest}


def catalogue(bundle_dir=None):
    return load_bundle(bundle_dir)["catalogue"]


def filter_observations(partitions=None, start=None, end=None, value_range=None, *, bundle_dir=None):
    """One inclusive mask shared by every EDA chart and its interpretation."""
    partitions = PARTITIONS if partitions is None else tuple(partitions)
    _require(set(partitions).issubset(PARTITIONS), "Filtro de partición inválido.")
    if start is not None:
        date.fromisoformat(start)
    if end is not None:
        date.fromisoformat(end)
    _require(start is None or end is None or start <= end, "El inicio debe ser anterior al fin.")
    if value_range is not None:
        _require(len(value_range) == 2 and all(math.isfinite(float(v)) for v in value_range) and
                 float(value_range[0]) <= float(value_range[1]), "Rango de volatilidad inválido.")
    return [row for row in load_bundle(bundle_dir)["observations"]
            if row["partition"] in partitions and (start is None or row["target_date"] >= start)
            and (end is None or row["target_date"] <= end)
            and (value_range is None or float(value_range[0]) <= row["target"] <= float(value_range[1]))]


def prediction_for(model_id, bundle_dir=None):
    bundle = load_bundle(bundle_dir)
    _require(model_id in bundle["predictions"]["models"], "Modelo desconocido.")
    result = deepcopy(bundle["predictions"]["models"][model_id])
    result.update(dates=bundle["predictions"]["dates"], targets=bundle["predictions"]["targets"])
    return result


def get_example(example_id, bundle_dir=None):
    matches = [row for row in load_bundle(bundle_dir)["examples"] if row["id"] == example_id]
    _require(len(matches) == 1, "Ejemplo desconocido.")
    return matches[0]
