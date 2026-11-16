#!/usr/bin/env python3
"""Audit the D03 dataset package with Python's standard library only.

Package layout (this file is copied to run.py):
  inputs/daily_market_development.csv
  inputs/daily_volatility_development.csv
  inputs/dataset_card_current.json
  inputs/contract_current.md
  inputs/raw_archive_manifest.json
  inputs/partition_audit.json

Offline: python run.py
Optional official raw download: python run.py --download --archive BTCUSDT-1m-2023-01.zip
Without --archive, --download selects the 24 development monthly archives only.
Downloading verifies pinned bytes; it does not create final benchmark X tensors.
An optional inputs/current_dataset_bundle.json admits the separately hash-bound
actual 655/53/608 dataset, including saved NPZ arrays, without NumPy or models.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import date, datetime, timedelta
import hashlib
import html
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import zipfile
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


VERSION = "e2-d03-standalone-dataset-v1"
BASE = Path(__file__).resolve().parent
SOURCES = {
    "daily_market_development.csv": {
        "sha256": "bf6bb5160af924aef6a15c3684aeae932df3017e8639888bb7e91ce787900171",
        "repository_path": "results/eda_development/daily_market_development.csv",
    },
    "daily_volatility_development.csv": {
        "sha256": "20b96374a659eb87dadfa68aa3d6b6b5fb6bea5b10eebe6f2c186ad69861844a",
        "repository_path": "results/eda_development/daily_volatility_development.csv",
    },
    "dataset_card_current.json": {
        "sha256": "43330913c2baf935fd02111f54c651f9c5bc7f99bf5ffcb4067eb5942042d14d",
        "repository_path": "book_e2/_static/e2_point1_review_v1/dataset_card_current.json",
    },
    "contract_current.md": {
        "sha256": "cf5cfa79754496befb9ffec2d3f86a4079ea331b30057c053656bf9a0a25d21f",
        "repository_path": "book_e2/sections/17_contrato_sincronizado.md",
    },
    "raw_archive_manifest.json": {
        "sha256": "af723bcca2caffcbe44ae5243e2863bbcdcd39ae8cd42df491af0a2857926bb2",
        "repository_path": "book_e2/_static/dataset_manifest_e1.json",
    },
    "partition_audit.json": {
        "sha256": "69ac07eebcfad0e946d2edaaf433871e4045696b95035e7b5252cf3409a7e02f",
        "repository_path": "book_e2/_static/e2_eda_current_v1/partition_audit.json",
    },
}
MARKET_HEADER = ("market_day_utc", "open", "high", "low", "close", "volume", "trades")
TARGET_HEADER = ("market_day_utc", "sigma_7", "sigma_14", "sigma_21", "sigma_28")
START, END = date(2023, 1, 1), date(2024, 12, 31)
ABS_TOL, REL_TOL = 1e-13, 1e-9
OFFICIAL_HOST = "data.binance.vision"
OFFICIAL_PATH = "/data/spot/"
CURRENT_BUNDLE = "current_dataset_bundle.json"
CURRENT_VERSION = "e2-standalone-current-dataset-v1"
CURRENT_FILES = {"daily_market_evaluation.csv", "current_cohort_arrays.npz", "audit_result.json",
                 "cohort_array_manifest.json", "reconstructed_training_scaler.json"}
EVALUATION_START, EVALUATION_END = date(2025, 1, 1), date(2026, 8, 31)
EVALUATION_DAILY_SHA = "9563ef557e180f34ee37dce3b1480c38aaefd7063df21c031644973310e28deb"
COHORT_COUNTS = {"training": 655, "calibration": 53, "evaluation": 608}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def calendar(first, last):
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def read_sources(directory):
    payloads, records = {}, []
    for name, source in SOURCES.items():
        path = directory / name
        require(path.is_file() and not path.is_symlink(), "Falta un archivo regular: " + name)
        payload = path.read_bytes()
        require(sha256(payload) == source["sha256"], "SHA256 distinto del insumo fijado: " + name)
        payloads[name] = payload
        records.append({"file": name, "bytes": len(payload), **source, "hash_verified": True})
    return payloads, records


def parse_daily(payload, header, first=START, last=END):
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    require(tuple(reader.fieldnames or ()) == header, "Columnas CSV distintas: " + header[1])
    rows, days = [], []
    for row in reader:
        require(None not in row and all(v is not None for v in row.values()), "Fila CSV incompleta")
        stamp = datetime.fromisoformat(row["market_day_utc"])
        require(stamp.tzinfo is not None and stamp.utcoffset() == timedelta(0), "Fecha sin UTC explícito")
        require((stamp.hour, stamp.minute, stamp.second, stamp.microsecond) == (0, 0, 0, 0),
                "Se requiere medianoche UTC")
        rows.append(row)
        days.append(stamp.date())
    require(days == calendar(first, last),
            "El calendario debe conservar todos los días UTC consecutivos y ordenados")
    return rows, days


def safe_input_path(directory, name):
    """Accept regular descendants only; never follow a bundled symlink."""
    require(isinstance(name, str) and bool(name) and "\\" not in name and ":" not in name,
            "Ruta relativa de bundle inválida")
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and all(part not in ("", ".", "..") for part in name.split("/")),
            "Ruta relativa de bundle inválida")
    current = directory
    for part in relative.parts:
        current = current / part
        require(not current.is_symlink(), "El bundle no admite enlaces simbólicos")
    require(current.is_file() and current.resolve().is_relative_to(directory.resolve()),
            "Falta un archivo regular dentro de inputs: " + name)
    return current


def read_current_bundle(directory):
    path = directory / CURRENT_BUNDLE
    if not path.exists() and not path.is_symlink():
        return None
    payload = safe_input_path(directory, CURRENT_BUNDLE).read_bytes()
    bundle = json.loads(payload)
    require(bundle["version"] == CURRENT_VERSION and bundle["cohorts"] == COHORT_COUNTS,
            "Versión o cohortes del bundle actual distintas")
    rows = bundle["files"]
    require(isinstance(rows, list) and len(rows) == len(CURRENT_FILES), "Se requieren los cinco insumos actuales")
    require({row["path"] for row in rows} == CURRENT_FILES, "Nombres de insumos actuales distintos o repetidos")
    payloads, records = {}, []
    for row in rows:
        name = row["path"]
        source = safe_input_path(directory, name)
        require(type(row["bytes"]) is int and 0 < row["bytes"] <= 16 * 1024 * 1024
                and isinstance(row["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
                and isinstance(row["role"], str) and bool(row["role"].strip()), "Binding de bundle inválido")
        require(source.stat().st_size == row["bytes"], "Tamaño de bundle distinto: " + name)
        data = source.read_bytes()
        require(sha256(data) == row["sha256"], "SHA256 de bundle distinto: " + name)
        payloads[name] = data
        records.append({"file": name, **{k: row[k] for k in ("sha256", "bytes", "role")}, "hash_verified": True})
    require(sha256(payloads["daily_market_evaluation.csv"]) == EVALUATION_DAILY_SHA,
            "La caché diaria de evaluación no es el insumo histórico fijado")
    return bundle, payloads, records, payload


def parse_npy(payload):
    """Read the bounded numeric/Unicode NPY subset used here, without NumPy/pickle."""
    require(payload[:6] == b"\x93NUMPY" and len(payload) >= 10, "Cabecera NPY inválida")
    version = tuple(payload[6:8])
    require(version in ((1, 0), (2, 0), (3, 0)), "Versión NPY no admitida")
    size = 2 if version == (1, 0) else 4
    require(len(payload) >= 8 + size, "Cabecera NPY truncada")
    length = int.from_bytes(payload[8:8 + size], "little")
    require(0 < length <= 65536 and 8 + size + length <= len(payload), "Longitud NPY inválida")
    try:
        header = ast.literal_eval(payload[8 + size:8 + size + length].decode("utf-8" if version == (3, 0) else "latin1"))
    except (ValueError, SyntaxError, UnicodeError) as exc:
        raise ValueError("Cabecera NPY no literal") from exc
    require(isinstance(header, dict) and set(header) == {"descr", "fortran_order", "shape"}
            and header["fortran_order"] is False and isinstance(header["shape"], tuple)
            and len(header["shape"]) <= 3 and all(type(n) is int and 0 < n <= 10000 for n in header["shape"]),
            "Forma u orden NPY no admitido")
    dtype, shape = header["descr"], header["shape"]
    require(dtype in ("<f4", "<f8", "<U10"), "Sólo f4/f8/U10 little-endian, sin objetos, son admisibles")
    width = {"<f4": 4, "<f8": 8, "<U10": 40}[dtype]
    count = math.prod(shape)
    raw = payload[8 + size + length:]
    require(len(raw) == count * width and len(raw) <= 4 * 1024 * 1024, "Bytes de array NPY distintos de su forma")
    if dtype == "<U10":
        try:
            values = [raw[i:i + width].decode("utf-32-le").rstrip("\0") for i in range(0, len(raw), width)]
        except UnicodeError as exc:
            raise ValueError("Fecha Unicode NPY inválida") from exc
        require(all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) for value in values), "ID de fecha NPY inválido")
    else:
        values = [x[0] for x in struct.iter_unpack("<f" if dtype == "<f4" else "<d", raw)]
        require(all(math.isfinite(value) for value in values), "Array NPY no finito")
    # The source array_sha uses np.ascontiguousarray, which promotes a scalar
    # to shape (1,) for hashing; the stored NPY/header shape remains ().
    digest_header = json.dumps({"shape": shape or (1,), "dtype": dtype}, sort_keys=True).encode()
    return {"shape": list(shape), "dtype": dtype, "array_sha256": sha256(digest_header + b"\0" + raw),
            "values": values}


def read_npz(payload, manifest):
    expected = manifest["arrays"]
    require(isinstance(expected, dict) and len(expected) == 21, "Se requieren los 21 arrays del contrato actual")
    result = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = archive.infolist()
        require(len(members) == len(expected) and {x.filename for x in members} == {k + ".npy" for k in expected},
                "Miembros NPZ distintos o duplicados")
        require(sum(x.file_size for x in members) <= 16 * 1024 * 1024, "NPZ descomprimido demasiado grande")
        for member in members:
            require(not member.is_dir() and not member.flag_bits & 1 and 0 < member.file_size <= 4 * 1024 * 1024
                    and "/" not in member.filename and "\\" not in member.filename,
                    "Miembro NPZ no admisible")
            key = member.filename[:-4]
            parsed = parse_npy(archive.read(member))
            require({k: parsed[k] for k in ("shape", "dtype", "array_sha256")} == expected[key],
                    "Array distinto del manifiesto: " + key)
            result[key] = parsed
    return result


def number(text, label):
    value = float(text)
    require(math.isfinite(value), "Valor no finito: " + label)
    return value


def audit_dataset(payloads, source_records):
    card = json.loads(payloads["dataset_card_current.json"])
    partitions = json.loads(payloads["partition_audit.json"])["final_partition"]
    manifest = json.loads(payloads["raw_archive_manifest.json"])
    target_contract = card["target"]
    require(target_contract["ddof"] == 0 and target_contract["return_count"] == 7
            and target_contract["required_consecutive_closes"] == 8
            and target_contract["annualized"] is False
            and target_contract["multiplied_by_100"] is False
            and target_contract["input_history_days"] == 7
            and target_contract["forecast_horizon_days"] == 1,
            "La Data Card no coincide con el contrato poblacional D7/h1")
    require(card["schema"]["model_features"] == ["open", "high", "low", "close", "volume"],
            "Los canales del contrato no coinciden")
    require(manifest["symbol"] == "BTCUSDT" and manifest["interval"] == "1m"
            and len(manifest["archives"]) == 75, "Inventario bruto distinto del fijado")

    market, days = parse_daily(payloads["daily_market_development.csv"], MARKET_HEADER)
    labels, label_days = parse_daily(payloads["daily_volatility_development.csv"], TARGET_HEADER)
    require(days == label_days, "Los calendarios de OHLCV y etiquetas difieren")
    closes, complete, saved_labels = [], [], []
    for row, label, day in zip(market, labels, days):
        fields = [row[k].strip() for k in MARKET_HEADER[1:]]
        if not any(fields):
            closes.append(None)
            complete.append(False)
        else:
            require(all(fields), "OHLCV parcialmente ausente: " + day.isoformat())
            op, high, low, close, volume, trades = [number(v, day.isoformat()) for v in fields]
            require(min(op, high, low, close) > 0 and volume >= 0 and trades >= 0
                    and trades.is_integer() and high >= max(op, close) and low <= min(op, close)
                    and high >= low, "Geometría OHLCV inválida: " + day.isoformat())
            closes.append(close)
            complete.append(True)
        for key in TARGET_HEADER[1:]:
            if label[key].strip():
                require(number(label[key], key) >= 0, "Volatilidad negativa: " + day.isoformat())
        saved_labels.append(number(label["sigma_7"], "sigma_7") if label["sigma_7"].strip() else None)
    invalid_days = [d.isoformat() for d, ok in zip(days, complete) if not ok]
    require(invalid_days == ["2023-03-24"], "Cambió la máscara del día incompleto registrado")

    reconstructed, eligible, errors = [], [], []
    for i, saved in enumerate(saved_labels):
        window = closes[i - 7:i + 1] if i >= 7 else []
        value = None
        if len(window) == 8 and all(c is not None for c in window):
            returns = [math.log(window[k]) - math.log(window[k - 1]) for k in range(1, 8)]
            mean_return = math.fsum(returns) / 7
            value = math.sqrt(math.fsum((r - mean_return) ** 2 for r in returns) / 7)
        require((saved is None) == (value is None), "La máscara de sigma_7 no coincide: " + days[i].isoformat())
        if value is not None:
            require(math.isclose(saved, value, rel_tol=REL_TOL, abs_tol=ABS_TOL),
                    "sigma_7 no coincide con la reconstrucción poblacional: " + days[i].isoformat())
            errors.append(abs(saved - value))
        reconstructed.append(value)
        eligible.append(value is not None and i >= 7 and all(complete[i - 7:i]))

    train_ids = [d.isoformat() for d, ok in zip(days, eligible) if ok and d <= date(2024, 10, 31)]
    cal_ids = [d.isoformat() for d, ok in zip(days, eligible)
               if ok and date(2024, 11, 8) <= d <= date(2024, 12, 30)]
    require(train_ids == partitions["training"]["training_sample_IDs"] and len(train_ids) == 655,
            "La cohorte train final no coincide con los 655 IDs registrados")
    require(cal_ids == partitions["calibration"]["sample_IDs"] and len(cal_ids) == 53,
            "La cohorte de calibración no coincide con los 53 IDs registrados")
    require(set(train_ids).isdisjoint(cal_ids), "Claves de objetivo train/cal repetidas")
    require(partitions["evaluation"]["count"] == 608, "El metadato de evaluación no coincide")
    by_year = {str(year): sum(d.year == year and v is not None for d, v in zip(days, saved_labels))
               for year in (2023, 2024)}
    require(by_year == {"2023": 350, "2024": 366}, "Conteos de etiquetas distintos de los registrados")

    joined = io.StringIO(newline="")
    fields = ["outcome_day_utc", *MARKET_HEADER[1:], "sigma_7", "sigma_7_reconstructed",
              "complete_daily_OHLCV", "D7_h1_eligible", "current_partition_role"]
    writer = csv.DictWriter(joined, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    train_set, cal_set = set(train_ids), set(cal_ids)
    for i, day in enumerate(days):
        key = day.isoformat()
        role = ("final_training" if key in train_set else "calibration" if key in cal_set
                else "causal_bridge_not_fit" if key == "2024-12-31"
                else "embargo_not_fit" if "2024-11-01" <= key <= "2024-11-07"
                else "ineligible_missing_support")
        writer.writerow({"outcome_day_utc": key, **{k: market[i][k] for k in MARKET_HEADER[1:]},
                         "sigma_7": labels[i]["sigma_7"],
                         "sigma_7_reconstructed": repr(reconstructed[i]) if reconstructed[i] is not None else "",
                         "complete_daily_OHLCV": str(complete[i]).lower(),
                         "D7_h1_eligible": str(eligible[i]).lower(), "current_partition_role": role})
    dataset = joined.getvalue().encode("utf-8")
    report = {
        "version": VERSION, "status": "passed", "mode": "offline_development_validation",
        "inputs": source_records,
        "contract": {"symbol": "BTCUSDT", "timezone": "UTC", "input_shape": [7, 5],
                     "target": "population SD of seven consecutive daily Close log returns",
                     "ddof": 0, "annualized": False, "multiplied_by_100": False,
                     "horizon_days": 1, "target_lags_added_as_features": False,
                     "window": "OHLCV[t-7:t], excluding outcome day t"},
        "development": {"calendar_start": START.isoformat(), "calendar_end": END.isoformat(),
                        "calendar_rows": len(days), "complete_daily_OHLCV_rows": sum(complete),
                        "incomplete_daily_dates": invalid_days,
                        "valid_sigma7_labels": len(errors), "valid_sigma7_labels_by_year": by_year,
                        "sigma7_mask_matches": True, "sigma7_values_match": True,
                        "sigma7_reconstruction_max_absolute_difference": max(errors),
                        "absolute_tolerance": ABS_TOL, "relative_tolerance": REL_TOL,
                        "current_final_training_ID_count": len(train_ids),
                        "current_calibration_ID_count": len(cal_ids),
                        "current_training_and_calibration_IDs_match_saved_metadata": True,
                        "calendar_gaps_compressed": False, "missing_values_imputed": False},
        "outputs": {"development_dataset.csv": {
                        "rows": len(days), "sha256": sha256(dataset),
                        "role": "Daily source audit table, not a model input tensor; OHLCV on outcome day t must not be used as X for t."},
                    "daily_market_development.csv": {"exact_copy_of_included_source": True},
                    "daily_volatility_development.csv": {"exact_copy_of_included_source": True},
                    "dataset_audit.json": {}, "dataset_audit.html": {}},
        "raw_inventory": {"archive_count": len(manifest["archives"]),
                          "manifest_is_included": True, "raw_archives_included": False,
                          "official_download_requires_explicit_download_flag": True},
        "limits": [
            "Los dos CSV numéricos incluidos cubren sólo desarrollo 2023–2024; las 608 fechas de evaluación son metadatos en este paquete.",
            "El CSV unido es una tabla diaria de auditoría: OHLCV del día t no es X para ese objetivo. La entrada causal usa sólo los siete días anteriores.",
            "El tensor X final completo, la caché OHLCV final y los escaladores ajustados no se incluyen ni se fabrican.",
            "La descarga oficial opcional verifica los ZIP fijados; este ejecutor no los agrega para crear X final ni ajusta transformaciones.",
            "La validez de datos y sus hashes no certifican ausencia global de fuga en cada port ni generalización a mercados futuros.",
            "La Data Card incluida conserva sus metadatos históricos; los 655 IDs train actuales se vinculan a partition_audit.json y se comprueban aquí.",
            "El manifiesto bruto incluye septiembre–octubre de 2026, fuera de la evaluación actual; ese periodo no se descarga por defecto.",
            "La reutilización y redistribución conservan los términos de Binance Vision y los límites por material descritos en la Data Card incluida.",
        ],
        "operations": {"new_model_fits": 0, "model_inference_calls": 0, "new_predictive_scores": 0,
                       "hypothesis_tests": 0, "bootstrap_draws": 0,
                       "source_files_modified": False, "raw_archives_opened": 0},
        "downloads": [],
    }
    return report, dataset, manifest


def audit_current_dataset(original, current):
    """Check the admitted current arrays and daily chronology, not model ports."""
    bundle, payloads, records, bundle_bytes = current
    audit = json.loads(payloads["audit_result.json"])
    manifest = json.loads(payloads["cohort_array_manifest.json"])
    scaler = json.loads(payloads["reconstructed_training_scaler.json"])
    npz_sha = sha256(payloads["current_cohort_arrays.npz"])
    require(audit["actual_cohorts_audited"] == COHORT_COUNTS and audit["data_ready_for_bounded_controls"] is True
            and audit["arrays_npz_sha256"] == npz_sha and manifest["sha256"] == npz_sha
            and audit["array_keys"] == bundle["array_keys"], "El bundle, audit y manifiesto no vinculan las mismas cohortes/NPZ")
    require(audit["reconstructed_scaler_sha256"] == sha256(payloads["reconstructed_training_scaler.json"]),
            "El escalador incluido difiere del audit admitido")
    arrays = read_npz(payloads["current_cohort_arrays.npz"], manifest)
    dev_rows, dev_days = parse_daily(original["daily_market_development.csv"], MARKET_HEADER)
    eval_rows, eval_days = parse_daily(payloads["daily_market_evaluation.csv"], MARKET_HEADER[:-1],
                                       EVALUATION_START, EVALUATION_END)
    all_rows, all_days = dev_rows + eval_rows, dev_days + eval_days
    raw_daily, closes = [], []
    for i, row in enumerate(all_rows):
        texts = [row[k].strip() for k in MARKET_HEADER[1:6]]
        if not any(texts):
            require(i < len(dev_days), "Barra de evaluación ausente")
            raw_daily.append(None)
            closes.append(None)
            continue
        require(all(texts), "Barra OHLCV parcialmente ausente")
        values = [number(text, all_days[i].isoformat()) for text in texts]
        op, high, low, close, volume = values
        require(min(op, high, low, close) > 0 and volume >= 0
                and low <= min(op, close) and high >= max(op, close) and high >= low,
                "Geometría OHLCV actual inválida")
        raw_daily.append(values)
        closes.append(close)
    reconstructed = []
    for i in range(len(closes)):
        window = closes[i - 7:i + 1] if i >= 7 else []
        value = None
        if len(window) == 8 and all(c is not None for c in window):
            returns = [math.log(window[k]) - math.log(window[k - 1]) for k in range(1, 8)]
            mean = math.fsum(returns) / 7
            value = math.sqrt(math.fsum((r - mean) ** 2 for r in returns) / 7)
        reconstructed.append(value)
    partition = json.loads(original["partition_audit.json"])["final_partition"]
    expected_dates = {"training": partition["training"]["training_sample_IDs"],
                      "calibration": partition["calibration"]["sample_IDs"],
                      "evaluation": [d.isoformat() for d in eval_days]}
    positions = {day.isoformat(): i for i, day in enumerate(all_days)}
    used_keys, cohort_records, all_ids = set(), {}, set()
    evaluation_y = None
    for split, count in COHORT_COUNTS.items():
        keys = bundle["array_keys"][split]
        shapes = {"dates": ([count], "<U10"), "raw": ([count, 7, 5], "<f8"),
                  "X_relative": ([count, 7, 5], "<f8"), "X_scaled": ([count, 7, 5], "<f4"),
                  "y_original": ([count], "<f8"), "y_scaled": ([count], "<f4")}
        require(set(keys) == set(shapes) and len(set(keys.values())) == 6,
                "Claves de arrays de cohorte inválidas")
        for role, (shape, dtype) in shapes.items():
            key = keys[role]
            require(key not in used_keys and key in arrays and arrays[key]["shape"] == shape
                    and arrays[key]["dtype"] == dtype, "Forma/dtype o clave de array incorrectos: " + split + "/" + role)
            used_keys.add(key)
        ids = arrays[keys["dates"]]["values"]
        require(ids == expected_dates[split] and len(ids) == count and len(set(ids)) == count
                and all_ids.isdisjoint(ids), "Los IDs D7 de cohorte no coinciden o se repiten")
        all_ids.update(ids)
        raw = arrays[keys["raw"]]["values"]
        targets = arrays[keys["y_original"]]["values"]
        target_errors, raw_errors = [], []
        for row_number, (day, saved) in enumerate(zip(ids, targets)):
            require(day in positions, "ID fuera del calendario diario")
            i = positions[day]
            expected_raw = raw_daily[i - 7:i] if i >= 7 else []
            value = reconstructed[i]
            require(len(expected_raw) == 7 and all(r is not None for r in expected_raw) and value is not None,
                    "Falta soporte causal de siete barras/ocho cierres: " + day)
            require(saved >= 0 and math.isclose(saved, value, rel_tol=REL_TOL, abs_tol=ABS_TOL),
                    "σ7 del NPZ difiere de la reconstrucción diaria: " + day)
            target_errors.append(abs(saved - value))
            for actual, expected in zip(raw[row_number * 35:(row_number + 1) * 35],
                                        [v for bar in expected_raw for v in bar]):
                # Python's decimal decoder and the original pandas default parser
                # can differ by a few FP64 ulps; the original bytes/hashes stay exact.
                require(math.isclose(actual, expected, rel_tol=2e-15, abs_tol=1e-12),
                        "Ventana raw no coincide con OHLCV anterior al objetivo: " + day)
                raw_errors.append(abs(actual - expected))
        cohort_records[split] = {"count": count, "first_outcome_day": ids[0], "last_outcome_day": ids[-1],
                                 "outcome_IDs_match": True, "raw_windows_match_daily_history": True,
                                 "sigma7_matches_daily_reconstruction": True,
                                 "maximum_sigma7_absolute_difference": max(target_errors),
                                 "maximum_raw_absolute_difference": max(raw_errors)}
        if split == "evaluation":
            evaluation_y = targets
    extras = {"training_target_scale": ([], [scaler["target_scale"]]),
              "training_channel_mean": ([5], scaler["mean"]), "training_channel_std": ([5], scaler["std"])}
    require(set(arrays) == used_keys | set(extras), "Arrays adicionales no declarados")
    require(scaler["train_dates"] == expected_dates["training"]
            and isinstance(scaler["training_sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", scaler["training_sha256"]), "Fechas/firma del escalador distintas")
    for key, (shape, values) in extras.items():
        require(arrays[key]["shape"] == shape and arrays[key]["dtype"] == "<f8"
                and arrays[key]["values"] == values, "Escalador JSON y arrays no coinciden: " + key)
    require(scaler["target_scale"] > 0 and all(v > 0 for v in scaler["std"]), "Escala no positiva")
    joined = io.StringIO(newline="")
    fields = ["outcome_day_utc", *MARKET_HEADER[1:6], "sigma_7", "sigma_7_reconstructed",
              "input_first_day_utc", "input_last_day_utc", "current_partition_role"]
    writer = csv.DictWriter(joined, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for day, row, saved in zip(eval_days, eval_rows, evaluation_y):
        writer.writerow({"outcome_day_utc": day.isoformat(), **{k: row[k] for k in MARKET_HEADER[1:6]},
                         "sigma_7": repr(saved), "sigma_7_reconstructed": repr(reconstructed[positions[day.isoformat()]]),
                         "input_first_day_utc": (day - timedelta(days=7)).isoformat(),
                         "input_last_day_utc": (day - timedelta(days=1)).isoformat(),
                         "current_partition_role": "inherited_known_evaluation"})
    dataset = joined.getvalue().encode()
    report = {"version": CURRENT_VERSION, "status": "passed", "mode": "offline_current_dataset_validation",
              "bundle_sha256": sha256(bundle_bytes), "inputs": records, "cohorts": cohort_records,
              "evaluation": {"calendar_start": EVALUATION_START.isoformat(), "calendar_end": EVALUATION_END.isoformat(),
                             "calendar_rows": 608, "complete_daily_OHLCV_rows": 608, "same_inherited_known_test": True},
              "arrays": {"file": "current_cohort_arrays.npz", "sha256": npz_sha, "count": len(arrays),
                         "NPY_headers_dtypes_shapes_and_content_hashes_verified": True,
                         "manifest_array_hashes_recomputed_without_numpy": True, "pickle_loading": False},
              "tolerances": {"sigma7_absolute": ABS_TOL, "sigma7_relative": REL_TOL,
                             "raw_CSV_to_array_absolute": 1e-12, "raw_CSV_to_array_relative": 2e-15,
                             "included_file_and_array_hashes": "exact"},
              "scaler": {"included_JSON_matches_saved_array_coefficients_and_train_dates": True,
                         "fitting_repeated": False, "model_specific_scalers_checked": False},
              "outputs": {"evaluation_dataset.csv": {"rows": 608, "sha256": sha256(dataset)},
                          **{name: {"exact_copy_of_included_source": True} for name in sorted(CURRENT_FILES)},
                          "current_dataset_bundle.json": {"exact_copy_of_included_source": True}},
              "limits": ["Es la evaluación histórica heredada y conocida de 608 fechas; no es una muestra ciega nueva.",
                         "El CSV diario incluye OHLCV del objetivo t sólo para auditoría; X[t] usa los siete días anteriores.",
                         "Se comprueban hashes, cabeceras, geometría, IDs, ventanas raw y σ7. No se recalculan aquí las transformaciones X_relative/X_scaled ni el ajuste del escalador.",
                         "La coherencia de datos no certifica ausencia global de fuga, los ports de modelos, sus checkpoints o todos los escaladores físicos.",
                         "No se abren ZIP brutos de mercado ni se producen pronósticos, métricas predictivas o pruebas estadísticas."],
              "operations": {"new_model_fits": 0, "model_inference_calls": 0, "new_predictive_scores": 0,
                             "hypothesis_tests": 0, "bootstrap_draws": 0, "raw_market_archives_opened": 0}}
    return report, dataset


def current_html_report(report):
    rows = "".join("<tr><th>" + html.escape(name) + "</th><td>" + str(row["count"]) + "</td><td>"
                   + html.escape(row["first_outcome_day"] + " a " + row["last_outcome_day"]) + "</td></tr>"
                   for name, row in report["cohorts"].items())
    limits = "".join("<li>" + html.escape(text) + "</li>" for text in report["limits"])
    return ("<!doctype html><html lang='es'><meta charset='utf-8'><title>Dataset actual E2</title>"
            "<body><h1>Dataset E2 actual: validación autónoma</h1><p>Estado: passed. Cohortes reales 655/53/608; "
            "σ₇ poblacional, ddof 0, ocho cierres y ventanas anteriores de 7×5. Sin modelos ni nuevos scores.</p>"
            "<table><tr><th>Cohorte</th><th>IDs</th><th>Fechas UTC</th></tr>" + rows + "</table>"
            "<p><a href='evaluation_dataset.csv'>Auditoría diaria de evaluación</a> · "
            "<a href='daily_market_evaluation.csv'>OHLCV original</a> · <a href='current_cohort_arrays.npz'>Arrays originales</a> · "
            "<a href='current_dataset_audit.json'>Informe JSON</a></p><ul>" + limits + "</ul></body></html>\n")


def validate_official_url(url):
    parsed = urlparse(url)
    require(parsed.scheme == "https" and parsed.hostname == OFFICIAL_HOST
            and parsed.port in (None, 443) and not parsed.username and not parsed.password
            and parsed.path.startswith(OFFICIAL_PATH) and not parsed.query and not parsed.fragment,
            "La descarga debe usar la fuente oficial HTTPS de Binance Vision")
    return url


class OfficialRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_official_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def choose_archives(manifest, names, scope):
    rows = manifest["archives"]
    if names:
        wanted = set(names)
        require(len(wanted) == len(names), "No repita --archive")
        selected = [r for r in rows if r["archive"] in wanted]
        require({r["archive"] for r in selected} == wanted, "Archivo no registrado en el manifiesto incluido")
    elif scope == "all":
        selected = rows
    else:
        first, last = ("2023-01-01", "2024-12-31") if scope == "development" else ("2025-01-01", "2026-08-31")
        selected = [r for r in rows if r["kind"] == "monthly"
                    and r["first_open_utc"][:10] >= first and r["last_open_utc"][:10] <= last]
    require(bool(selected), "Ningún archivo corresponde a la selección")
    for row in selected:
        name = row["archive"]
        require(Path(name).name == name and re.fullmatch(r"BTCUSDT-1m-\d{4}-\d{2}(?:-\d{2})?\.zip", name),
                "Nombre bruto inválido")
        prefix = f"https://{OFFICIAL_HOST}/data/spot/{row['kind']}/klines/BTCUSDT/1m/"
        require(row["kind"] in ("monthly", "daily") and row["url"] == prefix + name
                and row["checksum_url"] == row["url"] + ".CHECKSUM"
                and row["status"] == "verified" and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]),
                "La entrada bruta no coincide con el origen y hash fijados")
    return selected


def download_archives(manifest, names, scope, directory):
    selected = choose_archives(manifest, names, scope)
    opener = build_opener(OfficialRedirects())
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    for row in selected:
        name, expected = row["archive"], row["sha256"]
        checksum_request = Request(validate_official_url(row["checksum_url"]), headers={"User-Agent": VERSION})
        with opener.open(checksum_request, timeout=60) as response:
            checksum = response.read(8193)
        require(len(checksum) <= 8192, "CHECKSUM inesperadamente grande")
        tokens = checksum.decode("ascii").strip().split()
        require(len(tokens) == 2 and tokens[0].lower() == expected and tokens[1].lstrip("*") == name,
                "El CHECKSUM oficial cambió respecto del snapshot fijado: " + name)
        target = directory / name
        reused = target.exists()
        if reused:
            require(target.is_file() and not target.is_symlink(), "Destino bruto no regular")
            require(target.stat().st_size == row["bytes"] and sha256(target.read_bytes()) == expected,
                    "El archivo ya descargado no coincide con el snapshot: " + name)
        else:
            temporary = directory / (name + ".part")
            require(not temporary.exists() and not temporary.is_symlink(), "Destino temporal ya existente: " + temporary.name)
            digest, count = hashlib.sha256(), 0
            try:
                request = Request(validate_official_url(row["url"]), headers={"User-Agent": VERSION})
                with opener.open(request, timeout=60) as response, temporary.open("xb") as stream:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        count += len(chunk)
                        require(count <= row["bytes"], "Archivo mayor que la versión fijada: " + name)
                        digest.update(chunk)
                        stream.write(chunk)
                require(count == row["bytes"] and digest.hexdigest() == expected,
                        "Los bytes descargados no coinciden con el snapshot: " + name)
                temporary.replace(target)
            finally:
                if temporary.exists():
                    temporary.unlink()
        results.append({"archive": name, "url": row["url"], "checksum_url": row["checksum_url"],
                        "sha256": expected, "bytes": row["bytes"], "pinned_hash_verified": True,
                        "official_checksum_verified": True, "reused_verified_download": reused,
                        "raw_archive_contents_opened": False})
    return results


def html_report(report):
    dev = report["development"]
    facts = [("Estado", report["status"]), ("Calendario de desarrollo", f"{dev['calendar_start']} a {dev['calendar_end']}"),
             ("Filas de calendario / barras completas", f"{dev['calendar_rows']} / {dev['complete_daily_OHLCV_rows']}"),
             ("Día incompleto conservado", ", ".join(dev["incomplete_daily_dates"])),
             ("Etiquetas σ₇ válidas", str(dev["valid_sigma7_labels"])),
             ("Máxima diferencia de reconstrucción σ₇", f"{dev['sigma7_reconstruction_max_absolute_difference']:.17g}"),
             ("IDs train / calibración actuales", f"{dev['current_final_training_ID_count']} / {dev['current_calibration_ID_count']}"),
             ("Entrada X final de evaluación incluida", "Sí; bundle actual validado, consulte current_dataset_audit.html"
              if "current_dataset" in report else "No; las 608 fechas de evaluación son metadatos en este paquete"),
             ("ZIP oficiales descargados en esta ejecución", str(len(report["downloads"])))]
    rows = "".join("<tr><th>" + html.escape(k) + "</th><td>" + html.escape(v) + "</td></tr>" for k, v in facts)
    sources = "".join("<li>" + html.escape(r["file"]) + ": <code>" + r["sha256"] + "</code></li>" for r in report["inputs"])
    limits = "".join("<li>" + html.escape(s) + "</li>" for s in report["limits"])
    return ("<!doctype html><html lang='es'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
            "<title>Auditoría autónoma del dataset E2</title><style>body{font:16px system-ui;max-width:1050px;margin:2rem auto;padding:0 1rem;line-height:1.5}"
            "table{border-collapse:collapse;width:100%}th,td{padding:.6rem;border:1px solid #ccc;text-align:left}code{overflow-wrap:anywhere}</style>"
            "<h1>Dataset E2: validación autónoma de desarrollo</h1><p>σ₇ poblacional de siete retornos diarios, ocho cierres consecutivos, sin anualización. "
            "Se conservan el calendario y el día incompleto; no se imputan datos ni se entrena un modelo.</p><table>" + rows + "</table>"
            "<p><a href='development_dataset.csv'>Dataset unido</a> · <a href='daily_market_development.csv'>OHLCV original</a> · "
            "<a href='daily_volatility_development.csv'>Etiquetas originales</a> · <a href='dataset_audit.json'>Informe JSON</a>"
            + (" · <a href='current_dataset_audit.html'>Dataset actual 655/53/608</a>" if "current_dataset" in report else "") + "</p>"
            "<h2>Insumos fijados</h2><ul>" + sources + "</ul><h2>Alcance y límites</h2><ul>" + limits + "</ul></html>\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=BASE / "inputs", help="Directorio de los seis insumos incluidos")
    parser.add_argument("--output", type=Path, default=BASE / "output", help="Salida de informes y CSV, por defecto junto a run.py")
    parser.add_argument("--download", action="store_true", help="Descargar ZIP oficiales fijados y verificar también su CHECKSUM")
    parser.add_argument("--download-scope", choices=("development", "evaluation", "all"), default="development",
                        help="Sin --archive: development = 24 mensuales, evaluation = 20 mensuales; all = 75 archivos")
    parser.add_argument("--archive", action="append", default=[], help="Nombre registrado concreto; repetible; requiere --download")
    args = parser.parse_args(argv)
    try:
        require(not args.archive or args.download, "--archive requiere --download")
        input_dir, output = args.input_dir.resolve(), args.output.resolve()
        require(output != input_dir and input_dir not in output.parents, "La salida no puede estar dentro de inputs")
        payloads, records = read_sources(input_dir)
        report, dataset, manifest = audit_dataset(payloads, records)
        current = read_current_bundle(input_dir)
        current_report, evaluation_dataset = None, None
        if current is not None:
            current_report, evaluation_dataset = audit_current_dataset(payloads, current)
            report["mode"] = "offline_development_and_current_dataset_validation"
            report["current_dataset"] = {"status": "passed", "cohorts": COHORT_COUNTS,
                                         "report": "current_dataset_audit.json", "html": "current_dataset_audit.html",
                                         "same_inherited_known_test": True}
            report["limits"][0] = "Los seis insumos originales conservan su alcance de desarrollo; el bundle opcional añade los datos históricos heredados de evaluación, validados en current_dataset_audit.json."
            report["limits"][2] = "El bundle actual incluye los arrays y el escalador reconstruidos en la auditoría fuente; este ejecutor comprueba su integridad y coherencia, sin repetir el ajuste ni certificar ports de modelos."
        if args.download:
            report["mode"] += "_and_explicit_official_download" if current is not None else ""
            if current is None:
                report["mode"] = "development_validation_and_explicit_official_download"
            report["downloads"] = download_archives(manifest, args.archive, args.download_scope, output / "downloads")
        output.mkdir(parents=True, exist_ok=True)
        for name in ("daily_market_development.csv", "daily_volatility_development.csv"):
            (output / name).write_bytes(payloads[name])
        (output / "development_dataset.csv").write_bytes(dataset)
        if current_report is not None:
            for name, payload in current[1].items():
                (output / name).write_bytes(payload)
            (output / CURRENT_BUNDLE).write_bytes(current[3])
            (output / "evaluation_dataset.csv").write_bytes(evaluation_dataset)
            (output / "current_dataset_audit.json").write_text(json.dumps(current_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (output / "current_dataset_audit.html").write_text(current_html_report(current_report), encoding="utf-8")
        (output / "dataset_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output / "dataset_audit.html").write_text(html_report(report), encoding="utf-8")
        print("OK: 731 días, 730 barras completas, 716 etiquetas σ₇; train 655 / cal 53. Consulte output/dataset_audit.html o --output."
              + (" Bundle actual validado: evaluación heredada 608; current_dataset_audit.html." if current is not None else ""))
        return 0
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError, zipfile.BadZipFile, struct.error) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
