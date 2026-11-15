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
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta
import hashlib
import html
import io
import json
import math
from pathlib import Path
import re
import sys
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


def parse_daily(payload, header):
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
    require(days == calendar(START, END), "El calendario debe conservar 731 días UTC consecutivos y ordenados")
    return rows, days


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
             ("Entrada X final de evaluación incluida", "No; las 608 fechas de evaluación son metadatos en este paquete"),
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
            "<a href='daily_volatility_development.csv'>Etiquetas originales</a> · <a href='dataset_audit.json'>Informe JSON</a></p>"
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
        if args.download:
            report["mode"] = "development_validation_and_explicit_official_download"
            report["downloads"] = download_archives(manifest, args.archive, args.download_scope, output / "downloads")
        output.mkdir(parents=True, exist_ok=True)
        for name in ("daily_market_development.csv", "daily_volatility_development.csv"):
            (output / name).write_bytes(payloads[name])
        (output / "development_dataset.csv").write_bytes(dataset)
        (output / "dataset_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output / "dataset_audit.html").write_text(html_report(report), encoding="utf-8")
        print("OK: 731 días, 730 barras completas, 716 etiquetas σ₇; train 655 / cal 53. Consulte output/dataset_audit.html o --output.")
        return 0
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
