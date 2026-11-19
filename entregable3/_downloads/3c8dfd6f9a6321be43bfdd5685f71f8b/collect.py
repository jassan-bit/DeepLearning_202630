"""Inspect immutable daily OHLCV inputs with stdlib CPU; never run models.

Prepared with declared AI assistance, 2026-11-18. Failed or incomplete sources
are not admitted. Cached bytes are retained and verified on a local retry.
"""
from __future__ import annotations

import calendar
import csv
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import time
import urllib.error
import urllib.request
import zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE_COMMIT = "d571690eb66760763b0bd6e27e8cd35603c7e244"
LICENSE_ID = "CC-BY-NC-SA-4.0"
DAY_SECONDS = 86400
MAX_ARCHIVE_BYTES = 4 * 1024 * 1024
MAX_CSV_BYTES = 16 * 1024 * 1024
SOURCE_FILES = {
    "BINANCE_VISION_TERMS.md": {
        "path": "TERMS_AND_CONDITIONS.md",
        "sha256": "99c4921dbe023cd68c1a447775707d621453cc78aec6ac0bfe2e6c06a5114ee0",
    },
    "BINANCE_VISION_README.md": {
        "path": "README.md",
        "sha256": "2e133d9945a9263a02781369078e72805eff074680f305cd5460016471c6caad",
    },
}
DATASETS = (
    {"id": "D02", "symbol": "ETHUSDT", "market": "spot", "start": "2018-01-01",
     "end": "2021-12-31", "volume_units": "ETH", "product": "altcoin spot",
     "prefix": "https://data.binance.vision/data/spot/monthly/klines/ETHUSDT/1d/"},
    {"id": "D03", "symbol": "BTCUSD_PERP", "market": "coin_margined_perpetual",
     "start": "2020-09-01", "end": "2024-12-31", "volume_units": "contracts",
     "product": "coin-margined perpetual derivative",
     "prefix": "https://data.binance.vision/data/futures/cm/monthly/klines/BTCUSD_PERP/1d/"},
)
D01_INPUT_SHA = "5abadf99b2d394f920120a27de6f0df129859d71d4cfb0bf661bcc99446ab1fd"
D01_CALENDAR_SHA = "05911b2c78202d985f55a16f1a8919a8526fa48ccd82cffd6bf7eaa72ec0be60"
FIELDS = ("date", "open", "high", "low", "close", "volume")


class AdmissionError(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def atomic_write(path, raw, *, immutable=False):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise AdmissionError("symlink_destination_refused")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise AdmissionError("symlink_destination_refused")
    if path.exists() and immutable:
        if path.read_bytes() != raw:
            raise AdmissionError("cached_source_changed_use_a_new_output_directory")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".partial-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def days_between(start, end):
    start, end = date.fromisoformat(start), date.fromisoformat(end)
    return [(start + timedelta(days=i)).isoformat() for i in range((end - start).days + 1)]


def months_between(start, end):
    a, b = date.fromisoformat(start), date.fromisoformat(end)
    result = []
    year, month = a.year, a.month
    while (year, month) <= (b.year, b.month):
        result.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return result


def decimal_value(value, name):
    try:
        number = Decimal(value)
    except (InvalidOperation, TypeError, ValueError):
        raise AdmissionError("invalid_numeric_" + name) from None
    if not number.is_finite():
        raise AdmissionError("nonfinite_" + name)
    return number


def ohlcv(values):
    if len(values) != 5:
        raise AdmissionError("ohlcv_requires_five_values")
    numbers = [decimal_value(v, name) for v, name in zip(values, FIELDS[1:])]
    opn, high, low, close, volume = numbers
    if min(opn, high, low, close) <= 0 or volume < 0:
        raise AdmissionError("nonpositive_price_or_negative_volume")
    if high < max(opn, close, low) or low > min(opn, close, high):
        raise AdmissionError("incoherent_ohlc")
    return [format(x, "f") for x in numbers]


def checksum_digest(raw, expected_filename):
    try:
        lines = raw.decode("utf-8").strip().splitlines()
    except UnicodeError:
        raise AdmissionError("checksum_not_utf8") from None
    if len(lines) != 1:
        raise AdmissionError("checksum_requires_one_file")
    match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?([^\r\n]+)", lines[0])
    if not match or match.group(2).strip() != expected_filename:
        raise AdmissionError("checksum_filename_or_format_mismatch")
    return match.group(1).lower()


def timestamp_clock(open_text, close_text):
    try:
        opened, closed = int(open_text), int(close_text)
    except ValueError:
        raise AdmissionError("timestamps_must_be_integers") from None
    unit = 1_000_000 if abs(opened) >= 100_000_000_000_000 else 1_000
    if opened % (DAY_SECONDS * unit) != 0:
        raise AdmissionError("opening_not_UTC_midnight")
    if closed != opened + DAY_SECONDS * unit - 1:
        raise AdmissionError("closing_not_last_unit_of_UTC_day")
    try:
        label = datetime.fromtimestamp(opened // unit, timezone.utc).date().isoformat()
    except (OverflowError, OSError, ValueError):
        raise AdmissionError("timestamp_out_of_range") from None
    return label, ("microseconds" if unit == 1_000_000 else "milliseconds")


def parse_month(raw, filename, month):
    if len(raw) > MAX_ARCHIVE_BYTES:
        raise AdmissionError("archive_too_large")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = [e for e in archive.infolist() if not e.is_dir()]
            expected_csv = filename.removesuffix(".zip") + ".csv"
            if len(entries) != 1 or entries[0].filename != expected_csv:
                raise AdmissionError("unexpected_archive_members")
            if entries[0].file_size > MAX_CSV_BYTES:
                raise AdmissionError("uncompressed_csv_too_large")
            csv_raw = archive.read(entries[0])
    except (zipfile.BadZipFile, RuntimeError, OSError) as error:
        raise AdmissionError("invalid_zip_archive:" + type(error).__name__) from None
    records, units = [], set()
    header_seen, empty_lines = False, 0
    try:
        rows = csv.reader(io.StringIO(csv_raw.decode("utf-8-sig"), newline=""))
        for line_number, row in enumerate(rows, 1):
            if not row or all(not v.strip() for v in row):
                empty_lines += 1
                continue
            row = [v.strip() for v in row]
            if len(row) != 12:
                raise AdmissionError(f"expected_12_columns_at_line_{line_number}")
            if row[0].lower() in {"open_time", "opentime", "open time"}:
                if records or header_seen or [x.lower() for x in row[1:6]] != list(FIELDS[1:]):
                    raise AdmissionError("invalid_or_repeated_header")
                header_seen = True
                continue
            label, unit = timestamp_clock(row[0], row[6])
            if label[:7] != month:
                raise AdmissionError("row_outside_named_month")
            values = ohlcv(row[1:6])
            for index in (7, 9, 10):
                if decimal_value(row[index], "auxiliary_volume") < 0:
                    raise AdmissionError("negative_auxiliary_volume")
            if not re.fullmatch(r"[0-9]+", row[8]):
                raise AdmissionError("trade_count_not_nonnegative_integer")
            records.append(dict(zip(FIELDS, [label] + values)))
            units.add(unit)
    except UnicodeError:
        raise AdmissionError("csv_not_utf8") from None
    y, m = map(int, month.split("-"))
    expected = days_between(f"{month}-01", f"{month}-{calendar.monthrange(y, m)[1]:02d}")
    labels = [r["date"] for r in records]
    if labels != expected:
        raise AdmissionError("monthly_calendar_order_duplicates_or_missing_days")
    return records, {"csv_sha256": sha(csv_raw), "csv_bytes": len(csv_raw),
                     "rows": len(records), "first": labels[0], "last": labels[-1],
                     "timestamp_units": sorted(units), "header_present": header_seen,
                     "empty_non_data_lines_ignored": empty_lines}


def normalized_bytes(records):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return stream.getvalue().encode("utf-8")


def fetch(url, path, maximum, opener, pause):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise AdmissionError("cached_symlink_refused")
    if path.is_file():
        raw = path.read_bytes()
        if len(raw) > maximum:
            raise AdmissionError("cached_response_too_large")
        return raw, {"url": url, "cached": True, "bytes": len(raw), "sha256": sha(raw)}
    if pause:
        time.sleep(pause)
    request = urllib.request.Request(url, headers={"User-Agent": "E3-Academic-Dataset-Admission/1.0"})
    try:
        with opener(request, timeout=30) as response:
            raw = response.read(maximum + 1)
            status = response.status
            headers = {k: response.headers.get(k) for k in ("ETag", "Last-Modified", "Content-Type")}
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise AdmissionError("source_unavailable:" + type(error).__name__ + ":" + str(error)) from None
    if status != 200 or len(raw) > maximum:
        raise AdmissionError("unexpected_http_status_or_response_size")
    atomic_write(path, raw, immutable=True)
    return raw, {"url": url, "observed_at_utc": now(), "http_status": status,
                 "headers": headers, "cached": False, "bytes": len(raw), "sha256": sha(raw)}


def admit_d01(output):
    source = REPOSITORY_ROOT / "experiments/e3_dataset_admission_20261118/inputs/stocks.csv.gz"
    calendar_path = REPOSITORY_ROOT / "experiments/e3_d01_admission_20261118/NASDAQ_sessions_20000301_20130301.csv"
    decision_path = REPOSITORY_ROOT / "experiments/e3_d01_admission_20261118/admission_decision.json"
    raw, cal_raw = source.read_bytes(), calendar_path.read_bytes()
    if sha(raw) != D01_INPUT_SHA or sha(cal_raw) != D01_CALENDAR_SHA:
        raise AdmissionError("D01_pinned_source_or_calendar_hash_mismatch")
    decision = json.loads(decision_path.read_bytes())
    if decision.get("admitted") is not True or decision["policy"]["id"] != "RAW_KEEP_ALL":
        raise AdmissionError("D01_prior_admission_identity_mismatch")
    records = []
    with gzip.open(source, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["Ticker"] == "AAPL":
                label = date.fromisoformat(row["Date"]).isoformat()
                records.append(dict(zip(FIELDS, [label] + ohlcv([row[k] for k in ("Open", "High", "Low", "Close", "Volume")]))))
    labels = [r["date"] for r in records]
    calendar_labels = [r["date"] for r in csv.DictReader(io.StringIO(cal_raw.decode()))]
    if labels != calendar_labels or len(labels) != 3270:
        raise AdmissionError("D01_session_calendar_mismatch")
    raw_csv = normalized_bytes(records)
    atomic_write(output / "datasets/D01.csv", raw_csv, immutable=True)
    return {"id": "D01", "admitted": True, "status": "admitted_for_declared_RAW_session_task",
            "rows": len(records), "first": labels[0], "last": labels[-1],
            "path": "datasets/D01.csv", "bytes": len(raw_csv), "sha256": sha(raw_csv),
            "clock": "trading_session", "volume_units": "AAPL shares, raw",
            "source_sha256": sha(raw), "calendar_sha256": sha(cal_raw),
            "license": "Public domain per pinned WIKI/Bokeh declaration",
            "attribution": "Quandl WIKI Prices; archived by marketneutral; Bokeh sampledata",
            "policy": "RAW_KEEP_ALL; Adj Close excluded; corporate-action jumps retained",
            "role": "reserved_confirmation", "target_values_computed": False,
            "possible_window_target_pairs": len(records) - 7,
            "UTC_day_equivalence_claimed": False}


def admit_vision(spec, output, opener, pause):
    rows, sources = [], []
    months = months_between(spec["start"], spec["end"])
    directory = output / "responses" / spec["id"]
    receipt_path = directory / "download_receipts.json"
    cached_receipts = json.loads(receipt_path.read_bytes()) if receipt_path.is_file() else {}
    for position, month in enumerate(months, 1):
        filename = spec["symbol"] + "-1d-" + month + ".zip"
        checksum, checksum_receipt = fetch(spec["prefix"] + filename + ".CHECKSUM",
            directory / (filename + ".CHECKSUM"), 4096, opener, pause)
        expected = checksum_digest(checksum, filename)
        archive, archive_receipt = fetch(spec["prefix"] + filename,
            directory / filename, MAX_ARCHIVE_BYTES, opener, pause)
        if sha(archive) != expected:
            raise AdmissionError("official_archive_checksum_mismatch_use_new_output_if_source_changed")
        records, parsed = parse_month(archive, filename, month)
        if spec["market"] == "coin_margined_perpetual" and any(
                Decimal(r["volume"]) != Decimal(r["volume"]).to_integral_value() for r in records):
            raise AdmissionError("COIN_M_native_contract_volume_not_integral")
        rows.extend(records)
        current = {"month": month, "filename": filename, "checksum": checksum_receipt,
                   "archive": archive_receipt, "official_sha256": expected, "parsed": parsed}
        if month in cached_receipts:
            prior = cached_receipts[month]
            if (prior["official_sha256"] != expected or prior["parsed"] != parsed
                    or prior["archive"]["sha256"] != sha(archive)
                    or prior["checksum"]["sha256"] != sha(checksum)
                    or prior["archive"]["url"] != current["archive"]["url"]
                    or prior["checksum"]["url"] != current["checksum"]["url"]):
                raise AdmissionError("existing_pair_receipt_changed")
            current = prior  # Preserve the first observed HTTP metadata on a retry.
        else:
            cached_receipts[month] = current
            atomic_write(receipt_path, canonical(cached_receipts))
        sources.append(current)
        if position == 1 or position % 6 == 0 or position == len(months):
            print(json.dumps({"phase": "verified_archive", "dataset": spec["id"],
                              "complete_months": position, "required_months": len(months)}), flush=True)
    labels = [r["date"] for r in rows]
    expected_dates = days_between(spec["start"], spec["end"])
    if labels != expected_dates:
        raise AdmissionError("dataset_daily_calendar_mismatch")
    raw_csv = normalized_bytes(rows)
    relative = "datasets/" + spec["id"] + ".csv"
    atomic_write(output / relative, raw_csv, immutable=True)
    return {"id": spec["id"], "admitted": True, "status": "admitted_daily_OHLCV_bytes_and_calendar",
            "symbol": spec["symbol"], "market": spec["market"], "product": spec["product"],
            "rows": len(rows), "first": labels[0], "last": labels[-1],
            "path": relative, "bytes": len(raw_csv), "sha256": sha(raw_csv),
            "clock": "UTC_daily", "bar_frequency": "1d", "volume_units": spec["volume_units"],
            "volume_monetary_equivalence_claimed": False,
            "license": LICENSE_ID, "attribution": "Binance Vision https://data.binance.vision/",
            "license_source_commit": SOURCE_COMMIT,
            "policy": "RAW_KEEP_ALL; no filling, aggregation, filtering or price adjustment",
            "rows_removed": 0, "rows_filled": 0, "missing_dates": [], "duplicate_dates": [],
            "role": "reserved_confirmation", "target_values_computed": False,
            "possible_window_target_pairs": len(rows) - 7, "monthly_sources": sources}


def run(output, source_commit, *, opener=None, pause_seconds=0.15):
    if re.fullmatch(r"[0-9a-f]{40}", source_commit or "") is None:
        raise AdmissionError("exact_source_commit_required")
    if not 0 <= pause_seconds <= 5:
        raise AdmissionError("invalid_request_pause")
    output = Path(output)
    if output.is_symlink():
        raise AdmissionError("output_symlink_refused")
    output.mkdir(parents=True, exist_ok=True)
    config = {"kind": "e3_additional_OHLCV_admission", "source_commit": source_commit,
              "collector_sha256": sha(Path(__file__).read_bytes()),
              "specifications": list(DATASETS), "D01_source_sha256": D01_INPUT_SHA,
              "license_source_commit": SOURCE_COMMIT, "protocol_frozen": False}
    atomic_write(output / "run_config.json", canonical(config), immutable=True)
    source_evidence = []
    for name, metadata in SOURCE_FILES.items():
        raw = Path(__file__).with_name("sources").joinpath(name).read_bytes()
        if sha(raw) != metadata["sha256"]:
            raise AdmissionError("pinned_source_document_changed")
        atomic_write(output / "sources" / name, raw, immutable=True)
        source_evidence.append({"path": "sources/" + name, "sha256": sha(raw),
            "bytes": len(raw), "url": "https://github.com/binance/binance-public-data/blob/" + SOURCE_COMMIT + "/" + metadata["path"]})
    results = []
    try:
        results.append(admit_d01(output))
    except (AdmissionError, OSError, KeyError, ValueError) as error:
        results.append({"id": "D01", "admitted": False, "status": "blocked_current_input_validation",
                        "error": str(error), "prior_admission_revoked": False})
    for spec in DATASETS:
        try:
            results.append(admit_vision(spec, output, opener or urllib.request.urlopen, pause_seconds))
        except (AdmissionError, OSError, ValueError, csv.Error) as error:
            results.append({"id": spec["id"], "admitted": False, "status": "blocked",
                            "symbol": spec["symbol"], "requested_first": spec["start"],
                            "requested_last": spec["end"], "error_type": type(error).__name__,
                            "error": str(error), "missing_values_filled": False,
                            "target_values_computed": False})
    report = {"schema_version": 1, "kind": "e3_additional_OHLCV_admission",
              "recorded_at_utc": now(), "source_commit": source_commit, "datasets": results,
              "collector_sha256": sha(Path(__file__).read_bytes()),
              "datasets_admitted": sum(r["admitted"] is True for r in results),
              "datasets_required": 3, "datasets_evaluated_by_models": 0,
              "source_documents": source_evidence,
              "selection": "D02/D03 replace unadmitted metadata-only candidates before model performance; chosen by rights, access, product and period.",
              "dependence_and_exposure": [
                  "D02 and D03 share Binance as distributor and crypto market shocks.",
                  "D03 BTC derivative shares its underlying with legacy BTC Spot; 2023-2024 overlaps dates already known from legacy fitting.",
                  "D01 sessions and crypto UTC days are homologous clocks, not identical durations.",
                  "These are not three statistically independent markets or globally blind calendars."],
              "new_real_data_fits": 0, "new_market_predictions": 0,
              "new_statistical_contrasts": 0, "target_values_computed": False,
              "real_campaign_allowed": False, "protocol_frozen": False,
              "scope": "Physical file and declared task admission only; model protocol, common budget and prospective power remain separate gates."}
    atomic_write(output / "admission.json", canonical(report))
    files = {p.relative_to(output).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
             for p in sorted(output.rglob("*")) if p.is_file() and p.name != "raw_manifest.json" and not p.name.startswith(".partial-")}
    atomic_write(output / "raw_manifest.json", canonical({"kind": "admission_raw_file_manifest", "source_commit": source_commit, "files": files}))
    print(json.dumps({"datasets_admitted": report["datasets_admitted"], "datasets_required": 3,
                      "new_real_data_fits": 0, "new_market_predictions": 0,
                      "real_campaign_allowed": False, "output": str(output)}, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    run(args.output, args.source_commit)
