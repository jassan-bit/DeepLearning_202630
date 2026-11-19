"""Strict pre-protocol recovery of incomplete official monthly OHLCV archives.

AI assistance declared, 2026-11-19. Reuses the historical collector unchanged.
Only an otherwise valid, checksum-verified, incomplete monthly calendar may
switch to ALL official daily archives for that month. No row is imputed,
removed, averaged or shortened, and every overlapping OHLCV value must agree.
"""
from __future__ import annotations

import calendar
import csv
from decimal import Decimal
import importlib.util
import io
import json
from pathlib import Path
import re
import urllib.request
import zipfile

BASE_PATH = (Path(__file__).resolve().parents[2] /
             "e3_cierre_integral_20261118/data/collect.py")
_spec = importlib.util.spec_from_file_location("e3_historical_admission_base", BASE_PATH)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
AdmissionError = base.AdmissionError
POLICY = "checksum_valid_incomplete_month_to_all_official_daily_archives_v1"


def collector_components():
    return {"recovery_module_sha256": base.sha(Path(__file__).read_bytes()),
            "historical_base_sha256": base.sha(BASE_PATH.read_bytes())}


def collector_sha256():
    """Bind both executed source files, using canonical JSON, not a file alias."""
    return base.sha(base.canonical(collector_components()))


def _parse_rows(raw, filename, month):
    """Historical parse_month row validation copied with explicit provenance.

    Source: experiments/e3_cierre_integral_20261118/data/collect.py:parse_month.
    Only its final complete-calendar check is deferred to the caller. This
    permits diagnosis of missing dates, while retaining every other check.
    Complete monthly files always use the unchanged original parser instead.
    """
    if len(raw) > base.MAX_ARCHIVE_BYTES:
        raise AdmissionError("archive_too_large")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = [e for e in archive.infolist() if not e.is_dir()]
            if (len(entries) != 1 or entries[0].filename !=
                    filename.removesuffix(".zip") + ".csv"):
                raise AdmissionError("unexpected_archive_members")
            if entries[0].file_size > base.MAX_CSV_BYTES:
                raise AdmissionError("uncompressed_csv_too_large")
            csv_raw = archive.read(entries[0])
    except (zipfile.BadZipFile, RuntimeError, OSError) as error:
        raise AdmissionError("invalid_zip_archive:" + type(error).__name__) from None
    records, units = [], set()
    header_seen, empty_lines = False, 0
    try:
        for line_number, row in enumerate(csv.reader(io.StringIO(
                csv_raw.decode("utf-8-sig"), newline="")), 1):
            if not row or all(not v.strip() for v in row):
                empty_lines += 1
                continue
            row = [v.strip() for v in row]
            if len(row) != 12:
                raise AdmissionError(f"expected_12_columns_at_line_{line_number}")
            if row[0].lower() in {"open_time", "opentime", "open time"}:
                if records or header_seen or [x.lower() for x in row[1:6]] != list(base.FIELDS[1:]):
                    raise AdmissionError("invalid_or_repeated_header")
                header_seen = True
                continue
            label, unit = base.timestamp_clock(row[0], row[6])
            if label[:7] != month:
                raise AdmissionError("row_outside_named_month")
            values = base.ohlcv(row[1:6])
            for index in (7, 9, 10):
                if base.decimal_value(row[index], "auxiliary_volume") < 0:
                    raise AdmissionError("negative_auxiliary_volume")
            if not re.fullmatch(r"[0-9]+", row[8]):
                raise AdmissionError("trade_count_not_nonnegative_integer")
            records.append(dict(zip(base.FIELDS, [label] + values)))
            units.add(unit)
    except UnicodeError:
        raise AdmissionError("csv_not_utf8") from None
    labels = [r["date"] for r in records]
    return records, {"csv_sha256": base.sha(csv_raw), "csv_bytes": len(csv_raw),
                     "rows": len(records), "first": labels[0] if labels else None,
                     "last": labels[-1] if labels else None,
                     "timestamp_units": sorted(units), "header_present": header_seen,
                     "empty_non_data_lines_ignored": empty_lines}


def _contract_volumes(records, spec):
    if spec["market"] == "coin_margined_perpetual" and any(
            Decimal(r["volume"]) != Decimal(r["volume"]).to_integral_value() for r in records):
        raise AdmissionError("COIN_M_native_contract_volume_not_integral")


def _same_pair(prior, current):
    if any(prior.get(k) != current.get(k) for k in ("month", "date", "filename")):
        raise AdmissionError("existing_pair_receipt_changed")
    for field in ("official_sha256", "parsed"):
        if prior.get(field) != current.get(field):
            raise AdmissionError("existing_pair_receipt_changed")
    for field in ("archive", "checksum"):
        if any(prior.get(field, {}).get(k) != current[field][k] for k in ("sha256", "url")):
            raise AdmissionError("existing_pair_receipt_changed")
    if prior.get("recovery") != current.get("recovery"):
        raise AdmissionError("existing_pair_receipt_changed")
    return prior  # Preserve original HTTP metadata, including cached=False.


def _pair(url_prefix, filename, directory, opener, pause):
    checksum, checksum_receipt = base.fetch(url_prefix + filename + ".CHECKSUM",
        directory / (filename + ".CHECKSUM"), 4096, opener, pause)
    expected = base.checksum_digest(checksum, filename)
    archive, archive_receipt = base.fetch(url_prefix + filename,
        directory / filename, base.MAX_ARCHIVE_BYTES, opener, pause)
    if base.sha(archive) != expected:
        raise AdmissionError("official_archive_checksum_mismatch_use_new_output_if_source_changed")
    return archive, {"filename": filename, "checksum": checksum_receipt,
                     "archive": archive_receipt, "official_sha256": expected}


def _recover_month(spec, month, archive, monthly, directory, opener, pause):
    records, parsed = _parse_rows(archive, monthly["filename"], month)
    labels = [r["date"] for r in records]
    year, number = map(int, month.split("-"))
    expected = base.days_between(month + "-01", f"{month}-{calendar.monthrange(year, number)[1]:02d}")
    if not (0 < len(labels) < len(expected) and labels == sorted(set(labels))
            and set(labels).issubset(expected)):
        raise AdmissionError("fallback_requires_only_missing_dates_in_ordered_nonempty_month")
    _contract_volumes(records, spec)
    if spec["prefix"].count("/monthly/") != 1:
        raise AdmissionError("official_daily_prefix_cannot_be_derived")
    prefix = spec["prefix"].replace("/monthly/", "/daily/")
    daily_directory = directory / "daily_recovery" / month
    receipts_path = daily_directory / "download_receipts.json"
    receipts = json.loads(receipts_path.read_bytes()) if receipts_path.is_file() else {}
    attempt_path = directory / "recovery_attempts.json"
    attempts = json.loads(attempt_path.read_bytes()) if attempt_path.is_file() else {}
    attempt = {"policy": POLICY, "status": "started", "monthly_sha256": base.sha(archive),
               "monthly_parsed": parsed, "missing_dates": [d for d in expected if d not in labels],
               "all_daily_dates_required": expected, "rows_filled": 0, "rows_removed": 0,
               "target_values_computed": False}
    attempts[month] = attempt
    base.atomic_write(attempt_path, base.canonical(attempts))
    recovered, sources = [], []
    try:
        for day in expected:
            filename = spec["symbol"] + "-1d-" + day + ".zip"
            raw, current = _pair(prefix, filename, daily_directory, opener, pause)
            daily_records, daily_parsed = _parse_rows(raw, filename, month)
            if [r["date"] for r in daily_records] != [day]:
                raise AdmissionError("daily_calendar_requires_exactly_one_named_UTC_day")
            _contract_volumes(daily_records, spec)
            current.update(date=day, parsed=daily_parsed)
            if day in receipts:
                current = _same_pair(receipts[day], current)
            else:
                receipts[day] = current
                base.atomic_write(receipts_path, base.canonical(receipts))
            recovered.extend(daily_records)
            sources.append(current)
        if [r["date"] for r in recovered] != expected:
            raise AdmissionError("recovered_month_calendar_mismatch")
        daily_by_date = {r["date"]: r for r in recovered}
        for row in records:
            if any(Decimal(row[k]) != Decimal(daily_by_date[row["date"]][k]) for k in base.FIELDS[1:]):
                raise AdmissionError("monthly_daily_overlap_OHLCV_mismatch:" + row["date"])
    except (AdmissionError, OSError, ValueError, csv.Error) as error:
        attempt.update(status="blocked", error_type=type(error).__name__, error=str(error))
        base.atomic_write(attempt_path, base.canonical(attempts))
        raise
    recovery = {"policy": POLICY, "selection_trigger": "calendar_defect_before_model_performance",
                "source": "all_official_daily_archives_for_the_entire_month",
                "monthly_calendar_complete": False, "monthly_archive_preserved": True,
                "missing_monthly_dates": attempt["missing_dates"],
                "daily_dates": expected, "daily_rows": len(recovered),
                "daily_sources": sources, "overlap_dates": labels,
                "overlap_OHLCV_exact_decimal_equality": True,
                "rows_filled": 0, "rows_removed": 0, "period_shortened": False}
    attempt.update(status="complete", daily_rows=len(recovered), overlap_rows=len(records),
                   overlap_OHLCV_exact_decimal_equality=True)
    base.atomic_write(attempt_path, base.canonical(attempts))
    return recovered, parsed, recovery


def admit_vision(spec, output, opener, pause):
    rows, sources = [], []
    months = base.months_between(spec["start"], spec["end"])
    directory = Path(output) / "responses" / spec["id"]
    receipt_path = directory / "download_receipts.json"
    receipts = json.loads(receipt_path.read_bytes()) if receipt_path.is_file() else {}
    for position, month in enumerate(months, 1):
        filename = spec["symbol"] + "-1d-" + month + ".zip"
        archive, current = _pair(spec["prefix"], filename, directory, opener, pause)
        current = {"month": month, **current}
        try:
            records, parsed = base.parse_month(archive, filename, month)
        except AdmissionError as error:
            if str(error) != "monthly_calendar_order_duplicates_or_missing_days":
                raise
            records, parsed, recovery = _recover_month(
                spec, month, archive, current, directory, opener, pause)
            current["recovery"] = recovery
        _contract_volumes(records, spec)
        current["parsed"] = parsed
        if month in receipts:
            current = _same_pair(receipts[month], current)
        else:
            receipts[month] = current
            base.atomic_write(receipt_path, base.canonical(receipts))
        rows.extend(records)
        sources.append(current)
        if position == 1 or position % 6 == 0 or position == len(months):
            print(json.dumps({"phase": "verified_archive", "dataset": spec["id"],
                "complete_months": position, "required_months": len(months)}), flush=True)
    labels = [r["date"] for r in rows]
    if labels != base.days_between(spec["start"], spec["end"]):
        raise AdmissionError("dataset_daily_calendar_mismatch")
    raw_csv = base.normalized_bytes(rows)
    relative = "datasets/" + spec["id"] + ".csv"
    base.atomic_write(Path(output) / relative, raw_csv, immutable=True)
    return {"id": spec["id"], "admitted": True, "status": "admitted_daily_OHLCV_bytes_and_calendar",
            "symbol": spec["symbol"], "market": spec["market"], "product": spec["product"],
            "rows": len(rows), "first": labels[0], "last": labels[-1],
            "path": relative, "bytes": len(raw_csv), "sha256": base.sha(raw_csv),
            "clock": "UTC_daily", "bar_frequency": "1d", "volume_units": spec["volume_units"],
            "volume_monetary_equivalence_claimed": False,
            "license": base.LICENSE_ID, "attribution": "Binance Vision https://data.binance.vision/",
            "license_source_commit": base.SOURCE_COMMIT,
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
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise AdmissionError("output_symlink_refused")
    output.mkdir(parents=True, exist_ok=True)
    config = {"kind": "e3_additional_OHLCV_admission", "source_commit": source_commit,
              "collector_sha256": collector_sha256(), "collector_components": collector_components(),
              "recovery_policy": POLICY, "specifications": list(base.DATASETS),
              "D01_source_sha256": base.D01_INPUT_SHA,
              "license_source_commit": base.SOURCE_COMMIT, "protocol_frozen": False}
    # A cache imported from the historical collector keeps its original config.
    # The new executed code and source SHA have a separate immutable binding.
    base.atomic_write(output / "recovery_run_config.json", base.canonical(config), immutable=True)
    if not (output / "run_config.json").exists():
        base.atomic_write(output / "run_config.json", base.canonical(config), immutable=True)
    source_evidence = []
    for name, metadata in base.SOURCE_FILES.items():
        raw = BASE_PATH.with_name("sources").joinpath(name).read_bytes()
        if base.sha(raw) != metadata["sha256"]:
            raise AdmissionError("pinned_source_document_changed")
        base.atomic_write(output / "sources" / name, raw, immutable=True)
        source_evidence.append({"path": "sources/" + name, "sha256": base.sha(raw),
            "bytes": len(raw), "url": "https://github.com/binance/binance-public-data/blob/" +
            base.SOURCE_COMMIT + "/" + metadata["path"]})
    results = []
    try:
        results.append(base.admit_d01(output))
    except (AdmissionError, OSError, KeyError, ValueError) as error:
        results.append({"id": "D01", "admitted": False, "status": "blocked_current_input_validation",
                        "error": str(error), "prior_admission_revoked": False})
    for spec in base.DATASETS:
        try:
            results.append(admit_vision(spec, output, opener or urllib.request.urlopen, pause_seconds))
        except (AdmissionError, OSError, ValueError, csv.Error) as error:
            results.append({"id": spec["id"], "admitted": False, "status": "blocked",
                            "symbol": spec["symbol"], "requested_first": spec["start"],
                            "requested_last": spec["end"], "error_type": type(error).__name__,
                            "error": str(error), "missing_values_filled": False,
                            "target_values_computed": False})
    report = {"schema_version": 1, "kind": "e3_additional_OHLCV_admission",
              "recorded_at_utc": base.now(), "source_commit": source_commit, "datasets": results,
              "collector_sha256": collector_sha256(), "collector_components": collector_components(),
              "recovery_policy": POLICY,
              "datasets_admitted": sum(r["admitted"] is True for r in results),
              "datasets_required": 3, "datasets_evaluated_by_models": 0,
              "source_documents": source_evidence,
              "selection": "D02/D03 replace unadmitted metadata-only candidates before model performance; chosen by rights, access, product and period. Incomplete monthly calendars recover the entire month from official daily bytes before evaluation.",
              "dependence_and_exposure": [
                  "D02 and D03 share Binance as distributor and crypto market shocks.",
                  "D03 BTC derivative shares its underlying with legacy BTC Spot; 2023-2024 overlaps dates already known from legacy fitting.",
                  "D01 sessions and crypto UTC days are homologous clocks, not identical durations.",
                  "These are not three statistically independent markets or globally blind calendars."],
              "new_real_data_fits": 0, "new_market_predictions": 0,
              "new_statistical_contrasts": 0, "target_values_computed": False,
              "real_campaign_allowed": False, "protocol_frozen": False,
              "scope": "Physical file and declared task admission only; model protocol, common budget and prospective power remain separate gates."}
    report_path = output / "admission.json"
    if report_path.is_file():
        prior = json.loads(report_path.read_bytes())
        a, b = dict(prior), dict(report)
        a.pop("recorded_at_utc", None); b.pop("recorded_at_utc", None)
        if a == b:
            report = prior
    base.atomic_write(report_path, base.canonical(report))
    files = {p.relative_to(output).as_posix(): {"bytes": p.stat().st_size, "sha256": base.sha(p.read_bytes())}
             for p in sorted(output.rglob("*")) if p.is_file() and p.name != "raw_manifest.json" and not p.name.startswith(".partial-")}
    base.atomic_write(output / "raw_manifest.json", base.canonical({
        "kind": "admission_raw_file_manifest", "source_commit": source_commit, "files": files}))
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
