"""Acquire official BTCUSDT Spot 1m archives, with checksums and coverage metadata.

Uses complete calendar months, daily archives for partial months, and no API.
Only timestamps/schema are inspected; this does not perform target or price EDA.
"""
import argparse
import calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile

BASE = "https://data.binance.vision/data/spot"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def planned_archives(start, end):
    """Inclusive days; never include the next day/month outside the request."""
    current = start
    while current <= end:
        last = date(current.year, current.month,
                    calendar.monthrange(current.year, current.month)[1])
        if current.day == 1 and last <= end:
            yield "monthly", current.strftime("%Y-%m"), current, last
            current = last + timedelta(days=1)
        else:
            yield "daily", current.isoformat(), current, current
            current += timedelta(days=1)


def fetch(url, target):
    """Bounded retries for transient failures; TLS/proxy remain enabled."""
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={"User-Agent": "BitcoinAcademicData/1.0"}),
                         timeout=90) as response, target.open("wb") as output:
                shutil.copyfileobj(response, output)
            return
        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
        time.sleep(2 ** (attempt + 1))


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def audit_archive(path, start, end):
    lower = int(datetime.combine(start, datetime.min.time(), timezone.utc).timestamp())
    upper = int(datetime.combine(end + timedelta(days=1), datetime.min.time(),
                                 timezone.utc).timestamp())
    expected = (upper - lower) // 60
    seen = bytearray(expected)
    rows = duplicates = out_of_range = unaligned = invalid_close_times = 0
    units = set()
    first = last = None
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != 1 or not names[0].endswith(".csv"):
            raise ValueError("Expected one CSV in official ZIP")
        # Reading to EOF also checks ZIP CRC; no files are extracted.
        with archive.open(names[0]) as raw:
            for row in csv.reader(io.TextIOWrapper(raw, encoding="utf-8")):
                rows += 1
                if len(row) != 12:
                    raise ValueError(f"Row {rows}: expected 12 columns")
                stamp = int(row[0])
                unit = "us" if stamp >= 10**14 else "ms"
                divisor = 1_000_000 if unit == "us" else 1_000
                units.add(unit)
                seconds, remainder = divmod(stamp, divisor)
                first = seconds if first is None else min(first, seconds)
                last = seconds if last is None else max(last, seconds)
                if int(row[6]) != stamp + 60 * divisor - 1:
                    invalid_close_times += 1
                if seconds < lower or seconds >= upper:
                    out_of_range += 1
                elif remainder or seconds % 60:
                    unaligned += 1
                else:
                    position = (seconds - lower) // 60
                    duplicates += bool(seen[position])
                    seen[position] = 1
    fmt = lambda value: (datetime.fromtimestamp(value, timezone.utc).isoformat()
                         if value is not None else None)
    return {
        "csv_rows": rows, "expected_minutes": expected,
        "unique_minutes": sum(seen), "missing_minutes": expected - sum(seen),
        "duplicate_open_times": duplicates, "out_of_range_rows": out_of_range,
        "unaligned_open_times": unaligned,
        "unexpected_close_times": invalid_close_times,
        "first_open_utc": fmt(first), "last_open_utc": fmt(last),
        "timestamp_units": sorted(units), "csv_columns": 12,
    }


def acquire(task, output):
    kind, label, start, end = task
    name = f"BTCUSDT-1m-{label}.zip"
    url = f"{BASE}/{kind}/klines/BTCUSDT/1m/{name}"
    record = {"archive": name, "kind": kind, "url": url,
              "checksum_url": url + ".CHECKSUM", "status": "failed"}
    folder = output / "BTCUSDT" / "1m" / kind
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    check = folder / (name + ".CHECKSUM")
    checkpart = folder / (name + ".CHECKSUM.part")
    part = folder / (name + ".part")
    try:
        # Re-fetch official checksum so cached data is not silently assumed current.
        fetch(record["checksum_url"], checkpart)
        tokens = checkpart.read_text().strip().split()
        if len(tokens) != 2 or not re.fullmatch(r"[0-9a-fA-F]{64}", tokens[0]):
            raise ValueError("Invalid official checksum format")
        if tokens[1].lstrip("*") != name:
            raise ValueError("Checksum refers to another archive")
        expected = tokens[0].lower()
        if check.exists() and check.read_text().strip().split() != tokens:
            raise ValueError("Official checksum changed; old checksum and new .part preserved")
        reused = path.exists()
        if reused:
            if sha256(path) != expected:
                raise ValueError("Cached archive hash differs; preserved for inspection")
        else:
            fetch(url, part)
            if sha256(part) != expected:
                raise ValueError("Downloaded archive SHA-256 mismatch")
            part.replace(path)
        audit = audit_archive(path, start, end)
        checkpart.replace(check)
        record.update(audit)
        record.update(status="verified", sha256=expected,
                      bytes=path.stat().st_size,
                      relative_path=str(path.relative_to(output)),
                      reused=reused, verified_at_utc=utc_now())
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2023, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 10, 30))
    parser.add_argument("--output", type=Path, default=Path("/workspace/binance-spot-data"))
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    if args.start > args.end:
        parser.error("start must not be after end")
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = list(planned_archives(args.start, args.end))
    manifest = args.output / f"manifest_{args.start}_{args.end}.json"
    report = {"source": "Binance Vision / Binance Spot", "symbol": "BTCUSDT",
              "interval": "1m", "start_inclusive_utc": str(args.start),
              "end_inclusive_utc": str(args.end), "run_started_at_utc": utc_now(),
              "planned_archives": len(tasks), "archives": []}
    def save():
        tmp = manifest.with_suffix(".json.part")
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(manifest)
    save()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(acquire, task, args.output) for task in tasks]
        for future in as_completed(pending):
            record = future.result()
            report["archives"].append(record)
            save()
            print(f"{len(report['archives'])}/{len(tasks)} {record['archive']}: "
                  f"{record['status']} {record.get('error', '')}", flush=True)
    report["archives"].sort(key=lambda r: r["archive"])
    verified = [r for r in report["archives"] if r["status"] == "verified"]
    report["summary"] = {
        "verified_archives": len(verified), "failed_archives": len(tasks)-len(verified),
        "download_complete": len(verified) == len(tasks),
        "verified_bytes": sum(r["bytes"] for r in verified),
        "csv_rows": sum(r["csv_rows"] for r in verified),
        "expected_minutes_full_request": ((args.end-args.start).days+1)*1440,
        "missing_minutes_within_verified_archives": sum(r["missing_minutes"] for r in verified),
        "duplicate_open_times": sum(r["duplicate_open_times"] for r in verified),
        "out_of_range_rows": sum(r["out_of_range_rows"] for r in verified),
        "unaligned_open_times": sum(r["unaligned_open_times"] for r in verified),
        "unexpected_close_times": sum(r["unexpected_close_times"] for r in verified),
    }
    report["run_finished_at_utc"] = utc_now()
    save()
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"Manifest: {manifest}", flush=True)
    raise SystemExit(0 if report["summary"]["download_complete"] else 1)


if __name__ == "__main__":
    main()
