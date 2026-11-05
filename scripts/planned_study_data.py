"""Data boundary for the new, explicitly reserved historical holdout.

Importing this module and ``reserve_metadata`` never read ZIP payloads.  The
caller must freeze every selected configuration and checkpoint before calling
``load_reserved_data``.  Acquisition previously inspected ZIP timestamps and
schema; this module does not claim the market files were literally unopened.
All decoded observations and derivatives stay in the caller's private output.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
import pandas as pd

from dl_data import FEATURES, HISTORIES, make_windows

MANIFEST = "data/manifests/binance_spot_BTCUSDT_1m_20230101_20261030.json"
CONTEXT_START = "2026-08-01"
TEST_START = "2026-09-01"
TEST_END = "2026-10-30"
COLUMNS = ("open_time", "open", "high", "low", "close", "volume", "close_time",
           "quote_volume", "trades", "taker_base", "taker_quote", "ignore")
CALENDAR = pd.date_range(CONTEXT_START, TEST_END, freq="D", tz="UTC")
TEST_CALENDAR = pd.date_range(TEST_START, TEST_END, freq="D", tz="UTC")
_RECORD_FIELDS = ("archive", "kind", "relative_path", "url", "checksum_url", "sha256",
                  "bytes", "status", "csv_rows", "expected_minutes", "first_open_utc",
                  "last_open_utc", "timestamp_units")


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _identities():
    return [
        ("monthly", "2026-08", "2026-08-01", "2026-08-31"),
        ("monthly", "2026-09", "2026-09-01", "2026-09-30"),
        *[("daily", f"2026-10-{day:02d}", f"2026-10-{day:02d}", f"2026-10-{day:02d}")
          for day in range(1, 31)],
    ]


def _safe_path(raw_root, relative):
    relative = PurePosixPath(relative)
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
        raise ValueError("An archive path must remain relative to the raw root")
    root = Path(raw_root).resolve()
    candidate = (root / str(relative)).resolve()
    if not candidate.is_relative_to(root) or candidate.is_symlink():
        raise ValueError("An archive path escapes the raw root")
    return candidate


def reserve_metadata(project_root, raw_root=None):
    """Return identities from the acquisition JSON; optionally stat ZIP sizes.

    No archive bytes, CSV prices, targets or model scores are read here.  These
    are 31 holdout archives (September monthly plus 30 October daily archives)
    and one already-observed August context archive, not October31.
    """
    payload = (Path(project_root) / MANIFEST).read_bytes()
    acquisition = json.loads(payload)
    records = acquisition.get("archives", [])
    names = [r["archive"] for r in records]
    if len(names) != len(set(names)):
        raise ValueError("Acquisition metadata contains duplicate archive identities")
    by_name = {r["archive"]: r for r in records}
    chosen, stat_records = [], []
    for kind, label, first, last in _identities():
        name = f"BTCUSDT-1m-{label}.zip"
        if name not in by_name:
            raise ValueError("Missing reserved acquisition identity: " + name)
        record = {key: by_name[name][key] for key in _RECORD_FIELDS}
        expected_relative = f"BTCUSDT/1m/{kind}/{name}"
        checksum = record["sha256"]
        start = pd.Timestamp(first, tz="UTC")
        finish = pd.Timestamp(last, tz="UTC") + pd.Timedelta(days=1)
        if (record["kind"] != kind or record["relative_path"] != expected_relative
                or record["status"] != "verified" or record["bytes"] <= 0
                or not isinstance(checksum, str) or len(checksum) != 64
                or any(c not in "0123456789abcdef" for c in checksum)
                or pd.Timestamp(record["first_open_utc"]) != start
                or pd.Timestamp(record["last_open_utc"]) != finish - pd.Timedelta(minutes=1)
                or record["expected_minutes"] != int((finish - start).total_seconds() / 60)
                or not record["timestamp_units"]
                or not set(record["timestamp_units"]) <= {"ms", "us"}):
            raise ValueError("Reserved archive metadata differs from its fixed calendar: " + name)
        chosen.append(record)
        if raw_root is not None:
            path = _safe_path(raw_root, expected_relative)
            if path.stat().st_size != record["bytes"]:
                raise ValueError("Local ZIP size differs from acquisition metadata: " + name)
            stat_records.append({"archive": name, "size_matches": True})
    return {
        "schema_version": 1,
        "source_manifest": MANIFEST,
        "source_manifest_sha256": _sha(payload),
        "source": "Binance Vision / Binance Spot BTCUSDT 1m",
        "context_start": CONTEXT_START,
        "test_start": TEST_START,
        "test_end_inclusive": TEST_END,
        "calendar_days": len(TEST_CALENDAR),
        "archive_count": len(chosen) - 1,
        "context_archive": chosen[0],
        "archives": chosen[1:],
        "prior_access_scope": "Acquisition inspected timestamps/schema; outcomes were not evaluated by this reservation. This is a historical holdout, not a live forecast.",
        "payloads_opened_by_reservation": 0,
        "stat_verification": stat_records if raw_root is not None else None,
    }


def _canonical_reservation(record):
    result = dict(record)
    result.pop("stat_verification", None)
    return json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _aggregate_payload(payload, record):
    """Validate one monthly/daily archive, preserving missing minute/day masks."""
    if len(payload) != record["bytes"] or _sha(payload) != record["sha256"]:
        raise ValueError("Reserved ZIP bytes/SHA256 differ from their frozen identity")
    start = pd.Timestamp(record["first_open_utc"]).normalize()
    finish = pd.Timestamp(record["last_open_utc"]).normalize() + pd.Timedelta(days=1)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = archive.namelist()
        if (len(members) != 1 or not members[0].endswith(".csv")
                or PurePosixPath(members[0]).is_absolute() or ".." in PurePosixPath(members[0]).parts):
            raise ValueError("One safe CSV member is required in each reserved ZIP")
        with archive.open(members[0]) as stream:
            raw = pd.read_csv(stream, header=None)
    if raw.shape[1] != len(COLUMNS) or not len(raw):
        raise ValueError("A reserved Binance CSV must have exactly twelve columns")
    raw.columns = list(COLUMNS)
    for column in COLUMNS:
        raw[column] = pd.to_numeric(raw[column], errors="raise")
    times = raw.open_time.to_numpy()
    if not np.isfinite(times).all() or not np.equal(times, np.floor(times)).all():
        raise ValueError("Open timestamps must be finite integer epochs")
    unit = "us" if int(times[0]) >= 10**15 else "ms"
    if unit not in record["timestamp_units"]:
        raise ValueError("Timestamp unit differs from acquisition metadata")
    step = 60_000_000 if unit == "us" else 60_000
    times = times.astype(np.int64)
    if np.any(times % step):
        raise ValueError("Open timestamps must align to UTC minutes")
    index = pd.DatetimeIndex(pd.to_datetime(times, unit=unit, utc=True))
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Duplicate or unordered minute timestamps are inadmissible")
    grid = pd.date_range(start, finish, freq="min", inclusive="left")
    if not index.isin(grid).all():
        raise ValueError("An archive contains a minute outside its frozen calendar")
    raw.index = index
    finite = np.isfinite(raw.loc[:, list(COLUMNS)].to_numpy(dtype=float)).all(axis=1)
    positive = raw.loc[:, ["open", "high", "low", "close"]].gt(0).all(axis=1)
    coherent = raw.high.ge(raw[["open", "close", "low"]].max(axis=1)) & raw.low.le(raw[["open", "close", "high"]].min(axis=1))
    nonnegative = raw[["volume", "quote_volume", "trades", "taker_base", "taker_quote"]].ge(0).all(axis=1)
    close_ok = raw.close_time.eq(raw.open_time + step - 1)
    usable = pd.Series(finite, index=index) & positive & coherent & nonnegative & close_ok
    days = pd.date_range(start, finish, freq="D", inclusive="left")
    daily = raw.resample("D").agg(open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum"))
    counts = usable.astype(int).resample("D").sum().reindex(days, fill_value=0)
    daily = daily.reindex(days)
    daily.loc[counts.ne(1440), :] = np.nan
    daily.index.name = "market_day_utc"
    quality = {"archive": record["archive"], "kind": record["kind"], "sha256": _sha(payload),
        "timestamp_unit": unit, "csv_rows": len(raw), "expected_minutes": len(grid),
        "missing_minutes": len(grid) - len(raw), "invalid_minutes": int((~usable).sum()),
        "duplicate_open_times": 0, "out_of_range_minutes": 0,
        "invalid_close_time_minutes": int((~close_ok).sum()),
        "complete_days": int(counts.eq(1440).sum()),
        "excluded_days_utc": days[counts.ne(1440)].strftime("%Y-%m-%d").tolist()}
    return daily, quality


def _build_data(daily):
    if not daily.index.equals(CALENDAR):
        raise ValueError("The reserved derivative must retain every August1–October30 UTC day")
    values = daily.loc[:, list(FEATURES)].to_numpy(dtype=np.float64)
    returns = pd.Series(np.log(values[:, 3]), index=CALENDAR).diff()
    targets = returns.rolling(7, min_periods=7).std(ddof=0).to_numpy()
    persistence = np.concatenate(([np.nan], targets[:-1]))
    windows = {d: make_windows(values, d) for d in HISTORIES}
    eligible = {d: np.isfinite(w).all(axis=(1, 2)) & np.isfinite(targets) & np.isfinite(persistence)
                for d, w in windows.items()}
    selected = CALENDAR.isin(TEST_CALENDAR)
    # The registered new study is D7. Keep the other histories available for
    # transparent metadata without silently imposing their stronger mask.
    common = selected & eligible[7]
    cohort = {"calendar_days": len(TEST_CALENDAR), "primary_history_days": 7,
              "n_common_outcomes": int(common.sum()),
              "by_history": {str(d): int(np.sum(selected & valid)) for d, valid in eligible.items()},
              "excluded_outcome_days_utc": CALENDAR[selected & ~common].strftime("%Y-%m-%d").tolist()}
    data = {"calendar": CALENDAR.copy(), "features": values, "raw": values,
            "targets": targets, "target": targets, "persistence": persistence,
            "windows_by_history": windows, "eligible_by_history": eligible,
            "test_indices": np.flatnonzero(common), "valid": np.isfinite(values).all(axis=1),
            "input_usable": {d: np.isfinite(w).all(axis=(1, 2)) for d, w in windows.items()}}
    return data, cohort


def _read_payload(path):
    return path.read_bytes()


def load_reserved_data(project_root, raw_root, reservation, output_dir):
    """Decode once, after the caller verifies its complete frozen model gate.

    A signed daily cache can be resumed without reopening ZIPs. An interrupted
    first read without a complete cache fails closed and requires an audit.
    A cohort shorter than the fixed sixty days is reported and rejected rather
    than silently replaced or compressed.
    """
    expected = reserve_metadata(project_root)
    identity = _canonical_reservation(reservation)
    if identity != _canonical_reservation(expected):
        raise ValueError("The reservation differs from the frozen acquisition metadata")
    identity_hash = _sha(identity)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    access = out / "raw_access_started.json"
    csv_path, audit_path = out / "daily_market_reserved.csv", out / "raw_source_audit.json"
    if csv_path.exists() or audit_path.exists():
        if not csv_path.exists() or not audit_path.exists() or not access.exists():
            raise RuntimeError("A partial extraction requires audit; payloads will not be reopened")
        audit = json.loads(audit_path.read_text())
        marker = json.loads(access.read_text())
        if (audit.get("reservation_identity_sha256") != identity_hash
                or marker.get("reservation_identity_sha256") != identity_hash
                or _sha(csv_path.read_bytes()) != audit.get("daily_market_sha256")):
            raise ValueError("The reserved daily cache or its identity changed")
        # Python's float conversion preserves the written binary float across
        # the pandas2/3 parser change without its deprecated precision option.
        daily = pd.read_csv(csv_path, index_col="market_day_utc",
                            converters={column: lambda value: float(value) if value else np.nan
                                        for column in FEATURES})
        daily.index = pd.DatetimeIndex(pd.to_datetime(daily.index, utc=True))
        data, cohort = _build_data(daily)
        if cohort != audit.get("cohort"):
            raise ValueError("The reserved cached cohort changed")
    else:
        if access.exists():
            raise RuntimeError("Interrupted first payload access requires audit, not silent rereading")
        _write(access, {"at_utc": _now(), "reservation_identity_sha256": identity_hash,
            "source_manifest_sha256": reservation["source_manifest_sha256"],
            "scope": "New historical holdout payload read; caller has checked all frozen models/configurations"})
        frames, quality = [], []
        for record in [reservation["context_archive"], *reservation["archives"]]:
            payload = _read_payload(_safe_path(raw_root, record["relative_path"]))
            part, checks = _aggregate_payload(payload, record)
            frames.append(part)
            quality.append(checks)
        daily = pd.concat(frames)
        data, cohort = _build_data(daily)
        daily.to_csv(csv_path, mode="x")
        audit = {"completed_at_utc": _now(), "reservation_identity_sha256": identity_hash,
            "source_manifest_sha256": reservation["source_manifest_sha256"],
            "daily_market_sha256": _sha(csv_path.read_bytes()), "archives_opened": len(quality),
            "holdout_archives_opened": len(reservation["archives"]), "quality_by_archive": quality,
            "cohort": cohort, "model_refitting_by_loader": False,
            "historical_not_live_forecasts": True}
        _write(audit_path, audit)
    if cohort["n_common_outcomes"] != len(TEST_CALENDAR):
        raise ValueError("Fixed sixty-day holdout is incomplete; preserve the quality audit and do not replace dates")
    return data, audit
