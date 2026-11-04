"""Open the reserved minute archives only after configuration and models freeze.

The original development loaders remain unchanged. All aggregation uses complete
UTC days; a missing or invalid minute excludes the entire day and its dependent
windows. This module never downloads data or opens September--October 2026.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/filipe_multihistory_20261104"
FREEZE = ROOT / "experiments/filipe_multihistory_20261104/freeze.json"
PROTOCOL = ROOT / "experiments/filipe_multihistory_20261104/protocol.json"
RESERVE = ROOT / "data/manifests/test_raw_reserve_20250101_20260831.json"
RAW_ROOT = Path("/workspace/binance-spot-data")
HISTORIES = (7, 14, 21, 28)
FEATURES = ("open", "high", "low", "close", "volume")
COLUMNS = ("open_time", "open", "high", "low", "close", "volume", "close_time",
           "quote_volume", "trades", "taker_base", "taker_quote", "ignore")
TEST_CALENDAR = pd.date_range("2025-01-01", "2026-09-01", inclusive="left", tz="UTC")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def now():
    return pd.Timestamp.now(tz="UTC").isoformat()


def event(name, *, out=OUT, **details):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "final_test_events.jsonl").open("a") as stream:
        stream.write(json.dumps({"at_utc": now(), "event": name, **details}, ensure_ascii=False) + "\n")


def validate_freeze(freeze_path=FREEZE, *, root=ROOT, protocol_path=PROTOCOL):
    """Fail before raw access if registered code/protocol changed after closure."""
    freeze_path, root, protocol_path = Path(freeze_path), Path(root), Path(protocol_path)
    freeze = json.loads(freeze_path.read_text())
    if not freeze.get("final_jobs") or not freeze.get("code_sha256"):
        raise ValueError("Configuration freeze requires explicit jobs and signed code")
    if digest(protocol_path) != freeze.get("protocol_sha256"):
        raise ValueError("Protocol differs from its closed configuration")
    for name, expected in freeze["code_sha256"].items():
        candidate = Path(name)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("Frozen code names must be repository-relative")
        if digest(root / candidate) != expected:
            raise ValueError("Code changed after configuration closure: " + name)
    for name, expected in freeze.get("source_cv_sha256", {}).items():
        if digest(root / name) != expected:
            raise ValueError("Development selection evidence changed: " + name)
    return freeze


def validate_final_models(freeze_path=FREEZE, *, root=ROOT, out=OUT, protocol_path=PROTOCOL):
    freeze = validate_freeze(freeze_path, root=root, protocol_path=protocol_path)
    manifest_path = Path(out) / "final_models_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("freeze_sha256") != digest(freeze_path):
        raise ValueError("Final models do not belong to this configuration freeze")
    expected = {(int(j["history_days"]), j["model"], int(j["seed"])) for j in freeze["final_jobs"]}
    actual = {(int(j["history_days"]), j["model"], int(j["seed"])) for j in manifest["jobs"]}
    if actual != expected or len(manifest["jobs"]) != len(expected):
        raise ValueError("Every closed job must have exactly one frozen final model")
    for name, checksum in manifest["artifact_sha256"].items():
        if digest(Path(out) / name) != checksum:
            raise ValueError("A frozen final-fit artifact changed: " + name)
    return freeze, manifest


def safe_archive_path(raw_root, relative_name):
    relative = Path(relative_name)
    root = Path(raw_root).resolve()
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Raw archive path must remain below the configured raw root")
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError("Raw archive path escapes its configured root")
    return result


def aggregate_month_payload(payload, expected_month, *, expected_sha256=None):
    """Validate an in-memory ZIP and return complete daily bars and quality data.

    This pure function also supports synthetic test ZIPs. Production callers
    perform the freeze/model gates before obtaining the payload.
    """
    actual_hash = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and actual_hash != expected_sha256:
        raise ValueError("Reserved archive SHA256 does not match its acquisition identity")
    start = pd.Timestamp(expected_month + "-01", tz="UTC")
    finish = start + pd.offsets.MonthBegin(1)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        if len(names) != 1 or not names[0].endswith(".csv"):
            raise ValueError("A monthly archive must contain exactly one CSV")
        with archive.open(names[0]) as stream:
            raw = pd.read_csv(stream, names=list(COLUMNS), header=None)
    if raw.shape[1] != 12 or not len(raw):
        raise ValueError("Expected a nonempty twelve-column Binance monthly CSV")
    for name in COLUMNS:
        raw[name] = pd.to_numeric(raw[name], errors="raise")
    times = raw["open_time"].to_numpy()
    if not np.isfinite(times).all() or not np.equal(times, np.floor(times)).all():
        raise ValueError("Open timestamps must be finite integer epochs")
    # Binance Spot uses microseconds starting January 2025. Millisecond support
    # permits auditing compatible synthetic/source records without guessing dates.
    unit = "us" if int(times[0]) >= 10**15 else "ms"
    step = 60_000_000 if unit == "us" else 60_000
    integer_times = times.astype(np.int64)
    if np.any(integer_times % step):
        raise ValueError("An open timestamp is not aligned to a UTC minute")
    index = pd.DatetimeIndex(pd.to_datetime(integer_times, unit=unit, utc=True))
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Duplicate or unordered open timestamps are not admissible")
    if (index < start).any() or (index >= finish).any():
        raise ValueError("A reserved archive contains a timestamp outside its stated month")
    raw.index = index
    grid = pd.date_range(start, finish, freq="min", inclusive="left")
    if not index.isin(grid).all():
        raise ValueError("A reserved archive contains an off-grid timestamp")
    finite = np.isfinite(raw.loc[:, list(COLUMNS)].to_numpy(dtype=float)).all(axis=1)
    positive = raw.loc[:, ["open", "high", "low", "close"]].gt(0).all(axis=1).to_numpy()
    ohlc_ok = (raw.high.ge(raw[["open", "close", "low"]].max(axis=1)) &
               raw.low.le(raw[["open", "close", "high"]].min(axis=1))).to_numpy()
    nonnegative = raw[["volume", "quote_volume", "trades", "taker_base", "taker_quote"]].ge(0).all(axis=1).to_numpy()
    close_ok = raw.close_time.eq(raw.open_time + step - 1).to_numpy()
    usable = pd.Series(finite & positive & ohlc_ok & nonnegative & close_ok, index=index)
    daily = raw.resample("D").agg(open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum"))
    daily_calendar = pd.date_range(start, finish, freq="D", inclusive="left")
    counts = usable.astype(int).resample("D").sum().reindex(daily_calendar, fill_value=0)
    daily = daily.reindex(daily_calendar)
    daily.loc[counts.ne(1440), :] = np.nan
    daily.index.name = "market_day_utc"
    quality = {"month": expected_month, "sha256": actual_hash, "timestamp_unit": unit,
        "csv_rows": len(raw), "expected_minutes": len(grid), "missing_minutes": int(len(grid) - len(raw)),
        "invalid_minutes": int((~usable).sum()), "invalid_close_time_minutes": int((~close_ok).sum()),
        "complete_days": int(counts.eq(1440).sum()),
        "excluded_days_utc": counts.index[counts.ne(1440)].strftime("%Y-%m-%d").tolist()}
    return daily, quality


def build_combined_data(development_market, test_market):
    """Construct causal windows and intersection eligibility on full daily grid."""
    development_market = development_market.loc[:, list(FEATURES)]
    test_market = test_market.loc[:, list(FEATURES)]
    expected_development = pd.date_range("2023-01-01", "2025-01-01", inclusive="left", tz="UTC")
    if not development_market.index.equals(expected_development) or not test_market.index.equals(TEST_CALENDAR):
        raise ValueError("Development and test must retain their complete, disjoint UTC calendars")
    market = pd.concat([development_market, test_market])
    features = market.to_numpy(dtype=float)
    returns = np.log(market.close).diff()
    targets = returns.rolling(7, min_periods=7).std(ddof=0).to_numpy()
    persistence = np.concatenate(([np.nan], targets[:-1]))
    positions = np.arange(len(market))
    is_test = market.index >= TEST_CALENDAR[0]
    target_finite, persistence_finite = np.isfinite(targets), np.isfinite(persistence)
    input_masks, eligibility, rows, windows = {}, {}, [], {}
    for history in HISTORIES:
        values = np.full((len(market), history, 5), np.nan)
        for outcome in range(history, len(market)):
            values[outcome] = features[outcome - history:outcome]
        windows[history] = values
        input_masks[history] = np.isfinite(values).all(axis=(1, 2))
        eligibility[history] = input_masks[history] & target_finite & persistence_finite
    common = is_test & np.logical_and.reduce([eligibility[d] for d in HISTORIES])
    for outcome in positions[is_test]:
        for history in HISTORIES:
            rows.append({"outcome_day": market.index[outcome].strftime("%Y-%m-%d"),
                "history_days": history, "input_finite": bool(input_masks[history][outcome]),
                "target_finite": bool(target_finite[outcome]), "persistence_finite": bool(persistence_finite[outcome]),
                "eligible": bool(eligibility[history][outcome]), "common_eligible": bool(common[outcome])})
    excluded = {}
    for outcome in positions[is_test & ~common]:
        reasons = []
        if not target_finite[outcome]: reasons.append("sigma7_incomplete_close_support")
        if not persistence_finite[outcome]: reasons.append("persistence_incomplete_eight_close_support")
        for history in HISTORIES:
            if not input_masks[history][outcome]: reasons.append(f"incomplete_OHLCV_history_D{history}")
        excluded[market.index[outcome].strftime("%Y-%m-%d")] = reasons
    by_history = {str(d): {"eligible_days": int(np.sum(is_test & eligibility[d])),
        "excluded_outcome_days_utc": market.index[is_test & ~eligibility[d]].strftime("%Y-%m-%d").tolist()}
        for d in HISTORIES}
    manifest = {"calendar_days": int(is_test.sum()), "n_common_outcomes": int(common.sum()),
        "common_outcome_days_utc": market.index[common].strftime("%Y-%m-%d").tolist(),
        "by_history": by_history, "excluded_common": excluded,
        "policy": "Intersection across all four finite OHLCV histories, sigma7 target and lag-one sigma7 persistence; no compression or imputation"}
    data = {"calendar": market.index, "features": features, "targets": targets,
        "persistence": persistence, "windows_by_history": windows,
        "eligible_by_history": eligibility, "test_indices": np.flatnonzero(common)}
    return data, pd.DataFrame(rows), manifest


def load_reserved_data(freeze_path=FREEZE, *, raw_root=None, root=ROOT, out=OUT,
                       protocol_path=PROTOCOL, reserve_path=RESERVE):
    """Gate first access, aggregate once, and preserve the test cohort audit."""
    root, out = Path(root), Path(out)
    freeze, models = validate_final_models(freeze_path, root=root, out=out, protocol_path=protocol_path)
    test_out = out / "test"
    test_out.mkdir(parents=True, exist_ok=True)
    daily_path, audit_path = test_out / "daily_market_test.csv", test_out / "raw_test_audit.json"
    if daily_path.exists() or audit_path.exists():
        if not daily_path.exists() or not audit_path.exists():
            raise RuntimeError("Partial raw-test extraction requires an explicit audit; do not silently reopen archives")
        audit = json.loads(audit_path.read_text())
        if audit["freeze_sha256"] != digest(freeze_path) or digest(daily_path) != audit["daily_market_sha256"]:
            raise ValueError("Cached test derivative differs from its frozen audit")
        daily = pd.read_csv(daily_path, index_col="market_day_utc")
        daily.index = pd.DatetimeIndex(pd.to_datetime(daily.index, utc=True))
    else:
        reserve = json.loads(Path(reserve_path).read_text())
        records = reserve["archives"]
        labels = [r["archive"].removeprefix("BTCUSDT-1m-").removesuffix(".zip") for r in records]
        expected_months = pd.date_range("2025-01-01", "2026-08-01", freq="MS").strftime("%Y-%m").tolist()
        if labels != expected_months or reserve["calendar_days"] != 608:
            raise ValueError("Only the twenty reserved January2025--August2026 archives are admitted")
        if freeze.get("reserve_manifest_sha256") and digest(reserve_path) != freeze["reserve_manifest_sha256"]:
            raise ValueError("Raw test reservation changed after closure")
        configured_root = Path(raw_root or freeze.get("raw_root", RAW_ROOT))
        event("first_reserved_test_access_authorized", out=out, freeze_sha256=digest(freeze_path),
              final_models_manifest_sha256=digest(out / "final_models_manifest.json"))
        write_json(test_out / "raw_access_started.json", {"at_utc": now(),
            "freeze_sha256": digest(freeze_path),
            "final_models_manifest_sha256": digest(out / "final_models_manifest.json")})
        frames, qualities = [], []
        opened_at = now()
        for label, record in zip(labels, records):
            path = safe_archive_path(configured_root, record["relative_path"])
            daily_part, quality = aggregate_month_payload(path.read_bytes(), label, expected_sha256=record["sha256"])
            frames.append(daily_part)
            qualities.append(quality)
            event("reserved_archive_aggregated", out=out, archive=record["archive"], sha256=record["sha256"])
        daily = pd.concat(frames)
        if not daily.index.equals(TEST_CALENDAR):
            raise ValueError("Reserved aggregation did not retain the 608-day calendar")
        daily.to_csv(daily_path)
        audit = {"first_raw_access_at_utc": opened_at, "completed_at_utc": now(),
            "freeze_sha256": digest(freeze_path), "final_models_manifest_sha256": digest(out / "final_models_manifest.json"),
            "reserve_manifest_sha256": digest(reserve_path), "archives_opened": len(records),
            "minute_rows": sum(q["csv_rows"] for q in qualities),
            "daily_market_sha256": digest(daily_path), "quality_by_month": qualities,
            "excluded_days_utc": [d for q in qualities for d in q["excluded_days_utc"]],
            "september_october2026_opened": False, "model_refitting_after_access": False}
        write_json(audit_path, audit)
    development = pd.read_csv(root / "results/eda_development/daily_market_development.csv", index_col="market_day_utc")
    development.index = pd.DatetimeIndex(pd.to_datetime(development.index, utc=True))
    data, eligibility, cohort = build_combined_data(development, daily)
    cohort.update({"freeze_sha256": digest(freeze_path),
        "raw_test_audit_sha256": digest(audit_path), "target": "sigma7 population daily log returns, unannualized",
        "horizon_days": 1, "test_start_day_utc": "2025-01-01", "test_end_day_utc": "2026-08-31"})
    eligibility.to_csv(test_out / "eligibility.csv", index=False)
    cohort_path = test_out / "cohort_manifest.json"
    if cohort_path.exists():
        previous = json.loads(cohort_path.read_text())
        if any(previous.get(key) != value for key, value in cohort.items()):
            raise ValueError("The frozen common test cohort changed during a resumed evaluation")
        cohort = previous
    else:
        cohort["created_at_utc"] = now()
        write_json(cohort_path, cohort)
    pd.DataFrame({"market_day_utc": data["calendar"], "sigma_7": data["targets"],
        "persistence": data["persistence"]}).loc[lambda x: x.market_day_utc.ge(TEST_CALENDAR[0])].to_csv(test_out / "daily_volatility_test.csv", index=False)
    return data, cohort, audit
