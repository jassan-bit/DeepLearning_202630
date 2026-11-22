"""Causal, support-aware OHLCV admission. Importing this module performs no I/O.

Market acquisition is opt-in; no market data are distributed with this package.
Versioned extension of source A: Binance spot/USD-M/COIN-M base-volume decoding,
immutable resumable daily acquisition, and exact recovery. AI assistance declared.
No changes to the model, causal target, splits, statistical methods or seed bank.
The low-level test loader's allow_test flag is for the separate final controller,
not an authorization to evaluate. Authentic reserve evidence is still required.
"""
from __future__ import annotations

import argparse
import base64
import calendar
import csv
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import urllib.parse
import urllib.request
import zipfile

import numpy as np

PARTITIONS = ("fit", "selection", "planning", "calibration", "test")
TASK = "sigma7_next_bar_raw_shared_history_exclude_split_support_v1"
UTC = dt.timezone.utc
BAR_INDEX_CONVENTION = "zero_based_complete_observed_calendar_sessions_v1"
HISTORICAL_RESERVE_MODES = ("external_custodian", "documented_historical_reserve")


class AdmissionError(ValueError):
    """An explicit input/evidence requirement failed; no admission is inferred."""


def require(condition, message):
    if not condition:
        raise AdmissionError(message)


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(record):
    return sha256_bytes(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                   allow_nan=False).encode())


def iso_date(value):
    result = dt.date.fromisoformat(str(value))
    require(result.isoformat() == str(value), "Use canonical ISO YYYY-MM-DD dates")
    return result


def utc_now():
    return dt.datetime.now(UTC).isoformat()


def verify_evidence(receipt, expected_status):
    """Verify evidence BYTES, not the actor's identity or truth of its content."""
    require(receipt.get("status") == expected_status, "Evidence decision is incomplete")
    require(receipt.get("evidence_path") and receipt.get("evidence_sha256"), "Evidence path/hash missing")
    path = Path(receipt["evidence_path"])
    require(path.is_file() and sha256_bytes(path.read_bytes()) == receipt["evidence_sha256"], "Evidence bytes/hash mismatch")
    require(receipt.get("actor_name") and receipt.get("actor_contact"), "Evidence decision needs accountable actor/contact")
    if expected_status == "approved":
        require(all(isinstance(receipt.get(k), bool) for k in ("allow_derived_publication", "allow_raw_redistribution")),
                "Rights must explicitly distinguish derivative publication and raw redistribution")
        require(receipt.get("license") and receipt.get("scope") and receipt.get("source_url"), "Rights license/scope/primary URL missing")


def _rows_to_bars(rows, provider, metadata=None):
    require(bool(rows), "Provider returned no bars")
    dates = [str(r[0]) for r in rows]
    require(len(set(dates)) == len(dates), "Duplicate dates: reject; never silently deduplicate")
    order = np.argsort(dates)
    return {"dates": np.asarray(dates, dtype="U10")[order],
            "ohlcv": np.asarray([r[1:6] for r in rows], dtype=np.float64)[order],
            "split": np.asarray([r[6] for r in rows], dtype=np.float64)[order],
            "provider": provider, "metadata": metadata or {}}


def parse_stooq_csv(raw):
    """Format support only. Stooq rights are unresolved; collector is disabled."""
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    require(reader.fieldnames == ["Date", "Open", "High", "Low", "Close", "Volume"],
            "Unexpected Stooq schema/error/captcha")
    rows = [(r["Date"], *(float(r[k]) for k in reader.fieldnames[1:]), 1.) for r in reader]
    return _rows_to_bars(rows, "stooq", {"corporate_actions": "unknown_not_admissible"})


def parse_alphavantage(raw, symbol):
    """DAILY_ADJUSTED JSON: retain AS-TRADED OHLCV; never mix adjusted close."""
    def unique_object(pairs):
        obj = {}
        for key, value in pairs:
            require(key not in obj, "Duplicate JSON key/date")
            obj[key] = value
        return obj
    obj = json.loads(raw, object_pairs_hook=unique_object)
    require("Time Series (Daily)" in obj, "Alpha Vantage error, throttle or unavailable premium endpoint")
    meta = obj.get("Meta Data", {})
    require(meta.get("2. Symbol") == symbol, "Wrong Alpha Vantage instrument")
    require(meta.get("5. Time Zone") in ("US/Eastern", "America/New_York"), "Unverified source timezone")
    rows = []
    for date, r in obj["Time Series (Daily)"].items():
        vals = [float(r[k]) for k in ("1. open", "2. high", "3. low", "4. close", "6. volume")]
        rows.append((date, *vals, float(r["8. split coefficient"])))
    return _rows_to_bars(rows, "alphavantage", {"upstream": meta, "adjustments": "as_traded",
            "dividends": "price_return_includes_ex_dividend_move", "volume_units": "shares"})


def parse_nasdaq_wiki(raw, symbol):
    """Official WIKI/PRICES datatable JSON, raw OHLCV plus split-event ratio."""
    obj = json.loads(raw)
    require("datatable" in obj, "Nasdaq API error/throttle: no datatable")
    table = obj["datatable"]
    names = [column["name"] for column in table["columns"]]
    require(len(names) == len(set(names)), "Duplicate datatable column")
    needed = ("ticker", "date", "open", "high", "low", "close", "volume", "split_ratio")
    require(set(needed) <= set(names), "Unexpected WIKI schema")
    rows = []
    for values in table["data"]:
        require(len(values) == len(names), "Misaligned WIKI row")
        row = dict(zip(names, values))
        require(row["ticker"] == symbol, "Wrong WIKI instrument")
        rows.append((row["date"], *(float(row[k]) for k in needed[2:])))
    bars = _rows_to_bars(rows, "nasdaq_wiki", {"adjustments": "as_traded", "volume_units": "shares",
             "dividends": "raw_price_returns_include_ex_dividend_move", "no_live_updates_since": "2018-04-11"})
    return bars, obj.get("meta", {}).get("next_cursor_id")


BINANCE_MARKETS = {
    "binance_vision": {"market_type": "spot", "prefix": "spot", "symbols": ("BTCUSDT",)},
    "binance_vision_um": {"market_type": "um", "prefix": "futures/um", "symbols": ("BTCUSDT", "ETHUSDT")},
    "binance_vision_cm": {"market_type": "cm", "prefix": "futures/cm", "symbols": ("BTCUSD_PERP",)},
}
BINANCE_DATA_ADAPTER_VERSION = "crypto-markets-bounded-archives-base-volume-v2"


def parse_binance_zip(raw, *, acquired_utc=None, market_type="spot"):
    """Daily close-price klines, with source-specific base-asset volume.

    Source: Binance public-data README pinned in qa/data/primary. USD-M and
    spot volume is column5; COIN-M column5 counts contracts, so column7 supplies
    base volume. Funding payments are not included in the close-price target.
    """
    require(market_type in ("spot", "um", "cm"), "Unsupported Binance market type")
    acquired = dt.datetime.fromisoformat(acquired_utc) if acquired_utc else dt.datetime.now(UTC)
    require(acquired.tzinfo is not None, "Acquisition timestamp must include UTC offset")
    rows, units = [], set()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = [p for p in archive.namelist() if p.endswith(".csv")]
        require(len(members) == 1, "Expected exactly one CSV in Binance archive")
        for index, r in enumerate(csv.reader(io.StringIO(archive.read(members[0]).decode("utf-8-sig")))):
            require(len(r) == 12, "Unexpected Binance kline schema")
            # Published futures archives can carry a header. Skip only a
            # recognized first-line schema; never discard malformed price rows.
            if index == 0 and r[0].strip().lower().replace(" ", "_") == "open_time":
                names = [x.strip().lower().replace(" ", "_") for x in r]
                require(names[:7] == ["open_time","open","high","low","close","volume","close_time"],
                        "Unexpected Binance header order")
                expected_volume = ({"base_volume","base_asset_volume"} if market_type == "cm" else
                                   {"quote_volume","quote_asset_volume"})
                require(names[7] in expected_volume, "Wrong market-volume header")
                continue
            start, close = int(r[0]), int(r[6])
            unit = 1_000_000 if start >= 100_000_000_000_000 else 1_000
            require(start >= 1_000_000_000_000, "Expected epoch ms/us, not seconds")
            require(market_type == "spot" or unit == 1_000, "Futures timestamp unit changed; review source before admission")
            units.add(unit)
            stamp = dt.datetime.fromtimestamp(start / unit, UTC)
            require(stamp.time() == dt.time(), "Daily Binance bar must start at UTC midnight")
            require(close == start + 86400 * unit - 1, "Wrong daily close time/unit")
            require(dt.datetime.fromtimestamp((close + 1) / unit, UTC) <= acquired,
                    "Incomplete/unclosed daily bar")
            volume = float(r[7] if market_type == "cm" else r[5])
            require(np.isfinite(volume) and volume >= 0, "Invalid base-asset volume")
            if market_type == "cm":
                contracts = float(r[5])
                require(np.isfinite(contracts) and contracts >= 0, "Invalid COIN-M contract count")
            rows.append((stamp.date().isoformat(), *(float(x) for x in r[1:5]), volume, 1.))
    return _rows_to_bars(rows, "binance_vision" if market_type == "spot" else "binance_vision_"+market_type,
        {"epoch_units_per_second": sorted(units), "timezone": "UTC", "market_type": market_type,
         "close": "next_UTC_midnight_minus_one_tick", "volume_units": "base_asset",
         "volume_column_zero_based": 7 if market_type == "cm" else 5,
         "parser_version": BINANCE_DATA_ADAPTER_VERSION})


def binance_archive_plan(config):
    """Metadata-only archive itinerary, bounded before observing any prices.

    Complete calendar months inside the development range use monthly 1d ZIPs.
    Partial edge months use daily 1d ZIPs, so acquisition never fetches bars
    before/after the approved range. Missing archives halt; no source substitution
    or selection by model performance. A legacy daily-only policy is explicit.
    """
    provider, symbol = config["provider"], config["instrument"]
    require(provider in BINANCE_MARKETS and symbol in BINANCE_MARKETS[provider]["symbols"],
            "Unfrozen Binance provider/instrument")
    first, last = iso_date(config["development_start"]), iso_date(config["development_end"])
    require(first <= last, "Inverted acquisition bounds")
    policy = config.get("archive_policy", "bounded_monthly_with_daily_edges_v1")
    require(policy in ("bounded_monthly_with_daily_edges_v1", "daily_only_v1"), "Unknown archive policy")
    prefix = BINANCE_MARKETS[provider]["prefix"]
    plan, current = [], first
    while current <= last:
        month_end = current.replace(day=calendar.monthrange(current.year,current.month)[1])
        monthly = policy == "bounded_monthly_with_daily_edges_v1" and current.day == 1 and month_end <= last
        end = month_end if monthly else current
        frequency = "monthly" if monthly else "daily"
        suffix = current.strftime("%Y-%m") if monthly else current.isoformat()
        name = f"{symbol}-1d-{suffix}.zip"
        plan.append({"path":name,"url":f"https://data.binance.vision/data/{prefix}/{frequency}/klines/{symbol}/1d/{name}",
                     "archive_frequency":frequency,"first_session":current.isoformat(),"last_session":end.isoformat(),
                     "expected_bars":(end-current).days+1})
        current = end+dt.timedelta(days=1)
    return plan


def _validate_binance_archive(raw, item, provider, acquired_utc):
    bars = parse_binance_zip(raw, acquired_utc=acquired_utc,market_type=BINANCE_MARKETS[provider]["market_type"])
    require(bars["dates"].tolist() == expected_sessions(item["first_session"],item["last_session"],"24/7"),
            "Archive contains unexpected/missing/out-of-range sessions")
    return bars


def _atomic_new(path, raw):
    """Install immutable bytes, allowing exact reuse after interruption."""
    path = Path(path)
    require(not path.is_symlink(), "Acquisition symlink refused")
    if path.exists():
        require(path.is_file() and path.read_bytes() == raw, "Existing acquisition bytes differ; preserve and investigate")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name); stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    try:
        # There is one foreground collector. Refuse a raced different write.
        require(not path.exists(), "Acquisition target appeared during write")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _pending_record(config, rights_receipt, exposure_receipt, timestamp):
    result = {"kind": "binance_daily_partial_acquisition_v2", "adapter_version": BINANCE_DATA_ADAPTER_VERSION,
              "config": config, "config_sha256": identity(config),
              "rights_sha256": identity(rights_receipt), "exposure_sha256": identity(exposure_receipt),
              "acquisition_started_utc": timestamp}
    result["pending_sha256"] = identity(result)
    return result


def inspect_binance_pending(config, output_dir, *, rights_receipt, exposure_receipt):
    """Pure read-only validation of interrupted raw acquisition, before retry.

    Does not download, write, or mark partial bytes as admitted. Missing ZIP or
    checksum companions may be completed by the activated collector; every
    present pair is hash-verified before any retry contacts the provider.
    """
    verify_evidence(rights_receipt, "approved"); verify_evidence(exposure_receipt, "audited")
    provider, symbol = config["provider"], config["instrument"]
    require(provider in BINANCE_MARKETS, "Pending inspection requires a Binance provider")
    first, last = iso_date(config["development_start"]), iso_date(config["development_end"])
    require(first <= last, "Invalid pending development range")
    root = Path(output_dir)
    target = root/"acquisition_pending.json"
    require(target.is_file() and not target.is_symlink(), "No authentic pending acquisition metadata; preserve partial directory")
    pending = json.loads(target.read_text())
    claimed = pending.pop("pending_sha256", None)
    require(claimed == identity(pending), "Pending metadata hash mismatch")
    expected = _pending_record(config, rights_receipt, exposure_receipt, pending.get("acquisition_started_utc"))
    expected.pop("pending_sha256")
    require(pending == expected, "Pending config/evidence/adapter differs; do not reuse under a new campaign")
    stamp = dt.datetime.fromisoformat(pending["acquisition_started_utc"])
    require(stamp.tzinfo is not None, "Pending acquisition timestamp missing timezone")
    allowed = {"acquisition_pending.json", "acquisition_attempts.jsonl", "raw_manifest.json", "reacquisition_events.jsonl"}
    complete, incomplete = 0, 0
    plan = binance_archive_plan(config)
    for item in plan:
        name = item["path"]
        archive, checksum = root/name, root/(name+".CHECKSUM")
        allowed.update((name,name+".CHECKSUM"))
        require(not archive.is_symlink() and not checksum.is_symlink(), "Acquisition source symlink refused")
        if checksum.exists():
            tokens = checksum.read_text().split()
            require(len(tokens) >= 1 and len(tokens[0]) == 64 and all(x in "0123456789abcdefABCDEF" for x in tokens[0]),
                    "Invalid stored provider checksum")
            if archive.exists():
                raw = archive.read_bytes()
                require(sha256_bytes(raw) == tokens[0].lower(), "Partial stored source checksum mismatch")
                _validate_binance_archive(raw,item,provider,pending["acquisition_started_utc"])
                complete += 1
        if archive.exists() != checksum.exists():
            incomplete += 1
    require(all(p.name in allowed and p.is_file() for p in root.iterdir()),
            "Unknown acquisition files retained; do not silently discard or overwrite")
    return {"status": "VALIDATED_PARTIAL_RAW_NOT_ADMITTED", "complete_pairs": complete,
            "incomplete_pairs": incomplete, "expected_pairs": len(plan),
            "expected_raw_bars":sum(item["expected_bars"] for item in plan),
            "acquisition_started_utc": pending["acquisition_started_utc"]}


def _collect_binance_daily(config, root, rights, exposure):
    provider, symbol = config["provider"], config["instrument"]
    first, last = iso_date(config["development_start"]), iso_date(config["development_end"])
    pending_path = root/"acquisition_pending.json"
    if not pending_path.exists():
        require(all(p.name == "acquisition_attempts.jsonl" and p.is_file() for p in root.iterdir()),
                "Unbound pre-existing acquisition files retained; use original recovery metadata")
        pending = _pending_record(config, rights, exposure, utc_now())
        _atomic_new(pending_path, (json.dumps(pending,indent=2)+"\n").encode())
    state = inspect_binance_pending(config,root,rights_receipt=rights,exposure_receipt=exposure)
    manifest = {"provider":provider,"instrument":symbol,"scope":"development_only","files":[],
        "rights":rights,"exposure":exposure,"acquired_utc":state["acquisition_started_utc"],
        "acquisition_started_utc":state["acquisition_started_utc"],
        "acquired_utc_semantics":"First acquisition attempt; completed_utc and attempt ledgers report later actual events.",
        "market_type":BINANCE_MARKETS[provider]["market_type"],"adapter_version":BINANCE_DATA_ADAPTER_VERSION}
    manifest["archive_policy"] = config.get("archive_policy", "bounded_monthly_with_daily_edges_v1")
    downloaded = reused = 0
    plan = binance_archive_plan(config)
    for offset, item in enumerate(plan):
        name, url = item["path"], item["url"]
        checksum_path, archive_path = root/(name+".CHECKSUM"), root/name
        checksum = checksum_path.read_bytes() if checksum_path.exists() else _network_get(url+".CHECKSUM")
        tokens = checksum.decode().split()
        require(tokens and len(tokens[0]) == 64 and all(x in "0123456789abcdefABCDEF" for x in tokens[0]),
                "Invalid upstream Binance checksum")
        claimed = tokens[0].lower()
        if archive_path.exists():
            payload = archive_path.read_bytes(); reused += 1
        else:
            payload = _network_get(url); downloaded += 1
        require(sha256_bytes(payload) == claimed, "Upstream Binance checksum mismatch")
        _validate_binance_archive(payload,item,provider,utc_now())
        _atomic_new(archive_path,payload); _atomic_new(checksum_path,checksum)
        manifest["files"].append({**item,"sha256":claimed,
            "provider_checksum_sha256":sha256_bytes(checksum)})
        if (offset+1) % 12 == 0 or offset+1 == len(plan):
            print(json.dumps({"event":"development_acquisition_progress","provider":provider,"instrument":symbol,
                "completed_files":offset+1,"expected_files":len(plan),"downloaded_payloads":downloaded,
                "reused_payloads":reused,"new_market_fits":0,"test_opened":False}),flush=True)
    manifest["completed_utc"] = utc_now()
    manifest["manifest_sha256"] = identity(manifest)
    _atomic_new(root/"raw_manifest.json",(json.dumps(manifest,indent=2)+"\n").encode())
    return manifest


def _parse_binance_manifest_payload(raw, record):
    provider = record["provider"]
    expected = BINANCE_MARKETS[provider]["market_type"]
    require(record.get("market_type",expected) == expected, "Manifest provider/market type mismatch")
    return parse_binance_zip(raw,acquired_utc=record["acquired_utc"],market_type=expected)


def expected_sessions(start, end, calendar):
    """Official exchange_calendars schedule, including holidays and early closes."""
    first, last = iso_date(start), iso_date(end)
    require(first <= last, "Inverted calendar")
    if calendar == "24/7":
        return [(first + dt.timedelta(days=i)).isoformat() for i in range((last-first).days+1)]
    require(calendar in ("XNYS", "XNAS"), "Only explicitly supported market calendars allowed")
    import exchange_calendars as xcals
    import pandas as pd
    schedule = xcals.get_calendar(calendar, start=start, end=end)
    return [x.date().isoformat() for x in schedule.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))]


def validate_bars(bars, sessions):
    """Reject missing bars rather than forward-fill across sessions or weekends."""
    dates, values, splits = bars["dates"], bars["ohlcv"], bars["split"]
    require(values.shape == (len(dates), 5) and splits.shape == (len(dates),), "Misaligned OHLCV")
    require(len(dates) >= 8, "Eight closes are needed for the first label")
    for date in dates:
        iso_date(date)
    require(np.all(dates[1:] > dates[:-1]), "Dates must be strictly increasing and unique")
    require(list(dates) == list(sessions), "Missing/unexpected market session; no automatic imputation")
    require(np.isfinite(values).all() and np.isfinite(splits).all(), "Nonfinite OHLCV/corporate action")
    require((values[:, :4] > 0).all() and (values[:, 4] >= 0).all() and (splits > 0).all(), "Invalid price/volume/split")
    o, h, l, c, _ = values.T
    require((l <= np.minimum(o, c)).all() and (h >= np.maximum(o, c)).all() and (h >= l).all(), "Incoherent candle")
    return {"n_bars": len(dates), "first": str(dates[0]), "last": str(dates[-1]),
            "missing": 0, "duplicates": 0, "zero_volume": int(np.sum(values[:, 4] == 0)),
            "split_events": [str(dates[i]) for i in np.flatnonzero(splits != 1)]}


def build_windows(bars, *, target_start=None, target_end=None):
    """Raw bars [t-7,t-1], y=population SD of 7 log returns [t-7,t].

    Range checks run BEFORE extracting target closes, so development never
    constructs labels for other ranges. Caller must validate the full calendar.
    Exclude every support containing a split (fixed, loss-independent rule).
    """
    dates, values, splits = bars["dates"], bars["ohlcv"], bars["split"]
    samples = {k: [] for k in ("raw", "y", "dates", "support_start", "support_end",
                                "known_sigma_history", "known_floor_A", "history_end_positions")}
    # The series is past observed state, not future labels for a given endpoint.
    eligible = np.flatnonzero(dates <= target_end) if target_end else np.arange(len(dates))
    require(len(eligible) > 0, "No eligible target dates")
    stop_target = int(eligible[-1])
    observed = np.asarray([np.std(np.diff(np.log(values[u-7:u+1, 3])), ddof=0)
                           for u in range(7, stop_target)], dtype=np.float64)
    excluded = []
    for t in range(14, len(dates)):
        date = str(dates[t])
        if target_start and date < target_start:
            continue
        if target_end and date > target_end:
            continue
        if np.any(splits[t-14:t+1] != 1):
            excluded.append(date)
            continue
        samples["raw"].append(values[t-7:t].copy())
        samples["y"].append(float(np.std(np.diff(np.log(values[t-7:t+1, 3])), ddof=0)))
        samples["dates"].append(date)
        samples["support_start"].append(t-14)
        samples["support_end"].append(t)
        samples["known_sigma_history"].append(observed[t-14:t-7])
        six = np.diff(np.log(values[t-7:t, 3]))
        samples["known_floor_A"].append(float(np.sum((six-six.mean())**2)/7))
        samples["history_end_positions"].append(t-7)
    result = {"raw": np.asarray(samples["raw"], dtype=np.float64).reshape(-1, 7, 5),
              "y": np.asarray(samples["y"], dtype=np.float64),
              "dates": np.asarray(samples["dates"], dtype="U10"),
              "support_start": np.asarray(samples["support_start"], dtype=np.int64),
              "support_end": np.asarray(samples["support_end"], dtype=np.int64),
              "known_sigma_history": np.asarray(samples["known_sigma_history"], dtype=np.float64).reshape(-1, 7),
              "known_floor_A": np.asarray(samples["known_floor_A"], dtype=np.float64),
              "history_end_positions": np.asarray(samples["history_end_positions"], dtype=np.int64),
              "observed_sigma_since_fit": observed,
              "observed_sigma_dates": dates[7:stop_target].copy(),
              "observed_sigma_bar_indices": np.arange(7, stop_target, dtype=np.int64)}
    return result, {"split_support_excluded_dates": excluded, "task": TASK}


def split_by_support(samples, raw_bounds, *, embargo_bars=0, allow_test=False):
    """Assign only supports wholly inside disjoint inclusive RAW-BAR bounds.

    Boundary purging is fourteen target endpoints for adjacent bounds;
    an optional embargo removes extra raw bars at each later partition start.
    Trading-bar index, never calendar-day subtraction, defines this operation.
    """
    require(isinstance(embargo_bars, int) and embargo_bars >= 0, "Invalid embargo")
    require(set(raw_bounds) <= set(PARTITIONS), "Unknown partition")
    require(allow_test or "test" not in raw_bounds, "Development cannot prepare test")
    out, report, previous_end = {}, {}, -1
    for part in PARTITIONS:
        if part not in raw_bounds:
            continue
        lo, hi = raw_bounds[part]
        require(isinstance(lo, int) and isinstance(hi, int) and lo > previous_end and hi >= lo,
                "Ordered disjoint inclusive raw-bar bounds required")
        effective_lo = lo + (embargo_bars if part != "fit" else 0)
        keep = (samples["support_start"] >= effective_lo) & (samples["support_end"] <= hi)
        series_keys = ("observed_sigma_since_fit", "observed_sigma_dates", "observed_sigma_bar_indices")
        out[part] = {key: arr[keep] for key, arr in samples.items() if key not in series_keys}
        require(len(out[part]["y"]) > 0, f"No supported endpoints in {part}")
        # Historical ARIMA filtering state may span earlier partitions, like
        # learned weights. Its causal prefix is distinct from the finite local
        # support used for boundary purging. Do not reset ARIMA at boundaries.
        history_stop = int(out[part]["history_end_positions"].max())
        for key in series_keys:
            out[part][key] = samples[key][:history_stop]
        report[part] = {"raw_bounds": [lo, hi], "effective_raw_bounds": [effective_lo, hi],
                        "n": int(np.sum(keep)), "embargo_bars": effective_lo-lo}
        previous_end = hi
    return out, report


def _safe_path(root, relative):
    p = Path(relative)
    require(not p.is_absolute() and ".." not in p.parts, "Unsafe relative artifact path")
    resolved = (root / p).resolve()
    require(resolved.is_relative_to(root.resolve()), "Artifact escapes panel directory")
    return resolved


def validate_samples(data):
    require(set(data) == {"raw", "y", "dates", "support_start", "support_end", "known_sigma_history",
        "known_floor_A", "history_end_positions", "observed_sigma_since_fit", "observed_sigma_dates",
        "observed_sigma_bar_indices"}, "Unexpected NPZ fields")
    n = len(data["y"])
    require(n > 0 and data["raw"].shape == (n, 7, 5), "Expected raw[N,7,5]")
    require(all(data[k].shape == (n,) for k in ("y", "dates", "support_start", "support_end")), "Misaligned arrays")
    require(np.isfinite(data["raw"]).all() and np.isfinite(data["y"]).all() and (data["y"] >= 0).all(), "Invalid arrays")
    v = data["raw"]
    require((v[..., :4] > 0).all() and (v[..., 4] >= 0).all(), "Invalid OHLCV units")
    require(np.all(v[..., 2] <= np.minimum(v[..., 0], v[..., 3])) and
            np.all(v[..., 1] >= np.maximum(v[..., 0], v[..., 3])), "Incoherent input candle")
    require(data["dates"].dtype.kind in "US", "ISO dates required")
    for date in data["dates"]:
        iso_date(date)
    require(np.all(data["dates"][1:] > data["dates"][:-1]), "Unsorted/duplicate sample dates")
    for key in ("support_start", "support_end"):
        require(data[key].dtype.kind in "iu", "Support indices must be integers")
    require((data["support_start"] >= 0).all() and
            np.all(data["support_end"] - data["support_start"] == 14), "Incorrect fifteen-close local support")
    require(np.all(np.diff(data["support_end"]) > 0), "Unsorted support indices")
    require(data["known_sigma_history"].shape == (n, 7) and np.isfinite(data["known_sigma_history"]).all() and
            (data["known_sigma_history"] >= 0).all(), "Invalid known sigma history")
    require(data["known_floor_A"].shape == (n,) and np.isfinite(data["known_floor_A"]).all() and
            np.all(data["known_floor_A"] >= 0), "Invalid overlap floor")
    series, end = data["observed_sigma_since_fit"], data["history_end_positions"]
    require(series.ndim == 1 and np.isfinite(series).all() and (series >= 0).all(), "Invalid observed sigma prefix")
    require(end.shape == (n,) and end.dtype.kind in "iu" and (end >= 7).all() and (end <= len(series)).all(), "Invalid history prefix offset")
    require(data["observed_sigma_dates"].shape == series.shape and data["observed_sigma_bar_indices"].shape == series.shape, "Misaligned observed series")
    require(np.all(data["observed_sigma_bar_indices"] == np.arange(7, 7+len(series))), "Observed series must preserve bar indexing")
    require(np.all(data["observed_sigma_dates"][1:] > data["observed_sigma_dates"][:-1]), "Unsorted observed sigma dates")
    require(np.all(end+7 == data["support_end"]), "History end must be strictly before target bar")
    for i, stop in enumerate(end):
        require(np.array_equal(data["known_sigma_history"][i], series[stop-7:stop]), "Known history differs from causal prefix")
        require(data["observed_sigma_dates"][stop-1] < data["dates"][i], "Observed history includes target/future")
    six = np.diff(np.log(data["raw"][:, :, 3]), axis=1)
    floor = ((six-six.mean(axis=1, keepdims=True))**2).sum(axis=1)/7
    require(np.allclose(data["known_floor_A"], floor, rtol=1e-12, atol=1e-15), "Overlap floor does not match raw history")


def write_panel(panel_dir, partitions, *, provenance, rights, exposure, support_audit,
                fixture=False, final_authorization=None):
    """Write one NPZ per partition; no combined payload ever contains test."""
    require(bool(partitions) and set(partitions) <= set(PARTITIONS), "Unknown/empty partitions")
    if not fixture:
        verify_evidence(rights, "approved")
        verify_evidence(exposure, "audited")
        require(provenance.get("raw_manifest_sha256") and provenance.get("provider_version"), "Pinned raw provenance required")
        require(provenance.get("bar_index_origin") and
                provenance.get("bar_index_convention") == BAR_INDEX_CONVENTION,
                "Observed-calendar bar index origin/convention required")
        iso_date(provenance["bar_index_origin"])
    if "test" in partitions:
        require(final_authorization and final_authorization.get("authorize_test_preparation") is True,
                "Test creation requires explicit final authorization")
        require(final_authorization.get("protocol_sha256") and final_authorization.get("evidence_sha256"),
                "Bind protocol and authentic final authorization evidence")
    root = Path(panel_dir)
    root.mkdir(parents=True, exist_ok=True)
    require(not (root / "admission.json").exists(), "Immutable admission exists; use a new panel directory/version")
    admission = {"schema_version": 1, "task": TASK,
        "status": "software_fixture" if fixture else "development_admitted",
        "reserved_status": "explicit_final_preparation" if "test" in partitions else "test_not_acquired",
        "provenance": provenance, "rights": rights, "exposure": exposure,
        "support_audit": support_audit, "final_authorization": final_authorization,
        "partition_paths": {}, "partition_sha256": {}, "partition_counts": {},
        "partition_date_ranges": {}, "partition_support_bounds": {}, "created_utc": utc_now()}
    last_end = -1
    for part in PARTITIONS:
        if part not in partitions:
            continue
        data = partitions[part]
        validate_samples(data)
        lo, hi = int(data["support_start"].min()), int(data["support_end"].max())
        require(lo > last_end, "Partition raw/label support overlaps")
        last_end = hi
        name = part + ".npz"
        require(not (root/name).exists(), "Will not overwrite partition")
        np.savez_compressed(root/name, **data)
        admission["partition_paths"][part] = name
        admission["partition_sha256"][part] = sha256_bytes((root/name).read_bytes())
        admission["partition_counts"][part] = len(data["y"])
        admission["partition_date_ranges"][part] = [str(data["dates"][0]), str(data["dates"][-1])]
        admission["partition_support_bounds"][part] = [lo, hi]
    admission["admission_sha256"] = identity(admission)
    (root/"admission.json").write_text(json.dumps(admission, indent=2)+"\n")
    return admission


def read_admission(panel_dir):
    root = Path(panel_dir)
    record = json.loads((root/"admission.json").read_text())
    claimed = record.pop("admission_sha256", None)
    require(claimed == identity(record), "Admission metadata hash mismatch")
    require(record.get("schema_version") == 1 and record.get("task") == TASK, "Unsupported admission/task")
    require(record.get("status") in ("software_fixture", "development_admitted"), "Panel not admitted")
    if record["status"] != "software_fixture":
        require(record["rights"].get("status") == "approved" and record["exposure"].get("status") == "audited", "Incomplete admission")
    previous = -1
    for part in PARTITIONS:
        if part in record["partition_paths"]:
            lo, hi = record["partition_support_bounds"][part]
            require(lo > previous and hi >= lo, "Admission contains overlapping partition supports")
            previous = hi
    record["admission_sha256"] = claimed
    return record


def load_partition(panel_dir, part, *, allow_test=False):
    """Hash/read ONLY the requested NPZ; never touch another partition payload."""
    require(part in PARTITIONS, "Unknown partition")
    require(part != "test" or allow_test, "Final partition is sealed; use separate final controller")
    record = read_admission(panel_dir)
    require(part in record["partition_paths"], f"Partition not prepared: {part}")
    if part == "test":
        auth = record.get("final_authorization") or {}
        require(auth.get("authorize_test_preparation") is True, "No bound final-preparation authorization")
    payload = _safe_path(Path(panel_dir), record["partition_paths"][part]).read_bytes()
    require(sha256_bytes(payload) == record["partition_sha256"][part], "Partition byte SHA256 mismatch")
    with np.load(io.BytesIO(payload), allow_pickle=False) as source:
        data = {k: source[k].copy() for k in source.files}
    validate_samples(data)
    require(len(data["y"]) == record["partition_counts"][part], "Partition count mismatch")
    require([str(data["dates"][0]), str(data["dates"][-1])] == record["partition_date_ranges"][part], "Partition dates mismatch")
    require([int(data["support_start"].min()), int(data["support_end"].max())] ==
            record["partition_support_bounds"][part], "Partition support mismatch")
    return data


def _verify_historical_reserve(form):
    """Bind real historical access evidence, without requiring another person.

    Existing external-custodian exports retain their original schema. The
    owner-or-custodian mode additionally names the actual exports and access
    history; hashes/field presence do not establish the truth of that history.
    """
    require(bool(form.get("custodian_seal_sha256")) and bool(form.get("custodian_access_log_sha256")),
            "Historical reserve requires actual seal and access log")
    if form["reserve_mode"] == "documented_historical_reserve":
        require(isinstance(form.get("actual_history_accessed_per_panel"), dict) and
                bool(form["actual_history_accessed_per_panel"]),
                "Documented historical reserve requires actual panel exposure history")
        require(bool(form.get("who_could_access_prices_labels_predictions")),
                "Documented historical reserve requires accountable access history")
        for path_key, hash_key in (
                ("historical_seal_evidence_path", "custodian_seal_sha256"),
                ("historical_access_log_evidence_path", "custodian_access_log_sha256"),
                ("exposure_inventory_path", "exposure_inventory_sha256")):
            require(isinstance(form.get(path_key), str) and bool(form[path_key]),
                    "Documented historical reserve requires evidence file: "+path_key)
            path = Path(form[path_key])
            require(path.is_file() and sha256_bytes(path.read_bytes()) == form.get(hash_key),
                    "Historical reserve evidence bytes/hash mismatch: "+path_key)


def record_owner_evidence(form_path, evidence_path, output_path):
    """Bind externally supplied evidence; a hash is NOT a signature verification.

    The tool explicitly records who supplied the declaration and how the human
    author or custodian can be contacted. It never impersonates or certifies
    that actor, and does not require an external institution or approver.
    """
    form = json.loads(Path(form_path).read_text())
    for field in ("actor_name", "actor_contact", "actor_role", "statement_utc", "reserve_mode", "evidence_origin"):
        require(bool(form.get(field)), f"Authentic actor must complete {field}")
    statement = dt.datetime.fromisoformat(form["statement_utc"])
    require(statement.tzinfo is not None, "Actor statement must include timezone")
    require(form.get("acknowledge_prior_D00_D03_exposed") is True, "Prior exposure must be acknowledged")
    require(form["reserve_mode"] in ("development_only", *HISTORICAL_RESERVE_MODES, "post_protocol"), "Unsupported reserve mode")
    if form["reserve_mode"] == "development_only":
        require(form.get("all_data_declared_exposed_development") is True,
                "Development mode must explicitly designate all supplied data as exposed development")
        require(form.get("authorize_test_preparation") is not True,
                "Development designation cannot authorize test preparation")
        for field in ("observation_cutoff_utc", "dataset_scope", "source_url", "source_license_evidence_path", "source_license_evidence_sha256"):
            require(bool(form.get(field)), f"Development actor must complete {field}")
        cutoff = dt.datetime.fromisoformat(form["observation_cutoff_utc"])
        require(cutoff.tzinfo is not None, "Development observation cutoff must include timezone")
        require(cutoff <= statement, "Observation cutoff cannot be later than actor's statement")
        license_path = Path(form["source_license_evidence_path"])
        require(license_path.is_file() and sha256_bytes(license_path.read_bytes()) == form["source_license_evidence_sha256"],
                "Development primary-license evidence bytes/hash mismatch")
    else:
        require(bool(form.get("protocol_sha256")), "Authentic final reserve requires fixed protocol_sha256")
        require(bool(form.get("exposure_inventory_sha256")), "Authentic final reserve requires exposure inventory")
    if form["reserve_mode"] == "post_protocol":
        require(bool(form.get("protocol_lock_utc")), "Full protocol lock is missing")
        lock = dt.datetime.fromisoformat(form["protocol_lock_utc"])
        require(lock.tzinfo is not None, "Timezone-aware protocol lock required")
        require(iso_date(form["first_raw_support_session"]) > lock.astimezone(UTC).date(),
                "Every final support bar must follow the complete protocol lock")
    elif form["reserve_mode"] in HISTORICAL_RESERVE_MODES:
        _verify_historical_reserve(form)
    evidence = Path(evidence_path).read_bytes()
    require(len(evidence) > 0, "Empty authentic evidence")
    receipt = {"kind": "owner_supplied_evidence_not_identity_verification", "status": "audited",
               "actor_name": form["actor_name"], "actor_contact": form["actor_contact"], "form": form,
               "form_sha256": identity(form), "evidence_sha256": sha256_bytes(evidence),
               "evidence_path": str(Path(evidence_path).resolve()),
               "recorded_utc": utc_now(), "cryptographic_signature_verified": False}
    if form["reserve_mode"] == "development_only":
        receipt.update(scope="development_only", reserved_status="not_reserved_all_data_exposed",
            final_reserve_claim=False, final_protocol_required=False,
            designation="Actor designates all supplied dataset observations exposed development; no final-data reserve or final-protocol approval is claimed.")
    out = Path(output_path)
    require(not out.exists(), "Evidence receipts are immutable")
    out.write_text(json.dumps(receipt, indent=2)+"\n")
    return receipt


def _network_get(url):
    # Never propagate an exception containing a secret URL/API key to the notebook.
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "E3-admission/1.0"}), timeout=60) as response:
            require(response.status == 200, "Provider HTTP failure")
            return response.read()
    except Exception:
        raise AdmissionError("Provider acquisition failed; inspect private provider logs, never publish API keys") from None


def collect_development(config, output_dir, *, rights_receipt, exposure_receipt, enable_market_acquisition=False):
    """Acquire only development, retaining attempts including failure/retry cost."""
    require(enable_market_acquisition, "Market acquisition requires explicit author activation")
    verify_evidence(rights_receipt, "approved")
    verify_evidence(exposure_receipt, "audited")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    event = {"started_utc": utc_now(), "config_sha256": identity(config), "scope": "development_only",
             "provider": config.get("provider"), "instrument": config.get("instrument"), "status": "started"}
    log = root/"acquisition_attempts.jsonl"
    with log.open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True)+"\n")
    try:
        result = _collect_development(config, output_dir, rights_receipt=rights_receipt,
                    exposure_receipt=exposure_receipt, enable_market_acquisition=True)
    except Exception as exc:
        event.update(status="failed", finished_utc=utc_now(), error_type=type(exc).__name__,
                     error="Acquisition incomplete; existing bytes retained, no admission. No secret URLs recorded.")
        with log.open("a") as stream:
            stream.write(json.dumps(event, sort_keys=True)+"\n")
        raise
    event.update(status="complete", finished_utc=utc_now(), manifest_sha256=result["manifest_sha256"],
                 files=len(result["files"]), bytes=sum((root/r["path"]).stat().st_size for r in result["files"]))
    with log.open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True)+"\n")
    return result


def _collect_development(config, output_dir, *, rights_receipt, exposure_receipt, enable_market_acquisition=False):
    """Real collectors, disabled by default; this entrypoint CANNOT acquire test.

    Binance uses bounded per-day archives and provider checksum companions.
    Alpha Vantage cannot date-bound its full response: direct development
    acquisition is permitted only when ALL reserved support is still future.
    Historical sealed reserves require the real author's or custodian's
    appropriately filtered files and documented exposure history.
    """
    require(enable_market_acquisition, "Market acquisition requires explicit author activation")
    verify_evidence(rights_receipt, "approved")
    verify_evidence(exposure_receipt, "audited")
    provider, symbol = config["provider"], config["instrument"]
    first, last = iso_date(config["development_start"]), iso_date(config["development_end"])
    development_only = exposure_receipt.get("form", {}).get("reserve_mode") == "development_only"
    if development_only:
        require(exposure_receipt.get("final_reserve_claim") is False,
                "Development-only receipt must explicitly disclaim a final reserve")
        reserve = None
        cutoff = dt.datetime.fromisoformat(exposure_receipt["form"]["observation_cutoff_utc"])
        require(last <= cutoff.astimezone(UTC).date(), "Development range exceeds actor's observed cutoff")
    else:
        reserve = iso_date(config["first_reserved_raw_session"])
        require(last < reserve, "Development range overlaps reserved raw support")
    require(first <= last and last < dt.datetime.now(UTC).date(), "Only completed development bars are permitted")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    require(not (root/"raw_manifest.json").exists(), "Use a new immutable acquisition directory")
    manifest = {"provider": provider, "instrument": symbol, "scope": "development_only", "files": [],
                "rights": rights_receipt, "exposure": exposure_receipt, "acquired_utc": utc_now()}
    if provider in BINANCE_MARKETS:
        return _collect_binance_daily(config,root,rights_receipt,exposure_receipt)
    elif provider == "nasdaq_wiki":
        require(symbol in ("IBM", "MSFT", "XOM"), "Unfrozen WIKI instrument")
        require(last <= dt.date(2018, 3, 27), "WIKI is historical; it cannot supply a post-protocol future reserve")
        key = os.environ.get("NASDAQ_DATA_LINK_API_KEY")
        require(bool(key), "Set private NASDAQ_DATA_LINK_API_KEY")
        cursor, seen, page = None, set(), 0
        while True:
            params = {"ticker": symbol, "date.gte": first.isoformat(), "date.lte": last.isoformat()}
            if cursor:
                params["qopts.cursor_id"] = cursor
            public_url = "https://data.nasdaq.com/api/v3/datatables/WIKI/PRICES.json?"+urllib.parse.urlencode(params)
            payload = _network_get(public_url+"&api_key="+urllib.parse.quote(key))
            bars, cursor = parse_nasdaq_wiki(payload, symbol)
            require(str(bars["dates"][0]) >= str(first) and str(bars["dates"][-1]) <= str(last), "WIKI violated bounded development query")
            name = f"{symbol}-wiki-page-{page:04d}.json"
            require(not (root/name).exists(), "Will not overwrite source data")
            (root/name).write_bytes(payload)
            manifest["files"].append({"path": name, "url": public_url+"&api_key=REDACTED", "sha256": sha256_bytes(payload)})
            if not cursor:
                break
            require(cursor not in seen, "WIKI cursor cycle")
            seen.add(cursor)
            page += 1
    elif provider == "alphavantage":
        require(reserve is None or reserve > dt.datetime.now(UTC).date(), "Full AV response may contain historical reserve; use custodian filtering")
        require(symbol in ("AAPL", "GLD", "TLT"), "Unfrozen equity instrument")
        key = os.environ.get("ALPHAVANTAGE_API_KEY")
        require(bool(key), "Set private ALPHAVANTAGE_API_KEY (never commit it)")
        params = {"function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": symbol, "outputsize": "full", "datatype": "json"}
        public_url = "https://www.alphavantage.co/query?"+urllib.parse.urlencode(params)
        payload = _network_get(public_url+"&apikey="+urllib.parse.quote(key))
        bars = parse_alphavantage(payload, symbol)
        require(reserve is None or str(bars["dates"][-1]) < str(reserve), "Provider response contains reserved bars")
        # Retain exact upstream response privately; no adjusted data are inferred.
        name = f"{symbol}-daily-adjusted.json"
        require(not (root/name).exists(), "Will not overwrite source data")
        (root/name).write_bytes(payload)
        manifest["files"].append({"path": name, "url": public_url+"&apikey=REDACTED", "sha256": sha256_bytes(payload)})
    else:
        raise AdmissionError("Unsupported/unlicensed provider; Stooq network collector intentionally unavailable")
    manifest["manifest_sha256"] = identity(manifest)
    (root/"raw_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def prepare_development(manifest_path, split_config, panel_dir, *, rights, exposure,
                        record_recovery=True):
    """Parse local acquired development files, validate clock, then label ONLY dev."""
    manifest_path = Path(manifest_path)
    record = json.loads(manifest_path.read_text())
    declared = record.pop("manifest_sha256")
    require(declared == identity(record) and record["scope"] == "development_only", "Invalid development manifest")
    require("test" not in split_config["raw_bounds"], "Development cannot prepare test")
    parsed = []
    for row in record["files"]:
        payload = _safe_path(manifest_path.parent, row["path"]).read_bytes()
        require(sha256_bytes(payload) == row["sha256"], "Raw byte hash mismatch")
        if record["provider"] in BINANCE_MARKETS:
            parsed.append(_parse_binance_manifest_payload(payload,record))
        elif record["provider"] == "alphavantage":
            parsed.append(parse_alphavantage(payload, record["instrument"]))
        elif record["provider"] == "nasdaq_wiki":
            parsed.append(parse_nasdaq_wiki(payload, record["instrument"])[0])
        else:
            raise AdmissionError("Unadmitted source")
    require(bool(parsed), "Empty raw manifest")
    bars = {k: np.concatenate([p[k] for p in parsed]) for k in ("dates", "ohlcv", "split")}
    order = np.argsort(bars["dates"])
    bars = {k: values[order] for k, values in bars.items()}
    # Source response can contain earlier development; trim BEFORE labels/windows.
    keep = (bars["dates"] >= split_config["raw_start"]) & (bars["dates"] <= split_config["raw_end"])
    bars = {k: v[keep] for k, v in bars.items()}
    sessions = expected_sessions(split_config["raw_start"], split_config["raw_end"], split_config["calendar"])
    quality = validate_bars(bars, sessions)
    arrays, windows_audit = build_windows(bars, target_end=split_config["raw_end"])
    parts, support = split_by_support(arrays, split_config["raw_bounds"], embargo_bars=split_config.get("embargo_bars", 0))
    admission = write_panel(panel_dir, parts, provenance={"provider": record["provider"], "instrument": record["instrument"],
        "provider_version": declared, "raw_manifest_sha256": declared, "calendar": split_config["calendar"],
        "bar_index_origin": str(bars["dates"][0]), "bar_index_convention": BAR_INDEX_CONVENTION,
        "split_config_sha256": identity(split_config),
        **({"market_type":BINANCE_MARKETS[record["provider"]]["market_type"],
            "volume_units":"base_asset", "data_adapter_version":BINANCE_DATA_ADAPTER_VERSION}
           if record["provider"] in BINANCE_MARKETS else {})}, rights=rights, exposure=exposure,
        support_audit={"quality": quality, "windows": windows_audit, "partitions": support})
    if record_recovery:
        _write_recovery_metadata(manifest_path, split_config, panel_dir, admission)
    return admission


def _encoded_metadata(path, expected=None):
    path = Path(os.path.abspath(path))
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Recovery metadata path cannot traverse symlinks")
    require(path.suffix.lower() not in (".npz", ".npy", ".pt", ".pth", ".pkl", ".pickle"),
            "Recovery metadata cannot contain arrays or model weights")
    raw = path.read_bytes()
    require(len(raw) <= 16*1024*1024, "Individual metadata export exceeds 16 MiB")
    checksum = sha256_bytes(raw)
    require(expected is None or checksum == expected, "Original metadata evidence bytes/hash mismatch")
    return {"path": str(path), "sha256": checksum, "content_base64": base64.b64encode(raw).decode()}


def _decoded_metadata(item):
    require(isinstance(item, dict) and isinstance(item.get("content_base64"), str), "Encoded recovery metadata required")
    require(len(item["content_base64"]) <= 24*1024*1024, "Encoded metadata exceeds bound")
    try:
        raw = base64.b64decode(item["content_base64"], validate=True)
    except (ValueError, TypeError):
        raise AdmissionError("Invalid encoded recovery metadata") from None
    require(len(raw) <= 16*1024*1024 and sha256_bytes(raw) == item.get("sha256"), "Recovery metadata bytes/hash mismatch")
    return raw


def _write_recovery_metadata(manifest_path, split_config, panel_dir, admission):
    """Snapshot metadata/evidence only, after the original admission is fixed.

    No raw OHLCV arrays, upstream payloads, weights or credentials are included.
    Replaying original event metadata never claims a reacquisition happened at
    the old timestamp. The later replay event is recorded separately.
    """
    root = Path(panel_dir)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    artifacts = {}
    def visit(value):
        if isinstance(value, dict):
            for key, path in value.items():
                if isinstance(path, str) and key.endswith("_path"):
                    hash_key = key[:-5]+"_sha256"
                    if key == "historical_seal_evidence_path": hash_key = "custodian_seal_sha256"
                    if key == "historical_access_log_evidence_path": hash_key = "custodian_access_log_sha256"
                    if key == "exposure_inventory_path": hash_key = "exposure_inventory_sha256"
                    expected = value.get(hash_key)
                    if expected:
                        item = _encoded_metadata(path, expected)
                        previous = artifacts.get(item["path"])
                        require(previous is None or previous == item, "Conflicting original evidence bindings")
                        artifacts[item["path"]] = item
                visit(path)
        elif isinstance(value, list):
            for item in value: visit(item)
    visit(admission["rights"]); visit(admission["exposure"])
    visit(manifest.get("rights", {})); visit(manifest.get("exposure", {}))
    sources, companions = [], []
    for row in manifest["files"]:
        _recovery_download_url(row["url"], manifest["provider"], private=False)
        raw_path = _safe_path(manifest_path.parent, row["path"])
        require(sha256_bytes(raw_path.read_bytes()) == row["sha256"], "Original source changed before recovery snapshot")
        sources.append({"path": row["path"], "sha256": row["sha256"], "bytes": raw_path.stat().st_size})
        if row.get("provider_checksum_sha256"):
            checksum_path = _safe_path(manifest_path.parent, row["path"]+".CHECKSUM")
            item = _encoded_metadata(checksum_path, row["provider_checksum_sha256"])
            item["relative_path"] = row["path"]+".CHECKSUM"
            companions.append(item)
    for name in ("acquisition_attempts.jsonl", "acquisition_pending.json"):
        companion = manifest_path.parent/name
        if companion.is_file():
            item = _encoded_metadata(companion); item["relative_path"] = companion.name
            companions.append(item)
    result = {"schema_version": 1, "kind": "development_metadata_replay_v1", "created_utc": utc_now(),
              "contains_raw_arrays": False, "contains_model_weights": False, "contains_credentials": False,
              "raw_manifest": _encoded_metadata(manifest_path), "split_config": split_config,
              "original_admission": _encoded_metadata(root/"admission.json"),
              "original_panel_dir": str(root.absolute()), "original_raw_dir": str(manifest_path.parent.absolute()),
              "provider": manifest["provider"], "instrument": manifest["instrument"], "sources": sources,
              "evidence": list(artifacts.values()), "raw_metadata_companions": companions,
              "scope": "Replay the original byte identities; later acquisition events have separate timestamps/receipts."}
    result["metadata_sha256"] = identity(result)
    raw = (json.dumps(result, indent=2)+"\n").encode()
    require(len(raw) <= 64*1024*1024, "Recovery metadata bundle exceeds 64 MiB")
    target = root/"recovery_metadata.json"
    require(not target.exists(), "Recovery metadata is immutable")
    target.write_bytes(raw)
    return result


def _recovery_download_url(url, provider, *, private=True):
    parsed = urllib.parse.urlsplit(url)
    hosts = {"binance_vision": "data.binance.vision", "binance_vision_um": "data.binance.vision",
             "binance_vision_cm": "data.binance.vision", "nasdaq_wiki": "data.nasdaq.com",
             "alphavantage": "www.alphavantage.co"}
    require(parsed.scheme == "https" and parsed.hostname == hosts.get(provider) and
            parsed.username is None and parsed.password is None and not parsed.fragment,
            "Recovery source URL outside the frozen provider")
    if provider in BINANCE_MARKETS:
        root_prefix = "/data/"+BINANCE_MARKETS[provider]["prefix"]
        require(any(parsed.path.startswith(root_prefix+"/"+frequency+"/klines/") for frequency in ("daily","monthly")) and not parsed.query,
                "Binance recovery URL market or frequency differs from its provider")
    params = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    key_name = {"nasdaq_wiki": "api_key", "alphavantage": "apikey"}.get(provider)
    if key_name:
        require(sum(k == key_name for k, _ in params) == 1 and
                all(v == "REDACTED" for k, v in params if k == key_name),
                "Original recovery URL must redact its API key")
        if private:
            environment = "NASDAQ_DATA_LINK_API_KEY" if provider == "nasdaq_wiki" else "ALPHAVANTAGE_API_KEY"
            key = os.environ.get(environment)
            require(bool(key), "Set private "+environment+" for reacquisition")
            params = [(k, key if k == key_name else v) for k, v in params]
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path,
                                   urllib.parse.urlencode(params), ""))


def _restore_metadata_file(target, raw, workspace_root, *, allow_existing_outside=False):
    root = Path(os.path.abspath(workspace_root))
    target = Path(os.path.abspath(target))
    require(not any(p.is_symlink() for p in (root, *root.parents, target, *target.parents)),
            "Recovery path cannot traverse symlinks")
    inside = target == root or root in target.parents
    require(inside or allow_existing_outside and target.is_file(),
            "Recovery writes must stay inside the supplied workspace")
    if target.exists():
        require(target.is_file() and target.read_bytes() == raw, "Recovery refuses to overwrite different existing bytes")
        return
    require(inside, "Missing source-cache evidence outside workspace must be reacquired by the pinned source installer")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        temporary = Path(stream.name); stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    try:
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _read_recovery_bundle(bundle_path, expected_bundle_sha256=None):
    bundle_raw = Path(bundle_path).read_bytes()
    require(expected_bundle_sha256 is None or sha256_bytes(bundle_raw) == expected_bundle_sha256,
            "Recovery bundle differs from the frozen campaign")
    require(len(bundle_raw) <= 64*1024*1024, "Recovery bundle exceeds byte limit")
    bundle = json.loads(bundle_raw)
    claimed = bundle.pop("metadata_sha256", None)
    require(claimed == identity(bundle) and bundle.get("kind") == "development_metadata_replay_v1" and
            bundle.get("contains_raw_arrays") is False and bundle.get("contains_model_weights") is False and
            bundle.get("contains_credentials") is False,
            "Invalid development recovery metadata")
    manifest_raw = _decoded_metadata(bundle["raw_manifest"])
    original_raw = _decoded_metadata(bundle["original_admission"])
    manifest, original = json.loads(manifest_raw), json.loads(original_raw)
    require(manifest.get("scope") == "development_only" and "test" not in original["partition_paths"],
            "Recovery cannot acquire or prepare a final test")
    require(manifest["manifest_sha256"] == identity({k:v for k,v in manifest.items() if k != "manifest_sha256"}),
            "Original acquisition manifest identity differs")
    require(original["admission_sha256"] == identity({k:v for k,v in original.items() if k != "admission_sha256"}),
            "Original admission identity differs")
    return bundle_raw, bundle, manifest_raw, original_raw, manifest, original


def restore_development_metadata(bundle_path, workspace_root, *, expected_bundle_sha256=None):
    """Restore saved actor/license bytes without acquiring observations."""
    _, bundle, _, _, _, original = _read_recovery_bundle(bundle_path, expected_bundle_sha256)
    root = Path(os.path.abspath(workspace_root))
    require(not any(p.is_symlink() for p in (root, *root.parents)), "Recovery workspace cannot traverse symlinks")
    for artifact in bundle["evidence"]:
        _restore_metadata_file(artifact["path"], _decoded_metadata(artifact), root, allow_existing_outside=True)
    verify_evidence(original["rights"], "approved"); verify_evidence(original["exposure"], "audited")
    return {"status": "DEVELOPMENT_AUTHORIZATION_METADATA_RESTORED", "provider": bundle["provider"],
            "original_panel_dir": bundle["original_panel_dir"], "original_raw_dir": bundle["original_raw_dir"],
            "new_market_downloads": 0, "new_market_fits": 0, "reserved_test_opened": False}


def reacquire_development(bundle_path, panel_dir, raw_dir, workspace_root, *,
                          enable_market_acquisition=False, expected_bundle_sha256=None):
    """Rehydrate a saved development identity, not a newly dated campaign.

    The original manifest, partitions, admission and authorization bytes stay
    fixed. Private credentials are supplied again only to the upstream client.
    No new declaration or final-data access is performed.
    """
    require(enable_market_acquisition, "Development reacquisition requires explicit author activation")
    bundle_raw, bundle, manifest_raw, original_raw, manifest, original = _read_recovery_bundle(bundle_path, expected_bundle_sha256)
    root = Path(os.path.abspath(workspace_root)); panel_dir = Path(os.path.abspath(panel_dir)); raw_dir = Path(os.path.abspath(raw_dir))
    for directory in (root, panel_dir, raw_dir):
        require(directory == root or root in directory.parents, "Recovery directories must stay inside the supplied workspace")
        require(not any(p.is_symlink() for p in (directory, *directory.parents)), "Recovery directories cannot traverse symlinks")
    restore_development_metadata(bundle_path, root, expected_bundle_sha256=expected_bundle_sha256)
    if (panel_dir/"admission.json").exists():
        require((panel_dir/"admission.json").read_bytes() == original_raw, "Existing restored admission differs")
        for part in original["partition_paths"]: load_partition(panel_dir, part)
        return {"status": "DEVELOPMENT_ALREADY_RESTORED", "admission_sha256": original["admission_sha256"], "new_market_fits": 0}
    # Restore exact historical authorization/license exports. A source-cache
    # artifact outside WORK is read-only and must already match the pinned copy.
    expected_sources = {row["path"]: row for row in bundle["sources"]}
    require(set(expected_sources) == {row["path"] for row in manifest["files"]}, "Recovery source inventory differs")
    require(all(expected_sources[row["path"]]["sha256"] == row["sha256"] for row in manifest["files"]),
            "Recovery source hash inventory differs")
    raw_dir.mkdir(parents=True, exist_ok=True)
    actual_event = {"started_utc": utc_now(), "status": "started", "original_acquired_utc": manifest["acquired_utc"],
                    "original_manifest_replayed": True, "new_acquisition_manifest_created": False,
                    "bundle_sha256": sha256_bytes(bundle_raw), "original_manifest_sha256": manifest["manifest_sha256"],
                    "network_downloads": 0, "raw_files_reused": 0}
    event_path = raw_dir/"reacquisition_events.jsonl"
    def record_event():
        with event_path.open("a") as stream: stream.write(json.dumps(actual_event, sort_keys=True)+"\n")
    record_event()
    try:
        for row in manifest["files"]:
            target = _safe_path(raw_dir, row["path"])
            if target.is_file():
                payload = target.read_bytes(); actual_event["raw_files_reused"] += 1
            else:
                payload = _network_get(_recovery_download_url(row["url"], manifest["provider"]))
                actual_event["network_downloads"] += 1
            require(len(payload) == expected_sources[row["path"]]["bytes"] and sha256_bytes(payload) == row["sha256"],
                    "Reacquired raw bytes differ from the original saved source")
            _restore_metadata_file(target, payload, root)
        for item in bundle["raw_metadata_companions"]:
            _restore_metadata_file(_safe_path(raw_dir, item["relative_path"]), _decoded_metadata(item), root)
        manifest_target = raw_dir/"raw_manifest.json"
        _restore_metadata_file(manifest_target, manifest_raw, root)
        panel_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="."+panel_dir.name+"-recovery-", dir=panel_dir.parent))
        rebuilt = prepare_development(manifest_target, bundle["split_config"], staging,
                                     rights=original["rights"], exposure=original["exposure"], record_recovery=False)
        comparable = lambda value: {k:v for k,v in value.items() if k not in ("created_utc", "admission_sha256")}
        require(comparable(rebuilt) == comparable(original), "Rebuilt partitions or provenance differ from the original admission")
        # All partition hashes matched. Reinstall the preserved event record;
        # the new real reacquisition timestamp stays in its separate log.
        (staging/"admission.json").write_bytes(original_raw)
        _restore_metadata_file(staging/"recovery_metadata.json", bundle_raw, root)
        if panel_dir.exists():
            require(panel_dir.is_dir() and not any(panel_dir.iterdir()), "Nonempty incomplete panel retained; restore to a new directory")
        os.replace(staging, panel_dir)
        actual_event.update(status="complete", finished_utc=utc_now(), source_bytes_identical=True,
                            admission_sha256=original["admission_sha256"], new_owner_declaration=False)
        record_event()
    except Exception as error:
        actual_event.update(status="failed", finished_utc=utc_now(), error_type=type(error).__name__,
                            diagnostic="Reacquisition stopped; original identities and existing bytes retained.")
        record_event()
        raise
    return {"status": "DEVELOPMENT_RESTORED_ORIGINAL_IDENTITY", "admission_sha256": original["admission_sha256"],
            "reacquisition_receipt": str(event_path), "new_market_fits": 0, "reserved_test_opened": False}


def prepare_final_partition(manifest_path, split_config, panel_dir, *, rights, exposure,
                            authorization, enable_final_preparation=False):
    """Explicit final-only local preparation, after the outer final gate.

    Does not acquire data or evaluate models. The author or custodian supplies an immutable
    `scope=reserved_final` raw manifest. All current raw bytes and authorization
    bytes are verified BEFORE labels are constructed. Output is a new immutable
    panel directory; development directories cannot be silently overwritten.
    """
    require(enable_final_preparation, "Final preparation requires explicit author activation")
    verify_evidence(rights, "approved")
    verify_evidence(exposure, "audited")
    verify_evidence(authorization, "authorized")
    require(authorization.get("authorize_test_preparation") is True, "Author did not authorize final preparation")
    protocol_path = Path(authorization["protocol_path"])
    protocol_hash = sha256_bytes(protocol_path.read_bytes())
    require(protocol_hash == authorization.get("protocol_sha256") == split_config.get("protocol_sha256"), "Protocol hash mismatch")
    require(authorization.get("selected_checkpoints_sha256") and authorization.get("sources_methods_size_frozen") is True,
            "Outer final controller must fix and verify selected checkpoints, sources, methods, partitions and sample size")
    checkpoint_path = Path(authorization["selected_checkpoints_manifest_path"])
    require(sha256_bytes(checkpoint_path.read_bytes()) == authorization["selected_checkpoints_sha256"], "Selected checkpoints manifest mismatch")
    require(set(split_config["raw_bounds"]) == {"test"}, "Final preparation accepts only final bounds")
    manifest_path = Path(manifest_path)
    record = json.loads(manifest_path.read_text())
    claimed = record.pop("manifest_sha256")
    require(claimed == identity(record) and record.get("scope") == "reserved_final", "Require author/custodian reserved-final manifest")
    require(record.get("protocol_sha256") == protocol_hash, "Reserved raw manifest not bound to fixed protocol")
    reserve_form = exposure.get("form", {})
    require(reserve_form.get("protocol_sha256") == protocol_hash, "Exposure declaration not bound to final protocol")
    require(reserve_form.get("reserve_mode") in (*HISTORICAL_RESERVE_MODES, "post_protocol"), "Authentic reserve mode missing")
    if reserve_form["reserve_mode"] in HISTORICAL_RESERVE_MODES:
        _verify_historical_reserve(reserve_form)
        require(record.get("custodian_seal_sha256") == reserve_form["custodian_seal_sha256"], "Custodian seal binding mismatch")
    else:
        require(record["provider"] != "nasdaq_wiki", "Discontinued WIKI cannot supply a post-protocol future reserve")
    parsed = []
    for row in record["files"]:
        payload = _safe_path(manifest_path.parent, row["path"]).read_bytes()
        require(sha256_bytes(payload) == row["sha256"], "Reserved raw byte hash mismatch")
        if record["provider"] in BINANCE_MARKETS:
            parsed.append(_parse_binance_manifest_payload(payload,record))
        elif record["provider"] == "nasdaq_wiki":
            parsed.append(parse_nasdaq_wiki(payload, record["instrument"])[0])
        elif record["provider"] == "alphavantage":
            parsed.append(parse_alphavantage(payload, record["instrument"]))
        else:
            raise AdmissionError("Unsupported reserved source")
    require(bool(parsed), "No reserved source files")
    bars = {k: np.concatenate([p[k] for p in parsed]) for k in ("dates", "ohlcv", "split")}
    order = np.argsort(bars["dates"])
    bars = {k: v[order] for k, v in bars.items()}
    quality = validate_bars(bars, expected_sessions(split_config["raw_start"], split_config["raw_end"], split_config["calendar"]))
    # A local index such as 500 is only comparable with development index 500
    # when both clocks begin at the same actual observed calendar session.
    # This metadata check precedes any construction of reserved labels.
    origin = authorization.get("development_bar_index_origin")
    require(isinstance(origin, str), "Frozen development bar index origin required")
    iso_date(origin)
    require(str(bars["dates"][0]) == origin, "Reserved raw bar index origin differs from frozen development")
    arrays, windows = build_windows(bars, target_start=split_config["target_start"], target_end=split_config["target_end"])
    parts, support = split_by_support(arrays, split_config["raw_bounds"],
                                    embargo_bars=split_config.get("embargo_bars", 0), allow_test=True)
    if reserve_form["reserve_mode"] == "post_protocol":
        lock = dt.datetime.fromisoformat(reserve_form["protocol_lock_utc"])
        require(lock.tzinfo is not None, "Timezone-aware complete protocol lock required")
        first_support = int(parts["test"]["support_start"].min())
        require(iso_date(bars["dates"][first_support]) > lock.astimezone(UTC).date(), "Final support overlaps pre-protocol observations")
    require(len(parts["test"]["y"]) == split_config["fixed_test_count"], "Final endpoint count differs from frozen sizing")
    # Do not print labels, summaries or model comparisons during preparation.
    return write_panel(panel_dir, parts, provenance={"provider": record["provider"], "instrument": record["instrument"],
        "provider_version": claimed, "raw_manifest_sha256": claimed, "calendar": split_config["calendar"],
        "bar_index_origin": str(bars["dates"][0]), "bar_index_convention": BAR_INDEX_CONVENTION,
        "split_config_sha256": identity(split_config), "protocol_sha256": protocol_hash,
        **({"market_type":BINANCE_MARKETS[record["provider"]]["market_type"],
            "volume_units":"base_asset", "data_adapter_version":BINANCE_DATA_ADAPTER_VERSION}
           if record["provider"] in BINANCE_MARKETS else {})}, rights=rights, exposure=exposure,
        support_audit={"quality": quality, "windows": windows, "partitions": support}, final_authorization=authorization)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evidence = sub.add_parser("record-owner-evidence")
    for flag in ("form", "evidence", "output"):
        evidence.add_argument("--"+flag, required=True)
    acquire = sub.add_parser("collect-development")
    for flag in ("config", "rights", "exposure", "output"):
        acquire.add_argument("--"+flag, required=True)
    acquire.add_argument("--enable-market-acquisition", action="store_true")
    prepare = sub.add_parser("prepare-development")
    for flag in ("manifest", "split-config", "rights", "exposure", "output"):
        prepare.add_argument("--"+flag, required=True)
    recovery = sub.add_parser("reacquire-development")
    for flag in ("bundle", "output", "raw-output", "workspace-root"):
        recovery.add_argument("--"+flag, required=True)
    recovery.add_argument("--expected-bundle-sha256")
    recovery.add_argument("--enable-market-acquisition", action="store_true")
    recovery_metadata = sub.add_parser("restore-development-metadata")
    for flag in ("bundle", "workspace-root"):
        recovery_metadata.add_argument("--"+flag, required=True)
    recovery_metadata.add_argument("--expected-bundle-sha256")
    final = sub.add_parser("prepare-final")
    for flag in ("manifest", "split-config", "rights", "exposure", "authorization", "output"):
        final.add_argument("--"+flag, required=True)
    final.add_argument("--enable-final-preparation", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "record-owner-evidence":
        result = record_owner_evidence(args.form, args.evidence, args.output)
    elif args.command == "collect-development":
        result = collect_development(json.loads(Path(args.config).read_text()), args.output,
            rights_receipt=json.loads(Path(args.rights).read_text()), exposure_receipt=json.loads(Path(args.exposure).read_text()),
            enable_market_acquisition=args.enable_market_acquisition)
    elif args.command == "prepare-development":
        result = prepare_development(args.manifest, json.loads(Path(args.split_config).read_text()), args.output,
            rights=json.loads(Path(args.rights).read_text()), exposure=json.loads(Path(args.exposure).read_text()))
    elif args.command == "restore-development-metadata":
        result = restore_development_metadata(args.bundle, args.workspace_root,
            expected_bundle_sha256=args.expected_bundle_sha256)
    elif args.command == "reacquire-development":
        result = reacquire_development(args.bundle, args.output, args.raw_output, args.workspace_root,
            expected_bundle_sha256=args.expected_bundle_sha256, enable_market_acquisition=args.enable_market_acquisition)
    else:
        result = prepare_final_partition(args.manifest, json.loads(Path(args.split_config).read_text()), args.output,
            rights=json.loads(Path(args.rights).read_text()), exposure=json.loads(Path(args.exposure).read_text()),
            authorization=json.loads(Path(args.authorization).read_text()), enable_final_preparation=args.enable_final_preparation)
    print(json.dumps({k: result.get(k) for k in ("status", "kind", "admission_sha256", "manifest_sha256") if k in result}, indent=2))


if __name__ == "__main__":
    main()
