"""Explicit post-freeze target history for the endogenous ARIMA reference.

Metadata reservation does not open prices. The experiment runner, not a call to
this loader alone, enforces closure of ALL primary and branch-A predictors before
decoding the historical state. These already-known E1 dates are not a new test.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from planned_study_data import _aggregate_payload, _safe_path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "data/manifests/binance_spot_BTCUSDT_1m_20230101_20261030.json"
SOURCE_MONTHS = tuple(pd.date_range("2024-12-01", "2026-08-01", freq="MS").strftime("%Y-%m"))
STATE_START, STATE_END = date(2024, 12, 25), date(2026, 8, 31)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def reserve_final_state_metadata(project_root=ROOT, raw_root=None):
    """Bind 21 known archive identities, without reading any archive payload."""
    path = Path(project_root) / MANIFEST
    original = json.loads(path.read_bytes())
    if original.get("symbol") != "BTCUSDT" or original.get("interval") != "1m":
        raise ValueError("The inherited BTCUSDT minute-source identity differs")
    by_name = {r["archive"]: r for r in original["archives"]}
    records, stat = [], []
    for month in SOURCE_MONTHS:
        name = f"BTCUSDT-1m-{month}.zip"
        record = by_name.get(name)
        if not record or record.get("status") != "verified" or record.get("kind") != "monthly":
            raise ValueError("Missing verified history archive: " + name)
        if len(record.get("sha256", "")) != 64:
            raise ValueError("History archive needs its acquisition SHA256")
        expected = f"BTCUSDT/1m/monthly/{name}"
        if record.get("relative_path") != expected:
            raise ValueError("State-history archive path differs")
        records.append({key: record[key] for key in
                        ("archive", "kind", "relative_path", "bytes", "sha256",
                         "first_open_utc", "last_open_utc", "timestamp_units")})
        if raw_root is not None:
            local = _safe_path(raw_root, expected)
            if not local.is_file() or local.stat().st_size != record["bytes"]:
                raise ValueError("Missing or wrong-sized state archive: " + name)
            stat.append({"archive": name, "present_with_expected_size": True})
    return {"schema_version": 1, "scope": "post-global-freeze causal ARIMA state only",
            "manifest": MANIFEST, "source_manifest_sha256": digest(path),
            "source_months": list(SOURCE_MONTHS), "records": records,
            "first_state_label": STATE_START.isoformat(), "last_state_label": STATE_END.isoformat(),
            "archive_payloads_opened_by_reservation": 0,
            "known_historical_E1_period": True, "stat_verification": stat if raw_root is not None else None}


def _identity(reservation):
    payload = dict(reservation); payload.pop("stat_verification", None)
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def state_from_daily_closes(daily):
    """Keep gaps; the first 24 December days provide genuine return warm-up."""
    expected = pd.date_range("2024-12-01", "2026-08-31", freq="D", tz="UTC")
    if not daily.index.equals(expected):
        raise ValueError("State reconstruction requires the complete uncompressed daily calendar")
    closes = daily["close"].to_numpy(dtype=np.float64)
    finite = np.isfinite(closes)
    if np.isinf(closes).any() or np.any(closes[finite] <= 0):
        raise ValueError("State closes must be positive or explicitly missing")
    values = pd.Series(np.log(closes), index=expected).diff().rolling(7, min_periods=7).std(ddof=0)
    selected = values.loc[STATE_START.isoformat():STATE_END.isoformat()]
    return {timestamp.date(): float(value) if np.isfinite(value) else None
            for timestamp, value in selected.items()}


def load_final_state_history(project_root, raw_root, reservation, output_dir):
    """Decode only after the caller has verified the global checkpoint freeze.

    No fit or hyperparameter selection occurs here. Resume accepts only a closed
    cache with unchanged reservation and derivative bytes; an interrupted first
    decode is retained for audit instead of silently replacing its evidence.
    """
    expected = reserve_final_state_metadata(project_root)
    identity = _identity(reservation)
    if identity != _identity(expected):
        raise ValueError("State reservation differs from the registered acquisition manifest")
    output = Path(output_dir).resolve()
    if output.is_relative_to(Path(project_root).resolve()):
        raise ValueError("Historical state derivatives must remain private, outside the repository")
    output.mkdir(parents=True, exist_ok=True)
    marker, receipt_path, state_path = (output / name for name in
                                      ("first_state_payload_access.json", "state_history_receipt.json", "state_history.csv"))
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("reservation_identity_sha256") != identity
                or not marker.is_file() or digest(state_path) != receipt.get("state_history_sha256")):
            raise ValueError("Closed state-history cache identity changed")
        table = pd.read_csv(state_path, keep_default_na=False)
        dates = pd.to_datetime(table["outcome_day"], utc=True)
        wanted = pd.date_range(STATE_START, STATE_END, freq="D", tz="UTC")
        if not pd.DatetimeIndex(dates).equals(wanted):
            raise ValueError("State-history cache lost or reordered calendar days")
        history = {}
        for day_text, value in zip(table["outcome_day"], table["sigma7"]):
            number = None if value == "" else float(value)
            if number is not None and (not np.isfinite(number) or number < 0):
                raise ValueError("Invalid cached state label")
            history[date.fromisoformat(day_text)] = number
        return {"history": history, "receipt": receipt}
    if marker.exists() or state_path.exists():
        raise ValueError("Interrupted state decode requires an explicit audit, not an overwrite")
    _write(marker, {"at_utc": datetime.now(timezone.utc).isoformat(),
                    "reservation_identity_sha256": identity,
                    "caller_must_have_verified_all_primary_and_branch_A_weights_closed": True,
                    "scope": "Known E1 history decoded for causal state update, never for current HPO"})
    frames, checks = [], []
    for record in reservation["records"]:
        payload = _safe_path(raw_root, record["relative_path"]).read_bytes()
        daily, quality = _aggregate_payload(payload, record)
        frames.append(daily); checks.append(quality)
    history = state_from_daily_closes(pd.concat(frames))
    table = pd.DataFrame({"outcome_day": [day.isoformat() for day in history],
                          "sigma7": [history[day] for day in history]})
    table.to_csv(state_path, index=False, mode="x")
    receipt = {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
               "reservation_identity_sha256": identity,
               "source_manifest_sha256": expected["source_manifest_sha256"],
               "state_history_sha256": digest(state_path), "source_archives_opened": len(checks),
               "quality_by_archive": checks, "state_calendar_days": len(history),
               "missing_state_labels": sum(value is None for value in history.values()),
               "first_state_label": STATE_START.isoformat(), "last_state_label": STATE_END.isoformat(),
               "population_standard_deviation_ddof": 0, "return_count": 7,
               "model_refits": 0, "HPO_or_weight_selection": False,
               "known_historical_period_not_new_unseen_test": True}
    _write(receipt_path, receipt)
    return {"history": history, "receipt": receipt}
