"""Hash-pinned, admission-limited daily E1 development derivatives.

The only files accepted are the two registered 2023--2024 CSVs plus their
historical protocol. Whole raw sources are hashed as bytes. Only date fields
are inspected outside the admitted prefix: later feature/target cells are not
decoded or converted. There is no raw-archive loader or 2025+ data path.
"""
from __future__ import annotations

import csv
from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .e2_temporal_training import DatedPanel
except ImportError:
    from e2_temporal_training import DatedPanel

VERSION = "e2-development-admission-v1"
PROTOCOL_PATH = "experiments/dl_r2_cv_20261104/protocol.json"
EXPECTED_PROTOCOL_SHA256 = "e8da0f0098acd24a58664b83d0ba499b43aa9d597a36c52da6b3325c2c3c7bec"
MARKET_PATH = "results/eda_development/daily_market_development.csv"
TARGET_PATH = "results/eda_development/daily_volatility_development.csv"
EXPECTED_SOURCES = {
    MARKET_PATH: "bf6bb5160af924aef6a15c3684aeae932df3017e8639888bb7e91ce787900171",
    TARGET_PATH: "20b96374a659eb87dadfa68aa3d6b6b5fb6bea5b10eebe6f2c186ad69861844a",
}
SOURCE_CALENDAR = pd.date_range("2023-01-01", "2024-12-31", tz="UTC")
ADMISSIONS = {
    "2023-09-23": {"calendar_end": "2023-10-01", "purpose": "pre-2024 resource/noise pilot"},
    "2024-12-30": {"calendar_end": "2024-12-30", "purpose": "preselection development power pilot"},
}
FEATURES = ("open", "high", "low", "close", "volume")
HEADERS = {
    MARKET_PATH: ("market_day_utc", "open", "high", "low", "close", "volume", "trades"),
    TARGET_PATH: ("market_day_utc", "sigma_7", "sigma_14", "sigma_21", "sigma_28"),
}


def _sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _read_registered(root, relative, expected):
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError("A regular registered source file is required: "+relative)
    data = path.read_bytes()
    actual = _sha_bytes(data)
    if actual != expected:
        raise ValueError("Registered source SHA256 mismatch: "+relative)
    return data, actual


def _prefix_values(raw, path, admitted_end):
    """Parse all dates but no excluded numerical cells from a trusted source."""
    lines = raw.splitlines()
    if not lines:
        raise ValueError("Registered source is empty")
    header = tuple(next(csv.reader([lines[0].decode("utf-8")])))
    if header != HEADERS[path]:
        raise ValueError("Registered daily source columns changed")
    wanted = FEATURES if path == MARKET_PATH else ("sigma_7",)
    columns = [header.index(column) for column in wanted]
    dates, prefix_dates, values = [], [], []
    for line in lines[1:]:
        if not line:
            raise ValueError("Blank daily source rows are not admissible")
        # Parse only the first field on excluded dates; never decode its label
        # or feature cells. The registered source uses an unquoted date field.
        date_bytes, separator, _ = line.partition(b",")
        if not separator:
            raise ValueError("Daily source row has no date separator")
        stamp = pd.Timestamp(date_bytes.decode("utf-8"))
        if pd.isna(stamp) or stamp.tzinfo is None:
            raise ValueError("Source dates must explicitly identify UTC instants")
        stamp = stamp.tz_convert("UTC")
        if stamp != stamp.normalize():
            raise ValueError("Daily source timestamps must be UTC midnights")
        dates.append(stamp)
        if stamp > admitted_end:
            continue
        fields = next(csv.reader([line.decode("utf-8")]))
        if len(fields) != len(header):
            raise ValueError("Admitted daily source row has a different column count")
        row = []
        for column in columns:
            value = fields[column].strip()
            row.append(float(value) if value else float("nan"))
        values.append(row)
        prefix_dates.append(stamp)
    calendar = pd.DatetimeIndex(dates)
    if not calendar.equals(SOURCE_CALENDAR):
        raise ValueError("Registered source must retain the complete 2023--2024 UTC calendar")
    prefix = pd.DatetimeIndex(prefix_dates)
    expected = SOURCE_CALENDAR[SOURCE_CALENDAR <= admitted_end]
    if not prefix.equals(expected):
        raise ValueError("Admitted prefix is incomplete or reordered")
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(expected), len(wanted)):
        raise ValueError("Admitted numerical values differ from their dated prefix")
    if np.isinf(array).any():
        raise ValueError("Infinite values in admitted development observations")
    return prefix, array, {
        "date_fields_inspected": len(calendar),
        "numerical_rows_decoded": len(prefix),
        "numerical_values_after_admission_decoded": False,
        "source_calendar_start": str(calendar[0].date()),
        "source_calendar_end": str(calendar[-1].date()),
    }


def load_development(root, admitted_end="2023-09-23"):
    """Return (DatedPanel, receipt) for one of two explicit development prefixes.

    Resource/noise admission includes labels through 2023-09-23 only and appends
    unavailable rows through the 2023-10-01 origin. Power-pilot admission includes
    development labels through 2024-12-30, whose outer observations must remain
    outside the fit callbacks. The prepared split/replay API enforces that
    chronology; this loader is not a sandbox against deliberate array capture.
    """
    if not isinstance(admitted_end, str) or admitted_end not in ADMISSIONS:
        raise ValueError("Choose the registered 2023-09-23 or 2024-12-30 development admission")
    root = Path(root)
    if root.is_symlink():
        raise ValueError("The development cache root must not be a symlink")
    cutoff = pd.Timestamp(admitted_end, tz="UTC")
    protocol_raw, protocol_hash = _read_registered(root, PROTOCOL_PATH, EXPECTED_PROTOCOL_SHA256)
    protocol = json.loads(protocol_raw)
    source_map = protocol.get("sources", {})
    if any(source_map.get(path) != digest for path, digest in EXPECTED_SOURCES.items()):
        raise ValueError("Historical protocol does not identify the pinned daily derivatives")
    parsed, parsing, hashes = {}, {}, {}
    for path, expected in EXPECTED_SOURCES.items():
        raw, hashes[path] = _read_registered(root, path, expected)
        dates, parsed[path], parsing[path] = _prefix_values(raw, path, cutoff)
        if path == MARKET_PATH:
            market_dates = dates
        elif not market_dates.equals(dates):
            raise ValueError("Market and target admitted calendars differ")
    features = parsed[MARKET_PATH]
    targets = parsed[TARGET_PATH][:, 0]
    complete = np.isfinite(features).all(axis=1)
    incomplete_dates = market_dates[~complete].strftime("%Y-%m-%d").tolist()
    expected_missing = ["2023-03-24"] if cutoff >= pd.Timestamp("2023-03-24", tz="UTC") else []
    if incomplete_dates != expected_missing:
        raise ValueError("The established incomplete-day mask changed inside admission")
    observed = features[complete]
    if (np.any(observed[:, :4] <= 0) or np.any(observed[:, 4] < 0)
            or np.any(observed[:, 1] < np.maximum(observed[:, 0], observed[:, 3]))
            or np.any(observed[:, 2] > np.minimum(observed[:, 0], observed[:, 3]))
            or np.any(observed[:, 1] < observed[:, 2])):
        raise ValueError("Invalid admitted OHLCV geometry")
    returns = pd.Series(np.log(features[:, 3]), index=market_dates).diff()
    reconstructed = returns.rolling(7, min_periods=7).std(ddof=0).to_numpy()
    if not np.array_equal(np.isfinite(targets), np.isfinite(reconstructed)):
        raise ValueError("sigma_7 missingness differs from the full-calendar population target")
    if not np.allclose(targets, reconstructed, rtol=1e-9, atol=1e-13, equal_nan=True):
        raise ValueError("sigma_7 is not the population SD of seven daily log returns")
    calendar = pd.date_range("2023-01-01", ADMISSIONS[admitted_end]["calendar_end"], tz="UTC")
    if not calendar[:len(market_dates)].equals(market_dates):
        raise ValueError("Admission cannot extend beyond the selected output calendar")
    x = np.full((len(calendar), 5), np.nan)
    y = np.full(len(calendar), np.nan)
    quality = np.zeros(len(calendar), dtype=bool)
    x[:len(features)], y[:len(targets)], quality[:len(features)] = features, targets, complete
    panel = DatedPanel(calendar, x, y, quality)
    receipt = {
        "version": VERSION, "status": "registered_development_prefix_loaded",
        "admitted_end": admitted_end, "purpose": ADMISSIONS[admitted_end]["purpose"],
        "target_definition": "population sigma7 of daily log returns, ddof0, h1, unannualized",
        "target_is_intraday_realized_variance": False,
        "source_protocol": PROTOCOL_PATH, "source_protocol_sha256": protocol_hash,
        "source_sha256": hashes, "raw_sources_hashed_in_full_as_bytes": True,
        "source_values_parsed_only_through_admission": True,
        "parsing": parsing, "decoded_calendar_rows": len(market_dates),
        "output_calendar_start": str(calendar[0].date()),
        "output_calendar_end": str(calendar[-1].date()),
        "placeholder_rows": len(calendar)-len(market_dates),
        "placeholder_dates": calendar[len(market_dates):].strftime("%Y-%m-%d").tolist(),
        "incomplete_admitted_bar_dates": incomplete_dates,
        "invalid_admitted_sigma7_dates": market_dates[~np.isfinite(targets)].strftime("%Y-%m-%d").tolist(),
        "population_target_reconstruction_matches": True,
        "support_counts": {column: int(panel.support[column].sum()) for column in panel.support},
        "market_values_exported": False, "raw_archives_opened": 0,
        "external_2025_2026_files_opened": 0,
        "limits": [
            "Only the admitted numerical prefix is decoded; full file bytes are hashed and all date fields checked.",
            "Quality derives from registered daily derivatives; minute archives are not reaggregated.",
            "Development has been used historically; this does not create an untouched confirmatory sample.",
            "Power-pilot outer development labels must not feed selection or refits; use dated callbacks/replay.",
        ],
    }
    return panel, receipt

