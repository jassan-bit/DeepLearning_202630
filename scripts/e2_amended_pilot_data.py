"""Stage-limited numerical admissions for the amended E2 development pilots.

The immutable historical source hashes and prefix parser are reused. All source
bytes are hashed and all calendar fields are checked, but cells after the
selected stage's cutoff are never converted to numbers. Unavailable later rows
remain NaN/quality-false placeholders on the uncompressed daily calendar.

The runner owns phase closure: in particular, it must not request power_score
until the October HPO winners, fixed epoch counts and all power-pilot weights
are frozen. This loader is an admission boundary, not a sandbox or a claim that
the already-known development period has become an independent test.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import e2_development_data as historical
    from .e2_temporal_training import DatedPanel
except ImportError:
    import e2_development_data as historical
    from e2_temporal_training import DatedPanel

VERSION = "e2-amended-pilot-data-v1"
STAGES = {
    "noise": {"numerical_end": "2023-09-23", "calendar_end": "2023-10-01",
              "purpose": "TiDE noise/resource pilot before the first 2024 HPO anchor"},
    "hpo": {"numerical_end": "2024-09-23", "calendar_end": "2024-10-01",
            "purpose": "October 2024 HPO for the three registered power representatives"},
    "power_fit": {"numerical_end": "2024-10-31", "calendar_end": "2024-12-30",
                  "purpose": "Fresh fixed-epoch pilot weights; no November/December numerical cells"},
    "power_score": {"numerical_end": "2024-12-30", "calendar_end": "2024-12-30",
                    "purpose": "Causal scoring only after all pilot weights and recipes close"},
}


def _source_path(root, relative):
    """Keep the registered cache paths regular and inside the chosen root."""
    path = root / relative
    if (not path.resolve().is_relative_to(root.resolve())
            or any(part.is_symlink() for part in (path, *path.parents)
                   if part.is_relative_to(root))):
        raise ValueError("Registered pilot sources cannot escape the cache or use symlinks")
    return path


def load_pilot_panel(root, stage):
    """Return a sealed DatedPanel and admission receipt for one explicit stage.

    For power_fit, prepare_b(panel, '2024-11-08', '2024-12-30') is valid:
    refit_train ends at October 31 while every outer-pilot row is unavailable.
    The unused inner split must not be used to retune the frozen October recipe.
    Later power_score admission changes outer support, so its whole PreparedB
    signature differs; its refit_train tensor/scaler signature remains equal.
    """
    if type(stage) is not str or stage not in STAGES:
        raise ValueError("Choose an explicit noise, hpo, power_fit or power_score admission")
    config = deepcopy(STAGES[stage])
    root = Path(root).absolute()
    if root.is_symlink():
        raise ValueError("The pilot cache root must not be a symlink")
    cutoff = pd.Timestamp(config["numerical_end"], tz="UTC")
    _source_path(root, historical.PROTOCOL_PATH)
    protocol_raw, protocol_hash = historical._read_registered(
        root, historical.PROTOCOL_PATH, historical.EXPECTED_PROTOCOL_SHA256)
    source_protocol = json.loads(protocol_raw)
    source_map = source_protocol.get("sources", {})
    if any(source_map.get(path) != digest for path, digest in historical.EXPECTED_SOURCES.items()):
        raise ValueError("The historical protocol must identify the exact pinned pilot derivatives")

    parsed, parsing, hashes, calendars = {}, {}, {}, {}
    for relative, expected in historical.EXPECTED_SOURCES.items():
        _source_path(root, relative)
        raw, hashes[relative] = historical._read_registered(root, relative, expected)
        calendars[relative], parsed[relative], parsing[relative] = historical._prefix_values(
            raw, relative, cutoff)
    market_dates = calendars[historical.MARKET_PATH]
    if not market_dates.equals(calendars[historical.TARGET_PATH]):
        raise ValueError("Market and target pilot admission calendars differ")
    features = parsed[historical.MARKET_PATH]
    targets = parsed[historical.TARGET_PATH][:, 0]
    complete = np.isfinite(features).all(axis=1)
    incomplete_dates = market_dates[~complete].strftime("%Y-%m-%d").tolist()
    if incomplete_dates != ["2023-03-24"]:
        raise ValueError("The pinned incomplete-day mask changed inside the pilot admission")
    observed = features[complete]
    if (np.any(observed[:, :4] <= 0) or np.any(observed[:, 4] < 0)
            or np.any(observed[:, 1] < np.maximum(observed[:, 0], observed[:, 3]))
            or np.any(observed[:, 2] > np.minimum(observed[:, 0], observed[:, 3]))
            or np.any(observed[:, 1] < observed[:, 2])):
        raise ValueError("Invalid admitted pilot OHLCV geometry")
    returns = pd.Series(np.log(features[:, 3]), index=market_dates).diff()
    reconstructed = returns.rolling(7, min_periods=7).std(ddof=0).to_numpy()
    if (not np.array_equal(np.isfinite(targets), np.isfinite(reconstructed))
            or not np.allclose(targets, reconstructed, rtol=1e-9, atol=1e-13, equal_nan=True)):
        raise ValueError("Pilot sigma7 must retain the full-calendar population SD target")

    calendar = pd.date_range("2023-01-01", config["calendar_end"], tz="UTC")
    if not calendar[:len(market_dates)].equals(market_dates):
        raise ValueError("Pilot calendar cannot truncate the admitted numerical prefix")
    x = np.full((len(calendar), 5), np.nan)
    y = np.full(len(calendar), np.nan)
    quality = np.zeros(len(calendar), dtype=bool)
    x[:len(features)], y[:len(targets)], quality[:len(features)] = features, targets, complete
    panel = DatedPanel(calendar, x, y, quality)
    return panel, {
        "version": VERSION, "status": "pinned_stage_prefix_loaded", "stage": stage,
        "purpose": config["purpose"], "numerical_end": config["numerical_end"],
        "source_protocol": historical.PROTOCOL_PATH, "source_protocol_sha256": protocol_hash,
        "source_sha256": hashes, "source_bytes_hashed_in_full": True,
        "source_date_fields_inspected_in_full": True,
        "source_numerical_values_after_stage_cutoff_decoded": False,
        "parsing": parsing, "decoded_calendar_rows": len(market_dates),
        "output_calendar_start": str(calendar[0].date()),
        "output_calendar_end": str(calendar[-1].date()),
        "placeholder_rows": len(calendar)-len(market_dates),
        "placeholder_dates": calendar[len(market_dates):].strftime("%Y-%m-%d").tolist(),
        "incomplete_admitted_bar_dates": incomplete_dates,
        "invalid_admitted_sigma7_dates": market_dates[~np.isfinite(targets)].strftime("%Y-%m-%d").tolist(),
        "population_target_reconstruction_matches": True,
        "target_definition": "sigma7 of daily Close log returns, ddof0, h1, unannualized",
        "support_counts": {column: int(panel.support[column].sum()) for column in panel.support},
        "fitting_or_scoring_executed_by_loader": False,
        "power_score_requires_prior_runner_freeze": stage == "power_score",
        "market_values_exported": False, "raw_archives_opened": 0,
        "external_2025_2026_values_decoded": 0,
        "limits": [
            "The runner must freeze October winners and pilot weights before requesting power_score.",
            "PreparedB outer support changes when scoring is admitted; compare the refit_train signature, not the whole prepared signature.",
            "Full registered source bytes are hashed, but excluded numerical cells are not decoded.",
            "These are previously known development dates, not a new independent test.",
        ],
    }
