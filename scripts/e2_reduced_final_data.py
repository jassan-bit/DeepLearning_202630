"""Post-freeze admission of the known E1 calendar for reduced E2.

Reservation reads committed metadata only. December31 is an explicitly admitted
causal bridge. The 608 known E1 dates are decoded only after all sixteen systems,
six neural seeds, scalers, calibrators and analysis have frozen. This is not a
new blind/prospective sample. No fit, selection or imputation is implemented.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import urlsplit
from urllib.request import urlopen

PROTOCOL_ID = "e2-reduced-delivery-v1"
PROTOCOL_SHA256 = "867cf6837158c1353a1aa86ad9c51fc6fc260948ffe02dce8e779d3c329fb1cc"
NEURAL = ("FCN-1D", "ResNet-1D", "Inception-1D", "LSTM-autoencoder con predictor",
          "TimesNet", "TSMixer", "TiDE", "TimeMixer", "PatchTST", "iTransformer",
          "MLP16", "LSTM32", "GRU32")
CLASSICAL = ("Persistence", "SeasonalNaive", "ARIMA")
SYSTEMS = NEURAL + CLASSICAL
SEEDS = tuple(range(271801, 271807))
RESERVE = "data/manifests/test_raw_reserve_20250101_20260831.json"
RESERVE_SHA256 = "d2eb4e84fff85ee957e6b02890cead3cec37fd593ec2a8db487dbe137d2d3a10"
E1_AUDIT = "results/filipe_multihistory_20261104/test/raw_test_audit.json"
E1_AUDIT_SHA256 = "4f2ea5591ed83970ec1644aa9d8410fbf3817b05bc23721e13ad8e03b827c29c"
E1_DAILY = "results/filipe_multihistory_20261104/test/daily_market_test.csv"
E1_DAILY_SHA256 = "c81215f88dfc7fe9b8b579596c958f27da5bedb7580bb4a935808b23a414baf1"
FEATURES = ("open", "high", "low", "close", "volume")
HASH = re.compile(r"[0-9a-f]{64}\Z")


def _digest(raw):
    return sha256(raw).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def _path(root, relative):
    part = PurePosixPath(relative)
    if part.is_absolute() or ".." in part.parts or str(part) != relative or not relative:
        raise ValueError("An artifact/source path must be normalized and relative")
    root = Path(root).absolute()
    path = root.joinpath(*part.parts)
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("Symlinked data/artifact paths are not admitted")
    return path


def _read(root, relative, expected):
    path = _path(root, relative)
    if not path.is_file():
        raise ValueError("Missing registered source/artifact: " + relative)
    raw = path.read_bytes()
    if _digest(raw) != expected:
        raise ValueError("Source/artifact bytes changed: " + relative)
    return raw


def reserve_known_e1_evaluation(root):
    """Bind existing acquisition/E1 audit metadata, opening zero market payloads."""
    reserve = json.loads(_read(root, RESERVE, RESERVE_SHA256))
    audit = json.loads(_read(root, E1_AUDIT, E1_AUDIT_SHA256))
    if (reserve.get("test_start_day_utc"), reserve.get("test_end_day_utc_inclusive"),
        reserve.get("calendar_days"), audit.get("daily_market_sha256")) != (
            "2025-01-01", "2026-08-31", 608, E1_DAILY_SHA256):
        raise ValueError("The inherited known E1 calendar/source identity changed")
    records = reserve.get("archives", [])
    months = [f"2025-{m:02d}" for m in range(1, 13)] + [f"2026-{m:02d}" for m in range(1, 9)]
    if len(records) != 20 or [r.get("archive") for r in records] != [f"BTCUSDT-1m-{m}.zip" for m in months]:
        raise ValueError("Only the twenty known January2025--August2026 archives are admitted")
    for row in records:
        if (not HASH.fullmatch(row.get("sha256", "")) or type(row.get("bytes")) is not int
                or row["bytes"] < 1 or row.get("relative_path") != "BTCUSDT/1m/monthly/" + row["archive"]):
            raise ValueError("An archive reservation has invalid identity or path")
        parsed = urlsplit(row.get("url", ""))
        if parsed.scheme != "https" or parsed.hostname != "data.binance.vision":
            raise ValueError("Archive URL is outside the registered public acquisition source")
    result = {"version": "e2-reduced-known-E1-admission-v1", "protocol_id": PROTOCOL_ID,
        "start": "2025-01-01", "end": "2026-08-31", "calendar_days": 608,
        "reserve_manifest": RESERVE, "reserve_manifest_sha256": RESERVE_SHA256,
        "E1_audit": E1_AUDIT, "E1_audit_sha256": E1_AUDIT_SHA256,
        "known_daily_path": E1_DAILY, "known_daily_sha256": E1_DAILY_SHA256,
        "archives": records, "metadata_only": True, "market_payloads_opened": 0,
        "known_historical_E1_period": True, "blindness_recovered": False,
        "new_independent_prospective_confirmation": False}
    result["sha256"] = _digest(_canonical(result))
    return result


def validate_final_freeze(manifest, artifact_root):
    """Verify 13×6+3 closed predictors and their frozen physical artifacts."""
    if not isinstance(manifest, dict) or (manifest.get("protocol_id"), manifest.get("protocol_sha256")) != (PROTOCOL_ID, PROTOCOL_SHA256):
        raise ValueError("The reduced protocol identity is required before final data access")
    if tuple(manifest.get("systems", ())) != SYSTEMS or tuple(manifest.get("final_seed_ids", ())) != SEEDS:
        raise ValueError("All sixteen systems and all six final seeds must freeze")
    if manifest.get("pilot_power_MDE_closed") is not True or manifest.get("analysis_frozen") is not True:
        raise ValueError("Close the development pilot/power/MDE and analysis before final data access")
    files = manifest.get("frozen_files_sha256")
    if not isinstance(files, dict) or not files or any(not isinstance(k, str) or not isinstance(v, str) or not HASH.fullmatch(v) for k, v in files.items()):
        raise ValueError("Frozen physical artifacts and SHA256 identities are required")
    expected = {(name, seed) for name in NEURAL for seed in SEEDS} | {(name, "deterministic") for name in CLASSICAL}
    rows = manifest.get("models")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("Exactly 81 final predictor records must be closed")
    actual = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid predictor freeze record")
        key = (row.get("model"), row.get("seed"))
        if key not in expected:
            raise ValueError("Unexpected final model/seed identity")
        actual.append(key)
        checkpoint = row.get("checkpoint")
        if not isinstance(checkpoint, str) or files.get(checkpoint) != row.get("checkpoint_sha256"):
            raise ValueError("Each final checkpoint must be part of the physical artifact freeze")
        for field in ("scaler_sha256", "calibrator_sha256"):
            if row.get(field) not in files.values():
                raise ValueError("Each scaler and calibrator must be a physically frozen artifact")
    if set(actual) != expected or len(set(actual)) != len(actual):
        raise ValueError("A final predictor is missing or duplicated")
    for relative, checksum in files.items():
        _read(artifact_root, relative, checksum)
    return {"freeze_sha256": _digest(_canonical(manifest)), "systems": 16,
            "neural_seed_predictors": 78, "classical_predictors": 3,
            "physical_artifacts_verified": len(files), "all_weights_scalers_calibrators_frozen": True}


def _panel_from_frames(development, evaluation=None):
    """Pure calendar-preserving constructor, including the causal December31 bar."""
    import numpy as np
    import pandas as pd
    try:
        from .e2_temporal_training import DatedPanel
    except ImportError:
        from e2_temporal_training import DatedPanel
    expected_dev = pd.date_range("2023-01-01", "2024-12-31", tz="UTC")
    if not development.index.equals(expected_dev):
        raise ValueError("Development/bridge must retain every UTC date through December31")
    frames = [development.loc[:, list(FEATURES)]]
    if evaluation is not None:
        expected = pd.date_range("2025-01-01", "2026-08-31", tz="UTC")
        if not evaluation.index.equals(expected):
            raise ValueError("Evaluation must retain all 608 dates, without compression or imputation")
        frames.append(evaluation.loc[:, list(FEATURES)])
    full = pd.concat(frames)
    values = full.to_numpy(dtype=np.float64)
    if np.isinf(values).any():
        raise ValueError("Infinite market values are invalid")
    complete = np.isfinite(values).all(axis=1)
    observed = values[complete]
    if (np.any(observed[:, :4] <= 0) or np.any(observed[:, 4] < 0)
            or np.any(observed[:, 1] < np.maximum(observed[:, 0], observed[:, 3]))
            or np.any(observed[:, 2] > np.minimum(observed[:, 0], observed[:, 3]))
            or np.any(observed[:, 1] < observed[:, 2])):
        raise ValueError("Invalid positive-price OHLCV geometry")
    targets = pd.Series(np.log(values[:, 3]), index=full.index).diff().rolling(7, min_periods=7).std(ddof=0).to_numpy()
    return DatedPanel(full.index, values, targets, complete)


def load_development_bridge(root):
    """Admit the unchanged 2023--24 derivatives through December31 explicitly.

    The final label/feature row is a causal bridge only. The caller must keep
    training<=October31 and calibration<=December30; this panel is not a sandbox.
    No 2025+ path is touched by this call.
    """
    import numpy as np
    import pandas as pd
    try:
        from . import e2_development_data as old
    except ImportError:
        import e2_development_data as old
    raw, phash = old._read_registered(Path(root), old.PROTOCOL_PATH, old.EXPECTED_PROTOCOL_SHA256)
    protocol = json.loads(raw)
    if any(protocol.get("sources", {}).get(p) != h for p, h in old.EXPECTED_SOURCES.items()):
        raise ValueError("The inherited development source identities changed")
    parsed, hashes = {}, {}
    dates = None
    for relative, expected in old.EXPECTED_SOURCES.items():
        raw, hashes[relative] = old._read_registered(Path(root), relative, expected)
        current_dates, parsed[relative], _ = old._prefix_values(raw, relative, pd.Timestamp("2024-12-31", tz="UTC"))
        if dates is not None and not dates.equals(current_dates):
            raise ValueError("Development bridge market/target calendars differ")
        dates = current_dates
    development = pd.DataFrame(parsed[old.MARKET_PATH], index=dates, columns=FEATURES)
    panel = _panel_from_frames(development)
    targets = parsed[old.TARGET_PATH][:, 0]
    if not np.allclose(targets, panel.targets, equal_nan=True, rtol=1e-9, atol=1e-13):
        raise ValueError("Registered sigma7 labels differ from population seven-log-return reconstruction")
    bad = development.index[~panel.quality].strftime("%Y-%m-%d").tolist()
    if bad != ["2023-03-24"]:
        raise ValueError("The inherited incomplete development-day mask changed")
    return panel, {"version": "e2-reduced-development-bridge-v1", "admitted_end": "2024-12-31",
        "source_protocol_sha256": phash, "source_sha256": hashes, "source_bytes_unchanged": True,
        "training_label_end": "2024-10-31", "calibration_label_end": "2024-12-30",
        "bridge_dates": ["2024-12-31"], "bridge_role": "causal observations only; no fit or recalibration",
        "population_sigma7_ddof": 0, "external_2025_2026_files_opened": 0,
        "new_training_or_HPO_executed": False}


def _write_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_canonical(value) + b"\n")


def _download_archive(record, target, opener):
    if target.exists():
        if not target.is_file() or target.stat().st_size != record["bytes"]:
            raise ValueError("An existing final archive has an unexpected size; it was retained")
        return target.read_bytes()
    with opener(record["url"], timeout=60) as response:
        parsed = urlsplit(response.geturl())
        if parsed.scheme != "https" or parsed.hostname != "data.binance.vision":
            raise ValueError("Final archive redirected outside its pinned source")
        raw = response.read(record["bytes"] + 1)
    if len(raw) != record["bytes"] or _digest(raw) != record["sha256"]:
        raise ValueError("Downloaded final archive differs from the pinned acquisition bytes")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(raw)
    return raw


def load_known_e1_evaluation(root, reservation, freeze_manifest, *, artifact_root,
                             daily_csv=None, raw_root=None, allow_download=False,
                             output_dir=None, opener=None, development_root=None):
    """Decode the known inherited cohort after verifying the complete freeze.

    A matching cached E1 daily derivative is preferred. Otherwise the twenty
    pinned archives can be aggregated from raw_root. Downloads are opt-in and
    occur only after the freeze; neither September nor October2026 is admitted.
    A partially started decode is preserved and blocks an automatic retry.
    """
    import numpy as np
    import pandas as pd
    # All checks below precede any market payload access, including development.
    freeze = validate_final_freeze(freeze_manifest, artifact_root)
    current = reserve_known_e1_evaluation(root)
    if reservation != current:
        raise ValueError("Evaluation metadata changed after its reservation")
    if type(allow_download) is not bool:
        raise ValueError("Explicit boolean archive-download permission is required")
    # Missing inputs are a metadata/preparation issue, not a first payload read.
    source = Path(daily_csv) if daily_csv is not None else _path(root, E1_DAILY)
    closed_cache_available = (output_dir is not None
        and Path(output_dir, "known_E1_data_receipt.json").is_file()
        and Path(output_dir, "known_E1_daily.csv").is_file())
    if not source.is_file() and not closed_cache_available and raw_root is None:
        raise ValueError("Known E1 daily derivative is absent; supply a pinned raw archive root")
    if not source.is_file() and not closed_cache_available and not allow_download:
        for record in reservation["archives"]:
            path = _path(raw_root, record["relative_path"])
            if not path.is_file() or path.stat().st_size != record["bytes"]:
                raise ValueError("A reserved raw archive is missing; downloading was not enabled")
    output = None if output_dir is None else Path(output_dir)
    marker = None if output is None else output / "current_E2_first_known_E1_access.json"
    receipt_path = None if output is None else output / "known_E1_data_receipt.json"
    cached_market = None if output is None else output / "known_E1_daily.csv"
    cached = False
    if output is not None:
        if any(p.is_symlink() for p in (output.absolute(), *output.absolute().parents)):
            raise ValueError("The private data output must not resolve through symlinks")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_bytes())
            if (not marker.is_file() or not cached_market.is_file()
                    or receipt.get("freeze_sha256") != freeze["freeze_sha256"]
                    or receipt.get("reservation_sha256") != reservation["sha256"]
                    or _digest(cached_market.read_bytes()) != receipt.get("daily_market_sha256")):
                raise ValueError("Cached final derivative/freeze changed")
            cached = True
        elif marker.exists() or cached_market.exists():
            raise ValueError("A partial final decode needs reconciliation; no overwrite or silent retry")
        else:
            _write_exclusive(marker, {"observed_at_utc": datetime.now(timezone.utc).isoformat(),
                **freeze, "reservation_sha256": reservation["sha256"],
                "known_historical_E1_period": True, "new_unseen_confirmation": False})
    quality = []
    if cached:
        raw = cached_market.read_bytes()
        from io import BytesIO
        evaluation = pd.read_csv(BytesIO(raw), index_col="market_day_utc")
    elif source.is_file():
        if any(p.is_symlink() for p in (source.absolute(), *source.absolute().parents)):
            raise ValueError("Cached E1 daily source must not be symlinked")
        raw = source.read_bytes()
        if _digest(raw) != E1_DAILY_SHA256:
            raise ValueError("Cached E1 daily source is not the hash-pinned inherited derivative")
        from io import BytesIO
        evaluation = pd.read_csv(BytesIO(raw), index_col="market_day_utc")
    else:
        if raw_root is None:
            raise ValueError("Known E1 daily derivative is absent; supply a pinned raw archive root")
        try:
            from .load_filipe_reserved_test import aggregate_month_payload
        except ImportError:
            from load_filipe_reserved_test import aggregate_month_payload
        frames = []
        for record in reservation["archives"]:
            path = _path(raw_root, record["relative_path"])
            if allow_download:
                raw = _download_archive(record, path, urlopen if opener is None else opener)
            else:
                if not path.is_file() or path.stat().st_size != record["bytes"]:
                    raise ValueError("A reserved raw archive is missing; downloading was not enabled")
                raw = path.read_bytes()
            month = record["archive"].removeprefix("BTCUSDT-1m-").removesuffix(".zip")
            part, check = aggregate_month_payload(raw, month, expected_sha256=record["sha256"])
            frames.append(part); quality.append(check)
        evaluation = pd.concat(frames)
    evaluation.index = pd.DatetimeIndex(pd.to_datetime(evaluation.index, utc=True))
    development, dev_receipt = load_development_bridge(root if development_root is None else development_root)
    dev_frame = pd.DataFrame(development.features, index=development.calendar, columns=FEATURES)
    panel = _panel_from_frames(dev_frame, evaluation)
    evaluation_calendar = pd.date_range("2025-01-01", "2026-08-31", tz="UTC")
    support = panel.support.loc[evaluation_calendar]
    eligible = support.index[support["common"]].strftime("%Y-%m-%d").tolist()
    if not eligible:
        raise ValueError("The frozen common evaluation support is empty")
    receipt = {"version": "e2-reduced-known-E1-data-v1", **freeze,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "reservation_sha256": reservation["sha256"], "known_historical_E1_period": True,
        "previous_E1_exposure_disclosed": True, "blindness_recovered": False,
        "new_independent_prospective_confirmation": False, "calendar_days": 608,
        "evaluation_start": "2025-01-01", "evaluation_end": "2026-08-31",
        "common_eligible_dates": eligible, "common_eligible_count": len(eligible),
        "support_sha256": _digest(_canonical({"dates": [str(d.date()) for d in support.index],
            "columns": list(support.columns), "mask": support.to_numpy(dtype=bool).tolist()})),
        "quality_by_archive": quality, "development_bridge": dev_receipt,
        "model_refits_or_retuning_after_access": False, "September_October2026_opened": False,
        "target": "population sigma7 of daily Close log returns, ddof0, unannualized; h1"}
    if output is not None and not cached:
        output.mkdir(parents=True, exist_ok=True)
        evaluation.index.name = "market_day_utc"
        evaluation.to_csv(cached_market, mode="x")
        receipt["daily_market_sha256"] = _digest(cached_market.read_bytes())
        _write_exclusive(receipt_path, receipt)
    elif cached:
        # The original receipt also retains the original first-access evidence.
        previous = json.loads(receipt_path.read_bytes())
        if previous["support_sha256"] != receipt["support_sha256"]:
            raise ValueError("The frozen inherited cohort support changed")
        receipt = previous
    return panel, receipt
