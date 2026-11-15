"""Post-benchmark audit of the actual E2 655/53/608 data, without model fits.

Download-only work verifies the twenty original public archives. Numerical
aggregation/audit must run under e2_corrections_budget.py. No historical CLI,
model optimizer, checkpoint deserializer or prediction function is invoked.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from urllib.request import urlopen

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
VERSION = "e2-current-data-closure-v1"
ENGINE = "1e8745497b484518f766b6da5c19be8362d9add7"
PUBLISHED_EVIDENCE = "0bdbc42bd724c6ffbbb1dec67564327adedb9883"
FEATURES = ["open", "high", "low", "close", "volume"]
NEAR_THRESHOLD = 0.01
DAILY_SHA = "9563ef557e180f34ee37dce3b1480c38aaefd7063df21c031644973310e28deb"
E1_DAILY_SHA = "c81215f88dfc7fe9b8b579596c958f27da5bedb7580bb4a935808b23a414baf1"
PROTOCOL_SHA = "867cf6837158c1353a1aa86ad9c51fc6fc260948ffe02dce8e779d3c329fb1cc"
DEV_FILES = {
    "results/eda_development/daily_market_development.csv": "bf6bb5160af924aef6a15c3684aeae932df3017e8639888bb7e91ce787900171",
    "results/eda_development/daily_volatility_development.csv": "20b96374a659eb87dadfa68aa3d6b6b5fb6bea5b10eebe6f2c186ad69861844a",
}
CODE_FILES = ["e2_development_data.py", "e2_reduced_final_data.py", "e2_reduced_final.py",
              "e2_temporal_training.py", "e2_real_executor.py", "audit_e2_common_availability.py",
              "load_filipe_reserved_test.py"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    a = np.asarray(value)
    require(not a.dtype.hasobject, "Object arrays are not portable audit inputs")
    a = np.ascontiguousarray(a.astype(a.dtype.newbyteorder("<"), copy=False))
    header = json.dumps({"shape": a.shape, "dtype": a.dtype.str}, sort_keys=True).encode()
    return hashlib.sha256(header + b"\0" + a.tobytes()).hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def csv_write(path, rows):
    rows = list(rows)
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def date_ids(index):
    return pd.DatetimeIndex(index).strftime("%Y-%m-%d").tolist()


def relative_name(path, root=ROOT):
    path = Path(path).resolve()
    try:
        return path.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return "external_input/" + path.name


def validate_daily_frame(frame, first, last, allowed_missing=()):
    """Reject compressed/duplicate calendars, partial bars and bad OHLCV."""
    require(isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is not None,
            "Daily dates must explicitly identify UTC")
    require(str(frame.index.tz) == "UTC", "Daily timezone must be UTC")
    expected = pd.date_range(first, last, tz="UTC")
    require(frame.index.equals(expected), "Daily calendar must be complete, unique, ordered UTC midnights")
    require(all(c in frame.columns for c in FEATURES), "Missing OHLCV columns")
    values = frame[FEATURES].to_numpy(dtype=np.float64)
    require(not np.isinf(values).any(), "Infinite OHLCV value")
    finite = np.isfinite(values)
    require(np.all(finite.all(axis=1) | (~finite).all(axis=1)), "A daily OHLCV row is partially missing")
    missing = date_ids(frame.index[~finite.all(axis=1)])
    require(missing == sorted(allowed_missing), "Unexpected missing daily bars")
    good = values[finite.all(axis=1)]
    require(np.all(good[:, :4] > 0) and np.all(good[:, 4] >= 0)
            and np.all(good[:, 1] >= np.maximum(good[:, 0], good[:, 3]))
            and np.all(good[:, 2] <= np.minimum(good[:, 0], good[:, 3]))
            and np.all(good[:, 1] >= good[:, 2]), "Invalid OHLCV geometry")
    return frame


def _support(ids, first_offset, last_offset):
    return {(date.fromisoformat(d) + timedelta(days=k)).isoformat()
            for d in ids for k in range(first_offset, last_offset + 1)}


def support_intersections(left_ids, right_ids):
    return {
        "outcome_ID_intersection": sorted(set(left_ids) & set(right_ids)),
        "target_close_support_intersection": sorted(_support(left_ids, -7, 0) & _support(right_ids, -7, 0)),
        "D7_input_support_intersection": sorted(_support(left_ids, -7, -1) & _support(right_ids, -7, -1)),
        "target_return_support_intersection": sorted(_support(left_ids, -6, 0) & _support(right_ids, -6, 0)),
    }


def nearest_rows(query, reference, query_dates, reference_dates, scope, within=False):
    """Diagnostic RMS distance on35 train-scaled relative coordinates."""
    q, r = np.asarray(query, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    require(len(q) == len(query_dates) and len(r) == len(reference_dates), "Neighbor IDs and arrays differ")
    q, r = q.reshape(len(q), -1), r.reshape(len(r), -1)
    require(q.shape[1] == r.shape[1] == 35 and np.isfinite(q).all() and np.isfinite(r).all(),
            "Finite D7x5 arrays required for neighbors")
    distances = cdist(q, r, metric="euclidean") / np.sqrt(35)
    if within:
        for i, d in enumerate(query_dates):
            distances[i, np.asarray(reference_dates) == d] = np.inf
    require(np.isfinite(distances.min(axis=1)).all(), "No distinct reference window remains")
    nearest = distances.argmin(axis=1)
    result = []
    for i, j in enumerate(nearest):
        day, other = str(query_dates[i]), str(reference_dates[j])
        shared = support_intersections([day], [other])
        result.append({"scope": scope, "query_outcome_day_utc": day,
                       "nearest_reference_outcome_day_utc": other,
                       "rms_standardized_distance": float(distances[i, j]),
                       "below_fixed_threshold": bool(distances[i, j] <= NEAR_THRESHOLD),
                       "same_outcome_ID": day == other,
                       "input_calendar_days_shared_with_nearest": len(shared["D7_input_support_intersection"]),
                       "target_closes_shared_with_nearest": len(shared["target_close_support_intersection"]),
                       "interpretation": "pattern similarity, not automatic leakage or independence"})
    return result


def compare_scaler_records(expected, actual):
    """Bind scales to exact training IDs/signature, with explicit numeric tolerance."""
    identity = (expected.get("train_dates") == actual.get("train_dates")
                and expected.get("training_sha256") == actual.get("training_sha256"))
    try:
        a = np.asarray([*expected["mean"], *expected["std"], expected["target_scale"]], dtype=float)
        b = np.asarray([*actual["mean"], *actual["std"], actual["target_scale"]], dtype=float)
        require(a.shape == b.shape == (11,) and np.isfinite(a).all() and np.isfinite(b).all(),
                "Invalid numerical scaler record")
        difference = float(np.max(np.abs(a - b)))
    except (KeyError, ValueError, TypeError):
        return {"matches": False, "training_identity_matches": identity,
                "maximum_numeric_absolute_difference": None, "absolute_tolerance": 1e-12}
    return {"matches": bool(identity and difference <= 1e-12), "training_identity_matches": identity,
            "maximum_numeric_absolute_difference": difference, "absolute_tolerance": 1e-12}


def initialize_specification(output):
    path = output / "audit_specification.json"
    body = {"version": VERSION, "design_role": "post-benchmark corrective data verification",
            "prior_results_known": True, "historical_preregistration_claimed": False,
            "cohorts": {"training": ["2023-01-01", "2024-10-31", 655],
                        "calibration": ["2024-11-08", "2024-12-30", 53],
                        "evaluation": ["2025-01-01", "2026-08-31", 608]},
            "contract": {"D": 7, "features": FEATURES, "target_return_count": 7,
                         "target_close_count": 8, "ddof": 0, "horizon_days": 1,
                         "annualized": False, "future_or_current_outcome_features_allowed": False},
            "criteria": {"target_rtol": 1e-9, "target_atol": 1e-13,
                         "scaler_numeric_absolute_tolerance": 1e-12,
                         "near_RMS_train_standardized_threshold": NEAR_THRESHOLD,
                         "near_threshold_source": "book_e2/_static/eda_decisions/nearest_window_summary.csv",
                         "near_alert_policy": "report and review; never remove rows or infer leakage automatically",
                         "calibration_test_shared_history": "six target closes and five D7 input bars are legitimate past history"},
            "source_policy": "Exact pinned development derivatives and original20 hash-pinned evaluation archives/cache; no future data, fitted models or new prediction calls.",
            "status_definitions": {"PASS": "criterion verified for the declared actual data/code scope",
                                   "PARTIAL": "missing evidence or review needed, not a demonstrated failure",
                                   "FAIL": "an explicit data/identity/causality criterion failed",
                                   "NA": "the condition does not apply to this continuous nonhuman target"},
            "permitted_operations": ["raw download/hash verification", "daily aggregation", "target reconstruction",
                                     "causal window/scaler reconstruction", "exact fingerprints", "fixed-threshold neighbors",
                                     "read-only JSON identity comparison", "actual-data metamorphic preprocessing checks"],
            "prohibited_operations": ["model training", "hyperparameter selection", "model inference",
                                      "performance rescoring", "bootstrap", "hypothesis tests", "changing frozen A2"]}
    output.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = json.loads(path.read_text())
        require({k: v for k, v in previous.items() if k != "created_at_utc"} == body,
                "Existing corrective audit specification differs; preserve it")
    else:
        json_write(path, {"created_at_utc": utc(), **body})
    return path


def download_raw(root, raw_root, output):
    """I/O-only recovery. Numerical aggregation is left to the budgeted audit."""
    import e2_reduced_final_data as loader
    reservation = loader.reserve_known_e1_evaluation(root)
    started = time.monotonic()
    def fetch(row):
        target = raw_root / row["relative_path"]
        had = target.exists()
        raw = loader._download_archive(row, target, urlopen)
        require(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"],
                "Original archive bytes differ: " + row["archive"])
        return {"archive": row["archive"], "url": row["url"], "sha256": row["sha256"],
                "bytes": len(raw), "path": relative_name(target, root), "existing_file_reused": had}
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(fetch, reservation["archives"]))
    record = {"version": VERSION, "completed_at_utc": utc(), "purpose": "restore historical current-evaluation input bytes",
              "elapsed_wall_seconds": time.monotonic() - started, "archive_count": len(rows),
              "total_compressed_bytes": sum(r["bytes"] for r in rows), "archives": rows,
              "archive_contents_decoded": False, "model_fits": 0, "model_inference_calls": 0,
              "new_first_access_or_blinding_claimed": False}
    json_write(output / "raw_recovery_receipt.json", record)
    return record


def recover_daily(root, raw_root, daily_path):
    import e2_reduced_final_data as loader
    if daily_path.is_file():
        require(sha_file(daily_path) in (DAILY_SHA, E1_DAILY_SHA), "Unrecognized existing daily evaluation cache")
        return None
    reservation = loader.reserve_known_e1_evaluation(root)
    if not all((raw_root / r["relative_path"]).is_file() for r in reservation["archives"]):
        return None
    from load_filipe_reserved_test import aggregate_month_payload
    require(pd.__version__ == "2.2.3", "Exact original daily serialization requires pandas2.2.3")
    frames, quality = [], []
    for row in reservation["archives"]:
        raw = (raw_root / row["relative_path"]).read_bytes()
        frame, check = aggregate_month_payload(raw, row["archive"][11:-4], expected_sha256=row["sha256"])
        frames.append(frame)
        quality.append(check)
    full = pd.concat(frames)
    full.index = pd.DatetimeIndex(pd.to_datetime(full.index, utc=True))
    validate_daily_frame(full, "2025-01-01", "2026-08-31")
    full.index.name = "market_day_utc"
    stream = io.StringIO(newline="")
    full.to_csv(stream)
    raw = stream.getvalue().encode()
    require(hashlib.sha256(raw).hexdigest() == DAILY_SHA, "Aggregated daily bytes differ from current original admission")
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    with daily_path.open("xb") as handle:
        handle.write(raw)
    return {"archive_count": 20, "csv_rows": sum(r["csv_rows"] for r in quality),
            "daily_csv_sha256": DAILY_SHA, "quality_by_archive": quality,
            "status": "original_current_daily_cache_exactly_reconstructed"}


def duplicate_rows(arrays, cohort_dates):
    output = []
    for scope, a, b in [(k + "_within", k, k) for k in arrays] + [
            ("calibration_to_training", "calibration", "training"),
            ("evaluation_to_training", "evaluation", "training"),
            ("evaluation_to_calibration", "evaluation", "calibration")]:
        if a not in arrays or b not in arrays:
            continue
        aq, ar = arrays[a], arrays[b]
        qhash, rhash = [array_sha(x) for x in aq], [array_sha(x) for x in ar]
        reference = {}
        for i, fingerprint in enumerate(rhash):
            reference.setdefault(fingerprint, []).append(i)
        pairs = [(i, j) for i, h in enumerate(qhash) for j in reference.get(h, [])
                 if cohort_dates[a][i] != cohort_dates[b][j]]
        # Confirm candidate hash matches by numerical equality, rather than
        # treating a hash alone as content/causality certification.
        require(all(np.array_equal(aq[i], ar[j]) for i, j in pairs), "A duplicate fingerprint collision requires review")
        output.append({"scope": scope, "query_windows": len(aq), "reference_windows": len(ar),
                       "exact_duplicate_query_count": len({i for i, _ in pairs}),
                       "exact_duplicate_pairs": len(pairs), "same_outcome_ID_excluded": True})
    return output


def audit(root, output, raw_root, daily_path, assets_index):
    """Compute only data-integrity diagnostics; callers account its wall time."""
    import torch
    import scipy
    from e2_development_data import load_development
    from e2_reduced_final import final_training_split, _calibration_split
    from e2_reduced_final_data import load_development_bridge, _panel_from_frames
    from e2_temporal_training import relative_window

    started = time.monotonic()
    controls, provenance = [], []
    def control(key, status, scope, observed, remaining="", needs_fit=False):
        controls.append({"id": key, "status": status, "scope": scope, "observed": observed,
                         "remaining": remaining, "new_model_fit_needed_for_remaining": needs_fit})
        if status == "FAIL":
            print("FAIL " + key + ": " + str(observed), flush=True)
    def source(path, expected=None, role="input"):
        path = Path(path)
        checksum = sha_file(path)
        require(expected is None or checksum == expected, "Source hash mismatch: " + relative_name(path, root))
        provenance.append({"path": relative_name(path, root), "bytes": path.stat().st_size,
                           "sha256": checksum, "role": role, "expected_sha256_matched": expected is not None})
        return checksum

    protocol_path = root / "experiments/e2_reduced_protocol_20261112/protocol.json"
    source(protocol_path, PROTOCOL_SHA)
    protocol = json.loads(protocol_path.read_text())
    freeze_path = root / "book_e2/_static/e2_benchmark_final_20261113/final_freeze.json"
    payload_path = root / "book_e2/_static/e2_final_audit_inputs_20261114/e2_f2a_input.json"
    partition_path = root / "book_e2/_static/e2_eda_current_v1/partition_audit.json"
    fit_path = root / "book_e2/_static/e2_f2a_audit_20261113/fit_overfit_summary.csv"
    admission_path = root / "book_e2/_static/e2_benchmark_final_20261113/data_admission.json"
    for p in (freeze_path, payload_path, partition_path, fit_path, admission_path):
        historical = subprocess.check_output(["git", "--no-optional-locks", "-C", str(root), "show",
                                              PUBLISHED_EVIDENCE + ":" + p.relative_to(root).as_posix()])
        source(p, hashlib.sha256(historical).hexdigest(), "published frozen evidence at " + PUBLISHED_EVIDENCE)
    freeze, payload, registered, admission = [json.loads(p.read_text()) for p in
                                             (freeze_path, payload_path, partition_path, admission_path)]
    require(freeze["protocol_sha256"] == PROTOCOL_SHA and len(freeze["models"]) == 81
            and len(freeze["systems"]) == 16 and freeze["training_label_end"] == "2024-10-31",
            "The current16-system/81-predictor freeze is required")
    require(len(payload["predictors"]) == 81
            and {(r["model"], str(r["seed"])) for r in payload["predictors"]}
            == {(r["model"], str(r["seed"])) for r in freeze["models"]},
            "Saved target/prediction metadata do not match the frozen81 predictor IDs")
    require(len(protocol["hpo"]["folds"]) == 3, "Current reduced protocol must retain3HPO folds")
    source_identity = []
    for filename in CODE_FILES:
        path = root / "scripts" / filename
        original = subprocess.check_output(["git", "--no-optional-locks", "-C", str(root), "show",
                                             ENGINE + ":scripts/" + filename])
        original_sha = hashlib.sha256(original).hexdigest()
        current_sha = source(path, original_sha, "unchanged scientific data/preprocessing function source")
        source_identity.append({"path": "scripts/" + filename, "sha256": current_sha,
                                "identical_to_scientific_commit": ENGINE})
    for name, expected in DEV_FILES.items():
        source(root / name, expected)
    control("D01", "PASS", "current protocol and declared source identities",
            {"systems": 16, "frozen_predictors": 81, "HPO_folds": 3,
             "unchanged_scientific_data_source_files": len(source_identity)})

    # This is the final worker's actual admitted development path. Its targets
    # are the original CSV values, while the scoring bridge reconstructs them.
    # Keeping those paths distinct avoids inventing bitwise-equal target hashes.
    development, _ = load_development(root, "2024-12-30")
    bridge, _ = load_development_bridge(root)
    dev_frame = pd.DataFrame(bridge.features, index=bridge.calendar, columns=FEATURES)
    validate_daily_frame(dev_frame, "2023-01-01", "2024-12-31", ["2023-03-24"])
    train = final_training_split(development)
    calib = _calibration_split(development, train.scaler)
    train_ids, cal_ids = date_ids(train.dates), date_ids(calib.dates)
    expected_train = registered["final_partition"]["training"]["training_sample_IDs"]
    expected_cal = registered["final_partition"]["calibration"]["sample_IDs"]
    require(train_ids == expected_train and len(train_ids) == 655, "Actual train IDs differ")
    require(cal_ids == expected_cal and len(cal_ids) == 53, "Actual calibration IDs differ")
    independent_targets = []
    closes = bridge.features[:, 3]
    for i in range(len(closes)):
        window = closes[i - 7:i + 1] if i >= 7 else np.array([])
        independent_targets.append(float(np.std(np.diff(np.log(window)), ddof=0))
                                   if len(window) == 8 and np.isfinite(window).all() else np.nan)
    independent_targets = np.asarray(independent_targets)
    require(np.array_equal(np.isfinite(independent_targets), np.isfinite(bridge.targets)), "Development target mask differs")
    np.testing.assert_allclose(independent_targets, bridge.targets, rtol=1e-9, atol=1e-13, equal_nan=True)
    cal_truth = np.asarray(calib.y_original)
    for predictor in payload["predictors"]:
        require(predictor["calibration_dates"] == cal_ids, "Saved calibration IDs differ")
        np.testing.assert_allclose(cal_truth, predictor["calibration_targets"], rtol=1e-9, atol=1e-13)
    control("D02", "PASS", "actual development CSV calendar and OHLCV",
            {"calendar_rows": 731, "complete_daily_bars": 730, "incomplete_dates": ["2023-03-24"],
             "imputation": False, "calendar_compression": False})
    control("D03", "PASS", "actual train/calibration target construction and IDs",
            {"training": len(train_ids), "calibration": len(cal_ids), "population_sigma7_ddof": 0,
             "independent_direct8close_reconstruction_matches": True,
             "all81_saved_calibration_target_vectors_match": True})

    raw = {"training": development.raw_windows(train.dates),
           "calibration": development.raw_windows(calib.dates)}
    cohorts = {"training": train_ids, "calibration": cal_ids}
    original_y = {"training": np.asarray(train.y_original), "calibration": cal_truth}
    scaled_x = {"training": train.x.numpy(), "calibration": calib.x.numpy()}
    scaled_y = {"training": train.y.numpy(), "calibration": calib.y.numpy()}
    restored = recover_daily(root, raw_root, daily_path)
    if restored is not None:
        json_write(output / "evaluation_aggregation_receipt.json", restored)
    test_ids = pd.date_range("2025-01-01", "2026-08-31", tz="UTC")
    require(payload["dates"] == date_ids(test_ids) and all(payload["mask"]), "Saved evaluation mask/calendar changed")
    if daily_path.is_file():
        daily_hash = sha_file(daily_path)
        require(daily_hash in (DAILY_SHA, E1_DAILY_SHA), "Unrecognized evaluation daily source")
        source(daily_path, daily_hash, "original hash-pinned current or inherited daily evaluation cache")
        test_frame = pd.read_csv(daily_path, index_col="market_day_utc")
        test_frame.index = pd.DatetimeIndex(pd.to_datetime(test_frame.index, utc=True))
        validate_daily_frame(test_frame, "2025-01-01", "2026-08-31")
        full = _panel_from_frames(dev_frame, test_frame)
        require(full.support.loc[test_ids, "common"].all(), "Current common evaluation support is incomplete")
        test_y = full.targets_at(test_ids)
        np.testing.assert_allclose(test_y, payload["y_true"], rtol=1e-9, atol=1e-13)
        raw["evaluation"] = full.raw_windows(test_ids)
        cohorts["evaluation"] = date_ids(test_ids)
        original_y["evaluation"] = test_y
        scaled_x["evaluation"] = torch.as_tensor(train.scaler.transform(raw["evaluation"]), dtype=torch.float32).numpy()
        scaled_y["evaluation"] = torch.as_tensor(train.scaler.scale_targets(test_y), dtype=torch.float32).numpy()
        first_expected = full.features[full.calendar.get_indexer(pd.date_range("2024-12-25", "2024-12-31", tz="UTC"))]
        require(np.array_equal(raw["evaluation"][0], first_expected), "The causal December31 bridge changed")
        control("D04", "PASS", "actual608 evaluation targets, inputs and calendar",
                {"evaluation_rows": 608, "all_common_masks_true": True,
                 "target_max_abs_difference_from_frozen": float(np.max(np.abs(test_y - np.asarray(payload["y_true"])))),
                 "first_input_dates": date_ids(pd.date_range("2024-12-25", "2024-12-31", tz="UTC")),
                 "first_current_outcome_bar_excluded": True})
    else:
        full = bridge
        control("D04", "PARTIAL", "actual608 evaluation features and target reconstruction",
                {"saved_target_vector_and_IDs_available": True, "daily_OHLCV_available": False},
                "Recover the original daily cache or20 hash-pinned raw archives; do not fabricate X")

    support_rows = []
    all_ids = {**cohorts, "evaluation": date_ids(test_ids)}
    for left, right in (("training", "calibration"), ("training", "evaluation"), ("calibration", "evaluation")):
        found = support_intersections(all_ids[left], all_ids[right])
        require(not found["outcome_ID_intersection"], "Outcome IDs overlap between current cohorts")
        if left == "training":
            require(not found["target_close_support_intersection"] and not found["D7_input_support_intersection"],
                    "The registered seven-day train embargo failed")
        else:
            require(found["target_close_support_intersection"] == pd.date_range("2024-12-25", "2024-12-30").strftime("%Y-%m-%d").tolist(),
                    "Unexpected calibration/test target support")
            require(len(found["D7_input_support_intersection"]) == 5, "Unexpected calibration/test causal input history")
        support_rows.append({"pair": [left, right], **found,
                             "interpretation": "legitimate past rolling history, no reused fit outcome IDs" if left == "calibration"
                             else "embargo keeps target-close and D7 input supports disjoint"})
    json_write(output / "support_intersections.json", support_rows)
    control("D05", "PASS", "outcome IDs, embargo and rolling raw supports",
            {"pairs": 3, "all_outcome_IDs_disjoint": True, "train_raw_supports_disjoint": True,
             "cal_test_shared_target_closes": 6, "cal_test_shared_input_bars": 5,
             "cal_test_shared_returns": 5, "independence_of_calibration_and_test_claimed": False})

    # Check every admitted real window against its dated source, never only a fixture.
    for name in raw:
        panel = full if name == "evaluation" else development
        for i, day in enumerate(cohorts[name]):
            expected_days = pd.date_range(pd.Timestamp(day, tz="UTC") - pd.Timedelta(days=7),
                                          pd.Timestamp(day, tz="UTC") - pd.Timedelta(days=1))
            values = panel.features[panel.calendar.get_indexer(expected_days)]
            require(np.array_equal(values, raw[name][i]), "A current/future OHLCV entered a causal window")
    control("D06", "PASS" if "evaluation" in raw else "PARTIAL", "all available actual D7 input windows",
            {"real_windows_checked": sum(len(v) for v in raw.values()), "current_or_future_outcome_features": 0,
             "target_channel_added": False, "representation": "relative_window, five OHLCV channels"},
            "Evaluation windows still require actual daily features" if "evaluation" not in raw else "")

    relative = {k: relative_window(v) for k, v in raw.items()}
    manual_relative = {}
    for name, windows in raw.items():
        manual = np.empty_like(windows)
        manual[..., :4] = np.log(windows[..., :4]) - np.log(windows[:, -1, 3])[:, None, None]
        manual[..., 4] = np.log1p(windows[..., 4])
        require(np.array_equal(manual, relative[name]), "Production relative-window transformation differs")
        manual_relative[name] = manual
    pooled = manual_relative["training"]
    independent_training_signature = hashlib.sha256(json.dumps({
        "dates": train_ids, "relative_x": array_sha(pooled),
        "y": array_sha(original_y["training"])}, sort_keys=True, allow_nan=False).encode()).hexdigest()
    independent_scaler = {"mean": pooled.mean(axis=(0, 1)).tolist(), "std": pooled.std(axis=(0, 1), ddof=0).tolist(),
                          "target_scale": float(original_y["training"].mean()), "train_dates": train_ids,
                          "training_sha256": independent_training_signature}
    independent_scaler["std"] = [v if v else 1. for v in independent_scaler["std"]]
    comparison = compare_scaler_records(independent_scaler, train.scaler.record())
    require(comparison["matches"], "Final training-only scaler differs from independent algebraic reconstruction")
    for name in raw:
        manual_x = (manual_relative[name] - np.asarray(independent_scaler["mean"])) / np.asarray(independent_scaler["std"])
        require(np.array_equal(manual_x.astype(np.float32), scaled_x[name]), "Final float32 input conversion differs")
        np.testing.assert_allclose(train.scaler.inverse_targets(train.scaler.scale_targets(original_y[name])),
                                   original_y[name], rtol=1e-14, atol=1e-15)
    json_write(output / "reconstructed_training_scaler.json", train.scaler.record())
    control("D07", "PASS", "actual common final preprocessing, reconstructed only from655train",
            {"training_window_positions": 655 * 7, "parameter_comparison": comparison,
             "training_scaler_signature_independently_rehashed_from_actual_arrays": True,
             "float32_input_matrices_verified": list(raw), "target_scale_inverse_matches": True,
             "optimizer_or_predictor_fit_performed": False})
    # Metamorphic check changes real post-cutoff observations on a copy. This
    # verifies common preprocessing wiring, not every learned model behavior.
    changed_features = development.features
    later = development.calendar > pd.Timestamp("2024-10-31", tz="UTC")
    changed_features[later, :4] *= 1.125
    changed_features[later, 4] *= 1.25
    changed_targets = development.targets
    changed_targets[later & np.isfinite(changed_targets)] *= 1.5
    from e2_temporal_training import DatedPanel
    changed = DatedPanel(development.calendar, changed_features, changed_targets, development.quality)
    changed_train = final_training_split(changed)
    require(changed_train.signature == train.signature and changed_train.scaler.record() == train.scaler.record(),
            "Changing actual post-cutoff data changed training tensors/scaler")
    control("D08", "PASS", "common pipeline metamorphic check on copies of actual data",
            {"post_Oct31_values_changed": True, "training_signature_unchanged": True,
             "training_scaler_unchanged": True, "model_fit_or_inference": False},
            "This checks common preparation, not all author-port behavior")

    fits = list(csv.DictReader(fit_path.open()))
    final_fits = [r for r in fits if r["stage"] == "final"]
    require(len(final_fits) == 78 and all(r["train_signature"] == train.signature and r["train_count"] == "655" for r in final_fits),
            "Reconstructed real training signature differs from current78 saved fits")
    control("D09", "PASS", "current78 historical final-fit input signatures",
            {"matched_saved_final_rows": 78, "training_signature": train.signature,
             "original_fit_or_weights_reexecuted": False})

    duplicate_summary = duplicate_rows(raw, cohorts)
    csv_write(output / "exact_duplicate_windows.csv", duplicate_summary)
    exact_alerts = sum(r["exact_duplicate_query_count"] for r in duplicate_summary)
    control("D10", "PASS" if exact_alerts == 0 and "evaluation" in raw else "PARTIAL", "exact numerical D7 raw window fingerprints",
            {"cohort_scopes": len(duplicate_summary), "exact_duplicate_query_alerts": exact_alerts,
             "same_ID_excluded": True, "candidate_hash_matches_confirmed_by_array_equality": True},
            "Any content matches require review; duplicate patterns do not by themselves prove leakage" if exact_alerts else
            "Evaluation content unavailable" if "evaluation" not in raw else "No claim of universal absence of leakage")
    z = {k: train.scaler.transform(v) for k, v in raw.items()}
    neighbor_rows, summaries = [], []
    scopes = [(k + "_within", k, k, True) for k in raw] + [
        ("calibration_to_training", "calibration", "training", False),
        ("evaluation_to_training", "evaluation", "training", False),
        ("evaluation_to_calibration", "evaluation", "calibration", False)]
    for scope, query, reference, within in scopes:
        if query not in raw or reference not in raw:
            continue
        rows = nearest_rows(z[query], z[reference], cohorts[query], cohorts[reference], scope, within)
        neighbor_rows.extend(rows)
        distances = [r["rms_standardized_distance"] for r in rows]
        summaries.append({"scope": scope, "query_windows": len(rows), "reference_windows": len(raw[reference]),
                          "training_scaler_N": 655, "fixed_threshold": NEAR_THRESHOLD,
                          "near_query_count": sum(r["below_fixed_threshold"] for r in rows),
                          "minimum_RMS_distance": min(distances), "median_RMS_distance": float(np.median(distances)),
                          "interpretation": "descriptive similarity, no automatic exclusion or independence claim"})
    csv_write(output / "nearest_window_rows.csv", neighbor_rows)
    csv_write(output / "nearest_window_summary.csv", summaries)
    near_alerts = sum(r["near_query_count"] for r in summaries)
    control("D11", "PASS" if near_alerts == 0 and "evaluation" in raw else "PARTIAL", "actual available fixed-threshold near-window diagnostic",
            {"scopes": len(summaries), "query_rows_across_scopes": len(neighbor_rows),
             "near_query_alerts": near_alerts, "RMS_threshold": NEAR_THRESHOLD,
             "scaler_source": "training655 only", "target_or_current_day_used_as_covariate": False},
            "Review alerts if present; the threshold is descriptive and does not certify absence of leakage")

    deployed = []
    if assets_index.is_file():
        source(assets_index, role="recovered reproduction artifact index")
        index = json.loads(assets_index.read_text())
        rows = index.get("rows", [])
        freeze_rows = {(r["model"], str(r["seed"])): r for r in freeze["models"]}
        require({(r["model"], str(r["seed"])) for r in rows} == set(freeze_rows) and len(rows) == 81,
                "Recovered artifact index differs from the current81 predictor IDs")
        for row in rows:
            key = (row["model"], str(row["seed"]))
            path = Path(row["scaler_path"])
            source(path, row["scaler_sha256"], "recovered physically verified scaler JSON")
            require(row["scaler_sha256"] == freeze_rows[key]["scaler_sha256"], "Recovered scaler bytes differ from original frozen scaler")
            record = json.loads(path.read_text())
            neural = row["seed"] != "deterministic"
            result = compare_scaler_records(train.scaler.record(), record) if neural else {
                "matches": str(record.get("fit_cutoff", ""))[:10] == "2024-10-31",
                "scope": "classical frozen scaler identity and cutoff; not common neural standardization"}
            require(result["matches"], "Recovered frozen scaler does not match current training data: " + row["model"])
            deployed.append({"model": row["model"], "seed": row["seed"], "scaler_sha256": row["scaler_sha256"],
                             "original_freeze_scaler_bytes_match": True, **result})
        control("D12", "PASS", "all81 physical frozen scaler records and current training cutoff",
                {"neural_scalers_numerically_and_identity_checked": 78, "classical_scaler_hashes_and_cutoffs_checked": 3,
                 "checkpoint_weights_deserialized": False},
                "The recovered checkpoint files belong to the documented reproduction; this audit only reads JSON scalers")
    else:
        control("D12", "PARTIAL", "physical scaler records", {"available": False},
                "Recover SHA-bound scaler JSONs; no model fit is needed")
    json_write(output / "physical_scaler_checks.json", deployed)

    cohort_rows, npz_values, array_manifest = [], {}, {}
    for name in raw:
        ids = cohorts[name]
        for suffix, values in (("raw", raw[name]), ("relative", relative[name]), ("X", scaled_x[name]),
                               ("y_original", original_y[name]), ("y_scaled", scaled_y[name]),
                               ("dates", np.asarray(ids, dtype="U10"))):
            key = name + "_" + suffix
            npz_values[key] = values
            array_manifest[key] = {"shape": list(values.shape), "dtype": values.dtype.str, "array_sha256": array_sha(values)}
        for i, day in enumerate(ids):
            dt = date.fromisoformat(day)
            cohort_rows.append({"cohort": name, "outcome_day_utc": day,
                                "input_first_day": (dt - timedelta(days=7)).isoformat(),
                                "input_last_day": (dt - timedelta(days=1)).isoformat(),
                                "target_first_close_day": (dt - timedelta(days=7)).isoformat(),
                                "target_last_close_day": day, "raw_window_sha256": array_sha(raw[name][i]),
                                "relative_window_sha256": array_sha(relative[name][i]),
                                "X_float32_sha256": array_sha(scaled_x[name][i])})
    for key, value in (("training_target_scale", train.scaler.target_scale),
                       ("training_channel_mean", train.scaler.mean), ("training_channel_std", train.scaler.std)):
        values = np.asarray(value, dtype=np.float64)
        npz_values[key] = values
        array_manifest[key] = {"shape": list(values.shape), "dtype": values.dtype.str, "array_sha256": array_sha(values)}
    csv_write(output / "cohort_rows.csv", cohort_rows)
    np.savez_compressed(output / "current_cohort_arrays.npz", **npz_values)
    json_write(output / "cohort_array_manifest.json", {
        "version": VERSION, "file": "current_cohort_arrays.npz", "sha256": sha_file(output / "current_cohort_arrays.npz"),
        "arrays": array_manifest, "source_inputs": [r for r in provenance if r["role"] == "input" or "daily" in r["role"]],
        "scope": "Actual common data inputs for13 neural ports; classical references have declared endogenous histories.",
        "representation": "X=float32((relative_window(raw)-train_mean)/train_std); scaler from655 only; y_scaled=float32(y_original/train_target_mean)",
        "training_signature_matches_all78_saved_fits": True, "final_X_is_reconstructed_from_real_pinned_source_not_fabricated": "evaluation" in raw,
        "chronology_is_known_inherited_E1": True, "future_data_or_target_covariates_added": False,
        "original_predictor_training_or_inference": False})
    control("D13", "PARTIAL", "model-port behavior beyond common preprocessing",
            {"common_real_data_preparation_verified": True, "all_ports_executed_in_this_audit": False},
            "Port-specific learned behavior/negative controls require a separately recorded audit; this result is not an all-port leakage certificate")
    control("D14", "PARTIAL", "real permuted-target training negative control",
            {"executed_by_this_data_only_audit": False}, "Requires a separately registered bounded training control", True)
    control("D15", "PARTIAL", "adversarial train/test domain classifier",
            {"executed_by_this_data_only_audit": False}, "A classifier fit is a separate diagnostic of distribution difference, not proof of leakage", True)
    control("D16", "NA", "inter-annotator agreement/classification label-noise tests",
            {"target": "deterministic continuous sigma7", "human_annotators": False},
            "External measurement noise and a Bayes floor remain unidentified; arithmetic reconstruction is not their estimate")
    counts = dict(Counter(r["status"] for r in controls))
    report = {"version": VERSION, "completed_at_utc": utc(), "elapsed_compute_wall_seconds": time.monotonic() - started,
              "specification": {"path": "audit_specification.json", "sha256": sha_file(output / "audit_specification.json")},
              "status": "FAIL" if counts.get("FAIL") else "PARTIAL", "control_counts": counts,
              "data_ready_for_bounded_controls": "evaluation" in raw and all(r["status"] == "PASS" for r in controls[:12]),
              "arrays_npz_path": relative_name(output / "current_cohort_arrays.npz", root),
              "arrays_npz_sha256": sha_file(output / "current_cohort_arrays.npz"),
              "array_keys": {k: {"dates": k + "_dates", "X_scaled": k + "_X", "X_relative": k + "_relative",
                                  "raw": k + "_raw", "y_original": k + "_y_original", "y_scaled": k + "_y_scaled"}
                             for k in raw},
              "training_target_scale_key": "training_target_scale",
              "reconstructed_scaler_path": relative_name(output / "reconstructed_training_scaler.json", root),
              "reconstructed_scaler_sha256": sha_file(output / "reconstructed_training_scaler.json"),
              "original_frozen_neural_scaler_sha256_values": sorted({r["scaler_sha256"] for r in freeze["models"]
                                                                     if r["seed"] != "deterministic"}),
              "actual_cohorts_audited": {k: len(v) for k, v in cohorts.items()}, "controls": controls,
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                          "torch": torch.__version__, "scipy": scipy.__version__, "device": "CPU"},
              "causal_supports": support_rows, "declared_code_source_identity": source_identity,
              "provenance": provenance, "program_sha256": sha_file(Path(__file__)),
              "operations": {"model_fits": 0, "model_inference_calls": 0, "new_predictive_scores": 0,
                             "hypothesis_tests": 0, "bootstrap_draws": 0,
                             "training_scaler_algebraic_reconstructions": 2,
                             "actual_raw_archives_aggregated_this_run": 20 if restored else 0,
                             "frozen_scientific_files_modified": False, "new_blind_test_claimed": False}}
    json_write(output / "audit_result.json", report)
    csv_write(output / "controls.csv", [{**r, "observed": json.dumps(r["observed"], ensure_ascii=False)} for r in controls])
    json_write(output / "source_provenance.json", provenance)
    readme = ["# Auditoría correctiva de datos actuales", "", "Verificación posterior al benchmark, sin entrenamiento o inferencia de modelos.",
              "", "| Control | Estado | Alcance | Límite |", "| --- | --- | --- | --- |"]
    readme.extend(f"| {r['id']} | {r['status']} | {r['scope']} | {r['remaining']} |" for r in controls)
    readme.extend(["", "Los controles PASS se limitan a los datos y funciones indicados. PARTIAL no significa un fallo observado.",
                   "Las cinco barras de entrada y seis cierres objetivo compartidos entre calibración y test son historia causal legítima, sin claves de objetivo compartidas.",
                   "Los vecinos son un diagnóstico descriptivo con umbral heredado 0,01; no seleccionan transformaciones ni eliminan casos.",
                   "current_cohort_arrays.npz contiene datos reales reconstruidos y la conversión float32 común, no nuevas predicciones.",
                   "Los controles sintéticos unitarios comprueban el programa; la evidencia de mercado está en los resultados de esta ejecución real."])
    (output / "README.md").write_text("\n".join(readme) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "book_e2/_static/e2_corrections_20261115/data_audit")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "work/e2-corrections-assets/market_raw")
    parser.add_argument("--test-daily", type=Path, default=ROOT / "work/e2-corrections-assets/market_daily/known_E1_daily.csv")
    parser.add_argument("--assets-index", type=Path, default=ROOT / "work/e2-corrections-assets/metadata/checkpoint_index.json")
    parser.add_argument("--download-raw-only", action="store_true", help="I/O/hash verification only; no numerical data audit")
    parser.add_argument("--write-spec-only", action="store_true")
    args = parser.parse_args(argv)
    root, output = args.root.resolve(), args.output.resolve()
    sys.path.insert(0, str(root / "scripts"))
    operation_started = time.monotonic()
    try:
        initialize_specification(output)
        if args.write_spec_only:
            print("Corrective post-benchmark specification saved; no numerical audit executed.")
            return 0
        if args.download_raw_only:
            record = download_raw(root, args.raw_root.resolve(), output)
            print(json.dumps({k: record[k] for k in ("archive_count", "total_compressed_bytes", "elapsed_wall_seconds")}))
            return 0
        report = audit(root, output, args.raw_root.resolve(), args.test_daily.resolve(), args.assets_index.resolve())
        print(json.dumps({"status": report["status"], "actual_cohorts_audited": report["actual_cohorts_audited"],
                          "control_counts": report["control_counts"], "elapsed_seconds": report["elapsed_compute_wall_seconds"]}))
        return 1 if report["status"] == "FAIL" else 0
    except Exception as exc:
        recovery = args.download_raw_only
        failure = {"version": VERSION, "at_utc": utc(), "status": "PARTIAL" if recovery else "FAIL",
                   "phase": "source_recovery" if recovery else "data_audit", "error_type": type(exc).__name__,
                   "elapsed_wall_seconds": time.monotonic() - operation_started,
                   "error": str(exc), "scope": "Source recovery failed before numerical data audit; no data criterion was tested." if recovery
                   else "This audit failed; no success or global leakage certificate is issued.",
                   "model_fits": 0, "model_inference_calls": 0}
        output.mkdir(parents=True, exist_ok=True)
        json_write(output / ("source_recovery_failure.json" if recovery else "audit_failure.json"), failure)
        print(json.dumps(failure, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
