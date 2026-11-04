"""Materialize multi-history monthly blocks from real timeseries-cv daily splits.

Run with the dedicated timeseries-cv environment (NumPy1.26, timeseries-cv0.1.5).
The library supplies daily forward-chaining indices; this script groups their
2024 validation outcomes into twelve calendar months. It does not claim that
the library has a monthly API, fit a model, or open raw/reserved-test data.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path

import numpy as np
from tsxv.splitTrainVal import split_train_val_forwardChaining


ROOT = Path(__file__).resolve().parents[1]
HISTORIES = (7, 14, 21, 28)
SOURCES = (
    "results/eda_development/daily_market_development.csv",
    "results/eda_development/daily_volatility_development.csv",
)
FEATURES = ("open", "high", "low", "close", "volume")


def digest_indices(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_development(root, history):
    """Read only the two existing daily development CSVs, without pandas."""
    frames = []
    for name, columns in zip(SOURCES, (FEATURES, ("sigma_7",))):
        with (root / name).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        days = [date.fromisoformat(row["market_day_utc"][:10]) for row in rows]
        values = np.asarray([
            [float(row[column]) if row[column] else np.nan for column in columns]
            for row in rows
        ], dtype=np.float64)
        frames.append((days, values))
    days, features = frames[0]
    if days != frames[1][0]:
        raise ValueError("Development features and targets use different calendars")
    expected = [date(2023, 1, 1) + timedelta(days=i) for i in range(731)]
    if days != expected or days[-1] != date(2024, 12, 31):
        raise ValueError("Only the complete2023--2024 UTC calendar is admitted")
    targets = frames[1][1][:, 0]
    if np.isinf(features).any() or np.isinf(targets).any():
        raise ValueError("Infinite observations are not admissible")
    complete = np.isfinite(features).all(axis=1)
    if [d for d, valid in zip(days, complete) if not valid] != [date(2023, 3, 24)]:
        raise ValueError("The established incomplete-day mask changed")
    if (features[complete, :4] <= 0).any() or (features[complete, 4] < 0).any():
        raise ValueError("OHLC must be positive and volume nonnegative")
    returns = np.full(len(days), np.nan)
    returns[1:] = np.diff(np.log(features[:, 3]))
    reconstructed = np.full(len(days), np.nan)
    for outcome in range(7, len(days)):
        window = returns[outcome - 6:outcome + 1]
        if np.isfinite(window).all():
            reconstructed[outcome] = np.std(window, ddof=0)
    if not np.array_equal(np.isfinite(targets), np.isfinite(reconstructed)):
        raise ValueError("sigma7 eligibility does not match its full-calendar definition")
    if not np.allclose(targets, reconstructed, rtol=1e-9, atol=1e-13, equal_nan=True):
        raise ValueError("sigma7 labels are not aligned to the outcome day")
    eligible = np.zeros(len(days), dtype=bool)
    for outcome in range(history, len(days)):
        eligible[outcome] = (np.isfinite(targets[outcome])
                             and np.isfinite(features[outcome - history:outcome]).all())
    return days, features, targets, eligible


def build_manifest(history, root=ROOT):
    """Return audited index-only folds generated using the installed library."""
    root = Path(root)
    if isinstance(history, bool) or history not in HISTORIES:
        raise ValueError("History must be one of 7, 14, 21, 28 days")
    version = importlib.metadata.version("timeseries-cv")
    if version != "0.1.5" or not np.__version__.startswith("1.26."):
        raise ValueError("Use the registered timeseries-cv0.1.5 / NumPy1.26 environment")
    days, features, targets, eligible = load_development(root, history)
    sequence = np.arange(len(days), dtype=np.int64)
    xtrain, ytrain, xvalidation, yvalidation = split_train_val_forwardChaining(
        sequence, history, 1, 1
    )
    library = {}
    for key in xtrain:
        xt, yt = np.asarray(xtrain[key]), np.asarray(ytrain[key])
        xv, yv = np.asarray(xvalidation[key]), np.asarray(yvalidation[key])
        if yt.shape != (len(xt), 1) or xv.shape != (1, history) or yv.shape != (1, 1):
            raise ValueError("Unexpected installed library split shapes")
        outcome = int(yv[0, 0])
        if outcome in library:
            raise ValueError("Duplicate library validation outcome")
        expected_inputs = yt[:, 0, None] - np.arange(history, 0, -1)
        if not np.array_equal(xt, expected_inputs):
            raise ValueError("Library training windows do not match canonical outcome windows")
        if not np.array_equal(xv[0], np.arange(outcome - history, outcome)):
            raise ValueError("Library validation window does not match the canonical window")
        library[outcome] = {"key": int(key), "train": yt[:, 0], "input": xv[0]}

    def filtered_training(origin):
        candidates = library[origin]["train"]
        cutoff = origin - history - 1
        admitted = candidates[(candidates <= cutoff) & eligible[candidates]]
        if not len(admitted) or np.any(admitted + 1 > origin):
            raise ValueError("Empty training cohort or an unavailable training label")
        # The raw OHLCV values at those indices equal the canonical D-day view.
        for outcome in admitted:
            canonical = features[outcome - history:outcome]
            inputs = outcome - np.arange(history, 0, -1)
            if not np.array_equal(features[inputs], canonical):
                raise ValueError("Library-to-feature mapping changed the canonical input window")
        return admitted, cutoff

    def cohort_details(indices):
        values = np.asarray(indices, dtype=np.int64)
        return {"count": int(len(values)), "first_day": days[int(values[0])].isoformat(),
                "last_day": days[int(values[-1])].isoformat(),
                "indices_sha256": digest_indices(values)}

    position = {day: i for i, day in enumerate(days)}
    library_outcomes = np.asarray(sorted(library), dtype=np.int64)
    folds = []
    for month in range(1, 13):
        origin_day = date(2024, month, 1)
        origin = position[origin_day]
        inner_start = origin - 90
        train, cutoff = filtered_training(origin)
        inner_train, inner_cutoff = filtered_training(inner_start)
        inner_validation = library_outcomes[
            (library_outcomes >= inner_start) & (library_outcomes <= cutoff)
            & eligible[library_outcomes]
        ]
        test = np.asarray([i for i in library_outcomes
                           if days[i].year == 2024 and days[i].month == month and eligible[i]],
                          dtype=np.int64)
        if not len(inner_validation) or not len(test):
            raise ValueError("Every monthly fold needs inner validation and daily evaluation")
        if np.intersect1d(inner_train, inner_validation).size or np.intersect1d(train, test).size:
            raise ValueError("Training/evaluation outcomes overlap")
        if not np.array_equal(np.union1d(inner_train, inner_validation),
                              train[(train <= inner_cutoff) | (train >= inner_start)]):
            raise ValueError("Internal cohorts do not respect the registered internal embargo")
        test_inputs = np.asarray([library[int(i)]["input"] for i in test], dtype=np.int64)
        if not np.array_equal(test_inputs, test[:, None] - np.arange(history, 0, -1)):
            raise ValueError("Evaluation inputs cross their daily prediction origins")
        for outcome, inputs in zip(test, test_inputs):
            if not np.array_equal(features[inputs], features[outcome - history:outcome]):
                raise ValueError("Evaluation feature windows differ from canonical windows")
        folds.append({
            "fold_id": f"2024-{month:02d}", "fit_origin_index": int(origin),
            "fit_origin_day_utc": origin_day.isoformat(),
            "library_outer_fold_key": library[origin]["key"],
            "library_inner_fold_key": library[inner_start]["key"],
            "library_outer_train_candidate_last_index": int(library[origin]["train"][-1]),
            "outer_training_last_allowed_index": int(cutoff),
            "outer_training_last_allowed_day_utc": days[cutoff].isoformat(),
            "inner_validation_start_index": int(inner_start),
            "inner_validation_start_day_utc": days[inner_start].isoformat(),
            "inner_training_last_allowed_index": int(inner_cutoff),
            "inner_training_last_allowed_day_utc": days[inner_cutoff].isoformat(),
            "train_indices": train.tolist(), "inner_train_indices": inner_train.tolist(),
            "inner_validation_indices": inner_validation.tolist(), "test_indices": test.tolist(),
            "test_input_indices": test_inputs.tolist(),
            "test_input_indices_sha256": digest_indices(test_inputs),
            "train": cohort_details(train), "inner_train": cohort_details(inner_train),
            "inner_validation": cohort_details(inner_validation), "test": cohort_details(test),
            "daily_prediction_origins_utc": [days[i].isoformat() + "T00:00:00+00:00" for i in test],
            "daily_target_available_at_utc": [
                (days[i] + timedelta(days=1)).isoformat() + "T00:00:00+00:00" for i in test
            ],
            "availability_checks": {"all_train_labels_available_before_fit": bool(np.all(train + 1 <= origin)),
                                    "all_inner_labels_available_before_inner_origin": bool(np.all(inner_train + 1 <= inner_start)),
                                    "all_test_inputs_before_daily_origin": bool(np.all(test_inputs < test[:, None])),
                                    "test_labels_not_used_for_month_fit": True},
        })
    all_test = np.asarray([i for fold in folds for i in fold["test_indices"]], dtype=np.int64)
    expected_test = np.asarray([i for i, day in enumerate(days) if day.year == 2024], dtype=np.int64)
    if not np.array_equal(all_test, expected_test) or len(np.unique(all_test)) != 366:
        raise ValueError("Every2024 calendar date must appear exactly once as a daily holdout")
    july_test = [i for i in all_test if days[i].month >= 7]
    if len(july_test) != 184:
        raise ValueError("Expected184 July--December daily evaluation outcomes")
    gap = position[date(2023, 3, 24)]
    for fold in folds:
        for key in ("train_indices", "inner_train_indices", "inner_validation_indices"):
            indices = np.asarray(fold[key], dtype=np.int64)
            if np.any((indices[:, None] - np.arange(history, 0, -1)) == gap):
                raise ValueError("An admitted input window crosses the incomplete March2023 day")
            if not eligible[indices].all():
                raise ValueError("An admitted cohort contains an invalid target or input window")
    function_file = Path(inspect.getsourcefile(split_train_val_forwardChaining))
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Development-only monthly blocked forward CV, no model fitting or reserved-test evaluation",
        "generator": "scripts/prepare_filipe_multihistory.py", "generator_sha256": digest_file(__file__),
        "library": {"distribution": "timeseries-cv", "version": version, "numpy_version": np.__version__,
                    "function": "tsxv.splitTrainVal.split_train_val_forwardChaining",
                    "signature": str(inspect.signature(split_train_val_forwardChaining)),
                    "function_source_sha256": hashlib.sha256(inspect.getsource(split_train_val_forwardChaining).encode()).hexdigest(),
                    "module_source_sha256": digest_file(function_file)},
        "library_call": f"split_train_val_forwardChaining(np.arange(731),{history},1,1)",
        "adaptation": "Actual daily library validation origins are grouped into twelve calendar holdout blocks. The library has no monthly API. Training candidates come from the first daily library fold of each month and receive the stricter explicit embargo and eligibility filter.",
        "model_update_policy": "One month-specific model fit/refit procedure; fixed weights within month; daily h1 predictions use each day's already observed OHLCV history. This is neither a month-ahead forecast nor366 daily refits.",
        "inner_validation_policy": "Last90 calendar days before monthly fit origin, ending at outer training cutoff; inner training from the library fold at the inner start with its own embargo.",
        "history_days": history, "target_returns_days": 7, "forecast_days": 1,
        "embargo_rule": "training outcome u <= origin-D-1; input windows and label eligibility checked separately",
        "source_hashes": {name: digest_file(root / name) for name in SOURCES},
        "calendar_first_day": days[0].isoformat(), "calendar_last_day": days[-1].isoformat(),
        "calendar_days": len(days), "incomplete_input_day": "2023-03-24",
        "library_daily_validation_origins": len(library), "monthly_fold_count": len(folds),
        "evaluation_daily_outcomes": len(all_test), "july_december_daily_outcomes": len(july_test),
        "folds": folds, "models_fitted": 0, "raw_archives_opened": 0, "test_files_opened": 0,
        "test_indices_meaning": "Monthly CV validation outcomes inside development2024; not the reserved2025+ test",
    }


def write_manifest(manifest, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "folds.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    fields = ["fold_id", "fit_origin_day_utc", "inner_validation_start_day_utc",
              "outer_training_last_allowed_day_utc", "inner_training_last_allowed_day_utc"]
    fields += [f"{cohort}_{field}" for cohort in ("train", "inner_train", "inner_validation", "test")
               for field in ("count", "first_day", "last_day", "indices_sha256")]
    with (output_dir / "folds.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for fold in manifest["folds"]:
            row = {key: fold[key] for key in fields[:5]}
            row.update({f"{cohort}_{field}": value for cohort in ("train", "inner_train", "inner_validation", "test")
                        for field, value in fold[cohort].items()})
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/filipe_multihistory_20261104")
    parser.add_argument("--histories", type=int, nargs="+", choices=HISTORIES, default=[14, 21, 28])
    args = parser.parse_args()
    for history in args.histories:
        manifest = build_manifest(history)
        write_manifest(manifest, args.output_dir / f"D{history}")
        print(json.dumps({"history_days": history, "monthly_folds": manifest["monthly_fold_count"],
                          "daily_outcomes": manifest["evaluation_daily_outcomes"],
                          "library_version": manifest["library"]["version"],
                          "models_fitted": 0, "test_files_opened": 0}), flush=True)


if __name__ == "__main__":
    main()
