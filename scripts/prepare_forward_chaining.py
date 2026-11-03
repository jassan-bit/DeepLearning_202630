"""Validate the official tsxv example and materialize causal daily CV manifests.

The library receives calendar indices; OHLCV inputs and volatility labels are
mapped separately. No model is trained and no test archive is opened.
"""
import argparse
import base64
import csv
from datetime import date, timedelta
import hashlib
import inspect
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from tsxv.splitTrainVal import split_train_val_forwardChaining

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "8e9137c2774e92a6d7a80446e4022c27ca1b7991"
NOTEBOOK = f"https://raw.githubusercontent.com/DidierRLopes/timeseries-cv/{COMMIT}/example.ipynb"
PNG_SHA = "de6c7fff71c7fc0c76261508925b13d2ca5f6b31217cda80ed0f958249c80781"
RULE_VERSION = 2
PREVIOUS_FOLDS_SHA = "dbe66b20b72f28b7e477c79d8b1887fc80b96957fa3fbaab0e211f2a5b398dc5"


def illustration(refresh=False):
    path = ROOT / "book/_static/eda/08_forward_chaining_tsxv.png"
    if refresh or not path.exists():
        with urlopen(NOTEBOOK, timeout=30) as response:
            notebook = json.load(response)
        cell = [c for c in notebook["cells"] if "splitTrainVal.split_train_val_forwardChaining"
                in "".join(c.get("source", []))]
        if len(cell) != 1:
            raise ValueError("Official notebook structure changed")
        png = [o["data"]["image/png"] for o in cell[0]["outputs"]
               if "image/png" in o.get("data", {})]
        if len(png) != 1:
            raise ValueError("Expected one official illustration")
        encoded = "".join(png[0]) if isinstance(png[0], list) else png[0]
        payload = base64.b64decode(encoded)
        if hashlib.sha256(payload).hexdigest() != PNG_SHA:
            raise ValueError("Official illustration hash mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    if hashlib.sha256(path.read_bytes()).hexdigest() != PNG_SHA:
        raise ValueError("Cached illustration hash mismatch")
    return {"notebook_url": NOTEBOOK, "source_commit": COMMIT, "sha256": PNG_SHA,
            "license": "MIT", "copyright": "2021 DidierRLopes",
            "credit": "timeseries-cv; coauthorship Filipe R. Ramos, 2021",
            "role": "official synthetic example; not Bitcoin dates or a trained model"}


def load_csv(path, columns):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    dates = [date.fromisoformat(next(iter(row.values()))[:10]) for row in rows]
    values = np.asarray([[float(row[c]) if row[c] else np.nan for c in columns] for row in rows])
    return dates, values


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-illustration", action="store_true")
    args = parser.parse_args()
    out = ROOT / "results/cv_forward_chaining"
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / "results/eda_development"
    dates, features = load_csv(source / "daily_market_development.csv",
                               ["open", "high", "low", "close", "volume"])
    vol_dates, targets = load_csv(source / "daily_volatility_development.csv",
                                  [f"sigma_{n}" for n in (7, 14, 21, 28)])
    assert dates == vol_dates
    assert all(b-a == timedelta(days=1) for a, b in zip(dates, dates[1:]))
    assert dates[0] == date(2023, 1, 1) and dates[-1] == date(2024, 12, 31)
    assert all(d.year in (2023, 2024) for d in dates)
    fig = illustration(args.refresh_illustration)

    # Public README keyword aliases do not match the installed 0.1.5 signature.
    example = split_train_val_forwardChaining(np.arange(27), 4, 3, 2)
    X, y, Xcv, ycv = example
    assert len(X) == len(y) == len(Xcv) == len(ycv) == 8
    example_records = []
    for k in X:
        assert X[k].shape == (k+2, 4) and y[k].shape == (k+2, 3)
        assert int(y[k].max()) <= int(Xcv[k].max())
        assert int(Xcv[k].max()) < int(ycv[k].min())
        example_records.append({"fold": k+1, "X": X[k].tolist(), "y": y[k].tolist(),
                                "Xcv": Xcv[k].tolist(), "ycv": ycv[k].tolist()})
    report = {"manifest_rule_version": RULE_VERSION,
              "supersedes_fold_manifest_sha256": PREVIOUS_FOLDS_SHA,
              "previous_manifest_location": "Git history before explicit embargo rule version 2",
              "generator_script_sha256": digest(Path(__file__)),
              "package": "timeseries-cv", "version": "0.1.5",
              "function": "split_train_val_forwardChaining",
              "signature": str(inspect.signature(split_train_val_forwardChaining)),
              "illustration": fig, "example": {"length": 27, "input": 4,
              "forecast": 3, "jump": 2, "folds": example_records},
              "project": {"source_csv_sha256": digest(source / "daily_market_development.csv"),
              "volatility_csv_sha256": digest(source / "daily_volatility_development.csv"),
              "frequency": "daily UTC", "input_days": [7, 14, 21, 28],
              "volatility_days": [7, 14, 21, 28], "forecast_days": 1, "jump_days": 1,
              "validation_year": 2024, "test_archives_opened": 0,
              "test_files_opened": 0, "models_trained": 0,
              "data_preparation_only": True,
              "missing_market_days_preserved": [str(day) for day, row in zip(dates, features)
                                                if not np.isfinite(row).all()],
              "embargo": {
                  "guide_route": "6.4 temporal regression",
                  "days": "D = maximum covariate lag in calendar days = input_days",
                  "excluded_training_outcome_interval": "[t-D, t-1] inclusive",
                  "last_allowed_training_outcome": "t-D-1",
                  "minimum_distance_between_training_and_validation_outcomes_days": "D+1",
                  "additional_filters": "label availability, finite complete OHLCV windows and finite labels",
                  "target_support": "Y_t uses returns from t-n+1 through t inclusive",
                  "limitation": "An embargo of D does not guarantee disjoint target return support when n>D+1; shared returns are recorded and dependence must be addressed in later inference."
              }},
              "configurations": []}
    rows = []
    for history in (7, 14, 21, 28):
        tx, ty, vx, vy = split_train_val_forwardChaining(np.arange(len(dates)), history, 1, 1)
        config_counts = {n: [] for n in (7, 14, 21, 28)}
        for fold in tx:
            outcome = int(vy[fold][0, 0])
            if dates[outcome].year != 2024:
                continue
            train_inputs = tx[fold].astype(int)
            train_outcomes = ty[fold][:, 0].astype(int)
            val_inputs = vx[fold][0].astype(int)
            origin_previous_day = int(val_inputs.max())
            assert np.array_equal(val_inputs, np.arange(outcome-history, outcome))
            embargo_days = history
            last_allowed_outcome = outcome - embargo_days - 1
            # Enforce observability even if library defaults/other horizons change.
            known = train_outcomes <= origin_previous_day
            # Guide 6.4: omit D calendar outcome days immediately before t.
            # The library alone ends at t-D and omits only D-1 interior days.
            outside_embargo = train_outcomes <= last_allowed_outcome
            input_ok = np.isfinite(features[train_inputs]).all(axis=(1, 2))
            for column, window in enumerate((7, 14, 21, 28)):
                eligible_before_embargo = known & input_ok & np.isfinite(targets[train_outcomes, column])
                valid = eligible_before_embargo & outside_embargo
                selected = train_outcomes[valid]
                if (not len(selected) or not np.isfinite(features[val_inputs]).all()
                        or not np.isfinite(targets[outcome, column])):
                    continue
                assert max(selected) < outcome
                assert int(selected.max()) <= last_allowed_outcome
                assert (dates[outcome] - dates[int(selected.max())]).days >= embargo_days+1
                assert int(train_inputs[valid].max()) < outcome
                assert all(dates[i] <= dates[origin_previous_day] for i in selected)
                shared_return_days = max(0, int(selected.max()) - (outcome-window+1) + 1)
                config_counts[window].append(len(selected))
                rows.append({"input_days": history, "volatility_days": window,
                    "forecast_days": 1, "manifest_rule_version": RULE_VERSION,
                    "embargo_days": embargo_days,
                    "validation_outcome_day_utc": str(dates[outcome]),
                    "validation_input_first_day_utc": str(dates[int(val_inputs.min())]),
                    "validation_input_last_day_utc": str(dates[origin_previous_day]),
                    "embargo_first_outcome_day_utc": str(dates[outcome-embargo_days]),
                    "embargo_last_outcome_day_utc": str(dates[origin_previous_day]),
                    "last_allowed_train_outcome_day_utc": str(dates[last_allowed_outcome]),
                    "train_first_outcome_day_utc": str(dates[int(selected.min())]),
                    "train_last_outcome_day_utc": str(dates[int(selected.max())]),
                    "train_samples": len(selected),
                    "embargo_removed_train_samples": int((eligible_before_embargo & ~outside_embargo).sum()),
                    "validation_target_first_return_day_utc": str(dates[outcome-window+1]),
                    "validation_target_last_return_day_utc": str(dates[outcome]),
                    "latest_train_validation_shared_return_days": shared_return_days,
                    "train_outcome_indices_sha256": hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest(),
                    "library_fold": int(fold), "future_train_labels_removed": int((~known).sum())})
        for window, counts in config_counts.items():
            report["configurations"].append({"input_days": history, "volatility_days": window,
                "validation_folds": len(counts), "min_train_samples": min(counts),
                "max_train_samples": max(counts), "embargo_days": history,
                "latest_train_validation_shared_return_days": max(0, window-history-1)})
    assert len(rows) == 16 * 366, "Expected one eligible validation origin per 2024 day/configuration"
    with (out / "folds.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    report["total_configuration_folds"] = len(rows)
    report["folds_csv_sha256"] = digest(out / "folds.csv")
    report["checks"] = {
        "all_folds_respect_explicit_embargo": all(
            date.fromisoformat(row["train_last_outcome_day_utc"])
            <= date.fromisoformat(row["last_allowed_train_outcome_day_utc"])
            for row in rows),
        "all_expected_validation_origins_preserved": len(rows) == 16*366,
        "total_embargo_removed_eligible_samples_across_folds": sum(
            row["embargo_removed_train_samples"] for row in rows),
        "folds_with_shared_target_returns": sum(
            row["latest_train_validation_shared_return_days"] > 0 for row in rows),
        "source_csv_unchanged_during_generation": report["project"]["source_csv_sha256"]
            == digest(source / "daily_market_development.csv"),
        "volatility_csv_unchanged_during_generation": report["project"]["volatility_csv_sha256"]
            == digest(source / "daily_volatility_development.csv")}
    assert report["checks"]["all_folds_respect_explicit_embargo"]
    assert report["checks"]["source_csv_unchanged_during_generation"]
    assert report["checks"]["volatility_csv_unchanged_during_generation"]
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    assets = ROOT / "book/_static/cv_forward_chaining"
    assets.mkdir(parents=True, exist_ok=True)
    for name in ("folds.csv", "summary.json"):
        (assets / name).write_bytes((out / name).read_bytes())
    print(f"Example: 8 folds. Project: {len(rows)} configuration-folds, 366 per configuration.")


if __name__ == "__main__":
    main()
