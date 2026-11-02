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
    report = {"package": "timeseries-cv", "version": "0.1.5",
              "function": "split_train_val_forwardChaining",
              "signature": str(inspect.signature(split_train_val_forwardChaining)),
              "illustration": fig, "example": {"length": 27, "input": 4,
              "forecast": 3, "jump": 2, "folds": example_records},
              "project": {"source_csv_sha256": digest(source / "daily_market_development.csv"),
              "volatility_csv_sha256": digest(source / "daily_volatility_development.csv"),
              "frequency": "daily UTC", "input_days": [7, 14, 21, 28],
              "volatility_days": [7, 14, 21, 28], "forecast_days": 1, "jump_days": 1,
              "validation_year": 2024, "test_archives_opened": 0, "models_trained": 0},
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
            # Enforce observability even if library defaults/other horizons change.
            known = train_outcomes <= origin_previous_day
            input_ok = np.isfinite(features[train_inputs]).all(axis=(1, 2))
            for column, window in enumerate((7, 14, 21, 28)):
                valid = known & input_ok & np.isfinite(targets[train_outcomes, column])
                selected = train_outcomes[valid]
                if (not len(selected) or not np.isfinite(features[val_inputs]).all()
                        or not np.isfinite(targets[outcome, column])):
                    continue
                assert max(selected) < outcome
                assert int(train_inputs[valid].max()) < outcome
                assert all(dates[i] <= dates[origin_previous_day] for i in selected)
                config_counts[window].append(len(selected))
                rows.append({"input_days": history, "volatility_days": window,
                    "forecast_days": 1, "validation_outcome_day_utc": str(dates[outcome]),
                    "validation_input_first_day_utc": str(dates[int(val_inputs.min())]),
                    "validation_input_last_day_utc": str(dates[origin_previous_day]),
                    "train_first_outcome_day_utc": str(dates[int(selected.min())]),
                    "train_last_outcome_day_utc": str(dates[int(selected.max())]),
                    "train_samples": len(selected),
                    "train_outcome_indices_sha256": hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest(),
                    "library_fold": int(fold), "future_train_labels_removed": int((~known).sum())})
        for window, counts in config_counts.items():
            report["configurations"].append({"input_days": history, "volatility_days": window,
                "validation_folds": len(counts), "min_train_samples": min(counts),
                "max_train_samples": max(counts)})
    assert len(rows) == 16 * 366, "Expected one eligible validation origin per 2024 day/configuration"
    with (out / "folds.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report["total_configuration_folds"] = len(rows)
    report["folds_csv_sha256"] = digest(out / "folds.csv")
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Example: 8 folds. Project: {len(rows)} configuration-folds, 366 per configuration.")


if __name__ == "__main__":
    main()
