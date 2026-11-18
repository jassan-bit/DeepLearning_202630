"""Post-hoc physical-constraint audit of frozen sigma7 point predictions.

Only saved prediction arithmetic is performed. No model or training module is
imported; no fitting, model forward, recalibration or hypothesis test is run.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
NPZ_SHA256 = "186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683"
LOADER_SHA256 = "58d9cde8af881f9a987e449e6b70c301aa37c4b35fd4061e4d8bb5110b7e6302"
VIOLATION_TOLERANCE_SIGMA7 = 1e-12
NUMERIC_LOSS_TOLERANCE = 1e-15


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def frozen_loader(root=ROOT):
    path = Path(root) / "experiments/e3_estadistica_posterior_20261117/inputs.py"
    require(sha(path) == LOADER_SHA256, "Frozen prediction admission source differs")
    spec = importlib.util.spec_from_file_location("frozen_constraint_inputs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def observable_variance_floor(closes):
    """A from exactly seven past closes: six returns, population divisor seven.

    This function accepts no target, future return, fitted statistic or label.
    Its output is variance (sigma7 squared), not sigma7.
    """
    closes = np.asarray(closes, dtype=np.float64)
    require(closes.ndim == 2 and closes.shape[1] == 7,
            "Expected N-by-7 chronologically ordered past closes")
    require(np.isfinite(closes).all() and (closes > 0).all(),
            "Past closes must be finite and positive")
    returns = np.diff(np.log(closes), axis=1)
    return np.sum((returns - returns.mean(axis=1, keepdims=True)) ** 2, axis=1) / 7.0


def validate_chronology(raw, dates):
    """Check the audited t-7..t-1 contract using adjacent overlapping windows."""
    raw = np.asarray(raw, dtype=np.float64)
    require(raw.shape == (len(dates), 7, 5) and len(dates) >= 2,
            "Expected aligned N-by-7-by-5 OHLCV windows")
    days = [date.fromisoformat(str(day)) for day in dates]
    require([day.isoformat() for day in days] == list(dates), "Noncanonical date")
    require(all(b - a == timedelta(days=1) for a, b in zip(days, days[1:])),
            "Target dates must be unique, ordered and consecutive")
    require(np.isfinite(raw).all() and (raw[:, :, :4] > 0).all()
            and (raw[:, :, 4] >= 0).all(), "Invalid raw OHLCV")
    require((raw[:, :, 1] >= np.maximum(raw[:, :, 0], raw[:, :, 3])).all()
            and (raw[:, :, 2] <= np.minimum(raw[:, :, 0], raw[:, :, 3])).all(),
            "Invalid OHLC geometry")
    require(np.array_equal(raw[:-1, 1:, :], raw[1:, :-1, :]),
            "Adjacent windows do not preserve chronological six-bar overlap")
    return [{"target_date": day.isoformat(),
             "first_input_date": (day - timedelta(days=7)).isoformat(),
             "last_input_date": (day - timedelta(days=1)).isoformat()}
            for day in days]


def project(predictions, floor):
    predictions = np.asarray(predictions, dtype=np.float64)
    floor = np.asarray(floor, dtype=np.float64)
    require(predictions.ndim == 2 and floor.shape == (predictions.shape[1],),
            "Expected predictor-by-date predictions and a date vector floor")
    require(np.isfinite(predictions).all() and np.isfinite(floor).all()
            and (floor >= 0).all(), "Nonfinite prediction or invalid sigma7 floor")
    return np.maximum(predictions, floor[None, :])


def audit_loss_monotonicity(targets, before, after, floor):
    require(np.all(targets >= floor), "Target below floor: projection theorem inapplicable")
    mse_increase = (after - targets) ** 2 - (before - targets) ** 2
    mae_increase = np.abs(after - targets) - np.abs(before - targets)
    require(np.max(mse_increase) <= NUMERIC_LOSS_TOLERANCE,
            "Pointwise squared loss increased after projection")
    require(np.max(mae_increase) <= NUMERIC_LOSS_TOLERANCE,
            "Pointwise absolute loss increased after projection")
    return {"maximum_pointwise_squared_loss_increase": float(np.max(mse_increase)),
            "maximum_pointwise_absolute_loss_increase": float(np.max(mae_increase))}


def predictor_metrics(model, seed, before, after, targets, floor):
    shortfall = np.maximum(floor - before, 0.0)
    strict = shortfall > 0
    violations = shortfall > VIOLATION_TOLERANCE_SIGMA7
    mse_before, mse_after = np.mean((before - targets) ** 2), np.mean((after - targets) ** 2)
    rmse_before, rmse_after = float(np.sqrt(mse_before)), float(np.sqrt(mse_after))
    return {"model": model, "seed": seed, "dates": len(targets),
            "strict_below_floor_count": int(strict.sum()),
            "below_floor_count_tolerance": int(violations.sum()),
            "below_floor_rate_tolerance": float(violations.mean()),
            "shortfall_mean_all_sigma7": float(shortfall.mean()),
            "shortfall_mean_when_strict_sigma7": float(shortfall[strict].mean()) if strict.any() else None,
            "shortfall_max_sigma7": float(shortfall.max()),
            "RMSE_before": rmse_before, "RMSE_projected": rmse_after,
            "RMSE_reduction": rmse_before - rmse_after,
            "MAE_before": float(np.mean(np.abs(before - targets))),
            "MAE_projected": float(np.mean(np.abs(after - targets))),
            "MSE_reduction": float(mse_before - mse_after),
            "projection_changed_count": int(np.count_nonzero(before != after))}


def run(data_dir, npz_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loader, loader_path = frozen_loader()
    panel = loader.load_inputs(data_dir)  # All 11 pinned hashes plus semantic gates.
    require(sha(npz_path) == NPZ_SHA256, "Audited raw-cohort NPZ SHA256 differs")
    # Do not open scaled arrays, target scales, calibration or training data.
    members = ["evaluation_raw", "evaluation_y_original", "evaluation_dates"]
    with np.load(npz_path, allow_pickle=False) as source:
        raw, targets, dates_array = (np.asarray(source[key]).copy() for key in members)
    dates = dates_array.tolist()
    require(dates == panel["dates"] and len(dates) == 608, "Exact 608-date support differs")
    require(targets.shape == (608,) and np.array_equal(targets, panel["targets"]),
            "NPZ targets differ from the frozen prediction panel")
    supports = validate_chronology(raw, dates)
    A = observable_variance_floor(raw[:, :, 3])
    floor = np.sqrt(A)
    # Independent pairwise identity: sum_{i<j}(r_i-r_j)^2 / (6*7).
    r = np.diff(np.log(raw[:, :, 3]), axis=1)
    pairwise_A = sum((r[:, i] - r[:, j]) ** 2
                     for i in range(6) for j in range(i + 1, 6)) / 42.0
    np.testing.assert_allclose(A, pairwise_A, rtol=1e-13, atol=1e-18)
    np.testing.assert_allclose(floor, panel["known_floor"], rtol=1e-13, atol=1e-15)
    require(np.all(targets >= floor), "A saved target is below the observable floor")
    # Separate label-orientation audit. Future close is used ONLY to verify
    # 607 already frozen labels; it is NEVER an argument to the floor/projection.
    future_close = raw[1:, -1, 3]
    eight_closes = np.column_stack([raw[:-1, :, 3], future_close])
    reconstructed = np.std(np.diff(np.log(eight_closes), axis=1), axis=1, ddof=0)
    np.testing.assert_allclose(reconstructed, targets[:-1], rtol=1e-12, atol=1e-15)
    future_return = np.log(future_close) - np.log(raw[:-1, -1, 3])
    decomposed = A[:-1] + (6.0 / 49.0) * (future_return - r[:-1].mean(axis=1)) ** 2
    np.testing.assert_allclose(reconstructed ** 2, decomposed, rtol=1e-12, atol=1e-18)

    per_predictor, per_model, numerical_checks = [], [], []
    for model in panel["model_order"]:
        predictions = panel["predictions"][model]
        projected = project(predictions, floor)
        numerical_checks.append(audit_loss_monotonicity(targets, predictions, projected, floor))
        rows = [predictor_metrics(model, seed, predictions[i], projected[i], targets, floor)
                for i, seed in enumerate(panel["seeds"][model])]
        per_predictor.extend(rows)
        shortfall = np.maximum(floor - predictions, 0.0)
        violation = shortfall > VIOLATION_TOLERANCE_SIGMA7
        rmse_before = float(np.mean([row["RMSE_before"] for row in rows]))
        rmse_after = float(np.mean([row["RMSE_projected"] for row in rows]))
        per_model.append({"model": model, "predictors": len(rows), "dates_per_predictor": 608,
            "predictor_date_pairs": predictions.size,
            "strict_below_floor_count": int(np.count_nonzero(shortfall)),
            "below_floor_count_tolerance": int(violation.sum()),
            "below_floor_rate_tolerance": float(violation.mean()),
            "dates_any_predictor_violates": int(violation.any(axis=0).sum()),
            "dates_all_predictors_violate": int(violation.all(axis=0).sum()),
            "shortfall_mean_all_sigma7": float(shortfall.mean()),
            "shortfall_max_sigma7": float(shortfall.max()),
            "RMSE_mean_before": rmse_before, "RMSE_mean_projected": rmse_after,
            "RMSE_mean_reduction": rmse_before - rmse_after,
            "RMSE_mean_reduction_percent": 100 * (1 - rmse_after / rmse_before),
            "MAE_mean_before": float(np.mean([row["MAE_before"] for row in rows])),
            "MAE_mean_projected": float(np.mean([row["MAE_projected"] for row in rows])),
            "RMSE_seed_SD_before": float(np.std([row["RMSE_before"] for row in rows], ddof=1)) if len(rows) > 1 else None,
            "RMSE_seed_SD_projected": float(np.std([row["RMSE_projected"] for row in rows], ddof=1)) if len(rows) > 1 else None})
    require(len(per_predictor) == 143 and len(per_model) == 17, "Incomplete predictor/model family")
    by_model = {row["model"]: row for row in per_model}
    e3 = by_model[loader.E3_MODEL]
    for key, field in (("rank_before", "RMSE_mean_before"), ("rank_projected", "RMSE_mean_projected")):
        for rank, row in enumerate(sorted(per_model, key=lambda row: row[field]), 1):
            row[key] = rank
    for row in per_model:
        gap_before = row["RMSE_mean_before"] - e3["RMSE_mean_before"]
        gap_after = row["RMSE_mean_projected"] - e3["RMSE_mean_projected"]
        row.update({"RMSE_gap_vs_E3_before": gap_before, "RMSE_gap_vs_E3_projected": gap_after,
                    "descriptive_fraction_of_RMSE_gap_removed":
                        (gap_before - gap_after) / gap_before if gap_before > 0 else None})
    rivals = [row for row in per_model if row["model"] != loader.E3_MODEL]
    best_before = min(rivals, key=lambda row: row["RMSE_mean_before"])
    best_after = min(rivals, key=lambda row: row["RMSE_mean_projected"])
    raw_gap = best_before["RMSE_mean_before"] - e3["RMSE_mean_before"]
    projected_gap = best_after["RMSE_mean_projected"] - e3["RMSE_mean_projected"]
    summary = {"schema_version": 1,
        "scope": "exploratory_post_hoc_physical_projection_of_frozen_predictions",
        "formula": {"A": "sum_{i=t-6}^{t-1}((r_i-mean6)^2)/7; r_i=log(C_i/C_{i-1}); seven past closes C_{t-7},...,C_{t-1}",
                    "floor_sigma7": "sqrt(A)", "projection": "max(saved_prediction_sigma7, sqrt(A))",
                    "target_variance_identity": "sigma7^2=A+(6/49)*(r_t-mean6)^2"},
        "units": "daily unannualized population sigma7 of seven Close log returns; h=1",
        "metric_estimand": "arithmetic mean of individual predictor RMSEs, not ensemble RMSE",
        "model_count": 17, "predictor_count": 143, "dates_per_predictor": 608,
        "predictor_date_pairs": 143 * 608, "rows_dropped": 0,
        "violation_tolerance_sigma7": VIOLATION_TOLERANCE_SIGMA7,
        "projection_uses_no_tolerance": True,
        "best_rival_before": best_before["model"], "best_rival_after_projection": best_after["model"],
        "best_rival_RMSE_before": best_before["RMSE_mean_before"],
        "best_rival_RMSE_projected": best_after["RMSE_mean_projected"],
        "E3_RMSE_before": e3["RMSE_mean_before"], "E3_RMSE_projected": e3["RMSE_mean_projected"],
        "E3_MAE_before": e3["MAE_mean_before"], "E3_MAE_projected": e3["MAE_mean_projected"],
        "E3_relative_advantage_percent_before": 100 * raw_gap / best_before["RMSE_mean_before"],
        "E3_relative_advantage_percent_after_projection": 100 * projected_gap / best_after["RMSE_mean_projected"],
        "best_available_rival_gap_before": raw_gap, "best_available_rival_gap_after_projection": projected_gap,
        "descriptive_fraction_best_available_gap_removed": (raw_gap - projected_gap) / raw_gap,
        "floor_only_control_RMSE": float(np.sqrt(np.mean((floor - targets) ** 2))),
        "floor_only_control_MAE": float(np.mean(np.abs(floor - targets))),
        "floor_only_MAE_reduction_percent_vs_E3": 100 * (1 - float(np.mean(np.abs(floor - targets))) / e3["MAE_mean_before"]),
        "floor_only_control_is_separate_from_17_system_family": True,
        "all_models_pointwise_squared_and_absolute_loss_nonincreasing": True,
        "maximum_pointwise_squared_loss_increase": max(x["maximum_pointwise_squared_loss_increase"] for x in numerical_checks),
        "maximum_pointwise_absolute_loss_increase": max(x["maximum_pointwise_absolute_loss_increase"] for x in numerical_checks),
        "operations": {"new_model_fits": 0, "new_model_forwards": 0, "new_calibration_fits": 0,
                       "hypothesis_tests": 0, "new_datasets": 0, "saved_prediction_projections": 143 * 608},
        "claims": {"preregistered": False, "confirmatory": False, "R1_certified": False,
                   "R2_certified": False, "R3_certified": False, "causal_architecture_attribution": False,
                   "original_48_contrasts_replaced": False, "original_catalogue_modified": False}}
    admission = {"prediction_panel": panel["provenance"],
        "prediction_loader_source": {"path": str(loader_path.relative_to(ROOT)), "sha256": sha(loader_path)},
        "raw_cohort": {"commit": "af3f1223ad5899ed63016d2af2d09bae8470ab00",
            "path": "book_e2/_static/e2_scientific_closure_20261116/data_audit/current_cohort_arrays.npz",
            "sha256": NPZ_SHA256, "bytes": Path(npz_path).stat().st_size, "members_read": members},
        "scientific_formula_source": {"commit": "73d8958c50a69fbea46deb828f03f89385318789",
            "path": "src/e3_innovation_model.py", "sha256": sha(ROOT / "src/e3_innovation_model.py"),
            "function": "load_innovation_data", "past_returns": 6, "variance_denominator": 7},
        "dates": {"start": dates[0], "end": dates[-1], "n": 608,
            "input_support": "t-7 through t-1; no realized r_t argument in floor or projection",
            "consecutive_target_dates_verified": True, "adjacent_six_bar_overlap_verified": True},
        "minimum_target_minus_floor_sigma7": float(np.min(targets - floor)),
        "targets_below_floor_count": int(np.count_nonzero(targets < floor)),
        "maximum_abs_difference_from_E3_saved_floor": float(np.max(np.abs(floor - panel["known_floor"]))),
        "maximum_abs_difference_pairwise_variance_identity": float(np.max(np.abs(A - pairwise_A))),
        "label_reconstruction": {"dates": 607, "maximum_absolute_difference_sigma7": float(np.max(np.abs(reconstructed - targets[:-1]))),
            "last_date_unavailable_next_window": dates[-1],
            "future_close_used_only_for_frozen_label_audit_never_floor_or_projection": True,
            "all_608_targets_exactly_equal_pinned_cohort": True},
        "remaining_limit": "Chronology inherits the pinned raw-cohort audit's t-7..t-1 contract; checks do not independently re-download exchange data or close its three partial controls."}
    floor_rows = [{**support, "target_sigma7": float(targets[i]), "observable_A_variance": float(A[i]),
                   "observable_floor_sigma7": float(floor[i]), "target_minus_floor_sigma7": float(targets[i] - floor[i])}
                  for i, support in enumerate(supports)]
    write_csv(output_dir / "per_predictor.csv", per_predictor)
    write_csv(output_dir / "per_model.csv", per_model)
    write_csv(output_dir / "floor_by_date.csv", floor_rows)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "sources_and_admission.json", admission)
    write_json(output_dir / "execution.json", {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "numpy": np.__version__, "command": sys.argv,
        "data_dir": str(Path(data_dir).resolve()), "npz_path": str(Path(npz_path).resolve()),
        "script_sha256": sha(__file__), "model_training_imported": False})
    manifest = {path.name: {"bytes": path.stat().st_size, "sha256": sha(path)}
                for path in sorted(output_dir.iterdir()) if path.is_file() and path.name != "manifest.json"}
    write_json(output_dir / "manifest.json", {"files": manifest, "inputs_are_not_redistributed_here": True})
    print(json.dumps({key: summary[key] for key in ("model_count", "predictor_count", "best_rival_before",
        "best_rival_after_projection", "E3_relative_advantage_percent_before", "E3_relative_advantage_percent_after_projection")}, ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cohort-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    run(args.data_dir, args.cohort_npz, args.output_dir)


if __name__ == "__main__":
    main()
