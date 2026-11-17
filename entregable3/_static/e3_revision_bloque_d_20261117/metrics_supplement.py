"""Posterior descriptive metrics from immutable saved forecasts, with no model use.

The existing input validator checks eleven frozen sources and all 143 forecast
trajectories. This supplement adds sMAPE/R² and reconstructs E2's inherited
intervals from its frozen q; it never calibrates E3. Bootstrap intervals concern
optimization-seed means on this same calendar, not temporal/predictive coverage.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import platform
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / "experiments/e3_estadistica_posterior_20261117"
FROZEN_METRICS = ROOT / "book_e3/_static/e3_estadistica_posterior_20261117/metrics.csv"
CI_SEED, CI_RESAMPLES = 20261117, 10000
VERSION = "e3-posterior-descriptive-metrics-v1"
SOURCE_COMMIT = "8fccb216f36a3e4fc5cb38ecdec769b7f5379397"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validator():
    spec = importlib.util.spec_from_file_location("e3_frozen_metric_inputs", PRIOR / "inputs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def vector_metrics(targets, prediction):
    require(len(targets) == len(prediction) and len(targets) > 0, "Metric supports differ or are empty")
    require(all(math.isfinite(float(v)) for v in (*targets, *prediction)), "Metrics require finite values")
    errors = [float(p) - float(y) for y, p in zip(targets, prediction, strict=True)]
    squared = statistics.fmean(e * e for e in errors)
    observed_mean = statistics.fmean(float(y) for y in targets)
    total = math.fsum((float(y) - observed_mean) ** 2 for y in targets)
    smape = statistics.fmean(
        0.0 if abs(float(y)) + abs(float(p)) == 0 else
        200.0 * abs(float(y) - float(p)) / (abs(float(y)) + abs(float(p)))
        for y, p in zip(targets, prediction, strict=True)
    )
    return {"RMSE": math.sqrt(squared), "MAE": statistics.fmean(abs(e) for e in errors),
            "sMAPE_percent": smape,
            "R2": None if total == 0 else 1.0 - math.fsum(e * e for e in errors) / total}


def frozen_interval_metrics(targets, prediction, q):
    require(len(targets) == len(prediction) and len(targets) > 0, "Interval supports differ or are empty")
    require(type(q) in (int, float) and math.isfinite(q) and q >= 0, "Frozen interval q must be finite and nonnegative")
    require(all(math.isfinite(float(v)) for v in (*targets, *prediction)), "Intervals require finite values")
    lower = [max(0.0, float(p) - q) for p in prediction]
    upper = [float(p) + q for p in prediction]
    require(all(lo <= hi for lo, hi in zip(lower, upper, strict=True)), "Invalid frozen interval endpoints")
    return {"coverage90_percent": 100.0 * statistics.fmean(
                lo <= float(y) <= hi for lo, hi, y in zip(lower, upper, targets, strict=True)),
            "mean_interval_width90": statistics.fmean(hi - lo for lo, hi in zip(lower, upper, strict=True))}


def summarize(per_predictor):
    require(len(per_predictor) > 0, "No predictor metrics")
    values = [r["RMSE"] for r in per_predictor]
    return {"R": len(values), "RMSE_mean": statistics.fmean(values),
            "RMSE_sample_SD": statistics.stdev(values) if len(values) > 1 else None,
            "MAE_mean": statistics.fmean(r["MAE"] for r in per_predictor),
            "sMAPE_percent": statistics.fmean(r["sMAPE_percent"] for r in per_predictor),
            "R2_mean": None if any(r["R2"] is None for r in per_predictor) else
                statistics.fmean(r["R2"] for r in per_predictor)}


def seed_bootstrap(values, indices):
    """A fixed shared seed-index sample; a deterministic trajectory has no seed CI."""
    if len(values) == 1:
        return None
    values = np.asarray(values, dtype=np.float64)
    indices = np.asarray(indices)
    require(values.ndim == 1 and len(values) > 1 and np.isfinite(values).all(), "Invalid seed metrics")
    require(indices.ndim == 2 and indices.shape[1] == len(values) and
            np.issubdtype(indices.dtype, np.integer) and (indices >= 0).all() and
            (indices < len(values)).all(), "Invalid bootstrap seed indices")
    means = values[indices].mean(axis=1)
    return np.quantile(means, [0.025, 0.975], method="linear").tolist()


def csv_bytes(rows, fields):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fields, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def calculate(data_dir):
    lib = validator()
    documents = lib.read_documents(data_dir)
    panel = lib.validate_documents(documents)
    altered_dates = deepcopy(documents)
    altered_dates["e3_predictions"][0]["date"] = "2024-12-31"
    try:
        lib.validate_documents(altered_dates)
    except ValueError as error:
        require("E3 date" in str(error), "Date negative-control failed for an unrelated reason")
    else:
        raise ValueError("Misaligned E3 date was accepted")
    y = panel["targets"].tolist()
    original = documents["combined"]
    prior_metrics = list(csv.DictReader(io.StringIO(FROZEN_METRICS.read_bytes().decode())))
    require({r["model"] for r in prior_metrics} == set(panel["model_order"]), "Prior metric family differs")
    prior_by_model = {r["model"]: r for r in prior_metrics}
    by_coordinate = {(r["model"], r["seed"]): r for r in original["predictors"]}
    expected_cal_dates = original["predictors"][0]["calibration_dates"]
    expected_cal_targets = original["predictors"][0]["calibration_targets"]
    require(len(expected_cal_dates) == 53 and len(set(expected_cal_dates)) == 53 and
            expected_cal_dates[0] == "2024-11-08" and expected_cal_dates[-1] == "2024-12-30",
            "Inherited calibration calendar differs")
    rows, predictor_rows, uncertainty = [], [], []
    indices = np.random.default_rng(CI_SEED).choice(10, size=(CI_RESAMPLES, 10), replace=True)
    maximum_difference = {"RMSE_mean": 0.0, "MAE_mean": 0.0, "RMSE_sample_SD": 0.0}
    for model in panel["model_order"]:
        individuals, interval_values = [], []
        for seed, prediction in zip(panel["seeds"][model], panel["predictions"][model], strict=True):
            metric = vector_metrics(y, prediction.tolist())
            require(metric["R2"] is not None, "Real cohort R2 is undefined")
            individuals.append(metric)
            interval = {"coverage90_percent": None, "mean_interval_width90": None}
            if model != lib.E3_MODEL:
                inherited = by_coordinate[model, seed]
                require(inherited["calibration_dates"] == expected_cal_dates and
                        inherited["calibration_targets"] == expected_cal_targets and
                        len(inherited["calibration_predictions"]) == 53,
                        "A predictor's inherited calibration support differs")
                interval = frozen_interval_metrics(y, prediction.tolist(), inherited["calibrator_q"])
                interval_values.append(interval)
            predictor_rows.append({"model": model, "seed": seed, **metric, **interval})
        summary = summarize(individuals)
        previous = prior_by_model[model]
        require(int(previous["R"]) == summary["R"], "Prior seed count differs")
        for field in maximum_difference:
            old = previous[field]
            value = summary[field]
            if value is None:
                require(old == "", "Classical seed SD must be absent")
            else:
                difference = abs(value - float(old))
                maximum_difference[field] = max(maximum_difference[field], difference)
                require(difference <= 1e-14, "Supplement changed a frozen metric: " + model + "/" + field)
        rows.append({"model": model, "R": summary.pop("R"), "calendar_n": len(y), **summary,
                     "coverage90_percent": statistics.fmean(r["coverage90_percent"] for r in interval_values) if interval_values else None,
                     "mean_interval_width90": statistics.fmean(r["mean_interval_width90"] for r in interval_values) if interval_values else None,
                     "interval_status": "reconstructed_from_frozen_E2_q" if interval_values else "not_calibrated_no_frozen_interval"})
        ci = seed_bootstrap([r["RMSE"] for r in individuals], indices)
        uncertainty.append({"model": model, "R": len(individuals), "RMSE_mean": summary["RMSE_mean"],
                            "RMSE_sample_SD": summary["RMSE_sample_SD"],
                            "RMSE_seed_bootstrap_CI95_low": None if ci is None else ci[0],
                            "RMSE_seed_bootstrap_CI95_high": None if ci is None else ci[1],
                            "scope": "optimization_seed_mean_conditional_on_known_calendar" if ci is not None else "undefined_single_deterministic_trajectory"})
    # Independent historical six-seed table controls validate inherited formulas;
    # they do not supply the current ten-seed results.
    historical_differences = {"RMSE": 0.0, "MAE": 0.0, "R2_mean": 0.0,
                              "sMAPE_percent": 0.0, "coverage90_percent": 0.0, "mean_interval_width90": 0.0}
    for historical in original["original_analysis"]["table_A2"]:
        model = historical["model"]
        seeds = lib.OLD_SEEDS if model in lib.NEURAL else ("deterministic",)
        old_predictors = [by_coordinate[model, seed] for seed in seeds]
        old_metrics = [vector_metrics(y, r["prediction_original"]) for r in old_predictors]
        old_intervals = [frozen_interval_metrics(y, r["prediction_original"], r["calibrator_q"]) for r in old_predictors]
        check = {"RMSE": statistics.fmean(r["RMSE"] for r in old_metrics),
                 "MAE": statistics.fmean(r["MAE"] for r in old_metrics),
                 "R2_mean": statistics.fmean(r["R2"] for r in old_metrics),
                 "sMAPE_percent": statistics.fmean(r["sMAPE_percent"] for r in old_metrics),
                 "coverage90_percent": statistics.fmean(r["coverage90_percent"] for r in old_intervals),
                 "mean_interval_width90": statistics.fmean(r["mean_interval_width90"] for r in old_intervals)}
        reference = {**historical, "sMAPE_percent": 100.0 * historical["sMAPE"],
                     "coverage90_percent": 100.0 * historical["coverage90"]}
        for name, value in check.items():
            difference = abs(value - reference[name])
            historical_differences[name] = max(historical_differences[name], difference)
            require(difference <= 1e-12, "Historical formula control differs: " + model + "/" + name)
    e3_ci = next(r for r in uncertainty if r["model"] == lib.E3_MODEL)
    inherited_ci = documents["e3_summary"]["RMSE_bootstrap_CI95"]
    ci_difference = max(abs(e3_ci[key] - old) for key, old in zip(
        ("RMSE_seed_bootstrap_CI95_low", "RMSE_seed_bootstrap_CI95_high"), inherited_ci, strict=True))
    require(ci_difference <= 1e-14, "Shared bootstrap indices fail to reproduce E3's inherited seed CI")
    report = {
        "version": VERSION, "definition_date": "2026-11-17", "timezone": "America/Bogota",
        "source_repository_commit": SOURCE_COMMIT,
        "scope": "posterior_descriptive_supplement_on_known_calendar",
        "definitions": {
            "primary": "arithmetic mean of individual predictor RMSE, not ensemble RMSE",
            "RMSE_sample_SD": "sample SD across optimization seeds, ddof=1; absent at R=1",
            "sMAPE_percent": "200*mean(abs(y-p)/(abs(y)+abs(p))); zero/zero contributes zero",
            "R2_mean": "mean across predictors of 1-SSE/SST, same cohort target mean; undefined at SST=0",
            "E2_interval": "[max(0,p-calibrator_q),p+calibrator_q], frozen q for each predictor",
            "coverage90_percent": "100*mean over predictors of empirical datewise interval coverage",
            "mean_interval_width90": "mean over predictors and dates of upper-lower in original sigma7 units",
            "interval_nominal": 0.90,
            "E3_interval": "unavailable: no calibration q or frozen predictive interval; seed CI is not a predictive interval",
        },
        "bootstrap": {"resamples": CI_RESAMPLES, "seed": CI_SEED, "quantile_method": "linear",
                      "shared_indices_for_14_neural_systems": True,
                      "indices_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
                      "scope": "marginal optimization-seed mean conditional on this known calendar; not temporal, simultaneous or predictive",
                      "no_classical_seed_bootstrap": True,
                      "E3_inherited_CI95": inherited_ci,
                      "maximum_absolute_difference_from_inherited_E3_CI": ci_difference},
        "counts": {"systems": 17, "E2_predictors": 133, "E3_predictors": 10,
                   "common_dates": 608, "calibration_dates_E2": 53, "calibration_dates_E3": 0,
                   "neural_seed_ids": list(lib.SEEDS), "historical_seed_excluded": 20261116},
        "data_checks": {"dates_and_targets_exactly_matched": True, "rows_dropped": 0,
                        "misaligned_E3_date_negative_control_rejected": True,
                        "frozen_81_E2_predictors_preserved": True,
                        "maximum_difference_from_prior_metrics": maximum_difference,
                        "historical_six_seed_formula_control_maximum_differences": historical_differences},
        "sources": lib.SOURCES,
        "local_prior_outputs": {"metrics.csv": {"path": str(FROZEN_METRICS.relative_to(ROOT)),
                                                "sha256": digest(FROZEN_METRICS)},
                                "input_validator": {"path": str((PRIOR / "inputs.py").relative_to(ROOT)),
                                                    "sha256": digest(PRIOR / "inputs.py")}},
        "script_sha256": digest(__file__),
        "environment": {"python": platform.python_version(), "numpy": str(np.__version__)},
        "new_model_fits": 0, "new_model_inferences": 0, "new_hypothesis_tests": 0,
        "power_MDE_or_R3_certified": False,
        "limitations": ["Same known 608-date calendar; not independent markets.",
                        "E2 trained on 655 examples and calibrated on 53; E3 trained on 532 and validated on 116, with no calibration.",
                        "Inherited nominal 90% intervals do not guarantee coverage under temporal dependence or shift.",
                        "Coverage and width are reconstructed descriptions of inherited intervals, not a recalibration or export roundtrip.",
                        "Seed bootstrap quantifies initialization variation only; no deterministic replicate is invented."],
    }
    # Raw numeric tables preserve full precision; the display table is separate.
    raw_fields = ["model", "R", "calendar_n", "RMSE_mean", "RMSE_sample_SD", "MAE_mean",
                  "sMAPE_percent", "R2_mean", "coverage90_percent", "mean_interval_width90", "interval_status"]
    display = []
    for row in sorted(rows, key=lambda r: r["RMSE_mean"]):
        display.append({"Sistema": row["model"], "RMSE media": f'{row["RMSE_mean"]:.9f}',
                        "DE entre semillas": "—" if row["RMSE_sample_SD"] is None else f'{row["RMSE_sample_SD"]:.9f}',
                        "MAE": f'{row["MAE_mean"]:.9f}', "sMAPE (%)": f'{row["sMAPE_percent"]:.3f}',
                        "R² medio": f'{row["R2_mean"]:.6f}',
                        "Cobertura 90% (%)": "Pendiente" if row["coverage90_percent"] is None else f'{row["coverage90_percent"]:.3f}',
                        "Anchura media": "Pendiente" if row["mean_interval_width90"] is None else f'{row["mean_interval_width90"]:.9f}'})
    artifacts = {"metrics_extended.csv": csv_bytes(rows, raw_fields),
                 "metrics_extended_table.csv": csv_bytes(display, list(display[0])),
                 "metrics_uncertainty.csv": csv_bytes(uncertainty, list(uncertainty[0])),
                 "metrics_per_predictor.csv": csv_bytes(predictor_rows, list(predictor_rows[0]))}
    report["artifacts"] = {name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
                           for name, raw in artifacts.items()}
    return artifacts, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "work/e3_statistical_inputs")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    parser.add_argument("--mirror", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.download:
        validator().download_inputs(args.data_dir)
    artifacts, report = calculate(args.data_dir)
    if args.check:
        for name, raw in artifacts.items():
            require((args.output / name).read_bytes() == raw, "Stored supplement differs: " + name)
        saved = json.loads((args.output / "metrics_supplement.json").read_bytes())
        for key in set(report) - {"environment"}:
            require(saved[key] == report[key], "Stored supplement metadata differs: " + key)
        if args.mirror:
            for name in (*artifacts, "metrics_supplement.py", "metrics_supplement.json"):
                require((args.output / name).read_bytes() == (args.mirror / name).read_bytes(), "Mirror differs: " + name)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        for name, raw in {**artifacts, "metrics_supplement.json": (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()}.items():
            path = args.output / name
            require(not path.exists() or path.read_bytes() == raw, "Refusing to overwrite unequal supplement: " + name)
            path.write_bytes(raw)
        if args.mirror:
            args.mirror.mkdir(parents=True, exist_ok=True)
            for name in (*artifacts, "metrics_supplement.py", "metrics_supplement.json"):
                (args.mirror / name).write_bytes((args.output / name).read_bytes())
    print(json.dumps({"status": "verified" if args.check else "written", "systems": 17,
                      "predictors": 143, "E3_interval": "pending", "new_model_fits": 0,
                      "new_hypothesis_tests": 0}))


if __name__ == "__main__":
    main()
