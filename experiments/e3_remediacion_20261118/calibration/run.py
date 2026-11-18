"""Retrospective interval diagnostic for ten unchanged E3 predictors.

This explicitly opens calibration members forbidden to the ORIGINAL campaign.
It is a new posterior analysis, not an amendment, confirmatory preregistration,
retraining, model selection, serving update, or independent human reproduction.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SEEDS = tuple(range(271801, 271811))
SOURCE_HASHES = {
    "src/original_model.py": "dce48ff772cd284c63a672c599febc66dfc6eb14f286f523703fe6b01d086c2f",
    "src/e3_development.py": "33dd121f0dd36ef1933c5f4338045fa5a17e85ad6986dfd2c1a43c2662158ebe",
    "src/e3_innovation_model.py": "7f362d38ca83ed302dc9c822b4e4e19a0d446e4c01477f42b0e9a8da823ae7b1",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def write_csv(path, rows):
    require(bool(rows), "Empty output table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def finite_sample_quantile(residuals, alpha):
    """Order statistic, not interpolated np.quantile; only cal residuals enter."""
    values = [float(v) for v in residuals]
    require(values and all(math.isfinite(v) and v >= 0 for v in values), "Invalid calibration residuals")
    require(math.isfinite(alpha) and 0 < alpha < 1, "Invalid alpha")
    rank = math.ceil((len(values) + 1) * (1 - alpha))
    require(rank <= len(values), "Insufficient calibration size for finite rank at this alpha")
    return sorted(values)[rank - 1], rank


def fit_quantile(calibration_targets, calibration_predictions, alpha):
    require(len(calibration_targets) == len(calibration_predictions), "Calibration alignment differs")
    return finite_sample_quantile([abs(float(y) - float(p)) for y, p in
                                  zip(calibration_targets, calibration_predictions, strict=True)], alpha)


def interval_values(targets, predictions, q, alpha):
    """Interval metrics from saved point forecasts; no quantile fitting here."""
    require(math.isfinite(q) and q >= 0 and 0 < alpha < 1, "Invalid interval parameters")
    require(len(targets) == len(predictions) and len(targets) > 0, "Evaluation alignment differs")
    result = []
    for y, p in zip(targets, predictions, strict=True):
        y, p = float(y), float(p)
        require(math.isfinite(y) and math.isfinite(p) and y >= 0 and p >= 0, "Invalid interval inputs")
        lower, upper = max(0.0, p - q), p + q
        score = upper - lower + (2 / alpha) * max(lower - y, 0.0) + (2 / alpha) * max(y - upper, 0.0)
        result.append({"lower": lower, "upper": upper, "covered": lower <= y <= upper,
                       "width": upper - lower, "interval_score": score,
                       "lower_miss": y < lower, "upper_miss": y > upper})
    return result


def metric_summary(intervals):
    return {"n": len(intervals),
            "coverage_percent": 100 * statistics.fmean(float(r["covered"]) for r in intervals),
            "mean_width_sigma7": statistics.fmean(r["width"] for r in intervals),
            "mean_interval_score_alpha_0_1": statistics.fmean(r["interval_score"] for r in intervals),
            "lower_miss_percent": 100 * statistics.fmean(float(r["lower_miss"]) for r in intervals),
            "upper_miss_percent": 100 * statistics.fmean(float(r["upper_miss"]) for r in intervals)}


def validate_dates(fit_dates, validation_dates, calibration_dates, evaluation_dates):
    """Validate chronology, distinguishing known-input overlap from leakage."""
    groups = [list(x) for x in (fit_dates, validation_dates, calibration_dates, evaluation_dates)]
    for group in groups:
        require(group == sorted(set(group)), "Partition dates must be sorted and unique")
        require(all(date.fromisoformat(x).isoformat() == x for x in group), "Invalid ISO date")
    fit, validation, cal, evaluation = groups
    require(len(fit) == 532 and fit[-1] == "2024-06-30", "Original fit cohort differs")
    require(len(validation) == 116 and validation[0] == "2024-07-08" and validation[-1] == "2024-10-31", "Validation differs")
    require(len(cal) == 53 and cal[0] == "2024-11-08" and cal[-1] == "2024-12-30", "Calibration differs")
    require(len(evaluation) == 608 and evaluation[0] == "2025-01-01" and evaluation[-1] == "2026-08-31", "Evaluation differs")
    require(all(set(a).isdisjoint(b) for i, a in enumerate(groups) for b in groups[i + 1:]), "Partition target dates overlap")
    require(date.fromisoformat(cal[0]) - timedelta(days=7) > date.fromisoformat(validation[-1]), "Calibration inputs overlap fitting/selection target era")
    require(cal[-1] < evaluation[0], "Calibration does not precede evaluation")
    return {"target_dates_disjoint": True, "calibration_after_fit_and_validation": True,
            "calibration_latest_target_before_first_evaluation": True,
            "known_input_support_overlap_calibration_evaluation": True,
            "overlap_note": "The first evaluation input includes Dec25–31; calibration targets run through Dec30. Those values are known at the Jan1 forecast origin, but overlap/dependence precludes an iid coverage claim."}


def input_records():
    manifest = json.loads((HERE / "input_manifest.json").read_bytes())
    require(manifest["repository"] == "jassan-bit/DeepLearning_202630", "Unexpected source repository")
    rows = manifest["files"]
    require(len(rows) == 15 and len({r["name"] for r in rows}) == 15, "Input manifest differs")
    return rows


def admit_inputs(folder, download=False):
    folder = Path(folder)
    for record in input_records():
        relative = Path(record["name"])
        require(not relative.is_absolute() and ".." not in relative.parts, "Invalid input cache path")
        target = folder / relative
        if not target.exists():
            require(download, "Missing immutable input: " + record["name"] + "; pass --download explicitly")
            url = "https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/" + record["commit"] + "/" + record["path"]
            with urlopen(Request(url, headers={"User-Agent": "E3RetrospectiveCalibration/1"}), timeout=60) as response:
                raw = response.read(record["bytes"] + 1)
            require(len(raw) == record["bytes"] and hashlib.sha256(raw).hexdigest() == record["sha256"], "Downloaded input differs: " + record["name"])
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(raw)
        require(target.is_file() and not target.is_symlink() and target.stat().st_size == record["bytes"]
                and sha(target) == record["sha256"], "Immutable input differs: " + record["name"])
    return folder


def rmse(targets, predictions):
    return math.sqrt(statistics.fmean((float(p) - float(y)) ** 2 for y, p in zip(targets, predictions, strict=True)))


def run(args):
    started = perf_counter()
    import numpy as np
    import torch
    folder = admit_inputs(args.input_dir, args.download)
    for path, expected in SOURCE_HASHES.items():
        require(sha(ROOT / path) == expected, "Scientific model source changed: " + path)
    sys.path.insert(0, str(ROOT))
    from src.e3_innovation_model import InnovationModel, load_innovation_data
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    plan = json.loads((HERE / "analysis_plan.json").read_bytes())
    alpha = plan["calibration"]["alpha"]
    admission = json.loads((folder / "admission.json").read_bytes())
    original_plan = json.loads((folder / "plan.json").read_bytes())
    summary = json.loads((folder / "summary.json").read_bytes())
    require(original_plan["data"]["calibration_members_allowed"] is False
            and summary["calibration_members_opened"] is False, "Original campaign scope changed")
    original_plan_fingerprint = hashlib.sha256(json.dumps(original_plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(original_plan_fingerprint == summary["plan_sha256"], "Original plan binding differs")
    train = load_innovation_data(folder / "cohort.npz", admission["npz_sha256"])
    scaler = admission["scaler"]
    for key in ("mean", "std", "u_scale", "y_scale"):
        require(np.array_equal(np.asarray(train["scaler"][key]), np.asarray(scaler[key])), "Original fit-only scaler differs: " + key)
    with np.load(folder / "cohort.npz", allow_pickle=False) as saved:
        cal_raw = np.asarray(saved["calibration_raw"], dtype=np.float64).copy()
        cal_y = np.asarray(saved["calibration_y_original"], dtype=np.float64).copy()
        cal_dates = [str(x) for x in saved["calibration_dates"]]
        y = np.asarray(saved["evaluation_y_original"], dtype=np.float64).copy()
        dates = [str(x) for x in saved["evaluation_dates"]]
    chronology = validate_dates(admission["fit_dates"], admission["validation_dates"], cal_dates, dates)
    require(cal_raw.shape == (53, 7, 5) and cal_y.shape == (53,) and y.shape == (608,), "Unexpected cohort shapes")
    require(np.isfinite(cal_raw).all() and np.isfinite(cal_y).all() and np.isfinite(y).all()
            and (cal_raw[..., :4] > 0).all() and (cal_raw[..., 4] >= 0).all() and (cal_y >= 0).all() and (y >= 0).all(), "Invalid cohort values")
    relative = cal_raw.copy()
    relative[..., :4] = np.log(cal_raw[..., :4]) - np.log(cal_raw[:, -1, 3])[:, None, None]
    relative[..., 4] = np.log1p(cal_raw[..., 4])
    X = ((relative - np.asarray(scaler["mean"])) / np.asarray(scaler["std"])).astype(np.float32)
    returns = np.diff(np.log(cal_raw[..., 3]), axis=1)
    A = np.sum((returns - returns.mean(axis=1)[:, None]) ** 2, axis=1) / 7.0
    require(np.all(cal_y ** 2 >= A - 1e-12), "Calibration target violates observable variance floor")
    require(np.isfinite(X).all(), "Nonfinite transformed calibration inputs")
    with (folder / "evaluation_predictions.csv").open(newline="", encoding="utf-8") as stream:
        frozen = list(csv.DictReader(stream))
    require(len(frozen) == 6080 and {int(row["seed"]) for row in frozen} == set(SEEDS), "Frozen predictor coordinates differ")
    checkpoint_records = {r["seed"]: r for r in input_records() if "seed" in r}
    per_seed, calibration_rows, group_rows, all_intervals = [], [], [], []
    q25, q75 = (float(x) for x in np.quantile(train["train_y_original"], [.25, .75]))
    for seed in SEEDS:
        checkpoint_path = folder / checkpoint_records[seed]["name"]
        # admit_inputs verified the complete bytes before this safe-state load.
        cp = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        require(cp["seed"] == seed and cp["complete"] and cp["plan_sha256"] == summary["plan_sha256"], "Checkpoint campaign binding differs")
        require(checkpoint_records[seed]["sha256"] == summary["checkpoints_acknowledged_sha256"][str(seed)], "Checkpoint acknowledgment differs")
        require(cp["best_epoch"] == summary["seeds"][str(seed)]["best_epoch"], "Validation-selected checkpoint differs")
        require(all(cp["source_hashes"][path] == expected for path, expected in SOURCE_HASHES.items()), "Checkpoint source hashes differ")
        model = InnovationModel("additive_positive_head", u_scale=scaler["u_scale"], y_scale=scaler["y_scale"], seed=seed).cpu().eval()
        model.load_state_dict(cp["best_model_state"], strict=True)
        require(sum(p.numel() for p in model.parameters()) == 1210, "Wrong scientific architecture")
        require(float(model.y_scale) == scaler["y_scale"] and float(model.u_scale) == scaler["u_scale"], "Checkpoint scaler buffers differ")
        # One forward per calibration window, in the frozen original precision.
        with torch.inference_mode():
            cal_prediction = np.asarray([float(model(torch.from_numpy(x[None]), torch.as_tensor([a], dtype=torch.float64))[0])
                                         for x, a in zip(X, A, strict=True)], dtype=np.float64)
        require(np.isfinite(cal_prediction).all() and np.all(cal_prediction >= np.sqrt(A)), "Invalid frozen calibration forecasts")
        q, rank = fit_quantile(cal_y, cal_prediction, alpha)
        require(rank == 49, "Finite-sample calibration order differs")
        rows = [row for row in frozen if int(row["seed"]) == seed]
        require([row["date"] for row in rows] == dates, "Saved evaluation order differs")
        require(np.array_equal([float(row["target_sigma7"]) for row in rows], y), "Saved evaluation targets differ")
        prediction = np.asarray([float(row["prediction_sigma7"]) for row in rows])
        original_metric = next(row for row in summary["per_seed_metrics"] if row["seed"] == seed)
        point_rmse = rmse(y, prediction)
        require(abs(point_rmse - original_metric["RMSE"]) <= 1e-12, "Saved point metric changed")
        intervals = interval_values(y, prediction, q, alpha)
        all_intervals.append(intervals)
        cal_intervals = interval_values(cal_y, cal_prediction, q, alpha)
        per_seed.append({"seed": seed, "checkpoint_sha256": checkpoint_records[seed]["sha256"], "best_epoch": cp["best_epoch"],
                         "calibration_n": 53, "calibration_order_rank": rank, "q_sigma7": q,
                         "calibration_empirical_coverage_percent": metric_summary(cal_intervals)["coverage_percent"],
                         "unchanged_point_RMSE": point_rmse, **metric_summary(intervals)})
        calibration_rows += [{"seed": seed, "target_date": day, "target_sigma7": float(target), "frozen_CPU_prediction_sigma7": float(p),
                              "absolute_residual_sigma7": abs(float(target) - float(p)), "observable_floor_sigma7": math.sqrt(float(a))}
                             for day, target, p, a in zip(cal_dates, cal_y, cal_prediction, A, strict=True)]
        groups = {}
        for i, (day, target) in enumerate(zip(dates, y, strict=True)):
            quarter = day[:4] + "Q" + str((int(day[5:7]) - 1) // 3 + 1)
            level = "low_fit_q25" if target <= q25 else ("middle_fit_q25_q75" if target <= q75 else "high_fit_q75")
            groups.setdefault(("calendar_quarter", quarter), []).append(intervals[i])
            groups.setdefault(("realized_target_level_posthoc", level), []).append(intervals[i])
        group_rows += [{"seed": seed, "group_type": typ, "group": name, **metric_summary(values)}
                       for (typ, name), values in sorted(groups.items())]
        del model, cp
    # Calendar uncertainty at fixed fitted predictors and fixed q; paired days
    # are resampled together across the ten seeds, never treated as 6080 iid.
    bootstrap = plan["uncertainty"]
    rng = np.random.default_rng(bootstrap["random_seed"])
    repetitions, block = bootstrap["resamples"], bootstrap["block_days"]
    counts = math.ceil(608 / block)
    indices = (rng.integers(0, 608, size=(repetitions, counts, 1)) + np.arange(block)[None, None, :]) % 608
    indices = indices.reshape(repetitions, -1)[:, :608]
    ci = {}
    for field, factor in (("covered", 100), ("width", 1), ("interval_score", 1)):
        daily_mean = np.asarray([[float(row[field]) for row in seed] for seed in all_intervals]).mean(axis=0)
        estimates = factor * daily_mean[indices].mean(axis=1)
        ci[field] = {"estimate": factor * float(daily_mean.mean()), "marginal_calendar_bootstrap_CI95": np.quantile(estimates, [.025, .975]).tolist()}
    describe = lambda key: {"mean_over_10_seeds": statistics.fmean(row[key] for row in per_seed),
                            "sample_SD_between_seeds": statistics.stdev(row[key] for row in per_seed),
                            "min": min(row[key] for row in per_seed), "max": max(row[key] for row in per_seed)}
    result = {
        "schema_version": 1, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": plan["scope"], "analysis_plan_sha256": sha(HERE / "analysis_plan.json"), "script_sha256": sha(__file__),
        "input_manifest_sha256": sha(HERE / "input_manifest.json"), "source_model_sha256": SOURCE_HASHES,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "device": "cpu", "torch_threads": 1},
        "original_campaign": {"plan_sha256": summary["plan_sha256"], "calibration_members_allowed": False, "calibration_members_opened": False,
                              "amended": False, "point_prediction_file_sha256": sha(folder / "evaluation_predictions.csv")},
        "new_diagnostic": {"calibration_members_opened": True, "new_conformal_quantiles": 10, "calibration_forward_windows": 530,
                           "calibration_forward_calls": 530, "evaluation_forward_calls": 0, "model_fit_steps": 0,
                           "scaler_updates": 0, "original_calibrator_updates": 0, "serving_updates": 0,
                           "test_predictions_changed": False, "best_seed_selection": False},
        "calibration": {"n": 53, "first": cal_dates[0], "last": cal_dates[-1], "alpha": alpha, "rank": 49,
                        "same_q_for_each_predictor": False, "q_depends_on_evaluation_targets": False},
        "evaluation": {"n_dates": 608, "seeds": list(SEEDS), "first": dates[0], "last": dates[-1]},
        "chronology": chronology,
        "original_fit_target_quartiles": {"q25": q25, "q75": q75, "source_count": 532},
        "per_seed": per_seed,
        "summary": {key: describe(key) for key in ("q_sigma7", "coverage_percent", "mean_width_sigma7", "mean_interval_score_alpha_0_1", "unchanged_point_RMSE")},
        "conditional_calendar_uncertainty": {"method": bootstrap, "metrics": ci},
        "elapsed_seconds_including_admission_and_forward": perf_counter() - started,
        "guaranteed_90_percent_coverage_claimed": False, "R1_R2_R3_certified": False, "human_independent_reproduction": False,
        "deployed_E3_interval_unchanged_and_uncalibrated": True,
        "limits": [plan["dependence_limit"], "This explicitly post hoc analysis is not the forbidden calibration hidden inside the original campaign.",
                   "The public historical evaluation calendar was already known. Threshold choices are not a prospective claim.",
                   "Calendar bootstrap intervals condition on the fitted models and q; they omit calibration-quantile uncertainty and are not simultaneous.",
                   "Realized-target-level groups are retrospective diagnostics and are not available for forecast routing.",
                   "No dashboard or public deployment was changed or validated by this diagnostic."]}
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "per_seed_intervals.csv", per_seed)
    write_csv(destination / "calibration_predictions.csv", calibration_rows)
    write_csv(destination / "subgroup_diagnostics.csv", group_rows)
    write_json(destination / "results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "work/e3_remediation_calibration_inputs")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "work/e3_retrospective_calibration")
    arguments = parser.parse_args()
    result = run(arguments)
    print(json.dumps({"scope": result["scope"], "summary": result["summary"], "new_diagnostic": result["new_diagnostic"]}, indent=2))
