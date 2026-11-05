"""Independently audit the saved new-seed Filipe pilot without loading models.

Reads the two established development CSVs and this pilot's retained evidence.
Checkpoints are hashed as bytes only. No training runner or analyzer is imported,
no model is deserialized, and no market archive or individual old-test file is read.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy import integrate, stats
from statsmodels.stats.power import TTestPower


ROOT = Path(__file__).resolve().parents[1]
STUDY = "e1_filipe_pilot_20261105"
OUT = ROOT / "results" / STUDY
PLAN = ROOT / "experiments" / STUDY / "protocol.json"
HISTORIES = (7, 14, 21, 28)
MODELS = ("MLP", "LSTM", "CNN1D")
FEATURES = ("open", "high", "low", "close", "volume")
DEVELOPMENT = ("results/eda_development/daily_market_development.csv",
               "results/eda_development/daily_volatility_development.csv")
FOLDS = {d: ("results/dl_r2_cv_20261104/folds.json" if d == 7 else
             f"results/filipe_multihistory_20261104/D{d}/folds.json") for d in HISTORIES}
HISTORICAL = (
    "experiments/filipe_multihistory_20261104/protocol.json",
    "experiments/filipe_multihistory_20261104/freeze.json",
    "results/filipe_multihistory_20261104/test/execution_summary.json",
    "results/filipe_multihistory_20261104/test/raw_test_audit.json",
    "results/filipe_multihistory_20261104/test/analysis/metrics_summary.csv",
)
TRAINING_CODE = {
    "scripts/run_filipe_pilot.py", "scripts/analyze_filipe_pilot.py",
    "scripts/train_filipe_multihistory.py", "scripts/train_dl_benchmark.py",
    "scripts/dl_data.py", "scripts/dl_data_v2.py", "scripts/dl_models.py",
}
POWER_QUADRATURE_FALLBACKS = 0


class Audit:
    def __init__(self):
        self.counters = Counter()
        self.hash_cache = {}

    def sha(self, path):
        path = Path(path).resolve()
        relative = path.relative_to(ROOT).as_posix()
        if ("/test/" in relative and relative not in HISTORICAL
                or path.suffix.lower() in {".zip", ".gz", ".npz", ".npy"}):
            raise AssertionError("Audit must not open raw archives or old-test individual evidence")
        stat = path.stat()
        stamp = (stat.st_size, stat.st_mtime_ns)
        if path not in self.hash_cache or self.hash_cache[path][0] != stamp:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024*1024), b""):
                    digest.update(block)
            self.hash_cache[path] = (stamp, digest.hexdigest())
        self.counters["artifact_hash_comparisons"] += 1
        return self.hash_cache[path][1]

    def close(self, actual, expected, label, *, rtol=2e-10, atol=2e-13):
        if expected is None:
            assert actual is None or bool(pd.isna(actual)), label
        else:
            aa, bb = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
            assert aa.shape == bb.shape, label + ": shape"
            assert np.allclose(aa, bb, rtol=rtol, atol=atol, equal_nan=True), label
            self.counters["numeric_values_compared"] += int(aa.size)

    def artifact_manifest(self, directory, manifest):
        directory = directory.resolve()
        for name, expected in manifest.items():
            path = (directory / name).resolve()
            assert path.is_relative_to(directory), "Unsafe retained artifact path"
            assert self.sha(path) == expected, str(path)


def read(path):
    return json.loads(Path(path).read_text())


def instant(value):
    parsed = datetime.fromisoformat(value)
    assert parsed.tzinfo is not None, "Audit timestamps must include a timezone"
    return parsed


def git_bytes(commit, name):
    return subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)


def index_sha(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def registration(audit):
    plan, context = read(PLAN), read(OUT / "code_manifest_before_fitting.json")
    execution = read(OUT / "execution_summary.json")
    receipt = read(OUT / "preregistration_git_receipt.json")
    analysis = read(OUT / "analysis/summary.json")
    digest = audit.sha(PLAN)
    assert digest == context["protocol_sha256"] == execution["protocol_sha256"] == analysis["protocol_sha256"]
    assert (OUT / "protocol.json").read_bytes() == PLAN.read_bytes()
    assert plan["histories"] == list(HISTORIES) and list(plan["models"]) == list(MODELS)
    assert plan["seeds"] == [6201, 6202, 6203, 6204, 6205]
    assert plan["target_return_days"] == 7 and plan["forecast_horizon_days"] == 1
    assert plan["variant"] == "relative_window" and plan["workers"] == 3
    assert plan["test_files_opened_allowed"] == plan["raw_archives_opened_allowed"] == 0
    source_set = set(DEVELOPMENT) | set(FOLDS.values()) | set(HISTORICAL[:2])
    assert set(plan["source_sha256"]) == source_set
    assert set(plan["code_sha256"]) == TRAINING_CODE
    for name, expected in {**plan["source_sha256"], **plan["code_sha256"]}.items():
        assert audit.sha(ROOT / name) == expected, "Registered source changed: " + name
    assert plan["code_sha256"] == context["code_sha256"] == execution["code_sha256"]
    for name, expected in plan["code_sha256"].items():
        assert hashlib.sha256(git_bytes(context["source_commit"], name)).hexdigest() == expected
    commit = receipt["preregistration_commit"]
    assert hashlib.sha256(git_bytes(commit, PLAN.relative_to(ROOT).as_posix())).hexdigest() == digest
    assert receipt["publication_completed_before_pilot_run"] is True
    git_time = int(subprocess.check_output(["git", "show", "-s", "--format=%ct", commit], cwd=ROOT, text=True))
    started = read(OUT / "run_started.json")
    assert started["protocol_sha256"] == digest
    assert instant(plan["registered_at_utc"]) <= instant(context["registered_before_first_pilot_fit_at_utc"])
    assert git_time <= instant(started["started_at_utc"]).timestamp()
    starts = list((OUT / "trajectory_start_receipts").glob("*.json"))
    assert len(starts) == 60
    expected_jobs = set(product(HISTORIES, MODELS, plan["seeds"]))
    actual_jobs = set()
    receipts = {}
    for path in starts:
        row = read(path)
        job = row["job"]
        key = (job["history_days"], job["model"], job["seed"])
        assert key not in actual_jobs and row["protocol_sha256"] == digest
        actual_jobs.add(key)
        receipts[key] = instant(row["started_at_utc"])
        assert instant(context["registered_before_first_pilot_fit_at_utc"]) < receipts[key]
        assert instant(started["started_at_utc"]) <= receipts[key]
        audit.counters["new_trajectory_start_receipts"] += 1
    assert actual_jobs == expected_jobs
    assert instant(execution["first_new_trajectory_started_at_utc"]) == min(receipts.values())
    assert execution["registration_precedes_every_new_trajectory"] is True
    assert execution["historical_D7_reused"] is False and execution["all_fits_are_new_pilot_runs"] is True
    for key, expected in {"trajectories": 60, "monthly_segments": 720, "project_fits": 1440,
                          "unique_validation_predictions": 21960, "inference_forward_batches": 18000,
                          "test_files_opened": 0, "raw_archives_opened": 0, "new_test_forecasts": 0}.items():
        assert execution[key] == expected, key
    for name, expected in execution["output_sha256"].items():
        assert audit.sha(OUT / name) == expected
    assert set(receipt["historical_record_sha256"]) == set(HISTORICAL)
    for name, expected in receipt["historical_record_sha256"].items():
        assert audit.sha(ROOT / name) == expected, "Historical record changed: " + name
        audit.counters["historical_records_unchanged"] += 1
    signature = hashlib.sha256(json.dumps({"protocol": digest, "code": plan["code_sha256"],
        "versions": context["runtime"]["versions"]}, sort_keys=True).encode()).hexdigest()
    assert context["code_signature"] == execution["code_signature"] == signature
    assert context["runtime"]["device"] == "cpu" and context["runtime"]["deterministic_algorithms"] is True
    assert instant(execution["completed_at_utc"]) >= max(receipts.values())
    assert instant(analysis["analyzed_at_utc"]) >= instant(execution["completed_at_utc"])
    return plan, context, execution, analysis, receipts


def development(audit):
    calendar = pd.date_range("2023-01-01", "2025-01-01", inclusive="left", tz="UTC")
    market, labels = [pd.read_csv(ROOT / name) for name in DEVELOPMENT]
    for frame in (market, labels):
        assert pd.DatetimeIndex(pd.to_datetime(frame.market_day_utc, utc=True)).equals(calendar)
    features = market.loc[:, FEATURES].to_numpy(dtype=float)
    assert calendar[~np.isfinite(features).all(axis=1)].strftime("%Y-%m-%d").tolist() == ["2023-03-24"]
    daily_return = np.r_[np.nan, np.diff(np.log(features[:, 3]))]
    targets = np.full(len(calendar), np.nan)
    for i in range(6, len(calendar)):
        window = daily_return[i-6:i+1]
        if np.isfinite(window).all():
            targets[i] = np.std(window, ddof=0)
    audit.close(labels.sigma_7.to_numpy(), targets, "Independent sigma7 reconstruction", rtol=1e-9)
    audit.counters["development_days_reconstructed"] = len(calendar)
    return calendar, features, targets


def expected_preprocessor(calendar, features, targets, history, indices):
    indices = np.asarray(indices, dtype=int)
    positions = indices[:, None]-np.arange(history, 0, -1)
    window = features[positions]
    transformed = np.empty_like(window)
    transformed[:, :, :4] = np.log(window[:, :, :4])-np.log(window[:, -1:, 3:4])
    transformed[:, :, 4] = np.log1p(window[:, :, 4])
    mean = transformed.reshape(-1, 5).mean(axis=0)
    sd = transformed.reshape(-1, 5).std(axis=0, ddof=0)
    input_days = np.unique(positions)
    return {"x_mean": mean, "x_std_ddof0": sd, "x_scale": np.where(sd == 0, 1., sd),
            "y_scale": targets[indices].mean(), "fit_outcome_count": len(indices),
            "fit_window_positions": len(indices)*history,
            "fit_outcome_indices_sha256": index_sha(indices),
            "input_day_indices_sha256": index_sha(input_days), "input_day_indices": input_days.tolist(),
            "anchor_day_indices_sha256": index_sha(indices-1),
            "first_fit_outcome_day_utc": calendar[indices.min()].strftime("%Y-%m-%d"),
            "last_fit_outcome_day_utc": calendar[indices.max()].strftime("%Y-%m-%d")}


def fold_expectations(audit, calendar, features, targets):
    positions = np.arange(len(calendar))
    expected = {}
    for history in HISTORIES:
        manifest = read(ROOT / FOLDS[history])
        assert (OUT / f"D{history}/folds.json").read_bytes() == (ROOT / FOLDS[history]).read_bytes()
        assert manifest["library"]["version"] == "0.1.5" and len(manifest["folds"]) == 12
        eligible = np.zeros(len(calendar), dtype=bool)
        for i in range(history, len(calendar)):
            eligible[i] = np.isfinite(features[i-history:i]).all() and np.isfinite(targets[i])
        covered = []
        for fold in manifest["folds"]:
            origin = int(calendar.get_loc(pd.Timestamp(fold["fit_origin_day_utc"], tz="UTC")))
            inner_origin = origin-90
            cohorts = {"train_indices": np.flatnonzero(eligible & (positions <= origin-history-1)),
                       "inner_train_indices": np.flatnonzero(eligible & (positions <= inner_origin-history-1)),
                       "inner_validation_indices": np.flatnonzero(eligible & (positions >= inner_origin)
                                                                  & (positions <= origin-history-1)),
                       "test_indices": np.flatnonzero((calendar.year == 2024)
                                                      & (calendar.month == calendar[origin].month))}
            for key, values in cohorts.items():
                assert np.array_equal(values, fold[key]), f"Independent cohort: D{history}/{fold['fold_id']}/{key}"
            evaluation = cohorts["test_indices"]
            assert np.array_equal(evaluation[:, None]-np.arange(history, 0, -1), fold["test_input_indices"])
            assert len(cohorts["inner_validation_indices"]) == 90-history
            included = np.zeros(len(calendar), dtype=bool)
            included[cohorts["train_indices"]] = True
            consecutive = included[1:] & included[:-1] & np.isfinite(targets[1:]) & np.isfinite(targets[:-1])
            scale = np.abs(np.diff(targets))[consecutive].mean()
            expected[(history, fold["fold_id"])] = {
                "fold": fold, "cohorts": cohorts, "mase_scale": scale,
                "inner": expected_preprocessor(calendar, features, targets, history, cohorts["inner_train_indices"]),
                "refit": expected_preprocessor(calendar, features, targets, history, cohorts["train_indices"])}
            covered.extend(evaluation.tolist())
            audit.counters["independent_fold_cohorts"] += 1
        assert covered == list(np.flatnonzero(calendar.year == 2024))
    return expected


def metric_values(frame):
    y, pred = frame.y_true.to_numpy(), frame.y_pred.to_numpy()
    error = y-pred
    positive = y > 0
    mse = np.mean(error**2)
    return {"rmse": np.sqrt(mse), "mse": mse, "mae": np.mean(np.abs(error)),
            "r2": 1-np.sum(error**2)/np.sum((y-y.mean())**2),
            "mape": 100*np.mean(np.abs(error[positive]/y[positive])), "mape_coverage": positive.mean(),
            "mase": np.mean(np.abs(error)/frame.mase_train_scale.to_numpy()), "mase_coverage": 1.}


def curve_checks(audit, directory, configuration, training):
    inner, refit = [pd.read_csv(directory / name) for name in ("history_inner.csv", "history_refit.csv")]
    assert inner.epoch.tolist() == list(range(1, len(inner)+1)) and len(inner) <= training["max_epochs"]
    assert refit.epoch.tolist() == list(range(1, configuration["best_epoch"]+1))
    best, best_epoch, stale = np.inf, 0, 0
    for row in inner.itertuples(index=False):
        assert np.isfinite([row.train_mse_scaled_eval, row.validation_mse_scaled_eval]).all()
        if row.validation_mse_scaled_eval < best-training["min_delta"]:
            best, best_epoch, stale = row.validation_mse_scaled_eval, row.epoch, 0
        else:
            stale += 1
        assert row.best_epoch_so_far == best_epoch
        if stale >= training["patience"]:
            assert row.epoch == len(inner)
    assert best_epoch == configuration["best_epoch"]
    assert len(inner) == training["max_epochs"] or stale >= training["patience"]
    for curve in (inner, refit):
        assert np.isfinite(curve.train_mse_scaled_eval).all()
        audit.close(curve.learning_rate.to_numpy(), np.full(len(curve), configuration["learning_rate"]), "Curve learning rate")
    audit.counters["training_curves_checked"] += 2


def trajectories(audit, plan, context, execution, receipts, calendar, targets, expected):
    runs = pd.read_csv(OUT / "runs.csv")
    keys = [(int(row.history_days), row.model, int(row.seed)) for row in runs.itertuples(index=False)]
    assert len(keys) == len(set(keys)) == 60 and set(keys) == set(product(HISTORIES, MODELS, plan["seeds"]))
    folders = list((OUT / "runs").glob("*/summary.json"))
    assert len(folders) == 60
    merged = []
    lookup = runs.set_index(["history_days", "model", "seed"])
    for path in folders:
        directory, trajectory = path.parent, read(path)
        record = trajectory["record"]
        history, model, seed = (record[k] for k in ("history_days", "model", "seed"))
        key = (history, model, seed)
        audit.artifact_manifest(directory, trajectory["artifact_sha256"])
        assert trajectory["code_signature"] == context["code_signature"]
        assert instant(trajectory["completed_at_utc"]) >= receipts[key]
        assert record["n_dates"] == 366 and record["model_fits"] == 24 and record["folds_fitted"] == 12
        monthly = []
        for ordinal, month in enumerate(sorted(p for p in directory.iterdir() if p.is_dir())):
            exp = expected[(history, month.name)]
            configuration, completed = read(month / "configuration.json"), read(month / "completion.json")
            audit.artifact_manifest(month, completed["artifact_sha256"])
            assert completed["code_signature"] == context["code_signature"]
            assert receipts[key] <= instant(completed["completed_at_utc"]) <= instant(trajectory["completed_at_utc"])
            assert read(month / "splits.json") == exp["fold"]
            assert configuration["seed"] == seed+ordinal*100000
            assert configuration["trajectory_seed"] == seed
            assert configuration["inner_initialization_seed"] == seed+ordinal*100000
            assert configuration["refit_initialization_seed"] == seed+ordinal*100000+10000000
            assert configuration["training"] == plan["training"] and configuration["test_files_opened"] == 0
            assert configuration["model"] == model and configuration["history_days"] == history
            assert configuration["variant"] == "relative_window"
            assert configuration["width"] == plan["models"][model]["width"]
            audit.close(configuration["learning_rate"], plan["models"][model]["learning_rate"], "Recipe learning rate")
            assert configuration["fold_manifest_sha256"] == plan["source_sha256"][FOLDS[history]]
            preprocessing = read(month / "preprocessing.json")
            assert set(preprocessing) == {"inner", "refit"}
            for stage, pre in preprocessing.items():
                assert stage in ("inner", "refit") and pre["feature_order"] == list(FEATURES)
                for name, value in exp[stage].items():
                    if isinstance(value, (np.ndarray, float, np.floating)):
                        audit.close(pre[name], value, "Independent preprocessing: " + name)
                    else:
                        assert pre[name] == value, "Independent preprocessing: " + name
                assert pre["target_centered"] is False
                audit.counters["preprocessor_records_checked"] += 1
            parameters = {"MLP": 80*history+33, "LSTM": 5025, "CNN1D": 1057}[model]
            assert configuration["architecture"]["parameters"]["trainable"] == parameters
            curve_checks(audit, month, configuration, plan["training"])
            frame = pd.read_csv(month / "predictions.csv")
            evaluation = exp["cohorts"]["test_indices"]
            assert frame.outcome_day.tolist() == calendar[evaluation].strftime("%Y-%m-%d").tolist()
            assert frame.input_first_day.tolist() == calendar[evaluation-history].strftime("%Y-%m-%d").tolist()
            assert frame.input_last_day.tolist() == calendar[evaluation-1].strftime("%Y-%m-%d").tolist()
            assert pd.DatetimeIndex(pd.to_datetime(frame.prediction_origin_utc, utc=True)).equals(calendar[evaluation])
            assert pd.DatetimeIndex(pd.to_datetime(frame.target_available_at_utc, utc=True)).equals(calendar[evaluation]+pd.Timedelta(days=1))
            assert frame.model.eq(model).all() and frame.seed.eq(seed).all() and frame.history_days.eq(history).all()
            assert frame.fit_seed.eq(seed+ordinal*100000).all() and frame.horizon.eq(1).all()
            assert np.isfinite(frame.y_pred).all() and frame.y_pred.ge(0).all()
            audit.close(frame.y_true.to_numpy(), targets[evaluation], "Pilot target alignment", rtol=1e-9)
            audit.close(frame.persistence_pred.to_numpy(), targets[evaluation-1], "Persistence alignment", rtol=1e-9)
            audit.close(frame.mase_train_scale.to_numpy(), np.full(len(frame), exp["mase_scale"]), "Training-only MASE scale", rtol=1e-9)
            summary = read(month / "summary.json")
            assert summary == completed["record"]
            assert summary["train_samples"] == len(exp["cohorts"]["train_indices"])
            assert summary["inner_train_samples"] == len(exp["cohorts"]["inner_train_indices"])
            assert summary["inner_validation_samples"] == len(exp["cohorts"]["inner_validation_indices"])
            assert summary["evaluation_samples"] == len(evaluation)
            assert summary["best_epoch"] == configuration["best_epoch"]
            assert summary["parameters"] == parameters
            for name, value in metric_values(frame).items():
                audit.close(summary[name], value, "Independent monthly " + name, rtol=1e-8)
            audit.counters["monthly_segments_checked"] += 1
            monthly.append(frame)
        assert len(monthly) == 12
        annual = pd.read_csv(directory / "predictions.csv")
        pd.testing.assert_frame_equal(pd.concat(monthly, ignore_index=True), annual, check_exact=False, rtol=1e-11, atol=1e-13)
        assert annual.outcome_day.tolist() == calendar[365:].strftime("%Y-%m-%d").tolist()
        for name, value in metric_values(annual).items():
            audit.close(record[name], value, "Independent trajectory " + name, rtol=1e-8)
            audit.close(lookup.loc[key, name], value, "Independent run table " + name, rtol=1e-8)
        audit.counters["seed_trajectories_checked"] += 1
        audit.counters["unique_validation_predictions_checked"] += len(annual)
        merged.append(annual)
    aggregate = pd.read_csv(OUT / "predictions.csv")
    actual = pd.concat(merged, ignore_index=True)
    order = ["history_days", "model", "seed", "outcome_day"]
    assert not aggregate.duplicated(order).any() and len(aggregate) == 21960
    pd.testing.assert_frame_equal(actual.sort_values(order).reset_index(drop=True),
                                  aggregate.sort_values(order).reset_index(drop=True),
                                  check_exact=False, rtol=1e-11, atol=1e-13)
    audit.counters["model_fits_accounted"] = 2*audit.counters["monthly_segments_checked"]
    return runs


def verified_power(power, effect, n, alpha):
    """Use statsmodels, with an independent normal/chi-square mixture fallback.

    Some high-power noncentral-t CDF calls return NaN in this runtime. Numerical
    integration of the defining normal/chi-square ratio avoids that CDF entirely.
    """
    global POWER_QUADRATURE_FALLBACKS
    value = float(power.power(effect, n, alpha, alternative="two-sided"))
    if np.isfinite(value):
        return value
    POWER_QUADRATURE_FALLBACKS += 1
    df, nc = n-1, effect*np.sqrt(n)
    critical = stats.t.isf(alpha/2, df)
    lower, upper = stats.chi2.ppf([1e-13, 1-1e-13], df)
    def density(v):
        threshold = critical*np.sqrt(v/df)
        return (stats.norm.sf(threshold-nc)+stats.norm.sf(threshold+nc))*stats.chi2.pdf(v, df)
    value, error = integrate.quad(density, lower, upper, epsabs=1e-11, epsrel=1e-10, limit=100)
    assert np.isfinite(value) and error < 1e-8
    return float(np.clip(value, 0., 1.))


def independent_R(power, delta, sd, alpha, target, floor, maximum):
    if verified_power(power, delta/sd, maximum, alpha) < target:
        return None
    continuous = float(power.solve_power(effect_size=delta/sd, power=target, alpha=alpha, alternative="two-sided"))
    n = max(floor, int(np.ceil(continuous)))
    while verified_power(power, delta/sd, n, alpha) < target:
        n += 1
    while n > floor and verified_power(power, delta/sd, n-1, alpha) >= target:
        n -= 1
    assert n <= maximum
    return n


def independent_mde_effect(power, n, alpha, target):
    # Invert the independent statsmodels power curve by bisection. Its default
    # solve_power root tolerance is comparatively coarse for small MDE values.
    low, high = 0., 1.
    while verified_power(power, high, n, alpha) < target:
        high *= 2
    for _ in range(55):
        middle = (low+high)/2
        if verified_power(power, middle, n, alpha) < target:
            low = middle
        else:
            high = middle
    return (low+high)/2


def analysis_checks(audit, plan, runs, summary, calendar, targets):
    folder, setting = OUT / "analysis", plan["dimensioning"]
    for name, expected in summary["output_sha256"].items():
        assert audit.sha(folder / name) == expected
    for name, expected in summary["input_sha256"].items():
        assert audit.sha(ROOT / name) == expected
    assert summary["analysis_code_sha256"] == plan["code_sha256"]["scripts/analyze_filipe_pilot.py"]
    for key in ("achieved_power_claim", "joint_power_claim", "historical_study_repaired",
                "independent_market_evaluation_performed", "pilot_seeds_in_future_R", "definitive_training_started",
                "SD_bounds_are_simultaneous"):
        assert summary[key] is False, key
    assert summary["normality_not_certified_with_five_pairs"] is True
    assert summary["models_fitted_by_this_analyzer"] == summary["individual_predictions_read"] == summary["old_test_accesses"] == 0
    alpha, target, delta = setting["alpha_each"], setting["power_target"], setting["delta_absolute"]
    assert alpha == .05/12 and setting["alpha_global"] == .05 and target == .8
    own_persistence = np.sqrt(np.mean((targets[365:]-targets[364:-1])**2))
    audit.close(setting["reference_persistence_rmse"], own_persistence, "Independent fixed persistence anchor", rtol=1e-9)
    audit.close(delta, .05*own_persistence, "Independent fixed planning margin", rtol=1e-9)
    metrics = pd.read_csv(folder / "pilot_metrics.csv").set_index(["history_days", "model"])
    assert len(metrics) == 12
    for key, group in runs.groupby(["history_days", "model"]):
        mean, sd = group.rmse.mean(), group.rmse.std(ddof=1)
        half = stats.t.isf(.025, 4)*sd/np.sqrt(5)
        expected = {"rmse_mean": mean, "rmse_sample_sd": sd, "rmse_coefficient_variation_percent": 100*sd/mean,
                    "rmse_ci95_lower": mean-half, "rmse_ci95_upper": mean+half}
        for name, value in expected.items():
            audit.close(metrics.loc[key, name], value, "Independent pilot summary " + name)
    pairs = pd.read_csv(folder / "paired_dimensioning.csv").set_index(["history_days", "model_a", "model_b"])
    expected_keys = [(d, a, b) for d in HISTORIES for a, b in combinations(MODELS, 2)]
    assert len(pairs) == 12 and set(pairs.index) == set(expected_keys)
    power = TTestPower()
    computed = {}
    for key in expected_keys:
        history, a, b = key
        cell = runs.loc[runs.history_days.eq(history)].pivot(index="seed", columns="model", values="rmse")
        differences = (cell.loc[plan["seeds"], a]-cell.loc[plan["seeds"], b]).to_numpy()
        sd = float(differences.std(ddof=1))
        row = pairs.loc[key]
        audit.close(row.mean_difference_descriptive, differences.mean(), "Paired mean")
        audit.close(row.paired_difference_sample_sd, sd, "Paired SD")
        assert row.n_pilot_pairs == 5
        if sd <= 0:
            assert row.status == "degenerate_zero_pilot_SD_cannot_dimension"
            computed[key] = {"point": None, "upper": None, "sd": sd, "sd_upper": None}
            continue
        sd_upper = sd*np.sqrt(4/stats.chi2.ppf(.05, 4))
        normal_R = max(5, int(np.ceil((stats.norm.isf(alpha/2)+stats.norm.isf(1-target))**2*sd**2/delta**2)))
        point = independent_R(power, delta, sd, alpha, target, 5, setting["maximum_planning_R"])
        upper = independent_R(power, delta, sd_upper, alpha, target, 5, setting["maximum_planning_R"])
        mde_effect5 = independent_mde_effect(power, 5, alpha, target)
        shapiro = stats.shapiro(differences)
        expected = {"delta_absolute": delta, "standardized_planning_delta_over_sd": delta/sd,
                    "planning_alpha_each": alpha, "R_normal_approximation_ceiling": normal_R,
                    "R_exact_noncentral_t_point_SD": point, "R_exact_noncentral_t_upper_SD": upper,
                    "paired_SD_one_sided95_upper_normal_assumption": sd_upper,
                    "MDE_at_pilot_R5_point_SD": mde_effect5*sd,
                    "MDE_at_pilot_R5_upper_SD": mde_effect5*sd_upper,
                    "shapiro_W": shapiro.statistic, "shapiro_p": shapiro.pvalue}
        for name, value in expected.items():
            audit.close(row[name], value, "Independent statsmodels planning " + name, rtol=3e-7, atol=1e-12)
        computed[key] = {"point": point, "upper": upper, "sd": sd, "sd_upper": sd_upper}
        audit.counters["paired_planning_contrasts_checked"] += 1
    point_R = max(v["point"] for v in computed.values()) if all(v["point"] is not None for v in computed.values()) else None
    upper_R = max(v["upper"] for v in computed.values()) if all(v["upper"] is not None for v in computed.values()) else None
    assert summary["common_point_SD_R"] == point_R and summary["common_recommended_upper_SD_R"] == upper_R
    mde_effects = {n: independent_mde_effect(power, n, alpha, target)
                   for n in {point_R, upper_R} if n is not None}
    for key, estimates in computed.items():
        row = pairs.loc[key]
        audit.close(row.common_point_SD_R, point_R, "Common point-SD R")
        audit.close(row.common_recommended_upper_SD_R, upper_R, "Common upper-SD R")
        for suffix, n in (("common_point_R", point_R), ("common_recommended_R", upper_R)):
            for label, sd in (("point_SD", estimates["sd"]), ("upper_SD", estimates["sd_upper"])):
                value = (mde_effects[n]*sd
                         if n is not None and sd is not None and sd > 0 else None)
                audit.close(row[f"MDE_at_{suffix}_{label}"], value, "Independent statsmodels MDE", rtol=3e-7, atol=1e-12)
        value = (verified_power(power, delta/estimates["sd_upper"], upper_R, alpha)
                 if upper_R is not None and estimates["sd_upper"] is not None else None)
        audit.close(row.planned_power_at_common_recommended_R_upper_SD, value,
                    "Hypothetical marginal power cross-check", rtol=3e-7, atol=1e-10)
    assert summary["wilcoxon_minimum_discrete_resolution_pairs"] == max(6, int(np.ceil(1-np.log2(alpha))))
    audit.counters["nonfinite_library_power_quadrature_crosschecks"] = POWER_QUADRATURE_FALLBACKS
    return {"point_SD_common_R": point_R, "upper_SD_common_recommended_R": upper_R,
            "independent_power_implementation": "statsmodels.stats.power.TTestPower plus defining normal/chi-square mixture quadrature for nonfinite library CDFs; analyzer not imported",
            "achieved_or_joint_power_claim": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "references/audits/revision_filipe_pilot_20261105.json")
    args = parser.parse_args()
    audit = Audit()
    plan, context, execution, analysis, receipts = registration(audit)
    calendar, features, targets = development(audit)
    expected = fold_expectations(audit, calendar, features, targets)
    runs = trajectories(audit, plan, context, execution, receipts, calendar, targets, expected)
    planning = analysis_checks(audit, plan, runs, analysis, calendar, targets)
    report = {"status": "PASS", "audited_at_utc": datetime.now(timezone.utc).isoformat(),
              "study_id": STUDY, "protocol_sha256": audit.sha(PLAN),
              "auditor_code_sha256": audit.sha(Path(__file__)), "counters": dict(audit.counters),
              "unique_files_hashed": len(audit.hash_cache), "planning_cross_check": planning,
              "registration_precedes_all_60_new_trajectory_starts": True,
              "git_registration_commit_contains_sealed_protocol": True,
              "remote_publication_before_pilot_asserted_by_receipt": True,
              "scientific_scope": "Fresh training randomizations conditional on already known development2024; not independent market validation or a retrospective repair",
              "new_model_fits_by_auditor": 0, "checkpoints_deserialized": 0, "new_forward_calls": 0,
              "raw_market_archives_opened": 0, "old_test_individual_files_opened": 0,
              "publication_scope": "Aggregate audit counters, protocol/code hashes and planning counts only; no vectors, prices, scaler values or weights",
              "verified_output_sha256": {name: audit.sha(OUT / name) for name in
                  ("runs.csv", "fold_metrics.csv", "predictions.csv", "execution_summary.json", "analysis/summary.json")}}
    report["counters"] = dict(audit.counters)
    report["unique_files_hashed"] = len(audit.hash_cache)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    args.report.with_suffix(".md").write_text(
        "# Auditoría independiente del piloto Filipe\n\n"
        "**PASS.** Se verificaron registro, código, cronología, cohortes, escaladores, curvas y hashes de checkpoints sin cargar modelos. "
        "Las métricas se reconstruyeron desde las predicciones de validación retenidas y el dimensionamiento se contrastó con statsmodels.\n\n"
        f"- 60 trayectorias; {audit.counters['monthly_segments_checked']} segmentos; {audit.counters['model_fits_accounted']} ajustes contabilizados.\n"
        f"- {audit.counters['unique_validation_predictions_checked']} predicciones de validación; cinco semillas nuevas por combinación.\n"
        f"- R común con DE puntual: {planning['point_SD_common_R']}; recomendación con cota superior marginal de DE: {planning['upper_SD_common_recommended_R']}.\n\n"
        "El piloto es prospectivo respecto de estas nuevas ejecuciones, sobre el calendario ya conocido de 2024. "
        "No repara retrospectivamente el estudio anterior ni acredita independencia temporal, potencia lograda o potencia conjunta. "
        "La publicación remota previa queda atestada por el recibo; se verificó localmente el commit que contiene el protocolo sellado. "
        "Los cinco registros históricos permanecen sin cambios. El auditor no ajustó redes, no cargó checkpoints y no abrió datos individuales del test anterior.\n")
    print(json.dumps({"status": "PASS", "counters": dict(audit.counters), "planning": planning}, ensure_ascii=False))


if __name__ == "__main__":
    main()
