"""Independently audit saved multi-history CV and final-test evidence.

No training, inference, raw archive access, or project analysis helper is used.
The only writes are the requested JSON/Markdown audit reports. Run with the
training environment, which supplies NumPy, pandas, SciPy and statsmodels.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import bds

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / "results/filipe_multihistory_20261104"
OLD = ROOT / "results/dl_r2_cv_20261104"
PROTOCOL = ROOT / "experiments/filipe_multihistory_20261104/protocol.json"
FREEZE = PROTOCOL.with_name("freeze.json")
REPORT = ROOT / "references/audits/revision_filipe_multihistory_20261104.json"
HISTORIES = (7, 14, 21, 28)
MODELS = ("MLP", "LSTM", "CNN1D")
SEEDS = tuple(range(4101, 4111))
FEATURES = ("open", "high", "low", "close", "volume")
METRICS = ("rmse", "mae", "mse", "r2", "mape", "mase")


def read_json(path):
    return json.loads(Path(path).read_text())


def sigma_from_closes(closes):
    """Population SD from seven dated log ratios, without pandas rolling/std."""
    closes = np.asarray(closes, dtype=float)
    result = np.full(len(closes), np.nan)
    for day in range(7, len(closes)):
        window = closes[day-7:day+1]
        if not np.isfinite(window).all() or np.any(window <= 0):
            continue
        returns = [math.log(float(window[i])/float(window[i-1])) for i in range(1, 8)]
        mean = math.fsum(returns)/7
        result[day] = math.sqrt(math.fsum((value-mean)**2 for value in returns)/7)
    return result


def recompute_metrics(group):
    y = group.y_true.to_numpy(float)
    predicted = group.y_pred.to_numpy(float)
    residuals = y-predicted
    size = len(y)
    mse = math.fsum(float(value)**2 for value in residuals)/size
    mean = math.fsum(float(value) for value in y)/size
    sst = math.fsum((float(value)-mean)**2 for value in y)
    positive = y > 0
    scales = group.mase_train_scale.to_numpy(float)
    allowed = np.isfinite(scales) & (scales > 0)
    return {"n_dates": size, "mse": mse, "rmse": math.sqrt(mse),
            "mae": math.fsum(abs(float(value)) for value in residuals)/size,
            "r2": 1-size*mse/sst if sst > 0 else np.nan,
            "mape": 100*float(np.mean(np.abs(residuals[positive]/y[positive]))) if positive.any() else np.nan,
            "mape_coverage": float(positive.mean()),
            "mase": float(np.mean(np.abs(residuals[allowed])/scales[allowed])) if allowed.any() else np.nan,
            "mase_coverage": float(allowed.mean())}


def independent_holm(log_values):
    """Independent vectorized log-Holm, keeping undefined planned tests."""
    values = np.asarray(log_values, dtype=float)
    finite = np.isfinite(values)
    ordered_values = np.where(finite, np.minimum(values, 0), 0)
    order = np.argsort(ordered_values, kind="stable")
    factors = np.log(np.arange(len(values), 0, -1))
    ascending = np.minimum(0, np.maximum.accumulate(ordered_values[order]+factors))
    result = np.empty(len(values))
    result[order] = ascending
    result[~finite] = np.nan
    return result


class Audit:
    def __init__(self):
        self.counters = {}
        self.checks = []
        self.hash_cache = {}
        self.metric_max_error = 0.
        self.preproc_cache = {}

    def require(self, condition, detail):
        if not condition:
            raise AssertionError(detail)

    def count(self, name, value=1):
        self.counters[name] = self.counters.get(name, 0)+int(value)

    def near(self, actual, expected, detail, *, atol=2e-12, rtol=2e-9):
        a, e = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        self.require(a.shape == e.shape and np.allclose(a, e, atol=atol, rtol=rtol, equal_nan=True), detail)
        finite = np.isfinite(a) & np.isfinite(e)
        if finite.any():
            self.metric_max_error = max(self.metric_max_error, float(np.max(np.abs(a[finite]-e[finite]))))
        self.count("numeric_values_checked", a.size)

    def digest(self, path):
        path = Path(path)
        # Check actual bytes once per audit; all evaluated fit outputs are frozen.
        if path not in self.hash_cache:
            self.hash_cache[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        return self.hash_cache[path]

    def verify_hashes(self, base, mapping, detail):
        for name, wanted in mapping.items():
            self.require(self.digest(Path(base)/name) == wanted, f"{detail}: {name}")
            self.count("artifact_hashes_checked")

    def market(self, test=False):
        market = pd.read_csv(ROOT / "results/eda_development/daily_market_development.csv", index_col="market_day_utc")
        market.index = pd.DatetimeIndex(pd.to_datetime(market.index, utc=True))
        if test:
            future = pd.read_csv(NEW / "test/daily_market_test.csv", index_col="market_day_utc")
            future.index = pd.DatetimeIndex(pd.to_datetime(future.index, utc=True))
            market = pd.concat([market.loc[:, list(FEATURES)], future.loc[:, list(FEATURES)]])
        finish = "2026-08-31" if test else "2024-12-31"
        expected = pd.date_range("2023-01-01", finish, tz="UTC")
        self.require(market.index.equals(expected), "The full daily calendar must retain missing-day positions")
        return market

    def preprocessing(self, preproc, train, history, market, targets, cache_key):
        train = np.asarray(train, dtype=int)
        self.require(preproc["history_days"] == history, "Preprocessing history differs from its model")
        self.require(preproc["fit_outcome_count"] == len(train), "Scaler uses the wrong number of training outcomes")
        index_hash = hashlib.sha256(np.asarray(train, dtype="<i8").tobytes()).hexdigest()
        self.require(preproc["fit_outcome_indices_sha256"] == index_hash, "Scaler uses outcomes outside the frozen training cohort")
        if cache_key not in self.preproc_cache:
            values = market.loc[:, list(FEATURES)].to_numpy(float)
            windows = values[train[:, None]-np.arange(history, 0, -1)]
            transformed = np.empty_like(windows)
            transformed[..., :4] = np.log(windows[..., :4])-np.log(windows[:, -1:, 3:4])
            transformed[..., 4] = np.log1p(windows[..., 4])
            flattened = transformed.reshape(-1, 5)
            standard = flattened.std(axis=0, ddof=0)
            self.preproc_cache[cache_key] = (flattened.mean(axis=0), np.where(standard == 0, 1, standard),
                                            float(np.mean(targets[train])))
        mean, scale, target_scale = self.preproc_cache[cache_key]
        self.near(preproc["x_mean"], mean, "Frozen channel mean is not training-only")
        self.near(preproc["x_scale"], scale, "Frozen channel scale is not training-only")
        self.near(preproc["y_scale"], target_scale, "Frozen target scale is not training-only")
        self.count("training_preprocessors_checked")

    def verify_cv(self):
        plan = read_json(PROTOCOL)
        execution = read_json(NEW / "execution_summary.json")
        self.require(execution["all_registered_trajectories_complete"] and execution["trajectories"] == 90
                     and execution["completed_project_fits"] == 2160 and execution["test_files_opened"] == 0,
                     "All 90 registered new CV trajectories must complete before the audit")
        context = read_json(NEW / "code_manifest_before_fitting.json")
        self.require(context["protocol_sha256"] == self.digest(PROTOCOL), "CV protocol changed after registration")
        self.verify_hashes(ROOT, context["code_sha256"], "CV fitting code changed")
        self.verify_hashes(ROOT, plan["sources"], "CV source changed")
        market = self.market()
        values = market.loc[:, list(FEATURES)].to_numpy(float)
        targets = sigma_from_closes(values[:, 3])
        recorded_targets = pd.read_csv(ROOT / "results/eda_development/daily_volatility_development.csv").sigma_7.to_numpy()
        self.near(recorded_targets, targets, "Daily development sigma7 differs from seven dated log returns")
        self.count("independently_reconstructed_volatility_days", len(targets))
        old = pd.read_csv(OLD / "predictions.csv")
        old = old.loc[old.variant.eq("relative_window") & old.history_days.eq(7)].copy()
        new = pd.read_csv(NEW / "predictions.csv")
        frames = pd.concat([old, new], ignore_index=True)
        self.require(len(new) == 32940 and len(old) == 10980 and len(frames) == 43920, "Wrong CV prediction counts")
        expected_dates = pd.date_range("2024-01-01", "2024-12-31", tz="UTC").strftime("%Y-%m-%d").tolist()
        eligible = {}
        for history in HISTORIES:
            mask = np.zeros(len(values), dtype=bool)
            for outcome in range(history, len(values)):
                mask[outcome] = np.isfinite(values[outcome-history:outcome]).all() and np.isfinite(targets[outcome])
            eligible[history] = mask
        self.december_configurations = {}
        for (history, model, seed), frame in frames.groupby(["history_days", "model", "seed"]):
            self.require(history in HISTORIES and model in MODELS and seed in SEEDS, "Unexpected CV configuration")
            frame = frame.sort_values("outcome_day")
            self.require(frame.outcome_day.tolist() == expected_dates, "CV trajectory must cover each 2024 date once")
            self.require(frame.run_id.nunique() == 1, "A CV path mixes trajectories")
            base = OLD if history == 7 else NEW
            directory = base / "runs" / frame.run_id.iloc[0]
            complete = read_json(directory / "summary.json")
            self.verify_hashes(directory, complete["artifact_sha256"], "A completed CV artifact changed")
            self.check_metric_record(frame, complete["record"], "CV path metric")
            for ordinal, (month, group) in enumerate(frame.groupby("fold_id", sort=True)):
                part = directory / month
                split = read_json(part / "splits.json")
                train = np.asarray(split["train_indices"], dtype=int)
                inner = np.asarray(split["inner_train_indices"], dtype=int)
                validation = np.asarray(split["inner_validation_indices"], dtype=int)
                origin = market.index.get_loc(pd.Timestamp(month+"-01", tz="UTC"))
                positions = np.arange(len(market))
                expected_train = np.flatnonzero(eligible[history] & (positions <= origin-history-1))
                expected_inner = np.flatnonzero(eligible[history] & (positions <= origin-90-history-1))
                expected_validation = np.flatnonzero(eligible[history] & (positions >= origin-90) & (positions <= origin-history-1))
                self.require(np.array_equal(train, expected_train) and np.array_equal(inner, expected_inner)
                             and np.array_equal(validation, expected_validation), "CV outcomes violate expanding history or an embargo")
                evaluation = market.index.get_indexer(pd.to_datetime(group.outcome_day, utc=True))
                self.require(split["test_indices"] == evaluation.tolist(), "CV saved outcomes differ from monthly predictions")
                self.require(np.array_equal(split["test_input_indices"], evaluation[:, None]-np.arange(history, 0, -1)),
                             "CV input includes unknown current/future days")
                self.near(group.y_true, targets[evaluation], "CV target dates differ from reconstructed sigma7")
                config = read_json(part / "configuration.json")
                expected_parameters = 80*history+33 if model == "MLP" else 5025 if model == "LSTM" else 1057
                self.require(config["trajectory_seed"] == seed and config["history_days"] == history
                             and config["model"] == model and config["width"] == plan["models"][model]["width"]
                             and config["learning_rate"] == plan["models"][model]["learning_rate"]
                             and config["inner_initialization_seed"] == seed+ordinal*100000
                             and config["refit_initialization_seed"] == seed+ordinal*100000+10000000
                             and config["architecture"]["parameters"]["trainable"] == expected_parameters
                             and config["test_files_opened"] == 0, "CV architecture, seeds, or source contract changed")
                preprocessing = read_json(part / "preprocessing.json")
                self.preprocessing(preprocessing["inner"], inner, history, market, targets, (history, month, "inner"))
                self.preprocessing(preprocessing["refit"], train, history, market, targets, (history, month, "refit"))
                included = np.zeros(len(market), dtype=bool); included[train] = True
                adjacent = included[1:] & included[:-1] & np.isfinite(targets[1:]) & np.isfinite(targets[:-1])
                scale = np.mean(np.abs(targets[1:]-targets[:-1])[adjacent])
                self.near(group.mase_train_scale, np.full(len(group), scale), "CV MASE scale includes unadmitted dates")
                self.check_metric_record(group, read_json(part / "summary.json"), "CV monthly metric")
                if month == "2024-12":
                    self.december_configurations[(int(history), model, int(seed))] = (part / "configuration.json", config)
                self.count("cv_monthly_model_segments_checked")
            self.count("cv_trajectories_checked")
        self.require(self.counters["cv_trajectories_checked"] == 120, "CV grid must contain four histories, three models and ten seeds")
        self.verify_analysis(frames, "cv")
        self.cv_frames, self.development_market, self.development_targets = frames, market, targets
        self.checks.append("All saved CV cohorts, scaler fits, seeds, metrics, checkpoints and fold hashes match the four-history causal contract.")
        return frames

    def check_metric_record(self, frame, record, detail):
        values = recompute_metrics(frame)
        for name in (*METRICS, "mape_coverage", "mase_coverage"):
            self.near(record[name], values[name], detail+": "+name)
        self.count("prediction_metrics_recomputed", len(frame))

    def verify_analysis(self, frame, phase):
        directory = NEW / ("analysis_cv" if phase == "cv" else "test/analysis")
        summary = read_json(directory / "summary.json")
        self.require(summary["analysis_code_sha256"] == self.digest(ROOT / "scripts/analyze_filipe_multihistory.py"), "Analysis source changed after calculation")
        self.verify_hashes(ROOT, summary["source_sha256"], "Analysis input changed")
        self.verify_hashes(directory, summary["output_sha256"], "Analysis output changed")
        self.require(summary["neural_forecasts"] == len(frame), "Analysis prediction count differs")
        row_metrics, fold_metrics = {}, {}
        for keys, group in frame.groupby(["history_days", "model", "seed"]):
            row_metrics[tuple(keys)] = recompute_metrics(group)
            for month, part in group.groupby("fold_id"):
                fold_metrics[tuple(keys)+(month,)] = recompute_metrics(part)
        for history, group in frame.groupby("history_days"):
            reference = group.sort_values("outcome_day").drop_duplicates("outcome_day").copy()
            reference["y_pred"] = reference.persistence_pred
            row_metrics[(history, "Persistencia", None)] = recompute_metrics(reference)
            for month, part in reference.groupby("fold_id"):
                fold_metrics[(history, "Persistencia", None, month)] = recompute_metrics(part)
        for filename, calculations, monthly in [("run_metrics.csv", row_metrics, False), ("fold_metrics.csv", fold_metrics, True)]:
            table = pd.read_csv(directory / filename)
            self.require(len(table) == len(calculations), "Analysis metric grid has missing or extra rows")
            for row in table.to_dict("records"):
                key = (int(row["history_days"]), row["model"], int(row["seed"]) if np.isfinite(row["seed"]) else None)
                if monthly: key += (row["fold_id"],)
                expected = calculations[key]
                for metric in (*METRICS, "mape_coverage", "mase_coverage", "n_dates"):
                    self.near(row[metric], expected[metric], f"{phase} analysis {filename}: {key}/{metric}")
            self.count("analysis_metric_rows_checked", len(table))
        for filename, calculations, monthly in [("metrics_summary.csv", row_metrics, False), ("fold_metrics_summary.csv", fold_metrics, True)]:
            table = pd.read_csv(directory / filename)
            for row in table.to_dict("records"):
                history, model, metric = int(row["history_days"]), row["model"], row["metric"]
                selected = [values[metric] for key, values in calculations.items()
                            if key[0] == history and key[1] == model and (not monthly or key[3] == row["fold_id"])]
                selected = np.asarray(selected, dtype=float)
                valid = selected[np.isfinite(selected)]
                self.require(row["n_total"] == len(selected) and row["n_defined"] == len(valid), "Summary incorrectly hides undefined metric values")
                mean = np.mean(valid) if len(valid) else np.nan
                stochastic = model != "Persistencia"
                sd = np.std(valid, ddof=1) if len(valid) > 1 and stochastic else np.nan
                half = stats.t.ppf(.975, len(valid)-1)*sd/math.sqrt(len(valid)) if len(valid) > 1 and stochastic else np.nan
                for name, wanted in [("mean", mean), ("sample_sd", sd), ("ci95_lower", mean-half), ("ci95_upper", mean+half)]:
                    self.near(row[name], wanted, f"{phase} seed summary: {history}/{model}/{metric}/{name}")
            self.count("seed_summary_rows_checked", len(table))
        self.verify_bds(frame, phase, directory)

    def verify_bds(self, frame, phase, directory):
        raw = pd.read_csv(directory / "bds_by_fold.csv")
        months = pd.date_range("2024-01-01", "2024-12-01", freq="MS").strftime("%Y-%m").tolist() if phase == "cv" else pd.date_range("2025-01-01", "2026-08-01", freq="MS").strftime("%Y-%m").tolist()
        expected_count = len(HISTORIES)*len(MODELS)*len(SEEDS)*len(months)*2
        self.require(len(raw) == expected_count and not raw.duplicated(["history_days", "model", "seed", "fold_id", "dimension"]).any(), "BDS does not preserve its entire planned hypothesis family")
        keyed = raw.set_index(["history_days", "model", "seed", "fold_id", "dimension"])
        for (history, model, seed), trajectory in frame.groupby(["history_days", "model", "seed"]):
            for month in months:
                group = trajectory.loc[trajectory.fold_id.eq(month)].sort_values("outcome_day")
                dates = pd.DatetimeIndex(pd.to_datetime(group.outcome_day, utc=True))
                contiguous = len(dates) < 2 or bool(np.all((dates[1:]-dates[:-1]) == pd.Timedelta(days=1)))
                residuals = group.y_true.to_numpy()-group.y_pred.to_numpy()
                epsilon = 1.5*np.std(residuals, ddof=1) if len(residuals) > 1 else np.nan
                statistics = np.full(2, np.nan)
                if contiguous and len(residuals) > 5 and np.isfinite(residuals).all() and np.isfinite(epsilon) and epsilon > 0:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            statistics, _ = bds(residuals, max_dim=3, epsilon=epsilon)
                    except (ValueError, ZeroDivisionError, FloatingPointError, IndexError):
                        pass
                for dimension in (2, 3):
                    row = keyed.loc[(history, model, seed, month, dimension)]
                    value = float(statistics[dimension-2])
                    if not np.isfinite(value): value = np.nan
                    self.require(row.n_dates == len(group), "BDS sample count differs from its reporting block")
                    self.near(row.statistic, value, "BDS statistic differs from independent statsmodels evaluation")
                    if np.isfinite(value):
                        self.near(row.epsilon, epsilon, "BDS radius differs from1.5 sample residual SD")
                        log_p = math.log(2)+float(stats.norm.logsf(abs(value)))
                        self.near(row.log_p_value, log_p, "BDS normal-reference p-value differs")
                        self.near(row.p_value, math.exp(log_p), "BDS exact stored p-value differs")
                        self.require(row.status == "calculated_approximate_short_block", "A calculated BDS value is mislabeled")
                    else:
                        self.require(str(row.status).startswith("undefined_"), "Undefined BDS results must remain explicit")
                    self.count("bds_rows_recomputed")
        holm = independent_holm(raw.log_p_value.to_numpy())
        self.near(raw.log_p_holm, holm, "Global BDS Holm adjustment differs")
        self.require(raw.family_size_planned.eq(expected_count).all(), "BDS Holm family omits planned hypotheses")
        self.near(raw.p_holm, np.exp(holm), "Global BDS adjusted p-values differ")
        for filename, keys in [("bds_by_fold_summary.csv", ["history_days", "model", "dimension", "fold_id"]),
                               ("bds_mean_across_folds_descriptive.csv", ["history_days", "model", "seed", "dimension"]),
                               ("bds_pvalues_descriptive_summary.csv", ["history_days", "model", "dimension"])]:
            table = pd.read_csv(directory / filename)
            expected = {}
            for key, group in raw.groupby(keys):
                defined = group.p_value.dropna()
                expected[tuple(key)] = (defined.mean(), defined.std(ddof=1), len(defined))
            self.require(len(table) == len(expected), "Descriptive BDS summary grid differs")
            for row in table.to_dict("records"):
                mean, sd, count = expected[tuple(row[k] for k in keys)]
                self.near(row["mean_p_value_descriptive"], mean, "Descriptive BDS mean differs")
                self.near(row["sd_p_value_descriptive"], sd, "Descriptive BDS SD differs")
                self.require(row["n_defined"] == count, "BDS summary hides undefined p-values")
            self.count("bds_descriptive_rows_checked", len(table))

    def verify_freeze(self):
        freeze = read_json(FREEZE)
        self.require(freeze["protocol_sha256"] == self.digest(PROTOCOL), "Closed protocol hash changed")
        for key in ("code_sha256", "source_cv_sha256", "analysis_cv_sha256", "source_epoch_configuration_sha256"):
            self.verify_hashes(ROOT, freeze[key], "Closed evidence changed")
        rows = []
        for (history, model), group in self.cv_frames.groupby(["history_days", "model"]):
            rmses = [recompute_metrics(part)["rmse"] for _, part in group.groupby("seed")]
            rows.append({"history_days": int(history), "model": model,
                         "parameters": 80*history+33 if model == "MLP" else 5025 if model == "LSTM" else 1057,
                         "cv_rmse_mean": float(np.mean(rmses)), "cv_rmse_sample_sd": float(np.std(rmses, ddof=1))})
        def choose(candidates):
            minimum = min(row["cv_rmse_mean"] for row in candidates)
            return min((row for row in candidates if row["cv_rmse_mean"] <= minimum+1e-12),
                       key=lambda row: (row["history_days"], row["parameters"], row["model"]))
        for model in MODELS:
            selected = choose([row for row in rows if row["model"] == model])
            self.require(freeze["selection"]["selected_by_model"][model]["history_days"] == selected["history_days"], "D was not selected using development RMSE")
        candidate = choose(rows)
        frozen = freeze["selection"]["baseline"]
        self.require((frozen["model"], frozen["history_days"]) == (candidate["model"], candidate["history_days"]), "The neural baseline was selected outside the closed CV rule")
        self.near(frozen["cv_rmse_mean"], candidate["cv_rmse_mean"], "Frozen baseline score differs from its CV forecasts")
        actual_jobs = {(int(j["history_days"]), j["model"], int(j["seed"])) for j in freeze["final_jobs"]}
        expected_jobs = {(d, m, s) for d in HISTORIES for m in MODELS for s in SEEDS}
        self.require(actual_jobs == expected_jobs and len(freeze["final_jobs"]) == 120, "The closed final job grid differs")
        for job in freeze["final_jobs"]:
            key = (int(job["history_days"]), job["model"], int(job["seed"]))
            path, config = self.december_configurations[key]
            self.require(job["epochs"] == config["best_epoch"] and job["epoch_source_configuration"] == path.relative_to(ROOT).as_posix()
                         and job["epoch_source_configuration_sha256"] == self.digest(path), "Final epochs were not closed from the December inner CV fit")
        self.freeze = freeze
        self.checks.append("Configuration selection and all120 epoch counts were independently recovered from development CV, before test access.")

    def verify_test(self):
        self.verify_freeze()
        freeze_hash = self.digest(FREEZE)
        models = read_json(NEW / "final_models_manifest.json")
        self.require(models["freeze_sha256"] == freeze_hash and len(models["jobs"]) == 120 and models["test_files_opened"] == 0,
                     "Final models were not all frozen before test")
        self.verify_hashes(NEW, models["artifact_sha256"], "A frozen final model/configuration changed")
        plan = read_json(PROTOCOL)
        market, targets = self.development_market, self.development_targets
        features = market.loc[:, list(FEATURES)].to_numpy(float)
        frozen_jobs = {(j["history_days"], j["model"], j["seed"]): j for j in self.freeze["final_jobs"]}
        for job in models["jobs"]:
            history, model, seed = int(job["history_days"]), job["model"], int(job["seed"])
            directory = NEW / "final_fit" / job["run_id"]
            config = read_json(directory / "configuration.json")
            train = np.asarray(config["train_indices"], dtype=int)
            expected = [i for i in range(history, len(market)-history)
                        if np.isfinite(features[i-history:i]).all() and np.isfinite(targets[i])]
            self.require(train.tolist() == expected, "Final development fit violates its D embargo or eligibility")
            self.require(config["final_refit_epochs"] == frozen_jobs[(history, model, seed)]["epochs"]
                         and config["freeze_sha256"] == freeze_hash and config["initialization_seed"] == seed+10000000
                         and config["test_files_opened"] == 0, "Final training settings differ from the closed jobs")
            self.preprocessing(read_json(directory / "preprocessing.json"), train, history, market, targets, (history, "final"))
            self.count("final_models_checked")
        audit = read_json(NEW / "test/raw_test_audit.json")
        reserve = read_json(ROOT / "data/manifests/test_raw_reserve_20250101_20260831.json")
        self.require(audit["freeze_sha256"] == freeze_hash
                     and audit["final_models_manifest_sha256"] == self.digest(NEW / "final_models_manifest.json")
                     and audit["reserve_manifest_sha256"] == self.digest(ROOT / "data/manifests/test_raw_reserve_20250101_20260831.json")
                     and audit["archives_opened"] == 20 and not audit["september_october2026_opened"], "Reserved raw identities or admitted period changed")
        closed = pd.Timestamp(self.freeze["closed_at_utc"])
        final_times = [pd.Timestamp(j["completed_at_utc"]) for j in models["jobs"]]
        first_test = pd.Timestamp(audit["first_raw_access_at_utc"])
        self.require(closed < min(pd.Timestamp(j["started_at_utc"]) for j in models["jobs"])
                     and max(final_times) <= pd.Timestamp(models["completed_at_utc"]) < first_test,
                     "Test access preceded configuration/model closure")
        self.require(len(audit["quality_by_month"]) == 20, "Test quality audit does not cover all twenty archives")
        for recorded, reserved in zip(audit["quality_by_month"], reserve["archives"]):
            self.require(recorded["sha256"] == reserved["sha256"] and recorded["csv_rows"] == reserved["csv_rows"]
                         and recorded["month"] in reserved["archive"] and recorded["timestamp_unit"] == "us", "A reserved archive identity/epoch-unit differs")
        combined = self.market(test=True)
        combined_targets = sigma_from_closes(combined.close.to_numpy())
        cohort = read_json(NEW / "test/cohort_manifest.json")
        self.require(cohort["calendar_days"] == 608 and cohort["freeze_sha256"] == freeze_hash, "Test calendar or configuration changed")
        expected_dates = []
        values = combined.loc[:, list(FEATURES)].to_numpy(float)
        for outcome in range(731, len(combined)):
            inputs_ok = all(np.isfinite(values[outcome-history:outcome]).all() for history in HISTORIES)
            if inputs_ok and np.isfinite(combined_targets[outcome]) and np.isfinite(combined_targets[outcome-1]):
                expected_dates.append(combined.index[outcome].strftime("%Y-%m-%d"))
        self.require(cohort["common_outcome_days_utc"] == expected_dates and cohort["n_common_outcomes"] == len(expected_dates), "The test cohort was selected outside the predeclared finite-data rule")
        frame = pd.read_csv(NEW / "test/predictions.csv")
        self.require(len(frame) == 120*len(expected_dates), "Wrong final test forecast count")
        summary = read_json(NEW / "test/execution_summary.json")
        self.require(summary["model_runs"] == 120 and summary["test_period_model_refits"] == 0
                     and summary["actual_checkpoint_forward_calls"] == 120
                     and summary["monthly_reporting_blocks"] == 20, "Test should score every fixed checkpoint once without retraining")
        for (history, model, seed), group in frame.groupby(["history_days", "model", "seed"]):
            group = group.sort_values("outcome_day")
            self.require(group.outcome_day.tolist() == expected_dates, "Test configurations use unequal dates")
            days = pd.DatetimeIndex(pd.to_datetime(group.outcome_day, utc=True))
            positions = combined.index.get_indexer(days)
            self.near(group.y_true, combined_targets[positions], "Test sigma7 differs from independent dated close calculation")
            self.near(group.persistence_pred, combined_targets[positions-1], "Persistence is not the known prior sigma7")
            for field, wanted in [("input_first_day", days-pd.Timedelta(days=int(history))),
                                  ("input_last_day", days-pd.Timedelta(days=1)),
                                  ("prediction_origin_utc", days), ("target_available_at_utc", days+pd.Timedelta(days=1))]:
                self.require(np.array_equal(pd.to_datetime(group[field], utc=True), wanted), "Test input/target availability clocks are not causal")
            name = group.run_id.iloc[0]
            completion = read_json(NEW / "test/runs" / name / "completion.json")
            self.verify_hashes(NEW / "test/runs" / name, completion["artifact_sha256"], "A completed test prediction artifact changed")
            self.require(completion["freeze_sha256"] == freeze_hash
                         and completion["cohort_sha256"] == self.digest(NEW / "test/cohort_manifest.json")
                         and completion["model_sha256"] == self.digest(NEW / "final_fit" / name / "model.pt")
                         and completion["record"]["checkpoint_forward_calls"] == 1, "Test used an unclosed cohort/model or repeated forward call")
            self.check_metric_record(group, completion["record"], "Saved final test metric")
            final_config = read_json(NEW / "final_fit" / name / "configuration.json")
            self.near(group.mase_train_scale, np.full(len(group), final_config["mase_train_scale"]), "Final MASE denominator changed on test")
            self.count("test_checkpoint_outputs_checked")
        events = [json.loads(line) for line in (NEW / "final_test_events.jsonl").read_text().splitlines() if line]
        scored = [event for event in events if event["event"] == "test_run_scored_once"]
        self.require(len(scored) == 120 and len({event["run_id"] for event in scored}) == 120, "Event log does not show exactly one evaluation per checkpoint")
        self.require(not any(pd.Timestamp(event["at_utc"]) >= first_test and event["event"] == "final_fit_started_after_configuration_freeze" for event in events),
                     "A final training stage started after test access")
        self.verify_analysis(frame, "test")
        self.count("independently_reconstructed_volatility_days", len(combined_targets)-731)
        self.checks.append("All120 fixed final models precede first raw test access; twenty monthly test blocks contain no retraining, and all metrics/BDS diagnostics reproduce from saved predictions.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("cv", "test"), default="cv", help="test also audits the complete CV and configuration closure")
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    audit = Audit()
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "phase": args.phase,
              "verifier": "scripts/verify_filipe_multihistory.py", "independence": "No project analysis/training helpers imported; no raw archives, fitting or inference executed.",
              "raw_archives_opened": 0, "new_model_fits": 0, "checkpoint_forward_calls": 0}
    failed = None
    try:
        audit.verify_cv()
        if args.phase == "test": audit.verify_test()
        report["status"] = "PASS"
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        failed = exc
    report.update({"counters": audit.counters, "checks": audit.checks,
                   "maximum_numeric_absolute_difference": audit.metric_max_error,
                   "verifier_sha256": audit.digest(Path(__file__)),
                   "protocol_sha256": audit.digest(PROTOCOL),
                   "scope_limits": ["Saved forecasts are audited; test inference is deliberately not repeated.",
                                    "Monthly BDS values retain their short-sample asymptotic limitations.",
                                    "Seed intervals remain conditional on this observed calendar; no achieved-power or universal-equivalence claim."]})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    text = ["# Auditoría independiente: cuatro entradas y test reservado", "", f"Estado: **{report['status']}**. Fase: **{args.phase}**.", "",
            report["independence"], ""]
    text += ["- "+entry for entry in audit.checks]
    text += ["", "| Control | Cantidad |", "|---|---:|"]
    text += [f"| {key} | {value} |" for key, value in audit.counters.items()]
    text += ["", "Los promedios de p-valores BDS son descriptivos; los meses cortos limitan la aproximación asintótica. Los intervalos entre semillas no estiman variabilidad entre mercados independientes.", ""]
    if failed: text += ["Fallo: "+report["error"], ""]
    args.report.with_suffix(".md").write_text("\n".join(text))
    print(json.dumps({"status": report["status"], "phase": args.phase, "counters": audit.counters,
                      "error": report.get("error"), "report": str(args.report.relative_to(ROOT)) if args.report.is_relative_to(ROOT) else str(args.report)}))
    if failed: raise failed


if __name__ == "__main__":
    main()
