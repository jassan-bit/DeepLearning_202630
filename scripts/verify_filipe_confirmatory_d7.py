"""Audit retained fresh-seed D7 evidence independently, without fitting models.

No runner, analyzer or torch module is imported. Only established development
CSVs, frozen metadata, and the selected new phase's saved artifacts are read.
Checkpoints are hashed as bytes, never deserialized. Public push chronology is
attested by receipts, whose commit contents and local chronology are checked.
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
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestPower


ROOT = Path(__file__).resolve().parents[1]
STUDY = "e1_filipe_confirmatory_d7_20261105"
OUT = ROOT / "results" / STUDY
PLAN = ROOT / "experiments" / STUDY / "protocol.json"
MODELS = ("MLP", "LSTM", "CNN1D")
FEATURES = ("open", "high", "low", "close", "volume")
EXTENSION_SEEDS = list(range(6301, 6341, 2))
DEVELOPMENT = ("results/eda_development/daily_market_development.csv",
               "results/eda_development/daily_volatility_development.csv")
FOLD_SOURCE = "results/dl_r2_cv_20261104/folds.json"
DECISION = "references/literature_e1/selection_20261105/decision.json"
HISTORICAL = (
    "experiments/filipe_multihistory_20261104/protocol.json",
    "experiments/filipe_multihistory_20261104/freeze.json",
    "results/filipe_multihistory_20261104/test/execution_summary.json",
    "results/filipe_multihistory_20261104/test/raw_test_audit.json",
    "results/filipe_multihistory_20261104/test/analysis/metrics_summary.csv",
)
OLD_PILOT = "results/e1_filipe_pilot_20261105"
OLD_RECORDS = tuple(OLD_PILOT + "/" + name for name in
                    ("protocol.json", "execution_summary.json", "runs.csv"))
TRAINING_CODE = {
    "scripts/run_filipe_confirmatory_d7.py", "scripts/analyze_filipe_confirmatory_d7.py",
    "scripts/run_filipe_pilot.py", "scripts/analyze_filipe_pilot.py",
    "scripts/train_filipe_multihistory.py", "scripts/train_dl_benchmark.py",
    "scripts/dl_data.py", "scripts/dl_data_v2.py", "scripts/dl_models.py",
}
MONTHLY_FILES = {"predictions.csv", "history_inner.csv", "history_refit.csv",
                 "preprocessing.json", "splits.json", "configuration.json",
                 "model.pt", "summary.json"}
POWER_QUADRATURE_FALLBACKS = 0


def require(condition, label):
    if not condition:
        raise AssertionError(label)


def permitted(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), "Audit input outside the repository")
    relative = path.relative_to(ROOT).as_posix()
    require("/test/" not in relative or relative in HISTORICAL,
            "Audit must not open old-test individual evidence")
    require(path.suffix.lower() not in {".zip", ".gz", ".tar", ".npz", ".npy"},
            "Audit must not open market or retention archives")
    return path


def read(path):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON object member: " + key)
            result[key] = value
        return result
    return json.loads(permitted(path).read_text(), object_pairs_hook=unique_object)


def read_frame(path):
    return pd.read_csv(permitted(path))


def instant(value):
    value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(value.tzinfo is not None, "Audit timestamps must include their timezone")
    return value


def git_bytes(commit, name):
    return subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)


def ancestor(first, second):
    subprocess.run(["git", "merge-base", "--is-ancestor", first, second],
                   cwd=ROOT, check=True, capture_output=True)


def index_sha(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


class Audit:
    def __init__(self):
        self.counters = Counter()
        self.hash_cache = {}

    def sha(self, path):
        path = permitted(path)
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
            require(actual is None or bool(pd.isna(actual)), label)
            return
        aa, bb = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        require(aa.shape == bb.shape, label + ": shape")
        require(np.allclose(aa, bb, rtol=rtol, atol=atol, equal_nan=True), label)
        self.counters["numeric_values_compared"] += int(aa.size)

    def manifest(self, directory, values, *, excluded):
        directory = permitted(directory)
        paths = list(directory.rglob("*"))
        require(not any(p.is_symlink() for p in paths), "Symlink inside retained evidence")
        actual = {p.relative_to(directory).as_posix() for p in paths if p.is_file()} - set(excluded)
        require(set(values) == actual, "Retained manifest must cover every artifact: " + str(directory))
        for name, expected in values.items():
            path = (directory / name).resolve()
            require(path.is_relative_to(directory), "Unsafe retained artifact path")
            require(self.sha(path) == expected, "Artifact hash: " + str(path))


def seed_states(seeds):
    return {seed + ordinal*100000 + offset for seed in seeds
            for ordinal in range(12) for offset in (0, 17, 10000000, 10000017)}


def registered_inputs(audit):
    plan = read(PLAN)
    context = read(OUT / "code_manifest_before_fitting.json")
    require((OUT / "protocol.json").read_bytes() == PLAN.read_bytes(), "Protocol copy differs")
    digest = audit.sha(PLAN)
    require(digest == context["protocol_sha256"], "Registered protocol hash")
    require(plan["study_id"] == STUDY and plan["histories"] == [7], "D7 study identity")
    require(list(plan["models"]) == list(MODELS), "Three fixed family recipes")
    require(plan["variance_extension_seeds"] == EXTENSION_SEEDS, "Exactly twenty fresh extension seeds")
    require(plan["original_pilot_seeds"] == list(range(6201, 6206)), "Historical five remain sensitivity only")
    require(plan["definitive_seed_start"] == 50000001 and plan["definitive_seed_step"] == 2,
            "Predeclared definitive seed namespace")
    require(plan["variant"] == "relative_window" and plan["target_return_days"] == 7
            and plan["forecast_horizon_days"] == 1 and plan["workers"] == 4,
            "Frozen representation, horizon and concurrency")
    require(plan["monthly_folds"] == 12 and plan["embargo_days"] == 7, "D7 monthly cohort")
    require(plan["test_files_opened_allowed"] == plan["raw_archives_opened_allowed"] == 0,
            "Development-only data boundary")
    pairs = [{"history_days": 7, "model_a": a, "model_b": b} for a, b in combinations(MODELS, 2)]
    require(plan["primary_pairs"] == pairs, "Exactly three oriented primary neural pairs")
    dim = plan["dimensioning"]
    expected = {"primary_family_size": 3, "alpha_global": .05, "alpha_each": .05/3,
                "power_target": .8, "execution_floor": 30, "variance_extension_n": 20,
                "sd_upper_confidence": .95, "maximum_planning_R": 10000,
                "sd_source": "20 new extension seeds only"}
    for key, value in expected.items():
        require(dim[key] == value, "Predeclared planning field: " + key)
    require(plan["analysis"]["bootstrap_replicates"] == 4999
            and plan["analysis"]["bootstrap_seed"] == 20261105,
            "Predeclared paired bootstrap sensitivity")
    require(set(plan["source_sha256"]) == set(DEVELOPMENT) | {FOLD_SOURCE, DECISION}
            | set(HISTORICAL) | set(OLD_RECORDS), "Exact registered source guards")
    require(set(plan["code_sha256"]) == TRAINING_CODE, "Nine sealed code files")
    require(plan["code_sha256"] == context["code_sha256"], "Execution code guards")
    for name, expected_hash in {**plan["source_sha256"], **plan["code_sha256"]}.items():
        require(audit.sha(ROOT / name) == expected_hash, "Registered source changed: " + name)
    for name, expected_hash in plan["code_sha256"].items():
        require(hashlib.sha256(git_bytes(context["source_commit"], name)).hexdigest() == expected_hash,
                "Implementation commit differs: " + name)
    original = read(ROOT / HISTORICAL[0])
    require(plan["models"] == original["models"] and plan["training"] == original["training"],
            "Original architecture, optimizer and training rules remain frozen")
    signature = hashlib.sha256(json.dumps({"study": STUDY, "protocol": digest,
        "code": plan["code_sha256"], "versions": context["runtime"]["versions"]},
        sort_keys=True).encode()).hexdigest()
    require(signature == context["code_signature"], "Independent execution signature")
    require(context["runtime"]["device"] == "cpu"
            and context["runtime"]["deterministic_algorithms"] is True,
            "Deterministic CPU execution record")
    require(read(OUT / "runtime.json") == context["runtime"], "Runtime copy differs")
    require(instant(plan["registered_at_utc"]) <= instant(context["registered_at_utc"]),
            "Protocol precedes execution context")
    previous_seeds = set()
    for name in ("results/dl_r2_cv_20261104/runs.csv", "results/filipe_multihistory_20261104/runs.csv",
                 OLD_RECORDS[-1]):
        previous_seeds.update(read_frame(ROOT / name).seed.astype(int))
    prior_states = seed_states(previous_seeds)
    extension_states = seed_states(EXTENSION_SEEDS)
    future_states = seed_states([50000001+2*i for i in range(10000)])
    require(len(extension_states) == 20*48 and len(future_states) == 10000*48,
            "No within-phase initialization/minibatch state collisions")
    require(not extension_states.intersection(prior_states)
            and not future_states.intersection(prior_states | extension_states),
            "Fresh RNG namespaces, including refit and minibatch roles")
    old_execution = read(ROOT / OLD_RECORDS[1])
    require(old_execution["trajectories"] == 60, "Original sixty-trajectory pilot retained")
    for name, expected_hash in old_execution["output_sha256"].items():
        require(audit.sha(ROOT / OLD_PILOT / name) == expected_hash, "Historical pilot output changed")
        audit.counters["historical_pilot_outputs_unchanged"] += 1
    audit.counters["historical_records_unchanged"] = len(HISTORICAL) + len(OLD_RECORDS)
    return plan, context


def publication(audit, context, phase, lock=None):
    name = ("preregistration_git_receipt.json" if phase == "variance_extension"
            else "locked_design_publication_receipt.json")
    receipt = read(OUT / name)
    require(receipt["registered_protocol_sha256"] == context["protocol_sha256"]
            and receipt["publication_completed_before_runs"] is True, "Publication-before-runs receipt")
    commit = receipt["published_commit"]
    ancestor(context["source_commit"], commit)
    ancestor(commit, "HEAD")
    files = [PLAN] + [ROOT / n for n in TRAINING_CODE | {DECISION}]
    if lock is not None:
        files.append(OUT / "locked_design.json")
        require(receipt["locked_design_sha256"] == audit.sha(files[-1]), "Published design hash")
    for path in files:
        require(hashlib.sha256(git_bytes(commit, path.relative_to(ROOT).as_posix())).hexdigest()
                == audit.sha(path), "Published commit bytes differ: " + str(path))
    commit_time = int(subprocess.check_output(["git", "show", "-s", "--format=%ct", commit],
                                             cwd=ROOT, text=True))
    published = instant(receipt["published_at_utc"])
    lower = instant(lock["locked_at_utc"] if lock is not None else context["registered_at_utc"])
    require(lower <= published and commit_time <= published.timestamp(), "Publication chronology")
    return receipt


def development(audit):
    calendar = pd.date_range("2023-01-01", "2025-01-01", inclusive="left", tz="UTC")
    market, labels = [read_frame(ROOT / name) for name in DEVELOPMENT]
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
    require(len(inner) > 0 and len(refit) > 0 and configuration["best_epoch"] >= 1,
            "Nonempty fitting curves and a positive selected epoch")
    assert inner.epoch.tolist() == list(range(1, len(inner)+1)) and len(inner) <= training["max_epochs"]
    assert refit.epoch.tolist() == list(range(1, configuration["best_epoch"]+1))
    best, best_epoch, stale = np.inf, 0, 0
    for row in inner.itertuples(index=False):
        assert np.isfinite([row.train_mse_scaled_eval, row.validation_mse_scaled_eval]).all()
        require(row.train_mse_scaled_eval >= 0 and row.validation_mse_scaled_eval >= 0,
                "Nonnegative inner training and validation MSE")
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
        require(curve.train_mse_scaled_eval.ge(0).all(), "Nonnegative training MSE")
        audit.close(curve.learning_rate.to_numpy(), np.full(len(curve), configuration["learning_rate"]), "Curve learning rate")
    audit.counters["training_curves_checked"] += 2



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
    if sd is None or not np.isfinite(sd) or sd <= 0:
        return None
    if verified_power(power, delta/sd, maximum, alpha) < target:
        return None
    low, high = floor, maximum
    while low < high:
        middle = (low+high)//2
        if verified_power(power, delta/sd, middle, alpha) >= target:
            high = middle
        else:
            low = middle+1
    require(verified_power(power, delta/sd, low, alpha) >= target,
            "Independent R attains the planned marginal power")
    require(low == floor or verified_power(power, delta/sd, low-1, alpha) < target,
            "Independent R is the minimum integer under its floor")
    return low


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


def fold_expectations(audit, directory, calendar, features, targets):
    source = ROOT / FOLD_SOURCE
    require((directory / "D7/folds.json").read_bytes() == source.read_bytes(),
            "Byte-identical original D7 folds")
    manifest = read(source)
    require(manifest["library"]["version"] == "0.1.5" and len(manifest["folds"]) == 12
            and manifest["history_days"] == 7 and manifest["target_returns_days"] == 7
            and manifest["forecast_days"] == 1, "Original forward-fold identity")
    positions = np.arange(len(calendar))
    eligible = np.zeros(len(calendar), dtype=bool)
    for i in range(7, len(calendar)):
        eligible[i] = np.isfinite(features[i-7:i]).all() and np.isfinite(targets[i])
    result, covered = {}, []
    for ordinal, fold in enumerate(manifest["folds"]):
        origin = int(calendar.get_loc(pd.Timestamp(fold["fit_origin_day_utc"], tz="UTC")))
        inner_origin = origin-90
        cohorts = {"train_indices": np.flatnonzero(eligible & (positions <= origin-8)),
                   "inner_train_indices": np.flatnonzero(eligible & (positions <= inner_origin-8)),
                   "inner_validation_indices": np.flatnonzero(eligible & (positions >= inner_origin)
                                                              & (positions <= origin-8)),
                   "test_indices": np.flatnonzero((calendar.year == 2024)
                                                   & (calendar.month == ordinal+1))}
        require(calendar[origin].month == ordinal+1, "Chronological monthly fold order")
        for key, values in cohorts.items():
            require(np.array_equal(values, fold[key]), "Independent D7 cohort: " + key)
        evaluation = cohorts["test_indices"]
        require(np.array_equal(evaluation[:, None]-np.arange(7, 0, -1), fold["test_input_indices"]),
                "Causal input window at every validation origin")
        require(len(cohorts["inner_validation_indices"]) == 83, "Seven-day inner embargo")
        included = np.zeros(len(calendar), dtype=bool)
        included[cohorts["train_indices"]] = True
        consecutive = included[1:] & included[:-1] & np.isfinite(targets[1:]) & np.isfinite(targets[:-1])
        scale = np.abs(np.diff(targets))[consecutive].mean()
        result[fold["fold_id"]] = {
            "ordinal": ordinal, "fold": fold, "cohorts": cohorts, "mase_scale": scale,
            "inner": expected_preprocessor(calendar, features, targets, 7, cohorts["inner_train_indices"]),
            "refit": expected_preprocessor(calendar, features, targets, 7, cohorts["train_indices"])}
        covered.extend(evaluation.tolist())
        audit.counters["independent_fold_cohorts"] += 1
    require(covered == np.flatnonzero(calendar.year == 2024).tolist(), "All 366 dates once")
    return result


def same_record(audit, saved, expected, label):
    require(set(saved) == set(expected), label + ": columns")
    for name, value in expected.items():
        if value is None or isinstance(value, (int, float, np.integer, np.floating)):
            audit.close(saved[name], value, label + ": " + name, rtol=2e-9)
        else:
            require(saved[name] == value, label + ": " + name)


def trajectories(audit, phase, plan, context, calendar, targets, expected, seeds, publication_receipt, lock=None):
    directory = OUT / phase
    execution = read(directory / "execution_summary.json")
    lock_sha = audit.sha(OUT / "locked_design.json") if lock is not None else None
    counts = {"trajectories": 3*len(seeds), "seeds_per_model": len(seeds),
              "monthly_segments": 36*len(seeds), "project_fits": 72*len(seeds),
              "unique_validation_predictions": 1098*len(seeds),
              "inference_forward_batches": 900*len(seeds), "dates_per_trajectory": 366,
              "workers": 4, "test_files_opened": 0, "raw_archives_opened": 0}
    for key, value in counts.items():
        require(execution[key] == value, "Execution count: " + key)
    require(execution["phase"] == phase and execution["study_id"] == STUDY, "Execution phase")
    require(execution["seeds"] == seeds and execution["all_fits_are_new"] is True
            and execution["publication_precedes_every_trajectory"] is True,
            "Complete fresh registered seed family")
    require(execution["protocol_sha256"] == execution["registered_protocol_sha256"]
            == context["protocol_sha256"] and execution["code_signature"] == context["code_signature"],
            "Execution signature and protocol")
    require(execution["locked_design_sha256"] == lock_sha, "Execution design lock")
    require(set(execution["output_sha256"]) == {"runs.csv", "predictions.csv", "fold_metrics.csv"},
            "Every aggregate output is signed")
    for name, expected_hash in execution["output_sha256"].items():
        require(audit.sha(directory / name) == expected_hash, "Signed aggregate output: " + name)
    marker = read(directory / "run_started.json")
    require(marker["protocol_sha256"] == context["protocol_sha256"] and marker["phase"] == phase
            and marker["locked_design_sha256"] == lock_sha, "Phase start marker")
    published = instant(publication_receipt["published_at_utc"])
    require(published < instant(marker["started_at_utc"]), "Publication precedes phase launch")
    expected_keys = set(product(MODELS, seeds))
    starts = {}
    paths = list((directory / "trajectory_start_receipts").glob("*.json"))
    require(len(paths) == len(expected_keys), "Exact start receipt count")
    for path in paths:
        receipt = read(path)
        job = receipt["job"]
        key = (job["model"], job["seed"])
        require(key in expected_keys and key not in starts, "Unique expected trajectory start")
        require(path.name == f"D7_{key[0]}_s{key[1]}.json", "Start receipt filename")
        require(job == {"variant": "relative_window", "model": key[0], "seed": key[1],
                        "history_days": 7, **plan["models"][key[0]]}, "Frozen trajectory job")
        require(receipt["phase"] == phase and receipt["code_signature"] == context["code_signature"]
                and receipt["protocol_sha256"] == context["protocol_sha256"], "Signed start receipt")
        started = instant(receipt["started_at_utc"])
        require(instant(marker["started_at_utc"]) <= started
                and instant(context["registered_at_utc"]) < started and published < started,
                "Registration/publication precede every new trajectory")
        starts[key] = started
        audit.counters["new_trajectory_start_receipts"] += 1
    require(set(starts) == expected_keys, "Every registered job has a start receipt")
    require(instant(execution["first_new_trajectory_started_at_utc"]) == min(starts.values()),
            "Recorded first new trajectory")
    require(instant(execution["completed_at_utc"]) >= max(starts.values()), "Execution completion time")
    runs = read_frame(directory / "runs.csv")
    keys = list(runs[["model", "seed"]].itertuples(index=False, name=None))
    require(len(keys) == len(expected_keys) == len(set(keys)) and set(keys) == expected_keys,
            "Exact aggregate trajectory coverage")
    lookup = runs.set_index(["model", "seed"])
    folders = list((directory / "runs").glob("*/summary.json"))
    require(len(folders) == len(expected_keys), "Exact completed trajectory count")
    require(len(list((directory / "runs").iterdir())) == len(expected_keys), "No extra trajectory evidence")
    annual_frames, monthly_frames, observed = [], [], set()
    preprocessor_hashes = {}
    for path in folders:
        run_directory, trajectory = path.parent, read(path)
        record = trajectory["record"]
        model, seed = record["model"], record["seed"]
        key = (model, seed)
        require(key in expected_keys and key not in observed, "Unique completed job")
        observed.add(key)
        require(run_directory.name == f"relative_window_{model}_D7_s{seed}_{context['code_signature'][:10]}",
                "Signed trajectory identity")
        require(record["run_id"] == run_directory.name and record["history_days"] == 7
                and record["n_dates"] == 366 and record["model_fits"] == 24
                and record["folds_fitted"] == 12 and record["variant"] == "relative_window",
                "Trajectory dimensions and scope")
        require(trajectory["code_signature"] == context["code_signature"], "Trajectory execution signature")
        audit.manifest(run_directory, trajectory["artifact_sha256"], excluded={"summary.json"})
        completed_time = instant(trajectory["completed_at_utc"])
        require(starts[key] <= completed_time <= instant(execution["completed_at_utc"]),
                "Trajectory completion chronology")
        months = [p for p in run_directory.iterdir() if p.is_dir()]
        require({p.name for p in months} == set(expected) and len(months) == 12, "Exact twelve monthly artifacts")
        month_predictions, month_records = [], []
        for month in sorted(months):
            exp = expected[month.name]
            ordinal = exp["ordinal"]
            configuration, completed = read(month / "configuration.json"), read(month / "completion.json")
            require(set(completed["artifact_sha256"]) == MONTHLY_FILES, "Complete monthly artifact manifest")
            audit.manifest(month, completed["artifact_sha256"], excluded={"completion.json"})
            require(completed["code_signature"] == configuration["code_signature"] == context["code_signature"],
                    "Monthly execution signatures")
            require(starts[key] < instant(completed["completed_at_utc"]) <= completed_time,
                    "Monthly completion follows its new trajectory start")
            require(read(month / "splits.json") == exp["fold"], "Immutable split indices")
            require(configuration["seed"] == seed+ordinal*100000
                    and configuration["trajectory_seed"] == seed
                    and configuration["inner_initialization_seed"] == seed+ordinal*100000
                    and configuration["refit_initialization_seed"] == seed+ordinal*100000+10000000,
                    "Initialization and refit seed derivation")
            require(configuration["training"] == plan["training"] and configuration["test_files_opened"] == 0
                    and configuration["model"] == model and configuration["history_days"] == 7
                    and configuration["variant"] == "relative_window"
                    and configuration["width"] == plan["models"][model]["width"], "Fixed monthly recipe")
            audit.close(configuration["learning_rate"], plan["models"][model]["learning_rate"], "Fixed learning rate")
            require(configuration["fold_manifest_sha256"] == plan["source_sha256"][FOLD_SOURCE], "Monthly fold hash")
            pre_hash = audit.sha(month / "preprocessing.json")
            if month.name not in preprocessor_hashes:
                preprocessing = read(month / "preprocessing.json")
                require(set(preprocessing) == {"inner", "refit"}, "Two separately fitted preprocessors")
                for stage, pre in preprocessing.items():
                    require(pre["feature_order"] == list(FEATURES) and pre["target_centered"] is False,
                            "Preprocessing channel contract")
                    for name, value in exp[stage].items():
                        if isinstance(value, (np.ndarray, float, np.floating)):
                            audit.close(pre[name], value, "Independent preprocessing: " + name)
                        else:
                            require(pre[name] == value, "Independent preprocessing: " + name)
                    audit.counters["independently_reconstructed_preprocessors"] += 1
                preprocessor_hashes[month.name] = pre_hash
            else:
                require(pre_hash == preprocessor_hashes[month.name], "Identical seed-independent preprocessing bytes")
            audit.counters["preprocessor_records_checked"] += 2
            parameters = {"MLP": 593, "LSTM": 5025, "CNN1D": 1057}[model]
            require(configuration["architecture"]["parameters"]["trainable"] == parameters, "Architecture parameter count")
            curve_checks(audit, month, configuration, plan["training"])
            prediction = read_frame(month / "predictions.csv")
            evaluation = exp["cohorts"]["test_indices"]
            require(prediction.outcome_day.tolist() == calendar[evaluation].strftime("%Y-%m-%d").tolist(), "Monthly dates")
            require(prediction.input_first_day.tolist() == calendar[evaluation-7].strftime("%Y-%m-%d").tolist()
                    and prediction.input_last_day.tolist() == calendar[evaluation-1].strftime("%Y-%m-%d").tolist(),
                    "Past-only OHLCV window")
            require(pd.DatetimeIndex(pd.to_datetime(prediction.prediction_origin_utc, utc=True)).equals(calendar[evaluation])
                    and pd.DatetimeIndex(pd.to_datetime(prediction.target_available_at_utc, utc=True))
                    .equals(calendar[evaluation]+pd.Timedelta(days=1)), "Forecast availability clock")
            require(prediction.model.eq(model).all() and prediction.seed.eq(seed).all()
                    and prediction.history_days.eq(7).all() and prediction.fit_seed.eq(seed+ordinal*100000).all()
                    and prediction.horizon.eq(1).all() and prediction.fold_id.eq(month.name).all()
                    and prediction.run_id.eq(run_directory.name).all()
                    and prediction.variant.eq("relative_window").all(), "Every prediction belongs to its registered job")
            require(np.isfinite(prediction.y_pred).all() and prediction.y_pred.ge(0).all(), "Finite nonnegative predictions")
            audit.close(prediction.y_true.to_numpy(), targets[evaluation], "Independent target alignment", rtol=1e-9)
            audit.close(prediction.persistence_pred.to_numpy(), targets[evaluation-1], "Independent persistence alignment", rtol=1e-9)
            audit.close(prediction.mase_train_scale.to_numpy(), np.full(len(prediction), exp["mase_scale"]),
                        "Training-only MASE scale", rtol=1e-9)
            summary = read(month / "summary.json")
            require(summary == completed["record"], "Monthly completion record")
            require(summary["model"] == model and summary["seed"] == seed
                    and summary["history_days"] == 7 and summary["horizon"] == 1
                    and summary["fold_id"] == month.name and summary["variant"] == "relative_window",
                    "Monthly completion belongs to the same frozen job and fold")
            for name, cohort in (("train_samples", "train_indices"), ("inner_train_samples", "inner_train_indices"),
                                 ("inner_validation_samples", "inner_validation_indices"), ("evaluation_samples", "test_indices")):
                require(summary[name] == len(exp["cohorts"][cohort]), "Monthly sample count: " + name)
            require(summary["parameters"] == parameters and summary["best_epoch"] == configuration["best_epoch"],
                    "Monthly model/epoch metadata")
            for name, value in metric_values(prediction).items():
                audit.close(summary[name], value, "Independent monthly metric: " + name, rtol=1e-8)
            for name, scale in (("history_inner.csv", exp["inner"]["y_scale"]), ("history_refit.csv", exp["refit"]["y_scale"])):
                curve = read_frame(month / name)
                audit.close(curve.train_rmse_original, np.sqrt(curve.train_mse_scaled_eval)*scale,
                            "Training curve original scale")
                audit.close(curve.validation_rmse_original, np.sqrt(curve.validation_mse_scaled_eval)*scale,
                            "Validation curve original scale")
            month_predictions.append(prediction)
            month_records.append(summary)
            audit.counters["monthly_segments_checked"] += 1
        annual = read_frame(run_directory / "predictions.csv")
        pd.testing.assert_frame_equal(pd.concat(month_predictions, ignore_index=True), annual,
                                      check_exact=False, rtol=1e-11, atol=1e-13)
        require(annual.outcome_day.tolist() == calendar[365:].strftime("%Y-%m-%d").tolist(), "Annual 366-day cohort")
        for name, value in metric_values(annual).items():
            audit.close(record[name], value, "Independent annual metric: " + name, rtol=1e-8)
        same_record(audit, lookup.loc[key].to_dict(), {k: v for k, v in record.items() if k not in {"model", "seed"}},
                    "Run aggregate versus retained record")
        saved_months = read_frame(run_directory / "fold_metrics.csv")
        pd.testing.assert_frame_equal(saved_months, pd.DataFrame(month_records), check_dtype=False,
                                      check_exact=False, rtol=1e-10, atol=1e-13)
        annual_frames.append(annual)
        monthly_frames.append(saved_months)
        audit.counters["seed_trajectories_checked"] += 1
        audit.counters["unique_validation_predictions_checked"] += len(annual)
    require(observed == expected_keys, "Every registered trajectory independently audited")
    for filename, frames, order in (("predictions.csv", annual_frames, ["model", "seed", "outcome_day"]),
                                    ("fold_metrics.csv", monthly_frames, ["model", "seed", "fold_id"])):
        saved = read_frame(directory / filename)
        require(not saved.duplicated(order).any(), "Unique aggregate rows: " + filename)
        pd.testing.assert_frame_equal(pd.concat(frames, ignore_index=True).sort_values(order).reset_index(drop=True),
                                      saved.sort_values(order).reset_index(drop=True),
                                      check_exact=False, rtol=1e-10, atol=1e-13)
    audit.counters["model_fits_accounted"] = 2*audit.counters["monthly_segments_checked"]
    require(audit.counters["unique_validation_predictions_checked"] == counts["unique_validation_predictions"],
            "Independently counted distinct forecasts")
    return runs, execution


def planning_checks(audit, plan, calendar, targets):
    lock = read(OUT / "locked_design.json")
    settings = plan["dimensioning"]
    alpha, target, delta = settings["alpha_each"], settings["power_target"], settings["delta_absolute"]
    persistence = float(np.sqrt(np.mean((targets[365:]-targets[364:-1])**2)))
    audit.close(settings["reference_persistence_rmse"], persistence, "Independent persistence anchor", rtol=1e-9)
    audit.close(delta, .05*persistence, "Predeclared absolute planning margin", rtol=1e-9)
    require(lock["registered_protocol_sha256"] == audit.sha(PLAN)
            and lock["analysis_code_sha256"] == plan["code_sha256"]["scripts/analyze_filipe_confirmatory_d7.py"],
            "Sealed analysis produced the design lock")
    expected_inputs = {PLAN.relative_to(ROOT).as_posix(), OLD_RECORDS[-1],
                       (OUT / "variance_extension/runs.csv").relative_to(ROOT).as_posix(),
                       (OUT / "variance_extension/execution_summary.json").relative_to(ROOT).as_posix()}
    require(set(lock["input_sha256"]) == expected_inputs, "Exact four design inputs")
    for name, expected_hash in lock["input_sha256"].items():
        require(audit.sha(ROOT / name) == expected_hash, "Locked design input: " + name)
    for key, path in (("original_pilot_runs_sha256", ROOT / OLD_RECORDS[-1]),
                      ("variance_extension_runs_sha256", OUT / "variance_extension/runs.csv"),
                      ("variance_extension_summary_sha256", OUT / "variance_extension/execution_summary.json")):
        require(lock[key] == audit.sha(path), "Design direct input guard: " + key)
    execution = read(OUT / "variance_extension/execution_summary.json")
    require(execution["protocol_sha256"] == audit.sha(PLAN), "Variance extension protocol identity")
    for key, count in (("trajectories", 60), ("monthly_segments", 720), ("project_fits", 1440),
                       ("unique_validation_predictions", 21960)):
        require(execution[key] == count, "Complete variance extension: " + key)
    for name, expected_hash in execution["output_sha256"].items():
        require(audit.sha(OUT / "variance_extension" / name) == expected_hash,
                "Variance extension aggregate guard: " + name)
    require(instant(lock["locked_at_utc"]) > instant(execution["completed_at_utc"]),
            "Complete variance extension precedes the design lock")
    require(set(lock["design_output_sha256"]) == {"paired_dimensioning.csv", "variance_summary.json", "qq_new20.png", "README.md"},
            "All design outputs signed")
    for name, expected_hash in lock["design_output_sha256"].items():
        require(audit.sha(OUT / "planning" / name) == expected_hash, "Planning output guard: " + name)
    summary = read(OUT / "planning/variance_summary.json")
    for name, value in summary.items():
        require(lock[name] == value, "Design summary agrees with lock: " + name)
    for name in ("old_pilot_in_primary_variance", "pooled_variance_chooses_R", "budget_limit_caps_required_R",
                 "joint_or_achieved_power_claim", "future_advanced_boundary_test", "new_independent_market_evaluation"):
        require(lock[name] is False, "Planning scope flag: " + name)
    require(lock["all25_pilot_seeds_excluded_from_definitive"] is True
            and lock["SD_bound_is_marginal_normal_reference"] is True,
            "Variance bound and seed exclusions")
    old = read_frame(ROOT / OLD_RECORDS[-1])
    old = old.loc[old.history_days.eq(7)].copy()
    fresh = read_frame(OUT / "variance_extension/runs.csv")
    for data, seeds in ((old, plan["original_pilot_seeds"]), (fresh, EXTENSION_SEEDS)):
        keys = list(data[["model", "seed"]].itertuples(index=False, name=None))
        require(len(keys) == len(set(keys)) == 3*len(seeds)
                and set(keys) == set(product(MODELS, seeds)) and data.history_days.eq(7).all()
                and data.n_dates.eq(366).all() and np.isfinite(data.rmse).all(), "Exact variance source cohorts")
    old = old.pivot(index="seed", columns="model", values="rmse").loc[plan["original_pilot_seeds"], list(MODELS)]
    fresh = fresh.pivot(index="seed", columns="model", values="rmse").loc[EXTENSION_SEEDS, list(MODELS)]
    table = read_frame(OUT / "planning/paired_dimensioning.csv").set_index(["model_a", "model_b"])
    require(len(table) == 3 and set(table.index) == set(combinations(MODELS, 2)), "Three planning contrasts only")
    power = TTestPower()
    results = {}
    for a, b in combinations(MODELS, 2):
        before, new = (old[a]-old[b]).to_numpy(), (fresh[a]-fresh[b]).to_numpy()
        sd_old, sd, sd_pooled = before.std(ddof=1), new.std(ddof=1), np.r_[before, new].std(ddof=1)
        upper = float(sd*np.sqrt(19/stats.chi2.ppf(.05, 19))) if sd > 0 else None
        upper_pooled = float(sd_pooled*np.sqrt(24/stats.chi2.ppf(.05, 24))) if sd_pooled > 0 else None
        point = independent_R(power, delta, sd, alpha, target, 30, 10000)
        conservative = independent_R(power, delta, upper, alpha, target, 30, 10000)
        pooled = independent_R(power, delta, upper_pooled, alpha, target, 30, 10000)
        normal = max(30, int(np.ceil((stats.norm.isf(alpha/2)+stats.norm.ppf(target))**2*sd**2/delta**2))) if sd > 0 else None
        row = table.loc[(a, b)]
        values = {"history_days": 7, "n_primary_variance_seeds": 20, "n_old_seeds_sensitivity": 5,
                  "n_pooled_sensitivity": 25, "SD_original5_descriptive": sd_old, "SD_new20_primary": sd,
                  "SD_pooled25_sensitivity": sd_pooled, "SD_upper95_new20_primary": upper,
                  "SD_upper95_pooled25_sensitivity": upper_pooled, "delta_absolute": delta,
                  "standardized_planning_delta_primary": delta/sd if sd > 0 else None,
                  "planning_alpha_each": alpha, "marginal_power_target": target,
                  "R_point_new20": point, "R_upper_new20_primary": conservative,
                  "R_normal_approximation_new20": normal,
                  "R_upper_pooled25_sensitivity_not_used": pooled,
                  "MDE_at_primary_pilot20_point_SD": independent_mde_effect(power, 20, alpha, target)*sd if sd > 0 else None}
        for name, value in values.items():
            audit.close(row[name], value, "Independent planning: " + name, rtol=3e-7, atol=1e-12)
        require(row.status == ("conditional_normal_planning" if conservative is not None
                               else "zero_SD_or_planning_limit_no_sufficient_R"), "Planning limitation status")
        results[(a, b)] = {"point": point, "upper": conservative, "sd_upper": upper}
        audit.counters["paired_planning_contrasts_checked"] += 1
    common = max(r["upper"] for r in results.values()) if all(r["upper"] is not None for r in results.values()) else None
    point_common = max(r["point"] for r in results.values()) if all(r["point"] is not None for r in results.values()) else None
    require(lock["common_R"] == common and lock["common_point_R_new20"] == point_common,
            "R chosen exclusively by the twenty fresh seeds and marginal upper SD")
    expected_summary = {"primary_variance_seed_count": 20, "pooled_sensitivity_seed_count": 25,
                        "execution_floor": 30, "maximum_planning_R": 10000,
                        "workflow_budget_seed_limit": plan["maximum_definitive_seeds"],
                        "alpha_global": .05, "alpha_each": alpha, "marginal_power_target": .8,
                        "definitive_seed_step": 2,
                        "definitive_seed_first_last": [50000001, 50000001+2*(common-1)] if common else None,
                        "definitive_trajectories": 3*common if common else None,
                        "definitive_fits": 72*common if common else None}
    for name, value in expected_summary.items():
        require(lock[name] == value, "Independent design summary: " + name)
    status = ("blocked_unidentifiable_dimensioning" if common is None else
              "blocked_budget_and_plan_complete" if common > plan["maximum_definitive_seeds"]
              else "design_locked_ready_for_new_seed_runs")
    require(lock["stage_status"] == status, "Required R is never capped to a budget")
    for key, value in results.items():
        row = table.loc[key]
        audit.close(row.common_R, common, "Common sample size")
        sd = value["sd_upper"]
        mde = independent_mde_effect(power, common, alpha, target)*sd if common is not None and sd is not None else None
        planned = verified_power(power, delta/sd, common, alpha) if common is not None and sd is not None else None
        audit.close(row.MDE_at_common_R_upper_SD, mde, "Independent design MDE", rtol=3e-7, atol=1e-12)
        audit.close(row.planning_power_at_common_R_upper_SD, planned, "Independent design marginal power", rtol=3e-7)
    return lock, {"common_R": common, "common_point_R_new20": point_common,
                  "primary_variance_seeds": 20, "pooled25_sensitivity_chooses_R": False,
                  "power_implementation": "Independent statsmodels TTestPower; defining normal/chi-square quadrature for nonfinite CDFs",
                  "achieved_or_joint_power_claim": False}


def independent_bootstrap(scores, B, seed):
    """Draw paired seed rows jointly for all three contrasts, without analyzer helpers."""
    sampled_indices = np.random.default_rng(seed).integers(0, len(scores), size=(B, len(scores)))
    means, effects = [], []
    for a, b in combinations(range(3), 2):
        differences = (scores[:, a]-scores[:, b])[sampled_indices]
        mean = differences.mean(axis=1)
        sd = differences.std(axis=1, ddof=1)
        dz = np.full(B, np.nan)
        np.divide(mean, sd, out=dz, where=sd > 0)
        means.append(mean)
        effects.append(dz)
    return np.asarray(means), np.asarray(effects)


def recheck_extension_starts(audit, plan, context, receipt, lock):
    """Check every original phase start again when auditing the final phase."""
    directory = OUT / "variance_extension"
    marker = read(directory / "run_started.json")
    execution = read(directory / "execution_summary.json")
    require(marker["protocol_sha256"] == context["protocol_sha256"]
            and marker["phase"] == "variance_extension" and marker["locked_design_sha256"] is None,
            "Variance extension launch marker remains intact")
    published, launched = instant(receipt["published_at_utc"]), instant(marker["started_at_utc"])
    require(published < launched, "Protocol publication precedes variance extension launch")
    expected = set(product(MODELS, EXTENSION_SEEDS))
    paths = list((directory / "trajectory_start_receipts").glob("*.json"))
    require(len(paths) == 60, "Exactly sixty variance-extension starts retained")
    found, times = set(), []
    for path in paths:
        row = read(path)
        job = row["job"]
        key = (job["model"], job["seed"])
        require(key in expected and key not in found, "Unique expected variance-extension start")
        require(path.name == f"D7_{key[0]}_s{key[1]}.json"
                and job == {"variant": "relative_window", "model": key[0], "seed": key[1],
                            "history_days": 7, **plan["models"][key[0]]}, "Immutable extension job")
        require(row["phase"] == "variance_extension" and row["protocol_sha256"] == context["protocol_sha256"]
                and row["code_signature"] == context["code_signature"], "Immutable extension start signature")
        started = instant(row["started_at_utc"])
        require(published < started and launched <= started
                and instant(context["registered_at_utc"]) < started
                and started <= instant(execution["completed_at_utc"]) < instant(lock["locked_at_utc"]),
                "Every extension trajectory follows public registration and precedes the final lock")
        found.add(key)
        times.append(started)
    require(found == expected and min(times) == instant(execution["first_new_trajectory_started_at_utc"]),
            "All sixty public-registration chronology checks complete")
    audit.counters["variance_extension_start_receipts_rechecked"] = len(found)


def final_checks(audit, plan, lock, runs, execution):
    folder = OUT / "analysis"
    summary = read(folder / "summary.json")
    require(summary["registered_protocol_sha256"] == audit.sha(PLAN)
            and summary["locked_design_sha256"] == audit.sha(OUT / "locked_design.json")
            and summary["analysis_code_sha256"] == plan["code_sha256"]["scripts/analyze_filipe_confirmatory_d7.py"],
            "Final analysis uses the published protocol, lock and sealed implementation")
    expected_inputs = {PLAN.relative_to(ROOT).as_posix()}
    expected_inputs.update((OUT / name).relative_to(ROOT).as_posix() for name in
                          ("locked_design.json", "definitive/runs.csv", "definitive/execution_summary.json"))
    require(set(summary["input_sha256"]) == expected_inputs, "Every final-analysis input signed")
    for name, expected_hash in summary["input_sha256"].items():
        require(audit.sha(ROOT / name) == expected_hash, "Final analysis input: " + name)
    require(set(summary["output_sha256"]) == {"rmse_summary.csv", "metrics_summary.csv", "paired_comparisons.csv",
                                            "metrics.png", "qq_differences.png", "README.md"},
            "Every final-analysis output signed")
    for name, expected_hash in summary["output_sha256"].items():
        require(audit.sha(folder / name) == expected_hash, "Final analysis output: " + name)
    require(instant(summary["analyzed_at_utc"]) >= instant(execution["completed_at_utc"]),
            "Complete definitive trajectories precede inference")
    for name in ("normality_diagnostic_changes_primary_test", "pilot_seeds_in_definitive", "test2025_2026_read",
                 "new_market_generalization_claim", "historical_inference_relabelled_confirmatory",
                 "advanced_gain_boundary_tested", "achieved_or_joint_power_claim"):
        require(summary[name] is False, "Final inferential scope: " + name)
    n = lock["common_R"]
    alpha, delta = plan["dimensioning"]["alpha_each"], plan["dimensioning"]["delta_absolute"]
    seeds = [50000001+2*i for i in range(n)]
    scores = runs.pivot(index="seed", columns="model", values="rmse").loc[seeds, list(MODELS)].to_numpy()
    metrics = read_frame(folder / "rmse_summary.csv").set_index("model")
    require(len(metrics) == 3 and set(metrics.index) == set(MODELS), "Three definitive metric summaries")
    for j, model in enumerate(MODELS):
        values = scores[:, j]
        mean, sd = float(values.mean()), float(values.std(ddof=1))
        half = stats.t.isf(.025, n-1)*sd/np.sqrt(n)
        for name, value in {"history_days": 7, "n_seeds": n, "n_dates_per_trajectory": 366,
                            "rmse_mean": mean, "rmse_sample_sd": sd, "rmse_CV_percent": 100*sd/mean,
                            "rmse_ci95_lower": mean-half, "rmse_ci95_upper": mean+half}.items():
            audit.close(metrics.loc[model, name], value, "Independent definitive mean/uncertainty: " + name)
    secondary = read_frame(folder / "metrics_summary.csv").set_index(["model", "metric"])
    require(len(secondary) == 18 and set(secondary.index) == set(product(MODELS, ("rmse", "mse", "mae", "r2", "mape", "mase"))),
            "All six metrics summarized for every family")
    for (model, metric), row in secondary.iterrows():
        values = runs.loc[runs.model.eq(model), metric].to_numpy()
        values = values[np.isfinite(values)]
        mean, sd = float(values.mean()), float(values.std(ddof=1))
        half = stats.t.isf(.025, len(values)-1)*sd/np.sqrt(len(values))
        for name, value in {"n_seeds": n, "n_defined": len(values), "mean": mean, "sample_sd": sd,
                            "CV_percent": 100*sd/mean if metric != "r2" and mean > 0 else None,
                            "ci95_lower": mean-half, "ci95_upper": mean+half}.items():
            audit.close(row[name], value, "Independent secondary summary: " + metric + "/" + name)
    B, seed = plan["analysis"]["bootstrap_replicates"], plan["analysis"]["bootstrap_seed"]
    bootstrap_means, bootstrap_dz = independent_bootstrap(scores, B, seed)
    pairs = read_frame(folder / "paired_comparisons.csv").set_index(["model_a", "model_b"])
    require(len(pairs) == 3 and set(pairs.index) == set(combinations(MODELS, 2)), "Exactly three definitive comparisons")
    pvalues, stats_by_pair, warnings = [], [], 0
    for j, ((a, b), (ai, bi)) in enumerate(zip(combinations(MODELS, 2), combinations(range(3), 2))):
        difference = scores[:, ai]-scores[:, bi]
        mean, sd = float(difference.mean()), float(difference.std(ddof=1))
        se = sd/np.sqrt(n)
        half95, half_family = stats.t.isf(.025, n-1)*se, stats.t.isf(alpha/2, n-1)*se
        row = pairs.loc[(a, b)]
        defined = bootstrap_dz[j, np.isfinite(bootstrap_dz[j])]
        if sd > 0:
            test = stats.ttest_1samp(difference, 0.)
            statistic, p = float(test.statistic), float(test.pvalue)
            logp = float(np.log(2)+stats.t.logsf(abs(statistic), n-1))
            shapiro = stats.shapiro(difference)
            W, normal_p = float(shapiro.statistic), float(shapiro.pvalue)
        else:
            statistic = p = logp = W = normal_p = None
        expected = {"history_days": 7, "n_pairs": n, "mean_difference_A_minus_B": mean, "paired_SD": sd,
                    "mean_difference_ci95_lower": mean-half95, "mean_difference_ci95_upper": mean+half95,
                    "family_CI_level": 1-alpha, "mean_difference_family_ci_lower": mean-half_family,
                    "mean_difference_family_ci_upper": mean+half_family, "t_statistic": statistic, "df": n-1,
                    "log_p_value": logp, "paired_Cohen_d_z": mean/sd if sd > 0 else None,
                    "mean_difference_bootstrap95_lower_sensitivity": float(np.quantile(bootstrap_means[j], .025)),
                    "mean_difference_bootstrap95_upper_sensitivity": float(np.quantile(bootstrap_means[j], .975)),
                    "d_z_bootstrap95_lower_sensitivity": float(np.quantile(defined, .025)) if len(defined) else None,
                    "d_z_bootstrap95_upper_sensitivity": float(np.quantile(defined, .975)) if len(defined) else None,
                    "bootstrap_replicates": B, "bootstrap_defined_effect_replicates": len(defined),
                    "shapiro_W": W, "shapiro_p_value": normal_p, "delta_absolute": delta}
        for name, value in expected.items():
            audit.close(row[name], value, "Independent definitive inference: " + name, rtol=2e-8)
        audit.close(row.p_value, p, "Independent primary t p-value", rtol=2e-8, atol=0.)
        require(bool(row.p_underflow) == (p == 0.) if p is not None else not bool(row.p_underflow), "Primary p-value underflow flag")
        practical = ("A_lower_error_beyond_delta_by_family_CI" if mean+half_family < -delta else
                     "B_lower_error_beyond_delta_by_family_CI" if mean-half_family > delta else
                     "practical_advantage_beyond_delta_not_demonstrated")
        require(row.practical_interpretation == practical and bool(row.zero_null_and_practical_boundary_are_distinct),
                "Null-zero significance is separate from practical delta")
        require(row.status == ("paired_t_primary_predeclared_conditional_seed_mean" if sd > 0
                               else "degenerate_zero_SD_no_paired_t_claim"), "Predeclared primary test status")
        expected_warning = bool(normal_p < .05) if normal_p is not None else None
        if expected_warning is None:
            require(pd.isna(row.normality_warning_p_lt_005), "Undefined normality flag")
        else:
            require(bool(row.normality_warning_p_lt_005) == expected_warning, "Normality diagnostic flag")
            warnings += int(expected_warning)
        pvalues.append(1. if p is None else p)
        stats_by_pair.append((a, b, p))
        audit.counters["definitive_primary_contrasts_checked"] += 1
    rejection, adjusted, _, _ = multipletests(pvalues, alpha=.05, method="holm")
    rejected = 0
    for (a, b, original_p), corrected, reject in zip(stats_by_pair, adjusted, rejection):
        row = pairs.loc[(a, b)]
        audit.close(row.p_holm3, float(corrected) if original_p is not None else None,
                    "Independent statsmodels Holm3", rtol=2e-8, atol=0.)
        if original_p is None:
            require(pd.isna(row.reject_zero_null_holm3), "Undefined degenerate test decision")
        else:
            require(bool(row.reject_zero_null_holm3) == bool(reject), "Predeclared Holm3 decision")
            rejected += int(reject)
    for name, value in {"R_fixed_before_definitive": n, "fresh_definitive_trajectories": 3*n,
                        "status": "definitive_conditional_seed_analysis_complete",
                        "primary_comparisons": 3, "bootstrap_replicates_sensitivity": B,
                        "bootstrap_seed": seed, "reject_zero_null_pairs": rejected,
                        "normality_warning_pairs": warnings,
                        "lowest_mean_model_descriptive": MODELS[int(np.argmin(scores.mean(axis=0)))],
                        "family_CI_individual_adjusted_level": 1-alpha,
                        "family_CI_level": 1-alpha, "alpha_global": .05,
                        "delta_absolute": delta,
                        "family_CI_simultaneous_minimum_coverage": .95,
                        "planning_power_target_not_achieved_power": .8}.items():
        require(summary[name] == value, "Independent final summary: " + name)
    return {"primary_pairs": 3, "R_fixed_before_definitive": n, "Holm3_rejected_zero_null_pairs": rejected,
            "normality_diagnostic_warning_pairs": warnings, "paired_bootstrap_replicates": B,
            "independent_implementation": "SciPy one-sample paired differences and statsmodels Holm; analyzer not imported",
            "new_market_or_advanced_boundary_claim": False}


def main():
    if not __debug__:
        raise RuntimeError("Run this independent auditor without Python optimization (-O)")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["variance_extension", "definitive"], required=True)
    args = parser.parse_args()
    audit = Audit()
    plan, context = registered_inputs(audit)
    calendar, features, targets = development(audit)
    lock, planning = planning_checks(audit, plan, calendar, targets)
    if args.phase == "definitive":
        require(type(lock["common_R"]) is int and lock["common_R"] >= 30, "Identifiable fixed definitive R")
        require(lock["stage_status"] in {"design_locked_ready_for_new_seed_runs", "blocked_budget_and_plan_complete"},
                "Complete, identifiable R design; budget overrides are an execution authorization matter")
        # The first publication is also checked during the final audit.
        first_publication = publication(audit, context, "variance_extension")
        recheck_extension_starts(audit, plan, context, first_publication, lock)
        receipt = publication(audit, context, args.phase, lock)
        seeds = [50000001+2*i for i in range(lock["common_R"])]
        selected_lock = lock
    else:
        receipt = publication(audit, context, args.phase)
        seeds, selected_lock = EXTENSION_SEEDS, None
    expected = fold_expectations(audit, OUT / args.phase, calendar, features, targets)
    runs, execution = trajectories(audit, args.phase, plan, context, calendar, targets,
                                  expected, seeds, receipt, selected_lock)
    final = final_checks(audit, plan, lock, runs, execution) if args.phase == "definitive" else None
    for name, expected_hash in {**plan["source_sha256"], **plan["code_sha256"]}.items():
        require(audit.sha(ROOT / name) == expected_hash, "Input remained unchanged during audit: " + name)
    audit.counters["nonfinite_library_power_quadrature_crosschecks"] = POWER_QUADRATURE_FALLBACKS
    report = {"status": "PASS", "study_id": STUDY, "phase": args.phase,
              "audited_at_utc": datetime.now(timezone.utc).isoformat(),
              "protocol_sha256": audit.sha(PLAN), "locked_design_sha256": audit.sha(OUT / "locked_design.json"),
              "auditor_code_sha256": audit.sha(Path(__file__)), "counters": dict(audit.counters),
              "unique_files_hashed": len(audit.hash_cache), "planning_cross_check": planning,
              "final_inference_cross_check": final,
              "registration_publication_precedes_every_selected_phase_trajectory": True,
              "published_commits_contain_registered_protocol_and_code": True,
              "design_publication_precedes_every_definitive_trajectory": args.phase == "definitive",
              "remote_publication_evidence": "Attested by timestamped publication receipts; commit contents and chronology checked locally; no remote fetch by auditor",
              "original_sixty_trajectory_pilot_registered_records_and_aggregates_unchanged": True,
              "scientific_scope": "Fresh training randomizations conditional on the known2024 calendar and fixed D7 recipes; not independent market generalization or repair of the historical benchmark",
              "new_model_fits_by_auditor": 0, "checkpoints_deserialized": 0, "new_forward_calls": 0,
              "raw_market_archives_opened": 0, "old_test_individual_files_opened": 0,
              "September_October_2026_data_opened": 0,
              "publication_scope": "Counters, hashes and aggregate planning/inference checks only; no individual prices, predictions, scaler values or weights",
              "verified_output_sha256": {name: audit.sha(OUT / args.phase / name) for name in
                  ("runs.csv", "fold_metrics.csv", "predictions.csv", "execution_summary.json")}}
    report["counters"] = dict(audit.counters)
    report["unique_files_hashed"] = len(audit.hash_cache)
    stem = ROOT / "references/audits" / f"revision_filipe_confirmatory_d7_20261105_{args.phase}"
    markdown = ("# Auditoría independiente de la comparación D7\n\n"
        f"**PASS · {args.phase}.** Se verificaron protocolo, código sellado, publicación registrada, cohortes causales, "
        "preprocesamiento, artefactos y métricas reconstruidas.\n\n"
        f"- {audit.counters['seed_trajectories_checked']} trayectorias nuevas; "
        f"{audit.counters['monthly_segments_checked']} segmentos mensuales y "
        f"{audit.counters['unique_validation_predictions_checked']} predicciones de validación.\n"
        f"- R independiente con cota superior marginal normal de DE: {planning['common_R']}; "
        "las veinte semillas nuevas fijan R y las veinticinco quedan fuera de la fase definitiva.\n\n"
        "Los recibos atestiguan la publicación remota; el auditor verifica localmente sus commits, hashes y cronología. "
        "No ajusta redes, no carga checkpoints y no accede a ZIP de mercado, datos individuales del test anterior "
        "o septiembre–octubre de 2026. La inferencia se condiciona al calendario conocido de 2024; "
        "no acredita generalización temporal nueva, potencia lograda o conjunta, ni una mejora avanzada superior al 5 %.\n")
    stem.parent.mkdir(parents=True, exist_ok=True)
    targets_report = [stem]
    if args.phase == "definitive":
        targets_report.append(stem.parent / "revision_filipe_confirmatory_d7_20261105")
    for target in targets_report:
        target.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
        target.with_suffix(".md").write_text(markdown)
    print(json.dumps({"status": "PASS", "phase": args.phase, "counters": dict(audit.counters),
                      "planning": planning, "final_inference": final}, ensure_ascii=False))


if __name__ == "__main__":
    main()


