"""Registered prefix/future-only audit of the three frozen classical E2 ports.

No fit, selection, permutation of observed endogenous history, scoring or test
is supplied. Prepare after software checks; run after independent review through
the additive scientific budget wrapper. Every causal ARIMA forecast is persisted
by the original helper before its corresponding observation is revealed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import pickle
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "book_e2/_static/e2_scientific_closure_20261116/classical_causality"
VERSION = "e2-frozen-classical-causality-v1"
PREFIXES = (1, 7, 31, 90, 365, 608)
NAMES = ("Persistence", "SeasonalNaive", "ARIMA")
PACKAGES = {"statsmodels": "0.14.5", "numpy": "2.3.5", "pandas": "2.2.3", "scipy": "1.16.3"}
ENGINE_FILES = {
    "scripts/e2_temporal_controls.py": "36e5a8b15e51ddf7d36397e2b6e71734afb36c6894cac0a71c9364b277fcc139",
    "scripts/e2_reduced_final.py": "033e4bf7db0d4f69624063830d116a1b5cf159a401c310a79979aa46307dad19",
}
DAILY_SHA = "9563ef557e180f34ee37dce3b1480c38aaefd7063df21c031644973310e28deb"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def file_sha(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def binding(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), "Regular input required: " + str(path))
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha(path)}


def verify(record):
    require(binding(record["path"]) == record, "Frozen input changed: " + record["path"])


def prepare(args):
    out = args.output.absolute()
    require(not out.exists() or (out.is_dir() and
            {p.name for p in out.iterdir()} <= {"unit_checks.log", "unit_checks.json"}),
            "Use an unregistered directory containing at most the unit receipt/log")
    audit = read_json(args.data_audit)
    require(audit["actual_cohorts_audited"] == {"training": 655, "calibration": 53, "evaluation": 608}
            and audit["data_ready_for_bounded_controls"] is True,
            "The hash-verified actual 655/53/608 data admission is required")
    arrays = ROOT / audit["arrays_npz_path"]
    require(file_sha(arrays) == audit["arrays_npz_sha256"], "Audited full-cohort NPZ changed")
    daily = ROOT / "work/e2-scientific-market/known_E1_daily.csv"
    require(file_sha(daily) == DAILY_SHA, "Use the original hash-pinned daily evaluation cache")
    tests = read_json(args.unit_receipt)
    require(tests["status"] == "passed" and tests["returncode"] == 0 and tests["tests_run"] >= 7
            and tests["runner_sha256"] == file_sha(__file__), "Prior unit receipt must bind this runner")
    verify(tests["test_source"])
    verify(tests["log"])
    frozen_path = ROOT / "book_e2/_static/e2_benchmark_final_20261113/final_freeze.json"
    frozen = {r["model"]: r for r in read_json(frozen_path)["models"] if r["model"] in NAMES}
    index = read_json(args.checkpoint_index)
    require(index["status"] == "verified" and index["scientific_commit"] == "1e8745497b484518f766b6da5c19be8362d9add7", "Verified fixed-source recovery index required")
    rows = [r for r in index["rows"] if r["model"] in NAMES]
    require(len(rows) == 3 and {r["model"] for r in rows} == set(NAMES)
            and all(r["seed"] == "deterministic" for r in rows), "Exactly three deterministic classical predictors required")
    identities = {}
    for row in rows:
        name = row["model"]
        for kind in ("checkpoint", "scaler", "test_predictions"):
            require(file_sha(row[kind + "_path"]) == row["expectedsha"][kind], "Recovered file changed: " + name + "/" + kind)
        require(file_sha(row["source_result"]) == row["source_result_sha256"], "Historical reproduction receipt changed")
        if name != "ARIMA":
            config = read_json(row["checkpoint_path"])
            require(config == {"model": name, "lag": 1 if name == "Persistence" else 7, "parameters_fixed": True}, "Frozen naive lag changed")
            require(row["expectedsha"]["checkpoint"] == frozen[name]["checkpoint_sha256"], "Naive checkpoint differs from original freeze")
        identities[name] = {"checkpoint": binding(row["checkpoint_path"]), "scaler": binding(row["scaler_path"]),
            "saved_test_predictions": binding(row["test_predictions_path"]),
            "historical_reproduction_receipt": binding(row["source_result"]),
            "original_freeze_checkpoint_sha256": frozen[name]["checkpoint_sha256"],
            "checkpoint_bytes_match_original_freeze": row["expectedsha"]["checkpoint"] == frozen[name]["checkpoint_sha256"],
            "lineage": "Recovered clean-reproduction ARIMA pickle, with its own physical SHA; historical receipt records parameter and prediction agreement; never relabelled as original frozen pickle" if name == "ARIMA" else "Recovered JSON bytes equal original frozen JSON"}
    source = args.source_root.absolute()
    source_bindings = {}
    for relative, expected in ENGINE_FILES.items():
        record = binding(source / relative)
        require(record["sha256"] == expected, "Original classical scientific source changed")
        source_bindings[relative] = record
    for relative in ("scripts/e2_reduced_protocol.py", "scripts/e2_reduced_fit.py"):
        source_bindings[relative] = binding(source / relative)
    alias = ROOT / "scripts/e2_arima_prediction_compat.py"
    plan = {"version": VERSION, "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner": binding(__file__), "unit_receipt": binding(args.unit_receipt),
        "data_audit": binding(args.data_audit), "arrays": binding(arrays), "array_keys": audit["array_keys"],
        "daily_evaluation_csv": binding(daily), "development_daily_csv": binding(ROOT / "results/eda_development/daily_market_development.csv"),
        "original_freeze": binding(frozen_path), "recovery_index": binding(args.checkpoint_index),
        "scientific_source_root": str(source), "scientific_source_files": source_bindings,
        "existing_read_only_raw_name_adapter": binding(alias), "predictors": identities,
        "model_order": list(NAMES), "prefix_lengths": list(PREFIXES),
        "fixed_origins": [(date(2025, 1, 1) + timedelta(days=n - 1)).isoformat() for n in PREFIXES],
        "scope": "New post hoc causal implementation audit on actual inherited E1 development/evaluation chronology; no new blind sample, fitting, selection, performance score, calibration or statistical test",
        "data_construction": "Read development OHLCV with round-trip float parsing (matching registered Python float decoding), evaluation daily CSV with the original default pandas parser; reconstruct population sigma7 using the pinned controls.sigma7 and require exact evaluation y/raw-window agreement with the full audited NPZ. Training/calibration saved target labels are checked at the original reconstruction tolerance, not silently substituted into the causal history.",
        "branches": {"reference": "Original entire dated history; forecast sequentially from the recovered state through August31/2026",
            "truncated": "For each registered t, history contains dates only through t-1; use fresh identical checkpoint state; retain every forecast issued through t",
            "future_only_perturbed": "For each registered t, replace finite sigma7 labels at dates >=t by1.37*y+0.0001; retain past dates and values exactly; use fresh identical checkpoint state; retain every forecast issued through t",
            "ARIMA_bridge": "The checkpoint ends calibration on December30/2024; the original helper emits the December31 bridge forecast before observing that bridge label, then evaluates January1 onward",
            "endogenous_history": "Past sigma7 observations are legitimate model inputs. No permutation, deletion or alteration of any observed value strictly before emission t."},
        "criteria": {"forecast_prefixes": "Exact FP64 equality for every issued evaluation forecast in each prefix; same parameters, past observations, serial arithmetic and backend; no GPU comparison or accumulation-order change",
            "absolute_tolerance": 0., "relative_tolerance": 0.,
            "parameters_scale_order": "Exact unchanged fingerprint of actual statsmodels params, receipt params, order and training scale for every branch; filtered state is expected to advance causally",
            "same_day_target": "Forecast receipt is persisted by original _arima_walk before observe(t,y[t]); the truncated branch may observe NaN after issuing its final forecast",
            "saved_predictions": "Publish original saved608 values, new causal replay values and differences descriptively only; no parity threshold, replacement of original predictions or new predictive score",
            "failures": "Preserve/report all failures; no refit, repair, new order, tolerance adjustment, exclusions or backend retry"},
        "runtime_policy": {"packages": PACKAGES, "device": "cpu", "dtype": "float64",
            "blas_threads": 2, "fit_guards": ["FrozenARIMA.fit", "statsmodels.tsa.arima.model.ARIMA.fit"]},
        "budget": {"cap_seconds": 180, "ledger": "book_e2/_static/e2_scientific_closure_20261116/compute_ledger.jsonl"},
        "prohibited_operations": ["coefficient fitting", "ARIMA order selection", "changed naive lag", "permuting observed endogenous labels", "performance scoring", "bootstrap", "hypothesis testing", "changing original campaign evidence"]}
    out.mkdir(parents=True, exist_ok=True)
    write_new(out / "plan.json", plan)
    return {"status": "registered_before_forecasts", "plan_sha256": file_sha(out / "plan.json"),
            "weights_deserialized": False, "forecasts": 0}


def prefix_variant(history, target, branch):
    if branch == "truncated":
        return history.loc[history.index < target].copy()
    require(branch == "future_only_perturbed", "Unknown fixed branch")
    result = history.copy()
    mask = (result.index >= target) & result.notna()
    result.loc[mask] = result.loc[mask] * 1.37 + .0001
    return result


def exact_prefix(reference, observed, dates, np):
    expected = np.asarray([reference[d] for d in dates], dtype=np.float64)
    actual = np.asarray([observed[d] for d in dates], dtype=np.float64)
    require(expected.shape == actual.shape and bool(np.isfinite(expected).all() and np.isfinite(actual).all()), "Missing/nonfinite forecast in fixed prefix")
    return {"passed": bool(np.array_equal(expected, actual)), "count": len(dates),
            "maximum_absolute_difference": float(np.max(np.abs(expected - actual)))}


def parameter_record(model, np):
    return {"actual_result_parameters": np.asarray(model._result.params, dtype=np.float64).tolist(),
            "receipt_parameters": model.receipt["parameters"], "scale": float(model.scale),
            "order": model.receipt["order"], "fit_cutoff": model.receipt["fit_cutoff"]}


def forbid_fit(*args, **kwargs):
    raise RuntimeError("No fit is permitted in this frozen classical causality audit")


def history_from_inputs(plan, pd, np, controls):
    dev = pd.read_csv(plan["development_daily_csv"]["path"], index_col="market_day_utc", float_precision="round_trip")
    evaluation = pd.read_csv(plan["daily_evaluation_csv"]["path"], index_col="market_day_utc")
    for frame in (dev, evaluation):
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
    columns = ["open", "high", "low", "close", "volume"]
    full = pd.concat((dev.loc[:, columns], evaluation.loc[:, columns]))
    require(full.index.equals(pd.date_range("2023-01-01", "2026-08-31", tz="UTC")), "Full chronological calendar changed")
    quality = pd.Series(np.isfinite(full.to_numpy()).all(axis=1), index=full.index)
    history = controls.sigma7(full.close, quality)
    with np.load(plan["arrays"]["path"], allow_pickle=False) as arrays:
        records = []
        for split, count in (("training", 655), ("calibration", 53), ("evaluation", 608)):
            keys = plan["array_keys"][split]
            dates = pd.DatetimeIndex(pd.to_datetime(arrays[keys["dates"]], utc=True))
            require(len(dates) == count, "Cohort count changed")
            positions = full.index.get_indexer(dates)
            windows = full.to_numpy()[positions[:, None] - np.arange(7, 0, -1)]
            require(np.array_equal(windows, arrays[keys["raw"]]), "Raw windows differ from full audited cohort: " + split)
            actual = history.reindex(dates.tz_localize(None)).to_numpy()
            stored = arrays[keys["y_original"]]
            if split == "evaluation":
                require(np.array_equal(actual, stored), "Reconstructed evaluation target history differs from audited targets")
            else:
                require(np.allclose(actual, stored, atol=1e-13, rtol=1e-9), "Development target reconstruction differs from admitted tolerance")
            records.append({"split": split, "count": count, "raw_windows_exact": True,
                            "maximum_target_absolute_difference": float(np.max(np.abs(actual - stored))),
                            "targets_exact": bool(np.array_equal(actual, stored))})
    return history, records


def execute(args):
    out = args.output.absolute()
    plan_path = out / "plan.json"
    require(args.plan_sha256 == file_sha(plan_path), "Exact plan SHA is required")
    plan = read_json(plan_path)
    require(plan["version"] == VERSION and not (out / "progress.jsonl").exists(), "Use the registered, unexecuted plan")
    for key in ("runner", "unit_receipt", "data_audit", "arrays", "daily_evaluation_csv", "development_daily_csv", "original_freeze", "recovery_index", "existing_read_only_raw_name_adapter"):
        verify(plan[key])
    for record in plan["scientific_source_files"].values():
        verify(record)
    for row in plan["predictors"].values():
        for key in ("checkpoint", "scaler", "saved_test_predictions", "historical_reproduction_receipt"):
            verify(row[key])
    started = time.monotonic()
    log = (out / "progress.jsonl").open("x", encoding="utf-8")
    checks, runtime, parameters = [], None, None
    try:
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = "2"
        actual_packages = {name: version(name) for name in PACKAGES}
        require(actual_packages == PACKAGES, "Use the pinned classical CPU package versions")
        import numpy as np
        import pandas as pd
        sys.path.insert(0, str(Path(plan["scientific_source_root"]) / "scripts"))
        import e2_reduced_final as final
        import e2_temporal_controls as controls
        # This original, pre-existing alias changes receipt naming only.
        adapter_spec = __import__("importlib.util", fromlist=["spec_from_file_location"])
        module_spec = adapter_spec.spec_from_file_location("_e2_registered_arima_raw_alias", plan["existing_read_only_raw_name_adapter"]["path"])
        adapter = adapter_spec.module_from_spec(module_spec)
        module_spec.loader.exec_module(adapter)
        alias_receipt = adapter.install_prediction_raw_alias(plan["scientific_source_root"], controls)
        from statsmodels.tsa.arima.model import ARIMA
        controls.FrozenARIMA.fit = classmethod(forbid_fit)
        ARIMA.fit = forbid_fit
        runtime = {"python": platform.python_version(), "executable": sys.executable,
                   "packages": actual_packages, "device": "cpu", "dtype": "float64",
                   "thread_environment": {k: os.environ[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
                   "fit_methods_disabled": True}
        history, data_checks = history_from_inputs(plan, pd, np, controls)
        dates = pd.date_range("2025-01-01", "2026-08-31").strftime("%Y-%m-%d").tolist()
        write_new(out / "input_controls.json", {"full_cohort_checks": data_checks,
            "endogenous_history_csv_sha256": sha256(history.to_csv().encode()).hexdigest(),
            "history_start": history.index[0].isoformat(), "history_end": history.index[-1].isoformat(),
            "history_calendar_days": len(history), "raw_name_alias": alias_receipt, "runtime": runtime})
        references, lineage = {}, {}
        arima_bytes = Path(plan["predictors"]["ARIMA"]["checkpoint"]["path"]).read_bytes()

        def arima_forecasts(source_history, end, directory):
            model = pickle.loads(arima_bytes)
            require(type(model) is controls.FrozenARIMA, "Unexpected recovered ARIMA checkpoint type")
            require(model.receipt["order"] == [2, 0, 1]
                    and controls.day(model.receipt["fit_cutoff"]) == pd.Timestamp("2024-10-31")
                    and controls.day(model.current) == pd.Timestamp("2024-12-30")
                    and model._pending is None and model._failure is None, "Frozen ARIMA identity/start state changed")
            before = parameter_record(model, np)
            result = final._arima_walk(model, source_history, end, directory)
            after = parameter_record(model, np)
            require(before == after, "Frozen ARIMA coefficients, order or scale changed")
            return result, before, {"initial_current": "2024-12-30", "final_current": model.current.isoformat(),
                "filter_state_advanced": True, "coefficients_refitted": False,
                "forecast_receipts": len(result), "parameters_unchanged": True}

        for name in NAMES:
            identity = plan["predictors"][name]
            if name == "ARIMA":
                reference, parameters, trace = arima_forecasts(history, dates[-1], out / "causal_receipts/reference")
            else:
                config = read_json(identity["checkpoint"]["path"])
                reference = dict(zip(dates, final._naive_predictions(history, dates, config["lag"])))
                trace = {"lag": config["lag"], "parameters_unchanged": True, "coefficients_refitted": False}
            references[name] = reference
            lineage[name] = {**identity, "reference_trace": trace}
            for prefix, origin in zip(PREFIXES, plan["fixed_origins"]):
                target = pd.Timestamp(origin)
                for branch in ("truncated", "future_only_perturbed"):
                    changed = prefix_variant(history, target, branch)
                    past = history.loc[history.index < target]
                    require(changed.reindex(past.index).equals(past), "Perturbation changed observed endogenous history")
                    if name == "ARIMA":
                        observed, record, trace = arima_forecasts(changed, origin, out / "causal_receipts" / (branch + "_" + str(prefix)))
                        require(record == parameters, "ARIMA starting parameters differ across branches")
                    else:
                        observed = dict(zip(dates[:prefix], final._naive_predictions(changed, dates[:prefix], config["lag"])))
                        trace = {"lag": config["lag"], "parameters_unchanged": True}
                    result = {"model": name, "prefix_length": prefix, "origin": origin,
                              "branch": branch, "past_history_exactly_equal": True,
                              **exact_prefix(reference, observed, dates[:prefix], np), "trace": trace}
                    checks.append(result)
                    log.write(canonical(result).decode() + "\n")
                    log.flush()
                    print(json.dumps({k: result[k] for k in ("model", "prefix_length", "branch", "passed")}), flush=True)
        write_new(out / "prefix_checks.json", checks)
        write_new(out / "predictor_bindings.json", {"lineage": lineage, "arima_parameter_record": parameters})
        with (out / "saved_prediction_comparison.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["model", "date", "saved_prediction", "causal_replay_prediction", "replay_minus_saved", "role"], lineterminator="\n")
            writer.writeheader()
            descriptive = []
            for name in NAMES:
                saved = read_json(plan["predictors"][name]["saved_test_predictions"]["path"])
                require(saved["dates"] == dates and len(saved["prediction_original"]) == 608, "Saved trajectory calendar changed")
                differences = []
                for day, old in zip(dates, saved["prediction_original"]):
                    value = references[name][day]
                    differences.append(value - old)
                    writer.writerow({"model": name, "date": day, "saved_prediction": old,
                        "causal_replay_prediction": value, "replay_minus_saved": value - old,
                        "role": "descriptive numerical replay comparison, no new predictive score or parity decision"})
                descriptive.append({"model": name, "rows": 608, "maximum_absolute_prediction_difference": max(abs(v) for v in differences),
                                    "parity_gate_evaluated": False})
        for row in plan["predictors"].values():
            verify(row["checkpoint"])
            verify(row["scaler"])
        require(len(checks) == 36, "Incomplete registered prefix domain")
        passed = sum(r["passed"] for r in checks)
        report = {"version": VERSION, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASS" if passed == 36 else "FAIL", "plan_sha256": file_sha(plan_path),
            "scope": plan["scope"], "models": list(NAMES), "prefix_lengths": list(PREFIXES),
            "branch_checks": 36, "passed": passed, "failed": 36 - passed,
            "checked_forecast_values": sum(r["count"] for r in checks),
            "maximum_causal_prefix_difference": max(r["maximum_absolute_difference"] for r in checks),
            "arima_parameters": parameters, "checkpoint_and_scaler_files_unchanged": True,
            "saved_608_descriptive_comparison": descriptive,
            "original_ARIMA_checkpoint_bytes_recovered_claimed": False,
            "ARIMA_checkpoint_scope": plan["predictors"]["ARIMA"]["lineage"],
            "new_model_fits": 0, "new_predictive_scores": 0, "new_statistical_tests": 0,
            "original_GPU_parity_claimed": False, "elapsed_seconds": time.monotonic() - started,
            "runtime": runtime,
            "limitations": ["Six fixed emission origins, not a proof over every possible history", "Uses frozen endogenous sigma7 histories legitimately; no neural35-channel claim", "ARIMA initial pickle is recovered clean-reproduction state, not original frozen pickle bytes", "No labels-permuted fit, drift classifier or new uncertainty interval is performed"]}
        write_new(out / "audit_report.json", report)
        return report
    except Exception as error:
        write_new(out / "failure_report.json", {"version": VERSION, "status": "FAIL",
            "plan_sha256": file_sha(plan_path), "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(), "completed_branch_checks": len(checks),
            "elapsed_seconds": time.monotonic() - started, "outputs_preserved": True, "runtime": runtime})
        raise
    finally:
        log.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--unit-receipt", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--data-audit", type=Path, default=ROOT / "book_e2/_static/e2_scientific_closure_20261116/data_audit/audit_result.json")
    parser.add_argument("--checkpoint-index", type=Path, default=ROOT / "work/e2-corrections-assets/metadata/checkpoint_index.json")
    parser.add_argument("--source-root", type=Path, default=Path("/workspace/work/e2-corrections-source-20261115T222607Z"))
    args = parser.parse_args(argv)
    require(args.operation != "prepare" or args.unit_receipt is not None, "Prepare requires prior unit receipt")
    result = prepare(args) if args.operation == "prepare" else execute(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get("status") == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
