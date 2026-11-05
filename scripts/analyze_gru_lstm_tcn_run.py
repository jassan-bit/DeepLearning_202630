"""Analyze a fresh portable LSTM/GRU/TCN reproduction from its stored CSVs.

This entry point never calls the historical publication builder, fits a model,
loads a checkpoint or reads raw market files. It has no private snapshot/schema
dependency. Analyses of the known2024calendar and observed2025--August2026test
are exploratory; training-seed intervals do not establish market generalization.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy import stats

from build_gru_lstm_tcn_evidence import (
    MODELS, HISTORIES, SEEDS, KEY, METRICS,
    validate_predictions, metric_tables, summarize, pair_table,
)
from gru_lstm_tcn_diagnostics import diagnose_phase
from run_gru_lstm_tcn import clean, digest, project_path, verify_mapping, write_json

SCOPE = ("Exploratory analysis of a post-publication reproduction on previously known market dates; "
         "paired seed comparisons are conditional on that calendar, not independent temporal confirmation.")


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_plan(project_root, output_dir, phase):
    root, out = Path(project_root).resolve(), Path(output_dir).resolve()
    path = out / "protocol.json"
    plan = read_json(path)
    if (set(plan.get("models", {})) != set(MODELS) or plan.get("histories") != list(HISTORIES)
            or plan.get("seeds") != list(SEEDS)):
        raise ValueError("The analysis requires the registered three-model/four-history/ten-seed grid")
    if plan.get("code_sha256"):
        verify_mapping(plan["code_sha256"], root)
    sources = [path]
    selected = None
    freeze_path = out / "final_evaluation/freeze.json"
    if freeze_path.is_file():
        freeze = read_json(freeze_path)
        final_protocol = out / "final_evaluation/protocol.json"
        if digest(final_protocol) != freeze.get("protocol_sha256"):
            raise ValueError("The final protocol differs from its configuration freeze")
        if (freeze.get("test_selection_allowed") is not False
                or freeze.get("test_already_observed") is not True
                or read_json(final_protocol).get("test_already_observed") is not True):
            raise ValueError("The old-test role or prohibition of test selection changed")
        selected = freeze["selection"]["by_model"]
        if (set(selected) != set(MODELS)
                or any(row["model"] != family or row["history_days"] not in HISTORIES
                       for family, row in selected.items())):
            raise ValueError("Invalid CV-frozen model/history selections")
        sources += [freeze_path, final_protocol]
    elif phase == "test":
        raise FileNotFoundError("Test analysis requires the pre-test final configuration freeze")
    return plan, selected, sources


def validate_execution(output_dir, phase):
    out = Path(output_dir).resolve()
    prefix = out / ("results" if phase == "cv" else "final_evaluation/test")
    path = prefix / "execution_summary.json"
    execution = read_json(path)
    if phase == "cv":
        if (execution.get("trajectories") != 120 or execution.get("completed_project_fits") != 2880
                or execution.get("all_registered_trajectories_complete") is not True
                or execution.get("test_files_opened") != 0):
            raise ValueError("Complete all120CV trajectories and2880fits before analysis")
        extra = []
    else:
        if (execution.get("model_runs") != 120 or execution.get("common_prediction_dates") != 608
                or execution.get("predictions") != 72960 or execution.get("monthly_reporting_blocks") != 20
                or execution.get("test_period_model_refits") != 0
                or execution.get("actual_checkpoint_forward_calls") != 120
                or execution.get("test_already_observed") is not True
                or execution.get("test_used_for_configuration_selection") is not False):
            raise ValueError("A fresh reproduction requires120frozen model scores on608dates, with no test refits/selection")
        model_manifest = out / "final_evaluation/final_models_manifest.json"
        if digest(model_manifest) != execution.get("final_models_manifest_sha256"):
            raise ValueError("The score manifest differs from its declared frozen models")
        extra = [model_manifest]
    return prefix, [path, *extra]


def validate_stored_metrics(runs, monthly, annual, by_month, phase):
    runs, monthly = runs.copy(), monthly.copy()
    expected = {(d, m, s) for d in HISTORIES for m in MODELS for s in SEEDS}
    for frame in (runs, monthly):
        frame["history_days"] = frame.history_days.astype(int)
        frame["seed"] = frame.seed.astype(int)
    if (len(runs) != 120 or runs.duplicated(KEY).any()
            or set(runs[KEY].itertuples(index=False, name=None)) != expected):
        raise ValueError("Stored run metrics do not describe exactly120registered jobs")
    n_months = 12 if phase == "cv" else 20
    calculated = annual.loc[annual.model.isin(MODELS)]
    calculated_monthly = by_month.loc[by_month.model.isin(MODELS)]
    monthly_keys = KEY + ["fold_id"]
    if (len(monthly) != 120 * n_months or monthly.duplicated(monthly_keys).any()
            or set(monthly[monthly_keys].itertuples(index=False, name=None))
            != set(calculated_monthly[monthly_keys].itertuples(index=False, name=None))):
        raise ValueError("Stored monthly reporting intervals are incomplete or duplicated")
    for stored, recomputed, keys in ((runs, calculated, KEY), (monthly, calculated_monthly, monthly_keys)):
        joined = recomputed.merge(stored, on=keys, suffixes=("_computed", "_stored"), validate="one_to_one")
        for metric in METRICS:
            if not np.allclose(joined[metric + "_computed"], joined[metric + "_stored"],
                               rtol=1e-9, atol=1e-12, equal_nan=True):
                raise ValueError("Stored metric differs from the independent forecast calculation: " + metric)


def choose_cv_histories(annual):
    selected = {}
    for model in MODELS:
        part = annual.loc[annual.model.eq(model)].groupby("history_days", as_index=False).rmse.mean()
        minimum = float(part.rmse.min())
        history = int(part.loc[(part.rmse - minimum).abs().le(1e-12), "history_days"].min())
        selected[model] = {"model": model, "history_days": history, "cv_rmse_mean": minimum}
    return selected


def friedman(annual, phase):
    arrays = [annual.loc[annual.history_days.eq(7) & annual.model.eq(model)].sort_values("seed").rmse.to_numpy()
              for model in MODELS]
    if any(len(values) != 10 for values in arrays):
        raise ValueError("Friedman requires the same ten seed blocks")
    if np.allclose(arrays, arrays[0], rtol=0, atol=1e-15):
        statistic, pvalue, status = np.nan, np.nan, "all_models_tied_undefined"
    else:
        result = stats.friedmanchisquare(*arrays)
        statistic, pvalue, status = float(result.statistic), float(result.pvalue), "chi_square_approximation"
    return pd.DataFrame([{"phase": phase, "history_days": 7, "n_seed_blocks": 10,
        "n_models": 3, "statistic": statistic, "p_value_chi2_approx": pvalue,
        "degrees_of_freedom": 2, "status": status, "scope": SCOPE}])


def analyze_phase(project_root, output_dir, phase):
    root, out = Path(project_root).resolve(), Path(output_dir).resolve()
    _, selected, source_paths = validate_plan(root, out, phase)
    prefix, execution_paths = validate_execution(out, phase)
    paths = {name: prefix / f"{name}.csv" for name in ("runs", "predictions", "fold_metrics")}
    predictions = validate_predictions(pd.read_csv(paths["predictions"]), phase)
    annual, monthly = metric_tables(predictions)
    validate_stored_metrics(pd.read_csv(paths["runs"]), pd.read_csv(paths["fold_metrics"]), annual, monthly, phase)
    if phase == "cv":
        from_cv = choose_cv_histories(annual)
        if selected is not None and any(selected[model]["history_days"] != from_cv[model]["history_days"] for model in MODELS):
            raise ValueError("The final freeze disagrees with the recorded CV history selection")
        selected = from_cv
    # Test always uses the pre-existing final freeze, irrespective of test ranks.
    summary_table = summarize(annual)
    selected_metrics = pd.concat([summary_table.loc[summary_table.model.eq(model)
        & summary_table.history_days.eq(selected[model]["history_days"])] for model in MODELS], ignore_index=True)
    diagnostics = diagnose_phase(predictions, phase)
    outputs = {"run_metrics.csv": annual, "monthly_run_metrics.csv": monthly,
        "metric_summary.csv": summary_table, "monthly_summary.csv": summarize(monthly, monthly=True),
        "pairs_D7.csv": pair_table(annual, phase, (7,)),
        "pairs_allD.csv": pair_table(annual, phase, HISTORIES),
        "friedman_D7.csv": friedman(annual, phase), "selected_by_cv_metrics.csv": selected_metrics,
        **{name + ".csv": frame for name, frame in diagnostics.items()}}
    destination = out / "reproduction_analysis" / phase
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        if "scope" in frame:
            frame = frame.copy()
            frame["scope"] = SCOPE
        frame.to_csv(destination / name, index=False)
    source_paths += execution_paths + list(paths.values())
    code_names = ("scripts/analyze_gru_lstm_tcn_run.py", "scripts/build_gru_lstm_tcn_evidence.py",
                  "scripts/gru_lstm_tcn_diagnostics.py")
    sources = {path.relative_to(out).as_posix(): digest(path) for path in source_paths}
    code = {name: digest(project_path(root, name)) for name in code_names}
    summary = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "phase": phase,
        "scope": SCOPE, "models": list(MODELS), "histories": list(HISTORIES), "seeds": list(SEEDS),
        "model_runs": 120, "prediction_rows": len(predictions),
        "dates_per_run": 366 if phase == "cv" else 608, "monthly_reporting_intervals": 12 if phase == "cv" else 20,
        "D7_pair_family_size": 3, "allD_pair_family_size": 12,
        "cv_selected_by_model": selected, "winner_selected_on_test": False,
        "model_or_history_selection_on_test": False, "test_already_observed": True,
        "diagnostic_rows": {name: len(frame) for name, frame in diagnostics.items()},
        "training_performed": False, "checkpoint_inference_performed": False,
        "raw_data_opened": False, "publication_performed": False,
        "limitation": "Ten fixed training seeds are not a prospective80percent power calculation; monthly test groups are reporting intervals, not retraining folds"}
    write_json(destination / "summary.json", summary)
    write_json(destination / "source_manifest.json", {"phase": phase, "source_root": "--output-dir",
        "code_root": "--project-root", "source_sha256": sources, "code_sha256": code, "scope": SCOPE})
    artifacts = {path.name: digest(path) for path in destination.iterdir()
                 if path.is_file() and path.name != "output_manifest.json"}
    write_json(destination / "output_manifest.json", {"artifact_sha256": artifacts, "scope": SCOPE})
    return summary


def analyze(project_root, output_dir, phase="all"):
    if phase not in ("cv", "test", "all"):
        raise ValueError("phase must be cv, test or all")
    phases = ("cv", "test") if phase == "all" else (phase,)
    return {"phases": {name: analyze_phase(project_root, output_dir, name) for name in phases}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--phase", choices=("cv", "test", "all"), default="all")
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = args.output_dir.resolve() if args.output_dir else root / "results/gru_lstm_tcn_reproduction"
    print(json.dumps(clean(analyze(root, out, args.phase)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
