"""Posterior descriptive explanation of real train/test domain separability.

The only fitted targets are domain membership. Original forecasting models,
recipes, scalers, primary results, and test observations are read-only.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"book_e2/_static/e2_scientific_closure_20261116"
DEFAULT_OUTPUT = BASE/"drift_explanation"
SOURCE_FILES = {
    "arrays": BASE/"data_audit/current_cohort_arrays.npz",
    "adversarial_result": BASE/"controls/adversarial_result.json",
    "adversarial_specification": BASE/"controls/specification.json",
    "original_fold_manifest": BASE/"controls/adversarial_fold_manifest.json",
    "engine": ROOT/"scripts/e2_corrections_controls.py",
}
EXPECTED_ARRAYS_SHA = "186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683"
CHANNELS = ("Open", "High", "Low", "Close", "Volume")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(test, reason):
    if not test:
        raise ValueError(reason)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def register(output):
    require(not output.exists(), "Use a new output; preserve existing evidence")
    require(sha(SOURCE_FILES["arrays"]) == EXPECTED_ARRAYS_SHA, "Real cohort source changed")
    original = json.loads(SOURCE_FILES["adversarial_result"].read_text())
    require(original["status"] == "completed" and original["feature_count"] == 35,
            "Completed original full-domain diagnosis required")
    plan = {
        "version": "e2-domain-drift-explanation-v1",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "Posterior exploratory descriptive diagnosis after the full adversarial AUC and permutation result were known",
        "source_records": {name: {"path": path.relative_to(ROOT).as_posix(),
                                 "sha256": sha(path), "bytes": path.stat().st_size}
                           for name, path in SOURCE_FILES.items()},
        "script_sha256": sha(__file__),
        "groups": {"full_OHLCV_35": list(range(35)),
                   "relative_prices_OHLC_28": [i for i in range(35) if i % 5 < 4],
                   "log1p_volume_7": [i for i in range(35) if i % 5 == 4]},
        "cohorts": {"train_domain": 655, "test_domain": 608, "calibration_excluded": True},
        "classification_target": "domain membership only:0training-period,1evaluation-period",
        "forbidden_inputs": ["predictive target arrays", "current/future target as feature", "date as classifier feature"],
        "folds": "Identical five purged calendar folds,7daypurge,training-fold-only scaling from original engine",
        "classifier": "Same balanced L2logistic routine:lambda0.01,L-BFGS-B,maxiter200,gtol1e-7,ftol1e-12",
        "primary_replay_guard": {"full_AUC_reference": original["AUC_OOF"], "absolute_tolerance": 1e-12},
        "descriptive_features": {
            "flattening": "Corder day t-7 through t-1,then Open,High,Low,Close,Volume",
            "means_and_standard_deviations": "Bydomain,population SD ddof0,no standard errors",
            "quantile_probabilities": [0., .05, .25, .5, .75, .95, 1.],
            "univariate_AUC": "P(test feature>train feature)+half ties;orientation retained and max(AUC,1-AUC) separately descriptive",
            "standardized_mean_shift": "(testmean-trainmean)/trainSD;null for exactlyconstant trainfeature",
        },
        "coefficient_policy": "The unchanged frozen learner does not expose coefficients; none are reconstructed or reported as importance",
        "outputs": ["group_AUC_3.csv", "feature_descriptives_35.csv", "domain_calendar.json", "group_predictions.npz", "summary.json"],
        "new_confirmatory_tests": 0, "new_p_values": False, "new_bootstrap_draws": 0,
        "original_forecasting_fits": 0, "original_A2_RQ1_or_transform_changed": False,
        "interpretation_limits": [
            "A group retaining discrimination is not a causal explanation or a variance decomposition of full AUC.",
            "Overlapping windows and one chronology do not yield independent observations.",
            "Folds include later calendar rows for distribution diagnosis; this is not a causal forecasting backtest.",
            "Univariate orientation and differences between the three AUC values are descriptive, not significance claims.",
            "Shift can coexist with valid causal splitting and is not by itself information leakage.",
        ],
    }
    output.mkdir(parents=True)
    write_json(output/"analysis_plan.json", plan)
    return plan


def execute(output, phase):
    import e2_corrections_controls as original
    import numpy as np
    original.require_budget(phase)
    plan_path = output/"analysis_plan.json"
    plan = json.loads(plan_path.read_text())
    require(plan["script_sha256"] == sha(__file__), "Registered program changed")
    for record in plan["source_records"].values():
        path = ROOT/record["path"]
        require(sha(path) == record["sha256"] and path.stat().st_size == record["bytes"],
                "Registered source changed: "+record["path"])
    require(not (output/"summary.json").exists(), "Do not overwrite prior results")
    started = datetime.now(timezone.utc).isoformat()
    with np.load(ROOT/plan["source_records"]["arrays"]["path"], allow_pickle=False) as arrays:
        train = arrays["training_relative"].copy()
        test = arrays["evaluation_relative"].copy()
        train_dates = arrays["training_dates"].astype("datetime64[D]")
        test_dates = arrays["evaluation_dates"].astype("datetime64[D]")
    require(train.shape == (655, 7, 5) and test.shape == (608, 7, 5),
            "Wrong real-domain cohort")
    x = np.concatenate((train, test)).reshape(-1, 35)
    require(np.isfinite(x).all(), "Nonfinite classifier inputs")
    dates = np.r_[train_dates, test_dates]
    labels = np.r_[np.zeros(655, dtype=int), np.ones(608, dtype=int)]
    group_rows, predictions, receipts = [], {}, {}
    reference_folds = json.loads(SOURCE_FILES["original_fold_manifest"].read_text())
    for name, columns in plan["groups"].items():
        folds = original.purged_folds(x[:, columns], dates)
        for fold, reference in zip(folds, reference_folds):
            require(fold["train"].tolist() == reference["training_indices"]
                    and fold["validation"].tolist() == reference["validation_indices"],
                    "Diagnostic group used different calendar folds")
        scores, receipt = original.domain_oof(folds, labels)
        value = original.auc(labels, scores)
        if name == "full_OHLCV_35":
            require(abs(value-plan["primary_replay_guard"]["full_AUC_reference"]) <= 1e-12,
                    "Full35AUC does not reproduce the original real adversarial result")
        group_rows.append({"group": name, "feature_count": len(columns),
                           "AUC_OOF_descriptive": value, "train_count": 655, "test_count": 608,
                           "folds": 5, "purge_days": 7, "new_p_value": "",
                           "scope": "exploratory domain-membership diagnosis,not forecasting accuracy or causal attribution"})
        predictions[name] = scores
        receipts[name] = receipt
    write_csv(output/"group_AUC_3.csv", group_rows)
    rows, quantiles = [], plan["descriptive_features"]["quantile_probabilities"]
    for index in range(35):
        a, b = x[:655, index], x[655:, index]
        auc = original.auc(labels, x[:, index])
        stdev = float(a.std(ddof=0))
        row = {"feature_index": index, "lag_days": 7-index//5, "channel": CHANNELS[index % 5],
               "transform": "log(price)-log(last_available_Close)" if index % 5 < 4 else "log1p(raw_volume)",
               "train_mean": float(a.mean()), "test_mean": float(b.mean()),
               "train_std_ddof0": stdev, "test_std_ddof0": float(b.std(ddof=0)),
               "standardized_mean_shift_train_scale": float((b.mean()-a.mean())/stdev) if stdev > 0 else None,
               "univariate_AUC_high_value_means_test": auc,
               "univariate_separability_max_orientation_descriptive": max(auc, 1-auc),
               "independent_samples_assumed": False}
        for probability, qa, qb in zip(quantiles, np.quantile(a, quantiles), np.quantile(b, quantiles)):
            row["train_q"+str(probability)] = float(qa)
            row["test_q"+str(probability)] = float(qb)
        rows.append(row)
    write_csv(output/"feature_descriptives_35.csv", rows)
    write_json(output/"classifier_receipts.json", receipts)
    write_json(output/"domain_calendar.json", {
        "training": {"n": 655, "first": str(train_dates[0]), "last": str(train_dates[-1])},
        "evaluation": {"n": 608, "first": str(test_dates[0]), "last": str(test_dates[-1])},
        "pooled_n": 1263, "calibration_used": False,
        "date_as_classifier_feature": False, "folds_used_future_training_rows_for_distribution_diagnosis": True})
    np.savez_compressed(output/"group_predictions.npz", dates=dates.astype(str), labels=labels, **predictions)
    summary = {
        "version": plan["version"], "status": "completed_exploratory_description",
        "plan_sha256": sha(plan_path), "group_results": group_rows,
        "full_AUC_reference_reproduced": True, "features_described": len(rows),
        "new_domain_classifier_fits": 15, "new_forecasting_model_fits": 0,
        "predictive_target_arrays_read": False, "new_p_values": False, "new_resampling": False,
        "original_drift_flag_retained": True, "original_primary_results_changed": False,
        "drift_fixed_or_causal_mechanism_identified": False,
        "limits": plan["interpretation_limits"]}
    write_json(output/"summary.json", summary)
    write_json(output/"execution_receipt.json", {
        "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha(__file__), "plan_sha256": sha(plan_path),
        "inputs": plan["source_records"], "phase": phase,
        "python": sys.version.split()[0], "numpy": np.__version__,
        "scope": "15new domain-logistic fits and descriptive covariates;no forecasting fit or confirmatory test"})
    files = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(output.iterdir()) if p.is_file()}
    write_json(output/"manifest.json", {"files": files})
    print(json.dumps({"status": summary["status"], "groups": group_rows}, ensure_ascii=False))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--register-plan", action="store_true")
    parser.add_argument("--phase", default="e2_exploratory_domain_drift_description")
    args = parser.parse_args(argv)
    if args.register_plan:
        register(args.output)
        print(json.dumps({"plan_registered": str(args.output/"analysis_plan.json")}))
    else:
        execute(args.output, args.phase)


if __name__ == "__main__":
    main()
