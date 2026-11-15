"""Rebuild E3 Block B's selection table from frozen E2 validation receipts.

This script never trains, reads market data, chooses from test scores, or selects
the best final seed. Validation and historical test metrics have distinct fields.
Three temporal folds are not treated as independent seed repetitions for a CI.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile

BASE_COMMIT = "f65b14d38e38f8d5fd80c717d71f2abd9ea708e2"
BACKUP_COMMIT = "d604ede6bf0978e14a1b1d7bafaae6250fb8ed07"
OUTPUT = Path("experiments/e3_bloques_ab_20261115")
RECEIPTS = OUTPUT / "inputs/selection_receipts"
REGISTER = Path("book_e2/_static/e2_data_training_uncertainty_current_v1/model_training_register.csv")
COMPARISON = Path("book_e2/_static/e2_benchmark_current_v1/comparison_current.csv")
A2 = Path("experiments/e2_benchmark_final_20261113/benchmark_A2.csv")
FREEZE = Path("experiments/e2_benchmark_final_20261113/final_freeze.json")
MCS = Path("experiments/e2_f2a_s8_20261113/results/mcs_membership.csv")
FOLDS = ("2024-01-01", "2024-07-01", "2024-10-01")
CLASSICAL = ("Persistence", "SeasonalNaive", "ARIMA")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_positive(value, label):
    if isinstance(value, bool):
        raise ValueError(f"Invalid {label}")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid {label}")
    return value


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-14)


def pooled_rmse(values):
    values = [finite_positive(x, "validation RMSE") for x in values]
    if len(values) != 3:
        raise ValueError("All three registered validation folds are required")
    # The registered folds contain the same 83 target dates each.
    return math.sqrt(statistics.fmean(x*x for x in values))


def unique_rows(path, key="model"):
    with Path(path).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    result = {}
    for row in rows:
        name = row[key]
        if name in result:
            raise ValueError(f"Duplicate {key}: {name}")
        result[name] = row
    return result


def select_neural(receipt):
    """Verify the recorded winner from validation only, rejecting incomplete CV."""
    if receipt.get("criteria_only") or receipt.get("fold_count") != 3:
        raise ValueError("A completed three-fold validation receipt is required")
    if receipt.get("selection_scope") != "one_global_recipe_for_post_HPO_final_cohort":
        raise ValueError("Unexpected selection scope")
    if receipt.get("validation_count") != 249:
        raise ValueError("Unexpected validation support; cannot pool unequal folds")
    ledger = receipt["candidate_ledger"]
    if len(ledger) != 6 or len({r["configuration_id"] for r in ledger}) != 6:
        raise ValueError("Six distinct registered configuration outcomes are required")
    eligible = []
    for row in ledger:
        if row.get("status") != "completed":
            continue
        folds = row.get("fold_RMSE_original", {})
        if set(folds) != set(FOLDS) or row.get("validation_count") != 249:
            raise ValueError("A completed candidate lacks common validation support")
        values = [finite_positive(folds[f], "fold RMSE") for f in FOLDS]
        score = pooled_rmse(values)
        if not close(score, row["pooled_cv_RMSE_original"]):
            raise ValueError("Pooled validation score differs from fold evidence")
        if not close(statistics.fmean(values), row["mean_fold_RMSE_original_diagnostic"]):
            raise ValueError("Arithmetic fold mean differs from evidence")
        eligible.append((score, row["configuration_id"], values, row))
    if not eligible:
        raise ValueError("No eligible candidate")
    winner = min(eligible, key=lambda x: (x[0], x[1]))
    if winner[1] != receipt["winner_configuration_id"]:
        raise ValueError("Recorded winner is not the validation minimizer")
    if not close(winner[0], receipt["winner_pooled_cv_RMSE_original"]):
        raise ValueError("Winner score changed")
    arithmetic = min(eligible, key=lambda x: (statistics.fmean(x[2]), x[1]))
    return {
        "configuration_id": winner[1],
        "validation_pooled_rmse": winner[0],
        "validation_mean_fold_rmse": statistics.fmean(winner[2]),
        "validation_fold_sd_descriptive": statistics.stdev(winner[2]),
        "validation_fold_scores": dict(zip(FOLDS, winner[2])),
        "eligible_configurations": len(eligible),
        "candidate_configurations": len(ledger),
        "arithmetic_mean_winner_configuration_id": arithmetic[1],
        "pooled_and_arithmetic_winner_agree": arithmetic[1] == winner[1],
        "selected_parameters": receipt["winner_trial"],
        "fixed_phase_epochs": receipt["fixed_phase_epochs"],
    }


def select_classical(receipt):
    if receipt.get("calibration_used_for_selection") is not False:
        raise ValueError("Calibration must not select the configuration")
    if receipt.get("selection_metric") != "pooled_original_unit_CV_RMSE":
        raise ValueError("Unexpected classical selection metric")
    if receipt["model"] == "Persistence":
        if receipt.get("parameters") != {"lag": 1} or receipt.get("ledger"):
            raise ValueError("Persistence has a fixed lag-1 recipe")
        return {"selected_parameters": {"lag": 1}, "configuration_id": "fixed-lag1",
                "validation_pooled_rmse": None, "validation_mean_fold_rmse": None,
                "validation_fold_sd_descriptive": None, "validation_fold_scores": {},
                "eligible_configurations": 1, "candidate_configurations": 1,
                "arithmetic_mean_winner_configuration_id": None,
                "pooled_and_arithmetic_winner_agree": None, "fixed_phase_epochs": None}
    groups = {}
    for row in receipt["ledger"]:
        signature = json.dumps(row["parameters"], sort_keys=True, separators=(",", ":"))
        groups.setdefault(signature, []).append(row)
    eligible = []
    for signature, rows in groups.items():
        # A failed or missing fold is not dropped to improve a candidate's score.
        if len(rows) != 3 or {r["fold"] for r in rows} != set(FOLDS):
            continue
        if any(r.get("candidate_status") != "completed" for r in rows):
            continue
        byfold = {r["fold"]: r for r in rows}
        values = [finite_positive(byfold[f]["RMSE"], "classical CV RMSE") for f in FOLDS]
        eligible.append((pooled_rmse(values), signature, values))
    if not eligible:
        raise ValueError("No complete classical CV configuration")
    winner = min(eligible, key=lambda x: (x[0], x[1]))
    if json.loads(winner[1]) != receipt["parameters"]:
        raise ValueError("Classical winner differs from validation minimizer")
    arithmetic = min(eligible, key=lambda x: (statistics.fmean(x[2]), x[1]))
    return {"selected_parameters": receipt["parameters"], "configuration_id": winner[1],
            "validation_pooled_rmse": winner[0],
            "validation_mean_fold_rmse": statistics.fmean(winner[2]),
            "validation_fold_sd_descriptive": statistics.stdev(winner[2]),
            "validation_fold_scores": dict(zip(FOLDS, winner[2])),
            "eligible_configurations": len(eligible), "candidate_configurations": len(groups),
            "arithmetic_mean_winner_configuration_id": arithmetic[1],
            "pooled_and_arithmetic_winner_agree": arithmetic[1] == winner[1],
            "fixed_phase_epochs": None}


def number(row, key, integer=False):
    value = row.get(key)
    if value in (None, ""):
        return None
    value = finite_positive(value, key)
    return int(value) if integer else value


def build_table(root):
    root = Path(root)
    register = unique_rows(root/REGISTER)
    comparison = unique_rows(root/COMPARISON)
    a2 = unique_rows(root/A2)
    mcs = unique_rows(root/MCS)
    freeze = read_json(root/FREEZE)
    expected = freeze["systems"]
    if len(expected) != 16 or len(set(expected)) != 16:
        raise ValueError("Expected 16 distinct frozen systems")
    for data in (comparison, a2, mcs):
        if set(data) != set(expected):
            raise ValueError("Historical metric tables do not cover the frozen systems")
    inputs = [REGISTER, COMPARISON, A2, FREEZE, MCS]
    provenance_path = OUTPUT/"selection_receipt_provenance.json"
    provenance = read_json(root/provenance_path)
    if provenance.get("commit") != BACKUP_COMMIT:
        raise ValueError("Selection receipt provenance commit changed")
    declared = provenance["receipts"]
    if len(declared) != 13 or len({x["path"] for x in declared}) != 13:
        raise ValueError("Thirteen distinct receipt hashes are required")
    for item in declared:
        path = Path(item["path"])
        if path.is_absolute() or ".." in path.parts or path.parent != RECEIPTS:
            raise ValueError("Invalid receipt path")
        if sha256(root/path) != item["sha256"]:
            raise ValueError("Selection receipt hash mismatch")
    inputs.append(provenance_path)
    receipts = {}
    for p in sorted((root/RECEIPTS).glob("*.json")):
        doc = read_json(p)
        model = doc["model_id"]
        if model in receipts:
            raise ValueError("Duplicate neural selection receipt")
        receipts[model] = (p.relative_to(root), doc)
    if set(receipts) != set(register) or len(receipts) != 13:
        raise ValueError("Thirteen selected neural recipes with provenance are required")
    rows = []
    for model in expected:
        neural = model in register
        if neural:
            path, receipt = receipts[model]
            selected = select_neural(receipt)
            training = register[model]
            if (selected["configuration_id"] != training["configuration_id"] or
                    receipt["winner_trial"]["trial_id"] != training["trial_id"] or
                    receipt["winner_trial"]["recipe_id"] != training["recipe_id"]):
                raise ValueError("Validation winner does not bind to the frozen training recipe")
            for key in ("learning_rate", "weight_decay", "batch_size", "dropout"):
                if not close(receipt["winner_trial"][key], training[key]):
                    raise ValueError("Winner parameters differ from final recipe")
            role = training["role"]
            final_seeds = json.loads(training["final_seeds_json"])
            if final_seeds != freeze["final_seed_ids"]:
                raise ValueError("Registered seed cohort differs from freeze")
            per_seed = json.loads(training["primary_RMSE_per_seed_json"])
            if len(per_seed) != len(final_seeds) or not close(statistics.fmean(per_seed), a2[model]["RMSE"]):
                raise ValueError("Test-seed summary differs from A2")
            if not close(statistics.stdev(per_seed), a2[model]["RMSE_seed_SD"]):
                raise ValueError("Test-seed SD differs from A2")
        else:
            path = Path("experiments/e2_benchmark_final_20261113")/f"selection_{model}.json"
            receipt = read_json(root/path)
            if receipt["model"] != model:
                raise ValueError("Classical receipt identity mismatch")
            selected = select_classical(receipt)
            role, final_seeds, per_seed = "classical_comparator", [], []
        inputs.append(path)
        historical = comparison[model]
        if not close(historical["RMSE"], a2[model]["RMSE"]):
            raise ValueError("Descriptive test metrics differ from A2")
        mc = mcs[model]
        frozen_predictors = [x for x in freeze["models"] if x["model"] == model]
        if len(frozen_predictors) != (6 if neural else 1):
            raise ValueError("Frozen predictor count does not match system")
        row = dict(model=model, role=role, **selected)
        row.update({
            "selection_source": str(path), "selection_source_sha256": sha256(root/path),
            "selection_metric": ("pooled_original_unit_CV_RMSE" if model != "Persistence"
                                 else "fixed_recipe_no_hyperparameter_search"),
            "selected_by_validation": model != "Persistence",
            "validation_dates": 249 if model != "Persistence" else None,
            "validation_folds": list(FOLDS) if model != "Persistence" else [],
            "validation_optimization_seeds": [int(register[model]["hpo_execution_seed"])] if neural else [],
            "validation_ci95_lower": None, "validation_ci95_upper": None,
            "validation_ci_status": "not_estimated_correlated_temporal_folds" if model != "Persistence" else "no_CV_score_in_fixed_recipe_receipt",
            "seed_dispersion_in_validation_estimated": False,
            "winner_selection_bias_quantified": False,
            "final_seed_ids": final_seeds,
            "representative_export_seed": min(final_seeds) if neural else None,
            "representative_export_seed_policy": "lowest_registered_seed_not_test_selected" if neural else "deterministic_no_seed_replication",
            "frozen_predictor_count": len(frozen_predictors),
            "frozen_predictors": frozen_predictors,
            "test_mean_rmse": float(a2[model]["RMSE"]),
            "test_seed_rmse": per_seed,
            "test_seed_sd": number(historical, "RMSE_seed_SD"),
            "test_seed_ci95_lower": number(historical, "RMSE_seed_t_CI95_lower"),
            "test_seed_ci95_upper": number(historical, "RMSE_seed_t_CI95_upper"),
            "test_ci_scope": historical["seed_CI_scope"] if neural else "not_estimable_from_one_deterministic_predictor",
            "test_common_dates": int(a2[model]["n_common_dates"]),
            "parameters": number(historical, "actual_parameter_count", integer=True),
            "core_flops_min": number(historical, "core_forward_flops_observed_min", integer=True),
            "core_flops_max": number(historical, "core_forward_flops_observed_max", integer=True),
            "full_all_kernel_flops_measured": False,
            "gpu_latency_median_ms_existing": number(historical, "mean_latency_gpu_median_ms"),
            "cpu_latency_median_ms": None, "cpu_latency_p95_ms": None,
            "cpu_peak_memory_bytes": None,
            "mcs_90_retained": mc["retained_at_alpha"] == "True",
            "mcs_pvalue_existing": float(mc["MCS_pvalue"]),
            "mcs_scope": "E2_retrospective_temporal_MSE_family_not_selection_criterion",
            "export_roundtrip_status": "not_performed_by_selection_table",
            "exported_model_metric": None,
            "global_validation_winner": None,
        })
        rows.append(row)
    scored = [r for r in rows if r["validation_pooled_rmse"] is not None]
    best = min(scored, key=lambda r: (r["validation_pooled_rmse"], r["model"]))["model"]
    test_best = min(rows, key=lambda r: (r["test_mean_rmse"], r["model"]))["model"]
    for row in rows:
        row["leader_available_validation_15_systems"] = row["model"] == best
        row["historical_test_descriptive_leader"] = row["model"] == test_best
    source_report = [{"path": str(p), "sha256": sha256(root/p)} for p in sorted(set(inputs))]
    summary = {
        "base_main_commit": BASE_COMMIT, "selection_backup_commit": BACKUP_COMMIT,
        "selection_backup_branch": "results/e2-reduced-delivery-v1",
        "model_count": len(rows), "neural_recipes_verified": len(receipts),
        "all_neural_pooled_and_arithmetic_winners_agree": all(r["pooled_and_arithmetic_winner_agree"] for r in rows if r["model"] in register),
        "validation_scored_systems": len(scored), "available_validation_leader": best,
        "global_validation_winner_16_systems": None,
        "global_validation_status": "Persistence has no CV score in its fixed-recipe receipt; global claim remains unresolved",
        "available_validation_leader_mcs_retained": next(r["mcs_90_retained"] for r in rows if r["model"] == best),
        "historical_test_descriptive_leader": test_best,
        "historical_test_leader_mcs_retained": next(r["mcs_90_retained"] for r in rows if r["model"] == test_best),
        "validation_uncertainty_status": "fold SD descriptive; no independent-seed CI; winner-selection bias not quantified",
        "representative_seed_policy": "lowest registered final seed; never selected using historical test RMSE",
        "new_training_performed": False, "test_scores_used_for_configuration_selection": False,
        "cpu_profile_performed": False, "roundtrip_performed": False,
        "sources": source_report,
    }
    return rows, summary


def write_outputs(root, destination):
    rows, summary = build_table(root)
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    (destination/"master_configurations.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    fields = list(rows[0])
    with (destination/"master_configurations.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v for k,v in row.items()})
    (destination/"selection_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--output", type=Path)
    p.add_argument("--check", action="store_true", help="Rebuild offline and compare all three generated outputs without overwriting them")
    args = p.parse_args()
    destination = args.output or args.root/OUTPUT
    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            summary = write_outputs(args.root, Path(tmp))
            for name in ("master_configurations.json", "master_configurations.csv", "selection_report.json"):
                if not (destination/name).is_file() or (Path(tmp)/name).read_bytes() != (destination/name).read_bytes():
                    raise ValueError(f"Generated selection output is stale or modified: {name}")
        print("OK: three selection outputs reproduce exactly, offline")
    else:
        summary = write_outputs(args.root, destination)
    print(json.dumps({k:v for k,v in summary.items() if k != "sources"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
