"""Post-hoc descriptive seed covariance from the frozen exposed CPU pilot.

No model, random resampling, hypothesis test, confidence interval or power
calculation is executed. The 1210 rows represent 55 dates, not1210 independent
observations. The four-model seed covariance does not fill future-domain cells.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

PLAN_SHA = "2e6b71b53683d376ff24990b39bc591458816a3001cd62c432fdc1c7c8d8d819"
PRED_SHA = "3de0b8cad079c4b03f3d0cd7f7d35848b0acc9807a91e88c563bd34cf2a4c2c0"
NEURAL = ("sigma_crps_v4", "u_crps_control", "mse_only_control", "additive_positive_head")
ANALYTIC = ("floor_only", "floor_plus_mean_excess")
SEEDS = tuple(range(20261121, 20261126))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def describe(root, output):
    if output.exists():
        raise ValueError("Fresh descriptive output required; preserve previous bytes")
    report = json.loads((root / "pilot_result.json").read_text())
    if report.get("plan_sha256") != PLAN_SHA or sha(root / "assessment_predictions.csv") != PRED_SHA:
        raise ValueError("Fixed completed pilot identity differs")
    with (root / "assessment_predictions.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1210:
        raise ValueError("Expected1210 frozen development outputs")
    grouped, target_by_date = {}, {}
    for row in rows:
        name = row["name"]; seed = int(row["seed"]) if row["seed"] else None
        if (name in NEURAL and seed not in SEEDS) or (name in ANALYTIC and seed is not None) or name not in NEURAL + ANALYTIC:
            raise ValueError("Unexpected recipe/seed")
        target, prediction = float(row["target"]), float(row["prediction"])
        if not math.isfinite(target) or not math.isfinite(prediction):
            raise ValueError("Nonfinite frozen output")
        date = row["date"]
        if date in target_by_date and target_by_date[date] != target:
            raise ValueError("Targets disagree across trajectories")
        target_by_date[date] = target
        group = grouped.setdefault((name, seed), {})
        if date in group:
            raise ValueError("Duplicate trajectory/date")
        group[date] = (prediction, target)
    dates = sorted(target_by_date)
    if len(dates) != 55 or len(grouped) != 22 or any(sorted(values) != dates for values in grouped.values()):
        raise ValueError("Shared55-date/22-trajectory geometry differs")
    metrics = {}
    for key, values in grouped.items():
        differences = [values[date][0] - values[date][1] for date in dates]
        metrics[key] = {"RMSE": math.sqrt(math.fsum(v * v for v in differences) / len(dates)),
                        "MAE": math.fsum(abs(v) for v in differences) / len(dates)}
    summaries = []
    with (root / "recipe_summary.csv").open(newline="") as stream:
        original = {row["name"]: row for row in csv.DictReader(stream)}
    discrepancy = 0.0
    for name in NEURAL + ANALYTIC:
        values = [metrics[name, seed]["RMSE"] for seed in SEEDS] if name in NEURAL else [metrics[name, None]["RMSE"]]
        mean = statistics.mean(values)
        discrepancy = max(discrepancy, abs(mean - float(original[name]["mean_RMSE"])))
        summaries.append({"name": name, "trajectories": len(values), "distinct_dates": 55,
                          "stochastic_seeds": 5 if name in NEURAL else None,
                          "mean_individual_RMSE": mean,
                          "sample_SD_RMSE_across_seeds": statistics.stdev(values) if name in NEURAL else None,
                          "sample_variance_RMSE_across_seeds": statistics.variance(values) if name in NEURAL else None})
    covariance = []
    for left in NEURAL:
        a = [metrics[left, seed]["RMSE"] for seed in SEEDS]
        for right in NEURAL:
            b = [metrics[right, seed]["RMSE"] for seed in SEEDS]
            covariance.append({"left_recipe": left, "right_recipe": right, "paired_seeds": 5,
                               "fixed_exposed_dates": 55, "sample_covariance_RMSE": statistics.covariance(a, b),
                               "unit": "sigma7_squared"})
    paired = []
    for control in NEURAL[1:]:
        deltas = [metrics["sigma_crps_v4", seed]["RMSE"] - metrics[control, seed]["RMSE"] for seed in SEEDS]
        paired.append({"candidate": "sigma_crps_v4", "control": control, "paired_seeds": 5,
                       "fixed_exposed_dates": 55, "mean_RMSE_difference_candidate_minus_control": statistics.mean(deltas),
                       "sample_SD_paired_RMSE_difference": statistics.stdev(deltas),
                       "sample_variance_paired_RMSE_difference": statistics.variance(deltas),
                       "unit": "sigma7", "p_value": None, "confidence_interval": None})
    output.mkdir(parents=True)
    for name, data in (("descriptive_recipe_spread.csv", summaries),
                       ("conditional_four_recipe_seed_covariance.csv", covariance),
                       ("paired_seed_differences_descriptive.csv", paired)):
        write_csv(output / name, data)
    receipt = {"kind": "posthoc_descriptive_seed_spread_and_covariance_only",
               "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "source_plan_canonical_sha256": PLAN_SHA, "source_predictions_sha256": PRED_SHA,
               "source_rows": 1210, "distinct_dates": 55, "trajectories": 22,
               "neural_recipes": 4, "paired_seeds_per_neural_recipe": 5,
               "deterministic_controls_not_replicated_as_seeds": 2,
               "max_abs_recomputed_mean_RMSE_difference": discrepancy,
               "interpretation": "Conditional seed variability on one already exposed55-date development calendar; not calendar uncertainty or a complete-family planning pilot.",
               "future_four_panel_variance_cells_filled": 0,
               "p_values_computed": False, "confidence_intervals_computed": False,
               "power_or_MDE_estimated": False, "random_resamples": 0,
               "model_fits_or_forwards": 0, "R3_accredited": False,
               "new_candidate_selected_or_rule_changed": False,
               "files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha(p)}
                         for p in sorted(output.glob("*.csv"))], "AI_assistance_declared": True}
    (output / "descriptive_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(describe(args.input, args.output), indent=2))
