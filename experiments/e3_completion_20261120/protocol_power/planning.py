"""Replay conditional planning from V3's saved development forecasts only.

No torch/model/engine imports; no fitting, prediction, calibration or selection.
The registry precedes calculation. These are conditional design references,
not observed power and not a recommendation to proceed with the failed pilot.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

ROOT = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def replay(output: Path):
    registry = json.loads((ROOT / "planning_registry.json").read_text())
    if np.__version__ != "2.3.5" or scipy.__version__ != "1.17.0":
        raise RuntimeError("Replay requires NumPy 2.3.5 / SciPy 1.17.0")
    for relative, expected in registry["inputs"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Input hash mismatch: " + relative)
    source = ROOT / registry["historical_math_source"]["path"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != registry["historical_math_source"]["sha256"]:
        raise ValueError("Planning mathematics source hash mismatch")
    spec = importlib.util.spec_from_file_location("historical_math_only", source)
    math_only = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(math_only)
    # Deliberately never invoke historical_math_only.build(): its data are additive.
    scenarios = registry["fixed_scenarios"]
    alpha = scenarios["conservative_alpha"]  # 0.05 / 57, overrides legacy default /48.
    delta = scenarios["BTC_inherited_RMSE_delta"]
    seeds = scenarios["pilot_final_seeds"]
    records = read_csv(ROOT / "frozen_inputs/assessment_predictions.csv")
    by = {}
    for row in records:
        seed = int(row["seed"]) if row["seed"].isdigit() else row["seed"]
        key = row["name"], seed
        by.setdefault(key, {})[row["date"]] = (float(row["target"]), float(row["prediction"]))
    candidate = scenarios["candidate"]
    dates = sorted(by[(candidate, seeds[0])])
    if len(dates) != scenarios["pilot_dates"]:
        raise ValueError("The complete fixed pilot calendar is required")
    truth = np.array([by[(candidate, seeds[0])][d][0] for d in dates])
    trajectories = {}
    for name in [candidate] + scenarios["known_variance_controls"]:
        keys = [(name, s) for s in seeds] if name not in {"floor_only", "floor_plus_mean_excess"} else [k for k in by if k[0] == name]
        if len(keys) not in (1, 3):
            raise ValueError("Incomplete recipe: " + name)
        predictions = []
        for key in keys:
            if sorted(by[key]) != dates:
                raise ValueError("Calendar mismatch")
            if any(by[key][d][0] != truth[i] for i, d in enumerate(dates)):
                raise ValueError("Target mismatch")
            predictions.append([by[key][d][1] for d in dates])
        trajectories[name] = np.asarray(predictions)
    squared = {name: (values - truth[None, :]) ** 2 for name, values in trajectories.items()}
    rmses = {name: np.sqrt(values.mean(axis=1)) for name, values in squared.items()}
    influences = {name: ((values - values.mean(axis=1)[:, None]) / (2 * rmses[name][:, None])).mean(axis=0)
                  for name, values in squared.items()}
    variance_rows, seed_rows, time_rows = [], [], []
    target_names = scenarios["targets"]
    for rival in scenarios["known_variance_controls"]:
        # Deterministic rival has one forecast; subtraction broadcasts only for
        # paired initialization uncertainty, never as three rival fits or clocks.
        gains = rmses[rival] - rmses[candidate]
        sigma = float(np.std(gains, ddof=1))
        upper = math_only.sigma_upper_bound(sigma, len(seeds), tail_probability=0.05 / 6)
        variance_rows.append({"rival": rival, "seed_pairs": 3,
            "rival_saved_trajectories": len(trajectories[rival]), "seed_SD": sigma,
            "SD_upper95_simultaneous_six_normal_assumption": upper,
            "upper_to_sample_ratio": upper / sigma,
            "scope": "initialization on the same exposed55 calendar; not future calendar variability",
            "guide_pilot_minimum5_satisfied": False})
        for variance_label, planning_sd in (("observed_three_seed_sd", sigma),
                ("normal_theory_simultaneous_six_SD_upper95", upper),
                ("twice_observed_three_seed_sd", 2 * sigma)):
            for target_name, target in target_names.items():
                for alternative_multiplier in scenarios["alternative_gain_delta_multipliers"]:
                    gap = (alternative_multiplier - 1) * delta
                    row = {"rival": rival, "variance_reference": variance_label,
                        "conditional_SD": planning_sd, "alpha_reference": alpha,
                        "target_name": target_name, "target_reference_power": target,
                        "inherited_delta_pending_justification": delta,
                        "planned_gain": alternative_multiplier * delta,
                        "separation_g1_minus_delta": gap,
                        "conditional_required_pairs_minimum10": math_only.required_pairs(planning_sd, gap, target, alpha=alpha),
                        "final_R_selected": False, "actual_three_test_power_established": False}
                    for future_r in scenarios["future_R_sensitivity"]:
                        row[f"R{future_r}_conditional_t_reference_power"] = math_only.t_power(future_r, planning_sd, gap, alpha=alpha)
                        row[f"R{future_r}_MDE_separation"] = math_only.t_mde_separation(future_r, planning_sd, target, alpha=alpha)
                    seed_rows.append(row)
        difference = influences[rival] - influences[candidate]
        centered = difference - difference.mean()
        for lag in scenarios["temporal_HAC_Bartlett_lags"]:
            lrv = float(np.dot(centered, centered) / len(dates))
            for k in range(1, lag + 1):
                lrv += 2 * (1 - k / (lag + 1)) * float(np.dot(centered[k:], centered[:-k]) / len(dates))
            if lrv <= 0:
                raise ValueError("Nonpositive temporal variance reference")
            for sd_multiplier in scenarios["temporal_SD_multipliers"]:
                tau = math.sqrt(lrv) * sd_multiplier
                for target_name, target in target_names.items():
                    for alternative_multiplier in scenarios["alternative_gain_delta_multipliers"]:
                        gap = (alternative_multiplier - 1) * delta
                        row = {"rival": rival, "HAC_Bartlett_lags": lag,
                            "exposed_pilot_dates": len(dates), "temporal_SD_multiplier": sd_multiplier,
                            "conditional_long_run_SD": tau, "alpha_reference": alpha,
                            "target_name": target_name, "target_normal_reference_power": target,
                            "planned_gain": alternative_multiplier * delta,
                            "separation_g1_minus_delta": gap,
                            "asymptotic_reference_dates": math_only.normal_calendar_dates(tau, gap, target, alpha=alpha),
                            "reference_is_recommended_calendar_length": False,
                            "independent_future_dates_acquired": 0,
                            "guide_temporal_three_test_power_established": False}
                        for future_t in scenarios["future_T_sensitivity"]:
                            row[f"T{future_t}_conditional_normal_reference_power"] = float(stats.norm.sf(stats.norm.isf(alpha) - math.sqrt(future_t) * gap / tau))
                            row[f"T{future_t}_MDE_separation"] = tau * (stats.norm.isf(alpha) + stats.norm.ppf(target)) / math.sqrt(future_t)
                        time_rows.append(row)
    joint = {"kind": "dependence_sensitivity_only_not_estimated_joint_power",
        "main_rivals": 19, "required_components": 57,
        "variance_controls_available": 6, "main_rival_variances_available": 3,
        "ablation_control_variances_available": 3, "missing_main_rival_variance_count": 16,
        "marginal_0_90_independence_product_19_rivals_hypothesis": 0.9 ** 19,
        "marginal_0_90_independence_product_57_components_hypothesis": 0.9 ** 57,
        "marginal_0_90_perfect_common_normal_decision_hypothesis": 0.9,
        "marginal_0_90_arbitrary_dependence_union_lower_bound_57": max(0, 1 - 57 * 0.1),
        "marginal_sufficient_for_joint90_under_arbitrary_dependence_union_bound": 1 - 0.1 / 57,
        "guide_requires_joint90": False,
        "guide_requires_report_joint_and_marginal90_per_contrast": True,
        "joint90_is_optional_conservative_planning_scenario": True,
        "complete_family_actual_power_estimated": False,
        "reason": "All16 E2 rivals absent; only additive plus two analytical main-rival variances available, three other variances belong to separate ablations; only three final seeds, temporal procedures unvalidated, exposed55 calendar too short for stable tail/HAC variance"}
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "variance_inputs.csv", variance_rows)
    write_csv(output / "seed_sensitivity.csv", seed_rows)
    write_csv(output / "calendar_sensitivity.csv", time_rows)
    (output / "joint_sensitivity.json").write_text(json.dumps(joint, ensure_ascii=False, indent=2) + "\n")
    report = {"status": "conditional_sensitivity_only_final_R_unresolved",
        "registry_sha256": hashlib.sha256((ROOT / "planning_registry.json").read_bytes()).hexdigest(),
        "runtime": {"numpy": np.__version__, "scipy": scipy.__version__},
        "pilot_seed_pairs": 3, "guide_minimum_pilot_seed_pairs": 5,
        "pilot_calendar_dates": len(dates), "control_variances_available": len(variance_rows),
        "main_rival_variances_available": 3, "missing_main_rival_variances": 16,
        "full_family_rivals": 19, "procedure_components": 57,
        "seed_sensitivity_rows": len(seed_rows), "calendar_sensitivity_rows": len(time_rows),
        "analytic_scenario_rows": len(seed_rows) + len(time_rows),
        "Monte_Carlo_draws": 0,
        "no_observed_posthoc_power": True, "planned_alternative_from_observed_gain": False,
        "SD_source_is_V3_pilot": True, "additive_audit_SD_transferred": False,
        "seed_and_calendar_variance_combined_without_identification": False,
        "final_R": None, "final_calendar": None,
        "empirical_power_proven": False, "confirmation_go": False,
        "new_model_fits": 0, "new_model_predictions": 0, "AI_assistance_declared": True}
    (output / "planning_receipt.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    print(json.dumps(replay(args.output), ensure_ascii=False))
