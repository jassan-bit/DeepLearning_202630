"""Historical-variance planning inputs for a NEW E3 study, not observed power.

Only immutable saved forecasts are read. No model is imported, fitted, inferred,
calibrated, selected, or rescored for a new hypothesis. Planned alternatives are
defined above the inherited null boundary, never set to the observed mean gain.
Seed and calendar information remain distinct; this is not a frozen protocol.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import platform

import numpy as np
from scipy import optimize, stats
import scipy


ROOT = Path(__file__).resolve().parents[3]
FROZEN = ROOT / "experiments/e3_estadistica_posterior_20261117"
ALPHA = 0.05
RIVALS = 16
METHODS_PER_RIVAL = 3
TESTS = RIVALS * METHODS_PER_RIVAL
DELTA = 0.00025772842664981106
PLANNED_EXCESS_MULTIPLIERS = (0.25, 0.5, 1.0)
TARGETS = (
    ("one_procedure_90", 0.90),
    ("three_procedure_rival_union_bound_90", 1 - 0.10 / METHODS_PER_RIVAL),
    ("all_48_decisions_union_bound_90", 1 - 0.10 / TESTS),
)


def positive(value, name):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be positive and finite")
    return float(value)


def probability(value, name):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 < value < 1:
        raise ValueError(name + " must lie strictly between zero and one")
    return float(value)


def integer(value, minimum, name):
    if type(value) is not int or value < minimum:
        raise ValueError(name + " is an invalid integer")
    return value


def t_power(pairs, sigma, separation, alpha=ALPHA / TESTS):
    """Exact noncentral-t planning power under iid normal paired gains.

    separation is g1-delta for H0:g<=delta. It is NOT the observed gain.
    sigma is treated as the true future SD in this conditional calculation.
    """
    integer(pairs, 2, "pairs")
    sigma = positive(sigma, "sigma")
    if not isinstance(separation, (int, float)) or isinstance(separation, bool) or not math.isfinite(separation) or separation < 0:
        raise ValueError("separation must be nonnegative and finite")
    alpha = probability(alpha, "alpha")
    threshold = stats.t.isf(alpha, pairs - 1)
    return float(stats.nct.sf(threshold, pairs - 1, math.sqrt(pairs) * separation / sigma))


def required_pairs(sigma, separation, power, alpha=ALPHA / TESTS, minimum=10, maximum=1000000):
    positive(sigma, "sigma")
    positive(separation, "separation")
    probability(power, "power")
    probability(alpha, "alpha")
    integer(minimum, 2, "minimum")
    integer(maximum, minimum, "maximum")
    if t_power(minimum, sigma, separation, alpha) >= power:
        return minimum
    lo, hi = minimum, minimum
    while hi < maximum and t_power(hi, sigma, separation, alpha) < power:
        lo, hi = hi, min(maximum, hi * 2)
    if t_power(hi, sigma, separation, alpha) < power:
        return None
    while hi - lo > 1:
        mid = (hi + lo) // 2
        if t_power(mid, sigma, separation, alpha) >= power:
            hi = mid
        else:
            lo = mid
    return hi


def t_mde_separation(pairs, sigma, power, alpha=ALPHA / TESTS):
    integer(pairs, 2, "pairs")
    sigma = positive(sigma, "sigma")
    probability(power, "power")
    probability(alpha, "alpha")
    hi = sigma
    while t_power(pairs, sigma, hi, alpha) < power:
        hi *= 2
    return float(optimize.brentq(lambda gap: t_power(pairs, sigma, gap, alpha) - power,
                                0.0, hi, xtol=np.finfo(float).eps * sigma, rtol=1e-12))


def sigma_upper_bound(sample_sd, pairs, tail_probability=0.05 / RIVALS):
    """One-sided normal-theory SD bound; Bonferroni across 16 historical SDs."""
    sample_sd = positive(sample_sd, "sample_sd")
    integer(pairs, 2, "pairs")
    probability(tail_probability, "tail_probability")
    return float(sample_sd * math.sqrt((pairs - 1) / stats.chi2.ppf(tail_probability, pairs - 1)))


def normal_calendar_dates(long_run_sd, separation, power, alpha=ALPHA / TESTS):
    """Asymptotic calendar reference, not a finite-sample HAC power guarantee."""
    long_run_sd = positive(long_run_sd, "long_run_sd")
    separation = positive(separation, "separation")
    probability(power, "power")
    probability(alpha, "alpha")
    return (long_run_sd * (stats.norm.isf(alpha) + stats.norm.ppf(power)) / separation) ** 2


def marginal_target_for_joint(joint_power, decisions):
    probability(joint_power, "joint_power")
    integer(decisions, 1, "decisions")
    return 1 - (1 - joint_power) / decisions


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, FROZEN / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def aligned_seed_gains(candidate_rmse, rival_rmse, candidate_ids, rival_ids):
    candidate = np.asarray(candidate_rmse, dtype=float)
    rival = np.asarray(rival_rmse, dtype=float)
    expected = tuple(range(271801, 271811))
    if tuple(candidate_ids) != expected or candidate.shape != (10,):
        raise ValueError("The complete fixed ten candidate seed coordinates are required")
    if tuple(rival_ids) == ("deterministic",):
        if rival.shape != (1,):
            raise ValueError("A deterministic rival has exactly one saved trajectory")
        gains = rival[0] - candidate
    else:
        if tuple(rival_ids) != expected or rival.shape != (10,):
            raise ValueError("Rival seeds must align exactly; truncation or relabelling is not allowed")
        gains = rival - candidate
    if not np.isfinite(gains).all():
        raise ValueError("Paired metric gains must be finite")
    return gains


def build(folder):
    if np.__version__ != "2.3.5" or scipy.__version__ != "1.17.0":
        raise RuntimeError("Use NumPy 2.3.5 and SciPy 1.17.0")
    inputs = load_module("e3_frozen_inputs_for_planning", "inputs.py")
    analysis = load_module("e3_frozen_analysis_for_planning", "analysis.py")
    panel = inputs.load_inputs(folder)
    names, rivals = panel["model_order"], panel["rivals"]
    if len(names) != 17 or len(rivals) != RIVALS:
        raise ValueError("The full frozen 17-system family is required")
    squared = {m: (panel["predictions"][m] - panel["targets"][None, :]) ** 2 for m in names}
    rmses = {m: np.sqrt(squared[m].mean(axis=1)) for m in names}
    influences = {m: analysis.rmse_influence(squared[m]) for m in names}
    candidate = inputs.E3_MODEL
    seed_inputs, seed_rows, calendar_rows = [], [], []
    for rival in rivals:
        gains = aligned_seed_gains(rmses[candidate], rmses[rival], panel["seeds"][candidate], panel["seeds"][rival])
        sd = float(np.std(gains, ddof=1))
        upper = sigma_upper_bound(sd, len(gains))
        seed_inputs.append({"rival": rival, "candidate_seeds": 10,
            "rival_saved_trajectories": len(panel["seeds"][rival]),
            "seed_SD_RMSE_gain": sd, "SD_upper_95_family16_normal_theory": upper,
            "deterministic_rival_replicated_or_refitted": False,
            "variance_scope": "initialization_conditional_on_the_known_608_date_calendar"})
        for variance_name, planning_sd in (("historical_sample_SD", sd), ("historical_upper_SD_simultaneous16", upper),
                                           ("twice_historical_SD_sensitivity", 2 * sd)):
            for target_name, target in TARGETS:
                mde = t_mde_separation(10, planning_sd, target)
                for multiplier in PLANNED_EXCESS_MULTIPLIERS:
                    gap = multiplier * DELTA
                    pairs = required_pairs(planning_sd, gap, target)
                    seed_rows.append({"rival": rival, "variance_reference": variance_name,
                        "planning_SD": planning_sd, "target_scenario": target_name,
                        "target_seed_test_power": target, "alpha_seed_test_reference": ALPHA / TESTS,
                        "null_gain_boundary_delta": DELTA, "planned_alternative_gain": DELTA + gap,
                        "planned_separation_g1_minus_delta": gap,
                        "planned_separation_as_delta_multiple": multiplier,
                        "required_independent_seed_pairs_minimum10": pairs,
                        "sample_size_ratio_to_historical_ten": None if pairs is None else pairs / 10,
                        "R10_MDE_separation_from_null_boundary": mde,
                        "R10_minimum_alternative_gain_for_target": DELTA + mde,
                        "current_E3_power_estimated": False,
                        "full_three_procedure_family_power_established": False})
        difference = influences[rival] - influences[candidate]
        for lag in (7, 14, 28):
            se = analysis.hac_standard_error(difference, lag)
            if se is None:
                raise ValueError("Historical temporal variance is not estimable: " + rival)
            tau = se * math.sqrt(len(panel["dates"]))
            for target_name, target in TARGETS:
                mde = tau * (stats.norm.isf(ALPHA / TESTS) + stats.norm.ppf(target)) / math.sqrt(608)
                for multiplier in PLANNED_EXCESS_MULTIPLIERS:
                    gap = multiplier * DELTA
                    raw = normal_calendar_dates(tau, gap, target)
                    calendar_rows.append({"rival": rival, "HAC_Bartlett_lags": lag,
                        "historical_dates": 608, "conditional_long_run_SD": tau,
                        "historical_HAC_standard_error": se, "target_scenario": target_name,
                        "target_asymptotic_calendar_power": target, "alpha_reference": ALPHA / TESTS,
                        "planned_alternative_gain": DELTA + gap,
                        "planned_separation_g1_minus_delta": gap,
                        "normal_reference_calendar_dates": raw,
                        "calendar_dates_ceiling_minimum608": max(608, math.ceil(raw)),
                        "reference_length_ratio_to_historical608": raw / 608,
                        "reference_is_recommended_design_length": False,
                        "T608_MDE_separation_from_null_boundary": mde,
                        "T608_minimum_alternative_gain_for_target": DELTA + mde,
                        "new_calendar_power_established": False})
    correlation = np.corrcoef(np.stack([aligned_seed_gains(rmses[candidate], rmses[r], panel["seeds"][candidate],
                       panel["seeds"][r]) for r in rivals]))
    off = correlation[np.triu_indices(RIVALS, 1)]
    coupling = {"common_marginal_power_illustration": 0.90,
        "all16_if_contrast_decisions_were_independent": 0.90 ** RIVALS,
        "all48_if_test_decisions_were_independent": 0.90 ** TESTS,
        "all48_union_lower_bound_with_marginal_0_90": max(0.0, 1 - TESTS * 0.10),
        "marginal_target_sufficient_for_joint_0_90_by_union_bound": marginal_target_for_joint(.90, TESTS),
        "historical_seed_gain_correlation_min": float(off.min()),
        "historical_seed_gain_correlation_max": float(off.max()),
        "historical_correlation_scope": "16 gain vectors across ten initializations on one fixed calendar; not p-value dependence or future joint power",
        "actual_joint_power_estimated": False,
        "no_independence_product_is_used_as_actual_power": True}
    result = {"schema_version": 1, "status": "preparatory_historical_variance_planning_not_preregistered",
        "review_date_local": "2026-11-18", "timezone": "America/Bogota",
        "source_review_commit": "73d8958c50a69fbea46deb828f03f89385318789",
        "alpha": ALPHA, "rivals": RIVALS, "procedures_per_rival": METHODS_PER_RIVAL,
        "family_test_count": TESTS, "inherited_BTC_RMSE_margin": DELTA,
        "source_inputs": panel["provenance"],
        "code_sha256": {"power_planning.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "frozen_inputs.py": hashlib.sha256((FROZEN / "inputs.py").read_bytes()).hexdigest(),
            "frozen_analysis.py": hashlib.sha256((FROZEN / "analysis.py").read_bytes()).hexdigest()},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "observed_mean_gain_used_as_planned_alternative": False,
        "candidate_model_fits": 0, "rival_model_fits": 0, "new_model_inferences": 0,
        "new_calibrations": 0, "new_hypothesis_tests": 0, "R3_verified": False,
        "seed_variance_inputs": seed_inputs, "coupling": coupling,
        "seed_planning_rows": seed_rows, "calendar_planning_rows": calendar_rows,
        "limitations": [
            "The ten seeds and 608 dates were already known; their SDs are planning inputs, not a prospective pilot or observed power estimate.",
            "Noncentral-t results require iid normal paired gains and the assumed future SD; the SD upper bound additionally needs that model.",
            "The one-sided null is gain<=delta; planned excess is g1-delta. The observed positive mean gain is never inserted as the alternative.",
            "Bonferroni alpha/48 is a conservative reference for every Holm test. No power for the adaptive Holm procedure is claimed.",
            "The higher marginal targets imply a joint bound ONLY if all corresponding tests attain those marginal powers. Only the seed-t and a temporal normal reference are dimensioned here; bootstrap and finite-sample HAC battery power remain open.",
            "Historical gain correlations across ten seeds are descriptive, not future joint-power estimates. Seeds and dates are not interchangeable iid repetitions.",
            "Calendar calculations reuse a delta-method influence and HAC7/14/28. Future stationarity, dependence, marginal gain distribution and covariance can change; the ceilings are sensitivity references, not guaranteed forecast horizons.",
            "Extreme extrapolations are not operational sample-size recommendations. A uniform alternative just above delta for every weak and strong rival can be implausible; this diagnostic requires revisiting justified pilot assumptions before a future freeze, never weakening the standard after results.",
            "The inherited delta is in BTC sigma7 units. Other datasets require independently justified practical margins before their results are seen.",
            "A new admissible development pilot, common training/HPO policies, fixed procedures, selected datasets and a genuinely future protocol freeze are still required.",
        ]}
    return result


def family_summaries(result):
    """Worst conditional reference over sixteen rivals, separately by unit."""
    seed, calendar = [], []
    for variance in ("historical_sample_SD", "historical_upper_SD_simultaneous16", "twice_historical_SD_sensitivity"):
        for scenario, target in TARGETS:
            for multiplier in PLANNED_EXCESS_MULTIPLIERS:
                rows = [r for r in result["seed_planning_rows"] if r["variance_reference"] == variance
                        and r["target_scenario"] == scenario and r["planned_separation_as_delta_multiple"] == multiplier]
                if len(rows) != RIVALS:
                    raise ValueError("Do not summarize an incomplete family")
                nonestimable = [r["rival"] for r in rows if r["required_independent_seed_pairs_minimum10"] is None]
                maximum = None if nonestimable else max(r["required_independent_seed_pairs_minimum10"] for r in rows)
                seed.append({"variance_reference": variance, "target_scenario": scenario,
                    "marginal_target": target, "planned_excess_over_delta": multiplier,
                    "maximum_required_seed_pairs_across_16": maximum,
                    "binding_rivals": [] if maximum is None else [r["rival"] for r in rows if r["required_independent_seed_pairs_minimum10"] == maximum],
                    "nonestimable_rivals": nonestimable, "scope": "seed_t_component_only_not_full_battery"})
    for lag in (7, 14, 28):
        for scenario, target in TARGETS:
            for multiplier in PLANNED_EXCESS_MULTIPLIERS:
                gap = multiplier * DELTA
                rows = [r for r in result["calendar_planning_rows"] if r["HAC_Bartlett_lags"] == lag
                        and r["target_scenario"] == scenario and r["planned_separation_g1_minus_delta"] == gap]
                if len(rows) != RIVALS:
                    raise ValueError("Do not summarize an incomplete calendar family")
                maximum = max(r["calendar_dates_ceiling_minimum608"] for r in rows)
                calendar.append({"HAC_Bartlett_lags": lag, "target_scenario": scenario,
                    "marginal_target": target, "planned_excess_over_delta": multiplier,
                    "maximum_reference_calendar_dates_across_16": maximum,
                    "binding_rivals": [r["rival"] for r in rows if r["calendar_dates_ceiling_minimum608"] == maximum],
                    "scope": "temporal_normal_HAC_variance_reference_not_guaranteed_horizon"})
    return seed, calendar


def table_bytes(rows):
    fields = list(rows[0])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "work/e3_statistical_inputs")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.download:
        load_module("e3_frozen_inputs_for_download", "inputs.py").download_inputs(args.data)
    result = build(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    seed_summary, calendar_summary = family_summaries(result)
    summary = {k: v for k, v in result.items() if k not in {"seed_planning_rows", "calendar_planning_rows"}}
    summary.update({"seed_planning_row_count": len(result["seed_planning_rows"]),
        "calendar_planning_row_count": len(result["calendar_planning_rows"]),
        "seed_family_reference_summaries": seed_summary, "calendar_family_reference_summaries": calendar_summary})
    outputs = {"power_planning.json": (json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(),
        "seed_variance_inputs.csv": table_bytes(result["seed_variance_inputs"]),
        "seed_planning.csv": table_bytes(result["seed_planning_rows"]),
        "calendar_planning.csv": table_bytes(result["calendar_planning_rows"])}
    for name, raw in outputs.items():
        p = args.output / name
        if p.exists() and p.read_bytes() != raw:
            raise RuntimeError("Preserve existing unequal output; choose a new folder: " + str(p))
        p.write_bytes(raw)
    manifest = {name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} for name, raw in outputs.items()}
    (args.output / "manifest.json").write_text(json.dumps({"files": manifest}, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "seed_rows": len(result["seed_planning_rows"]),
        "calendar_rows": len(result["calendar_planning_rows"]), "R3_verified": False, "output": str(args.output)}))


if __name__ == "__main__":
    main()
