"""Conservative development-pilot decisions and conditional power planning.

This module fits no models and opens no data files. A resource/preselection
pilot cannot impersonate the original postselection power study. Power values
below are conditional planning sensitivities, never achieved final-study power.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import itertools
import math

import numpy as np
from scipy.stats import chi2, norm, rankdata, t as student_t

try:
    from . import e2_search_plan as planner
except ImportError:
    import e2_search_plan as planner

VERSION = "e2-development-pilot-decisions-v1"


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name+" must be a finite scalar")
    return float(value)


def _pattern_id(row):
    explicit = row.get("pattern_id")
    architecture = row.get("architecture", row)
    widths, bits = architecture.get("widths"), architecture.get("dropout_pattern")
    derived = None
    if widths is not None or bits is not None:
        if (not isinstance(widths, (list, tuple)) or len(widths) != 3
                or not isinstance(bits, (list, tuple)) or len(bits) != 3):
            raise ValueError("Explicit architecture widths/bits must have three coordinates")
        derived = "w"+"-".join(map(str,widths))+":b"+"".join(map(str,bits))
    if explicit is not None and derived is not None and explicit != derived:
        raise ValueError("Pattern ID does not match supplied architecture")
    if explicit is None and derived is None:
        raise ValueError("Noise run needs a registered pattern_id or widths/bits")
    return explicit if explicit is not None else derived


def decide_k(protocol, noise_runs):
    """Evaluate the exact 12x5x6 rank rule, or choose the conservative upper k.

    Missing/censored rows do not evaluate the original rank rule. Their k40 is
    explicitly a conservative operational decision, not a synthetic completion
    of unexecuted models or architectures.
    """
    planner.validate_protocol(protocol)
    names = [row["model"] for row in protocol["models"] if row["nas"]]
    patterns = [row["pattern_id"] for row in planner.noise_pilot_architectures(protocol)]
    seeds = list(protocol["budget"]["noise_pilot_seed_values"])
    if len(patterns) != 12 or len(seeds) != 5 or len(names) != 6:
        raise ValueError("This rank decision implements exactly the registered12x5x6 pilot")
    expected = set(itertools.product(names, patterns, seeds))
    rows = {}
    acceptable = {"completed", "success", "failed", "censored", "timeout",
                  "resource_limit", "not_run"}
    for run in noise_runs:
        if not isinstance(run, dict):
            raise TypeError("Noise runs must be explicit dictionaries")
        key = (run.get("model"), _pattern_id(run), run.get("seed"))
        if type(run.get("seed")) is not int:
            raise ValueError("Noise-pilot seed must be an exact integer")
        if key not in expected:
            raise ValueError("Noise run outside the registered family/pattern/seed domain")
        if key in rows:
            raise ValueError("A noise-pilot attempt cannot be repeated or replaced")
        status = run.get("status")
        if status not in acceptable:
            raise ValueError("Unknown noise-pilot run status")
        copied = dict(run)
        if status in ("completed", "success"):
            value = _finite(run.get("validation_rmse"), "validation_rmse")
            if value < 0:
                raise ValueError("Validation RMSE cannot be negative")
            copied["validation_rmse"] = value
        rows[key] = copied
    missing = expected-set(rows)
    unfinished = [key for key,row in rows.items() if row["status"] not in ("completed", "success")]
    full = not missing and not unfinished
    per_model = {}
    any_unstable = False
    for name in names:
        complete = all(key in rows and rows[key]["status"] in ("completed", "success")
                       for key in itertools.product((name,), patterns, seeds))
        record = {"expected_rows": len(patterns)*len(seeds),
                  "completed_rows": sum(key[0] == name and row["status"] in ("completed", "success")
                                        for key,row in rows.items()),
                  "median_pairwise_spearman": None, "pairwise_correlations": [],
                  "complete": complete}
        if complete:
            scores = np.array([[rows[(name, pattern, seed)]["validation_rmse"]
                                for pattern in patterns] for seed in seeds], dtype=float)
            ranks = np.stack([rankdata(values, method="average") for values in scores])
            centered = ranks-ranks.mean(axis=1,keepdims=True)
            correlations = []
            for first,second in itertools.combinations(range(len(seeds)),2):
                denominator = float(np.linalg.norm(centered[first])*np.linalg.norm(centered[second]))
                value = None if denominator == 0 else float(np.clip(
                    np.dot(centered[first],centered[second])/denominator,-1,1))
                correlations.append(value)
                record["pairwise_correlations"].append({"seeds":[seeds[first],seeds[second]],
                                                         "spearman":value})
            if all(value is not None for value in correlations):
                record["median_pairwise_spearman"] = float(np.median(correlations))
            # Do not selectively discard a constant-seed ranking to obtain an
            # apparently stable median from only the remaining pairs.
            if record["median_pairwise_spearman"] is None or record["median_pairwise_spearman"] < .5:
                any_unstable = True
        per_model[name] = record
    recommendation = 40 if not full or any_unstable else 20
    return {"version":VERSION, "k_recommendation":recommendation,
            "status":"full_rank_rule_evaluated" if full else "not_full_rank_rule_evaluated",
            "full_rank_rule_evaluated":full,
            "decision_basis":"registered_rank_rule" if full else "conservative_upper_option_for_incomplete_or_censored_pilot",
            "expected_rows":len(expected), "provided_rows":len(rows),
            "completed_rows":sum(row["status"] in ("completed", "success") for row in rows.values()),
            "missing_rows":len(missing), "unfinished_rows":len(unfinished),
            "per_model":per_model,
            "limits":["k40 is not a guarantee of recovering the best of1000 architectures.",
                      "Missing or censored fits do not count as observed rank correlations."]}


def decide_R(protocol, *, selected_recipes=None, multiplicity=None, power_runs=None,
             reference_analysis=None):
    """Gate the original postselection study separately from a reference pilot."""
    planner.validate_protocol(protocol)
    reasons = []
    expected_names = protocol["budget"]["power_pilot_model_names"]
    if not isinstance(selected_recipes,dict) or any(name not in selected_recipes for name in expected_names):
        reasons.append("Missing actual recipes selected at the declared2024-10-01 inner origin")
    if not isinstance(multiplicity,dict) or multiplicity.get("preregistered") is not True:
        reasons.append("Missing preregistered inferential family and multiplicity specification")
    if not power_runs:
        reasons.append("Original postselection power pilot is not complete")
    if not reasons:
        reasons.append("Original pilot still requires its preregistered temporal power analysis; metadata alone cannot close R")
    return {"version":VERSION,"status":"original_postselection_R_not_closed", "R_selected":None,
            "achieved_power":None,"blocking_reasons":reasons,
            "R_cap":protocol["budget"]["max_final_seeds"],
            "R_scenarios":[{"R":value,"achieved_power":None,"interpretation":"resource scenario, not achieved power"}
                           for value in protocol["budget"]["R_budget_cases"]],
            "reference_pilot_analysis_available":reference_analysis is not None,
            "reference_pilot_R_proposal":None if reference_analysis is None else reference_analysis.get("reference_pilot_R_recommendation"),
            "limits":["A fixed-reference preselection pilot does not become the original postselection pilot.",
                      "R increases initialization replicates, not independent dates or markets."]}


def budget_report(protocol, k, R, *, resource_runs=()):
    counts = planner.budget(protocol,k,R)
    short_cap = protocol["hardware"]["short_stage_wall_limit_seconds"]
    full_cap = protocol["hardware"]["full_stage_wall_limit_seconds"]
    statuses = {}
    for run in resource_runs:
        if not isinstance(run,dict) or not isinstance(run.get("status"),str):
            raise ValueError("Resource runs need explicit statuses")
        statuses[run["status"]] = statuses.get(run["status"],0)+1
    return {"version":VERSION,"phase_counts":counts,
            "attempt_time_allowances":{
                "short_stage_cap_seconds":short_cap,"full_stage_cap_seconds":full_cap,
                "short_phase_allowance_seconds":counts["short_phase_total"]*short_cap,
                "full_phase_allowance_seconds":counts["full_phase_total"]*full_cap,
                "neural_allowance_sum_seconds":counts["short_phase_total"]*short_cap+counts["full_phase_total"]*full_cap,
                "classical_timing_included":False,
                "interpretation":"Sum of declared per-stage attempt allowances, not an estimated duration or complete-campaign wall bound"},
            "resource_run_status_counts":statuses,"measured_conditional_projection":None,
            "total_campaign_estimated_hours":None,"total_campaign_monetary_cost":None,
            "limits":["Reference measurements do not identify the average cost of all1000 architectures.",
                      "A censored180-second reconnaissance is distinct from the7200-second full-stage allocation.",
                      "Classical timing and all components excluded by the budget remain additional.",
                      "Boundary-checked limits alone are not hard process isolation."]}


def _calendar(dates, length):
    if dates is None or len(dates) != length:
        raise ValueError("Explicit daily calendar is required; compressed dates cannot be assumed consecutive")
    parsed = []
    for value in dates:
        text = value.isoformat() if hasattr(value,"isoformat") else str(value)
        stamp = datetime.fromisoformat(text.replace("Z","+00:00"))
        stamp = stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp.astimezone(timezone.utc)
        if stamp.hour or stamp.minute or stamp.second or stamp.microsecond:
            raise ValueError("Power-pilot dates must be explicit UTC midnights")
        if stamp >= datetime(2025,1,1,tzinfo=timezone.utc):
            raise ValueError("This development-pilot analysis does not accept the final period")
        parsed.append(stamp)
    if any(b-a != timedelta(days=1) for a,b in zip(parsed,parsed[1:])):
        raise ValueError("Power-pilot calendar must retain every consecutive day")
    return [value.date().isoformat() for value in parsed]


def _wilson(successes, total):
    z = float(norm.ppf(.975))
    p = successes/total
    denominator = 1+z*z/total
    center = (p+z*z/(2*total))/denominator
    radius = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [max(0.,center-radius),min(1.,center+radius)]


def conditional_power_analysis(predictions, y, delta, m=153, *, alpha=.05,
                               power_target=.8, R_values=(6,10,20,30), draws=5000,
                               block_length=7, seed=272001, model_names=None,
                               seed_ids=None, dates=None):
    """Preselection fixed-reference pilot: conditional planning, not achieved power.

    Seed variability is estimated from paired per-seed RMSEs. Temporal circular
    blocks are shared by every model/seed. Their common-date variability remains
    a variance floor when R increases. Seed residuals are paired and inflated to
    a Gaussian-assumption95% upper SD sensitivity, and the analytical t critical
    value is used;5000 draws cannot resolve the extreme Bonferroni tail itself.
    """
    prediction = np.asarray(predictions,dtype=np.float64)
    truth = np.asarray(y,dtype=np.float64)
    if prediction.ndim != 3 or prediction.shape[0] not in (2,3) or truth.shape != (prediction.shape[-1],):
        raise ValueError("Expected predictions[2or3 models,seeds,dates] and aligned scalar targets")
    models,seeds_n,n = prediction.shape
    if (not np.isfinite(prediction).all() or not np.isfinite(truth).all()
            or (prediction < 0).any() or (truth < 0).any() or truth.mean() <= 0):
        raise ValueError("Complete nonnegative finite predictions/targets are required; do not delete failed dates")
    delta = _finite(delta,"delta")
    alpha = _finite(alpha,"alpha")
    power_target = _finite(power_target,"power_target")
    if delta <= 0 or not 0 < alpha < 1 or not 0 < power_target < 1:
        raise ValueError("Positive delta and probability-valued alpha/power required")
    if type(m) is not int or m < 1 or type(draws) is not int or draws < 1000:
        raise ValueError("Positive integer family size and at least1000 simulation draws required")
    if type(block_length) is not int or block_length < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Positive block length and nonnegative simulation seed required")
    R_values = tuple(R_values)
    if (not R_values or len(set(R_values)) != len(R_values)
            or any(type(value) is not int or not 6 <= value <= 30 for value in R_values)):
        raise ValueError("Unique R scenarios must lie in6..30")
    calendar = _calendar(dates,n)
    names = list(model_names) if model_names is not None else ["model_"+str(i) for i in range(models)]
    if len(names) != models or any(not isinstance(name,str) for name in names) or len(set(names)) != models:
        raise ValueError("Models need unique labels")
    if seed_ids is not None and (len(seed_ids) != seeds_n or len(set(seed_ids)) != seeds_n
                                 or any(type(value) is not int for value in seed_ids)):
        raise ValueError("Seed IDs must be distinct and match the prediction axis")
    base = {"version":VERSION,"study_role":"amended_fixed_reference_preselection_pilot",
            "achieved_final_power":None,"alpha":alpha,"multiplicity_family_size":m,
            "per_comparison_alpha":alpha/m,"delta":delta,"power_target":power_target,
            "models":names,"seed_ids":seed_ids,"calendar":calendar,"draws_per_R":draws,
            "block_length":block_length,"simulation_seed":seed,
            "reference_pilot_R_recommendation":None}
    if seeds_n < 5 or n < 4*block_length:
        return {**base,"status":"insufficient_pilot_support",
                "reason":"Need at least5 seeds and4 calendar blocks for this conditional sensitivity",
                "pairs":[]}
    errors = prediction-truth[None,None,:]
    seed_rmse = np.sqrt(np.mean(errors*errors,axis=-1))
    if not np.isfinite(seed_rmse).all():
        raise ValueError("RMSE overflow; scores are not estimable")
    rng = np.random.default_rng(seed)
    blocks = math.ceil(n/block_length)
    starts = rng.integers(0,n,size=(draws,blocks))
    indices = ((starts[:,:,None]+np.arange(block_length)[None,None,:])%n).reshape(draws,-1)[:,:n]
    squared = errors*errors
    # [model,seed,draw] scores, paired on exactly the same resampled dates.
    boot_scores = np.sqrt(squared[:,:,indices].mean(axis=-1))
    alpha_pair = alpha/m
    z_critical = float(norm.ppf(1-alpha_pair/2))
    z_power = float(norm.ppf(power_target))
    df = seeds_n-1
    sd_upper_factor = math.sqrt(df/float(chi2.ppf(.05,df)))
    seed_draws = {R:np.random.default_rng(seed+1000+R).integers(0,seeds_n,size=(draws,R))
                  for R in range(6,31)}
    pairs = []
    for first,second in itertools.combinations(range(models),2):
        differences = seed_rmse[first]-seed_rmse[second]
        sd = float(differences.std(ddof=1))
        record = {"models":[names[first],names[second]],
                  "difference_definition":"RMSE(first)-RMSE(second)",
                  "paired_seed_differences":differences.tolist(),
                  "observed_mean_difference":float(differences.mean()),
                  "sigma_d":sd,"effect_delta_over_sigma_d":None,
                  "normal_R_approximation":None,"normal_R_SD_upper_sensitivity":None,
                  "scenarios":[],"status":"not_estimable"}
        if not math.isfinite(sd) or sd <= 0:
            record["reason"]="Seed variability is zero/nonestimable; do not infer perfect power"
            pairs.append(record)
            continue
        upper = sd*sd_upper_factor
        record.update({"status":"conditional_planning_only", "sigma_d_upper95_normal_assumption":upper,
                       "effect_delta_over_sigma_d":delta/sd,
                       "normal_R_approximation":int(math.ceil(((z_critical+z_power)*sd/delta)**2)),
                       "normal_R_SD_upper_sensitivity":int(math.ceil(((z_critical+z_power)*upper/delta)**2))})
        boot_difference = (boot_scores[first]-boot_scores[second]).T  # draw x existing seed
        temporal_mean = boot_difference.mean(axis=1)
        temporal_shift = temporal_mean-temporal_mean.mean()
        temporal_sd = float(temporal_mean.std(ddof=1))
        centered = boot_difference-temporal_mean[:,None]
        # Resampling a finite empirical population uses ddof0. This factor
        # restores its ddof1 scale before applying the upper-SD sensitivity.
        inflation = sd_upper_factor*math.sqrt(seeds_n/(seeds_n-1))
        record["temporal_SD_floor"] = temporal_sd
        simulations = []
        for R in range(6,31):
            noise = temporal_shift+np.take_along_axis(centered,seed_draws[R],axis=1).mean(axis=1)*inflation
            standard_error = math.sqrt(upper*upper/R+temporal_sd*temporal_sd)
            critical = float(student_t.ppf(1-alpha_pair/2,R-1))*standard_error
            hits = int(np.count_nonzero(np.abs(noise+delta)>critical))
            interval = _wilson(hits,draws)
            approximate_mde = (z_critical+z_power)*standard_error
            # Conditional simulated effect needed for the80% planning target,
            # under the same declared temporal/initialization noise model.
            low,high = 0.,max(delta,approximate_mde)*4
            for _ in range(20):
                if np.mean(np.abs(noise+high)>critical) >= power_target:
                    break
                high *= 2
            for _ in range(35):
                middle = (low+high)/2
                if np.mean(np.abs(noise+middle)>critical) >= power_target:
                    high = middle
                else:
                    low = middle
            simulations.append({"R":R,"conditional_simulated_rejection_probability":hits/draws,
                                "conditional_simulated_null_rejection_probability":float(np.mean(np.abs(noise)>critical)),
                                "Monte_Carlo_CI95":interval,
                                "conditional_MDE_normal":approximate_mde,
                                "conditional_MDE_simulated":high,
                                "includes_temporal_floor":True})
        # Require the MC lower bound at this R and every larger inspected R,
        # rather than selecting an isolated upward Monte Carlo fluctuation.
        acceptable = [row["R"] for row in simulations if all(
            other["Monte_Carlo_CI95"][0] >= power_target for other in simulations if other["R"] >= row["R"])]
        record["conditional_R_required_in6_to30"] = min(acceptable) if acceptable else None
        record["conditional_target_reached_within_cap"] = bool(acceptable)
        record["scenarios"] = [row for row in simulations if row["R"] in R_values]
        record["all_R_simulation_results"] = simulations
        pairs.append(record)
    all_estimable = all(row["status"] == "conditional_planning_only" for row in pairs)
    normal_upper_required = max(row["normal_R_SD_upper_sensitivity"] for row in pairs) if all_estimable else None
    attained = (all_estimable and normal_upper_required <= 30
                and all(row["conditional_R_required_in6_to30"] is not None for row in pairs))
    proposal = max(6,normal_upper_required,max(row["conditional_R_required_in6_to30"] for row in pairs)) if attained else 30
    return {**base,"status":"conditional_preselection_planning_not_final_power",
            "reference_pilot_R_recommendation":proposal,
            "reference_pilot_R_decision_basis":"observed_pilot_pairs_conditional_target" if attained else "resource_cap_with_target_unclosed",
            "all_observed_pilot_pairs_meet_conditional_target_by30":bool(attained),
            "R_required_normal_upper_across_observed_pairs":normal_upper_required,
            "pairs":pairs,"independent_calendar_blocks_approx":n/block_length,
            "normal_SD_upper_factor":sd_upper_factor,
            "limits":[
                "This fixed-reference preselection pilot is an amendment, not the original postselection study.",
                "The simulation is conditional planning sensitivity, not achieved final-study power or a distribution-free bound.",
                "Power is against zero with a true difference delta; it does not prove superiority beyond the practical margin delta.",
                "For a practical-superiority null at delta, the same effective shift corresponds to an alternative at2delta.",
                "Multiplicity153 governs the critical value; variances of unobserved model pairs are not estimated.",
                "Five seeds give a fragile SD estimate; its95% upper sensitivity assumes Gaussian paired seed scores.",
                "Temporal block variability is not removed by adding initialization seeds.",
                "Only a few calendar blocks are observed; block7 and the empirical residual model may miss future regimes.",
                "At alpha=.05/153,5000 draws cannot resolve the two-sided Bonferroni tail empirically; the critical value is analytical.",
                "Monte Carlo intervals are pointwise uncertainty of the simulation, not scientific confidence intervals for performance.",
                "The selected resource proposal cannot close R for the final selected recipes or all18-system comparisons."
            ]}
