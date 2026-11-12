"""Conditional paired power planning from the amended, dated E2 pilot.

No market loader or model fitting is provided. The caller verifies artifact
bytes and provenance before analysis; hashes here bind those verified receipts
to the supplied predictions. Synthetic fixtures can exercise the calculation
but can never close R. Reported power is conditional on a five-seed empirical
pilot model, not achieved power or measured power for unpiloted contrasts.
"""
from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
import math
import re

import numpy as np

VERSION = "e2-amended-pilot-power-v1"
DELTA = 0.00025772842664981106
ALPHA_PAIR = .05 / 20
MODELS = ("TimeMixer", "PatchTST", "GRU32")
SEEDS = tuple(range(271701, 271706))
RECIPES = {"TimeMixer": "TM-D7-PUBLISHED-v1", "PatchTST": "PT-D7-PUBLISHED-v1",
           "GRU32": "GRU32-E1-ref-E2-HPO-v1"}
CONTRASTS = (("TimeMixer", "GRU32"), ("TimeMixer", "Persistence"),
             ("PatchTST", "GRU32"), ("PatchTST", "Persistence"))
R_VALUES = tuple(range(6, 31))
Z95_ONE_SIDED = 1.6448536269514722


def canonical_sha256(value):
    """Canonical JSON digest used to bind payload arrays and frozen context."""
    def plain(item):
        if isinstance(item, np.ndarray):
            return plain(item.tolist())
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, dict):
            return {str(key): plain(val) for key, val in item.items()}
        if isinstance(item, (tuple, list)):
            return [plain(val) for val in item]
        return item
    data = json.dumps(plain(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def _hash(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(name + " must be an explicit SHA256")


def _calendar(values):
    expected = [(date(2024, 11, 8) + timedelta(days=i)).isoformat() for i in range(53)]
    if list(values) != expected:
        raise ValueError("Keep all 53 registered UTC dates 2024-11-08 through 2024-12-30, without compressing gaps")
    return expected


def _numeric(values, mask, name):
    raw = list(values)
    if len(raw) != len(mask):
        raise ValueError(name + " must align with the complete registered calendar")
    array = np.zeros(len(mask), dtype=np.float64)
    try:
        array[mask] = np.asarray([raw[i] for i in np.flatnonzero(mask)], dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(name + " has invalid eligible values") from error
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(name + " requires finite nonnegative values on every common eligible date")
    return array


def _validate(payload, expected_context, scope):
    if scope not in ("registered_market_pilot", "synthetic_fixture"):
        raise ValueError("Declare registered_market_pilot or synthetic_fixture")
    dates = _calendar(payload["dates"])
    raw_mask = payload.get("eligible_mask", [True] * len(dates))
    if len(raw_mask) != len(dates) or any(type(value) is not bool for value in raw_mask):
        raise ValueError("Use one explicit common boolean eligibility mask")
    mask = np.asarray(raw_mask, dtype=bool)
    if mask.sum() < 28:
        raise ValueError("Fewer than 28 common eligible dates cannot support this registered pilot analysis")
    if not isinstance(expected_context, dict) or expected_context.get("model_recipes") != RECIPES:
        raise ValueError("Bind the three actual selected recipe IDs in expected_context")
    for key in ("data_sha256", "recipe_freeze_sha256", "execution_policy_sha256"):
        _hash(expected_context.get(key), "expected_context." + key)
    provenance = payload.get("provenance", {})
    if scope == "registered_market_pilot":
        if (provenance.get("synthetic_only") is not False
                or provenance.get("market_training_executed") is not True
                or provenance.get("artifact_hashes_verified") is not True):
            raise ValueError("A synthetic/unverified pilot cannot close R; verified real-market artifacts are required")
    if provenance.get("context_sha256") != canonical_sha256(expected_context):
        raise ValueError("Pilot provenance is not bound to the frozen expected context")
    for key, raw in (("truth_sha256", payload["truth"]), ("persistence_sha256", payload["persistence"]),
                     ("dates_sha256", dates), ("eligible_mask_sha256", raw_mask)):
        if provenance.get(key) != canonical_sha256(raw):
            raise ValueError("Pilot " + key + " does not match its supplied payload")
    supplied = payload.get("predictions")
    if not isinstance(supplied, dict) or set(supplied) != set(MODELS):
        raise ValueError("Exactly TimeMixer, PatchTST and GRU32 must share the pilot")
    predictions = []
    for model in MODELS:
        rows = supplied[model]
        if not isinstance(rows, dict) or set(map(str, rows)) != set(map(str, SEEDS)) or len(rows) != 5:
            raise ValueError("Every pilot model requires the five registered seed IDs")
        canonical = {str(seed): values for seed, values in rows.items()}
        predictions.append([_numeric(canonical[str(seed)], mask, model) for seed in SEEDS])
    records = provenance.get("records", [])
    if not isinstance(records, list) or len(records) != 15:
        raise ValueError("Fifteen distinct completed pilot-fit receipts are required")
    seen, model_identities = set(), {}
    for record in records:
        model, seed = record.get("model_id"), record.get("seed")
        if model not in MODELS or type(seed) is not int or seed not in SEEDS or (model, seed) in seen:
            raise ValueError("The pilot receipt model/seed domain differs or contains duplicates")
        seen.add((model, seed))
        if (record.get("status") != "completed" or record.get("fresh_initialization") is not True
                or type(record.get("optimizer_steps")) is not int or record["optimizer_steps"] < 1):
            raise ValueError("Pilot fits must be completed, fresh and have known optimizer updates")
        if record.get("recipe_id") != RECIPES[model]:
            raise ValueError("An old or different recipe cannot impersonate the selected pilot body")
        for key in ("prepared_signature", "model_source_sha256", "hpo_sha256", "artifact_sha256"):
            _hash(record.get(key), "pilot receipt " + key)
        for key in ("data_sha256", "execution_policy_sha256"):
            if record.get(key) != expected_context[key]:
                raise ValueError("Pilot receipt " + key + " differs from the frozen context")
        identity = {key: record[key] for key in
                    ("recipe_id", "prepared_signature", "model_source_sha256", "hpo_sha256")}
        if model in model_identities and model_identities[model] != identity:
            raise ValueError("The five seeds must share one selected recipe, preparation and HPO configuration per model")
        model_identities[model] = identity
        if record.get("dates_sha256") != provenance["dates_sha256"]:
            raise ValueError("A pilot fit changed the common prediction calendar")
        if record.get("eligible_mask_sha256") != provenance["eligible_mask_sha256"]:
            raise ValueError("A pilot fit changed the common eligibility mask")
        raw = supplied[model].get(str(seed), supplied[model].get(seed))
        if record.get("predictions_sha256") != canonical_sha256(raw):
            raise ValueError("Pilot predictions differ from their verified fit receipt")
        if record.get("train_cutoff") != "2024-10-31" or record.get("recipe_selection_origin") != "2024-10-01":
            raise ValueError("The registered pilot cutoff and October recipe basis must remain fixed")
    truth = _numeric(payload["truth"], mask, "pilot truth")
    persistence = _numeric(payload["persistence"], mask, "Persistence")
    array = np.asarray(predictions, dtype=np.float64)
    errors = array - truth[None, None, :]
    errors[:, :, ~mask] = 0
    baseline_errors = persistence - truth
    baseline_errors[~mask] = 0
    squared = errors * errors
    baseline_squared = baseline_errors * baseline_errors
    if not np.isfinite(squared).all() or not np.isfinite(baseline_squared).all():
        raise ValueError("Pilot squared errors overflow")
    return dates, mask, squared, baseline_squared


def _wilson_lower(successes, total):
    p, z = successes / total, Z95_ONE_SIDED
    return max(0., (p + z*z/(2*total) - z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))) / (1 + z*z/total))


def _required_successes(total, target=.8):
    low, high = 0, total
    while low < high:
        middle = (low + high) // 2
        if _wilson_lower(middle, total) >= target:
            high = middle
        else:
            low = middle + 1
    return low


def _conservative_quantile_index(total, tail_alpha=ALPHA_PAIR, confidence=.95):
    """Upper one-sided binomial/order-statistic bound for the null quantile.

    If K~Binomial(total, tail_alpha), choose its lower .05 quantile m. Then
    X_(total-m+1) exceeds the desired null quantile with probability >=.95.
    This protects calibration Monte Carlo error, not uncertainty in pilot data.
    """
    log_q, log_p = math.log1p(-tail_alpha), math.log(tail_alpha)
    cdf, count = 0., 0
    while cdf < 1 - confidence and count <= total:
        log_probability = (math.lgamma(total + 1) - math.lgamma(count + 1)
                           - math.lgamma(total - count + 1) + count*log_p + (total-count)*log_q)
        cdf += math.exp(log_probability)
        if cdf < 1 - confidence:
            count += 1
    index = total - count  # zero-based index for X_(total-m+1)
    if index >= total or count > total:
        raise ValueError("Insufficient null simulations to calibrate the Bonferroni tail")
    return index


def _block_counts(draws, days, length, rng):
    starts = rng.integers(0, days, size=(draws, math.ceil(days / length)))
    indices = ((starts[:, :, None] + np.arange(length)[None, None, :]) % days).reshape(draws, -1)[:, :days]
    counts = np.zeros((draws, days), dtype=np.int16)
    np.add.at(counts, (np.repeat(np.arange(draws), days), indices.ravel()), 1)
    return counts


def analyze_pilot_power(payload, *, expected_context, scope="registered_market_pilot",
                        simulations=5000, calibration_simulations=20000, seed=272801):
    """Select common R only from verified real pilot artifacts and L14 scenarios.

    The working data-generating model is a paired temporal circular bootstrap
    plus a paired five-seed empirical distribution. A shared temporal component
    remains a variance floor as R grows. Seed resampling uses sqrt(5/4) to restore
    the sample-variance scale. The null is gain<=delta and the planning gain is
    2delta. The critical value comes from an independent, centered null sample.
    """
    if type(simulations) is not int or simulations < 2000:
        raise ValueError("At least 2000 alternative simulations are required")
    if type(calibration_simulations) is not int or calibration_simulations < 20000:
        raise ValueError("At least 20000 independent null-calibration simulations are required")
    if type(seed) is not int or seed < 0:
        raise ValueError("A nonnegative integer simulation seed is required")
    dates, mask, squared, baseline_squared = _validate(payload, expected_context, scope)
    names = {name: index for index, name in enumerate(MODELS)}
    observed = np.sqrt(squared[:, :, mask].mean(axis=-1))
    persistence_rmse = float(np.sqrt(baseline_squared[mask].mean()))
    null_n, alt_n = calibration_simulations, simulations
    draws = null_n + alt_n
    order_index = _conservative_quantile_index(null_n)
    successes_needed = _required_successes(alt_n)
    scenario_results = {}
    for length in (14, 7, 28):
        rng = np.random.default_rng(seed + length * 1000)
        # Independent temporal draws under the same empirical pilot model.
        counts = _block_counts(draws, len(dates), length, rng)
        eligible_counts = counts @ mask.astype(np.int16)
        if np.any(eligible_counts == 0):
            scenario_results[str(length)] = {"status": "not_estimable", "reason": "A temporal draw has no common eligible dates", "curves": []}
            continue
        boot = np.sqrt((counts.astype(np.float64) @ squared.reshape(15, len(dates)).T)
                       / eligible_counts[:, None]).reshape(draws, 3, 5)
        persistence_boot = np.sqrt((counts @ baseline_squared) / eligible_counts)
        differences = []
        contrast_info = []
        for model, baseline in CONTRASTS:
            modeled = boot[:, names[model], :]
            base = boot[:, names[baseline], :] if baseline != "Persistence" else persistence_boot[:, None]
            current = base - modeled
            differences.append(current)
            original_base = observed[names[baseline]] if baseline != "Persistence" else np.full(5, persistence_rmse)
            original = original_base - observed[names[model]]
            base_mean = float(original_base.mean())
            contrast_info.append({"model": model, "baseline": baseline,
                "observed_mean_gain": float(original.mean()),
                "observed_paired_seed_SD": float(original.std(ddof=1)),
                "baseline_mean_seed_RMSE": base_mean,
                "planning_alternative_feasible": base_mean >= 2*DELTA})
        difference = np.stack(differences, axis=1)  # draw × contrast × seed
        temporal = difference.mean(axis=2)
        temporal = temporal - temporal.mean(axis=0, keepdims=True)
        centered_seed = difference - difference.mean(axis=2, keepdims=True)
        temporal_sd = temporal.std(axis=0, ddof=1)
        curves = []
        for R in R_VALUES:
            weights = rng.multinomial(R, np.full(5, .2), size=draws) / R
            noise = temporal + math.sqrt(5/4) * np.sum(centered_seed * weights[:, None, :], axis=2)
            # Centers are removed once at the pilot-distribution level. Null and
            # alternative simulation sections remain independently sampled.
            for index, info in enumerate(contrast_info):
                calibration = noise[:null_n, index]
                alternative = noise[null_n:, index]
                estimable = (info["observed_paired_seed_SD"] > 0
                             and float(np.std(calibration, ddof=1)) > 0
                             and info["planning_alternative_feasible"])
                row = {**info, "R": R, "block_days": length,
                       "temporal_SD_floor": float(temporal_sd[index]),
                       "independent_calendar_blocks_approx": len(dates)/length,
                       "status": "conditional_planning_only" if estimable else "not_estimable"}
                if not estimable:
                    row.update(conditional_power=None, Wilson_one_sided95_lower=None,
                               reason="Zero/nonestimable paired seed variability, null variability, or planning gain exceeds observed baseline RMSE")
                    curves.append(row)
                    continue
                critical_noise = float(np.partition(calibration.copy(), order_index)[order_index])
                hits = int(np.count_nonzero(alternative + DELTA > critical_noise))
                mde_boundary = float(np.partition(alternative.copy(), alt_n-successes_needed)[alt_n-successes_needed])
                mde_gain = max(DELTA, float(np.nextafter(DELTA + critical_noise - mde_boundary, np.inf)))
                row.update(conditional_power=hits/alt_n, rejections=hits, alternative_simulations=alt_n,
                    Wilson_one_sided95_lower=_wilson_lower(hits, alt_n),
                    critical_gain_under_margin_null=DELTA+critical_noise,
                    null_boundary_rejection_rate_diagnostic=float(np.mean(alternative > critical_noise)),
                    conditional_MDE_raw_gain_for_Wilson80=mde_gain,
                    conditional_MDE_separation_from_delta=mde_gain-DELTA,
                    calibration_simulations=null_n, null_quantile_order_index_one_based=order_index+1,
                    calibration_quantile_confidence=.95)
                curves.append(row)
        scenario_results[str(length)] = {"status": "conditional_planning_only", "curves": curves}
    primary = scenario_results["14"]
    acceptable, marginal_curve = [], []
    if primary["status"] == "conditional_planning_only":
        for R in R_VALUES:
            at_R = [row for row in primary["curves"] if row["R"] == R]
            if len(at_R) == 4 and all(row.get("Wilson_one_sided95_lower") is not None
                                    and row["Wilson_one_sided95_lower"] >= .8 for row in at_R):
                acceptable.append(R)
            estimable = len(at_R) == 4 and all(row.get("Wilson_one_sided95_lower") is not None for row in at_R)
            marginal_curve.append({"R": R, "all_four_estimable": estimable,
                "minimum_conditional_power": min(row["conditional_power"] for row in at_R) if estimable else None,
                "minimum_Wilson_one_sided95_lower": min(row["Wilson_one_sided95_lower"] for row in at_R) if estimable else None})
    candidate = min(acceptable) if acceptable else None
    real = scope == "registered_market_pilot"
    cap_rows = [row for row in primary.get("curves", []) if row["R"] == 30]
    mdes = [row.get("conditional_MDE_raw_gain_for_Wilson80") for row in cap_rows]
    return {"version": VERSION, "scope": scope,
        "status": "conditional_R_planning_closed" if real and candidate is not None else
                  "synthetic_fixture_not_R_closure" if not real else "power_target_unclosed",
        "R_selected": candidate if real else None, "fixture_R_candidate": candidate if not real else None,
        "R_operative_resource_recommendation": None if candidate is not None else 30,
        "resource_recommendation_closes_power": False,
        "power_target_met_for_measured_primary_scenarios": bool(candidate is not None),
        "achieved_final_power": None, "measured_power_for_all_twenty_contrasts": False,
        "alpha_family": .05, "family_size": 20, "planning_per_contrast_alpha": ALPHA_PAIR,
        "margin_null_gain": DELTA, "planning_true_gain": 2*DELTA,
        "primary_block_days": 14, "diagnostic_block_days": [7, 28],
        "models": list(MODELS), "pilot_seed_ids": list(SEEDS), "dates": dates,
        "common_eligible_dates": int(mask.sum()), "calendar_days": len(dates),
        "simulation_seed": seed, "simulations_per_R": simulations,
        "calibration_simulations_per_R": calibration_simulations,
        "context_sha256": canonical_sha256(expected_context),
        "conditional_worst_MDE_gain_at_R30": max(mdes) if len(mdes) == 4 and all(v is not None for v in mdes) else None,
        "primary_minimum_marginal_curve": marginal_curve,
        "unpiloted_envelope_scope": "Worst measured representative scenarios are a planning proxy only; they are not a proven variance upper bound for the other sixteen contrasts.",
        "scenarios": scenario_results,
        "method": "Paired circular date blocks; temporal mean floor plus centered paired empirical seed residuals scaled sqrt(5/4); independent null calibration with conservative upper95% order-statistic quantile; unilateral alternative2delta versus margin null delta.",
        "limits": [
            "Conditional plugin planning, not achieved final-study power or a distribution-free guarantee.",
            "Only four representative contrasts are measured; the other sixteen and a future regime are unobserved.",
            "Wilson bounds describe alternative Monte Carlo error only; calibration order bounds address calibration Monte Carlo error only.",
            "Neither Monte Carlo bound includes uncertainty from five pilot seeds, temporal dependence estimates or regime transport.",
            "The 53-day pilot has about3.79 length14 blocks; simulation draws are not new market observations.",
            "The temporal variance floor is not divided by R and Persistence has no artificial seed replication.",
            "The alternative is a local location-shift model on the mean-individual-RMSE contrast, not a generative prediction model; variance/dependence are assumed unchanged under the hypothetical gain.",
            "Pointwise Monte Carlo intervals across inspected R values are not a simultaneous confidence statement.",
            "The caller remains responsible for actual artifact-byte/provenance verification; flags alone cannot establish market origin.",
            "The synthetic fixture scope can never produce a registered R selection."]}
