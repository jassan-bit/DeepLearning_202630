#!/usr/bin/env python3
"""Secondary frozen-predictor intervals and temporal OOF error analysis.

Default invocation describes a proposed procedure; it never loads prices or
trains/scores a model. Numerical functions accept explicitly supplied forecasts.
The calibration predictor is NOT the primary fit through December 2024.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "experiments/e2_remediation_20261106/uncertainty_errors.json"
DEV_START, DEV_END = date(2024, 1, 1), date(2024, 10, 31)
CAL_START, CAL_END = date(2024, 11, 1), date(2024, 12, 30)
FIT_MAX = date(2024, 10, 24)
EVAL_START, EVAL_END = date(2026, 9, 1), date(2026, 10, 30)
TAXONOMY_TRAIN_MAX = date(2023, 12, 24)
STAGES = {"P0_diagnostic", "E2_nested_development"}
CATEGORIES = ("low_reference_error", "moderate_under", "moderate_over", "moderate_mixed",
              "large_under", "large_over", "large_mixed")


def numeric():
    import numpy as np
    return np


def day(value):
    """ISO dates mean UTC by the explicit daily contract; offsets must be UTC."""
    if isinstance(value, datetime):
        value = value.isoformat()
    if isinstance(value, date):
        return value
    text = str(value)
    if len(text) == 10:
        return date.fromisoformat(text)
    moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if moment.tzinfo is None or moment.utcoffset() != timedelta(0) or moment.time().replace(tzinfo=None) != datetime.min.time():
        raise ValueError("Forecast origins require UTC midnight or an explicit ISO UTC date")
    return moment.date()


def calendar(start, end):
    start, end = day(start), day(end)
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def vector(values, name, nonnegative=False):
    np = numeric(); result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError(name + " must be a finite, nonempty vector")
    if nonnegative and (result < 0).any():
        raise ValueError(name + " must remain nonnegative in original volatility units")
    return result


def _origin(row):
    values = [day(row[k]) for k in ("origin_utc", "outcome_day", "origin") if k in row]
    if not values or len(set(values)) != 1:
        raise ValueError("Missing or conflicting daily forecast origins")
    return values[0]


def _eligible(value):
    if value in (True, "true", "True", 1, "1"):
        return True
    if value in (False, "false", "False", 0, "0"):
        return False
    raise ValueError("Eligibility must be explicit")


def validate_calibration_support(receipt, calibration_origins, evaluation_origins=None):
    """Validate nominal information supports and actual freeze/scoring ordering.

    This retrospective experiment cannot claim weights existed in November2024.
    Fit/calendar cutoffs are nominal; actual UTC timestamps only order computing
    operations. Hashes identify receipts, not the sufficiency of their assertions.
    """
    required = {"branch", "model", "fit_origin", "fit_outcome_max", "scaler_fit_outcome_max",
                "configuration_selection_outcome_max", "branch_selection_outcome_max", "primary_fit_replaced",
                "point_predictor_kind", "ensemble_seeds", "member_checkpoint_sha256", "member_run_ids",
                "member_scaler_sha256", "member_config_sha256",
                "checkpoint_closed_at_utc", "calibration_scoring_started_at_utc", "target_scale"}
    if not required.issubset(receipt):
        raise ValueError("Incomplete frozen-predictor receipt")
    if receipt["target_scale"] != "original_unannualized_daily_volatility":
        raise ValueError("Calibration and evaluation must use the inherited original target scale")
    if receipt.get("branch") != "secondary_frozen_calibration_A" or receipt.get("primary_fit_replaced") is not False:
        raise ValueError("Intervals require the explicit secondary branch, not a silently replaced primary fit")
    if day(receipt["fit_origin"]) != CAL_START:
        raise ValueError("The secondary predictor has nominal fit origin 2024-11-01")
    for key in ("fit_outcome_max", "scaler_fit_outcome_max"):
        if day(receipt[key]) > FIT_MAX:
            raise ValueError("Fit/scaler outcomes must stop by 2024-10-24, before the seven-day embargo")
    for key in ("configuration_selection_outcome_max", "branch_selection_outcome_max"):
        if day(receipt[key]) > FIT_MAX:
            raise ValueError("Branch-A configuration/Top3 selection must also stop by 2024-10-24")
    for key in ("refit_during_calibration", "calibration_used_for_weight_selection", "test_used_for_selection"):
        if receipt.get(key) is not False:
            raise ValueError("Calibration/test labels cannot refit, early-stop or select this frozen predictor")
    seeds = [str(s) for s in receipt["ensemble_seeds"]]
    kind = receipt.get("point_predictor_kind")
    if kind == "deterministic_frozen_predictor":
        if seeds != ["deterministic"]:
            raise ValueError("A deterministic reference is one predictor, not five independent replicas")
    elif kind != "mean_frozen_seed_members" or len(seeds) < 5 or len(set(seeds)) != len(seeds):
        raise ValueError("A learned secondary predictor requires at least five predeclared frozen seed members")
    for name in ("member_checkpoint_sha256", "member_scaler_sha256", "member_config_sha256"):
        hashes = receipt[name]
        if set(map(str, hashes)) != set(seeds) or any(len(str(h)) != 64 or any(c not in "0123456789abcdef" for c in str(h)) for h in hashes.values()):
            raise ValueError("Every frozen member's weights, scaler and configuration need SHA-256 receipts")
    if set(map(str, receipt["member_run_ids"])) != set(seeds) or len(set(receipt["member_run_ids"].values())) != len(seeds):
        raise ValueError("Frozen member run identities must be explicit and unique")
    frozen = datetime.fromisoformat(receipt["checkpoint_closed_at_utc"].replace("Z", "+00:00"))
    scoring = datetime.fromisoformat(receipt["calibration_scoring_started_at_utc"].replace("Z", "+00:00"))
    if (frozen.tzinfo is None or scoring.tzinfo is None or frozen.utcoffset() != timedelta(0)
            or scoring.utcoffset() != timedelta(0) or not frozen < scoring):
        raise ValueError("Every member must close before actual calibration scoring, with UTC timestamps")
    dates = [day(x) for x in calibration_origins]
    if dates != calendar(CAL_START, CAL_END):
        raise ValueError("Keep the full sixty-day calibration calendar 2024-11-01 through 2024-12-30")
    if evaluation_origins is not None:
        evaluation = [day(x) for x in evaluation_origins]
        if evaluation != calendar(EVAL_START, EVAL_END) or set(dates).intersection(evaluation):
            raise ValueError("Secondary evaluation retains the known Sep1-Oct30/2026 calendar, disjoint from calibration")
    return {"calibration_days": len(dates), "fit_outcome_max_allowed": FIT_MAX.isoformat(),
            "first_calibration_label_close_support": [(CAL_START-timedelta(days=7)).isoformat(), CAL_START.isoformat()],
            "overlap_with_fit_label_closes": False, "actual_freeze_precedes_scoring": True,
            "new_unknown_test": False, "nominal_dates_are_not_live_issuance_times": True}


def weighted_split_quantile(scores, weights, alpha, *, test_weight=1.0):
    """Finite weighted residual quantile, including a fixed future-point mass.

    This numerical correction does not establish exchangeability or a coverage
    theorem for recency weights, dependent sigma7 labels or distribution shift.
    Insufficient weighted mass raises, rather than returning zero or infinity.
    """
    np = numeric(); scores = vector(scores, "calibration scores", True); weights = vector(weights, "weights")
    if len(scores) != len(weights) or (weights <= 0).any() or not 0 < float(alpha) < 1:
        raise ValueError("Positive matching weights and alpha in (0,1) are required")
    if not math.isfinite(float(test_weight)) or test_weight <= 0:
        raise ValueError("The fixed test mass must be finite and positive")
    threshold = (1-float(alpha)) * (float(weights.sum())+float(test_weight))
    if not math.isfinite(threshold) or threshold > float(weights.sum()):
        raise ValueError("Insufficient calibration mass for a finite interval at this nominal level")
    order = np.argsort(scores, kind="stable"); cumulative = np.cumsum(weights[order])
    position = int(np.searchsorted(cumulative, threshold, side="left"))
    if position >= len(scores):
        raise ValueError("Weighted quantile is not finite at the requested level")
    return float(scores[order[position]])


def fit_temporal_calibrator(rows, receipt, *, alphas=(.20, .10, .05), half_life_days=28.0):
    """Calibrate the fixed mean-of-seeds predictor on its own disjoint residuals."""
    np = numeric()
    if not math.isfinite(float(half_life_days)) or half_life_days <= 0:
        raise ValueError("A positive predeclared half life is required")
    dates = calendar(CAL_START, CAL_END); support = validate_calibration_support(receipt, dates)
    model = receipt["model"]; seeds = list(map(str, receipt["ensemble_seeds"]))
    indexed = {}; truth = {}
    for row in rows:
        origin = _origin(row); seed = str(row["seed"])
        if row["model"] != model or seed not in seeds or origin not in dates or (seed, origin) in indexed:
            raise ValueError("Calibration must use exactly this model's frozen ensemble and calendar, without duplicates")
        if row.get("run_id") != receipt["member_run_ids"][seed]:
            raise ValueError("OOF or other-fit predictions cannot replace this frozen calibration predictor")
        if row.get("checkpoint_sha256", receipt["member_checkpoint_sha256"][seed]) != receipt["member_checkpoint_sha256"][seed]:
            raise ValueError("Calibration forecast checkpoint differs from the frozen predictor")
        for field,member_field in (("scaler_sha256","member_scaler_sha256"),("config_sha256","member_config_sha256")):
            if row.get(field, receipt[member_field][seed]) != receipt[member_field][seed]:
                raise ValueError("Calibration scaler/configuration differs from the frozen predictor")
        y = float(row["y_true"]); pred = float(row["y_pred"])
        if not math.isfinite(y) or not math.isfinite(pred) or min(y, pred) < 0:
            raise ValueError("Calibration forecasts and targets must be finite, nonnegative original-scale volatility")
        if origin in truth and truth[origin] != y:
            raise ValueError("Seeds must share identical calibration targets")
        truth[origin] = y; indexed[seed, origin] = pred
    if len(indexed) != len(seeds)*len(dates):
        raise ValueError("The sixty-day ensemble calibration panel is incomplete")
    pred = np.array([[indexed[s, d] for d in dates] for s in seeds]).mean(axis=0)
    y = np.array([truth[d] for d in dates]); scores = np.abs(pred-y)
    # Weights are fixed relative to CAL_END, not silently extrapolated as a new
    # online window across the twenty-month evaluation gap.
    weights = np.exp2(-np.array([(CAL_END-d).days for d in dates])/float(half_life_days))
    if len(set(float(a) for a in alphas)) != len(alphas):
        raise ValueError("Nominal interval levels must be unique")
    radii = {str(float(a)): weighted_split_quantile(scores, weights, float(a)) for a in alphas}
    signature = sha256(json.dumps({"model": model, "seeds": seeds, "checkpoints": receipt["member_checkpoint_sha256"],
                                  "scalers": receipt["member_scaler_sha256"], "configs": receipt["member_config_sha256"],
                                  "target_scale": receipt["target_scale"], "kind": receipt["point_predictor_kind"],
                                  "fit_outcome_max": receipt["fit_outcome_max"]}, sort_keys=True).encode()).hexdigest()
    return {"model": model, "predictor_signature": signature, "point_predictor": "fixed mean of frozen branch-A seed forecasts",
            "calibration_origins": len(dates), "radii_by_alpha": radii, "half_life_days": float(half_life_days),
            "weights_indexed_to_fixed_calibration_end": CAL_END.isoformat(), "effective_weight_count": float(weights.sum()**2/(weights@weights)),
            "support": support, "exact_finite_sample_coverage_claimed": False,
            "long_gap_and_regime_shift_validity_established": False, "primary_benchmark_fit_changed": False}


def symmetric_intervals(point, radius):
    np = numeric(); point = vector(point, "point predictions", True)
    if not math.isfinite(float(radius)) or radius < 0:
        raise ValueError("A finite nonnegative calibrated radius is required")
    return np.maximum(0, point-float(radius)), point+float(radius)


def pinball(target, quantile_prediction, tau):
    np = numeric(); y = vector(target, "target", True); q = vector(quantile_prediction, "quantile forecast", True)
    if len(y) != len(q) or not 0 < float(tau) < 1:
        raise ValueError("Aligned forecasts and a quantile level in (0,1) are required")
    error = y-q
    return np.maximum(float(tau)*error, (float(tau)-1)*error)


def interval_scores(target, point, intervals):
    """Per-origin coverage/width/pinball and WIS for declared quantile forecasts.

    The central point is submitted as the forecast at quantile .5; it is not
    asserted to equal the true conditional median. CRPS is not inferred from
    these few intervals, their widths or the variability of optimization seeds.
    """
    np = numeric(); y = vector(target, "target", True); center = vector(point, "central forecast", True)
    if len(y) != len(center) or not intervals or len(set(map(float, intervals))) != len(intervals):
        raise ValueError("Aligned points and at least one interval are required")
    result = {}; weighted = .5*np.abs(y-center); previous = None
    for alpha in sorted(map(float, intervals), reverse=True):
        original_key = next(k for k in intervals if float(k) == alpha)
        lower, upper = (vector(x, "interval bounds", True) for x in intervals[original_key])
        if (not 0 < alpha < 1 or len(lower) != len(y) or len(upper) != len(y)
                or (lower > upper).any() or (lower > center).any() or (upper < center).any()):
            raise ValueError("Finite nonnegative ordered intervals must contain the central forecast")
        if previous is not None and ((lower > previous[0]).any() or (upper < previous[1]).any()):
            raise ValueError("Intervals at larger nominal coverage must nest")
        score = upper-lower+2/alpha*np.maximum(lower-y, 0)+2/alpha*np.maximum(y-upper, 0)
        result[str(alpha)] = {"coverage": ((lower <= y)&(y <= upper)).astype(float), "width": upper-lower,
                              "interval_score": score, "pinball_lower": pinball(y, lower, alpha/2),
                              "pinball_upper": pinball(y, upper, 1-alpha/2)}
        weighted += alpha/2*score; previous = lower, upper
    result["WIS"] = weighted/(len(intervals)+.5)
    return result


def empirical_crps(target, distribution_samples, *, weights=None):
    """CRPS only for an explicitly supplied predictive empirical distribution.

    A caller must justify the sample distribution; five optimizer-seed forecasts
    alone do not establish predictive draws of future volatility.
    """
    np = numeric(); y = vector(target, "target", True); draws = np.asarray(distribution_samples, dtype=float)
    if draws.ndim != 2 or draws.shape[0] != len(y) or draws.shape[1] < 2 or not np.isfinite(draws).all() or (draws < 0).any():
        raise ValueError("CRPS requires finite nonnegative [origin, predictive_draw] samples")
    p = np.ones(draws.shape[1]) if weights is None else vector(weights, "empirical predictive weights")
    if len(p) != draws.shape[1] or (p <= 0).any():
        raise ValueError("Predictive distribution weights must match the finite draws and be positive")
    p = p/p.sum(); order = np.argsort(draws, axis=1, kind="stable")
    ordered = np.take_along_axis(draws, order, axis=1); probabilities = p[order]
    prior_mass = np.cumsum(probabilities, axis=1)-probabilities
    prior_value = np.cumsum(probabilities*ordered, axis=1)-probabilities*ordered
    pair_half = (probabilities*(ordered*prior_mass-prior_value)).sum(axis=1)
    return (np.abs(draws-y[:, None])*p[None,:]).sum(axis=1)-pair_half


def signed_residual_distribution(point, calibration_signed_residuals):
    """Explicit empirical CDF: frozen point plus calibration (truth-prediction).

    Zero clipping is part of the nonnegative predictive distribution, not an
    arbitrary Gaussian assumption. Its historical residual-transfer assumption
    is unproved across the 2024/2026 gap; report that limitation with CRPS.
    """
    np = numeric(); center = vector(point, "point forecast", True)
    residuals = vector(calibration_signed_residuals, "signed calibration residuals")
    return np.maximum(0, center[:,None]+residuals[None,:])


def block_indices(n, length, rng):
    np = numeric()
    if int(length) != length or int(n) != n or not 1 <= int(length) <= int(n):
        raise ValueError("Block length must lie between one and the full calendar length")
    n, length = int(n), int(length)
    starts = rng.integers(0, n, size=math.ceil(n/length))
    return ((starts[:, None]+np.arange(length)) % n).ravel()[:n]


def temporal_mean_summary(values, origins, *, block_length=14, repetitions=2000, seed=20261106, eligible_mask=None):
    np = numeric(); values = vector(values, "per-origin statistic"); dates = [day(x) for x in origins]
    if (len(values) != len(dates) or dates != calendar(dates[0], dates[-1])
            or int(repetitions) != repetitions or repetitions < 2000):
        raise ValueError("Use a complete daily calendar and at least 2000 block resamples")
    repetitions = int(repetitions)
    mask = np.ones(len(values), bool) if eligible_mask is None else np.asarray(eligible_mask, bool)
    if mask.shape != values.shape:
        raise ValueError("Subgroup eligibility must align to the full daily calendar")
    rng = np.random.default_rng(seed); boot = []
    for _ in range(repetitions):
        idx = block_indices(len(values), block_length, rng)
        if mask[idx].any():
            boot.append(float(values[idx][mask[idx]].mean()))
    estimable = bool(mask.any()) and len(boot) >= .95*repetitions
    return {"mean": float(values[mask].mean()) if mask.any() else None,
            "ci95_nominal": np.quantile(boot, [.025, .975]).tolist() if estimable else None,
            "eligible_group_origins": int(mask.sum()), "status": "estimated" if estimable else "not_estimable",
            "calendar_origins": len(values), "block_length_days": int(block_length), "bootstrap_replicates": int(repetitions),
            "method": "circular moving-block percentile; approximate temporal uncertainty, not independent markets",
            "simultaneous_or_confirmatory_claim": False}


def evaluate_interval_forecasts(target, point, origins, calibrator, *, evaluation_predictor_signature,
                                primary_comparable_point=None, subgroup_labels=None, forecast_bin_edges=None,
                                subgroup_receipt=None, subgroup_levels=None, predictive_distribution_samples=None,
                                predictive_distribution_weights=None, repetitions=2000, block_length=14):
    """Evaluate fixed intervals without replacing the primary benchmark predictor.

    Supplying future forecasts is a separate explicit action. This function does
    not fit, update calibration or select an alpha after observing errors.
    """
    np = numeric(); dates = [day(x) for x in origins]
    if dates != calendar(EVAL_START, EVAL_END):
        raise ValueError("Retain the sixty known historical evaluation origins; no new test is claimed")
    if evaluation_predictor_signature != calibrator["predictor_signature"]:
        raise ValueError("The evaluation predictor differs from the one that generated calibration residuals")
    y = vector(target, "target", True); point = vector(point, "branch-A point", True)
    if len(y) != len(dates) or len(point) != len(dates):
        raise ValueError("Predictions must align to the assigned evaluation calendar")
    intervals = {float(a): symmetric_intervals(point, r) for a,r in calibrator["radii_by_alpha"].items()}
    scores = interval_scores(y, point, intervals); result = {"intervals": {}}
    for alpha in intervals:
        result["intervals"][str(alpha)] = {name: temporal_mean_summary(values, dates, block_length=block_length,
                                                                         repetitions=repetitions)
                                          for name,values in scores[str(alpha)].items()}
    result["WIS"] = temporal_mean_summary(scores["WIS"], dates, block_length=block_length, repetitions=repetitions)
    result["CRPS"] = (temporal_mean_summary(empirical_crps(y, predictive_distribution_samples,
                                                            weights=predictive_distribution_weights), dates,
                                            block_length=block_length, repetitions=repetitions)
                      if predictive_distribution_samples is not None else
                      {"status": "not_computed", "reason": "No explicit predictive distribution supplied; interval width or seed spread is not CRPS"})
    groups = dict(subgroup_labels or {}); declared_levels = dict(subgroup_levels or {})
    if groups and set(groups) != set(declared_levels):
        raise ValueError("Predeclare the full subgroup-level registry, including empty groups")
    if forecast_bin_edges is not None:
        edges = vector(forecast_bin_edges, "train-frozen forecast-bin edges", True)
        if (np.diff(edges) <= 0).any():
            raise ValueError("Forecast-bin thresholds must be strictly increasing and frozen in training")
        groups["forecast_range_train_thresholds"] = np.digitize(point, edges).tolist()
        declared_levels["forecast_range_train_thresholds"] = list(range(len(edges)+1))
    if groups and (not subgroup_receipt or subgroup_receipt.get("threshold_source") != "training_only"
                   or day(subgroup_receipt["threshold_outcome_max"]) > FIT_MAX):
        raise ValueError("Conditional-coverage groups/ranges need pre-calibration train-only threshold receipts")
    result["conditional_coverage"] = {}
    for name,labels in groups.items():
        if len(labels) != len(dates):
            raise ValueError("Subgroup labels must align to the dense forecast calendar")
        result["conditional_coverage"][name] = {}
        levels = list(map(str, declared_levels[name]))
        if len(levels) != len(set(levels)) or not set(map(str,labels)).issubset(levels):
            raise ValueError("Observed subgroup labels must belong to the fixed, unique level registry")
        for label in levels:
            mask = np.array([str(v)==label for v in labels])
            result["conditional_coverage"][name][label] = {
                str(a): temporal_mean_summary(scores[str(a)]["coverage"], dates, eligible_mask=mask,
                                             block_length=block_length, repetitions=repetitions) for a in intervals}
    result["point_performance_side_by_side"] = {
        "branch_A_fixed_ensemble_RMSE": float(np.sqrt(((point-y)**2).mean())),
        "primary_comparable_fixed_ensemble_RMSE": None,
        "scope": "Secondary point-predictor comparison on the same dates; not the primary mean-of-seed-RMSE table"}
    if primary_comparable_point is not None:
        primary = vector(primary_comparable_point, "primary comparable point", True)
        if len(primary) != len(y):
            raise ValueError("Primary and branch-A forecasts must use identical evaluation origins")
        result["point_performance_side_by_side"]["primary_comparable_fixed_ensemble_RMSE"] = float(np.sqrt(((primary-y)**2).mean()))
    result.update({"model": calibrator["model"], "scope": "secondary_branch_A_known_historical_evaluation",
                   "primary_fit_replaced": False, "new_unknown_test": False,
                   "coverage_exact_guarantee": False, "nominal_CIs_not_global_FWER_decisions": True})
    return result


def fit_error_taxonomy(reference_absolute_errors, training_origins, *, reference_name):
    np = numeric(); values = vector(reference_absolute_errors, "training reference absolute errors", True); dates = [day(x) for x in training_origins]
    if len(values) != len(dates) or len(values) < 20 or dates != sorted(set(dates)) or dates[-1] > TAXONOMY_TRAIN_MAX:
        raise ValueError("Taxonomy thresholds require at least twenty distinct training outcomes ending by 2023-12-24")
    if not reference_name:
        raise ValueError("The fixed training reference must be identified")
    return {"reference_name": reference_name, "threshold_source": "training_reference_absolute_errors_only",
            "training_outcome_max": dates[-1].isoformat(), "training_cases": len(values),
            "q50_abs_error": float(np.quantile(values, .5)), "q90_abs_error": float(np.quantile(values, .9)),
            "categories": list(CATEGORIES), "primary_RMSE_delta_reused_as_case_threshold": False}


def validate_oof_panel(rows, models, seeds, *, source_stage, deterministic_models=()):
    """One case is a daily origin; repeated seeds remain paired trajectories."""
    np = numeric()
    if source_stage not in STAGES:
        raise ValueError("Do not relabel E1 results, pilot Nov-Dec or ETTh1 as new E2 development OOF")
    models, seeds = list(models), list(map(str, seeds)); dates = calendar(DEV_START, DEV_END)
    if not models or not seeds or len(set(models)) != len(models) or len(set(seeds)) != len(seeds):
        raise ValueError("Unique model and seed registries are required")
    deterministic = set(deterministic_models)
    if not deterministic.issubset(models):
        raise ValueError("Deterministic references must belong to the explicit model registry")
    seen = {}; truth = {}; eligibility = {}; fold_receipts = {}
    for row in rows:
        origin = _origin(row); model = row["model"]; seed = str(row["seed"]); key = model, seed, origin
        valid_seed = seed == "deterministic" if model in deterministic else seed in seeds
        if model not in models or not valid_seed or origin not in dates or key in seen:
            raise ValueError("OOF rows must match the model/seed/Jan-Oct2024 calendar without duplicates")
        if row.get("source_stage", source_stage) != source_stage:
            raise ValueError("Mixed or conflicting OOF source stages")
        # Inherited validation uses monthly frozen outer fits. Merely stopping
        # at t-8 would silently admit a daily refit within the validation month.
        cutoff = origin.replace(day=1)-timedelta(days=8)
        for field in ("fit_outcome_max", "scaler_fit_outcome_max", "configuration_selection_outcome_max"):
            if day(row[field]) > cutoff:
                raise ValueError("OOF fits, scaling and configuration selection must respect the monthly outer-origin embargo")
        fold_key = model, seed, origin.year, origin.month
        support = tuple(day(row[f]) for f in ("fit_outcome_max", "scaler_fit_outcome_max", "configuration_selection_outcome_max"))
        support += tuple(row.get(f) for f in ("run_id", "checkpoint_sha256"))
        if fold_key in fold_receipts and fold_receipts[fold_key] != support:
            raise ValueError("A monthly OOF fold cannot silently refit or switch its frozen run/checkpoint")
        fold_receipts[fold_key] = support
        eligible = _eligible(row.get("eligible", True))
        if origin in eligibility and eligibility[origin] != eligible:
            raise ValueError("All models and seeds must retain the same eligible dates")
        eligibility[origin] = eligible
        if eligible:
            y, pred = float(row["y_true"]), float(row["y_pred"])
            if not math.isfinite(y) or not math.isfinite(pred) or min(y, pred) < 0:
                raise ValueError("OOF targets/forecasts must be nonnegative original-scale volatility")
            if origin in truth and truth[origin] != y:
                raise ValueError("Daily ground truth differs between models or seeds")
            truth[origin] = y; seen[key] = pred
        else:
            if row.get("y_true") not in (None, "") or row.get("y_pred") not in (None, ""):
                raise ValueError("Ineligible origins remain blank placeholders, not fabricated zeros")
            seen[key] = np.nan
    expected_count = ((len(models)-len(deterministic))*len(seeds)+len(deterministic))*len(dates)
    if len(seen) != expected_count:
        raise ValueError("Retain all305 calendar origins, including ineligible placeholders")
    mask = np.array([eligibility[d] for d in dates], bool)
    if int(mask.sum()) < 100:
        raise ValueError("At least100 unique eligible origins per model are required; seed×date is not case count")
    predictions = np.array([[[seen[m,"deterministic" if m in deterministic else s,d] for d in dates] for s in seeds] for m in models])
    target = np.array([truth.get(d, np.nan) for d in dates])
    return {"models": models, "seeds": seeds, "dates": dates, "eligible": mask, "predictions": predictions,
            "target": target, "source_stage": source_stage, "unique_eligible_origins_per_model": int(mask.sum()),
            "deterministic_models": sorted(deterministic)}


def select_top3_from_development(panel, *, expected_candidate_models):
    np = numeric()
    if panel["source_stage"] != "E2_nested_development":
        raise ValueError("P0 diagnostic recipes are not the final Top3; await complete nested-development evidence")
    if len(panel["models"]) < 3:
        raise ValueError("At least three model families are required")
    if set(expected_candidate_models) != set(panel["models"]) or len(set(expected_candidate_models)) != len(expected_candidate_models):
        raise ValueError("Branch-A selection must cover the entire predeclared candidate cohort")
    mask = panel["eligible"] & np.array([d <= FIT_MAX for d in panel["dates"]])
    if int(mask.sum()) < 100:
        raise ValueError("Branch-A selection needs at least100 common eligible origins ending Oct24")
    y = panel["target"][mask]
    scores = {m: float(np.sqrt(((panel["predictions"][i][:,mask]-y[None,:])**2).mean(axis=1)).mean())
              for i,m in enumerate(panel["models"])}
    ordered = sorted(scores, key=lambda m: (scores[m], m))
    return {"selected_models": ordered[:3], "selection_source": "secondary_branch_A_common_OOF_Jan1-Oct24_2024_only",
            "selection_outcome_max": FIT_MAX.isoformat(), "primary_JanOct31_ranking_changed": False,
            "selection_candidate_cohort": list(expected_candidate_models),
            "primary_metric": "mean_seed_RMSE_original_scale", "scores": scores,
            "test_used": False, "P0_ranking_substituted": False, "tiebreak": "lexicographic frozen model ID"}


def category_for_case(seed_errors, taxonomy):
    np = numeric(); error = vector(seed_errors, "paired seed errors"); magnitude = float(np.abs(error).mean())
    low, high = float(taxonomy["q50_abs_error"]), float(taxonomy["q90_abs_error"])
    if not 0 <= low <= high or not math.isfinite(high):
        raise ValueError("Ordered finite training thresholds are required")
    if magnitude <= low:
        return CATEGORIES[0]
    size = "moderate" if magnitude <= high else "large"
    sign = "under" if (error < 0).all() else "over" if (error > 0).all() else "mixed"
    return size+"_"+sign


def stratified_case_sample(categories, origins, *, sample_size=100, seed=20261106):
    """Sample unique origins, retaining taxonomy×calendar-quarter strata."""
    np = numeric(); dates = [day(x) for x in origins]
    if (len(categories) != len(dates) or len(set(dates)) != len(dates)
            or int(sample_size) != sample_size or sample_size < 100):
        raise ValueError("The audit sample needs at least100 distinct daily origins, not repeated seeds")
    sample_size = int(sample_size)
    strata = defaultdict(list)
    for i,(cat,d) in enumerate(zip(categories, dates)):
        if cat not in CATEGORIES:
            raise ValueError("Unknown MECE taxonomy category")
        strata[(cat, (d.month-1)//3+1)].append(i)
    if len(dates) < sample_size or len(strata) > sample_size:
        raise ValueError("Requested unique stratified sample is not available")
    keys = sorted(strata); capacities = np.array([len(strata[k])-1 for k in keys]); remaining = sample_size-len(keys)
    shares = remaining*capacities/capacities.sum() if capacities.sum() else np.zeros(len(keys))
    extra = np.floor(shares).astype(int)
    for pos in sorted(range(len(keys)), key=lambda i: (-(shares[i]-extra[i]), keys[i]))[:remaining-int(extra.sum())]:
        extra[pos] += 1
    rng = np.random.default_rng(seed); selected = []
    for i,key in enumerate(keys):
        take = int(extra[i])+1; indices = rng.choice(strata[key], take, replace=False)
        for j in indices:
            selected.append({"origin_utc": dates[int(j)].isoformat(), "category": key[0], "calendar_quarter": key[1],
                             "stratum_population": len(strata[key]), "stratum_sampled": take,
                             "inverse_inclusion_weight": len(strata[key])/take})
    return sorted(selected, key=lambda row: row["origin_utc"])


def analyze_oof_errors(panel, taxonomy, *, sample_size=100, repetitions=2000, block_length=14, seed=20261106):
    np = numeric()
    if (taxonomy.get("threshold_source") != "training_reference_absolute_errors_only"
            or day(taxonomy["training_outcome_max"]) > TAXONOMY_TRAIN_MAX
            or int(repetitions) != repetitions or repetitions < 2000):
        raise ValueError("Use pre-2024 training-only taxonomy and at least2000 temporal resamples")
    repetitions = int(repetitions)
    dates = panel["dates"]; mask = panel["eligible"]; n = len(dates)
    rng = np.random.default_rng(seed); draws = [block_indices(n, block_length, rng) for _ in range(repetitions)]
    out = {}; large = []; signed = []
    for i,model in enumerate(panel["models"]):
        error = panel["predictions"][i]-panel["target"][None,:]
        categories = [category_for_case(error[:,j], taxonomy) if mask[j] else None for j in range(n)]
        encoded = np.array([CATEGORIES.index(c) if c else -1 for c in categories]); rows = []
        for k,cat in enumerate(CATEGORIES):
            boot = [float((encoded[idx][mask[idx]]==k).mean()) for idx in draws if mask[idx].any()]
            rows.append({"category": cat, "cases": int((encoded==k).sum()),
                         "frequency": float((encoded[mask]==k).mean()),
                         "ci95_temporal_nominal": np.quantile(boot,[.025,.975]).tolist() if len(boot)>=.95*repetitions else None})
        sample = stratified_case_sample([c for c in categories if c is not None], [d for d,keep in zip(dates,mask) if keep], sample_size=sample_size, seed=seed+i)
        out[model] = {"unique_eligible_cases": int(mask.sum()),
                      "seed_trajectories": 1 if model in panel.get("deterministic_models",[]) else len(panel["seeds"]),
                      "categories": rows, "private_stratified_audit_sample": sample,
                      "frequency_denominator": "all eligible OOF daily origins, not the unweighted stratified subsample"}
        loss = (error[:,mask]**2).mean(axis=0); eligible_dates = [d for d,keep in zip(dates,mask) if keep]
        out[model]["private_high_loss_frontier_origins"] = [eligible_dates[int(j)].isoformat() for j in np.argsort(-loss,kind="stable")[:10]]
        out[model]["independent_randomized_seed_trajectories"] = 0 if model in panel.get("deterministic_models",[]) else len(panel["seeds"])
        large.append(np.isin(encoded,[4,5,6])); signed.append(error[:,mask].mean(axis=0))
    pairs = []
    for i in range(len(panel["models"])):
        for j in range(i+1,len(panel["models"])):
            union = (large[i]|large[j])[mask].sum(); intersection = (large[i]&large[j])[mask].sum()
            corr = float(np.corrcoef(signed[i],signed[j])[0,1]) if np.std(signed[i])>0 and np.std(signed[j])>0 else None
            pairs.append({"models": [panel["models"][i],panel["models"][j]], "large_error_jaccard": float(intersection/union) if union else None,
                          "signed_error_correlation": corr, "not_a_significance_or_architecture_causality_claim": True})
    common_dense = np.stack(large).all(axis=0); common = common_dense[mask]
    common_boot = [float(common_dense[idx][mask[idx]].mean()) for idx in draws if mask[idx].any()]
    member_centers = panel["predictions"].mean(axis=1)[:,mask]
    individual_ensemble_rmse = np.sqrt(((member_centers-panel["target"][mask])**2).mean(axis=1))
    ensemble_rmse = float(np.sqrt(((member_centers.mean(axis=0)-panel["target"][mask])**2).mean()))
    return {"source_stage": panel["source_stage"], "case_unit": "unique daily development origin",
            "models": out, "shared_large_error_frequency": float(common.mean()), "pairwise_error_overlap": pairs,
            "shared_large_error_ci95_temporal_nominal": np.quantile(common_boot,[.025,.975]).tolist() if len(common_boot)>=.95*repetitions else None,
            "complementarity_diagnostic": {"equal_model_weight_ensemble_point_RMSE": ensemble_rmse,
                "best_member_ensemble_point_RMSE_on_same_OOF": float(individual_ensemble_rmse.min()),
                "observed_RMSE_gain": float(individual_ensemble_rmse.min()-ensemble_rmse),
                "scope": "Descriptive OOF complementarity; selected-best reference is optimistic and no future ensemble gain or primary-RMSE improvement is established"},
            "block_length_days": block_length, "bootstrap_replicates": repetitions,
            "intervals_nominal_not_simultaneous": True, "new_hypothesis_decisions": 0,
            "formal_diagnostic_family_budget_not_spent_by_descriptive_CIs": True,
            "final_top3_identified": False, "test_used": False, "raw_predictions_exported": False}


def public_error_summary(value):
    """Strip the explicit private audit/frontier ledgers before public export."""
    if isinstance(value, dict):
        return {key: public_error_summary(item) for key,item in value.items() if not key.startswith("private_")}
    if isinstance(value, list):
        return [public_error_summary(item) for item in value]
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--oof-csv", type=Path)
    parser.add_argument("--taxonomy-json", type=Path)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seeds", nargs="+")
    parser.add_argument("--deterministic-models", nargs="*", default=[])
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args(argv); plan = json.loads(args.plan.read_text())
    if args.oof_csv is not None:
        if not all((args.taxonomy_json, args.models, args.seeds, args.private_output)):
            parser.error("OOF mode requires taxonomy, complete model/seed registries and an explicit private output")
        try:
            destination = args.private_output.resolve()
            if destination.is_relative_to(ROOT.resolve()) or (destination.exists() and any(destination.iterdir())):
                raise ValueError("Use a new empty private directory outside the repository; no public case-ledger export")
            with args.oof_csv.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            if not rows or any(row.get("source_stage") != "P0_diagnostic" or not row.get("run_id") for row in rows):
                raise ValueError("This CLI accepts explicitly identified P0 diagnostic OOF runs, not legacy or final benchmark results")
            closed, scored = [], []
            for row in rows:
                checksum = row.get("checkpoint_sha256", "")
                if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
                    raise ValueError("Every OOF row must identify its frozen checkpoint")
                a = datetime.fromisoformat(row["checkpoint_closed_at_utc"].replace("Z", "+00:00"))
                b = datetime.fromisoformat(row["scoring_started_at_utc"].replace("Z", "+00:00"))
                if a.tzinfo is None or b.tzinfo is None or a.utcoffset() != timedelta(0) or b.utcoffset() != timedelta(0):
                    raise ValueError("OOF fit/scoring receipts need explicit UTC timestamps")
                closed.append(a); scored.append(b)
            if not max(closed) < min(scored):
                raise ValueError("All recorded OOF checkpoints must close before the first recorded OOF scoring")
            taxonomy = json.loads(args.taxonomy_json.read_text())
            if int(taxonomy.get("training_cases", 0)) < 20 or not taxonomy.get("reference_name"):
                raise ValueError("Taxonomy needs a named pre-2024 training reference with at least20 distinct outcomes")
            panel = validate_oof_panel(rows, args.models, args.seeds, source_stage="P0_diagnostic",
                                       deterministic_models=args.deterministic_models)
            result = analyze_oof_errors(panel, taxonomy, repetitions=args.bootstrap_replicates)
            destination.mkdir(parents=True, exist_ok=True)
            (destination/"error_summary.json").write_text(json.dumps(public_error_summary(result), ensure_ascii=False, indent=2, allow_nan=False)+"\n")
            with (destination/"private_audit_cases.csv").open("w", newline="") as stream:
                fields = ["model", "origin_utc", "category", "calendar_quarter", "stratum_population", "stratum_sampled", "inverse_inclusion_weight"]
                writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
                for model, record in result["models"].items():
                    writer.writerows({"model": model, **row} for row in record["private_stratified_audit_sample"])
            receipt = {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                       "prediction_CSV_sha256": sha256(args.oof_csv.read_bytes()).hexdigest(),
                       "taxonomy_JSON_sha256": sha256(args.taxonomy_json.read_bytes()).hexdigest(),
                       "plan_sha256": sha256(args.plan.read_bytes()).hexdigest(),
                       "source_stage": "P0_diagnostic", "unique_eligible_origins_per_model": panel["unique_eligible_origins_per_model"],
                       "private_audit_cases_per_model": 100, "all_checkpoint_receipts_closed_before_first_score": True,
                       "model_fits_performed": 0, "final_benchmark_evaluated": False, "final_Top3_identified": False,
                       "hashes_identify_bytes_not_scientific_sufficiency": True}
            (destination/"analysis_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n")
            print(json.dumps({"source_stage": "P0_diagnostic", "unique_eligible_origins_per_model": panel["unique_eligible_origins_per_model"],
                              "model_fits": 0, "final_benchmark_evaluated": False}))
            return 0
        except (ValueError, KeyError, OSError, TypeError) as error:
            print(str(error), file=sys.stderr)
            return 1
    if any((args.taxonomy_json, args.models, args.seeds, args.private_output)):
        parser.error("Provide the OOF CSV together with all explicit diagnostic inputs")
    print(json.dumps({"status": plan["status"], "calibration_branch": plan["predictive_intervals"]["branch"],
                      "OOF_calendar_origins": len(calendar(DEV_START, DEV_END)), "minimum_unique_error_cases_per_model": 100,
                      "actual_intervals_evaluated": 0, "actual_error_cases_analyzed_by_this_command": 0,
                      "model_fits": 0, "new_Bitcoin_test_claimed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
