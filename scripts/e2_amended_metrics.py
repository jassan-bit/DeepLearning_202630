"""Pure scoring, local selection and resource checks for the E2 amendment.

No files, market loaders, model constructors, fitting or remote writes occur.
Scores use the complete common support; scale denominators inspect only dated
training labels at or before the supplied cutoff. Cache checks compare explicit
identities and do not themselves certify the bytes of a stored artifact.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, timedelta, timezone
from itertools import combinations
import math
from numbers import Real
import re
from statistics import mean, median

VERSION = "e2-amended-metrics-v1"
DELTA = 0.00025772842664981106
HPO_SEEDS = (271201, 271202)
NOISE_SEEDS = (271501, 271502, 271503, 271504, 271505)
CACHE_IDENTITY_FIELDS = (
    "model_id", "origin", "recipe_id", "model_source_sha256", "hpo_sha256",
    "architecture_id", "architecture_sha256", "data_sha256",
    "prepared_signature", "seed", "execution_policy_sha256",
)
HASH_FIELDS = tuple(k for k in CACHE_IDENTITY_FIELDS if k.endswith("sha256") or k == "prepared_signature")
RESTORED_RECIPES = {
    "LSTM-autoencoder con predictor": "SAE-D7-PUBLISHED-v1",
    "TimesNet": "TN-D7-PUBLISHED-v1", "TimeMixer": "TM-D7-PUBLISHED-v1",
    "PatchTST": "PT-D7-PUBLISHED-v1",
}


def _finite(value, name, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(name + " must be a finite numeric scalar")
    try:
        value = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(name + " must be a finite numeric scalar") from error
    if not math.isfinite(value):
        raise ValueError(name + " must be a finite numeric scalar")
    if nonnegative and value < 0:
        raise ValueError(name + " cannot be negative")
    return value


def _day(value):
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("Naive timestamps are not declared UTC dates")
        return value.astimezone(timezone.utc).date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("Use explicit ISO calendar dates")
    return date.fromisoformat(value)


def _vector(values, name):
    try:
        copied = list(values)
    except TypeError as error:
        raise ValueError(name + " must be a nonempty one-dimensional sequence") from error
    if not copied:
        raise ValueError(name + " cannot be empty")
    return [_finite(value, name, nonnegative=True) for value in copied]


def _moments(truth, predictions):
    truth, predictions = _vector(truth, "truth"), _vector(predictions, "predictions")
    if len(truth) != len(predictions):
        raise ValueError("Complete aligned predictions on common support are required")
    errors = [prediction - target for prediction, target in zip(predictions, truth)]
    mse = mean([error * error for error in errors])
    mae = mean([abs(error) for error in errors])
    if not math.isfinite(mse) or not math.isfinite(mae):
        raise ValueError("Metric arithmetic overflow")
    return truth, errors, mse, mae


def dated_training_scales(dates, targets, *, cutoff):
    """MASE lag-1 absolute scale and RMSSE lag-1 squared scale, train-only.

    Future labels are sliced before numeric conversion. Missing/nonfinite past
    labels break pairs, as does every calendar gap; neither is bridged.
    """
    days, values, cutoff = [_day(day) for day in dates], list(targets), _day(cutoff)
    if len(days) != len(values) or any(b <= a for a, b in zip(days, days[1:])):
        raise ValueError("Unique chronological dates must align with training targets")
    past = []
    for day, value in zip(days, values):
        if day > cutoff:
            continue
        if value is None:
            past.append((day, float("nan")))
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("Past training labels must be numeric or explicit nonfinite missing values")
        try:
            value = float(value)
        except (ValueError, OverflowError) as error:
            raise ValueError("Training label numeric conversion overflow") from error
        if math.isfinite(value) and value < 0:
            raise ValueError("Volatility training labels cannot be negative")
        past.append((day, value))
    differences = [b[1] - a[1] for a, b in zip(past, past[1:])
                   if b[0] - a[0] == timedelta(days=1)
                   and math.isfinite(a[1]) and math.isfinite(b[1])]
    absolute = mean([abs(v) for v in differences]) if differences else None
    squared = mean([v * v for v in differences]) if differences else None
    if any(value is not None and not math.isfinite(value) for value in (absolute, squared)):
        raise ValueError("Training denominator arithmetic overflow")
    def record(value):
        reason = "no_consecutive_training_pairs" if value is None else "zero_training_scale" if value == 0 else None
        return {"value": value if value is not None and value > 0 else None, "reason": reason}
    return {"cutoff": cutoff.isoformat(), "pairs": len(differences),
            "MASE": record(absolute), "RMSSE": record(squared),
            "past_values": len(past),
            "nonfinite_past_values": sum(not math.isfinite(value) for _, value in past)}


def _scale(scales, name):
    if not isinstance(scales, Mapping) or not isinstance(scales.get(name), Mapping):
        raise ValueError("Supply explicit dated " + name + " training scale metadata")
    row = scales[name]
    value = row.get("value")
    if value is None:
        return None, row.get("reason") or "training_scale_unavailable"
    value = _finite(value, name + " denominator", nonnegative=True)
    return (value, None) if value > 0 else (None, "zero_training_scale")


def regression_metrics(truth, predictions, *, training_scales, persistence_predictions=None):
    """Original-unit metrics without dropping zero labels from the primary score."""
    truth, errors, mse, mae = _moments(truth, predictions)
    rmse = math.sqrt(mse)
    center = mean(truth)
    target_variance = mean([(value - center) ** 2 for value in truth])
    if not math.isfinite(target_variance):
        raise ValueError("Target variance arithmetic overflow")
    mase_scale, mase_reason = _scale(training_scales, "MASE")
    rmsse_scale, rmsse_reason = _scale(training_scales, "RMSSE")
    positive = [(target, error) for target, error in zip(truth, errors) if target > 0]
    mape = mean([abs(error) / target * 100 for target, error in positive]) if positive else None
    reference_rmse = None
    reference_reason = "reference_not_supplied"
    if persistence_predictions is not None:
        _, _, reference_mse, _ = _moments(truth, persistence_predictions)
        reference_rmse = math.sqrt(reference_mse)
        reference_reason = None if reference_rmse > 0 else "zero_reference_rmse"
    result = {"n": len(truth), "RMSE": rmse, "MAE": mae, "MSE": mse,
        "R2": 1 - mse / target_variance if target_variance > 0 else None,
        "R2_reason": None if target_variance > 0 else "constant_target",
        "MAPE_percent": mape, "MAPE_coverage": len(positive) / len(truth),
        "MAPE_n": len(positive), "MAPE_reason": None if positive else "no_positive_targets",
        "MASE": mae / mase_scale if mase_scale is not None else None, "MASE_reason": mase_reason,
        "RMSSE": math.sqrt(mse / rmsse_scale) if rmsse_scale is not None else None,
        "RMSSE_reason": rmsse_reason,
        "RMRMSE": rmse / reference_rmse if reference_reason is None else None,
        "RMRMSE_reason": reference_reason, "RMRMSE_reference": "Persistence",
        "Persistence_RMSE": reference_rmse, "training_scales": dict(training_scales)}
    if any(isinstance(value, Real) and not math.isfinite(value) for value in result.values()):
        raise ValueError("Metric ratio arithmetic overflow; no silent finite replacement")
    return result


def mean_seed_rmse(truth, predictions_by_seed):
    """Average individual RMSEs, preserving all dates within every seed."""
    truth = _vector(truth, "truth")
    rows = list(predictions_by_seed.values()) if isinstance(predictions_by_seed, Mapping) else list(predictions_by_seed)
    if not rows:
        raise ValueError("At least one independently fitted seed prediction is required")
    return mean([math.sqrt(_moments(truth, row)[2]) for row in rows])


def select_local_hpo(candidates, *, anchor, seed_ids=HPO_SEEDS,
                     scope="registered_domains", registered_trial_ids=None):
    """Select locally only after accounting for every registered HPO point.

    The caller supplies eighteen IDs from the hash-pinned model registry.
    Smaller domains require explicit synthetic_fixture scope and never close a
    registered search. This pure criteria helper does not verify run receipts.
    """
    anchor = _day(anchor).isoformat()
    if tuple(seed_ids) != HPO_SEEDS:
        raise ValueError("The local HPO comparison uses the two registered seeds")
    if scope not in ("registered_domains", "synthetic_fixture"):
        raise ValueError("Declare registered_domains or an explicit synthetic_fixture")
    rows, ledger, successes, seen = list(candidates), [], [], set()
    if not rows:
        raise ValueError("A nonempty explicit candidate ledger is required")
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("HPO ledger rows must be explicit mappings")
    if scope == "registered_domains":
        if registered_trial_ids is None or isinstance(registered_trial_ids, str):
            raise ValueError("Supply the eighteen exact registered trial IDs from the pinned model registry")
        ids = list(registered_trial_ids)
        if (len(ids) != 18 or any(not isinstance(identity, str) or not identity for identity in ids)
                or len(set(ids)) != 18):
            raise ValueError("The registered HPO domain must contain eighteen unique trial IDs")
        actual = [row.get("trial_id") for row in rows]
        if len(rows) != 18 or any(not isinstance(identity, str) for identity in actual) or set(actual) != set(ids):
            raise ValueError("Account for all eighteen registered configurations; incomplete or foreign ledgers cannot select a winner")
        if any(row.get("status") not in ("completed", "failed") for row in rows):
            raise ValueError("Every registered HPO row needs explicit completed or failed status")
    for row in rows:
        resources = row.get("resource_use", {})
        if not isinstance(resources, Mapping):
            raise ValueError("Run resource metadata must be a mapping when supplied")
        flags = [source.get(key, False) for source in (row, resources)
                 for key in ("fatal", "systematic_failure")]
        if any(type(flag) is not bool for flag in flags):
            raise ValueError("Fatal/systematic failure declarations must be explicit booleans")
        if any(flags):
            raise ValueError("A fatal or systematic implementation failure blocks local selection")
        trial_id = row.get("trial_id")
        if not isinstance(trial_id, str) or not trial_id or trial_id in seen:
            raise ValueError("Trial IDs must be unique nonempty strings")
        seen.add(trial_id)
        if row.get("anchor", anchor) != anchor or row.get("origin", anchor) != anchor:
            raise ValueError("An HPO decision cannot pool different anchors")
        status = row.get("status", "completed")
        record = {"trial_id": trial_id, "anchor": anchor, "status": status}
        if status == "failed":
            if not isinstance(row.get("reason"), str) or not row["reason"]:
                raise ValueError("Failed configurations remain in the ledger with a reason")
            record["reason"] = row["reason"]
        elif status == "completed":
            scores = row.get("seed_rmse")
            if not isinstance(scores, Mapping) or len(scores) != 2:
                raise ValueError("Every completed configuration needs exactly two seed RMSEs")
            scores = {str(key): _finite(value, "seed RMSE", nonnegative=True) for key, value in scores.items()}
            if set(scores) != {str(seed) for seed in HPO_SEEDS}:
                raise ValueError("HPO seed IDs differ from the registered pair")
            record.update(seed_rmse=scores, mean_seed_RMSE=mean(scores.values()))
            successes.append((record["mean_seed_RMSE"], trial_id))
        else:
            raise ValueError("Use completed or explicitly failed candidate status")
        ledger.append(record)
    if not successes:
        raise ValueError("No complete configuration; no replacement search is permitted")
    score, trial_id = min(successes)
    return {"winner_trial_id": trial_id, "winner_mean_rmse": score, "anchor": anchor,
            "selection_scope": "local_anchor_only", "tie_rule": "lowest stable trial_id on exact mean tie",
            "configuration_count": len(ledger), "scope": scope,
            "registered_domain_accounted_for": scope == "registered_domains",
            "registered_selection_complete": scope == "registered_domains",
            "criteria_only": True, "run_receipts_verified": False,
            "fixture_does_not_complete_registered_search": scope == "synthetic_fixture",
            "ledger": ledger}


def nas_parsimony_choice(candidates, *, delta=DELTA):
    """Nominal-CI overlap is a selection rule, never an equivalence declaration."""
    delta = _finite(delta, "delta", nonnegative=True)
    if delta != DELTA:
        raise ValueError("The inherited practical RMSE margin cannot be silently changed")
    rows, seen = [], set()
    for candidate in candidates:
        identity = candidate.get("architecture_id")
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("Unique architecture IDs are required")
        seen.add(identity)
        score = _finite(candidate["mean_rmse"], "mean RMSE", nonnegative=True)
        interval = candidate["ci95"]
        if len(interval) != 2:
            raise ValueError("Supply the nominal mean-score CI95 bounds")
        lower, upper = [_finite(value, "CI bound") for value in interval]
        count = candidate["parameters"]
        if lower > score or upper < score or type(count) is not int or count < 1:
            raise ValueError("CI must enclose the mean and parameter count must be positive")
        rows.append((score, identity, lower, upper, count))
    if not rows:
        raise ValueError("Complete NAS finalists are required")
    best = min(rows)
    tied = [row for row in rows if row[0] - best[0] <= delta or max(row[2], best[2]) <= min(row[3], best[3])]
    winner = min(tied, key=lambda row: (row[4], row[0], row[1]))
    return {"winner_architecture_id": winner[1], "technical_tie_ids": sorted(row[1] for row in tied),
            "equivalence_proven": False, "nominal_CI_scope": "selection_only",
            "phase2_ledger_verified": False, "criteria_only": True,
            "indistinguishable_set_analysis": "TOST/ROPE/MCS still required separately"}


def _ranks(values):
    ordered = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks, index = [0.] * len(values), 0
    while index < len(ordered):
        stop = index + 1
        while stop < len(ordered) and ordered[stop][1] == ordered[index][1]:
            stop += 1
        rank = (index + 1 + stop) / 2
        for position, _ in ordered[index:stop]:
            ranks[position] = rank
        index = stop
    return ranks


def choose_nas_k(pilot_scores, *, pilot_origin="2023-10-01", last_label="2023-09-23"):
    """Apply the registered 12-architecture × five-seed TiDE rank rule."""
    if _day(pilot_origin) != date(2023, 10, 1) or _day(last_label) != date(2023, 9, 23):
        raise ValueError("Noise pilot must precede 2024 selection and use labels only through 2023-09-23")
    if not isinstance(pilot_scores, Mapping) or set(pilot_scores) != {"TiDE"}:
        raise ValueError("The current mandatory NAS pilot covers only TiDE")
    scores = [list(row) for row in pilot_scores["TiDE"]]
    if len(scores) != 5 or any(len(row) != 12 for row in scores):
        raise ValueError("Provide the complete five by twelve score table, with missing values explicit")
    if any(value is not None and (isinstance(value, bool) or not isinstance(value, Real))
           for row in scores for value in row):
        raise ValueError("Pilot scores must be numeric, with unavailable entries explicitly missing")
    complete = all(isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
                   for row in scores for value in row)
    correlations, rho = [], None
    if complete:
        if any(value < 0 for row in scores for value in row):
            raise ValueError("Pilot RMSE cannot be negative")
        ranks = [_ranks(row) for row in scores]
        for first, second in combinations(ranks, 2):
            a, b = [v - mean(first) for v in first], [v - mean(second) for v in second]
            denominator = math.sqrt(sum(v*v for v in a) * sum(v*v for v in b))
            correlations.append(sum(x*y for x, y in zip(a, b)) / denominator if denominator else None)
        if all(value is not None for value in correlations):
            rho = median(correlations)
    return {"k": 40 if rho is None or rho < .5 else 20,
            "median_pairwise_spearman": rho, "pairwise_correlations": correlations,
            "full_score_table_complete": complete, "freeze_before_NAS1": True,
            "pilot_origin": pilot_origin, "last_label": last_label,
            "decision_scope": "noise screening rule, not proof of optimality"}


def remaining_budget(consumed_seconds, reserved_seconds, limit_hours=64):
    """Account cumulative historical consumption before admitting a reservation.

    reserved_seconds is the reservation being considered, not additional credit.
    This arithmetic does not authorize a new limit or reconcile unknown charges.
    """
    consumed = _finite(consumed_seconds, "consumed seconds", nonnegative=True)
    reserved = _finite(reserved_seconds, "reserved seconds", nonnegative=True)
    limit = _finite(limit_hours, "total limit hours", nonnegative=True)
    if limit <= 0:
        raise ValueError("The total authorized limit must be positive")
    total = limit * 3600
    if not math.isfinite(total) or not math.isfinite(consumed + reserved):
        raise ValueError("Budget arithmetic overflow")
    balance = total - consumed - reserved
    return {"limit_hours_total": limit, "limit_seconds_total": total,
            "consumed_seconds": consumed, "reserved_seconds": reserved,
            "remaining_before_reservation_seconds": max(0., total - consumed),
            "remaining_seconds": max(0., balance), "deficit_seconds": max(0., -balance),
            "can_admit": balance >= 0, "budget_reset": False}


def cache_reuse_decision(expected, observed):
    """Require complete exact identities; the caller verifies artifact bytes first."""
    if not isinstance(expected, Mapping) or not isinstance(observed, Mapping):
        raise ValueError("Expected and observed cache identities must be explicit mappings")
    missing = sorted({key for key in CACHE_IDENTITY_FIELDS if key not in expected or key not in observed})
    invalid = []
    for label, identity in (("expected", expected), ("observed", observed)):
        for key in HASH_FIELDS:
            if key in identity and (not isinstance(identity[key], str) or not re.fullmatch(r"[0-9a-f]{64}", identity[key])):
                invalid.append(label + "." + key)
        if "seed" in identity and (type(identity["seed"]) is not int or identity["seed"] < 0):
            invalid.append(label + ".seed")
        for key in ("model_id", "origin", "recipe_id"):
            if key in identity and (not isinstance(identity[key], str) or not identity[key]):
                invalid.append(label + "." + key)
        if "origin" in identity:
            try:
                _day(identity["origin"])
            except (ValueError, TypeError):
                invalid.append(label + ".origin")
        architecture = identity.get("architecture_id")
        if architecture is not None and (not isinstance(architecture, str) or not architecture):
            invalid.append(label + ".architecture_id")
        if identity.get("model_id") == "TiDE" and not architecture:
            invalid.append(label + ".architecture_id")
    model = expected.get("model_id")
    if model in RESTORED_RECIPES and expected.get("recipe_id") != RESTORED_RECIPES[model]:
        invalid.append("expected.recipe_id_not_current")
    different = [key for key in CACHE_IDENTITY_FIELDS
                 if key in expected and key in observed and expected[key] != observed[key]]
    complete = observed.get("status") == "completed"
    reusable = not missing and not invalid and not different and complete
    return {"reusable": reusable, "missing_fields": missing,
            "invalid_fields": sorted(invalid), "mismatched_fields": different,
            "completed_result": complete, "scope": "identity_match_only",
            "artifact_bytes_must_be_verified": True, "historical_result_relabelled": False}
