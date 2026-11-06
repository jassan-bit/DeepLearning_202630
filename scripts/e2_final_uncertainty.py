"""Frozen branch-A uncertainty and provenance-qualified development error analysis.

All inputs are explicit private forecasts and hash-bound receipts. This module
does not train a predictor, open market files, select final models or tune a
calibrator using evaluation labels. Absolute-residual intervals and the signed
residual empirical CDF are distinct, explicitly reported predictive procedures.
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

import numpy as np

import e2_predictive_uncertainty as base

ROOT = Path(__file__).resolve().parents[1]
TOP10 = ("TSMixer", "TiDE", "TimeMixer", "TimesNet", "PatchTST", "Informer",
         "Autoformer", "iTransformer", "Crossformer", "S-Mamba")
CATALOGUE = (*TOP10, "GRU-E2", "LSTM-E2", "TCN-E2", "GRU/D7-c03", "LSTM/D7-c02",
             "TCN/D7-c02", "Ridge", "Persistence", "SeasonalNaive", "ARIMA")
DETERMINISTIC = frozenset(("Ridge", "ARIMA", "Persistence", "SeasonalNaive"))
LEGACY_CANDIDATES = {"GRU/D7-c03": "GRU_D7-c03-fixed", "LSTM/D7-c02": "LSTM_D7-c02-fixed",
                     "TCN/D7-c02": "TCN_D7-c02-fixed"}
A_SEEDS = tuple(str(s) for s in range(680101, 680106))
OOF_SEEDS = ("2026110611", "2026110612")
CAL_STAGE, EVAL_STAGE, PRIMARY_STAGE = "E2_branch_A_calibration", "E2_branch_A_evaluation", "E2_final_primary"
HIGH_THRESHOLD = .024247163097617024
ALPHAS = (.2, .1, .05)
QUANTILES = (.025, .05, .1, .5, .9, .95, .975)
PIT_GRID = tuple(i/20 for i in range(1, 20))
BOOTSTRAPS = 2000
BLOCKS = (14, 7, 28)
MC_SEED = 202611067
MIN_GROUP_CASES = 10
FIT_MAX = date(2024, 10, 31)
CAL_START, CAL_END = date(2024, 11, 8), date(2024, 12, 30)
A_BRANCH = "secondary_frozen_calibration_A53"


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def canonical_bytes(value):
    return (json.dumps(clean(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()


def digest_json(value):
    return sha256(canonical_bytes(value)).hexdigest()


def digest_file(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def timestamp(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() != timedelta(0):
        raise ValueError("Actual phase timestamps must explicitly use UTC")
    return result


def policy():
    return {
        "protocol": "E2_completion_frozen_uncertainty_A53_all20",
        "catalogue": list(CATALOGUE), "neural_models": 16,
        "A_seed_ids": list(A_SEEDS), "deterministic_models": sorted(DETERMINISTIC),
        "A_fit_outcome_max": "2024-10-31", "A_selection_outcome_max": "2024-10-31",
        "calibration_calendar": ["2024-11-08", "2024-12-30"],
        "calibration_origins": 53, "nominal_fit_origin": "2024-11-08",
        "embargo_calendar": ["2024-11-01", "2024-11-07"],
        "A_recipe": "same frozen primary HPO recipe selected throughOctober31; no new truncated recipe search",
        "amendment_reason": "preserve the inherited E1 recipe-selection exposure throughOctober31 while maintaining disjoint label-close support",
        "evaluation_calendar": ["2026-09-01", "2026-10-30"],
        "all20_calibrated_before_primary_top3_reporting": True,
        "phase_order": "all primary+A weights closed < calibration scoring < calibrators sealed < evaluation scoring",
        "absolute_residual_intervals": {"nominal_coverage": [.8, .9, .95], "half_life_days": 28,
                                         "fixed_test_mass": 1., "zero_lower_clipping": True,
                                         "WIS_center": "A mean-members point submitted as quantile.5 forecast; equality to the true conditional median not asserted",
                                         "exact_temporal_coverage_guarantee": False},
        "signed_empirical_distribution": {"residual": "calibration truth minus frozen A ensemble point",
            "weights": "same recency weights, normalized to probability mass; no extra mass at infinity",
            "samples": "max(0,A_point+signed_calibration_residual)", "quantile_levels": list(QUANTILES),
            "scores": ["CRPS", "pinball", "randomized PIT"],
            "same_as_symmetric_absolute_residual_interval_CDF": False},
        "subgroups": {"high_volatility": "sigma7(t-1)>fixed2023q75", "high_threshold": HIGH_THRESHOLD,
                       "forecast_bin_edges": "training2023 sigma7 target quantiles1/3,2/3; frozen before calibration",
                       "minimum_cases_for_conditional_CI": MIN_GROUP_CASES,
                       "calibration_metrics": ["PIT L1 uniform-grid deviation, not classification ECE", "OLS truth-on-Apoint slope and intercept"],
                       "conditional_tests": "descriptive nominal temporal CIs, no additional hypothesis decisions"},
        "temporal_bootstrap": {"draws": BOOTSTRAPS, "primary_block": 14, "sensitivities": [7, 28],
                                "seed": MC_SEED, "filter_subgroups_after_calendar_resampling": True,
                                "conditional_on_frozen_calibrators": True, "simultaneous_CIs": False},
        "PIT_diagnostic": {"grid": list(PIT_GRID), "randomization_seed": MC_SEED,
            "statistic": "max_grid_abs(empirical_PIT_CDF(tau)-tau)",
            "calibration_L1": "mean_grid_abs(empirical_PIT_CDF(tau)-tau); continuous-PIT calibration score, not classification ECE",
            "null": "uniform marginal randomized PIT on the declared grid",
            "resampling": "centered circular-block empirical process; no iid KS reference distribution",
            "family": "20 separate exploratory PIT diagnostics, Holm alpha.01",
            "global_final_primary_FWER_extension_claimed": False},
        "OOF_errors": {"stage": "E2_nested_development", "calendar": ["2024-01-01", "2024-10-31"],
                       "stochastic_seeds": list(OOF_SEEDS), "cases_required": 100,
                       "audit_sample_per_model": 100, "training_taxonomy_max": "2023-12-24",
                       "future_calibration_not_used_for_2024_confidence": True,
                       "pruned_weights_require_original_completion_and_closed_score_chain": True,
                       "nested_configuration_selection_models": 17,
                       "conditional_inherited_references": list(LEGACY_CANDIDATES),
                       "legacy_configuration_selection_outcome_max": "2024-10-31",
                       "legacy_hyperparameter_selection_not_unbiased": True,
                       "legacy_exception_extends_to_new_tuning": False,
                       "all20_monthly_fit_scaler_and_epoch_selection_embargo_retained": True},
        "private_evaluation_frontier": {"high_loss": "ten largest squared errors, known2026evaluation only",
            "confidence_proxy": "ten widest95percent signed-empirical-CDF quantile intervals, ties explicitly disclosed",
            "confidence_uses_future_labels": False, "retrospective_OOF_confidence_from2024NovDec_claimed": False},
        "new_unseen_test_claimed": False, "model_fits_performed": 0,
    }


def bound_path(root, relative):
    root = Path(root).resolve(); entry = Path(str(relative))
    if root.is_relative_to(ROOT):
        raise ValueError("Individual prediction and checkpoint evidence must remain outside the source repository")
    if entry.is_absolute() or ".." in entry.parts:
        raise ValueError("Evidence paths must be relative to the explicit private artifact root")
    path = root/entry
    if (not path.resolve().is_relative_to(root) or path.resolve().is_relative_to(ROOT)
            or path.is_symlink() or not path.is_file()):
        raise ValueError("Evidence file is missing, unsafe or outside its artifact root")
    return path


def verify_descriptor(root, descriptor):
    if not isinstance(descriptor, dict) or set(("path", "sha256"))-set(descriptor):
        raise ValueError("A real evidence path and SHA-256 are required")
    path = bound_path(root, descriptor["path"])
    if digest_file(path) != descriptor["sha256"]:
        raise ValueError("Closed evidence bytes have changed: " + str(descriptor["path"]))
    return path


def _row(row):
    d = base._origin(row)
    result = {"model": str(row["model"]), "seed": str(row["seed"]), "outcome_day": d.isoformat(),
              "y_true": float(row["y_true"]), "y_pred": float(row["y_pred"]),
              "run_id": str(row["run_id"]), "checkpoint_sha256": str(row["checkpoint_sha256"]),
              "scaler_sha256": str(row["scaler_sha256"]), "config_sha256": str(row["config_sha256"])}
    if not np.isfinite([result["y_true"], result["y_pred"]]).all() or min(result["y_true"], result["y_pred"]) < 0:
        raise ValueError("Forecasts and truth must be finite nonnegative original-scale sigma7")
    for field in ("fit_origin", "fit_outcome_max", "scaler_fit_outcome_max", "configuration_selection_outcome_max",
                  "epoch_selection_outcome_max"):
        if field in row:
            result[field] = base.day(row[field]).isoformat()
    for field in ("checkpoint_closed_at_utc", "scoring_started_at_utc"):
        if field in row:
            result[field] = timestamp(row[field]).isoformat()
    for field in ("persistence_pred", "mase_scale"):
        if field in row:
            result[field] = float(row[field])
    for field in ("eligible", "high_volatility_known", "historical_selection_exposure_disclosed",
                  "E2_new_configuration_selection", "unbiased_nested_selection_claimed"):
        if field in row:
            result[field] = base._eligible(row[field])
    for field in ("candidate_id", "configuration_selection_kind", "source_stage"):
        if field in row:
            result[field] = str(row[field])
    return result


def panel_hash(rows):
    return digest_json(sorted((_row(r) for r in rows), key=lambda r: (r["model"], r["seed"], r["outcome_day"])))


def _verify_csv_identity(rows, path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        stored = list(csv.DictReader(stream))
    if panel_hash(rows) != panel_hash(stored):
        raise ValueError("Supplied rows differ from the hash-bound closed prediction CSV")


def validate_joint_order(receipt, *, artifact_root, evaluation=False):
    if not isinstance(receipt, dict):
        raise ValueError("A joint primary+A phase-order receipt is required")
    closed = timestamp(receipt["all_primary_and_A_weights_closed_at_utc"])
    start = timestamp(receipt["calibration_scoring_started_at_utc"])
    if not closed < start:
        raise ValueError("All primary and A weights must close before the first calibration forecast")
    verified = {}
    for name in ("primary_fit_manifest", "branch_A_fit_manifest", "calibration_predictions", "subgroup_receipt"):
        verified[name] = verify_descriptor(artifact_root, receipt[name])
    if evaluation:
        calibration_closed = timestamp(receipt["calibration_closed_at_utc"])
        evaluation_started = timestamp(receipt["evaluation_scoring_started_at_utc"])
        if not start < calibration_closed < evaluation_started:
            raise ValueError("All calibrators must seal before the first E2 evaluation forecast")
        for name in ("calibrators_file", "evaluation_predictions", "primary_predictions"):
            verified[name] = verify_descriptor(artifact_root, receipt[name])
    return verified


def _verify_member_files(receipt, *, artifact_root):
    seeds = [str(s) for s in receipt["ensemble_seeds"]]
    for field in ("member_files", "member_completion_sha256", "member_checkpoint_closed_at_utc"):
        if set(map(str, receipt.get(field, {}))) != set(seeds):
            raise ValueError("Every A member requires real files, completion hash and closure timestamp")
    for seed in seeds:
        files = receipt["member_files"][seed]
        actual = {}
        for field, hashes in (("checkpoint", "member_checkpoint_sha256"), ("scaler", "member_scaler_sha256"),
                              ("config", "member_config_sha256"), ("completion", "member_completion_sha256")):
            actual[field] = verify_descriptor(artifact_root, {"path": files[field], "sha256": receipt[hashes][seed]})
        completion = json.loads(actual["completion"].read_bytes())
        if completion.get("status") != "passed":
            raise ValueError("An A member did not close successfully")
        if timestamp(completion["completed_at_utc"]) != timestamp(receipt["member_checkpoint_closed_at_utc"][seed]):
            raise ValueError("A member's closure timestamp differs from its real completion")
        if timestamp(completion["completed_at_utc"]) > timestamp(receipt["checkpoint_closed_at_utc"]):
            raise ValueError("The cohort closure precedes one of its A members")
        setup = json.loads(actual["config"].read_bytes())
        if setup.get("model") != receipt["model"] or str(setup.get("seed")) != seed:
            raise ValueError("An A checkpoint belongs to a different model/seed")
        if base.day(setup["fit_origin"]) != CAL_START:
            raise ValueError("A member must be the nominal November8 frozen fit")
        for field in ("fit_outcome_max", "scaler_fit_outcome_max"):
            if base.day(setup[field]) > FIT_MAX or base.day(setup[field]) != base.day(receipt[field]):
                raise ValueError("A member's real setup violates the October31 fit/scaler cutoff")
        if (base.day(setup["configuration_selection_outcome_max"]) > FIT_MAX
                or base.day(setup["configuration_selection_outcome_max"]) != base.day(receipt["configuration_selection_outcome_max"])
                or base.day(setup["epoch_selection_outcome_max"]) > FIT_MAX):
            raise ValueError("An A member's actual recipe and epoch selection must respect the October31 cutoff")
        preprocessing = json.loads(actual["scaler"].read_bytes())
        if base.day(preprocessing["last_fit_outcome_day_utc"]) != base.day(receipt["scaler_fit_outcome_max"]):
            raise ValueError("An A member's scaler bytes disagree with its declared fit support")
        artifacts = completion.get("artifacts", completion.get("artifact_sha256", {}))
        for field in ("checkpoint", "scaler", "config"):
            if artifacts.get(actual[field].name) != digest_file(actual[field]):
                raise ValueError("A member artifact is absent or changed in its original fit completion")


def validate_calibration_support(receipt, calibration_origins, evaluation_origins=None):
    """Strict new53-date branch; does not weaken or modify the archived A60 guard."""
    required = {"branch", "model", "fit_origin", "fit_outcome_max", "scaler_fit_outcome_max",
                "configuration_selection_outcome_max", "branch_selection_outcome_max", "primary_fit_replaced",
                "point_predictor_kind", "ensemble_seeds", "member_checkpoint_sha256", "member_run_ids",
                "member_scaler_sha256", "member_config_sha256", "checkpoint_closed_at_utc",
                "calibration_scoring_started_at_utc", "target_scale"}
    if not required.issubset(receipt) or receipt["model"] not in CATALOGUE:
        raise ValueError("Incomplete registered frozen A53 predictor receipt")
    if receipt["branch"] != A_BRANCH or receipt["primary_fit_replaced"] is not False:
        raise ValueError("Use the explicitly amended A53 branch, not the archived A60 or primary fit")
    if receipt["target_scale"] != "original_unannualized_daily_volatility":
        raise ValueError("A53 intervals must use original unannualized daily sigma7")
    if base.day(receipt["fit_origin"]) != CAL_START:
        raise ValueError("A53 has the nominal fit origin November8/2024")
    for field in ("fit_outcome_max", "scaler_fit_outcome_max", "configuration_selection_outcome_max", "branch_selection_outcome_max"):
        if base.day(receipt[field]) > FIT_MAX:
            raise ValueError("A53 fit, scalers and recipe selection must end byOctober31 before the seven-day embargo")
    for field in ("refit_during_calibration", "calibration_used_for_weight_selection", "test_used_for_selection"):
        if receipt.get(field) is not False:
            raise ValueError("A53 calibration/test must not refit or select weights, recipes or thresholds")
    expected = ("deterministic",) if receipt["model"] in DETERMINISTIC else A_SEEDS
    if tuple(map(str, receipt["ensemble_seeds"])) != expected:
        raise ValueError("A53 uses the five new seed IDs or one deterministic reference")
    expected_kind = "deterministic_frozen_predictor" if receipt["model"] in DETERMINISTIC else "mean_frozen_seed_members"
    if receipt["point_predictor_kind"] != expected_kind:
        raise ValueError("A deterministic reference cannot be declared a randomized seed ensemble")
    for field in ("member_checkpoint_sha256", "member_scaler_sha256", "member_config_sha256"):
        hashes = receipt[field]
        if set(map(str, hashes)) != set(expected) or any(len(str(v)) != 64 or any(c not in "0123456789abcdef" for c in str(v)) for v in hashes.values()):
            raise ValueError("Each A53 member requires real checkpoint/scaler/config SHA-256 identities")
    if set(map(str, receipt["member_run_ids"])) != set(expected) or len(set(receipt["member_run_ids"].values())) != len(expected):
        raise ValueError("Every A53 member requires a unique frozen run identity")
    if not timestamp(receipt["checkpoint_closed_at_utc"]) < timestamp(receipt["calibration_scoring_started_at_utc"]):
        raise ValueError("All A53 members must close before actual calibration scoring")
    dates = [base.day(d) for d in calibration_origins]
    if dates != base.calendar(CAL_START, CAL_END):
        raise ValueError("Retain all53 calibration dates November8–December30/2024")
    if evaluation_origins is not None and [base.day(d) for d in evaluation_origins] != base.calendar(base.EVAL_START, base.EVAL_END):
        raise ValueError("A53 evaluation preserves all60 inherited September1–October30/2026 dates")
    return {"calibration_days": 53, "fit_and_selection_outcome_max_allowed": FIT_MAX.isoformat(),
            "first_calibration_label_close_support": ["2024-11-01", "2024-11-08"],
            "overlap_with_fit_or_selection_label_closes": False,
            "actual_checkpoint_closure_precedes_scoring": True, "new_unseen_test": False,
            "nominal_historical_dates_are_not_actual_live_issuance_times": True}


def _fit_one_calibrator(rows, receipt):
    dates = base.calendar(CAL_START, CAL_END)
    support = validate_calibration_support(receipt, dates)
    point, truth, _ = _dense_point(rows, receipt["ensemble_seeds"], dates)
    weights = np.exp2(-np.array([(CAL_END-d).days for d in dates])/28.)
    radii = {str(a): base.weighted_split_quantile(np.abs(truth-point), weights, a, test_weight=1.) for a in ALPHAS}
    signature = digest_json({"branch": A_BRANCH, "model": receipt["model"],
        "seeds": receipt["ensemble_seeds"], "checkpoints": receipt["member_checkpoint_sha256"],
        "scalers": receipt["member_scaler_sha256"], "configs": receipt["member_config_sha256"],
        "target_scale": receipt["target_scale"], "kind": receipt["point_predictor_kind"],
        "fit_outcome_max": receipt["fit_outcome_max"]})
    return {"model": receipt["model"], "predictor_signature": signature,
            "point_predictor": "fixed mean of five A53 seed members, or one deterministic reference",
            "calibration_origins": 53, "radii_by_alpha": radii, "half_life_days": 28.,
            "weights_indexed_to_fixed_calibration_end": CAL_END.isoformat(),
            "effective_weight_count": float(weights.sum()**2/(weights@weights)),
            "support": support, "exact_finite_sample_coverage_claimed": False,
            "long_gap_and_regime_shift_validity_established": False,
            "primary_benchmark_fit_changed": False}


def _validate_A_rows(rows, receipts, stage):
    if set(receipts) != set(CATALOGUE):
        raise ValueError("Calibrate and evaluate the complete twenty-model A cohort")
    dates = (base.calendar(CAL_START, CAL_END) if stage == CAL_STAGE
             else base.calendar(base.EVAL_START, base.EVAL_END))
    expected_dates = set(dates); grouped = defaultdict(list); seen = set(); truth = {}
    for row in rows:
        model, seed, origin = str(row["model"]), str(row["seed"]), base._origin(row)
        if model not in receipts or row.get("source_stage") != stage:
            raise ValueError("A rows cannot be substituted by P0, OOF or another forecast stage")
        receipt = receipts[model]
        expected_seeds = ("deterministic",) if model in DETERMINISTIC else A_SEEDS
        if tuple(map(str, receipt["ensemble_seeds"])) != expected_seeds:
            raise ValueError("A members must use the five fresh registered seeds or the single deterministic reference")
        key = model, seed, origin
        if seed not in expected_seeds or origin not in expected_dates or key in seen or not base._eligible(row.get("eligible", True)):
            raise ValueError("A forecasts must preserve every eligible member/date exactly once")
        normalized = _row(row)
        for field, target in (("run_id", "member_run_ids"), ("checkpoint_sha256", "member_checkpoint_sha256"),
                              ("scaler_sha256", "member_scaler_sha256"), ("config_sha256", "member_config_sha256")):
            if normalized[field] != receipt[target][seed]:
                raise ValueError("A calibration/evaluation run, checkpoint, scaler or config identity changed")
        if timestamp(row["checkpoint_closed_at_utc"]) != timestamp(receipt["member_checkpoint_closed_at_utc"][seed]):
            raise ValueError("A forecast closure does not match its frozen member")
        if not timestamp(row["checkpoint_closed_at_utc"]) < timestamp(row["scoring_started_at_utc"]):
            raise ValueError("A member must close before its forecast scoring begins")
        y = normalized["y_true"]
        if origin in truth and truth[origin] != y:
            raise ValueError("All A models and seeds must share identical daily targets")
        truth[origin] = y; seen.add(key); grouped[model].append(row)
    if len(seen) != (16*5+4)*len(dates):
        raise ValueError("The complete A cohort requires84 member trajectories on each registered calibration/evaluation date")
    return grouped


def _dense_point(rows, seeds, dates):
    indexed = {(str(r["seed"]), base._origin(r)): float(r["y_pred"]) for r in rows}
    y = {base._origin(r): float(r["y_true"]) for r in rows}
    matrix = np.array([[indexed[str(s), d] for d in dates] for s in seeds])
    return matrix.mean(0), np.array([y[d] for d in dates]), matrix


def fit_calibration_cohort(calibration_rows, receipts, *, artifact_root, cohort_fit_receipt=None):
    """Fit all20 residual calibrators before any final evaluation is decoded."""
    verified = validate_joint_order(cohort_fit_receipt, artifact_root=artifact_root)
    _verify_csv_identity(calibration_rows, verified["calibration_predictions"])
    subgroup = json.loads(verified["subgroup_receipt"].read_bytes())
    _validate_subgroup_receipt(subgroup, cohort_fit_receipt["calibration_scoring_started_at_utc"])
    grouped = _validate_A_rows(calibration_rows, receipts, CAL_STAGE)
    if min(timestamp(r["scoring_started_at_utc"]) for r in calibration_rows) < timestamp(cohort_fit_receipt["calibration_scoring_started_at_utc"]):
        raise ValueError("A calibration forecast predates the declared first calibration scoring event")
    dates = base.calendar(CAL_START, CAL_END)
    weights = np.exp2(-np.array([(CAL_END-d).days for d in dates])/28.)
    fitted = {}
    for model in CATALOGUE:
        receipt = receipts[model]
        validate_calibration_support(receipt, dates)
        _verify_member_files(receipt, artifact_root=artifact_root)
        if timestamp(receipt["checkpoint_closed_at_utc"]) > timestamp(cohort_fit_receipt["all_primary_and_A_weights_closed_at_utc"]):
            raise ValueError("Joint fit closure predates an A model closure")
        calibrator = _fit_one_calibrator(grouped[model], receipt)
        center, target, _ = _dense_point(grouped[model], receipt["ensemble_seeds"], dates)
        fitted[model] = {**calibrator, "signed_calibration_residuals": (target-center).tolist(),
                         "calibration_weights": weights.tolist(), "calibration_panel_sha256": panel_hash(grouped[model]),
                         "frozen_receipt_sha256": digest_json(receipt)}
    return {"models": fitted, "calibrated_at_utc": datetime.now(timezone.utc).isoformat(),
            "settings_sha256": digest_json(policy()), "calibration_panel_sha256": panel_hash(calibration_rows),
            "subgroup_receipt_sha256": digest_json(subgroup),
            "all20_calibrated": True, "calibration_fit_uses_evaluation_labels": False,
            "model_fits_performed": 0}


def build_subgroup_receipt(training_targets, training_origins, *, source_sha256):
    targets = base.vector(training_targets, "training2023 reference targets", True)
    dates = [base.day(d) for d in training_origins]
    if len(targets) != len(dates) or len(dates) < 20 or dates != sorted(set(dates)) or dates[-1] > base.TAXONOMY_TRAIN_MAX:
        raise ValueError("Conditional-coverage thresholds require at least20 distinct2023 training targets")
    edges = np.quantile(targets, [1/3, 2/3])
    if not np.isfinite(edges).all() or edges[0] >= edges[1]:
        raise ValueError("Training forecast-bin thresholds are not distinct; no arbitrary replacement is permitted")
    if len(str(source_sha256)) != 64:
        raise ValueError("The reference training source must be hash identified")
    return {"threshold_source": "training_only", "threshold_outcome_max": dates[-1].isoformat(),
            "training_cases": len(dates), "reference_variable": "sigma7_training_targets",
            "forecast_bin_quantiles": [1/3, 2/3], "forecast_bin_edges": edges.tolist(),
            "high_volatility_threshold": HIGH_THRESHOLD, "source_sha256": source_sha256,
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "test_or_calibration_used": False}


def _validate_subgroup_receipt(receipt, calibration_started_at):
    if (receipt.get("threshold_source") != "training_only"
            or base.day(receipt["threshold_outcome_max"]) > base.TAXONOMY_TRAIN_MAX
            or receipt.get("test_or_calibration_used") is not False
            or receipt.get("high_volatility_threshold") != HIGH_THRESHOLD
            or not timestamp(receipt["frozen_at_utc"]) < timestamp(calibration_started_at)):
        raise ValueError("Subgroup/range thresholds must freeze from2023 training before calibration")
    edges = base.vector(receipt["forecast_bin_edges"], "training forecast-bin edges", True)
    if len(edges) != 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("Keep the two distinct q33/q67 training thresholds")
    return edges


def _draws(n, block):
    rng = np.random.default_rng(MC_SEED+block)
    return np.array([base.block_indices(n, block, rng) for _ in range(BOOTSTRAPS)])


def _summary(values, mask, draws, *, block, conditional=False):
    values = base.vector(values, "per-origin score"); mask = np.asarray(mask, bool)
    if mask.shape != values.shape:
        raise ValueError("Subgroup masks must align to the full calendar")
    sampled_mask = mask[draws]; counts = sampled_mask.sum(1); usable = counts > 0
    averages = np.divide((values[draws]*sampled_mask).sum(1), counts,
                         out=np.full(len(draws), np.nan), where=usable)
    minimum = MIN_GROUP_CASES if conditional else 1
    sufficient = int(mask.sum()) >= minimum and usable.sum() >= .95*len(draws)
    return {"mean": float(values[mask].mean()) if mask.any() else None,
            "ci95_nominal": np.quantile(averages[usable], [.025, .975]).tolist() if sufficient else None,
            "eligible_group_origins": int(mask.sum()), "calendar_origins": len(mask),
            "block_length_days": block, "bootstrap_replicates": len(draws),
            "status": "estimated" if sufficient else "not_estimable_empty_or_small_group",
            "group_filtered_after_calendar_resampling": True, "simultaneous_CI": False,
            "zero_frequency_is_not_zero_population_probability": True}


def empirical_quantile_samples(samples, weights, tau):
    samples = np.asarray(samples, float); weights = base.vector(weights, "CDF weights")
    if samples.ndim != 2 or len(weights) != samples.shape[1] or (weights <= 0).any() or not 0 < tau < 1:
        raise ValueError("Weighted empirical quantiles require finite aligned distribution samples")
    if not np.isfinite(samples).all() or (samples < 0).any():
        raise ValueError("Predictive sigma7 samples must be finite and nonnegative")
    order = samples.argsort(axis=1, kind="stable")
    ordered = np.take_along_axis(samples, order, axis=1)
    cumulative = np.cumsum(weights[order]/weights.sum(), axis=1)
    indices = np.minimum((cumulative < tau).sum(1), samples.shape[1]-1)
    return ordered[np.arange(len(samples)), indices]


def randomized_empirical_pit(targets, samples, weights, *, seed=MC_SEED):
    targets = base.vector(targets, "PIT targets", True); samples = np.asarray(samples, float)
    weights = base.vector(weights, "PIT distribution weights")
    if samples.shape != (len(targets), len(weights)) or (weights <= 0).any() or not np.isfinite(samples).all():
        raise ValueError("PIT requires the explicit weighted empirical predictive CDF")
    probabilities = weights/weights.sum()
    below = ((samples < targets[:, None])*probabilities).sum(1)
    at_or_below = ((samples <= targets[:, None])*probabilities).sum(1)
    uniform = np.random.default_rng(seed).uniform(size=len(targets))
    return below + uniform*(at_or_below-below)


def pit_diagnostic(pit, draws, *, block):
    pit = base.vector(pit, "randomized PIT")
    if (pit < 0).any() or (pit > 1).any() or draws.shape[1] != len(pit):
        raise ValueError("PIT must remain in[0,1] on the complete evaluation calendar")
    grid = np.array(PIT_GRID)
    indicators = (pit[:, None] <= grid[None, :]).astype(float)
    observed_process = indicators.mean(0)-grid
    observed = float(np.max(np.abs(observed_process)))
    bootstrap_process = indicators[draws].mean(1)-indicators.mean(0)
    null_statistics = np.abs(bootstrap_process).max(1)
    raw_process = indicators[draws].mean(1)-grid
    l1_draws = np.abs(raw_process).mean(1)
    p = float((1+(null_statistics >= observed).sum())/(len(draws)+1))
    return {"PIT_calibration_L1_grid": float(np.abs(observed_process).mean()),
            "PIT_L1_ci95_nominal": np.quantile(l1_draws, [.025, .975]).tolist(),
            "max_grid_deviation": observed, "p_temporal_centered_bootstrap": p,
            "grid": list(PIT_GRID), "block_length_days": block, "bootstrap_replicates": len(draws),
            "null": "uniform marginal randomized PIT at the fixed grid",
            "method": "centered circular-block empirical process, not iid KS",
            "exploratory_only": True, "stationarity_and_residual_transfer_not_guaranteed": True,
            "classification_ECE_claimed": False, "finite_sample_exact_pvalue_claimed": False}


def subgroup_calibration(pit, point, truth, mask, draws):
    """Continuous calibration and OLS slope, preserving the full daily calendar."""
    pit, point, truth = (base.vector(v, "conditional calibration input") for v in (pit, point, truth))
    mask = np.asarray(mask, bool)
    if len(pit) != len(point) or len(point) != len(truth) or mask.shape != point.shape:
        raise ValueError("Subgroup calibration must retain one aligned dense forecast calendar")
    sampled_mask = mask[draws]; counts = sampled_mask.sum(1); usable = counts > 0
    n = int(mask.sum()); minimum = n >= MIN_GROUP_CASES
    if n:
        indicator = (pit[:, None] <= np.asarray(PIT_GRID)[None, :]).astype(float)
        process = indicator[mask].mean(0)-np.asarray(PIT_GRID)
        observed_l1 = float(np.abs(process).mean())
        sampled_cdf = np.divide((indicator[draws]*sampled_mask[:, :, None]).sum(1), counts[:, None],
                                out=np.full((len(draws), len(PIT_GRID)), np.nan), where=usable[:, None])
        l1 = np.abs(sampled_cdf-np.asarray(PIT_GRID)).mean(1)
    else:
        observed_l1, l1 = None, np.full(len(draws), np.nan)
    pit_estimated = minimum and usable.sum() >= .95*len(draws)
    pit_result = {"value": observed_l1, "ci95_nominal": np.quantile(l1[usable], [.025, .975]).tolist() if pit_estimated else None,
                  "status": "estimated" if pit_estimated else "not_estimable_empty_or_small_group",
                  "cases": n, "classification_ECE_claimed": False}
    slope_result = {"slope": None, "intercept": None, "slope_ci95_nominal": None, "intercept_ci95_nominal": None,
                    "status": "not_estimable_empty_small_or_constant_prediction", "cases": n,
                    "ideal_point_calibration": {"slope": 1., "intercept": 0.}, "new_hypothesis_decision": False}
    if n >= 2:
        x, y = point[mask], truth[mask]
        variance = float(((x-x.mean())**2).sum())
        if variance > np.finfo(float).eps*max(float((x*x).sum()), np.finfo(float).tiny):
            slope = float(((x-x.mean())*(y-y.mean())).sum()/variance)
            slope_result.update(slope=slope, intercept=float(y.mean()-slope*x.mean()))
            sx = (point[draws]*sampled_mask).sum(1); sy = (truth[draws]*sampled_mask).sum(1)
            sxx = (point[draws]**2*sampled_mask).sum(1); sxy = (point[draws]*truth[draws]*sampled_mask).sum(1)
            denom = sxx-np.divide(sx*sx, counts, out=np.zeros(len(counts)), where=usable)
            stable = usable & (counts >= 2) & (denom > np.finfo(float).eps*np.maximum(sxx, np.finfo(float).tiny))
            if minimum and stable.sum() >= .95*len(draws):
                slopes = (sxy[stable]-sx[stable]*sy[stable]/counts[stable])/denom[stable]
                intercepts = (sy[stable]-slopes*sx[stable])/counts[stable]
                slope_result.update(status="estimated", slope_ci95_nominal=np.quantile(slopes, [.025, .975]).tolist(),
                                    intercept_ci95_nominal=np.quantile(intercepts, [.025, .975]).tolist())
    return {"PIT_calibration_L1_grid": pit_result, "point_calibration_OLS": slope_result,
            "group_filtered_after_calendar_resampling": True, "nominal_not_simultaneous": True}


def _holm(values):
    array = np.array([1. if p is None else float(p) for p in values]); order = array.argsort(kind="stable")
    adjusted = np.empty(len(array)); adjusted[order] = np.minimum(1., np.maximum.accumulate(array[order]*np.arange(len(array), 0, -1)))
    return [None if p is None else float(a) for p, a in zip(values, adjusted)]


def _primary_panel(rows):
    dates = base.calendar(base.EVAL_START, base.EVAL_END); seen = {}; targets = {}; seeds = defaultdict(set); identities = {}
    for row in rows:
        model, seed, origin = str(row["model"]), str(row["seed"]), base._origin(row)
        if model not in CATALOGUE or row.get("source_stage") != PRIMARY_STAGE or origin not in dates:
            raise ValueError("Comparable primary forecasts must be the registered final E2 cohort")
        key = model, seed, origin
        if key in seen:
            raise ValueError("Primary forecast model/seed/date duplicated")
        normalized = _row(row); seen[key] = normalized["y_pred"]; seeds[model].add(seed)
        support = tuple(normalized[k] for k in ("run_id", "checkpoint_sha256", "scaler_sha256", "config_sha256"))
        if (model, seed) in identities and identities[model, seed] != support:
            raise ValueError("A primary member cannot switch weights, scaler, recipe or run within evaluation")
        identities[model, seed] = support
        if not timestamp(row["checkpoint_closed_at_utc"]) < timestamp(row["scoring_started_at_utc"]):
            raise ValueError("Primary member closure must precede its final scoring")
        if origin in targets and targets[origin] != normalized["y_true"]:
            raise ValueError("Primary forecasts do not share one common target calendar")
        targets[origin] = normalized["y_true"]
    if set(seeds) != set(CATALOGUE):
        raise ValueError("Primary comparable panel must contain all20 models")
    randomized = [seeds[m] for m in CATALOGUE if m not in DETERMINISTIC]
    if len(randomized[0]) < 5 or any(s != randomized[0] for s in randomized):
        raise ValueError("Primary neural models require the same declared seed cohort, with at least5 members")
    if len(randomized[0]) > 200 or randomized[0] != {str(s) for s in range(670101, 670101+len(randomized[0]))}:
        raise ValueError("Primary member IDs must be the registered final prefix, never P0 or A seeds")
    output = {}
    for model in CATALOGUE:
        if model in DETERMINISTIC and seeds[model] != {"deterministic"}:
            raise ValueError("A deterministic reference must not be duplicated into fictitious independent runs")
        members = sorted(seeds[model]); expected = {(model, s, d) for s in members for d in dates}
        if not expected.issubset(seen):
            raise ValueError("A primary comparable member/date is missing")
        matrix = np.array([[seen[model, s, d] for d in dates] for s in members])
        target = np.array([targets[d] for d in dates])
        output[model] = {"point": matrix.mean(0), "target": target,
                         "mean_seed_RMSE": float(np.sqrt(((matrix-target)**2).mean(1)).mean()),
                         "fixed_ensemble_RMSE": float(np.sqrt(((matrix.mean(0)-target)**2).mean()))}
    if len(seen) != sum(len(seeds[m])*len(dates) for m in CATALOGUE):
        raise ValueError("The primary comparable panel is not a complete common calendar")
    return output


def evaluate_final_uncertainty(calibration_rows, evaluation_rows, receipts, *, primary_rows,
                               subgroup_receipt, top3, artifact_root, cohort_receipt,
                               frozen_calibrators):
    verified = validate_joint_order(cohort_receipt, artifact_root=artifact_root, evaluation=True)
    for rows, name in ((calibration_rows, "calibration_predictions"), (evaluation_rows, "evaluation_predictions"),
                       (primary_rows, "primary_predictions")):
        _verify_csv_identity(rows, verified[name])
    stored_calibrators = json.loads(verified["calibrators_file"].read_bytes())
    if digest_json(stored_calibrators) != digest_json(frozen_calibrators):
        raise ValueError("The supplied calibrators differ from their sealed pre-evaluation bytes")
    if (frozen_calibrators.get("settings_sha256") != digest_json(policy())
            or frozen_calibrators.get("calibration_panel_sha256") != panel_hash(calibration_rows)
            or not timestamp(frozen_calibrators["calibrated_at_utc"]) < timestamp(cohort_receipt["calibration_closed_at_utc"])):
        raise ValueError("Frozen calibrators do not match the registered settings, calibration panel or phase order")
    _validate_A_rows(calibration_rows, receipts, CAL_STAGE)
    grouped = _validate_A_rows(evaluation_rows, receipts, EVAL_STAGE)
    first_test = timestamp(cohort_receipt["evaluation_scoring_started_at_utc"])
    if min(timestamp(r["scoring_started_at_utc"]) for r in evaluation_rows+primary_rows) < first_test:
        raise ValueError("An evaluation forecast predates the registered first E2 scoring event")
    if max(timestamp(r["scoring_started_at_utc"]) for r in calibration_rows) >= timestamp(cohort_receipt["calibration_closed_at_utc"]):
        raise ValueError("The calibration phase closed before its predictions were generated")
    if max(timestamp(r["checkpoint_closed_at_utc"]) for r in evaluation_rows+primary_rows) > timestamp(cohort_receipt["all_primary_and_A_weights_closed_at_utc"]):
        raise ValueError("The joint weight closure predates one of its actual forecast members")
    primary = _primary_panel(primary_rows)
    if len(top3) != 3 or len(set(top3)) != 3 or not set(top3).issubset(CATALOGUE):
        raise ValueError("Report exactly three models selected by the primary development procedure")
    actual_subgroup = json.loads(verified["subgroup_receipt"].read_bytes())
    if (digest_json(subgroup_receipt) != digest_json(actual_subgroup)
            or digest_json(actual_subgroup) != frozen_calibrators.get("subgroup_receipt_sha256")):
        raise ValueError("Conditional-coverage thresholds changed after calibrator sealing")
    edges = _validate_subgroup_receipt(subgroup_receipt, cohort_receipt["calibration_scoring_started_at_utc"])
    dates = base.calendar(base.EVAL_START, base.EVAL_END); draws = {b: _draws(len(dates), b) for b in BLOCKS}
    output = {}; evaluation_frontier = {}
    for index, model in enumerate(CATALOGUE):
        receipt = receipts[model]; calibrator = frozen_calibrators["models"][model]
        validate_calibration_support(receipt, base.calendar(CAL_START, CAL_END), dates)
        _verify_member_files(receipt, artifact_root=artifact_root)
        if calibrator.get("frozen_receipt_sha256") != digest_json(receipt):
            raise ValueError("An A model receipt changed after its calibrator was fitted")
        point, truth, _ = _dense_point(grouped[model], receipt["ensemble_seeds"], dates)
        if not np.array_equal(truth, primary[model]["target"]):
            raise ValueError("A and primary point comparisons must share exactly the same truth")
        by_date = {base._origin(r): r for r in grouped[model]}
        persistence = np.array([float(by_date[d]["persistence_pred"]) for d in dates])
        high = persistence > HIGH_THRESHOLD
        for j, d in enumerate(dates):
            if base._eligible(by_date[d]["high_volatility_known"]) != bool(high[j]):
                raise ValueError("The declared high-volatility group differs from causal persistence and its frozen threshold")
        forecast_bins = np.digitize(point, edges)
        masks = {"all": np.ones(len(dates), bool), "high_volatility": high, "normal_volatility": ~high,
                 **{"forecast_bin_"+str(i): forecast_bins == i for i in range(3)}}
        intervals = {float(a): base.symmetric_intervals(point, radius) for a, radius in calibrator["radii_by_alpha"].items()}
        score = base.interval_scores(truth, point, intervals)
        signed = base.signed_residual_distribution(point, calibrator["signed_calibration_residuals"])
        weights = base.vector(calibrator["calibration_weights"], "frozen residual weights")
        crps = base.empirical_crps(truth, signed, weights=weights)
        pit = randomized_empirical_pit(truth, signed, weights, seed=MC_SEED+index)
        per_quantile = {str(tau): base.pinball(truth, empirical_quantile_samples(signed, weights, tau), tau) for tau in QUANTILES}
        block_results = {}
        for block, indices in draws.items():
            groups = {}
            for group, mask in masks.items():
                conditional = group != "all"
                summary = lambda values: _summary(values, mask, indices, block=block, conditional=conditional)
                groups[group] = {"absolute_residual_intervals": {
                    str(1-float(a)): {name: summary(values) for name, values in metrics.items()}
                    for a, metrics in score.items() if a != "WIS"},
                    "absolute_interval_WIS": summary(score["WIS"]),
                    "signed_empirical_CDF_CRPS": summary(crps),
                    "signed_empirical_CDF_pinball": {tau: summary(values) for tau, values in per_quantile.items()},
                    "conditional_calibration": subgroup_calibration(pit, point, truth, mask, indices)}
            block_results[str(block)] = {"groups": groups, "PIT": pit_diagnostic(pit, indices, block=block),
                                        "principal_uncertainty_block": block == 14}
        output[model] = {"radii_by_alpha": calibrator["radii_by_alpha"],
                         "residual_weight_effective_count_not_independent_dates": calibrator["effective_weight_count"],
                         "block_summaries": block_results,
                         "point_performance_side_by_side": {
                             "branch_A_fixed_ensemble_RMSE": float(np.sqrt(((point-truth)**2).mean())),
                             "primary_fixed_ensemble_RMSE": primary[model]["fixed_ensemble_RMSE"],
                             "primary_mean_seed_RMSE_distinct_estimand": primary[model]["mean_seed_RMSE"],
                             "causal_improvement_from_calibration_claimed": False}}
        width95 = empirical_quantile_samples(signed, weights, .975)-empirical_quantile_samples(signed, weights, .025)
        high_loss_indices = np.argsort(-(point-truth)**2, kind="stable")[:10]
        low_confidence_indices = np.argsort(-width95, kind="stable")[:10]
        spread = float(np.ptp(width95)); scale = max(float(np.max(np.abs(width95))), np.finfo(float).tiny)
        evaluation_frontier[model] = {
            "high_loss_origins": [dates[int(i)].isoformat() for i in high_loss_indices],
            "widest_calibrated95_CDF_interval_origins": [dates[int(i)].isoformat() for i in low_confidence_indices],
            "confidence_proxy": "width of signed empirical CDF95percent interval, determined without evaluation truth",
            "all_widths_numerically_tied": spread <= 100*np.finfo(float).eps*scale,
            "width_min": float(width95.min()), "width_max": float(width95.max()),
            "tie_rule": "earliest forecast origins; tied widths do not establish a meaningful confidence ordering",
            "calibration_period_precedes_these2026_origins": True,
            "no_confidence_retrofitted_to_2024_OOF": True}
    for block in BLOCKS:
        tests = [output[m]["block_summaries"][str(block)]["PIT"]["p_temporal_centered_bootstrap"] for m in CATALOGUE]
        for model, adjusted in zip(CATALOGUE, _holm(tests)):
            record = output[model]["block_summaries"][str(block)]["PIT"]
            record.update(p_Holm_exploratory_family20=adjusted, alpha=.01,
                          principal_diagnostic=block == 14,
                          reject_exploratory_uniform_grid=(adjusted < .01 if block == 14 else None))
    public = {"scope": "all20 frozen branchA on known historical evaluation; primary benchmark unchanged",
              "models": output, "reported_primary_top3": list(top3), "all20_precalibrated": True,
              "intervals_and_CDF_are_distinct_procedures": True,
              "coverage_exact_or_global_FWER_claimed": False, "new_unseen_test": False,
              "PIT_family_separate_exploratory_not_extension_of_primary_global_alpha": True,
              "frontier_confidence_limit": "additive residual uncertainty can have tied widths; no artificial confidence ordering is claimed",
              "CI_scope": "calendar bootstrap conditional on frozen fitted members and calibration residuals; temporal shift and20month transfer unproved",
              "model_fits_performed": 0, "private_cases_or_predictions_exported": False,
              "settings": policy()}
    return {"summary": clean(public), "private_calibrators": frozen_calibrators,
            "private_evaluation_frontier": clean(evaluation_frontier)}


def validate_final_oof_panel(rows, *, models=CATALOGUE, seeds=OOF_SEEDS,
                             deterministic_models=DETERMINISTIC):
    """Keep the17-model nested guard; expose the three inherited E1 selections.

    A reused E1 recipe is conditional on a configuration selected using2024
    outcomes throughOctober31. Its new monthly weights, preprocessing and epoch
    selection still obey the outer-origin embargo. No date is rewritten before
    calling the archived validator and no exception is granted to new tuning.
    """
    models, seeds = list(models), list(map(str, seeds))
    if (set(models) != set(CATALOGUE) or len(models) != len(CATALOGUE)
            or tuple(seeds) != OOF_SEEDS or set(deterministic_models) != set(DETERMINISTIC)):
        raise ValueError("Final OOF must use all20 registered models, two HPO seeds and four deterministic references")
    required = {"model", "seed", "source_stage", "candidate_id", "fit_origin", "fit_outcome_max",
                "scaler_fit_outcome_max", "configuration_selection_outcome_max", "epoch_selection_outcome_max",
                "historical_selection_exposure_disclosed", "E2_new_configuration_selection",
                "unbiased_nested_selection_claimed", "configuration_selection_kind", "eligible", "run_id",
                "checkpoint_sha256", "scaler_sha256", "config_sha256", "y_true", "y_pred",
                "checkpoint_closed_at_utc", "scoring_started_at_utc"}
    if any(not required.issubset(row) for row in rows):
        raise ValueError("Final OOF rows require complete dates, frozen identities and explicit selection-provenance claims")
    if any(r.get("source_stage") != "E2_nested_development" for r in rows):
        raise ValueError("Use the final development stage, not P0, recipe-search or test rows")
    nested_models = [m for m in models if m not in LEGACY_CANDIDATES]
    nested_rows = [r for r in rows if r.get("model") in nested_models]
    panel = base.validate_oof_panel(nested_rows, nested_models, seeds,
        source_stage="E2_nested_development", deterministic_models=deterministic_models)
    if not panel["eligible"].all():
        raise ValueError("The audited final cohort retains all305 common eligible calendar dates")
    dates = panel["dates"]; positions = {d: i for i, d in enumerate(dates)}
    legacy_predictions, frozen = {}, {}
    for row in rows:
        model, seed, origin = row["model"], str(row["seed"]), base._origin(row)
        if model not in models or origin not in positions or not base._eligible(row["eligible"]):
            raise ValueError("Every final development row must retain a registered model and common eligible date")
        cutoff = origin.replace(day=1)-timedelta(days=8)
        if (base.day(row["fit_origin"]) != origin.replace(day=1)
                or any(base.day(row[k]) > cutoff for k in
                       ("fit_outcome_max", "scaler_fit_outcome_max", "epoch_selection_outcome_max"))):
            raise ValueError("All20 monthly fitting, scaling and epoch selection must respect the outer-origin embargo")
        inherited = model in LEGACY_CANDIDATES
        if (base._eligible(row["historical_selection_exposure_disclosed"]) != inherited
                or base._eligible(row["E2_new_configuration_selection"]) == inherited
                or base._eligible(row["unbiased_nested_selection_claimed"]) == inherited):
            raise ValueError("The inherited-selection exposure and nested-selection claims must be explicit and truthful")
        if inherited:
            if (row["candidate_id"] != LEGACY_CANDIDATES[model]
                    or base.day(row["configuration_selection_outcome_max"]) != FIT_MAX
                    or row["configuration_selection_kind"] != "inherited_fixed_E1_recipe"
                    or seed not in seeds):
                raise ValueError("Only the exact three fixed E1 recipes may disclose the historical October31 selection support")
            y, prediction = float(row["y_true"]), float(row["y_pred"])
            if not np.isfinite([y, prediction]).all() or min(y, prediction) < 0:
                raise ValueError("Inherited-reference targets and forecasts must be finite nonnegative volatility")
            if y != panel["target"][positions[origin]]:
                raise ValueError("Inherited references must use the same daily ground truth as the nested models")
            key = model, seed, origin
            if key in legacy_predictions:
                raise ValueError("An inherited reference cannot duplicate a model/seed/date")
            legacy_predictions[key] = prediction
        elif row["configuration_selection_kind"] != "registered_nested_selection_or_rule":
            raise ValueError("New model selections must retain their registered nested-selection provenance")
        monthly = model, seed, origin.year, origin.month
        support = tuple(str(row[k]) for k in ("candidate_id", "run_id", "checkpoint_sha256", "scaler_sha256",
                         "config_sha256", "fit_origin", "fit_outcome_max", "scaler_fit_outcome_max",
                         "configuration_selection_outcome_max", "epoch_selection_outcome_max"))
        if monthly in frozen and frozen[monthly] != support:
            raise ValueError("A monthly frozen predictor cannot switch candidate, checkpoint, scaler or selection support")
        frozen[monthly] = support
    if len(legacy_predictions) != len(LEGACY_CANDIDATES)*len(seeds)*len(dates):
        raise ValueError("Every inherited reference must retain both registered seeds and all305 dates")
    predictions = {m: panel["predictions"][i] for i, m in enumerate(nested_models)}
    for model in LEGACY_CANDIDATES:
        predictions[model] = np.array([[legacy_predictions[model, s, d] for d in dates] for s in seeds])
    return {**panel, "models": models, "predictions": np.array([predictions[m] for m in models]),
            "nested_configuration_selection_models": nested_models,
            "conditional_inherited_references": list(LEGACY_CANDIDATES),
            "legacy_hyperparameter_selection_not_unbiased": True,
            "all_models_unbiased_nested_configuration_selection": False}


def _verify_oof_prediction_chain(rows, records, *, artifact_root, registration_inputs_sha256):
    """Accept retained or explicitly pruned weights through their closed score chain."""
    from run_e2_final_benchmark import verify_job, hash_json, fixed_candidate
    if len(str(registration_inputs_sha256)) != 64:
        raise ValueError("The OOF panel must identify the registered completion input hash")
    by_run = defaultdict(list)
    for row in rows:
        run = str(row.get("run_id", ""))
        if len(run) != 64 or any(c not in "0123456789abcdef" for c in run):
            raise ValueError("Final OOF must use the registered engine run IDs, not relabeled P0 trajectories")
        by_run[run].append(row)
    if set(by_run) != set(records):
        raise ValueError("Every final OOF run requires its own complete fit/scoring evidence record")
    pruned = 0
    for run, group in by_run.items():
        descriptor = records[run]
        folder_relative = str(descriptor["folder"])
        completion_path = bound_path(artifact_root, str(Path(folder_relative)/"completion.json"))
        folder = completion_path.parent
        completion = verify_job(folder, registration_inputs_sha256)
        if completion.get("status") != "passed" or hash_json({"fit": completion["identity"]}) != run:
            raise ValueError("Final OOF fit status or source identity is inconsistent")
        score_path = bound_path(artifact_root, descriptor["scoring_receipt"])
        score = json.loads(score_path.read_bytes())
        if (score.get("status") != "passed" or score.get("registration_inputs_sha256") != registration_inputs_sha256
                or score.get("fit_completion_sha256") != digest_file(completion_path)):
            raise ValueError("Final OOF scoring is not bound to this exact closed fit")
        predictions_path = bound_path(artifact_root, descriptor["predictions_file"])
        if (predictions_path.parent != score_path.parent
                or score.get("artifacts", {}).get(predictions_path.name) != digest_file(predictions_path)):
            raise ValueError("Final OOF predictions differ from the closed scoring artifact")
        for relative, expected in score.get("artifacts", {}).items():
            actual = bound_path(artifact_root, str(score_path.parent.relative_to(Path(artifact_root).resolve())/relative))
            if digest_file(actual) != expected:
                raise ValueError("An OOF score artifact changed after closure")
        with predictions_path.open(encoding="utf-8-sig", newline="") as stream:
            original = [r for r in csv.DictReader(stream) if str(r.get("run_id")) == run]
        if not original or any(not str(r.get("source_stage", "")).startswith("E2_") for r in original):
            raise ValueError("An archived P0 score cannot be relabeled as a final nested-development run")
        current_dates = {base._origin(r) for r in group}
        original = [r for r in original if base._origin(r) in current_dates]
        if panel_hash(group) != panel_hash(original):
            raise ValueError("Final OOF rows were modified relative to the original closed score CSV")
        setup_path = folder/"setup.json"; preprocessing_path = folder/"preprocessing.json"
        setup = json.loads(setup_path.read_bytes()); preprocessing = json.loads(preprocessing_path.read_bytes())
        identity = completion["identity"]
        for row in group:
            if (row["checkpoint_sha256"] not in completion.get("artifacts", {}).values()
                    or row["config_sha256"] != digest_file(setup_path)
                    or row["scaler_sha256"] != digest_file(preprocessing_path)
                    or timestamp(row["checkpoint_closed_at_utc"]) != timestamp(completion["completed_at_utc"])
                    or not timestamp(row["checkpoint_closed_at_utc"]) < timestamp(row["scoring_started_at_utc"])):
                raise ValueError("A final OOF row no longer matches its checkpoint/scaler/configuration closure")
            if (setup.get("model") != row["model"] or str(setup.get("seed")) != str(row["seed"])
                    or identity.get("model") != row["model"] or str(identity.get("seed")) != str(row["seed"])
                    or setup.get("candidate", {}).get("candidate_id") != row["candidate_id"]
                    or identity.get("candidate_id") != row["candidate_id"]
                    or identity.get("candidate_sha256") != hash_json(setup.get("candidate"))
                    or base.day(identity["origin"]) != base.day(row["fit_origin"])
                    or base.day(identity["fit_outcome_max"]) != base.day(row["fit_outcome_max"])
                    or any(base.day(setup[k]) != base.day(row[k]) for k in
                           ("fit_origin", "fit_outcome_max", "scaler_fit_outcome_max", "epoch_selection_outcome_max",
                            "configuration_selection_outcome_max"))
                    or base.day(preprocessing["last_fit_outcome_day_utc"]) != base.day(row["scaler_fit_outcome_max"])):
                raise ValueError("OOF support dates and model/candidate identity must match the real frozen fit setup")
            if row["model"] in LEGACY_CANDIDATES:
                source_candidate = fixed_candidate(row["model"])
                if (setup["candidate"] != source_candidate or identity.get("candidate_sha256") != hash_json(source_candidate)
                        or base.day(setup["configuration_selection_outcome_max"]) != FIT_MAX
                        or setup.get("configuration_selection_kind") != "inherited_fixed_E1_recipe"
                        or setup.get("historical_selection_exposure_disclosed") is not True
                        or setup.get("E2_new_configuration_selection") is not False):
                    raise ValueError("The actual E1 fit setup must retain the exact pinned recipe and historical selection exposure")
        if (folder/"checkpoint_pruned_after_development_scoring.json").exists():
            pruned += 1
    return {"closed_runs_verified": len(by_run), "runs_with_explicit_checkpoint_pruning": pruned,
            "pruned_weights_claimed_present": False, "original_fit_and_closed_score_chains_verified": True}


def analyze_final_oof(rows, taxonomy, *, models=CATALOGUE, seeds=OOF_SEEDS,
                      deterministic_models=DETERMINISTIC, artifact_root, fit_records,
                      registration_inputs_sha256, top3=None):
    if set(models) != set(CATALOGUE) or len(models) != len(CATALOGUE):
        raise ValueError("Final error analysis must cover all20 registered candidates/references")
    if tuple(map(str, seeds)) != OOF_SEEDS or set(deterministic_models) != set(DETERMINISTIC):
        raise ValueError("Final OOF uses the registered two HPO seeds and four single deterministic references")
    if any(r.get("source_stage") != "E2_nested_development" for r in rows):
        raise ValueError("P0, recipe-search or final-test rows cannot replace final nested OOF")
    panel = validate_final_oof_panel(rows, models=models, seeds=seeds, deterministic_models=deterministic_models)
    chain = _verify_oof_prediction_chain(rows, fit_records, artifact_root=artifact_root,
                                         registration_inputs_sha256=registration_inputs_sha256)
    if (taxonomy.get("threshold_source") != "training_reference_absolute_errors_only"
            or base.day(taxonomy["training_outcome_max"]) > base.TAXONOMY_TRAIN_MAX
            or int(taxonomy.get("training_cases", 0)) < 20):
        raise ValueError("Use the pre-2024 frozen training-reference error taxonomy")
    if "created_at_utc" in taxonomy and timestamp(taxonomy["created_at_utc"]) >= min(timestamp(r["scoring_started_at_utc"]) for r in rows):
        raise ValueError("Freeze the training taxonomy before scoring final OOF errors")
    result = base.analyze_oof_errors(panel, taxonomy, sample_size=100,
                                     repetitions=BOOTSTRAPS, block_length=14, seed=MC_SEED)
    audit_cases = {model: result["models"][model]["private_stratified_audit_sample"] for model in models}
    frontier = {model: {"high_loss_origins": result["models"][model]["private_high_loss_frontier_origins"],
                        "confidence_frontier_status": "not_assigned_using_futureNovDeccalibration",
                        "outlier_definition": "largest mean squared prediction loss on unique OOF origins"}
                for model in models}
    if any(len(v) != 100 or len({r["origin_utc"] for r in v}) != 100 for v in audit_cases.values()):
        raise ValueError("Each final model must retain exactly100 distinct audit dates")
    public = base.public_error_summary(result)
    public.pop("final_top3_identified", None)
    public.update({"scope": "Final development errors:17 nested-selection models and3 historically exposed inherited conditional references",
                   "registered_cohort": list(models), "cases_audited_per_model": 100,
                   "private_unique_audit_dates_total_model_pairs": 2000,
                   "primary_top3_reporting": list(top3) if top3 is not None else None,
                   "primary_top3_available_from_separate_development_selection": top3 is not None,
                   "top3_selected_by_this_analysis": False, "checkpoint_evidence": chain,
                   "nested_configuration_selection_models": panel["nested_configuration_selection_models"],
                   "conditional_inherited_references": panel["conditional_inherited_references"],
                   "legacy_hyperparameter_selection_not_unbiased": True,
                   "all_models_unbiased_nested_configuration_selection": False,
                   "inherited_reference_configuration_selection_outcome_max": FIT_MAX.isoformat(),
                   "postOOF_calibration_used_for_2024_confidence": False,
                   "zero_category_frequency_is_not_zero_population_probability": True})
    if top3 is not None and (len(top3) != 3 or len(set(top3)) != 3 or not set(top3).issubset(models)):
        raise ValueError("Report the three already selected primary models, without selecting by error taxonomy")
    return {"summary": clean(public), "private_audit_cases": clean(audit_cases), "private_frontier": clean(frontier)}


def _read_json(path):
    return json.loads(Path(path).read_bytes())


def _read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_json(path, value):
    Path(path).write_bytes(canonical_bytes(value))


def _destination(path):
    destination = Path(path).resolve()
    if destination.is_relative_to(ROOT) or (destination.exists() and any(destination.iterdir())):
        raise ValueError("Use a new private output directory outside the repository; no silent overwrite")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "calibrate", "evaluate", "oof"), nargs="?", default="plan")
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--calibration-csv", type=Path)
    parser.add_argument("--evaluation-csv", type=Path)
    parser.add_argument("--primary-csv", type=Path)
    parser.add_argument("--receipts-json", type=Path)
    parser.add_argument("--cohort-receipt-json", type=Path)
    parser.add_argument("--calibrators-json", type=Path)
    parser.add_argument("--subgroup-receipt-json", type=Path)
    parser.add_argument("--top3", nargs=3)
    parser.add_argument("--oof-csv", type=Path)
    parser.add_argument("--taxonomy-json", type=Path)
    parser.add_argument("--fit-records-json", type=Path)
    parser.add_argument("--registration-inputs-sha256")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "plan":
        print(json.dumps(policy(), ensure_ascii=False, indent=2)); return 0
    try:
        if args.artifact_root is None or args.output_dir is None:
            parser.error("An explicit private artifact root and new output directory are required")
        if args.artifact_root.resolve().is_relative_to(ROOT):
            raise ValueError("Individual forecasts, checkpoints and case ledgers must remain outside the repository")
        common = {"artifact_root": args.artifact_root}
        paths = {}
        if args.command in ("calibrate", "evaluate"):
            for field in ("calibration_csv", "receipts_json", "cohort_receipt_json"):
                value = getattr(args, field)
                if value is None:
                    parser.error("Missing --"+field.replace("_", "-"))
                paths[field] = value
            calibration = _read_csv(args.calibration_csv); receipts = _read_json(args.receipts_json)
            cohort = _read_json(args.cohort_receipt_json)
            if args.command == "calibrate":
                value = fit_calibration_cohort(calibration, receipts, cohort_fit_receipt=cohort, **common)
                output = _destination(args.output_dir)
                _write_json(output/"calibrators_private.json", value)
                public = {"all20_calibrated": True, "calibration_origins": 53, "fit_selection_max": "2024-10-31",
                          "calibrated_at_utc": value["calibrated_at_utc"], "settings_sha256": value["settings_sha256"],
                          "model_fits": 0, "evaluation_data_opened": False}
            else:
                for field in ("evaluation_csv", "primary_csv", "calibrators_json", "subgroup_receipt_json", "top3"):
                    if getattr(args, field) is None:
                        parser.error("Missing --"+field.replace("_", "-"))
                    if field != "top3":
                        paths[field] = getattr(args, field)
                # Inspect metadata/order before decoding any evaluation rows.
                validate_joint_order(cohort, artifact_root=args.artifact_root, evaluation=True)
                result = evaluate_final_uncertainty(calibration, _read_csv(args.evaluation_csv), receipts,
                    primary_rows=_read_csv(args.primary_csv), subgroup_receipt=_read_json(args.subgroup_receipt_json),
                    top3=args.top3, cohort_receipt=cohort, frozen_calibrators=_read_json(args.calibrators_json), **common)
                output = _destination(args.output_dir); public = result["summary"]
                _write_json(output/"private_evaluation_frontier.json", result["private_evaluation_frontier"])
            _write_json(output/"summary.json", public)
        else:
            for field in ("oof_csv", "taxonomy_json", "fit_records_json", "registration_inputs_sha256"):
                if getattr(args, field) is None:
                    parser.error("Missing --"+field.replace("_", "-"))
                if field != "registration_inputs_sha256":
                    paths[field] = getattr(args, field)
            result = analyze_final_oof(_read_csv(args.oof_csv), _read_json(args.taxonomy_json),
                fit_records=_read_json(args.fit_records_json), registration_inputs_sha256=args.registration_inputs_sha256,
                top3=args.top3, **common)
            output = _destination(args.output_dir); public = result["summary"]
            _write_json(output/"error_summary.json", public)
            _write_json(output/"private_frontier.json", result["private_frontier"])
            with (output/"private_audit_cases.csv").open("w", newline="") as stream:
                fields = ["model", "origin_utc", "category", "calendar_quarter", "stratum_population", "stratum_sampled", "inverse_inclusion_weight"]
                writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
                for model, records in result["private_audit_cases"].items():
                    writer.writerows({"model": model, **r} for r in records)
        _write_json(output/"analysis_receipt.json", {"command": args.command,
            "generator_sha256": digest_file(__file__), "numeric_helpers_sha256": digest_file(base.__file__),
            "input_sha256": {field: digest_file(p) for field, p in paths.items()},
            "settings_sha256": digest_json(policy()), "model_fits_performed": 0,
            "public_export_allowlist": ["summary.json", "error_summary.json", "analysis_receipt.json"],
            "individual_predictions_or_error_cases_publicly_written": False})
        print(json.dumps({"command": args.command, "completed": True, "model_fits": 0})); return 0
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(str(error), file=sys.stderr); return 1


if __name__ == "__main__":
    raise SystemExit(main())
