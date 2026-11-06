"""Prospective, diagnostic controls for temporal leakage and domain shift.

This module never opens market data or fits a benchmark model.  The registered
runner trains label-shuffled predictors before the common freeze and calls the
analyses only in its single, already-known historical test phase.  A good or bad
negative control is evidence to investigate, never proof of absence/presence of
leakage.  Domain classification is a distribution-shift diagnostic, not HPO.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from hashlib import sha256
import json
import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

TOP10 = ("TSMixer", "TiDE", "TimeMixer", "TimesNet", "PatchTST", "Informer",
         "Autoformer", "iTransformer", "Crossformer", "S-Mamba")
NEURAL = (*TOP10, "GRU-E2", "LSTM-E2", "TCN-E2", "GRU/D7-c03", "LSTM/D7-c02", "TCN/D7-c02")
CONTROL_MODELS = (*NEURAL, "Ridge")
ENDOGENOUS = ("ARIMA", "Persistence", "SeasonalNaive")
CONTROL_SEED = 690101
SHUFFLE_SEEDS = {"outer": 690101, "inner_train": 690118, "inner_validation": 690132}
ADVERSARIAL_SEED = 690201
DELTA = .00025772842664981106
TRAIN_MAX = date(2024, 12, 24)
EVAL_START, EVAL_END = date(2026, 9, 1), date(2026, 10, 30)
REPETITIONS = 2000
BLOCK = 7
ALPHA = .01
CONTROL_STAGE = "E2_label_shuffle_negative_control"


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def _hash(value):
    return sha256((json.dumps(_clean(value), sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()).hexdigest()


def _day(value):
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("Control calendar timestamps require explicit UTC")
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value)
    if len(text) == 10:
        return date.fromisoformat(text)
    return _day(datetime.fromisoformat(text.replace("Z", "+00:00")))


def _stamp(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() != timedelta(0):
        raise ValueError("Actual control timestamps require explicit UTC")
    return result


def _calendar(start, end):
    return [start+timedelta(days=i) for i in range((end-start).days+1)]


def policy():
    return {"protocol": "E2_completion_temporal_leakage_controls_v1",
        "guide": "Entregable2DeepLearningProject.pdf section10.1 p45",
        "label_shuffle_models": list(CONTROL_MODELS), "label_shuffle_models_count": 17,
        "neural_seed": CONTROL_SEED, "Ridge_seed": "deterministic",
        "permutation_seeds": dict(SHUFFLE_SEEDS), "permutation_block_observations": BLOCK,
        "permutation_unit": "consecutive admitted daily observations, including a moved final partial block; gaps reported, never filled or relabeled",
        "permutation_scope": "outer, inner-training and inner-validation labels independently permuted; inner stopping never uses unpermuted validation labels",
        "recipe": "exact primary development-selected hyperparameters; independent negative-control weights, not a competitor or HPO candidate",
        "fit_origin": "2025-01-01", "fit_and_epoch_selection_max": TRAIN_MAX.isoformat(),
        "no_hyperparameter_or_architecture_search": True,
        "freeze_order": "all primary+A+negative-control weights <= global closure < all calibrators closed < the single E2 test scoring phase",
        "test_calendar": [EVAL_START.isoformat(), EVAL_END.isoformat()],
        "test_is_already_known_historically": True,
        "regression_reference": "constant original-scale training-target mean, not classification accuracy or an assumed zero score",
        "gain": "RMSE(training-mean constant)-RMSE(shuffled-label predictor)",
        "diagnostic_family": {"size": 17, "alpha": ALPHA, "adjustment": "Holm across17joint alerts; each intersection-union pvalue=max(practicalgainp,positiveassociationp)",
            "practical_gain_null": "gain<=inheriteddelta", "practical_gain_delta": DELTA,
            "gain_pvalue": "centered paired circular-block bootstrap at the null boundary delta",
            "association_null": "no positive temporal association under block exchangeability",
            "association_pvalue": "block-permute evaluation targets, fixed control forecasts; one-sided Spearman",
            "replicates": REPETITIONS, "block_observations": BLOCK,
            "alert_rule": "joint IUT Holm17 pvalue<alpha and observed gain>delta and observed Spearman>0",
            "nonalert_is_proof_of_no_leakage": False},
        "endogenous_reference_control": {"models": list(ENDOGENOUS),
            "label_shuffle_training_claimed": False,
            "reason": "ARIMA and naive forecasts explicitly consume only past target history; shuffling supervised OHLCV labels does not preserve their input contract",
            "alternative": "existing fixed-parameter causal state/support audits and controlled past-history perturbation curves, evaluated in the same sealed scoring phase",
            "causal_target_history_is_not_suspicious_association": True},
        "adversarial_validation": {"features": "the35flattened seven-day OHLCV cells from the common frozen primary preprocessing; no target, outcome-derived covariate or calendar feature",
            "class_labels": "0=original study training,1=the60historical test inputs",
            "classifier": "fixed balanced logistic regression, scipy L-BFGS-B, L2coefficient.01, unpenalized intercept, maximum200iterations",
            "folds": 5, "heldout": "one contiguous date chunk from each population per fold",
            "purge_calendar_days": 7, "scaler": "mean/SD learned exclusively from each classifier training fold",
            "score": "pooled out-of-fold ROC AUC via Mann-Whitney ranks, tiecorrected; direction predeclared test=1",
            "resampling": "permute domain-label blocks and refit the entire cross-validation procedure for every replicate",
            "replicates": REPETITIONS, "block_observations": BLOCK, "seed": ADVERSARIAL_SEED,
            "alpha": ALPHA, "two_sided_distinguishability": True,
            "CI": "paired circular-block percentile bootstrap of frozen out-of-fold scores and domain labels",
            "handling_if_shift": "retain chronological evaluation, fixed A53 calibration and registered robustness analyses; explain change without reselection or retraining feedback",
            "iid_exchangeability_or_exact_pvalue_claimed": False,
            "no_shift_is_proof_of_no_leakage": False},
        "status_before_execution": "prospectively specified pending public registration and execution",
        "benchmark_or_control_model_fits_performed_by_this_module": 0}


def _block_permutation(n, rng, length=BLOCK, *, ensure_changed=False):
    blocks = [np.arange(i, min(i+length, n)) for i in range(0, n, length)]
    order = rng.permutation(len(blocks))
    # Rule fixed before data: avoid an entirely unchanged negative control.
    if ensure_changed and len(order) > 1 and np.array_equal(order, np.arange(len(order))):
        order = np.roll(order, 1)
    return np.concatenate([blocks[i] for i in order]), order


def shuffled_training_targets(targets, outcome_days, *, seed=CONTROL_SEED, block_days=BLOCK):
    """Return an independently block-permuted training target and its receipt.

    Both scaled and original targets are accepted: a permutation preserves the
    original values and mean exactly.  The runner provides the dates associated
    with the admitted array, not a compressed replacement calendar.
    """
    original = np.asarray(targets)
    if original.ndim != 1 or len(original) < 2*BLOCK or not np.isfinite(original).all():
        raise ValueError("Shuffle controls require at least14 finite scalar training targets")
    if block_days != BLOCK or isinstance(seed, bool) or int(seed) != seed:
        raise ValueError("The registered label permutation uses seven-observation blocks and an explicit integer seed")
    days = [_day(d) for d in outcome_days]
    if len(days) != len(original) or days != sorted(set(days)) or days[-1] > TRAIN_MAX:
        raise ValueError("Control training dates must be unique, ordered and end by2024-12-24")
    indices, blocks = _block_permutation(len(original), np.random.default_rng(int(seed)), ensure_changed=True)
    result = original[indices].copy()
    gaps = sum(max(0, (b-a).days-1) for a, b in zip(days, days[1:]))
    return result, {"method": "block_permutation_of_admitted_training_targets", "seed": int(seed),
        "block_observations": BLOCK, "training_cases": len(days),
        "training_outcome_min": days[0].isoformat(), "training_outcome_max": days[-1].isoformat(),
        "missing_calendar_dates_between_admitted_rows": gaps,
        "block_unit": "consecutive admitted daily observations; no gap filling",
        "partial_last_block_observations": len(days)%BLOCK,
        "permutation_sha256": _hash(indices), "block_order_sha256": _hash(blocks),
        "training_calendar_sha256": _hash([d.isoformat() for d in days]),
        "original_targets_sha256": sha256(np.ascontiguousarray(original).tobytes()).hexdigest(),
        "permuted_targets_sha256": sha256(np.ascontiguousarray(result).tobytes()).hexdigest(),
        "dtype": str(original.dtype), "targets_multiset_preserved": True,
        "nonidentity_index_permutation": not np.array_equal(indices, np.arange(len(days))),
        "local_and_nonstationary_target_dependence_may_remain": True,
        "selection_uses_true_evaluation_labels": False}


def _circular_indices(n, rng, block=BLOCK):
    starts = rng.integers(n, size=math.ceil(n/block))
    return ((starts[:, None]+np.arange(block))%n).ravel()[:n]


def _holm(pvalues):
    p = np.asarray(pvalues, float); order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p)); adjusted[order] = np.minimum(1., np.maximum.accumulate(p[order]*np.arange(len(p), 0, -1)))
    return adjusted


def _spearman(y, prediction):
    a, b = rankdata(y), rankdata(prediction)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def analyze_shuffled_predictions(rows, *, training_mean_by_model, control_receipts, phase_receipt,
                                 repetitions=REPETITIONS, fixture=False):
    """Describe the17 pre-frozen shuffled-label fits on the same60test dates."""
    if repetitions < REPETITIONS and not fixture:
        raise ValueError("Real negative-control inference requires2000replicates")
    if set(control_receipts) != set(CONTROL_MODELS) or set(training_mean_by_model) != set(CONTROL_MODELS):
        raise ValueError("Negative controls must retain all16neuralmodels andRidge")
    closed = _stamp(phase_receipt["control_weights_closed_at_utc"])
    primary_A = _stamp(phase_receipt["all_primary_and_A_weights_closed_at_utc"])
    common = _stamp(phase_receipt["all_weights_including_controls_closed_at_utc"])
    calibrated = _stamp(phase_receipt["calibration_closed_at_utc"])
    scored = _stamp(phase_receipt["evaluation_scoring_started_at_utc"])
    if not (closed <= common and primary_A <= common < calibrated < scored) or phase_receipt["checkpoint_count"] != 17:
        raise ValueError("All17negative controls must close before common weights/calibration/test scoring")
    dates = _calendar(EVAL_START, EVAL_END); grouped = {}; truth = {}
    for row in rows:
        model, d = str(row["model"]), _day(row["outcome_day"])
        expected_seed = "deterministic" if model == "Ridge" else str(CONTROL_SEED)
        if (model not in CONTROL_MODELS or str(row["seed"]) != expected_seed
                or row.get("source_stage") != CONTROL_STAGE or d not in dates or (model, d) in grouped):
            raise ValueError("Use exactly the registered model/seed/test-calendar negative-control rows")
        receipt = control_receipts[model]
        if (str(receipt["seed"]) != expected_seed or receipt["model"] != model
                or _day(receipt["fit_origin"]) != date(2025, 1, 1)
                or _day(receipt["fit_outcome_max"]) > TRAIN_MAX
                or _day(receipt["configuration_selection_outcome_max"]) > TRAIN_MAX):
            raise ValueError("A shuffled-label fit changed its registered seed or training/selection support")
        if (row["checkpoint_sha256"] != receipt["checkpoint_sha256"]
                or _stamp(row["checkpoint_closed_at_utc"]) != _stamp(receipt["checkpoint_closed_at_utc"])
                or _stamp(row["checkpoint_closed_at_utc"]) > closed
                or _stamp(row["scoring_started_at_utc"]) < scored):
            raise ValueError("A shuffled predictor changed its checkpoint or precedes the sealed scoring phase")
        permutations = receipt["label_permutation"]
        expected_parts = ("outer",) if model == "Ridge" else tuple(SHUFFLE_SEEDS)
        if set(permutations) != set(expected_parts):
            raise ValueError("Every fitted/stopping target array requires its own registered permutation receipt")
        for part in expected_parts:
            p = permutations[part]
            if (p["seed"] != SHUFFLE_SEEDS[part] or p["block_observations"] != BLOCK
                    or _day(p["training_outcome_max"]) > TRAIN_MAX
                    or p.get("targets_multiset_preserved") is not True
                    or p.get("nonidentity_index_permutation") is not True):
                raise ValueError("The negative-control labels do not match the registered permutation scheme")
        target, forecast = float(row["y_true"]), float(row["y_pred"])
        if not np.isfinite([target, forecast]).all() or min(target, forecast) < 0:
            raise ValueError("Control truth/forecasts must be nonnegative original-scale sigma7")
        if d in truth and target != truth[d]:
            raise ValueError("Negative controls must share the identical daily ground truth")
        truth[d] = target; grouped[model, d] = forecast
    if len(grouped) != len(CONTROL_MODELS)*len(dates):
        raise ValueError("Retain all17models and all60test dates; seed×date is not a case count")
    target = np.array([truth[d] for d in dates]); results = {}; joint_pvalues = []
    rng = np.random.default_rng(CONTROL_SEED+700)
    boot = np.array([_circular_indices(len(dates), rng) for _ in range(repetitions)])
    shuffled = np.array([_block_permutation(len(dates), rng)[0] for _ in range(repetitions)])
    for model in CONTROL_MODELS:
        prediction = np.array([grouped[model, d] for d in dates])
        constant = float(training_mean_by_model[model])
        if not math.isfinite(constant) or constant < 0:
            raise ValueError("The frozen training-mean reference must use original-scale finite sigma7")
        baseline_loss, control_loss = (constant-target)**2, (prediction-target)**2
        baseline_rmse, control_rmse = np.sqrt(baseline_loss.mean()), np.sqrt(control_loss.mean())
        gain = float(baseline_rmse-control_rmse)
        boot_gain = np.sqrt(baseline_loss[boot].mean(1))-np.sqrt(control_loss[boot].mean(1))
        p_gain = float((1+((boot_gain-gain) >= gain-DELTA).sum())/(repetitions+1))
        association = _spearman(target, prediction)
        if association is None:
            p_association = 1.
        else:
            rank_y, rank_prediction = rankdata(target), rankdata(prediction)
            rank_y -= rank_y.mean(); rank_prediction -= rank_prediction.mean()
            null = (rank_y[shuffled]@rank_prediction)/np.sqrt((rank_y@rank_y)*(rank_prediction@rank_prediction))
            p_association = float((1+(null >= association).sum())/(repetitions+1))
        joint_pvalues.append(max(p_gain, p_association))
        results[model] = {"test_dates": len(dates), "independent_negative_control_fits": 1,
            "RMSE_training_mean_reference": float(baseline_rmse), "RMSE_shuffled_predictor": float(control_rmse),
            "RMSE_gain_over_training_mean": gain, "gain_ci95_nominal_temporal": np.quantile(boot_gain, [.025, .975]).tolist(),
            "gain_delta": DELTA, "p_gain_gt_delta_temporal": p_gain,
            "positive_Spearman_association": association, "p_positive_association_block_permutation": p_association,
            "constant_forecast_association_not_estimable": association is None}
    adjusted = _holm(joint_pvalues)
    for i, model in enumerate(CONTROL_MODELS):
        r = results[model]; adjusted_joint = float(adjusted[i])
        alert = adjusted_joint < ALPHA and r["RMSE_gain_over_training_mean"] > DELTA and (r["positive_Spearman_association"] or 0) > 0
        r.update(p_joint_IUT=joint_pvalues[i], p_joint_IUT_Holm17=adjusted_joint,
                 diagnostic_alert_requires_investigation=bool(alert),
                 conclusion="investigate unexpected predictive association; this result alone does not prove leakage" if alert else
                            "no registered combined alert; absence of leakage is not established")
    return _clean({"models": results, "diagnostic_family_size": 17, "component_diagnostics_count": 34, "alpha": ALPHA,
        "bootstrap_and_permutation_replicates": repetitions, "block_observations": BLOCK,
        "temporal_block_exchangeability_assumed_not_guaranteed": True,
        "no_false_pass_by_low_RMSE_or_nonrejection": True, "confirmatory_or_new_unseen_test": False,
        "HPO_or_model_selection_feedback_allowed": False, "endogenous_control": policy()["endogenous_reference_control"],
        "synthetic_fixture_only": bool(fixture), "private_predictions_exported": False})


def mann_whitney_auc(labels, scores):
    y, score = np.asarray(labels), np.asarray(scores, float)
    if y.ndim != 1 or score.shape != y.shape or not np.isfinite(score).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("AUC requires aligned finite scores and explicit binary domain labels")
    n1, n0 = int(y.sum()), int((y == 0).sum())
    if min(n1, n0) == 0:
        return None
    return float((rankdata(score)[y == 1].sum()-n1*(n1+1)/2)/(n1*n0))


def _adversarial_folds(train_days, test_days):
    populations = [list(map(_day, train_days)), list(map(_day, test_days))]
    if any(len(p) < 25 or p != sorted(set(p)) for p in populations):
        raise ValueError("Adversarial CV requires at least25distinct ordered dates in each population")
    days = populations[0]+populations[1]; offset = len(populations[0]); allidx = np.arange(len(days))
    chunks = [np.array_split(np.arange(len(populations[0])), 5),
              np.array_split(np.arange(offset, len(days)), 5)]
    folds = []
    for f in range(5):
        heldout = np.concatenate([parts[f] for parts in chunks])
        forbidden = {d+timedelta(days=lag) for i in heldout for d in [days[i]] for lag in range(-BLOCK, BLOCK+1)}
        training = np.array([i for i in allidx if i not in heldout and days[i] not in forbidden], dtype=int)
        if len(training) < 20:
            raise ValueError("Purged adversarial training fold is too small")
        folds.append((training, heldout))
    return folds


def _logistic_fit_predict(train_x, labels, heldout_x):
    labels = np.asarray(labels, float)
    if len(np.unique(labels)) != 2:
        return None
    mean, scale = train_x.mean(0), train_x.std(0)
    scale[scale < 1e-12] = 1.
    train, heldout = (train_x-mean)/scale, (heldout_x-mean)/scale
    class_counts = np.bincount(labels.astype(int), minlength=2)
    weights = .5/class_counts[labels.astype(int)]
    def objective(theta):
        linear = theta[0]+train@theta[1:]
        loss = np.sum(weights*(np.logaddexp(0., linear)-labels*linear))+.005*(theta[1:]@theta[1:])
        residual = weights*(expit(linear)-labels)
        gradient = np.r_[residual.sum(), train.T@residual+.01*theta[1:]]
        return loss, gradient
    result = minimize(objective, np.zeros(train.shape[1]+1), method="L-BFGS-B", jac=True,
                      options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-7})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError("Adversarial logistic fit did not converge; retain the failure, do not silently replace it")
    return expit(result.x[0]+heldout@result.x[1:])


def _cross_validated_domain_scores(x, labels, folds):
    out = np.full(len(labels), np.nan)
    for training, heldout in folds:
        scores = _logistic_fit_predict(x[training], labels[training], x[heldout])
        if scores is None:
            return None
        out[heldout] = scores
    if not np.isfinite(out).all():
        raise ValueError("The adversarial OOF calendar must be complete")
    return out


def adversarial_validation(train_x, train_days, test_x, test_days, *, repetitions=REPETITIONS,
                           seed=ADVERSARIAL_SEED, fixture=False):
    """Purged temporal CV domain classification, with full refits under the null.

    Study-test features may be supplied only after the registered common freeze.
    No study-test target is accepted or read.  The runner enforces that phase gate.
    """
    if repetitions < REPETITIONS and not fixture:
        raise ValueError("Real adversarial validation requires2000block permutations")
    td, ed = list(map(_day, train_days)), list(map(_day, test_days))
    if not td or td[-1] > TRAIN_MAX or ed != _calendar(EVAL_START, EVAL_END):
        raise ValueError("Adversarial populations must preserve the registered training and60test calendars")
    tx, ex = np.asarray(train_x, float), np.asarray(test_x, float)
    if tx.shape[0] != len(td) or ex.shape[0] != len(ed):
        raise ValueError("Adversarial features must align to their actual admitted dates")
    tx, ex = tx.reshape(len(td), -1), ex.reshape(len(ed), -1)
    if tx.shape[1] != 35 or ex.shape[1] != 35 or not np.isfinite(tx).all() or not np.isfinite(ex).all():
        raise ValueError("Use exactly35finite OHLCV window cells; targets and calendar covariates are forbidden")
    x = np.concatenate([tx, ex]); labels = np.r_[np.zeros(len(td), int), np.ones(len(ed), int)]
    folds = _adversarial_folds(td, ed)
    observed_scores = _cross_validated_domain_scores(x, labels, folds)
    if observed_scores is None:
        raise ValueError("Both domains must be represented in every purged classifier training fold")
    observed = mann_whitney_auc(labels, observed_scores)
    rng = np.random.default_rng(int(seed)); null = []; failed_null = 0
    for _ in range(repetitions):
        permutation, _ = _block_permutation(len(labels), rng)
        permuted = labels[permutation]
        scores = _cross_validated_domain_scores(x, permuted, folds)
        value = None if scores is None else mann_whitney_auc(permuted, scores)
        if value is None:
            failed_null += 1
        else:
            null.append(value)
    estimable = len(null) >= .95*repetitions
    p = (float((1+np.sum(np.abs(np.asarray(null)-.5) >= abs(observed-.5)))/(len(null)+1)) if estimable else None)
    boot = []
    # Resample each population independently, retaining its original size.
    for _ in range(repetitions):
        indices = np.r_[_circular_indices(len(td), rng), len(td)+_circular_indices(len(ed), rng)]
        boot.append(mann_whitney_auc(labels[indices], observed_scores[indices]))
    distinguishable = None if p is None else bool(p < ALPHA)
    return _clean({"method": policy()["adversarial_validation"], "training_cases": len(td), "test_cases": len(ed),
        "classifier_count_observed": 5, "maximum_classifier_refits_under_null_scheduled": 5*repetitions,
        "OOF_AUC_Mann_Whitney": observed, "AUC_ci95_nominal_temporal_conditional_on_OOF_scores": np.quantile(boot, [.025, .975]).tolist(),
        "p_two_sided_block_permutation_refit": p, "alpha": ALPHA,
        "permutation_replicates_attempted": repetitions, "permutation_replicates_estimable": len(null),
        "permutation_replicates_without_two_classes": failed_null,
        "distinguishable_from_AUC_half": distinguishable,
        "status": "not_estimable" if p is None else "distribution_shift_to_explain_and_treat" if distinguishable else "no_detected_shift_under_this_diagnostic",
        "fold_case_counts": [{"training": len(tr), "heldout": len(va)} for tr, va in folds],
        "purge_calendar_days": BLOCK, "HPO_or_model_selection_feedback_allowed": False,
        "label_or_outcome_features_used": False,
        "exact_iid_inference_or_absence_of_leakage_claimed": False,
        "exchangeability_limit": "domain labels arise from distant chronological periods; block exchangeability is a diagnostic approximation, not a randomized study",
        "handling": policy()["adversarial_validation"]["handling_if_shift"],
        "private_features_or_domain_scores_exported": False, "synthetic_fixture_only": bool(fixture)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan",), nargs="?", default="plan")
    parser.parse_args(argv)
    print(json.dumps(policy(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
