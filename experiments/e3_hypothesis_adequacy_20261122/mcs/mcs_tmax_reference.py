"""Supplementary Tmax model confidence set on a common historical calendar.

This is the Hansen--Lunde--Nason sequential maximum-statistic construction.
Its loss is the per-date mean of individual-seed squared errors. That loss is
different from the primary arithmetic mean of individual-seed RMSEs. This
auxiliary calculation cannot establish a practical RMSE-margin gain or R3.

No source files are loaded and no fits are performed by this module. The
caller owns input verification and must record the analysis plan first.
"""
from __future__ import annotations

import math

import numpy as np


ORIGINAL_MODEL = "CNN-LSTM-atencion-aditiva E3"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _settings(plan):
    settings = plan.get("mcs")
    _require(isinstance(settings, dict), "An explicit MCS analysis plan is required")
    required = {"alpha", "block_days", "repetitions", "seed"}
    _require(required <= set(settings), "MCS alpha, block, draws and RNG seed must be fixed")
    alpha, block, draws, seed = (settings[k] for k in ("alpha", "block_days", "repetitions", "seed"))
    _require(type(alpha) in (int, float) and math.isfinite(alpha) and 0 < alpha < 1,
             "MCS alpha must lie between zero and one")
    _require(type(block) is int and block >= 1, "MCS block length must be a positive integer")
    _require(type(draws) is int and draws >= 99, "At least 99 bootstrap draws are required")
    _require(type(seed) is int and seed >= 0, "An explicit nonnegative integer RNG seed is required")
    return float(alpha), block, draws, seed


def _loss_panel(panel):
    predictions = panel.get("predictions")
    _require(isinstance(predictions, dict) and len(predictions) >= 2,
             "MCS requires an unfiltered mapping of at least two models")
    _require(all(isinstance(name, str) and name for name in predictions), "Invalid model names")
    candidate = panel.get("original_model", ORIGINAL_MODEL)
    _require(candidate in predictions, "The original E3 model must appear in the initial MCS roster")
    target = np.asarray(panel.get("targets"), dtype=np.float64)
    _require(target.ndim == 1 and len(target) >= 4 and np.isfinite(target).all(),
             "A common finite one-dimensional target calendar is required")
    if "mask" in panel:
        mask = np.asarray(panel["mask"])
        _require(mask.shape == target.shape and mask.dtype.kind == "b" and bool(mask.all()),
                 "MCS does not compress a calendar with excluded or missing dates")
    if "dates" in panel:
        _require(len(panel["dates"]) == len(target) and len(set(panel["dates"])) == len(target),
                 "MCS calendar dates must be unique and match the targets")
    models = sorted(predictions)  # Stable ties and floating-point summation order.
    losses = []
    for name in models:
        prediction = np.asarray(predictions[name], dtype=np.float64)
        _require(prediction.ndim == 2 and prediction.shape[0] >= 1
                 and prediction.shape[1] == len(target) and np.isfinite(prediction).all(),
                 "Every model needs finite R by T predictions on the common calendar")
        with np.errstate(over="ignore", invalid="ignore"):
            individual_squared = np.square(prediction - target[None, :])
            loss = individual_squared.mean(axis=0)
        _require(np.isfinite(loss).all(), "Squared-error loss is not finite")
        losses.append(loss)
    return models, np.column_stack(losses), candidate


def _bootstrap_means(losses, block, draws, seed):
    """Common circular blocks, sampled once and reused for every elimination."""
    n, m = losses.shape
    rng = np.random.Generator(np.random.PCG64(seed))
    means = np.empty((draws, m), dtype=np.float64)
    offsets = np.arange(block, dtype=np.int64)
    for first in range(0, draws, 256):
        last = min(first + 256, draws)
        starts = rng.integers(0, n, size=(last - first, math.ceil(n / block)))
        indices = ((starts[:, :, None] + offsets[None, None, :]) % n).reshape(last - first, -1)[:, :n]
        counts = np.zeros((last - first, n), dtype=np.float64)
        np.add.at(counts, (np.arange(last - first)[:, None], indices), 1.0)
        means[first:last] = counts @ losses / n
    return means


def run_mcs(panel, plan):
    """Return the level-1-alpha Tmax MCS and the sequential elimination path.

    For the active set S, d_i,t = loss_i,t - mean_j-in-S(loss_j,t).
    Its observed t statistic uses the bootstrap SE of mean_t(d_i,t).
    Each null bootstrap statistic centers its relative mean by the observed
    relative mean. The model with the largest standardized relative loss is
    eliminated only when the stepwise maximum p-value remains below alpha.

    Exactly tied zero-variance contrasts contribute zero. A nonzero contrast
    with zero estimated relative variance makes the studentized procedure
    unidentifiable here: retain the current set, disclose that limitation and
    report no invented p-value for that step.
    """
    alpha, block, draws, seed = _settings(plan)
    models, losses, candidate = _loss_panel(panel)
    n = len(losses)
    _require(block <= n, "MCS block length exceeds the calendar")
    # Contrast invariance permits subtracting a common loss series. Doing it
    # before resampling makes identical models have exact zero differences.
    anchored = losses - losses[:, :1]
    observed_means = anchored.mean(axis=0)
    bootstrap = _bootstrap_means(anchored, block, draws, seed)
    active = list(range(len(models)))
    steps, excluded = [], {}
    running_p = 0.0
    status = "completed"
    stop_reason = "one_model_remaining"
    while len(active) > 1:
        # Re-anchor to a currently active model before pool centering. The
        # initial anchor may have been eliminated; subtracting its large common
        # mean again can leave tiny false effects among identical survivors.
        mean = observed_means[active] - observed_means[active[0]]
        effect = mean - mean.mean()
        active_bootstrap = bootstrap[:, active] - bootstrap[:, active[0], None]
        bootstrap_relative = active_bootstrap - active_bootstrap.mean(axis=1)[:, None]
        centered = bootstrap_relative - effect[None, :]
        standard_error = centered.std(axis=0, ddof=1)
        zero_variance = standard_error == 0
        step = {
            "step": len(steps) + 1,
            "active_models": [models[i] for i in active],
            "relative_mean_loss": {models[i]: float(effect[k]) for k, i in enumerate(active)},
            "bootstrap_standard_error": {models[i]: float(standard_error[k]) for k, i in enumerate(active)},
        }
        if np.any(zero_variance & (effect != 0)):
            status = "not_estimable_zero_relative_variance"
            stop_reason = "Nonzero relative mean loss has zero bootstrap standard error"
            step.update(status=status, Tmax=None, p=None, p_stepwise=None, eliminated_model=None)
            steps.append(step)
            break
        observed_t = np.zeros(len(active), dtype=np.float64)
        null_t = np.zeros_like(centered)
        np.divide(effect, standard_error, out=observed_t, where=~zero_variance)
        np.divide(centered, standard_error[None, :], out=null_t, where=~zero_variance[None, :])
        statistic = float(observed_t.max())
        null_maximum = null_t.max(axis=1)
        p = float((1 + np.count_nonzero(null_maximum >= statistic)) / (draws + 1))
        running_p = max(running_p, p)
        step.update(status="estimated", Tmax=statistic, p=p, p_stepwise=running_p, eliminated_model=None)
        if running_p >= alpha:
            stop_reason = "Equal predictive ability not rejected for the remaining set"
            steps.append(step)
            break
        removed_position = int(np.argmax(observed_t))
        removed = active.pop(removed_position)
        excluded[models[removed]] = running_p
        step["eliminated_model"] = models[removed]
        steps.append(step)
    included = [models[i] for i in active]
    return {
        "version": "e3-supplementary-mcs-tmax-v1",
        "status": status,
        "stop_reason": stop_reason,
        "initial_models": models,
        "original_model": candidate,
        "candidate_in_initial_set": True,
        "candidate_included": candidate in included,
        "included_models": included,
        "excluded_pvalues": excluded,
        "steps": steps,
        "alpha": alpha,
        "level": 1.0 - alpha,
        "block_days": block,
        "repetitions": draws,
        "seed": seed,
        "rng": "numpy.Generator(PCG64)",
        "calendar_dates": n,
        "mean_loss": {name: float(losses[:, i].mean()) for i, name in enumerate(models)},
        "loss": "per-date arithmetic mean of individual-seed squared errors",
        "uncertainty_scope": "calendar conditional on supplied frozen fitted trajectories; no seed resampling",
        "separate_auxiliary_family": True,
        "primary_Holm_family_member": False,
        "does_not_test_RMSE_margin": True,
        "membership_does_not_prove_equivalence": True,
        "R3_verified": False,
        "assumptions": [
            "Circular-block bootstrap approximation requires suitable weak dependence and temporal stability",
            "Fixed fitted trajectories; architecture selection and training-population uncertainty are not resampled",
            "Known historical test; this calculation does not restore a blinded confirmatory design",
        ],
        "reference": "Hansen, Lunde and Nason (2011), The Model Confidence Set, Econometrica 79(2):453-497",
    }
