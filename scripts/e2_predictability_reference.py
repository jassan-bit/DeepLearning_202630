"""A structural, assumption-bound reference for sigma7 predictability.

D7 contains seven closes and therefore six of the next sigma7's seven returns.
Integrating the remaining return over an empirical TRAINING innovation law gives
an estimated conditional risk under that law. It is not a universal Bayes bound:
conditional heteroscedasticity, regime change, and information in other covariates
can invalidate the law. No function reads market files or trains a neural model.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from hashlib import sha256
import math

import numpy as np

from e2_predictive_uncertainty import day


TRAIN_MIN = date(2023, 1, 1)
TRAIN_MAX = date(2023, 12, 24)
GAMMA_DENOMINATOR_MIN = 1e-14


def _dates(values):
    dates = [day(x) for x in values]
    if not dates or dates != [dates[0] + timedelta(days=i) for i in range(len(dates))]:
        raise ValueError("Retain an ordered complete daily calendar, including missing positions")
    return dates


def permutation_entropy(values, dimension=3, delay=1):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or dimension < 2 or delay < 1:
        raise ValueError("One dimensional calendar data and valid ordinal dimensions required")
    counts, excluded, tied = {}, 0, 0
    for start in range(len(values) - (dimension-1)*delay):
        window = values[start:start+dimension*delay:delay]
        if not np.isfinite(window).all():
            excluded += 1; continue
        tied += len(np.unique(window)) != dimension
        pattern = tuple(np.argsort(window, kind="stable"))
        counts[pattern] = counts.get(pattern, 0) + 1
    total = sum(counts.values())
    if not total:
        return {"status": "not_estimable", "normalized_entropy": None, "patterns": 0}
    probabilities = np.asarray(list(counts.values()), dtype=float) / total
    entropy = float(-(probabilities*np.log(probabilities)).sum() / math.lgamma(dimension+1))
    return {"status": "estimated", "normalized_entropy": entropy, "dimension": dimension,
            "delay_calendar_days": delay, "patterns": total, "missing_windows_excluded": excluded,
            "tied_windows": int(tied), "tie_rule": "stable order, explicitly counted",
            "meaning": "Ordinal complexity diagnostic; not converted into accuracy or a proven risk bound"}


def _block_counts(length, block, repetitions, rng):
    if not 1 <= block <= length:
        raise ValueError("Block length must fit the retained calendar")
    result = np.zeros((repetitions, length), dtype=np.float64)
    starts = rng.integers(0, length, (repetitions, math.ceil(length/block)))
    indices = ((starts[..., None]+np.arange(block)) % length).reshape(repetitions, -1)[:, :length]
    np.add.at(result, (np.repeat(np.arange(repetitions), length), indices.ravel()), 1.)
    return result


@dataclass(frozen=True)
class InnovationReference:
    daily_innovations: np.ndarray
    dates: tuple[date, ...]

    @classmethod
    def from_training_daily(cls, dates, closes):
        days = _dates(dates)
        values = np.asarray(closes, dtype=np.float64)
        if values.shape != (len(days),) or days[0] < TRAIN_MIN or days[-1] > TRAIN_MAX:
            raise ValueError("Innovation law uses daily training observations within 2023-01-01--2023-12-24")
        finite = np.isfinite(values)
        if np.isinf(values).any() or (values[finite] <= 0).any():
            raise ValueError("Training closes must be positive or explicitly missing")
        innovations = np.diff(np.log(values), prepend=np.nan)
        if np.isfinite(innovations).sum() < 30:
            raise ValueError("At least thirty finite training innovations are required")
        innovations.setflags(write=False)
        return cls(innovations, tuple(days))

    def specification(self):
        finite = np.isfinite(self.daily_innovations)
        return {"method": "six known returns plus one empirical training innovation",
                "training_start": self.dates[0].isoformat(), "training_end": self.dates[-1].isoformat(),
                "training_calendar_positions": len(self.dates), "finite_innovations": int(finite.sum()),
                "training_innovation_bytes_sha256": sha256(self.daily_innovations.tobytes()).hexdigest(),
                "target": "population std of seven log returns, daily/unannualized",
                "conditioning": "six returns derivable from the same D7 closes",
                "assumptions": ["Future innovation independent of observed window under the proxy law",
                                "Empirical 2023 innovation distribution transfers to evaluation context",
                                "Other OHLCV/history information does not improve the innovation law under this proxy"],
                "universal_or_empirically_proven_Bayes_bound": False,
                "permutation_entropy": permutation_entropy(self.daily_innovations)}

    def volatility_matrix(self, window_closes):
        windows = np.asarray(window_closes, dtype=np.float64)
        if windows.ndim != 2 or windows.shape[1] != 7 or not len(windows):
            raise ValueError("Each context must contain the same seven observed D7 closes")
        if not np.isfinite(windows).all() or (windows <= 0).any():
            raise ValueError("Use the common eligible finite D7 contexts, without fabricated closes")
        past = np.diff(np.log(windows), axis=1)
        future = self.daily_innovations[np.isfinite(self.daily_innovations)]
        first, second = past.sum(axis=1)[:, None], (past**2).sum(axis=1)[:, None]
        variance = (second + future[None, :]**2)/7 - ((first+future[None, :])/7)**2
        if variance.min() < -1e-12:
            raise ValueError("Negative conditional volatility variance exceeds roundoff")
        return np.sqrt(np.maximum(variance, 0.))

    def predict(self, window_closes):
        sigma = self.volatility_matrix(window_closes)
        return {"conditional_mean_sigma7": sigma.mean(axis=1),
                "conditional_variance_sigma7": sigma.var(axis=1, ddof=0),
                "assumption_bound_reference_RMSE": float(np.sqrt(sigma.var(axis=1, ddof=0).mean()))}


def _predictions(value, n):
    result = np.asarray(value, dtype=float)
    if result.ndim == 1:
        result = result[None, :]
    if result.ndim != 2 or result.shape[1] != n or not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("Predictor trajectories must retain the common original-scale calendar")
    return result


def predictability_summary(reference, context_dates, window_closes, y_true,
                           baseline_predictions, best_predictions, *,
                           baseline_name, best_name, best_selection_scope,
                           seed_ids=None, repetitions=2000, block_length=7, seed=202611066):
    """Joint calendar/law bootstrap; gamma is reported without clipping.

    Intervals condition on supplied trained predictors. They do not capture every
    source of model selection or certify the empirical law's transfer assumption.
    A violated reference or unstable gamma denominator is explicitly flagged.
    """
    dates = _dates(context_dates); truth = np.asarray(y_true, dtype=float); n = len(dates)
    if truth.shape != (n,) or not np.isfinite(truth).all() or (truth < 0).any():
        raise ValueError("Finite nonnegative truth required on the whole retained evaluation calendar")
    if best_selection_scope not in {"frozen_development_selection", "observed_test_minimum_descriptive"}:
        raise ValueError("Declare whether the reported best was selected before or after test results")
    if repetitions < 2000 or not isinstance(repetitions, int):
        raise ValueError("At least2000 registered temporal resamples are required")
    baseline, best = _predictions(baseline_predictions, n), _predictions(best_predictions, n)
    r = max(len(baseline), len(best))
    if len(baseline) not in (1, r) or len(best) not in (1, r):
        raise ValueError("Compare common paired seed registries; deterministic references retain one trajectory")
    if r > 1 and (seed_ids is None or len(seed_ids) != r or len(set(map(str, seed_ids))) != r):
        raise ValueError("Explicit unique paired seed identities are required for training-randomness resampling")
    sigma = reference.volatility_matrix(window_closes)
    if sigma.shape[0] != n:
        raise ValueError("Dates and D7 context count differ")
    b_rmse = float(np.sqrt(((baseline-truth)**2).mean(axis=1)).mean())
    m_rmse = float(np.sqrt(((best-truth)**2).mean(axis=1)).mean())
    ref = reference.predict(window_closes); floor = ref["assumption_bound_reference_RMSE"]
    denominator = b_rmse-floor
    gamma = (m_rmse-floor)/denominator if denominator > GAMMA_DENOMINATOR_MIN else None
    rng = np.random.default_rng(seed)
    law_counts = _block_counts(len(reference.dates), block_length, repetitions, rng)
    law_counts = law_counts[:, np.isfinite(reference.daily_innovations)]
    law_mass = law_counts.sum(axis=1)
    valid_law = law_mass > 0
    law_counts = np.divide(law_counts, law_mass[:, None], out=np.zeros_like(law_counts),
                           where=valid_law[:, None])
    calendar_counts = _block_counts(n, block_length, repetitions, rng) / n
    conditional_mean = law_counts @ sigma.T
    conditional_second = law_counts @ (sigma**2).T
    conditional_var = np.maximum(0., conditional_second-conditional_mean**2)
    floors = np.sqrt((calendar_counts*conditional_var).sum(axis=1))
    floors[~valid_law] = np.nan
    br_by_seed = np.sqrt(calendar_counts @ ((baseline-truth)**2).T)
    mr_by_seed = np.sqrt(calendar_counts @ ((best-truth)**2).T)
    seed_indices = rng.integers(0, r, (repetitions, r))
    br = (np.take_along_axis(br_by_seed, seed_indices, axis=1).mean(axis=1)
          if len(baseline) > 1 else br_by_seed[:, 0])
    mr = (np.take_along_axis(mr_by_seed, seed_indices, axis=1).mean(axis=1)
          if len(best) > 1 else mr_by_seed[:, 0])
    denominators = br-floors
    valid = valid_law & np.isfinite(denominators) & (denominators > GAMMA_DENOMINATOR_MIN)
    gammas = (mr[valid]-floors[valid])/denominators[valid]
    enough = gamma is not None and valid.mean() >= .95
    risk_enough = valid_law.mean() >= .95
    observed_proxy_error = float(np.sqrt(((ref["conditional_mean_sigma7"]-truth)**2).mean()))
    return {"reference": reference.specification(), "common_context_origins": n,
            "best_model": best_name, "baseline": baseline_name,
            "best_selection_scope": best_selection_scope,
            "model_selection_uncertainty_included": False,
            "selection_limit": "Intervals condition on the supplied chosen model; an observed test minimum remains descriptive",
            "best_mean_seed_RMSE": m_rmse, "baseline_mean_seed_RMSE": b_rmse,
            "estimated_conditional_risk_RMSE_under_training_law": floor,
            "reference_point_forecaster_observed_RMSE": observed_proxy_error,
            "risk_reference_ci95_nominal": (np.quantile(floors[valid_law], [.025, .975]).tolist()
                                             if risk_enough and valid_law.any() else None),
            "risk_reference_estimable_draws": int(valid_law.sum()),
            "risk_reference_unestimable_law_draws": int((~valid_law).sum()),
            "risk_reference_interval_status": ("estimated" if risk_enough and valid_law.any()
                                                else "not_estimable_insufficient_valid_innovation_draws"),
            "gamma": gamma,
            "gamma_point_denominator": denominator,
            "gamma_minimum_positive_denominator": GAMMA_DENOMINATOR_MIN,
            "gamma_point_status": "estimated" if gamma is not None else "not_estimable_unstable_denominator",
            "gamma_definition_for_lower_is_better_RMSE": "(RMSE_best-RMSE_reference)/(RMSE_baseline-RMSE_reference)",
            "gamma_ci95_nominal": np.quantile(gammas, [.025, .975]).tolist() if enough and len(gammas) else None,
            "gamma_estimable_draws": int(valid.sum()), "gamma_unstable_denominator_draws": int((~valid).sum()),
            "gamma_invalid_law_draws": int((~valid_law).sum()),
            "gamma_nonpositive_or_small_denominator_draws": int((valid_law & ~valid).sum()),
            "gamma_interval_status": "estimated" if enough and len(gammas) else "not_estimable_unstable_denominator_or_innovation_draws",
            "gamma_clipped_to_unit_interval": False,
            "reference_exceeded_by_observed_best": bool(m_rmse < floor),
            "universal_performance_ceiling_established": False,
            "interpretation": "Assumption-bound predictability reference only; negative gamma or best below reference challenges proxy assumptions, not mathematical optimality",
            "bootstrap": {"draws": repetitions, "seed": seed, "block_calendar_days": block_length,
                          "joint_training_law_and_evaluation_calendar_resampling": True,
                          "fitted_predictor_seed_variability_resampled": r > 1,
                          "seed_resampling_paired_between_randomized_predictors": True,
                          "minimum_estimable_draw_fraction_for_nominal_CI": .95,
                          "zero_innovation_mass_draws_retained_as_unestimable": True,
                          "unestimable_draws_not_silently_replaced_or_regenerated": True,
                          "intervals_nominal_not_simultaneous": True},
            "new_neural_fits": 0, "individual_forecasts_or_prices_exported": False}
