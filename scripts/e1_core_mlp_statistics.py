"""Planning and aggregate-only analysis for the new LSTM/MLP/GRU extension.

No function trains, reads a checkpoint, opens market data or writes a file.
The paired-t planning concerns training randomness conditional on a fixed
60-day calendar. It does not assert 80% power for temporal generalization.
The secondary time analysis resamples the same dates jointly across models
and seeds; its estimand averages per-seed RMSE, never ensemble predictions.
"""
from __future__ import annotations

import hashlib
import itertools
import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import bds

from gru_lstm_tcn_diagnostics import diebold_mariano_hln, holm_with_undefined

MODELS = ("GRU", "LSTM", "MLP")
PAIRS = tuple(itertools.combinations(MODELS, 2))
METRICS = ("rmse", "mae", "mse", "r2", "mape", "mase")
PERSISTENCE = "Persistence"
SCOPE = ("New coherent historical exploratory extension on an already observed reserve; "
         "seed uncertainty is conditional on its fixed calendar and frozen recipes, "
         "not uncertainty over markets or a fresh confirmatory holdout.")
REFERENCE_SOURCE_SHA256 = {
    "planned_study_statistics.py": "703492f299649ac5f87a2c37e30f0ae5fc4253595fba0fce36ec1b1b817fcdf2",
    "gru_lstm_tcn_diagnostics.py": "8cf66f3c169d90a677ae1fe16d2b77636d19c42bfa718d0b1beec77980d49ec8",
}


def verify_reference_sources():
    """Check immutable method sources, without fitting or reading market data."""
    root = Path(__file__).resolve().parent
    observed = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in REFERENCE_SOURCE_SHA256}
    if observed != REFERENCE_SOURCE_SHA256:
        raise ValueError("An immutable reference statistics source has changed")
    return observed


def _coherent_origin(frame, design, *, plan=None, pilot=False):
    """Require one new execution; a matching label alone never proves provenance."""
    required = {"study_id", "execution_id"}
    if not required <= set(frame):
        raise ValueError("Every row requires study_id and execution_id provenance")
    identity = {}
    for name in ("study_id", "execution_id"):
        values = frame[name]
        if (values.isna().any() or values.astype(str).str.strip().eq("").any()
                or values.nunique() != 1):
            raise ValueError("Mixed or missing execution provenance: " + name)
        identity[name] = str(values.iloc[0])
    expected_study = design.get("study_id")
    if not expected_study or identity["study_id"] != expected_study:
        raise ValueError("Metrics must belong to the registered new study")
    expected_execution = design.get("execution_id", (plan or {}).get("execution_id"))
    if expected_execution and identity["execution_id"] != expected_execution:
        raise ValueError("Metrics do not belong to the registered execution")
    for name in ("data_manifest_sha256", "source_manifest_sha256", "dataset_sha256"):
        if name in frame and (frame[name].isna().any() or frame[name].nunique() != 1):
            raise ValueError("Mixed scientific sources: " + name)
    if pilot:
        for name in ("evaluation_days", "n_dates"):
            if name in frame and not frame[name].eq(60).all():
                raise ValueError("Pilot metrics must use the same registered 60-day cohort")
        if "phase" in frame and not frame.phase.eq("pilot").all():
            raise ValueError("Only new pilot metrics may determine sample size")
        settings = design.get("pilot", {})
        for name, expected in (("fit_origin_day_utc", settings.get("origin", "2024-11-01")),
                               ("evaluation_start", settings.get("outcome_start", "2024-11-01")),
                               ("evaluation_end", settings.get("outcome_end", "2024-12-30"))):
            if name in frame and not pd.to_datetime(frame[name], utc=True).dt.strftime("%Y-%m-%d").eq(expected).all():
                raise ValueError("Pilot metrics mix incompatible cohorts: " + name)
    return identity


class PlanningLimitError(ValueError):
    """The requested power is not attainable within the declared numerical bound."""


def _settings(design):
    s = design.get("statistics", {})
    result = {"delta": float(s.get("delta", .00025772842664981106)),
              "alpha": float(s.get("alpha", .05)), "power": float(s.get("power", .8)),
              "min_runs": s.get("min_runs", 10), "max_runs": s.get("max_runs", 10000),
              "confidence": float(s.get("sd_upper_simultaneous_confidence", .8)),
              "bootstrap_draws": s.get("bootstrap_draws", 4999),
              "bootstrap_seed": s.get("bootstrap_seed", 20261107),
              "temporal_bootstrap_seed": s.get("temporal_bootstrap_seed", 20261108),
              "final_runs_floor": s.get("final_runs_floor", design.get("definitive", {}).get("minimum_runs", 465))}
    for key in ("min_runs", "max_runs", "bootstrap_draws", "bootstrap_seed", "temporal_bootstrap_seed", "final_runs_floor"):
        if isinstance(result[key], bool) or int(result[key]) != result[key]:
            raise ValueError("Integer setting required: " + key)
        result[key] = int(result[key])
    if (not np.isfinite(result["delta"]) or result["delta"] <= 0
            or not 0 < result["alpha"] < result["power"] < 1
            or not 0 < result["confidence"] < 1
            or not 2 <= result["min_runs"] <= result["max_runs"]
            or not 465 <= result["final_runs_floor"] <= result["max_runs"]
            or result["bootstrap_draws"] < 100):
        raise ValueError("Invalid pre-specified statistics settings")
    if (s.get("family_size",len(PAIRS)) != len(PAIRS)
            or tuple(map(tuple,s.get("primary_pairs",PAIRS))) != PAIRS):
        raise ValueError("The registered primary family must be exactly the three specified pairs")
    return result


def difference_power(n, difference, sd, alpha):
    """Exact noncentral-t reference under independent normal paired differences."""
    if (isinstance(n, bool) or int(n) != n or n < 2 or not np.isfinite(difference)
            or difference < 0 or not np.isfinite(sd) or sd <= 0 or not 0 < alpha < 1):
        raise ValueError("Invalid paired-t planning arguments")
    critical = stats.t.ppf(1 - alpha / 2, n - 1)
    ncp = float(difference / sd * math.sqrt(n))
    power = float(stats.nct.sf(critical, n - 1, ncp)
                  + stats.nct.sf(critical, n - 1, -ncp))
    if not np.isfinite(power):
        raise PlanningLimitError("Nonfinite noncentral-t power")
    return float(np.clip(power, 0, 1))


def minimum_runs(delta, sd, alpha, target=.8, floor=10, maximum=10000):
    if (not np.isfinite(delta) or delta <= 0 or not np.isfinite(sd) or sd <= 0
            or not 0 < alpha < target < 1 or isinstance(floor, bool)
            or isinstance(maximum, bool) or int(floor) != floor or int(maximum) != maximum
            or not 2 <= floor <= maximum):
        raise ValueError("Invalid sample-size inputs")
    if difference_power(maximum, delta, sd, alpha) < target:
        raise PlanningLimitError(f"Required repetitions exceed {maximum}; do not truncate the design")
    low, high = int(floor), int(maximum)
    while low < high:
        mid = (low + high) // 2
        if difference_power(mid, delta, sd, alpha) >= target:
            high = mid
        else:
            low = mid + 1
    return low


def detectable_difference(n, sd, alpha, target=.8):
    difference_power(n, 0, sd, alpha)
    high = 1.0
    while difference_power(n, high, 1., alpha) < target:
        high *= 2
        if high > 1e6:
            raise PlanningLimitError("Standardized MDE outside numerical planning range")
    standardized = optimize.brentq(lambda d: difference_power(n, d, 1., alpha) - target,
                                   0., high, xtol=1e-12)
    return float(standardized * sd)


def _paired_panel(metrics, *, expected_seeds=None):
    frame = pd.DataFrame(metrics).copy()
    if not {"model", "seed", "rmse"} <= set(frame):
        raise ValueError("Metrics need model, seed and rmse")
    if (frame.empty or set(frame.model) != set(MODELS)
            or frame.duplicated(["model", "seed"]).any()
            or not np.isfinite(frame.rmse.to_numpy(float)).all() or frame.rmse.lt(0).any()):
        raise ValueError("Need unique finite nonnegative metrics for exactly three models")
    if not np.isfinite(frame.seed.to_numpy(float)).all() or not (frame.seed % 1).eq(0).all():
        raise ValueError("Seed identifiers must be finite integers")
    frame["seed"] = frame.seed.astype(int)
    panel = frame.pivot(index="seed", columns="model", values="rmse").sort_index()
    if panel.isna().any().any():
        raise ValueError("Every model must have exactly the same paired seeds")
    if expected_seeds is not None and panel.index.tolist() != sorted(expected_seeds):
        raise ValueError("Observed seeds differ from the registered seed set")
    return panel


def plan_sample_size(pilot_metrics, design):
    """Plan the maximum R required across the three pre-specified paired tests.

    A Bonferroni lower chi-square tail gives simultaneous upper SD bounds with
    at least the specified confidence, assuming normal paired differences.
    Planning alpha/3 is conservative for the eventual three-test Holm family.
    """
    s = _settings(design)
    identity = _coherent_origin(pd.DataFrame(pilot_metrics), design, pilot=True)
    seeds = design.get("pilot", {}).get("seeds", list(range(7101, 7111)))
    panel = _paired_panel(pilot_metrics, expected_seeds=seeds)
    n = len(panel)
    if n < 5:
        raise ValueError("The guide requires at least five independent pilot seeds")
    alpha_each = s["alpha"] / len(PAIRS)
    chi_tail = (1 - s["confidence"]) / len(PAIRS)
    factor = float(math.sqrt((n - 1) / stats.chi2.ppf(chi_tail, n - 1)))
    rows = []
    for a, b in PAIRS:
        delta_values = (panel[a] - panel[b]).to_numpy(float)
        sd = float(delta_values.std(ddof=1))
        if sd <= 0:
            raise ValueError("Zero pilot paired SD cannot justify power sizing: " + a + "/" + b)
        upper = sd * factor
        point_r = minimum_runs(s["delta"], sd, alpha_each, s["power"], s["min_runs"], s["max_runs"])
        upper_r = minimum_runs(s["delta"], upper, alpha_each, s["power"], s["min_runs"], s["max_runs"])
        shapiro = stats.shapiro(delta_values)
        rows.append({"model_a": a, "model_b": b, "n_pilot_pairs": n,
                     "difference_a_minus_b_mean_descriptive": float(delta_values.mean()),
                     "paired_sd": sd, "paired_sd_upper": upper,
                     "sd_upper_factor": factor, "sd_upper_chi2_lower_tail": chi_tail,
                     "simultaneous_sd_confidence": s["confidence"],
                     "delta": s["delta"], "standardized_delta_point_sd": s["delta"] / sd,
                     "standardized_delta_upper_sd": s["delta"] / upper,
                     "planning_alpha_each": alpha_each,
                     "point_sd_required_R": point_r, "upper_sd_required_R": upper_r,
                     "pilot_R_mde_point_sd": detectable_difference(n, sd, alpha_each, s["power"]),
                     "pilot_R_mde_upper_sd": detectable_difference(n, upper, alpha_each, s["power"]),
                     "shapiro_W": float(shapiro.statistic), "shapiro_p": float(shapiro.pvalue),
                     "normality_diagnostic_changes_test": False})
    required = max(row["upper_sd_required_R"] for row in rows)
    final = max(s["final_runs_floor"], required)
    for row in rows:
        row.update(planned_R=final,
                   noncentrality_at_planned_R=s["delta"] / row["paired_sd_upper"] * math.sqrt(final),
                   power_at_planned_R_point_sd=difference_power(final, s["delta"], row["paired_sd"], alpha_each),
                   power_at_planned_R_upper_sd=difference_power(final, s["delta"], row["paired_sd_upper"], alpha_each),
                   mde_at_planned_R_upper_sd=detectable_difference(final, row["paired_sd_upper"], alpha_each, s["power"]))
        sensitivity=[]
        for confidence in design.get("statistics",{}).get("sd_sensitivity_confidences",[.8,.9,.95]):
            if not 0 < float(confidence) < 1:
                raise ValueError("Invalid SD sensitivity confidence")
            upper=row["paired_sd"]*math.sqrt((n-1)/stats.chi2.ppf((1-float(confidence))/len(PAIRS),n-1))
            try:
                repetitions=minimum_runs(s["delta"],upper,alpha_each,s["power"],s["min_runs"],s["max_runs"])
                status="defined"
            except PlanningLimitError:
                repetitions=None
                status="exceeds_numerical_planning_bound"
            sensitivity.append({"simultaneous_sd_confidence":float(confidence),"paired_sd_upper":upper,
                                "required_R":repetitions,"status":status,
                                "changes_registered_R":False})
        row["sd_confidence_sensitivity"]=sensitivity
    seed_start = int(design.get("definitive", {}).get("seed_start", 8101))
    final_seeds = list(range(seed_start, seed_start + final))
    hpo_seeds = design.get("search", design.get("hpo",{})).get("seeds", [6101, 6102])
    if set(final_seeds) & (set(seeds) | set(hpo_seeds)) or set(seeds) & set(hpo_seeds):
        raise ValueError("Tuning, pilot and definitive seeds must be disjoint")
    return {"R": final, "R_final": final, "R_required": required, "planned_R": final,
            "final_runs_floor": s["final_runs_floor"], "floor_predeclared_not_empirical_power": True,
            **identity, "seeds": final_seeds,
            "pilot_seeds": list(seeds), "n_pilot_pairs": n,
            "delta": s["delta"], "alpha": s["alpha"], "planning_alpha_each": alpha_each,
            "target_power": s["power"], "family_size": len(PAIRS),
            "point_sd_required_R_max": max(row["point_sd_required_R"] for row in rows),
            "sd_upper_simultaneous_confidence": s["confidence"], "pairs": rows,
            "normality_assumed_not_proven": True,
            "temporal_power_claimed": False,
            "assumptions": ["Independent training seeds conditional on the same 60-day calendar",
                            "Normal paired RMSE differences; Shapiro is descriptive and does not select the test",
                            "Pilot difference SD transfers from the analogous development period to the known historical reserve",
                            "At least target marginal power per pair at the simultaneous upper SD under those assumptions",
                            "Marginal planning power is not joint power, achieved post-hoc power, or power over future markets"],
            "scope": SCOPE}


def _metric_values(y, prediction, mase_scale):
    error = y - prediction
    squared = error * error
    denominator = float(np.sum((y - y.mean()) ** 2))
    nonzero = y != 0
    return {"rmse": float(np.sqrt(squared.mean())), "mae": float(np.abs(error).mean()),
            "mse": float(squared.mean()),
            "r2": float(1 - squared.sum() / denominator) if denominator > 0 else np.nan,
            "mape": float(100 * np.mean(np.abs(error[nonzero] / y[nonzero]))) if nonzero.any() else np.nan,
            "mase": float(np.abs(error).mean() / mase_scale) if np.isfinite(mase_scale) and mase_scale > 0 else np.nan,
            "mape_defined_dates": int(nonzero.sum()), "n_dates": len(y)}


def _validate_predictions(predictions, design, plan):
    frame = pd.DataFrame(predictions).copy()
    _coherent_origin(frame, design, plan=plan)
    required = {"model", "seed", "outcome_day", "y_true", "y_pred"}
    if not required <= set(frame):
        raise ValueError("Missing planned forecast fields: " + str(required - set(frame)))
    frame["outcome_day"] = pd.to_datetime(frame.outcome_day, utc=True).dt.strftime("%Y-%m-%d")
    if "mase_scale" in frame and "mase_train_scale" not in frame:
        frame["mase_train_scale"] = frame["mase_scale"]
    if "mase_train_scale" not in frame:
        raise ValueError("Frozen train-only MASE denominator is required")
    frame["model"] = frame.model.replace({"Persistencia": PERSISTENCE, "persistence": PERSISTENCE})
    if not np.isfinite(frame.seed.to_numpy(float)).all() or not (frame.seed % 1).eq(0).all():
        raise ValueError("Seed identifiers must be finite integers")
    frame["seed"] = frame.seed.astype(int)
    if (frame.empty or not set(frame.model) <= set(MODELS) | {PERSISTENCE}
            or frame.duplicated(["model", "seed", "outcome_day"]).any()
            or not np.isfinite(frame[["y_true", "y_pred", "mase_train_scale"]].to_numpy(float)).all()
            or frame.mase_train_scale.le(0).any()
            or frame[["y_true", "y_pred"]].lt(0).any().any()):
        raise ValueError("Forecasts must be finite, nonnegative, unique and use a positive frozen scale")
    neural = frame.loc[frame.model.isin(MODELS)]
    expected_seeds = list(plan["seeds"])
    if (len(expected_seeds) < _settings(design)["final_runs_floor"]
            or len(set(expected_seeds)) != len(expected_seeds)
            or plan.get("R",len(expected_seeds)) != len(expected_seeds)
            or plan.get("R_final",len(expected_seeds)) != len(expected_seeds)):
        raise ValueError("The definitive sample-size plan must have unique seeds matching R")
    reserve=design.get("reserve",{})
    registered_dates=pd.date_range(reserve.get("start","2026-09-01"),
                                   reserve.get("end","2026-10-30")).strftime("%Y-%m-%d").tolist()
    if len(registered_dates)!=60:
        raise ValueError("The registered outcome calendar must contain exactly60dates")
    observed = set(neural[["model", "seed"]].itertuples(index=False, name=None))
    if observed != set(itertools.product(MODELS, expected_seeds)):
        raise ValueError("Incomplete or unexpected definitive model/seed grid")
    reference = None
    for (_, _), group in neural.groupby(["model", "seed"], sort=True):
        group = group.sort_values("outcome_day")
        dates = group.outcome_day.tolist()
        if dates != registered_dates:
            raise ValueError("The planned evaluation requires exactly the60registered reserve dates")
        current = group[["outcome_day", "y_true"]].reset_index(drop=True)
        if reference is None:
            reference = current
        elif not reference.equals(current):
            raise ValueError("All models and seeds must share dates and observed outcomes")
        if group.mase_train_scale.nunique() != 1:
            raise ValueError("The final MASE training scale must remain fixed throughout scoring")
    if "persistence_pred" in neural:
        if not np.isfinite(neural.persistence_pred.to_numpy(float)).all() or neural.persistence_pred.lt(0).any():
            raise ValueError("Invalid causal persistence predictions")
        per_date = neural.groupby("outcome_day").persistence_pred.nunique()
        if not per_date.eq(1).all():
            raise ValueError("Persistence must be identical across neural trajectories")
    if PERSISTENCE not in set(frame.model):
        if "persistence_pred" not in neural:
            raise ValueError("The registered six temporal pairs require causal persistence predictions")
        first = neural.loc[neural.model.eq(MODELS[0]) & neural.seed.eq(expected_seeds[0])].copy()
        first["model"], first["seed"], first["y_pred"] = PERSISTENCE, 0, first.persistence_pred
        frame = pd.concat([frame, first], ignore_index=True)
    persistent = frame.loc[frame.model.eq(PERSISTENCE)].sort_values("outcome_day")
    if (persistent.seed.nunique() != 1 or len(persistent) != 60
            or not persistent[["outcome_day", "y_true"]].reset_index(drop=True).equals(reference)):
        raise ValueError("Persistence must have exactly one deterministic, matched 60-day trajectory")
    if "persistence_pred" in neural:
        expected = neural.groupby("outcome_day").persistence_pred.first().sort_index().to_numpy(float)
        if not np.array_equal(persistent.y_pred.to_numpy(float), expected):
            raise ValueError("Explicit persistence differs from the registered reference predictions")
    return frame


def _run_metrics(frame):
    rows = []
    for (model, seed), group in frame.groupby(["model", "seed"], sort=True):
        group = group.sort_values("outcome_day")
        values = _metric_values(group.y_true.to_numpy(float), group.y_pred.to_numpy(float),
                                float(group.mase_train_scale.iloc[0]))
        rows.append({"model": model, "seed": int(seed), **values, "scope": SCOPE})
    return pd.DataFrame(rows)


def _summarize(runs, alpha):
    rows = []
    for model in (*MODELS, PERSISTENCE):
        group = runs.loc[runs.model.eq(model)]
        stochastic = model != PERSISTENCE
        for metric in METRICS:
            values = group[metric].to_numpy(float)
            valid = values[np.isfinite(values)]
            n = len(valid)
            mean = float(valid.mean()) if n else np.nan
            sd = float(valid.std(ddof=1)) if stochastic and n > 1 else np.nan
            half = float(stats.t.ppf(1-alpha/2, n-1) * sd / math.sqrt(n)) if n > 1 and stochastic else np.nan
            rows.append({"model": model, "metric": metric, "n_total": len(values), "n_defined": n,
                         "n_dates": int(group.n_dates.iloc[0]), "mean": mean, "sample_sd": sd,
                         "ci95_lower": mean-half, "ci95_upper": mean+half,
                         "stochastic_training": stochastic,
                         "ci_method": "Marginal Student-t across independent training seeds" if stochastic else "Not applicable: deterministic reference",
                         "scope": SCOPE})
    return pd.DataFrame(rows)


def _paired_tests(runs, design, plan):
    s = _settings(design)
    panel = _paired_panel(runs.loc[runs.model.isin(MODELS)], expected_seeds=plan["seeds"])
    n = len(panel)
    rng = np.random.default_rng(s["bootstrap_seed"])
    draws = rng.integers(0, n, size=(s["bootstrap_draws"], n))
    rows = []
    for a, b in PAIRS:
        values = (panel[a]-panel[b]).to_numpy(float)
        mean, sd = float(values.mean()), float(values.std(ddof=1))
        se = sd / math.sqrt(n)
        tvalue = mean / se if se > 0 else np.nan
        pvalue = float(2 * stats.t.sf(abs(tvalue), n-1)) if se > 0 else np.nan
        half = float(stats.t.ppf(1-s["alpha"]/2, n-1) * se) if se>0 else np.nan
        family_half = float(stats.t.ppf(1-s["alpha"]/(2*len(PAIRS)), n-1) * se) if se>0 else np.nan
        shapiro = stats.shapiro(values)
        bootstrap = values[draws]
        boot_sd = bootstrap.std(axis=1, ddof=1)
        boot_dz = np.divide(bootstrap.mean(axis=1), boot_sd,
                            out=np.full(s["bootstrap_draws"], np.nan), where=boot_sd > 0)
        valid_dz = boot_dz[np.isfinite(boot_dz)]
        effect_ci = np.quantile(valid_dz, [s["alpha"]/2, 1-s["alpha"]/2]) if len(valid_dz) else [np.nan,np.nan]
        mean_a, mean_b = float(panel[a].mean()), float(panel[b].mean())
        rows.append({"model_a": a, "model_b": b, "n_paired": n, "mean_a": mean_a, "mean_b": mean_b,
                     "difference_a_minus_b": mean, "difference_sample_sd": sd,
                     "ci95_lower": mean-half, "ci95_upper": mean+half,
                     "ci_family_lower": mean-family_half, "ci_family_upper": mean+family_half,
                     "statistic": tvalue, "degrees_of_freedom": n-1, "p_value_unadjusted": pvalue,
                     "status": "defined_normal_reference" if se > 0 else "degenerate_difference_sd_not_tested",
                     "cohen_d_z": mean/sd if sd > 0 else np.nan,
                     "effect_ci95_lower": float(effect_ci[0]), "effect_ci95_upper": float(effect_ci[1]),
                     "effect_bootstrap_n": s["bootstrap_draws"], "effect_bootstrap_defined": len(valid_dz),
                     "gain_a_over_b_absolute": -mean,
                     "gain_a_over_b_relative_percent_point": 100*(mean_b-mean_a)/mean_b if mean_b > 0 else np.nan,
                     "absolute_planning_delta": s["delta"],
                     "gain_a_over_b_family_ci_lower": -mean-family_half,
                     "gain_a_over_b_family_ci_upper": -mean+family_half,
                     "a_improvement_family_ci_exceeds_absolute_delta": bool(-mean-family_half > s["delta"]),
                     "b_improvement_family_ci_exceeds_absolute_delta": bool(mean-family_half > s["delta"]),
                     "shapiro_W": float(shapiro.statistic), "shapiro_p": float(shapiro.pvalue),
                     "normality_diagnostic_changes_test": False,
                     "direction": "Negative A-minus-B favors A; the zero-null test does not test the 5-percent practical margin",
                     "scope": SCOPE})
    result = pd.DataFrame(rows)
    result["p_value_holm3"] = holm_with_undefined(result.p_value_unadjusted)
    result["reject_holm3"] = result.p_value_holm3.lt(s["alpha"])
    result["multiplicity_family_size"] = len(PAIRS)
    return result


def circular_block_counts(n_dates, block_length, draws, rng):
    if not 1 <= block_length <= n_dates or draws < 1:
        raise ValueError("Invalid circular-block bootstrap dimensions")
    blocks = math.ceil(n_dates/block_length)
    starts = rng.integers(0, n_dates, size=(draws, blocks))
    indices = ((starts[...,None] + np.arange(block_length)) % n_dates).reshape(draws,-1)[:,:n_dates]
    weights = np.zeros((draws,n_dates), dtype=float)
    np.add.at(weights, (np.repeat(np.arange(draws),n_dates),indices.reshape(-1)),1)
    return weights


def _temporal_analysis(frame, design):
    s = _settings(design)
    order = (*MODELS, PERSISTENCE)
    default_pairs=list(PAIRS)+[(model,PERSISTENCE) for model in MODELS]
    settings=design.get("statistics",{}).get("temporal_secondary",{})
    temporal_pairs=list(map(tuple,settings.get("pairs",default_pairs)))
    if (len(temporal_pairs)!=6 or set(temporal_pairs)!=set(default_pairs)
            or settings.get("dm_hac_lags",6)!=6 or settings.get("dm_horizon",1)!=1
            or settings.get("dm_family_size",6)!=6):
        raise ValueError("The temporal analysis requires the six fixed pairs, HAC6 and h1")
    loss = {}
    for model in order:
        group = frame.loc[frame.model.eq(model)].copy()
        group["loss"] = (group.y_true-group.y_pred)**2
        panel = group.pivot(index="seed",columns="outcome_day",values="loss").sort_index(axis=1)
        if panel.isna().any().any():
            raise ValueError("Temporal analysis requires identical complete date support")
        loss[model] = panel.to_numpy(float)
    baseline = design.get("baseline_model", design.get("selection",{}).get("baseline_model"))
    dm_rows = []
    for a,b in temporal_pairs:
        dm = diebold_mariano_hln(loss[a].mean(axis=0),loss[b].mean(axis=0),horizon=1,hac_lags=6)
        dm.update(model_a=a, model_b=b,
                  scope="Registered-method exploratory temporal comparison on an already observed 60-day historical reserve; no temporal power guarantee",
                  limitations="Bartlett HAC lag6 and HLN h1 are fixed; short series, regime change and dependence beyond lag6 can invalidate the approximation",
                  loss_estimand="Mean squared residual across seeds, not squared ensemble residual",
                  baseline_selected_before_reserve=baseline)
        dm_rows.append(dm)
    dm_frame = pd.DataFrame(dm_rows)
    dm_frame["p_value_holm6"] = holm_with_undefined(dm_frame.p_value)
    dm_frame["multiplicity_family_size"] = len(temporal_pairs)
    dm_frame["reject_holm6"] = dm_frame.p_value_holm6.lt(s["alpha"])
    point = {m:float(np.sqrt(loss[m].mean(axis=1)).mean()) for m in order}
    bootstrap_rows=[]
    for block in (7,3,14):
        rng = np.random.default_rng(s["temporal_bootstrap_seed"]+block)
        boot = {m:np.empty(s["bootstrap_draws"]) for m in order}
        for start in range(0,s["bootstrap_draws"],128):
            size=min(128,s["bootstrap_draws"]-start)
            counts=circular_block_counts(60,block,size,rng)
            for model in order:
                boot[model][start:start+size]=np.sqrt(loss[model] @ counts.T / 60).mean(axis=0)
        for a,b in temporal_pairs:
            difference=boot[a]-boot[b]
            relative=np.divide(100*(boot[b]-boot[a]),boot[b],out=np.full(len(difference),np.nan),where=boot[b]>0)
            valid=relative[np.isfinite(relative)]
            reverse=np.divide(100*(boot[a]-boot[b]),boot[a],out=np.full(len(difference),np.nan),where=boot[a]>0)
            reverse_valid=reverse[np.isfinite(reverse)]
            lower,upper=np.quantile(difference,[s["alpha"]/2,1-s["alpha"]/2])
            family_lower,family_upper=np.quantile(difference,[s["alpha"]/(2*len(temporal_pairs)),1-s["alpha"]/(2*len(temporal_pairs))])
            rlo,rhi=np.quantile(valid,[s["alpha"]/2,1-s["alpha"]/2]) if len(valid) else (np.nan,np.nan)
            rflo,rfhi=np.quantile(valid,[s["alpha"]/(2*len(temporal_pairs)),1-s["alpha"]/(2*len(temporal_pairs))]) if len(valid) else (np.nan,np.nan)
            rblo,rbhi=np.quantile(reverse_valid,[s["alpha"]/(2*len(temporal_pairs)),1-s["alpha"]/(2*len(temporal_pairs))]) if len(reverse_valid) else (np.nan,np.nan)
            bootstrap_rows.append({"model_a":a,"model_b":b,"block_length":block,
                "role":"pre-specified secondary temporal analysis, main block length" if block==7 else "descriptive block-length sensitivity; do not select favorable significance",
                "n_dates":60,"draws":s["bootstrap_draws"],"bootstrap_seed":s["temporal_bootstrap_seed"]+block,
                "difference_a_minus_b":point[a]-point[b],"ci95_lower":float(lower),"ci95_upper":float(upper),
                "ci_family_lower":float(family_lower),"ci_family_upper":float(family_upper),
                "relative_gain_a_over_b_percent":100*(point[b]-point[a])/point[b] if point[b]>0 else np.nan,
                "relative_gain_ci95_lower":float(rlo),"relative_gain_ci95_upper":float(rhi),
                "relative_gain_family_lower":float(rflo),"relative_gain_family_upper":float(rfhi),
                "a_relative_gain_family_lower_exceeds_5percent":bool(rflo>5),
                "relative_gain_b_over_a_percent":100*(point[a]-point[b])/point[a] if point[a]>0 else np.nan,
                "relative_gain_b_over_a_family_lower":float(rblo),"relative_gain_b_over_a_family_upper":float(rbhi),
                "b_relative_gain_family_lower_exceeds_5percent":bool(rblo>5),
                "multiplicity_family_size":len(temporal_pairs),
                "ci_method":"Circular moving-block percentile bootstrap; Bonferroni percentile tails for six intervals",
                "estimand":"Mean of individual-seed RMSEs after each common resampled calendar, not RMSE of ensemble predictions",
                "scope":"Secondary, conditional on frozen models; 60 dates and assumed short-range stationarity; approximate coverage, not new markets",
                "baseline_selected_before_reserve":baseline})
    return dm_frame,pd.DataFrame(bootstrap_rows)


def _diagnostics(frame, alpha):
    rows=[]
    for (model,seed),part in frame.loc[frame.model.isin(MODELS)].groupby(["model","seed"],sort=True):
        residual=(part.sort_values("outcome_day").y_true-part.sort_values("outcome_day").y_pred).to_numpy(float)
        sd=float(residual.std(ddof=1))
        base={"model":model,"seed":int(seed),"n_residuals":len(residual),
              "scope":"Pre-specified exploratory residual diagnostic; overlapping sigma7 and fitted networks limit null calibration"}
        if sd <= 0:
            rows.extend([{**base,"diagnostic":name,"lag":lag,"statistic":np.nan,"p_value":np.nan,"status":"zero_residual_sd"}
                         for name,lag in (("BDS",2),("Ljung-Box",1),("Ljung-Box",5))])
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                statistic,pvalue=bds(residual,max_dim=2,epsilon=1.5*sd)
                stat,p=float(np.asarray(statistic).ravel()[0]),float(np.asarray(pvalue).ravel()[0])
                rows.append({**base,"diagnostic":"BDS","lag":2,"epsilon":1.5*sd,"statistic":stat,
                             "p_value":p,"status":"defined" if np.isfinite(stat+p) else "nonfinite"})
            except Exception as error:
                rows.append({**base,"diagnostic":"BDS","lag":2,"statistic":np.nan,"p_value":np.nan,
                             "status":type(error).__name__})
            try:
                lb=acorr_ljungbox(residual,lags=[1,5],model_df=0,return_df=True)
                for lag in (1,5):
                    stat,p=float(lb.loc[lag,"lb_stat"]),float(lb.loc[lag,"lb_pvalue"])
                    rows.append({**base,"diagnostic":"Ljung-Box","lag":lag,"model_df":0,
                                 "statistic":stat,"p_value":p,"status":"defined" if np.isfinite(stat+p) else "nonfinite"})
            except Exception as error:
                for lag in (1,5):
                    rows.append({**base,"diagnostic":"Ljung-Box","lag":lag,"statistic":np.nan,"p_value":np.nan,
                                 "status":type(error).__name__})
    result=pd.DataFrame(rows)
    result["p_value_holm_diagnostic_family"]=np.nan
    result["multiplicity_family_size"]=0
    for name,group in result.groupby("diagnostic"):
        result.loc[group.index,"p_value_holm_diagnostic_family"]=holm_with_undefined(group.p_value)
        result.loc[group.index,"multiplicity_family_size"]=len(group)
    result["reject_holm"]=result.p_value_holm_diagnostic_family.lt(alpha)
    return result


def analyze_predictions(predictions, design, plan):
    """Return aggregate frames and an analysis summary; never choose a winner on test."""
    s=_settings(design)
    identity=_coherent_origin(pd.DataFrame(predictions),design,plan=plan)
    frame=_validate_predictions(predictions,design,plan)
    runs=_run_metrics(frame)
    for name,value in identity.items():
        runs[name]=value
    summary_metrics=_summarize(runs,s["alpha"])
    paired=_paired_tests(runs,design,plan)
    dm,bootstrap=_temporal_analysis(frame,design)
    diagnostics=_diagnostics(frame,s["alpha"])
    baseline=design.get("baseline_model",design.get("selection",{}).get("baseline_model"))
    if baseline is not None and baseline not in MODELS:
        raise ValueError("Baseline must be the development-frozen neural family")
    summary={**identity,"models":list(MODELS),"history_days":7,"target_n_days":7,"horizon_days":1,
             "R":len(plan["seeds"]),"dates_per_trajectory":60,"neural_prediction_rows":len(frame.loc[frame.model.isin(MODELS)]),
             "baseline_selected_before_reserve":baseline,"selection_using_holdout":False,
             "primary_family":"Three registered-method exploratory paired RMSE zero-null tests, Holm FWER alpha0.05",
             "planned_delta":s["delta"],"power_target":s["power"],"power_scope":"Conditional training-seed normal-difference planning; no temporal or achieved power claim",
             "temporal_secondary_family":"Six fixed DM comparisons, Holm6; circular block7 bootstrap sensitivity3/14",
             "normality_diagnostic_selects_tests":False,"bootstrap_draws":s["bootstrap_draws"],
             "bootstrap_seed":s["bootstrap_seed"],"temporal_bootstrap_seed":s["temporal_bootstrap_seed"],
             "R_required":plan.get("R_required"),"R_final":len(plan["seeds"]),
             "final_runs_floor":s["final_runs_floor"],
             "known_reserve_exploratory":True,"fresh_confirmatory_holdout":False,
             "financial_utility_claimed":False,"delta_role":"Unchanged academic planning convention",
             "reference_source_sha256":verify_reference_sources(),
             "scope":SCOPE,"live_prospective_forecasts":False,
             "temporal_limitations":"Already observed historical outcome cohort of only60days; market-regime generalization and 80percent temporal power are not established",
             "significance_is_not_practical_relevance":True}
    return {"metric_summary":summary_metrics,"run_metrics":runs,"pairwise_rmse":paired,
            "dm_pairs":dm,"temporal_bootstrap":bootstrap,"diagnostics":diagnostics,"summary":summary}
