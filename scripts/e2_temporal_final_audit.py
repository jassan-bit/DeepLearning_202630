"""Read-only numerical refinement and relative forecast fluctuation diagnostics.

The original 4,999-draw analysis is verified and retained. This program only
reads saved predictions and epoch-risk metadata: it cannot fit or infer with a
model. The new analysis is post hoc on the already known E1 chronology.

GR means Giacomini--Rossi's equal-predictive-ability fluctuation statistic, NOT a
test that an arbitrary nonzero relative mean is constant. The latter diagnostic
is separately named, centered, and calibrated. No rolling parameter refits are
asserted. All bootstrap inference requires weak dependence and an adequate
long-run variance approximation; it is not finite-sample distribution free.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from e2_reduced_final_analysis import ANALYSIS_SEED, bootstrap_contrast, holm
from e2_reduced_protocol import DELTA, NEURAL, TOP10, SYSTEMS

VERSION = "e2-temporal-final-audit-v1"
INPUT_SHA256 = "65ae74b547a6073b2c286dea8039a2e7a1b1f6063b1d9ac0eb760c9ce727d187"
SEEDS = tuple(range(271801, 271807))
GR_SEED = 2026111401


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def write_csv(path, rows):
    rows = list(rows)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def predictions_by_model(payload):
    """Validate the complete, dated, fixed-seed input without compressing gaps."""
    y = np.asarray(payload["y_true"], dtype=float)
    mask = np.asarray(payload["mask"])
    dates = payload["dates"]
    if len(dates) != 608 or y.shape != (608,) or mask.shape != (608,) or mask.dtype != bool or not mask.all():
        raise ValueError("This audit requires the frozen complete 608-date calendar")
    from datetime import date, timedelta
    if dates != [(date(2025, 1, 1)+timedelta(days=i)).isoformat() for i in range(608)]:
        raise ValueError("Dates must be the original consecutive calendar")
    records = {}
    for row in payload["predictors"]:
        key = (row["model"], row["seed"])
        if key in records:
            raise ValueError("Duplicate fixed predictor")
        records[key] = np.asarray(row["prediction_original"], dtype=float)
    result = {}
    for model in dict.fromkeys(row["model"] for row in payload["predictors"]):
        seeds = SEEDS if model in NEURAL else ("deterministic",)
        selected = [records[(model, seed)] for seed in seeds]
        result[model] = np.asarray(selected)
        if result[model].shape != (len(seeds), 608) or not np.isfinite(result[model]).all():
            raise ValueError("Incomplete or nonfinite forecast trajectory")
    if set(result) != set(SYSTEMS) or len(records) != 81 or not np.isfinite(y).all():
        raise ValueError("The original complete 16-system panel is required")
    return y, result, mask


def verify_original(payload, predictions, targets, mask):
    """Prove that the same code, data, seed and estimand reproduce original B."""
    original = payload["original_analysis"]["primary_contrasts"]
    computed = []
    expected_pairs = list(itertools.product(TOP10, ("GRU32", "Persistence")))
    if [(row["model"], row["baseline"]) for row in original] != expected_pairs:
        raise ValueError("The original contrast family or ordering changed")
    for index, (model, baseline) in enumerate(expected_pairs):
        value = bootstrap_contrast(targets, predictions[model], predictions[baseline], mask=mask,
                                   draws=4999, seed=ANALYSIS_SEED+index)
        computed.append({"model": model, "baseline": baseline, **value})
    computed = holm(computed)
    checked = []
    for old, new in zip(original, computed):
        for key in ("gain", "delta", "p_one_sided", "p_Holm", "CI95", "draws", "block_days", "seed",
                    "test", "resampling_unit", "reject_at_FWER05"):
            if old[key] != new[key]:
                raise ValueError("Original 4999-draw result is not exactly reproduced: "+key)
        checked.append({"model": old["model"], "baseline": old["baseline"],
                        "exact_original_4999_reproduction": True,
                        "original_p_raw": old["p_one_sided"], "original_p_Holm": old["p_Holm"]})
    return checked


def primary_refinement(payload, predictions, targets, mask, draws):
    if draws < 10000:
        raise ValueError("The separately registered refinement requires at least 10000 draws")
    rows = []
    for index, (model, baseline) in enumerate(itertools.product(TOP10, ("GRU32", "Persistence"))):
        value = bootstrap_contrast(targets, predictions[model], predictions[baseline], mask=mask,
                                   draws=draws, seed=ANALYSIS_SEED+index)
        old = payload["original_analysis"]["primary_contrasts"][index]
        rows.append({"model": model, "baseline": baseline, **value,
                     "original_draws": old["draws"], "original_p_raw": old["p_one_sided"],
                     "original_p_Holm": old["p_Holm"], "original_CI95": old["CI95"],
                     "posthoc_computational_refinement": True,
                     "original_result_replaced": False})
        print(json.dumps({"event": "primary_refinement_completed", "model": model,
                          "baseline": baseline, "draws": draws}), flush=True)
    return holm(rows)


def hac_long_run_variance(values, maxlags=14):
    """Global Bartlett HAC, denominator T; batch axis allowed for bootstrap."""
    values = np.asarray(values, dtype=float)
    if values.ndim not in (1, 2) or values.shape[-1] < maxlags+2 or not np.isfinite(values).all():
        raise ValueError("Finite, sufficiently long scalar loss trajectories required")
    n = values.shape[-1]
    centered = values-values.mean(axis=-1, keepdims=True)
    variance = np.mean(centered*centered, axis=-1)
    for lag in range(1, maxlags+1):
        variance += 2*(1-lag/(maxlags+1))*np.sum(centered[..., lag:]*centered[..., :-lag], axis=-1)/n
    # Bartlett is nonnegative in exact arithmetic. Never manufacture precision.
    scale = np.mean(centered*centered, axis=-1)
    if np.any(variance < -1e-12*np.maximum(scale, np.finfo(float).tiny)):
        raise ValueError("Negative long-run variance beyond rounding tolerance")
    return np.maximum(variance, 0)


def rolling_sums(values, window):
    values = np.asarray(values, dtype=float)
    if values.ndim not in (1, 2) or type(window) is not int or not 1 <= window <= values.shape[-1]:
        raise ValueError("A valid rolling window is required")
    cumulative = np.cumsum(values, axis=-1)
    zeros = np.zeros(values.shape[:-1]+(1,))
    cumulative = np.concatenate((zeros, cumulative), axis=-1)
    return cumulative[..., window:]-cumulative[..., :-window]


def fluctuation_statistics(difference, *, window, maxlags=14):
    """Uncentered GR-EPA and centered constant-mean diagnostic, global HAC."""
    d = np.asarray(difference, dtype=float)
    variance = hac_long_run_variance(d, maxlags)
    if np.any(variance <= 0):
        raise ValueError("Fluctuation studentization requires positive long-run variance")
    denominator = np.sqrt(window*variance)[..., None]
    local = rolling_sums(d, window)/denominator
    centered_local = rolling_sums(d-d.mean(axis=-1, keepdims=True), window)/denominator
    return {"GR_EPA": np.max(np.abs(local), axis=-1),
            "relative_mean_constancy": np.max(np.abs(centered_local), axis=-1),
            "GR_path": local, "constancy_path": centered_local,
            "long_run_variance": variance}


def fluctuation_bootstrap(difference, *, window=182, maxlags=14, block_days=14,
                          draws=20000, seed=GR_SEED, chunk_size=128):
    """Centered circular calendar blocks with global HAC reestimated per draw.

    Centering of the input bootstrap imposes EPA. The observed GR statistic is
    intentionally NOT centered. The separate constancy statistic centers each
    observed/bootstrap path by its own full-period mean.
    """
    d = np.asarray(difference, dtype=float)
    if d.ndim != 1 or draws < 99 or type(draws) is not int or type(block_days) is not int or block_days < 1:
        raise ValueError("Explicit finite trajectory and bootstrap settings required")
    if not np.isfinite(d).all() or not 2 <= window <= len(d):
        raise ValueError("Invalid loss difference or window")
    variance = float(hac_long_run_variance(d, maxlags))
    if variance <= np.finfo(float).tiny:
        status = "identical_loss_path" if np.all(d == 0) else "not_estimable_zero_variance"
        return {"status": status, "GR_statistic": 0. if status == "identical_loss_path" else None,
                "constancy_statistic": 0. if status == "identical_loss_path" else None,
                "p_EPA": 1. if status == "identical_loss_path" else None,
                "p_constancy": 1. if status == "identical_loss_path" else None,
                "critical_EPA_95": None, "critical_constancy_95": None,
                "long_run_variance": variance, "draws": draws, "seed": seed}
    observed = fluctuation_statistics(d, window=window, maxlags=maxlags)
    rng = np.random.default_rng(seed)
    null = d-d.mean()
    epa, constancy = np.empty(draws), np.empty(draws)
    for start in range(0, draws, chunk_size):
        count = min(chunk_size, draws-start)
        starts = rng.integers(0, len(d), size=(count, math.ceil(len(d)/block_days)))
        indices = ((starts[:, :, None]+np.arange(block_days)) % len(d)).reshape(count, -1)[:, :len(d)]
        simulated = fluctuation_statistics(null[indices], window=window, maxlags=maxlags)
        epa[start:start+count] = simulated["GR_EPA"]
        constancy[start:start+count] = simulated["relative_mean_constancy"]
    return {"status": "computed", "GR_statistic": float(observed["GR_EPA"]),
            "constancy_statistic": float(observed["relative_mean_constancy"]),
            "p_EPA": float((1+np.count_nonzero(epa >= observed["GR_EPA"]))/ (draws+1)),
            "p_constancy": float((1+np.count_nonzero(constancy >= observed["relative_mean_constancy"]))/ (draws+1)),
            "critical_EPA_95": float(np.quantile(epa, .95)),
            "critical_constancy_95": float(np.quantile(constancy, .95)),
            "long_run_variance": variance, "draws": draws, "seed": seed,
            "GR_path": observed["GR_path"].tolist(),
            "constancy_path": observed["constancy_path"].tolist()}


def temporal_audit(predictions, targets, dates, settings):
    daily_losses = {model: np.mean((value-targets[None, :])**2, axis=0)
                    for model, value in predictions.items()}
    rows, paths, family = [], [], []
    for index, (model, baseline) in enumerate(itertools.product(TOP10, ("GRU32", "Persistence"))):
        # Negative values mean the candidate's MSE is lower on that date.
        d = daily_losses[model]-daily_losses[baseline]
        result = fluctuation_bootstrap(d, window=settings["window_days"], maxlags=settings["hac_lags"],
                                      block_days=settings["block_days"], draws=settings["draws"],
                                      seed=settings["seed_base"]+index)
        gr_path, constant_path = result.pop("GR_path", []), result.pop("constancy_path", [])
        row = {"model": model, "baseline": baseline, "mean_loss_difference": float(d.mean()),
               "window_days": settings["window_days"], "mu": settings["mu"],
               "hac_lags": settings["hac_lags"], "block_days": settings["block_days"],
               "n_calendar_dates": len(d), "n_overlapping_windows": len(gr_path), **result,
               "GR_original_asymptotic_critical_95_mu03": 3.012,
               "GR_crosses_original_unadjusted_critical": None if result["GR_statistic"] is None else result["GR_statistic"] > 3.012,
               "calibration": "posthoc_centered_circular_block_bootstrap_adaptation",
               "loss": "mean_individual_seed_squared_error_not_ensemble_squared_error",
               "equal_ability_rejection_proves_temporal_break": False,
               "rolling_parameter_refits_performed": False}
        rows.append(row)
        for label, pvalue in (("EPA", result["p_EPA"]), ("constancy", result["p_constancy"])):
            family.append({"model": model, "baseline": baseline, "test": label,
                           "p_one_sided": 1. if pvalue is None else pvalue, "estimable": pvalue is not None})
        for offset, (gr, constant) in enumerate(zip(gr_path, constant_path)):
            paths.append({"model": model, "baseline": baseline, "window_start": dates[offset],
                          "window_end": dates[offset+settings["window_days"]-1],
                          "GR_EPA_statistic": gr, "centered_constancy_statistic": constant})
        print(json.dumps({"event": "relative_fluctuation_completed", "model": model,
                          "baseline": baseline, "draws": settings["draws"]}), flush=True)
    adjusted = holm(family)
    for index, row in enumerate(rows):
        for offset, name in enumerate(("EPA", "constancy")):
            value = adjusted[2*index+offset]
            row["p_"+name+"_Holm40"] = value["p_Holm"] if value["estimable"] else None
            row["reject_"+name+"_FWER05_Holm40"] = value["reject_at_FWER05"] if value["estimable"] else None
    return rows, paths


def descriptive_gap_sensitivity(path, multipliers=(1, 2, 5, 10)):
    """No margin is chosen as confirmatory; no seed t-test manufactures one."""
    with Path(path).open() as stream:
        records = list(csv.DictReader(stream))
    rows, seed_rows = [], []
    for model in NEURAL:
        selected = [r for r in records if r["stage"] == "final" and r["model"] == model]
        selected.sort(key=lambda r: int(r["seed"]))
        if [int(row["seed"]) for row in selected] != list(SEEDS):
            raise ValueError("Six original final training-risk metadata records required")
        gaps = np.asarray([float(row["calibration_minus_train_rmse_original"]) for row in selected])
        if not np.isfinite(gaps).all() or not all(row["same_final_predictor_metadata_binding_verified"] == "True" for row in selected):
            raise ValueError("Unbound or nonfinite saved training/calibration gap")
        for row, gap in zip(selected, gaps):
            seed_rows.append({"model": model, "seed": int(row["seed"]),
                              "training_RMSE": float(row["selected_train_rmse_original"]),
                              "calibration_RMSE": float(row["calibration_rmse_original"]),
                              "gap_calibration_minus_training_RMSE": float(gap),
                              "training_batch_size": int(row["epoch_evaluation_batch_size"]),
                              "calibration_prediction_batch_size": int(row["persisted_prediction_batch_size"]),
                              "formal_a_priori_overfit_test": False})
        for multiplier in multipliers:
            margin = DELTA*multiplier
            rows.append({"model": model, "illustrative_margin_delta_multiple": multiplier,
                         "illustrative_margin_RMSE_units": margin,
                         "mean_gap_calibration_minus_training": float(gaps.mean()),
                         "median_gap": float(np.median(gaps)), "min_gap": float(gaps.min()),
                         "max_gap": float(gaps.max()), "seeds_gap_below_margin": int((gaps < margin).sum()),
                         "seeds_total": len(gaps), "a_priori_delta_g_available": False,
                         "absence_of_overfit_established": False,
                         "interpretation": "descriptive_tolerance_sensitivity_not_train_validation_noninferiority",
                         "limitations": "different_time_windows; training_epoch_batch_vs_calibration_batch1; six_seeds_not_six_markets"})
    return rows, seed_rows


def validate_plan(plan):
    if plan["version"] != VERSION or plan["input_sha256"] != INPUT_SHA256:
        raise ValueError("A matching separately recorded analysis plan is required")
    primary, fluctuation = plan["primary"], plan["fluctuation"]
    for key, value in {"draws": 20000, "original_draws": 4999, "seed_base": ANALYSIS_SEED,
                       "block_days": 14, "delta": DELTA, "family_size": 20}.items():
        if primary[key] != value:
            raise ValueError("Registered primary setting mismatch: "+key)
    for key, value in {"draws": 20000, "seed_base": GR_SEED, "block_days": 14, "hac_lags": 14,
                       "mu": .3, "window_days": 182}.items():
        if fluctuation[key] != value:
            raise ValueError("Registered fluctuation setting mismatch: "+key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--training-summary", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-registration-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if sha256(args.input) != INPUT_SHA256:
        raise ValueError("The fixed input digest changed")
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    if len(args.plan_registration_commit) != 40 or any(c not in "0123456789abcdef" for c in args.plan_registration_commit):
        raise ValueError("A source-control registration commit is required")
    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.input.read_text())
    y, predictions, mask = predictions_by_model(payload)
    original_checks = verify_original(payload, predictions, y, mask)
    write_csv(args.output/"original_4999_exact_checks.csv", original_checks)
    primary = primary_refinement(payload, predictions, y, mask, plan["primary"]["draws"])
    write_csv(args.output/"primary_refinement_20000.csv", primary)
    fluctuation, paths = temporal_audit(predictions, y, payload["dates"], plan["fluctuation"])
    write_csv(args.output/"relative_fluctuation_tests.csv", fluctuation)
    write_csv(args.output/"relative_fluctuation_paths.csv", paths)
    gaps, seeds = descriptive_gap_sensitivity(args.training_summary, plan["gap_sensitivity"]["margins_delta_multipliers"])
    write_csv(args.output/"gap_tolerance_sensitivity_descriptive.csv", gaps)
    write_csv(args.output/"saved_training_calibration_gaps.csv", seeds)
    summary = {"version": VERSION, "plan_registration_commit": args.plan_registration_commit,
               "analysis_status": plan["analysis_status"], "original_4999_exactly_reproduced": len(original_checks),
               "primary_family_size": len(primary), "primary_refinement_draws": 20000,
               "primary_refinement_rejections_Holm20": sum(r["reject_at_FWER05"] for r in primary),
               "original_primary_rejections_Holm20": sum(r["reject_at_FWER05"] for r in payload["original_analysis"]["primary_contrasts"]),
               "original_results_overwritten": False,
               "GR_EPA_bootstrap_rejections_Holm40": sum(r["reject_EPA_FWER05_Holm40"] is True for r in fluctuation),
               "constant_relative_mean_rejections_Holm40": sum(r["reject_constancy_FWER05_Holm40"] is True for r in fluctuation),
               "GR_EPA_unadjusted_original_asymptotic_critical_crossings": sum(r["GR_crosses_original_unadjusted_critical"] is True for r in fluctuation),
               "fluctuation_draws_per_contrast": 20000, "fluctuation_total_hypotheses_family": 40,
               "GR_null_is_equal_predictive_ability_not_constant_nonzero_relative_mean": True,
               "new_model_fits": 0, "model_inference_calls": 0, "new_predictions": 0,
               "newly_blind_test_claimed": False, "a_priori_delta_g_repaired": False,
               "rolling_parameter_refits_claimed": False,
               "scope": "conditional_fixed_forecasts_on_known_E1_chronology; weak_dependence_and_global_HAC_assumptions; posthoc_diagnostics"}
    write_json(args.output/"summary.json", summary)
    write_json(args.output/"analysis_plan.json", plan)
    source_files = [Path(__file__), Path(__file__).with_name("e2_reduced_final_analysis.py"),
                    Path(__file__).with_name("e2_reduced_protocol.py")]
    manifest = {"version": VERSION, "input_sha256": INPUT_SHA256,
                "training_summary_sha256": sha256(args.training_summary),
                "plan_sha256": sha256(args.plan), "plan_registration_commit": args.plan_registration_commit,
                "source_sha256": {str(p.name): sha256(p) for p in source_files},
                "output_sha256": {p.name: sha256(p) for p in sorted(args.output.iterdir()) if p.is_file() and p.name != "manifest.json"},
                "training_performed": False, "model_inference_performed": False,
                "scientific_engine_commit": plan["scientific_engine_commit"]}
    write_json(args.output/"manifest.json", manifest)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
