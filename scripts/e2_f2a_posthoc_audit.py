"""Read-only post hoc F2a diagnostics of the frozen reduced E2 predictions.

This module consumes saved numbers. It never loads a model, selects a recipe,
retrains, recalibrates, or reruns the original twenty final comparisons. All
inference is exploratory and conditional on the already observed chronology.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import hashlib
import json
import math
from pathlib import Path
import warnings

import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import bds

from e2_reduced_protocol import CLASSICAL, NEURAL, SYSTEMS

VERSION = "e2-f2a-posthoc-audit-v1"
SEEDS = tuple(range(271801, 271807))
FAMILY_SIZE = 826
ALPHA = .05
HAC_LAGS = 14
BLOCK_DAYS = 14
BOOTSTRAP_SEED = 202611131
RMSE_TOLERANCE = 1e-10
FINAL_DATES = [(date(2025, 1, 1) + timedelta(days=i)).isoformat() for i in range(608)]
CALIBRATION_DATES = [(date(2024, 11, 8) + timedelta(days=i)).isoformat() for i in range(53)]
ASSUMPTIONS = (
    "Exploratory nominal asymptotic p-values; overlapping sigma7 targets and "
    "serial dependence limit calibration. HAC14 assumes adequate bandwidth and "
    "weak dependence. Holm does not repair invalid marginal-test assumptions. "
    "Seeds share one market chronology and are not independent market samples."
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value, *, n=None, ndim=1):
    result = np.asarray(value, dtype=float)
    _require(result.ndim == ndim and result.size > 0 and np.isfinite(result).all(),
             "Finite nonempty aligned numeric arrays are required")
    if n is not None:
        _require(result.shape[-1] == n, "Array length does not match the calendar")
    return result


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def holm(rows, family_size, alpha=.05):
    """Undefined hypotheses remain in the fixed family as conservative p=1.

    Their stored raw/adjusted p-values stay null: missing evidence is never
    represented as an observed nonsignificant result.
    """
    _require(family_size >= len(rows) and 0 < alpha < 1, "Invalid Holm family")
    result = [dict(row, p_holm=None, reject=False) for row in rows]
    available = []
    for i, row in enumerate(result):
        p = row.get("p_raw")
        if p is not None:
            _require(math.isfinite(p) and 0 <= p <= 1, "Invalid raw p-value")
            available.append((float(p), i))
    previous = 0.
    for rank, (p, i) in enumerate(sorted(available)):
        previous = max(previous, min(1., (family_size-rank)*p))
        result[i].update(p_holm=previous, reject=previous < alpha)
    return result


def _wald(difference, covariance):
    difference, covariance = np.asarray(difference), np.asarray(covariance)
    _require(np.isfinite(covariance).all(), "Nonfinite HAC covariance")
    _require(np.linalg.matrix_rank(covariance) == len(difference), "Singular HAC covariance")
    statistic = float(difference @ np.linalg.solve(covariance, difference))
    _require(math.isfinite(statistic) and statistic >= -1e-10, "Invalid Wald statistic")
    statistic = max(0., statistic)
    return statistic, float(stats.chi2.sf(statistic, len(difference)))


def _mz(y, p, maxlags=HAC_LAGS):
    y, p = _array(y), _array(p)
    _require(y.shape == p.shape, "Forecast/target alignment differs")
    result = dict(statistic=None, p_raw=None, status="not_estimable", reason=None)
    try:
        X = np.column_stack([np.ones(len(y)), p])
        _require(np.linalg.matrix_rank(X) == 2,
                 "Constant forecast prevents intercept/slope identification")
        fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags, "use_correction": True})
        _require(np.linalg.norm(fit.resid) > 100*np.finfo(float).eps*np.linalg.norm(y),
                 "Zero regression residual variance prevents sampling uncertainty estimation")
        statistic, pvalue = _wald(fit.params-np.array([0., 1.]), fit.cov_params())
        result.update(statistic=statistic, p_raw=pvalue, status="estimable",
                      intercept=float(fit.params[0]), slope=float(fit.params[1]),
                      intercept_se=float(fit.bse[0]), slope_se=float(fit.bse[1]),
                      reference_distribution="chi2_df2")
    except (ValueError, np.linalg.LinAlgError, FloatingPointError, ZeroDivisionError) as error:
        result["reason"] = str(error)
    return result


def _coverage_hac(y, p, q):
    lower, upper = np.maximum(0., p-q), p+q
    hits = ((y >= lower) & (y <= upper)).astype(float)
    _require(np.ptp(hits) > 0, "Coverage indicator variance is zero; HAC sampling uncertainty is not estimable")
    fit = sm.OLS(hits, np.ones((len(y), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS, "use_correction": True})
    statistic, pvalue = _wald(np.array([hits.mean()-.9]), fit.cov_params())
    return dict(statistic=statistic, p_raw=pvalue, coverage90=float(hits.mean()),
                reference_distribution="chi2_df1")


def trajectory_metrics(y, p, q):
    y, p = _array(y), _array(p)
    _require(y.shape == p.shape and math.isfinite(q) and q >= 0, "Invalid trajectory or interval")
    lower, upper = np.maximum(0., p-q), p+q
    _require(np.all(upper >= lower), "Invalid frozen nonnegative interval")
    error = y-p
    return dict(RMSE=float(np.sqrt(np.mean(error**2))), MAE=float(np.mean(np.abs(error))),
                bias=float(np.mean(error)), MSE_mean_individual=float(np.mean(error**2)),
                coverage90=float(np.mean((y >= lower) & (y <= upper))),
                mean_interval_width90=float(np.mean(upper-lower)), n_dates=len(y))


def summarize_system(y, predictions, qs):
    y, P, qs = _array(y), _array(predictions, ndim=2), _array(qs)
    _require(P.shape == (len(qs), len(y)), "System trajectories are not aligned")
    members = [trajectory_metrics(y, p, float(q)) for p, q in zip(P, qs)]
    result = {field: float(np.mean([r[field] for r in members])) for field in
              ("RMSE", "MAE", "bias", "MSE_mean_individual", "coverage90", "mean_interval_width90")}
    result.update(R=len(P), n_dates=len(y), primary_is_ensemble_RMSE=False,
                  RMSE_seed_SD=float(np.std([r["RMSE"] for r in members], ddof=1)) if len(P) > 1 else None)
    return result


def diagnostic_tests(y, p, dates, identity, calibrator_q):
    """Ten tests for one frozen trajectory; no pooling across seed rows."""
    y, p = _array(y), _array(p)
    _require(y.shape == p.shape and len(y) == len(dates), "Unaligned residual dates")
    _require(all((date.fromisoformat(b)-date.fromisoformat(a)).days == 1
                 for a, b in zip(dates, dates[1:])), "Residual calendar must be dense daily")
    q = float(calibrator_q)
    _require(math.isfinite(q) and q >= 0, "Invalid frozen calibrator q")
    error = y-p
    subtraction_scale = max(float(np.max(np.abs(y))), float(np.max(np.abs(p))))
    specs = [(f"LB_{kind}_lag{lag}", {"lags": lag, "model_df": 0})
             for kind in ("residual", "squared") for lag in (7, 14)]
    specs += [("ARCH_LM_lag7", {"lags": 7, "ddof": 0}),
              ("BDS_dim2", {"dimension": 2, "radius_SD": 1.5, "SD_ddof": 1}),
              ("BDS_dim3", {"dimension": 3, "radius_SD": 1.5, "SD_ddof": 1}),
              ("Jarque_Bera", {}),
              ("Mincer_Zarnowitz", {"null_intercept": 0., "null_slope": 1., "HAC_lags": HAC_LAGS,
                                    "HAC_small_sample_correction": True}),
              ("Coverage90_HAC", {"nominal": .9, "HAC_lags": HAC_LAGS,
                                  "HAC_small_sample_correction": True, "frozen_q": q})]
    rows = []
    bds_values = None
    for test, parameters in specs:
        row = {**identity, "test": test, "parameters": parameters, "n_dates": len(y),
               "statistic": None, "p_raw": None, "status": "not_estimable", "reason": None,
               "assumptions": ASSUMPTIONS}
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if test == "Mincer_Zarnowitz":
                    row.update(_mz(y, p))
                elif test == "Coverage90_HAC":
                    row.update(_coverage_hac(y, p, q))
                else:
                    _require(np.std(error, ddof=1) >
                             100*np.finfo(float).eps*subtraction_scale,
                             "Residual variance is zero at floating point precision")
                    if test.startswith("LB_"):
                        values = error**2 if test.startswith("LB_squared") else error
                        values_scale = (max(float(np.max(error**2)),
                                            2*float(np.max(np.abs(error)))*subtraction_scale)
                                        if test.startswith("LB_squared") else subtraction_scale)
                        _require(np.std(values, ddof=1) > 100*np.finfo(float).eps*values_scale,
                                 "Diagnostic series variance is zero at floating point precision")
                        lag = parameters["lags"]
                        value = acorr_ljungbox(values, lags=[lag], model_df=0, return_df=True).iloc[0]
                        row.update(statistic=float(value["lb_stat"]), p_raw=float(value["lb_pvalue"]))
                    elif test.startswith("ARCH_"):
                        lm, pvalue, _, _ = het_arch(error, nlags=7, ddof=0)
                        row.update(statistic=float(lm), p_raw=float(pvalue))
                    elif test.startswith("BDS_"):
                        if bds_values is None:
                            radius = 1.5*np.std(error, ddof=1)
                            z, pvalue = bds(error, max_dim=3, epsilon=radius)
                            bds_values = (np.atleast_1d(z), np.atleast_1d(pvalue))
                        index = parameters["dimension"]-2
                        row.update(statistic=float(bds_values[0][index]), p_raw=float(bds_values[1][index]))
                        row["parameters"]["radius_original_units"] = float(1.5*np.std(error, ddof=1))
                    elif test == "Jarque_Bera":
                        value = stats.jarque_bera(error)
                        row.update(statistic=float(value.statistic), p_raw=float(value.pvalue))
                if caught:
                    row["warnings"] = sorted({str(w.message) for w in caught})
            if row["p_raw"] is not None and row["statistic"] is not None:
                _require(math.isfinite(row["p_raw"]) and 0 <= row["p_raw"] <= 1 and
                         math.isfinite(row["statistic"]), "Nonfinite diagnostic result")
                row.update(status="estimable", reason=None)
            else:
                row["reason"] = row["reason"] or "Undefined diagnostic"
        except (ValueError, np.linalg.LinAlgError, FloatingPointError, ZeroDivisionError) as exc:
            row.update(statistic=None, p_raw=None, status="not_estimable", reason=str(exc))
        rows.append(row)
    return rows


def _quarter(d):
    return d[:4]+"Q"+str((int(d[5:7])-1)//3+1)


def stability_test(y, P, dates, identity):
    labels = [_quarter(d) for d in dates]
    quarters = list(dict.fromkeys(labels))
    row = {**identity, "seed": "mean_individual_seed_losses", "test": "Calendar_quarter_loss_stability",
           "parameters": {"quarters": quarters, "HAC_lags": HAC_LAGS,
                          "HAC_small_sample_correction": True,
                          "loss": "daily mean of individual seed squared errors; not ensemble loss"},
           "n_dates": len(y), "statistic": None, "p_raw": None, "status": "not_estimable",
           "reason": None, "assumptions": ASSUMPTIONS}
    try:
        losses = np.mean((P-y[None, :])**2, axis=0)
        _require(np.ptp(losses) > 100*np.finfo(float).eps*np.max(np.abs(losses)),
                 "Daily loss variance is zero at floating point precision; stability uncertainty is not estimable")
        X = np.column_stack([np.ones(len(y))] +
                            [np.array([label == q for label in labels], dtype=float) for q in quarters[1:]])
        _require(len(quarters) > 1 and np.linalg.matrix_rank(X) == len(quarters), "Unidentified calendar segments")
        fit = sm.OLS(losses, X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS, "use_correction": True})
        _require(np.linalg.norm(fit.resid) > 100*np.finfo(float).eps*np.linalg.norm(losses),
                 "Zero regression residual variance prevents stability sampling uncertainty estimation")
        statistic, pvalue = _wald(fit.params[1:], fit.cov_params()[1:, 1:])
        row.update(statistic=statistic, p_raw=pvalue, status="estimable", df=len(quarters)-1,
                   reference_distribution=f"chi2_df{len(quarters)-1}")
    except (ValueError, np.linalg.LinAlgError, FloatingPointError, ZeroDivisionError) as exc:
        row["reason"] = str(exc)
    return row


def validate_input(payload):
    _require(isinstance(payload, dict) and bool(payload.get("version")), "Versioned input is required")
    _require(payload.get("dates") == FINAL_DATES, "Retain the exact 608 registered daily dates")
    y = _array(payload["y_true"], n=608)
    _require(np.all(y >= 0), "sigma7 targets must be nonnegative")
    mask = payload.get("mask")
    _require(isinstance(mask, list) and len(mask) == 608 and all(v is True for v in mask),
             "The original complete 608-date common mask is required; no compressed gaps")
    predictors = payload.get("predictors", [])
    expected = {(m, seed) for m in NEURAL for seed in SEEDS} | {(m, "deterministic") for m in CLASSICAL}
    actual = {(r.get("model"), r.get("seed")) for r in predictors}
    _require(len(predictors) == 81 and actual == expected, "Require all 81 exact frozen predictor identities")
    calibration_truth = None
    for row in predictors:
        p = _array(row["prediction_original"], n=608)
        q = row.get("calibrator_q")
        _require(isinstance(q, (int, float)) and math.isfinite(q) and q >= 0, "Finite frozen q is required")
        _require(np.all(p+q >= np.maximum(0., p-q)), "Invalid frozen intervals")
        checksum = row.get("checkpoint_sha256", "")
        _require(isinstance(checksum, str) and len(checksum) == 64 and
                 all(c in "0123456789abcdef" for c in checksum), "Predictor checksum is required")
        present = [row.get(k) is not None for k in
                   ("calibration_dates", "calibration_targets", "calibration_predictions")]
        _require(all(present) or not any(present), "Partial saved calibration input is inadmissible")
        if all(present):
            _require(row["calibration_dates"] == CALIBRATION_DATES, "Retain all 53 original calibration dates")
            cy = _array(row["calibration_targets"], n=53)
            cp = _array(row["calibration_predictions"], n=53)
            if calibration_truth is None:
                calibration_truth = cy
            _require(np.array_equal(cy, calibration_truth), "Calibration targets differ between predictors")
            observed_q = float(np.sort(np.abs(cy-cp))[48])
            _require(abs(observed_q-q) <= 1e-12, "Saved calibration residuals disagree with frozen q")
    _require(isinstance(payload.get("original_analysis"), dict), "Untouched original analysis is required")
    return payload


def verify_original_rmse(payload):
    original = {r["model"]: r for r in payload["original_analysis"]["table_A2"]}
    _require(set(original) == set(SYSTEMS), "Original A2 must contain exactly sixteen systems")
    y = _array(payload["y_true"])
    rows = []
    for predictor in payload["predictors"]:
        model, seed = predictor["model"], predictor["seed"]
        old = original[model]
        index = 0 if seed == "deterministic" else SEEDS.index(seed)
        _require(len(old["RMSE_per_seed"]) == (1 if model in CLASSICAL else 6), "Original seed RMSE registry differs")
        expected = float(old["RMSE_per_seed"][index])
        observed = float(np.sqrt(np.mean((y-_array(predictor["prediction_original"]))**2)))
        delta = observed-expected
        _require(abs(delta) <= RMSE_TOLERANCE, f"Original RMSE mismatch: {model}/{seed}: {delta}")
        rows.append({"model": model, "seed": seed, "checkpoint_sha256": predictor["checkpoint_sha256"],
                     "original_RMSE": expected, "audit_RMSE": observed, "difference": delta,
                     "absolute_tolerance": RMSE_TOLERANCE, "passed": True})
    for model in SYSTEMS:
        values = [r["audit_RMSE"] for r in rows if r["model"] == model]
        _require(abs(float(np.mean(values))-float(original[model]["RMSE"])) <= RMSE_TOLERANCE,
                 "Original mean-individual seed RMSE mismatch")
    return rows


def _block_indices(rng, n, draws, block=BLOCK_DAYS):
    starts = rng.integers(0, n, size=(draws, math.ceil(n/block)))
    return ((starts[:, :, None]+np.arange(block)[None, None, :]) % n).reshape(draws, -1)[:, :n]


def bootstrap_intervals(payload, draws=2000):
    _require(draws >= 2000, "At least 2000 exploratory bootstrap draws are required")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    final_indices = _block_indices(rng, 608, draws)
    calibration_indices = _block_indices(rng, 53, draws)
    y = _array(payload["y_true"])
    coverage, gaps = [], []
    for row in payload["predictors"]:
        identity = {k: row[k] for k in ("model", "seed", "checkpoint_sha256")}
        p, q = _array(row["prediction_original"]), row["calibrator_q"]
        hits = ((y >= np.maximum(0., p-q)) & (y <= p+q)).astype(float)
        samples = hits[final_indices].mean(axis=1)
        coverage.append({**identity, "coverage90": float(hits.mean()),
                         "coverage90_CI95_temporal": np.quantile(samples, [.025, .975]).tolist(),
                         "bootstrap_draws": draws, "block_days": BLOCK_DAYS,
                         "scope": "nominal circular-block interval conditional on frozen trajectories"})
        comparison = {**identity, "status": "not_estimable", "reason": "Saved calibration predictions absent",
                      "formal_test": False, "delta_g": None,
                      "absence_of_overfitting_established": False}
        if row.get("calibration_predictions") is not None:
            cy, cp = _array(row["calibration_targets"]), _array(row["calibration_predictions"])
            ce, fe = (cy-cp)**2, (y-p)**2
            bootstrap_gap = np.sqrt(fe[final_indices].mean(axis=1))-np.sqrt(ce[calibration_indices].mean(axis=1))
            cm, fm = float(np.sqrt(ce.mean())), float(np.sqrt(fe.mean()))
            comparison.update(status="descriptive", reason=None, calibration_RMSE=cm,
                              final_RMSE=fm, final_minus_calibration_RMSE=fm-cm,
                              final_to_calibration_RMSE_ratio=None if cm == 0 else fm/cm,
                              gap_CI95_temporal_nominal=np.quantile(bootstrap_gap, [.025, .975]).tolist(),
                              calibration_dates=53, final_dates=608, bootstrap_draws=draws,
                              block_days=BLOCK_DAYS, calendars_resampled_separately=True,
                              scope="Same frozen predictor, different periods; gap may reflect distribution shift. "
                                    "Calibration labels set q and do not select these point predictions. "
                                    "No a priori delta_g or train-origin predictions; not an overfit noninferiority test.")
        gaps.append(comparison)
    return coverage, gaps


def audit(payload, *, bootstrap_draws=2000):
    validate_input(payload)
    rmse_checks = verify_original_rmse(payload)
    y = _array(payload["y_true"])
    dates, predictors = payload["dates"], payload["predictors"]
    tests, quarterly, rolling, systems = [], [], [], []
    for row in predictors:
        identity = {k: row[k] for k in ("model", "seed", "checkpoint_sha256")}
        tests.extend(diagnostic_tests(y, row["prediction_original"], dates, identity, row["calibrator_q"]))
    quarter_labels = np.array([_quarter(d) for d in dates])
    for model in SYSTEMS:
        members = sorted((r for r in predictors if r["model"] == model), key=lambda r: str(r["seed"]))
        P = np.asarray([r["prediction_original"] for r in members])
        qs = np.asarray([r["calibrator_q"] for r in members])
        identity = {"model": model, "checkpoint_sha256": [r["checkpoint_sha256"] for r in members]}
        tests.append(stability_test(y, P, dates, identity))
        systems.append({"model": model, **summarize_system(y, P, qs)})
        for quarter in dict.fromkeys(quarter_labels):
            mask = quarter_labels == quarter
            selected = np.flatnonzero(mask)
            quarterly.append({"model": model, "quarter": quarter, "start_date": dates[selected[0]],
                              "end_date": dates[selected[-1]], "partial_calendar_quarter": quarter == "2026Q3",
                              **summarize_system(y[mask], P[:, mask], qs)})
        for end in range(89, len(y)):
            start = end-89
            rolling.append({"model": model, "start_date": dates[start], "end_date": dates[end],
                            "window_days": 90, **summarize_system(y[start:end+1], P[:, start:end+1], qs)})
    _require(len(tests) == FAMILY_SIZE, "Exactly 826 post hoc diagnostic hypotheses are required")
    tests = holm(tests, FAMILY_SIZE, ALPHA)
    coverage, gaps = bootstrap_intervals(payload, draws=bootstrap_draws)
    return _clean({"version": VERSION, "scope": "exploratory post hoc audit of saved frozen predictions",
                   "post_hoc": True, "fits_performed": 0, "model_inference_calls": 0,
                   "original_twenty_comparisons_repeated": False, "original_analysis_modified": False,
                   "calendar_start": dates[0], "calendar_end": dates[-1], "calendar_dates": 608,
                   "systems": 16, "frozen_predictors": 81, "assumptions": ASSUMPTIONS,
                   "family": {"size": FAMILY_SIZE, "alpha": ALPHA, "adjustment": "Holm",
                              "undefined_counted_conservatively": True,
                              "interpretation": "Approximate exploratory control conditional on valid marginal p-values; "
                                                "does not extend or alter original Holm20 family"},
                   "tests": tests, "system_metrics": systems, "quarterly_metrics": quarterly,
                   "rolling90_metrics": rolling, "coverage_intervals": coverage,
                   "calibration_final_comparison": gaps, "original_rmse_checks": rmse_checks,
                   "original_RMSE_max_absolute_difference": max(abs(r["difference"]) for r in rmse_checks),
                   "PIT": {"status": "not_estimable", "reason": "Only a frozen90% absolute-residual interval exists; "
                           "no full predictive CDF. Seed variation is not a predictive distribution."},
                   "overfit": {"status": "not_estimable", "delta_g": None,
                               "absence_of_overfitting_established": False,
                               "reason": "No a priori overfit margin or saved per-date, batch1 training predictions. "
                                         "Calibration/final changes cannot identify model overfitting."},
                   "scope_limits": ["Residual iid/normality tests are nominal asymptotic diagnostics, not proof of usable predictability.",
                                    "Failure to reject does not establish normality, calibration, stability, or equivalence.",
                                    "Calendar stability is not an implemented Giacomini-Rossi test.",
                                    "This audit does not close robustness, interpretability, ablation, or the full F2a battery."],
                   "input_provenance": payload.get("provenance", {})})


def _write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, allow_nan=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args(argv)
    raw = args.input.read_bytes()
    payload = json.loads(raw)
    result = audit(payload, bootstrap_draws=args.bootstrap_draws)
    result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key in ("tests", "system_metrics", "quarterly_metrics", "rolling90_metrics", "coverage_intervals",
                "calibration_final_comparison", "original_rmse_checks"):
        _write_csv(args.output_dir/(key+".csv"), result[key])
    (args.output_dir/"audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    summary = {"version": VERSION, "tests": len(result["tests"]),
               "estimable": sum(r["status"] == "estimable" for r in result["tests"]),
               "rejected_Holm": sum(r["reject"] for r in result["tests"]),
               "RMSE_max_absolute_difference": result["original_RMSE_max_absolute_difference"],
               "output": str(args.output_dir/"audit.json")}
    print(json.dumps(summary, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
