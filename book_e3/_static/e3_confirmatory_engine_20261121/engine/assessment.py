"""R3 assessment of saved predictions, never a forecasting/model entry point.

All input CSVs must already have been produced by the sealed controller. This
module does not import a dataset loader, fit a model, or call a model forward.
Calibration fitting is a separate, deterministic operation BEFORE test opening.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import csv
import datetime as dt
import inspect
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
from scipy import stats

from .common import digest, read_json, require, sha, write_json
from .models import CANDIDATE, DETERMINISTIC, METHODS
from .power import statistical_rule
from .statistics import PROTOCOL_VERSION, analyze, hac_covariance, holm, stationary_indices

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("fit", "selection", "planning", "calibration", "test")
REQUIRED_CRITERIA = (
    "complete_four_panel_family", "protocol_closed_and_final_size_fixed",
    "sealed_evaluation_evidence", "method_calibration_admitted",
    "adequacy_thresholds_admitted", "all_primary_rivals_superior",
    "historical_primary_mean_rmse_favorable", "additional_two_of_three_per_rival",
    "no_significant_additional_inferiority", "effects_intervals_and_practical_margin",
    "power_per_contrast_at_least_90_and_joint_reported", "between_panel_tests_reported", "adequacy", "robustness",
)


def _json(value):
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, Mapping):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _load_spec(value, base=ROOT):
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_absolute():
            path = Path(base) / path
        return read_json(path), {"path": str(path), "sha256": sha(path)}
    require(isinstance(value, Mapping), "Expected JSON object or JSON path")
    return deepcopy(dict(value)), {"sha256": digest(value), "in_memory": True}


def _records(source, default_panel=None):
    """Normalize controller aliases, retaining their original-unit floor A."""
    if isinstance(source, Mapping) and "path" in source:
        require(source.get("sha256") == sha(source["path"]), "Saved prediction artifact checksum mismatch")
        source = source["path"]
    if isinstance(source, (str, Path)):
        with Path(source).open(newline="") as handle:
            raw = list(csv.DictReader(handle))
    elif isinstance(source, Mapping) and "rows" in source:
        raw = source["rows"]
    else:
        raw = list(source)
    rows, seen = [], set()
    for row in raw:
        item = {"panel": str(row.get("panel", default_panel or "F00")),
                "system": str(row.get("system", row.get("method", ""))),
                "seed": str(row["seed"]), "date": str(row.get("date", row.get("time", ""))),
                "y": float(row.get("y", row.get("y_true"))),
                "prediction": float(row.get("prediction", row.get("y_pred"))),
                "known_floor_A": float(row.get("known_floor_A", 0.)),
                "rfit": float(row.get("rfit", 1.))}
        dt.date.fromisoformat(item["date"])
        require(item["system"] in METHODS, "Unknown system outside frozen 21-system family")
        values = [item[k] for k in ("y", "prediction", "known_floor_A", "rfit")]
        require(np.isfinite(values).all() and item["known_floor_A"] >= 0 and item["rfit"] > 0,
                "Predictions require finite values, nonnegative known floor and positive fit scale")
        floor = math.sqrt(item["known_floor_A"])
        require(item["y"] + 1e-10 >= floor and item["prediction"] + 1e-10 >= floor,
                "Label/prediction violates the frozen common projected support")
        key = tuple(item[k] for k in ("panel", "system", "seed", "date"))
        require(key not in seen, "Duplicate saved prediction key")
        seen.add(key)
        rows.append(item)
    require(bool(rows), "Empty predictions are not evaluation evidence")
    return sorted(rows, key=lambda r: (r["panel"], r["system"], r["seed"], r["date"]))


def fit_calibrators(calibration, output_path=None, *, level=.9, target_floor=0.):
    """Fit absolute-residual conformal radii from calibration predictions only.

    ``known_floor_A`` is the known variance contribution, so the point/interval
    support is sqrt(A), not A. ``target_floor`` is an optional additional sigma
    floor. A rank exceeding n is unavailable rather than a fabricated finite
    quantile. The returned SHA is over the canonical payload excluding SHA.
    Repeating the operation on the same rows is byte/content deterministic.
    """
    require(0 < level < 1 and math.isfinite(target_floor) and target_floor >= 0,
            "Invalid calibration level or target floor")
    rows = _records(calibration)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["panel"], row["system"], row["seed"])].append(row)
    entries = []
    for key, group in sorted(grouped.items()):
        residuals = np.sort([abs(r["prediction"]-r["y"]) for r in group])
        n = len(residuals)
        # Tiny floating error at an integer must not select a larger order stat.
        rank = math.ceil((n+1)*level-1e-12)
        entries.append({"panel": key[0], "system": key[1], "seed": key[2], "n": n,
                        "rank_one_based": rank, "q": float(residuals[rank-1]) if rank <= n else None,
                        "status": "ok" if rank <= n else "insufficient_calibration_sample",
                        "first_date": group[0]["date"], "last_date": group[-1]["date"],
                        "dates_sha256": digest([r["date"] for r in group])})
    result = {"schema_version": 1, "kind": "frozen_absolute_residual_calibrators",
              "version": "finite-order-quantile-floor-v1", "level": float(level),
              "target_floor": float(target_floor), "calibration_input_sha256": digest(rows),
              "calibration_file_sha256": sha(calibration) if isinstance(calibration, (str, Path)) else None,
              "calibrators": entries,
              "distribution_claim": "point forecast MSE atoms are not a calibrated predictive distribution",
              "coverage_guarantee": "none under dependent time; empirical coverage/width only"}
    result["sha256"] = digest(result)
    if output_path is not None:
        write_json(output_path, result, immutable=True)
    return result


def _verified_calibrators(value, calibration_rows):
    obj, evidence = _load_spec(value)
    payload = {k: v for k, v in obj.items() if k != "sha256"}
    require(obj.get("sha256") == digest(payload), "Calibrator payload checksum mismatch")
    require(obj.get("kind") == "frozen_absolute_residual_calibrators", "Wrong calibration artifact kind")
    require(obj.get("calibration_input_sha256") == digest(calibration_rows), "Calibration predictions changed after fitting")
    return obj, evidence


def _matrix(rows, panel, seeds, systems=METHODS):
    by_key = {(r["system"], r["seed"], r["date"]): r for r in rows if r["panel"] == panel}
    require(by_key, "Missing saved panel: "+panel)
    present = {r[0] for r in by_key}
    require(present == set(systems), "Assessment requires all 21 systems in every panel")
    reference_dates = None
    for system in systems:
        observed_seeds = {s for m, s, _ in by_key if m == system}
        expected = {"deterministic"} if system in DETERMINISTIC else set(seeds)
        require(observed_seeds == expected, "Missing/extra seed or repeated deterministic fit: "+system)
        for seed in expected:
            dates = {d for m, s, d in by_key if m == system and s == seed}
            if reference_dates is None:
                reference_dates = dates
            require(dates == reference_dates, "Saved endpoints differ across systems/seeds")
    dates = sorted(reference_dates)
    p = np.empty((len(systems), len(seeds), len(dates)))
    y, floors, scales = [], [], []
    for ti, date in enumerate(dates):
        reference = None
        for mi, system in enumerate(systems):
            for ri, seed in enumerate(seeds):
                row = by_key[(system, "deterministic" if system in DETERMINISTIC else seed, date)]
                values = (row["y"], row["known_floor_A"], row["rfit"])
                if reference is None:
                    reference = values
                require(np.allclose(reference, values, rtol=0, atol=1e-10), "Paired labels/floors/fit scales disagree")
                p[mi, ri, ti] = row["prediction"]
        y.append(reference[0]); floors.append(math.sqrt(reference[1])); scales.append(reference[2])
    require(np.allclose(scales, scales[0], rtol=0, atol=1e-10), "Fit scale changes within a panel")
    instruments = np.column_stack([np.ones(len(dates)), np.asarray(floors)/scales[0]])
    return {"p": p, "y": np.asarray(y), "floor": np.asarray(floors), "rfit": scales[0],
            "dates": dates, "seeds": list(seeds), "instruments": instruments}


def _settings(protocol):
    value = dict(protocol.get("statistical_settings", protocol.get("statistics", {})))
    required = ("bootstrap_reps", "block_length", "hac_lags")
    require(all(k in value for k in required), "Protocol lacks fixed temporal settings")
    return ({k: value[k] for k in required} | {"horizon": value.get("horizon", 1),
            "margin": value.get("margin", .02), "seed": value.get("seed", 20261121)}
            | {k: value[k] for k in ("reference_reps", "reference_seed", "alpha") if k in value})


def _coverage_bootstrap_means(time_values, settings):
    """Bounded joint stationary resampling for empirical coverage indicators."""
    count_series, n = time_values.shape
    B = settings["bootstrap_reps"]
    means = np.empty((B, count_series))
    rng = np.random.default_rng(settings["seed"])
    chunk = max(1, min(512, (64*1024**2)//(24*n)))
    for start in range(0, B, chunk):
        count = min(chunk, B-start)
        idx = stationary_indices(n, count, settings["block_length"], rng)
        counts = np.bincount((idx.astype(np.int64)+np.arange(count)[:, None]*n).ravel(), minlength=count*n)
        means[start:start+count] = counts.reshape(count, n)@time_values.T/n
    return means


def _analyze_panel(matrix, settings):
    return analyze(matrix["p"], matrix["y"], METHODS, instruments=matrix["instruments"], **settings)


def _panel_rule(results, margin):
    alphas = {r.get("family", {}).get("alpha", .05) for r in results}
    require(len(alphas) == 1, "Panel operational alpha settings differ")
    alpha = float(alphas.pop())
    value = _json(statistical_rule(results, margin))
    slots = []
    for pi, result in enumerate(results[1:], start=1):
        for c in result["comparisons"]:
            for method, tests in c["methods"].items():
                test = tests["epa"]
                slots.append({"panel_index": pi, "rival": c["rival"], "method": method,
                              "raw_p": test["raw_p"], "status": test["status"],
                              "direction": test["direction"], "mean_mse_difference": c["mean_mse_difference"]})
    adjusted = holm([s["raw_p"] if s["status"] == "ok" else 1. for s in slots])
    for slot, adj in zip(slots, adjusted):
        slot["global_additional_epa_adjusted_p"] = adj
        slot["conditional_adverse_signal"] = bool(slot["method"] == "gw_conditional" and slot["mean_mse_difference"] < 0 and adj < alpha)
        slot["significant_inferiority"] = bool(slot["method"] in ("dm_hln_hac", "stationary_bootstrap") and slot["mean_mse_difference"] < 0 and adj < alpha)
    value["additional_inferiority_family"] = {"size": len(slots), "alpha": alpha, "adjustment": "Holm across 3 additional panels x 20 rivals x 3 EPA tests",
                                              "slots": slots}
    value["additional_significant_inferiority"] = any(s["significant_inferiority"] for s in slots)
    value["additional_conditional_adverse_signal"] = any(s["conditional_adverse_signal"] for s in slots)
    # GW is an unsigned conditional-moment test. Even with a negative sample
    # mean, its rejection alone is not directional mean inferiority.
    value["joint_statistical_rule_pass"] = bool(all(value["contrast_pass"][0])
        and value["historical_main_all_mean_rmse_favorable"]
        and value["historical_additional_mean_rmse_two_of_three"]
        and value["additional_statistical_two_of_three"] and not value["additional_significant_inferiority"])
    value["additional_mean_pooled_mse_two_of_three"] = bool(all(
        sum(results[pi]["comparisons"][ri]["mean_mse_difference"] > 0 for pi in range(1, 4)) >= 2
        for ri in range(20)))
    return value


def residual_diagnostics(y, prediction, *, fit_scale, hac_lags, bds_enabled=False, bds_maximum_n=2000):
    """Per-trace HAC Mincer-Zarnowitz and residual diagnostics in time units.

    Normality/whiteness null nonrejection is never treated as calibrated-law
    evidence. MZ uses original point predictions, not a manufactured density.
    """
    import statsmodels
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    target, pred = np.asarray(y, float)/fit_scale, np.asarray(prediction, float)/fit_scale
    require(target.shape == pred.shape and target.ndim == 1 and np.isfinite(target).all() and np.isfinite(pred).all(), "Invalid residual trace")
    n = len(target)
    require(1 <= hac_lags < n//3, "Insufficient diagnostic dates for preregistered HAC lag")
    residual = target-pred
    result = {"n": n, "fit_scale": float(fit_scale), "versions": {"statsmodels": statsmodels.__version__},
              "bias_over_fit_scale": float(residual.mean()),
              "nonrejection_certifies_calibration": False, "tests": {}}
    lrv = float(hac_covariance(residual, hac_lags)[0, 0])
    se = math.sqrt(max(0., lrv)/n)
    result["bias_ci95_over_fit_scale"] = [float(residual.mean()-1.96*se), float(residual.mean()+1.96*se)]
    x = np.column_stack([np.ones(n), pred])
    degenerate_residual = float(np.std(residual)) <= np.finfo(float).eps*max(1., float(np.max(np.abs(residual))))*32
    if degenerate_residual:
        result["tests"]["mz_hac"] = {"status": "degenerate_residual_variance", "raw_p": 1., "statistic": None}
    elif np.linalg.matrix_rank(x) < 2:
        result["tests"]["mz_hac"] = {"status": "singular_predictor_design", "raw_p": 1., "statistic": None}
    else:
        gram = x.T@x/n
        beta = np.linalg.solve(gram, x.T@target/n)
        scores = x*(target-x@beta)[:, None]
        inverse = np.linalg.inv(gram)
        covariance = inverse@hac_covariance(scores, hac_lags)@inverse/n
        covariance = (covariance+covariance.T)/2
        eigenvalues = np.linalg.eigvalsh(covariance)
        if eigenvalues[0] <= max(eigenvalues[-1], np.finfo(float).tiny)*1e-10:
            test = {"status": "singular_hac_constraint_covariance", "raw_p": 1., "statistic": None}
        else:
            restriction = beta-np.array([0., 1.])
            wald = max(0., float(restriction@np.linalg.solve(covariance, restriction)))
            test = {"status": "ok", "raw_p": float(stats.chi2.sf(wald, 2)), "statistic": wald, "df": 2,
                    "null": "intercept=0 and slope=1", "reference": "asymptotic HAC Wald chi-square",
                    "intercept_over_fit_scale": float(beta[0]), "slope": float(beta[1]),
                    "coefficient_ci95": [[float(b-1.96*s), float(b+1.96*s)] for b, s in zip(beta, np.sqrt(np.maximum(np.diag(covariance), 0.)))]}
        result["tests"]["mz_hac"] = test
    tests = result["tests"]
    if degenerate_residual:
        for name in ("ljung_box", "arch_lm", "jarque_bera"):
            tests[name] = {"status": "degenerate_residual_variance", "raw_p": 1., "statistic": None}
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lb = acorr_ljungbox(residual, lags=[hac_lags], return_df=True)
            arch = het_arch(residual, nlags=hac_lags)
            jb = stats.jarque_bera(residual)
        for name, statistic, p in (("ljung_box", lb["lb_stat"].iloc[0], lb["lb_pvalue"].iloc[0]),
                                   ("arch_lm", arch[0], arch[1]), ("jarque_bera", jb.statistic, jb.pvalue)):
            tests[name] = {"status": "ok" if np.isfinite([statistic, p]).all() else "nonfinite_diagnostic",
                           "raw_p": float(p) if np.isfinite(p) else 1., "statistic": float(statistic) if np.isfinite(statistic) else None}
    if bds_enabled and n <= bds_maximum_n and np.std(residual) > 0:
        from statsmodels.tsa.stattools import bds
        try:
            stat, p = bds(residual, max_dim=2, distance=2.)
            stat, p = float(np.asarray(stat)), float(np.asarray(p))
            tests["bds_iid"] = {"status": "ok" if np.isfinite([stat, p]).all() else "nonfinite_diagnostic",
                                "statistic": stat if math.isfinite(stat) else None, "raw_p": p if math.isfinite(p) else 1.,
                                "null": "iid residuals; dependence can be expected for overlapping labels", "dimension": 2, "epsilon": "2 times residual standard deviation"}
        except ValueError as exc:
            tests["bds_iid"] = {"status": "unavailable", "reason": str(exc), "raw_p": 1., "statistic": None}
    else:
        tests["bds_iid"] = {"status": "not_preregistered_or_outside_fixed_size_cap", "raw_p": 1., "statistic": None}
    return _json(result)


def score_intervals(rows, calibrators, *, settings):
    """Score saved test traces against an already frozen calibration artifact."""
    lookup = {(c["panel"], c["system"], c["seed"]): c for c in calibrators["calibrators"]}
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["panel"], row["system"], row["seed"])].append(row)
    results, coverage_groups = [], defaultdict(list)
    for key, group in sorted(grouped.items()):
        cal = lookup.get(key)
        if cal is None or cal["status"] != "ok":
            results.append({"panel": key[0], "system": key[1], "seed": key[2], "status": "calibrator_unavailable"})
            continue
        require(group[0]["date"] > cal["last_date"], "Calibration must precede test endpoints")
        q = cal["q"]
        pred = np.array([r["prediction"] for r in group]); y = np.array([r["y"] for r in group])
        lower = np.maximum(np.array([math.sqrt(r["known_floor_A"]) for r in group]), np.maximum(calibrators["target_floor"], pred-q))
        upper = pred+q
        require(np.all(upper >= lower), "Interval upper bound falls below known support")
        covered = ((y >= lower) & (y <= upper)).astype(float)
        results.append({"panel": key[0], "system": key[1], "seed": key[2], "status": "ok", "n": len(group), "q": q,
                        "coverage": float(covered.mean()),
                        "mean_width": float(np.mean(upper-lower)), "mean_width_over_fit_rms": float(np.mean(upper-lower)/group[0]["rfit"]),
                        "minimum_lower": float(lower.min()), "floor_clipped_endpoints": int(np.count_nonzero(lower > pred-q)),
                        "exchangeable_coverage_guarantee": False})
        coverage_groups[len(group)].append((len(results)-1, covered))
    # Joint date resampling for equal-length traces, with bounded bootstrap
    # temporaries; do not regenerate B*T random indices once per model/seed.
    for n, group in coverage_groups.items():
        bootstrap = _coverage_bootstrap_means(np.stack([covered for _, covered in group]), settings)
        ci = np.quantile(bootstrap, [.025, .975], axis=0)
        for column, (index, _) in enumerate(group):
            results[index]["coverage_ci95_temporal"] = ci[:, column].tolist()
    return results


def _adequacy(split_rows, matrices, calibrators, config, settings):
    residual_config = config["residuals"]
    intervals = score_intervals(split_rows["test"], calibrators, settings=settings) if calibrators is not None else []
    interval_lookup = {(r["panel"], r["system"], r["seed"]): r for r in intervals}
    diagnostics, metrics, candidates = [], {}, []
    for panel, m in matrices.items():
        metrics[panel] = {}
        for split in ("fit", "selection", "test"):
            rows = [r for r in split_rows.get(split, []) if r["panel"] == panel]
            for system in METHODS:
                group = [r for r in rows if r["system"] == system]
                if not group:
                    continue
                by_seed = defaultdict(list)
                for row in group:
                    by_seed[row["seed"]].append((row["prediction"]-row["y"])**2)
                metrics[panel].setdefault(system, {})[split] = {"rmse_pooled": float(math.sqrt(np.mean(list(v for seq in by_seed.values() for v in seq)))),
                    "mean_seed_rmse": float(np.mean([math.sqrt(np.mean(v)) for v in by_seed.values()])), "n_saved_seeds": len(by_seed)}
        cp = metrics[panel].get(CANDIDATE, {})
        fit, selection, test = (cp.get(k) for k in ("fit", "selection", "test"))
        gap = selection["rmse_pooled"]-fit["rmse_pooled"] if fit and selection else None
        consistency = abs(test["rmse_pooled"]-selection["rmse_pooled"])/selection["rmse_pooled"] if test and selection and selection["rmse_pooled"] > 0 else None
        trivial = metrics[panel].get("Persistence", {}).get("test")
        gap_limit = min(config["gap"]["selection_minus_fit_upper"], config["gap"]["maximum_fraction_of_fit_rmse"]*fit["rmse_pooled"]) if fit and fit["rmse_pooled"] > 0 else None
        candidate = {"panel": panel, "selection_minus_fit_rmse": gap, "gap_limit_rmse": gap_limit,
                     "absolute_test_minus_selection_over_selection_rmse": consistency,
                     "gap_below_margin": gap is not None and gap_limit is not None and gap <= gap_limit,
                     "test_selection_consistent": consistency is not None and consistency <= config["consistency"]["maximum_absolute_test_minus_selection_over_selection_rmse"],
                     "beats_trivial_persistence": bool(test and trivial and test["rmse_pooled"] < trivial["rmse_pooled"] and test["mean_seed_rmse"] < trivial["mean_seed_rmse"]),
                     "residual_positive_tolerance_evidence": True, "empirical_interval_criteria": True}
        for ri, seed in enumerate(m["seeds"]):
            d = residual_diagnostics(m["y"], m["p"][0, ri], fit_scale=m["rfit"], hac_lags=settings["hac_lags"],
                                     bds_enabled=residual_config["bds_enabled"], bds_maximum_n=residual_config["bds_maximum_n"])
            d.update(panel=panel, system=CANDIDATE, seed=seed)
            mz = d["tests"]["mz_hac"]
            bci = d["bias_ci95_over_fit_scale"]
            mci = mz.get("coefficient_ci95")
            bias_tol = residual_config["maximum_absolute_bias_over_fit_rms"]
            intercept_tol = residual_config["maximum_absolute_mz_intercept_over_fit_rms"]
            slope_tol = residual_config["maximum_mz_slope_deviation"]
            d["positive_tolerance_evidence"] = bool(mz["status"] == "ok" and mci
                and bci[0] >= -bias_tol and bci[1] <= bias_tol
                and mci[0][0] >= -intercept_tol and mci[0][1] <= intercept_tol
                and mci[1][0] >= 1-slope_tol and mci[1][1] <= 1+slope_tol)
            candidate["residual_positive_tolerance_evidence"] &= d["positive_tolerance_evidence"]
            diagnostics.append(d)
            iv = interval_lookup.get((panel, CANDIDATE, seed))
            cal_cfg = config["calibration"]
            candidate["empirical_interval_criteria"] &= bool(iv and iv["status"] == "ok"
                and abs(iv["coverage"]-cal_cfg["level"]) <= cal_cfg["coverage_shortfall_tolerance"]+1e-12
                and iv["coverage_ci95_temporal"][0] >= cal_cfg["level"]-cal_cfg["coverage_shortfall_tolerance"]
                and iv["mean_width_over_fit_rms"] <= cal_cfg["maximum_mean_width_over_fit_rms"])
        candidate["passed"] = all(candidate[k] for k in ("gap_below_margin", "test_selection_consistent", "beats_trivial_persistence", "residual_positive_tolerance_evidence", "empirical_interval_criteria"))
        candidates.append(candidate)
    slots = [(d, name, test) for d in diagnostics for name, test in d["tests"].items() if test["status"] == "ok"]
    for (_, _, test), adj in zip(slots, holm([s[2]["raw_p"] for s in slots])):
        test["adjusted_p"] = adj
        test["reject"] = bool(adj < residual_config["diagnostic_alpha"])
    return {"candidate_by_panel": candidates, "all_saved_system_split_metrics": metrics, "intervals_by_trace": intervals,
            "residual_diagnostics": diagnostics, "diagnostic_family_size": len(slots),
            "diagnostic_adjustment": "Holm across all available candidate panel/seed MZ, LB, ARCH, JB and preregistered BDS diagnostics",
            "rejections": [{"panel": d["panel"], "seed": d["seed"], "diagnostic": name, "adjusted_p": test["adjusted_p"]} for d, name, test in slots if test["reject"]],
            "passed": bool(candidates and all(c["passed"] for c in candidates)),
            "nonrejection_is_positive_calibration_evidence": False,
            "density_pit": {"status": "not_applicable", "reason": "MSE point head/finite atoms do not define an admitted calibrated density; no fabricated PIT or log-score"}}


def _subset_matrix(matrix, seed_indices=None, time_indices=None):
    r = deepcopy(matrix)
    if seed_indices is not None:
        r["p"] = r["p"][:, seed_indices]
        r["seeds"] = [r["seeds"][i] for i in seed_indices]
    if time_indices is not None:
        r["p"] = r["p"][:, :, time_indices]
        for k in ("y", "floor", "instruments"):
            r[k] = r[k][time_indices]
        r["dates"] = [r["dates"][i] for i in time_indices]
    return r


def _robustness(matrices, config, settings, stress_predictions, protocol, evaluation_receipt=None):
    cfg = config["robustness"]
    scenarios = []
    for alt in cfg["temporal_alternatives"]:
        new_settings = settings | {k: alt[k] for k in ("hac_lags", "block_length")}
        scenarios.append((alt["id"], matrices, new_settings))
    for subset in cfg["seed_subsets"]:
        reduced = {}
        for panel, matrix in matrices.items():
            n = len(matrix["seeds"])
            indices = list(range(n//2)) if subset == "first_half" else list(range(0, n, 2)) if subset == "even_positions" else None
            require(indices is not None and len(indices) >= 2, "Unknown/too small predeclared seed subset")
            reduced[panel] = _subset_matrix(matrix, seed_indices=indices)
        scenarios.append(("seed_"+subset, reduced, settings))
    for block in cfg["temporal_blocks"]:
        reduced = {}
        for panel, matrix in matrices.items():
            n = len(matrix["dates"])
            indices = list(range(n//2)) if block == "first_half" else list(range(n//2, n)) if block == "second_half" else None
            require(indices is not None, "Unknown predeclared temporal block")
            reduced[panel] = _subset_matrix(matrix, time_indices=indices)
        scenarios.append(("time_"+block, reduced, settings))
    outputs = []
    for name, values, scenario_settings in scenarios:
        try:
            results = [_analyze_panel(m, scenario_settings) for m in values.values()]
            rule = _panel_rule(results, settings["margin"])
            favorable = all(c["mean_mse_difference"] > 0 and c["historical_mean_seed_rmse_favorable"] for r in results for c in r["comparisons"])
            passed = rule["joint_statistical_rule_pass"] if cfg["require_full_statistical_conclusion"] else favorable and not rule["additional_significant_inferiority"]
            outputs.append({"id": name, "status": "executed_on_saved_predictions", "settings": scenario_settings,
                            "results": results, "rule": rule, "passed": bool(passed)})
        except ValueError as exc:
            outputs.append({"id": name, "status": "unavailable_for_fixed_settings", "reason": str(exc), "passed": False})
    mae_results = []
    for panel, m in matrices.items():
        # The existing temporal engine expects squared losses. This exact
        # deterministic transformation yields |y-p| as its squared loss and
        # performs zero model calls. A zero margin avoids a false RMSE margin.
        p = np.sqrt(np.abs(m["p"]-m["y"][None, None]))
        result = analyze(p, np.zeros(len(m["y"])), METHODS, instruments=m["instruments"], **(settings | {"margin": 0.}))
        operational_alpha = result.get("family", {}).get("alpha", .05)
        ok = all(c["mean_mse_difference"] > 0 and c["ci95_mse_difference"][0] > 0
                 and all(test["epa"]["status"] == "ok" and test["epa"]["adjusted_p"] < operational_alpha for test in c["methods"].values())
                 for c in result["comparisons"])
        mae_results.append({"panel": panel, "loss": "absolute_error", "engine_squared_loss_is": "MAE exactly; sqrt(MAE) engine transform is not RMSE",
                            "mae": {system: result["metrics"][system]["mse"] for system in METHODS},
                            "comparisons": [{"rival": c["rival"], "mae_difference": c["mean_mse_difference"],
                                             "ci95_mae_difference": c["ci95_mse_difference"], "methods": c["methods"]} for c in result["comparisons"]],
                            "passed": bool(ok)})
    stress = []
    for name in cfg["required_saved_stress_scenarios"]:
        spec = (stress_predictions or {}).get(name)
        plan = protocol.get("robustness_stress_plans", {}).get(name)
        plan_payload = {k: v for k, v in (plan or {}).items() if k != "scenario_sha256"}
        if spec is None or not plan or plan.get("frozen_before_test") is not True or plan.get("scenario_sha256") != digest(plan_payload):
            stress.append({"id": name, "status": "unexecuted_missing_saved_predictions_or_frozen_scenario", "passed": False})
            continue
        try:
            saved = (evaluation_receipt or {}).get("stress_prediction_files", {}).get(name)
            require(saved is not None and (evaluation_receipt or {}).get("status") == "completed"
                    and evaluation_receipt.get("stress_plans_sha256") == digest(protocol.get("robustness_stress_plans", {})),
                    "Saved stress predictions need the completed controller evaluation and exact pretest scenario identity")
            stress_path = spec.get("path") if isinstance(spec, Mapping) else spec
            require(isinstance(stress_path, (str, Path)) and saved.get("sha256") == sha(stress_path), "Stress predictions differ from the controller cache checksum")
            require(saved.get("scenario_sha256") == plan["scenario_sha256"]
                    and saved.get("selected_manifest_sha256") == evaluation_receipt.get("selected_manifest_sha256")
                    and saved.get("eval_id") == evaluation_receipt.get("eval_id")
                    and saved.get("protocol_sha256") == evaluation_receipt.get("protocol_sha256")
                    and saved.get("no_refit") is True and saved.get("completed") is True,
                    "Stress cache does not bind the unchanged selected state and frozen scenario")
            rows = _records(spec)
            results = []
            for panel, m in matrices.items():
                stress_matrix = _matrix(rows, panel, m["seeds"])
                require(stress_matrix["dates"] == m["dates"] and np.array_equal(stress_matrix["y"], m["y"]), "Stress endpoints/labels differ")
                require(np.array_equal(stress_matrix["floor"], m["floor"]) and stress_matrix["rfit"] == m["rfit"], "Stress changed target support or fit-only reference scale")
                results.append(_analyze_panel(stress_matrix, settings))
            rule = _panel_rule(results, settings["margin"])
            stress.append({"id": name, "status": "executed_on_saved_predictions", "scenario": plan,
                           "results": results, "rule": rule, "passed": rule["joint_statistical_rule_pass"]})
        except ValueError as exc:
            stress.append({"id": name, "status": "invalid_saved_stress_evidence", "reason": str(exc), "passed": False})
    return {"sensitivities": outputs, "secondary_MAE": mae_results, "saved_stress_scenarios": stress,
            "new_model_forwards": 0, "nadeau_bengio": cfg["nadeau_bengio"],
            "passed": bool(outputs and mae_results and all(x["passed"] for x in outputs+mae_results+stress))}


def decide_r3(criteria):
    """Scientific conjunction only; this never grants institutional approval."""
    require(set(criteria) == set(REQUIRED_CRITERIA), "Incomplete or unknown R3 criterion set")
    return bool(all(value is True for value in criteria.values()))


def _method_admission(spec, matrices, settings):
    if spec is None:
        return {"passed": False, "reason": "No saved method-admission evidence supplied", "evidence": None}
    try:
        if isinstance(spec, Mapping) and "path" in spec:
            path = Path(spec["path"])
            if not path.is_absolute():
                path = ROOT/path
            require(spec.get("sha256") == sha(path), "Method admission report hash mismatch")
            report = read_json(path)
            evidence = {"path": str(path), "sha256": sha(path)}
        else:
            report, evidence = _load_spec(spec)
        required = {"status": "PASSED_PREDECLARED_VALIDATION", "admitted": True,
                    "family_rivals": 20, "all_three_methods": True,
                    "complete_and_partial_nulls": True, "extreme_tail_validated": True, "coverage_validated": True}
        api_parameters = inspect.signature(analyze).parameters
        reference_settings = {key: settings.get(key, api_parameters[key].default)
                              for key in ("alpha", "reference_reps", "reference_seed") if key in api_parameters}
        expected_settings = [{"R": len(m["seeds"]), "T": len(m["dates"]), "B": settings["bootstrap_reps"],
                              "block_length": settings["block_length"], "hac_lags": settings["hac_lags"],
                              "horizon": settings["horizon"], "margin": settings["margin"]} | reference_settings for m in matrices.values()]
        settings_covered = all(any(all(actual.get(k) == v for k, v in expected.items())
                                  for actual in report.get("validated_settings", [])) for expected in expected_settings)
        alpha_matches = ("alpha" not in reference_settings or report.get("operational_alpha") == reference_settings["alpha"])
        passed = (all(report.get(k) == v for k, v in required.items())
                  and report.get("statistics_source_sha256", report.get("statistics_sha256")) == sha(ROOT/"engine/statistics.py")
                  and settings_covered and alpha_matches)
        verified_bundles = []
        if passed:
            # The summary fields above are useful for a readable diagnostic,
            # but a plausible caller-authored certificate is not evidence.
            # Reuse the same archived-plan/runner/replicate and CP recount
            # required before final training, including gzip/plaintext hashes.
            require("path" in evidence, "Executed method evidence must be a hash-bound archived report")
            from .controller import _method_admission as verify_archived_validation
            for exact_settings in {digest(value): value for value in expected_settings}.values():
                verified_bundles.append(verify_archived_validation(ROOT, evidence, exact_settings))
        return {"passed": bool(passed), "reason": "Exact executed source/scope/settings required; software tests do not establish finite-sample calibration",
                "artifact": evidence, "verified_bundles": verified_bundles, "evidence": report}
    except (ValueError, OSError, RuntimeError) as exc:
        return {"passed": False, "reason": str(exc), "evidence": locals().get("report")}


def _threshold_admission(config, config_evidence, protocol):
    admission = config["threshold_admission"]
    ok = (admission.get("status") == "DEFINED_PROSPECTIVELY_NOT_EMPIRICALLY_SATISFIED" and admission.get("frozen_before_new_data") is True
          and protocol.get("adequacy", {}).get("config_sha256") == config_evidence["sha256"]
          and config.get("primary_estimand_version") == PROTOCOL_VERSION and bool(admission.get("rationale")))
    return {"passed": bool(ok), "status": admission.get("status"), "rationale": admission.get("rationale"),
            "config": config_evidence, "sources": config.get("sources", [])}


def _power_evidence(spec, final=None, panels=None, *, protocol=None, protocol_evidence=None):
    if not spec:
        return {"passed": False, "reason": "Final empirical power receipt missing"}
    try:
        if isinstance(spec, Mapping) and "path" in spec:
            path = Path(spec["path"])
            if not path.is_absolute():
                path = ROOT/path
            require(spec.get("sha256") == sha(path), "Power receipt checksum mismatch")
            report = read_json(path)
        else:
            report, _ = _load_spec(spec)
        validation = report.get("validation_result", {})
        scenarios = validation.get("scenarios", [])
        fixed = report.get("fixed_before_validation", {})
        expected_contrasts = {(p, m) for p in panels or [] for m in METHODS[1:]}
        size_matches = final is None or all(report.get("final_"+key) == final.get(key) for key in ("R", "T", "B"))
        passed = (size_matches and report.get("status") == "DIMENSIONED_CONDITIONAL_EMPIRICAL_DGP"
                  and report.get("temporal_calibration_verified") is True
                  and report.get("fixed_before_validation_sha256") == digest(fixed)
                  and validation.get("cell") == fixed.get("cell")
                  and validation.get("meets_power_target") is True and not validation.get("incomplete")
                  and validation.get("minimum_per_contrast_simultaneous_mc_lower_bound", 0) >= .9
                  and bool(scenarios) and all(len(s.get("contrasts", [])) == 80
                      and (not expected_contrasts or {(c.get("panel"), c.get("rival")) for c in s["contrasts"]} == expected_contrasts)
                      and all(c.get("simultaneous_mc_lower_bound", 0) >= .9 for c in s["contrasts"])
                      and isinstance(s.get("joint_statistical_rule_power"), (float, int)) for s in scenarios))
        verified_final_admission = None
        if passed:
            # Recompute the simultaneous CP bounds from successes/trials,
            # verify planning files and design hashes, and check the actual
            # frozen candidate/method/R/T/B contract used by the controller.
            # Numeric lower-bound claims alone must never qualify.
            require(isinstance(spec, Mapping) and "path" in spec
                    and isinstance(protocol, Mapping) and protocol_evidence is not None,
                    "Power evidence requires an archived receipt and the full frozen protocol")
            from .controller import _admit_final
            verified_final_admission = _admit_final(ROOT, protocol, protocol_evidence["sha256"], path)
        return {"passed": bool(passed), "verified_final_admission": verified_final_admission, "evidence": report}
    except (ValueError, OSError, RuntimeError) as exc:
        return {"passed": False, "reason": str(exc)}


def _sealed_evaluation(case_root, receipt, protocol_evidence, calibrator, test_rows, split_rows=None, protocol=None):
    """Verify only derived controller artifacts, never a source test partition."""
    reasons, data_reasons = [], []
    if not (receipt.get("status") == "completed" and receipt.get("evaluation_sealed", receipt.get("sealed")) is True
            and receipt.get("selected_checkpoints_verified_before_test") is True
            and receipt.get("calibrators_frozen_before_test") is True
            and calibrator is not None and receipt.get("calibrators_sha256") == calibrator["sha256"]
            and receipt.get("protocol_sha256") == protocol_evidence["sha256"]):
        reasons.append("completed evaluation, selected-state, calibrator and protocol bindings missing or inconsistent")
    selected_path = case_root/"selected_manifest.json"
    seal_path = case_root/"training_seal.json"
    try:
        require(selected_path.is_file() and seal_path.is_file(), "saved selected manifest/training seal missing")
        require(receipt.get("selected_manifest_sha256") == sha(selected_path), "selected manifest checksum differs")
        selected, seal = read_json(selected_path), read_json(seal_path)
        require(selected.get("protocol_sha256") == protocol_evidence["sha256"] and seal.get("protocol_sha256") == protocol_evidence["sha256"], "selected manifest/training seal protocol differs")
        observed = {(r["panel"], r["system"], r["seed"]) for r in test_rows}
        jobs = selected.get("jobs", [])
        expected = {(j["panel"], j["model"], "deterministic" if j["model"] in DETERMINISTIC else str(j["seed"])) for j in jobs}
        require(len(jobs) == len(expected) and expected == observed, "selected manifest does not cover exactly every saved test trace")
        require(seal.get("selected_manifest_sha256") == sha(selected_path) and seal.get("test_opened") is False, "pretest selected-state seal differs")
        cal_path = case_root/"metrics/calibrators.json"
        require(cal_path.is_file() and receipt.get("calibrators_file_sha256") == sha(cal_path)
                and seal.get("calibrators_sha256") == sha(cal_path), "calibrator file differs from pretest seal")
        for split in ("fit", "selection", "calibration"):
            path = case_root/"predictions"/(split+"_predictions.csv")
            identity = read_json(case_root/"receipts"/(split+"_prediction_identity.json"))
            require(identity.get("sha256") == sha(path) and identity.get("selected_manifest_sha256") == sha(selected_path), "saved gap/calibration predictions came from another selected state")
            if split_rows is not None:
                require(digest(_records(path)) == digest(split_rows.get(split, [])), "supplied gap/calibration rows differ from sealed saved predictions")
        require(receipt.get("predictions_sha256") == sha(case_root/"predictions/test_predictions.csv"), "saved final prediction checksum differs")
        require(digest(_records(case_root/"predictions/test_predictions.csv")) == digest(_records(test_rows)), "supplied test rows differ from sealed saved predictions")
        require(receipt.get("final_admissions_file_sha256") == sha(case_root/"receipts/final_admissions.json"), "saved final data admission identity differs")
        admissions = read_json(case_root/"receipts/final_admissions.json")
        panel_ids = {r["panel"] for r in test_rows}
        if set(admissions) != panel_ids:
            data_reasons.append("saved final admissions must cover exactly the four evaluated panels")
        else:
            for panel, admission in admissions.items():
                require(isinstance(admission, Mapping), "Invalid saved final admission metadata")
                if not (admission.get("status") == "development_admitted"
                        and admission.get("reserved_status") == "explicit_final_preparation"
                        and admission.get("rights", {}).get("status") == "approved"
                        and admission.get("exposure", {}).get("status") == "audited"
                        and admission.get("exposure", {}).get("form", {}).get("reserve_mode") in ("external_custodian", "post_protocol")):
                    data_reasons.append(panel+": genuine admitted final data required; software/exposed-development fixtures cannot confirm R3")
                if not (admission.get("admission_sha256") == digest({k: v for k, v in admission.items() if k != "admission_sha256"})
                        and receipt.get("panels", {}).get(panel, {}).get("admission_sha256") == admission.get("admission_sha256")
                        and receipt.get("panels", {}).get(panel, {}).get("test_sha256") == admission.get("partition_sha256", {}).get("test")):
                    data_reasons.append(panel+": final admission metadata/payload bindings differ")
        identity_keys = ("case_id", "plan_sha256", "protocol_sha256", "selected_manifest_sha256", "calibrators_file_sha256", "calibrators_sha256", "calibrators_frozen_before_test", "selected_checkpoints_verified_before_test", "final_admissions_file_sha256", "stress_plans_sha256", "panels")
        require(all(k in receipt for k in identity_keys) and receipt.get("eval_id") == digest({k: receipt[k] for k in identity_keys}), "unique controller evaluation identity differs")
        if protocol is not None:
            require(receipt.get("stress_plans_sha256") == digest(protocol.get("robustness_stress_plans", {})), "pretest stress plans differ from frozen protocol")
        for name, saved in receipt.get("stress_prediction_files", {}).items():
            path = case_root/"predictions"/("test_"+name+"_predictions.csv")
            require(saved.get("sha256") == sha(path), "saved stress prediction checksum differs")
    except (OSError, ValueError, KeyError) as exc:
        reasons.append(str(exc))
    return {"passed": not reasons and not data_reasons, "artifact_bindings_passed": not reasons,
            "final_data_admission_passed": not data_reasons, "reasons": reasons+data_reasons,
            "scope": "saved derived manifests, receipts and prediction bytes only; underlying checkpoints were verified by the controller before test"}


def between_panel_tests(results):
    """Separate four-panel rank summary; never a third temporal contrast.

    These panels can share economic shocks. The classical independent-dataset
    reference is therefore explicitly conditional, and not used to manufacture
    candidate/rival directional p-values or proof of broad generalization.
    """
    risks = np.array([[r["metrics"][m]["mean_seed_rmse"] for m in METHODS] for r in results])
    n, k = risks.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        q, p = stats.friedmanchisquare(*[risks[:, j] for j in range(k)])
    denominator = n*(k-1)-q
    f = (n-1)*q/denominator if np.isfinite(q) and denominator > np.finfo(float).eps else None
    global_test = {"status": "ok" if np.isfinite([q, p]).all() else "degenerate_panel_ranks",
                   "friedman_statistic": float(q) if np.isfinite(q) else None,
                   "raw_p": float(p) if np.isfinite(p) else 1., "df": k-1,
                   "iman_davenport_F": float(f) if f is not None else None,
                   "iman_davenport_p": float(stats.f.sf(f, k-1, (k-1)*(n-1))) if f is not None else None}
    posthoc = []
    for j, rival in enumerate(METHODS[1:], start=1):
        difference = risks[:, j]-risks[:, 0]
        if np.all(difference == 0):
            statistic, pvalue, status = 0., 1., "all_tied"
        else:
            # Four observations permit exact sign enumeration. scipy's exact
            # Wilcoxon with zeros is not exact; sign enumeration of absolute
            # signed-rank sums also handles zero differences transparently.
            active = difference[difference != 0]
            ranks = stats.rankdata(np.abs(active), method="average")
            observed = abs(float(np.sign(active)@ranks))
            null = np.array([abs(sum((1 if mask & (1 << i) else -1)*rank for i, rank in enumerate(ranks))) for mask in range(2**len(active))])
            statistic, pvalue, status = observed, float(np.mean(null >= observed-1e-12)), "exact_signs_of_signed_ranks"
        posthoc.append({"rival": rival, "statistic": float(statistic), "raw_p": float(pvalue), "status": status,
                        "panel_mean_seed_rmse_differences": difference.tolist(), "favorable_panel_count": int(np.count_nonzero(difference > 0))})
    for row, adj in zip(posthoc, holm([r["raw_p"] for r in posthoc])):
        row["adjusted_p"] = adj
    return {"n_panels": n, "global": global_test, "wilcoxon_signed_rank_sign_enumeration_holm": posthoc,
            "mean_ranks": stats.rankdata(risks, axis=1).mean(axis=0).tolist(),
            "alpha": results[0].get("family", {}).get("alpha", .05), "independent_panel_reference_established": False,
            "limitations": "Four economically dependent panels provide limited rank resolution. Exact sign enumeration needs independent symmetric dataset differences; this assumption is not certified. These tests are reported separately and not counted toward the three within-panel procedures."}


def assess_case(case_root, protocol, *, predictions=None, calibrators=None,
                method_admission=None, adequacy_config=None, evaluation_receipt=None,
                stress_predictions=None, output_path=None):
    """Assess previously evaluated CSVs/dicts, preserving every adverse result.

    ``predictions`` is split->saved CSV/list of rows, or panel->split->rows.
    With no argument, read only controller-derived ``predictions/*_predictions.csv``;
    no source market split or checkpoint is opened. Incomplete evidence returns
    an unaccredited report; malformed paired records raise instead of silently
    dropping observations or family members.
    """
    case_root = Path(case_root)
    protocol, protocol_evidence = _load_spec(protocol)
    config, config_evidence = _load_spec(adequacy_config or protocol.get("adequacy", {}).get("config_path", "configs/adequacy.json"))
    settings = _settings(protocol)
    panels = list(protocol.get("panel_ids", protocol.get("panels", [])))
    require(len(panels) == 4 and len(set(panels)) == 4, "Exactly four predeclared panels required")
    require(tuple(protocol.get("family", {}).get("method_order", METHODS)) == METHODS, "Protocol changed the frozen ordered 21-system family")
    if predictions is None:
        predictions = {split: case_root/"predictions"/(split+"_predictions.csv") for split in SPLITS if (case_root/"predictions"/(split+"_predictions.csv")).is_file()}
    split_rows = defaultdict(list)
    if set(predictions).issubset(set(SPLITS)):
        for split, source in predictions.items():
            split_rows[split] = _records(source)
    else:
        for panel, splits in predictions.items():
            require(panel in panels, "Unknown assessment panel")
            for split, source in splits.items():
                require(split in SPLITS, "Unknown prediction partition")
                split_rows[split].extend(_records(source, panel))
        for split in split_rows:
            split_rows[split] = _records(split_rows[split])
    require(split_rows.get("test"), "Already evaluated saved test predictions required; assessment never opens a market reserve")
    test_rows = split_rows["test"]
    seeds = sorted({r["seed"] for r in test_rows if r["system"] == CANDIDATE}, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
    final = protocol.get("final", {})
    expected_seeds = final.get("seed_ids", final.get("final_seed_ids"))
    if expected_seeds is not None:
        require(set(seeds) == {str(s) for s in expected_seeds}, "Final prediction seeds differ from frozen prefix")
        seeds = [str(s) for s in expected_seeds]
    seed_bank = protocol.get("seed_bank")
    seed_prefix_verified = False
    if seed_bank:
        bank_path = Path(seed_bank["path"])
        if not bank_path.is_absolute():
            bank_path = ROOT/bank_path
        require(seed_bank.get("sha256") == sha(bank_path), "Seed bank checksum mismatch")
        prefix = [str(s) for s in read_json(bank_path)["seed_ids"][:len(seeds)]]
        require(set(seeds) == set(prefix), "Final seeds must be the frozen bank prefix, never performance-selected IDs")
        seeds, seed_prefix_verified = prefix, True
    matrices = {panel: _matrix(test_rows, panel, seeds) for panel in panels}
    require({r["panel"] for r in test_rows} == set(panels), "Unexpected saved test panel")
    for split in ("fit", "selection", "calibration"):
        if split_rows.get(split):
            for panel in panels:
                _matrix(split_rows[split], panel, seeds)
    # No cross-partition reuse. Purged input support is controller/data evidence;
    # checking endpoint disjointness here cannot itself prove a support purge.
    for panel in panels:
        previous = None
        for split in SPLITS:
            dates = sorted({r["date"] for r in split_rows.get(split, []) if r["panel"] == panel})
            if dates:
                require(previous is None or dates[0] > previous, "Saved prediction partitions overlap or are out of order")
                previous = dates[-1]
    if calibrators is None:
        for path in (case_root/"metrics/calibrators.json", case_root/"calibrators.json"):
            if path.is_file():
                calibrators = path
                break
    calibration_object, calibration_evidence = (None, None)
    if calibrators is not None:
        require(split_rows.get("calibration"), "Calibration predictions required to verify frozen calibrator identity")
        calibration_object, calibration_evidence = _verified_calibrators(calibrators, split_rows["calibration"])
        require(calibration_object["level"] == config["calibration"]["level"], "Calibrator level differs from frozen adequacy config")
    if evaluation_receipt is None and (case_root/"evaluation_receipt.json").is_file():
        evaluation_receipt = case_root/"evaluation_receipt.json"
    receipt, _ = _load_spec(evaluation_receipt) if evaluation_receipt is not None else ({}, {})
    seal_verification = _sealed_evaluation(case_root, receipt, protocol_evidence, calibration_object, test_rows, split_rows, protocol)
    if stress_predictions is None:
        stress_predictions = {name: {"path": str(case_root/"predictions"/("test_"+name+"_predictions.csv")), "sha256": item["sha256"]}
            for name, item in receipt.get("stress_prediction_files", {}).items()}
    results = [_analyze_panel(matrix, settings) for matrix in matrices.values()]
    rule = _panel_rule(results, settings["margin"])
    adequate = _adequacy(split_rows, matrices, calibration_object, config, settings)
    robust = _robustness(matrices, config, settings, stress_predictions, protocol, receipt)
    method = _method_admission(method_admission or final.get("method_admission") or protocol.get("method_validation"), matrices, settings)
    threshold = _threshold_admission(config, config_evidence, protocol)
    power = _power_evidence(final.get("power_receipt"), final, panels,
                            protocol=protocol, protocol_evidence=protocol_evidence)
    between = between_panel_tests(results)
    complete = seed_prefix_verified and 10 <= len(seeds) <= 4091 and all(r["family"]["rivals"] == 20 for r in results)
    size_fixed = (final.get("closed") is True and final.get("R") == len(seeds)
                  and final.get("B") == settings["bootstrap_reps"]
                  and all(final.get("T") == len(m["dates"]) for m in matrices.values()))
    contrasts = rule["contrast_pass"]
    criteria = {
        "complete_four_panel_family": bool(complete),
        "protocol_closed_and_final_size_fixed": bool(size_fixed),
        "sealed_evaluation_evidence": seal_verification["passed"], "method_calibration_admitted": method["passed"],
        "adequacy_thresholds_admitted": threshold["passed"],
        "all_primary_rivals_superior": bool(all(contrasts[0])),
        "historical_primary_mean_rmse_favorable": bool(rule["historical_main_all_mean_rmse_favorable"]),
        "additional_two_of_three_per_rival": bool(rule["historical_additional_mean_rmse_two_of_three"] and rule["additional_statistical_two_of_three"]),
        "no_significant_additional_inferiority": not rule["additional_significant_inferiority"],
        "effects_intervals_and_practical_margin": bool(all(contrasts[0])),
        "power_per_contrast_at_least_90_and_joint_reported": power["passed"],
        "between_panel_tests_reported": bool(between["n_panels"] == 4 and len(between["wilcoxon_signed_rank_sign_enumeration_holm"]) == 20),
        "adequacy": adequate["passed"], "robustness": robust["passed"],
    }
    recovery_history = read_json(case_root/"recovery_history.json") if (case_root/"recovery_history.json").is_file() else []
    require(isinstance(recovery_history, list), "Invalid saved recovery history")
    uncertain_compute = bool(receipt.get("unobserved_remote_compute_possible", False)
                             or any(r.get("unobserved_remote_compute_possible") is True for r in recovery_history))
    report = {"schema_version": 1, "version": "saved-prediction-r3-assessment-v1", "candidate": CANDIDATE,
              "protocol": protocol_evidence, "panels": panels, "systems": list(METHODS), "seed_ids": seeds, "seed_bank_prefix_verified": seed_prefix_verified,
              "partition_prediction_hashes": {s: digest(r) for s, r in split_rows.items()},
              "calibrators": calibration_evidence, "calibrators_payload_sha256": calibration_object["sha256"] if calibration_object else None,
              "evaluation_seal_verification": seal_verification,
              "compute_uncertainty": {"unobserved_remote_compute_possible": uncertain_compute,
                  "recovery_history": recovery_history,
                  "cost_comparability_claimed": False,
                  "limitation": "Charged active time and optimizer steps are lower bounds when an open-session remote recovery may omit unuploaded compute; no exact total-cost claim is made."},
              "panel_statistics": {p: r for p, r in zip(panels, results)}, "joint_rule": rule,
              "adequacy": adequate, "robustness": robust, "method_admission": method,
              "threshold_admission": threshold, "power": power, "between_panel_tests": between, "criteria": criteria,
              "r3_scientific_criteria_satisfied": decide_r3(criteria),
              "institutional_approval_claimed": False, "new_model_fits": 0, "new_model_forwards": 0,
              "limitations": ["Fixed-seed temporal inference conditions on fitted models; it does not integrate training uncertainty.",
                              "Four panel means and Holm families do not establish economic independence or universal generalization. GW rejection alone is not directional mean inferiority.",
                              "Marginal bootstrap intervals rely on weak dependence/stationarity; finite-sample method admission is separately required.",
                              "No density/PIT calibration is inferred from the MSE head. Dependent-time conformal radii have no exact exchangeable coverage guarantee.",
                              "Fit/selection gap requires saved predictions from the same selected state in original units; training loss/SAE auxiliary loss is not substituted.",
                              "Numerical adequacy tolerances are prospective scientific design choices; the final protocol must bind the exact configuration before test opening.",
                              "A scientific conjunction is computed by code; institutional/academic approval is not claimed."]}
    report = _json(report)
    if output_path is not None:
        write_json(output_path, report, immutable=True)
    return report
