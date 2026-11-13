"""Read-only, explicitly post hoc F2a S8 comparisons of frozen predictions.

No model loading/fitting or recomputation of the twenty original hypotheses.
Temporal loss is the daily mean of *individual* squared errors over fixed seeds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy import stats

from e2_reduced_protocol import SYSTEMS, NEURAL, CLASSICAL

VERSION = "e2-f2a-s8-posthoc-comparisons-v1"
INPUT_SHA256 = "65ae74b547a6073b2c286dea8039a2e7a1b1f6063b1d9ac0eb760c9ce727d187"
SEEDS = tuple(range(271801, 271807))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def array(value, ndim=1):
    result = np.asarray(value, dtype=float)
    require(result.ndim == ndim and result.size and np.isfinite(result).all(),
            "Expected finite nonempty numeric array")
    return result


def holm(pvalues):
    """Fixed family including unavailable hypotheses conservatively as p=1."""
    available = [(float(p), i) for i, p in enumerate(pvalues) if p is not None]
    require(all(0 <= p <= 1 and math.isfinite(p) for p, _ in available), "Invalid p")
    adjusted = [None] * len(pvalues)
    previous = 0.
    for rank, (p, i) in enumerate(sorted(available)):
        previous = max(previous, min(1., (len(pvalues)-rank)*p))
        adjusted[i] = previous
    return adjusted


def daily_loss(y, predictions):
    y, predictions = array(y), array(predictions, 2)
    require(predictions.shape[1] == len(y), "Predictions and targets must align")
    return np.mean((predictions-y[None, :])**2, axis=0)


def hac_mean_variance(difference, maxlags=14):
    """Bartlett HAC, gamma_l denominator T; no extra OLS df correction."""
    d = array(difference)
    n = len(d)
    require(type(maxlags) is int and 0 <= maxlags < n, "Invalid HAC bandwidth")
    centered = d-d.mean()
    long_run = float(centered @ centered / n)
    for lag in range(1, maxlags+1):
        long_run += 2*(1-lag/(maxlags+1))*float(centered[lag:] @ centered[:-lag]/n)
    require(long_run >= -1e-12*float(np.mean(d*d)), "Negative HAC variance")
    return max(0., long_run)/n


def dm_hln(loss_a, loss_b, *, maxlags=14, horizon=1):
    """Two-sided DM-HLN with t_(T-1); positive d means A has higher MSE."""
    a, b = array(loss_a), array(loss_b)
    require(a.shape == b.shape, "Loss paths must align")
    n = len(a)
    require(type(horizon) is int and 1 <= horizon < n, "Invalid forecast horizon")
    d = a-b
    mean = float(d.mean())
    correction = math.sqrt((n+1-2*horizon+horizon*(horizon-1)/n)/n)
    result = dict(mean_MSE_difference=mean, n_dates=n, HAC_lags=maxlags,
                  horizon=horizon, HLN_factor=correction, p_raw=None,
                  statistic=None, CI95_low=None, CI95_high=None,
                  reference="Student_t_df_%d" % (n-1), status="estimable",
                  fraction_dates_A_lower_loss=float(np.mean(d < 0)+.5*np.mean(d == 0)))
    if np.array_equal(a, b):
        result.update(status="identical_loss_path", statistic=0., p_raw=1.,
                      variance_of_mean=0., CI95_low=0., CI95_high=0.)
        return result
    variance = hac_mean_variance(d, maxlags)
    result["variance_of_mean"] = variance
    tolerance = (64*np.finfo(float).eps*max(float(np.max(np.abs(d))), np.finfo(float).tiny))**2
    if variance <= tolerance:
        result.update(status="not_estimable_zero_variance",
                      reason="Nonidentical constant differential has no estimable sampling variance")
        return result
    se = math.sqrt(variance)
    statistic = correction*mean/se
    radius = float(stats.t.ppf(.975, n-1))*se/correction
    result.update(statistic=statistic, p_raw=float(2*stats.t.sf(abs(statistic), n-1)),
                  CI95_low=mean-radius, CI95_high=mean+radius,
                  standardized_daily_difference=mean/float(np.std(d, ddof=1)))
    return result


def circular_indices(n, block_days, draws, seed):
    require(type(n) is int and n > 1 and type(block_days) is int and 1 <= block_days <= n,
            "Invalid circular block design")
    require(type(draws) is int and draws > 1, "Need multiple bootstrap draws")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(draws, math.ceil(n/block_days)))
    return ((starts[:, :, None]+np.arange(block_days)) % n).reshape(draws, -1)[:, :n]


def bootstrap_loss_means(losses, *, draws=10000, block_days=14, seed=202611132):
    losses = array(losses, 2)  # dates x systems
    indices = circular_indices(len(losses), block_days, draws, seed)
    result = np.empty((draws, losses.shape[1]))
    for start in range(0, draws, 256):
        result[start:start+256] = losses[indices[start:start+256]].mean(axis=1)
    return result


def mcs_range(losses, names, *, alpha=.1, draws=10000, block_days=14, seed=202611132,
              bootstrap_means=None):
    """Hansen T_R EPA test and e_R elimination with centered block bootstrap.

    SE_ij^2 = mean_b[(dbar*_ij-dbar_ij)^2]. Common indices are used across
    every model and elimination step. The bootstrap maximum is centered under
    EPA; ranking uncentered bootstrap losses would not implement the MCS test.
    """
    losses = array(losses, 2)
    require(losses.shape[1] == len(names) and len(set(names)) == len(names) and len(names) >= 2,
            "Unique model names must match loss columns")
    require(0 < alpha < 1, "Invalid MCS alpha")
    means = losses.mean(axis=0)
    boot = (bootstrap_loss_means(losses, draws=draws, block_days=block_days, seed=seed)
            if bootstrap_means is None else array(bootstrap_means, 2))
    require(boot.shape == (draws, len(names)), "Invalid bootstrap means")
    centered = boot-means
    differences = means[:, None]-means[None, :]
    null_differences = centered[:, :, None]-centered[:, None, :]
    se = np.sqrt(np.mean(null_differences**2, axis=0))
    tolerance = 64*np.finfo(float).eps*max(float(np.max(np.abs(losses))), np.finfo(float).tiny)
    zero = se <= tolerance
    nonidentical_zero = zero & (np.abs(differences) > tolerance)
    if np.any(nonidentical_zero):
        return dict(status="not_estimable_zero_pair_variance", steps=[], membership=[],
                    retained=list(names), reason="Nonzero mean difference has zero bootstrap pair variance")
    safe = np.where(zero, 1., se)
    observed_t = np.where(zero, 0., differences/safe)
    null_t = np.where(zero[None, :, :], 0., null_differences/safe[None, :, :])
    active = list(range(len(names)))
    model_p = np.ones(len(names))
    elimination_step = [None]*len(names)
    steps = []
    cumulative_p = 0.
    # Continue the sequence for monotone per-model MCS p-values, even after the
    # alpha-level retained set is reached; never eliminate a retained model at alpha.
    while len(active) > 1:
        ii, jj = np.triu_indices(len(active), 1)
        ai, aj = np.asarray(active)[ii], np.asarray(active)[jj]
        observed = float(np.max(np.abs(observed_t[ai, aj])))
        null_max = np.max(np.abs(null_t[:, ai, aj]), axis=1)
        pvalue = float((1+np.count_nonzero(null_max >= observed))/(draws+1))
        cumulative_p = max(cumulative_p, pvalue)
        # e_R uses maximum standardized disadvantage, not mean-loss ordering.
        disadvantages = observed_t[np.ix_(active, active)].copy()
        np.fill_diagonal(disadvantages, -np.inf)
        candidate = active[int(np.argmax(np.max(disadvantages, axis=1)))]
        step = len(steps)+1
        steps.append(dict(step=step, n_models=len(active), models=[names[i] for i in active],
                          T_R=observed, p_raw=pvalue, p_monotone=cumulative_p,
                          EPA_reject_at_alpha=pvalue < alpha,
                          candidate_e_R=names[candidate],
                          excluded_at_alpha=cumulative_p < alpha))
        model_p[candidate] = cumulative_p
        elimination_step[candidate] = step
        active.remove(candidate)
    model_p[active[0]] = 1.
    membership = [dict(model=name, MCS_pvalue=float(model_p[i]),
                       retained_at_alpha=bool(model_p[i] >= alpha), confidence_level=1-alpha,
                       elimination_order_step=elimination_step[i],
                       mean_MSE=float(means[i])) for i, name in enumerate(names)]
    return dict(status="estimable", alpha=alpha, confidence=1-alpha, draws=draws,
                block_days=block_days, seed=seed, statistic="T_R", elimination="e_R",
                variance="mean squared centered bootstrap pair differences",
                common_bootstrap_indices=True, null_centered=True,
                sequence_continued_for_model_pvalues=True,
                steps=steps, membership=membership,
                retained=[r["model"] for r in membership if r["retained_at_alpha"]])


def conditional_seed_ranks(scores, names, *, draws=10000, seed=202611133):
    """Optimization-only diagnostic: seed blocks x neural systems, fixed market.

    The block permutation requires exchangeable model rankings under H0. Common
    numeric seeds do not certify matching or exchangeability; neither seeds nor
    posthoc calendar divisions become independent markets/origins.
    """
    scores = array(scores, 2)
    n, k = scores.shape
    require(n >= 3 and k >= 3 and k == len(names), "Need aligned neural seed blocks")
    require(type(draws) is int and draws > 1, "Invalid permutation draw count")
    ranks = stats.rankdata(scores, axis=1)
    correction = 1-sum(sum(c**3-c for c in np.unique(row, return_counts=True)[1])
                       for row in scores)/(n*k*(k*k-1))
    require(correction > 0, "All within-block scores tied; rank test not estimable")
    friedman = stats.friedmanchisquare(*scores.T)
    q = float(friedman.statistic)
    fstat = (n-1)*q/(n*(k-1)-q) if q < n*(k-1) else math.inf
    fp = float(stats.f.sf(fstat, k-1, (k-1)*(n-1)))
    rng = np.random.default_rng(seed)
    exceed = 0
    for start in range(0, draws, 512):
        size = min(512, draws-start)
        # Independent uniform model-label permutations within each seed row.
        order = np.argsort(rng.random((size, n, k)), axis=2)
        permuted = np.take_along_axis(np.broadcast_to(ranks, (size, n, k)), order, axis=2)
        qstar = (12*n/(k*(k+1))*np.sum((permuted.mean(axis=1)-(k+1)/2)**2, axis=1))/correction
        exceed += int(np.count_nonzero(qstar >= q-1e-12))
    rows = []
    if fp < .05:
        for i, j in itertools.combinations(range(k), 2):
            d = scores[:, i]-scores[:, j]
            if np.all(d == 0):
                statistic, pvalue, status = 0., 1., "identical_seed_scores"
            elif np.any(d == 0) or len(np.unique(np.abs(d))) != len(d):
                # Do not silently call a tied/zero signed-rank result 'exact'.
                statistic, pvalue, status = None, None, "not_estimable_exact_ties_or_zeros"
            else:
                test = stats.wilcoxon(d, alternative="two-sided", method="exact")
                statistic, pvalue, status = float(test.statistic), float(test.pvalue), "exact"
            rows.append(dict(model_A=names[i], model_B=names[j], statistic=statistic,
                             p_raw=pvalue, status=status,
                             mean_seed_RMSE_difference=float(d.mean())))
        for row, adjusted in zip(rows, holm([r["p_raw"] for r in rows])):
            row.update(p_Holm78=adjusted, reject_Holm05=adjusted is not None and adjusted < .05)
    summary = dict(scope="conditional_optimization_diagnostic_not_independent_markets",
                   blocks=n, systems=k, metric="per_seed_RMSE", Friedman_Q=q,
                   tie_correction=correction, chi2_df=k-1, chi2_p=float(friedman.pvalue),
                   Iman_Davenport_F=fstat if math.isfinite(fstat) else None,
                   Iman_Davenport_F_is_infinite=not math.isfinite(fstat),
                   Iman_Davenport_df=[k-1, (k-1)*(n-1)], Iman_Davenport_p=fp,
                   permutation_p=(1+exceed)/(draws+1), permutation_draws=draws,
                   permutation_seed=seed, posthoc_gate="Iman_Davenport_p_lt_0.05",
                   posthoc_executed=bool(rows), posthoc_family_size=k*(k-1)//2,
                   minimum_exact_bilateral_p_no_ties=2/2**n,
                   assumptions=["Independent seed vectors conditional on fixed data and recipes",
                       "Within-block model-rank exchangeability for permutation null",
                       "Same seed labels alone do not establish matching or shared random effects",
                       "Pairwise signed-rank inference additionally requires symmetric seed differences"],
                   deterministic_comparators_excluded=True,
                   population_Friedman_over_independent_series_or_origins=False)
    rank_rows = [dict(model=name, mean_rank=float(ranks[:, i].mean()),
                      min_rank=float(ranks[:, i].min()), max_rank=float(ranks[:, i].max()),
                      mean_seed_RMSE=float(scores[:, i].mean())) for i, name in enumerate(names)]
    return summary, rank_rows, rows


def load_frozen(path):
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == INPUT_SHA256, "Frozen input SHA256 mismatch")
    payload = json.loads(raw)
    require(len(payload["dates"]) == 608 and all(payload["mask"]), "Expected complete frozen calendar")
    require(len(set(payload["dates"])) == 608 and payload["dates"] == sorted(payload["dates"]),
            "Date alignment is invalid")
    require(np.all(np.diff(np.asarray(payload["dates"], dtype="datetime64[D]")) == np.timedelta64(1, "D")),
            "Temporal lags require contiguous calendar days")
    require(len(payload["original_analysis"]["primary_contrasts"]) == 20, "Original family changed")
    y = array(payload["y_true"])
    require(len(y) == 608, "Expected 608 targets")
    grouped = {}
    for name in SYSTEMS:
        records = [r for r in payload["predictors"] if r["model"] == name]
        expected = SEEDS if name in NEURAL else ("deterministic",)
        require(tuple(r["seed"] for r in records) == expected, "Frozen model/seed support changed")
        grouped[name] = array([r["prediction_original"] for r in records], 2)
        require(grouped[name].shape == (len(expected), 608), "Forecast support changed")
    require(len(payload["predictors"]) == 81 and set(r["model"] for r in payload["predictors"]) == set(SYSTEMS),
            "Frozen system support changed")
    original = {r["model"]: r for r in payload["original_analysis"]["table_A2"]}
    maximum_rmse_delta = 0.
    for name, P in grouped.items():
        rmses = np.sqrt(np.mean((P-y)**2, axis=1))
        expected = original[name].get("RMSE_per_seed")
        if expected is None:
            expected = [original[name]["RMSE"]]
        maximum_rmse_delta = max(maximum_rmse_delta, float(np.max(np.abs(rmses-expected))))
    require(maximum_rmse_delta <= 1e-10, "Stored per-seed original RMSE verification failed")
    return payload, y, grouped, maximum_rmse_delta


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n")


def write_csv(path, rows):
    require(bool(rows), "Do not write empty result CSV")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def run(input_path, output_dir):
    require((output_dir/"analysis_plan.json").exists(), "Write the post hoc plan before computation")
    plan_bytes = (output_dir/"analysis_plan.json").read_bytes()
    payload, y, grouped, rmse_delta = load_frozen(input_path)
    original_before = digest(payload["original_analysis"])
    losses = np.column_stack([daily_loss(y, grouped[name]) for name in SYSTEMS])
    system_rows = []
    for i, name in enumerate(SYSTEMS):
        P = grouped[name]
        system_rows.append(dict(model=name, frozen_trajectories=len(P), dates=len(y),
                               mean_MSE=float(losses[:, i].mean()),
                               sqrt_mean_MSE=float(np.sqrt(losses[:, i].mean())),
                               original_mean_seed_RMSE=float(np.sqrt(np.mean((P-y)**2, axis=1)).mean()),
                               MSE_ensemble_prediction=float(np.mean((P.mean(axis=0)-y)**2))))
    dm_rows = [dict(model_A=SYSTEMS[i], model_B=SYSTEMS[j],
                    **dm_hln(losses[:, i], losses[:, j]))
               for i, j in itertools.combinations(range(len(SYSTEMS)), 2)]
    for row, adjusted in zip(dm_rows, holm([r["p_raw"] for r in dm_rows])):
        row.update(p_Holm120=adjusted, reject_Holm05=adjusted is not None and adjusted < .05)
    mcs = mcs_range(losses, SYSTEMS)
    require(mcs["status"] == "estimable", "MCS is not estimable: "+mcs["status"])
    seed_scores = np.column_stack([np.sqrt(np.mean((grouped[name]-y)**2, axis=1)) for name in NEURAL])
    rank_test, rank_rows, posthoc = conditional_seed_ranks(seed_scores, NEURAL)
    original_after = digest(json.loads(input_path.read_bytes())["original_analysis"])
    require(original_before == original_after and hashlib.sha256(input_path.read_bytes()).hexdigest() == INPUT_SHA256,
            "Read-only frozen input changed")
    summary = dict(version=VERSION, post_hoc=True, original_benchmark_repeated=False,
                   scientific_fit=False, model_inference=False, seeds_resampled_in_temporal_tests=False,
                   input_sha256=INPUT_SHA256, original_analysis_sha256_before=original_before,
                   original_analysis_sha256_after=original_after, original_primary_count=20,
                   original_primary_contrasts_sha256=digest(payload["original_analysis"]["primary_contrasts"]),
                   plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
                   maximum_per_seed_RMSE_verification_difference=rmse_delta,
                   loss="mean fixed-seed individual squared error per date; original sigma7 units squared",
                   original_primary_metric="mean seed RMSE; unchanged, not the temporal loss estimand",
                   scope="known single market chronology 2025-01-01 through 2026-08-31",
                   dm=dict(family_size=120, alternative="two-sided", maxlags=14, horizon=1,
                           reject_Holm05=sum(r["reject_Holm05"] for r in dm_rows),
                           unavailable=sum(r["p_raw"] is None for r in dm_rows)),
                   mcs=mcs, conditional_seed_rank_test=rank_test,
                   conditional_posthoc_Holm_rejections=sum(r["reject_Holm05"] for r in posthoc),
                   assumptions=["Post hoc exploratory p-values conditional on already known chronology",
                       "Weak dependence/local stationarity, adequate HAC14 and circular block14 span",
                       "HLN h1 correction combined with longer HAC14 is an approximate sensitivity analysis",
                       "Sigma7 target overlap motivates serial adjustment, not horizon7",
                       "No inference across independent markets, no equivalence claim, no overfit clearance"],
                   runtime=dict(Python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__))
    write_csv(output_dir/"system_loss_summary.csv", system_rows)
    write_csv(output_dir/"dm_hln_hac14_all120.csv", dm_rows)
    write_csv(output_dir/"mcs_steps.csv", mcs["steps"])
    write_csv(output_dir/"mcs_membership.csv", mcs["membership"])
    write_csv(output_dir/"conditional_seed_ranks.csv", rank_rows)
    if posthoc:
        write_csv(output_dir/"conditional_wilcoxon_holm78.csv", posthoc)
    write_json(output_dir/"comparisons_summary.json", summary)
    manifest = dict(version=VERSION, input_sha256=INPUT_SHA256,
                    source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    test_code_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/"tests"/"test_e2_f2a_s8_comparisons.py").read_bytes()).hexdigest(),
                    files={p.name:dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                           for p in sorted(output_dir.iterdir()) if p.is_file() and p.name != "manifest.json"})
    write_json(output_dir/"manifest.json", manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input, args.output)
    print(json.dumps(dict(version=VERSION, DM_Holm120_rejections=result["dm"]["reject_Holm05"],
                          MCS90=result["mcs"]["retained"],
                          seed_posthoc_rejections=result["conditional_posthoc_Holm_rejections"])))


if __name__ == "__main__":
    main()
