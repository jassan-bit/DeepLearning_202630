"""Independent numerical cross-check of current Filipe-CV statistics.

This verifier reads stored development predictions, registered metadata and
December stopping logs. It never imports a trainer, fits a model or opens raw
market/reserved-test files. Recalculation uses SciPy/reference definitions rather
than importing the supplement or its numerical helper implementation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import bds

ROOT = Path(__file__).resolve().parents[1]
CV = ROOT / "results/dl_r2_cv_20261104"
SUPPLEMENT = ROOT / "results/filipe_statistics_20261104"
METHODS = ("MLP", "LSTM", "CNN1D", "Persistencia")
SEEDS = tuple(range(4101, 4111))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def near(actual, expected, *, rtol=1e-10, atol=1e-13):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)


def current_path(recorded):
    recorded = Path(recorded)
    if recorded.is_relative_to(ROOT):
        return recorded
    if not recorded.is_absolute():
        return ROOT / recorded
    for marker in ("results", "experiments", "scripts"):
        if marker in recorded.parts:
            return ROOT.joinpath(*recorded.parts[recorded.parts.index(marker):])
    raise ValueError("Unrecognized recorded path")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/tmp/filipe_statistics_independent_audit.json"))
    args = parser.parse_args()
    report = json.loads((SUPPLEMENT / "summary.json").read_text())
    protocol_path = ROOT / "experiments/filipe_statistics_20261104/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    assert pd.Timestamp(protocol["registered_at_utc"]) < pd.Timestamp(report["created_at_utc"])
    assert not protocol["test_evaluation_allowed"] and not protocol["new_training_allowed"]
    assert not protocol["raw_market_reads_allowed"]
    assert report["new_model_fits"] == report["raw_market_files_opened"] == 0
    assert report["reserved_test_evaluated"] is False
    assert sha(protocol_path) == report["protocol"]["sha256"]
    hashes_checked = 0
    for name, expected in protocol["source_sha256"].items():
        assert sha(ROOT / name) == expected, name
        hashes_checked += 1
    assert sha(ROOT / "scripts/analyze_filipe_statistics.py") == report["analysis_script_sha256"]
    assert sha(ROOT / "scripts/analyze_dl_benchmark.py") == report["numeric_helper_sha256"]
    for name, expected in report["outputs"].items():
        path = (SUPPLEMENT / name).resolve()
        assert path.is_relative_to(SUPPLEMENT.resolve()) and sha(path) == expected, name
        hashes_checked += 1

    frame = pd.read_csv(CV / "predictions.csv")
    runs = pd.read_csv(CV / "runs.csv")
    dates = pd.date_range("2024-01-01", "2024-12-31").strftime("%Y-%m-%d").tolist()
    assert len(frame) == 21960 and len(runs) == 60
    assert set(frame.variant) == {"absolute_log", "relative_window"}
    assert not frame.duplicated(["variant", "model", "seed", "outcome_day"]).any()
    cube = np.empty((4, 10, 366), dtype=float)
    absolute = np.empty((3, 10, 366), dtype=float)
    target = persistence = None
    for variant in ("relative_window", "absolute_log"):
        for i, model in enumerate(METHODS[:3]):
            for j, seed in enumerate(SEEDS):
                part = frame[(frame.variant == variant) & (frame.model == model)
                             & (frame.seed == seed)].sort_values("outcome_day")
                assert part.outcome_day.tolist() == dates and part.history_days.eq(7).all()
                assert part.fold_id.tolist() == [value[:7] for value in dates]
                y, p = part.y_true.to_numpy(), part.persistence_pred.to_numpy()
                if target is None:
                    target, persistence = y, p
                np.testing.assert_array_equal(target, y)
                np.testing.assert_array_equal(persistence, p)
                if variant == "relative_window":
                    cube[i, j] = part.y_pred
                else:
                    absolute[i, j] = part.y_pred
    cube[3] = persistence
    losses = (cube-target)**2
    score = np.sqrt(losses.mean(axis=2))
    absolute_score = np.sqrt(((absolute-target)**2).mean(axis=2))
    paired = pd.read_csv(SUPPLEMENT / "paired_tests.csv")
    global_tests = pd.read_csv(SUPPLEMENT / "global_seed_tests.csv")
    assert len(paired) == 9 and len(global_tests) == 12
    differences = [score[a]-score[b] for a, b in combinations(range(4), 2)]
    differences.extend([1-score[i]/score[3, 0] for i in range(3)])
    pvalues = []
    for index, difference in enumerate(differences):
        boundary = .05 if index >= 6 else 0.
        alternative = "greater" if index >= 6 else "two-sided"
        test = stats.ttest_1samp(difference, popmean=boundary, alternative=alternative)
        sd = difference.std(ddof=1)
        half = stats.t.ppf(.975, 9)*sd/np.sqrt(10)
        row = paired.iloc[index]
        near([row.mean_difference, row.sd_difference, row.t_statistic, row.p_value,
              row.ci95_lower, row.ci95_upper, row.paired_cohen_d_z,
              row.standardized_distance_from_null],
             [difference.mean(), sd, test.statistic, test.pvalue,
              difference.mean()-half, difference.mean()+half,
              difference.mean()/sd, (difference.mean()-boundary)/sd])
        diagnostic = stats.shapiro(difference)
        near([row.shapiro_W, row.shapiro_p], [diagnostic.statistic, diagnostic.pvalue])
        pvalues.append(float(test.pvalue))
    near(paired.p_holm9, multipletests(pvalues, method="holm")[1])
    for i in range(3):
        difference = score[i]-absolute_score[i]
        pvalues.append(float(stats.ttest_1samp(difference, popmean=0.).pvalue))
    near(global_tests.p_value, pvalues)
    near(global_tests.p_holm, multipletests(pvalues, method="holm")[1])
    near(paired.p_holm12, global_tests.p_holm.iloc[:9])
    friedman = stats.friedmanchisquare(*score[:3])
    near([report["friedman"]["statistic"], report["friedman"]["p_value"]],
         [friedman.statistic, friedman.pvalue])

    sensitivity = pd.read_csv(SUPPLEMENT / "design_sensitivity.csv")
    superiority = pd.read_csv(SUPPLEMENT / "probability_superiority.csv")
    zsum = stats.norm.ppf(1-(.05/12)/2)+stats.norm.ppf(.8)
    for index, (a, b) in enumerate(combinations(range(4), 2)):
        d = score[a]-score[b]
        sd = d.std(ddof=1)
        row = sensitivity.iloc[index]
        near([row.sd_seed_RMSE_difference, row.normal_approximation_MDE0],
             [sd, zsum*sd/np.sqrt(10)])
        if b == 3:
            delta = .05*score[b, 0]
            near(row.planning_delta_vs_zero, delta)
            assert row.normal_approximation_R_for_delta == math.ceil((zsum*sd/delta)**2)
        else:
            assert pd.isna(row.planning_delta_vs_zero) and pd.isna(row.normal_approximation_R_for_delta)
        wins, ties = int(np.sum(d < 0)), int(np.sum(d == 0))
        row = superiority.iloc[index]
        assert row.strict_wins_a == wins and row.ties == ties
        near(row.probability_superiority_half_ties, (wins+.5*ties)/10)
        if not ties:
            interval = stats.binomtest(wins, 10).proportion_ci(confidence_level=.95, method="exact")
            near([row.ci95_lower, row.ci95_upper], [interval.low, interval.high])

    dm = pd.read_csv(SUPPLEMENT / "temporal_dm.csv")
    temporal = losses.mean(axis=1)
    primary_p = []
    for row in dm.itertuples():
        a, b = METHODS.index(row.model_a), METHODS.index(row.model_b)
        d = temporal[a]-temporal[b]
        centered, lag = d-d.mean(), int(row.HAC_lag)
        variance = np.dot(centered, centered)/366
        variance += 2*sum((1-k/(lag+1))*np.dot(centered[k:], centered[:-k])/366
                          for k in range(1, lag+1))
        statistic = d.mean()/np.sqrt(variance/366)
        corrected = statistic*np.sqrt(365/366)
        p = 2*stats.t.sf(abs(corrected), 365)
        near([row.mean_squared_loss_difference, row.long_run_variance, row.DM, row.DM_HLN, row.p_value],
             [d.mean(), variance, statistic, corrected, p])
        if lag == 14:
            primary_p.append(p)
    near(dm.loc[dm.HAC_lag == 14, "p_holm"], multipletests(primary_p, method="holm")[1])

    diagnostics = pd.read_csv(SUPPLEMENT / "diagnostics.csv")
    acf_frame = pd.read_csv(SUPPLEMENT / "acf.csv")
    diagnostic_p = []
    for row in diagnostics.itertuples():
        i = METHODS.index(row.model)
        error = target-cube[i, 0]
        centered = error-error.mean()
        if row.test == "Ljung-Box":
            lag = int(row.lag_or_dimension)
            correlations = np.array([np.dot(centered[k:], centered[:-k])/np.dot(centered, centered)
                                     for k in range(1, lag+1)])
            statistic = 366*368*np.sum(correlations**2/(366-np.arange(1, lag+1)))
            p = stats.chi2.sf(statistic, lag)
        else:
            dim = int(row.lag_or_dimension)
            radius = 1.5*error.std(ddof=1)
            statistic = np.atleast_1d(bds(error, max_dim=3, epsilon=radius)[0])[dim-2]
            p = 2*stats.norm.sf(abs(statistic))
            near(row.epsilon, radius)
        near([row.statistic, row.p_value], [statistic, p], rtol=1e-9, atol=0.)
        diagnostic_p.append(p)
    near(diagnostics.p_holm, multipletests(diagnostic_p, method="holm")[1], rtol=1e-9, atol=0.)
    for i, model in enumerate(METHODS[:3]):
        e = target-cube[i, 0]
        e = e-e.mean()
        expected = [1.] + [np.dot(e[k:], e[:-k])/np.dot(e, e) for k in range(1, 29)]
        near(acf_frame[acf_frame.model == model].sort_values("lag").acf, expected)

    bootstrap = pd.read_csv(SUPPLEMENT / "block_bootstrap.csv")
    ranking = pd.read_csv(SUPPLEMENT / "ranking_bootstrap.csv")
    for length in (14, 7, 28):
        rng = np.random.default_rng(20261104)
        starts = rng.integers(0, 366, size=(2000, math.ceil(366/length)))
        indices = ((starts[:, :, None]+np.arange(length)) % 366).reshape(2000, -1)[:, :366]
        samples = np.array([np.sqrt(losses[:, :, index].mean(axis=2)).mean(axis=1)
                            for index in indices])
        for row in bootstrap[bootstrap.block_length == length].itertuples():
            a, b = METHODS.index(row.model_a), METHODS.index(row.model_b)
            difference = samples[:, a]-samples[:, b]
            gain = 1-samples[:, a]/samples[:, b]
            near([row.ci95_difference_lower, row.ci95_difference_upper], np.quantile(difference, [.025, .975]))
            near([row.ci95_relative_lower, row.ci95_relative_upper], np.quantile(gain, [.025, .975]))
        for count, label in [(3, "three_DL"), (4, "three_DL_and_persistence")]:
            winner = samples[:, :count] == samples[:, :count].min(axis=1, keepdims=True)
            probability = (winner/winner.sum(axis=1, keepdims=True)).mean(axis=0)
            for i, model in enumerate(METHODS[:count]):
                row = ranking[(ranking.block_length == length) & (ranking.comparison_set == label)
                              & (ranking.model == model)].iloc[0]
                near(row.bootstrap_lowest_error_frequency, probability[i])

    curves = report["epoch_curves"]
    assert curves["status"] == "plotted" and len(curves["sources"]) == 3
    curve_details = []
    for source in curves["sources"]:
        history_path = current_path(source["history_path"])
        config_path = current_path(source["configuration_path"])
        assert sha(history_path) == source["history_sha256"]
        assert sha(config_path) == source["configuration_sha256"]
        curve, config = pd.read_csv(history_path), json.loads(config_path.read_text())
        preproc = json.loads((history_path.parent / "preprocessing.json").read_text())["inner"]
        assert curve.epoch.tolist() == list(range(1, len(curve)+1))
        assert config["best_epoch"] == source["best_epoch"] == curve.best_epoch_so_far.iloc[-1]
        assert source["fold"] == "2024-12" and source["seed"] == 4101
        near(curve.train_rmse_original, np.sqrt(curve.train_mse_scaled_eval)*preproc["y_scale"])
        near(curve.validation_rmse_original, np.sqrt(curve.validation_mse_scaled_eval)*preproc["y_scale"])
        curve_details.append({"model": source["model"], "fold": source["fold"],
                              "seed": 4101, "epochs_logged": len(curve), "best_epoch": source["best_epoch"]})

    audit = {"audited_at_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS",
        "failures": [], "scope": "Independent recalculation from current CV predictions only; no optimizer or raw/test reads",
        "checks": {"prediction_rows": len(frame), "CV_dates": 366, "seed_trajectories_per_model": 10,
            "primary_seed_tests": 9, "global_Holm_tests": 12, "DM_tests": 18, "residual_diagnostics": 15,
            "acf_rows": len(acf_frame), "design_sensitivity_rows": 6, "superiority_rows": 6,
            "bootstrap_intervals": 18, "ranking_rows": 21, "bootstrap_draws_per_length": 2000,
            "input_and_output_hashes_checked": hashes_checked, "epoch_curves": curve_details,
            "protocol_registered_before_calculation": True, "analysis_and_helper_hashes_match": True},
        "conclusions": {"global_Holm12_model_pair_rejections": int((global_tests.iloc[[0, 1, 3]].p_holm < .05).sum()),
            "global_Holm12_network_vs_persistence_rejections": int((global_tests.iloc[[2, 4, 5]].p_holm < .05).sum()),
            "primary_DM_Holm6_rejections": int((dm[dm.HAC_lag == 14].p_holm < .05).sum()),
            "residual_Holm15_rejections": int((diagnostics.p_holm < .05).sum())},
        "registered_protocol_commit": subprocess.check_output(["git", "log", "-1", "--format=%H", "--",
            "experiments/filipe_statistics_20261104/protocol.json"], cwd=ROOT, text=True).strip(),
        "new_model_fits": 0, "raw_market_files_opened": 0, "reserved_test_files_opened": 0,
        "limitations": ["Development2024 was already exposed; registration precedes this supplement, not training or data inspection.",
            "Seed t intervals and exact-binomial superiority intervals condition on training randomization and the shared calendar.",
            "DM compares seed-mean squared losses with a Holm family; date-bootstrap intervals are marginal mean-seed RMSE functionals.",
            "MDE and oriented R calibrate observed dispersion retrospectively; they are not a prospective pilot or observed-effect power.",
            "BDS rejection is a departure from i.i.d., not proof of exclusively exploitable nonlinearity."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
