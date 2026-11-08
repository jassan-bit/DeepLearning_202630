"""Independent synthetic checks for the new 608-date framework; never fit models."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestPower
from statsmodels.tsa.stattools import bds

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from e1_608_statistics import (
    MODELS, PAIRS, METRICS, N_DATES, STUDY_ID, REFERENCE_SOURCE_SHA256,
    PlanningLimitError, analyze_predictions, circular_block_counts,
    detectable_difference, difference_power, minimum_runs, plan_sample_size,
    verify_reference_sources, _validate_predictions,
    _rank_and_qq_diagnostics,
)

EXECUTION = "synthetic-new-608-execution"


def design():
    return {"study_id": STUDY_ID, "execution_id": EXECUTION,
        "source_signature": "synthetic-current-source", "baseline_model": "GRU",
        "statistics": {"delta": .00025772842664981106, "alpha": .05, "power": .8,
            "min_runs": 10, "max_runs": 10000, "final_runs_floor": 30,
            "sd_upper_simultaneous_confidence": .8, "family_size": 3,
            "primary_pairs": [list(pair) for pair in PAIRS], "bootstrap_draws": 101,
            "bootstrap_seed": 20261109, "temporal_bootstrap_seed": 20261110},
        "pilot": {"seeds": list(range(17101, 17111)), "origin": "2024-11-01",
            "outcome_start": "2024-11-01", "outcome_end": "2024-12-30"},
        "search": {"seeds": [16101, 16102]},
        "definitive": {"seed_start": 18101, "minimum_runs": 30},
        "reserve": {"start": "2025-01-01", "end": "2026-08-31"}}


def pilot_metrics(spread=1.):
    rng = np.random.default_rng(99283)
    rows = []
    for index, model in enumerate(MODELS):
        for seed, value in zip(range(17101, 17111),
                1 + .0004 * index + rng.normal(0, .0003 * spread, 10)):
            rows.append({"model": model, "seed": seed, "rmse": value,
                "study_id": STUDY_ID, "execution_id": EXECUTION, "phase": "pilot",
                "evaluation_days": 60, "fit_origin_day_utc": "2024-11-01",
                "evaluation_start": "2024-11-01", "evaluation_end_inclusive": "2024-12-30",
                "source_signature": "synthetic-current-source"})
    return pd.DataFrame(rows)


def predictions(n=30):
    dates = pd.date_range("2025-01-01", "2026-08-31").strftime("%Y-%m-%d")
    day = np.arange(len(dates))
    target = .02 + .003 * np.sin(day / 3.7) + .000003 * day
    persistence = np.r_[.0192, target[:-1]]
    rows = []
    for index, model in enumerate(MODELS):
        for seed in range(18101, 18101 + n):
            rng = np.random.default_rng(seed + 10000 * index)
            error = (.0002 * (index + 1) + .0001 * np.sin(day / 8)
                     + rng.normal(0, .00015 * (1 + (seed - 18101) / 10), len(dates)))
            rows.append(pd.DataFrame({"model": model, "seed": seed,
                "outcome_day": dates, "y_true": target, "y_pred": target + error,
                "persistence_pred": persistence, "mase_scale": .001,
                "study_id": STUDY_ID, "execution_id": EXECUTION,
                "source_signature": "synthetic-current-source"}))
    return pd.concat(rows, ignore_index=True)


def plan(n=30):
    return {"R": n, "R_final": n, "R_required": 10,
        "seeds": list(range(18101, 18101 + n)), "execution_id": EXECUTION}


class PlanningTests(unittest.TestCase):
    def test_independent_noncentral_t_and_smallest_integer(self):
        alpha = .05 / 3
        reference = TTestPower().power(effect_size=.3, nobs=100,
                                       alpha=alpha, alternative="two-sided")
        self.assertAlmostEqual(difference_power(100, .0003, .001, alpha), reference, places=11)
        n = minimum_runs(.0003, .001, alpha)
        self.assertGreaterEqual(difference_power(n, .0003, .001, alpha), .8)
        self.assertLess(difference_power(n - 1, .0003, .001, alpha), .8)
        mde = detectable_difference(47, .001, alpha)
        self.assertAlmostEqual(TTestPower().power(effect_size=mde / .001, nobs=47,
            alpha=alpha, alternative="two-sided"), .8, places=9)
        with self.assertRaises(PlanningLimitError):
            minimum_runs(.000001, .01, alpha, maximum=10000)
        for n in (np.nan, np.inf, 1, 3.5, True):
            with self.assertRaises(ValueError):
                difference_power(n, .0003, .001, alpha)

    def test_floor_30_and_pilot_required_R_are_distinct_no_465_floor(self):
        result = plan_sample_size(pilot_metrics(.03), design())
        self.assertLess(result["R_required"], 30)
        self.assertEqual(result["R_final"], 30)
        self.assertEqual(result["seeds"], list(range(18101, 18131)))
        high = plan_sample_size(pilot_metrics(3), design())
        self.assertGreater(high["R_required"], 30)
        self.assertEqual(high["R_final"], high["R_required"])
        self.assertNotEqual(high["R_final"], 465)
        self.assertFalse(high["temporal_power_claimed"])
        self.assertEqual(high["R_required"], max(r["upper_sd_required_R"] for r in high["pairs"]))
        pilot = pilot_metrics(3).pivot(index="seed", columns="model", values="rmse")
        for row in high["pairs"]:
            sd = np.std(pilot[row["model_a"]] - pilot[row["model_b"]], ddof=1)
            upper = sd * np.sqrt(9 / stats.chi2.ppf(.2 / 3, 9))
            self.assertAlmostEqual(row["paired_sd_upper"], upper, places=14)
            self.assertGreaterEqual(row["power_at_planned_R_upper_sd"], .8)
            self.assertEqual([r["simultaneous_sd_confidence"] for r in
                row["sd_confidence_sensitivity"]], [.8, .9, .95])
        for row in high["pilot_summary"]:
            self.assertAlmostEqual(row["coefficient_of_variation"],
                pilot[row["model"]].std(ddof=1) / pilot[row["model"]].mean(), places=14)
        with self.assertRaises(PlanningLimitError):
            plan_sample_size(pilot_metrics(100), design())

    def test_pilot_rejects_old_mixed_incomplete_and_nonfinite(self):
        frame = pilot_metrics()
        for column, value in (("study_id", "e1_core_mlp_extension_20261107"),
                ("execution_id", "old-run"), ("source_signature", "old-source"),
                ("evaluation_days", 608), ("evaluation_end_inclusive", "2024-12-31"),
                ("fit_origin_day_utc", "2025-01-01"), ("seed", 7101),
                ("rmse", np.inf), ("rmse", -1), ("phase", "final")):
            broken = frame.copy()
            broken.loc[0, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                plan_sample_size(broken, design())
        for broken in (frame.iloc[:-1], pd.concat([frame, frame.iloc[:1]]),
                frame.drop(columns="execution_id"), frame.drop(columns="source_signature")):
            with self.assertRaises(ValueError):
                plan_sample_size(broken, design())
        for group, key, value in (("statistics", "final_runs_floor", 465),
                ("definitive", "seed_start", 8101), ("search", "seeds", [6101, 6102]),
                ("pilot", "seeds", list(range(7101, 7111)))):
            broken = deepcopy(design())
            broken[group][key] = value
            with self.assertRaises(ValueError):
                plan_sample_size(frame, broken)

    def test_reference_sources_immutable(self):
        self.assertEqual(verify_reference_sources(), REFERENCE_SOURCE_SHA256)
        self.assertIn("e1_core_mlp_statistics.py", REFERENCE_SOURCE_SHA256)


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.forecasts = predictions()
        cls.output = analyze_predictions(cls.forecasts, design(), plan())
        cls.panel = cls.output["run_metrics"].query("model!='Persistence'").pivot(
            index="seed", columns="model", values="rmse")

    def test_six_per_seed_metrics_and_summary(self):
        runs = self.output["run_metrics"]
        self.assertEqual(len(runs), 91)
        self.assertEqual(len(self.output["metric_summary"]), 24)
        self.assertTrue(set(METRICS) <= set(runs))
        self.assertTrue(runs.n_dates.eq(608).all())
        raw = self.forecasts.query("model=='GRU' and seed==18101")
        run = runs.query("model=='GRU' and seed==18101").iloc[0]
        error = raw.y_true.to_numpy() - raw.y_pred.to_numpy()
        self.assertAlmostEqual(run.rmse, np.sqrt(np.mean(error ** 2)), places=15)
        self.assertAlmostEqual(run.mae, np.abs(error).mean(), places=15)
        self.assertAlmostEqual(run.mse, np.mean(error ** 2), places=15)
        self.assertAlmostEqual(run.r2, 1 - np.sum(error ** 2) /
            np.sum((raw.y_true - raw.y_true.mean()) ** 2), places=13)
        self.assertAlmostEqual(run.mape, 100 * np.mean(np.abs(error / raw.y_true)), places=13)
        self.assertAlmostEqual(run.mase, np.abs(error).mean() / .001, places=14)
        persistence = self.output["metric_summary"].query("model=='Persistence'")
        self.assertTrue(persistence.sample_sd.isna().all())
        self.assertTrue(persistence.n_total.eq(1).all())

    def test_primary_t_CI_holm_and_actual_R_MDE(self):
        expected = []
        for row in self.output["pairwise_rmse"].itertuples():
            reference = stats.ttest_rel(self.panel[row.model_a], self.panel[row.model_b])
            expected.append(reference.pvalue)
            self.assertEqual(row.n_paired, 30)
            self.assertAlmostEqual(row.statistic, reference.statistic, places=10)
            self.assertAlmostEqual(row.p_value_unadjusted, reference.pvalue, places=13)
            values = self.panel[row.model_a] - self.panel[row.model_b]
            self.assertAlmostEqual(row.ci95_lower,
                values.mean() - stats.t.ppf(.975, 29) * stats.sem(values), places=15)
            self.assertLessEqual(row.ci_family_lower, row.ci95_lower)
            self.assertGreaterEqual(row.ci_family_upper, row.ci95_upper)
            self.assertAlmostEqual(row.cohen_d_z, values.mean() / values.std(ddof=1), places=13)
            self.assertFalse(row.normality_diagnostic_changes_test)
            self.assertAlmostEqual(difference_power(30, row.MDE_at_actual_R_point_sd,
                values.std(ddof=1), .05 / 3), .8, places=9)
        np.testing.assert_allclose(self.output["pairwise_rmse"].p_value_holm3,
            multipletests(expected, method="holm")[1])

    def test_registered_friedman_wilcoxon_and_actual_QQ(self):
        reference = stats.friedmanchisquare(*(self.panel[m] for m in MODELS))
        row = self.output["friedman_rmse"].iloc[0]
        self.assertAlmostEqual(row.statistic, reference.statistic, places=12)
        self.assertAlmostEqual(row.p_value, reference.pvalue, places=14)
        self.assertAlmostEqual(row.kendall_W, reference.statistic / 60, places=14)
        expected = []
        for row in self.output["wilcoxon_rmse"].itertuples():
            values = self.panel[row.model_a] - self.panel[row.model_b]
            reference = stats.wilcoxon(values, zero_method="wilcox", correction=False,
                alternative="two-sided", method="approx")
            expected.append(reference.pvalue)
            self.assertEqual(row.statistic, reference.statistic)
            self.assertAlmostEqual(row.p_value_unadjusted, reference.pvalue, places=14)
            qq = self.output["qq_data"].query("model_a==@row.model_a and model_b==@row.model_b")
            theoretical, ordered = stats.probplot(values, dist="norm", fit=False)
            np.testing.assert_allclose(qq.theoretical_normal_quantile, theoretical)
            np.testing.assert_allclose(qq.ordered_difference_a_minus_b, ordered)
        np.testing.assert_allclose(self.output["wilcoxon_rmse"].p_value_holm3,
            multipletests(expected, method="holm")[1])
        self.assertEqual(len(self.output["qq_data"]), 90)

    def test_rank_degeneracy_keeps_all_registered_family_slots(self):
        runs = self.output["run_metrics"].copy()
        runs["rmse"] = 0.
        friedman, wilcoxon, qq = _rank_and_qq_diagnostics(runs, design(), plan())
        self.assertEqual(friedman.iloc[0].status, "undefined_all_tied")
        self.assertEqual(len(wilcoxon), 3)
        self.assertTrue(wilcoxon.p_value_holm3.isna().all())
        self.assertTrue(wilcoxon.multiplicity_family_size.eq(3).all())
        self.assertFalse(wilcoxon.reject_holm3.any())
        self.assertEqual(len(qq), 90)
        self.assertTrue(qq.ordered_standardized_difference.isna().all())

    def test_DM_HAC6_and_bootstrap_use_individual_model_losses(self):
        loss = {}
        for model in MODELS:
            group = self.forecasts.query("model==@model").copy()
            group["loss"] = (group.y_true - group.y_pred) ** 2
            loss[model] = group.pivot(index="seed", columns="outcome_day", values="loss").to_numpy()
        for row in self.output["dm_pairs"].itertuples():
            self.assertEqual(row.n_dates, 608)
            self.assertEqual(row.horizon, 1)
            self.assertEqual(row.hac_lags, 6)
        row = self.output["dm_pairs"].query("model_a=='GRU' and model_b=='LSTM'").iloc[0]
        diff = loss["GRU"].mean(axis=0) - loss["LSTM"].mean(axis=0)
        centered = diff - diff.mean()
        covariance = lambda lag: np.dot(centered[lag:], centered[:608-lag]) / 608
        lrv = covariance(0) + 2 * sum((1 - lag / 7) * covariance(lag) for lag in range(1, 7))
        statistic = diff.mean() / np.sqrt(lrv / 608) * np.sqrt(607 / 608)
        self.assertAlmostEqual(row.long_run_variance, lrv, places=26)
        self.assertAlmostEqual(row.dm_hln_statistic, statistic, places=11)
        self.assertAlmostEqual(row.mean_squared_loss_a, loss["GRU"].mean(), places=18)
        bootstrap = self.output["temporal_bootstrap"]
        self.assertEqual(len(bootstrap), 18)
        self.assertEqual(set(bootstrap.block_length), {3, 7, 14})
        self.assertEqual(set(bootstrap.bootstrap_seed), {20261113, 20261117, 20261124})
        expected = np.sqrt(loss["GRU"].mean(axis=1)).mean() - np.sqrt(loss["LSTM"].mean(axis=1)).mean()
        group = bootstrap.query("model_a=='GRU' and model_b=='LSTM'")
        np.testing.assert_allclose(group.difference_a_minus_b, expected)
        self.assertTrue((group.ci_family_lower <= group.ci95_lower).all())
        self.assertTrue((group.ci_family_upper >= group.ci95_upper).all())
        counts = circular_block_counts(608, 7, 101, np.random.default_rng(81))
        np.testing.assert_array_equal(counts.sum(axis=1), np.full(101, 608))
        for invalid in ((608, 7.5, 101), (608, True, 101), (608, 609, 101)):
            with self.assertRaises(ValueError):
                circular_block_counts(*invalid, np.random.default_rng(81))

    def test_full_residual_diagnostics_and_multiplicity(self):
        diagnostics = self.output["diagnostics"]
        self.assertTrue(diagnostics.n_residuals.eq(608).all())
        self.assertEqual(len(diagnostics), 270)
        raw = self.forecasts.query("model=='GRU' and seed==18101").sort_values("outcome_day")
        residual = (raw.y_true - raw.y_pred).to_numpy()
        group = diagnostics.query("model=='GRU' and seed==18101")
        row = group.query("diagnostic=='BDS'").iloc[0]
        expected = bds(residual, max_dim=2, epsilon=1.5 * residual.std(ddof=1))
        self.assertAlmostEqual(row.statistic, float(expected[0]), places=12)
        self.assertEqual(row.multiplicity_family_size, 90)
        lb = acorr_ljungbox(residual, lags=[1, 5], model_df=0, return_df=True)
        for row in group.query("diagnostic=='Ljung-Box'").itertuples():
            self.assertAlmostEqual(row.p_value, lb.loc[row.lag, "lb_pvalue"], places=14)
            self.assertEqual(row.multiplicity_family_size, 180)
        for _, group in diagnostics.groupby("diagnostic"):
            np.testing.assert_allclose(group.p_value_holm_diagnostic_family,
                multipletests(group.p_value, method="holm")[1])

    def test_rejects_old_or_mixed_cohorts_predictions_and_sources(self):
        for column, value in (("execution_id", "old-run"), ("source_signature", "old-source"),
                ("study_id", "e1_core_mlp_extension_20261107"), ("model", "TCN"),
                ("outcome_day", "2026-09-01"), ("y_true", .123),
                ("y_pred", np.inf), ("mase_scale", .002), ("seed", 8101)):
            broken = self.forecasts.copy()
            broken.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                _validate_predictions(broken, design(), plan())
        with self.assertRaises(ValueError):
            _validate_predictions(self.forecasts.iloc[:-1], design(), plan())
        with self.assertRaises(ValueError):
            _validate_predictions(predictions(29), design(), plan(29))
        validated = _validate_predictions(predictions(31), design(), plan(31))
        self.assertEqual(len(validated), 608 * (3 * 31 + 1))
        old_dates = deepcopy(design())
        old_dates["reserve"] = {"start": "2026-09-01", "end": "2026-10-30"}
        with self.assertRaises(ValueError):
            _validate_predictions(self.forecasts, old_dates, plan())

    def test_scope_discloses_known_608_history_and_approximation(self):
        summary = self.output["summary"]
        self.assertEqual(summary["dates_per_trajectory"], N_DATES)
        self.assertEqual(summary["final_runs_floor"], 30)
        self.assertEqual(summary["R_final"], 30)
        self.assertTrue(summary["known_reserve_exploratory"])
        self.assertFalse(summary["fresh_confirmatory_holdout"])
        self.assertFalse(summary["normality_diagnostic_selects_tests"])
        self.assertFalse(summary["financial_utility_claimed"])
        self.assertEqual(set(self.output["run_metrics"].source_signature), {"synthetic-current-source"})
        self.assertEqual(summary["reference_source_sha256"], REFERENCE_SOURCE_SHA256)


if __name__ == "__main__":
    unittest.main()
