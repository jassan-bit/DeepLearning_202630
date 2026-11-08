"""Independent numerical/provenance checks; all inputs are synthetic, no fits."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestPower

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from e1_core_mlp_statistics import (
    METRICS, MODELS, PAIRS, REFERENCE_SOURCE_SHA256, PlanningLimitError,
    analyze_predictions, circular_block_counts, difference_power,
    minimum_runs, plan_sample_size, verify_reference_sources,
)

STUDY = "e1_core_mlp_extension_20261107"
EXECUTION = "synthetic-new-coherent-execution"


def design():
    return {"study_id": STUDY,
            "statistics": {"delta": .00025772842664981106, "alpha": .05,
                "power": .8, "min_runs": 10, "max_runs": 10000,
                "final_runs_floor": 465, "sd_upper_simultaneous_confidence": .8,
                "primary_pairs": [list(pair) for pair in PAIRS], "family_size": 3,
                "bootstrap_draws": 101, "bootstrap_seed": 20261107,
                "temporal_bootstrap_seed": 20261108},
            "pilot": {"seeds": list(range(7101, 7111)), "origin": "2024-11-01",
                "outcome_start": "2024-11-01", "outcome_end": "2024-12-30"},
            "search": {"seeds": [6101, 6102]},
            "definitive": {"seed_start": 8101, "minimum_runs": 465},
            "reserve": {"start": "2026-09-01", "end": "2026-10-30"},
            "baseline_model": "GRU"}


def pilot_metrics(spread=1.):
    rng = np.random.default_rng(9283)
    rows = []
    for index, model in enumerate(MODELS):
        noise = rng.normal(0, .0003 * spread, 10)
        for seed, value in zip(range(7101, 7111), .01 + index * .0004 + noise):
            rows.append({"model": model, "seed": seed, "rmse": value,
                "study_id": STUDY, "execution_id": EXECUTION, "phase": "pilot",
                "evaluation_days": 60, "fit_origin_day_utc": "2024-11-01"})
    return pd.DataFrame(rows)


def predictions():
    dates = pd.date_range("2026-09-01", periods=60).strftime("%Y-%m-%d")
    day = np.arange(60)
    target = .02 + .003 * np.sin(day / 3.7) + .00003 * day
    persistence = np.r_[.0192, target[:-1]]
    rows = []
    for index, model in enumerate(MODELS):
        for seed in range(8101, 8566):
            rng = np.random.default_rng(seed + 10000 * index)
            error = .0004 * (index + 1) + rng.normal(0, .0002, 60)
            rows.append(pd.DataFrame({"model": model, "seed": seed,
                "outcome_day": dates, "y_true": target, "y_pred": target + error,
                "persistence_pred": persistence, "mase_scale": .001,
                "study_id": STUDY, "execution_id": EXECUTION}))
    return pd.concat(rows, ignore_index=True)


class CorePlanningTests(unittest.TestCase):
    def test_core_family_and_reference_sources_are_immutable(self):
        self.assertEqual(MODELS, ("GRU", "LSTM", "MLP"))
        self.assertEqual(PAIRS, (("GRU", "LSTM"), ("GRU", "MLP"), ("LSTM", "MLP")))
        self.assertNotIn("TCN", MODELS)
        self.assertEqual(verify_reference_sources(), REFERENCE_SOURCE_SHA256)

    def test_power_matches_independent_statsmodels_and_bound_is_not_a_cap(self):
        reference = TTestPower().power(effect_size=.3, nobs=100,
                                      alpha=.05 / 3, alternative="two-sided")
        self.assertAlmostEqual(difference_power(100, .0003, .001, .05 / 3),
                               reference, places=11)
        with self.assertRaises(PlanningLimitError):
            minimum_runs(.000001, .01, .05 / 3, maximum=465)

    def test_floor_and_required_sample_size_are_distinct(self):
        result = plan_sample_size(pilot_metrics(), design())
        self.assertEqual(result["R_required"], max(row["upper_sd_required_R"] for row in result["pairs"]))
        self.assertLess(result["R_required"], 465)
        self.assertEqual(result["R"], 465)
        self.assertEqual(result["R_final"], 465)
        self.assertEqual(result["seeds"], list(range(8101, 8566)))
        self.assertEqual(result["execution_id"], EXECUTION)
        high = plan_sample_size(pilot_metrics(spread=5), design())
        self.assertGreater(high["R_required"], 465)
        self.assertEqual(high["R_final"], high["R_required"])
        self.assertFalse(high["temporal_power_claimed"])
        for row in high["pairs"]:
            self.assertGreaterEqual(row["power_at_planned_R_upper_sd"], .8)
            self.assertLess(difference_power(row["upper_sd_required_R"] - 1,
                high["delta"], row["paired_sd_upper"], .05 / 3), .8)

    def test_pilot_requires_matching_seeds_cohort_and_new_execution(self):
        frame = pilot_metrics()
        for malformed in (frame.iloc[:-1], pd.concat([frame, frame.iloc[:1]]),
                          frame.drop(columns="execution_id")):
            with self.assertRaises(ValueError):
                plan_sample_size(malformed, design())
        for column, value in (("model", "TCN"), ("execution_id", "old-run"),
                              ("study_id", "old-study"), ("evaluation_days", 61),
                              ("fit_origin_day_utc", "2025-01-01"),
                              ("phase", "definitive")):
            broken = frame.copy()
            broken.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                plan_sample_size(broken, design())
        overlap = design()
        overlap["definitive"]["seed_start"] = 7101
        with self.assertRaises(ValueError):
            plan_sample_size(frame, overlap)
        wrong_pairs = design()
        wrong_pairs["statistics"]["primary_pairs"][1][1] = "TCN"
        with self.assertRaises(ValueError):
            plan_sample_size(frame, wrong_pairs)


class CoreAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.forecasts = predictions()
        cls.plan = {"R": 465, "R_final": 465, "R_required": 110,
                    "seeds": list(range(8101, 8566)), "execution_id": EXECUTION}
        cls.output = analyze_predictions(cls.forecasts, design(), cls.plan)

    def test_paired_tests_match_independent_scipy_and_holm(self):
        panel = self.output["run_metrics"].query("model!='Persistence'").pivot(
            index="seed", columns="model", values="rmse")
        expected = []
        for row in self.output["pairwise_rmse"].itertuples():
            reference = stats.ttest_rel(panel[row.model_a], panel[row.model_b])
            expected.append(reference.pvalue)
            self.assertAlmostEqual(row.statistic, reference.statistic, places=9)
            self.assertAlmostEqual(row.p_value_unadjusted, reference.pvalue, places=12)
            self.assertLessEqual(row.ci_family_lower, row.ci95_lower)
            self.assertGreaterEqual(row.ci_family_upper, row.ci95_upper)
        np.testing.assert_allclose(self.output["pairwise_rmse"].p_value_holm3,
                                  multipletests(expected, method="holm")[1])

    def test_complete_metrics_and_temporal_estimands(self):
        runs = self.output["run_metrics"]
        self.assertEqual(len(runs), 3 * 465 + 1)
        self.assertTrue(set(METRICS) <= set(runs))
        self.assertEqual(len(self.output["metric_summary"]), 24)
        persistent = self.output["metric_summary"].query("model=='Persistence'")
        self.assertTrue(persistent.sample_sd.isna().all())
        self.assertTrue(persistent.n_total.eq(1).all())
        dm = self.output["dm_pairs"]
        self.assertEqual(len(dm), 6)
        self.assertEqual(set(map(tuple, dm[["model_a", "model_b"]].to_numpy())),
                         set(PAIRS) | {(model, "Persistence") for model in MODELS})
        self.assertTrue(dm.hac_lags.eq(6).all())
        self.assertTrue(dm.multiplicity_family_size.eq(6).all())
        group = self.forecasts.query("model=='GRU'")
        self.assertAlmostEqual(dm.iloc[0].mean_squared_loss_a,
                               ((group.y_true - group.y_pred) ** 2).mean(), places=14)
        bootstrap = self.output["temporal_bootstrap"]
        self.assertEqual(set(bootstrap.block_length), {3, 7, 14})
        self.assertEqual(set(bootstrap.bootstrap_seed), {20261111, 20261115, 20261122})
        self.assertEqual(len(bootstrap), 18)
        counts = circular_block_counts(60, 7, 101, np.random.default_rng(81))
        np.testing.assert_array_equal(counts.sum(axis=1), np.full(101, 60))
        self.assertEqual(set(self.output["diagnostics"].diagnostic), {"BDS", "Ljung-Box"})

    def test_summary_does_not_promote_known_reserve_or_training_power(self):
        summary = self.output["summary"]
        self.assertTrue(summary["known_reserve_exploratory"])
        self.assertFalse(summary["fresh_confirmatory_holdout"])
        self.assertFalse(summary["financial_utility_claimed"])
        self.assertEqual(summary["baseline_selected_before_reserve"], "GRU")
        self.assertEqual(summary["reference_source_sha256"], REFERENCE_SOURCE_SHA256)

    def test_predictions_reject_mixed_old_metrics_dates_and_missing_seed(self):
        for column, value in (("execution_id", "old-run"), ("study_id", "old-study"),
                              ("model", "TCN"), ("outcome_day", "2026-08-31"),
                              ("y_true", .123)):
            broken = self.forecasts.copy()
            broken.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                analyze_predictions(broken, design(), self.plan)
        with self.assertRaises(ValueError):
            analyze_predictions(self.forecasts.iloc[:-1], design(), self.plan)
        wrong = deepcopy(self.plan)
        wrong["execution_id"] = "other-new-run"
        with self.assertRaises(ValueError):
            analyze_predictions(self.forecasts, design(), wrong)


if __name__ == "__main__":
    unittest.main()
