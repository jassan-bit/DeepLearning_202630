"""Independent design/inference contracts; never trains a neural network."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestPower

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from analyze_filipe_confirmatory_d7 import dimension_from_metrics, final_analysis, holm_adjust, verify_dimension_lock


def plan():
    return {"study_id": "e1_filipe_confirmatory_d7_20261105", "histories": [7],
            "models": {"MLP": {}, "LSTM": {}, "CNN1D": {}},
            "variance_extension_seeds": list(range(6301, 6341, 2)),
            "original_pilot_seeds": list(range(6201, 6206)), "definitive_seed_start": 50000001,
            "definitive_seed_step": 2,
            "primary_pairs": [{"history_days": 7, "model_a": a, "model_b": b}
                              for a, b in [("MLP", "LSTM"), ("MLP", "CNN1D"), ("LSTM", "CNN1D")]],
            "dimensioning": {"n_pilot": 20, "primary_family_size": 3,
                "alpha_global": .05, "alpha_each": .05/3, "execution_floor": 30,
                "power_target": .8, "sd_upper_confidence": .95,
                "sd_estimation": "new20_only; pooled25 sensitivity, never choosesR",
                "delta_absolute": .00023944847859645605, "maximum_planning_R": 10000},
            "analysis": {"bootstrap_replicates": 4999, "bootstrap_seed": 20261105}}


def frame(seeds, random_seed=7, sd=.0002):
    rng = np.random.default_rng(random_seed)
    return pd.DataFrame([{"history_days": 7, "model": model, "seed": seed,
                          "n_dates": 366, "rmse": .006+j*.00015+sd*rng.normal()}
                         for j, model in enumerate(["MLP", "LSTM", "CNN1D"]) for seed in seeds])


class ConfirmatoryDesignTests(unittest.TestCase):
    def test_old_five_never_choose_primary_R(self):
        p = plan()
        old, new = frame(p["original_pilot_seeds"]), frame(p["variance_extension_seeds"], 99)
        table, first = dimension_from_metrics(old, new, p)
        old.loc[old.model.eq("MLP"), "rmse"] *= 1000
        second_table, second = dimension_from_metrics(old, new, p)
        self.assertEqual(first["common_R"], second["common_R"])
        np.testing.assert_allclose(table.SD_new20_primary, second_table.SD_new20_primary)
        self.assertFalse(np.allclose(table.SD_pooled25_sensitivity, second_table.SD_pooled25_sensitivity))

    def test_planning_ignores_observed_mean_effect(self):
        p = plan()
        old, new = frame(p["original_pilot_seeds"]), frame(p["variance_extension_seeds"], 99)
        _, first = dimension_from_metrics(old, new, p)
        new.loc[new.model.eq("MLP"), "rmse"] += .3
        _, second = dimension_from_metrics(old, new, p)
        self.assertEqual(first["common_R"], second["common_R"])

    def test_upper_SD_rule_reaches_independent_power_target(self):
        p = plan()
        table, summary = dimension_from_metrics(frame(p["original_pilot_seeds"]), frame(p["variance_extension_seeds"], 99), p)
        for row in table.to_dict("records"):
            effect = p["dimensioning"]["delta_absolute"]/row["SD_upper95_new20_primary"]
            probability = TTestPower().power(effect, summary["common_R"], alpha=.05/3, alternative="two-sided")
            self.assertGreaterEqual(probability, .8)
        self.assertGreaterEqual(summary["common_R"], 30)

    def test_bad_or_reused_seed_sets_rejected(self):
        p = plan()
        old, new = frame(p["original_pilot_seeds"]), frame(p["variance_extension_seeds"])
        with self.assertRaises(ValueError):
            dimension_from_metrics(old, new.iloc[:-1], p)
        duplicate = pd.concat([new.iloc[:-1], new.iloc[:1]], ignore_index=True)
        with self.assertRaises(ValueError):
            dimension_from_metrics(old, duplicate, p)
        with self.assertRaises(ValueError):
            final_analysis(frame(p["original_pilot_seeds"]), p, {"common_R": 30})

    def test_design_lock_recomputation_rejects_changed_R(self):
        p = plan()
        old, new = frame(p["original_pilot_seeds"]), frame(p["variance_extension_seeds"], 99)
        _, summary = dimension_from_metrics(old, new, p)
        self.assertEqual(verify_dimension_lock(old, new, p, summary), summary)
        wrong = {**summary, "common_R": summary["common_R"]+1}
        with self.assertRaises(ValueError):
            verify_dimension_lock(old, new, p, wrong)

    def test_Holm_matches_independent_statsmodels(self):
        for raw in [[.02, .001, .4], [.049, .03, .02], [0., .5, 1.]]:
            np.testing.assert_allclose(holm_adjust(raw), multipletests(raw, alpha=.05, method="holm")[1])

    def test_final_paired_statistics_and_family_interval_reference(self):
        p = plan()
        f = frame([p["definitive_seed_start"]+2*i for i in range(40)], 33)
        metrics, pairs, summary = final_analysis(f, p, {"common_R": 40})
        pivot = f.pivot(index="seed", columns="model", values="rmse")
        for row in pairs.to_dict("records"):
            d = (pivot[row["model_a"]]-pivot[row["model_b"]]).to_numpy()
            reference = stats.ttest_1samp(d, 0.)
            self.assertAlmostEqual(row["t_statistic"], reference.statistic, places=12)
            self.assertAlmostEqual(row["p_value"], reference.pvalue, places=12)
            expected_half = stats.t.ppf(1-.05/6, 39)*d.std(ddof=1)/np.sqrt(40)
            self.assertAlmostEqual(row["mean_difference_family_ci_upper"]-d.mean(), expected_half, places=14)
            self.assertAlmostEqual(row["family_CI_level"], 1-.05/3)
        self.assertEqual(len(metrics), 3)
        self.assertEqual(summary["primary_comparisons"], 3)
        self.assertFalse(summary["advanced_gain_boundary_tested"])
        self.assertFalse(summary["achieved_or_joint_power_claim"])

    def test_bootstrap_is_reproducible_and_secondary(self):
        p = plan()
        f = frame([p["definitive_seed_start"]+2*i for i in range(30)], 33)
        _, first, _ = final_analysis(f, p, {"common_R": 30})
        _, second, _ = final_analysis(f, p, {"common_R": 30})
        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(first.bootstrap_replicates.eq(4999).all())
        self.assertTrue(first.bootstrap_defined_effect_replicates.eq(4999).all())


if __name__ == "__main__":
    unittest.main()
