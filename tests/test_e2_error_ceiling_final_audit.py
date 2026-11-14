"""Scientific boundary checks for the post hoc, frozen-prediction audit."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/e2_error_ceiling_final_audit.py"
spec = importlib.util.spec_from_file_location("error_ceiling", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ScientificContractTests(unittest.TestCase):
    def test_exclusive_categories_at_both_boundaries_and_signs(self):
        errors = [-3, -2, -1, -.1, 0, .1, 1, 2, 3]
        got = audit.classify_errors(errors, 1, 2)
        self.assertEqual(got.tolist(), [3, 1, 0, 0, 0, 0, 0, 2, 4])
        with self.assertRaises(ValueError):
            audit.classify_errors([float("nan")], 1, 2)

    def test_development_changes_never_bridge_missing_or_calendar_gap(self):
        dates = pd.date_range("2023-01-01", periods=110, tz="UTC")
        values = np.linspace(1, 5, 110) ** 2
        values[50] = np.nan
        frame = pd.DataFrame({"market_day_utc": dates, "sigma_7": values})
        got = audit.development_thresholds(frame)
        self.assertEqual(got["finite_target_days"], 109)
        self.assertEqual(got["consecutive_finite_change_pairs"], 107)
        frame.loc[51, "market_day_utc"] += pd.Timedelta(hours=1)
        changed = audit.development_thresholds(frame)
        self.assertEqual(changed["consecutive_finite_change_pairs"], 106)
        frame.loc[0, "market_day_utc"] = pd.Timestamp("2025-01-01", tz="UTC")
        with self.assertRaises(ValueError):
            audit.development_thresholds(frame)

    def test_stratified_sample_has_100_distinct_dates_and_correct_weights(self):
        strata = np.repeat(["2025:low", "2025:middle", "2025:high", "2026:low"], [220, 270, 117, 1])
        selected, design = audit.stratified_sample(strata)
        repeated, _ = audit.stratified_sample(strata)
        self.assertEqual(len(selected), 100)
        self.assertEqual(len(set(selected)), 100)
        self.assertTrue(np.array_equal(selected, repeated))
        self.assertEqual(set(strata[selected]), set(strata))
        self.assertEqual(sum(r["sample_dates"] for r in design), 100)
        self.assertAlmostEqual(sum(r["sample_dates"] * r["design_weight"] for r in design), 608)
        for row in design:
            self.assertEqual(int((strata[selected] == row["stratum"]).sum()), row["sample_dates"])

    def test_blocks_wrap_without_destroying_daily_order(self):
        indices = audit.block_indices(np.random.default_rng(10), 23, 100, 7)
        self.assertEqual(indices.shape, (100, 23))
        for start in (0, 7, 14):
            self.assertTrue(np.all(np.diff(indices[:, start:start + 7], axis=1) % 23 == 1))

    def test_constant_frequency_intervals_and_minimum_draws(self):
        profiles = np.column_stack((np.ones(28), np.zeros(28)))
        ci = audit.bootstrap_profiles(profiles, draws=10000, block=7)
        np.testing.assert_array_equal(ci, [[1, 1], [0, 0]])
        with self.assertRaises(ValueError):
            audit.bootstrap_profiles(profiles, draws=9999)

    def test_invalid_empirical_ceiling_is_not_forced_to_fraction(self):
        invalid = audit.gamma_reference(.0038, .0041, .0042)
        self.assertIsNone(invalid["gamma_remaining_fraction"])
        self.assertFalse(invalid["valid_order_for_ceiling_gap"])
        ideal = audit.gamma_reference(.0038, .0041, 0.)
        self.assertAlmostEqual(ideal["gamma_remaining_fraction"], .0038 / .0041)
        degenerate = audit.gamma_reference(.0041, .0041, .0041)
        self.assertIsNone(degenerate["raw_algebraic_gamma"])

    def test_ordinal_entropy_handles_monotonic_series_and_missing_windows(self):
        monotonic = audit.permutation_entropy(np.arange(20.), order=3)
        self.assertEqual(monotonic["normalized_entropy"], 0.)
        self.assertEqual(monotonic["complete_windows"], 18)
        values = np.arange(20.)
        values[10] = np.nan
        missing = audit.permutation_entropy(values, order=3)
        self.assertEqual(missing["complete_windows"], 15)
        tied = audit.permutation_entropy(np.ones(5), order=3)
        self.assertEqual(tied["windows_with_ties"], 3)
        random = audit.permutation_entropy(np.random.default_rng(1).normal(size=10000), order=3)
        self.assertGreater(random["normalized_entropy"], .99)


if __name__ == "__main__":
    unittest.main()
