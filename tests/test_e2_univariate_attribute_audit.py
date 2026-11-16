"""Causal schema and train-only univariate calculations, synthetic fixtures."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_univariate_attribute_audit as audit


class UnivariateAttributeTests(unittest.TestCase):
    def test_reject_current_future_or_reordered_coordinate(self):
        rows = audit.coordinate_rows()
        audit.validate_coordinates(rows)
        for key, value in (("lag_days", 0), ("latest_required_bar_offset_days", 1),
                           ("normalization_anchor_offset_days", 0), ("channel", "target")):
            bad = copy.deepcopy(rows); bad[0][key] = value
            with self.assertRaises(ValueError):
                audit.validate_coordinates(bad)
        bad = copy.deepcopy(rows); bad[0], bad[1] = bad[1], bad[0]
        with self.assertRaises(ValueError):
            audit.validate_coordinates(bad)

    def test_reject_duplicate_unordered_noncanonical_ids(self):
        for ids in (["2025-01-01", "2025-01-01"], ["2025-01-02", "2025-01-01"],
                    ["20250101", "2025-01-02"], ["2025-01-01", "2025-01-32"]):
            with self.assertRaises(ValueError):
                audit.validate_ids(ids, "evaluation", expected_count=2)

    def test_frozen_training_fit_does_not_learn_scoring_relation(self):
        fitted = audit.fit_ols(np.arange(5.), 2 + 3*np.arange(5.))
        before = fitted.copy()
        self.assertAlmostEqual(fitted["intercept"], 2)
        self.assertAlmostEqual(fitted["slope"], 3)
        scored = audit.score_ols(fitted, np.arange(5.), 100 + 9*np.arange(5.))
        self.assertLess(scored["R2"], 0)
        self.assertGreater(scored["RMSE"], 90)
        self.assertEqual(before, fitted)

    def test_constant_coordinate_predicts_only_training_mean(self):
        fitted = audit.fit_ols(np.ones(4), np.array([1., 2., 3., 4.]))
        self.assertTrue(fitted["constant_training_coordinate"])
        self.assertEqual(fitted["intercept"], 2.5)
        self.assertEqual(fitted["slope"], 0)
        scored = audit.score_ols(fitted, np.full(3, 900.), np.array([10., 11., 12.]))
        self.assertLess(scored["R2"], 0)
        self.assertGreater(scored["RMSE"], 8)

    def test_constant_target_has_undefined_r2_with_reported_rmse(self):
        scored = audit.score_ols({"intercept": 2., "slope": 0.}, [0., 1.], [3., 3.])
        self.assertIsNone(scored["R2"])
        self.assertEqual(scored["RMSE"], 1.)

    def test_nonfinite_or_differing_sample_pairs_rejected(self):
        with self.assertRaises(ValueError):
            audit.fit_ols([0., 1.], [0., float("nan")])
        with self.assertRaises(ValueError):
            audit.fit_ols([0., 1.], [0.])
        with self.assertRaises(ValueError):
            audit.score_ols({"intercept": 0., "slope": 1.}, [0.], [float("inf")])


if __name__ == "__main__":
    unittest.main()
