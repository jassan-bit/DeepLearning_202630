"""Independent checks for floor units, chronology, projection and loss theorem."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("constraint_audit", HERE / "run_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def closes_from_returns(returns):
    return 100 * np.exp(np.concatenate([[0.0], np.cumsum(returns)]))


class ConstraintTests(unittest.TestCase):
    def test_hand_variance_uses_seven_not_six_and_returns_variance(self):
        closes = closes_from_returns([-.02, -.01, 0, .01, .02, .03])
        # Sum centered squared returns is .00175; sigma floor is sqrt(.00025).
        A = audit.observable_variance_floor(closes[None, :])[0]
        self.assertAlmostEqual(A, .00025, places=16)
        self.assertAlmostEqual(np.sqrt(A), np.sqrt(.00025), places=14)
        self.assertNotAlmostEqual(A, .00175 / 6, places=8)

    def test_population_seven_return_identity_and_attainable_boundary(self):
        known = np.array([-.03, .012, .007, -.009, .021, .004])
        A = audit.observable_variance_floor(closes_from_returns(known)[None, :])[0]
        for future in [-1, -.03, known.mean(), .07, 2]:
            expected = np.var(np.append(known, future), ddof=0)
            self.assertAlmostEqual(expected, A + 6 / 49 * (future - known.mean()) ** 2, places=14)
            self.assertGreaterEqual(expected + 1e-17, A)
        self.assertAlmostEqual(np.var(np.append(known, known.mean())), A, places=16)

    def test_no_future_or_nonclose_feature_enters_floor(self):
        past = closes_from_returns([.01, -.02, .03, -.01, .02, .005])
        A = audit.observable_variance_floor(past[None, :])
        for future_return in [-4, 0, 4]:
            eight = np.append(past, past[-1] * np.exp(future_return))
            np.testing.assert_array_equal(audit.observable_variance_floor(eight[None, :7]), A)
            with self.assertRaises(ValueError):
                audit.observable_variance_floor(eight[None, :])

    def test_close_price_scale_invariance(self):
        past = closes_from_returns([.01, -.02, .03, -.01, .02, .005])
        np.testing.assert_allclose(audit.observable_variance_floor(past[None, :]),
            audit.observable_variance_floor((past * 5000)[None, :]), rtol=1e-12, atol=1e-18)

    def test_chronology_rejects_reversal_shuffle_duplicate_dates(self):
        bars = np.zeros((9, 5)); bars[:, :4] = np.arange(100, 109)[:, None]
        windows = np.stack([bars[i:i + 7] for i in range(3)])
        dates = ["2025-01-08", "2025-01-09", "2025-01-10"]
        supports = audit.validate_chronology(windows, dates)
        self.assertEqual(supports[0]["first_input_date"], "2025-01-01")
        self.assertEqual(supports[0]["last_input_date"], "2025-01-07")
        for wrong in [windows[:, ::-1, :], windows[[1, 0, 2]]]:
            with self.assertRaises(ValueError):
                audit.validate_chronology(wrong, dates)
        with self.assertRaises(ValueError):
            audit.validate_chronology(windows, [dates[0], dates[0], dates[2]])

    def test_projection_is_max_in_sigma_units_and_not_clipping_to_target(self):
        predictions = np.array([[-.1, .01, .025, .04]])
        floor = np.array([.015, .015, .015, .015])
        np.testing.assert_array_equal(audit.project(predictions, floor), [[.015, .015, .025, .04]])
        with self.assertRaises(ValueError):
            audit.project(predictions.T, floor)

    def test_projection_theorem_both_losses_and_inapplicable_case_rejected(self):
        floor = np.array([.02, .01, 0, .2])
        targets = np.array([.03, .01, .04, .3])
        before = np.array([[-.1, .005, .08, .6], [.025, .02, .01, .1]])
        after = audit.project(before, floor)
        checks = audit.audit_loss_monotonicity(targets, before, after, floor)
        self.assertEqual(checks["maximum_pointwise_squared_loss_increase"], 0)
        self.assertEqual(checks["maximum_pointwise_absolute_loss_increase"], 0)
        with self.assertRaises(ValueError):
            audit.audit_loss_monotonicity(np.zeros(4), before, after, floor)

    def test_rank_metric_is_mean_individual_rmse_not_ensemble_or_pooled(self):
        y = np.array([.02, .02]); floor = np.array([0., 0.])
        ps = np.array([[.03, .01], [.06, .02]])
        rows = [audit.predictor_metrics("toy", seed, ps[i], ps[i], y, floor)
                for i, seed in enumerate([1, 2])]
        mean_individual = np.mean([row["RMSE_before"] for row in rows])
        self.assertAlmostEqual(mean_individual, (.01 + np.sqrt(.0008)) / 2)
        self.assertNotAlmostEqual(mean_individual, np.sqrt(np.mean((ps - y) ** 2)))
        self.assertNotAlmostEqual(mean_individual, np.sqrt(np.mean((ps.mean(axis=0) - y) ** 2)))

    def test_tolerance_only_counts_violations_projection_always_exact_max(self):
        floor = np.array([.02]); ps = np.array([[.02 - 1e-13]])
        projected = audit.project(ps, floor)
        row = audit.predictor_metrics("toy", 1, ps[0], projected[0], np.array([.03]), floor)
        self.assertEqual(row["below_floor_count_tolerance"], 0)
        self.assertEqual(row["strict_below_floor_count"], 1)
        self.assertEqual(row["projection_changed_count"], 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ConstraintTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if args.receipt:
        args.receipt.write_text(json.dumps({"checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
            "successful": result.wasSuccessful(), "scope": "synthetic numerical/unit checks; not hypothesis tests",
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [HERE / "run_audit.py", Path(__file__)]}}, indent=2) + "\n")
    sys.exit(0 if result.wasSuccessful() else 1)
