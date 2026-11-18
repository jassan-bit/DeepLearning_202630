"""Guard finite-sample calibration, chronology, and test-set separation."""
import importlib.util
from datetime import date, timedelta
from pathlib import Path
import unittest


SOURCE = Path(__file__).with_name("run.py")
SPEC = importlib.util.spec_from_file_location("e3_retrospective_calibration", SOURCE)
cal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cal)


class CalibrationContracts(unittest.TestCase):
    def test_uses_49th_observed_residual_not_an_interpolated_quantile(self):
        q, rank = cal.finite_sample_quantile(list(reversed(range(53))), .1)
        self.assertEqual(rank, 49)
        self.assertEqual(q, 48)

    def test_rejects_invalid_or_unattainable_finite_calibration_rank(self):
        for residuals, alpha in (([], .1), ([0, float("nan")], .1), ([-1, 0], .1),
                                 ([1, 2, 3], .1), ([0] * 53, 0), ([0] * 53, 1)):
            with self.subTest(residuals=residuals, alpha=alpha), self.assertRaises(ValueError):
                cal.finite_sample_quantile(residuals, alpha)

    def test_calibration_quantile_does_not_depend_on_evaluation_targets(self):
        cal_y, cal_prediction = [i / 100 for i in range(53)], [0.0] * 53
        q, rank = cal.fit_quantile(cal_y, cal_prediction, .1)
        before = cal.metric_summary(cal.interval_values([.01], [.01], q, .1))
        after = cal.metric_summary(cal.interval_values([5.0], [.01], q, .1))
        self.assertEqual(cal.fit_quantile(cal_y, cal_prediction, .1), (q, rank))
        self.assertEqual(before["coverage_percent"], 100)
        self.assertEqual(after["coverage_percent"], 0)

    def test_proper_interval_score_penalizes_misses_with_declared_alpha(self):
        values = cal.interval_values([.01, .1], [.02, .02], .03, .1)
        self.assertEqual(values[0]["lower"], 0)
        self.assertEqual(values[0]["upper"], .05)
        self.assertTrue(values[0]["covered"])
        self.assertAlmostEqual(values[0]["interval_score"], .05)
        self.assertFalse(values[1]["covered"])
        self.assertAlmostEqual(values[1]["interval_score"], 1.05)

    def test_saved_point_predictions_are_not_modified_by_interval_construction(self):
        targets, predictions = [.01, .02], [.015, .025]
        old = predictions.copy()
        cal.interval_values(targets, predictions, .005, .1)
        self.assertEqual(predictions, old)
        with self.assertRaises(ValueError):
            cal.interval_values([.01], predictions, .005, .1)

    def test_chronology_explicitly_distinguishes_input_overlap_from_target_leakage(self):
        # Dates are picked only to exercise the admission contract. Fit supports
        # include an allowed gap, as the audited series itself does.
        fit = [(date(2023, 1, 8) + timedelta(days=i)).isoformat() for i in range(531)] + ["2024-06-30"]
        validation = [(date(2024, 7, 8) + timedelta(days=i)).isoformat() for i in range(116)]
        calibration = [(date(2024, 11, 8) + timedelta(days=i)).isoformat() for i in range(53)]
        evaluation = [(date(2025, 1, 1) + timedelta(days=i)).isoformat() for i in range(608)]
        report = cal.validate_dates(fit, validation, calibration, evaluation)
        self.assertTrue(report["target_dates_disjoint"])
        self.assertTrue(report["known_input_support_overlap_calibration_evaluation"])
        corrupted = calibration.copy()
        corrupted[0] = "2024-10-31"
        with self.assertRaises(ValueError):
            cal.validate_dates(fit, validation, corrupted, evaluation)


if __name__ == "__main__":
    unittest.main(verbosity=2)
