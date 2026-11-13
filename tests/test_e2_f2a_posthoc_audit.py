"""Synthetic checks for the frozen-output audit, never market-result tests."""

import copy
from datetime import date, timedelta
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_f2a_posthoc_audit as audit


class PosthocAuditTests(unittest.TestCase):
    @staticmethod
    def input_fixture():
        y = np.linspace(.02, .04, 608)
        predictors = []
        metrics_by_model = {}
        identities = [(model, seed) for model in audit.NEURAL for seed in audit.SEEDS]
        identities += [(model, "deterministic") for model in audit.CLASSICAL]
        for i, (model, seed) in enumerate(identities):
            offset = (i+1)*1e-5
            predictors.append({
                "model": model, "seed": seed,
                "prediction_original": (y+offset).tolist(),
                "calibrator_q": .01, "checkpoint_sha256": f"{i+1:064x}",
            })
            metrics_by_model.setdefault(model, []).append(offset)
        original = {"table_A2": [
            {"model": model, "RMSE_per_seed": values,
             "RMSE": float(np.mean(values))}
            for model, values in metrics_by_model.items()
        ]}
        return {
            "version": "synthetic-frozen-audit-fixture-v1",
            "dates": list(audit.FINAL_DATES), "y_true": y.tolist(),
            "mask": [True]*608, "predictors": predictors,
            "original_analysis": original,
        }

    def test_holm_keeps_unestimable_members_in_the_family(self):
        rows = [
            {"test": "a", "p_raw": .02, "status": "estimable"},
            {"test": "b", "p_raw": .04, "status": "estimable"},
            {"test": "c", "p_raw": None, "status": "not_estimable"},
        ]
        original = copy.deepcopy(rows)
        result = audit.holm(rows, family_size=3)
        self.assertEqual(rows, original)
        self.assertAlmostEqual(result[0]["p_holm"], .06)
        self.assertAlmostEqual(result[1]["p_holm"], .08)
        self.assertIsNone(result[2]["p_holm"])
        self.assertFalse(any(row["reject"] for row in result))

    def test_holm_is_order_invariant_and_monotone(self):
        rows = [{"test": "c", "p_raw": .20},
                {"test": "a", "p_raw": .001},
                {"test": "b", "p_raw": .03}]
        adjusted = audit.holm(rows, family_size=3)
        by_id = {row["test"]: row for row in adjusted}
        reversed_by_id = {row["test"]: row for row in
                          audit.holm(rows[::-1], family_size=3)}
        for name in by_id:
            self.assertEqual(by_id[name]["p_holm"], reversed_by_id[name]["p_holm"])
        self.assertTrue(by_id["a"]["reject"])
        self.assertFalse(by_id["b"]["reject"])
        self.assertLessEqual(by_id["a"]["p_holm"], by_id["b"]["p_holm"])
        self.assertLessEqual(by_id["b"]["p_holm"], by_id["c"]["p_holm"])

    @staticmethod
    def calibrated_case():
        rng = np.random.default_rng(20261113)
        prediction = rng.uniform(.5, 1.5, size=608)
        design = np.column_stack((np.ones(len(prediction)), prediction))
        noise = rng.normal(scale=.03, size=len(prediction))
        # Orthogonal noise makes the population-null coefficients exact in
        # this finite synthetic sample while keeping positive HAC variance.
        residual = noise - design @ np.linalg.lstsq(design, noise, rcond=None)[0]
        return prediction + residual, prediction

    def test_mincer_zarnowitz_accepts_exact_calibration_and_detects_distortion(self):
        y, prediction = self.calibrated_case()
        calibrated = audit._mz(y, prediction, maxlags=14)
        self.assertEqual(calibrated["status"], "estimable")
        self.assertAlmostEqual(calibrated["intercept"], 0., places=9)
        self.assertAlmostEqual(calibrated["slope"], 1., places=9)
        self.assertGreater(calibrated["p_raw"], .99)

        distorted = audit._mz(.4 + 1.3*y, prediction, maxlags=14)
        self.assertEqual(distorted["status"], "estimable")
        self.assertLess(distorted["p_raw"], 1e-8)

    def test_constant_forecast_has_no_identifiable_mz_slope(self):
        y, _ = self.calibrated_case()
        result = audit._mz(y, np.ones(len(y)), maxlags=14)
        self.assertEqual(result["status"], "not_estimable")
        self.assertIsNone(result["p_raw"])
        self.assertTrue(result["reason"])

    def test_exact_fit_does_not_create_hac_significance_from_roundoff(self):
        prediction = np.linspace(.5, 1.5, 608)
        result = audit._mz(prediction, prediction, maxlags=14)
        self.assertEqual(result["status"], "not_estimable")
        self.assertIsNone(result["p_raw"])

    def test_constant_coverage_indicator_has_no_hac_sampling_variance(self):
        prediction = np.linspace(.5, 1.5, 608)
        rows = audit.diagnostic_tests(
            prediction, prediction, audit.FINAL_DATES,
            {"model": "synthetic", "seed": 271801}, calibrator_q=.1)
        coverage = next(row for row in rows if row["test"] == "Coverage90_HAC")
        self.assertEqual(coverage["status"], "not_estimable")
        self.assertIsNone(coverage["p_raw"])
        self.assertTrue(coverage["reason"])

    def test_constant_losses_do_not_create_spurious_calendar_instability(self):
        y = np.linspace(.5, 1.5, 608)
        result = audit.stability_test(
            y, np.array([y+.1]), audit.FINAL_DATES, {"model": "synthetic"})
        self.assertEqual(result["status"], "not_estimable")
        self.assertIsNone(result["p_raw"])
        self.assertTrue(result["reason"])

    def test_constant_bias_roundoff_is_not_residual_sampling_variation(self):
        y = np.linspace(.5, 1.5, 608)
        rows = audit.diagnostic_tests(
            y, y+.1, audit.FINAL_DATES,
            {"model": "synthetic", "seed": 271801}, calibrator_q=.2)
        for row in rows:
            if row["test"] in {"Mincer_Zarnowitz", "Coverage90_HAC"}:
                continue
            with self.subTest(test=row["test"]):
                self.assertEqual(row["status"], "not_estimable")
                self.assertIsNone(row["p_raw"])
                self.assertTrue(row["reason"])

    def test_residual_audit_detects_serial_dependence(self):
        rng = np.random.default_rng(117)
        n = 608
        residual = np.zeros(n)
        innovations = rng.normal(scale=.01, size=n)
        for i in range(1, n):
            residual[i] = .95*residual[i-1] + innovations[i]
        prediction = 1. + .1*np.sin(np.arange(n)/19.)
        dates = [(date(2025, 1, 1) + timedelta(days=i)).isoformat()
                 for i in range(n)]
        rows = audit.diagnostic_tests(
            prediction + residual, prediction, dates,
            {"model": "synthetic", "seed": 271801}, calibrator_q=.1)
        by_id = {row["test"]: row for row in rows}
        self.assertEqual(set(by_id), {
            "LB_residual_lag7", "LB_residual_lag14",
            "LB_squared_lag7", "LB_squared_lag14", "ARCH_LM_lag7",
            "BDS_dim2", "BDS_dim3", "Jarque_Bera", "Mincer_Zarnowitz",
            "Coverage90_HAC",
        })
        for test_id in ("LB_residual_lag7", "LB_residual_lag14"):
            self.assertEqual(by_id[test_id]["status"], "estimable")
            self.assertLess(by_id[test_id]["p_raw"], .001)

    def test_system_rmse_averages_members_without_creating_an_ensemble(self):
        y = np.ones(4)
        predictions = np.array([[0., 0., 0., 0.], [2., 2., 2., 2.]])
        result = audit.summarize_system(y, predictions, [.25, .25])
        self.assertAlmostEqual(result["RMSE"], 1.)
        self.assertAlmostEqual(result["MAE"], 1.)
        self.assertAlmostEqual(result["bias"], 0.)
        self.assertEqual(result["R"], 2)
        self.assertEqual(result["n_dates"], 4)
        ensemble_rmse = np.sqrt(np.mean((predictions.mean(axis=0)-y)**2))
        self.assertEqual(ensemble_rmse, 0.)
        self.assertNotEqual(result["RMSE"], ensemble_rmse)

    def test_interval_coverage_uses_inclusive_clipped_endpoints(self):
        result = audit.trajectory_metrics(
            np.array([0., .5, 1., 1.5]),
            np.array([0., .5, .5, 2.]), .5)
        self.assertEqual(result["coverage90"], 1.)
        self.assertAlmostEqual(result["mean_interval_width90"], .875)

    def test_input_preserves_complete_calendar_mask_and_predictor_registry(self):
        payload = self.input_fixture()
        audit.validate_input(payload)
        variants = []
        missing_date = copy.deepcopy(payload)
        missing_date["dates"].pop(45)
        variants.append(missing_date)
        reordered = copy.deepcopy(payload)
        reordered["dates"][45:47] = reordered["dates"][45:47][::-1]
        variants.append(reordered)
        masked_date = copy.deepcopy(payload)
        masked_date["mask"][45] = False
        variants.append(masked_date)
        missing_predictor = copy.deepcopy(payload)
        missing_predictor["predictors"].pop()
        variants.append(missing_predictor)
        duplicate_predictor = copy.deepcopy(payload)
        duplicate_predictor["predictors"][1] = copy.deepcopy(duplicate_predictor["predictors"][0])
        variants.append(duplicate_predictor)
        wrong_seed = copy.deepcopy(payload)
        wrong_seed["predictors"][0]["seed"] = audit.SEEDS[0]-100
        variants.append(wrong_seed)
        for i, invalid in enumerate(variants):
            with self.subTest(invalid_variant=i), self.assertRaises(ValueError):
                audit.validate_input(invalid)

    def test_saved_calibration_requires_complete_shared_truth_and_frozen_q(self):
        payload = self.input_fixture()
        cy = np.linspace(.02, .03, 53)
        saved = {
            "calibration_dates": list(audit.CALIBRATION_DATES),
            "calibration_targets": cy.tolist(),
            "calibration_predictions": (cy+.01).tolist(),
        }
        payload["predictors"][0].update(copy.deepcopy(saved))
        payload["predictors"][1].update(copy.deepcopy(saved))
        audit.validate_input(payload)
        partial = copy.deepcopy(payload)
        del partial["predictors"][0]["calibration_predictions"]
        changed_q = copy.deepcopy(payload)
        changed_q["predictors"][0]["calibrator_q"] = .011
        changed_truth = copy.deepcopy(payload)
        changed_truth["predictors"][1]["calibration_targets"][0] += .001
        changed_dates = copy.deepcopy(payload)
        changed_dates["predictors"][0]["calibration_dates"] = list(audit.FINAL_DATES[:53])
        for i, invalid in enumerate((partial, changed_q, changed_truth, changed_dates)):
            with self.subTest(invalid_variant=i), self.assertRaises(ValueError):
                audit.validate_input(invalid)

    def test_reproduction_check_rejects_changed_saved_final_predictions(self):
        payload = self.input_fixture()
        checks = audit.verify_original_rmse(payload)
        self.assertEqual(len(checks), 81)
        self.assertLess(max(abs(row["difference"]) for row in checks), 1e-12)
        payload["predictors"][0]["prediction_original"][0] += .01
        with self.assertRaisesRegex(ValueError, "RMSE mismatch"):
            audit.verify_original_rmse(payload)


if __name__ == "__main__":
    unittest.main()
