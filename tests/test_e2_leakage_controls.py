"""Synthetic leakage-control contracts and numerical evidence, no market data."""
from __future__ import annotations

import copy
from datetime import date, timedelta
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import e2_leakage_controls as module


def synthetic_controls():
    dates = module._calendar(module.EVAL_START, module.EVAL_END)
    training_dates = module._calendar(date(2024, 1, 1), date(2024, 2, 25))
    training = np.linspace(.01, .03, len(training_dates))
    permutations = {name: module.shuffled_training_targets(training, training_dates, seed=seed)[1]
                    for name, seed in module.SHUFFLE_SEEDS.items()}
    receipts, rows, means = {}, [], {}
    truth = .02+.008*np.sin(np.arange(60)/4)
    phase = {"control_weights_closed_at_utc": "2026-01-01T02:00:00+00:00",
        "all_primary_and_A_weights_closed_at_utc": "2026-01-01T01:00:00+00:00",
        "all_weights_including_controls_closed_at_utc": "2026-01-01T02:00:00+00:00",
        "calibration_closed_at_utc": "2026-01-01T03:00:00+00:00",
        "evaluation_scoring_started_at_utc": "2026-01-01T04:00:00+00:00", "checkpoint_count": 17}
    for index, model in enumerate(module.CONTROL_MODELS):
        seed = "deterministic" if model == "Ridge" else str(module.CONTROL_SEED)
        receipts[model] = {"model": model, "seed": seed, "fit_origin": "2025-01-01",
            "fit_outcome_max": "2024-12-24", "configuration_selection_outcome_max": "2024-10-31",
            "label_permutation": copy.deepcopy(permutations if model != "Ridge" else {"outer": permutations["outer"]}),
            "checkpoint_sha256": f"{index:064x}", "checkpoint_closed_at_utc": phase["control_weights_closed_at_utc"]}
        means[model] = float(training.mean())
        # Deliberately suspicious perfect control for the numerical alarm test;
        # a constant forecast for the nonestimable-correlation case.
        forecast = truth if index == 0 else np.full(60, training.mean())
        rows.extend({"model": model, "seed": seed, "outcome_day": d.isoformat(),
            "y_true": float(y), "y_pred": float(p), "source_stage": module.CONTROL_STAGE,
            "checkpoint_sha256": receipts[model]["checkpoint_sha256"],
            "checkpoint_closed_at_utc": phase["control_weights_closed_at_utc"],
            "scoring_started_at_utc": phase["evaluation_scoring_started_at_utc"]}
                    for d, y, p in zip(dates, truth, forecast))
    return rows, means, receipts, phase


class PermutationTests(unittest.TestCase):
    def test_block_shuffle_preserves_multiset_dtype_and_actual_calendar(self):
        dates = module._calendar(date(2024, 1, 1), date(2024, 1, 31))
        dates.pop(10)
        values = np.arange(len(dates), dtype=np.float32)
        original = values.copy()
        shuffled, receipt = module.shuffled_training_targets(values, dates)
        np.testing.assert_array_equal(values, original)
        np.testing.assert_array_equal(np.sort(shuffled), original)
        self.assertEqual(shuffled.dtype, values.dtype)
        self.assertFalse(np.array_equal(shuffled, values))
        self.assertEqual(receipt["missing_calendar_dates_between_admitted_rows"], 1)
        self.assertEqual(receipt["partial_last_block_observations"], 2)
        self.assertEqual(receipt["training_outcome_max"], "2024-01-31")

    def test_same_seed_reproduces_target_bytes_and_receipt(self):
        dates = module._calendar(date(2024, 1, 1), date(2024, 2, 1))
        values = np.arange(len(dates), dtype=float)
        a, ar = module.shuffled_training_targets(values, dates)
        b, br = module.shuffled_training_targets(values, dates)
        np.testing.assert_array_equal(a, b); self.assertEqual(ar, br)

    def test_no_future_training_or_silent_calendar_compression(self):
        values = np.arange(28, dtype=float)
        with self.assertRaises(ValueError):
            module.shuffled_training_targets(values, module._calendar(date(2024, 12, 1), date(2024, 12, 28)))
        dates = module._calendar(date(2024, 1, 1), date(2024, 1, 28))
        with self.assertRaises(ValueError):
            module.shuffled_training_targets(values, dates[::-1])
        with self.assertRaises(ValueError):
            module.shuffled_training_targets(values, dates, block_days=1)

    def test_training_shuffle_moves_whole_blocks_including_partial_block(self):
        dates = module._calendar(date(2024, 1, 1), date(2024, 1, 31))
        values, _ = module.shuffled_training_targets(np.arange(31), dates)
        for start in range(0, 31, 7):
            original = list(range(start, min(start+7, 31)))
            where = np.flatnonzero(values == start)[0]
            self.assertEqual(values[where:where+len(original)].tolist(), original)


class NegativeControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.means, cls.receipts, cls.phase = synthetic_controls()

    def test_perfect_control_alerts_but_constant_does_not_claim_no_leakage(self):
        result = module.analyze_shuffled_predictions(self.rows, training_mean_by_model=self.means,
            control_receipts=self.receipts, phase_receipt=self.phase)
        self.assertEqual(result["diagnostic_family_size"], 17)
        perfect = result["models"][module.CONTROL_MODELS[0]]
        self.assertTrue(perfect["diagnostic_alert_requires_investigation"])
        self.assertGreater(perfect["RMSE_gain_over_training_mean"], module.DELTA)
        constant = result["models"][module.CONTROL_MODELS[1]]
        self.assertTrue(constant["constant_forecast_association_not_estimable"])
        self.assertFalse(constant["diagnostic_alert_requires_investigation"])
        self.assertIn("absence of leakage is not established", constant["conclusion"])
        self.assertFalse(result["private_predictions_exported"])

    def test_control_fits_must_close_before_global_freeze_and_calibration(self):
        bad = copy.deepcopy(self.phase)
        bad["control_weights_closed_at_utc"] = "2026-01-01T03:30:00+00:00"
        with self.assertRaisesRegex(ValueError, "before common"):
            module.analyze_shuffled_predictions(self.rows, training_mean_by_model=self.means,
                control_receipts=self.receipts, phase_receipt=bad)

    def test_reused_primary_seed_or_missing_model_is_rejected(self):
        rows = copy.deepcopy(self.rows); rows[0]["seed"] = "670101"
        with self.assertRaises(ValueError):
            module.analyze_shuffled_predictions(rows, training_mean_by_model=self.means,
                control_receipts=self.receipts, phase_receipt=self.phase)
        with self.assertRaises(ValueError):
            module.analyze_shuffled_predictions(self.rows[:-1], training_mean_by_model=self.means,
                control_receipts=self.receipts, phase_receipt=self.phase)

    def test_real_inference_cannot_silently_reduce_registered_resampling_budget(self):
        with self.assertRaises(ValueError):
            module.analyze_shuffled_predictions(self.rows, training_mean_by_model=self.means,
                control_receipts=self.receipts, phase_receipt=self.phase, repetitions=5)

    def test_inner_validation_labels_must_also_be_permuted(self):
        receipts = copy.deepcopy(self.receipts)
        del receipts[module.CONTROL_MODELS[0]]["label_permutation"]["inner_validation"]
        with self.assertRaisesRegex(ValueError, "stopping target"):
            module.analyze_shuffled_predictions(self.rows, training_mean_by_model=self.means,
                control_receipts=receipts, phase_receipt=self.phase)


class AdversarialTests(unittest.TestCase):
    def test_Mann_Whitney_AUC_handles_ties_and_direction(self):
        self.assertEqual(module.mann_whitney_auc([0, 0, 1, 1], [.1, .2, .2, .3]), .875)
        self.assertEqual(module.mann_whitney_auc([0, 0, 1, 1], [1, 1, 1, 1]), .5)
        self.assertEqual(module.mann_whitney_auc([0, 0, 1, 1], [.4, .3, .2, .1]), 0.)
        self.assertIsNone(module.mann_whitney_auc([1, 1], [.1, .9]))

    def test_purged_folds_never_train_on_neighbors_of_heldout_windows(self):
        train = module._calendar(date(2024, 1, 1), date(2024, 5, 19))
        test = module._calendar(module.EVAL_START, module.EVAL_END)
        days = train+test; seen = []
        for tr, va in module._adversarial_folds(train, test):
            seen.extend(va.tolist())
            self.assertFalse(set(tr)&set(va))
            for a in tr:
                self.assertTrue(all(abs((days[a]-days[b]).days) > 7 for b in va))
        self.assertEqual(sorted(seen), list(range(len(days))))

    def test_full_permutation_refits_CV_and_detects_large_synthetic_shift(self):
        train_days = module._calendar(date(2024, 1, 1), date(2024, 3, 10))
        test_days = module._calendar(module.EVAL_START, module.EVAL_END)
        rng = np.random.default_rng(18)
        tx = rng.normal(size=(len(train_days), 35))
        ex = rng.normal(3., size=(len(test_days), 35))
        with patch.object(module, "_cross_validated_domain_scores", wraps=module._cross_validated_domain_scores) as cv:
            result = module.adversarial_validation(tx, train_days, ex, test_days, repetitions=19, fixture=True)
        self.assertEqual(cv.call_count, 20)
        self.assertGreater(result["OOF_AUC_Mann_Whitney"], .99)
        self.assertEqual(result["maximum_classifier_refits_under_null_scheduled"], 95)
        self.assertTrue(result["synthetic_fixture_only"])
        self.assertFalse(result["HPO_or_model_selection_feedback_allowed"])
        self.assertFalse(result["exact_iid_inference_or_absence_of_leakage_claimed"])
        self.assertNotIn("domain_scores", result)

    def test_classifier_scaler_is_fitted_to_training_fold_only(self):
        x = np.array([[-1.], [-.5], [.5], [1.]])
        y = np.array([0, 0, 1, 1])
        point = module._logistic_fit_predict(x, y, np.array([[.25]]))[0]
        with_outlier = module._logistic_fit_predict(x, y, np.array([[.25], [1e12]]))[0]
        self.assertEqual(point, with_outlier)

    def test_actual_calendars_and_35_feature_contract_are_required(self):
        train_days = module._calendar(date(2024, 1, 1), date(2024, 2, 1))
        test_days = module._calendar(module.EVAL_START, module.EVAL_END)
        with self.assertRaises(ValueError):
            module.adversarial_validation(np.ones((32, 36)), train_days,
                np.ones((60, 36)), test_days, repetitions=1, fixture=True)
        with self.assertRaises(ValueError):
            module.adversarial_validation(np.ones((32, 35)), train_days,
                np.ones((59, 35)), test_days[:-1], repetitions=1, fixture=True)

    def test_plan_is_data_free_and_distinguishes_endogenous_controls(self):
        with patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(module.main([]), 0)
            policy = json.loads(output.getvalue())
        self.assertEqual(policy["label_shuffle_models_count"], 17)
        self.assertEqual(policy["adversarial_validation"]["replicates"], 2000)
        self.assertFalse(policy["endogenous_reference_control"]["label_shuffle_training_claimed"])


if __name__ == "__main__":
    unittest.main()
