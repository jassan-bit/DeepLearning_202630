"""Numerical estimands, no-future denominators, local selection and budget guards."""
from copy import deepcopy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_amended_metrics as amended


class AmendedMetricTests(unittest.TestCase):
    def setUp(self):
        self.scales = amended.dated_training_scales(
            ["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05"],
            [1., 3., 4., 8.], cutoff="2024-01-05")

    def test_original_unit_auxiliary_metrics_keep_zero_targets_in_primary_support(self):
        self.assertEqual(self.scales["pairs"], 2)
        self.assertEqual(self.scales["MASE"]["value"], 3)
        self.assertEqual(self.scales["RMSSE"]["value"], 10)
        result = amended.regression_metrics([0., 2.], [2., 1.], training_scales=self.scales,
                                           persistence_predictions=[0., 0.])
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["MSE"], 2.5)
        self.assertAlmostEqual(result["RMSE"], math.sqrt(2.5))
        self.assertEqual(result["MAE"], 1.5)
        self.assertEqual(result["R2"], -1.5)
        self.assertEqual(result["MAPE_percent"], 50)
        self.assertEqual(result["MAPE_coverage"], .5)
        self.assertEqual(result["MAPE_n"], 1)
        self.assertEqual(result["MASE"], .5)
        self.assertEqual(result["RMSSE"], .5)
        self.assertAlmostEqual(result["RMRMSE"], math.sqrt(1.25))

    def test_denominators_do_not_bridge_calendar_gaps_or_missing_past_labels(self):
        result = amended.dated_training_scales(
            ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-06"],
            [1., None, 4., 7., 99.], cutoff="2024-01-06")
        self.assertEqual(result["pairs"], 1)
        self.assertEqual(result["MASE"]["value"], 3)
        self.assertEqual(result["RMSSE"]["value"], 9)
        self.assertEqual(result["nonfinite_past_values"], 1)
        with self.assertRaises(ValueError):
            amended.regression_metrics([1., 2.], [1., None], training_scales=self.scales)

    def test_future_labels_are_not_even_converted_to_numbers(self):
        class ForbiddenFutureValue:
            def __float__(self):
                raise AssertionError("Future label was inspected")
        result = amended.dated_training_scales(
            ["2024-01-01", "2024-01-02", "2024-01-03"], [1., 2., ForbiddenFutureValue()],
            cutoff="2024-01-02")
        self.assertEqual(result["pairs"], 1)
        self.assertEqual(result["MASE"]["value"], 1)

    def test_zero_denominators_are_explicitly_unavailable_not_epsilon_replaced(self):
        scales = amended.dated_training_scales(["2024-01-01", "2024-01-02"], [0., 0.], cutoff="2024-01-02")
        result = amended.regression_metrics([0., 0.], [0., 0.], training_scales=scales,
                                           persistence_predictions=[0., 0.])
        self.assertEqual(result["RMSE"], 0)
        for field in ("MASE", "RMSSE", "R2", "MAPE_percent", "RMRMSE"):
            self.assertIsNone(result[field], field)
        self.assertEqual(result["MASE_reason"], "zero_training_scale")
        self.assertEqual(result["RMRMSE_reason"], "zero_reference_rmse")
        self.assertEqual(result["MAPE_coverage"], 0)
        self.assertEqual(result["MAPE_reason"], "no_positive_targets")
        no_pairs = amended.dated_training_scales(["2024-01-01", "2024-01-03"], [1., 2.], cutoff="2024-01-03")
        self.assertIsNone(no_pairs["RMSSE"]["value"])
        self.assertEqual(no_pairs["RMSSE"]["reason"], "no_consecutive_training_pairs")

    def test_primary_is_mean_of_individual_seed_rmse_not_ensemble_or_pooled_mse(self):
        truth, predictions = [1., 1.], {271201: [0., 1.], 271202: [1., 3.]}
        value = amended.mean_seed_rmse(truth, predictions)
        self.assertAlmostEqual(value, (math.sqrt(.5) + math.sqrt(2)) / 2)
        self.assertNotAlmostEqual(value, math.sqrt(1.25))
        self.assertNotAlmostEqual(value, math.sqrt(.625))

    def test_nonfinite_predictions_and_shape_changes_are_rejected_without_dropping_dates(self):
        for prediction in ([1.], [1., float("nan")], [1., float("inf")], [-1., 1.]):
            with self.subTest(prediction=prediction), self.assertRaises(ValueError):
                amended.regression_metrics([1., 2.], prediction, training_scales=self.scales)
        with self.assertRaises(ValueError):
            amended.regression_metrics([0.], [1e308], training_scales=self.scales)
        with self.assertRaises(ValueError):
            amended.dated_training_scales(["2024-01-02", "2024-01-01"], [1., 2.], cutoff="2024-01-02")


class LocalSelectionTests(unittest.TestCase):
    def test_hpo_is_local_two_seed_mean_with_stable_id_tie_and_failed_ledger(self):
        rows = [
            {"trial_id": "B", "seed_rmse": {271201: 1., 271202: 3.}},
            {"trial_id": "A", "seed_rmse": {271201: 2., 271202: 2.}},
            {"trial_id": "C", "status": "failed", "reason": "timeout"},
        ]
        result = amended.select_local_hpo(rows, anchor="2024-01-01", scope="synthetic_fixture")
        self.assertEqual(result["winner_trial_id"], "A")
        self.assertEqual(result["winner_mean_rmse"], 2)
        self.assertEqual(result["configuration_count"], 3)
        self.assertFalse(result["registered_selection_complete"])
        self.assertTrue(result["fixture_does_not_complete_registered_search"])
        self.assertEqual(result["ledger"][-1]["reason"], "timeout")
        wrong = deepcopy(rows)
        wrong[1]["anchor"] = "2024-10-01"
        with self.assertRaises(ValueError):
            amended.select_local_hpo(wrong, anchor="2024-01-01", scope="synthetic_fixture")
        with self.assertRaises(ValueError):
            amended.select_local_hpo([{"trial_id": "A", "seed_rmse": {271201: 1.}}],
                                     anchor="2024-01-01", scope="synthetic_fixture")

    def test_incomplete_registered_hpo_domain_cannot_produce_a_normal_winner(self):
        ids = [f"recipe:hpo-{i:03d}" for i in range(18)]
        row = {"trial_id": ids[0], "status": "completed", "seed_rmse": {271201: 1., 271202: 1.}}
        with self.assertRaisesRegex(ValueError, "eighteen"):
            amended.select_local_hpo([row], anchor="2024-01-01", registered_trial_ids=ids)
        with self.assertRaisesRegex(ValueError, "registered trial IDs"):
            amended.select_local_hpo([row], anchor="2024-01-01")
        with self.assertRaisesRegex(ValueError, "eighteen"):
            amended.select_local_hpo([row], anchor="2024-01-01", registered_trial_ids=ids[:1])

    def test_registered_hpo_accounts_for_all_eighteen_ids_and_aborts_systematic_failures(self):
        ids = [f"recipe:hpo-{i:03d}" for i in range(18)]
        rows = [{"trial_id": identity, "status": "completed", "seed_rmse": {271201: 1., 271202: 1.}}
                for identity in reversed(ids)]
        rows[0] = {"trial_id": rows[0]["trial_id"], "status": "failed", "reason": "declared timeout"}
        result = amended.select_local_hpo(rows, anchor="2024-01-01", registered_trial_ids=ids)
        self.assertEqual(result["winner_trial_id"], ids[0])
        self.assertTrue(result["registered_domain_accounted_for"])
        self.assertEqual(result["configuration_count"], 18)
        self.assertEqual(result["ledger"][0]["status"], "failed")
        foreign = deepcopy(rows)
        foreign[-1]["trial_id"] = "foreign:hpo-000"
        with self.assertRaisesRegex(ValueError, "foreign"):
            amended.select_local_hpo(foreign, anchor="2024-01-01", registered_trial_ids=ids)
        unjustified = deepcopy(rows)
        unjustified[0]["reason"] = ""
        with self.assertRaisesRegex(ValueError, "reason"):
            amended.select_local_hpo(unjustified, anchor="2024-01-01", registered_trial_ids=ids)
        for field in ("fatal", "systematic_failure"):
            blocked = deepcopy(rows)
            blocked[0][field] = True
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "blocks"):
                amended.select_local_hpo(blocked, anchor="2024-01-01", registered_trial_ids=ids)

    def test_nas_ci_overlap_is_only_a_parsimony_rule_not_equivalence(self):
        rows = [
            {"architecture_id": "large", "mean_rmse": .01, "ci95": [.009, .011], "parameters": 1000},
            {"architecture_id": "small", "mean_rmse": .011, "ci95": [.0105, .0115], "parameters": 100},
            {"architecture_id": "bad", "mean_rmse": .02, "ci95": [.019, .021], "parameters": 1},
        ]
        result = amended.nas_parsimony_choice(rows)
        self.assertEqual(result["winner_architecture_id"], "small")
        self.assertFalse(result["equivalence_proven"])
        self.assertFalse(result["phase2_ledger_verified"])
        self.assertTrue(result["criteria_only"])
        self.assertEqual(result["technical_tie_ids"], ["large", "small"])
        with self.assertRaises(ValueError):
            amended.nas_parsimony_choice(rows, delta=.1)

    def test_k_rule_uses_complete_rank_table_and_keeps_conservative_undefined_case(self):
        stable = [list(range(12)) for _ in range(5)]
        result = amended.choose_nas_k({"TiDE": stable})
        self.assertEqual(result["k"], 20)
        self.assertAlmostEqual(result["median_pairwise_spearman"], 1)
        unstable = [list(range(12)), list(range(11, -1, -1)), list(range(12)),
                    list(range(11, -1, -1)), list(range(12))]
        self.assertEqual(amended.choose_nas_k({"TiDE": unstable})["k"], 40)
        stable[0] = [0.] * 12
        result = amended.choose_nas_k({"TiDE": stable})
        self.assertEqual(result["k"], 40)
        self.assertIsNone(result["median_pairwise_spearman"])
        stable[0][0] = float("nan")
        result = amended.choose_nas_k({"TiDE": stable})
        self.assertEqual(result["k"], 40)
        self.assertFalse(result["full_score_table_complete"])
        with self.assertRaises(ValueError):
            amended.choose_nas_k({"TiDE": stable}, last_label="2024-09-23")


class BudgetAndCacheTests(unittest.TestCase):
    def identity(self, model="TimesNet", recipe="TN-D7-PUBLISHED-v1"):
        result = {key: "a" * 64 for key in amended.HASH_FIELDS}
        result.update(model_id=model, recipe_id=recipe, origin="2024-01-01", architecture_id=None,
                      seed=271201, status="completed")
        return result

    def test_budget_is_cumulative_and_a_reservation_cannot_reset_it(self):
        result = amended.remaining_budget(215448.629525904, 7200)
        self.assertEqual(result["limit_seconds_total"], 230400)
        self.assertAlmostEqual(result["remaining_before_reservation_seconds"], 14951.370474096)
        self.assertAlmostEqual(result["remaining_seconds"], 7751.370474096)
        self.assertTrue(result["can_admit"])
        self.assertFalse(result["budget_reset"])
        blocked = amended.remaining_budget(230399, 2)
        self.assertFalse(blocked["can_admit"])
        self.assertEqual(blocked["remaining_seconds"], 0)
        self.assertEqual(blocked["deficit_seconds"], 1)
        over = amended.remaining_budget(230401, 0)
        self.assertFalse(over["can_admit"])
        self.assertEqual(over["consumed_seconds"], 230401)

    def test_unknown_negative_and_nonfinite_budget_values_never_become_zero_consumption(self):
        for consumed, reserved in ((None, 0), (-1, 0), (0, -1), (float("nan"), 0), (0, float("inf")), (True, 0), (10**500, 0)):
            with self.subTest(consumed=consumed, reserved=reserved), self.assertRaises(ValueError):
                amended.remaining_budget(consumed, reserved)
        with self.assertRaises(ValueError):
            amended.remaining_budget(0, 0, limit_hours=0)

    def test_cache_needs_every_identity_and_exact_execution_recipe_data_and_seed(self):
        expected = self.identity()
        self.assertTrue(amended.cache_reuse_decision(expected, deepcopy(expected))["reusable"])
        for key in ("hpo_sha256", "data_sha256", "model_source_sha256", "prepared_signature", "execution_policy_sha256"):
            observed = deepcopy(expected)
            observed[key] = "b" * 64
            result = amended.cache_reuse_decision(expected, observed)
            self.assertFalse(result["reusable"], key)
            self.assertIn(key, result["mismatched_fields"])
        observed = deepcopy(expected)
        del observed["data_sha256"]
        self.assertFalse(amended.cache_reuse_decision(expected, observed)["reusable"])
        observed = deepcopy(expected)
        observed["status"] = "unknown"
        self.assertFalse(amended.cache_reuse_decision(expected, observed)["reusable"])

    def test_legacy_restorations_and_tide_winner_architectures_cannot_alias_reference_hpo(self):
        for model, recipe in amended.RESTORED_RECIPES.items():
            expected = self.identity(model, recipe)
            old = self.identity(model, recipe.replace("PUBLISHED", "NAS"))
            self.assertFalse(amended.cache_reuse_decision(expected, old)["reusable"])
            self.assertFalse(amended.cache_reuse_decision(old, old)["reusable"])
        expected = self.identity("TiDE", "TIDE-D7-NAS-v1")
        expected["architecture_id"] = "reference-64-64-64-010"
        observed = deepcopy(expected)
        observed["architecture_id"] = "selected-32-64-16-000"
        observed["architecture_sha256"] = "b" * 64
        result = amended.cache_reuse_decision(expected, observed)
        self.assertFalse(result["reusable"])
        self.assertIn("architecture_id", result["mismatched_fields"])


if __name__ == "__main__":
    unittest.main()
