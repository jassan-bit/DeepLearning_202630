"""Synthetic checks of conditional planning; fixtures can never close real R."""
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_amended_pilot_power as power


def fixture(*, constant_seeds=False):
    dates = [(date(2024, 11, 8) + timedelta(days=i)).isoformat() for i in range(53)]
    x = np.arange(53, dtype=float)
    truth = (.02 + .001*np.sin(x/3)).tolist()
    persistence = (np.asarray(truth) + .004 + .00001*np.sin(x/4)).tolist()
    mask = [True] * 53
    context = {"data_sha256": "a"*64, "recipe_freeze_sha256": "b"*64,
               "execution_policy_sha256": "c"*64, "model_recipes": dict(power.RECIPES)}
    predictions, records = {}, []
    for i, model in enumerate(power.MODELS):
        predictions[model] = {}
        for j, seed in enumerate(power.SEEDS):
            seed_offset = 0 if constant_seeds else (j-2)*.000012*(i+1)
            values = (np.asarray(truth) + .0045 + i*.0002 + seed_offset
                      + .00001*np.cos(x/(i+2))).tolist()
            predictions[model][str(seed)] = values
            records.append({"model_id": model, "seed": seed, "status": "completed",
                "fresh_initialization": True, "optimizer_steps": 5,
                "recipe_id": power.RECIPES[model], "prepared_signature": "d"*64,
                "model_source_sha256": "e"*64, "hpo_sha256": "f"*64,
                "artifact_sha256": "1"*64, "data_sha256": context["data_sha256"],
                "execution_policy_sha256": context["execution_policy_sha256"],
                "dates_sha256": power.canonical_sha256(dates),
                "eligible_mask_sha256": power.canonical_sha256(mask),
                "predictions_sha256": power.canonical_sha256(values),
                "train_cutoff": "2024-10-31", "recipe_selection_origin": "2024-10-01"})
    payload = {"dates": dates, "truth": truth, "persistence": persistence,
               "eligible_mask": mask, "predictions": predictions,
               "provenance": {"synthetic_only": True, "market_training_executed": False,
                    "artifact_hashes_verified": False, "context_sha256": power.canonical_sha256(context),
                    "truth_sha256": power.canonical_sha256(truth),
                    "persistence_sha256": power.canonical_sha256(persistence),
                    "dates_sha256": power.canonical_sha256(dates),
                    "eligible_mask_sha256": power.canonical_sha256(mask), "records": records}}
    return payload, context


class PilotPowerValidationTests(unittest.TestCase):
    def setUp(self):
        self.payload, self.context = fixture()

    def analyze(self, payload=None, context=None, **kwargs):
        return power.analyze_pilot_power(self.payload if payload is None else payload,
            expected_context=self.context if context is None else context,
            scope="synthetic_fixture", simulations=2000, calibration_simulations=20000, **kwargs)

    def test_synthetic_flags_cannot_close_registered_market_pilot(self):
        with self.assertRaisesRegex(ValueError, "synthetic/unverified"):
            power.analyze_pilot_power(self.payload, expected_context=self.context)

    def test_incomplete_dates_missing_seed_and_duplicate_receipts_fail_before_simulation(self):
        shortened = deepcopy(self.payload)
        shortened["dates"].pop()
        with self.assertRaisesRegex(ValueError, "53"):
            self.analyze(shortened)
        missing_seed = deepcopy(self.payload)
        del missing_seed["predictions"]["TimeMixer"][str(power.SEEDS[0])]
        with self.assertRaisesRegex(ValueError, "five"):
            self.analyze(missing_seed)
        duplicate = deepcopy(self.payload)
        duplicate["provenance"]["records"][1] = deepcopy(duplicate["provenance"]["records"][0])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            self.analyze(duplicate)

    def test_payload_changes_or_mixed_recipe_identity_are_rejected(self):
        changed = deepcopy(self.payload)
        changed["predictions"]["TimeMixer"][str(power.SEEDS[0])][0] += .001
        with self.assertRaisesRegex(ValueError, "predictions differ"):
            self.analyze(changed)
        wrong = deepcopy(self.payload)
        wrong["provenance"]["records"][0]["recipe_id"] = "TM-D7-NAS-v1"
        with self.assertRaisesRegex(ValueError, "different recipe"):
            self.analyze(wrong)
        wrong = deepcopy(self.payload)
        wrong["provenance"]["records"][1]["hpo_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "share one selected recipe"):
            self.analyze(wrong)

    def test_simulation_minima_are_not_silently_shortened(self):
        with self.assertRaisesRegex(ValueError, "2000"):
            power.analyze_pilot_power(self.payload, expected_context=self.context,
                scope="synthetic_fixture", simulations=1999)
        with self.assertRaisesRegex(ValueError, "20000"):
            power.analyze_pilot_power(self.payload, expected_context=self.context,
                scope="synthetic_fixture", calibration_simulations=19999)

    def test_mc_lower_bound_and_conservative_tail_quantile_are_distinct(self):
        self.assertEqual(power._wilson_lower(0, 2000), 0)
        self.assertLess(power._wilson_lower(1600, 2000), .8)
        required = power._required_successes(2000)
        self.assertGreater(required, 1600)
        self.assertGreaterEqual(power._wilson_lower(required, 2000), .8)
        index = power._conservative_quantile_index(20000)
        self.assertGreater(index, int((1-power.ALPHA_PAIR)*20000)-1)
        self.assertLess(index, 20000)
        with self.assertRaises(ValueError):
            power._conservative_quantile_index(100)


class ConditionalPowerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload, cls.context = fixture()
        cls.result = power.analyze_pilot_power(cls.payload, expected_context=cls.context,
            scope="synthetic_fixture", simulations=2000, calibration_simulations=20000, seed=71)

    def test_fixture_can_exercise_choice_but_never_close_R(self):
        result = self.result
        self.assertEqual(result["status"], "synthetic_fixture_not_R_closure")
        self.assertIsNone(result["R_selected"])
        self.assertEqual(result["fixture_R_candidate"], 6)
        self.assertIsNone(result["achieved_final_power"])
        self.assertFalse(result["measured_power_for_all_twenty_contrasts"])
        self.assertEqual(result["planning_per_contrast_alpha"], .05/20)
        self.assertEqual(result["margin_null_gain"], power.DELTA)
        self.assertEqual(result["planning_true_gain"], 2*power.DELTA)

    def test_minimum_marginal_rule_and_calendar_floor_are_explicit(self):
        result = self.result
        self.assertEqual(result["calendar_days"], 53)
        self.assertEqual(result["primary_block_days"], 14)
        self.assertEqual(result["diagnostic_block_days"], [7, 28])
        for length in (14, 7, 28):
            curves = result["scenarios"][str(length)]["curves"]
            self.assertEqual(len(curves), 25*4)
            for R in power.R_VALUES:
                rows = [row for row in curves if row["R"] == R]
                self.assertEqual(len(rows), 4)
                self.assertTrue(all(row["alternative_simulations"] == 2000 for row in rows))
                self.assertTrue(all(row["calibration_simulations"] == 20000 for row in rows))
            reference = next(row for row in curves if row["R"] == 6 and row["baseline"] == "Persistence")
            last = next(row for row in curves if row["R"] == 30 and row["model"] == reference["model"]
                        and row["baseline"] == "Persistence")
            self.assertGreater(reference["temporal_SD_floor"], 0)
            self.assertEqual(reference["temporal_SD_floor"], last["temporal_SD_floor"])
            self.assertAlmostEqual(reference["independent_calendar_blocks_approx"], 53/length)

    def test_unestimable_seed_variability_does_not_become_perfect_power(self):
        payload, context = fixture(constant_seeds=True)
        result = power.analyze_pilot_power(payload, expected_context=context,
            scope="synthetic_fixture", simulations=2000, calibration_simulations=20000, seed=72)
        self.assertIsNone(result["R_selected"])
        self.assertIsNone(result["fixture_R_candidate"])
        self.assertFalse(result["power_target_met_for_measured_primary_scenarios"])
        self.assertEqual(result["R_operative_resource_recommendation"], 30)
        self.assertFalse(result["resource_recommendation_closes_power"])
        self.assertIsNone(result["conditional_worst_MDE_gain_at_R30"])


if __name__ == "__main__":
    unittest.main()
