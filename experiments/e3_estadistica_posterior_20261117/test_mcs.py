"""Synthetic tests of the supplementary MCS; no historical data are read."""
from copy import deepcopy
import json
import unittest

import numpy as np

import mcs


def settings(**changes):
    plan = {"mcs": {"alpha": .10, "block_days": 7, "repetitions": 999, "seed": 271811}}
    plan["mcs"].update(changes)
    return plan


def panel(predictions, targets=None):
    length = next(iter(predictions.values())).shape[1]
    return {"predictions": predictions,
            "targets": np.zeros(length) if targets is None else targets,
            "dates": ["synthetic-" + str(i) for i in range(length)],
            "mask": np.ones(length, dtype=bool)}


class MCSTests(unittest.TestCase):
    def test_identical_trajectories_retain_all_with_exact_unit_p(self):
        series = (1.0 + np.sin(np.arange(210) / 8.0))[None, :]
        data = panel({mcs.ORIGINAL_MODEL: series.copy(), "same B": series.copy(), "same A": series.copy()})
        result = mcs.run_mcs(data, settings())
        self.assertEqual(result["status"], "completed")
        self.assertEqual(set(result["included_models"]), set(data["predictions"]))
        self.assertEqual(result["excluded_pvalues"], {})
        self.assertEqual(result["steps"][0]["Tmax"], 0.0)
        self.assertEqual(result["steps"][0]["p"], 1.0)

    def test_clearly_dominated_models_are_removed_and_p_values_monotone(self):
        x = np.arange(350)
        strong = (.10 + .01 * np.sin(x / 7))[None, :]
        moderate = (.65 + .10 * np.sin(x / 11))[None, :]
        weak = (1.40 + .12 * np.cos(x / 9))[None, :]
        data = panel({mcs.ORIGINAL_MODEL: strong, "moderate": moderate, "weak": weak})
        result = mcs.run_mcs(data, settings())
        self.assertEqual(result["included_models"], [mcs.ORIGINAL_MODEL])
        self.assertEqual(set(result["excluded_pvalues"]), {"moderate", "weak"})
        self.assertTrue(all(p < .10 for p in result["excluded_pvalues"].values()))
        path_p = [row["p_stepwise"] for row in result["steps"]]
        self.assertEqual(path_p, sorted(path_p))
        self.assertTrue(result["candidate_in_initial_set"])
        self.assertTrue(result["candidate_included"])

    def test_fixed_rng_is_deterministic_and_input_order_does_not_change_inference(self):
        x = np.arange(210)
        data = panel({"z": (.7 + .1*np.cos(x/9))[None, :],
                      mcs.ORIGINAL_MODEL: (.6 + .15*np.sin(x/11))[None, :],
                      "a": (.65 + .1*np.sin(x/13))[None, :]})
        result = mcs.run_mcs(data, settings())
        reverse = {**data, "predictions": dict(reversed(list(data["predictions"].items())))}
        self.assertEqual(result, mcs.run_mcs(data, settings()))
        self.assertEqual(result, mcs.run_mcs(reverse, settings()))
        json.dumps(result, allow_nan=False)

    def test_identical_survivors_stay_tied_after_initial_anchor_is_eliminated(self):
        for length in (210, 608):
            with self.subTest(calendar=length):
                x = np.arange(length)
                good = (1 + .01*np.sin(x/7))[None, :]
                bad = (10 + .12*np.cos(x/9))[None, :]
                data = panel({"AAA worse": bad, mcs.ORIGINAL_MODEL: good,
                              "copy b": good.copy(), "copy c": good.copy()})
                result = mcs.run_mcs(data, settings(repetitions=199))
                self.assertEqual(result["included_models"],
                                 sorted((mcs.ORIGINAL_MODEL, "copy b", "copy c")))
                self.assertEqual(set(result["excluded_pvalues"]), {"AAA worse"})
                self.assertEqual(result["steps"][-1]["Tmax"], 0.0)
                self.assertEqual(result["steps"][-1]["p"], 1.0)

    def test_seed_averaging_keeps_classical_trajectory_once(self):
        x = np.arange(210)
        first = .2+.01*np.sin(x/7)
        second = .4+.01*np.cos(x/7)
        data = panel({mcs.ORIGINAL_MODEL: np.stack((first, second)), "deterministic": first[None, :]})
        result = mcs.run_mcs(data, settings())
        self.assertEqual(result["mean_loss"][mcs.ORIGINAL_MODEL],
                         float(np.mean(np.mean(np.square(np.stack((first, second))), axis=0))))
        self.assertEqual(result["mean_loss"]["deterministic"], float(np.mean(first**2)))
        self.assertTrue(result["does_not_test_RMSE_margin"])
        self.assertFalse(result["primary_Holm_family_member"])

    def test_zero_variance_nonzero_difference_stops_without_invented_p(self):
        data = panel({mcs.ORIGINAL_MODEL: np.ones((1, 210)), "constant worse": np.full((1, 210), 2.0)})
        result = mcs.run_mcs(data, settings())
        self.assertEqual(result["status"], "not_estimable_zero_relative_variance")
        self.assertEqual(set(result["included_models"]), set(data["predictions"]))
        self.assertIsNone(result["steps"][0]["p"])
        self.assertEqual(result["excluded_pvalues"], {})
        json.dumps(result, allow_nan=False)

    def test_candidate_must_be_in_initial_roster(self):
        with self.assertRaisesRegex(ValueError, "original E3 model"):
            mcs.run_mcs(panel({"a": np.ones((1, 20)), "b": np.ones((1, 20))}), settings())

    def test_invalid_inputs_and_settings_refuse(self):
        good = panel({mcs.ORIGINAL_MODEL: np.ones((1, 20)), "other": np.ones((1, 20))})
        for values in ({"alpha": 0}, {"alpha": 1}, {"block_days": 0}, {"block_days": 21},
                       {"repetitions": 9}, {"seed": -1}, {"seed": True}):
            with self.subTest(settings=values), self.assertRaises(ValueError):
                mcs.run_mcs(good, settings(**values))
        bad = deepcopy(good); bad["predictions"]["other"][0, 3] = np.nan
        with self.assertRaisesRegex(ValueError, "finite R by T"):
            mcs.run_mcs(bad, settings())
        bad = deepcopy(good); bad["mask"][3] = False
        with self.assertRaisesRegex(ValueError, "does not compress"):
            mcs.run_mcs(bad, settings())


if __name__ == "__main__":
    unittest.main()
