"""Safety checks for a new pilot registration, without registering or fitting it."""
from copy import deepcopy
from itertools import combinations
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_filipe_pilot as pilot


class PilotRegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This only reads the frozen recipe, development inputs and code hashes.
        # It does not call register(), write a protocol, or create study output.
        cls.template = pilot.make_plan()
        cls.previous_seeds = set()
        for name in ("results/dl_r2_cv_20261104/runs.csv",
                     "results/filipe_multihistory_20261104/runs.csv"):
            cls.previous_seeds.update(pd.read_csv(ROOT / name, usecols=["seed"])["seed"].astype(int))

    def setUp(self):
        self.plan = deepcopy(self.template)
        # Validation must remain a read-only gate, even when plans are invalid.
        for owner, name in ((pilot.fitting, "optimization_fit"),
                            (pilot.fitting, "build_model"),
                            (pilot.fitting, "infer"),
                            (pilot, "write")):
            guard = patch.object(owner, name, side_effect=AssertionError(
                f"Pure registration validation must not call {name}"))
            guard.start()
            self.addCleanup(guard.stop)

    def test_actual_plan_validates_all_four_histories_without_fitting(self):
        pilot.validate_plan(self.plan)
        manifests = pilot.validate_inputs(self.plan)
        self.assertEqual(set(manifests), {7, 14, 21, 28})
        self.assertEqual(sum(len(value["folds"]) for value in manifests.values()), 48)
        for history, manifest in manifests.items():
            with self.subTest(history=history):
                outcomes = [i for fold in manifest["folds"] for i in fold["test_indices"]]
                self.assertEqual(outcomes, list(range(365, 731)))
                self.assertEqual(len(set(outcomes)), 366)
                for fold in manifest["folds"]:
                    self.assertLessEqual(max(fold["train_indices"]),
                                         fold["fit_origin_index"]-history-1)
                    self.assertLessEqual(max(fold["inner_train_indices"]),
                                         min(fold["inner_validation_indices"])-history-1)
                    self.assertEqual(len(fold["inner_validation_indices"]), 90-history)

    def test_fresh_seed_namespace_includes_every_month_and_refit_stage(self):
        def initialization_namespace(seeds):
            return {int(seed)+month*100000+offset
                    for seed in seeds for month in range(12)
                    for offset in (0, 10000000)}
        self.assertEqual(len(self.plan["seeds"]), 5)
        self.assertEqual(len(set(self.plan["seeds"])), 5)
        self.assertTrue(initialization_namespace(self.plan["seeds"]).isdisjoint(
            initialization_namespace(self.previous_seeds)))
        expected_pairs = [{"history_days": history, "model_a": a, "model_b": b}
                          for history in (7, 14, 21, 28)
                          for a, b in combinations(("MLP", "LSTM", "CNN1D"), 2)]
        self.assertEqual(self.plan["primary_pairs"], expected_pairs)

    def test_historical_seed_reuse_is_rejected(self):
        self.plan["seeds"] = sorted(self.previous_seeds)[:5]
        with self.assertRaises(ValueError):
            pilot.validate_plan(self.plan)
        # Also exercise the actual freshness gate when the prescribed namespace
        # has appeared in an earlier execution record.
        self.plan = deepcopy(self.template)
        with patch.object(pilot.pd, "read_csv", return_value=pd.DataFrame(
                {"seed": [self.plan["seeds"][0]]})):
            with self.assertRaisesRegex(ValueError, "new, not reused"):
                pilot.validate_plan(self.plan)

    def test_fixed_recipe_target_and_data_boundaries_cannot_be_changed(self):
        changes = [
            ("width", lambda p: p["models"]["MLP"].update(width=32)),
            ("learning_rate", lambda p: p["models"]["LSTM"].update(learning_rate=.001)),
            ("epoch_budget", lambda p: p["training"].update(max_epochs=101)),
            ("variant", lambda p: p.update(variant="absolute_log")),
            ("target", lambda p: p.update(target_return_days=14)),
            ("horizon", lambda p: p.update(forecast_horizon_days=2)),
            ("history", lambda p: p.update(histories=[7, 14, 21])),
            ("test_access", lambda p: p.update(test_files_opened_allowed=1)),
            ("raw_access", lambda p: p.update(raw_archives_opened_allowed=1)),
        ]
        for label, change in changes:
            with self.subTest(change=label):
                modified = deepcopy(self.template)
                change(modified)
                with self.assertRaises(ValueError):
                    pilot.validate_plan(modified)

    def test_comparison_family_multiplicity_and_margin_are_frozen(self):
        changes = [
            ("missing_pair", lambda p: p["primary_pairs"].pop()),
            ("duplicate_pair", lambda p: p["primary_pairs"].__setitem__(
                -1, deepcopy(p["primary_pairs"][0]))),
            ("family_size", lambda p: p["dimensioning"].update(primary_family_size=3)),
            ("global_alpha", lambda p: p["dimensioning"].update(alpha_global=.1)),
            ("per_pair_alpha", lambda p: p["dimensioning"].update(alpha_each=.05/3)),
            ("power", lambda p: p["dimensioning"].update(power_target=.9)),
            ("minimum_runs", lambda p: p["dimensioning"].update(execution_floor=3)),
            ("margin", lambda p: p["dimensioning"].update(delta_absolute=.1)),
        ]
        for label, change in changes:
            with self.subTest(change=label):
                modified = deepcopy(self.template)
                change(modified)
                with self.assertRaises(ValueError):
                    pilot.validate_plan(modified)

    def test_changed_registered_source_hash_is_rejected_before_loading_data(self):
        source = next(iter(self.plan["source_sha256"]))
        self.plan["source_sha256"][source] = "0"*64
        with patch.object(pilot, "load_development", side_effect=AssertionError(
                "Source mismatch must fail before loading development data")):
            with self.assertRaisesRegex(ValueError, "Registered source changed"):
                pilot.validate_inputs(self.plan)

    def test_persistence_anchor_is_checked_against_the_actual_development_cohort(self):
        dimensioning = self.plan["dimensioning"]
        dimensioning["reference_persistence_rmse"] *= 2
        dimensioning["delta_absolute"] *= 2
        # A self-consistent 5% formula alone must not admit another anchor.
        pilot.validate_plan(self.plan)
        with self.assertRaisesRegex(ValueError, "persistence anchor differs"):
            pilot.validate_inputs(self.plan)


if __name__ == "__main__":
    unittest.main()
