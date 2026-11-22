"""Independent parity checks for exact duplicate bootstrap compression.

Uses deterministic loss/date fixtures. These checks do not calibrate tests.
"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods"))
import primary_rmse as production


def fixture(seed_classes=(0, 1, 0, 1, 1, 0), models=7):
    t = np.arange(29, dtype=float)
    patterns = np.empty((4, 2, len(t)))
    for j in range(4):
        for s in range(2):
            patterns[j, s] = 0.03 + (0.5 + 0.4 * j + 0.3 * s + 0.13 * np.sin(t / (2 + j + s) + 0.2 * j)) ** 2
    model_map = [0, 1, 2, 1, 0, 3, 2] + [2] * 14
    return np.ascontiguousarray(patterns[model_map[:models]][:, seed_classes, :])


class ExactCompressionChecks(unittest.TestCase):
    def test_joint_seed_equal_multiplicity_and_model_duplicates(self):
        losses = fixture()
        reduced, expansion, info = production.exact_bootstrap_reduction(losses)
        self.assertEqual(info["computed_shape"], [5, 2, 29])
        self.assertEqual(info["joint_seed_class_multiplicities"], [3, 3])
        self.assertTrue(info["equal_multiplicity_seed_reduction"])
        self.assertEqual(info["statistical_family_slots_removed"], 0)
        self.assertEqual(info["model_representatives"], [0, 1, 2, 4, 5])
        self.assertNotEqual(expansion[4], 0, "Candidate-identical rival must retain its own channel")
        theta, psi, _, _ = production.functional_and_influence(losses)
        theta_small, psi_small, _, _ = production.functional_and_influence(reduced)
        np.testing.assert_allclose(theta, theta_small[expansion], rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(psi, psi_small[expansion], rtol=2e-13, atol=2e-14)

    def test_unequal_joint_seed_classes_keep_every_seed(self):
        losses = fixture((0, 1, 0, 1, 0))
        reduced, _, info = production.exact_bootstrap_reduction(losses)
        self.assertEqual(info["joint_seed_class_multiplicities"], [3, 2])
        self.assertFalse(info["equal_multiplicity_seed_reduction"])
        self.assertEqual(info["retained_seed_indices"], list(range(5)))
        self.assertEqual(reduced.shape[1], 5)

    def test_seed_equivalence_is_joint_not_per_model(self):
        losses = fixture()
        # Every system still has two seed classes of equal size, but one model
        # has a different partition. The joint classes now have unequal sizes.
        losses[5] = losses[5, [0, 2, 1, 3, 4, 5]]
        _, _, info = production.exact_bootstrap_reduction(losses)
        self.assertFalse(info["equal_multiplicity_seed_reduction"])
        self.assertEqual(info["retained_seed_indices"], list(range(6)))

    def test_bitwise_distinct_traces_are_not_approximately_merged(self):
        losses = fixture()
        losses[3, 0, 0] = np.nextafter(losses[3, 0, 0], np.inf)
        _, _, info = production.exact_bootstrap_reduction(losses)
        self.assertIn(3, info["model_representatives"])
        self.assertNotEqual(info["model_expansion_indices"][1], info["model_expansion_indices"][3])
        self.assertFalse(info["equal_multiplicity_seed_reduction"])

    def test_bootstrap_parity_with_and_without_compression(self):
        for seed_classes in [(0, 1, 0, 1, 1, 0), (0, 1, 0, 1, 0)]:
            losses = fixture(seed_classes)
            n = losses.shape[-1]
            indices = np.stack([np.arange(n), np.roll(np.arange(n), 4), np.arange(n)[::-1], (7 * np.arange(n) + 3) % n])
            settings = dict(reps=len(indices), block_length=4, lags=3, margin=0.02, seed=1, indices=indices)
            for backend in ["native", "numpy"]:
                original = production.bootstrap_functional(losses, backend=backend, compress_exact=False, **settings)
                compressed = production.bootstrap_functional(losses, backend=backend, compress_exact=True, **settings)
                for key in ["theta", "epa_t", "practical_t", "log_ratio_t"]:
                    np.testing.assert_allclose(compressed[key], original[key], rtol=3e-9, atol=2e-11)

    def test_full_analysis_expands_before_holm_and_preserves_original_seed_panel(self):
        losses = fixture(models=21)
        t = np.arange(losses.shape[-1], dtype=float)
        settings = dict(bootstrap_reps=99, block_length=4, hac_lags=3,
                        instruments=np.column_stack([np.ones(len(t)), 1 + 0.3 * np.sin(t / 3)]),
                        reference_reps=999, seed=123, reference_seed=456, bootstrap_backend="native")
        original_function = production.bootstrap_functional

        def uncompressed(*args, **kwargs):
            kwargs["compress_exact"] = False
            return original_function(*args, **kwargs)

        with mock.patch.object(production, "bootstrap_functional", side_effect=uncompressed):
            reference = production.analyze_losses(losses, list(production.EXPECTED_SYSTEMS), **settings)
        actual = production.analyze_losses(losses, list(production.EXPECTED_SYSTEMS), **settings)
        self.assertEqual(actual["n_fixed_seeds"], 6)
        self.assertEqual(actual["family"]["tests"], 120)
        self.assertEqual(actual["family"], reference["family"])
        self.assertEqual(actual["metrics"], reference["metrics"])
        self.assertEqual(len(actual["comparisons"]), 20)
        for first, second in zip(actual["comparisons"], reference["comparisons"]):
            for key in ["mean_rmse_difference", "practical_mean", "relative_rmse_gain", "temporal_effect_cohen_d", "ci95_rmse_difference", "ci99_rmse_difference", "ci95_relative_rmse_gain", "ci99_relative_rmse_gain"]:
                if first[key] is None:
                    self.assertIsNone(second[key])
                else:
                    np.testing.assert_allclose(first[key], second[key], rtol=3e-9, atol=2e-11)
            for method in production.METHODS:
                for null in ["epa", "practical"]:
                    self.assertEqual(first["methods"][method][null]["status"], second["methods"][method][null]["status"])
                    self.assertEqual(first["methods"][method][null]["raw_p"], second["methods"][method][null]["raw_p"])
                    self.assertEqual(first["methods"][method][null]["adjusted_p"], second["methods"][method][null]["adjusted_p"])
            self.assertEqual(first["provisional_statistical_gate"], second["provisional_statistical_gate"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
