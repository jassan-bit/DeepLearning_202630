"""Mathematical boundary checks independent of pilot outcomes."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("planning_math", ROOT / "frozen_sources/historical_planning_math.py")
math_only = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math_only)


class ProspectivePlanningMath(unittest.TestCase):
    def test_null_boundary_power_equals_one_sided_alpha(self):
        # A zero separation is the null boundary, not an observed effect.
        alpha = 0.05 / 57
        self.assertAlmostEqual(math_only.t_power(10, 0.01, 0.0, alpha=alpha), alpha, places=12)

    def test_mde_is_inverse_of_power_for_a_fixed_future_design(self):
        alpha = 0.05 / 57
        gap = math_only.t_mde_separation(20, 0.001, 0.9, alpha=alpha)
        self.assertAlmostEqual(math_only.t_power(20, 0.001, gap, alpha=alpha), 0.9, places=10)

    def test_required_pairs_is_minimum_and_tighter_alpha_cannot_help(self):
        alpha = 0.05 / 57
        needed = math_only.required_pairs(0.001, 0.0003, 0.9, alpha=alpha)
        self.assertGreaterEqual(math_only.t_power(needed, 0.001, 0.0003, alpha=alpha), 0.9)
        self.assertLess(math_only.t_power(needed - 1, 0.001, 0.0003, alpha=alpha), 0.9)
        weaker_family = math_only.required_pairs(0.001, 0.0003, 0.9, alpha=0.05 / 48)
        self.assertGreaterEqual(needed, weaker_family)


if __name__ == "__main__":
    unittest.main()
