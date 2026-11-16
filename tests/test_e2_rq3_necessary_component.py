"""Synthetic validation and monotone-bound checks; no new scientific tests."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_rq3_necessary_component as rq3


class NecessaryComponentTests(unittest.TestCase):
    def rows(self):
        return [{"candidate": name, "reference": "GRU32", "p_cost_lower_raw": "1",
                 "blocks": "100", "seeds_per_block": "6"} for name in rq3.MODELS]

    def test_missing_and_duplicate_contrasts_are_not_closed(self):
        with self.assertRaises(ValueError): rq3.derive_bounds(self.rows()[:-1])
        rows = self.rows(); rows[-1] = rows[0]
        with self.assertRaises(ValueError): rq3.derive_bounds(rows)

    def test_nonfinite_and_outside_unit_interval_are_rejected(self):
        for value in ("nan", "inf", "-0.1", "1.1", ""):
            rows = self.rows(); rows[0]["p_cost_lower_raw"] = value
            with self.subTest(value=value), self.assertRaises(ValueError): rq3.derive_bounds(rows)

    def test_subunit_component_never_becomes_one_or_a_full_IUT(self):
        rows = self.rows(); rows[0]["p_cost_lower_raw"] = ".2"; rows[1]["p_cost_lower_raw"] = ".01"
        result = rq3.derive_bounds(rows)
        self.assertEqual(result[0]["IUT_lower_bound"], .2)
        self.assertFalse(result[0]["lower_bound_reaches_one"])
        self.assertTrue(result[0]["compatible_CPU_IUT_rejection_excluded_by_cost"])
        self.assertEqual(result[1]["decision"], "undetermined_from_necessary_component")
        self.assertTrue(all(r["CPU_predictive_gain_p"] is None for r in result))
        self.assertTrue(all(r["full_CPU_IUT_executed"] is False and r["IUT_p_value_estimated"] is False for r in result))


if __name__ == "__main__": unittest.main()
