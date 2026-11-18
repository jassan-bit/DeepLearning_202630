"""Negative controls for semantic, keyed scientific-output comparison."""
import importlib.util
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "experiments/e3_remediacion_20261118/reproduction/reproduce.py"
SPEC = importlib.util.spec_from_file_location("e3_scientific_replay", SOURCE)
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


class ScientificTableContracts(unittest.TestCase):
    def test_row_permutation_is_aligned_by_model_not_position(self):
        before = b"model,RMSE_mean\na,0.2\nb,0.5\n"
        after = b"model,RMSE_mean\nb,0.5\na,0.2\n"
        result = replay.compare_csv(before, after, ["model"])
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["same_row_order"])
        self.assertEqual(result["field_discrepancies"][0]["maximum_absolute_difference"], 0)

    def test_mispaired_model_values_fail_even_with_same_value_multiset(self):
        before = b"model,RMSE_mean\na,0.2\nb,0.5\n"
        after = b"model,RMSE_mean\na,0.5\nb,0.2\n"
        result = replay.compare_csv(before, after, ["model"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["failures"]), 2)

    def test_duplicate_model_or_missing_model_cannot_silently_drop_rows(self):
        before = b"model,RMSE_mean\na,0.2\nb,0.5\n"
        for after in (b"model,RMSE_mean\na,0.2\na,0.5\n", b"model,RMSE_mean\na,0.2\n"):
            with self.subTest(after=after), self.assertRaises(ValueError):
                replay.compare_csv(before, after, ["model"])

    def test_absent_calibration_is_not_equivalent_to_zero(self):
        before = b"model,coverage90_percent\na,\n"
        after = b"model,coverage90_percent\na,0\n"
        with self.assertRaisesRegex(ValueError, "cell type changed"):
            replay.compare_csv(before, after, ["model"])

    def test_schema_changes_are_rejected(self):
        before = b"model,RMSE_mean,MAE_mean\na,0.2,0.1\n"
        after = b"model,RMSE_mean\na,0.2\n"
        with self.assertRaisesRegex(ValueError, "schema changed"):
            replay.compare_csv(before, after, ["model"])

    def test_nonfinite_scalar_and_structured_values_are_rejected(self):
        before = b"model,RMSE_mean\na,0.2\n"
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            replay.compare_csv(before, b"model,RMSE_mean\na,nan\n", ["model"])
        before = b'model,gain_ci95\na,"[0.1, 0.2]"\n'
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            replay.compare_csv(before, b'model,gain_ci95\na,"[0.1, NaN]"\n', ["model"])

    def test_structured_interval_change_reports_correct_leaf(self):
        before = b'model,gain_ci95\na,"[0.1, 0.2]"\n'
        after = b'model,gain_ci95\na,"[0.1, 0.3]"\n'
        result = replay.compare_csv(before, after, ["model"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failures"][0]["leaf"], "gain_ci95[1]")

    def test_discrepancies_do_not_pool_metrics_and_probabilities(self):
        before = b"model,RMSE_mean,p_holm\na,0.2,0.001\n"
        after = b"model,RMSE_mean,p_holm\na,0.200000000000001,0.002\n"
        result = replay.compare_csv(before, after, ["model"])
        self.assertEqual(result["status"], "failed")
        records = {row["field"]: row for row in result["field_discrepancies"]}
        self.assertEqual(records["RMSE_mean"]["unit"], "sigma7_original")
        self.assertEqual(records["p_holm"]["unit"], "dimensionless")
        self.assertEqual(records["RMSE_mean"]["tolerance_violations"], 0)
        self.assertEqual(records["p_holm"]["tolerance_violations"], 1)
        self.assertNotIn("maximum_absolute_difference", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
