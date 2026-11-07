"""Data/provenance guards only: these tests never create or fit a model."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import reproduce_e1_final_curves as replay


class CurveReproductionGuards(unittest.TestCase):
    def test_prepared_plan_preserves_the_complete_registered_panel(self):
        plan = replay.protocol()
        self.assertEqual(plan["planned_trajectories"], 3 * 465)
        self.assertEqual(plan["planned_fit_stages"], 2 * 3 * 465)
        self.assertEqual(plan["representative_figure_seed"], 8101)
        self.assertEqual(plan["baseline_model_unchanged"], "GRU")
        self.assertEqual(plan["data_scope"]["calendar"], ["2023-01-01", "2024-12-31"])
        self.assertFalse(plan["data_scope"]["reserve_scoring_allowed"])
        self.assertFalse(plan["data_scope"]["new_forecasts_allowed"])

    def test_private_artifacts_cannot_be_written_inside_source(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            replay.private_destination(replay.ROOT / "results" / "private-run")

    def test_private_artifacts_cannot_be_written_in_another_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").mkdir()
            with self.assertRaisesRegex(ValueError, "another Git"):
                replay.private_destination(root / "outputs")

    def test_changed_input_rejected_before_training(self):
        with patch.object(replay, "sha", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "source or input changed"):
                replay.protocol()

    def test_unpublished_or_wrong_source_cut_cannot_register(self):
        with patch.object(replay, "git", return_value="a" * 40):
            with self.assertRaisesRegex(ValueError, "already-published"):
                replay.register(Path(tempfile.gettempdir()) / "unused-e1-test", "b" * 40)


if __name__ == "__main__":
    unittest.main()
