"""Protect the development-only choice and the gate before final/test access."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import close_filipe_multihistory as closure


class ConfigurationClosureTests(unittest.TestCase):
    def test_selects_development_rmse_before_complexity(self):
        scores = [dict(model="MLP", history_days=7, parameters=593, cv_rmse_mean=.012),
                  dict(model="LSTM", history_days=28, parameters=5025, cv_rmse_mean=.010)]
        self.assertEqual(closure.select_configuration(scores)["model"], "LSTM")

    def test_registered_numeric_tie_prefers_smaller_input(self):
        scores = [dict(model="MLP", history_days=28, parameters=2273, cv_rmse_mean=.01),
                  dict(model="MLP", history_days=7, parameters=593, cv_rmse_mean=.01+5e-13)]
        self.assertEqual(closure.select_configuration(scores)["history_days"], 7)

    def test_same_history_tie_prefers_parameter_count_before_name(self):
        scores = [dict(model="CNN1D", history_days=7, parameters=1057, cv_rmse_mean=.01),
                  dict(model="MLP", history_days=7, parameters=593, cv_rmse_mean=.01)]
        self.assertEqual(closure.select_configuration(scores)["model"], "MLP")

    def test_invalid_development_scores_cannot_close_selection(self):
        for rows in ([], [dict(cv_rmse_mean=float("nan"))]):
            with self.assertRaises(ValueError):
                closure.select_configuration(rows)

    def test_partial_development_execution_rejected_before_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.json"
            protocol.write_text("{}")
            (root / "execution_summary.json").write_text(json.dumps({
                "trajectories": 89, "completed_project_fits": 2136,
                "all_registered_trajectories_complete": False, "test_files_opened": 0}))
            with patch.object(closure, "OUT", root), patch.object(closure, "PROTOCOL", protocol):
                with self.assertRaisesRegex(ValueError, "90 new development"):
                    closure.prepare_freeze()

    def test_test_access_marker_blocks_configuration_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.json"
            protocol.write_text("{}")
            (root / "execution_summary.json").write_text(json.dumps({
                "trajectories": 90, "completed_project_fits": 2160,
                "all_registered_trajectories_complete": True, "test_files_opened": 0}))
            (root / "test").mkdir()
            (root / "test/raw_access_started.json").write_text("{}")
            with patch.object(closure, "OUT", root), patch.object(closure, "PROTOCOL", protocol):
                with self.assertRaisesRegex(ValueError, "must not precede"):
                    closure.prepare_freeze()


if __name__ == "__main__":
    unittest.main()
