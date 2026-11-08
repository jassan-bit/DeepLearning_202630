"""Synthetic architecture/selection and pre-fit provenance guards only."""
from pathlib import Path
import csv
import json
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e1_core_models_mlp as models
import run_e1_core_mlp_extension as runner


class CoreExtensionTests(unittest.TestCase):
    def test_mlp_capacity_brackets_and_positive_shape_contract(self):
        import torch
        for width, count in [(32, 1185), (128, 4737)]:
            model = models.build_model("MLP", 7, width, .1)
            self.assertEqual(models.count_parameters(model)["trainable"], count)
            spec = models.architecture_spec(model)
            self.assertEqual(spec["hidden_layers"], 1)
            self.assertEqual(spec["hidden_activation"], "ReLU")
            with torch.inference_mode():
                output = model(torch.zeros(3, 7, 5))
            self.assertEqual(tuple(output.shape), (3,))
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue((output >= 0).all())
            with self.assertRaises(ValueError):
                model(torch.zeros(3, 5, 7))

    def test_imported_recurrent_classes_and_complementary_exclusion(self):
        from dl_models import SingleLayerLSTM
        from dl_models_gru_lstm_tcn import SingleLayerGRU
        self.assertIsInstance(models.build_model("GRU", 7, 16, 0), SingleLayerGRU)
        self.assertIsInstance(models.build_model("LSTM", 7, 16, 0), SingleLayerLSTM)
        with self.assertRaisesRegex(ValueError, "exactly"):
            models.build_model("TCN", 7, 16)

    def test_equal_candidate_effort_with_declared_different_capacities(self):
        recurrent, mlp = runner.grid((16, 32)), runner.grid((32, 128))
        self.assertEqual(len(recurrent), 8)
        self.assertEqual(len(mlp), 8)
        self.assertEqual({row["width"] for row in mlp}, {32, 128})
        self.assertEqual({row["learning_rate"] for row in mlp}, {.001, .003})
        self.assertEqual({row["dropout"] for row in mlp}, {0, .1})
        self.assertEqual(3 * len(mlp) * 2 * 10 * 2, 960)

    def test_new_hpo_can_select_mlp_baseline_and_reject_missing_seed(self):
        candidates, rows = [], []
        for family in runner.MODELS:
            for i, item in enumerate(runner.grid((32, 128) if family == "MLP" else (16, 32))):
                candidate = {**item, "model": family, "history_days": 7}
                candidates.append(candidate)
                for seed in [6101, 6102]:
                    rows.append({**candidate, "study_id": runner.STUDY, "execution_id": "new-only",
                                 "trajectory_seed": seed, "parameters": item["width"] * 37 + 1,
                                 "rmse": (.003 if family == "MLP" else .006) + i * .00001})
        recipes, baseline, summary = runner.select_candidates(rows, candidates)
        self.assertEqual(baseline, "MLP")
        self.assertEqual(set(recipes), set(runner.MODELS))
        self.assertEqual(len(summary), 24)
        with self.assertRaises(ValueError):
            runner.select_candidates(rows[:-1], candidates)
        rows[0]["execution_id"] = "older-execution"
        with self.assertRaisesRegex(ValueError, "same new execution"):
            runner.select_candidates(rows, candidates)

    def test_wrong_or_unpublished_source_cannot_register(self):
        with patch.object(runner, "design", return_value={}), patch.object(runner, "git", return_value="a" * 40):
            with self.assertRaisesRegex(ValueError, "already-published"):
                runner.register(Path(tempfile.gettempdir()) / "unused-e1-core", "b" * 40, "unused")

    def test_remote_plan_receipt_requires_exact_content_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_text(json.dumps({"sha": "a" * 40, "observed_at_utc": "2020-01-01T00:00:00Z", "file_sha256": "c" * 64}))
            runner.publication(path, "a" * 40, file_hash="c" * 64)
            with self.assertRaisesRegex(ValueError, "exact published file"):
                runner.publication(path, "a" * 40, file_hash="d" * 64)

    def test_provenance_tagging_keeps_numeric_decimal_strings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "predictions.csv").write_text("model,y_true,y_pred\nMLP,0.10000000000000001,2.0000000000000001e-3\n")
            (path / "completion.json").write_text(json.dumps({"artifact_sha256": {"predictions.csv": "old"}}))
            runner.tag_predictions(path, {"registration_signature": "new-execution"})
            with (path / "predictions.csv").open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["y_true"], "0.10000000000000001")
            self.assertEqual(row["y_pred"], "2.0000000000000001e-3")
            self.assertEqual(row["execution_id"], "new-execution")
            self.assertEqual(row["study_id"], runner.STUDY)
            self.assertEqual(runner.read(path / "completion.json")["artifact_sha256"]["predictions.csv"], runner.digest(path / "predictions.csv"))

    def test_private_payloads_cannot_be_written_inside_source(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            runner.safe_private(runner.ROOT / "private")

    def test_final_fit_is_blocked_before_published_dimensional_plan(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "validate_plan", return_value={}), patch.object(runner, "execute") as execute:
            with self.assertRaises(FileNotFoundError):
                runner.fit_final(Path(directory), {})
            execute.assert_not_called()
            self.assertFalse((Path(directory) / "final_started.json").exists())

    def test_score_rejects_missing_freeze_before_any_raw_loader_call(self):
        import planned_study_data
        with patch.object(runner, "validate_final", side_effect=ValueError("missing freeze")), patch.object(planned_study_data, "load_reserved_data") as loader:
            with self.assertRaisesRegex(ValueError, "missing freeze"):
                runner.score(Path(tempfile.gettempdir()) / "unused-e1-core", {}, Path("unused-raw"))
            loader.assert_not_called()

    def test_failure_cancels_pending_work_without_fitting_replacements(self):
        failed, queued = Mock(), Mock()
        failed.result.side_effect = RuntimeError("synthetic worker failure")
        pool = Mock()
        pool.submit.side_effect = [failed, queued]
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "ProcessPoolExecutor", return_value=pool), patch.object(runner, "as_completed", return_value=iter([failed])):
            with self.assertRaisesRegex(RuntimeError, "synthetic worker failure"):
                runner.execute(Path(directory), {"registration_signature": "synthetic"}, [{}, {}], Mock(), "pilot")
            failed.cancel.assert_called_once()
            queued.cancel.assert_called_once()
            pool.shutdown.assert_called_once_with(wait=True, cancel_futures=True)


if __name__ == "__main__":
    unittest.main()
