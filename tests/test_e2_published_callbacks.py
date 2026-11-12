"""Tiny real optimization and guards; this is not a market campaign."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_published_registry as registry
import e2_published_training as callbacks
import e2_training_bindings as previous
from e2_temporal_training import prepare_b, refit_b
from run_e2_training_synthetic import artificial_panel


class PublishedCallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        panel = artificial_panel()
        cls.prepared = prepare_b(panel, panel.calendar[220], panel.calendar[222])

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def kwargs(self, name):
        return dict(hpo_trials=registry.registered_trials(name)[:1], architectures=[],
            hpo_seeds=[271201, 271202], phase1_seeds=[271301],
            phase2_seeds=[271401, 271402, 271403, 271404, 271405],
            k=40, delta=registry.historical.protocol()["contract"]["delta"], scope="synthetic_fixture")

    def test_fixed_sae_runs_three_stages_without_a_nas_call_and_refits_fresh(self):
        name = "LSTM-autoencoder con predictor"
        with tempfile.TemporaryDirectory() as directory:
            store = previous.CheckpointStore(Path(directory) / "checkpoints")
            adapter = callbacks.PublishedTrainingAdapter(name, store, full_epochs=1, short_epochs=1)
            with patch.object(previous, "run_neural_selection", side_effect=AssertionError("Fixed SAE cannot enter old NAS")):
                selected, stages = callbacks.run_neural_selection(self.prepared, name, adapter, **self.kwargs(name))
            self.assertEqual(tuple(stages), ("hpo",))
            self.assertEqual(selected.fixed_epochs, {"ae1": 1, "ae2": 1, "supervised": 1})
            runs = refit_b(self.prepared, selected, [271801], adapter.refit_candidate)
            self.assertEqual(runs[0].status, "completed")
            self.assertTrue(runs[0].fresh_initialization)
            self.assertEqual(len(adapter.run_records), 3)
            for record in adapter.run_records:
                self.assertEqual(record["resource_use"]["optimization_stages"], 3)
                self.assertTrue(record["resource_use"]["model_construction_metadata"]["pretraining_executed"])
                self.assertIn(record["run_id"], adapter._seen)
                self.assertGreater(record["resource_use"]["optimizer_steps"], 0)
            self.assertFalse(any("decoder" in k for k in runs[0].artifact.model.state_dict()))

    def test_legacy_adapter_architectures_and_registered_hpo_pruning_are_rejected(self):
        name = "LSTM-autoencoder con predictor"
        with tempfile.TemporaryDirectory() as directory:
            store = previous.CheckpointStore(Path(directory) / "checkpoints")
            old = previous.NeuralTrainingAdapter(name, store, full_epochs=1, short_epochs=1)
            with self.assertRaisesRegex(ValueError, "versioned"):
                callbacks.run_neural_selection(self.prepared, name, old, **self.kwargs(name))
            current = callbacks.PublishedTrainingAdapter(name, store)
            kwargs = self.kwargs(name)
            kwargs["architectures"] = [{"widths": [64, 64, 64]}]
            with self.assertRaisesRegex(ValueError, "NAS"):
                callbacks.run_neural_selection(self.prepared, name, current, **kwargs)
            kwargs = self.kwargs(name)
            kwargs["scope"] = "registered_domains"
            with self.assertRaisesRegex(ValueError, "HPO effort"):
                callbacks.run_neural_selection(self.prepared, name, current, **kwargs)
            self.assertEqual(list(store.directory.iterdir()), [])

    def test_checkpoint_failure_keeps_the_reserved_run_id_and_completed_stage_cost(self):
        name = "LSTM-autoencoder con predictor"
        with tempfile.TemporaryDirectory() as directory:
            store = previous.CheckpointStore(Path(directory) / "checkpoints")
            adapter = callbacks.PublishedTrainingAdapter(name, store, full_epochs=1, short_epochs=1)
            candidate = previous.candidate_for(name, registry.registered_trials(name)[0], None, "hpo")
            train = self.prepared.inner_train.clone()
            run_id = adapter._run_id(candidate, 271201, train)
            with patch.object(store, "save", side_effect=OSError("injected disk failure")):
                run = adapter.fit_candidate(candidate, 271201, train, self.prepared.inner_validation.clone())
            self.assertEqual(run.status, "failed")
            self.assertTrue(run.fatal)
            self.assertEqual(run.run_id, run_id)
            self.assertIn(run_id, adapter._seen)
            self.assertEqual(adapter.run_records[-1]["run_id"], run_id)
            self.assertEqual(run.resource_use["optimization_stages"], 3)
            self.assertGreater(run.resource_use["optimizer_steps"], 0)

    def test_tide_retains_the_existing_full_nas_selection_algorithm(self):
        kwargs = self.kwargs("TiDE")
        with patch.object(previous, "run_neural_selection", return_value="retained-nas") as delegate:
            result = callbacks.run_neural_selection("prepared", "TiDE", "old-compatible-adapter", **kwargs)
        self.assertEqual(result, "retained-nas")
        delegate.assert_called_once_with("prepared", "TiDE", "old-compatible-adapter", **kwargs)

    def test_prediction_failure_preserves_completed_training_and_checkpoint(self):
        name = "LSTM-autoencoder con predictor"
        class BrokenPrediction(torch.nn.Module):
            def forward(self, values):
                raise RuntimeError("injected post-fit inference failure")
        with tempfile.TemporaryDirectory() as directory:
            store = previous.CheckpointStore(Path(directory) / "checkpoints")
            adapter = callbacks.PublishedTrainingAdapter(name, store, full_epochs=1, short_epochs=1)
            original_fit = adapter._fit
            def fitted_then_broken(*args, **kwargs):
                run_id, resources, epochs, artifact = original_fit(*args, **kwargs)
                artifact.model = BrokenPrediction()
                return run_id, resources, epochs, artifact
            candidate = previous.candidate_for(name, registry.registered_trials(name)[0], None, "hpo")
            with patch.object(adapter, "_fit", side_effect=fitted_then_broken):
                result = adapter.fit_candidate(candidate, 271201, self.prepared.inner_train.clone(),
                                               self.prepared.inner_validation.clone())
            self.assertEqual(result.status, "failed")
            self.assertTrue(result.fatal)
            self.assertGreater(result.resource_use["optimizer_steps"], 0)
            self.assertEqual(result.resource_use["optimization_stages"], 3)
            self.assertTrue((store.directory / result.resource_use["checkpoint"]["file"]).is_file())
            self.assertEqual(len(adapter.run_records), 1)
            self.assertEqual(adapter.run_records[0]["status"], "failed")
            self.assertEqual(adapter.run_records[0]["run_id"], result.run_id)

    def test_gpu_campaign_migration_cannot_be_silently_started_by_cpu_callback(self):
        with tempfile.TemporaryDirectory() as directory:
            store = previous.CheckpointStore(Path(directory) / "checkpoints")
            with self.assertRaisesRegex(ValueError, "CPU-verified"):
                callbacks.PublishedTrainingAdapter("TimesNet", store, device="cuda:0")


if __name__ == "__main__":
    unittest.main()
