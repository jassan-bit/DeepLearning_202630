"""Published fixed SAE staging on artificial D7 tensors, without market access."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import torch
from torch.nn import functional as F
import e2_recipe_models as recipes
import e2_staged_training as training
import e2_published_sae as published


class PublishedSAETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        generator = torch.Generator().manual_seed(632)
        self.x = torch.randn(16, 7, 5, generator=generator) * .2
        self.y = torch.full((16,), 1.5)
        self.vx = torch.randn(4, 7, 5, generator=generator) * .2
        self.vy = torch.full((4,), 1.4)
        self.config = training.TrainConfig(.003, .0001, 16, 4, 20, 27911)

    def fit(self, **kwargs):
        return published.fit_published_sae(self.x, self.y, self.vx, self.vy,
            .02, self.config, fold_id="synthetic/train-before-validation", **kwargs)

    def test_fixed_body_matches_verified_existing_recipe_and_preserves_rng(self):
        before = torch.random.get_rng_state().clone()
        model = published.build_sae(dropout=.3, seed=19).eval()
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            reference = recipes.SAERecipe().eval()
        self.assertEqual(training.state_sha256(model), training.state_sha256(reference))
        torch.testing.assert_close(model(self.x), reference(self.x), atol=0, rtol=0)
        self.assertEqual((model.encoder1.hidden_size, model.encoder2.hidden_size), (25, 26))
        self.assertEqual(model.head.in_features, 26)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 8739)
        self.assertFalse(hasattr(model, "dense"))
        self.assertFalse(any("decoder" in name for name in model.state_dict()))
        self.assertEqual(model.architecture_metadata["recipe_id"], "SAE-D7-PUBLISHED-v1")
        self.assertFalse(model.architecture_metadata["source"]["pdf_downloaded_or_hashed"])
        self.assertTrue(model.architecture_metadata["pretraining_required"])
        self.assertFalse(model.architecture_metadata["pretraining_executed"])

    def test_existing_dropout_rates_change_without_structural_search(self):
        zero = published.build_sae(dropout=0, seed=13)
        half = published.build_sae(dropout=.5, seed=13)
        self.assertEqual(training.state_sha256(zero), training.state_sha256(half))
        self.assertEqual((half.dropout1.p, half.dropout2.p), (.5, .5))
        for value in (True, -.1, .6, float("nan"), "0.3"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                published.build_sae(dropout=value, seed=13)
        for seed in (None, True, -1):
            with self.assertRaises(ValueError):
                published.build_sae(seed=seed)

    def test_three_real_stages_fresh_optimizers_original_targets_and_frozen_transfer(self):
        result = self.fit()
        self.assertEqual(tuple(result.phases), ("ae1", "ae2", "supervised"))
        self.assertEqual([p.optimizer_initial_state_entries for p in result.phases.values()], [0, 0, 0])
        self.assertEqual([p.optimizer_steps for p in result.phases.values()], [4, 4, 4])
        self.assertEqual([p.scheduler_total_steps for p in result.phases.values()], [4, 4, 4])
        self.assertTrue(all(p.initial_state_sha256 != p.selected_state_sha256 for p in result.phases.values()))
        self.assertEqual(result.auxiliary_scale, max(1., float(self.x.abs().max())))
        expected = published._tensor_sha256(self.x / result.auxiliary_scale)
        self.assertEqual(result.audit["auxiliary_training_target_sha256"], expected)
        self.assertEqual(result.audit["encoder1_before_ae2_sha256"], result.audit["encoder1_after_ae2_sha256"])
        self.assertTrue(result.audit["encoder1_frozen_during_ae2"])
        self.assertTrue(result.audit["encoder1_transfer_exact"])
        self.assertTrue(result.audit["encoder2_transfer_exact"])
        for n in (1, 2):
            self.assertNotEqual(result.audit[f"encoder{n}_before_supervised_sha256"],
                                result.audit[f"encoder{n}_after_supervised_sha256"])
        self.assertTrue(result.model.architecture_metadata["pretraining_executed"])
        self.assertTrue(all(p.requires_grad for p in result.model.parameters()))
        self.assertTrue(all(torch.isfinite(p).all() for p in result.model.parameters()))
        record = result.to_dict()
        self.assertEqual(record["optimizer_steps"], 12)
        self.assertFalse(record["decoder_parameters_in_predictor"])
        json.dumps(record, allow_nan=False)

    def test_one_batch_updates_reduce_each_stage_training_objective(self):
        original_reconstruction = training.fit_reconstruction
        original_supervised = training.fit_supervised
        observed = []

        def wrapped_reconstruction(model, x, y, *args, **kwargs):
            model.eval()
            with torch.no_grad():
                initial = float(F.mse_loss(model(x), y))
            result = original_reconstruction(model, x, y, *args, **kwargs)
            observed.append((initial, result.logs[0]["train_mse_scaled"]))
            return result

        def wrapped_supervised(model, x, y, *args, **kwargs):
            model.eval()
            with torch.no_grad():
                initial = float(F.mse_loss(model(x), y))
            result = original_supervised(model, x, y, *args, **kwargs)
            observed.append((initial, result.logs[0]["train_mse_scaled"]))
            return result

        with patch.object(training, "fit_reconstruction", side_effect=wrapped_reconstruction), \
             patch.object(training, "fit_supervised", side_effect=wrapped_supervised):
            self.fit(dropout=0)
        self.assertEqual(len(observed), 3)
        for initial, after_first_step in observed:
            self.assertLess(after_first_step, initial)

    def test_validation_cannot_change_scale_targets_or_one_epoch_training_states(self):
        config = replace(self.config, max_epochs=1)
        a = published.fit_published_sae(self.x, self.y, self.vx, self.vy, .02,
                                      config, fold_id="synthetic/train")
        b = published.fit_published_sae(self.x, self.y, self.vx * 10000, self.vy * 10000,
                                      .02, config, fold_id="synthetic/train")
        self.assertEqual(a.auxiliary_scale, b.auxiliary_scale)
        self.assertEqual(a.audit["auxiliary_training_target_sha256"], b.audit["auxiliary_training_target_sha256"])
        for name in a.phases:
            self.assertEqual(a.phases[name].initial_state_sha256, b.phases[name].initial_state_sha256)
            self.assertEqual(a.phases[name].selected_state_sha256, b.phases[name].selected_state_sha256)
        self.assertGreater(b.audit["validation_auxiliary_fraction_outside_unit_range"], .9)

    def test_fresh_fixed_refit_uses_three_counts_and_no_validation(self):
        epochs = {"ae1": 1, "ae2": 2, "supervised": 3}
        result = published.fit_published_sae(self.x, self.y, None, None, .02,
            self.config, fold_id="synthetic/refit", fixed_epochs=epochs)
        self.assertEqual({n: p.epochs_completed for n, p in result.phases.items()}, epochs)
        self.assertTrue(all(p.stop_reason == "fixed_epochs" for p in result.phases.values()))
        self.assertTrue(all(p.scheduler_total_steps == 4 for p in result.phases.values()))
        self.assertTrue(all(row["validation_score"] is None for p in result.phases.values() for row in p.logs))
        self.assertIsNone(result.audit["validation_auxiliary_fraction_outside_unit_range"])
        with patch.object(training, "fit_reconstruction") as fit:
            for malformed in ({"supervised": 1}, {"ae1": 0, "ae2": 1, "supervised": 1}):
                with self.assertRaises(ValueError):
                    published.fit_published_sae(self.x, self.y, None, None, .02,
                        self.config, fold_id="synthetic/refit", fixed_epochs=malformed)
            with self.assertRaises(ValueError):
                self.fit(fixed_epochs=epochs)
            fit.assert_not_called()

    def test_failed_second_phase_preserves_prior_work_without_retry(self):
        original = training.fit_reconstruction
        calls = []

        def fail_second(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise training.TrainingFailure("simulated interruption", {"optimizer_steps": 1, "retry_performed": False})
            return original(*args, **kwargs)

        with patch.object(training, "fit_reconstruction", side_effect=fail_second):
            with self.assertRaises(training.TrainingFailure) as raised:
                self.fit()
        record = raised.exception.receipt
        self.assertEqual(len(calls), 2)
        self.assertEqual(record["sae_phase"], "ae2")
        self.assertEqual(tuple(record["completed_phases"]), ("ae1",))
        self.assertEqual(record["known_optimizer_steps_all_sae_phases"], 5)
        self.assertFalse(record["retry_performed"])
        json.dumps(record, allow_nan=False)

    def test_resource_cap_fails_before_first_update_and_does_not_retry(self):
        with self.assertRaises(training.TrainingFailure) as raised:
            published.fit_published_sae(self.x, self.y, self.vx, self.vy, .02,
                replace(self.config, memory_limit_bytes=1), fold_id="synthetic/train")
        record = raised.exception.receipt
        self.assertEqual(record["sae_phase"], "ae1")
        self.assertEqual(record["known_optimizer_steps_all_sae_phases"], 0)
        self.assertFalse(record["retry_performed"])
        self.assertEqual(record["completed_phases"], {})

    def test_changed_verified_body_and_unavailable_cuda_block_before_fitting(self):
        with patch.object(published, "BODY_SOURCE_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "body source changed"):
                published.build_sae(seed=13)
        import e2_cuda_training as cuda
        with patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(cuda, "fit_reconstruction") as fit:
            with self.assertRaises(cuda.CudaUnavailableError):
                self.fit(device="cuda:0")
            fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
