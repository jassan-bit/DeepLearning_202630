"""Synthetic optimization semantics and real-CUDA checks; no market loader.

CPU runs use the explicitly test-only opt-in and compare with the unchanged
historical trainer. CUDA integration tests are skipped on a CPU-only host.
"""
from copy import deepcopy
from contextlib import nullcontext
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import torch
from torch import nn
from torch.nn import functional as F
import e2_staged_training as historical
import e2_cuda_training as cuda
from e2_nas_common import NASConfig


class TinyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(35, 8),
                                    nn.BatchNorm1d(8), nn.LayerNorm(8),
                                    nn.ReLU(), nn.Dropout(.2), nn.Linear(8, 1))

    def forward(self, values):
        return F.softplus(self.layers(values)).squeeze(-1)


class ConstantPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.0))

    def forward(self, values):
        return F.softplus(self.value).expand(len(values))


class NonfiniteAfterOneBatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(35, 1)
        self.calls = 0

    def forward(self, values):
        self.calls += 1
        output = F.softplus(self.linear(values.flatten(1))).squeeze(-1)
        return output if self.calls == 1 else output * float("nan")


class TinyRecurrent(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(5, 4, batch_first=True)
        self.dropout = nn.Dropout(.2)
        self.output = nn.Linear(4, 1)

    def forward(self, values):
        hidden = self.lstm(values)[0][:, -1]
        return F.softplus(self.output(self.dropout(hidden))).squeeze(-1)


class CudaTrainingCPUContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(321001)
        self.x = torch.randn(17, 7, 5)
        self.y = torch.linspace(.3, 1.3, 17)
        self.vx = torch.randn(8, 7, 5)
        self.vy = torch.linspace(.4, 1.2, 8)
        self.config = historical.TrainConfig(.001, .0001, 16, 3, 20, 321002)
        self.cpu = {"device": "cpu", "allow_cpu_for_tests": True}

    def assert_same_stage(self, before, after):
        fields = ("initial_state_sha256", "selected_state_sha256", "best_epoch",
                  "epochs_completed", "optimizer_steps", "optimizer_initial_state_entries",
                  "stop_reason", "selection_metric", "fixed_epochs", "scheduler_total_steps", "seed")
        for name in fields:
            self.assertEqual(getattr(before, name), getattr(after, name), name)
        self.assertEqual(before.logs, after.logs)
        self.assertEqual(set(before.state_dict), set(after.state_dict))
        for name, tensor in before.state_dict.items():
            self.assertTrue(torch.equal(tensor, after.state_dict[name]), name)
            self.assertEqual(after.state_dict[name].device.type, "cpu")

    def test_selection_matches_historical_optimizer_rng_metrics_and_batch_partition(self):
        first = TinyPredictor()
        second = deepcopy(first)
        x_before, y_before = self.x.clone(), self.y.clone()
        expected = historical.fit_supervised(first, self.x, self.y, self.vx, self.vy,
                                              .02, self.config)
        expected_rng = torch.get_rng_state().clone()
        actual = cuda.fit_supervised(second, self.x, self.y, self.vx, self.vy,
                                     .02, self.config, **self.cpu)
        self.assert_same_stage(expected, actual)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        self.assertTrue(torch.equal(self.x, x_before))
        self.assertTrue(torch.equal(self.y, y_before))
        self.assertEqual(actual.optimizer_steps, 6)
        self.assertTrue(all(row["batch_sizes"] == [15, 2] for row in actual.logs))
        self.assertTrue(all(row["examples_seen"] == 17 for row in actual.logs))
        with torch.no_grad():
            rmse = float(F.mse_loss(second(self.vx), self.vy).sqrt())*.02
        self.assertAlmostEqual(rmse, actual.logs[actual.best_epoch-1]["validation_score"], places=8)
        self.assertEqual(actual.to_dict()["device"], "cpu")
        json.dumps(actual.to_dict(), allow_nan=False)
        old_checkpoint = historical.state_sha256(actual.state_dict)
        with torch.no_grad():
            next(second.parameters()).add_(1)
        self.assertEqual(historical.state_sha256(actual.state_dict), old_checkpoint)

    def test_fixed_refit_retains_declared_scheduler_horizon_and_has_no_validation(self):
        config = replace(self.config, max_epochs=200, warmup_fraction=.05)
        original = TinyPredictor()
        expected = historical.fit_supervised(deepcopy(original), self.x, self.y,
                                              None, None, .02, config, fixed_epochs=2)
        actual = cuda.fit_supervised(deepcopy(original), self.x, self.y,
                                     None, None, .02, config, fixed_epochs=2, **self.cpu)
        self.assert_same_stage(expected, actual)
        self.assertEqual(actual.scheduler_total_steps, 400)
        self.assertEqual(actual.optimizer_steps, 4)
        self.assertEqual(actual.logs[0]["learning_rate_first"], .001/20)
        self.assertTrue(all(row["validation_score"] is None for row in actual.logs))
        with self.assertRaises(ValueError):
            cuda.fit_supervised(TinyPredictor(), self.x, self.y, self.vx, self.vy,
                                .02, config, fixed_epochs=2, **self.cpu)

    def test_exact_tie_keeps_first_checkpoint_and_counts_patience_identically(self):
        config = replace(self.config, weight_decay=0, max_epochs=10, patience=2)
        targets = F.softplus(torch.tensor(0.)).expand(17).clone()
        valid = F.softplus(torch.tensor(0.)).expand(8).clone()
        expected = historical.fit_supervised(ConstantPredictor(), self.x, targets,
                                              self.vx, valid, 1., config)
        actual = cuda.fit_supervised(ConstantPredictor(), self.x, targets,
                                     self.vx, valid, 1., config, **self.cpu)
        self.assert_same_stage(expected, actual)
        self.assertEqual((actual.best_epoch, actual.epochs_completed, actual.stop_reason),
                         (1, 3, "patience"))

    def test_sae_three_fresh_stages_match_historical_and_use_train_only_scale(self):
        architecture = NASConfig((16, 16, 16), (1, 0, 1))
        config = replace(self.config, max_epochs=2)
        counts = {"ae1": 2, "ae2": 2, "supervised": 2}
        expected = historical.fit_sae(architecture, self.x, self.y, None, None,
                                      .02, config, fold_id="synthetic-only", fixed_epochs=counts)
        actual = cuda.fit_sae(architecture, self.x, self.y, None, None, .02, config,
                             fold_id="synthetic-only", fixed_epochs=counts, **self.cpu)
        self.assertEqual(actual.auxiliary_scale, max(1., float(self.x.abs().max())))
        self.assertEqual(expected.auxiliary_scale, actual.auxiliary_scale)
        self.assertEqual(set(actual.phases), {"ae1", "ae2", "supervised"})
        for name in expected.phases:
            self.assert_same_stage(expected.phases[name], actual.phases[name])
            self.assertEqual(actual.phases[name].optimizer_initial_state_entries, 0)
        self.assertEqual(historical.state_sha256(expected.model), historical.state_sha256(actual.model))
        self.assertFalse(any("decoder" in name for name in actual.model.state_dict()))
        self.assertEqual(actual.to_dict()["optimizer_steps"], 12)
        with self.assertRaises(ValueError):
            cuda.fit_sae(architecture, self.x, self.y, self.vx*100., self.vy, .02,
                         config, fold_id="synthetic-only", fixed_epochs=counts, **self.cpu)

    def test_nonfinite_failure_preserves_completed_step_and_never_retries(self):
        model = NonfiniteAfterOneBatch()
        with self.assertRaises(historical.TrainingFailure) as caught:
            cuda.fit_supervised(model, self.x, self.y, self.vx, self.vy,
                                .02, self.config, **self.cpu)
        receipt = caught.exception.receipt
        self.assertEqual(receipt["optimizer_steps"], 1)
        self.assertEqual(receipt["error_code"], "nonfinite")
        self.assertFalse(receipt["retry_performed"])
        self.assertEqual(model.calls, 2)

    def test_sae_allocation_boundaries_retain_every_completed_auxiliary_stage(self):
        from e2_nas_autoencoder import SAEPretrainingNAS
        architecture = NASConfig((16, 16, 16), (0, 0, 0))
        config = replace(self.config, max_epochs=2)
        counts = {"ae1": 2, "ae2": 2, "supervised": 2}
        cases = (("prepare_phase2", {"ae1"}, 4),
                 ("transfer_predictor", {"ae1", "ae2"}, 8))
        for method, completed, known_steps in cases:
            with self.subTest(boundary=method):
                with patch.object(SAEPretrainingNAS, method,
                                  side_effect=torch.OutOfMemoryError("Synthetic allocation boundary")):
                    with self.assertRaises(historical.TrainingFailure) as caught:
                        cuda.fit_sae(architecture, self.x, self.y, None, None, .02,
                                     config, fold_id="synthetic-boundary", fixed_epochs=counts,
                                     **self.cpu)
                receipt = caught.exception.receipt
                self.assertEqual(receipt["error_code"], "resource_limit")
                self.assertEqual(receipt["optimizer_steps"], 0)
                self.assertEqual(set(receipt["completed_phases"]), completed)
                self.assertEqual(sum(item["optimizer_steps"]
                                     for item in receipt["completed_phases"].values()), known_steps)
                self.assertEqual(receipt["known_optimizer_steps_all_sae_phases"], known_steps)
                self.assertFalse(receipt["partial_optimizer_update_may_have_occurred"])
                self.assertFalse(receipt["retry_performed"])

    def test_wall_failure_has_zero_known_steps_and_resource_classification(self):
        config = replace(self.config, wall_limit_seconds=1e-12)
        with self.assertRaises(historical.TrainingFailure) as caught:
            cuda.fit_supervised(TinyPredictor(), self.x, self.y, self.vx, self.vy,
                                .02, config, **self.cpu)
        self.assertEqual(caught.exception.receipt["optimizer_steps"], 0)
        self.assertEqual(caught.exception.receipt["error_code"], "resource_limit")

    def test_missing_cuda_rejects_before_mutating_model_or_rng(self):
        model = TinyPredictor()
        state = historical.state_sha256(model)
        torch_state = torch.get_rng_state().clone()
        random_state = random.getstate()
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaises((ValueError, RuntimeError)):
                cuda.fit_supervised(model, self.x, self.y, self.vx, self.vy,
                                    .02, self.config, device="cuda:0")
        self.assertEqual(historical.state_sha256(model), state)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertEqual(random.getstate(), random_state)
        self.assertEqual(next(model.parameters()).device.type, "cpu")

    def test_production_device_resolution_never_accepts_cpu_or_invalid_cuda_indices(self):
        for device in ("cpu", "cuda:-1", "cuda:999999", "mps"):
            with self.subTest(device=device), self.assertRaises((ValueError, RuntimeError)):
                cuda.validate_cuda_device(device)
        with self.assertRaises((ValueError, RuntimeError)):
            cuda.fit_supervised(TinyPredictor(), self.x, self.y, self.vx, self.vy,
                                .02, self.config, device="cpu")

    def test_prediction_returns_cpu_and_restores_mode_without_changing_state_or_rng(self):
        model = TinyPredictor()
        model.train()
        state = historical.state_sha256(model)
        rng = torch.get_rng_state().clone()
        prediction = cuda.predict_scaled(model, self.vx, batch_size=3, **self.cpu)
        self.assertEqual(tuple(prediction.shape), (8,))
        self.assertEqual(prediction.dtype, torch.float32)
        self.assertEqual(prediction.device.type, "cpu")
        self.assertTrue(model.training)
        self.assertEqual(historical.state_sha256(model), state)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        model.eval()
        with torch.no_grad():
            expected = model(self.vx)
        torch.testing.assert_close(prediction, expected, rtol=1e-6, atol=1e-7)
        with self.assertRaises(ValueError):
            cuda.predict_scaled(model, self.vx.double(), **self.cpu)

    def test_simulated_cuda_policy_restores_exact_precision_even_when_body_raises(self):
        # Torch couples allow_tf32 and the named matmul precision. Restoring
        # allow_tf32 last used to turn an original medium setting into high.
        # Only the device/autocast scopes are mocked: flag setters/getters are
        # the real Torch APIs, so this covers that coupling without a GPU.
        def capture():
            return {"precision": torch.get_float32_matmul_precision(),
                    "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                    "benchmark": torch.backends.cudnn.benchmark,
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "deterministic": torch.are_deterministic_algorithms_enabled(),
                    "warn_only": torch.is_deterministic_algorithms_warn_only_enabled()}

        def restore(state):
            torch.use_deterministic_algorithms(state["deterministic"],
                                               warn_only=state["warn_only"])
            torch.backends.cudnn.allow_tf32 = state["cudnn_tf32"]
            torch.backends.cudnn.benchmark = state["benchmark"]
            torch.backends.cudnn.deterministic = state["cudnn_deterministic"]
            torch.backends.cuda.matmul.allow_tf32 = state["matmul_tf32"]
            torch.set_float32_matmul_precision(state["precision"])

        class ProbeError(RuntimeError):
            pass

        original = capture()
        try:
            for precision, allow_tf32 in (("medium", True), ("high", True), ("highest", False)):
                for raises in (False, True):
                    with self.subTest(precision=precision, body_raises=raises):
                        target = {**original, "precision": precision, "matmul_tf32": allow_tf32}
                        restore(target)
                        before = capture()
                        self.assertEqual((before["precision"], before["matmul_tf32"]),
                                         (precision, allow_tf32))
                        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}), \
                             patch.object(torch.cuda, "device", return_value=nullcontext()), \
                             patch.object(torch, "autocast", side_effect=lambda **kwargs: nullcontext()):
                            def exercise():
                                with cuda._execution_policy(torch.device("cuda:0"), True):
                                    self.assertEqual(torch.get_float32_matmul_precision(), "highest")
                                    self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                                    self.assertFalse(torch.backends.cudnn.allow_tf32)
                                    self.assertFalse(torch.backends.cudnn.benchmark)
                                    self.assertTrue(torch.backends.cudnn.deterministic)
                                    self.assertTrue(torch.are_deterministic_algorithms_enabled())
                                    if raises:
                                        raise ProbeError("Synthetic failure inside CUDA policy")
                            if raises:
                                with self.assertRaises(ProbeError):
                                    exercise()
                            else:
                                exercise()
                        self.assertEqual(capture(), before)
        finally:
            restore(original)


@unittest.skipUnless(torch.cuda.is_available(), "Real CUDA unavailable; GPU integration remains pending")
class RealCudaTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    def setUp(self):
        torch.manual_seed(321101)
        self.x = torch.randn(17, 7, 5)
        self.y = torch.linspace(.3, 1.3, 17)
        self.config = historical.TrainConfig(.001, .0001, 16, 2, 20, 321102)

    def test_native_lstm_optimization_uses_cuda_and_exports_cpu_checkpoints(self):
        original = TinyRecurrent()
        first, second = deepcopy(original), deepcopy(original)
        before_flags = (torch.backends.cuda.matmul.allow_tf32,
                        torch.backends.cudnn.allow_tf32,
                        torch.backends.cudnn.benchmark,
                        torch.backends.cudnn.deterministic,
                        torch.are_deterministic_algorithms_enabled())
        result = cuda.fit_supervised(first, self.x, self.y, None, None,
                                     .02, self.config, fixed_epochs=1, device="cuda:0")
        other = cuda.fit_supervised(second, self.x, self.y, None, None,
                                    .02, self.config, fixed_epochs=1, device="cuda:0")
        self.assertEqual(result.selected_state_sha256, other.selected_state_sha256)
        self.assertEqual(next(first.parameters()).device.type, "cuda")
        self.assertTrue(all(value.device.type == "cpu" for value in result.state_dict.values()))
        self.assertEqual(result.to_dict()["device"], "cuda:0")
        self.assertEqual(result.optimizer_steps, 2)
        self.assertNotEqual(result.initial_state_sha256, result.selected_state_sha256)
        prediction = cuda.predict_scaled(first, self.x[:3], device="cuda:0")
        self.assertEqual(prediction.device.type, "cpu")
        self.assertEqual(tuple(prediction.shape), (3,))
        self.assertTrue(bool(torch.isfinite(prediction).all()))
        self.assertEqual(before_flags,
                         (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                          torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                          torch.are_deterministic_algorithms_enabled()))

    def test_device_memory_budget_fails_without_replacement_or_unknown_steps(self):
        with self.assertRaises(historical.TrainingFailure) as caught:
            cuda.fit_supervised(TinyRecurrent(), self.x, self.y, None, None, .02,
                                self.config, fixed_epochs=1, device="cuda:0",
                                gpu_memory_limit_bytes=1)
        receipt = caught.exception.receipt
        self.assertEqual(receipt["error_code"], "resource_limit")
        self.assertEqual(receipt["optimizer_steps"], 0)
        self.assertFalse(receipt["retry_performed"])

    def test_sae_three_real_cuda_stages_export_cpu_states_and_keep_train_only_scale(self):
        train_x, train_y = self.x[:4].clone(), self.y[:4].clone()
        validation_x = self.x[4:6].clone()*1000.
        validation_y = self.y[4:6].clone()
        config = replace(self.config, max_epochs=1)
        result = cuda.fit_sae(NASConfig((16, 16, 16), (1, 0, 1)),
                              train_x, train_y, validation_x, validation_y,
                              .02, config, fold_id="synthetic-cuda-sae", device="cuda:0")
        expected_scale = max(1., float(train_x.abs().max()))
        self.assertEqual(result.auxiliary_scale, expected_scale)
        self.assertGreater(float(validation_x.abs().max()), result.auxiliary_scale)
        self.assertEqual(set(result.phases), {"ae1", "ae2", "supervised"})
        for stage in result.phases.values():
            self.assertEqual(stage.optimizer_initial_state_entries, 0)
            self.assertEqual(stage.optimizer_steps, 1)
            self.assertEqual(stage.epochs_completed, 1)
            self.assertEqual(stage.to_dict()["device"], "cuda:0")
            self.assertTrue(all(value.device.type == "cpu" for value in stage.state_dict.values()))
        self.assertTrue(all(parameter.device.type == "cuda" for parameter in result.model.parameters()))
        self.assertFalse(any("decoder" in name for name in result.model.state_dict()))
        prediction = cuda.predict_scaled(result.model, validation_x, device="cuda:0")
        self.assertEqual(prediction.device.type, "cpu")
        self.assertEqual(tuple(prediction.shape), (2,))
        self.assertTrue(bool(torch.isfinite(prediction).all()))
        json.dumps(result.to_dict(), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
