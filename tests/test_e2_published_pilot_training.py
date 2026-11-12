"""Synthetic parity/durability contracts; never opens market data."""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import torch
from torch import nn
from torch.nn import functional as F
import e2_staged_training as frozen
import e2_published_pilot_training as pilot


class MemoryStore:
    """Test event sink with exclusive admission and immutable snapshots."""
    def __init__(self):
        self.started = False
        self.events = []
        self.snapshots = []
        self.status = "prepared"

    def begin_training(self, metadata):
        if self.started:
            raise ValueError("Existing or interrupted training requires reconciliation")
        self.started = True
        self.events.append(("training_started", deepcopy(metadata)))

    def event(self, kind, payload):
        self.events.append((kind, deepcopy(payload)))

    def checkpoint(self, payload, label="latest"):
        self.snapshots.append((label, deepcopy(payload)))
        return {"path": label + str(len(self.snapshots)), "sha256": frozen.state_sha256(payload["state_dict"]),
                "label": label}

    def finish(self, status, receipt):
        self.status = status
        self.events.append(("terminal_" + status, deepcopy(receipt)))


class TinyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(35, 8), nn.BatchNorm1d(8),
                                    nn.LayerNorm(8), nn.ReLU(), nn.Dropout(.2), nn.Linear(8, 1))

    def forward(self, x):
        return F.softplus(self.layers(x)).squeeze(-1)


class NoncontiguousPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(35, 4).transpose(0, 1))
        self.output = nn.Linear(4, 1)

    def forward(self, x):
        return F.softplus(self.output(F.linear(x.flatten(1), self.weight))).squeeze(-1)


class InterruptAfterOneBatch(TinyPredictor):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        if self.calls == 2:
            raise KeyboardInterrupt("Synthetic kernel/session interruption")
        return super().forward(x)


class ConstantPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        return F.softplus(self.value).expand(len(x))


class PilotTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(11)
        self.x = torch.randn(17, 7, 5)
        self.y = torch.linspace(.3, 1.3, 17)
        self.vx = torch.randn(8, 7, 5)
        self.vy = torch.linspace(.4, 1.2, 8)
        self.config = frozen.TrainConfig(.001, .0001, 16, 3, 20, 271501)
        self.cpu = {"device": "cpu", "allow_cpu_for_tests": True}

    def assert_parity(self, model, *, fixed_epochs=None, config=None):
        config = config or self.config
        args = (None, None) if fixed_epochs else (self.vx, self.vy)
        expected = frozen.fit_supervised(deepcopy(model), self.x, self.y, *args,
                                         .02, config, fixed_epochs=fixed_epochs)
        expected_rng = torch.get_rng_state().clone()
        store = MemoryStore()
        actual_model = deepcopy(model)
        actual = pilot.fit_pilot(actual_model, self.x, self.y, config, store,
                                *args, fixed_epochs=fixed_epochs, target_scale=.02, **self.cpu)
        for field in ("best_epoch", "epochs_completed", "optimizer_steps"):
            self.assertEqual(getattr(expected, field), getattr(actual, field))
        self.assertEqual(expected.logs, actual.logs)
        self.assertEqual(frozen.state_sha256(expected.state_dict), frozen.state_sha256(actual.state_dict))
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        self.assertEqual(expected.scheduler_total_steps, actual.receipt["scheduler_total_steps"])
        self.assertFalse(actual.receipt["resume_exact_supported"])
        self.assertFalse(actual.receipt["retry_performed"])
        self.assertEqual(store.status, "prepared")  # Caller closes prediction/scoring artifacts.
        self.assertEqual(store.events[-1][0], "training_finished")
        json.dumps(actual.to_dict(), allow_nan=False)
        return actual_model, actual, store

    def test_selection_exactly_matches_frozen_optimizer_dropout_and_singleton(self):
        _, actual, store = self.assert_parity(TinyPredictor())
        self.assertTrue(all(row["batch_sizes"] == [15, 2] for row in actual.logs))
        self.assertEqual(actual.optimizer_steps, 6)
        self.assertEqual(len([kind for kind, _ in store.events if kind == "optimizer_step_completed"]), 0)
        self.assertEqual(len([kind for kind, _ in store.events if kind == "epoch_started"]), 3)
        self.assertEqual(len([kind for kind, _ in store.events if kind == "epoch_completed"]), 3)
        self.assertEqual(actual.receipt["durable_event_cadence"], "epoch")
        initial = store.snapshots[0][1]
        self.assertEqual(store.snapshots[0][0], "initial")
        self.assertEqual(initial["optimizer_state_dict"]["state"], {})
        self.assertEqual(initial["optimizer_steps"], 0)
        self.assertEqual(frozen.state_sha256(initial["state_dict"]), actual.receipt["initial_state_sha256"])
        self.assertTrue(any(label == "best" for label, _ in store.snapshots))
        for label, payload in store.snapshots:
            self.assertIn("batch_generator", payload["rng"])
            self.assertIn("torch_cpu", payload["rng"])
            self.assertEqual(payload["model_state_sha256"], frozen.state_sha256(payload["state_dict"]))
            if label != "initial":
                self.assertTrue(payload["optimizer_state_dict"]["state"])

    def test_per_step_diagnostic_events_are_optional_and_preserve_math(self):
        model, store = TinyPredictor(), MemoryStore()
        config = replace(self.config, max_epochs=1)
        expected = frozen.fit_supervised(deepcopy(model), self.x, self.y,
                                         self.vx, self.vy, 1., config)
        actual = pilot.fit_pilot(deepcopy(model), self.x, self.y, config, store,
                                self.vx, self.vy, step_events=True, **self.cpu)
        self.assertEqual(actual.logs, expected.logs)
        self.assertEqual(frozen.state_sha256(actual.state_dict), expected.selected_state_sha256)
        self.assertEqual(actual.receipt["durable_event_cadence"], "optimizer_step")
        self.assertEqual(len([kind for kind, _ in store.events if kind == "optimizer_step_started"]), 2)
        self.assertEqual(len([kind for kind, _ in store.events if kind == "optimizer_step_completed"]), 2)

    def test_fixed_epochs_do_not_accept_or_score_validation(self):
        config = replace(self.config, max_epochs=200, warmup_fraction=.05, patience=1)
        model, actual, _ = self.assert_parity(TinyPredictor(), fixed_epochs=2, config=config)
        self.assertEqual(actual.receipt["scheduler_total_steps"], 400)
        self.assertEqual(actual.optimizer_steps, 4)
        self.assertTrue(all(row["validation_score"] is None for row in actual.logs))
        self.assertEqual(actual.receipt["stop_reason"], "fixed_epochs")
        with self.assertRaises(ValueError):
            pilot.fit_pilot(model, self.x, self.y, config, MemoryStore(), self.vx, self.vy,
                            fixed_epochs=2, **self.cpu)

    def test_author_noncontiguous_parameters_preserve_optimizer_semantics(self):
        model = NoncontiguousPredictor()
        self.assertFalse(model.weight.is_contiguous())
        _, _, store = self.assert_parity(model)
        self.assertFalse(store.snapshots[0][1]["state_dict"]["weight"].is_contiguous())

    def test_retained_tide_module_preserves_frozen_training_semantics(self):
        from e2_nas_factory import build_nas_model
        model = build_nas_model("TiDE", (16, 16, 16), (0, 1, 0))
        self.assert_parity(model, config=replace(self.config, max_epochs=1))

    def test_strict_ties_keep_first_epoch_and_stop_at_patience(self):
        self.y = F.softplus(torch.tensor(0.)).expand(17).clone()
        self.vy = F.softplus(torch.tensor(0.)).expand(8).clone()
        _, result, store = self.assert_parity(ConstantPredictor(),
            config=replace(self.config, weight_decay=0, max_epochs=10, patience=2))
        self.assertEqual((result.best_epoch, result.epochs_completed), (1, 3))
        self.assertEqual(result.receipt["stop_reason"], "patience")
        self.assertEqual(len([label for label, _ in store.snapshots if label == "best"]), 1)

    def test_dirty_interruption_preserves_initial_and_known_steps_without_retry(self):
        store = MemoryStore()
        model = InterruptAfterOneBatch()
        with self.assertRaises(frozen.TrainingFailure) as caught:
            pilot.fit_pilot(model, self.x, self.y, self.config, store, self.vx, self.vy, **self.cpu)
        receipt = caught.exception.receipt
        self.assertEqual(receipt["status"], "interrupted")
        self.assertEqual(receipt["completed_optimizer_steps"], 1)
        self.assertFalse(receipt["partial_optimizer_update_may_have_occurred"])
        self.assertTrue(receipt["requires_reconciliation"])
        self.assertEqual(store.snapshots[0][0], "initial")
        self.assertEqual(model.calls, 2)
        before = frozen.state_sha256(model)
        with self.assertRaisesRegex(ValueError, "reconciliation"):
            pilot.fit_pilot(model, self.x, self.y, self.config, store, self.vx, self.vy, **self.cpu)
        self.assertEqual(frozen.state_sha256(model), before)

    def test_resource_failure_is_terminal_with_zero_updates(self):
        store = MemoryStore()
        with self.assertRaises(frozen.TrainingFailure) as caught:
            pilot.fit_pilot(TinyPredictor(), self.x, self.y,
                            replace(self.config, wall_limit_seconds=1e-12), store,
                            self.vx, self.vy, **self.cpu)
        self.assertEqual(caught.exception.receipt["error_code"], "resource_limit")
        self.assertEqual(caught.exception.receipt["completed_optimizer_steps"], 0)
        self.assertEqual(store.status, "failed")
        self.assertEqual(store.snapshots[0][0], "initial")

    def test_interruption_inside_optimizer_update_is_unknown_and_cannot_retry(self):
        store = MemoryStore()
        original_step = torch.optim.AdamW.step

        def interrupted_step(optimizer, *args, **kwargs):
            original_step(optimizer, *args, **kwargs)
            raise KeyboardInterrupt("Synthetic interruption before update receipt")

        with patch.object(torch.optim.AdamW, "step", interrupted_step):
            with self.assertRaises(frozen.TrainingFailure) as caught:
                pilot.fit_pilot(TinyPredictor(), self.x, self.y, self.config, store,
                                self.vx, self.vy, **self.cpu)
        receipt = caught.exception.receipt
        self.assertEqual(receipt["completed_optimizer_steps"], 0)
        self.assertTrue(receipt["partial_optimizer_update_may_have_occurred"])
        self.assertEqual(store.status, "interrupted")
        self.assertFalse(receipt["retry_performed"])
        self.assertEqual([label for label, _ in store.snapshots], ["initial"])

    def test_cuda_unavailable_is_recorded_and_never_falls_back(self):
        model, store = TinyPredictor(), MemoryStore()
        before = frozen.state_sha256(model)
        rng = torch.get_rng_state().clone()
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaises(frozen.TrainingFailure) as caught:
                pilot.fit_pilot(model, self.x, self.y, self.config, store, self.vx, self.vy)
        self.assertEqual(caught.exception.receipt["error_code"], "cuda_unavailable")
        self.assertEqual(store.status, "failed")
        self.assertEqual(store.snapshots, [])
        self.assertEqual(frozen.state_sha256(model), before)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        with self.assertRaises(ValueError):
            pilot.fit_pilot(model, self.x, self.y, self.config, MemoryStore(),
                            self.vx, self.vy, device="cpu")

    def test_inference_is_separate_scaled_and_label_free(self):
        model, _, _ = self.assert_parity(TinyPredictor())
        expected = model(self.vx).detach()
        actual = pilot.predict_pilot(model, self.vx, batch_size=3, **self.cpu)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-7))

    def test_durable_store_integration_and_weights_only_snapshots(self):
        from e2_amended_pilot_artifacts import PilotArtifactStore, ArtifactReconciliationRequired
        identity = {"protocol": "synthetic-test-only", "seed": self.config.seed}
        metadata = {"config": {"test": True}, "source": {"fixture": True},
                    "data": {"synthetic": True}, "seed": self.config.seed,
                    "resources": {"cpu_test_only": True}}
        with tempfile.TemporaryDirectory() as root:
            with PilotArtifactStore(root, "fixture-completed", identity) as store:
                store.prepare(metadata)
                outcome = pilot.fit_pilot(TinyPredictor(), self.x, self.y, self.config,
                                         store, self.vx, self.vy, **self.cpu)
                self.assertEqual(store.status, "training")
                self.assertEqual(store.manifest["optimizer_steps_known_lower_bound"], outcome.optimizer_steps)
                self.assertGreater(store.manifest["known_cost_seconds"], 0)
                for label in ("initial", "latest", "best"):
                    reference = store.manifest["checkpoints"][label]
                    payload = torch.load(store.path / reference["path"], weights_only=True)
                    self.assertIn("optimizer_state_dict", payload)
                    self.assertIn("rng", payload)
                store.write_json("score", {"synthetic": True, "score": 1.})
                store.complete({"synthetic": True})
                store.require_reconciled()
            with PilotArtifactStore(root, "fixture-interrupted", identity) as store:
                store.prepare(metadata)
                with self.assertRaises(frozen.TrainingFailure):
                    pilot.fit_pilot(InterruptAfterOneBatch(), self.x, self.y, self.config,
                                    store, self.vx, self.vy, **self.cpu)
                self.assertEqual(store.status, "interrupted")
                self.assertEqual(store.manifest["optimizer_steps_known_lower_bound"], 1)
                with self.assertRaises(ArtifactReconciliationRequired):
                    store.require_reconciled()
            with PilotArtifactStore(root, "fixture-interrupted", identity) as store:
                with self.assertRaises(ArtifactReconciliationRequired):
                    pilot.fit_pilot(TinyPredictor(), self.x, self.y, self.config,
                                    store, self.vx, self.vy, **self.cpu)

    def test_published_time_mixer_and_patchtst_author_bodies(self):
        roots_file = os.environ.get("E2_PUBLISHED_SOURCE_ROOTS")
        if not roots_file:
            self.skipTest("Pinned author roots are supplied explicitly for integration")
        import e2_published_registry as registry
        roots = json.loads(Path(roots_file).read_text())
        config = replace(self.config, max_epochs=1)
        for name in ("TimeMixer", "PatchTST", "GRU32"):
            with self.subTest(model=name):
                model = registry.build_train_model(name, registry.registered_trials(name)[0],
                                                   source_roots=roots, seed=config.seed)
                self.assert_parity(model, config=config)

    @unittest.skipUnless(torch.cuda.is_available(), "A real CUDA device is required")
    def test_real_cuda_matches_frozen_cuda_backend(self):
        import e2_cuda_training as frozen_cuda
        model = TinyPredictor()
        config = replace(self.config, max_epochs=1)
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}):
            expected = frozen_cuda.fit_supervised(deepcopy(model), self.x, self.y,
                                                 self.vx, self.vy, .02, config)
            result = pilot.fit_pilot(deepcopy(model), self.x, self.y, config,
                                     MemoryStore(), self.vx, self.vy, target_scale=.02)
        self.assertEqual(result.logs, expected.logs)
        self.assertEqual(frozen.state_sha256(result.state_dict), expected.selected_state_sha256)
        self.assertEqual(result.receipt["device"], "cuda:0")
        self.assertFalse(result.receipt["runtime"]["test_only_cpu_execution"])


if __name__ == "__main__":
    unittest.main()
