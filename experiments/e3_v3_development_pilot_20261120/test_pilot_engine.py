"""Synthetic software tests only: no market files, data download, or market fits."""
import contextlib
from datetime import date, timedelta
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import pilot_engine as engine

_source_env = os.environ.get("E3_PILOT_SOURCE_ROOT")
_source_candidates = [Path(_source_env)] if _source_env else list(Path(__file__).resolve().parents)
SOURCE = next((candidate for candidate in _source_candidates
               if (candidate / "src" / "e3_overlap_model_v3.py").is_file()), None)
if SOURCE is None:
    raise RuntimeError("Set E3_PILOT_SOURCE_ROOT to the frozen source checkout when testing staged files")
PLUGIN = Path(__file__).resolve().parent / "plugin_control.py"
if not PLUGIN.is_file():
    PLUGIN = Path(__file__).resolve().parents[1] / "model" / "plugin_control.py"
sys.path.insert(0, str(SOURCE))


def synthetic_npz(path):
    """Synthetic coherent OHLCV; poison non-training members must never be read."""
    n = (date(2024, 10, 31) - date(2023, 1, 16)).days + 1
    generator = np.random.default_rng(773)
    closes = np.exp(4. + generator.normal(0., .01, (n, 7)).cumsum(1))
    raw = np.stack([closes, closes * 1.01, closes * .99, closes,
                    generator.uniform(1., 5., (n, 7))], axis=-1)
    returns = np.diff(np.log(closes), axis=1)
    a = ((returns - returns.mean(1)[:, None])**2).sum(1) / 7.
    magnitude = generator.uniform(.001, .04, n)
    y = np.sqrt(a + (6. / 49.) * magnitude**2)
    dates = np.array([(date(2023, 1, 16) + timedelta(days=i)).isoformat() for i in range(n)])
    np.savez(path, training_raw=raw, training_y_original=y, training_dates=dates,
             calibration_DO_NOT_READ=np.array([object()], dtype=object),
             test_DO_NOT_READ=np.array([object()], dtype=object))


def small_plan(data):
    return {"config": {**engine.CONFIG, "max_epochs": 2, "batch_size": 532},
            "budget_seconds_per_fit": 30., "admission_sha256": engine.digest(data["admission"]),
            "cohort": engine.cohort("cpu"), "synthetic_fixture": True}


class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory(prefix="e3-synthetic-only-")
        cls.root = Path(cls.temporary.name)
        cls.npz = cls.root / "SYNTHETIC.npz"
        synthetic_npz(cls.npz)
        cls.data = engine.load_data(SOURCE, cls.npz, engine.sha(cls.npz))
        cls.split = engine.split_development(cls.data)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def phase(self, label):
        return engine.module_from_path("synthetic_phase_" + label, SOURCE / engine.SOURCE_FILES[-1])

    def test_train_only_loader_never_indexes_forbidden_members(self):
        original = np.load
        reads = []

        class Guard:
            def __init__(self, archive): self.archive = archive
            def __enter__(self): return self
            def __exit__(self, *args): self.archive.close()
            def __getitem__(self, key):
                reads.append(key)
                if not key.startswith("training_"):
                    raise AssertionError("Forbidden array materialized")
                return self.archive[key]

        with patch("numpy.load", side_effect=lambda *a, **kw: Guard(original(*a, **kw))):
            data = engine.load_data(SOURCE, self.npz, engine.sha(self.npz))
        self.assertEqual(reads, ["training_raw", "training_y_original", "training_dates"])
        self.assertEqual((len(data["train_X"]), len(data["val_X"])), (532, 116))
        self.assertFalse(data["admission"]["forbidden_arrays_loaded"])

    def test_fixed_split_has_seven_row_raw_support_purge(self):
        self.assertEqual([len(self.split[k]) for k in ("select", "purged", "assessment")], [54, 7, 55])
        dates = self.data["val_dates"].astype("datetime64[D]")
        self.assertEqual(int((dates[61] - dates[53]).astype(int)), 8)

    def test_recipe_selection_requires_all_three_and_ignores_assessment(self):
        rows = [{"learning_rate": rate, "selection_RMSE": .2, "assessment": -rate}
                for rate in engine.LEARNING_RATES]
        self.assertEqual(engine.select_recipe(rows), .0003)
        with self.assertRaises(ValueError): engine.select_recipe(rows[:2])

    def test_five_variants_fit_only_synthetic_and_completed_cache_does_not_retrain(self):
        phase = self.phase("all_variants")
        plan = small_plan(self.data)
        output = self.root / "all_variants"
        for name in engine.NEURAL:
            with contextlib.redirect_stdout(io.StringIO()):
                model, first = engine.fit_job(name, engine.HPO_SEED, .001, "synthetic_HPO",
                    self.data, self.split, output, phase, PLUGIN, plan, "cpu")
            checkpoint = output / "jobs" / first["job_id"] / "supervised" / "checkpoint.pt"
            before = engine.sha(checkpoint)
            with contextlib.redirect_stdout(io.StringIO()):
                _, second = engine.fit_job(name, engine.HPO_SEED, .001, "synthetic_HPO",
                    self.data, self.split, output, phase, PLUGIN, plan, "cpu")
            self.assertEqual(before, engine.sha(checkpoint))
            self.assertEqual(first["phase"]["optimizer_steps"], second["phase"]["optimizer_steps"])
            self.assertEqual(sum(p.numel() for p in model.parameters()), engine.PARAMETERS[name])
            pred, crps = engine.evaluate_model(model, name, self.data, self.split["assessment"], "cpu")
            self.assertEqual(pred.shape, (55,))
            self.assertEqual(crps is None, name == "additive_positive_head")

    def test_interrupted_checkpoint_resumes_optimizer_rng_and_original_deadline(self):
        phase = self.phase("interrupted")
        plan = small_plan(self.data)
        original = phase._torch_save

        def stop_after_epoch(path, state):
            original(path, state)
            if state["epoch"] == 1:
                raise KeyboardInterrupt("Synthetic injected interruption after durable epoch")

        output = self.root / "interrupted"
        phase._torch_save = stop_after_epoch
        with self.assertRaises(KeyboardInterrupt), contextlib.redirect_stdout(io.StringIO()):
            engine.fit_job("atom_crps", engine.HPO_SEED, .001, "synthetic", self.data,
                           self.split, output, phase, PLUGIN, plan, "cpu")
        marker = next(output.rglob("phase_started.json"))
        before = marker.read_bytes()
        checkpoint = next(output.rglob("checkpoint.pt"))
        saved = torch.load(checkpoint, weights_only=False)
        self.assertEqual(saved["epoch"], 1)
        self.assertTrue(saved["optimizer"]["state"])
        self.assertIn("rng", saved)
        phase._torch_save = original
        with contextlib.redirect_stdout(io.StringIO()):
            _, record = engine.fit_job("atom_crps", engine.HPO_SEED, .001, "synthetic", self.data,
                                      self.split, output, phase, PLUGIN, plan, "cpu")
        self.assertEqual(marker.read_bytes(), before)
        resumed = torch.load(checkpoint, weights_only=False)
        self.assertEqual(resumed["epoch"], 2)
        self.assertGreater(resumed["steps"], saved["steps"])
        self.assertEqual(resumed["resume_segments"], 1)

    def test_expired_phase_without_trained_checkpoint_rejects_random_fallback(self):
        phase = self.phase("expired")
        model = torch.nn.Linear(2, 1)
        x, y = torch.ones((3, 2)), torch.ones((3, 1))
        with self.assertRaises(phase.PhaseIncomplete):
            phase.phase(model, x, y, x, y, self.root / "expired", {**engine.CONFIG, "learning_rate": .001},
                        identity={"synthetic_fixture": True}, budget_seconds=1e-8)

    def test_changed_completed_request_is_rejected(self):
        phase = self.phase("mismatch")
        plan = small_plan(self.data)
        output = self.root / "mismatch"
        with contextlib.redirect_stdout(io.StringIO()):
            engine.fit_job("atom_crps", engine.HPO_SEED, .001, "synthetic", self.data,
                           self.split, output, phase, PLUGIN, plan, "cpu")
        plan["config"]["max_epochs"] = 3
        with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
            engine.fit_job("atom_crps", engine.HPO_SEED, .001, "synthetic", self.data,
                           self.split, output, phase, PLUGIN, plan, "cpu")

    def test_publication_receipt_cannot_enable_a_different_plan(self):
        plan = {"admission_sha256": "admission", "master_run_id": "synthetic",
                "identity": {"source_commit": "a" * 40}}
        receipt = self.root / "wrong_publication.json"
        engine.write_json(receipt, {"plan_sha256": "wrong", "git_commit": "b" * 40})
        with self.assertRaises(ValueError): engine.verify_publication(plan, receipt)

    def test_preparation_seals_exact_thirty_fit_plan_before_fits_and_rejects_cohort_change(self):
        output = self.root / "prepared"
        output.mkdir()
        engine.write_json(output / "hardware_profile.json", {"synthetic_fixture": True, "device": "cpu"})
        engine.write_json(output / "launcher_identity.json", {"master_run_id": "synthetic-test-only"})
        engine.write_json(output / "launcher_lock.json", {"synthetic_fixture": True})
        with contextlib.redirect_stdout(io.StringIO()):
            plan, _ = engine.prepare(output, SOURCE, PLUGIN, self.npz, engine.sha(self.npz), "a" * 40, "cpu")
        self.assertEqual(plan["total_required_fits"], 30)
        self.assertEqual(plan["required_hpo_fits"], 15)
        self.assertEqual(plan["required_final_fits"], 15)
        self.assertEqual(plan["nominal_fit_budget_seconds"], 3600.)
        self.assertEqual(plan["master_run_id"], "synthetic-test-only")
        self.assertFalse(list(output.rglob("checkpoint.pt")))
        with patch.object(engine, "cohort", return_value={"device": "changed"}), self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
            engine.prepare(output, SOURCE, PLUGIN, self.npz, engine.sha(self.npz), "a" * 40, "cpu")

    def test_durability_barrier_rejects_nonverified_ack(self):
        directory = self.root / "barriers"
        directory.mkdir()
        binding = {"synthetic_fixture": True}
        barrier_id = engine.digest({"stage": "before_fit", "job": "synthetic-job", "binding": binding})
        engine.write_json(directory / (barrier_id + ".json"), {"barrier_id": barrier_id, "remote_verified": False})
        with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
            engine.durability_barrier(directory, "before_fit", "synthetic-job", binding)

    def test_first_validated_checkpoint_barrier_follows_durable_optimizer_state(self):
        phase = self.phase("checkpoint_barrier")
        output = self.root / "checkpoint_barrier"
        stages = []

        def checked_barrier(directory, stage, job, binding):
            stages.append(stage)
            job_directory = output / "jobs" / job
            if stage == "before_fit":
                self.assertTrue((job_directory / "supervised" / "phase_started.json").is_file())
            elif stage == "first_validated_checkpoint":
                checkpoint = job_directory / "supervised" / "checkpoint.pt"
                self.assertEqual(engine.sha(checkpoint), binding["checkpoint_sha256"])
                state = torch.load(checkpoint, weights_only=False)
                self.assertGreater(state["steps"], 0)
                self.assertIsNotNone(state["best_score"])
                self.assertTrue(state["optimizer"]["state"])
            elif stage == "after_job":
                self.assertTrue((job_directory / "job.json").is_file())

        with patch.object(engine, "durability_barrier", side_effect=checked_barrier), contextlib.redirect_stdout(io.StringIO()):
            engine.fit_job("atom_crps", engine.HPO_SEED, .001, "synthetic", self.data,
                           self.split, output, phase, PLUGIN, small_plan(self.data), "cpu", self.root / "acks")
        self.assertEqual(stages, ["before_fit", "first_validated_checkpoint", "after_job"])

    def test_complete_synthetic_pipeline_has_thirty_fits_before_any_assessment_and_cached_restart(self):
        output = self.root / "synthetic_complete_pipeline"
        output.mkdir()
        engine.write_json(output / "hardware_profile.json", {"synthetic_fixture": True, "device": "cpu"})
        engine.write_json(output / "launcher_identity.json", {"master_run_id": "SYNTHETIC_COMPLETE_PIPELINE"})
        engine.write_json(output / "launcher_lock.json", {"synthetic_fixture": True})
        with contextlib.redirect_stdout(io.StringIO()):
            plan, data = engine.prepare(output, SOURCE, PLUGIN, self.npz, engine.sha(self.npz), "a" * 40, "cpu")
        # A separate prospective SOFTWARE-ONLY fixture plan: one epoch, one batch;
        # these overrides are never available through the real pilot CLI.
        plan["config"] = {**plan["config"], "max_epochs": 1, "batch_size": 532}
        plan["synthetic_fixture"] = True
        engine.write_json(output / "plan.json", plan)
        publication = output / "publication_receipt.json"
        engine.write_json(publication, {"plan_sha256": engine.digest(plan), "admission_sha256": plan["admission_sha256"],
                     "source_commit": "a" * 40, "master_run_id": plan["master_run_id"], "git_commit": "b" * 40})
        original_evaluate = engine.evaluate_model

        def assessment_after_all_fits(*args, **kwargs):
            self.assertEqual(len(list((output / "jobs").rglob("job.json"))), 30)
            self.assertTrue((output / "closed_selection.json").is_file())
            return original_evaluate(*args, **kwargs)

        with patch.object(engine, "evaluate_model", side_effect=assessment_after_all_fits), contextlib.redirect_stdout(io.StringIO()):
            receipt = engine.run(output, SOURCE, PLUGIN, data, plan, "cpu", publication)
        self.assertEqual(receipt["completed_fits"], 30)
        self.assertFalse(receipt["R1_accredited"])
        self.assertFalse(receipt["R3_accredited"])
        self.assertEqual(receipt["market_test_arrays_opened"], 0)
        checkpoints = {str(path): engine.sha(path) for path in output.rglob("checkpoint.pt")}
        self.assertEqual(len(checkpoints), 30)
        with contextlib.redirect_stdout(io.StringIO()):
            engine.run(output, SOURCE, PLUGIN, data, plan, "cpu", publication)
        self.assertEqual(checkpoints, {str(path): engine.sha(path) for path in output.rglob("checkpoint.pt")})

    def test_missing_final_variant_cannot_pass_continuation_heuristic(self):
        with self.assertRaises(ValueError): engine.summarize([], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
