"""Offline worker admission and synthetic original-model contracts.

Run with PYTHONDONTWRITEBYTECODE=1 python -B -m unittest test_worker.
The only optimizer calls below use artificial arrays and the original library's
explicit CPU test flag. Production worker entry points remain CUDA-only.
"""
from copy import deepcopy
import ast
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import worker


ROOT = Path(__file__).resolve().parent
BUNDLE = Path(os.environ.get("E2_TEST_BUNDLE_ROOT", str(ROOT / "bundle/D02")))


def schedule():
    return [{"id": "job_" + str(index).zfill(2), "model": model, "seed": seed}
            for index, (model, seed) in enumerate(
                (model, seed) for model in worker.NEURAL for seed in worker.NEW_SEEDS)]


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="e2-worker-contract-")
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.plan = {"jobs": schedule()}
        self.manifest = {"version": worker.VERSION, "run_id": "synthetic-run",
            "plan_fingerprint": hashlib.sha256(worker.canonical(self.plan)).hexdigest(),
            "jobs": deepcopy(self.plan["jobs"]),
            "code_hashes": {"worker.py": worker.sha(ROOT / "worker.py")}}
        self.job = self.manifest["jobs"][0]
        self.reservation = {"sequence": 0, "previous_sha256": "0" * 64,
            "event": "reserve", "reservation_id": "synthetic-reservation",
            "run_id": self.manifest["run_id"],
            "plan_fingerprint": self.manifest["plan_fingerprint"],
            "job_id": self.job["id"], "phase": "fit", "cap_seconds": 600.0}
        self.claim = {"version": worker.VERSION,
            **{key: self.reservation[key] for key in
               ("reservation_id", "run_id", "plan_fingerprint", "job_id", "phase", "cap_seconds")},
            "ack_commit": "a" * 40}
        self.write_ledger([self.reservation])
        self.claim["ledger_event_sha256"] = self.ledger_rows[0]["sha256"]
        self.save()

    def save(self):
        (self.run / "plan.json").write_bytes(worker.canonical(self.plan) + b"\n")
        (self.run / "run_manifest.json").write_bytes(worker.canonical(self.manifest) + b"\n")
        (self.run / "claim.json").write_bytes(worker.canonical(self.claim) + b"\n")

    def rebind_grid(self):
        """Keep fixture identities valid so malformed-grid tests reach that gate."""
        self.plan = {"jobs": deepcopy(self.manifest["jobs"])}
        fingerprint = hashlib.sha256(worker.canonical(self.plan)).hexdigest()
        for row in (self.manifest, self.claim, self.reservation):
            row["plan_fingerprint"] = fingerprint
        self.write_ledger([self.reservation])
        self.claim["ledger_event_sha256"] = self.ledger_rows[0]["sha256"]
        self.save()

    def write_ledger(self, values):
        rows = []
        previous = "0" * 64
        for index, value in enumerate(values):
            row = {**value, "sequence": index, "previous_sha256": previous}
            row.pop("sha256", None)
            row["sha256"] = hashlib.sha256(worker.canonical(row)).hexdigest()
            rows.append(row)
            previous = row["sha256"]
        (self.run / "ledger.jsonl").write_bytes(
            b"".join(worker.canonical(row) + b"\n" for row in rows))
        self.ledger_rows = rows

    def context(self):
        return worker.claim_context(self.run, self.run / "claim.json", self.job["id"], "fit")

    def test_positive_claim_binds_the_active_hash_chained_reservation(self):
        manifest, claim, job = self.context()
        self.assertEqual(manifest, self.manifest)
        self.assertEqual(claim, self.claim)
        self.assertEqual(job, self.job)
        self.assertEqual(len(manifest["jobs"]), 52)
        self.assertEqual({row["seed"] for row in manifest["jobs"]}, set(range(271807, 271811)))

    def test_spent_reservation_cannot_be_replayed(self):
        self.write_ledger([self.reservation, {"event": "finish",
            "reservation_id": self.reservation["reservation_id"], "elapsed_seconds": 1.0}])
        with self.assertRaisesRegex(ValueError, "active coordinator"):
            self.context()

    def test_plan_and_registered_worker_fingerprints_are_enforced(self):
        self.plan["altered_after_registration"] = True
        self.save()
        with self.assertRaisesRegex(ValueError, "plan fingerprint"):
            self.context()
        self.plan.pop("altered_after_registration")
        self.manifest["code_hashes"]["worker.py"] = "0" * 64
        self.save()
        with self.assertRaisesRegex(ValueError, "registered code manifest"):
            self.context()

    def test_only_one_reservation_can_be_active(self):
        self.write_ledger([self.reservation,
            {**self.reservation, "reservation_id": "another-active-reservation"}])
        with self.assertRaisesRegex(ValueError, "active coordinator"):
            self.context()

    def test_phase_reservations_respect_fit_and_evaluation_caps(self):
        self.claim["cap_seconds"] = 1801
        self.save()
        with self.assertRaisesRegex(ValueError, "bounded coordinator"):
            self.context()
        self.claim.update(phase="evaluate", cap_seconds=301)
        self.save()
        with self.assertRaisesRegex(ValueError, "bounded coordinator"):
            worker.claim_context(self.run, self.run / "claim.json", self.job["id"], "evaluate")

    def test_altered_claim_and_ledger_bytes_are_rejected(self):
        for field, value in (("cap_seconds", 599.0), ("ledger_event_sha256", "b" * 64),
                             ("ack_commit", "not-a-commit"), ("run_id", "another-run")):
            with self.subTest(field=field):
                original = deepcopy(self.claim)
                self.claim[field] = value
                self.save()
                with self.assertRaises(ValueError):
                    self.context()
                self.claim = original
        self.save()
        rows = deepcopy(self.ledger_rows)
        rows[0]["cap_seconds"] = 599.0
        (self.run / "ledger.jsonl").write_text(json.dumps(rows[0]) + "\n")
        with self.assertRaisesRegex(ValueError, "hash chain"):
            self.context()

    def test_schedule_requires_exactly_thirteen_models_times_four_new_seeds(self):
        original = deepcopy(self.manifest)
        changes = (lambda jobs: jobs.pop(),
                   lambda jobs: jobs.append(deepcopy(jobs[0])),
                   lambda jobs: jobs[1].update(seed=271801),
                   lambda jobs: jobs[1].update(model=jobs[0]["model"], seed=jobs[0]["seed"]))
        for index, change in enumerate(changes):
            with self.subTest(change=index):
                self.manifest = deepcopy(original)
                change(self.manifest["jobs"])
                self.rebind_grid()
                with self.assertRaisesRegex(ValueError, "exact thirteen-by-four"):
                    self.context()

    def test_old_seed_is_rejected_before_original_imports_or_inputs(self):
        for seed in range(271801, 271807):
            with self.subTest(seed=seed), \
                    patch.object(worker, "source_inventory") as inventory, \
                    patch.object(worker, "original_module") as original, \
                    patch.object(worker, "input_paths") as inputs:
                with self.assertRaisesRegex(ValueError, "additional E2 seed"):
                    worker.fit_job(Path("absent-bundle"), Path("absent-inputs"), self.run,
                                   {**self.job, "seed": seed}, self.claim, self.manifest, {})
                inventory.assert_not_called()
                original.assert_not_called()
                inputs.assert_not_called()

    def test_evaluation_freeze_gate_precedes_all_cohort_and_predictor_access(self):
        worker.immutable_json(self.run / "all_fits_remote_ack.json", {
            "version": worker.VERSION, "run_id": self.manifest["run_id"],
            "plan_fingerprint": self.manifest["plan_fingerprint"], "fits": []})
        with patch.object(worker, "cohort") as cohort, \
                patch.object(worker, "source_inventory") as inventory, \
                patch.object(worker, "input_paths") as inputs, \
                patch.object(worker, "load_predictor") as predictor:
            with self.assertRaisesRegex(ValueError, "All 52"):
                worker.evaluate_job(Path("absent-bundle"), Path("absent-inputs"), self.run,
                                    self.job, self.claim, self.manifest, {})
            for mocked in (cohort, inventory, inputs, predictor):
                mocked.assert_not_called()

    def test_completed_fit_rejects_a_changed_checkpoint(self):
        directory = self.run / "jobs" / self.job["id"]
        directory.mkdir(parents=True)
        receipt = {"version": worker.VERSION, "status": "completed",
            "run_id": self.manifest["run_id"], "plan_fingerprint": self.manifest["plan_fingerprint"]}
        for key in ("checkpoint", "scaler", "calibrator", "calibration_predictions"):
            name = key + ".json"
            (directory / name).write_text("{}\n")
            receipt[key] = name
            receipt[key + "_sha256"] = worker.sha(directory / name)
        receipt["artifacts_sha256"] = {receipt[key]: receipt[key + "_sha256"] for key in
            ("checkpoint", "scaler", "calibrator", "calibration_predictions")}
        worker.immutable_json(directory / "fit_receipt.json", receipt)
        self.assertEqual(worker.completed_fit(self.run, self.job["id"],
            run_id=self.manifest["run_id"], plan_fingerprint=self.manifest["plan_fingerprint"]), receipt)
        (directory / receipt["checkpoint"]).write_text('{"tampered":true}\n')
        with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
            worker.completed_fit(self.run, self.job["id"])

    def test_production_fitter_never_enables_the_cpu_test_route(self):
        tree = ast.parse(inspect.getsource(worker.fit_job))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "fit_reduced_job"]
        self.assertEqual(len(calls), 1)
        keywords = {item.arg: item.value for item in calls[0].keywords}
        self.assertIs(keywords["allow_cpu_for_tests"].value, False)
        self.assertEqual(keywords["device"].value, "cuda:0")
        with self.assertRaisesRegex(ValueError, "require cuda:0"):
            worker.runtime_identity("cpu", Path("absent"))


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch is not available in this test interpreter")
class OriginalLibrarySyntheticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BUNDLE.is_dir():
            raise unittest.SkipTest("The hash-pinned D02 source bundle is not present")
        if not os.environ.get("PYTHONDONTWRITEBYTECODE") == "1":
            raise RuntimeError("Run source-bundle tests with PYTHONDONTWRITEBYTECODE=1 and Python -B")
        import torch
        cls.torch = torch
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, cls.previous_threads)
        cls.report, cls.recipes, cls.roots = worker.source_inventory(BUNDLE)
        cls.code = BUNDLE / "code"

    def synthetic_training(self):
        import numpy as np
        import pandas as pd
        temporal = worker.original_module(self.code, "e2_temporal_training")
        dated = worker.original_module(self.code, "e2_real_executor")
        n = 17
        raw = np.linspace(100., 120., n * 7 * 5).reshape(n, 7, 5)
        raw[:, :, 4] = np.linspace(1000., 2000., n * 7).reshape(n, 7)
        original = np.linspace(.01, .03, n)
        dates = tuple(pd.date_range("2000-01-01", periods=n, tz="UTC"))
        scaler = temporal.FoldScaler.fit(raw, original, dates)
        x = self.torch.tensor(scaler.transform(raw), dtype=self.torch.float32)
        y = self.torch.tensor(scaler.scale_targets(original), dtype=self.torch.float32)
        originals = tuple(map(float, original))
        name = "synthetic_original_fit_api"
        signature = dated.canonical_hash({"name": name, "dates": dated.date_strings(dates),
            "x": temporal._tensor_hash(x), "y": temporal._tensor_hash(y),
            "y_original": list(originals), "scaler": scaler.record()})
        train = temporal.PreparedSplit(name, x, y, originals, dates, scaler, signature)
        train.validate()
        return train

    def test_pinned_source_inventory(self):
        self.assertEqual(self.report["author_files_verified"], 76)
        self.assertEqual(len(self.recipes), 13)
        self.assertEqual(set(self.recipes), set(worker.NEURAL))

    def test_production_runtime_metadata_passes_unchanged_strict_artifact_writer(self):
        """Exercise actual TorchVersion through the production metadata adapter.

        CUDA properties and package admission are mocked; no CUDA operation,
        predictor, optimizer or real cohort is used by this metadata test.
        """
        from contextlib import ExitStack
        from dataclasses import asdict
        from types import SimpleNamespace
        import importlib.metadata
        import re
        from torch.torch_version import TorchVersion

        expected = {}
        for line in (BUNDLE / "requirements_cuda.txt").read_text().splitlines():
            match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+)", line.strip())
            if match:
                expected[match[1]] = match[2]
        self.assertEqual(len(expected), 43)
        actual_torch_version_type = TorchVersion("2.8.0+cu126")
        self.assertIsInstance(actual_torch_version_type, str)
        self.assertIsNot(type(actual_torch_version_type), str)
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}))
            stack.enter_context(patch.object(worker.sys, "version_info", (3, 12, 12)))
            stack.enter_context(patch.object(self.torch, "__version__", actual_torch_version_type))
            stack.enter_context(patch.object(self.torch.version, "cuda", "12.6"))
            stack.enter_context(patch.object(self.torch.cuda, "is_available", return_value=True))
            stack.enter_context(patch.object(self.torch.cuda, "get_device_properties",
                return_value=SimpleNamespace(name="Synthetic metadata GPU", total_memory=85094825984)))
            stack.enter_context(patch.object(importlib.metadata, "version", side_effect=expected.__getitem__))
            for name in ("set_num_threads", "set_num_interop_threads", "use_deterministic_algorithms",
                         "set_float32_matmul_precision"):
                stack.enter_context(patch.object(self.torch, name))
            stack.enter_context(patch.object(self.torch, "backends", SimpleNamespace(
                cudnn=SimpleNamespace(benchmark=False, deterministic=False, allow_tf32=False),
                cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)))))
            runtime = worker.runtime_identity("cuda:0", BUNDLE)

        self.assertIs(type(runtime["torch"]), str)
        self.assertEqual(runtime["torch"], "2.8.0+cu126")
        durable = worker.original_module(self.code, "e2_amended_pilot_artifacts")
        # Reproduce the exact old failure, proving the original validator still
        # rejects a string subclass instead of weakening its admission policy.
        with self.assertRaisesRegex(TypeError, "Artifact metadata must contain only finite JSON values"):
            durable._canonical({"resources": {**runtime, "torch": actual_torch_version_type}})
        historical = worker.original_module(self.code, "e2_staged_training")
        trial = self.recipes["MLP16"]["trial"]
        config = historical.TrainConfig(learning_rate=trial["learning_rate"],
            weight_decay=trial["weight_decay"], batch_size=trial["batch_size"],
            max_epochs=200, patience=20, seed=271807,
            warmup_fraction=trial.get("warmup_fraction", 0), wall_limit_seconds=1800.,
            memory_limit_bytes=12 * 1024**3)
        metadata = {"config": asdict(config), "source": {"run_id": "synthetic-metadata-only"},
            "data": {"train": self.synthetic_training().record(), "validation": None},
            "seed": 271807, "resources": runtime}
        with tempfile.TemporaryDirectory(prefix="e2-production-metadata-") as temporary:
            store = durable.PilotArtifactStore(Path(temporary), "supervised",
                {"seed": 271807, "phase": "supervised", "request_sha256": "a" * 64})
            try:
                store.prepare(metadata)
                self.assertEqual(store.status, "prepared")
                self.assertEqual(store.manifest["checkpoints"], {})
            finally:
                store.close()

    def test_original_mlp_and_three_phase_sae_optimize_only_synthetic_targets(self):
        fitter = worker.original_module(self.code, "e2_reduced_fit")
        train = self.synthetic_training()
        with tempfile.TemporaryDirectory(prefix="e2-original-synthetic-") as temporary:
            for name, batches in (("MLP16", 1), (worker.NEURAL[3], 2)):
                with self.subTest(model=name):
                    phases = ("ae1", "ae2", "supervised") if batches == 2 else ("supervised",)
                    output = Path(temporary) / ("sae" if batches == 2 else "mlp")
                    result = fitter.fit_reduced_job(name, self.recipes[name]["trial"], 271807,
                        train, None, output, fixed_epochs={phase: 1 for phase in phases},
                        source_roots=self.roots, device="cpu", fit_wall_seconds=60,
                        allow_cpu_for_tests=True)
                    self.assertEqual(result["status"], "completed")
                    self.assertTrue(result["synthetic"])
                    self.assertEqual(result["seed"], 271807)
                    self.assertEqual(set(result["phase_records"]), set(phases))
                    for phase, record in result["phase_records"].items():
                        self.assertNotEqual(record["initial_state_sha256"], record["selected_state_sha256"])
                        self.assertEqual(record["fixed_epochs"], 1)
                        self.assertEqual(record["optimizer_steps"], batches)
                        self.assertEqual(record["scheduler_total_steps"], 200 * batches)
                        self.assertEqual(record["stop_reason"], "fixed_epochs")
                        self.assertFalse(record["resume_exact_supported"])
                        self.assertFalse(record["retry_performed"])
                        self.assertTrue(fitter.verify_phase_archive(output / "phases" / phase))
                    request = json.loads((output / "request.json").read_text())
                    self.assertEqual(fitter.verified_fit(output, request)["status"], "completed")
                    self.assertIsNone(request["validation_signature"])
                    self.assertEqual(request["train_signature"], train.signature)
                    checkpoint = output / result["selected_checkpoint"]["path"]
                    historical = worker.original_module(self.code, "e2_staged_training")
                    with patch.object(self.torch, "load", wraps=self.torch.load) as loader:
                        predictor = worker.load_predictor(self.code, self.recipes, self.roots,
                            name, 271807, checkpoint, train.scaler.record())
                    self.assertIs(loader.call_args.kwargs["weights_only"], True)
                    self.assertEqual(loader.call_args.kwargs["map_location"], "cpu")
                    self.assertEqual(historical.state_sha256(predictor),
                        result["phase_records"]["supervised"]["selected_state_sha256"])
                    self.assertFalse(predictor.training)
                    changed_scaler = {**train.scaler.record(),
                                      "target_scale": train.scaler.target_scale * 2}
                    with self.assertRaisesRegex(ValueError, "Checkpoint scaler differs"):
                        worker.load_predictor(self.code, self.recipes, self.roots,
                            name, 271807, checkpoint, changed_scaler)

    def test_all_thirteen_original_builders_forward_on_artificial_inputs(self):
        missing = [name for name in ("einops", "reformer_pytorch")
                   if importlib.util.find_spec(name) is None]
        if missing:
            self.skipTest("Optional synthetic-forward dependencies missing: " + ", ".join(missing))
        standalone = worker.original_module(self.code, "e2_standalone_code")
        result = standalone.smoke_cpu(self.code, self.recipes, self.roots, worker.NEURAL)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(result["models"]), 13)
        self.assertTrue(all(row["real_forward_executed"] and row["finite_nonnegative"]
                            and row["output_shape"] == [1] for row in result["models"]))
        self.assertFalse(result["optimizer_created"])
        self.assertFalse(result["training_performed"])
        self.assertFalse(result["trained_checkpoint_loaded"])
        self.assertFalse(result["scientific_benchmark_reproduced"])


if __name__ == "__main__":
    unittest.main()
