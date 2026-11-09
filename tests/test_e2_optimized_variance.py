"""Metadata and supervisor tests for compute31; no model fit is executed."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_e2_optimized_variance as pilot


COMMIT = "b" * 40
REGISTRATION_SHA = "a" * 64
RUNTIME_POLICY = {
    "registration_sha256": REGISTRATION_SHA,
    "registration_commit": COMMIT,
    "threads": 2,
    "crop": False,
    "kernel_backend": "dense",
    "kernels_summary_sha256": "c" * 64,
    "threads_summary_sha256": "d" * 64,
    "kernel_policy": {"selected_backend": "dense"},
    "thread_policy": {"selected_threads": 2},
}


def registered_fixture():
    return {
        "version": pilot.REGISTRATION_VERSION,
        "source_protocol_sha256": pilot.PROTOCOL_SHA256,
        "cpu_profile": {},
        "resources": {"sampled_worker_group_rss_limit_bytes": 12 * 1024**3},
        "preselection_variance": {
            "models": list(pilot.MODELS),
            "seed_pool": list(pilot.SEEDS),
            "epoch_selection_origin": "2024-10-01",
            "epoch_selection_train_end": "2024-06-25",
            "epoch_selection_validation": ["2024-07-03", "2024-09-23"],
            "refit_origin": "2024-11-08",
            "fresh_refit_training_end": "2024-10-31",
            "evaluation": ["2024-11-08", "2024-12-30"],
            "evaluation_days": 53,
            "max_epochs_per_stage": 200,
            "patience_epochs": 20,
            "internal_stage_wall_seconds": 600,
            "external_worker_wall_seconds": 660,
        },
    }


class OptimizedVarianceHelpers(unittest.TestCase):
    def _registration_file(self, root, value=None):
        path = Path(root) / "registration.json"
        path.write_text(json.dumps(registered_fixture() if value is None else value) + "\n")
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def _spec(self):
        return {
            "version": pilot.VERSION,
            "model": pilot.MODELS[0],
            "seed": pilot.SEEDS[0],
            "phase": "epoch_selection",
            "fixed_epochs": None,
            "registration_commit": COMMIT,
            "expected_registration_sha256": REGISTRATION_SHA,
            "runtime_policy": deepcopy(RUNTIME_POLICY),
        }

    def _fake_campaign(self, root, *, failed_selection=None, censored_selection=None,
                       mutate_worker=None):
        calls, supervision = [], []

        def supervise(command, output_dir, **kwargs):
            spec = json.loads(Path(command[-1]).read_text())
            calls.append(spec)
            supervision.append(kwargs)
            directory = Path(output_dir)
            directory.mkdir()
            identity = (spec["model"], spec["seed"])
            selection = spec["phase"] == "epoch_selection"
            censored = selection and identity == censored_selection
            failed = selection and identity == failed_selection
            worker = None if censored else {
                "version": pilot.VERSION,
                "status": "failed" if failed else "completed",
                "model": spec["model"],
                "seed": spec["seed"],
                "phase": spec["phase"],
                "registration_commit": COMMIT,
                "registration_file_sha256": REGISTRATION_SHA,
                "expected_registration_sha256": REGISTRATION_SHA,
                "runtime_policy": deepcopy(RUNTIME_POLICY),
                "torch_num_threads": 2,
                "torch_num_interop_threads": 1,
                "threads": 2,
                "source_hashes_unchanged": True,
                "best_epoch": None if failed else spec["seed"] - pilot.SEEDS[0] + 1,
                "optimizer_steps": 17 if failed else 5,
            }
            if worker is not None:
                if mutate_worker is not None:
                    mutate_worker(worker)
                pilot.json_new(directory / "worker_profile.json", worker)
            result = {
                "status": "timeout" if censored else "worker_failure" if failed else "success",
                "worker_profile": worker,
                "elapsed_wall_seconds": 660.1 if censored else .01,
                "sampled_peak_process_group_rss_bytes": 1024,
            }
            pilot.json_new(directory / "result.json", result)
            return result

        with patch.object(pilot, "registration", return_value=registered_fixture()), \
             patch.object(pilot, "load_runtime_policy", return_value=deepcopy(RUNTIME_POLICY)), \
             patch("profile_e2_training.run_isolated_profile", side_effect=supervise):
            result = pilot.run_optimized_variance(
                Path(root) / "unused-data", Path(root) / "unused-registration",
                Path(root) / "campaign", registration_commit=COMMIT,
                expected_registration_sha256=REGISTRATION_SHA,
                kernels_summary=Path(root) / "kernels.json",
                threads_summary=Path(root) / "threads.json")
        return result, calls, supervision

    def test_registration_requires_exact_new_bytes_and_full_commit(self):
        with tempfile.TemporaryDirectory() as root:
            path, sha = self._registration_file(root)
            self.assertEqual(pilot.registration(path, sha, COMMIT)["version"],
                             pilot.REGISTRATION_VERSION)
            for wrong_sha, wrong_commit in [("0" * 64, COMMIT), (sha, "abc123"),
                                             (sha, "z" * 40)]:
                with self.assertRaises(ValueError):
                    pilot.registration(path, wrong_sha, wrong_commit)
            path.write_text(path.read_text() + "\n")
            with self.assertRaises(ValueError):
                pilot.registration(path, sha, COMMIT)

    def test_old_registration_and_old_seed_pool_are_rejected_even_with_valid_hash(self):
        for change in ("version", "seed_pool"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as root:
                value = registered_fixture()
                if change == "version":
                    value["version"] = "e2-development-pilot-registration-v1"
                else:
                    value["preselection_variance"]["seed_pool"] = list(range(271901, 271906))
                path, sha = self._registration_file(root, value)
                with self.assertRaises(ValueError):
                    pilot.registration(path, sha, COMMIT)

    def test_worker_rejects_prior_seeds_and_refit_without_frozen_epochs(self):
        spec = self._spec()
        pilot._validate_spec(spec)
        for key, value in [("seed", 271901), ("model", "TimesNet"),
                           ("fixed_epochs", 3), ("registration_commit", "short")]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                pilot._validate_spec({**spec, key: value})
        refit = {**spec, "phase": "fresh_refit", "fixed_epochs": 8}
        pilot._validate_spec(refit)
        for epochs in (None, 0, 201, True):
            with self.subTest(epochs=epochs), self.assertRaises(ValueError):
                pilot._validate_spec({**refit, "fixed_epochs": epochs})

    def test_worker_rejects_unmeasured_thread_count_or_unbound_summary(self):
        for key, value in [("threads", 3), ("threads", True),
                           ("kernel_backend", "cropped"),
                           ("kernels_summary_sha256", "short")]:
            spec = self._spec()
            spec["runtime_policy"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                pilot._validate_spec(spec)

    def test_runtime_policy_cannot_be_bound_to_another_registration(self):
        pilot._bind_policy_registration(RUNTIME_POLICY, REGISTRATION_SHA, COMMIT)
        for sha, commit in [("e" * 64, COMMIT), (REGISTRATION_SHA, "f" * 40)]:
            with self.subTest(sha=sha, commit=commit), self.assertRaises(ValueError):
                pilot._bind_policy_registration(RUNTIME_POLICY, sha, commit)

    def test_epoch_count_requires_exactly_five_new_completed_seeds(self):
        records = [{"seed": seed, "status": "completed", "best_epoch": epoch}
                   for seed, epoch in zip(pilot.SEEDS, [9, 2, 7, 20, 4])]
        self.assertEqual(pilot.median_epoch_count(records, seeds=pilot.SEEDS), 7)
        changed = deepcopy(records)
        changed[2]["seed"] = 271901
        for bad in (records[:-1], records + [deepcopy(records[0])], changed):
            with self.assertRaises(ValueError):
                pilot.median_epoch_count(bad, seeds=pilot.SEEDS)

    def test_all_selection_attempts_precede_fresh_refits_and_caps_are_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            result, calls, supervision = self._fake_campaign(root)
        self.assertEqual(len(calls), 30)
        self.assertTrue(all(c["phase"] == "epoch_selection" for c in calls[:15]))
        self.assertTrue(all(c["phase"] == "fresh_refit" for c in calls[15:]))
        self.assertTrue(all(c["fixed_epochs"] == 3 for c in calls[15:]))
        self.assertEqual({c["seed"] for c in calls}, set(range(273001, 273006)))
        self.assertTrue(all(c["runtime_policy"]["threads"] == 2 for c in calls))
        self.assertTrue(all(c["timeout_seconds"] == 660 for c in supervision))
        self.assertTrue(all(c["memory_limit_bytes"] == 12 * 1024**3 for c in supervision))
        self.assertTrue(result["paired_3x5_complete"])
        self.assertFalse(result["final_R_set"])
        self.assertFalse(result["final_power_claimed"])
        self.assertEqual(result["replacement_attempts"], 0)

    def test_failed_selection_keeps_cost_and_blocks_only_that_model_refits(self):
        failed = (pilot.MODELS[0], pilot.SEEDS[2])
        with tempfile.TemporaryDirectory() as root:
            result, calls, _ = self._fake_campaign(root, failed_selection=failed)
        self.assertEqual(len(calls), 25)
        rows = result["models"][failed[0]]["selection"]
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[2]["optimizer_steps"], 17)
        self.assertEqual(rows[2]["status"], "failed")
        self.assertEqual(result["models"][failed[0]]["refit"], [])
        self.assertIn(failed[0], result["blocked_refits"])
        self.assertFalse(result["paired_3x5_complete"])
        self.assertEqual(result["replacement_attempts"], 0)

    def test_external_censor_keeps_unknown_steps_null_without_replacement(self):
        failed = (pilot.MODELS[1], pilot.SEEDS[1])
        with tempfile.TemporaryDirectory() as root:
            result, calls, _ = self._fake_campaign(root, censored_selection=failed)
        self.assertEqual(len(calls), 25)
        row = result["models"][failed[0]]["selection"][1]
        self.assertEqual(row["supervisor_status"], "timeout")
        self.assertIsNone(row["optimizer_steps"])
        self.assertGreaterEqual(row["elapsed_wall_seconds"], 660)
        self.assertEqual(result["models"][failed[0]]["refit"], [])
        self.assertEqual(result["replacement_attempts"], 0)

    def test_durable_fit_receipt_recovers_known_steps_after_outer_kill(self):
        spec = self._spec()
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(pilot._known_steps(None, root, spec))
            path = Path(root) / "fit_receipt.json"
            receipt = {key: spec[key] for key in ("model", "seed", "phase")}
            receipt.update(optimizer_steps=91, registration_file_sha256=REGISTRATION_SHA)
            pilot.json_new(path, receipt)
            self.assertEqual(pilot._known_steps(None, root, spec), 91)
            for field, changed in [("seed", 271901), ("registration_file_sha256", "e" * 64)]:
                bad = {**receipt, field: changed}
                path.write_text(json.dumps(bad))
                self.assertIsNone(pilot._known_steps(None, root, spec))

    def test_worker_steps_require_matching_seed_and_registration(self):
        spec = self._spec()
        worker = {key: spec[key] for key in ("model", "seed", "phase")}
        worker.update(optimizer_steps=17, registration_file_sha256=REGISTRATION_SHA)
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(pilot._known_steps(worker, root, spec), 17)
            for field, value in [("seed", 271901), ("registration_file_sha256", "e" * 64)]:
                with self.subTest(field=field):
                    self.assertIsNone(pilot._known_steps({**worker, field: value}, root, spec))

    def test_resource_totals_include_failed_steps_and_keep_unknown_attempt_count(self):
        records = {pilot.MODELS[0]: {
            "selection": [
                {"status": "failed", "optimizer_steps": 17, "elapsed_wall_seconds": 2},
                {"status": "failed", "optimizer_steps": None, "elapsed_wall_seconds": 3},
            ],
            "refit": [{"status": "completed", "optimizer_steps": 5,
                       "elapsed_wall_seconds": 1.5}],
        }}
        costs = pilot._costs(records)
        self.assertEqual(costs["fit_attempts"], 3)
        self.assertEqual(costs["known_optimizer_steps"], 22)
        self.assertEqual(costs["known_failed_optimizer_steps"], 17)
        self.assertEqual(costs["unknown_optimizer_step_attempts"], 1)
        self.assertEqual(costs["elapsed_worker_wall_seconds"], 6.5)

    def test_success_status_without_source_invariance_does_not_qualify_for_refits(self):
        def change_worker(worker):
            if (worker["model"] == pilot.MODELS[0] and worker["seed"] == pilot.SEEDS[0]
                    and worker["phase"] == "epoch_selection"):
                worker["source_hashes_unchanged"] = False
        with tempfile.TemporaryDirectory() as root:
            result, calls, _ = self._fake_campaign(root, mutate_worker=change_worker)
        self.assertEqual(len(calls), 25)
        self.assertIn(pilot.MODELS[0], result["blocked_refits"])
        self.assertFalse(result["paired_3x5_complete"])

    def test_campaign_refuses_existing_output_before_any_worker(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "campaign").mkdir()
            with self.assertRaises(FileExistsError):
                self._fake_campaign(root)

    def test_exclusive_metadata_never_replaces_an_existing_record(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "record.json"
            pilot.json_new(path, {"first": True})
            with self.assertRaises(FileExistsError):
                pilot.json_new(path, {"first": False})
            self.assertEqual(json.loads(path.read_text()), {"first": True})


if __name__ == "__main__":
    unittest.main()
