"""GPU pilot admission, bounded-domain and public-evidence contract checks.

No market fit is executed by this suite. A real CUDA admission smoke check is
skipped explicitly on CPU runners; synthetic fixtures never claim GPU timing.
"""
from copy import deepcopy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

import torch
import run_e2_gpu_pilot as runner


def registration():
    protocol_path = ROOT/"book_e2/_static/search_protocol_20261109/protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    return {
        "version": "e2-gpu-pilot-registration-v1",
        "status": "registered_before_gpu_market_fits",
        "source_protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "profile": {
            "models": list(runner.NAS_NAMES+runner.FIXED_NAMES),
            "cohorts": ["small", "large"], "default_cohorts": ["small"],
            "epochs": 2, "seed": 274201, "optional_largest_timesnet": True,
        },
        "resources": {
            "parallel_fits": 1, "cpu_threads": 2, "interop_threads": 1,
            "host_rss_limit_bytes": 12*1024**3,
            "gpu_memory_fraction": .8, "gpu_memory_cap_bytes": 12*1024**3,
            "internal_wall_seconds": 300, "external_wall_seconds": 360,
            "campaign_wall_seconds": 900, "minimum_worker_start_seconds": 5,
        },
        "execution": {"device": "cuda", "dtype": "float32", "amp": False,
                      "tf32": False, "deterministic": True,
                      "cublas_workspace_config": ":4096:8", "cudnn_benchmark": False,
                      "cpu_fallback": False},
        "fixed_recipe": deepcopy(protocol["budget"]["noise_and_resource_pilot_fixed_nonarchitecture"]),
        "cohorts": {"small": {"train_n": 161, "validation_n": 83},
                    "large": {"train_n": 619, "validation_n": 83}},
    }


def write_registration(directory, value):
    path = Path(directory)/"registration.json"
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def walk(value):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def identity(row):
    return row.get("job", row)


def result_rows(value):
    return [node for node in walk(value) if isinstance(node, dict)
            and "optimizer_steps" in node and "status" in node
            and ("job_id" in node or isinstance(node.get("job"), dict))]


def public_fixture():
    jobs = runner.build_jobs(registration(), cohorts=["large"], include_largest=True)
    rows = []
    for i, job in enumerate(jobs):
        steps = 120 if job["model"] == "LSTM-autoencoder con predictor" else 40
        worker = {
            "status": "completed", "model": job["model"], "train_n": 619,
            "validation_n": 83, "parameter_count": 125, "state_bytes": 500,
            "phase_receipts": {"supervised": {"epochs_completed": 2,
                "optimizer_steps": steps, "best_epoch": 2, "fixed_epochs": 2,
                "elapsed_seconds": 1.0, "optimizer_initial_state_entries": 0}},
            "timing": {"build_seconds": .1, "fit_seconds": 1.0,
                       "validation_seconds": .2},
            "cuda_metrics": {"max_memory_allocated_bytes": 4000,
                             "max_memory_reserved_bytes": 5000,
                             "effective_limit_bytes": 12*1024**3},
        }
        rows.append({"job": deepcopy(job), "status": "success", "optimizer_steps": steps,
                     "optimizer_steps_censored": False, "elapsed_wall_seconds": 2.+i/10,
                     "sampled_peak_process_group_rss_bytes": 1000000,
                     "worker_profile": worker})
    largest = next(row for row in rows if row["job"].get("pattern")
                   and row["job"]["pattern"]["widths"] == [256, 256, 256])
    largest.update(status="worker_failure", optimizer_steps=38)
    largest["worker_profile"].update(status="failed", error_code="resource_limit")
    timeout = next(row for row in rows if row is not largest and row["job"]["model"] == "TiDE")
    timeout.update(status="timeout", optimizer_steps=None,
                   optimizer_steps_censored=True, worker_profile=None)
    unused = next(row for row in rows if row["job"]["model"] == "TimeMixer")
    unused.update(status="not_attempted_campaign_budget", optimizer_steps=None,
                  optimizer_steps_censored=False, elapsed_wall_seconds=0., worker_profile=None)
    known = sum(row["optimizer_steps"] for row in rows if type(row["optimizer_steps"]) is int)
    return {
        "version": runner.VERSION, "status": "completed_all_attempts",
        "registration_sha256": "a"*64, "registration_commit": "b"*40,
        "plan": {"jobs": jobs, "hardware": {"gpu_name": "Synthetic fixture only",
                  "total_memory_bytes": 16*1024**3}, "parallel_fits": 1},
        "results": rows, "known_optimizer_steps": known,
        "unknown_optimizer_step_workers": 1,
        "elapsed_workers_sum_seconds": sum(row["elapsed_wall_seconds"] for row in rows),
    }


class CUDAAdmissionTests(unittest.TestCase):
    def test_cpu_only_host_cannot_silently_enter_training(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "[Cc][Uu][Dd][Aa]|GPU"):
                runner.require_cuda_environment("cuda:0")

    def test_campaign_denies_missing_cuda_before_data_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"must-not-be-created"
            with mock.patch.object(runner, "require_cuda_environment",
                                   side_effect=RuntimeError("CUDA admission denied")) as gate, \
                    mock.patch.object(runner, "load_registration") as load, \
                    mock.patch.object(runner, "prepare_cohort") as prepare:
                with self.assertRaisesRegex(RuntimeError, "CUDA admission denied"):
                    runner.run_campaign(Path(directory)/"missing-data",
                        Path(directory)/"missing-registration.json", "a"*64, "b"*40, output)
            gate.assert_called_once()
            load.assert_not_called()
            prepare.assert_not_called()
            self.assertFalse(output.exists())

    @unittest.skipUnless(torch.cuda.is_available(), "No CUDA GPU available: real admission is not executed")
    def test_real_cuda_environment_reports_actual_available_device(self):
        result = runner.require_cuda_environment("cuda:0")
        self.assertIsInstance(result, dict)
        self.assertTrue(any(type(v) is int and v > 0 for k, v in result.items()
                            if "memory" in k or "bytes" in k))


class RegisteredDomainTests(unittest.TestCase):
    def test_campaign_budget_limits_next_worker_and_marks_unstarted_jobs(self):
        # The supervisor is a fixture, not a CUDA claim. Exercise the real
        # campaign journal/spec/output path without loading any market values.
        for remaining in (900., 100.):
            with self.subTest(first_remaining=remaining), tempfile.TemporaryDirectory() as directory:
                path, sha = write_registration(directory, registration())
                output = Path(directory)/"pilot"
                clock = iter((100., 100.+900.-remaining, 996., 996.))

                def supervise(command, worker_output, **kwargs):
                    spec = json.loads(Path(command[-1]).read_text())
                    return {"status": "success", "elapsed_wall_seconds": remaining-4.,
                            "sampled_peak_process_group_rss_bytes": 10000,
                            "worker_profile": {"status": "completed", "job": spec["job"],
                                               "optimizer_steps": 12}}

                with mock.patch.object(runner, "require_cuda_environment", return_value={"device": "cuda:0"}), \
                        mock.patch.object(runner, "_sources", return_value={"scripts/model.py": "a"*64}), \
                        mock.patch.object(runner, "time", SimpleNamespace(monotonic=lambda: next(clock))), \
                        mock.patch.object(runner, "run_isolated_profile", side_effect=supervise) as supervised, \
                        mock.patch.object(runner, "prepare_cohort") as prepare, \
                        redirect_stdout(io.StringIO()):
                    actual = runner.run_campaign(Path(directory), path, sha, "b"*40, output,
                                                 models=["TimesNet", "TiDE", "GRU32"])
                prepare.assert_not_called()
                supervised.assert_called_once()
                self.assertEqual(supervised.call_args.kwargs["timeout_seconds"], min(360., remaining))
                self.assertEqual([row["status"] for row in actual["results"]],
                                 ["success", "not_attempted_campaign_budget", "not_attempted_campaign_budget"])
                self.assertEqual(actual["known_optimizer_steps"], 12)
                self.assertEqual(actual["unknown_optimizer_step_workers"], 0)
                self.assertFalse(actual["all_fits_completed"])
                public = json.loads((output/"scientific_report.json").read_text())
                self.assertEqual(public["profile_attempts"], 1)
                self.assertEqual(public["costs"]["not_attempted_jobs"], 2)
                self.assertEqual(public["costs"]["failed_attempts"], 0)

    def test_default_and_maximum_domains_are_small_finite_pilots(self):
        default = runner.build_jobs(registration())
        self.assertEqual(len(default), 14)
        self.assertEqual({row["cohort"] for row in default}, {"small"})
        self.assertTrue(all(row["epochs"] == 2 and row["seed"] == 274201 for row in default))
        self.assertEqual(len({row["job_id"] for row in default}), 14)
        for row in default:
            if row["model"] in runner.NAS_NAMES:
                self.assertEqual(row["pattern"]["widths"], [64, 64, 64])
                self.assertEqual(row["pattern"]["dropout_pattern"], [0, 1, 0])
        largest = runner.build_jobs(registration(), cohorts=["small", "large"], include_largest=True)
        self.assertEqual(len(largest), 30)
        self.assertEqual(len({row["job_id"] for row in largest}), 30)
        extremes = [row for row in largest if row.get("pattern")
                    and row["pattern"]["widths"] == [256, 256, 256]]
        self.assertEqual(len(extremes), 2)
        self.assertTrue(all(row["model"] == "TimesNet" for row in extremes))

    def test_subsets_keep_registered_order_and_cannot_add_hidden_trials(self):
        selected = ["GRU32", "TimesNet", "TiDE"]
        jobs = runner.build_jobs(registration(), models=selected)
        expected = [name for name in runner.NAS_NAMES+runner.FIXED_NAMES if name in selected]
        self.assertEqual([row["model"] for row in jobs], expected)
        for options in ({"models": ["GRU32", "GRU32"]}, {"models": ["unknown"]},
                        {"models": []}, {"cohorts": ["final"]},
                        {"cohorts": ["small", "small"]}, {"cohorts": []},
                        {"models": ["GRU32"], "include_largest": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                runner.build_jobs(registration(), **options)

    def test_exact_registration_hash_and_commit_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path, sha = write_registration(directory, registration())
            self.assertEqual(runner.load_registration(path, sha, "b"*40)["profile"]["epochs"], 2)
            with self.assertRaises(ValueError):
                runner.load_registration(path, "0"*64, "b"*40)
            for commit in ("main", "b"*39, "X"*40, ""):
                with self.subTest(commit=commit), self.assertRaises(ValueError):
                    runner.load_registration(path, sha, commit)
            path.write_text(path.read_text()+"\n")
            with self.assertRaises(ValueError):
                runner.load_registration(path, sha, "b"*40)

    def test_registration_cannot_weaken_precision_resources_or_data_boundaries(self):
        cases = [(("status",), "draft"), (("source_protocol_sha256",), "0"*64),
                 (("execution", "amp"), True), (("execution", "tf32"), True),
                 (("execution", "cublas_workspace_config"), ":16:8"),
                 (("execution", "cudnn_benchmark"), True), (("execution", "cpu_fallback"), True),
                 (("execution", "deterministic"), False), (("resources", "parallel_fits"), 2),
                 (("resources", "campaign_wall_seconds"), 86400),
                 (("profile", "epochs"), 200), (("profile", "seed"), 273001),
                 (("cohorts", "large", "train_n"), 620),
                 (("fixed_recipe", "learning_rate"), .003)]
        with tempfile.TemporaryDirectory() as directory:
            for keys, value in cases:
                candidate = registration()
                target = candidate
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                path, sha = write_registration(directory, candidate)
                with self.subTest(keys=keys), self.assertRaises(ValueError):
                    runner.load_registration(path, sha, "b"*40)


class ScientificPublicReportTests(unittest.TestCase):
    def test_oom_counts_completed_updates_as_a_lower_bound_without_double_counting_sae(self):
        partial = runner._failure_steps({}, {"completed_optimizer_steps": 13,
            "partial_optimizer_update_may_have_occurred": True},
            fit_entered=True, fit_completed=False)
        self.assertIsNone(partial["optimizer_steps"])
        self.assertEqual(partial["optimizer_steps_known_lower_bound"], 13)
        self.assertTrue(partial["partial_optimizer_update_may_have_occurred"])
        phases = {"ae1": SimpleNamespace(optimizer_steps=24)}
        between_phases = runner._failure_steps(phases, {
            "completed_phases": {"ae1": {"optimizer_steps": 24}},
            "completed_optimizer_steps": 0, "known_optimizer_steps_all_sae_phases": 24,
            "partial_optimizer_update_may_have_occurred": False},
            fit_entered=True, fit_completed=False)
        self.assertEqual(between_phases["optimizer_steps"], 24)
        self.assertEqual(between_phases["optimizer_steps_known_lower_bound"], 24)
        within_next = runner._failure_steps(phases, {
            "completed_phases": {"ae1": {"optimizer_steps": 24}},
            "completed_optimizer_steps": 7,
            "partial_optimizer_update_may_have_occurred": True},
            fit_entered=True, fit_completed=False)
        self.assertIsNone(within_next["optimizer_steps"])
        self.assertEqual(within_next["optimizer_steps_known_lower_bound"], 31)

    def test_public_failure_retains_known_lower_bound_when_exact_updates_are_unknown(self):
        source = public_fixture()
        failed = next(row for row in source["results"] if row["optimizer_steps"] == 38)
        failed.update(optimizer_steps=None, optimizer_steps_known_lower_bound=38,
                      partial_optimizer_update_may_have_occurred=True)
        public = runner.public_scientific_report(source)
        actual = next(row for row in result_rows(public)
                      if identity(row)["job_id"] == failed["job"]["job_id"])
        self.assertIsNone(actual["optimizer_steps"])
        self.assertEqual(actual["optimizer_steps_known_lower_bound"], 38)
        self.assertEqual(public["costs"]["known_optimizer_steps"], source["known_optimizer_steps"])
        self.assertEqual(public["costs"]["unknown_optimizer_step_workers"], 2)

    def test_failures_unknown_steps_and_unstarted_jobs_remain_distinct(self):
        source = public_fixture()
        public = runner.public_scientific_report(source)
        rows = result_rows(public)
        self.assertEqual(len(rows), len(source["results"]))
        by_id = {identity(row)["job_id"]: row for row in rows}
        self.assertEqual(len(by_id), len(rows))
        for row in source["results"]:
            actual = by_id[row["job"]["job_id"]]
            self.assertEqual(actual["status"], row["status"])
            self.assertEqual(actual["optimizer_steps"], row["optimizer_steps"])
        costs = public["costs"]
        self.assertEqual(costs["known_optimizer_steps"], source["known_optimizer_steps"])
        self.assertEqual(costs["unknown_optimizer_step_workers"], 1)
        self.assertEqual(costs["not_attempted_jobs"], 1)
        self.assertEqual(costs["failed_attempts"], 2)
        self.assertEqual(costs["completed_attempts"], 12)
        self.assertIsNone(public["final_R"])
        self.assertIsNone(public["full_campaign_wall_hours"])
        self.assertFalse(public["scientific_HPO_or_NAS_executed"])

    def test_public_export_ignores_unknown_private_fields_recursively(self):
        source = public_fixture()
        expected = runner.public_scientific_report(source)
        private = deepcopy(source)
        for node in list(walk(private)):
            if isinstance(node, dict):
                node["__private_extra__"] = {"command": "/workspace/private/token",
                                            "env": {"API_KEY": "synthetic sentinel"}}
                node["__unlisted_private_number__"] = 991828381
        for row in private["results"]:
            row["worker_result_file"] = "/home/private/result.json"
            if row.get("worker_profile"):
                row["worker_profile"]["message"] = "/tmp/private stderr"
                row["worker_profile"]["source_sha256_before"] = {"/workspace/private.py": "a"*64}
                row["worker_profile"]["source_sha256_after"] = {"/workspace/private.py": "a"*64}
        actual = runner.public_scientific_report(private)
        self.assertEqual(actual, expected)
        text = json.dumps(actual, ensure_ascii=False, allow_nan=False)
        for forbidden in ("/workspace/", "/home/", "/tmp/", "API_KEY", "__private_extra__", "991828381"):
            self.assertNotIn(forbidden, text)

    def test_allowed_identity_field_cannot_smuggle_a_private_path(self):
        source = public_fixture()
        source["results"][0]["job"]["model"] = "/workspace/private/model"
        with self.assertRaises(ValueError):
            runner.public_scientific_report(source)

    def test_cuda_platform_is_public_without_internal_hardware_metadata(self):
        source = public_fixture()
        source["plan"] = {"hardware": {"device": "cuda:0", "device_name": "NVIDIA T4",
            "torch": "2.8.0+cu126", "torch_cuda": "12.6", "total_memory_bytes": 16*1024**3,
            "effective_memory_limit_bytes": 12*1024**3, "cudnn": 91002,
            "hostname": "/workspace/private", "commands": ["PRIVATE_SENTINEL"],
            "environment": {"API_KEY": "PRIVATE_SENTINEL"}}}
        hardware = runner.public_scientific_report(source)["hardware"]
        self.assertEqual(hardware["device_name"], "NVIDIA T4")
        self.assertEqual(hardware["torch_cuda"], "12.6")
        self.assertFalse(hardware["amp"])
        self.assertNotIn("hostname", hardware)
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(hardware))
        source["plan"]["hardware"]["device_name"] = "/workspace/private"
        with self.assertRaises(ValueError):
            runner.public_scientific_report(source)

    def test_registration_identifiers_are_public_but_cannot_contain_private_text(self):
        source = public_fixture()
        report = runner.public_scientific_report(source)
        self.assertEqual(report["registration_commit"], "b"*40)
        self.assertEqual(report["registration_sha256"], "a"*64)
        for name in ("registration_commit", "registration_sha256"):
            invalid = deepcopy(source)
            invalid[name] = "/workspace/private"
            with self.assertRaises(ValueError):
                runner.public_scientific_report(invalid)


if __name__ == "__main__":
    unittest.main()
