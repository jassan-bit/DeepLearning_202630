"""Read-only step1 runtime/budget boundaries, no market fitting/network."""
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_e2_reduced_environment as checker
spec = importlib.util.spec_from_file_location("reduced_colab_check_test", ROOT / "notebooks/comprobar_e2_reducido_colab.py")
colab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(colab)


def evidence():
    value = {"version": "e2-amended-pilot-budget-v1", "status": "reconciled_before_new_pilot_reservations",
        "authorized_combined_seconds": 230400., "authorized_combined_hours": 64.,
        "historical_charged_seconds": 222927.98951942776,
        "remaining_seconds_before_pilots": 7472.010480572237, "budget_reset": False,
        "historical_heads": {"campaigns/e2-cuda-selection-v1": "a" * 40,
                             "campaigns/e2-cuda-crop-11-v2": "b" * 40}}
    return {**value, "sha256": checker.digest(value)}


def event_sequence(events, anchor):
    rows, previous = [], "0" * 64
    for i, values in enumerate(events):
        row = {"version": "e2-amended-pilot-budget-v1", "sequence": i,
            "previous_sha256": previous, "reconciliation_sha256": anchor, **values}
        row["sha256"] = checker.digest(row)
        rows.append(row); previous = row["sha256"]
    return b"".join(checker.canonical(row) + b"\n" for row in rows)


class RuntimeTests(unittest.TestCase):
    def test_probe_and_versions_are_distinct_acceptance_conditions(self):
        good = {"python_major_minor": [3, 12], "torch": "2.8.0+cu126", "torch_cuda": "12.6",
                "cuda_available": True, "cuda_kernel_probe_passed": True}
        self.assertTrue(checker.policy_compatible(good))
        for change in ({"python_major_minor": [3, 13]}, {"torch": "2.8.0+cpu"},
                       {"torch_cuda": "12.8"}, {"cuda_available": False}, {"cuda_kernel_probe_passed": False}):
            self.assertFalse(checker.policy_compatible(good | change))

    def test_cpu_build_never_runs_a_tensor_or_training_fallback(self):
        torch = SimpleNamespace(__version__="2.8.0+cpu", version=SimpleNamespace(cuda=None),
                                cuda=SimpleNamespace(is_available=lambda: False))
        with tempfile.TemporaryDirectory() as folder, patch.object(checker, "physical_gpu_probe", return_value={"devices": [{"name": "T4"}]}), patch.object(checker, "cuda_probe") as kernel:
            result = checker.inspect_runtime(torch, proc_root=Path(folder))
        kernel.assert_not_called()
        self.assertFalse(result["production_policy_compatible"])
        self.assertFalse(result["cuda_available"])
        self.assertFalse(result["CPU_training_fallback_performed"])
        self.assertFalse(result["all_model_kernels_validated"])
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertEqual(result["physical_gpu_probe"]["devices"][0]["name"], "T4")

    def test_missing_torch_is_reported_without_installation(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(checker, "physical_gpu_probe", return_value={"devices": []}):
            result = checker.inspect_runtime(None, proc_root=Path(folder))
        self.assertEqual(result["status"], "runtime_check_failed")
        self.assertFalse(result["market_training_started"])
        self.assertFalse(result["cuda_kernel_probe_passed"])

    def test_controller_detection_uses_argv_not_embedded_shell_code(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for pid, data in (("12345", b"python\0/content/run_e2_optimized_selection_v2.py\0"),
                              ("23456", b"bash\0-lc\0python /content/run_e2_optimized_selection_v2.py --arg\0")):
                (root / pid).mkdir(); (root / pid / "cmdline").write_bytes(data)
            self.assertEqual(checker.controller_pids(root), [12345])

    def test_real_cuda_probe_only_when_a_cuda_device_is_present(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch is not installed in the stdlib checking environment")
        if not torch.cuda.is_available():
            self.skipTest("No actual CUDA device in this environment")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        result = checker.cuda_probe(torch)
        self.assertTrue(result["passed"])
        self.assertEqual(result["device"], "cuda:0")
        self.assertEqual(result["optimizer_steps"], 0)


class BudgetTests(unittest.TestCase):
    def setup_records(self, folder):
        value = evidence()
        path = Path(folder) / "budget.json"
        path.write_text(json.dumps(value))
        return path, value

    def test_snapshot_reads_preserve_charges_without_authorizing_training(self):
        with tempfile.TemporaryDirectory() as folder:
            path, value = self.setup_records(folder)
            before = {p.name: p.read_bytes() for p in Path(folder).iterdir()}
            result = checker.inspect_budget(path)
            self.assertEqual(result["status"], "budget_snapshot_verified")
            self.assertAlmostEqual(result["remaining_seconds"], 7472.010480572237)
            self.assertFalse(result["training_authorized"])
            self.assertFalse(result["fresh_remote_heads_checked_here"])
            self.assertEqual(before, {p.name: p.read_bytes() for p in Path(folder).iterdir()})

    def test_open_reservation_keeps_its_full_cost_and_reader_writes_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            path, value = self.setup_records(folder)
            ledger = Path(folder) / "pilot_budget.jsonl"
            raw = event_sequence([
                {"kind": "reserved", "attempt_id": "closed", "reservation_seconds": 500},
                {"kind": "charged", "attempt_id": "closed", "reservation_seconds": 500,
                 "elapsed_seconds": 10, "status": "completed"},
                {"kind": "overhead", "elapsed_seconds": 2},
                {"kind": "reserved", "attempt_id": "unknown", "reservation_seconds": 300},
                {"kind": "progress", "attempt_id": "unknown", "elapsed_seconds_lower_bound": 20,
                 "optimizer_steps_lower_bound": 7}], value["sha256"])
            ledger.write_bytes(raw)
            result = checker.inspect_budget(path, ledger=ledger)
            self.assertEqual(result["status"], "reconciliation_required")
            self.assertEqual(result["unclosed_reservations"], {"unknown": 300})
            self.assertAlmostEqual(result["remaining_seconds"], value["remaining_seconds_before_pilots"]-312)
            self.assertEqual(raw, ledger.read_bytes())
            self.assertFalse((Path(folder) / ".pilot-budget.lock").exists())

    def test_altered_chain_unknown_release_and_truncated_tail_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            path, value = self.setup_records(folder)
            ledger = Path(folder) / "ledger.jsonl"
            records = [{"kind": "reserved", "attempt_id": "x", "reservation_seconds": 300},
                {"kind": "charged", "attempt_id": "x", "reservation_seconds": 300,
                 "elapsed_seconds": 0, "status": "reconciliation_required"}]
            ledger.write_bytes(event_sequence(records, value["sha256"]))
            with self.assertRaises(ValueError):
                checker.inspect_budget(path, ledger=ledger)
            ledger.write_bytes(event_sequence(records[:1], value["sha256"])[:-1])
            with self.assertRaisesRegex(ValueError, "truncated"):
                checker.inspect_budget(path, ledger=ledger)
            ledger.write_bytes(event_sequence(records[:1], "c" * 64))
            with self.assertRaisesRegex(ValueError, "anchor"):
                checker.inspect_budget(path, ledger=ledger)

    def test_no_budget_evidence_never_becomes_zero_cost_or_ready(self):
        result = checker.inspect_budget()
        self.assertEqual(result["status"], "budget_evidence_pending")
        self.assertNotIn("remaining_seconds", result)
        self.assertFalse(result["training_authorized"])

    def test_rehashed_budget_extension_and_tampered_snapshot_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path, value = self.setup_records(folder)
            value["authorized_combined_hours"] = 72
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                checker.inspect_budget(path)
            value["sha256"] = checker.digest({k: v for k, v in value.items() if k != "sha256"})
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "64-hour"):
                checker.inspect_budget(path)


class ColabHelperTests(unittest.TestCase):
    def test_existing_isolated_python_is_preferred_without_bootstrap(self):
        with tempfile.TemporaryDirectory() as folder:
            python = Path(folder) / "python312"
            python.write_bytes(b"fixture"); python.chmod(0o700)
            chosen, isolated = colab.choose_python({"GPU_PYTHON": python})
            self.assertEqual(chosen, str(python))
            self.assertTrue(isolated)

    def test_bad_source_pin_never_opens_network(self):
        with patch.object(colab.urllib.request, "urlopen") as request:
            for source in ("main", "a" * 39, "__REDUCED_SOURCE_COMMIT__"):
                with self.assertRaises(ValueError):
                    colab.download_pinned(source, "scripts/check_e2_reduced_environment.py", "b" * 64)
            request.assert_not_called()

    def test_helper_runs_only_diagnostic_and_removes_tokens_from_child_environment(self):
        source, raw = "a" * 40, b"# synthetic checker fixture\n"
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "diagnostics"
            def run(args, **kwargs):
                self.assertNotIn("--execute", args)
                self.assertNotIn("GITHUB_TOKEN", kwargs["env"])
                self.assertNotIn("GH_TOKEN", kwargs["env"])
                report = Path(args[args.index("--output") + 1])
                report.write_text(json.dumps({"runtime": {"cuda_available": False}, "budget": {"status": "budget_evidence_pending"}}))
                return SimpleNamespace(returncode=3, stdout="", stderr="")
            with patch.object(colab, "download_pinned", return_value=raw), patch.object(colab, "choose_python", return_value=("/existing/python", True)), patch.object(colab.subprocess, "run", side_effect=run) as child, patch.dict(os.environ, {"GITHUB_TOKEN": "unit-test-secret", "GH_TOKEN": "second-secret"}), redirect_stdout(io.StringIO()):
                result = colab.run_colab_check(source, hashlib.sha256(raw).hexdigest(), output_directory=target)
            self.assertEqual(child.call_count, 1)
            self.assertFalse(result["training_started"])
            self.assertEqual(result["exit_code"], 3)


if __name__ == "__main__":
    unittest.main()
