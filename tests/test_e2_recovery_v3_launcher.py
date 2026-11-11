"""In-place Colab launch admission and interruption tests; no GPU or network."""
import contextlib
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = Path(os.environ.get("E2_ENGINE_ROOT", ROOT)).resolve()
SPEC = importlib.util.spec_from_file_location("recovery_v3_launcher_tested", ROOT / "scripts/colab_recovery_v3_launcher.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class FakeProcess:
    next_pid = 4100

    def __init__(self, command, *, returncode=0, waits=None, payload=None, **kwargs):
        FakeProcess.next_pid += 1
        self.pid = FakeProcess.next_pid
        self.command = list(command)
        self.kwargs = dict(kwargs, env=dict(kwargs["env"]))
        self.returncode = None
        self.final_returncode = returncode
        self.waits = list(waits or [])
        self.signals = []
        self.wait_calls = 0
        self.payload = payload

    def poll(self):
        return self.returncode

    def send_signal(self, value):
        self.signals.append(value)

    def wait(self):
        self.wait_calls += 1
        if self.waits:
            value = self.waits.pop(0)
            if isinstance(value, BaseException):
                raise value
        self.returncode = self.final_returncode
        if self.payload is not None:
            report = Path(self.command[self.command.index("--external-report") + 1])
            with report.open("x") as stream:
                json.dump(self.payload, stream)
        return self.returncode


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="e2-launcher-v3-test-")
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name) / "workspace"
        self.campaign = self.workspace / "search-campaigns" / launcher._CAMPAIGN
        self.campaign.mkdir(parents=True)
        (self.campaign / "campaign.json").write_text('{"sealed":true}')
        (self.campaign / ".campaign.lock").write_bytes(b"existing-ledger-lock")
        self.engine = self.workspace / "search-source"
        self.engine.mkdir()
        self.python = self.workspace / "venv-py312-cu126/bin/python"
        self.python.parent.mkdir(parents=True)
        self.python.write_text("prepared-python-placeholder")
        self.data = self.workspace / "development_cache"
        self.data.mkdir()
        self.controller = self.workspace / ("recovery-v3-source-" + "a" * 12)
        self.controller.mkdir()
        self.auth = self.controller / launcher._AUTHORIZATION_PATH
        self.auth.parent.mkdir(parents=True)
        self.auth.write_text("{}")
        self.ns = dict(RUN_SEARCH=True, RETRY_LOST_ONCE=True, RECOVERY_COMMIT="a" * 40,
                       EXPECTED_RECOVERY_AUTHORIZATION_SHA256="b" * 64, WORKSPACE=self.workspace)
        self.calls = []
        self.processes = []
        self.output = io.StringIO()

    def prepared(self, *, waits=None, payload=None, returncode=0):
        stack = contextlib.ExitStack()
        stack.enter_context(contextlib.redirect_stdout(self.output))
        stack.enter_context(patch.object(launcher, "_verify_engine"))
        stack.enter_context(patch.object(launcher, "_verify_hardware"))
        stack.enter_context(patch.object(launcher, "_checkout_controller", return_value=self.auth))
        stack.enter_context(patch.object(launcher, "_verify_cached_inputs"))
        stack.enter_context(patch.object(launcher, "_github_environment", side_effect=lambda clean: dict(clean, GITHUB_TOKEN="SECRET-MUST-NOT-PRINT")))
        def run(args, environment, **kwargs):
            self.calls.append((list(map(str, args)), dict(environment), kwargs))
            return SimpleNamespace(stdout=json.dumps({"status": "inspection_only", "previous_completed_jobs": 501}))
        stack.enter_context(patch.object(launcher, "_run", side_effect=run))
        def spawn(command, **kwargs):
            process = FakeProcess(command, waits=waits, payload=payload, returncode=returncode, **kwargs)
            self.processes.append(process)
            return process
        stack.enter_context(patch.object(launcher.subprocess, "Popen", side_effect=spawn))
        return stack

    def test_false_flags_are_inert_preview_even_without_pins_or_paths(self):
        for flags in ({}, {"RUN_SEARCH": True}, {"RETRY_LOST_ONCE": True}):
            with self.subTest(flags=flags), contextlib.ExitStack() as stack:
                for name in ("_no_symlinks", "_clean_environment", "_github_environment", "_launcher_lock"):
                    stack.enter_context(patch.object(launcher, name, side_effect=AssertionError("No side effects during preview")))
                stack.enter_context(patch.object(launcher.subprocess, "Popen", side_effect=AssertionError("No process")))
                stack.enter_context(contextlib.redirect_stdout(self.output))
                ns = dict(flags)
                result = launcher.run_launcher(ns)
                self.assertEqual(result["status"], "preview_only")
                self.assertEqual(ns, flags)
                self.assertFalse(result["network_used"])
                self.assertFalse(result["credentials_read"])
                self.assertFalse(result["recovery_claimed"])
                self.assertEqual(result["initial_campaign_hours"], 24)
                self.assertEqual(result["registered_effective_campaign_hours"], 36)
                self.assertFalse(result["budget_extension"])
                self.assertIn("restauración remota", result["missing_campaign_cache"])

    def test_execution_requires_literal_boolean_flags_and_full_pins(self):
        for update in ({"RUN_SEARCH": "True"}, {"RETRY_LOST_ONCE": 1}, {"RECOVERY_COMMIT": "main"},
                       {"RECOVERY_COMMIT": launcher._A3_SOURCE}, {"EXPECTED_RECOVERY_AUTHORIZATION_SHA256": "x"}):
            with self.subTest(update=update), patch.object(launcher, "_no_symlinks", side_effect=AssertionError("No disk admission")):
                with self.assertRaises(launcher.LauncherBlocked):
                    launcher.run_launcher(dict(self.ns, **update))

    def test_active_process_blocks_without_overwriting_success_or_using_disk(self):
        previous = {"status": "selection_paused", "external_report": "/old/unique-success.json"}
        process = SimpleNamespace(poll=lambda: None)
        self.ns.update(E2_RECOVERY_V3_ACTIVE_PROCESS=process, E2_RECOVERY_V3_LAST_PID=999,
                       E2_RECOVERY_V3_RUN_HISTORY=[previous.copy()])
        with patch.object(launcher, "_no_symlinks", side_effect=AssertionError("No disk")):
            with self.assertRaisesRegex(launcher.LauncherBlocked, "runner_already_active"):
                launcher.run_launcher(self.ns)
        self.assertEqual(self.ns["E2_RECOVERY_V3_RUN_HISTORY"], [previous])
        self.assertEqual(self.ns["E2_RECOVERY_V3_LAST_PID"], 999)

    def test_readonly_ledger_lock_blocks_before_source_gpu_auth_or_any_write(self):
        lock = self.campaign / ".campaign.lock"
        before = {p.relative_to(self.workspace).as_posix(): p.read_bytes() for p in self.workspace.rglob("*") if p.is_file()}
        with lock.open("rb") as held:
            fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(launcher, "_launcher_lock", side_effect=AssertionError("No write")), \
                 patch.object(launcher, "_verify_engine", side_effect=AssertionError("No source")), \
                 patch.object(launcher, "_verify_hardware", side_effect=AssertionError("No GPU")), \
                 patch.object(launcher, "_github_environment", side_effect=AssertionError("No auth")):
                with self.assertRaisesRegex(launcher.LauncherBlocked, "runner_already_active"):
                    launcher.run_launcher(self.ns)
        after = {p.relative_to(self.workspace).as_posix(): p.read_bytes() for p in self.workspace.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_missing_preparation_fails_without_installing_or_fetching(self):
        self.python.unlink()
        with patch.object(launcher, "_run", side_effect=AssertionError("No install or fetch")):
            with self.assertRaisesRegex(launcher.LauncherBlocked, "preparación anterior"):
                launcher.run_launcher(self.ns)
        self.assertFalse((self.controller / "scripts").exists())

    def test_missing_workspace_blocks_before_lock_fetch_gpu_or_credentials(self):
        missing = Path(self.temp.name) / "not-prepared"
        with patch.object(launcher, "_launcher_lock", side_effect=AssertionError("No locks")), \
             patch.object(launcher, "_run", side_effect=AssertionError("No fetch or setup")), \
             patch.object(launcher, "_github_environment", side_effect=AssertionError("No credentials")):
            with self.assertRaisesRegex(launcher.LauncherBlocked, "secciones 1–7"):
                launcher.run_launcher(dict(self.ns, WORKSPACE=missing))
        self.assertFalse(missing.exists())

    def test_missing_campaign_cache_is_admitted_without_initializing_campaign(self):
        shutil.rmtree(self.campaign)
        with self.prepared(payload={"status": "blocked_recovery", "error_code": "remote_validation_failed",
                                    "error_phase": "campaign_restore", "logical_completed_jobs": None,
                                    "pending_job_ids": None}, returncode=3):
            result = launcher.run_launcher(self.ns)
        self.assertFalse(self.campaign.exists(), "Only the fixed controller may restore the campaign")
        self.assertFalse(result["campaign_cache_present_at_admission"])
        self.assertEqual(result["status"], "blocked_recovery")
        self.assertIsNone(result["logical_completed_jobs"])
        self.assertEqual(result["error_phase"], "campaign_restore")
        command = self.processes[0].command
        self.assertEqual(command[command.index("--output") + 1], str(self.campaign))
        self.assertEqual(command[command.index("--campaign-hours") + 1], "24")
        self.assertNotIn("--extend-campaign-hours", command)
        self.assertEqual(result["registered_effective_campaign_hours"], 36)
        self.assertFalse(result["additional_budget_extension_requested"])

    def test_missing_campaign_parent_is_left_for_the_fixed_restore_controller(self):
        shutil.rmtree(self.campaign)
        self.campaign.parent.rmdir()
        with self.prepared():
            result = launcher.run_launcher(self.ns)
        self.assertEqual(len(self.processes), 1)
        self.assertFalse(result["campaign_cache_present_at_admission"])
        self.assertFalse(self.campaign.parent.exists())
        self.assertEqual(result["status"], "unconfirmed")
        self.assertEqual(result["error_code"], "external_report_missing")
        self.assertIsNone(result["logical_completed_jobs"])

    def test_existing_unrelated_campaign_directory_is_preserved_and_blocks(self):
        (self.campaign / "campaign.json").unlink()
        sentinel = self.campaign / "unrelated.json"
        sentinel.write_text('{"preserve": true}')
        before = {p.name: p.read_bytes() for p in self.campaign.iterdir() if p.is_file()}
        with patch.object(launcher, "_launcher_lock", side_effect=AssertionError("No writes")), \
             patch.object(launcher, "_github_environment", side_effect=AssertionError("No credentials")):
            with self.assertRaisesRegex(launcher.LauncherBlocked, "caché existente"):
                launcher.run_launcher(self.ns)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.campaign.iterdir() if p.is_file()})

    def test_existing_campaign_file_is_not_replaced_by_directory(self):
        shutil.rmtree(self.campaign)
        self.campaign.write_bytes(b"unrelated existing file")
        with patch.object(launcher, "_launcher_lock", side_effect=AssertionError("No writes")):
            with self.assertRaisesRegex(launcher.LauncherBlocked, "caché existente"):
                launcher.run_launcher(self.ns)
        self.assertEqual(self.campaign.read_bytes(), b"unrelated existing file")

    def test_symlink_campaign_is_not_followed_or_restored(self):
        shutil.rmtree(self.campaign)
        outside = Path(self.temp.name) / "other-campaign"
        outside.mkdir()
        self.campaign.symlink_to(outside, target_is_directory=True)
        with patch.object(launcher, "_launcher_lock", side_effect=AssertionError("No writes")):
            with self.assertRaisesRegex(launcher.LauncherBlocked, "enlace simbólico"):
                launcher.run_launcher(self.ns)
        self.assertEqual(list(outside.iterdir()), [])

    def test_previous_version_active_or_unconfirmed_launch_also_blocks(self):
        for version in ("V1", "V2"):
            for state in ("active", "unconfirmed"):
                with self.subTest(version=version, state=state):
                    prefix = "E2_RECOVERY_" + version
                    values = {prefix + "_RUN_HISTORY": [{"status": "selection_paused"}]}
                    if state == "active":
                        values[prefix + "_ACTIVE_PROCESS"] = SimpleNamespace(poll=lambda: None)
                        expected = "runner_already_active"
                    else:
                        values[prefix + "_LAUNCH_UNCONFIRMED"] = True
                        expected = "launch_unconfirmed"
                    ns = dict(self.ns, **values)
                    with patch.object(launcher, "_no_symlinks", side_effect=AssertionError("No disk admission")):
                        with self.assertRaisesRegex(launcher.LauncherBlocked, expected):
                            launcher.run_launcher(ns)
                    self.assertEqual(ns[prefix + "_RUN_HISTORY"], [{"status": "selection_paused"}])

    def test_prepared_execution_uses_scoped_token_unique_report_and_new_session(self):
        payload = {"status": "selection_paused", "error_code": None, "error_phase": None,
                   "logical_completed_jobs": 502, "pending_job_ids": []}
        with self.prepared(payload=payload):
            result = launcher.run_launcher(self.ns)
        process = self.processes[0]
        self.assertTrue(process.kwargs["start_new_session"])
        self.assertEqual(process.kwargs["cwd"], self.controller)
        self.assertEqual(process.kwargs["env"]["GITHUB_TOKEN"], "SECRET-MUST-NOT-PRINT")
        self.assertNotIn("SECRET-MUST-NOT-PRINT", " ".join(process.command))
        self.assertNotIn("SECRET-MUST-NOT-PRINT", self.output.getvalue())
        self.assertEqual(result["logical_completed_jobs"], 502)
        self.assertEqual(result["pending_job_ids"], [])
        report = Path(result["external_report"])
        self.assertEqual(report.parent, self.campaign.parent)
        self.assertNotIn(report, list(self.campaign.rglob("*")))
        self.assertIn("--retry-lost-once", process.command)
        self.assertNotIn("--extend-campaign-hours", process.command)
        for command, environment, _ in self.calls:
            self.assertNotIn("GITHUB_TOKEN", environment)
            self.assertNotIn("--execute", command)
            if "unittest" in command:
                self.assertEqual(environment["PYTHONPATH"], os.pathsep.join([str(self.controller / "scripts"), str(self.engine / "scripts")]))
                self.assertEqual(environment["E2_ENGINE_ROOT"], str(self.engine))
                self.assertEqual(self.calls[[item[0] for item in self.calls].index(command)][2]["cwd"], self.controller)
        self.assertEqual(self.ns["E2_RECOVERY_V3_LAST_PID"], process.pid)
        self.assertEqual(self.ns["E2_RECOVERY_V3_RUN_HISTORY"][0]["external_report"], str(report))
        self.assertTrue(result["campaign_cache_present_at_admission"])
        self.assertEqual(process.command[process.command.index("--campaign-hours") + 1], "24")
        # The launcher's original ledger lock is released for the controller.
        with (self.campaign / ".campaign.lock").open("rb") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_repeated_interrupts_signal_only_controller_once_and_wait_for_exit(self):
        with self.prepared(waits=[KeyboardInterrupt(), KeyboardInterrupt()], payload={"status": "blocked_recovery", "pending_job_ids": ["unknown"], "logical_completed_jobs": 501}, returncode=130):
            result = launcher.run_launcher(self.ns)
        process = self.processes[0]
        self.assertEqual(process.signals, [signal.SIGINT])
        self.assertEqual(process.wait_calls, 3)
        self.assertTrue(result["interrupted"])
        self.assertEqual(result["returncode"], 130)
        self.assertEqual(result["pending_job_ids"], ["unknown"])
        self.assertEqual(self.ns["E2_RECOVERY_V3_ACTIVE_PROCESS"].poll(), 130)

    def test_interrupted_notebook_display_cannot_abandon_shutdown_wait(self):
        process = FakeProcess([], env={}, waits=[KeyboardInterrupt(), KeyboardInterrupt()])
        with patch("builtins.print", side_effect=KeyboardInterrupt()):
            returncode, interrupted = launcher._wait_gracefully(process)
        self.assertEqual(process.signals, [signal.SIGINT])
        self.assertEqual(process.wait_calls, 3)
        self.assertEqual(returncode, 0)
        self.assertTrue(interrupted)

    def test_interrupt_after_popen_before_wait_still_signals_and_waits(self):
        original_print = print
        injected = False
        def interrupt_once(*args, **kwargs):
            nonlocal injected
            if args and args[0] == "Controlador PID:" and not injected:
                injected = True
                raise KeyboardInterrupt()
            return original_print(*args, **kwargs)
        with self.prepared(payload={"status": "blocked_recovery"}), patch("builtins.print", side_effect=interrupt_once):
            result = launcher.run_launcher(self.ns)
        self.assertEqual(self.processes[0].signals, [signal.SIGINT])
        self.assertEqual(self.processes[0].wait_calls, 1)
        self.assertTrue(result["interrupted"])

    def test_unknown_popen_return_blocks_replay_and_keeps_intended_report_path(self):
        with self.prepared(), patch.object(launcher.subprocess, "Popen", side_effect=KeyboardInterrupt()):
            with self.assertRaisesRegex(launcher.LauncherBlocked, "launch_unconfirmed"):
                launcher.run_launcher(self.ns)
        pending = self.ns["E2_RECOVERY_V3_LAUNCH_PENDING"]
        self.assertTrue(self.ns["E2_RECOVERY_V3_LAUNCH_UNCONFIRMED"])
        self.assertIsNone(pending["pid"])
        self.assertEqual(Path(pending["external_report"]).parent, self.campaign.parent)
        with patch.object(launcher, "_no_symlinks", side_effect=AssertionError("No new attempt")):
            with self.assertRaisesRegex(launcher.LauncherBlocked, "launch_unconfirmed"):
                launcher.run_launcher(self.ns)

    def test_unexpected_output_error_waits_for_controller_before_unlocking(self):
        original_print = print
        def failed_output(*args, **kwargs):
            if args and args[0] == "Controlador PID:":
                raise RuntimeError("output transport failed")
            return original_print(*args, **kwargs)
        with self.prepared(payload={"status": "blocked_recovery"}), patch("builtins.print", side_effect=failed_output):
            with self.assertRaisesRegex(RuntimeError, "output transport failed"):
                launcher.run_launcher(self.ns)
        self.assertEqual(self.processes[0].signals, [signal.SIGINT])
        self.assertEqual(self.processes[0].wait_calls, 1)
        self.assertIsNotNone(self.processes[0].poll())

    def test_real_stdlib_controller_receives_sigint_and_closes_receipt(self):
        ready = Path(self.temp.name) / "controller-ready"
        receipt = Path(self.temp.name) / "controller-closed.json"
        source = """import json,signal,sys,time
from pathlib import Path
ready,receipt=map(Path,sys.argv[1:])
def close(signum,frame):
    receipt.write_text(json.dumps({'signal':signum,'closed':True}))
    raise SystemExit(0)
signal.signal(signal.SIGINT,close)
ready.write_text('ready')
deadline=time.monotonic()+5
while time.monotonic()<deadline: time.sleep(.01)
"""
        process = subprocess.Popen([sys.executable, "-S", "-c", source, str(ready), str(receipt)], start_new_session=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        class InterruptWait:
            def __init__(self): self.first = True
            def wait(self):
                if self.first:
                    self.first = False
                    raise KeyboardInterrupt()
                return process.wait(timeout=7)
            def send_signal(self, value): process.send_signal(value)
        try:
            deadline = time.monotonic() + 2
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertTrue(ready.exists(), "stdlib controller failed to become ready")
            with contextlib.redirect_stdout(self.output):
                returncode, interrupted = launcher._wait_gracefully(InterruptWait())
            self.assertTrue(interrupted)
            self.assertEqual(returncode, 0)
            self.assertEqual(json.loads(receipt.read_text()), {"signal": signal.SIGINT, "closed": True})
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                process.wait(timeout=7)
    def test_second_invocation_preserves_first_success_and_uses_different_report(self):
        with self.prepared(payload={"status": "selection_paused", "logical_completed_jobs": 502}):
            first = launcher.run_launcher(self.ns)
            first_bytes = Path(first["external_report"]).read_bytes()
            second = launcher.run_launcher(self.ns)
        self.assertNotEqual(first["external_report"], second["external_report"])
        self.assertEqual(Path(first["external_report"]).read_bytes(), first_bytes)
        self.assertEqual(len(self.ns["E2_RECOVERY_V3_RUN_HISTORY"]), 2)

    def test_corrected_pin_preserves_old_controller_caches_and_blocked_report(self):
        legacy = self.workspace / "recovery-v3-source"
        legacy.mkdir()
        (legacy / "controller.py").write_bytes(b"# earlier unversioned R3 cache\n")
        previous_pin = self.workspace / ("recovery-v3-source-" + "c" * 12)
        previous_pin.mkdir()
        (previous_pin / "authorization.json").write_bytes(b'{"previous": "sealed"}\n')
        old_report = self.campaign.parent / "recovery-v3-prior-blocked.json"
        old_report.write_bytes(b'{"status":"blocked_recovery","error_code":"private_or_unapproved_text"}\n')
        old_record = {"external_report": str(old_report), "status": "blocked_recovery", "returncode": 2,
                      "controller_source_commit": "c" * 40, "pid": 2333}
        self.ns["E2_RECOVERY_V3_RUN_HISTORY"] = [dict(old_record)]
        self.ns["E2_RECOVERY_V3_ACTIVE_PROCESS"] = SimpleNamespace(poll=lambda: 2)
        self.ns["E2_RECOVERY_V3_LAST_PID"] = 2333
        original = {p: p.read_bytes() for root in (legacy, previous_pin) for p in root.rglob("*") if p.is_file()}
        original[old_report] = old_report.read_bytes()
        with self.prepared(payload={"status": "selection_paused", "logical_completed_jobs": 1552}), \
             patch.object(launcher, "_checkout_controller", return_value=self.auth) as checkout:
            result = launcher.run_launcher(self.ns)
        expected_controller = self.workspace / ("recovery-v3-source-" + self.ns["RECOVERY_COMMIT"][:12])
        self.assertEqual(expected_controller, self.controller)
        self.assertEqual(checkout.call_count, 2)
        self.assertTrue(all(call.args[0] == expected_controller for call in checkout.call_args_list))
        self.assertEqual(self.processes[0].kwargs["cwd"], expected_controller)
        self.assertEqual(Path(self.processes[0].command[1]).parent.parent, expected_controller)
        for p, payload in original.items():
            self.assertEqual(p.read_bytes(), payload, str(p))
        self.assertEqual(self.ns["E2_RECOVERY_V3_RUN_HISTORY"][0], old_record)
        self.assertEqual(len(self.ns["E2_RECOVERY_V3_RUN_HISTORY"]), 2)
        self.assertNotEqual(result["external_report"], str(old_report))
        self.assertEqual(result["logical_completed_jobs"], 1552)

    def test_missing_report_is_unknown_not_a_zero_result(self):
        with self.prepared():
            result = launcher.run_launcher(self.ns)
        self.assertEqual(result["status"], "unconfirmed")
        self.assertEqual(result["error_code"], "external_report_missing")
        self.assertIsNone(result["logical_completed_jobs"])
        self.assertIsNone(result["pending_job_ids"])

    def test_failed_new_helper_tests_prevent_data_auth_and_fit(self):
        with self.prepared(), patch.object(launcher, "_verify_cached_inputs", side_effect=AssertionError("No data")), \
             patch.object(launcher, "_github_environment", side_effect=AssertionError("No credentials")):
            saved_run = launcher._run
            def fail_tests(args, environment, **kwargs):
                if "unittest" in list(map(str, args)):
                    raise launcher.LauncherBlocked("synthetic failure")
                return saved_run(args, environment, **kwargs)
            with patch.object(launcher, "_run", side_effect=fail_tests):
                with self.assertRaisesRegex(launcher.LauncherBlocked, "synthetic failure"):
                    launcher.run_launcher(self.ns)
        self.assertEqual(self.processes, [])

    def test_non_t4_blocks_before_python_gpu_check(self):
        calls = []
        def driver(args, environment, **kwargs):
            calls.append(args)
            return SimpleNamespace(stdout="NVIDIA L4\n")
        with patch.object(launcher, "_run", side_effect=driver):
            with self.assertRaisesRegex(launcher.LauncherBlocked, "Tesla T4"):
                launcher._verify_hardware(self.python, self.engine, {})
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "nvidia-smi")

    def test_existing_controller_with_wrong_head_is_never_rewritten(self):
        sentinel = self.controller / "keep-me.txt"
        sentinel.write_text("existing source")
        with patch.object(launcher, "_run", return_value=SimpleNamespace(stdout="c" * 40 + "\n")) as run:
            with self.assertRaisesRegex(launcher.LauncherBlocked, "otro commit"):
                launcher._checkout_controller(self.controller, "a" * 40, "b" * 64, {})
        self.assertEqual(sentinel.read_text(), "existing source")
        self.assertEqual(run.call_count, 1)

    def test_sparse_checkout_admits_exactly_three_helpers_three_tests_and_auth(self):
        self.assertEqual(len(launcher._CONTROLLER_PATHS), 7)
        self.assertTrue(all(path.endswith("_v3.py") for path in launcher._CONTROLLER_PATHS[:6]))
        self.assertEqual(launcher._CONTROLLER_PATHS[-1], "experiments/e2_recovery_20261111_r3/authorization.json")
        self.assertNotIn("colab_recovery_v3_launcher.py", launcher._CONTROLLER_PATHS)

    def test_environment_preserves_proxy_ca_but_removes_worker_credentials(self):
        with patch.dict(os.environ, {"GITHUB_TOKEN": "SECRET", "GH_TOKEN": "OTHER", "PYTHONPATH": "/unsafe",
                                     "HTTPS_PROXY": "https://proxy.example", "SSL_CERT_FILE": "/ca.pem"}):
            clean = launcher._clean_environment()
        self.assertNotIn("GITHUB_TOKEN", clean)
        self.assertNotIn("GH_TOKEN", clean)
        self.assertNotIn("PYTHONPATH", clean)
        self.assertEqual(clean["HTTPS_PROXY"], "https://proxy.example")
        self.assertEqual(clean["SSL_CERT_FILE"], "/ca.pem")

    def test_github_auth_failure_never_leaks_response_or_secret(self):
        import urllib.error
        import urllib.request
        secret = "SECRET-MUST-NOT-PRINT"
        opener = SimpleNamespace(open=lambda *args, **kwargs: (_ for _ in ()).throw(
            urllib.error.HTTPError("https://api.github.com/repos/" + launcher._REPOSITORY, 403, secret, {}, io.BytesIO(secret.encode()))))
        with patch.dict(os.environ, {"GITHUB_TOKEN": secret}), patch.object(urllib.request, "build_opener", return_value=opener):
            with self.assertRaises(launcher.LauncherBlocked) as error:
                launcher._github_environment({})
        self.assertIn("HTTP 403", str(error.exception))
        self.assertNotIn(secret, str(error.exception))

    def test_github_token_enters_only_returned_controller_environment(self):
        import urllib.request
        secret = "SECRET-MUST-NOT-PRINT"
        requests = []
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, maximum): return json.dumps({"full_name": launcher._REPOSITORY}).encode()
        def open_request(request, timeout):
            requests.append(request)
            return Response()
        with patch.dict(os.environ, {"GITHUB_TOKEN": secret}), patch.object(urllib.request, "build_opener", return_value=SimpleNamespace(open=open_request)):
            clean = {"safe": "value"}
            scoped = launcher._github_environment(clean)
        self.assertEqual(scoped["GITHUB_TOKEN"], secret)
        self.assertEqual(clean, {"safe": "value"})
        self.assertEqual(requests[0].full_url, "https://api.github.com/repos/" + launcher._REPOSITORY)
        self.assertEqual(requests[0].get_header("Authorization"), "Bearer " + secret)

    def test_actual_engine_hash_guard_rejects_mutation_without_importing_engine(self):
        hashes = {}
        for index in range(61):
            relative = "scripts/source_" + str(index) + ".py"
            path = self.engine / relative
            path.parent.mkdir(exist_ok=True)
            path.write_text("# sealed " + str(index))
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        registration = self.engine / "experiments/e2_cuda_search_20261109/registration.json"
        registration.parent.mkdir(parents=True)
        registration.write_text(json.dumps({"implementation_sha256": hashes}))
        protocol = self.engine / "book_e2/_static/search_protocol_20261109/protocol.json"
        protocol.parent.mkdir(parents=True)
        protocol.write_text("{}")
        def git(args, environment, **kwargs):
            return SimpleNamespace(stdout=launcher._A3_SOURCE + "\n" if "rev-parse" in args else "")
        with patch.object(launcher, "_A3_REGISTRATION", launcher._sha256(registration)), \
             patch.object(launcher, "_A3_PROTOCOL", launcher._sha256(protocol)), patch.object(launcher, "_run", side_effect=git):
            launcher._verify_engine(self.engine, {})
            (self.engine / "scripts/source_31.py").write_text("# modified")
            with self.assertRaisesRegex(launcher.LauncherBlocked, "implementación científica"):
                launcher._verify_engine(self.engine, {})

    def test_prepared_paths_and_hashes_match_actual_frozen_a3_and_setup_notebook(self):
        registration = ENGINE_ROOT / "experiments/e2_cuda_search_20261109/registration.json"
        self.assertEqual(launcher._sha256(registration), launcher._A3_REGISTRATION)
        registered = json.loads(registration.read_text())
        self.assertEqual(len(registered["implementation_sha256"]), 61)
        for relative, expected in registered["implementation_sha256"].items():
            self.assertEqual(launcher._sha256(ENGINE_ROOT / relative), expected, relative)
        for relative, expected in launcher._DATA_HASHES.items():
            self.assertEqual(registered["development_sources"][relative], expected)
        notebook = json.loads((ENGINE_ROOT / "notebooks/entregable2_busqueda_gpu.ipynb").read_text())
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        for path in ('WORKSPACE/"search-source"', 'WORKSPACE/"venv-py312-cu126"', 'WORKSPACE/"development_cache"',
                     'REPO/"work/author_sources/iTransformer"'):
            self.assertIn(path, source)

    def test_lost_controller_report_diagnostic_does_not_replace_prior_success(self):
        old = self.campaign.parent / "prior-success.json"
        old.write_text('{"logical_completed_jobs":502}')
        self.ns["E2_RECOVERY_V3_RUN_HISTORY"] = [{"external_report": str(old), "status": "selection_paused"}]
        with self.prepared(payload={"status": "blocked_recovery", "error_code": "runner_already_active", "error_phase": "local_lock_check", "logical_completed_jobs": 502, "pending_job_ids": []}, returncode=3):
            result = launcher.run_launcher(self.ns)
        self.assertEqual(result["error_code"], "runner_already_active")
        self.assertEqual(old.read_text(), '{"logical_completed_jobs":502}')
        self.assertEqual(self.ns["E2_RECOVERY_V3_RUN_HISTORY"][0]["external_report"], str(old))


if __name__ == "__main__":
    unittest.main()
