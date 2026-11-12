"""Colab launcher boundaries, using stdlib mocks and no real GPU or network."""
from contextlib import ExitStack, nullcontext, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("e2_colab_launcher_test", ROOT / "notebooks/run_e2_pilots_colab.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)
SOURCE = "a" * 40
TOKEN = "test-secret-never-command-line"


class ColabLauncherTests(unittest.TestCase):
    def prepared(self, directory, runner):
        drive, repo = Path(directory) / "MyDrive", Path(directory) / "repo"
        drive.mkdir(); repo.mkdir()
        (repo / "requirements-e2-training-gpu.lock.txt").write_text("torch==2.8.0+cu126\n")
        stack = ExitStack()
        for name, value in (("_assert_no_controller", Mock()),
            ("_local_launch_lock", lambda workspace: nullcontext()), ("_mount_drive", lambda: drive),
            ("_checkout", Mock(return_value=repo)), ("_bootstrap_python", Mock(return_value=Path("/fake/python312"))),
            ("_gpu_environment", Mock(return_value={"device": "mock CUDA, not executed"})),
            ("_token", Mock(return_value=TOKEN)), ("_run", Mock()), ("_stream_runner", runner)):
            stack.enter_context(patch.object(launcher, name, value))
        stack.enter_context(patch.object(launcher.platform, "system", return_value="Linux"))
        stack.enter_context(patch.object(launcher.platform, "machine", return_value="x86_64"))
        stack.enter_context(patch.object(launcher.shutil, "which", return_value="/fake/nvidia-smi"))
        return stack, drive

    def test_immutable_source_is_required_before_any_runtime_work(self):
        with patch.object(launcher, "_assert_no_controller") as processes:
            for source in ("main", "__PILOT_SOURCE_COMMIT__", "a" * 39, "z" * 40):
                with self.subTest(source=source), self.assertRaises(ValueError):
                    launcher.run_colab_pilots(source)
            processes.assert_not_called()

    def test_no_nvidia_has_no_cpu_fallback_or_drive_setup(self):
        with patch.object(launcher, "_assert_no_controller"), patch.object(launcher.shutil, "which", return_value=None), patch.object(launcher, "_mount_drive") as mount:
            with self.assertRaisesRegex(RuntimeError, "fallback CPU"):
                launcher.run_colab_pilots(SOURCE)
            mount.assert_not_called()

    def test_budget_exit_three_never_fetches_development_or_executes(self):
        def blocked(args, **kwargs):
            target = Path(args[args.index("--output") + 1]); target.mkdir(parents=True)
            (target / "preflight.json").write_text(json.dumps({"ready": False, "status": "paused_budget_insufficient"}))
            return 3
        runner = Mock(side_effect=blocked)
        with tempfile.TemporaryDirectory() as directory:
            stack, drive = self.prepared(directory, runner)
            with stack, redirect_stdout(io.StringIO()):
                result = launcher.run_colab_pilots(SOURCE, execute=True)
                calls = launcher._run.call_args_list
            self.assertEqual(result["exit_code"], 3)
            self.assertFalse(result["training_started_by_launcher"])
            self.assertEqual(runner.call_count, 1)
            self.assertTrue(Path(result["preflight"]).is_relative_to(drive))
            self.assertFalse(any("fetch_e2_development_sources.py" in str(call.args[0]) for call in calls))
            self.assertNotIn("--execute", runner.call_args.args[0])

    def test_output_must_be_persistent_drive_before_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            stack, drive = self.prepared(directory, Mock())
            with stack, self.assertRaisesRegex(ValueError, "MyDrive"):
                launcher.run_colab_pilots(SOURCE, output=Path(directory) / "ephemeral")
            self.assertFalse((drive / "amended-pilots-v1").exists())

    def test_ready_preflight_precedes_csv_and_execute_token_stays_in_environment(self):
        def admitted(args, **kwargs):
            target = Path(args[args.index("--output") + 1]); target.mkdir(parents=True, exist_ok=True)
            (target / "preflight.json").write_text(json.dumps({"ready": True}))
            return 0
        runner = Mock(side_effect=admitted)
        with tempfile.TemporaryDirectory() as directory:
            stack, drive = self.prepared(directory, runner)
            with stack, patch.dict(os.environ, {"GITHUB_TOKEN": TOKEN}), redirect_stdout(io.StringIO()):
                result = launcher.run_colab_pilots(SOURCE, execute=True)
                setup_calls = launcher._run.call_args_list
            self.assertEqual(runner.call_count, 2)
            self.assertNotIn("--execute", runner.call_args_list[0].args[0])
            self.assertIn("--execute", runner.call_args_list[1].args[0])
            for call in runner.call_args_list:
                self.assertNotIn(TOKEN, " ".join(map(str, call.args[0])))
                self.assertEqual(call.kwargs["env"]["GITHUB_TOKEN"], TOKEN)
            self.assertTrue(any("fetch_e2_development_sources.py" in str(call.args[0]) for call in setup_calls))
            self.assertTrue(all("GITHUB_TOKEN" not in call.kwargs["env"] for call in setup_calls))
            self.assertTrue(result["execution_requested"])
            self.assertFalse(result["training_completion_claimed"])
            self.assertTrue(Path(result["output"]).is_relative_to(drive))

    def test_stream_redacts_token_from_visible_and_persistent_logs(self):
        process = SimpleNamespace(stdout=iter(["line " + TOKEN + "\n"]), wait=lambda: 0)
        visible = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, patch.object(launcher.subprocess, "Popen", return_value=process) as popen, redirect_stdout(visible):
            log = Path(directory) / "runner.log"
            code = launcher._stream_runner(["python", "runner.py"], cwd=directory,
                env={"GITHUB_TOKEN": TOKEN}, log_path=log, token=TOKEN)
            logged = log.read_text()
        self.assertEqual(code, 0)
        self.assertNotIn(TOKEN, visible.getvalue() + logged)
        self.assertIn("[REDACTED]", logged)
        self.assertNotIn(TOKEN, " ".join(popen.call_args.args[0]))

    def test_sparse_code_checkout_excludes_numerical_development_inputs(self):
        self.assertTrue(any("scripts" in path for path in launcher.SPARSE_PATHS))
        self.assertFalse(any("/results/" in path for path in launcher.SPARSE_PATHS))
        self.assertFalse(any("daily_market_development.csv" in path for path in launcher.SPARSE_PATHS))

    def test_process_guard_distinguishes_shell_code_from_real_controller_argument(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for pid, arguments in (("987001", ["bash", "-lc", "python run_e2_amended_pilots.py --execute"]),
                                   ("987002", ["python", "/code/run_e2_amended_pilots.py", "--execute"])):
                proc = root / pid; proc.mkdir()
                (proc / "cmdline").write_bytes(b"\0".join(value.encode() for value in arguments) + b"\0")
            self.assertEqual(launcher._controller_pids(root), [987002])


if __name__ == "__main__":
    unittest.main()
