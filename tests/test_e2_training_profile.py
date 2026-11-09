"""Tests for bounded profiling supervision; no model fitting in this file."""
from __future__ import annotations
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from profile_e2_training import run_isolated_profile, profile_models


def alive(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat[stat.rfind(")") + 2:].split()[0] != "Z"


class ProfileSupervisorTests(unittest.TestCase):
    def test_success_preserves_worker_scope_and_observed_resources(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "new"
            worker = {"synthetic_only": True, "market_data_read": False, "optimizer_steps": 2}
            code = ("from pathlib import Path;import json;"
                    f"Path({str(output / 'worker_profile.json')!r}).write_text(json.dumps({worker!r}));"
                    "print('artificial fixture completed')")
            result = run_isolated_profile([sys.executable, "-c", code], output,
                                          timeout_seconds=5, poll_interval=.005)
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["returncode"], 0)
            self.assertEqual(result["worker_profile"], worker)
            self.assertFalse(result["rss_watchdog_is_hard_memory_limit"])
            self.assertFalse(result["supervisor_reads_market_files"])
            self.assertGreaterEqual(result["sampled_peak_process_group_rss_bytes"], 0)
            self.assertIn("artificial fixture completed", (output / "stdout.log").read_text())
            self.assertEqual(json.loads((output / "result.json").read_text())["status"], "success")

    def test_worker_failure_is_retained_with_no_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "failed"
            result = run_isolated_profile(
                [sys.executable, "-c", "import sys;print('failure detail',file=sys.stderr);sys.exit(7)"],
                output, timeout_seconds=5)
            self.assertEqual(result["status"], "worker_failure")
            self.assertEqual(result["returncode"], 7)
            self.assertIsNone(result["worker_profile"])
            self.assertIn("failure detail", (output / "stderr.log").read_text())
            self.assertEqual(json.loads((output / "result.json").read_text())["returncode"], 7)

    def test_timeout_terminates_only_the_spawned_process_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "timeout"
            own_pid = os.getpid()
            child_file = root / "child.pid"
            code = ("import subprocess,sys,time;from pathlib import Path;"
                    "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);"
                    f"Path({str(child_file)!r}).write_text(str(child.pid));"
                    "time.sleep(30)")
            result = run_isolated_profile([sys.executable, "-c", code], output,
                                          timeout_seconds=.6, poll_interval=.01)
            self.assertEqual(result["status"], "timeout")
            self.assertNotEqual(result["returncode"], 0)
            self.assertTrue(alive(own_pid))
            self.assertTrue(child_file.is_file())
            child = int(child_file.read_text())
            deadline = time.monotonic() + 2
            while alive(child) and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertFalse(alive(child))

    def test_memory_watchdog_records_observed_breach(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "memory"
            result = run_isolated_profile(
                [sys.executable, "-c", "import time;data=bytearray(20*1024**2);time.sleep(3)"],
                output, timeout_seconds=5, memory_limit_bytes=1024**2, poll_interval=.005)
            self.assertEqual(result["status"], "memory_limit_observed")
            self.assertGreater(result["sampled_peak_process_group_rss_bytes"], 1024**2)
            self.assertFalse(result["rss_watchdog_is_hard_memory_limit"])

    def test_existing_outputs_and_invalid_supervision_values_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing = root / "existing"
            existing.mkdir()
            marker = existing / "keep.txt"
            marker.write_text("unchanged")
            with self.assertRaises(FileExistsError):
                run_isolated_profile([sys.executable, "-c", "pass"], existing)
            self.assertEqual(marker.read_text(), "unchanged")
            for values in [{"timeout_seconds": 0}, {"memory_limit_bytes": 0},
                           {"poll_interval": 0}, {"memory_limit_bytes": True}]:
                with self.subTest(values=values), self.assertRaises(ValueError):
                    run_isolated_profile([sys.executable, "-c", "pass"], root / "invalid", **values)
                self.assertFalse((root / "invalid").exists())
            with self.assertRaises(ValueError):
                run_isolated_profile("python -c pass", root / "invalid-shell")

    def test_invalid_worker_json_is_retained_as_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bad-json"
            code = f"from pathlib import Path;Path({str(output / 'worker_profile.json')!r}).write_text('not JSON')"
            result = run_isolated_profile([sys.executable, "-c", code], output,
                                          timeout_seconds=5, poll_interval=.005)
            self.assertEqual(result["status"], "invalid_worker_profile")
            self.assertIsNotNone(result["failure"])
            self.assertEqual(result["returncode"], 0)

    def test_campaign_refuses_unbounded_epochs_before_any_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = root / "protocol.json"
            protocol.write_text(json.dumps({"models": [{"model": "Example"}],
                                            "neural_reference_profiles": []}))
            output = root / "campaign"
            with self.assertRaises(ValueError):
                profile_models(protocol, output, ["Example"], epochs=4)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

