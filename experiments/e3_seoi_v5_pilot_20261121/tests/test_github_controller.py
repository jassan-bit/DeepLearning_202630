"""Persistence-control fixtures only: no network, training or optimizer steps."""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from seoi_v5.github_controller import ControllerError, GithubController, credentials, main, storage_code

C = "a" * 40
TOKEN = "fixture-token-secret-never-log"
KEY = "fixture-backup-key-32characters-or-more"


class DurabilityError(RuntimeError):
    pass


class StorageError(RuntimeError):
    pass


class Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value


class FakeRunner:
    DurabilityError = DurabilityError
    NPZ_BYTES = 4
    NPZ_SHA256 = hashlib.sha256(b"DATA").hexdigest()

    def __init__(self):
        self.prepare_calls = self.run_calls = self.assess_calls = 0
        self.run_behavior = None
        self.next_job_started = False

    @staticmethod
    def safe(path):
        path = Path(path).absolute()
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("symlink")
        return path

    @staticmethod
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def read_json(path):
        return json.loads(Path(path).read_text())

    @staticmethod
    def write_json(path, value, **kwargs):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    @staticmethod
    def atomic_bytes(path, value, **kwargs):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(value)

    @staticmethod
    def utc_now():
        return "2026-11-21T00:00:00Z"

    def verify_sources(self, root):
        return {"manifest_sha256": self.sha(Path(root) / "SOURCE_MANIFEST.json")}

    @staticmethod
    def case_lock(case):
        return nullcontext()

    def prepare(self, source, data, case):
        self.prepare_calls += 1
        Path(case).mkdir()
        self.write_case(case)
        self.atomic_bytes(Path(case) / "inputs/cohort.npz", Path(data).read_bytes())

    def write_case(self, case):
        self.write_json(Path(case) / "plan.json", {"case_id": "fixture-case", "jobs": [1, 2]})
        self.write_json(Path(case) / "PUBLIC_SOURCE_ANCHOR.json", {"source_commit": C})

    def load_case(self, source, case, *, load_arrays=False):
        raw = Path(case) / "inputs/cohort.npz"
        if not raw.is_file() or self.sha(raw) != self.NPZ_SHA256:
            raise ValueError("raw identity")
        return self.read_json(Path(case) / "plan.json"), None, None

    def status(self, source, case):
        return {"case_id": "fixture-case", "completed_neural_fits": 0}

    def run(self, source, case, *, on_checkpoint):
        self.run_calls += 1
        if self.run_behavior:
            return self.run_behavior(case, on_checkpoint)
        return {"status": "fixture-no-training"}

    def assess(self, source, case):
        self.assess_calls += 1
        return {"status": "fixture-no-predictions"}

    def export_case(self, source, case, output):
        Path(output).mkdir()
        return {"status": "fixture-export"}


class FakeStore:
    def __init__(self, runner):
        self.runner, self.exists = runner, False
        self.head, self.serial = None, 0
        self.plan, self.restore_calls = None, 0
        self.stages, self.sessions = [], []
        self.fail_stage = None

    def status(self):
        result = {"exists": self.exists, "source_commit": C}
        if self.exists:
            result.update({"head": self.head, "case_id": self.plan["case_id"],
                           "plan_sha256": self.runner.digest(self.plan)})
        return result

    def backup(self, case, stage=None):
        if stage == self.fail_stage:
            raise RuntimeError(TOKEN + KEY)
        self.stages.append(stage)
        session = Path(case) / "github_session.json"
        self.sessions.append(self.runner.read_json(session) if session.exists() else {})
        self.exists = True; self.serial += 1; self.head = f"{self.serial:040x}"
        self.plan = self.runner.read_json(Path(case) / "plan.json")
        receipt = {"status": "verified", "commit": self.head, "source_commit": C,
                   "plan_sha256": self.runner.digest(self.plan), "case_id": self.plan["case_id"],
                   "remote_verified": True}
        self.runner.write_json(Path(case) / ".github_backup_receipt.json", receipt)
        return receipt

    def restore(self, case):
        self.restore_calls += 1
        if Path(case).exists():
            raise AssertionError("Never overwrite a local case")
        Path(case).mkdir()
        self.runner.write_case(case)
        self.runner.write_json(Path(case) / "remote_restore.json", {"restored": True})
        receipt = {"commit": self.head, "source_commit": C, "remote_verified": True}
        self.runner.write_json(Path(case) / ".github_backup_receipt.json", receipt)
        return receipt


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"; self.source.mkdir()
        self.data = self.root / "cache.npz"; self.data.write_bytes(b"DATA")
        self.case = self.root / "case"
        self.runner, self.clock = FakeRunner(), Clock()
        (self.source / "SOURCE_MANIFEST.json").write_text("fixture-manifest")
        self.runner.write_json(self.source / "PUBLIC_SOURCE_ANCHOR.json", {
            "source_commit": C, "manifest_sha256": self.runner.sha(self.source / "SOURCE_MANIFEST.json")})
        self.store = FakeStore(self.runner)
        self.controller = GithubController(self.source, self.case, self.data, C, self.store,
                                           runner_module=self.runner, monotonic=self.clock)

    def tearDown(self):
        self.temp.cleanup()

    def test_setup_prepares_and_verifies_remote_before_any_run(self):
        result = self.controller.setup()
        self.assertTrue(result["backup"]["remote_verified"])
        self.assertEqual(self.runner.prepare_calls, 1)
        self.assertEqual(self.runner.run_calls, 0)
        self.assertFalse(self.store.sessions[-1]["session_open"])

    def test_setup_restores_authenticated_case_without_repreparing(self):
        self.store.exists = True; self.store.head = "b" * 40
        self.store.plan = {"case_id": "fixture-case", "jobs": [1, 2]}
        result = self.controller.setup()
        self.assertTrue(result["restored"])
        self.assertEqual(self.store.restore_calls, 1)
        self.assertEqual(self.runner.prepare_calls, 0)
        self.assertEqual((self.case / "inputs/cohort.npz").read_bytes(), b"DATA")
        self.assertTrue((self.case / "remote_restore.json").exists())

    def test_existing_case_is_preserved_and_not_restored(self):
        self.controller.setup()
        self.controller.setup()
        self.assertEqual(self.runner.prepare_calls, 1)
        self.assertEqual(self.store.restore_calls, 0)

    def test_stale_local_case_is_not_overwritten_or_pushed(self):
        self.controller.setup(); count = len(self.store.stages)
        self.store.head = "f" * 40
        with self.assertRaisesRegex(ControllerError, "stale"):
            self.controller.setup()
        self.assertEqual(len(self.store.stages), count)
        self.assertEqual(self.store.restore_calls, 0)

    def test_remote_case_plan_mismatch_stops_before_any_run(self):
        self.controller.setup(); self.store.plan = {"case_id": "other", "jobs": [1, 2]}
        with self.assertRaisesRegex(ControllerError, "plan_identity"):
            self.controller.run()
        self.assertEqual(self.runner.run_calls, 0)

    def test_missing_remote_branch_does_not_recreate_previous_history(self):
        self.controller.setup(); self.store.exists = False
        with self.assertRaisesRegex(ControllerError, "remote_branch_missing"):
            self.controller.setup()

    def test_data_tamper_stops_restore_without_training(self):
        self.store.exists = True; self.store.head = "b" * 40
        self.store.plan = {"case_id": "fixture-case", "jobs": [1, 2]}
        self.data.write_bytes(b"FAIL")
        with self.assertRaisesRegex(ControllerError, "cache_identity"):
            self.controller.setup()
        self.assertEqual(self.runner.run_calls, 0)
        self.assertFalse((self.case / "inputs/cohort.npz").exists())

    def test_public_source_anchor_mismatch_stops_before_preparation(self):
        self.runner.write_json(self.source / "PUBLIC_SOURCE_ANCHOR.json", {"source_commit": "c" * 40})
        with self.assertRaisesRegex(ControllerError, "anchor_differs"):
            self.controller.setup()
        self.assertEqual(self.runner.prepare_calls, 0)

    def test_backup_failure_before_run_starts_no_jobs(self):
        self.controller.setup(); self.store.fail_stage = "run_started"
        with self.assertRaises(DurabilityError):
            self.controller.run()
        self.assertEqual(self.runner.run_calls, 0)
        self.assertTrue(self.runner.read_json(self.case / "github_session.json")["session_open"])

    def test_job_backups_forced_and_epoch_backups_throttled(self):
        self.controller.setup()

        def behavior(case, hook):
            self.clock.value += 10; hook(case, {"kind": "epoch_checkpoint", "job": "one"})
            self.clock.value += 5; hook(case, {"kind": "job_completed", "job": "one"})
            self.clock.value += 61; hook(case, {"kind": "epoch_checkpoint", "job": "two"})
            return {"status": "fixture-no-training"}

        self.runner.run_behavior = behavior
        self.controller.run()
        self.assertEqual(self.store.stages[1:], ["run_started", "job_completed", "epoch_checkpoint", "run_finished"])
        self.assertTrue(self.store.sessions[1]["session_open"])
        self.assertFalse(self.store.sessions[-1]["session_open"])

    def test_synchronous_job_save_failure_stops_next_job_and_keeps_open_session(self):
        self.controller.setup(); self.store.fail_stage = "job_completed"

        def behavior(case, hook):
            hook(case, {"kind": "job_completed", "job": "one"})
            self.runner.next_job_started = True

        self.runner.run_behavior = behavior
        with self.assertRaises(DurabilityError):
            self.controller.run()
        self.assertFalse(self.runner.next_job_started)
        self.assertTrue(self.runner.read_json(self.case / "github_session.json")["session_open"])
        self.assertNotIn("run_finished", self.store.stages)

    def test_failed_fit_record_is_backed_up_before_next_job(self):
        self.controller.setup()

        def behavior(case, hook):
            hook(case, {"kind": "fit_failed", "job": "one"})
            return {"status": "fixture-incomplete"}

        self.runner.run_behavior = behavior
        self.controller.run()
        self.assertEqual(self.store.stages[-2:], ["fit_failed", "run_finished"])

    def test_existing_pending_reconciliation_is_followed_by_latest_state_backup(self):
        self.controller.setup()
        (self.case / ".github_pending").mkdir()
        original = self.store.backup
        calls = []

        def reconcile(case, stage=None):
            calls.append(stage)
            pending = Path(case) / ".github_pending"
            if pending.exists():
                pending.rmdir()
            return original(case, stage=stage)

        self.store.backup = reconcile
        self.controller._backup("latest-state")
        self.assertEqual(calls, ["latest-state", "latest-state"])

    def test_final_save_failure_is_not_swallowed(self):
        self.controller.setup(); self.store.fail_stage = "run_finished"
        with self.assertRaises(DurabilityError):
            self.controller.run()

    def test_caught_interrupt_closes_session_and_backs_up(self):
        self.controller.setup()

        def behavior(case, hook):
            raise KeyboardInterrupt()

        self.runner.run_behavior = behavior
        with self.assertRaises(KeyboardInterrupt):
            self.controller.run()
        self.assertEqual(self.store.stages[-1], "run_interrupted")
        self.assertFalse(self.store.sessions[-1]["session_open"])

    def test_assess_and_export_backup_without_running_fits(self):
        self.controller.setup()
        self.controller.assess()
        self.controller.export(self.root / "export")
        self.assertEqual(self.runner.run_calls, 0)
        self.assertEqual(self.runner.assess_calls, 1)
        self.assertEqual(self.store.stages[-3:], ["assessment_started", "assessment_finished", "public_export_created"])

    def test_cli_uses_stdin_and_never_prints_exception_credentials(self):
        args = ["--source-root", str(self.source), "--case-root", str(self.case),
                "--data-file", str(self.data), "--source-commit", C, "setup"]
        self.assertNotIn(TOKEN, args); self.assertNotIn(KEY, args)
        output = io.StringIO()

        def factory(**kwargs):
            self.assertEqual(kwargs["token"], TOKEN); self.assertEqual(kwargs["backup_secret"], KEY)
            raise RuntimeError(TOKEN + KEY)

        code = main(args, stdin=io.StringIO(json.dumps({"token": TOKEN, "backup_secret": KEY})),
                    stdout=output, store_factory=factory, runner_module=self.runner)
        self.assertEqual(code, 1)
        self.assertNotIn(TOKEN, output.getvalue()); self.assertNotIn(KEY, output.getvalue())
        self.assertEqual(json.loads(output.getvalue())["status"], "stopped")

    def test_safe_auth_diagnostic_survives_controller_wrapping(self):
        self.controller.setup()

        def denied(case, stage=None):
            raise StorageError("github_auth_or_contents_write_permission_required status=403")

        self.store.backup = denied
        with self.assertRaises(DurabilityError) as caught:
            self.controller.run()
        self.assertEqual(storage_code(caught.exception), "github_auth_or_contents_write_permission_required status=403")

    def test_cli_does_not_write_credentials_to_case_files(self):
        args = ["--source-root", str(self.source), "--case-root", str(self.case),
                "--data-file", str(self.data), "--source-commit", C, "setup"]
        output = io.StringIO()
        before_environment = dict(os.environ)
        code = main(args, stdin=io.StringIO(json.dumps({"token": TOKEN, "backup_secret": KEY})),
                    stdout=output, store_factory=lambda **kwargs: self.store, runner_module=self.runner)
        self.assertEqual(code, 0)
        self.assertEqual(dict(os.environ), before_environment)
        for path in self.root.rglob("*"):
            if path.is_file():
                raw = path.read_bytes()
                self.assertNotIn(TOKEN.encode(), raw); self.assertNotIn(KEY.encode(), raw)


class CredentialTests(unittest.TestCase):
    def test_reject_short_backup_key_or_extra_keys(self):
        for value in ({"token": TOKEN, "backup_secret": "short"},
                      {"token": TOKEN, "backup_secret": KEY, "action": "run"}):
            with self.assertRaises(ControllerError):
                credentials(io.StringIO(json.dumps(value)))

    def test_duplicate_credentials_keys_rejected(self):
        with self.assertRaises(ControllerError):
            credentials(io.StringIO('{"token":"one","token":"two","backup_secret":"' + KEY + '"}'))

    def test_arbitrary_exception_text_is_not_a_storage_diagnostic(self):
        self.assertEqual(storage_code(RuntimeError(TOKEN + KEY)), "storage_detail_unavailable")
        self.assertEqual(storage_code(StorageError("an HTTP response body / with secrets")), "storage_detail_unavailable")
        self.assertEqual(storage_code(StorageError("github_secondary_rate_limit status=403")),
                         "github_secondary_rate_limit status=403")


if __name__ == "__main__":
    unittest.main()
