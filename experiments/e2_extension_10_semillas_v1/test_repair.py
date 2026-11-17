"""Exact failed-snapshot migration only; every worker below is synthetic."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import bootstrap as b
import repair
from test_bootstrap import MemoryRemote, CoordinatorTests

ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "failed_remote_snapshot.json"


@unittest.skipUnless(FIXTURE.is_file(), "exact reviewed failed snapshot unavailable")
class RepairTests(unittest.TestCase):
    def setUp(self):
        self.failed = json.loads(FIXTURE.read_bytes())
        self.hashes = {name: b.sha((ROOT / name).read_bytes()) for name in
            ("bootstrap.py", "worker.py", "prepare.py", "aggregate.py", "plan.json", "repair.py")}
        self.payloads = repair.build_repair_payloads(self.failed, self.hashes,
            source_commit="b" * 40, now="2026-11-17T14:00:00+00:00")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        self.remote = MemoryRemote(self.payloads)

    def test_pure_reopening_preserves_original_bytes_cost_and_science(self):
        original = {p: v.encode() for p, v in self.failed.items() if p != "snapshot.json"}
        self.assertTrue(self.payloads["ledger.jsonl"].startswith(original["ledger.jsonl"]))
        self.assertEqual(self.payloads["plan.json"], original["plan.json"])
        self.assertEqual(self.payloads["runtime.json"], original["runtime.json"])
        ledger, state = json.loads(self.payloads["ledger.json"]), json.loads(self.payloads["state.json"])
        self.assertEqual(ledger["charged_seconds"], 3.62394042599999)
        self.assertEqual(ledger["remaining_seconds"], 7196.3760595740005)
        self.assertEqual(ledger["limit_seconds"], 7200.0)
        self.assertEqual(state["status"], "running")
        self.assertEqual(state["jobs"], {})
        self.assertEqual(state["prior_failure"]["outcome"], "failed")
        events = b.events_from(self.payloads["ledger.jsonl"])
        self.assertEqual(len(events), 3)
        self.assertEqual(events[2]["event"], "reopen")
        self.assertNotIn(repair.FAILED_REQUEST, self.payloads)
        self.assertEqual(self.payloads[b.REPAIR_ROOT + "/failed_request.json"], original[repair.FAILED_REQUEST])
        self.assertEqual(self.payloads[b.REPAIR_ROOT + "/failed_worker.log"], original[repair.FAILED_LOG])
        b.validate_snapshot(b.make_snapshot(self.payloads), self.payloads)

    def test_rejects_other_head_or_any_phase_checkpoint_or_evidence_change(self):
        with self.assertRaisesRegex(RuntimeError, "exact reviewed failed remote head"):
            repair.build_repair_payloads(self.failed, self.hashes, expected_head="c"*40)
        for name, value in ((repair.FAILED_REQUEST, "changed"),
                ("jobs/759a1b80df7f02c7-271807/fit/phases/supervised/prepared.json", "{}"),
                ("jobs/759a1b80df7f02c7-271807/fit/selected_model.pt", "weights")):
            mutated = dict(self.failed); mutated[name] = value
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                repair.build_repair_payloads(mutated, self.hashes)
        for name in ("plan.json", "prepare.py", "aggregate.py", "worker.py", "bootstrap.py", "repair.py"):
            hashes = dict(self.hashes); hashes[name] = "0"*64
            with self.subTest(code=name), self.assertRaises(RuntimeError):
                repair.build_repair_payloads(self.failed, hashes)

    def test_old_local_job_folder_including_empty_lock_is_preserved(self):
        old = {p: v.encode() for p, v in self.failed.items() if p != "snapshot.json"}
        b.restore(self.workspace / "run", old)
        lock = self.workspace / "run/jobs" / b.FAILED_JOB_ID / "fit/phases/supervised/.writer.lock"
        lock.parent.mkdir(parents=True); lock.touch()
        head = repair.install_remote_repair(self.workspace, self.remote.read())
        self.assertEqual(head, self.remote.read()[0])
        archives = list((self.workspace / "failed_preflight_copies").iterdir())
        self.assertEqual(len(archives), 1)
        self.assertEqual((archives[0] / "fit/request.json").read_bytes(), old[repair.FAILED_REQUEST])
        self.assertTrue((archives[0] / "fit/phases/supervised/.writer.lock").is_file())
        self.assertFalse((self.workspace / "run/jobs" / b.FAILED_JOB_ID).exists())
        self.assertEqual(b.collect(self.workspace / "run"), self.payloads)
        repair.install_remote_repair(self.workspace, self.remote.read())
        self.assertEqual(len(list((self.workspace / "failed_preflight_copies").iterdir())), 1)

    def test_refuses_local_optimizer_phase_artifact_even_if_snapshot_would_ignore_lock(self):
        old = {p: v.encode() for p, v in self.failed.items() if p != "snapshot.json"}
        b.restore(self.workspace / "run", old)
        artifact = self.workspace / "run/jobs" / b.FAILED_JOB_ID / "fit/phases/supervised/checkpoints/00000001-initial.pt"
        b.atomic(artifact, b"weights")
        with self.assertRaises(RuntimeError):
            repair.install_remote_repair(self.workspace, self.remote.read())
        self.assertEqual(artifact.read_bytes(), b"weights")
        self.assertFalse((self.workspace / "failed_preflight_copies").exists())

    def test_torn_migration_after_folder_move_reconciles(self):
        old = {p: v.encode() for p, v in self.failed.items() if p != "snapshot.json"}
        b.restore(self.workspace / "run", old)
        archived = self.workspace / "failed_preflight_copies" / (b.FAILED_JOB_ID + "-" + repair.FAILED_REQUEST_SHA256[:12])
        archived.parent.mkdir(); (self.workspace / "run/jobs" / b.FAILED_JOB_ID).rename(archived)
        repair.install_remote_repair(self.workspace, self.remote.read())
        self.assertEqual(b.collect(self.workspace / "run"), self.payloads)
        self.assertTrue((archived / "fit/request.json").is_file())

    def test_second_failure_is_terminal_and_old_charge_is_added_once(self):
        repair.install_remote_repair(self.workspace, self.remote.read())
        directory = self.workspace / "run"
        reserve = b.append_event(directory, {"event": "reserve", "job_id": b.FAILED_JOB_ID,
            "phase": "fit", "reservation_id": "a"*32, "cap_seconds": 1800, "owner_id": "synthetic",
            "lease_until_utc": (b.utc() + b.timedelta(seconds=1860)).isoformat()})
        b.finish_job(directory, reserve, "failed", 2.0)
        self.assertEqual(b.read_json(directory / "state.json")["status"], "failed")
        self.assertEqual(b.read_json(directory / "ledger.json")["charged_seconds"], b.FAILED_CHARGED_SECONDS + 2)
        b.validate_snapshot(b.make_snapshot(b.collect(directory)), b.collect(directory))
        # A second reopen record, even with the same authorization, is forbidden.
        event = dict(b.events_from(self.payloads["ledger.jsonl"])[2]); event.pop("sha256")
        with self.assertRaisesRegex(RuntimeError, "restricted"):
            b.append_event(directory, event)

    def test_new_worker_needs_remote_reservation_ack_and_lost_finish_ack_never_repeats(self):
        harness = CoordinatorTests(methodName="test_worker_environment_strips_credentials")
        harness.workspace, harness.plan, harness.remote = self.workspace, json.loads(self.payloads["plan.json"]), self.remote
        harness.calls, harness.on_execute = [], lambda: setattr(self.remote, "fail_after_write", True)
        repair.install_remote_repair(self.workspace, self.remote.read())
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ConnectionError):
            b.run_coordinator(self.workspace, harness.plan, self.remote, ("python", {}), harness.command,
                executor=harness.executor, max_jobs=1)
        self.assertEqual(harness.calls, [("fit", b.FAILED_JOB_ID)])
        # Repair installation must leave completed unacknowledged local progress
        # and its previous ACK available for the base reconciliation protocol.
        before = b.collect(self.workspace / "run")
        repair.install_remote_repair(self.workspace, self.remote.read())
        self.assertEqual(b.collect(self.workspace / "run"), before)
        harness.on_execute = None
        with contextlib.redirect_stdout(io.StringIO()):
            b.run_coordinator(self.workspace, harness.plan, self.remote, ("python", {}), harness.command,
                executor=harness.executor, max_jobs=1)
        self.assertEqual(len(harness.calls), 2)
        self.assertNotEqual(harness.calls[0][1], harness.calls[1][1])
        self.assertEqual(b.read_json(self.workspace / "run/ledger.json")["charged_seconds"], b.FAILED_CHARGED_SECONDS + 2)


if __name__ == "__main__":
    unittest.main()
