"""Crash and budget safety for the one-time optimized campaign recovery.

These tests use the frozen durable ledger, fake monotonic time, and atomic
in-memory remote acknowledgements. No CUDA, market values or network is used.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
if not (ROOT / "scripts/e2_search_session.py").is_file():
    # Local isolated staging; in the repository the frozen ledger is in scripts.
    sys.path.append(str(ROOT.parent / "e2opt/scripts"))
import e2_search_session as engine
import e2_optimized_reconciliation_v1 as recovery

LEGACY_SECONDS = 156553.03253392776
INITIAL_SECONDS = 16246.967466072238
EFFECTIVE_SECONDS = 45046.96746607224
FIRST_CLOSED_SECONDS = 8748.729611191997
OLD_ALLOCATION_SECONDS = 28800.0
RESERVATION_SECONDS = 7500.0


class Clock:
    def __init__(self, value=1000.0):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def public_records(directory):
    return {p.relative_to(directory).as_posix(): p.read_bytes()
            for p in Path(directory).rglob("*.json")}


def observed_outcome(identity=None):
    return {"status": "completed", "run_id": "c" * 64,
            "resource_use": {"optimizer_steps": 4, "optimizer_steps_known_lower_bound": 4,
                "optimizer_steps_unknown": False, "partial_optimizer_update_may_have_occurred": False,
                "phase_records": {}, "checkpoint_policy": "selection_metadata_only",
                "cuda_metrics": {}, **({"identity": identity} if identity else {})},
            "predictions_original": [0.1, 0.2],
            "prediction_dates": ["2024-03-01T00:00:00+00:00", "2024-03-02T00:00:00+00:00"],
            "phase_epochs": {"supervised": 2}, "reason": None, "fatal": False}


class MemoryTransport:
    """Create-only remote storage; acknowledgement loss retains committed bytes."""
    def __init__(self, order):
        self.files = {}
        self.order = order
        self.lose_observed_ack = False

    def read(self, path):
        return self.files.get(path)

    def create(self, path, payload):
        if path in self.files:
            if self.files[path] != payload:
                raise RuntimeError("immutable_receipt_conflict")
            return self.files[path]
        self.files[path] = payload
        record = json.loads(payload)
        phase = record.get("phase", "authorization")
        self.order.append("receipt_" + phase)
        if phase == "observed" and self.lose_observed_ack:
            self.lose_observed_ack = False
            raise RuntimeError("observed_ack_lost")
        return payload


class CampaignMirror:
    def __init__(self, directory, head, target_job, order):
        self.directory = Path(directory)
        self._head = head
        self.original = public_records(self.directory)
        self.remote = deepcopy(self.original)
        self.target_path = "results/" + engine.SearchSession._job_filename(target_job)
        self.order = order
        self.receipts = None
        self.push_count = 0
        self.fail_logical_push_once = False
        self.lose_logical_ack_once = False

    def inspect_remote(self):
        self.order.append("inspect_campaign")
        return {"remote_commit": self._head}

    def push(self, directory, *, paths=None, message=""):
        local = public_records(Path(directory))
        for name, payload in self.original.items():
            if local.get(name) != payload:
                raise AssertionError("Existing scientific record was changed: " + name)
        chosen = tuple(local) if paths is None else tuple(paths)
        publishing = self.target_path in chosen and self.target_path in local
        if publishing:
            if self.receipts is None or self.receipts.inspect().get("observed") is None:
                raise AssertionError("Logical result published before durable observation")
            if self.fail_logical_push_once:
                self.fail_logical_push_once = False
                raise RuntimeError("logical_push_failed")
        changes = {name: local[name] for name in chosen if self.remote.get(name) != local[name]}
        self.remote.update(changes)
        self.push_count += 1
        if changes:
            self._head = hashlib.sha1((self._head + str(self.push_count)).encode()).hexdigest()
        self.order.append("logical_result_published" if publishing else "markers_published")
        if publishing and self.lose_logical_ack_once:
            self.lose_logical_ack_once = False
            raise RuntimeError("logical_ack_lost")
        return {"remote_commit": self._head, "synced_paths": chosen}


class OptimizedRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed_temp = tempfile.TemporaryDirectory()
        cls.seed = Path(cls.seed_temp.name) / "campaign"
        cls.seed_clock = Clock()
        cls.plan = {"version": "e2-optimized-fixture", "source_commit": "a" * 40,
                    "registration_sha256": "b" * 64,
                    "legacy_cache": {"spent_seconds": LEGACY_SECONDS},
                    "initial_controls": {"campaign_budget_seconds": INITIAL_SECONDS},
                    "checkpoint_policy": "selection_metadata_only_refit_separate"}
        def session():
            return engine.SearchSession(cls.seed, cls.plan,
                session_budget_seconds=OLD_ALLOCATION_SECONDS,
                campaign_budget_seconds=INITIAL_SECONDS,
                max_storage_bytes=10 * 1024 ** 3, clock=cls.seed_clock)
        first = session()
        overhead = 2.168943662998572
        for number in range(331):
            payload = {"candidate": "prior-first-" + str(number), "seed": 271201}
            job = engine.digest({"plan_sha256": engine.digest(cls.plan), "job": payload})
            def result(deadline):
                cls.seed_clock.advance((FIRST_CLOSED_SECONDS - overhead) / 331)
                return observed_outcome()
            first.run_job(job, payload, result, reservation_seconds=RESERVATION_SECONDS)
        first._paused = "campaign_budget"
        first._append_event("session_paused", session_id=first._session_id,
            reason="campaign_budget", details={"reservation_seconds": RESERVATION_SECONDS})
        cls.seed_clock.advance(overhead)
        first.close()
        second = session()
        second.extend_campaign_budget(EFFECTIVE_SECONDS,
            reason="Explicit user authorization: combined total 48 -> 56 hours.")
        cls.extension = second._events[-1]
        for number in range(558):
            payload = {"candidate": "prior-second-" + str(number), "seed": 271201}
            job = engine.digest({"plan_sha256": engine.digest(cls.plan), "job": payload})
            def result(deadline):
                cls.seed_clock.advance(10)
                return observed_outcome()
            second.run_job(job, payload, result, reservation_seconds=RESERVATION_SECONDS)
        cls.identity = {"model": "TSMixer", "origin": "2024-04-01", "candidate_id": "hpo:fixture:hpo-012",
            "payload_sha256": "e" * 64, "seed": 271202, "stage": "hpo", "prepared_signature": "d" * 64}
        cls.payload = deepcopy(cls.identity)
        cls.job = engine.digest({"plan_sha256": engine.digest(cls.plan), "job": cls.payload})
        cls.start = second._append_event("job_started", job_id=cls.job, session_id=second._session_id,
            job_payload_sha256=engine.digest(cls.payload), reservation_seconds=RESERVATION_SECONDS)
        second._closed = True
        second._release_lock()
        assert cls.start["sequence"] == 1783
        assert len(list((cls.seed / "results").glob("*.json"))) == 889

    @classmethod
    def tearDownClass(cls):
        cls.seed_temp.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "campaign"
        shutil.copytree(self.seed, self.directory)
        self.original = public_records(self.directory)
        self.clock = Clock()
        self.order = []
        self.calls = []
        self.incident = replace(recovery.INCIDENT,
            sealed_head="1" * 40, plan_sha256=engine.digest(self.plan),
            scientific_source_commit=self.plan["source_commit"], previous_completed_jobs=889,
            target_job_id=self.job, target_payload_sha256=engine.digest(self.payload),
            target_started_sha256=self.start["sha256"], target_sequence=self.start["sequence"],
            old_session_id=self.start["session_id"], reservation_seconds=RESERVATION_SECONDS,
            initial_optimized_budget_seconds=INITIAL_SECONDS,
            effective_optimized_budget_seconds=EFFECTIVE_SECONDS,
            old_session_allocation_seconds=OLD_ALLOCATION_SECONDS,
            legacy_spent_seconds=LEGACY_SECONDS, authorized_combined_hours=56,
            budget_extension_event_sha256=self.extension["sha256"])
        patcher = mock.patch.object(recovery, "INCIDENT", self.incident)
        patcher.start()
        self.addCleanup(patcher.stop)

    def view(self, authorization=None):
        return recovery.read_snapshot(self.directory, self.plan,
            session_class=engine.SearchSession, authorization=authorization)

    def authorization(self, total=57):
        return recovery.build_authorization(self.view(), total_hours=total,
            implementation_sha256={"scripts/e2_optimized_reconciliation_v1.py": "f" * 64})

    def prepare(self, total=57):
        self.auth = self.authorization(total)
        self.transport = MemoryTransport(self.order)
        self.receipts = recovery.RecoveryReceiptStore(self.transport, self.auth)
        self.mirror = CampaignMirror(self.directory, self.incident.sealed_head, self.job, self.order)
        self.mirror.receipts = self.receipts
        return self.reconciler()

    def reconciler(self):
        return recovery.ManualReconciler(self.view(self.auth), self.auth, self.mirror, self.receipts,
            authorization_sha256=engine.digest(self.auth), recovery_source_commit="f" * 40,
            physical_session_started_at=self.clock(), clock=self.clock,
            executor_absence_confirmed=True)

    def callback(self, deadline):
        self.assertIsNotNone(self.receipts.inspect().get("started"))
        self.calls.append(deadline)
        self.order.append("physical_fit")
        self.clock.advance(5)
        return observed_outcome(deepcopy(self.identity))

    def recover(self, reconciler=None, **kwargs):
        return (self.reconciler() if reconciler is None else reconciler).recover(
            kwargs.pop("job_id", self.job), kwargs.pop("payload", deepcopy(self.payload)),
            kwargs.pop("callback", self.callback), reservation_seconds=RESERVATION_SECONDS, **kwargs)

    def assert_preserved(self):
        for name, payload in self.original.items():
            self.assertEqual((self.directory / name).read_bytes(), payload, name)

    def test_snapshot_is_read_only_and_56_hours_cannot_admit_another_fit(self):
        view = self.view()
        self.assertEqual(len(view.completed_job_ids), 889)
        self.assertEqual(view.unresolved_job_ids, (self.job,))
        remaining = EFFECTIVE_SECONDS - view.charged_wall_seconds
        self.assertAlmostEqual(remaining, 7498.23785488025, places=5)
        self.assertLess(remaining, RESERVATION_SECONDS)
        for total in (None, 0, 56, 56 + 1 / 3600):
            with self.subTest(total=total), self.assertRaises((ValueError, recovery.ReconciliationError)):
                self.authorization(total)
        self.assertEqual(self.calls, [])
        self.assert_preserved()
        self.assertEqual(public_records(self.directory), self.original)

    def test_explicit_57_hour_retry_keeps_889_original_receipts_and_counts_physical_cost(self):
        answer = self.recover(self.prepare())
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(answer["outcome"], observed_outcome(self.identity))
        self.assertIsNotNone(self.receipts.inspect()["completed"])
        self.assert_preserved()
        with engine.SearchSession(self.directory, self.plan, session_budget_seconds=28800,
                campaign_budget_seconds=INITIAL_SECONDS, max_storage_bytes=10 * 1024 ** 3,
                clock=self.clock) as session:
            self.assertEqual(len(session.completed_job_ids), 890)
            self.assertEqual(session.get_result(self.job, self.payload), observed_outcome(self.identity))
            self.assertAlmostEqual(session.charged_wall_seconds, FIRST_CLOSED_SECONDS + OLD_ALLOCATION_SECONDS + 5, places=5)
            self.assertAlmostEqual(session.campaign_budget_seconds + LEGACY_SECONDS, 57 * 3600, places=5)
            self.assertEqual(session.initial_campaign_budget_seconds, INITIAL_SECONDS)
        self.assertIsNone(self.auth["original_unknown_attempt"]["elapsed_seconds"])
        self.assertIsNone(self.auth["original_unknown_attempt"]["optimizer_steps"])

    def test_elapsed_recovery_setup_can_exhaust_the_new_reservation_without_a_fit(self):
        reconciler = self.prepare()
        # An explicit increase does not waive a second admission check after
        # setup has used more than the newly added hour's available margin.
        self.clock.advance(3600)
        with self.assertRaises(recovery.ReconciliationBudgetPause):
            self.recover(reconciler)
        self.assertEqual(self.calls, [])
        self.assertIsNone(self.receipts.inspect()["started"])
        self.assertEqual(public_records(self.directory), self.original)

    def test_receipt_and_result_acknowledgements_are_in_safe_order(self):
        self.recover(self.prepare())
        self.assertLess(self.order.index("receipt_started"), self.order.index("physical_fit"))
        self.assertLess(self.order.index("receipt_observed"), self.order.index("logical_result_published"))
        self.assertLess(self.order.index("logical_result_published"), self.order.index("receipt_completed"))

    def test_changed_remote_head_rejects_fit_without_claim(self):
        reconciler = self.prepare()
        self.mirror._head = "9" * 40
        with self.assertRaises(recovery.ReconciliationError):
            self.recover(reconciler)
        self.assertEqual(self.calls, [])
        self.assertIsNone(self.receipts.inspect()["started"])
        self.assert_preserved()

    def test_changed_job_or_payload_rejects_fit_without_scientific_writes(self):
        reconciler = self.prepare()
        for kwargs in ({"job_id": "9" * 64}, {"payload": {**self.payload, "seed": 7}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(recovery.ReconciliationError):
                self.recover(reconciler, **kwargs)
        self.assertEqual(self.calls, [])
        self.assertEqual(public_records(self.directory), self.original)

    def test_completed_recovery_is_returned_without_second_physical_fit(self):
        self.recover(self.prepare())
        self.recover()
        self.assertEqual(len(self.calls), 1)
        self.assert_preserved()

    def test_worker_interruption_with_started_receipt_blocks_second_physical_fit(self):
        reconciler = self.prepare()
        def interrupted(deadline):
            self.calls.append(deadline)
            self.clock.advance(3)
            raise KeyboardInterrupt
        with self.assertRaises(recovery.ReconciliationError):
            self.recover(reconciler, callback=interrupted)
        self.assertIsNotNone(self.receipts.inspect()["started"])
        self.assertIsNone(self.receipts.inspect()["observed"])
        # A fresh runtime has a new nonce and only the durable remote receipt.
        self.receipts = recovery.RecoveryReceiptStore(self.transport, self.auth)
        self.mirror.receipts = self.receipts
        with self.assertRaises(recovery.ReconciliationError):
            self.recover()
        self.assertEqual(len(self.calls), 1)
        self.assert_preserved()

    def test_started_claim_blocks_new_reconciler_even_when_receipt_store_is_reused(self):
        reconciler = self.prepare()
        def interrupted(deadline):
            self.calls.append(deadline)
            raise KeyboardInterrupt
        with self.assertRaises(recovery.ReconciliationError):
            self.recover(reconciler, callback=interrupted)
        # Reusing the in-memory store must not let a new reconciler forget that
        # a physical call was already claimed but its outcome is unknown.
        with self.assertRaises(recovery.ReconciliationError):
            self.recover()
        self.assertEqual(len(self.calls), 1)
        self.assertIsNone(self.receipts.inspect()["observed"])
        self.assert_preserved()

    def test_observed_outcome_survives_logical_push_failure_without_refitting(self):
        reconciler = self.prepare()
        self.mirror.fail_logical_push_once = True
        with self.assertRaises(Exception):
            self.recover(reconciler)
        self.assertIsNotNone(self.receipts.inspect()["observed"])
        self.assertIsNone(self.receipts.inspect()["completed"])
        self.recover()
        self.assertEqual(len(self.calls), 1)
        self.assertIsNotNone(self.receipts.inspect()["completed"])
        self.assert_preserved()

    def test_observed_acknowledgement_loss_does_not_repeat_the_fit(self):
        reconciler = self.prepare()
        self.transport.lose_observed_ack = True
        # The store may resolve the lost acknowledgement by reading back the
        # exact immutable bytes. That is a successful durable observation.
        self.recover(reconciler)
        self.assertIsNotNone(self.receipts.inspect()["observed"])
        self.recover()
        self.assertEqual(len(self.calls), 1)
        self.assert_preserved()

    def test_fresh_runtime_uses_remote_observation_after_unacknowledged_local_result_loss(self):
        reconciler = self.prepare()
        self.mirror.fail_logical_push_once = True
        with self.assertRaises(Exception):
            self.recover(reconciler)
        self.assertIsNotNone(self.receipts.inspect()["observed"])
        self.assertIsNone(self.receipts.inspect()["completed"])
        # Restore only acknowledged remote files into an entirely new cache,
        # modeling the loss of a Colab VM rather than deleting production data.
        fresh = Path(self.temp.name) / "fresh_campaign"
        for folder in ("events", "results", "sessions"):
            (fresh / folder).mkdir(parents=True)
        for relative, payload in self.mirror.remote.items():
            destination = fresh / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        self.directory = fresh
        self.clock = Clock()
        self.receipts = recovery.RecoveryReceiptStore(self.transport, self.auth)
        self.mirror.receipts = self.receipts
        answer = self.recover()
        self.assertFalse(answer["physical_reexecution_performed"])
        self.assertEqual(len(self.calls), 1)
        self.assertIsNotNone(self.receipts.inspect()["completed"])
        self.assert_preserved()

    def test_logical_acknowledgement_loss_does_not_repeat_the_fit(self):
        reconciler = self.prepare()
        self.mirror.lose_logical_ack_once = True
        with self.assertRaises(Exception):
            self.recover(reconciler)
        self.assertIsNotNone(self.receipts.inspect()["observed"])
        self.recover()
        self.assertEqual(len(self.calls), 1)
        self.assert_preserved()

    def test_absence_confirmation_is_required_before_any_physical_attempt(self):
        self.prepare()
        with self.assertRaises(recovery.ReconciliationError):
            recovery.ManualReconciler(self.view(self.auth), self.auth, self.mirror, self.receipts,
                authorization_sha256=engine.digest(self.auth), recovery_source_commit="f" * 40,
                physical_session_started_at=self.clock(), clock=self.clock)
        self.assertEqual(self.calls, [])
        self.assertEqual(public_records(self.directory), self.original)

    def test_corrupt_observed_receipt_is_rejected_without_a_fit_or_scientific_write(self):
        self.recover(self.prepare())
        before = public_records(self.directory)
        observed_path = next(path for path, payload in self.transport.files.items()
                             if json.loads(payload).get("phase") == "observed")
        changed = json.loads(self.transport.files[observed_path])
        changed["outcome"]["predictions_original"][0] = 99
        self.transport.files[observed_path] = json.dumps(changed).encode()
        with self.assertRaises(recovery.ReconciliationError):
            self.recover()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(public_records(self.directory), before)


if __name__ == "__main__":
    unittest.main()
