"""Durable scientific-search admission and restart contracts; no model fitting."""
from copy import deepcopy
import errno
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import e2_search_session as sessions


class Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def snapshot(directory):
    directory = Path(directory)
    if not directory.exists():
        return {}
    return {p.relative_to(directory).as_posix(): p.read_bytes()
            for p in sorted(directory.rglob("*")) if p.is_file() and not p.is_symlink()}


class SearchSessionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)/"campaign"
        self.clock = Clock()
        self.plan = {
            "version": "synthetic-session-contract-v1",
            "source_commit": "a"*40,
            "protocol_sha256": "b"*64,
            "device": "cuda:0",
            "domain": {"candidates": ["candidate-0", "candidate-1"], "seeds": [11, 12]},
            "scientific_market_fit": False,
        }

    def session(self, *, plan=None, session_budget=100, campaign_budget=1000,
                max_storage=8*1024**2, **kwargs):
        return sessions.SearchSession(
            self.directory, self.plan if plan is None else plan,
            session_budget_seconds=session_budget,
            campaign_budget_seconds=campaign_budget,
            max_storage_bytes=max_storage, clock=self.clock, **kwargs)

    def callback(self, seconds=1, outcome=None, calls=None):
        outcome = {"status": "completed", "score": 0.25, "tag": "OUTCOME42"} if outcome is None else outcome
        def invoke(deadline):
            if calls is not None:
                calls.append(deadline)
            self.assertGreaterEqual(deadline, self.clock())
            self.clock.advance(seconds)
            return deepcopy(outcome)
        return invoke

    def fit(self, session, *, name="candidate-0/seed11", payload=None, seconds=1,
            reservation=10, outcome=None, calls=None):
        payload = {"candidate": "candidate-0", "seed": 11} if payload is None else payload
        return session.run_job(name, payload, self.callback(seconds, outcome, calls),
                               reservation_seconds=reservation)

    def test_completed_result_is_not_repeated_after_restart(self):
        calls = []
        with self.session() as active:
            expected = self.fit(active, calls=calls)
        self.assertEqual(len(calls), 1)
        before = snapshot(self.directory)
        with self.session() as resumed:
            actual = self.fit(resumed, calls=calls)
        self.assertEqual(actual, expected)
        self.assertEqual(len(calls), 1)
        self.assertEqual(snapshot(self.directory), before)

    def test_cached_outcome_is_a_copy(self):
        with self.session() as active:
            returned = self.fit(active, outcome={"status": "completed", "nested": {"score": 7}})
            returned["nested"]["score"] = -999
            cached = self.fit(active, seconds=999, outcome={"replayed": True})
        self.assertEqual(cached, {"status": "completed", "nested": {"score": 7}})

    def test_plan_change_is_rejected_without_mutation(self):
        with self.session() as active:
            self.fit(active)
        changed = deepcopy(self.plan)
        changed["domain"]["seeds"].append(13)
        before = snapshot(self.directory)
        with self.assertRaises(sessions.InvalidPlan):
            self.session(plan=changed)
        self.assertEqual(snapshot(self.directory), before)

    def test_original_plan_input_is_not_retained_by_reference(self):
        submitted = deepcopy(self.plan)
        with self.session(plan=submitted) as active:
            submitted["domain"]["seeds"].append(999)
            self.fit(active)
        with self.session() as resumed:
            self.assertEqual(self.fit(resumed)["status"], "completed")

    def test_campaign_budget_change_requires_an_amendment(self):
        with self.session():
            pass
        before = snapshot(self.directory)
        with self.assertRaises(sessions.InvalidPlan):
            self.session(campaign_budget=2000)
        self.assertEqual(snapshot(self.directory), before)

    def test_storage_budget_change_is_rejected_without_mutation(self):
        with self.session():
            pass
        before = snapshot(self.directory)
        with self.assertRaises(sessions.InvalidPlan):
            self.session(max_storage=16*1024**2)
        self.assertEqual(snapshot(self.directory), before)

    def test_same_job_id_cannot_be_reused_for_changed_candidate(self):
        with self.session() as active:
            self.fit(active)
        before = snapshot(self.directory)
        with self.session() as resumed:
            with self.assertRaises(sessions.InvalidPlan):
                self.fit(resumed, payload={"candidate": "replacement", "seed": 99})
        self.assertEqual(snapshot(self.directory), before)

    def test_plan_must_be_strict_json_with_string_keys_and_finite_numbers(self):
        for invalid in ({1: "not a string key"}, {"value": float("nan")},
                        {"value": float("inf")}, {"value": Path("not-json")}):
            with self.subTest(invalid=repr(invalid)):
                with self.assertRaises((sessions.InvalidPlan, TypeError, ValueError)):
                    self.session(plan=invalid)
                self.assertFalse(self.directory.exists())

    def test_invalid_budget_values_are_rejected_before_creating_campaign(self):
        for keyword, value in (("session_budget", 0), ("session_budget", -1),
                               ("campaign_budget", float("inf")),
                               ("campaign_budget", True), ("max_storage", -1),
                               ("max_storage", True)):
            with self.subTest(keyword=keyword, value=value):
                with self.assertRaises((sessions.InvalidPlan, TypeError, ValueError)):
                    self.session(**{keyword: value})
                self.assertFalse(self.directory.exists())

    def test_session_reserves_complete_job_before_callback(self):
        calls = []
        with self.session(session_budget=10) as active:
            with self.assertRaises(sessions.SessionPaused):
                self.fit(active, reservation=11, calls=calls)
        self.assertEqual(calls, [])

    def test_campaign_charge_survives_restart_and_blocks_unaffordable_job(self):
        calls = []
        with self.session(session_budget=10, campaign_budget=10) as active:
            self.fit(active, seconds=6, reservation=6, calls=calls)
        with self.session(session_budget=10, campaign_budget=10) as resumed:
            with self.assertRaises(sessions.SessionPaused):
                self.fit(resumed, name="candidate-1/seed11", payload={"candidate": "candidate-1", "seed": 11},
                         reservation=5, calls=calls)
        self.assertEqual(len(calls), 1)

    def test_campaign_budget_can_be_extended_explicitly_and_survives_restart(self):
        with self.session(session_budget=10, campaign_budget=10) as active:
            self.fit(active, seconds=6, reservation=6)
        before_config = (self.directory/"campaign.json").read_bytes()
        with self.session(session_budget=10, campaign_budget=10) as resumed:
            resumed.extend_campaign_budget(20, reason="Explicit additional session allocation")
            actual = self.fit(resumed, name="candidate-1/seed11",
                              payload={"candidate": "candidate-1", "seed": 11}, seconds=5, reservation=5)
        self.assertEqual(actual["status"], "completed")
        self.assertEqual((self.directory/"campaign.json").read_bytes(), before_config)
        with self.session(session_budget=10, campaign_budget=10) as again:
            self.fit(again, name="candidate-1/seed12", payload={"candidate": "candidate-1", "seed": 12},
                     seconds=2, reservation=2)

    def test_budget_extension_requires_strict_increase_and_reason(self):
        with self.session() as active:
            for value, reason in ((1000, "same"), (999, "decrease"), (float("inf"), "invalid"),
                                  (2000, ""), (2000, "   ")):
                before = snapshot(self.directory)
                with self.subTest(value=value, reason=reason):
                    with self.assertRaises((sessions.InvalidPlan, ValueError)):
                        active.extend_campaign_budget(value, reason=reason)
                    self.assertEqual(snapshot(self.directory), before)

    def test_campaign_lock_excludes_second_writer_and_is_released(self):
        first = self.session()
        try:
            with self.assertRaises(sessions.CampaignLocked):
                self.session()
        finally:
            first.close()
        with self.session():
            pass

    def test_unsupported_filesystem_lock_is_reported(self):
        with mock.patch.object(sessions.fcntl, "flock", side_effect=OSError(errno.ENOTSUP, "unsupported")):
            with self.assertRaises(sessions.LockUnavailable):
                self.session()

    def test_callback_exception_blocks_automatic_retry_of_any_job(self):
        calls = []
        def interrupted(deadline):
            calls.append(deadline)
            self.clock.advance(3)
            raise RuntimeError("Synthetic interruption")
        with self.assertRaises(sessions.UnresolvedAttempt):
            with self.session() as active:
                active.run_job("candidate-0/seed11", {"candidate": "candidate-0", "seed": 11},
                               interrupted, reservation_seconds=10)
        with self.assertRaises(sessions.UnresolvedAttempt):
            with self.session() as resumed:
                self.fit(resumed, name="candidate-1/seed12", calls=calls)
        self.assertEqual(len(calls), 1)

    def test_completed_scientific_failure_is_charged_and_not_retried(self):
        calls = []
        failure = {"status": "failed", "reason": "resource_limit", "optimizer_steps": None,
                   "optimizer_steps_known_lower_bound": 2}
        with self.session() as active:
            expected = self.fit(active, outcome=failure, calls=calls)
        with self.session() as resumed:
            actual = self.fit(resumed, calls=calls)
        self.assertEqual(actual, expected)
        self.assertIsNone(actual["optimizer_steps"])
        self.assertEqual(actual["optimizer_steps_known_lower_bound"], 2)
        self.assertEqual(len(calls), 1)

    def test_tampered_result_cannot_be_loaded_as_a_completed_trial(self):
        with self.session() as active:
            self.fit(active)
        selected = [p for p in self.directory.rglob("*.json") if b"OUTCOME42" in p.read_bytes()]
        self.assertEqual(len(selected), 1)
        selected[0].write_bytes(selected[0].read_bytes().replace(b"OUTCOME42", b"OUTCOME43"))
        with self.assertRaises(sessions.InvalidPlan):
            with self.session() as resumed:
                self.fit(resumed)

    def test_symlink_campaign_root_is_rejected(self):
        with self.session():
            pass
        alias = self.directory.parent/"alias"
        alias.symlink_to(self.directory, target_is_directory=True)
        before = snapshot(self.directory)
        with self.assertRaises(sessions.InvalidPlan):
            sessions.SearchSession(alias, self.plan, session_budget_seconds=100,
                                   campaign_budget_seconds=1000, max_storage_bytes=8*1024**2,
                                   clock=self.clock)
        self.assertEqual(snapshot(self.directory), before)

    def test_result_commit_reserve_blocks_callback_before_storage_is_exhausted(self):
        calls = []
        with self.session(max_storage=2*1024**2) as active:
            (self.directory/"scratch.bin").write_bytes(b"x"*(1536*1024))
            with self.assertRaises(sessions.SessionPaused):
                self.fit(active, calls=calls)
        self.assertEqual(calls, [])

    def test_deadline_is_bounded_by_reserved_job_time(self):
        calls = []
        before = self.clock()
        with self.session(session_budget=100) as active:
            self.fit(active, reservation=9, seconds=2, calls=calls)
        self.assertEqual(len(calls), 1)
        self.assertLessEqual(calls[0], before+9)

    def test_reservation_overrun_keeps_completed_outcome_and_pauses(self):
        calls = []
        with self.session(session_budget=100) as active:
            with self.assertRaises(sessions.SessionPaused) as caught:
                self.fit(active, seconds=11, reservation=10, calls=calls)
            self.assertEqual(caught.exception.reason, "job_reservation_overrun")
        with self.session(session_budget=100) as resumed:
            actual = self.fit(resumed, calls=calls)
        self.assertEqual(actual["status"], "completed")
        self.assertEqual(len(calls), 1)

    def test_committed_outcome_survives_missing_finished_event_without_refit(self):
        calls = []
        active = self.session()
        original = active._append_event
        def fail_finished_event(kind, **fields):
            if kind == "job_finished":
                raise OSError("Synthetic loss after result publication")
            return original(kind, **fields)
        try:
            with mock.patch.object(active, "_append_event", side_effect=fail_finished_event):
                with self.assertRaises(OSError):
                    self.fit(active, calls=calls)
        finally:
            # Simulate process loss after a committed result, before any clean
            # session-close record. Only release the OS lock for this test.
            active._closed = True
            active._release_lock()
        before = snapshot(self.directory)
        with self.session() as resumed:
            actual = self.fit(resumed, calls=calls)
            audit = resumed.summary()
            self.assertIn("candidate-0/seed11", audit["recovered_result_commits"])
            self.assertEqual(audit["charged_wall_seconds"], 100)
        self.assertEqual(actual["status"], "completed")
        self.assertEqual(len(calls), 1)
        self.assertEqual(snapshot(self.directory), before)

    def test_non_json_callback_result_keeps_unresolved_attempt(self):
        calls = []
        with self.assertRaises(sessions.UnresolvedAttempt):
            with self.session() as active:
                self.fit(active, outcome={"score": float("nan")}, calls=calls)
        with self.assertRaises(sessions.UnresolvedAttempt):
            self.session()
        self.assertEqual(len(calls), 1)

    def test_oversized_result_commit_does_not_become_a_retriable_job(self):
        calls = []
        with self.assertRaises(sessions.UnresolvedAttempt):
            with self.session(max_storage=2*1024**2) as active:
                self.fit(active, outcome={"oversized": "x"*(3*1024**2)}, calls=calls)
        with self.assertRaises(sessions.UnresolvedAttempt):
            self.session(max_storage=2*1024**2)
        self.assertEqual(len(calls), 1)

    def test_pending_jobs_block_stage_completion(self):
        with self.session() as active:
            self.fit(active)
            active.require_complete(["candidate-0/seed11"])
            with self.assertRaises(sessions.IncompleteCampaign):
                active.require_complete(["candidate-0/seed11", "candidate-1/seed12"])
            self.assertEqual(active.pending_jobs(["candidate-0/seed11", "candidate-1/seed12"]),
                             ("candidate-1/seed12",))

    def test_unsupported_directory_fsync_requires_explicit_relaxed_policy(self):
        original = sessions.os.fsync
        def fsync_files_only(descriptor):
            if stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise OSError(errno.EINVAL, "Synthetic FUSE directory fsync unsupported")
            return original(descriptor)
        with mock.patch.object(sessions.os, "fsync", side_effect=fsync_files_only):
            with self.assertRaises(sessions.LockUnavailable):
                self.session()
            self.assertFalse((self.directory/"campaign.json").exists())
            with self.session(require_directory_fsync=False) as recovered:
                actual = self.fit(recovered)
            self.assertEqual(actual["status"], "completed")

    def test_relaxed_directory_fsync_is_recorded_and_cannot_be_silently_changed(self):
        original = sessions.os.fsync
        def fsync_files_only(descriptor):
            if stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise OSError(errno.EINVAL, "Synthetic FUSE directory fsync unsupported")
            return original(descriptor)
        with mock.patch.object(sessions.os, "fsync", side_effect=fsync_files_only):
            with self.session(require_directory_fsync=False) as active:
                actual = self.fit(active)
                self.assertFalse(active.summary()["directory_fsync_supported"])
        self.assertEqual(actual["status"], "completed")
        sealed = json.loads((self.directory/"campaign.json").read_text())
        self.assertIs(sealed["controls"]["require_directory_fsync"], False)
        before = snapshot(self.directory)
        with self.assertRaises(sessions.InvalidPlan):
            self.session(require_directory_fsync=True)
        self.assertEqual(snapshot(self.directory), before)

    def test_sync_acknowledgements_are_incremental_and_restart_reseeds_records(self):
        with self.session() as active:
            initial = active.pending_record_paths
            self.assertEqual(initial, ("campaign.json",))
            before = snapshot(self.directory)
            active.mark_records_synced(initial)
            self.assertEqual(active.pending_record_paths, ())
            self.assertEqual(snapshot(self.directory), before)
            self.fit(active)
            pending = active.pending_record_paths
            self.assertEqual(pending, tuple(sorted(pending)))
            self.assertTrue(pending)
            self.assertNotIn("campaign.json", pending)
            self.assertTrue(all((self.directory/path).is_file() and path.endswith(".json")
                                for path in pending))
            committed = snapshot(self.directory)
            active.mark_records_synced(pending)
            self.assertEqual(active.pending_record_paths, ())
            self.assertEqual(snapshot(self.directory), committed)
            self.fit(active)
            self.assertEqual(active.pending_record_paths, ())
        expected = tuple(sorted(p.relative_to(self.directory).as_posix()
                                for p in self.directory.rglob("*.json")))
        with self.session() as resumed:
            self.assertEqual(resumed.pending_record_paths, expected)

    def test_sync_acknowledgement_rejects_unknown_records_atomically(self):
        with self.session() as active:
            before = snapshot(self.directory)
            pending = active.pending_record_paths
            for paths in (["campaign.json", "results/unknown.json"], "campaign.json", [Path("campaign.json")]):
                with self.subTest(paths=repr(paths)):
                    with self.assertRaises(sessions.InvalidPlan):
                        active.mark_records_synced(paths)
                    self.assertEqual(active.pending_record_paths, pending)
                    self.assertEqual(snapshot(self.directory), before)

    def test_cached_attempt_cannot_change_its_scientific_reservation(self):
        calls = []
        with self.session() as active:
            self.fit(active, reservation=10, calls=calls)
        before = snapshot(self.directory)
        with self.session() as resumed:
            with self.assertRaises(sessions.InvalidPlan):
                self.fit(resumed, reservation=20, calls=calls)
        self.assertEqual(snapshot(self.directory), before)
        self.assertEqual(len(calls), 1)

    def test_budget_amendment_can_resume_campaign_pause_in_same_instance(self):
        calls = []
        with self.session(session_budget=20, campaign_budget=5) as active:
            with self.assertRaises(sessions.SessionPaused) as caught:
                self.fit(active, reservation=6, calls=calls)
            self.assertEqual(caught.exception.reason, "campaign_budget")
            self.assertEqual(calls, [])
            active.extend_campaign_budget(10, reason="Explicit extra allocation before fitting")
            actual = self.fit(active, reservation=6, seconds=3, calls=calls)
        self.assertEqual(actual["status"], "completed")
        self.assertEqual(len(calls), 1)

    def test_expired_reservation_after_start_journal_prevents_callback_launch(self):
        calls = []
        with self.assertRaises(sessions.UnresolvedAttempt):
            with self.session() as active:
                original = active._append_event
                def delayed_journal(kind, **fields):
                    record = original(kind, **fields)
                    if kind == "job_started":
                        self.clock.advance(11)
                    return record
                with mock.patch.object(active, "_append_event", side_effect=delayed_journal):
                    self.fit(active, reservation=10, calls=calls)
        self.assertEqual(calls, [])
        with self.assertRaises(sessions.UnresolvedAttempt):
            self.session()


if __name__ == "__main__":
    unittest.main()
