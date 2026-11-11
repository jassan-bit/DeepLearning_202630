"""Manually authorized recovery of one unknown physical search attempt.

Fixtures use the frozen session engine and the frozen public mirror validator.
No GPU, market fit, credential, or external GitHub transport is used.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import e2_github_campaign as github
import e2_manual_reconciliation_v4 as recovery
import e2_recovery_receipts_v4 as physical_receipts
import e2_search_session as engine


OUTCOME = {"status":"completed", "run_id":"c"*64,
    "resource_use":{"optimizer_steps":4,"optimizer_steps_known_lower_bound":4,"phase_records":{},
        "optimizer_steps_unknown":False,"partial_optimizer_update_may_have_occurred":False,
        "checkpoint_policy":"selection_metadata_only","cuda_metrics":{}},
    "predictions_original":[.1,.2],
    "prediction_dates":["2024-12-23T00:00:00+00:00","2024-12-24T00:00:00+00:00"],
    "phase_epochs":{"supervised":2},"reason":None,"fatal":False}


class Clock:
    def __init__(self):self.value=1000.0
    def __call__(self):return self.value
    def advance(self,seconds):self.value+=seconds


def records(directory):
    return {p.relative_to(directory).as_posix():p.read_bytes()
            for p in sorted(Path(directory).rglob("*.json"))}


class FakeMirror:
    """A CAS acknowledgement stub that still validates every public record."""
    def __init__(self,directory,auth,order):
        self.directory=Path(directory)
        self.auth=auth
        self.repository=auth["repository"]
        self.branch=auth["campaign_branch"]
        self.prefix=auth["campaign_prefix"]
        self.recovery_source_commit="f"*40
        self._head=auth["sealed_campaign_head"]
        self.remote=records(self.directory)
        self.original=dict(self.remote)
        self.order=order
        self.pushes=[]
        self.fail_before_commit=False
        self.lose_ack=False
        self.receipts=None

    def inspect_remote(self):
        self.order.append("campaign_inspected")
        return {"plan_sha256":self.auth["plan_sha256"],"record_count":len(self.remote)}

    def push(self,directory,*,paths=None,message="manual recovery"):
        directory=Path(directory)
        local=records(directory)
        if any(local.get(path)!=payload for path,payload in self.original.items()):
            raise AssertionError("Recovery modified original scientific bytes")
        chosen=sorted(local) if paths is None else list(paths)
        for path in chosen:github.validate_record(path,local[path])
        target="results/"+hashlib.sha256(self.auth["target"]["job_id"].encode()).hexdigest()+".json"
        publishes_result=target in local and target not in self.original
        if publishes_result:
            if not self.receipts or self.receipts.state["observed"] is None:
                raise AssertionError("A logical result was published before an observed physical receipt")
        self.order.append("campaign_push_attempt")
        if self.fail_before_commit and publishes_result:
            self.fail_before_commit=False
            raise github.CampaignSyncError("synthetic_campaign_push_failure")
        changed={path:local[path] for path in chosen if self.remote.get(path)!=local[path]}
        self.remote.update(changed)
        self.pushes.append(tuple(chosen))
        if changed:
            self._head=hashlib.sha1((self._head+str(len(self.pushes))).encode()).hexdigest()
        self.order.append("campaign_committed")
        if target in changed:self.order.append("campaign_result_committed")
        if self.lose_ack and publishes_result:
            self.lose_ack=False
            raise github.CampaignSyncError("synthetic_commit_ack_lost")
        return {"status":"synced" if changed else "unchanged","remote_commit":self._head,
                "synced_paths":tuple(chosen)}


class FakeReceipts:
    """Immutable receipts with explicit phase acknowledgement ordering."""
    def __init__(self,directory,auth,authorization_sha256,order):
        self.directory=Path(directory)
        self.auth=auth
        self.authorization_sha256=authorization_sha256
        self.recovery_source_commit="f"*40
        self.order=order
        self.state={"started":None,"observed":None,"completed":None}
        self.calls=[]
        self.lose_observed_ack=False

    def inspect(self):
        self.order.append("receipts_inspected")
        physical_receipts.RecoveryReceiptStore._links(self.state)
        return deepcopy(self.state)

    def _save(self,phase,record):
        record=deepcopy(record)
        record.setdefault("phase",phase)
        record.setdefault("authorization_sha256",self.authorization_sha256)
        check={k:v for k,v in record.items() if k!="sha256"}
        record["sha256"]=engine.digest(check)
        # R4 binds a verified authorization instead of magic incident constants.
        physical_receipts.validate_receipt(record,phase,self.authorization_sha256,self.auth)
        physical_receipts.RecoveryReceiptStore._links({**self.state,phase:record})
        previous=self.state[phase]
        if previous is not None and previous!=record:
            raise RuntimeError("Synthetic immutable receipt conflict")
        self.state[phase]=record
        self.calls.append(phase)
        self.order.append("receipt_"+phase+"_committed")
        return deepcopy(record)

    def claim_started(self,record):
        if self.state["started"] is not None:
            raise RuntimeError("Synthetic started receipt already exists")
        return self._save("started",record)

    def save_observed(self,record):
        target="results/"+hashlib.sha256(self.auth["target"]["job_id"].encode()).hexdigest()+".json"
        if self.state["observed"] is None and (self.directory/target).exists():
            raise AssertionError("Logical result exists before observed receipt acknowledgement")
        for p in (self.directory/"events").glob("*.json"):
            event=json.loads(p.read_text())
            if self.state["observed"] is None and event.get("kind")=="job_finished" and event.get("job_id")==self.auth["target"]["job_id"]:
                raise AssertionError("Logical finish exists before observed receipt acknowledgement")
        saved=self._save("observed",record)
        if self.lose_observed_ack:
            self.lose_observed_ack=False
            raise github.CampaignSyncError("synthetic_observed_ack_lost")
        self.order.append("observed_ack_returned")
        return saved

    def save_completed(self,record):
        if self.state["observed"] is None:
            raise AssertionError("Completion without an observed physical outcome")
        return self._save("completed",record)


class ManualReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory=Path(self.temporary.name)/"campaign"
        self.clock=Clock()
        self.plan={"version":"e2-cuda-search-plan-v1","source_commit":"a"*40,
            "protocol_sha256":"b"*64,"registration_sha256":"d"*64,
            "models":["GRU32"],"origins":["2025-01-01"],
            "checkpoint_policy":"selection_metadata_only_refit_separate"}
        # The production receipt validator keeps the frozen scientific bindings.
        # Only those three constants are substituted for this synthetic campaign.
        bindings=mock.patch.multiple(physical_receipts,PLAN_SHA256=engine.digest(self.plan),
            SOURCE_COMMIT=self.plan["source_commit"],REGISTRATION_SHA256=self.plan["registration_sha256"])
        bindings.start()
        self.addCleanup(bindings.stop)
        self.payload={"candidate":"pending-hpo-candidate","seed":17}
        self.job_id=engine.digest({"plan_sha256":engine.digest(self.plan),"job":self.payload})
        self.finished_job=engine.digest({"logical_job":"already-completed"})
        session=self.open_original()
        session.run_job(self.finished_job,{"candidate":"completed","seed":11},
                        lambda deadline:self.original_result(),reservation_seconds=30)
        start=session._append_event("job_started",job_id=self.job_id,session_id=session._session_id,
            job_payload_sha256=engine.digest(self.payload),reservation_seconds=5700)
        session._closed=True
        session._release_lock()
        self.bind_authorization(start,completed_count=1)

    def bind_authorization(self,start,*,completed_count):
        self.original_records=records(self.directory)
        # Keep fixtures independent of a particular incident or registration file.
        self.auth={"version":"e2-manual-recovery-authorization-v4",
            "registered_at_utc":"2026-11-10T17:00:00+00:00",
            "registered_at_bogota":"2026-11-10T12:00:00-05:00",
            "repository":physical_receipts.REPOSITORY,"campaign_branch":physical_receipts.CAMPAIGN_BRANCH,
            "campaign_prefix":physical_receipts.CAMPAIGN_BRANCH,
            "receipt_branch":"recoveries/e2-cuda-selection-v1/20261111-r4",
            "original_unknown_attempt":deepcopy(physical_receipts.UNKNOWN),
            "decision":"manual_retry_same_sealed_logical_job_once_after_confirmed_executor_and_outcome_absent",
            "evidence":"Synthetic diagnostic confirms no running executor or target outcome; prior results exist.",
            "limits":{"session_seconds":28800,"initial_campaign_seconds":86400,"campaign_seconds":172800,
                "budget_extension_event_sha256":self.extension_event_sha,
                "old_session_conservative_allocation_seconds":28800,"budget_extension_authorized":False},
            "ledger_interpretation":"New physical retry resolves the sealed logical job; old physical costs remain unknown."}
        self.auth.update(status="registered_before_recovery_execution",previous_completed_jobs=completed_count,
            sealed_campaign_head="1"*40,plan_sha256=engine.digest(self.plan),
            scientific_source_commit=self.plan["source_commit"],
            scientific_registration_sha256=self.plan["registration_sha256"])
        self.auth["implementation_sha256"]={name:"a"*64 for name in physical_receipts.IMPLEMENTATION_PATHS}
        prior_events=[json.loads(path.read_text()) for path in (self.directory/"events").glob("*.json")]
        extension=next(event for event in reversed(sorted(prior_events,key=lambda e:e["sequence"])) if event.get("kind")=="campaign_budget_extended")
        self.auth["limits"]["budget_extension_event_sha256"]=extension["sha256"]
        # A synthetic real extension has its own authentic hash. Production
        # bindings remain fixed to the already published 36-to-48-hour amendment.
        for module in (recovery,physical_receipts):
            extension_binding=mock.patch.object(module,"BUDGET_EXTENSION_EVENT_SHA256",extension["sha256"])
            extension_binding.start()
            self.addCleanup(extension_binding.stop)
        old_start=next(event for event in prior_events if event.get("kind")=="session_started"
            and event.get("session_id")==start["session_id"])
        self.auth["limits"]["old_session_conservative_allocation_seconds"]=old_start["session_budget_seconds"]
        self.auth["target"]={"job_id":self.job_id,"job_payload_sha256":engine.digest(self.payload),
            "started_event_sha256":start["sha256"],"sequence":start["sequence"],
            "old_session_id":start["session_id"],"reservation_seconds":5700}
        self.authorization_sha256=engine.digest(self.auth)
        self.order=[]
        self.mirror=FakeMirror(self.directory,self.auth,self.order)
        self.receipts=FakeReceipts(self.directory,self.auth,self.authorization_sha256,self.order)
        self.mirror.receipts=self.receipts
        self.callback_calls=[]
        self.physical_start=self.clock()

    def original_result(self):
        self.clock.advance(2)
        return deepcopy(OUTCOME)

    def open_original(self):
        session=engine.SearchSession(self.directory,self.plan,session_budget_seconds=28800,
            campaign_budget_seconds=86400,max_storage_bytes=10*1024**3,clock=self.clock)
        if session.campaign_budget_seconds==86400:
            session.extend_campaign_budget(129600, reason="operator_requested_budget_extension")
            session.extend_campaign_budget(172800,
                reason=getattr(self,"fixture_extension_reason","operator_requested_budget_extension"))
            self.extension_event_sha=session._events[-1]["sha256"]
        return session

    def view(self,auth=None,plan=None):
        return recovery.read_snapshot(self.directory,self.plan if plan is None else plan,
                                      self.auth if auth is None else auth,session_class=engine.SearchSession)

    def reconciler(self,*,auth=None,deadline=None):
        authorization=self.auth if auth is None else auth
        return recovery.ManualReconciler(self.view(auth=authorization),authorization,self.mirror,self.receipts,
            authorization_sha256=self.authorization_sha256,physical_session_started_at=self.physical_start,
            recovery_source_commit="f"*40,
            physical_deadline=self.physical_start+28800 if deadline is None else deadline,clock=self.clock)

    def callback(self,deadline):
        self.callback_calls.append(deadline)
        self.order.append("physical_fit_started")
        self.assertIsNotNone(self.receipts.state["started"])
        closures=[json.loads(p.read_text()) for p in (self.directory/"events").glob("*.json")]
        self.assertTrue(any(e.get("kind")=="session_closed" and e.get("session_id")==self.auth["target"]["old_session_id"]
                            for e in closures),"Unknown old session must be closed conservatively before retry")
        self.clock.advance(5)
        return deepcopy(OUTCOME)

    def recover(self,instance=None,**kwargs):
        return (self.reconciler() if instance is None else instance).recover(
            kwargs.pop("job_id",self.job_id),kwargs.pop("payload",self.payload),
            kwargs.pop("callback",self.callback),reservation_seconds=kwargs.pop("reservation_seconds",5700),**kwargs)

    def assert_old_bytes_preserved(self):
        for path,payload in self.original_records.items():
            self.assertEqual((self.directory/path).read_bytes(),payload,path)

    def test_snapshot_read_is_read_only_and_reports_one_unknown_target(self):
        before=records(self.directory)
        self.view()
        self.assertEqual(records(self.directory),before)
        self.assertEqual(self.callback_calls,[])

    def test_existing_extension_preserves_initial_controls_and_changes_only_effective_cap(self):
        before=records(self.directory)
        view=self.view()
        self.assertEqual(view.controls["campaign_budget_seconds"],86400)
        self.assertEqual(view._loader.campaign_budget_seconds,172800)
        matching=[event for event in view.events if event.get("kind")=="campaign_budget_extended"]
        self.assertEqual(len(matching),2)
        self.assertEqual(matching[-1]["sha256"],self.auth["limits"]["budget_extension_event_sha256"])
        self.assertEqual(records(self.directory),before)
        self.recover()
        with self.open_original() as resumed:
            self.assertEqual(resumed.initial_campaign_budget_seconds,86400)
            self.assertEqual(resumed.campaign_budget_seconds,172800)
            self.assertEqual(sum(event["kind"]=="campaign_budget_extended" for event in resumed._events),2)
        self.assert_old_bytes_preserved()

    def test_authorization_cannot_replace_initial_cap_effective_cap_or_existing_extension(self):
        before=records(self.directory)
        for field,value in (("initial_campaign_seconds",172800),("campaign_seconds",86400),
                            ("campaign_seconds",216000),("budget_extension_event_sha256","0"*64),
                            ("budget_extension_authorized",True)):
            changed=deepcopy(self.auth)
            changed["limits"][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(recovery.ReconciliationBlocked):
                self.view(auth=changed)
            self.assertEqual(records(self.directory),before)
            self.assertEqual(self.callback_calls,[])

    def test_registered_extension_hash_must_exist_in_verified_event_prefix(self):
        changed=deepcopy(self.auth)
        changed["limits"]["budget_extension_event_sha256"]="e"*64
        before=records(self.directory)
        with mock.patch.object(recovery,"BUDGET_EXTENSION_EVENT_SHA256","e"*64):
            with self.assertRaisesRegex(recovery.ReconciliationBlocked,"sealed_budget_extension_changed"):
                self.view(auth=changed)
        self.assertEqual(records(self.directory),before)
        self.assertEqual(self.callback_calls,[])

    def test_extension_must_have_the_recorded_operator_reason(self):
        self.fixture_extension_reason="unregistered_extra_compute"
        self.campaign_with_prior_closed_sessions([])
        before=records(self.directory)
        with self.assertRaisesRegex(recovery.ReconciliationBlocked,"sealed_budget_extension_changed"):
            self.view()
        self.assertEqual(records(self.directory),before)
        self.assertEqual(self.callback_calls,[])

    def test_manual_retry_preserves_history_and_charges_old_allocation_plus_new_fit(self):
        authorization_before=deepcopy(self.auth)
        answer=self.recover()
        self.assertEqual(answer["status"],"logical_job_reconciled")
        self.assertEqual(answer["outcome"],OUTCOME)
        self.assertEqual(len(self.callback_calls),1)
        self.assert_old_bytes_preserved()
        self.assertIsNotNone(self.receipts.state["completed"])
        self.assertEqual(self.auth,authorization_before)
        self.assertIsNone(self.auth["original_unknown_attempt"]["optimizer_steps"])
        self.assertIsNone(self.auth["original_unknown_attempt"]["elapsed_seconds"])
        with self.open_original() as resumed:
            self.assertEqual(len(resumed.completed_job_ids),2)
            self.assertEqual(resumed.charged_wall_seconds,28805)
            self.assertEqual(resumed.get_result(self.job_id,self.payload),OUTCOME)

    def test_completed_authorized_recovery_is_never_physically_replayed(self):
        self.recover()
        answer=self.recover()
        self.assertEqual(answer["status"],"already_reconciled")
        self.assertEqual(answer["outcome"],OUTCOME)
        self.assertEqual(len(self.callback_calls),1)
        self.assert_old_bytes_preserved()

    def test_new_physical_overhead_is_charged_once_without_charging_unknown_old_time_twice(self):
        claim=self.receipts.claim_started
        observe=self.receipts.save_observed
        push=self.mirror.push
        def claim_with_elapsed_ack(record):
            result=claim(record)
            self.clock.advance(1)
            return result
        def observed_with_elapsed_ack(record):
            result=observe(record)
            self.clock.advance(3)
            return result
        def marker_push_with_elapsed_ack(directory,**kwargs):
            before_callback=not self.callback_calls
            result=push(directory,**kwargs)
            if before_callback:self.clock.advance(2)
            return result
        with mock.patch.object(self.receipts,"claim_started",side_effect=claim_with_elapsed_ack),\
             mock.patch.object(self.receipts,"save_observed",side_effect=observed_with_elapsed_ack),\
             mock.patch.object(self.mirror,"push",side_effect=marker_push_with_elapsed_ack):
            answer=self.recover()
        self.assertEqual(answer["actual_fit_wall_seconds"],5)
        self.assertEqual(answer["ledger_accounted_recovery_wall_seconds"],11)
        self.assertEqual(answer["campaign_charged_seconds"],28811)
        with self.open_original() as resumed:self.assertEqual(resumed.charged_wall_seconds,28811)
        self.assert_old_bytes_preserved()

    def test_different_logical_job_or_payload_is_rejected_before_fitting(self):
        for kwargs in ({"job_id":self.finished_job},{"payload":{"candidate":"replacement","seed":17}}):
            before=records(self.directory)
            with self.subTest(kwargs=kwargs),self.assertRaises(recovery.ReconciliationError):
                self.recover(**kwargs)
            self.assertEqual(records(self.directory),before)
            self.assertEqual(self.callback_calls,[])

    def test_changed_plan_or_started_binding_is_rejected_without_fitting(self):
        for field in ("plan_sha256","started_event_sha256"):
            changed=deepcopy(self.auth)
            if field=="plan_sha256":changed[field]="e"*64
            else:changed["target"][field]="e"*64
            before=records(self.directory)
            with self.subTest(field=field),self.assertRaises(recovery.ReconciliationError):
                self.view(auth=changed)
            self.assertEqual(records(self.directory),before)
            self.assertEqual(self.callback_calls,[])

    def test_old_cache_absence_decision_cannot_authorize_this_incident(self):
        changed=deepcopy(self.auth)
        changed["decision"]="manual_retry_same_sealed_logical_job_once_after_colab_local_cache_absent"
        before=records(self.directory)
        with self.assertRaisesRegex(recovery.ReconciliationBlocked,"manual_retry_policy_changed"):
            self.view(auth=changed)
        self.assertEqual(records(self.directory),before)
        self.assertEqual(self.callback_calls,[])

    def test_remote_head_conflict_blocks_new_physical_execution(self):
        self.mirror._head="f"*40
        with self.assertRaises(recovery.ReconciliationError):self.recover()
        self.assertEqual(self.callback_calls,[])
        self.assertIsNone(self.receipts.state["started"])
        self.assert_old_bytes_preserved()

    def test_physical_receipt_is_observed_before_logical_result_publication(self):
        self.recover()
        self.assertLess(self.order.index("receipt_started_committed"),self.order.index("physical_fit_started"))
        self.assertLess(self.order.index("observed_ack_returned"),self.order.index("campaign_result_committed"))
        self.assertLess(self.order.index("campaign_result_committed"),self.order.index("receipt_completed_committed"))

    def test_known_observation_recovers_after_logical_push_failure_without_refit(self):
        self.mirror.fail_before_commit=True
        with self.assertRaises(Exception):self.recover()
        self.assertIsNotNone(self.receipts.state["observed"])
        self.assertIsNone(self.receipts.state["completed"])
        self.recover()
        self.assertEqual(len(self.callback_calls),1)
        self.assertIsNotNone(self.receipts.state["completed"])
        self.assert_old_bytes_preserved()

    def test_lost_logical_push_acknowledgement_is_resolved_without_refit(self):
        self.mirror.lose_ack=True
        with self.assertRaises(Exception):self.recover()
        self.assertIsNotNone(self.receipts.state["observed"])
        self.recover()
        self.assertEqual(len(self.callback_calls),1)
        self.assertIsNotNone(self.receipts.state["completed"])
        self.assert_old_bytes_preserved()

    def test_lost_observation_acknowledgement_cannot_trigger_another_physical_fit(self):
        self.receipts.lose_observed_ack=True
        with self.assertRaises(Exception):self.recover()
        self.assertIsNotNone(self.receipts.state["observed"])
        self.recover()
        self.assertEqual(len(self.callback_calls),1)
        self.assertIsNotNone(self.receipts.state["completed"])
        self.assert_old_bytes_preserved()

    def test_fresh_process_uses_remote_observation_after_local_result_was_lost(self):
        def first_physical_fit(deadline):
            self.callback_calls.append(deadline)
            self.clock.advance(20)
            return deepcopy(OUTCOME)
        self.mirror.fail_before_commit=True
        with self.assertRaises(github.CampaignSyncError):
            self.recover(callback=first_physical_fit)
        self.assertIsNotNone(self.receipts.state["observed"])
        self.assertEqual(self.receipts.state["observed"]["actual_fit_wall_seconds"],20)
        self.assertIsNone(self.receipts.state["completed"])
        # A new Colab runtime sees only acknowledged campaign files. Discard
        # solely the unacknowledged files created by this test's recovery; all
        # original scientific bytes remain untouched.
        remote=deepcopy(self.mirror.remote)
        for relative in set(records(self.directory))-set(remote):
            self.assertNotIn(relative,self.original_records)
            (self.directory/relative).unlink()
        for relative,payload in remote.items():
            path=self.directory/relative
            path.parent.mkdir(parents=True,exist_ok=True)
            if relative in self.original_records:
                self.assertEqual(path.read_bytes(),payload)
            else:path.write_bytes(payload)
        self.assert_old_bytes_preserved()
        self.clock=Clock()
        self.physical_start=self.clock()
        push=self.mirror.push
        def repair_ack_elapsed(directory,**kwargs):
            answer=push(directory,**kwargs)
            self.clock.advance(2)
            return answer
        with mock.patch.object(self.mirror,"push",side_effect=repair_ack_elapsed):
            answer=self.recover()
        self.assertFalse(answer["physical_reexecution_performed"])
        self.assertEqual(len(self.callback_calls),1)
        self.assertEqual(answer["actual_fit_wall_seconds"],20)
        self.assertEqual(answer["ledger_accounted_recovery_wall_seconds"],28800)
        self.assertEqual(answer["campaign_charged_seconds"],57600)
        completed=self.receipts.state["completed"]
        self.assertGreaterEqual(completed["recovery_elapsed_seconds"],
                                self.receipts.state["observed"]["recovery_elapsed_seconds"])
        self.assertTrue(completed["recovery_elapsed_lower_bound_only"])
        self.assertEqual(completed["current_repair_elapsed_seconds"],2)
        with self.open_original() as resumed:
            self.assertEqual(resumed.get_result(self.job_id,self.payload),OUTCOME)
            self.assertEqual(resumed.charged_wall_seconds,57600)
        self.assert_old_bytes_preserved()

    def test_completed_receipt_cannot_bind_a_different_result_hash(self):
        self.recover()
        forged=self.receipts.state["completed"]
        forged["result_sha256"]="0"*64
        forged["sha256"]=engine.digest({key:value for key,value in forged.items() if key!="sha256"})
        with self.assertRaises(recovery.ReconciliationBlocked):self.recover()
        self.assertEqual(len(self.callback_calls),1)
        self.assert_old_bytes_preserved()

    def test_new_retry_interruption_remains_unknown_and_blocks_automatic_retry(self):
        def interrupted(deadline):
            self.callback_calls.append(deadline)
            self.clock.advance(3)
            raise RuntimeError("Synthetic interruption of the explicitly authorized new fit")
        with self.assertRaises(Exception):self.recover(callback=interrupted)
        self.assertIsNotNone(self.receipts.state["started"])
        self.assertIsNone(self.receipts.state["observed"])
        with self.assertRaises(recovery.ReconciliationBlocked):self.recover()
        self.assertEqual(len(self.callback_calls),1)
        self.assert_old_bytes_preserved()

    def test_session_admission_refuses_full_retry_before_claiming_or_fitting(self):
        instance=self.reconciler(deadline=self.clock()+5699)
        with self.assertRaises(recovery.ReconciliationBudgetPause):self.recover(instance)
        self.assertEqual(self.callback_calls,[])
        self.assertIsNone(self.receipts.state["started"])
        self.assert_old_bytes_preserved()

    def test_an_additional_unknown_logical_job_is_not_covered_by_this_authorization(self):
        events=self.directory/"events"
        previous=json.loads(sorted(events.glob("*.json"))[-1].read_text())
        extra={"version":engine.VERSION,"sequence":previous["sequence"]+1,"kind":"job_started",
            "plan_sha256":self.auth["plan_sha256"],"previous_sha256":previous["sha256"],
            "utc":previous["utc"],"job_id":"e"*64,"session_id":self.auth["target"]["old_session_id"],
            "job_payload_sha256":"f"*64,"reservation_seconds":30}
        extra["sha256"]=engine.digest(extra)
        (events/("%09d.json"%extra["sequence"])).write_bytes(engine.canonical(extra)+b"\n")
        with self.assertRaises((recovery.ReconciliationError,engine.InvalidPlan)):self.view()
        self.assertEqual(self.callback_calls,[])
        self.assert_old_bytes_preserved()

    def test_corrupt_original_result_is_rejected_before_manual_execution(self):
        path=next((self.directory/"results").glob("*.json"))
        changed=json.loads(path.read_text())
        changed["outcome"]["predictions_original"][0]=999
        path.write_text(json.dumps(changed))
        with self.assertRaises((recovery.ReconciliationError,engine.InvalidPlan)):self.view()
        self.assertEqual(self.callback_calls,[])

    def test_observed_scientific_failure_is_terminal_and_preserves_unknown_step_count(self):
        failed=deepcopy(OUTCOME)
        failed.update(status="failed",predictions_original=None,prediction_dates=None,phase_epochs=None,reason="resource_limit")
        failed["resource_use"].update(optimizer_steps=None,optimizer_steps_known_lower_bound=2,
            optimizer_steps_unknown=True,partial_optimizer_update_may_have_occurred=True,error_code="resource_limit")
        def observed_failure(deadline):
            self.callback_calls.append(deadline)
            self.clock.advance(5)
            return deepcopy(failed)
        answer=self.recover(callback=observed_failure)
        self.assertEqual(answer["outcome"],failed)
        self.recover()
        self.assertEqual(len(self.callback_calls),1)
        with self.open_original() as resumed:
            result=resumed.get_result(self.job_id,self.payload)
            self.assertIsNone(result["resource_use"]["optimizer_steps"])
            self.assertEqual(result["resource_use"]["optimizer_steps_known_lower_bound"],2)
        self.assert_old_bytes_preserved()

    def test_global_budget_counts_closed_sessions_and_unknown_old_session_allocation(self):
        self.campaign_with_prior_closed_sessions([28800,28800,28800,28800,28800])
        with self.assertRaises(recovery.ReconciliationBudgetPause):self.recover()
        self.assertEqual(self.callback_calls,[])
        self.assertIsNone(self.receipts.state["started"])
        self.assert_old_bytes_preserved()

    def test_later_incident_preserves_prior_recovery_and_exact_fractional_session_allocation(self):
        # Reproduce a prior recovery: its actual fit is bound to the original
        # logical session, while its physical session closes with known overhead.
        def first_retry(deadline):
            self.callback_calls.append(deadline)
            self.clock.advance(19.5)
            return deepcopy(OUTCOME)
        observe=self.receipts.save_observed
        def observed_with_overhead(record):
            answer=observe(record)
            self.clock.advance(.75)
            return answer
        with mock.patch.object(self.receipts,"save_observed",side_effect=observed_with_overhead):
            first=self.recover(callback=first_retry)
        first_target=self.job_id
        first_recovery_session=first["recovery_session_id"]
        first_receipts=deepcopy(self.receipts.state)
        self.assertEqual(first["campaign_charged_seconds"],28820.25)
        remaining=first["continuation_session_budget_seconds"]
        self.assertEqual(remaining,28779.75)

        # The continuing physical session uses precisely the residual allocation.
        # It commits another result, then loses its executor on a different job.
        self.payload={"candidate":"later-incident-candidate","seed":29}
        self.job_id=engine.digest({"plan_sha256":engine.digest(self.plan),"job":self.payload})
        later=engine.SearchSession(self.directory,self.plan,session_budget_seconds=remaining,
            campaign_budget_seconds=86400,max_storage_bytes=10*1024**3,clock=self.clock)
        later.run_job("completed-after-prior-recovery",{"candidate":"later-completed","seed":23},
            lambda deadline:self.original_result(),reservation_seconds=30)
        start=later._append_event("job_started",job_id=self.job_id,session_id=later._session_id,
            job_payload_sha256=engine.digest(self.payload),reservation_seconds=5700)
        later._closed=True
        later._release_lock()
        self.bind_authorization(start,completed_count=3)
        self.assertEqual(self.auth["limits"]["old_session_conservative_allocation_seconds"],28779.75)
        self.assertEqual(self.view().charged_wall_seconds,57600)
        before=dict(self.original_records)
        seen=[]
        def actual_later_retry(deadline):
            self.callback_calls.append(deadline)
            # Verify the conservative old closure was acknowledged before fit.
            snapshot=self.mirror.remote
            rows=[json.loads(payload) for name,payload in snapshot.items() if name.startswith("events/")]
            closure=next(row for row in rows if row["kind"]=="session_closed"
                and row["session_id"]==start["session_id"])
            self.assertEqual(closure["charged_overhead_seconds"],28777.75)
            seen.append(True)
            self.clock.advance(5)
            return deepcopy(OUTCOME)
        second=self.recover(callback=actual_later_retry)
        self.assertEqual(seen,[True])
        self.assertEqual(second["campaign_charged_seconds"],57605)
        self.assertEqual(second["actual_fit_wall_seconds"],5)
        self.assertNotEqual(first_recovery_session,second["recovery_session_id"])
        with self.open_original() as resumed:
            self.assertEqual(resumed.campaign_budget_seconds,172800)
            self.assertEqual(resumed.charged_wall_seconds,57605)
            self.assertEqual(len(resumed.completed_job_ids),4)
            self.assertEqual(resumed.get_result(first_target),OUTCOME)
            self.assertEqual(resumed.get_result(self.job_id,self.payload),OUTCOME)
        self.assertEqual(first_receipts["completed"]["job_id"],first_target)
        self.assert_old_bytes_preserved()
        self.assertTrue(all((self.directory/name).read_bytes()==payload for name,payload in before.items()))

    def campaign_with_prior_closed_sessions(self,durations):
        self.directory=Path(self.temporary.name)/"global-budget-campaign"
        for index,duration in enumerate(durations):
            with self.open_original() as session:
                def full_session(deadline):
                    self.clock.advance(duration)
                    return deepcopy(OUTCOME)
                session.run_job("finished-"+str(index),{"candidate":index},full_session,reservation_seconds=duration)
        session=self.open_original()
        session._start_session()
        start=session._append_event("job_started",job_id=self.job_id,session_id=session._session_id,
            job_payload_sha256=engine.digest(self.payload),reservation_seconds=5700)
        session._closed=True
        session._release_lock()
        self.bind_authorization(start,completed_count=len(durations))

    def test_global_budget_is_rechecked_after_prefit_mirror_acknowledgement(self):
        self.campaign_with_prior_closed_sessions([28800,28800,28800,28800,7200])
        push=self.mirror.push
        def very_slow_prefit_ack(directory,**kwargs):
            answer=push(directory,**kwargs)
            self.clock.advance(15901)
            return answer
        with mock.patch.object(self.mirror,"push",side_effect=very_slow_prefit_ack):
            with self.assertRaises(recovery.ReconciliationBudgetPause):self.recover()
        self.assertEqual(self.callback_calls,[])
        self.assertIsNotNone(self.receipts.state["started"])
        self.assertIsNone(self.receipts.state["observed"])
        self.assert_old_bytes_preserved()


if __name__=="__main__":unittest.main()
