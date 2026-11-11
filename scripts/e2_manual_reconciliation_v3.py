"""Registered one-time logical-job reconciliation without editing frozen records.

R3 requires confirmed absence of both a running executor and this job outcome.
An actual new worker outcome is recorded externally before the legacy logical
result is committed. The original physical result and costs remain unknown.
"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,importlib,json,math,re,time,uuid
from pathlib import Path

VERSION="e2-manual-reconciliation-v3"
RECEIPT_VERSION="e2-manual-recovery-physical-v1"
INITIAL_CAMPAIGN_SECONDS=86400
EFFECTIVE_CAMPAIGN_SECONDS=129600
BUDGET_EXTENSION_EVENT_SHA256="899ab1bc8c440b0e90be19abc38e3f22785aaa87902b8b531752080ad444cf9a"
HEX40=re.compile(r"[0-9a-f]{40}")
HEX64=re.compile(r"[0-9a-f]{64}")

class ReconciliationError(RuntimeError):
    def __init__(self,code):
        self.code=code
        super().__init__(code)
class ReconciliationBlocked(ReconciliationError):pass
class ReconciliationBudgetPause(ReconciliationError):pass

def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def _positive(v):return type(v) in (int,float) and math.isfinite(v) and v>0
def _utc():return datetime.now(timezone.utc).isoformat()

def _hash_records(directory):
    paths=[directory/"campaign.json"]
    if (directory/"status.json").exists():paths.append(directory/"status.json")
    for name in ("events","results"):
        folder=directory/name
        if folder.is_symlink() or not folder.is_dir():
            raise ReconciliationBlocked("campaign_record_directory_invalid")
        paths.extend(sorted(folder.glob("*.json")))
    result={}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ReconciliationBlocked("campaign_record_file_invalid")
        result[path.relative_to(directory).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result

def _validate_authorization(a,plan):
    if (type(a) is not dict or a.get("version")!="e2-manual-recovery-authorization-v3"
            or a.get("status")!="registered_before_recovery_execution"):
        raise ReconciliationBlocked("manual_authorization_not_registered")
    hashes=a.get("implementation_sha256")
    if not isinstance(hashes,dict) or not hashes or any(not isinstance(v,str) or not HEX64.fullmatch(v) for v in hashes.values()):
        raise ReconciliationBlocked("manual_implementation_hashes_missing")
    if (a.get("plan_sha256")!=digest(plan) or a.get("scientific_source_commit")!=plan.get("source_commit")
            or ("registration_sha256" in plan and a.get("scientific_registration_sha256")!=plan["registration_sha256"])):
        raise ReconciliationBlocked("manual_authorization_plan_changed")
    t=a.get("target",{});limits=a.get("limits",{})
    if any(not isinstance(t.get(k),str) or not HEX64.fullmatch(t[k]) for k in ("job_id","job_payload_sha256","started_event_sha256")):
        raise ReconciliationBlocked("manual_target_identifier_invalid")
    if not isinstance(a.get("sealed_campaign_head"),str) or not HEX40.fullmatch(a["sealed_campaign_head"]):
        raise ReconciliationBlocked("manual_campaign_head_invalid")
    if (type(t.get("sequence")) is not int or t["sequence"]<1 or type(a.get("previous_completed_jobs")) is not int
            or a["previous_completed_jobs"]<0 or not isinstance(t.get("old_session_id"),str) or not t["old_session_id"]
            or not _positive(t.get("reservation_seconds"))):
        raise ReconciliationBlocked("manual_target_domain_invalid")
    if (set(limits)!={"session_seconds","initial_campaign_seconds","campaign_seconds",
                         "budget_extension_event_sha256","old_session_conservative_allocation_seconds",
                         "budget_extension_authorized"}
            or any(not _positive(limits.get(k)) for k in ("session_seconds","initial_campaign_seconds",
                       "campaign_seconds","old_session_conservative_allocation_seconds"))
            or limits["session_seconds"]!=28800
            or limits["initial_campaign_seconds"]!=INITIAL_CAMPAIGN_SECONDS
            or limits["campaign_seconds"]!=EFFECTIVE_CAMPAIGN_SECONDS
            or limits["budget_extension_event_sha256"]!=BUDGET_EXTENSION_EVENT_SHA256
            or limits["budget_extension_authorized"] is not False
            or t["reservation_seconds"]>limits["session_seconds"]
            or limits["old_session_conservative_allocation_seconds"]>limits["session_seconds"]):
        raise ReconciliationBlocked("manual_budget_controls_invalid")
    expected={"outcome":"unknown","elapsed_seconds":None,"optimizer_steps":None,
        "optimizer_steps_known_lower_bound":0,"partial_optimizer_update_may_have_occurred":True}
    if a.get("original_unknown_attempt")!=expected:
        raise ReconciliationBlocked("original_unknown_attempt_must_remain_unknown")
    if a.get("decision")!="manual_retry_same_sealed_logical_job_once_after_confirmed_executor_and_outcome_absent":
        raise ReconciliationBlocked("manual_retry_policy_changed")

class SnapshotView:
    def __init__(self,loader,a,session_class):
        self._loader=loader;self.session_class=session_class
        self.engine=importlib.import_module(session_class.__module__)
        self.directory=loader.directory;self.plan=loader.plan;self.plan_sha256=loader.plan_sha256
        self.controls=deepcopy(loader._controls);self.events=deepcopy(loader._events)
        self.completed_job_ids=tuple(loader.completed_job_ids)
        self.unresolved_job_ids=tuple(sorted(set(loader._started)-set(loader._results)))
        self.charged_wall_seconds=loader.charged_wall_seconds
        t=a["target"];n=t["sequence"]
        if len(self.events)<=n:raise ReconciliationBlocked("sealed_event_prefix_missing")
        start=self.events[n]
        checks={"kind":"job_started","job_id":t["job_id"],"job_payload_sha256":t["job_payload_sha256"],
            "sha256":t["started_event_sha256"],"session_id":t["old_session_id"],"reservation_seconds":t["reservation_seconds"]}
        if any(start.get(k)!=v for k,v in checks.items()):
            raise ReconciliationBlocked("sealed_started_event_changed")
        self.target_start=deepcopy(start)
        baseline=[j for j in loader._results if loader._started[j]["sequence"]<n]
        if len(baseline)!=a["previous_completed_jobs"]:raise ReconciliationBlocked("original_completed_result_count_changed")
        if any(j!=t["job_id"] for j in self.unresolved_job_ids):
            raise ReconciliationBlocked("unexpected_additional_unresolved_job")
        if (self.controls["campaign_budget_seconds"]!=a["limits"]["initial_campaign_seconds"]
                or loader.campaign_budget_seconds!=a["limits"]["campaign_seconds"]):
            raise ReconciliationBlocked("sealed_campaign_budget_changed")
        old=loader._sessions.get(t["old_session_id"])
        if old is None or old["session_budget_seconds"]!=a["limits"]["old_session_conservative_allocation_seconds"]:
            raise ReconciliationBlocked("sealed_old_session_allocation_changed")
        # The 24-to-36-hour amendment already exists in the sealed prefix. R3
        # observes it; it never writes a new amendment or changes initial controls.
        extensions=[e for e in self.events if e.get("kind")=="campaign_budget_extended"
                    and e.get("sha256")==a["limits"]["budget_extension_event_sha256"]]
        if (len(extensions)!=1
                or extensions[0].get("previous_budget_seconds")!=a["limits"]["initial_campaign_seconds"]
                or extensions[0].get("new_budget_seconds")!=a["limits"]["campaign_seconds"]
                or extensions[0].get("reason")!="operator_requested_budget_extension"
                or extensions[0].get("sequence",n)>=old["sequence"]):
            raise ReconciliationBlocked("sealed_budget_extension_changed")
        self.record_hashes=_hash_records(self.directory)
    def get_result(self,job_id,job_payload=None):return self._loader.get_result(job_id,job_payload)
    def pending_jobs(self,job_ids):return self._loader.pending_jobs(job_ids)

def read_snapshot(directory,plan,authorization,*,session_class=None):
    """Frozen validation catches only its expected unresolved-attempt stop."""
    _validate_authorization(authorization,plan)
    directory=Path(directory).absolute()
    if not (directory/"campaign.json").is_file():raise ReconciliationBlocked("restored_campaign_required")
    if session_class is None:
        from e2_search_session import SearchSession
        session_class=SearchSession
    engine=importlib.import_module(session_class.__module__)
    controls=engine._read(directory/"campaign.json")["controls"]
    class ReadOnlySession(session_class):
        def _write_new(self,*args,**kwargs):raise ReconciliationBlocked("snapshot_reader_cannot_write_records")
    loader=ReadOnlySession.__new__(ReadOnlySession)
    try:
        session_class.__init__(loader,directory,plan,session_budget_seconds=authorization["limits"]["session_seconds"],
            campaign_budget_seconds=controls["campaign_budget_seconds"],max_storage_bytes=controls["max_storage_bytes"],
            completion_reserve_bytes=controls["completion_reserve_bytes"],require_directory_fsync=controls["require_directory_fsync"])
    except engine.UnresolvedAttempt:pass
    else:loader.close()
    return SnapshotView(loader,authorization,session_class)

open_snapshot=read_snapshot

class ManualReconciler:
    def __init__(self,view,authorization,mirror,receipts,*,authorization_sha256,recovery_source_commit,
                 physical_session_started_at=None,physical_deadline=None,clock=time.monotonic):
        _validate_authorization(authorization,view.plan)
        if not isinstance(authorization_sha256,str) or not HEX64.fullmatch(authorization_sha256):
            raise ReconciliationBlocked("manual_authorization_hash_invalid")
        if not isinstance(recovery_source_commit,str) or not HEX40.fullmatch(recovery_source_commit):
            raise ReconciliationBlocked("recovery_source_commit_invalid")
        self.view=view;self.authorization=deepcopy(authorization);self.authorization_sha256=authorization_sha256
        self.recovery_source_commit=recovery_source_commit
        self.mirror=mirror;self.receipts=receipts;self.clock=clock;self._last_clock=None
        now=self._now()
        self.physical_session_started_at=now if physical_session_started_at is None else physical_session_started_at
        self.physical_deadline=(self.physical_session_started_at+authorization["limits"]["session_seconds"]
            if physical_deadline is None else physical_deadline)
        if (not math.isfinite(self.physical_session_started_at) or not math.isfinite(self.physical_deadline)
                or self.physical_session_started_at>now or self.physical_deadline<=self.physical_session_started_at
                or self.physical_deadline-self.physical_session_started_at>authorization["limits"]["session_seconds"]+1e-8):
            raise ReconciliationBlocked("physical_session_deadline_invalid")
        self.owner_nonce=getattr(receipts,"owner_nonce",None) or uuid.uuid4().hex
        self.recovery_session_id=digest({"authorization":authorization_sha256,"job":authorization["target"]["job_id"]})[:32]
        self._callback_called=False;self._expected_hashes=deepcopy(view.record_hashes)
    def _now(self):
        v=self.clock()
        if type(v) not in (int,float) or not math.isfinite(v):raise ReconciliationBlocked("recovery_clock_invalid")
        if self._last_clock is not None and v<self._last_clock:raise ReconciliationBlocked("recovery_clock_moved_backwards")
        self._last_clock=float(v)
        return float(v)
    def _assert_preserved(self):
        for p,sha in self._expected_hashes.items():
            path=self.view.directory/p
            if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
                raise ReconciliationBlocked("existing_campaign_bytes_changed")
    def _reload(self,w):
        w._events=[];w._results={};w._started={};w._sessions={};w._finished_sessions=set()
        w._recovered_results=[];w._unclosed_allocations={};w._charged=0.
        w._effective_budget=w.initial_campaign_budget_seconds;w._pending_record_paths=set()
        w._validate_config(self.view.engine._read(w.directory/"campaign.json"))
        w._storage_bytes=w._scan_storage();w._external_bytes=w._scan_external_storage();w._load()
    def _writer(self):
        w=self.view.session_class.__new__(self.view.session_class)
        w.__dict__=deepcopy(self.view._loader.__dict__)
        w._lock=None;w._lock_held=False;w._closed=False;w._session_id=None;w._session_start=None;w._charge_point=None
        w._clock=self.clock;w._last_clock=None
        try:
            w._acquire_lock();self._assert_preserved();self._reload(w);return w
        except BaseException:
            w._release_lock();w._closed=True;raise
    def _head(self):
        self.mirror.inspect_remote()
        return getattr(self.mirror,"_head",getattr(self.mirror,"head",None))
    def _sync(self,w,paths,message):
        self._assert_preserved()
        reply=self.mirror.push(w.directory,paths=tuple(paths),message=message)
        commit=reply.get("remote_commit") if isinstance(reply,dict) else None
        if not isinstance(commit,str) or not HEX40.fullmatch(commit):raise ReconciliationBlocked("campaign_commit_ack_missing")
        return commit
    def _common(self,owner=None):
        a=self.authorization;t=a["target"]
        return {"version":RECEIPT_VERSION,"authorization_sha256":self.authorization_sha256,"plan_sha256":a["plan_sha256"],
            "scientific_source_commit":a["scientific_source_commit"],"recovery_source_commit":self.recovery_source_commit,
            "sealed_campaign_head":a["sealed_campaign_head"],
            "job_id":t["job_id"],"job_payload_sha256":t["job_payload_sha256"],
            "original_started_event_sha256":t["started_event_sha256"],"old_session_id":t["old_session_id"],
            "recovery_session_id":self.recovery_session_id,"owner_nonce":self.owner_nonce if owner is None else owner,
            "original_unknown_attempt":deepcopy(a["original_unknown_attempt"])}
    def _check_receipt(self,r,phase):
        if not isinstance(r,dict):raise ReconciliationBlocked("physical_receipt_missing")
        if any(r.get(k)!=v for k,v in self._common(r.get("owner_nonce")).items()) or r.get("phase")!=phase:
            raise ReconciliationBlocked("physical_receipt_binding_changed")
        c=dict(r);claimed=c.pop("sha256",None)
        if claimed!=digest(c):raise ReconciliationBlocked("physical_receipt_hash_changed")
    def _append(self,w,kind,**fields):
        e=w._append_event(kind,**fields)
        return "events/%09d.json"%e["sequence"]
    def _markers(self,w):
        a=self.authorization;t=a["target"];old=t["old_session_id"];paths=[]
        if old not in w._finished_sessions:
            if len(w._events)!=t["sequence"]+1:raise ReconciliationBlocked("unregistered_campaign_suffix_before_retry")
            paths.append(self._append(w,"job_interrupted",job_id=t["job_id"],session_id=old,
                exception_type="ColabSessionLostOutcomeUnknown",known_elapsed_seconds=None))
            paths.append(self._append(w,"session_paused",session_id=old,
                reason="manual_reconciliation_registration_"+self.authorization_sha256,details={"reservation_seconds":t["reservation_seconds"]}))
            charged=sum(r["charged_wall_seconds"] for r in w._results.values() if r["session_id"]==old)
            gap=max(0.,a["limits"]["old_session_conservative_allocation_seconds"]-charged)
            paths.append(self._append(w,"session_closed",session_id=old,session_elapsed_seconds=None,
                charged_overhead_seconds=gap,paused_reason="conservative_unclosed_session_allocation",
                directory_fsync_supported=w._directory_fsync_supported))
            self._reload(w)
        if self.recovery_session_id not in w._sessions:
            paths.append(self._append(w,"session_started",session_id=self.recovery_session_id,
                session_budget_seconds=a["limits"]["session_seconds"],campaign_budget_seconds=w.campaign_budget_seconds,
                charged_before_session_seconds=w.charged_wall_seconds,directory_fsync_required=w.require_directory_fsync,
                directory_fsync_supported=w._directory_fsync_supported))
        self._reload(w)
        self._expected_hashes=_hash_records(w.directory)
        return paths
    def _envelope(self,job,outcome,fit,overrun):
        a=self.authorization;t=a["target"]
        r={"version":self.view.engine.VERSION,"plan_sha256":a["plan_sha256"],"job_id":job,
            "job_payload_sha256":t["job_payload_sha256"],"session_id":t["old_session_id"],
            "started_event_sha256":t["started_event_sha256"],"outcome":deepcopy(outcome),"outcome_sha256":digest(outcome),
            "charged_wall_seconds":fit,"callback_wall_seconds":fit,"reservation_seconds":t["reservation_seconds"],
            "reservation_overrun":bool(overrun),"session_elapsed_seconds":None}
        r["sha256"]=digest(r);return r
    def _validate_outcome(self,job,payload,outcome,fit):
        from e2_github_campaign import validate_record
        r=self._envelope(job,outcome,fit,False)
        validate_record("results/"+self.view.session_class._job_filename(job),canonical(r)+b"\n")
        if outcome.get("status") not in ("completed","failed") or type(outcome.get("fatal")) is not bool:
            raise ReconciliationBlocked("actual_outcome_status_invalid")
        keys=("model","origin","candidate_id","payload_sha256","seed","stage","prepared_signature")
        if all(k in payload for k in keys) and outcome["resource_use"].get("identity")!={k:payload[k] for k in keys}:
            raise ReconciliationBlocked("actual_outcome_job_identity_changed")
        if outcome["status"]=="completed":
            p=outcome.get("predictions_original");d=outcome.get("prediction_dates");e=outcome.get("phase_epochs")
            if (not isinstance(p,list) or not p or not isinstance(d,list) or len(p)!=len(d)
                    or any(type(x) not in (int,float) or not math.isfinite(x) for x in p)
                    or not isinstance(e,dict) or not e or any(type(x) is not int or x<1 for x in e.values())):
                raise ReconciliationBlocked("actual_outcome_predictions_invalid")
    def _result(self,w,outcome,fit,commit,performed,status,accounted,accounted_through):
        self._assert_preserved();self._reload(w);now=self._now()
        return {"status":status,"outcome":deepcopy(outcome),"job_id":self.authorization["target"]["job_id"],
            "recovery_session_id":self.recovery_session_id,"actual_fit_wall_seconds":fit,
            "recovery_elapsed_seconds":now-self.physical_session_started_at,
            "session_seconds_remaining":max(0.,self.physical_deadline-now),
            "campaign_charged_seconds":w.charged_wall_seconds,"completed_jobs":len(w.completed_job_ids),
            "physical_reexecution_performed":performed,"campaign_commit":commit,
            "original_unknown_attempt":deepcopy(self.authorization["original_unknown_attempt"]),
            "ledger_accounted_recovery_wall_seconds":accounted,
            "campaign_accounted_through_monotonic":accounted_through,
            "continuation_session_budget_seconds":max(0.,self.physical_deadline-accounted_through)}
    def recover(self,job_id,job_payload,callback,*,reservation_seconds):
        a=self.authorization;t=a["target"]
        if (job_id!=t["job_id"] or digest(job_payload)!=t["job_payload_sha256"] or reservation_seconds!=t["reservation_seconds"]
                or digest({"plan_sha256":a["plan_sha256"],"job":job_payload})!=job_id):
            raise ReconciliationBlocked("replayed_target_job_changed")
        state=self.receipts.inspect()
        for phase in ("started","observed","completed"):
            if state.get(phase) is not None:self._check_receipt(state[phase],phase)
        started,observed,completed=(state.get(p) for p in ("started","observed","completed"))
        if completed is not None and observed is None:raise ReconciliationBlocked("completed_receipt_has_no_observed_outcome")
        if started is not None and observed is None and (started["owner_nonce"]!=self.owner_nonce or self._callback_called):
            raise ReconciliationBlocked("manual_reexecution_already_claimed_without_observed_outcome")
        w=self._writer();performed=False
        fresh=observed is not None and observed.get("owner_nonce")==self.owner_nonce and self._callback_called
        try:
            if completed is not None:
                cached=w.get_result(job_id,job_payload)
                if cached!=observed["outcome"]:raise ReconciliationBlocked("completed_recovery_campaign_result_missing")
                result=w._results[job_id]
                finish=[e for e in w._events if e["kind"]=="job_finished" and e["job_id"]==job_id]
                close=[e for e in w._events if e["kind"]=="session_closed" and e["session_id"]==self.recovery_session_id]
                if (result["sha256"]!=completed["result_sha256"] or len(finish)!=1
                        or finish[0]["session_id"]!=self.recovery_session_id
                        or finish[0]["result_sha256"]!=result["sha256"] or len(close)!=1):
                    raise ReconciliationBlocked("completed_receipt_campaign_link_changed")
                point=self.physical_session_started_at
                if fresh and type(close[0].get("session_elapsed_seconds")) in (int,float):
                    point+=close[0]["session_elapsed_seconds"]
                return self._result(w,cached,observed["actual_fit_wall_seconds"],completed["campaign_commit"],False,
                    "already_reconciled",completed.get("recovery_wall_seconds",completed["recovery_elapsed_seconds"]),
                    point)
            if observed is None:
                now=self._now()
                if now+reservation_seconds>self.physical_deadline:raise ReconciliationBudgetPause("physical_session_reservation_unaffordable")
                if w.charged_wall_seconds+now-self.physical_session_started_at+reservation_seconds>w.campaign_budget_seconds:
                    raise ReconciliationBudgetPause("campaign_recovery_reservation_unaffordable")
                w._ensure_space(w.completion_reserve_bytes)
                if started is None:
                    if self._head()!=a["sealed_campaign_head"]:raise ReconciliationBlocked("sealed_campaign_head_changed_before_manual_retry")
                    if len(w._events)!=t["sequence"]+1 or len(w._results)!=a["previous_completed_jobs"]:
                        raise ReconciliationBlocked("sealed_campaign_baseline_changed_before_manual_retry")
                    started=self.receipts.claim_started({**self._common(),"phase":"started","reservation_seconds":reservation_seconds,
                        "new_session_budget_seconds":a["limits"]["session_seconds"],"campaign_budget_seconds":a["limits"]["campaign_seconds"],
                        "old_conservative_allocation_seconds":a["limits"]["old_session_conservative_allocation_seconds"],"started_at_utc":_utc()})
                    self._check_receipt(started,"started")
                paths=self._markers(w)
                if paths:self._sync(w,paths,"Acknowledge registered manual recovery before one physical retry")
                now=self._now();deadline=min(self.physical_deadline,now+reservation_seconds)
                if now+reservation_seconds>self.physical_deadline:raise ReconciliationBudgetPause("manual_retry_deadline_spent_before_worker")
                before=w._sessions[self.recovery_session_id]["charged_before_session_seconds"]
                if before+now-self.physical_session_started_at+reservation_seconds>w.campaign_budget_seconds:
                    raise ReconciliationBudgetPause("campaign_recovery_reservation_spent_before_worker")
                self._callback_called=True;performed=True;at=self._now()
                try:outcome=callback(deadline)
                except BaseException as error:raise ReconciliationBlocked("physical_retry_interrupted_without_observed_outcome") from error
                stopped=self._now();fit=stopped-at
                if fit<0:raise ReconciliationBlocked("recovery_clock_moved_backwards")
                self._validate_outcome(job_id,job_payload,outcome,fit)
                observed=self.receipts.save_observed({**self._common(started["owner_nonce"]),"phase":"observed",
                    "outcome":deepcopy(outcome),"outcome_sha256":digest(outcome),"actual_fit_wall_seconds":fit,
                    "recovery_elapsed_seconds":stopped-self.physical_session_started_at,"observed_at_utc":_utc(),
                    "started_receipt_sha256":started["sha256"]})
                self._check_receipt(observed,"observed");fresh=True
            outcome=observed["outcome"];fit=observed["actual_fit_wall_seconds"]
            self._validate_outcome(job_id,job_payload,outcome,fit)
            if (not _positive(fit) and fit!=0) or observed.get("outcome_sha256")!=digest(outcome):
                raise ReconciliationBlocked("observed_outcome_or_duration_invalid")
            if observed.get("started_receipt_sha256")!=(started or {}).get("sha256"):
                raise ReconciliationBlocked("observed_started_receipt_changed")
            # The external OBSERVED acknowledgement precedes every logical write.
            paths=[];existing=w._results.get(job_id)
            if existing is None:
                if t["old_session_id"] not in w._finished_sessions or self.recovery_session_id not in w._sessions:
                    raise ReconciliationBlocked("registered_recovery_markers_missing")
                envelope=self._envelope(job_id,outcome,fit,fit>reservation_seconds)
                relative="results/"+w._job_filename(job_id);w._write_new(w.directory/relative,envelope);paths.append(relative)
            else:
                envelope=existing
                if envelope["outcome"]!=outcome or envelope["callback_wall_seconds"]!=fit:
                    raise ReconciliationBlocked("logical_result_differs_from_observed_retry")
                paths.append("results/"+w._job_filename(job_id))
            finishes=[e for e in w._events if e["kind"]=="job_finished" and e["job_id"]==job_id]
            if not finishes:paths.append(self._append(w,"job_finished",job_id=job_id,session_id=self.recovery_session_id,result_sha256=envelope["sha256"]))
            elif len(finishes)!=1 or finishes[0]["result_sha256"]!=envelope["sha256"] or finishes[0]["session_id"]!=self.recovery_session_id:
                raise ReconciliationBlocked("logical_finish_differs_from_physical_recovery")
            else:paths.append("events/%09d.json"%finishes[0]["sequence"])
            closes=[e for e in w._events if e["kind"]=="session_closed" and e["session_id"]==self.recovery_session_id]
            closed_at=self._now()
            elapsed=closed_at-self.physical_session_started_at
            accounted_through=closed_at if fresh else self.physical_session_started_at
            if not closes:
                # A new process cannot reconstruct prior post-observation time.
                accounted=elapsed if fresh else a["limits"]["session_seconds"];overhead=max(0.,accounted-fit)
                paths.append(self._append(w,"session_closed",session_id=self.recovery_session_id,
                    session_elapsed_seconds=elapsed if fresh else None,charged_overhead_seconds=overhead,
                    paused_reason="manual_recovery_observed_result_committed" if fresh else "conservative_recovery_session_allocation",
                    directory_fsync_supported=w._directory_fsync_supported))
            elif len(closes)!=1:raise ReconciliationBlocked("duplicate_recovery_session_close")
            else:
                overhead=closes[0]["charged_overhead_seconds"];accounted=fit+overhead
                if fresh and type(closes[0].get("session_elapsed_seconds")) in (int,float):
                    accounted_through=self.physical_session_started_at+closes[0]["session_elapsed_seconds"]
                paths.append("events/%09d.json"%closes[0]["sequence"])
            self._assert_preserved();self._expected_hashes=_hash_records(w.directory)
            commit=self._sync(w,paths,"Commit actual observed manual retry as one sealed logical job")
            completed=self.receipts.save_completed({**self._common(observed["owner_nonce"]),"phase":"completed",
                "result_sha256":envelope["sha256"],"campaign_commit":commit,"completed_at_utc":_utc(),
                "actual_fit_wall_seconds":fit,
                "recovery_elapsed_seconds":self._now()-self.physical_session_started_at if fresh else observed["recovery_elapsed_seconds"],
                "recovery_elapsed_lower_bound_only":not fresh,
                "current_repair_elapsed_seconds":self._now()-self.physical_session_started_at,
                "observed_receipt_sha256":observed["sha256"],"recovery_wall_seconds":accounted,
                "charged_recovery_overhead_seconds":overhead,
                "old_session_conservative_allocation_seconds":a["limits"]["old_session_conservative_allocation_seconds"]})
            self._check_receipt(completed,"completed")
            return self._result(w,outcome,fit,commit,performed,"logical_job_reconciled",accounted,accounted_through)
        finally:
            w._release_lock();w._closed=True
