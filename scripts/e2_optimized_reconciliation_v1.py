"""One registered physical recovery for the interrupted optimized E2 logical job.

This adapter never edits the frozen engine, old results, plan, epochs or model
configuration. A missing original outcome stays unknown. Only an explicitly
entered new combined total can enlarge the budget. The lost session is charged
its complete original allocation; the recovery receives its own durable receipt.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib, importlib, json, math, re, time, uuid
from pathlib import Path

VERSION = "e2-optimized-reconciliation-v1"
RECEIPT_VERSION = "e2-optimized-recovery-physical-v1"
AUTH_VERSION = "e2-optimized-recovery-authorization-v1"
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
UNKNOWN = {"outcome": "unknown", "elapsed_seconds": None, "optimizer_steps": None,
           "optimizer_steps_known_lower_bound": 0,
           "partial_optimizer_update_may_have_occurred": True}
DECISION = "retry_same_sealed_logical_job_once_after_executor_and_outcome_absence_confirmed"

@dataclass(frozen=True)
class PinnedIncident:
    repository: str = "jassan-bit/DeepLearning_202630"
    branch: str = "campaigns/e2-cuda-crop-11-v2"
    sealed_head: str = "30c121072f97917c234fea2269dad50d00113850"
    plan_sha256: str = "877a8ab4a0d34599f43c5dd16a90fe87fe03d37fe535466bf8e009c379cce742"
    scientific_source_commit: str = "564ba4762f84a800b02fa4c734f5752a169f5223"
    controller_source_commit: str = "8e3ab6cfb1b3fc3c804f0c914fa5d92b661058d9"
    previous_completed_jobs: int = 889
    target_job_id: str = "2228300649374ed72f7d223cafafe60815fc7a20767979d96f6f709b1dcad2fe"
    target_payload_sha256: str = "65bd47f453de7b7cf501b6ba3f2e485893891fe036711154860c5c1af0095a9c"
    target_started_sha256: str = "cb313756ba677a378ee96f9ec572c56b5cfde0d29dbc5ba4728f8aeba045c8cc"
    target_sequence: int = 1783
    old_session_id: str = "9f75b7f0c76e4e88993f03ef2c296531"
    reservation_seconds: float = 7500.0
    initial_optimized_budget_seconds: float = 16246.967466072238
    effective_optimized_budget_seconds: float = 45046.96746607224
    old_session_allocation_seconds: float = 28800.0
    legacy_spent_seconds: float = 156553.03253392776
    authorized_combined_hours: float = 56.0
    budget_extension_event_sha256: str = "946a31b7809f28cfb8651c3560d210f6610cd1b0867a37f7601c9e765f71b936"
    receipt_branch: str = "recoveries/e2-cuda-crop-11-v2/colab-loss-20261112-r1"

INCIDENT = PinnedIncident()

class ReconciliationError(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)
class ReconciliationBlocked(ReconciliationError): pass
class ReconciliationBudgetPause(ReconciliationError): pass

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
def digest(value): return hashlib.sha256(canonical(value)).hexdigest()
def _positive(value): return type(value) in (int, float) and math.isfinite(value) and value > 0
def _utc(): return datetime.now(timezone.utc).isoformat()
def _same_number(a, b):
    return type(a) in (int, float) and type(b) in (int, float) and math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=0, abs_tol=1e-7)

def _hash_records(directory):
    paths = [directory / "campaign.json"]
    if (directory / "status.json").exists(): paths.append(directory / "status.json")
    for name in ("events", "results"):
        folder = directory / name
        if folder.is_symlink() or not folder.is_dir():
            raise ReconciliationBlocked("campaign_record_directory_invalid")
        paths.extend(sorted(folder.glob("*.json")))
    result = {}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ReconciliationBlocked("campaign_record_file_invalid")
        result[path.relative_to(directory).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result

def _target():
    i = INCIDENT
    return {"job_id": i.target_job_id, "job_payload_sha256": i.target_payload_sha256,
            "started_event_sha256": i.target_started_sha256, "sequence": i.target_sequence,
            "old_session_id": i.old_session_id, "reservation_seconds": i.reservation_seconds}

def _validate_authorization(a, plan):
    i = INCIDENT
    if not isinstance(a, dict) or a.get("version") != AUTH_VERSION or a.get("status") != "registered_before_recovery_execution":
        raise ReconciliationBlocked("manual_authorization_not_registered")
    expected = {"repository": i.repository, "campaign_branch": i.branch, "receipt_branch": i.receipt_branch,
                "sealed_campaign_head": i.sealed_head, "plan_sha256": i.plan_sha256,
                "scientific_source_commit": i.scientific_source_commit,
                "controller_source_commit": i.controller_source_commit,
                "previous_completed_jobs": i.previous_completed_jobs,
                "registration_sha256": plan.get("registration_sha256"),
                "target": _target(), "original_unknown_attempt": UNKNOWN, "decision": DECISION}
    if any(a.get(k) != v for k, v in expected.items()) or digest(plan) != i.plan_sha256:
        raise ReconciliationBlocked("manual_authorization_binding_changed")
    if plan.get("source_commit") != i.scientific_source_commit:
        raise ReconciliationBlocked("scientific_source_binding_changed")
    hashes = a.get("implementation_sha256")
    if not isinstance(hashes, dict) or not hashes or any(not isinstance(k, str) or not k or not isinstance(v, str) or not HEX64.fullmatch(v) for k, v in hashes.items()):
        raise ReconciliationBlocked("manual_implementation_hashes_missing")
    if not isinstance(a.get("registered_at_utc"), str) or not a["registered_at_utc"]:
        raise ReconciliationBlocked("manual_registration_timestamp_missing")
    try:
        stamp = datetime.fromisoformat(a["registered_at_utc"].replace("Z", "+00:00"))
        if stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
            raise ValueError("UTC required")
    except (ValueError, AttributeError):
        raise ReconciliationBlocked("manual_registration_timestamp_invalid") from None
    limits = a.get("limits", {})
    required = {"session_seconds", "initial_campaign_seconds", "campaign_seconds", "new_campaign_seconds",
                "old_session_conservative_allocation_seconds", "combined_total_hours", "legacy_spent_seconds",
                "budget_extension_event_sha256", "budget_extension_authorized"}
    if set(limits) != required:
        raise ReconciliationBlocked("manual_budget_controls_invalid")
    expected_limits = {"session_seconds": i.old_session_allocation_seconds,
        "initial_campaign_seconds": i.initial_optimized_budget_seconds,
        "campaign_seconds": i.effective_optimized_budget_seconds,
        "old_session_conservative_allocation_seconds": i.old_session_allocation_seconds,
        "legacy_spent_seconds": i.legacy_spent_seconds,
        "budget_extension_event_sha256": i.budget_extension_event_sha256,
        "budget_extension_authorized": True}
    if limits.get("budget_extension_authorized") is not True or any(limits.get(k) != v for k, v in expected_limits.items()):
        raise ReconciliationBlocked("manual_budget_binding_changed")
    total = limits.get("combined_total_hours")
    if (not _positive(total) or total <= i.authorized_combined_hours
            or a.get("user_explicit_total_hours") != total
            or not _same_number(limits.get("new_campaign_seconds"), total * 3600 - i.legacy_spent_seconds)):
        raise ReconciliationBlocked("new_combined_total_not_explicitly_authorized")
    if not _positive(limits["new_campaign_seconds"]) or limits["new_campaign_seconds"] <= i.effective_optimized_budget_seconds:
        raise ReconciliationBlocked("new_campaign_limit_must_increase")
    evidence = a.get("evidence", {})
    if evidence.get("executor_absence_required") is not True or evidence.get("durable_outcome_absence_required") is not True:
        raise ReconciliationBlocked("manual_recovery_absence_requirements_missing")
    canonical(a)

class SnapshotView:
    def __init__(self, loader, session_class, authorization=None):
        i = INCIDENT
        self._loader = loader
        self.session_class = session_class
        self.engine = importlib.import_module(session_class.__module__)
        self.directory = loader.directory
        self.plan = loader.plan
        self.plan_sha256 = loader.plan_sha256
        self.controls = deepcopy(loader._controls)
        self.events = deepcopy(loader._events)
        self.completed_job_ids = tuple(loader.completed_job_ids)
        self.unresolved_job_ids = tuple(sorted(set(loader._started) - set(loader._results)))
        self.charged_wall_seconds = loader.charged_wall_seconds
        self.campaign_budget_seconds = loader.campaign_budget_seconds
        if self.plan_sha256 != i.plan_sha256 or self.plan.get("source_commit") != i.scientific_source_commit:
            raise ReconciliationBlocked("sealed_scientific_plan_changed")
        if len(self.events) <= i.target_sequence:
            raise ReconciliationBlocked("sealed_event_prefix_missing")
        start = self.events[i.target_sequence]
        checks = {"kind": "job_started", "job_id": i.target_job_id,
            "job_payload_sha256": i.target_payload_sha256, "sha256": i.target_started_sha256,
            "session_id": i.old_session_id, "reservation_seconds": i.reservation_seconds}
        if any(start.get(k) != v for k, v in checks.items()):
            raise ReconciliationBlocked("sealed_started_event_changed")
        self.target_start = deepcopy(start)
        baseline = [j for j in loader._results if loader._started[j]["sequence"] < i.target_sequence]
        if len(baseline) != i.previous_completed_jobs or any(j != i.target_job_id for j in self.unresolved_job_ids):
            raise ReconciliationBlocked("original_completed_result_count_changed")
        if any(loader._started[j]["sequence"] > i.target_sequence for j in loader._results):
            raise ReconciliationBlocked("unexpected_result_after_sealed_target")
        if not _same_number(self.controls["campaign_budget_seconds"], i.initial_optimized_budget_seconds):
            raise ReconciliationBlocked("sealed_initial_campaign_budget_changed")
        old = loader._sessions.get(i.old_session_id)
        if old is None or not _same_number(old["session_budget_seconds"], i.old_session_allocation_seconds):
            raise ReconciliationBlocked("sealed_old_session_allocation_changed")
        prior = [e for e in self.events if e.get("kind") == "campaign_budget_extended" and e.get("sha256") == i.budget_extension_event_sha256]
        if len(prior) != 1 or not _same_number(prior[0].get("new_budget_seconds"), i.effective_optimized_budget_seconds):
            raise ReconciliationBlocked("sealed_budget_extension_changed")
        if authorization is None:
            if (len(self.events) != i.target_sequence + 1 or self.unresolved_job_ids != (i.target_job_id,)
                    or not _same_number(loader.campaign_budget_seconds, i.effective_optimized_budget_seconds)):
                raise ReconciliationBlocked("sealed_campaign_suffix_changed")
        else:
            _validate_authorization(authorization, self.plan)
            allowed = (i.effective_optimized_budget_seconds, authorization["limits"]["new_campaign_seconds"])
            if not any(_same_number(loader.campaign_budget_seconds, value) for value in allowed):
                raise ReconciliationBlocked("sealed_campaign_budget_changed")
            _validate_recovery_suffix(loader, authorization)
        self.record_hashes = _hash_records(self.directory)
    def get_result(self, job_id, job_payload=None): return self._loader.get_result(job_id, job_payload)
    def pending_jobs(self, job_ids): return self._loader.pending_jobs(job_ids)


def read_snapshot(directory, plan, *, session_class=None, authorization=None):
    """Validate durable records without new session receipts, data reads or fits."""
    directory = Path(directory).absolute()
    if not (directory / "campaign.json").is_file():
        raise ReconciliationBlocked("restored_campaign_required")
    if session_class is None:
        from e2_search_session import SearchSession
        session_class = SearchSession
    engine = importlib.import_module(session_class.__module__)
    controls = engine._read(directory / "campaign.json")["controls"]
    class ReadOnlySession(session_class):
        def _write_new(self, *args, **kwargs):
            raise ReconciliationBlocked("snapshot_reader_cannot_write_records")
    loader = ReadOnlySession.__new__(ReadOnlySession)
    try:
        session_class.__init__(loader, directory, plan,
            session_budget_seconds=INCIDENT.old_session_allocation_seconds,
            campaign_budget_seconds=controls["campaign_budget_seconds"], max_storage_bytes=controls["max_storage_bytes"],
            completion_reserve_bytes=controls["completion_reserve_bytes"], require_directory_fsync=controls["require_directory_fsync"])
    except engine.UnresolvedAttempt:
        pass
    else:
        loader.close()
    return SnapshotView(loader, session_class, authorization)

open_snapshot = read_snapshot


def budget_report(view):
    """Includes the entire unclosed allocation; never reports it as observed fit time."""
    i = INCIDENT
    total = i.legacy_spent_seconds + view.charged_wall_seconds
    remaining = max(0.0, view.campaign_budget_seconds - view.charged_wall_seconds)
    return {"status": "inspection_only", "new_completed_jobs": len(view.completed_job_ids),
        "unresolved_jobs": list(view.unresolved_job_ids), "original_unknown_attempt": deepcopy(UNKNOWN),
        "combined_authorized_hours": (i.legacy_spent_seconds + view.campaign_budget_seconds) / 3600,
        "combined_conservatively_charged_seconds": total,
        "remaining_seconds": remaining, "next_reservation_seconds": i.reservation_seconds,
        "reservation_fits": remaining >= i.reservation_seconds,
        "market_training_started": False, "remote_writes": False}


def build_authorization(view, *, total_hours, implementation_sha256, registered_at_utc=None):
    i = INCIDENT
    if not _positive(total_hours) or total_hours <= i.authorized_combined_hours:
        raise ReconciliationBlocked("new_combined_total_not_explicitly_authorized")
    new_budget = total_hours * 3600 - i.legacy_spent_seconds
    if view.charged_wall_seconds + i.reservation_seconds > new_budget:
        raise ReconciliationBudgetPause("new_total_does_not_fit_recovery_reservation")
    value = {"version": AUTH_VERSION, "status": "registered_before_recovery_execution",
        "registered_at_utc": _utc() if registered_at_utc is None else registered_at_utc,
        "repository": i.repository, "campaign_branch": i.branch, "receipt_branch": i.receipt_branch,
        "sealed_campaign_head": i.sealed_head, "plan_sha256": i.plan_sha256,
        "scientific_source_commit": i.scientific_source_commit, "controller_source_commit": i.controller_source_commit,
        "registration_sha256": view.plan.get("registration_sha256"),
        "previous_completed_jobs": i.previous_completed_jobs, "target": _target(),
        "original_unknown_attempt": deepcopy(UNKNOWN), "decision": DECISION,
        "user_explicit_total_hours": total_hours,
        "evidence": {"executor_absence_required": True, "durable_outcome_absence_required": True},
        "limits": {"session_seconds": i.old_session_allocation_seconds,
            "initial_campaign_seconds": i.initial_optimized_budget_seconds,
            "campaign_seconds": i.effective_optimized_budget_seconds,
            "new_campaign_seconds": new_budget,
            "old_session_conservative_allocation_seconds": i.old_session_allocation_seconds,
            "combined_total_hours": total_hours, "legacy_spent_seconds": i.legacy_spent_seconds,
            "budget_extension_event_sha256": i.budget_extension_event_sha256,
            "budget_extension_authorized": True},
        "implementation_sha256": deepcopy(implementation_sha256)}
    _validate_authorization(value, view.plan)
    return value


def _validate_recovery_suffix(loader, authorization):
    """Only this recovery's append-only markers may follow the sealed prefix."""
    target = authorization["target"]
    auth_sha = digest(authorization)
    recovery_session = digest({"authorization": auth_sha, "job": target["job_id"]})[:32]
    expected_order = [
        ("campaign_budget_extended", None), ("job_interrupted", target["old_session_id"]),
        ("session_paused", target["old_session_id"]), ("session_closed", target["old_session_id"]),
        ("session_started", recovery_session), ("job_finished", recovery_session),
        ("session_closed", recovery_session),
    ]
    suffix = loader._events[target["sequence"] + 1:]
    if len(suffix) > len(expected_order):
        raise ReconciliationBlocked("unexpected_recovery_ledger_suffix")
    for position, event in enumerate(suffix):
        kind = event["kind"]
        if (kind, event.get("session_id")) != expected_order[position]:
            raise ReconciliationBlocked("unexpected_recovery_ledger_suffix")
        if kind == "campaign_budget_extended":
            if (not _same_number(event.get("previous_budget_seconds"), authorization["limits"]["campaign_seconds"])
                    or not _same_number(event.get("new_budget_seconds"), authorization["limits"]["new_campaign_seconds"])
                    or event.get("reason") != "explicit_combined_total_registered_recovery_" + auth_sha):
                raise ReconciliationBlocked("recovery_budget_amendment_changed")
        if kind in ("job_interrupted", "job_finished") and event.get("job_id") != target["job_id"]:
            raise ReconciliationBlocked("unexpected_recovery_job")
        if kind == "job_interrupted" and (event.get("known_elapsed_seconds") is not None
                or event.get("exception_type") != "ColabSessionLostOutcomeUnknown"):
            raise ReconciliationBlocked("unknown_original_attempt_changed")
        if kind == "session_paused" and event.get("reason") != "manual_reconciliation_registration_" + auth_sha:
            raise ReconciliationBlocked("recovery_pause_authorization_changed")
        if kind == "session_started":
            if (not _same_number(event.get("session_budget_seconds"), authorization["limits"]["session_seconds"])
                    or not _same_number(event.get("campaign_budget_seconds"), authorization["limits"]["new_campaign_seconds"])):
                raise ReconciliationBlocked("recovery_session_controls_changed")
        if kind == "session_closed" and event.get("session_id") == target["old_session_id"]:
            observed_charge = sum(r["charged_wall_seconds"] for job, r in loader._results.items()
                                  if r["session_id"] == target["old_session_id"] and job != target["job_id"])
            expected_gap = max(0.0, authorization["limits"]["old_session_conservative_allocation_seconds"] - observed_charge)
            if (event.get("session_elapsed_seconds") is not None
                    or event.get("paused_reason") != "conservative_unclosed_session_allocation"
                    or not _same_number(event.get("charged_overhead_seconds"), expected_gap)):
                raise ReconciliationBlocked("old_session_conservative_charge_changed")

class ManualReconciler:
    def __init__(self,view,authorization,mirror,receipts,*,authorization_sha256,recovery_source_commit,
                 physical_session_started_at=None,physical_deadline=None,clock=time.monotonic,
                 executor_absence_confirmed=False):
        _validate_authorization(authorization,view.plan)
        if executor_absence_confirmed is not True:
            raise ReconciliationBlocked("executor_absence_not_confirmed")
        if (not isinstance(authorization_sha256,str) or not HEX64.fullmatch(authorization_sha256)
                or authorization_sha256 != digest(authorization)):
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
        _validate_recovery_suffix(w, self.authorization)
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
        _validate_recovery_suffix(w, a)
        if _same_number(w.campaign_budget_seconds, a["limits"]["campaign_seconds"]):
            paths.append(self._append(w,"campaign_budget_extended",
                previous_budget_seconds=w.campaign_budget_seconds,
                new_budget_seconds=a["limits"]["new_campaign_seconds"],
                reason="explicit_combined_total_registered_recovery_"+self.authorization_sha256))
            self._reload(w)
        elif not _same_number(w.campaign_budget_seconds,a["limits"]["new_campaign_seconds"]):
            raise ReconciliationBlocked("recovery_budget_changed_before_markers")
        if old not in w._finished_sessions:
            if any(e["kind"] in ("job_interrupted","session_paused","session_closed","session_started","job_finished")
                   for e in w._events[t["sequence"]+1:]):
                raise ReconciliationBlocked("partial_recovery_markers_without_observed_outcome")
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
        if started is not None and observed is None:
            raise ReconciliationBlocked("manual_reexecution_already_claimed_without_observed_outcome")
        self.receipts.ensure_authorization()
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
                if w.charged_wall_seconds+now-self.physical_session_started_at+reservation_seconds>a["limits"]["new_campaign_seconds"]:
                    raise ReconciliationBudgetPause("campaign_recovery_reservation_unaffordable")
                w._ensure_space(w.completion_reserve_bytes)
                if started is None:
                    if self._head()!=a["sealed_campaign_head"]:raise ReconciliationBlocked("sealed_campaign_head_changed_before_manual_retry")
                    if len(w._events)!=t["sequence"]+1 or len(w._results)!=a["previous_completed_jobs"]:
                        raise ReconciliationBlocked("sealed_campaign_baseline_changed_before_manual_retry")
                    started=self.receipts.claim_started({**self._common(),"phase":"started","reservation_seconds":reservation_seconds,
                        "new_session_budget_seconds":a["limits"]["session_seconds"],"campaign_budget_seconds":a["limits"]["new_campaign_seconds"],
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


class RecoveryReceiptStore:
    """Create-only durable evidence, independent of the scientific campaign.

    ``transport.read(path)`` returns bytes or None. ``transport.create(path,
    bytes)`` must atomically create an immutable file and return the bytes
    actually persisted, including an existing racing winner. A lost write
    acknowledgement is resolved with a read; the physical worker is not retried.
    """
    PHASES = ("started", "observed", "completed")

    def __init__(self, transport, authorization, *, owner_nonce=None):
        self.transport = transport
        self.authorization = deepcopy(authorization)
        self.authorization_sha256 = digest(authorization)
        self.owner_nonce = uuid.uuid4().hex if owner_nonce is None else owner_nonce
        if not isinstance(self.owner_nonce, str) or not re.fullmatch(r"[0-9a-f]{32}", self.owner_nonce):
            raise ReconciliationBlocked("invalid_recovery_owner_nonce")
        self._authorization_bytes = canonical(self.authorization) + b"\n"

    def _read(self, path):
        raw = self.transport.read(path)
        if raw is not None and not isinstance(raw, bytes):
            raise ReconciliationBlocked("recovery_transport_bytes_required")
        return raw

    def _create(self, path, raw):
        try:
            actual = self.transport.create(path, raw)
        except Exception:
            # Confirm only; no second mutation after an ambiguous acknowledgement.
            actual = self._read(path)
            if actual is None:
                raise
        if actual != raw:
            raise ReconciliationBlocked("immutable_recovery_receipt_collision")
        return actual

    def ensure_authorization(self):
        current = self._read("authorization.json")
        if current is None:
            current = self._create("authorization.json", self._authorization_bytes)
        if current != self._authorization_bytes:
            raise ReconciliationBlocked("durable_recovery_authorization_changed")
        return self.authorization_sha256

    def _validate(self, receipt, phase):
        a = self.authorization
        t = a["target"]
        if not isinstance(receipt, dict) or receipt.get("phase") != phase:
            raise ReconciliationBlocked("physical_recovery_receipt_phase_invalid")
        expected = {"version": RECEIPT_VERSION, "authorization_sha256": self.authorization_sha256,
            "plan_sha256": a["plan_sha256"], "scientific_source_commit": a["scientific_source_commit"],
            "sealed_campaign_head": a["sealed_campaign_head"], "job_id": t["job_id"],
            "job_payload_sha256": t["job_payload_sha256"], "original_started_event_sha256": t["started_event_sha256"],
            "old_session_id": t["old_session_id"],
            "recovery_session_id": digest({"authorization": self.authorization_sha256, "job": t["job_id"]})[:32],
            "original_unknown_attempt": a["original_unknown_attempt"]}
        if any(receipt.get(k) != v for k, v in expected.items()):
            raise ReconciliationBlocked("physical_recovery_receipt_binding_changed")
        if (not isinstance(receipt.get("owner_nonce"), str)
                or not re.fullmatch(r"[0-9a-f]{32}", receipt["owner_nonce"])
                or not isinstance(receipt.get("recovery_source_commit"), str)
                or not HEX40.fullmatch(receipt["recovery_source_commit"])):
            raise ReconciliationBlocked("physical_recovery_receipt_identity_invalid")
        unsigned = {k: v for k, v in receipt.items() if k != "sha256"}
        if receipt.get("sha256") != digest(unsigned):
            raise ReconciliationBlocked("physical_recovery_receipt_hash_changed")
        if phase == "started":
            expected_start = {"reservation_seconds": t["reservation_seconds"],
                "new_session_budget_seconds": a["limits"]["session_seconds"],
                "campaign_budget_seconds": a["limits"]["new_campaign_seconds"],
                "old_conservative_allocation_seconds": a["limits"]["old_session_conservative_allocation_seconds"]}
            if any(receipt.get(k) != v for k, v in expected_start.items()) or not receipt.get("started_at_utc"):
                raise ReconciliationBlocked("physical_recovery_started_controls_changed")
        elif phase == "observed":
            if (receipt.get("outcome_sha256") != digest(receipt.get("outcome"))
                    or not isinstance(receipt.get("started_receipt_sha256"), str)
                    or not HEX64.fullmatch(receipt["started_receipt_sha256"])):
                raise ReconciliationBlocked("physical_recovery_observed_outcome_changed")
            for name in ("actual_fit_wall_seconds", "recovery_elapsed_seconds"):
                value = receipt.get(name)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ReconciliationBlocked("physical_recovery_observed_time_invalid")
        else:
            for name, size in (("result_sha256", 64), ("observed_receipt_sha256", 64), ("campaign_commit", 40)):
                value = receipt.get(name)
                if not isinstance(value, str) or not (HEX64 if size == 64 else HEX40).fullmatch(value):
                    raise ReconciliationBlocked("physical_recovery_completed_link_invalid")
        return receipt

    def inspect(self):
        authorization = self._read("authorization.json")
        if authorization is not None and authorization != self._authorization_bytes:
            raise ReconciliationBlocked("durable_recovery_authorization_changed")
        state = {}
        for phase in self.PHASES:
            raw = self._read(phase + ".json")
            if raw is None:
                state[phase] = None
                continue
            if authorization is None:
                raise ReconciliationBlocked("physical_receipt_has_no_authorization")
            try:
                receipt = json.loads(raw)
            except (ValueError, UnicodeError):
                raise ReconciliationBlocked("physical_receipt_json_invalid") from None
            state[phase] = self._validate(receipt, phase)
        started, observed, completed = (state[p] for p in self.PHASES)
        if (observed is not None and started is None) or (completed is not None and observed is None):
            raise ReconciliationBlocked("physical_recovery_receipt_chain_incomplete")
        if observed is not None:
            if (observed["started_receipt_sha256"] != started["sha256"]
                    or observed["owner_nonce"] != started["owner_nonce"]
                    or observed["recovery_source_commit"] != started["recovery_source_commit"]):
                raise ReconciliationBlocked("physical_observed_started_link_changed")
        if completed is not None:
            if (completed["observed_receipt_sha256"] != observed["sha256"]
                    or completed["owner_nonce"] != observed["owner_nonce"]
                    or completed["recovery_source_commit"] != observed["recovery_source_commit"]):
                raise ReconciliationBlocked("physical_completed_observed_link_changed")
        return state

    def _save(self, value, phase):
        receipt = deepcopy(value)
        if "sha256" in receipt:
            raise ReconciliationBlocked("caller_supplied_physical_receipt_hash")
        receipt["sha256"] = digest(receipt)
        self._validate(receipt, phase)
        actual = self._create(phase + ".json", canonical(receipt) + b"\n")
        return self._validate(json.loads(actual), phase)

    def claim_started(self, value):
        self.ensure_authorization()
        state = self.inspect()
        if state["started"] is not None:
            raise ReconciliationBlocked("manual_reexecution_already_claimed_without_observed_outcome")
        if value.get("owner_nonce") != self.owner_nonce:
            raise ReconciliationBlocked("physical_claim_owner_changed")
        return self._save(value, "started")

    def save_observed(self, value):
        state = self.inspect()
        started = state["started"]
        if (started is None or started["owner_nonce"] != self.owner_nonce
                or value.get("owner_nonce") != self.owner_nonce
                or value.get("started_receipt_sha256") != started["sha256"]):
            raise ReconciliationBlocked("physical_observation_without_owned_claim")
        return self._save(value, "observed")

    def save_completed(self, value):
        state = self.inspect()
        observed = state["observed"]
        if (observed is None or value.get("owner_nonce") != observed["owner_nonce"]
                or value.get("observed_receipt_sha256") != observed["sha256"]):
            raise ReconciliationBlocked("physical_completion_without_observed_outcome")
        return self._save(value, "completed")
