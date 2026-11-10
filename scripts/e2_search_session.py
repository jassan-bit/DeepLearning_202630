"""Durable, budgeted recovery between completed scientific search attempts.

This module never trains or chooses a winner. A caller supplies an immutable
JSON plan and converts scientific outcomes (including charged model failures)
to JSON. Callback exceptions mean an unresolved infrastructure interruption,
not an unsuccessful scientific candidate, and block further fitting.

Locks coordinate cooperating processes on one mounted filesystem. They are not
a distributed lock across independently cached cloud mounts: use one Colab
runtime for a campaign at a time. File fsync is mandatory; directory fsync can
be explicitly relaxed for FUSE and that choice is sealed in campaign.json.
The storage bound covers observed/managed logical file bytes, not a filesystem
quota or an operating-system bound on callback-created checkpoints.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
import uuid

VERSION = "e2-search-session-v1"
ZERO = "0"*64


class SessionError(RuntimeError):
    pass


class InvalidPlan(SessionError):
    pass


class CampaignLocked(SessionError):
    pass


class LockUnavailable(SessionError):
    pass


class StorageLimit(SessionError):
    pass


class IncompleteCampaign(SessionError):
    pass


class SessionPaused(SessionError):
    def __init__(self, reason, details=None, *, completed_result=None):
        self.reason = reason
        self.details = deepcopy(details or {})
        self.completed_result = deepcopy(completed_result)
        super().__init__("Search session paused: "+reason)


class UnresolvedAttempt(SessionError):
    def __init__(self, job_ids, details=None):
        self.job_ids = tuple(sorted(job_ids))
        self.details = deepcopy(details or {})
        super().__init__("Unresolved attempt requires reconciliation; automatic retry is forbidden")


def _json_value(value):
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise InvalidPlan("JSON numbers must be finite")
        return value
    if type(value) is list:
        return [_json_value(v) for v in value]
    if type(value) is dict:
        if any(type(k) is not str for k in value):
            raise InvalidPlan("JSON object keys must be strings")
        return {k:_json_value(v) for k,v in value.items()}
    raise InvalidPlan("Search metadata must contain JSON-native values")


def canonical(value):
    try:
        return json.dumps(_json_value(value), sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (UnicodeError, ValueError, TypeError) as error:
        raise InvalidPlan("Invalid JSON search metadata") from error


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _number(value, label, *, integer=False):
    if isinstance(value, bool) or type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise InvalidPlan(label+" must be positive and finite")
    if integer and type(value) is not int:
        raise InvalidPlan(label+" must be an integer")
    return value if integer else float(value)


def _pairs(pairs):
    result = {}
    for key,value in pairs:
        if key in result:
            raise InvalidPlan("Duplicate JSON object key")
        result[key] = value
    return result


def _read(path):
    try:
        value=json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
        canonical(value)
        return value
    except (OSError, ValueError, UnicodeError) as error:
        raise InvalidPlan("Unreadable or invalid campaign record: "+path.name) from error


def _checked_hash(record):
    if type(record) is not dict:
        raise InvalidPlan("Campaign record must be a JSON object")
    value=deepcopy(record)
    claimed=value.pop("sha256", None)
    if claimed != digest(value):
        raise InvalidPlan("Campaign record hash changed")
    return value


def _no_symlinks(path):
    path=Path(path).absolute()
    if any(p.is_symlink() for p in (path,*path.parents)):
        raise InvalidPlan("Campaign paths must not contain symbolic links")


class SearchSession:
    """Single-writer campaign ledger with lazy per-execution session receipts.

    ``run_job`` invokes ``callback(deadline_monotonic)`` after reserving the
    whole declared worker bound. The callback must supervise its own process
    against that deadline. This generic in-process wrapper detects overruns but
    cannot preempt an arbitrary Python callback. A committed result survives a
    missing job_finished event. A started attempt without a committed result
    blocks the campaign; it is never retried automatically.

    The campaign's original cap and metadata/storage policy remain immutable.
    An explicit, reasoned ``extend_campaign_budget`` appends a budget amendment.
    An unclosed prior session whose fits all have durable results conservatively
    consumes its entire session allocation; this is labelled allocation charging,
    not an observed training duration.
    """
    def __init__(self, directory, plan, *, session_budget_seconds,
                 campaign_budget_seconds, max_storage_bytes, clock=time.monotonic,
                 completion_reserve_bytes=1024*1024, require_directory_fsync=True):
        if type(plan) is not dict:
            raise InvalidPlan("Plan must be a JSON object")
        self._plan=_json_value(plan)
        self.plan_sha256=digest(self._plan)
        self.session_budget_seconds=_number(session_budget_seconds,"Session budget")
        self.initial_campaign_budget_seconds=_number(campaign_budget_seconds,"Campaign budget")
        self.max_storage_bytes=_number(max_storage_bytes,"Storage budget",integer=True)
        self.completion_reserve_bytes=_number(completion_reserve_bytes,"Completion reserve",integer=True)
        if type(require_directory_fsync) is not bool:
            raise InvalidPlan("Directory fsync policy must be explicit boolean")
        self.require_directory_fsync=require_directory_fsync
        self.directory=Path(directory).absolute()
        self._clock=clock
        self._last_clock=None
        self._lock=None
        self._lock_held=False
        self._closed=False
        self._session_id=None
        self._session_start=None
        self._charge_point=None
        self._session_initial_charge=None
        self._paused=None
        self._directory_fsync_supported=True
        self._events=[]
        self._results={}
        self._started={}
        self._sessions={}
        self._finished_sessions=set()
        self._recovered_results=[]
        self._unclosed_allocations={}
        self._charged=0.0
        self._effective_budget=self.initial_campaign_budget_seconds
        self._storage_bytes=0
        self._external_bytes=0
        self._pending_record_paths=set()
        self._controls={"campaign_budget_seconds":self.initial_campaign_budget_seconds,
                        "max_storage_bytes":self.max_storage_bytes,
                        "completion_reserve_bytes":self.completion_reserve_bytes,
                        "require_directory_fsync":self.require_directory_fsync}
        _no_symlinks(self.directory)
        config=self.directory/"campaign.json"
        # Reject mismatched existing plans before creating even a lock file.
        if config.exists():
            self._validate_config(_read(config))
        elif self.directory.exists() and any(p.name!=".campaign.lock" for p in self.directory.iterdir()):
            raise InvalidPlan("Existing campaign directory has no valid campaign.json")
        self.directory.mkdir(parents=True,exist_ok=True)
        try:
            self._acquire_lock()
            if config.exists():
                self._validate_config(_read(config))
            else:
                if any(p.name!=".campaign.lock" for p in self.directory.iterdir()):
                    raise InvalidPlan("Refusing to initialize a nonempty campaign")
                # Check a strict filesystem policy before sealing any config.
                # A failed check leaves only an unlocked lock file, permitting
                # an explicit relaxed FUSE initialization at the same location.
                self._fsync_directory(self.directory)
                for name in ("events","results","sessions"):
                    (self.directory/name).mkdir()
                value={"version":VERSION,"plan":self._plan,"plan_sha256":self.plan_sha256,
                       "controls":self._controls,"created_at_utc":self._utc()}
                value["sha256"]=digest(value)
                self._write_new(config,value)
            self._storage_bytes=self._scan_storage()
            self._external_bytes=self._scan_external_storage()
            self._load()
            unresolved=set(self._started)-set(self._results)
            if unresolved:
                raise UnresolvedAttempt(unresolved,{"known_charge_seconds":self._charged,
                    "unknown_attempt_reservations":{j:self._started[j]["reservation_seconds"] for j in unresolved}})
        except BaseException:
            self._release_lock()
            self._closed=True
            raise

    @property
    def plan(self):
        return deepcopy(self._plan)

    @property
    def campaign_budget_seconds(self):
        return self._effective_budget

    @property
    def charged_wall_seconds(self):
        return self._charged

    @property
    def completed_job_ids(self):
        return tuple(self._results)

    @property
    def storage_used_bytes(self):
        return self._storage_bytes

    @property
    def pending_record_paths(self):
        """Managed immutable JSON needing a remote acknowledgement.

        Existing records are seeded once at startup so an earlier unsuccessful
        upload cannot silently disappear. After acknowledgement this is O(new
        records), without a full directory scan after every scientific fit.
        """
        return tuple(sorted(self._pending_record_paths))

    def mark_records_synced(self,paths):
        """Acknowledge only paths whose remote commit has been confirmed."""
        self._assert_open()
        if not isinstance(paths,(tuple,list)) or any(type(p) is not str for p in paths):
            raise InvalidPlan("Acknowledged record paths must be an explicit string list")
        if any(p not in self._pending_record_paths for p in paths):
            raise InvalidPlan("Cannot acknowledge an unknown campaign record")
        self._pending_record_paths.difference_update(paths)

    def _utc(self):
        return datetime.now(timezone.utc).isoformat()

    def _now(self):
        value=self._clock()
        if isinstance(value,bool) or type(value) not in (int,float) or not math.isfinite(value):
            raise InvalidPlan("Clock must return finite monotonic seconds")
        if self._last_clock is not None and value<self._last_clock:
            raise InvalidPlan("Clock moved backwards")
        self._last_clock=float(value)
        return float(value)

    def _validate_config(self,record):
        value=_checked_hash(record)
        if (value.get("version")!=VERSION or value.get("plan")!=self._plan or
                value.get("plan_sha256")!=self.plan_sha256 or value.get("controls")!=self._controls):
            raise InvalidPlan("Plan or original campaign controls differ from the sealed campaign")

    def _acquire_lock(self):
        path=self.directory/".campaign.lock"
        _no_symlinks(path)
        descriptor=os.open(path,os.O_RDWR|os.O_CREAT|getattr(os,"O_NOFOLLOW",0),0o600)
        self._lock=os.fdopen(descriptor,"r+b")
        try:
            fcntl.flock(self._lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            self._lock_held=True
        except BlockingIOError as error:
            raise CampaignLocked("Another process holds this campaign") from error
        except OSError as error:
            raise LockUnavailable("Filesystem does not support required exclusive campaign locking") from error

    def _release_lock(self):
        if self._lock is not None:
            try:
                if self._lock_held:
                    fcntl.flock(self._lock.fileno(),fcntl.LOCK_UN)
            finally:
                self._lock.close()
                self._lock=None
                self._lock_held=False

    def _fsync_directory(self,path):
        descriptor=None
        try:
            descriptor=os.open(path,os.O_RDONLY|getattr(os,"O_DIRECTORY",0))
            os.fsync(descriptor)
        except OSError as error:
            if error.errno not in (errno.EINVAL,errno.ENOTSUP,errno.EOPNOTSUPP,errno.EBADF):
                raise
            self._directory_fsync_supported=False
            if self.require_directory_fsync:
                raise LockUnavailable("Directory fsync unsupported; a relaxed FUSE policy must be explicit") from error
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _scan_storage(self):
        size=0
        for path in self.directory.rglob("*"):
            _no_symlinks(path)
            if path.is_file():
                size+=path.stat().st_size
        return size

    def refresh_storage_usage(self):
        """Explicit scan for caller-created artifacts; avoid O(n²) per-fit scans."""
        self._storage_bytes=self._scan_storage()
        self._external_bytes=self._scan_external_storage()
        return self._storage_bytes

    def _scan_external_storage(self):
        """Engine-owned append-only records have O(1) byte accounting."""
        total=0
        managed={"events","results","sessions","campaign.json",".campaign.lock"}
        for path in self.directory.iterdir():
            if path.name in managed:
                continue
            _no_symlinks(path)
            if path.is_file():
                total+=path.stat().st_size
            elif path.is_dir():
                for child in path.rglob("*"):
                    _no_symlinks(child)
                    if child.is_file():
                        total+=child.stat().st_size
        return total

    def _refresh_external_storage(self):
        total=self._scan_external_storage()
        self._storage_bytes+=total-self._external_bytes
        self._external_bytes=total

    def _ensure_space(self,additional):
        if self._storage_bytes+additional>self.max_storage_bytes:
            raise StorageLimit("Campaign storage admission limit reached")
        if shutil.disk_usage(self.directory).free<additional:
            raise StorageLimit("Filesystem lacks space for a durable campaign record")

    def _write_new(self,path,value):
        _no_symlinks(path)
        if path.exists():
            raise InvalidPlan("Refusing to overwrite an existing campaign record")
        payload=canonical(value)+b"\n"
        self._ensure_space(len(payload))
        descriptor,temporary=tempfile.mkstemp(prefix=".pending-",dir=path.parent)
        # Failed temporary files are retained for reconciliation, not discarded.
        with os.fdopen(descriptor,"wb") as stream:
            stream.write(payload);stream.flush();os.fsync(stream.fileno())
        if path.exists() or path.is_symlink():
            raise InvalidPlan("A campaign target appeared while its lock was held")
        os.rename(temporary,path)  # Single-writer lock makes target publication exclusive.
        self._fsync_directory(path.parent)
        self._storage_bytes+=len(payload)
        self._pending_record_paths.add(path.relative_to(self.directory).as_posix())

    def _append_event(self,kind,**fields):
        value={"version":VERSION,"sequence":len(self._events),"kind":kind,
               "plan_sha256":self.plan_sha256,"previous_sha256":self._events[-1]["sha256"] if self._events else ZERO,
               "utc":self._utc(),**_json_value(fields)}
        value["sha256"]=digest(value)
        self._write_new(self.directory/"events"/("%09d.json" % len(self._events)),value)
        self._events.append(value)
        return value

    def _load(self):
        self._pending_record_paths.add("campaign.json")
        for name in ("events","results","sessions"):
            path=self.directory/name
            _no_symlinks(path)
            if not path.is_dir():
                raise InvalidPlan("Campaign record directories are missing")
        for index,path in enumerate(sorted((self.directory/"events").glob("*.json"))):
            if path.name!="%09d.json" % index:
                raise InvalidPlan("Campaign event sequence is incomplete")
            row=_read(path);value=_checked_hash(row)
            previous=self._events[-1]["sha256"] if self._events else ZERO
            if value.get("version")!=VERSION or value.get("sequence")!=index or value.get("plan_sha256")!=self.plan_sha256 or value.get("previous_sha256")!=previous:
                raise InvalidPlan("Campaign event chain changed")
            kind=value.get("kind")
            if kind=="session_started":
                identity=value["session_id"]
                if identity in self._sessions:
                    raise InvalidPlan("Duplicate session identity")
                _number(value["session_budget_seconds"],"Recorded session budget")
                self._sessions[identity]=row
            elif kind=="job_started":
                job=value["job_id"]
                if job in self._started or value.get("session_id") not in self._sessions:
                    raise InvalidPlan("Duplicate or unbound attempt start")
                _number(value["reservation_seconds"],"Recorded job reservation")
                self._started[job]=row
            elif kind=="campaign_budget_extended":
                budget=_number(value["new_budget_seconds"],"Amended campaign budget")
                if value.get("previous_budget_seconds")!=self._effective_budget or budget<=self._effective_budget or not value.get("reason"):
                    raise InvalidPlan("Invalid campaign budget amendment")
                self._effective_budget=budget
            elif kind=="session_closed":
                identity=value["session_id"]
                if identity not in self._sessions or identity in self._finished_sessions:
                    raise InvalidPlan("Invalid closed session")
                overhead=value.get("charged_overhead_seconds")
                if type(overhead) not in (int,float) or isinstance(overhead,bool) or not math.isfinite(overhead) or overhead<0:
                    raise InvalidPlan("Invalid session overhead charge")
                self._charged+=overhead
                self._finished_sessions.add(identity)
            elif kind not in ("job_finished","job_interrupted","session_paused"):
                raise InvalidPlan("Unknown campaign event type")
            self._events.append(row)
            self._pending_record_paths.add(path.relative_to(self.directory).as_posix())
        finished={}
        for row in self._events:
            if row["kind"]=="job_finished":
                job=row["job_id"]
                if job in finished:
                    raise InvalidPlan("Duplicate finished attempt")
                finished[job]=row
        charges_by_session={identity:0.0 for identity in self._sessions}
        for path in sorted((self.directory/"results").glob("*.json")):
            record=_read(path);value=_checked_hash(record)
            job=value.get("job_id");start=self._started.get(job)
            if (not start or path.name!=self._job_filename(job) or value.get("version")!=VERSION or
                    value.get("plan_sha256")!=self.plan_sha256 or value.get("started_event_sha256")!=start["sha256"] or
                    value.get("job_payload_sha256")!=start["job_payload_sha256"] or
                    value.get("session_id")!=start["session_id"] or value.get("outcome_sha256")!=digest(value.get("outcome"))):
                raise InvalidPlan("Committed result does not match its sealed attempt")
            charge=value.get("charged_wall_seconds")
            if type(charge) not in (int,float) or isinstance(charge,bool) or not math.isfinite(charge) or charge<0:
                raise InvalidPlan("Invalid committed attempt charge")
            finish=finished.get(job)
            if finish and finish.get("result_sha256")!=record["sha256"]:
                raise InvalidPlan("Finished event differs from committed result")
            if not finish:
                self._recovered_results.append(job)
            self._results[job]=record
            self._pending_record_paths.add(path.relative_to(self.directory).as_posix())
            self._charged+=charge
            charges_by_session[value["session_id"]]+=charge
        if any(job not in self._results for job in finished):
            raise InvalidPlan("Finished result file is missing")
        # A cleanly closed session has exact known charges. A process that died
        # after committing its last result may have unrecorded idle/bookkeeping
        # time; charge its full allocation conservatively, without replaying fits.
        for identity,start in self._sessions.items():
            if identity not in self._finished_sessions:
                extra=max(0.0,start["session_budget_seconds"]-charges_by_session[identity])
                self._unclosed_allocations[identity]=extra
                self._charged+=extra

    def _assert_open(self):
        if self._closed:
            raise SessionError("Search session is closed")

    @staticmethod
    def _job_filename(job_id):
        return hashlib.sha256(job_id.encode("utf-8")).hexdigest()+".json"

    def get_result(self,job_id,job_payload=None):
        result=self._results.get(job_id)
        if result is None:
            return None
        if job_payload is not None and digest(job_payload)!=result["job_payload_sha256"]:
            raise InvalidPlan("A completed job identity was reused with different inputs")
        return deepcopy(result["outcome"])

    def pending_jobs(self,job_ids):
        return tuple(job for job in job_ids if job not in self._results)

    def require_complete(self,job_ids):
        if self.pending_jobs(job_ids):
            raise IncompleteCampaign("A stage with pending jobs cannot select a winner")

    def extend_campaign_budget(self,new_seconds,*,reason):
        self._assert_open()
        value=_number(new_seconds,"New campaign budget")
        if value<=self._effective_budget or type(reason) is not str or not reason.strip() or len(reason)>2048:
            raise InvalidPlan("Budget extensions require a strict increase and an explicit reason")
        self._append_event("campaign_budget_extended",previous_budget_seconds=self._effective_budget,
                           new_budget_seconds=value,reason=reason)
        self._effective_budget=value
        if self._paused=="campaign_budget":
            self._paused=None

    def _start_session(self):
        self._session_id=uuid.uuid4().hex
        self._session_start=self._charge_point=self._now()
        self._session_initial_charge=self._charged
        row=self._append_event("session_started",session_id=self._session_id,
            session_budget_seconds=self.session_budget_seconds,campaign_budget_seconds=self._effective_budget,
            charged_before_session_seconds=self._charged,
            directory_fsync_required=self.require_directory_fsync,
            directory_fsync_supported=self._directory_fsync_supported)
        self._sessions[self._session_id]=row

    def _pause(self,reason,**details):
        self._paused=reason
        if self._session_id is not None:
            self._append_event("session_paused",session_id=self._session_id,reason=reason,details=details)
        raise SessionPaused(reason,details)

    def run_job(self,job_id,job_payload,callback,*,reservation_seconds):
        self._assert_open()
        if type(job_id) is not str or not job_id or len(job_id.encode("utf-8"))>4096:
            raise InvalidPlan("Job identity must be a nonempty bounded string")
        payload=_json_value(job_payload)
        reservation=_number(reservation_seconds,"Job reservation")
        if job_id in self._results:
            if self._results[job_id]["reservation_seconds"]!=reservation:
                raise InvalidPlan("A completed job's scientific time reservation changed")
            return self.get_result(job_id,payload)
        if set(self._started)-set(self._results):
            raise UnresolvedAttempt(set(self._started)-set(self._results))
        if self._paused:
            raise SessionPaused(self._paused)
        self._refresh_external_storage()
        # Impossible reservations and storage refusal do not open a session.
        if reservation>self.session_budget_seconds:
            self._pause("session_budget",reservation_seconds=reservation,remaining_seconds=self.session_budget_seconds)
        if reservation>self._effective_budget-self._charged:
            self._pause("campaign_budget",reservation_seconds=reservation,remaining_seconds=self._effective_budget-self._charged)
        try:
            self._ensure_space(self.completion_reserve_bytes)
        except StorageLimit:
            self._pause("storage_budget",storage_used_bytes=self._storage_bytes)
        if self._session_id is None:
            self._start_session()
        now=self._now()
        gap=now-self._charge_point
        session_remaining=self.session_budget_seconds-(now-self._session_start)
        campaign_remaining=self._effective_budget-self._charged-gap
        if reservation>session_remaining:
            self._pause("session_budget",reservation_seconds=reservation,remaining_seconds=session_remaining)
        if reservation>campaign_remaining:
            self._pause("campaign_budget",reservation_seconds=reservation,remaining_seconds=campaign_remaining)
        deadline=min(now+reservation,self._session_start+self.session_budget_seconds)
        start=self._append_event("job_started",job_id=job_id,session_id=self._session_id,
            job_payload_sha256=digest(payload),reservation_seconds=reservation)
        self._started[job_id]=start
        callback_at=self._now()
        try:
            if callback_at>=deadline:
                raise TimeoutError("Reserved time elapsed before callback launch")
            outcome=_json_value(callback(deadline))
            stopped=self._now()
            charge=stopped-self._charge_point
            envelope={"version":VERSION,"plan_sha256":self.plan_sha256,"job_id":job_id,
                "job_payload_sha256":digest(payload),"session_id":self._session_id,
                "started_event_sha256":start["sha256"],"outcome":outcome,"outcome_sha256":digest(outcome),
                "charged_wall_seconds":charge,"callback_wall_seconds":stopped-callback_at,
                "reservation_seconds":reservation,"reservation_overrun":stopped>deadline,
                "session_elapsed_seconds":stopped-self._session_start}
            envelope["sha256"]=digest(envelope)
            self._refresh_external_storage()
            self._write_new(self.directory/"results"/self._job_filename(job_id),envelope)
        except BaseException as error:
            # A durable result may exist if directory fsync or the caller's
            # artificial crash occurred after rename. Constructor validates it
            # on recovery; this live instance stops instead of retrying.
            try:
                self._append_event("job_interrupted",job_id=job_id,session_id=self._session_id,
                    exception_type=type(error).__name__,known_elapsed_seconds=max(0.0,self._now()-callback_at))
            except BaseException:
                pass
            raise UnresolvedAttempt([job_id],{"exception_type":type(error).__name__}) from error
        self._results[job_id]=envelope
        self._charged+=charge
        self._charge_point=stopped
        # Outcome is already the commit marker. If this redundant event cannot
        # be written, recovery still uses the validated result without a refit.
        self._append_event("job_finished",job_id=job_id,session_id=self._session_id,result_sha256=envelope["sha256"])
        if envelope["reservation_overrun"]:
            self._paused="job_reservation_overrun"
            raise SessionPaused("job_reservation_overrun",{"charged_wall_seconds":charge},completed_result=outcome)
        return deepcopy(outcome)

    def summary(self):
        return {"version":VERSION,"plan_sha256":self.plan_sha256,"completed_jobs":len(self._results),
                "charged_wall_seconds":self._charged,"campaign_budget_seconds":self._effective_budget,
                "remaining_campaign_seconds":max(0.0,self._effective_budget-self._charged),
                "storage_used_bytes":self._storage_bytes,"max_storage_bytes":self.max_storage_bytes,
                "recovered_result_commits":self._recovered_results[:],
                "conservative_unclosed_session_charges":deepcopy(self._unclosed_allocations),
                "directory_fsync_supported":self._directory_fsync_supported,
                "paused_reason":self._paused}

    def close(self):
        if self._closed:
            return
        try:
            if self._session_id is not None:
                stopped=self._now();overhead=max(0.0,stopped-self._charge_point)
                self._append_event("session_closed",session_id=self._session_id,
                    session_elapsed_seconds=stopped-self._session_start,
                    charged_overhead_seconds=overhead,paused_reason=self._paused,
                    directory_fsync_supported=self._directory_fsync_supported)
                self._charged+=overhead
                self._finished_sessions.add(self._session_id)
        finally:
            self._closed=True
            self._release_lock()

    def __enter__(self):
        self._assert_open()
        return self

    def __exit__(self,kind,value,traceback):
        try:
            self.close()
        except BaseException:
            if kind is None:
                raise
        return False
