"""Shared, exactly-once accounting for newly registered reduced E2 workers.

This adapter keeps the immutable historical ledgers and their conservative
charges. Its shorter reservations apply only after an explicit new migration;
they never reinterpret old attempts, grant new hours, or authorize retries.
The lease is cooperative on one canonical filesystem/runtime, not a global
distributed lock across disconnected Drive mounts.
"""
from __future__ import annotations

from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import re
import time

try:
    from . import e2_amended_pilot_budget as historical
except ImportError:
    import e2_amended_pilot_budget as historical

VERSION = "e2-reduced-shared-budget-v1"
MIGRATION_VERSION = "e2-reduced-worker-budget-migration-v1"
BudgetError = historical.BudgetError
BudgetPause = historical.BudgetPause
BudgetReconciliationRequired = historical.BudgetReconciliationRequired
canonical = historical.canonical
digest = historical.digest


def _json(path):
    return historical._read(path)


def _number(value, *, positive=False):
    return historical._number(value, positive=positive)


def validate_migration(migration, reconciliation, registration_commit):
    historical._checked(reconciliation)
    value = historical._checked(migration)
    if not isinstance(registration_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", registration_commit):
        raise BudgetError("actual_migration_registration_commit_required")
    required = {"version": MIGRATION_VERSION, "authorized_total_hours": 64,
        "budget_reset": False, "budget_extension_authorized": False,
        "historical_reconciliation_sha256": reconciliation["sha256"],
        "scope": "new_reduced_jobs_only", "stage_wall_seconds": 600,
        "worker_overhead_seconds": 60, "max_optimization_phases": 3,
        "max_epochs": 200, "patience_epochs": 20}
    if any(value.get(key) != expected for key, expected in required.items()):
        raise BudgetError("registered_reduced_worker_migration_required")
    if (reconciliation.get("budget_reset") is not False
            or reconciliation.get("authorized_combined_hours") != 64
            or reconciliation.get("authorized_combined_seconds") != 230400):
        raise BudgetError("unchanged_shared_64_hour_ceiling_required")
    return deepcopy(migration)


def inspect_inherited_ledger(path, reconciliation):
    """Replay an optional earlier local pilot ledger without mutating it."""
    path = historical._no_links(path)
    if path.is_dir():
        path = path / "pilot_budget.jsonl"
    if not path.exists():
        return {"path": str(path), "status": "absent_at_migration", "charged_seconds": 0.,
                "reserved_seconds": 0., "records": 0, "sha256": None}
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise BudgetReconciliationRequired("inherited_pilot_ledger_has_partial_tail")
    reader = object.__new__(historical.PilotBudgetLedger)
    reader._events, reader._active, reader._charges = [], {}, 0.
    reader._attempts, reader._progress, reader.reconciliation = set(), {}, deepcopy(reconciliation)
    for index, line in enumerate(raw.splitlines()):
        record = json.loads(line, object_pairs_hook=historical._pairs)
        value = historical._checked(record)
        if (value.get("version") != historical.VERSION or value.get("sequence") != index
                or value.get("previous_sha256") != (reader._events[-1]["sha256"] if index else historical.ZERO)
                or value.get("reconciliation_sha256") != reconciliation["sha256"]):
            raise BudgetReconciliationRequired("inherited_pilot_chain_or_anchor_changed")
        if value.get("kind") == "charged" and value.get("status") not in ("completed", "failed", "paused_clean"):
            raise BudgetReconciliationRequired("unknown_inherited_attempt_cannot_release_reservation")
        reader._apply(record)
    if path.read_bytes() != raw:
        raise BudgetReconciliationRequired("inherited_pilot_ledger_changed_while_reading")
    import hashlib
    return {"path": str(path), "status": "replayed", "charged_seconds": reader._charges,
        "reserved_seconds": sum(reader._active.values()), "unresolved_reservations": dict(reader._active),
        "records": len(reader._events), "sha256": hashlib.sha256(raw).hexdigest()}


def _write_new(path, value):
    path = historical._no_links(path)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush(); os.fsync(stream.fileno())


class ReducedBudgetLedger:
    """One canonical ledger shared sequentially by HPO, refit and final runners.

    The caller verifies the Git registration before constructing this object
    and bounds the *entire* child process using worker_timeout_for(phases).
    All measured time since reserve is charged, including startup, IO, scoring
    and termination. Time between workers is charged automatically as overhead.
    """
    def __init__(self, root, reconciliation, *, migration, migration_registration_commit,
                 inherited_ledgers=(), clock=time.monotonic):
        self._clock, self._inner, self._inherited_locks = clock, None, []
        self._closed, self._gap_started, self._active_started = False, None, {}
        self._close_warning = None
        began = clock()
        self.root = historical._no_links(root)
        self.reconciliation = deepcopy(reconciliation)
        self.migration = validate_migration(migration, reconciliation, migration_registration_commit)
        self.migration_registration_commit = migration_registration_commit
        self._inner = historical.PilotBudgetLedger(self.root, reconciliation)
        try:
            # Earlier cooperative writers use these same files. Missing earlier
            # ledgers are recorded explicitly and their parent is not created.
            inherited = []
            seen = set()
            for supplied in inherited_ledgers:
                path = historical._no_links(supplied)
                if path.is_dir():
                    path = path / "pilot_budget.jsonl"
                if path == self.root / "pilot_budget.jsonl" or str(path) in seen:
                    raise BudgetError("duplicate_or_self_inherited_budget_path")
                seen.add(str(path))
                if path.exists():
                    descriptor = os.open(historical._no_links(path.parent / ".pilot-budget.lock"),
                        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
                    lock = os.fdopen(descriptor, "r+b")
                    try:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except OSError:
                        lock.close()
                        raise BudgetError("an_earlier_local_pilot_controller_is_active") from None
                    self._inherited_locks.append(lock)
                inherited.append(inspect_inherited_ledger(path, reconciliation))
            if any(row["reserved_seconds"] for row in inherited):
                raise BudgetReconciliationRequired("unresolved_inherited_attempt_preserves_full_reservation")
            self._inherited = inherited
            anchor = {"version": VERSION, "historical_reconciliation_sha256": reconciliation["sha256"],
                "migration_sha256": migration["sha256"], "migration_registration_commit": migration_registration_commit,
                "inherited_ledgers": inherited, "authorized_total_hours": 64, "budget_reset": False}
            anchor["sha256"] = digest(anchor)
            anchor_path = self.root / "reduced_migration.json"
            if anchor_path.exists():
                if _json(anchor_path) != anchor:
                    raise BudgetReconciliationRequired("shared_budget_anchor_policy_or_inherited_history_changed")
            else:
                if self._inner._events:
                    raise BudgetReconciliationRequired("existing_unbound_budget_cannot_be_adopted")
                _write_new(anchor_path, anchor)
            self._anchor = anchor
            marker = "inherited_local_ledger_charges:" + digest(inherited)
            inherited_total = sum(row["charged_seconds"] for row in inherited)
            if not self._inner._events:
                self._inner.charge_overhead(inherited_total, description=marker)
            first = self._inner._events[0]
            if (first.get("kind") != "overhead" or first.get("description") != marker
                    or not math.isclose(first.get("elapsed_seconds", -1), inherited_total, abs_tol=1e-9)):
                raise BudgetReconciliationRequired("inherited_budget_transfer_not_durable_or_changed")
            # Existing complete ledgers must retain a newline-delimited tail.
            raw = (self.root / "pilot_budget.jsonl").read_bytes()
            if raw and not raw.endswith(b"\n"):
                raise BudgetReconciliationRequired("shared_budget_has_partial_tail")
            self._logical_attempts = {}
            for row in self._inner._events:
                if row["kind"] == "reserved":
                    logical = row.get("metadata", {}).get("logical_job_id", row["attempt_id"])
                    if logical in self._logical_attempts:
                        raise BudgetReconciliationRequired("a_logical_fit_was_repeated")
                    self._logical_attempts[logical] = row["attempt_id"]
            self._gap_started = began
            self._charge_gap("reduced_controller_initialization")
        except BaseException:
            self._release()
            raise

    @property
    def stage_wall_seconds(self):
        return self.migration["stage_wall_seconds"]

    def reservation_for(self, phases=1):
        if type(phases) is not int or phases not in (1, 2, 3):
            raise BudgetError("one_to_three_optimizer_phases_required")
        return self.stage_wall_seconds * phases + self.migration["worker_overhead_seconds"]

    def worker_timeout_for(self, phases=1):
        # Reserve five seconds for terminating/reaping the child and persisting
        # its receipt. The trainer separately checks600s for each optimizer phase.
        return self.reservation_for(phases) - 5

    def _ensure_open(self):
        if self._closed:
            raise BudgetError("reduced_shared_budget_is_closed")

    def _pending_gap(self):
        return 0. if self._gap_started is None else max(0., self._clock()-self._gap_started)

    def _charge_gap(self, description):
        if self._gap_started is not None:
            amount = self._pending_gap()
            self._gap_started = None
            self._inner.charge_overhead(amount, description=description)
            self._gap_started = self._clock()

    @property
    def remaining_seconds(self):
        return max(0., self._inner.remaining_seconds - self._pending_gap() - self._known_unclosed_overrun())

    def _known_unclosed_overrun(self):
        extra = 0.
        for identity, reservation in self._inner._active.items():
            lower = self._inner._progress.get(identity, {}).get("elapsed_seconds_lower_bound", 0.)
            if not self._closed and identity in self._active_started:
                lower = max(lower, self._clock()-self._active_started[identity])
            extra += max(0., lower-reservation)
        return extra

    def reserve(self, attempt_id, seconds, *, metadata=None):
        self._ensure_open()
        if self._active_started:
            raise BudgetReconciliationRequired("finish_the_current_worker_before_another_reservation")
        self._charge_gap("reduced_controller_between_workers")
        amount = _number(seconds, positive=True)
        if amount not in {self.reservation_for(p) for p in (1, 2, 3)}:
            raise BudgetError("reservation_must_equal_the_registered_whole_worker_bound")
        metadata = {} if metadata is None else deepcopy(metadata)
        logical = metadata.get("logical_job_id", attempt_id)
        if not isinstance(logical, str) or not logical or logical in self._logical_attempts:
            raise BudgetReconciliationRequired("logical_fit_already_attempted_no_automatic_retry")
        metadata.update(logical_job_id=logical, reduced_migration_sha256=self.migration["sha256"],
            migration_registration_commit=self.migration_registration_commit)
        result = self._inner.reserve(attempt_id, amount, metadata=metadata)
        self._logical_attempts[logical] = attempt_id
        self._gap_started = None
        self._active_started[attempt_id] = self._clock()
        return result

    def charge(self, attempt_id, elapsed_seconds, *, status, resource_use=None):
        self._ensure_open()
        if attempt_id not in self._active_started:
            raise BudgetReconciliationRequired("worker_does_not_belong_to_this_live_lease")
        amount = max(_number(elapsed_seconds), self._clock()-self._active_started[attempt_id])
        try:
            result = self._inner.charge(attempt_id, amount, status=status, resource_use=resource_use)
        finally:
            # Unknown status intentionally leaves the underlying reservation.
            if attempt_id not in self._inner._active:
                del self._active_started[attempt_id]
                self._gap_started = self._clock()
        return result

    def charge_overhead(self, elapsed_seconds, *, description="reduced_controller_overhead"):
        self._ensure_open()
        if self._active_started:
            raise BudgetError("worker_elapsed_already_includes_controller_overhead")
        amount = max(_number(elapsed_seconds), self._pending_gap())
        self._gap_started = None
        try:
            return self._inner.charge_overhead(amount, description=description)
        finally:
            self._gap_started = self._clock()

    def record_progress(self, attempt_id, elapsed_seconds_lower_bound, optimizer_steps_lower_bound):
        self._ensure_open()
        if attempt_id not in self._active_started:
            raise BudgetError("progress_requires_a_live_reserved_worker")
        elapsed = max(_number(elapsed_seconds_lower_bound), self._clock()-self._active_started[attempt_id])
        return self._inner.record_progress(attempt_id, elapsed, optimizer_steps_lower_bound)

    def completed_receipt(self, attempt_id, *, expected_resource_use=None):
        """Reusable artifacts need their corresponding durable successful cost."""
        matches = [row for row in self._inner._events if row["kind"] == "charged" and row["attempt_id"] == attempt_id]
        if not matches:
            if attempt_id in self._inner._attempts:
                raise BudgetReconciliationRequired("existing_attempt_has_no_completed_cost_receipt")
            return None
        row = matches[-1]
        if row["status"] != "completed" or row.get("reservation_overrun") is True:
            raise BudgetReconciliationRequired("failed_or_paused_fit_cannot_be_retried_automatically")
        for key, value in (expected_resource_use or {}).items():
            if row.get("resource_use", {}).get(key) != value:
                raise BudgetReconciliationRequired("artifact_and_cost_receipt_binding_changed")
        return deepcopy(row)

    def report(self):
        base = self._inner.report()
        return {**base, "version": VERSION, "migration_sha256": self.migration["sha256"],
            "migration_registration_commit": self.migration_registration_commit,
            "stage_wall_seconds": self.stage_wall_seconds,
            "worker_overhead_seconds": self.migration["worker_overhead_seconds"],
            "pending_controller_overhead_seconds": self._pending_gap(),
            "known_unclosed_overrun_seconds": self._known_unclosed_overrun(),
            "reserved_seconds": base["reserved_seconds"] + self._known_unclosed_overrun(),
            "remaining_seconds": self.remaining_seconds,
            "inherited_local_ledger_seconds": sum(row["charged_seconds"] for row in self._inherited),
            "lease_scope": "One canonical filesystem/runtime; other disconnected Drive mounts are not locked",
            "last_close_warning": self._close_warning}

    def _release(self):
        if self._inner is not None:
            self._inner.close()
        for lock in self._inherited_locks:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            finally:
                lock.close()
        self._inherited_locks = []
        self._closed = True

    def close(self):
        if self._closed:
            return
        try:
            if self._active_started:
                for identity, started in self._active_started.items():
                    prior = self._inner._progress.get(identity, {})
                    self._inner.record_progress(identity,
                        max(self._clock()-started, prior.get("elapsed_seconds_lower_bound", 0.)),
                        prior.get("optimizer_steps_lower_bound", 0))
            else:
                self._charge_gap("reduced_controller_close")
                self._gap_started = None
        except BudgetError as error:
            self._close_warning = str(error)
        finally:
            self._release()

    def __enter__(self):
        self._ensure_open(); return self

    def __exit__(self, *unused):
        self.close()


SharedBudgetLedger = ReducedBudgetLedger
