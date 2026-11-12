"""Read-only reconciliation and a durable, shared-cap pilot budget.

Historical ledgers remain immutable. Replay verifies every event and result,
including conservatively charged unclosed allocations. This module cannot
authorize an increase above the already registered combined 64-hour ceiling.
The local lock coordinates one runtime on one filesystem; a caller must also
pin/recheck remote heads and must not run another Colab on a separate mount.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import io
import stat
import subprocess
import tempfile
import urllib.error
import urllib.request
import zipfile

VERSION = "e2-amended-pilot-budget-v1"
LEGACY_BRANCH = "campaigns/e2-cuda-selection-v1"
OPTIMIZED_BRANCH = "campaigns/e2-cuda-crop-11-v2"
LEGACY_PLAN = "4eae2d2ff9ef6f83a2f9eb0dc5ba647208d695e1665a8e522d812fed67893bda"
OPTIMIZED_PLAN = "877a8ab4a0d34599f43c5dd16a90fe87fe03d37fe535466bf8e009c379cce742"
AUTHORIZED_TOTAL_HOURS = 64.0
ZERO = "0" * 64
HEX40 = re.compile(r"[0-9a-f]{40}")
REPOSITORY = "jassan-bit/DeepLearning_202630"


class BudgetError(RuntimeError):
    pass


class BudgetPause(BudgetError):
    pass


class BudgetReconciliationRequired(BudgetError):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _number(value, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (positive and value <= 0)):
        raise BudgetError("nonnegative_finite_budget_number_required")
    return float(value)


def _same(left, right):
    return math.isclose(_number(left), _number(right), rel_tol=0, abs_tol=1e-6)


def _no_links(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise BudgetError("symlink_budget_path_refused")
    return path


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BudgetError("duplicate_json_key")
        result[key] = value
    return result


def _read(path):
    path = _no_links(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
        canonical(value)
        return value
    except (OSError, ValueError, TypeError):
        raise BudgetError("invalid_budget_record") from None


def _checked(value):
    if type(value) is not dict:
        raise BudgetError("hashed_budget_object_required")
    clean = deepcopy(value)
    claimed = clean.pop("sha256", None)
    if claimed != digest(clean):
        raise BudgetError("budget_record_hash_changed")
    return clean


def _records(directory, name):
    folder = _no_links(directory / name)
    if not folder.is_dir():
        raise BudgetError("missing_budget_record_directory")
    paths = sorted(folder.iterdir())
    if any(not p.is_file() or p.suffix != ".json" or p.is_symlink() for p in paths):
        raise BudgetError("unexpected_budget_record_file")
    return paths


def replay_campaign(directory, *, branch, head_sha, expected_plan_sha256=None,
                    verify_inventory=True):
    """Validate a pinned local mirror without creating a session or lock.

    ``head_sha`` is an externally pinned Git commit, not proof of remote state.
    Reconciliation separately requires both observed current branch heads. A
    public snapshot manifest, when required, binds every file byte by Merkle
    inventory. Outcomes are hashed; their predictions are never interpreted.
    """
    directory = _no_links(directory)
    if branch not in (LEGACY_BRANCH, OPTIMIZED_BRANCH) or not HEX40.fullmatch(head_sha):
        raise BudgetError("pinned_campaign_branch_and_commit_required")
    campaign = _read(directory / "campaign.json")
    config = _checked(campaign)
    plan = config["plan"]
    plan_sha = digest(plan)
    if (config.get("version") != "e2-search-session-v1"
            or config.get("plan_sha256") != plan_sha
            or (expected_plan_sha256 is not None and plan_sha != expected_plan_sha256)
            or plan.get("remote_storage", {}).get("branch") != branch):
        raise BudgetError("sealed_campaign_plan_changed")
    effective = _number(config["controls"]["campaign_budget_seconds"], positive=True)
    events, sessions, started, finished, closed, extensions = [], {}, {}, {}, set(), []
    overhead = 0.0
    for index, path in enumerate(_records(directory, "events")):
        if path.name != "%09d.json" % index:
            raise BudgetError("campaign_event_sequence_incomplete")
        record = _read(path)
        row = _checked(record)
        if (row.get("version") != "e2-search-session-v1"
                or row.get("plan_sha256") != plan_sha or row.get("sequence") != index
                or row.get("previous_sha256") != (events[-1]["sha256"] if events else ZERO)):
            raise BudgetError("campaign_event_chain_changed")
        kind = row.get("kind")
        if kind == "session_started":
            identity = row["session_id"]
            if identity in sessions:
                raise BudgetError("duplicate_campaign_session")
            _number(row["session_budget_seconds"], positive=True)
            if not _same(row["campaign_budget_seconds"], effective):
                raise BudgetError("session_campaign_cap_changed")
            sessions[identity] = record
        elif kind == "job_started":
            identity = row["job_id"]
            if identity in started or row.get("session_id") not in sessions:
                raise BudgetError("duplicate_or_unbound_campaign_attempt")
            _number(row["reservation_seconds"], positive=True)
            started[identity] = record
        elif kind == "job_finished":
            identity = row["job_id"]
            # Registered external recoveries may append a finish in their own
            # receipt session. The durable result still binds the ORIGINAL
            # start/session below; changing that binding remains forbidden.
            if (identity in finished or identity not in started
                    or row.get("session_id") not in sessions):
                raise BudgetError("invalid_campaign_finish")
            finished[identity] = record
        elif kind == "campaign_budget_extended":
            new = _number(row["new_budget_seconds"], positive=True)
            if not _same(row["previous_budget_seconds"], effective) or new <= effective or not row.get("reason"):
                raise BudgetError("invalid_campaign_budget_extension")
            effective = new
            extensions.append({"sha256": record["sha256"], "new_budget_seconds": new,
                               "reason": row["reason"]})
        elif kind == "session_closed":
            identity = row["session_id"]
            if identity not in sessions or identity in closed:
                raise BudgetError("invalid_campaign_session_close")
            overhead += _number(row["charged_overhead_seconds"])
            closed.add(identity)
        elif kind in ("job_interrupted", "session_paused"):
            if row.get("session_id") not in sessions:
                raise BudgetError("unbound_campaign_pause")
        else:
            raise BudgetError("unknown_campaign_event")
        events.append(record)
    results, charges_by_session, charged = {}, dict.fromkeys(sessions, 0.0), overhead
    for path in _records(directory, "results"):
        record = _read(path)
        row = _checked(record)
        identity = row.get("job_id")
        start = started.get(identity)
        if (not start or identity in results
                or path.name != hashlib.sha256(identity.encode()).hexdigest() + ".json"
                or row.get("version") != "e2-search-session-v1"
                or row.get("plan_sha256") != plan_sha
                or row.get("started_event_sha256") != start["sha256"]
                or row.get("job_payload_sha256") != start["job_payload_sha256"]
                or row.get("session_id") != start["session_id"]
                or row.get("outcome_sha256") != digest(row.get("outcome"))
                or (identity in finished and finished[identity]["result_sha256"] != record["sha256"])):
            raise BudgetError("campaign_result_binding_changed")
        charge = _number(row["charged_wall_seconds"])
        charged += charge
        charges_by_session[row["session_id"]] += charge
        results[identity] = record
    if set(finished) - set(results):
        raise BudgetError("finished_campaign_result_missing")
    conservative = {}
    for identity in set(sessions) - closed:
        conservative[identity] = max(0.0, sessions[identity]["session_budget_seconds"] - charges_by_session[identity])
        charged += conservative[identity]
    unresolved = sorted(set(started) - set(results))
    inventory_sha = None
    if verify_inventory:
        from e2_github_campaign import _Inventory, _descriptor
        manifest_record = _read(directory / "snapshot.json")
        manifest = _checked(manifest_record)
        inventory = _Inventory()
        paths = [directory / "campaign.json", directory / "status.json"]
        paths += _records(directory, "events") + _records(directory, "results")
        for path in paths:
            _no_links(path)
            relative = path.relative_to(directory).as_posix()
            inventory.set(relative, _descriptor(relative, path.read_bytes()))
        if (manifest.get("repository") != "jassan-bit/DeepLearning_202630"
                or manifest.get("branch") != branch or manifest.get("prefix") != branch
                or manifest.get("plan_sha256") != plan_sha
                or manifest.get("inventory_algorithm") != "sha256-path-bucket-merkle-v1"
                or manifest.get("record_count") != len(inventory.records)
                or manifest.get("record_bytes") != inventory.bytes
                or manifest.get("inventory_root_sha256") != inventory.root):
            raise BudgetError("campaign_snapshot_inventory_changed")
        inventory_sha = manifest_record["sha256"]
    # Derived status is checked, never used to calculate the budget.
    status = _read(directory / "status.json")
    summary = status.get("budget", {})
    if (status.get("plan_sha256") != plan_sha
            or summary.get("plan_sha256") != plan_sha
            or summary.get("completed_jobs") != len(results)
            or not _same(summary.get("charged_wall_seconds"), charged)
            or not _same(summary.get("campaign_budget_seconds"), effective)
            or not _same(summary.get("remaining_campaign_seconds"), max(0, effective - charged))
            or summary.get("conservative_unclosed_session_charges") != conservative):
        raise BudgetError("derived_campaign_budget_report_disagrees_with_journal")
    value = {"version": VERSION, "branch": branch, "head_sha": head_sha,
             "plan_sha256": plan_sha, "legacy_cache": plan.get("legacy_cache"),
             "journal_verified": True, "snapshot_manifest_sha256": inventory_sha,
             "event_count": len(events), "last_event_sha256": events[-1]["sha256"] if events else ZERO,
             "completed_jobs": len(results), "unresolved_job_ids": unresolved,
             "unclosed_session_ids": sorted(set(sessions) - closed),
             "conservative_unclosed_session_charges": conservative,
             "charged_wall_seconds": charged, "campaign_budget_seconds": effective,
             "remaining_campaign_seconds": max(0, effective - charged),
             "budget_extensions": extensions, "status": status.get("status")}
    value["sha256"] = digest(value)
    return value


def reconcile_snapshots(legacy, optimized, *, observed_heads, authorized_total_hours=AUTHORIZED_TOTAL_HOURS):
    """Bind fully replayed snapshots to current heads; no fresh 64-hour grant."""
    if not _same(authorized_total_hours, AUTHORIZED_TOTAL_HOURS):
        raise BudgetError("this_registration_only_authorizes_64_combined_hours")
    for snapshot, branch, plan_sha in ((legacy, LEGACY_BRANCH, LEGACY_PLAN),
                                        (optimized, OPTIMIZED_BRANCH, OPTIMIZED_PLAN)):
        _checked(snapshot)
        if (snapshot.get("branch") != branch or snapshot.get("plan_sha256") != plan_sha
                or snapshot.get("journal_verified") is not True
                or not snapshot.get("snapshot_manifest_sha256")
                or observed_heads.get(branch) != snapshot.get("head_sha")):
            raise BudgetError("verified_snapshot_or_current_head_mismatch")
        if snapshot["unresolved_job_ids"] or snapshot["unclosed_session_ids"]:
            raise BudgetReconciliationRequired("historical_attempt_or_session_requires_reconciliation")
    cache = optimized.get("legacy_cache")
    if (type(cache) is not dict or cache.get("commit") != legacy["head_sha"]
            or cache.get("plan_sha256") != legacy["plan_sha256"]
            or cache.get("completed_jobs") != legacy["completed_jobs"]
            or not _same(cache.get("spent_seconds"), legacy["charged_wall_seconds"])
            or not _same(cache.get("limit_seconds"), legacy["campaign_budget_seconds"])):
        raise BudgetError("optimized_transferred_budget_does_not_match_legacy_journal")
    cap = AUTHORIZED_TOTAL_HOURS * 3600
    if not _same(legacy["charged_wall_seconds"] + optimized["campaign_budget_seconds"], cap):
        raise BudgetError("combined_registered_limit_does_not_equal_64_hours")
    charged = legacy["charged_wall_seconds"] + optimized["charged_wall_seconds"]
    report = {"version": VERSION, "status": "reconciled_before_new_pilot_reservations",
              "authorized_combined_hours": AUTHORIZED_TOTAL_HOURS,
              "authorized_combined_seconds": cap,
              "historical_charged_seconds": charged,
              "remaining_seconds_before_pilots": max(0, cap - charged),
              "historical_heads": {LEGACY_BRANCH: legacy["head_sha"], OPTIMIZED_BRANCH: optimized["head_sha"]},
              "snapshot_evidence_sha256": {LEGACY_BRANCH: legacy["sha256"], OPTIMIZED_BRANCH: optimized["sha256"]},
              "conservative_historical_charges_preserved": True,
              "budget_reset": False, "market_training_started": False}
    report["sha256"] = digest(report)
    return report


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


def _public_get(url, *, token=None, maximum=1024 * 1024):
    # Only API requests receive the token. Immutable codeload archives never do.
    headers = {"User-Agent": "E2-registered-pilot-budget", "Accept": "application/vnd.github+json"}
    if token is not None:
        if (not isinstance(token, str) or not token or any(c.isspace() for c in token)
                or not url.startswith("https://api.github.com/repos/" + REPOSITORY + "/")):
            raise BudgetError("invalid_budget_read_credential_scope")
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(url, headers=headers)
    try:
        opener = urllib.request.build_opener(_NoRedirect())
        with opener.open(request, timeout=45) as stream:
            payload = stream.read(maximum + 1)
        if len(payload) > maximum:
            raise BudgetError("budget_snapshot_download_limit")
        return payload
    except (OSError, urllib.error.HTTPError):
        raise BudgetError("budget_remote_read_unavailable") from None


def fetch_public_heads(*, token=None):
    """Read current branch heads; tokens are never printed or persisted."""
    heads = {}
    for branch in (LEGACY_BRANCH, OPTIMIZED_BRANCH):
        try:
            payload = _public_get("https://api.github.com/repos/" + REPOSITORY
                                  + "/git/ref/heads/" + branch, token=token)
        except BudgetError as error:
            if str(error) != "budget_remote_read_unavailable":
                raise
            # Public read-only Git transport is usable when an API proxy or
            # anonymous API quota rejects GET. It receives no write token.
            return _git_public_heads()
        try:
            value = json.loads(payload, object_pairs_hook=_pairs)
            sha = value["object"]["sha"]
            if (value.get("ref") != "refs/heads/" + branch
                    or value["object"].get("type") != "commit" or not HEX40.fullmatch(sha)):
                raise ValueError()
        except (KeyError, TypeError, ValueError):
            raise BudgetError("budget_remote_head_invalid") from None
        heads[branch] = sha
    return heads


def _git_public_heads():
    environment = {key: value for key, value in os.environ.items()
                   if (not key.startswith("GIT_")
                       or key in {"GIT_SSL_CAINFO", "GIT_SSL_CAPATH"})
                   and key not in {"GITHUB_TOKEN", "GH_TOKEN", "SSH_ASKPASS"}}
    environment.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1",
                       GIT_TERMINAL_PROMPT="0", GIT_ASKPASS=os.devnull)
    refs = ["refs/heads/" + branch for branch in (LEGACY_BRANCH, OPTIMIZED_BRANCH)]
    try:
        result = subprocess.run(["git", "-c", "credential.helper=", "-c", "http.followRedirects=false",
                                 "-c", "protocol.ext.allow=never", "ls-remote", "--exit-code",
                                 "https://github.com/" + REPOSITORY + ".git", *refs],
                                capture_output=True, text=True, timeout=45,
                                cwd=tempfile.gettempdir(), env=environment)
        rows = [line.split("\t") for line in result.stdout.splitlines()]
        if result.returncode or len(rows) != 2:
            raise ValueError()
        mapping = {}
        for sha, ref in rows:
            if ref not in refs or not HEX40.fullmatch(sha) or ref in mapping:
                raise ValueError()
            mapping[ref] = sha
        if set(mapping) != set(refs):
            raise ValueError()
        return {ref[len("refs/heads/"):]: sha for ref, sha in mapping.items()}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        raise BudgetError("budget_remote_head_read_unavailable") from None


def _restore_public_snapshot(output, branch, head):
    """Extract only the pinned orphan campaign; no credentials go to codeload."""
    output = _no_links(output)
    if (output / "snapshot.json").is_file():
        return
    if output.exists() and any(output.iterdir()):
        raise BudgetError("partial_snapshot_requires_reconciliation")
    payload = _public_get("https://codeload.github.com/" + REPOSITORY + "/zip/" + head,
                          maximum=64 * 1024 * 1024)
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except (ValueError, zipfile.BadZipFile):
        raise BudgetError("budget_snapshot_archive_invalid") from None
    records = {}
    archive_root = None
    total = 0
    with archive:
        for item in archive.infolist():
            path = Path(item.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in item.filename:
                raise BudgetError("budget_snapshot_archive_path_invalid")
            parts = item.filename.split("/")
            if not parts or not parts[0]:
                raise BudgetError("budget_snapshot_archive_path_invalid")
            if archive_root is None:
                archive_root = parts[0]
            if parts[0] != archive_root:
                raise BudgetError("budget_snapshot_archive_root_changed")
            mode = item.external_attr >> 16
            if stat.S_ISLNK(mode) or item.flag_bits & 1:
                raise BudgetError("budget_snapshot_archive_link_or_encryption")
            if item.is_dir():
                continue
            relative = "/".join(parts[1:])
            prefix = branch + "/"
            if not relative.startswith(prefix):
                raise BudgetError("non_campaign_archive_file_refused")
            relative = relative[len(prefix):]
            if (relative in records or not relative.endswith(".json") or item.file_size > 8 * 1024 * 1024
                    or ("/" in relative and not relative.startswith(("events/", "results/")))
                    or ("/" not in relative and relative not in ("campaign.json", "snapshot.json", "status.json"))):
                raise BudgetError("unapproved_campaign_snapshot_file")
            total += item.file_size
            if total > 512 * 1024 * 1024:
                raise BudgetError("budget_snapshot_uncompressed_limit")
            records[relative] = archive.read(item)
    if not {"campaign.json", "snapshot.json", "status.json"} <= set(records):
        raise BudgetError("budget_snapshot_required_files_missing")
    output.mkdir(parents=True, exist_ok=True)
    # Publish snapshot.json last: incomplete extraction is never called valid.
    for relative in sorted(records, key=lambda p: p == "snapshot.json"):
        target = _no_links(output / relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "xb") as stream:
            stream.write(records[relative])
            stream.flush()
            os.fsync(stream.fileno())


def fetch_reconciliation(output, token=None):
    """Restore both current ledgers, replay every record, and persist evidence.

    Returns the hashed reconciliation object and writes
    ``output/budget_evidence.json``. A changed evidence anchor is refused, since
    accepting a new anchor could erase pilot spending already recorded there.
    No model, data panel, outcome scoring, or remote write is involved.
    """
    output = _no_links(output)
    output.mkdir(parents=True, exist_ok=True)
    secret = token if token is not None else os.environ.get("GITHUB_TOKEN")
    before = fetch_public_heads(token=secret)
    snapshots = []
    for branch, expected in ((LEGACY_BRANCH, LEGACY_PLAN), (OPTIMIZED_BRANCH, OPTIMIZED_PLAN)):
        directory = output / "snapshots" / branch.split("/")[-1] / before[branch]
        _restore_public_snapshot(directory, branch, before[branch])
        snapshots.append(replay_campaign(directory, branch=branch, head_sha=before[branch],
                                         expected_plan_sha256=expected, verify_inventory=True))
    after = fetch_public_heads(token=secret)
    if before != after:
        raise BudgetError("campaign_head_changed_during_budget_reconciliation")
    report = reconcile_snapshots(*snapshots, observed_heads=after)
    target = output / "budget_evidence.json"
    if target.exists():
        if _read(target) != report:
            raise BudgetReconciliationRequired("historical_anchor_changed_pilot_spending_must_not_reset")
    else:
        with open(_no_links(target), "xb") as stream:
            stream.write(canonical(report) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    for snapshot in snapshots:
        path = output / (snapshot["branch"].split("/")[-1] + "_replay.json")
        if path.exists():
            if _read(path) != snapshot:
                raise BudgetReconciliationRequired("stored_journal_replay_changed")
        else:
            with open(_no_links(path), "xb") as stream:
                stream.write(canonical(snapshot) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
    return report


class PilotBudgetLedger:
    """Single-runtime append journal. A reserved attempt survives a VM death.

    An unresolved reservation blocks reopening. It is reported conservatively
    as its full reservation, never silently cleared or replayed. The caller
    must supervise the entire fit against its reserved wall-clock deadline.
    """
    def __init__(self, root, reconciliation):
        _checked(reconciliation)
        if (reconciliation.get("version") != VERSION
                or reconciliation.get("status") != "reconciled_before_new_pilot_reservations"
                or reconciliation.get("budget_reset") is not False
                or not _same(reconciliation.get("authorized_combined_hours"), AUTHORIZED_TOTAL_HOURS)
                or not _same(reconciliation.get("authorized_combined_seconds"), AUTHORIZED_TOTAL_HOURS * 3600)
                or not _same(reconciliation.get("remaining_seconds_before_pilots"),
                             max(0, AUTHORIZED_TOTAL_HOURS * 3600
                                 - _number(reconciliation.get("historical_charged_seconds"))))):
            raise BudgetError("verified_reconciliation_required")
        self.root = _no_links(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = None
        self._events = []
        self._active = {}
        self._charges = 0.0
        self._attempts = set()
        self._progress = {}
        self.reconciliation = deepcopy(reconciliation)
        self._path = self.root / "pilot_budget.jsonl"
        self._closed = False
        try:
            descriptor = os.open(_no_links(self.root / ".pilot-budget.lock"),
                                 os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
            self._lock = os.fdopen(descriptor, "r+b")
            try:
                fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise BudgetError("another_pilot_budget_controller_is_active") from None
            if self._path.exists():
                _no_links(self._path)
                for index, line in enumerate(self._path.read_bytes().splitlines()):
                    record = json.loads(line, object_pairs_hook=_pairs)
                    row = _checked(record)
                    if (row.get("version") != VERSION or row.get("sequence") != index
                            or row.get("reconciliation_sha256") != reconciliation["sha256"]
                            or row.get("previous_sha256") != (self._events[-1]["sha256"] if self._events else ZERO)):
                        raise BudgetError("pilot_budget_chain_or_snapshot_binding_changed")
                    self._apply(record)
            if self._active:
                raise BudgetReconciliationRequired("unresolved_pilot_reservation_no_automatic_retry")
        except BaseException:
            self.close()
            raise

    def _apply(self, record):
        row = _checked(record)
        kind = row["kind"]
        if kind == "reserved":
            attempt = row["attempt_id"]
            if not isinstance(attempt, str) or not attempt or attempt in self._attempts:
                raise BudgetError("duplicate_pilot_attempt_id")
            amount = _number(row["reservation_seconds"], positive=True)
            if amount > self.remaining_seconds + 1e-9:
                raise BudgetError("pilot_reservation_exceeds_shared_cap")
            self._active[attempt] = amount
            self._attempts.add(attempt)
        elif kind == "charged":
            attempt = row["attempt_id"]
            if attempt not in self._active or not _same(row["reservation_seconds"], self._active[attempt]):
                raise BudgetError("unbound_pilot_charge")
            self._charges += _number(row["elapsed_seconds"])
            del self._active[attempt]
        elif kind == "overhead":
            self._charges += _number(row["elapsed_seconds"])
        elif kind == "progress":
            attempt = row["attempt_id"]
            prior = self._progress.get(attempt, {"elapsed_seconds_lower_bound": 0, "optimizer_steps_lower_bound": 0})
            elapsed = _number(row["elapsed_seconds_lower_bound"])
            steps = row["optimizer_steps_lower_bound"]
            if (attempt not in self._active or elapsed < prior["elapsed_seconds_lower_bound"]
                    or type(steps) is not int or steps < prior["optimizer_steps_lower_bound"]):
                raise BudgetError("invalid_or_decreasing_pilot_cost_lower_bound")
            self._progress[attempt] = {"elapsed_seconds_lower_bound": elapsed,
                                      "optimizer_steps_lower_bound": steps}
        else:
            raise BudgetError("unknown_pilot_budget_event")
        self._events.append(record)

    def _append(self, kind, **fields):
        if self._closed:
            raise BudgetError("pilot_budget_ledger_closed")
        row = {"version": VERSION, "sequence": len(self._events), "kind": kind,
               "reconciliation_sha256": self.reconciliation["sha256"],
               "previous_sha256": self._events[-1]["sha256"] if self._events else ZERO,
               "utc": datetime.now(timezone.utc).isoformat(), **fields}
        row["sha256"] = digest(row)
        # Validate on a disposable state first, so a failed write never clears
        # an in-memory reservation. Reopening a partial write fails closed.
        clone = object.__new__(PilotBudgetLedger)
        clone._events, clone._active = list(self._events), dict(self._active)
        clone._charges, clone._attempts = self._charges, set(self._attempts)
        clone._progress = deepcopy(self._progress)
        clone.reconciliation = self.reconciliation
        clone._apply(row)
        with open(_no_links(self._path), "ab") as stream:
            stream.write(canonical(row) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._apply(row)
        return deepcopy(row)

    @property
    def remaining_seconds(self):
        return max(0.0, self.reconciliation["remaining_seconds_before_pilots"]
                   - self._charges - sum(self._active.values()))

    def reserve(self, attempt_id, seconds, *, metadata=None):
        amount = _number(seconds, positive=True)
        if self._active:
            raise BudgetReconciliationRequired("finish_current_pilot_before_another_reservation")
        if amount > self.remaining_seconds + 1e-9:
            raise BudgetPause("pilot_shared_campaign_budget")
        return self._append("reserved", attempt_id=attempt_id, reservation_seconds=amount,
                            metadata={} if metadata is None else metadata)

    def charge(self, attempt_id, elapsed_seconds, *, status, resource_use=None):
        if status not in ("completed", "failed", "paused_clean", "reconciliation_required"):
            raise BudgetError("explicit_terminal_pilot_status_required")
        if attempt_id not in self._active:
            raise BudgetError("unbound_pilot_charge")
        if status == "reconciliation_required":
            # Persist a lower bound without releasing the reservation. The
            # attempt's actual duration/optimizer effect remains unresolved.
            raise BudgetReconciliationRequired("unknown_attempt_keeps_its_full_reservation")
        reservation = self._active[attempt_id]
        amount = _number(elapsed_seconds)
        result = self._append("charged", attempt_id=attempt_id,
                              reservation_seconds=reservation, elapsed_seconds=amount,
                              reservation_overrun=amount > reservation,
                              status=status, resource_use={} if resource_use is None else resource_use)
        if amount > reservation + 1e-9:
            raise BudgetPause("pilot_attempt_reservation_overrun_charged")
        return result

    def record_progress(self, attempt_id, elapsed_seconds_lower_bound, optimizer_steps_lower_bound):
        """Record known costs without releasing any reservation after a crash."""
        return self._append("progress", attempt_id=attempt_id,
                            elapsed_seconds_lower_bound=_number(elapsed_seconds_lower_bound),
                            optimizer_steps_lower_bound=optimizer_steps_lower_bound)

    def charge_overhead(self, elapsed_seconds, *, description="controller_overhead"):
        result = self._append("overhead", elapsed_seconds=_number(elapsed_seconds), description=description)
        if self.remaining_seconds <= 0:
            raise BudgetPause("pilot_shared_campaign_budget")
        return result

    def report(self):
        return {"version": VERSION, "reconciliation_sha256": self.reconciliation["sha256"],
                "historical_charged_seconds": self.reconciliation["historical_charged_seconds"],
                "pilot_charged_seconds": self._charges,
                "reserved_seconds": sum(self._active.values()),
                "unresolved_reservations": dict(self._active),
                "attempt_cost_lower_bounds": deepcopy(self._progress),
                "remaining_seconds": self.remaining_seconds,
                "last_event_sha256": self._events[-1]["sha256"] if self._events else ZERO,
                "reconciliation_required": bool(self._active), "budget_reset": False}

    def close(self):
        if self._lock is not None:
            try:
                fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            finally:
                self._lock.close()
                self._lock = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
