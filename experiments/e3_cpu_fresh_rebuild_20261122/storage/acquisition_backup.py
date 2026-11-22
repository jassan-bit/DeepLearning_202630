"""Metadata-only acquisition journal; extends, never edits, frozen S2 storage.

No requests, prompts, downloads, or secret reads occur at import. All private
metadata is encrypted by the original S2 chunked GithubStore. Public archives
are empty; this journal is neither a planning experiment nor a confirmation.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile

VERSION = "e3-acquisition-metadata-journal-1"
SOURCE_COMMIT = "db74afd0763596f71bcbafb4b9150a44a2e83339"
STORAGE_SHA256 = "63ee17b4db5791bad7dcce34afc6066c4ebb6322c63e38856d4449fbbb7fac7e"
MAX_METADATA_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 10 * 1024 * 1024
MAX_METADATA_FILES = 256
SAFE_SUFFIXES = {".json", ".jsonl", ".txt", ".md", ".py"}


class JournalError(RuntimeError):
    """Fixed codes only; never includes private file contents or credentials."""


def _need(ok, code):
    if not ok:
        raise JournalError(code)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _need(key not in result, "duplicate_metadata_key")
            result[key] = value
        return result
    try:
        return json.loads(raw, object_pairs_hook=unique,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, TypeError, UnicodeError):
        raise JournalError("invalid_metadata_json") from None


def _relative(name):
    _need(isinstance(name, str) and bool(name) and "\\" not in name and "\0" not in name,
          "unsafe_metadata_path")
    path = PurePosixPath(name)
    _need(not path.is_absolute() and path.as_posix() == name
          and all(x not in ("", ".", "..") for x in name.split("/")), "unsafe_metadata_path")
    _need(path.suffix.lower() in SAFE_SUFFIXES, "nonmetadata_file_refused")
    _need(not any(x.lower() in {"tokens", "secrets", ".git"} for x in path.parts),
          "secret_or_git_path_refused")
    return name


def _path(root, name):
    name = _relative(name)
    root = Path(root).absolute()
    path = root / name
    current = path
    while current != root:
        _need(not current.is_symlink(), "metadata_symlink_refused")
        current = current.parent
    _need(not root.is_symlink() and path.resolve().is_relative_to(root.resolve()),
          "metadata_path_escapes_root")
    return path


def metadata_store_class(storage):
    """Typed neutral identity, strict metadata allowlist; S2 crypto unchanged."""
    _need(_sha(Path(storage.__file__).read_bytes()) == STORAGE_SHA256,
          "frozen_S2_storage_source_differs")

    class MetadataStore(storage.GithubStore):
        def _context(self, plan):
            _need(isinstance(plan, dict) and plan.get("kind") == "development_acquisition_journal"
                  and plan.get("format") == VERSION and plan.get("stage") == "metadata_only"
                  and isinstance(plan.get("case_id"), str)
                  and re.fullmatch(r"data-acquisition-[A-Za-z0-9_-]{1,80}", plan["case_id"])
                  and plan.get("jobs") == [] and plan.get("market_fits") == 0
                  and plan.get("R3_accredited") is False and plan.get("test_opened") is False,
                  "neutral_metadata_journal_identity_required")
            _need(plan.get("source_commit") == self.source_commit == SOURCE_COMMIT,
                  "metadata_journal_source_differs")
            return {"version": storage.VERSION, "repository": self.repository,
                    "branch": self.branch, "prefix": self.prefix,
                    "source_commit": self.source_commit, "case_id": plan["case_id"],
                    "plan_sha256": _sha(storage._canonical(plan))}

        def _eligible(self, name):
            return name == "plan.json" or (isinstance(name, str) and
                re.fullmatch(r"receipts/[0-9a-f]{64}\.json", name) is not None)

        def _members(self, case, plan):
            self._context(plan)
            names = ["plan.json"]
            directory = case / "receipts"
            _need(not directory.is_symlink(), "metadata_symlink_refused")
            for path in sorted(directory.glob("*.json")):
                name = path.relative_to(case).as_posix()
                _need(self._eligible(name), "unexpected_capsule_member")
                self._case_path(case, name)
                names.append(name)
            _need(1 < len(names) <= MAX_METADATA_FILES + 2, "metadata_inventory_empty_or_too_large")
            _need(sum(self._case_path(case, n).stat().st_size for n in names)
                  <= MAX_TOTAL_BYTES * 2, "metadata_inventory_too_large")
            return names

        def _public_archive(self, case, plan, public_files):
            _need(not public_files, "public_metadata_publication_refused")
            return super()._public_archive(case, plan, None)

        def _request(self, method, path, payload=None, **kwargs):
            if method == "POST" and path.endswith("/git/commits") and isinstance(payload, dict):
                payload = dict(payload)
                payload["message"] = "Preserve encrypted development acquisition metadata; no model fits"
                for field in ("author", "committer"):
                    payload[field] = {**payload[field], "name": "E3 data acquisition metadata journal",
                                      "email": "e3-data@users.noreply.github.com"}
            return super()._request(method, path, payload, **kwargs)

    return MetadataStore


class MetadataJournal:
    """Caller holds one instance/credentials; checkpoint must precede acquisition.

    Paths are confined to work_root; put metadata about any external code overlay
    under work_root first. Raw archives, CSV, arrays and checkpoints are refused.
    Every supplied file is immutable once journaled, including owner declarations.
    """
    def __init__(self, storage, *, token, backup_secret, repository, branch, prefix,
                 work_root, journal_dir, case_id, source_commit=SOURCE_COMMIT, **store_options):
        self.storage = storage
        self.work_root = Path(work_root).absolute()
        self.case = Path(journal_dir).absolute()
        _need(not self.work_root.is_symlink() and not self.case.is_symlink(), "metadata_symlink_refused")
        _need(self.case.resolve().is_relative_to(self.work_root.resolve()), "journal_outside_work_root")
        current = self.case
        while current != self.work_root:
            _need(not current.is_symlink(), "metadata_symlink_refused")
            current = current.parent
        self.plan = {"format": VERSION, "kind": "development_acquisition_journal", "case_id": case_id,
                     "source_commit": source_commit, "stage": "metadata_only", "jobs": [],
                     "market_fits": 0, "test_opened": False, "R3_accredited": False}
        self.store = metadata_store_class(storage)(token, backup_secret, repository, branch,
            source_commit=source_commit, prefix=prefix, **store_options)
        self.store._context(self.plan)

    def __repr__(self):
        return "MetadataJournal(credentials=<hidden>, kind='development_acquisition_journal')"

    def close(self):
        """Drop this instance's credential references; no zeroization claim."""
        self.store._token = ""
        self.store._secret = ""

    def _read_capsules(self, case=None):
        case = self.case if case is None else Path(case)
        records = {}
        total = 0
        directory = case / "receipts"
        _need(not directory.is_symlink(), "metadata_symlink_refused")
        for path in sorted(directory.glob("*.json")):
            _need(not path.is_symlink() and path.stat().st_size <= MAX_METADATA_BYTES * 2,
                  "invalid_capsule_file")
            record = _json(path.read_bytes())
            _need(isinstance(record, dict) and record.get("format") == VERSION
                  and record.get("kind") in {"exact_metadata_bytes", "acquisition_journal_events"},
                  "invalid_metadata_capsule")
            if record["kind"] == "acquisition_journal_events":
                _need(path.name == _sha(b"acquisition-journal-events") + ".json",
                      "invalid_events_capsule_identity")
                continue
            name = _relative(record.get("path"))
            _need(path.name == _sha(name.encode()) + ".json" and name not in records,
                  "capsule_path_identity_differs")
            try:
                raw = base64.b64decode(record.get("base64", ""), validate=True)
            except (ValueError, TypeError):
                raise JournalError("invalid_capsule_encoding") from None
            _need(len(raw) <= MAX_METADATA_BYTES and record.get("bytes") == len(raw)
                  and record.get("sha256") == _sha(raw), "capsule_hash_differs")
            self.store._no_secrets(raw)
            total += len(raw)
            _need(total <= MAX_TOTAL_BYTES and len(records) < MAX_METADATA_FILES, "metadata_inventory_too_large")
            records[name] = (record, raw)
        return records

    def checkpoint(self, metadata_paths, *, event):
        _need(isinstance(event, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", event), "invalid_journal_event")
        _need(isinstance(metadata_paths, (list, tuple)) and bool(metadata_paths), "metadata_paths_required")
        # Finish an interrupted, already encrypted snapshot before staging any
        # new metadata. Otherwise S2 would correctly replay the older pending
        # snapshot while a caller could mistake it for the newly added files.
        if (self.case / ".github_pending").exists():
            _need(_json((self.case / "plan.json").read_bytes()) == self.plan, "journal_plan_changed")
            self.store.backup(self.case, public_files=None, stage="data_acquisition_pending_retry")
        # Discover and authenticate the remote first. Existing remotes must be
        # restored, never silently replaced by a fresh journal on a new VM.
        remote = self.store.status()
        if remote["exists"] and not (self.case / ".github_backup_receipt.json").is_file():
            _need(not self.case.exists(), "restore_existing_remote_before_checkpoint")
            self.restore()
        existing = self._read_capsules()
        pending = []
        for supplied in metadata_paths:
            supplied = Path(supplied)
            if supplied.is_absolute():
                _need(supplied.is_relative_to(self.work_root), "metadata_path_escapes_root")
                name = supplied.relative_to(self.work_root).as_posix()
            else:
                name = supplied.as_posix()
            path = _path(self.work_root, name)
            _need(not path.is_relative_to(self.case), "journal_recursion_refused")
            _need(path.is_file() and path.stat().st_size <= MAX_METADATA_BYTES,
                  "metadata_file_missing_or_too_large")
            raw = path.read_bytes()
            self.store._no_secrets(raw)
            _need(name not in existing or existing[name][1] == raw, "original_metadata_change_refused")
            record = {"format": VERSION, "kind": "exact_metadata_bytes", "path": name,
                      "bytes": len(raw), "sha256": _sha(raw), "base64": base64.b64encode(raw).decode()}
            pending.append((name, record))
        total = {n: len(r[1]) for n, r in existing.items()}
        total.update({n: r["bytes"] for n, r in pending})
        _need(len(total) <= MAX_METADATA_FILES and sum(total.values()) <= MAX_TOTAL_BYTES,
              "metadata_inventory_too_large")
        self.case.mkdir(parents=True, exist_ok=True)
        plan_path = self.case / "plan.json"
        if plan_path.exists():
            _need(plan_path.read_bytes() == self.storage._canonical(self.plan), "journal_plan_changed")
        else:
            self.storage._atomic(plan_path, self.storage._canonical(self.plan))
        for name, record in pending:
            self.storage._atomic(self.case / "receipts" / (_sha(name.encode()) + ".json"),
                                 self.storage._canonical(record))
        events_path = self.case / "receipts" / (_sha(b"acquisition-journal-events") + ".json")
        events = _json(events_path.read_bytes()) if events_path.exists() else {
            "format": VERSION, "kind": "acquisition_journal_events", "events": []}
        _need(isinstance(events.get("events"), list), "invalid_journal_events")
        if not events["events"] or events["events"][-1].get("event") != event:
            events["events"].append({"event": event, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                                      "metadata_file_count": len(total), "market_fits": 0})
        self.storage._atomic(events_path, self.storage._canonical(events))
        receipt = self.store.backup(self.case, public_files=None, stage="data_acquisition_" + event)
        _need(receipt.get("remote_verified") is True, "metadata_backup_not_verified_acquisition_must_stop")
        return receipt

    def restore(self):
        """Authenticate/decrypt first, preflight every destination, then restore.

        A mismatch causes no owner/source metadata writes; exact existing bytes
        are reused. No raw archives or arrays can occur in a capsule.
        """
        _need(not self.case.exists(), "fresh_journal_destination_required")
        self.case.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".metadata-restore-", dir=self.case.parent))
        staging = temporary / "case"
        try:
            receipt = self.store.restore(staging)
            if receipt is None:
                return None
            _need(_json((staging / "plan.json").read_bytes()) == self.plan, "restored_journal_plan_differs")
            records = self._read_capsules(staging)
            targets = []
            for name, (_, raw) in records.items():
                path = _path(self.work_root, name)
                _need(not path.is_relative_to(self.case), "journal_recursion_refused")
                _need(not path.exists() or (path.is_file() and path.read_bytes() == raw),
                      "restore_metadata_conflict_no_overwrite")
                targets.append((path, raw))
            for path, raw in targets:
                if not path.exists():
                    self.storage._atomic(path, raw)
            _need(not self.case.exists(), "journal_destination_appeared")
            staging.rename(self.case)
            return {**receipt, "metadata_files_restored": len(records),
                    "raw_inputs_restored": False, "market_fits": 0}
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
