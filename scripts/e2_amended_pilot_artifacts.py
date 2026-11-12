"""Durable, single-writer artifacts for the amended E2 pilots.

This module imports no training framework until a tensor checkpoint is saved.
An interrupted/open attempt is evidence to reconcile, never permission to run
the fit again.  Only an explicitly sealed clean pause can be resumed.
"""
from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from collections.abc import Mapping
from copy import deepcopy


VERSION = "e2-amended-pilot-artifacts-v1"
_ZERO = "0" * 64
_SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\Z")
_RESERVED = {"prepared", "training_started", "checkpoint_saved", "artifact_saved",
             "completed", "paused_clean", "resumed_clean", "failed", "interrupted"}
_TERMINAL = {"completed", "paused_clean", "failed", "interrupted"}


class PilotArtifactError(RuntimeError):
    pass


class ArtifactIdentityMismatch(PilotArtifactError):
    pass


class ArtifactReconciliationRequired(PilotArtifactError):
    pass


class ArtifactStateError(PilotArtifactError):
    pass


def _canonical(value):
    def checked(item):
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise TypeError("Artifact JSON keys must be strings")
            return {key: checked(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [checked(val) for val in item]
        if item is None or type(item) in (str, bool, int, float):
            return item
        raise TypeError("Artifact metadata must contain only finite JSON values")
    return json.dumps(checked(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(name):
    if not isinstance(name, str) or not _SAFE.fullmatch(name):
        raise ValueError("Use a simple artifact/attempt name without path components")
    return name


def _fsync_directory(path):
    """Some FUSE mounts reject directory fsync; file fsync remains mandatory."""
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        try:
            os.fsync(descriptor)
            return True
        except OSError as error:
            if error.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP):
                raise
            return False
    finally:
        os.close(descriptor)


class PilotArtifactStore:
    """One immutable scientific identity, journal, and versioned checkpoints.

    ``prepare`` requires config/source/data/seed/resources metadata. The caller
    prepares before constructing an optimizer; the trainer calls
    ``begin_training`` exactly once. ``complete`` belongs to the caller after
    predictions and scores have also been persisted. Use as a context manager
    or call ``close`` to release the cooperative writer lock.
    """

    def __init__(self, root, attempt_id, identity):
        self.attempt_id = _safe_name(attempt_id)
        if not isinstance(identity, Mapping) or not identity:
            raise ValueError("A nonempty scientific identity is required")
        self.identity = json.loads(_canonical(identity))
        self.identity_sha256 = _sha(_canonical(self.identity))
        self.path = Path(root) / self.attempt_id
        if self.path.is_symlink():
            raise ArtifactStateError("Attempt directory must not be a symlink")
        self.path.mkdir(parents=True, exist_ok=True)
        self._closed = False
        self._io_failed = False
        self._fresh = not (self.path / "identity.json").exists()
        self._prepared_here = False
        self._active_here = False
        lock = self.path / ".writer.lock"
        self._reject_symlink(lock)
        self._lock = lock.open("a+b")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self._lock.close()
            self._closed = True
            raise ArtifactStateError("Another writer owns this pilot attempt") from error
        self._state = self._empty_state()
        try:
            if not self._fresh:
                self._read_existing()
            elif any(p.name != ".writer.lock" for p in self.path.iterdir()):
                raise ArtifactReconciliationRequired("Unidentified/orphan attempt files require reconciliation")
        except BaseException:
            self.close()
            raise

    def _empty_state(self):
        return {"version": VERSION, "attempt_id": self.attempt_id,
                "identity_sha256": self.identity_sha256, "state": "new",
                "event_count": 0, "last_event_sha256": _ZERO,
                "optimizer_step_open": False, "optimizer_steps_known_lower_bound": 0,
                "known_cost_seconds": 0.0, "checkpoints": {}, "checkpoint_history": [], "artifacts": {},
                "durability": {"file_fsync_required": True,
                               "directory_fsync": "unless_unsupported_by_filesystem",
                               "publication": "os.replace", "hardlinks": False}}

    @property
    def status(self):
        return self._state["state"]

    @property
    def manifest(self):
        return deepcopy(self._state)

    @staticmethod
    def _reject_symlink(path):
        if Path(path).is_symlink():
            raise ArtifactReconciliationRequired("Artifact symlinks are not accepted: " + str(path))

    def _assert_usable(self):
        if self._closed:
            raise ArtifactStateError("Artifact store is closed")
        if self._io_failed:
            raise ArtifactReconciliationRequired("A durable write failed; this attempt requires reconciliation")

    def _assert_active(self):
        self._assert_usable()
        if not self._active_here:
            raise ArtifactReconciliationRequired("Reopened attempts are read-only until an explicit clean resume")

    def _atomic_write(self, path, writer):
        self._reject_symlink(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._reject_symlink(path.parent)
        descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                writer(handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            _fsync_directory(path.parent)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def _write_json(self, path, value):
        content = _canonical(value) + b"\n"
        self._atomic_write(path, lambda handle: handle.write(content))

    def _read_json(self, path):
        self._reject_symlink(path)
        return json.loads(Path(path).read_bytes())

    def _read_existing(self):
        claimed = self._read_json(self.path / "identity.json")
        expected = {"version": VERSION, "attempt_id": self.attempt_id,
                    "identity": self.identity, "identity_sha256": self.identity_sha256}
        if _canonical(claimed) != _canonical(expected):
            raise ArtifactIdentityMismatch("Attempt identity changed; use a different registered attempt")
        journal = self.path / "events.jsonl"
        self._reject_symlink(journal)
        try:
            raw = journal.read_bytes()
            state = self._empty_state()
            for line in raw.splitlines(keepends=True):
                if not line.endswith(b"\n"):
                    raise ValueError("Incomplete journal tail")
                row = json.loads(line)
                digest = row.pop("sha256")
                if (row["sequence"] != state["event_count"]
                        or row["previous_sha256"] != state["last_event_sha256"]
                        or row["identity_sha256"] != self.identity_sha256
                        or digest != _sha(_canonical(row))):
                    raise ValueError("Changed journal chain")
                row["sha256"] = digest
                state = self._advance(state, row)
            saved = self._read_json(self.path / "manifest.json")
            if not raw or _canonical(saved) != _canonical(state):
                raise ValueError("Journal/manifest publication is incomplete")
            prepared = self._read_json(self.path / "prepared.json")
            if _sha(_canonical(prepared)) != state["prepared_metadata_sha256"]:
                raise ValueError("Prepared metadata changed")
            self._state = state
            self._verify_artifacts()
        except ArtifactReconciliationRequired:
            raise
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ArtifactReconciliationRequired("Invalid/incomplete pilot artifact journal: " + str(error)) from error

    def _advance(self, previous, row):
        state = deepcopy(previous)
        kind, payload = row["kind"], row["payload"]
        if kind == "prepared":
            state["state"] = "prepared"
            state["prepared_metadata_sha256"] = payload["metadata_sha256"]
        elif kind == "training_started":
            state["state"] = "training"
        elif kind == "checkpoint_saved":
            state["checkpoints"][payload["label"]] = payload
            state["checkpoint_history"].append(payload)
        elif kind == "artifact_saved":
            state["artifacts"][payload["name"]] = payload
        elif kind == "resumed_clean":
            state["state"] = "training"
        elif kind in _TERMINAL:
            state["state"] = kind
            state["terminal_metadata"] = payload
        elif kind == "optimizer_step_started":
            if state["optimizer_step_open"]:
                raise ArtifactStateError("Optimizer step already open")
            state["optimizer_step_open"] = True
        elif kind == "optimizer_step_completed":
            if not state["optimizer_step_open"]:
                raise ArtifactStateError("No optimizer step is open")
            state["optimizer_step_open"] = False
        for field, target in (("optimizer_steps", "optimizer_steps_known_lower_bound"),
                              ("optimizer_steps_known_lower_bound", "optimizer_steps_known_lower_bound"),
                              ("known_cost_seconds", "known_cost_seconds")):
            if field in payload:
                value = payload[field]
                if (type(value) not in (int, float) or value < state[target]
                        or (target.endswith("lower_bound") and (type(value) is not int))):
                    raise ArtifactStateError("Durable optimizer/cost counters cannot decrease")
                state[target] = value
        state["event_count"] += 1
        state["last_event_sha256"] = row["sha256"]
        return state

    def _append(self, kind, payload):
        self._assert_usable()
        payload = json.loads(_canonical(payload))
        if not isinstance(payload, dict):
            raise TypeError("Event payload must be a mapping")
        row = {"sequence": self._state["event_count"], "kind": kind,
               "utc_unix_seconds": time.time(), "identity_sha256": self.identity_sha256,
               "previous_sha256": self._state["last_event_sha256"], "payload": payload}
        row["sha256"] = _sha(_canonical(row))
        advanced = self._advance(self._state, row)
        try:
            journal = self.path / "events.jsonl"
            self._reject_symlink(journal)
            with journal.open("ab") as handle:
                handle.write(_canonical(row) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._write_json(self.path / "manifest.json", advanced)
        except BaseException:
            self._io_failed = True
            raise
        self._state = advanced
        return deepcopy(row)

    def prepare(self, metadata):
        self._assert_usable()
        metadata = json.loads(_canonical(metadata))
        if not isinstance(metadata, dict) or not {"config", "source", "data", "seed", "resources"} <= metadata.keys():
            raise ValueError("Prepare needs config, source, data, seed, and resources metadata")
        if any(not isinstance(metadata[key], dict) or not metadata[key]
               for key in ("config", "source", "data", "resources")) or type(metadata["seed"]) is not int:
            raise ValueError("Config/source/data/resources must be nonempty mappings and seed an integer")
        if "seed" in self.identity and self.identity["seed"] != metadata["seed"]:
            raise ArtifactIdentityMismatch("Prepared seed differs from immutable attempt identity")
        digest = _sha(_canonical(metadata))
        if self._prepared_here and self.status == "prepared":
            if digest != self._state["prepared_metadata_sha256"]:
                raise ArtifactIdentityMismatch("Prepared metadata changed")
            return self.manifest
        if not self._fresh or self.status != "new":
            raise ArtifactReconciliationRequired("Existing attempts cannot be prepared or retried automatically")
        try:
            self._write_json(self.path / "identity.json", {
                "version": VERSION, "attempt_id": self.attempt_id,
                "identity": self.identity, "identity_sha256": self.identity_sha256})
            self._write_json(self.path / "prepared.json", metadata)
            self._append("prepared", {"metadata_sha256": digest})
        except BaseException:
            self._io_failed = True
            raise
        self._fresh = False
        self._prepared_here = True
        self._active_here = True
        return self.manifest

    def begin_training(self, metadata=None):
        self._assert_usable()
        if not self._prepared_here or self.status != "prepared":
            raise ArtifactReconciliationRequired("Training requires a freshly prepared attempt; no automatic retry")
        return self._append("training_started", metadata or {})

    def event(self, kind, payload=None):
        self._assert_active()
        if not isinstance(kind, str) or not kind or kind in _RESERVED:
            raise ValueError("Use the dedicated method for reserved artifact events")
        if self.status not in ("prepared", "training"):
            raise ArtifactStateError("Events require an active prepared/training attempt")
        return self._append(kind, payload or {})

    emit = event

    def _save(self, name, writer, file_format, collection, label_key):
        self._assert_active()
        if self.status not in ("prepared", "training"):
            raise ArtifactStateError("Saving requires an active attempt")
        name = _safe_name(name)
        if collection == "checkpoints" and name == "initial" and name in self._state[collection]:
            raise ArtifactStateError("Initial weights are immutable")
        if collection == "artifacts" and name in self._state[collection]:
            raise ArtifactStateError("Named result artifacts are immutable")
        relative = Path(collection) / (f"{self._state['event_count']:08d}-{name}.{file_format}")
        destination = self.path / relative
        if destination.exists():
            raise ArtifactReconciliationRequired("Unjournaled artifact already exists")
        try:
            self._atomic_write(destination, writer)
            receipt = {label_key: name, "path": str(relative), "sha256": _file_sha(destination),
                       "bytes": destination.stat().st_size, "format": file_format,
                       "identity_sha256": self.identity_sha256,
                       "optimizer_steps_known_lower_bound": self._state["optimizer_steps_known_lower_bound"],
                       "known_cost_seconds": self._state["known_cost_seconds"]}
            self._append("checkpoint_saved" if collection == "checkpoints" else "artifact_saved", receipt)
            return deepcopy(receipt)
        except BaseException:
            self._io_failed = True
            raise

    def checkpoint(self, payload, label="latest"):
        if isinstance(payload, (bytes, bytearray, memoryview)):
            return self._save(label, lambda handle: handle.write(bytes(payload)), "bin", "checkpoints", "label")
        # Importing torch is deliberately deferred; stdlib integrity tests need no GPU/framework.
        import torch
        return self._save(label, lambda handle: torch.save(payload, handle), "pt", "checkpoints", "label")

    def save_checkpoint(self, label, payload):
        return self.checkpoint(payload, label=label)

    def save_bytes(self, name, payload):
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("save_bytes expects bytes")
        return self._save(name, lambda handle: handle.write(bytes(payload)), "bin", "artifacts", "name")

    def write_json(self, name, payload):
        content = _canonical(payload) + b"\n"
        return self._save(name, lambda handle: handle.write(content), "json", "artifacts", "name")

    def _verify_artifacts(self):
        for collection in ("checkpoints", "artifacts"):
            receipts = (self._state["checkpoint_history"] if collection == "checkpoints"
                        else self._state[collection].values())
            for receipt in receipts:
                relative = Path(receipt["path"])
                if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != collection:
                    raise ArtifactReconciliationRequired("Invalid artifact receipt path")
                path = self.path / relative
                self._reject_symlink(path.parent)
                self._reject_symlink(path)
                if (not path.is_file() or path.stat().st_size != receipt["bytes"]
                        or _file_sha(path) != receipt["sha256"]):
                    raise ArtifactReconciliationRequired("Missing/changed durable artifact: " + str(relative))

    def require_reconciled(self):
        self._assert_usable()
        if self.status != "new":
            self._read_existing()
        if self.status not in ("completed", "paused_clean") or self._state["optimizer_step_open"]:
            raise ArtifactReconciliationRequired("Open, failed, or interrupted attempt requires explicit reconciliation")
        return self.manifest

    def complete(self, metadata=None):
        self._assert_active()
        if self.status != "training" or self._state["optimizer_step_open"]:
            raise ArtifactStateError("Only a training attempt without an open optimizer step can complete")
        if not {"initial", "latest", "best"} <= self._state["checkpoints"].keys():
            raise ArtifactStateError("Completion requires durable initial/latest/best checkpoints")
        self._verify_artifacts()
        return self._append("completed", metadata or {})

    def pause_clean(self, metadata):
        self._assert_active()
        if self.status != "training" or self._state["optimizer_step_open"]:
            raise ArtifactStateError("A clean pause cannot seal an open optimizer step")
        latest = self._state["checkpoints"].get("latest")
        if not latest or metadata.get("checkpoint_sha256") != latest["sha256"] or metadata.get("epoch_boundary") is not True:
            raise ArtifactStateError("Clean pause must attest an epoch boundary and the durable latest checkpoint")
        self._verify_artifacts()
        return self._append("paused_clean", metadata)

    def resume_clean(self, metadata):
        self.require_reconciled()
        latest = self._state["checkpoints"].get("latest")
        if (self.status != "paused_clean" or not latest
                or metadata.get("checkpoint_sha256") != latest["sha256"]
                or metadata.get("resume_state_restored") is not True):
            raise ArtifactStateError("Resume requires explicit restoration of the clean checkpoint, optimizer, and RNG state")
        result = self._append("resumed_clean", metadata)
        self._active_here = True
        return result

    def finish(self, status, metadata=None):
        if status == "completed":
            return self.complete(metadata)
        if status == "paused_clean":
            return self.pause_clean(metadata or {})
        self._assert_active()
        if status not in ("failed", "interrupted") or self.status not in ("prepared", "training"):
            raise ArtifactStateError("Invalid terminal artifact state")
        payload = dict(metadata or {})
        payload["requires_reconciliation"] = True
        return self._append(status, payload)

    def close(self):
        if not getattr(self, "_closed", True):
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()
