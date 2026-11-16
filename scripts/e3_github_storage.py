"""Small, non-force Git snapshots of the E3 development run, never E2/main.

Credentials exist only in the private Git subprocess environment. Snapshot
contents are positively selected and validated before any checkpoint can be
deserialized by the separate scientific runner.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import tempfile
import uuid

REPOSITORY = "jassan-bit/DeepLearning_202630"
BRANCH = "e3-desarrollo-resultados-v1"
PREFIX = "runs/desarrollo_v1"
MANIFEST_PATH = PREFIX + "/snapshot_manifest.json"
VERSION = "e3-github-development-storage-v1"
MAX_FILE_BYTES = 25 * 1024 ** 2
MAX_SNAPSHOT_BYTES = 64 * 1024 ** 2
BASE_FILES = {"manifest.json", "plan.json", "runtime.json", "data_admission.json",
              "state.json", "ledger.json", "history.jsonl", "summary.json"}
MODEL_FILES = {"last.pt", "last.pt.sha256", "best.pt", "best.pt.sha256",
               "validation_predictions.csv", "curves.csv"}
REQUIRED = BASE_FILES - {"summary.json"}
REQUIRED |= {name + "/" + item for name in ("full", "M00") for item in ("last.pt", "last.pt.sha256")}
IGNORED = {".run.lock", ".github_storage.lock", ".github_storage_receipt.json", "colab_console.log"}


class StorageError(RuntimeError):
    """A fixed safe error code; never includes Git stderr, token or response body."""


class StorageConflict(StorageError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def _decode(raw):
    try:
        return json.loads(raw.decode("utf-8"), parse_constant=lambda _x: (_ for _ in ()).throw(ValueError()))
    except (ValueError, UnicodeError, AttributeError):
        raise StorageError("invalid_snapshot_json") from None


def _now(value=None):
    value = value() if callable(value) else value
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise StorageError("timezone_aware_clock_required")
    return value.astimezone(timezone.utc)


def approved_path(relative):
    if not isinstance(relative, str) or "\\" in relative or "\x00" in relative:
        return False
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or str(path) != relative:
        return False
    return relative in BASE_FILES or (len(path.parts) == 2 and path.parts[0] in {"full", "M00"}
                                      and path.parts[1] in MODEL_FILES)


def _validate_digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_snapshot(manifest, files):
    """Validate complete bytes, run identity and ledger without loading pickle."""
    try:
        if (manifest.get("version") != VERSION or not isinstance(files, dict)
                or set(files) != set(manifest["files"]) or not REQUIRED <= set(files)):
            raise StorageError("incomplete_snapshot")
        if sum(len(value) for value in files.values()) > MAX_SNAPSHOT_BYTES:
            raise StorageError("snapshot_too_large")
        for path, data in files.items():
            record = manifest["files"][path]
            if (not approved_path(path) or not isinstance(data, bytes) or len(data) > MAX_FILE_BYTES
                    or record["bytes"] != len(data) or not _validate_digest(record["sha256"])
                    or sha256(data) != record["sha256"]):
                raise StorageError("snapshot_path_size_or_hash_mismatch")
        identity, ledger = _decode(files["manifest.json"]), _decode(files["ledger.json"])
        state = _decode(files["state.json"])
        if (not isinstance(identity["run_id"], str)
                or re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", identity["run_id"]) is None
                or identity["run_id"] != manifest["run_id"]):
            raise StorageError("snapshot_run_identity_mismatch")
        for field in ("plan_sha256", "data_fingerprint", "source_hashes"):
            if identity[field] != manifest[field]:
                raise StorageError("snapshot_scientific_fingerprint_mismatch")
        if manifest["state_status"] != state["status"] or manifest["model_epochs"] != {name: state["models"][name]["epoch"] for name in ("full", "M00")}:
            raise StorageError("snapshot_progress_mismatch")
        if any(type(value) is not int or value < 0 for value in manifest["model_epochs"].values()):
            raise StorageError("invalid_model_epoch")
        if sha256(canonical(_decode(files["plan.json"]))) != identity["plan_sha256"]:
            raise StorageError("snapshot_plan_hash_mismatch")
        if sha256(canonical(_decode(files["data_admission.json"]))) != identity["data_fingerprint"]:
            raise StorageError("snapshot_data_hash_mismatch")
        charged, limit = ledger["charged_seconds"], ledger["limit_seconds"]
        if (type(charged) not in (int, float) or type(limit) not in (int, float)
                or not math.isfinite(charged) or not math.isfinite(limit)
                or charged < 0 or not 0 < limit <= 1200
                or manifest["charged_seconds"] != charged or manifest["limit_seconds"] != limit
                or ledger.get("active_session") != manifest.get("active_session")):
            raise StorageError("snapshot_budget_mismatch")
        for name in ("full", "M00"):
            for item in ("last.pt", "best.pt"):
                path, receipt = name + "/" + item, name + "/" + item + ".sha256"
                if path in files and (receipt not in files or sha256(files[path]) not in files[receipt].decode().splitlines()):
                    raise StorageError("snapshot_checkpoint_receipt_mismatch")
        active = manifest.get("active_session")
        if active:
            _now(active["last_wall_utc"])
            if not manifest.get("lease_owner") or _now(manifest["lease_until_utc"]) <= _now(manifest["lease_claimed_at_utc"]):
                raise StorageError("invalid_active_lease")
        return manifest
    except StorageError:
        raise
    except (KeyError, TypeError, ValueError, UnicodeError):
        raise StorageError("invalid_snapshot_schema") from None


def build_snapshot(run_dir, *, owner_id=None, now=None):
    """Read a quiescent run under the caller's lock; keep raw NPZ/source/token out."""
    root = Path(run_dir)
    if root.is_symlink() or not root.is_dir():
        raise StorageError("regular_run_directory_required")
    files = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise StorageError("snapshot_symlink_refused")
        if path.is_dir():
            if relative not in {"full", "M00"}:
                raise StorageError("snapshot_directory_refused")
            continue
        if relative in IGNORED or path.name.endswith(".tmp"):
            continue
        if not approved_path(relative):
            raise StorageError("snapshot_file_not_allowlisted")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise StorageError("snapshot_file_too_large")
        files[relative] = path.read_bytes()
    identity, ledger = _decode(files.get("manifest.json", b"")), _decode(files.get("ledger.json", b""))
    active = ledger.get("active_session")
    stamp = _now(now)
    state = _decode(files["state.json"])
    manifest = {"version": VERSION, "run_id": identity["run_id"],
                "plan_sha256": identity["plan_sha256"], "data_fingerprint": identity["data_fingerprint"],
                "source_hashes": identity["source_hashes"],
                "charged_seconds": ledger["charged_seconds"], "limit_seconds": ledger["limit_seconds"],
                "active_session": active, "state_status": state["status"],
                "model_epochs": {name: state["models"][name]["epoch"] for name in ("full", "M00")},
                "files": {path: {"sha256": sha256(data), "bytes": len(data)} for path, data in sorted(files.items())}}
    if active:
        if not isinstance(owner_id, str) or not owner_id:
            raise StorageError("active_snapshot_requires_owner")
        manifest.update(lease_owner=owner_id, lease_claimed_at_utc=stamp.isoformat(),
                        lease_until_utc=(stamp + timedelta(seconds=max(0, ledger["limit_seconds"] - ledger["charged_seconds"]) + 60)).isoformat())
    validate_snapshot(manifest, files)
    return manifest, files


def assert_claimable(manifest, owner_id=None, *, now=None):
    if (manifest.get("active_session") and manifest.get("lease_owner") != owner_id
            and _now(now) < _now(manifest["lease_until_utc"])):
        raise StorageConflict("remote_active_lease_owned_by_another_session")


def restore_snapshot(manifest, files, destination):
    validate_snapshot(manifest, files)
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise StorageConflict("restore_destination_must_be_new")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".e3-restore-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "run"
        staging.mkdir()
        for relative, data in files.items():
            target = staging / relative
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(data)
        # Active ledger bytes/time are preserved exactly; the v1 runner charges
        # their gap before changing the active session on a legitimate resume.
        os.rename(staging, destination)


def check_progress(previous, current):
    for field in ("run_id", "plan_sha256", "data_fingerprint", "source_hashes", "limit_seconds"):
        if previous[field] != current[field]:
            raise StorageConflict("remote_run_identity_or_plan_conflict")
    if current["charged_seconds"] < previous["charged_seconds"]:
        raise StorageConflict("remote_budget_cannot_decrease")
    if any(current["model_epochs"][name] < previous["model_epochs"][name] for name in ("full", "M00")):
        raise StorageConflict("remote_model_progress_cannot_decrease")


class GitTransport:
    """Bare Git cache, exact destination allowlist, non-force compare-and-swap."""
    def __init__(self, token, repository=REPOSITORY, branch=BRANCH, timeout=45,
                 _test_remote=None, _allow_local_for_tests=False):
        if repository != REPOSITORY or branch != BRANCH:
            raise StorageError("only_dedicated_e3_repository_branch_is_allowed")
        if not isinstance(token, str) or not token or any(x.isspace() for x in token):
            raise StorageError("github_token_required")
        if _test_remote is not None and not _allow_local_for_tests:
            raise StorageError("local_transport_is_test_only")
        self.repository, self.branch, self.timeout = repository, branch, timeout
        self._temporary = tempfile.TemporaryDirectory(prefix="e3-git-storage-")
        self._private = Path(self._temporary.name)
        self._cache = self._private / "objects.git"
        self._remote = str(Path(_test_remote).resolve()) if _test_remote else "https://github.com/" + repository + ".git"
        self._ref = "refs/heads/" + branch
        askpass = self._private / "askpass.sh"
        askpass.write_text('#!/bin/sh\ncase "$1" in\n*Username*|*username*) printf "%s\\n" "x-access-token";;\n*Password*|*password*) printf "%s\\n" "$E3_GITHUB_TOKEN";;\n*) exit 1;;\nesac\n')
        askpass.chmod(0o700)
        self._environment = {key: value for key, value in os.environ.items()
                             if not key.startswith("GIT_") and key not in {"GH_TOKEN", "GITHUB_TOKEN", "SSH_ASKPASS", "E3_GITHUB_TOKEN"}}
        self._environment.update(E3_GITHUB_TOKEN=token, GIT_ASKPASS=str(askpass), GIT_TERMINAL_PROMPT="0",
                                 GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1", LC_ALL="C",
                                 GIT_AUTHOR_NAME="E3 development storage", GIT_AUTHOR_EMAIL="e3@users.noreply.github.com",
                                 GIT_COMMITTER_NAME="E3 development storage", GIT_COMMITTER_EMAIL="e3@users.noreply.github.com")
        self._options = ["-c", "credential.helper=", "-c", "core.hooksPath=" + os.devnull,
                         "-c", "http.followRedirects=false", "-c", "protocol.ext.allow=never",
                         "-c", "maintenance.auto=false", "-c", "protocol.file.allow=" + ("always" if _test_remote else "never")]
        self._closed = False
        try:
            self._run(["init", "--bare", "--object-format=sha1", str(self._cache)], bare=False)
            self._run(["remote", "add", "origin", self._remote])
        except BaseException:
            self.close()
            raise

    def __repr__(self):
        return f"GitTransport(repository={self.repository!r}, branch={self.branch!r}, token=<redacted>)"

    def _run(self, args, *, data=None, env=None, bare=True, allow_failure=False):
        if self._closed:
            raise StorageError("git_transport_closed")
        command = ["git", *self._options]
        if bare:
            command += ["--git-dir=" + str(self._cache)]
        command += list(args)
        environment = dict(self._environment)
        if env:
            environment.update(env)
        process = None
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       env=environment, cwd=self._private, start_new_session=True)
            stdout, stderr = process.communicate(input=data, timeout=self.timeout)
        except BaseException as error:
            if process:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate()
            if isinstance(error, (OSError, subprocess.TimeoutExpired)):
                raise StorageError("git_operation_unconfirmed") from None
            raise
        result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        if result.returncode and not allow_failure:
            raise StorageError("git_operation_failed")
        return result

    @staticmethod
    def _sha(value):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise StorageError("invalid_git_object")
        return value

    def head(self):
        result = self._run(["ls-remote", "--exit-code", "origin", self._ref], allow_failure=True)
        if result.returncode == 2:
            return None
        if result.returncode:
            raise StorageError("remote_read_failed_not_an_empty_branch")
        self._run(["fetch", "--depth=1", "--no-tags", "origin", self._ref])
        return self._sha(self._run(["rev-parse", "FETCH_HEAD"]).stdout.decode().strip())

    def read_snapshot(self, head):
        head = self._sha(head)
        raw = self._run(["show", head + ":" + MANIFEST_PATH]).stdout
        manifest = _decode(raw)
        files = {}
        for path in manifest.get("files", {}):
            if not approved_path(path):
                raise StorageError("remote_snapshot_path_refused")
            files[path] = self._run(["show", head + ":" + PREFIX + "/" + path]).stdout
        validate_snapshot(manifest, files)
        return manifest, files

    def publish(self, expected_head, manifest, files):
        validate_snapshot(manifest, files)
        actual_head = self.head()
        if actual_head != expected_head:
            raise StorageConflict("remote_head_changed")
        index = self._private / "snapshot-index"
        index.unlink(missing_ok=True)
        env = {"GIT_INDEX_FILE": str(index)}
        if expected_head:
            tree = self._run(["show", "-s", "--format=%T", self._sha(expected_head)]).stdout.decode().strip()
            self._run(["read-tree", tree], env=env)
        else:
            self._run(["read-tree", "--empty"], env=env)
        payloads = {PREFIX + "/" + path: value for path, value in files.items()}
        payloads[MANIFEST_PATH] = canonical(manifest) + b"\n"
        lines = []
        for path, value in sorted(payloads.items()):
            blob = self._sha(self._run(["hash-object", "-w", "--stdin"], data=value).stdout.decode().strip())
            lines.append(b"100644 " + blob.encode() + b"\t" + path.encode() + b"\0")
        self._run(["update-index", "-z", "--index-info"], data=b"".join(lines), env=env)
        tree = self._sha(self._run(["write-tree"], env=env).stdout.decode().strip())
        arguments = ["commit-tree", tree]
        if expected_head:
            arguments += ["-p", self._sha(expected_head)]
        commit = self._sha(self._run(arguments, data=b"Save E3 development checkpoint snapshot\n").stdout.decode().strip())
        result = self._run(["push", "--porcelain", "origin", commit + ":" + self._ref], allow_failure=True)
        if result.returncode:
            response = (result.stdout + result.stderr).lower()
            if any(x in response for x in (b"[rejected]", b"non-fast-forward", b"fetch first")):
                raise StorageConflict("remote_head_changed")
            raise StorageError("git_push_unconfirmed")
        return commit

    def close(self):
        if not self._closed:
            self._closed = True
            self._environment.pop("E3_GITHUB_TOKEN", None)
            self._temporary.cleanup()


class GitHubRunStore:
    def __init__(self, transport, token=None):
        self.transport, self._token = transport, token

    def read(self):
        head = self.transport.head()
        if head is None:
            return None
        manifest, files = self.transport.read_snapshot(head)
        validate_snapshot(manifest, files)
        return head, manifest, files

    def publish(self, run_dir, *, expected_head, owner_id=None, now=None):
        manifest, files = build_snapshot(run_dir, owner_id=owner_id, now=now)
        if self._token and any(self._token.encode() in value for value in files.values()):
            raise StorageError("credential_detected_in_snapshot")
        current = self.read()
        # Exact replay is read-only and safe even when the prior ACK or its
        # readback was lost. A stale expected head cannot mutate divergent data.
        if current and canonical(current[1]) == canonical(manifest) and current[2] == files:
            return {"remote_commit": current[0], "status": "unchanged", "run_id": manifest["run_id"]}
        if (current[0] if current else None) != expected_head:
            raise StorageConflict("remote_head_changed")
        if current:
            previous = current[1]
            check_progress(previous, manifest)
            assert_claimable(previous, owner_id=owner_id, now=now)
            if canonical(previous) == canonical(manifest):
                return {"remote_commit": current[0], "status": "unchanged", "run_id": manifest["run_id"]}
        try:
            commit = self.transport.publish(expected_head, manifest, files)
        except StorageError:
            # A push can succeed while its ACK is lost. Read back exact bytes;
            # never repeat scientific fitting or overwrite a different head.
            found = self.read()
            if found and canonical(found[1]) == canonical(manifest) and found[2] == files:
                return {"remote_commit": found[0], "status": "ack_reconciled", "run_id": manifest["run_id"]}
            raise
        return {"remote_commit": commit, "status": "published", "run_id": manifest["run_id"]}

    def restore(self, destination, head=None):
        found = self.read()
        if found is None:
            raise StorageError("remote_snapshot_missing")
        if head is not None and found[0] != head:
            raise StorageConflict("remote_head_changed")
        restore_snapshot(found[1], found[2], destination)
        return {"remote_commit": found[0], "status": "restored", "run_id": found[1]["run_id"],
                "active_session_preserved": bool(found[1].get("active_session"))}
