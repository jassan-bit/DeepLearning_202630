"""GitHub-backed orchestration around the unchanged scientific development CLI.

This helper receives an isolated Python executable and environment explicitly.
It does not mount Drive, ask for credentials, install packages, change the plan,
or reset a ledger. Every training chunk requires an acknowledged ACTIVE Git
snapshot, followed by an acknowledged CLOSED snapshot before the next chunk.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e3_github_storage import (BRANCH, PREFIX, REPOSITORY, GitHubRunStore,
                                       GitTransport, StorageConflict, StorageError,
                                       _now, assert_claimable, build_snapshot,
                                       canonical, check_progress, sha256)

# Deliberately not serialized or restored from a portable file. A new VM does
# not become the owner merely by reading another VM's ACTIVE snapshot.
_OWNERS = {}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode() + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _lock(path):
    try:
        import fcntl
    except ImportError:
        raise StorageError("POSIX_lock_required_for_GitHub_orchestration") from None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            raise StorageConflict("local_run_already_owned_by_another_process") from None
        yield


def child_environment(environment, token):
    if not isinstance(environment, dict):
        raise StorageError("explicit_isolated_child_environment_required")
    return {key: str(value) for key, value in environment.items()
            if key not in {"GITHUB_TOKEN", "GH_TOKEN", "E3_GITHUB_TOKEN"}
            and not re.search(r"(?:^|_)(?:TOKEN|PAT|SECRET|PASSWORD)(?:_|$)", key.upper())
            and str(value) != token}


def _command_runner(argv, env):
    """Stream the token-free scientific CLI, terminate its process group on cancel."""
    process = subprocess.Popen(argv, env=env, start_new_session=True)
    try:
        return process.wait()
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise


def _cli(command, *, code_root, python, environment, npz_path, expected_sha256,
         run_dir, plan_path, device, command_runner):
    arguments = [str(python), str(Path(code_root) / "scripts/e3_run_development.py"), command,
                 "--npz", str(npz_path), "--expected-sha256", expected_sha256,
                 "--run-dir", str(run_dir), "--device", device]
    if command == "preflight":
        arguments += ["--plan", str(plan_path)]
    else:
        arguments += ["--stop-after-epochs", "2"]
    result = command_runner(arguments, environment)
    return int(result.returncode if hasattr(result, "returncode") else result)


def _validate_local_identity(run_dir, code_root, npz_path, expected_sha256):
    manifest = _read(run_dir / "manifest.json")
    if (manifest["npz_sha256"] != expected_sha256.lower()
            or sha256(Path(npz_path).read_bytes()) != expected_sha256.lower()):
        raise StorageConflict("local_input_identity_mismatch")
    for name, expected in manifest["source_hashes"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or sha256((Path(code_root) / relative).read_bytes()) != expected:
            raise StorageConflict("local_scientific_source_identity_mismatch")
    return manifest


def _receipt(run_dir, result, manifest):
    value = {**result, "repository": REPOSITORY, "branch": BRANCH, "prefix": PREFIX,
             "run_id": manifest["run_id"], "url": "https://github.com/" + REPOSITORY + "/tree/" + BRANCH + "/" + PREFIX}
    _atomic_json(run_dir / ".github_storage_receipt.json", value)
    print(json.dumps({"event": "github_snapshot_acknowledged", **value}), flush=True)
    return value


def _publish(store, run_dir, expected_head, owner, now):
    with _lock(run_dir / ".run.lock"):
        result = store.publish(run_dir, expected_head=expected_head, owner_id=owner, now=now)
        manifest = _read(run_dir / "manifest.json")
        return _receipt(run_dir, result, manifest)


def _prepare(store, *, token, npz_path, expected_sha256, run_dir, plan_path,
             code_root, python, environment, device, command_runner, now):
    remote = store.read()  # Auth/network failures must never masquerade as empty storage.
    if remote and not run_dir.exists():
        store.restore(run_dir, head=remote[0])
    if not run_dir.exists():
        status = _cli("preflight", code_root=code_root, python=python, environment=environment,
                      npz_path=npz_path, expected_sha256=expected_sha256, run_dir=run_dir,
                      plan_path=plan_path, device=device, command_runner=command_runner)
        if status:
            raise StorageError("scientific_preflight_failed_no_training")
    with _lock(run_dir / ".run.lock"):
        manifest = _validate_local_identity(run_dir, code_root, npz_path, expected_sha256)
        owner = _OWNERS.setdefault(manifest["run_id"], uuid.uuid4().hex)
        local_manifest, local_files = build_snapshot(run_dir, owner_id=owner, now=now)
        if remote:
            check_progress(remote[1], local_manifest)
            if local_files == remote[2]:
                try:
                    assert_claimable(remote[1], owner_id=owner, now=now)
                    ready = True
                except StorageConflict:
                    ready = False
                return {"remote_commit": remote[0], "run_id": manifest["run_id"], "ready": ready,
                        "status": "restored_active_inspection_only" if not ready else "already_synchronized",
                        "owner": owner}
            assert_claimable(remote[1], owner_id=owner, now=now)
        result = store.publish(run_dir, expected_head=remote[0] if remote else None,
                               owner_id=owner, now=now)
        receipt = _receipt(run_dir, result, manifest)
        return {**receipt, "ready": True, "owner": owner}


@contextmanager
def _store_context(*, token, run_dir, transport):
    run_dir = Path(run_dir)
    outer = run_dir.parent / ("." + run_dir.name + ".github_storage.lock")
    with _lock(outer):
        owned = transport is None
        actual = GitTransport(token) if owned else transport
        try:
            yield GitHubRunStore(actual, token=token)
        finally:
            if owned:
                actual.close()


def prepare_github_run(*, token, npz_path, expected_sha256, run_dir, plan_path,
                       code_root, python=None, environment=None, device="auto",
                       transport=None, command_runner=None, now=None):
    """Restore or preflight, then acknowledge GitHub before allowing any fitting."""
    if python is None:
        raise StorageError("explicit_isolated_python_required")
    environment = child_environment(environment, token)
    run_dir = Path(run_dir)
    with _store_context(token=token, run_dir=run_dir, transport=transport) as store:
        result = _prepare(store, token=token, npz_path=npz_path, expected_sha256=expected_sha256,
                          run_dir=run_dir, plan_path=plan_path, code_root=code_root, python=python,
                          environment=environment, device=device,
                          command_runner=command_runner or _command_runner, now=now)
        result.pop("owner", None)
        return result


def run_github_development(*, token, npz_path, expected_sha256, run_dir, plan_path,
                           code_root, python=None, environment=None, device="auto",
                           max_chunks=40, transport=None, command_runner=None, now=None):
    """Run two-model-epoch chunks with acknowledged ACTIVE/CLOSED Git boundaries."""
    if python is None or type(max_chunks) is not int or not 1 <= max_chunks <= 40:
        raise StorageError("explicit_python_and_maximum_40_chunks_required")
    environment = child_environment(environment, token)
    run_dir = Path(run_dir)
    runner = command_runner or _command_runner
    with _store_context(token=token, run_dir=run_dir, transport=transport) as store:
        prepared = _prepare(store, token=token, npz_path=npz_path, expected_sha256=expected_sha256,
                            run_dir=run_dir, plan_path=plan_path, code_root=code_root, python=python,
                            environment=environment, device=device, command_runner=runner, now=now)
        if not prepared["ready"]:
            raise StorageConflict("restored_foreign_active_run_is_inspection_only_until_lease_expires")
        owner, head = prepared["owner"], prepared["remote_commit"]
        for _chunk in range(max_chunks):
            with _lock(run_dir / ".run.lock"):
                state, ledger = _read(run_dir / "state.json"), _read(run_dir / "ledger.json")
                if state["status"] in {"completed", "budget_exhausted"} or ledger["charged_seconds"] >= ledger["limit_seconds"]:
                    return {"status": state["status"], "remote_commit": head,
                            "summary": _read(run_dir / "summary.json") if (run_dir / "summary.json").is_file() else None}
                remote = store.read()
                if remote is None or remote[0] != head:
                    raise StorageConflict("remote_changed_before_training_claim")
                assert_claimable(remote[1], owner_id=owner, now=now)
                # Preserve an ACTIVE timestamp from recovery. Replacing it would
                # erase uncharged downtime before the v1 runner can account it.
                if ledger.get("active_session") is None:
                    ledger["active_session"] = {"session_id": "github-" + uuid.uuid4().hex,
                                                "last_wall_utc": _now(now).isoformat()}
                ledger["events"].append({"kind": "github_chunk_reserved", "owner": owner,
                                          "utc": _now(now).isoformat()})
                _atomic_json(run_dir / "ledger.json", ledger)
                result = store.publish(run_dir, expected_head=head, owner_id=owner, now=now)
                head = result["remote_commit"]
                _receipt(run_dir, result, _read(run_dir / "manifest.json"))
            command = "run" if state["status"] == "initialized" else "resume"
            failure = None
            try:
                return_code = _cli(command, code_root=code_root, python=python, environment=environment,
                                   npz_path=npz_path, expected_sha256=expected_sha256, run_dir=run_dir,
                                   plan_path=plan_path, device=device, command_runner=runner)
            except BaseException as error:
                failure, return_code = error, 1
            # No next fit is possible until this exact local result is ACKed.
            receipt = _publish(store, run_dir, head, owner, now)
            head = receipt["remote_commit"]
            ledger = _read(run_dir / "ledger.json")
            if failure is not None:
                raise failure
            if return_code or ledger.get("active_session") is not None:
                raise StorageError("scientific_child_failed_snapshot_preserved_no_training_retry")
        state = _read(run_dir / "state.json")
        return {"status": state["status"], "remote_commit": head,
                "summary": _read(run_dir / "summary.json") if (run_dir / "summary.json").is_file() else None}
