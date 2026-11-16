"""Offline snapshot/CAS/credential checks; synthetic weights, no market fits."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "scripts", ROOT / "notebooks"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import e3_github_storage as storage
from src.e3_development import create_run


NOW = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)
TOKEN = "ghp_SYNTHETIC_TEST_VALUE_NOT_A_REAL_CREDENTIAL_74192"


class MemoryRemote:
    """An independent immutable-commit store with non-forcing CAS semantics."""

    def __init__(self):
        self.current = None
        self.commits = {}
        self.fail_head = False
        self.fail_publish = None

    def head(self):
        if self.fail_head:
            raise RuntimeError("Synthetic authentication failure (HTTP 403)")
        return self.current

    def read_snapshot(self, head):
        manifest, files = self.commits[head]
        return deepcopy(manifest), dict(files)

    def publish(self, expected_head, manifest, files):
        if self.fail_publish == "before":
            self.fail_publish = None
            raise storage.StorageError("Synthetic remote failure before ref update")
        if expected_head != self.current:
            raise RuntimeError("Synthetic non-fast-forward compare-and-swap conflict")
        new_head = hashlib.sha1(str(len(self.commits) + 1).encode()).hexdigest()
        self.commits[new_head] = (deepcopy(manifest), dict(files))
        self.current = new_head
        if self.fail_publish == "after":
            self.fail_publish = None
            raise storage.StorageError("Synthetic acknowledgement lost after ref update")
        return new_head


def _new_run(tmp_path, *, name="synthetic_run", active=False):
    dates = np.array(["2024-06-29", "2024-06-30", "2024-07-08", "2024-07-09"])
    raw = np.empty((4, 7, 5), dtype=np.float64)
    for sample in range(4):
        for lag in range(7):
            close = 100 + sample + lag * 0.2
            raw[sample, lag] = [close + 0.1, close + 1, close - 1, close, 500 + sample * 5 + lag]
    npz = tmp_path / (name + ".npz")
    np.savez(npz, training_raw=raw, training_y_original=np.array([.01, .02, .03, .04]),
             training_dates=dates)
    run_dir = tmp_path / name
    create_run(npz, hashlib.sha256(npz.read_bytes()).hexdigest(), run_dir, device="cpu",
               plan={"seed": 7, "training": {"max_epochs": 1, "batch_size": 2}})
    ledger_path = run_dir / "ledger.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["charged_seconds"] = 19.5
    ledger["events"] = [{"kind": "synthetic_charge", "seconds": 19.5}]
    if active:
        ledger["active_session"] = {"session_id": "synthetic-owner-A", "last_wall_utc": NOW.isoformat()}
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    return run_dir


def _ledger(run_dir):
    return json.loads((run_dir / "ledger.json").read_text())


def test_snapshot_roundtrip_restores_checkpoint_bytes_and_active_budget_exactly(tmp_path):
    run_dir = _new_run(tmp_path, active=True)
    manifest, files = storage.build_snapshot(run_dir, owner_id="owner-A", now=NOW)
    storage.validate_snapshot(manifest, files)
    restored = tmp_path / "restored"
    storage.restore_snapshot(manifest, files, restored)
    for name, payload in files.items():
        assert (restored / name).read_bytes() == payload
    assert (restored / "full" / "last.pt").read_bytes() == (run_dir / "full" / "last.pt").read_bytes()
    assert (restored / "M00" / "last.pt").read_bytes() == (run_dir / "M00" / "last.pt").read_bytes()
    assert _ledger(restored) == _ledger(run_dir)
    assert _ledger(restored)["charged_seconds"] == 19.5
    assert _ledger(restored)["active_session"]["last_wall_utc"] == NOW.isoformat()
    assert not any(name.endswith(".lock") for name in files)
    with pytest.raises((ValueError, RuntimeError, FileExistsError)):
        storage.restore_snapshot(manifest, files, restored)


@pytest.mark.parametrize("regression", ["budget", "run_identity"])
def test_publishing_cannot_reset_spent_budget_or_replace_run_identity(tmp_path, regression):
    run_dir = _new_run(tmp_path)
    remote = MemoryRemote()
    store = storage.GitHubRunStore(remote)
    store.publish(run_dir, expected_head=None)
    original_head = remote.head()
    if regression == "budget":
        path = run_dir / "ledger.json"
        ledger = json.loads(path.read_text())
        ledger["charged_seconds"] = 0.0
        path.write_text(json.dumps(ledger))
    else:
        path = run_dir / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["run_id"] = "a-different-synthetic-run"
        path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, RuntimeError)):
        store.publish(run_dir, expected_head=original_head)
    assert remote.head() == original_head
    assert len(remote.commits) == 1


@pytest.mark.parametrize("bad_path", ["../escaped.pt", "/escaped.pt", ".git/config"])
def test_self_consistent_malicious_paths_cannot_escape_restore(tmp_path, bad_path):
    run_dir = _new_run(tmp_path)
    manifest, files = storage.build_snapshot(run_dir)
    payload = b"synthetic untrusted content"
    files[bad_path] = payload
    manifest["files"][bad_path] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    target = tmp_path / "unsafe_restore"
    with pytest.raises((ValueError, RuntimeError)):
        storage.restore_snapshot(manifest, files, target)
    assert not target.exists()
    assert not (tmp_path / "escaped.pt").exists()


def test_corrupt_blob_and_local_symlink_are_rejected_without_partial_restore(tmp_path):
    run_dir = _new_run(tmp_path)
    manifest, files = storage.build_snapshot(run_dir)
    damaged = dict(files)
    checkpoint_name = "full/last.pt"
    damaged[checkpoint_name] = damaged[checkpoint_name] + b"corruption"
    target = tmp_path / "corrupt_restore"
    with pytest.raises((ValueError, RuntimeError)):
        storage.restore_snapshot(manifest, damaged, target)
    assert not target.exists()
    external = tmp_path / "external_checkpoint"
    external.write_bytes(files[checkpoint_name])
    (run_dir / checkpoint_name).unlink()
    (run_dir / checkpoint_name).symlink_to(external)
    with pytest.raises((ValueError, RuntimeError)):
        storage.build_snapshot(run_dir)


def test_competing_claims_are_cas_protected_and_fresh_active_lease_blocks_other_owner(tmp_path):
    run_dir = _new_run(tmp_path)
    remote = MemoryRemote()
    first_store, second_store = storage.GitHubRunStore(remote), storage.GitHubRunStore(remote)
    first_store.publish(run_dir, expected_head=None)
    previous_head = remote.head()
    assert first_store.read()[0] == second_store.read()[0] == previous_head
    second_run = tmp_path / "second_writer"
    shutil.copytree(run_dir, second_run)
    for directory, charge in ((run_dir, 20.0), (second_run, 21.0)):
        ledger = _ledger(directory)
        ledger["charged_seconds"] = charge
        (directory / "ledger.json").write_text(json.dumps(ledger))
    first_store.publish(run_dir, expected_head=previous_head)
    winner = remote.head()
    with pytest.raises((ValueError, RuntimeError)):
        second_store.publish(second_run, expected_head=previous_head)
    assert remote.head() == winner and len(remote.commits) == 2
    _, winning_files = remote.read_snapshot(winner)
    assert json.loads(winning_files["ledger.json"])["charged_seconds"] == 20.0

    active_run = _new_run(tmp_path, name="active_run", active=True)
    active, active_files = storage.build_snapshot(active_run, owner_id="owner-A", now=NOW)
    with pytest.raises((ValueError, RuntimeError)):
        storage.assert_claimable(active, owner_id="owner-B", now=NOW + timedelta(seconds=1))
    storage.assert_claimable(active, owner_id="owner-A", now=NOW + timedelta(seconds=1))
    # Even after lease expiry, transport must preserve old UTC and the charge;
    # the unchanged v1 runner, rather than storage, accounts for the wall gap.
    storage.assert_claimable(active, owner_id="owner-B", now=NOW + timedelta(hours=1))
    restored = tmp_path / "active_recovered"
    storage.restore_snapshot(active, active_files, restored)
    assert _ledger(restored) == _ledger(active_run)


@pytest.mark.parametrize("failure", ["before", "after"])
def test_failed_git_publish_is_safely_retryable_including_lost_acknowledgement(tmp_path, failure):
    run_dir = _new_run(tmp_path)
    remote = MemoryRemote()
    store = storage.GitHubRunStore(remote)
    remote.fail_publish = failure
    try:
        store.publish(run_dir, expected_head=None)
    except (ConnectionError, RuntimeError):
        pass
    if failure == "before":
        assert remote.head() is None and not remote.commits
    else:
        assert remote.head() is not None and len(remote.commits) == 1
    store.publish(run_dir, expected_head=None)
    assert remote.head() is not None and len(remote.commits) == 1
    _, manifest, files = store.read()
    storage.validate_snapshot(manifest, files)
    assert json.loads(files["ledger.json"])["charged_seconds"] == 19.5


def test_authentication_failure_is_not_interpreted_as_missing_remote_run():
    remote = MemoryRemote()
    remote.fail_head = True
    with pytest.raises(RuntimeError, match="authentication"):
        storage.GitHubRunStore(remote).read()
    assert not remote.commits


def test_real_local_git_transport_keeps_token_out_of_snapshots_urls_and_errors(tmp_path, monkeypatch):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    run_dir = _new_run(tmp_path)
    transport = storage.GitTransport(TOKEN, _test_remote=str(remote), _allow_local_for_tests=True)
    store = storage.GitHubRunStore(transport, token=TOKEN)
    store.publish(run_dir, expected_head=None)
    head, manifest, files = store.read()
    assert head and TOKEN not in repr(transport)
    assert TOKEN.encode() not in json.dumps(manifest).encode()
    assert all(TOKEN.encode() not in payload for payload in files.values())
    assert TOKEN not in (remote / "config").read_text()
    observed_argv = []
    askpass_paths = []

    class AuthenticationFailure:
        returncode = 128

        def communicate(self, input=None, timeout=None):
            return b"", ("Synthetic auth error " + TOKEN).encode()

    def authentication_failure(argv, *args, **kwargs):
        observed_argv.append(argv)
        environment = kwargs["env"]
        askpass = Path(environment["GIT_ASKPASS"])
        askpass_paths.append(askpass)
        assert TOKEN not in askpass.read_text()
        return AuthenticationFailure()

    monkeypatch.setattr(storage.subprocess, "Popen", authentication_failure)
    with pytest.raises(storage.StorageError) as error:
        transport.head()
    assert TOKEN not in str(error.value)
    assert observed_argv and all(TOKEN not in repr(argv) for argv in observed_argv)
    transport.close()
    assert askpass_paths and all(not path.exists() for path in askpass_paths)


def _helper_arguments(run_dir, remote, command_runner):
    npz = run_dir.parent / (run_dir.name + ".npz")
    return {"token": TOKEN, "npz_path": npz,
            "expected_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
            "run_dir": run_dir, "plan_path": run_dir / "plan.json", "code_root": ROOT,
            "device": "cpu", "transport": remote, "command_runner": command_runner,
            "python": sys.executable,
            "environment": {"PATH": os.environ.get("PATH", ""), "EXPECTED_SAFE_VALUE": "preserved",
                            "GITHUB_TOKEN": TOKEN, "GH_TOKEN": TOKEN, "E3_GITHUB_TOKEN": TOKEN,
                            "OTHER_TOKEN_COPY": TOKEN},
            "now": NOW}


def _simulate_closed_child(run_dir, *, completed):
    ledger = _ledger(run_dir)
    ledger["active_session"] = None
    ledger["charged_seconds"] += 0.5
    (run_dir / "ledger.json").write_text(json.dumps(ledger))
    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text())
    state["status"] = "completed" if completed else "paused"
    state_path.write_text(json.dumps(state))


def test_helper_confirms_active_before_child_and_closed_before_next_chunk_with_isolated_environment(tmp_path):
    import run_e3_github_colab as helper

    run_dir = _new_run(tmp_path)
    events, fit_calls = [], []

    class TracingRemote(MemoryRemote):
        def publish(self, expected_head, manifest, files):
            head = super().publish(expected_head, manifest, files)
            events.append(("publish", bool(manifest.get("active_session"))))
            return head

    remote = TracingRemote()

    def child(argv, environment):
        assert argv[0] == sys.executable
        assert str(ROOT / "scripts" / "e3_run_development.py") in argv
        assert environment["EXPECTED_SAFE_VALUE"] == "preserved"
        assert not {"GITHUB_TOKEN", "GH_TOKEN", "E3_GITHUB_TOKEN"} & set(environment)
        assert TOKEN not in environment.values() and TOKEN not in repr(argv)
        assert any(command in argv for command in ("run", "resume"))
        remote_manifest, remote_files = remote.read_snapshot(remote.head())
        assert remote_manifest["active_session"]
        assert json.loads(remote_files["ledger.json"])["active_session"]
        fit_calls.append(argv)
        events.append(("fit", len(fit_calls)))
        _simulate_closed_child(run_dir, completed=len(fit_calls) == 2)
        return 0

    helper.run_github_development(**_helper_arguments(run_dir, remote, child), max_chunks=2)
    fit_positions = [i for i, event in enumerate(events) if event[0] == "fit"]
    assert len(fit_positions) == 2
    for index, position in enumerate(fit_positions):
        preceding = [event for event in events[:position] if event[0] == "publish"]
        assert preceding and preceding[-1] == ("publish", True)
        following_end = fit_positions[index + 1] if index + 1 < len(fit_positions) else len(events)
        assert ("publish", False) in events[position + 1:following_end]
    assert events[-1] == ("publish", False)
    final_manifest, final_files = remote.read_snapshot(remote.head())
    assert not final_manifest["active_session"]
    assert json.loads(final_files["ledger.json"])["charged_seconds"] == 20.5


@pytest.mark.parametrize("failure", ["authentication", "active_publish", "closed_publish", "foreign_active"])
def test_helper_does_not_fit_after_auth_or_required_publish_failure(tmp_path, failure):
    import run_e3_github_colab as helper

    run_dir = _new_run(tmp_path)
    fit_calls = []

    class FailingRemote(MemoryRemote):
        def publish(self, expected_head, manifest, files):
            active = bool(manifest.get("active_session"))
            if failure == "active_publish" and active:
                raise storage.StorageError("Synthetic active publication failure")
            if failure == "closed_publish" and not active and fit_calls:
                raise storage.StorageError("Synthetic closed publication failure")
            return super().publish(expected_head, manifest, files)

    remote = FailingRemote()
    remote.fail_head = failure == "authentication"
    arguments = _helper_arguments(run_dir, remote, None)
    protected_head = None
    if failure == "foreign_active":
        ledger = _ledger(run_dir)
        ledger["active_session"] = {"session_id": "other-vm-session", "last_wall_utc": NOW.isoformat()}
        (run_dir / "ledger.json").write_text(json.dumps(ledger))
        storage.GitHubRunStore(remote).publish(run_dir, expected_head=None,
                                               owner_id="other-vm-owner", now=NOW)
        protected_head = remote.head()
        arguments["run_dir"] = tmp_path / "fresh_vm_restore"

    def child(argv, environment):
        fit_calls.append(argv)
        _simulate_closed_child(run_dir, completed=False)
        return 0

    arguments["command_runner"] = child
    with pytest.raises(RuntimeError):
        helper.run_github_development(**arguments, max_chunks=2)
    assert len(fit_calls) == (1 if failure == "closed_publish" else 0)
    if failure == "foreign_active":
        assert remote.head() == protected_head
        assert _ledger(arguments["run_dir"]) == _ledger(run_dir)
