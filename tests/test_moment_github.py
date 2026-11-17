"""Offline global-budget/CAS tests; synthetic bytes and workers, no market fit."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import e3_moment_github as storage

NOW = datetime(2026, 11, 16, 12, tzinfo=timezone.utc)
TOKEN = "ghp_SYNTHETIC_NOT_A_REAL_TOKEN_89405"



class MemoryRemote:
    """Immutable commits and independent non-forcing compare-and-swap."""
    def __init__(self):
        self.current = None
        self.commits = {}
        self.events = []
        self.fail = None
        self.fail_head = False

    def head(self):
        if self.fail_head:
            raise storage.StorageError("synthetic_authentication_failed")
        return self.current

    def read_snapshot(self, head):
        manifest, files = self.commits[head]
        return deepcopy(manifest), dict(files)

    def publish(self, expected_head, manifest, files):
        if self.fail == "before" or (self.fail == "closed" and not manifest["registry"].get("active_session")):
            self.fail = None
            raise storage.StorageError("synthetic_publication_failed")
        if expected_head != self.current:
            raise storage.StorageConflict("synthetic_cas_lost")
        if self.fail == "race":
            self.fail = None
            self.current = "f" * 40
            self.commits[self.current] = (deepcopy(manifest), dict(files))
            # The competing claim has a different canonical identity.
            other = self.commits[self.current][0]["registry"]
            other["run_id"] = "other-vm"
            self.commits[self.current][1]["global_registry.json"] = storage.core.canonical(other) + b"\n"
            record = self.commits[self.current][0]["files"]["global_registry.json"]
            payload = self.commits[self.current][1]["global_registry.json"]
            record.update(bytes=len(payload), sha256=storage.core.sha256(payload))
            raise storage.StorageConflict("synthetic_cas_lost")
        head = hashlib.sha1(str(len(self.commits) + 1).encode()).hexdigest()
        self.commits[head] = (deepcopy(manifest), dict(files))
        self.current = head
        self.events.append(("ack", bool(manifest["registry"].get("active_session")), head))
        if self.fail == "after":
            self.fail = None
            raise storage.StorageError("synthetic_acknowledgement_lost")
        return head


class PinnedPrior:
    """Entire predecessor is artificial: no market arrays, curves or predictions."""
    def __init__(self):
        self.current = storage.PRIOR_COMMIT
        registry = {"version": "e3-global-development-budget-v2", "run_id": storage.PRIOR_RUN_ID,
                    "v1_commit": storage.V1_COMMIT, "prior_charged_seconds": storage.V1_CHARGED,
                    "charged_seconds": storage.V2_CHARGED, "active_session": None, "phase": "ready"}
        models = {name: {"complete": True, "completed_epochs": 40} for name in storage.PRIOR_NAMES}
        rows = {
            "global_registry.json": registry,
            "ledger.json": {"charged_seconds": storage.V2_CHARGED, "active_session": None},
            "summary.json": {"charged_seconds": storage.V2_CHARGED, "run_id": storage.PRIOR_RUN_ID,
                             "status": "completed", "models": models},
            "state.json": {"status": "completed", "models": {name: {"complete": True, "epoch": 40}
                                                                    for name in storage.PRIOR_NAMES}},
            "manifest.json": {"initialized": True, "source_hashes": {"synthetic.py": "a" * 64}},
            "data_admission.json": {"source": "fully synthetic metadata"},
            "plan.json": {"scope": "synthetic terminal predecessor"},
        }
        self.files = {name: storage.core.canonical(row) + b"\n" for name, row in rows.items()}
        self.files["additive_positive_head/validation_predictions.csv"] = b"date,y_true,prediction\n2024-07-08,0.01,0.02\n"
        self.manifest = {"version": "e3-innovation-github-snapshot-v1", "registry": registry,
                         "progress": {"state_status": "completed"},
                         "files": {name: {"bytes": len(value), "sha256": storage.core.sha256(value)}
                                   for name, value in self.files.items()}}
        self.fail_head = False

    def head(self):
        if self.fail_head:
            raise storage.StorageError("synthetic_prior_authentication_failed")
        return self.current

    def read_snapshot(self, head):
        assert head == storage.PRIOR_COMMIT
        return deepcopy(self.manifest), dict(self.files)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(storage.core.canonical(value) + b"\n")


class SyntheticWorker:
    """Checks remote authority before simulating bounded state transitions."""
    def __init__(self, remote):
        self.remote = remote
        self.calls = []
        self.fail_partial = False
        self.advance_prior = None

    def __call__(self, argv, env):
        command = argv[2]
        options = dict(zip(argv[3::2], argv[4::2]))
        directory = Path(options["--run-dir"])
        claim = read(options["--global-claim"])
        head = self.remote.head()
        manifest, _ = self.remote.read_snapshot(head)
        registry = manifest["registry"]
        assert registry["active_session"] is not None, "fitting must follow acknowledged ACTIVE"
        assert claim["registry_commit"] == head, "each invocation must receive its current ACTIVE ACK"
        assert claim["run_id"] == registry["run_id"] and claim["claim_id"] == registry["claim_id"]
        assert not any(TOKEN in str(value) for value in env.values())
        assert not {"GITHUB_TOKEN", "GH_TOKEN", "E3_GITHUB_TOKEN", "OTHER_SECRET"} & env.keys()
        assert argv[0] == sys.executable
        self.calls.append((command, claim, deepcopy(registry)))
        self.remote.events.append(("worker", command, head))
        if command == "preflight":
            directory.mkdir()
            ledger = read(options["--initial-ledger"])
            assert ledger["active_session"] == registry["active_session"]
            write(directory / "plan.json", read(options["--plan"]))
            identity = {key: claim[key] for key in ("run_id", "claim_id", "plan_sha256", "npz_sha256")}
            identity.update(initialized=False, source_hashes={"synthetic.py": "b" * 64})
            write(directory / "manifest.json", identity)
            write(directory / "runtime.json", {"device_type": "cpu"})
            write(directory / "state.json", {"status": "initializing", "models": {}})
        else:
            ledger = read(directory / "ledger.json")
            assert ledger["charged_seconds"] == registry["charged_seconds"]
            assert ledger["active_session"] == registry["active_session"]
        ledger["charged_seconds"] += 2.0
        if self.fail_partial:
            mode = self.fail_partial
            self.fail_partial = False
            # Emulate death before fitted preparation completes, preserving ACTIVE.
            if mode == "closed":
                ledger["active_session"] = None
            write(directory / "ledger.json", ledger)
            raise RuntimeError("synthetic_preflight_process_died")
        identity = read(directory / "manifest.json")
        if not identity["initialized"]:
            admission = {"source": "synthetic-training-only", "fit_count": 4}
            identity.update(initialized=True, data_fingerprint=storage.core.sha256(storage.core.canonical(admission)))
            write(directory / "data_admission.json", admission)
            write(directory / "controls.json", {"scope": "synthetic"})
            (directory / "controls").mkdir(exist_ok=True)
            (directory / "controls/validation_predictions.csv").write_bytes(b"date,prediction\nsynthetic,0.1\n")
            state = {"status": "initialized", "models": {name: {"epoch": 0, "complete": False} for name in storage.NAMES}}
        else:
            state = read(directory / "state.json")
            for row in state["models"].values():
                row["epoch"] += 1
            state["status"] = "completed" if min(row["epoch"] for row in state["models"].values()) == 2 else "paused"
            for row in state["models"].values():
                row["complete"] = state["status"] == "completed"
        write(directory / "manifest.json", identity)
        write(directory / "state.json", state)
        for name in storage.NAMES:
            path = directory / name / "last.pt"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes((name + ":" + str(state["models"][name]["epoch"])).encode())
            path.with_name("last.pt.sha256").write_text(storage.core.sha256(path.read_bytes()) + "\n")
        ledger["active_session"] = None
        write(directory / "ledger.json", ledger)
        if self.advance_prior is not None:
            self.advance_prior.current = "a" * 40
        return 0


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    storage.OWNERS.clear()
    remote, prior = MemoryRemote(), PinnedPrior()
    monkeypatch.setattr(storage, "PRIOR_SNAPSHOT_SHA256", storage.core.sha256(storage.core.canonical(prior.manifest) + b"\n"))
    worker = SyntheticWorker(remote)
    data = tmp_path / "synthetic.npz"
    data.write_bytes(b"not market data; worker is synthetic")
    plan = tmp_path / "plan.json"
    write(plan, {"scope": "synthetic-storage-test", "budget": {"additional_seconds": 600}})
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    snapshot = tmp_path / "v2_snapshot_manifest.json"
    snapshot.write_bytes(storage.core.canonical(prior.manifest) + b"\n")
    args = dict(token=TOKEN, npz_path=data, expected_sha256=storage.core.sha256(data.read_bytes()),
                run_dir=tmp_path / "run", plan_path=plan, code_root=ROOT, python=sys.executable,
                environment={"PATH": os.environ.get("PATH", ""), "SAFE": "retained", "GITHUB_TOKEN": TOKEN,
                             "GH_TOKEN": TOKEN, "E3_GITHUB_TOKEN": TOKEN, "OTHER_SECRET": "secret", "VALUE": TOKEN,
                             "CUSTOM_URL": "https://example.invalid/" + TOKEN},
                device="cpu", transport=remote, prior_transport=prior, command_runner=worker, now=NOW,
                reference_dir=frozen, snapshot_manifest_path=snapshot)
    yield args, remote, prior, worker
    storage.OWNERS.clear()


def test_active_ack_covers_preparation_and_every_chunk_closed_ack_precedes_next(campaign):
    args, remote, _, worker = campaign
    result = storage.run_github_moments(**args, max_chunks=4)
    assert result["status"] == "completed"
    assert [call[0] for call in worker.calls] == ["preflight", "run", "resume"]
    assert [event[:2] for event in remote.events] == [
        ("ack", True), ("worker", "preflight"), ("ack", False),
        ("ack", True), ("worker", "run"), ("ack", False),
        ("ack", True), ("worker", "resume"), ("ack", False)]
    assert result["cumulative_charged_seconds"] == storage.PRIOR + 6
    saved = storage.MomentStore(remote).read()
    assert saved[1]["registry"]["additional_limit_seconds"] == 600
    assert saved[1]["registry"]["prior_charged_seconds"] == storage.PRIOR
    assert all(TOKEN.encode() not in value for value in saved[2].values())


def test_restore_keeps_exact_checkpoints_shared_charge_and_permanent_run_binding(campaign, tmp_path):
    args, remote, _, worker = campaign
    first = storage.prepare_github_moments(**args)
    head, manifest, files = storage.MomentStore(remote).read()
    destination = tmp_path / "another_vm"
    storage.pull_github_moments(token=TOKEN, run_dir=destination, transport=remote)
    assert all((destination / path).read_bytes() == payload for path, payload in files.items())
    storage.OWNERS.clear()
    second = storage.prepare_github_moments(**{**args, "run_dir": destination})
    assert first["run_id"] == second["run_id"]
    assert second["moment_charged_seconds"] == 2 and len(worker.calls) == 1
    registry, local = storage._capture(destination, manifest["registry"])
    store = storage.MomentStore(remote)
    with pytest.raises(storage.StorageError, match="binding_cannot_change|claim_or_input_mismatch"):
        store.publish({**registry, "run_id": "new-run-cannot-get-new-600"}, local, remote.head(), "owner", NOW)
    ledger = read(destination / "ledger.json")
    ledger["charged_seconds"] = 0
    write(destination / "ledger.json", ledger)
    registry, local = storage._capture(destination, manifest["registry"])
    with pytest.raises(storage.StorageError, match="charge_cannot_reset"):
        store.publish(registry, local, remote.head(), "owner", NOW)


@pytest.mark.parametrize("failure", ["auth", "before", "race", "prior-advanced", "prior-auth"])
def test_failed_authority_or_initial_active_ack_permits_zero_fitting(campaign, failure):
    args, remote, prior, worker = campaign
    if failure == "auth":
        remote.fail_head = True
    elif failure == "prior-advanced":
        prior.current = "a" * 40
    elif failure == "prior-auth":
        prior.fail_head = True
    else:
        remote.fail = failure
    with pytest.raises(storage.StorageError):
        storage.prepare_github_moments(**args)
    assert worker.calls == [] and not args["run_dir"].exists()


def test_unconfirmed_closed_publication_stops_before_next_chunk(campaign):
    args, remote, _, worker = campaign
    storage.prepare_github_moments(**args)
    remote.fail = "closed"
    with pytest.raises(storage.StorageError, match="publication_failed"):
        storage.run_github_moments(**args, max_chunks=3)
    assert [call[0] for call in worker.calls] == ["preflight", "run"]
    assert storage.MomentStore(remote).read()[1]["registry"]["active_session"] is not None


def test_partial_preflight_is_restored_but_foreign_active_cannot_fit_or_reset(campaign, tmp_path):
    args, remote, _, worker = campaign
    worker.fail_partial = True
    with pytest.raises(RuntimeError, match="process_died"):
        storage.prepare_github_moments(**args)
    old_head, manifest, files = storage.MomentStore(remote).read()
    old_active = manifest["registry"]["active_session"]
    assert manifest["registry"]["phase"] == "initializing"
    assert manifest["registry"]["charged_seconds"] == 2
    owners = dict(storage.OWNERS)
    storage.OWNERS.clear()
    destination = tmp_path / "foreign_vm"
    with pytest.raises(storage.StorageConflict, match="foreign_active"):
        storage.prepare_github_moments(**{**args, "run_dir": destination, "now": NOW + timedelta(seconds=1)})
    assert (destination / "ledger.json").read_bytes() == files["ledger.json"]
    assert remote.head() == old_head and len(worker.calls) == 1
    storage.OWNERS.clear(); storage.OWNERS.update(owners)
    result = storage.prepare_github_moments(**args)
    assert result["ready"] and result["run_id"] == manifest["registry"]["run_id"]
    assert result["moment_charged_seconds"] == 4
    assert worker.calls[-1][0] == "resume"
    assert worker.calls[-1][2]["active_session"] == old_active


def test_failed_initialization_with_closed_ledger_reopens_acknowledged_local_active(campaign):
    args, remote, _, worker = campaign
    worker.fail_partial = "closed"
    with pytest.raises(RuntimeError, match="process_died"):
        storage.prepare_github_moments(**args)
    assert read(args["run_dir"] / "ledger.json")["active_session"] is None
    assert storage.MomentStore(remote).read()[1]["registry"]["active_session"] is None
    result = storage.prepare_github_moments(**args)
    assert result["ready"] and result["moment_charged_seconds"] == 4
    assert worker.calls[-1][0] == "resume"
    assert worker.calls[-1][2]["active_session"] is not None
    assert read(args["run_dir"] / "ledger.json")["active_session"] is None


@pytest.mark.parametrize("mutation", ["source", "claim", "data", "epoch", "weights", "initialized"])
def test_publish_rejects_scientific_identity_and_checkpoint_rollback(campaign, mutation):
    args, remote, _, _ = campaign
    storage.run_github_moments(**args, max_chunks=1)
    head, manifest, files = storage.MomentStore(remote).read()
    registry = deepcopy(manifest["registry"])
    changed = dict(files)
    identity = json.loads(changed["manifest.json"])
    state = json.loads(changed["state.json"])
    if mutation == "source":
        identity["source_hashes"] = {"synthetic.py": "c" * 64}
    elif mutation == "claim":
        identity["claim_id"] = "different-claim"
    elif mutation == "data":
        identity["data_fingerprint"] = "e" * 64
    elif mutation == "epoch":
        state["models"][storage.NAMES[0]]["epoch"] = 0
    elif mutation == "initialized":
        identity["initialized"] = False
    else:
        path = storage.NAMES[0] + "/last.pt"
        changed[path] = b"old or arbitrary weights at unchanged expenditure"
        changed[path + ".sha256"] = (storage.core.sha256(changed[path]) + "\n").encode()
    changed["manifest.json"] = storage.core.canonical(identity) + b"\n"
    changed["state.json"] = storage.core.canonical(state) + b"\n"
    with pytest.raises(storage.StorageError):
        storage.MomentStore(remote).publish(registry, changed, head, "owner", NOW)
    assert remote.head() == head


def test_corruption_traversal_and_credentials_never_restore_or_publish(campaign, tmp_path):
    args, remote, _, _ = campaign
    storage.prepare_github_moments(**args)
    head, manifest, files = storage.MomentStore(remote).read()
    corrupt = dict(files)
    corrupt[storage.NAMES[0] + "/last.pt"] = b"corrupt"
    with pytest.raises(storage.StorageError):
        storage._restore(manifest, corrupt, tmp_path / "corrupt")
    assert not (tmp_path / "corrupt").exists()
    traversal = dict(files); traversal["../escaped"] = b"attack"
    with pytest.raises(storage.StorageError):
        storage._snapshot(manifest["registry"], traversal, "owner", NOW)
    secrets = dict(files); secrets["events.jsonl"] = TOKEN.encode()
    with pytest.raises(storage.StorageError, match="credential_in_snapshot"):
        storage.MomentStore(remote, TOKEN).publish(manifest["registry"], secrets, head, "owner", NOW)
    assert remote.head() == head and not (tmp_path / "escaped").exists()


def test_lost_ack_retry_is_idempotent_and_source_advance_blocks_next_fit(campaign):
    args, remote, prior, worker = campaign
    remote.fail = "after"
    storage.prepare_github_moments(**args)
    assert len(worker.calls) == 1
    worker.advance_prior = prior
    with pytest.raises(storage.StorageError, match="source_advanced"):
        storage.run_github_moments(**args, max_chunks=3)
    assert [call[0] for call in worker.calls] == ["preflight", "run"]
    assert storage.MomentStore(remote).read()[1]["registry"]["active_session"] is None


def test_exhausted_shared_allocation_cannot_restart_in_a_new_directory(campaign, tmp_path):
    args, remote, _, worker = campaign
    storage.prepare_github_moments(**args)
    ledger = read(args["run_dir"] / "ledger.json")
    ledger["charged_seconds"] = storage.ADDITIONAL
    write(args["run_dir"] / "ledger.json", ledger)
    write(args["run_dir"] / "state.json", {"status": "budget_exhausted", "models": {
        name: {"epoch": 0, "complete": False} for name in storage.NAMES}})
    storage.prepare_github_moments(**args)
    storage.OWNERS.clear()
    result = storage.run_github_moments(**{**args, "run_dir": tmp_path / "new_vm"})
    assert result["status"] == "budget_exhausted"
    assert result["cumulative_charged_seconds"] == storage.PRIOR + 600
    assert len(worker.calls) == 1
    assert read(tmp_path / "new_vm/ledger.json")["charged_seconds"] == 600


def test_missing_remote_and_local_competitor_never_initialize_a_second_run(campaign):
    args, remote, _, worker = campaign
    lock_path = args["run_dir"].parent / ("." + args["run_dir"].name + ".github_storage.lock")
    with storage._lock(lock_path):
        with pytest.raises(storage.StorageConflict, match="already_locked"):
            storage.prepare_github_moments(**args)
    assert len(worker.calls) == 0
    storage.prepare_github_moments(**args)
    remote.current = None
    with pytest.raises(storage.StorageError, match="local_state_without_remote"):
        storage.prepare_github_moments(**args)
    assert len(worker.calls) == 1


def test_real_local_git_transport_uses_only_moment_branch_and_prefix(campaign, tmp_path):
    args, memory, _, _ = campaign
    storage.prepare_github_moments(**args)
    _, manifest, files = storage.MomentStore(memory).read()
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)
    transport = storage.MomentGitTransport(TOKEN, _test_remote=bare, _allow_local_for_tests=True)
    try:
        head = transport.publish(None, manifest, files)
        restored_manifest, restored_files = transport.read_snapshot(head)
        assert restored_manifest == manifest and restored_files == files
        refs = subprocess.check_output(["git", "--git-dir", str(bare), "for-each-ref", "--format=%(refname)"]).decode().splitlines()
        assert refs == ["refs/heads/" + storage.BRANCH]
        paths = subprocess.check_output(["git", "--git-dir", str(bare), "ls-tree", "-r", "--name-only", head]).decode().splitlines()
        assert all(path.startswith(storage.PREFIX + "/") for path in paths)
        assert TOKEN not in transport._run(["config", "--list"]).stdout.decode()
    finally:
        transport.close()


def test_production_prior_allocation_and_source_constants_are_explicit():
    assert storage.PRIOR_COMMIT == "1ca029e869a517cb8de704e573c78ca1e0cbad3c"
    assert storage.PRIOR_SNAPSHOT_SHA256 == "5fe65596b99b334de197f3fc08811514c5ee6ea48b8447318057fea1b7cf9be1"
    assert storage.V1_CHARGED + storage.V2_CHARGED == storage.PRIOR
    assert storage.PRIOR + storage.ADDITIONAL < storage.GLOBAL
    assert storage.BRANCH == "e3-desarrollo-resultados-momentos"
    assert storage.PREFIX == "runs/momentos_v1"


@pytest.mark.parametrize("status", ["completed", "budget_exhausted", "failed", "failed_numerical"])
def test_terminal_state_cannot_reopen_or_create_another_run(campaign, tmp_path, status):
    args, remote, _, worker = campaign
    storage.prepare_github_moments(**args)
    state = read(args["run_dir"] / "state.json")
    state["status"] = status
    if status == "completed":
        for row in state["models"].values():
            row["complete"] = True
            row["epoch"] = 1
        ledger = read(args["run_dir"] / "ledger.json")
        ledger["charged_seconds"] += 1
        write(args["run_dir"] / "ledger.json", ledger)
    write(args["run_dir"] / "state.json", state)
    storage.prepare_github_moments(**args)
    original = storage.MomentStore(remote).read()
    storage.OWNERS.clear()
    result = storage.run_github_moments(**{**args, "run_dir": tmp_path / "another-vm"})
    assert result["status"] == status
    assert len(worker.calls) == 1
    registry = deepcopy(original[1]["registry"])
    changed = dict(original[2])
    state["status"] = "paused"
    changed["state.json"] = storage.core.canonical(state) + b"\n"
    with pytest.raises(storage.StorageError, match="terminal_snapshot_is_immutable|terminal_run_cannot_reopen"):
        storage.MomentStore(remote).publish(registry, changed, remote.head(), "owner", NOW)


@pytest.mark.parametrize("path", ["controls.json", "controls/validation_predictions.csv",
                                  "runtime.json", "moment_joint/validation_predictions.csv"])
def test_fixed_reference_and_uncharged_prediction_mutations_are_refused(campaign, path):
    args, remote, _, _ = campaign
    storage.prepare_github_moments(**args)
    head, manifest, files = storage.MomentStore(remote).read()
    files[path] = b"fabricated or unaccounted changed report"
    with pytest.raises(storage.StorageError):
        storage.MomentStore(remote).publish(manifest["registry"], files, head, "owner", NOW)
    assert remote.head() == head


def test_reference_subset_never_downloads_checkpoint_or_git_fetches_old_tree(campaign, monkeypatch):
    _, _, prior, _ = campaign
    transport = object.__new__(storage.V2ReferenceTransport)
    transport._reference_cache = None
    calls = []

    def read_pinned(path):
        calls.append(("raw", path))
        return storage.core.canonical(prior.manifest) + b"\n" if path == "snapshot_manifest.json" else prior.files[path]

    def run(args, **kwargs):
        calls.append(("git", args))
        output = (storage.V1_COMMIT + "\trefs/heads/" + storage.core.BRANCH + "\n"
                  + storage.PRIOR_COMMIT + "\trefs/heads/" + storage.PRIOR_BRANCH + "\n").encode()
        return subprocess.CompletedProcess(args, 0, output, b"")

    transport._run = run
    transport._read_pinned = read_pinned
    transport._ref = "refs/heads/" + storage.PRIOR_BRANCH
    assert transport.head() == storage.PRIOR_COMMIT
    first = transport.read_snapshot(storage.PRIOR_COMMIT)
    first[0]["registry"]["phase"] = "tampered caller copy"
    assert transport.read_snapshot(storage.PRIOR_COMMIT)[0]["registry"]["phase"] == "ready"
    raw_paths = [path for kind, path in calls if kind == "raw"]
    assert set(raw_paths) == storage.REFERENCE_FILES | {"snapshot_manifest.json"}
    assert len(raw_paths) == len(set(raw_paths))
    assert not any(path.endswith(".pt") for path in raw_paths)
    assert all(args[0] == "ls-remote" for kind, args in calls if kind == "git")
    with pytest.raises(storage.StorageError, match="read_only"):
        transport.publish(None, {}, {})


@pytest.mark.parametrize("source", ["v1", "v2", "empty"])
def test_live_predecessor_heads_fail_closed_without_fetch_or_worker(source):
    transport = object.__new__(storage.V2ReferenceTransport)
    transport._ref = "refs/heads/" + storage.PRIOR_BRANCH
    lines = (storage.V1_COMMIT + "\trefs/heads/" + storage.core.BRANCH + "\n"
             + storage.PRIOR_COMMIT + "\trefs/heads/" + storage.PRIOR_BRANCH + "\n")
    if source == "empty":
        lines = ""
    else:
        lines = lines.replace(storage.V1_COMMIT if source == "v1" else storage.PRIOR_COMMIT, "f" * 40)
    transport._run = lambda args, **kwargs: subprocess.CompletedProcess(args, 0, lines.encode(), b"")
    with pytest.raises(storage.StorageError, match="authoritative"):
        transport.head()


@pytest.mark.parametrize("mutation", ["checkpoint", "missing", "corrupt", "active", "budget"])
def test_predecessor_reference_must_be_exact_terminal_text_and_cost(campaign, mutation):
    args, remote, prior, worker = campaign
    if mutation == "checkpoint":
        prior.files["additive_positive_head/best.pt"] = b"must never download or restore"
    elif mutation == "missing":
        prior.files.pop("additive_positive_head/validation_predictions.csv")
    elif mutation == "corrupt":
        prior.files["additive_positive_head/validation_predictions.csv"] += b"altered"
    elif mutation == "active":
        prior.manifest["registry"]["active_session"] = {"session_id": "not-sealed"}
    else:
        prior.manifest["registry"]["charged_seconds"] += 1
    with pytest.raises(storage.StorageError):
        storage.prepare_github_moments(**args)
    assert worker.calls == [] and remote.head() is None


def test_reference_existing_local_tampering_and_symlink_are_refused(campaign, tmp_path):
    _, _, prior, _ = campaign
    target = tmp_path / "reference"
    receipt = tmp_path / "snapshot.json"
    storage._restore_reference(prior.manifest, prior.files, target, receipt)
    (target / "summary.json").write_bytes(b"local mismatch")
    with pytest.raises(storage.StorageError, match="existing_file_mismatch"):
        storage._restore_reference(prior.manifest, prior.files, target, receipt)
    link = tmp_path / "linked-reference"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(storage.StorageError, match="symlink"):
        storage._restore_reference(prior.manifest, prior.files, link, receipt)


def test_failed_initialization_is_terminal_without_minting_another_active_claim(campaign, tmp_path):
    args, remote, _, worker = campaign
    worker.fail_partial = "closed"
    with pytest.raises(RuntimeError):
        storage.prepare_github_moments(**args)
    write(args["run_dir"] / "state.json", {"status": "failed", "models": {}})
    current = storage.MomentStore(remote).read()
    registry, files = storage._capture(args["run_dir"], current[1]["registry"])
    storage.MomentStore(remote).publish(registry, files, remote.head(), "owner", NOW)
    calls, active_acknowledgements = len(worker.calls), sum(event[:2] == ("ack", True) for event in remote.events)
    storage.OWNERS.clear()
    result = storage.run_github_moments(**{**args, "run_dir": tmp_path / "new-vm"})
    assert result["status"] == "failed" and len(worker.calls) == calls
    assert sum(event[:2] == ("ack", True) for event in remote.events) == active_acknowledgements
    assert storage.MomentStore(remote).read()[1]["registry"]["active_session"] is None


@pytest.mark.parametrize("mutation", ["summary", "history", "epochs", "cost", "checkpoint"])
def test_completed_snapshot_cannot_be_rewritten_even_with_higher_cost(campaign, mutation):
    args, remote, _, _ = campaign
    storage.run_github_moments(**args, max_chunks=4)
    head, manifest, files = storage.MomentStore(remote).read()
    registry = deepcopy(manifest["registry"])
    if mutation == "summary":
        files["summary.json"] = b'{"winner":"fabricated-after-completion"}\n'
    elif mutation == "history":
        files["history.jsonl"] = b'{"replacement":"after-completion"}\n'
    elif mutation == "epochs":
        state = json.loads(files["state.json"])
        state["models"][storage.NAMES[0]]["epoch"] += 1
        files["state.json"] = storage.core.canonical(state) + b"\n"
        registry["charged_seconds"] += 1
    elif mutation == "checkpoint":
        path = storage.NAMES[0] + "/last.pt"
        files[path] = b"post-terminal newly fitted or replaced weights"
        files[path + ".sha256"] = (storage.core.sha256(files[path]) + "\n").encode()
        registry["charged_seconds"] += 1
    else:
        registry["charged_seconds"] += 1
    ledger = json.loads(files["ledger.json"])
    ledger["charged_seconds"] = registry["charged_seconds"]
    files["ledger.json"] = storage.core.canonical(ledger) + b"\n"
    with pytest.raises(storage.StorageError, match="terminal_snapshot_is_immutable"):
        storage.MomentStore(remote).publish(registry, files, head, "owner", NOW)
    assert remote.head() == head


def test_recipe_epoch_cannot_advance_without_accounted_execution(campaign):
    args, remote, _, _ = campaign
    storage.prepare_github_moments(**args)
    head, manifest, files = storage.MomentStore(remote).read()
    state = json.loads(files["state.json"])
    state["models"][storage.NAMES[0]]["epoch"] += 1
    files["state.json"] = storage.core.canonical(state) + b"\n"
    with pytest.raises(storage.StorageError, match="progress_without_accounted"):
        storage.MomentStore(remote).publish(manifest["registry"], files, head, "owner", NOW)


def test_public_reference_request_has_no_authentication_or_unapproved_path(monkeypatch):
    transport = object.__new__(storage.V2ReferenceTransport)
    calls = []

    def open_public(query, timeout):
        calls.append(query)
        assert query.full_url == ("https://raw.githubusercontent.com/" + storage.core.REPOSITORY
            + "/" + storage.PRIOR_COMMIT + "/" + storage.PRIOR_PREFIX + "/summary.json")
        assert TOKEN not in query.full_url
        assert not {"authorization", "cookie"} & {key.lower() for key in query.headers}
        return io.BytesIO(b"synthetic text")

    monkeypatch.setattr(storage.request, "urlopen", open_public)
    assert transport._read_pinned("summary.json") == b"synthetic text"
    for path in ("../summary.json", "additive_positive_head/best.pt", "summary.json?token=" + TOKEN):
        with pytest.raises(storage.StorageError, match="path_not_allowlisted"):
            transport._read_pinned(path)
    assert len(calls) == 1


def test_real_credential_free_worker_end_to_end_with_synthetic_reference_and_remote(tmp_path, monkeypatch):
    """Exercise CLI + real tiny CPU fit, using no network, market or old weights."""
    import test_moment_training as synthetic

    f = synthetic.fixture.__wrapped__(tmp_path)
    reference = f["reference"]
    registry = read(reference / "global_registry.json")
    registry.update(version="e3-global-development-budget-v2", run_id=storage.PRIOR_RUN_ID,
                    v1_commit=storage.V1_COMMIT, phase="ready")
    write(reference / "global_registry.json", registry)
    summary = read(reference / "summary.json")
    summary.update(run_id=storage.PRIOR_RUN_ID, charged_seconds=storage.V2_CHARGED,
                   models={name: {"complete": True, "completed_epochs": 40} for name in storage.PRIOR_NAMES})
    write(reference / "summary.json", summary)
    state = {"status": "completed", "models": {name: {"complete": True, "epoch": 40}
                                               for name in storage.PRIOR_NAMES}}
    write(reference / "state.json", state)
    files = {name: (reference / name).read_bytes() for name in storage.REFERENCE_FILES}
    manifest = {"version": "e3-innovation-github-snapshot-v1", "registry": registry,
                "progress": {"state_status": "completed", "models": state["models"]},
                "scientific_identity": {"data_fingerprint": f["prior_manifest"]["data_fingerprint"]},
                "files": {name: {"bytes": len(value), "sha256": storage.core.sha256(value)}
                          for name, value in files.items()}}
    reference_sha = storage.core.sha256(storage.core.canonical(manifest) + b"\n")
    monkeypatch.setattr(storage, "PRIOR_SNAPSHOT_SHA256", reference_sha)
    plan = f["plan"]
    plan["frozen_control"]["snapshot_manifest_sha256"] = reference_sha
    plan_path = tmp_path / "plan-for-coordinator.json"
    write(plan_path, plan)
    prior = object.__new__(PinnedPrior)
    prior.current, prior.manifest, prior.files, prior.fail_head = storage.PRIOR_COMMIT, manifest, files, False
    remote = MemoryRemote()
    invocations = []

    def worker(argv, environment):
        assert not any(TOKEN in value for value in environment.values())
        options = dict(zip(argv[3::2], argv[4::2]))
        assert read(options["--global-claim"])["registry_commit"] == remote.head()
        invocations.append(argv[2])
        return storage._run_child(argv, environment)

    args = dict(token=TOKEN, npz_path=f["npz"], expected_sha256=f["sha"],
                run_dir=tmp_path / "coordinated-run", plan_path=plan_path, code_root=ROOT,
                python=sys.executable, environment={"PATH": os.environ.get("PATH", ""),
                    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "GITHUB_TOKEN": TOKEN},
                device="cpu", transport=remote, prior_transport=prior,
                command_runner=worker, stop_after_epochs=3, max_chunks=4)
    try:
        result = storage.run_github_moments(**args)
        assert result["status"] == "completed"
        assert invocations == ["preflight", "run", "resume"]
        head, saved, payloads = storage.MomentStore(remote).read()
        assert saved["progress"]["state_status"] == "completed"
        assert all(row == {"complete": True, "epoch": 2} for row in saved["progress"]["models"].values())
        assert saved["registry"]["active_session"] is None
        assert storage.PRIOR < result["cumulative_charged_seconds"] <= storage.PRIOR + storage.ADDITIONAL
        assert all(TOKEN.encode() not in value for value in payloads.values())
        before = list(invocations)
        assert storage.run_github_moments(**args)["remote_commit"] == head
        assert invocations == before
    finally:
        storage.OWNERS.clear()
