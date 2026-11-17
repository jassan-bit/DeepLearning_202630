"""Acknowledged, resumable GitHub storage for one exploratory moment campaign.

The completed v1/v2 sources remain sealed. The new, permanently bound registry
allocates at most 600 charged seconds against their cumulative 556.212659362
seconds and the inherited 1200-second ceiling. The predecessor is read as
hash-verified text only; no old neural checkpoint is downloaded or loaded.
Each fitted worker invocation follows a confirmed remote ACTIVE claim.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
from urllib import request, error
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import e3_github_storage as core

BRANCH, PREFIX = "e3-desarrollo-resultados-momentos", "runs/momentos_v1"
MANIFEST_PATH = PREFIX + "/snapshot_manifest.json"
PRIOR_COMMIT = "1ca029e869a517cb8de704e573c78ca1e0cbad3c"
PRIOR_SNAPSHOT_SHA256 = "5fe65596b99b334de197f3fc08811514c5ee6ea48b8447318057fea1b7cf9be1"
PRIOR = 556.2126593619992
V1_COMMIT = "f7826d120f3a6eb8e9a0952dd9842cd77f152f34"
V1_CHARGED = 369.4779596389991
V2_CHARGED = 186.73469972300006
PRIOR_RUN_ID = "6cc32b1a91224c4f9144a1c831394746"
PRIOR_BRANCH, PRIOR_PREFIX = "e3-desarrollo-resultados-v2", "runs/innovacion_v2"
PRIOR_NAMES = ("additive_positive_head", "innovation_mean_only", "innovation_density_mean")
REFERENCE_FILES = frozenset({"summary.json", "ledger.json", "global_registry.json", "state.json",
                             "manifest.json", "data_admission.json", "plan.json",
                             "additive_positive_head/validation_predictions.csv"})
ADDITIONAL, GLOBAL = 600.0, 1200.0
CAMPAIGN = "e3-moment-pilot-20261116"
VERSION = "e3-moment-github-snapshot-v1"
REGISTRY_VERSION = "e3-moment-development-budget-v1"
NAMES = ("moment_joint", "moment_mean_only", "direct_moment_aux")
ROOT_FILES = {"global_registry.json", "plan.json", "manifest.json", "data_admission.json", "ledger.json",
              "state.json", "runtime.json", "summary.json", "events.jsonl", "history.jsonl", "global_claim.json", "controls.json"}
MODEL_FILES = {"last.pt", "last.pt.sha256", "best.pt", "best.pt.sha256", "validation_predictions.csv", "curves.csv"}
IGNORED = {".run.lock", ".github_storage_receipt.json"}
OWNERS = {}
TERMINAL_STATUSES = {"completed", "budget_exhausted", "failed", "failed_numerical"}
StorageError, StorageConflict = core.StorageError, core.StorageConflict


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(core.canonical(value) + b"\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _lock(path):
    import fcntl
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise StorageConflict("moment_local_session_already_locked") from None
        yield


def _require(value, message):
    if not value:
        raise StorageError(message)


def _approved(path):
    if not isinstance(path, str) or "\\" in path or "\x00" in path:
        return False
    parts = Path(path).parts
    return path in ROOT_FILES or (len(parts) == 2 and parts[0] in NAMES and parts[1] in MODEL_FILES
                                 and "/".join(parts) == path) or path == "controls/validation_predictions.csv"


def validate_registry(registry):
    _require(registry.get("version") == REGISTRY_VERSION and registry.get("campaign_id") == CAMPAIGN,
             "moment_campaign_identity_mismatch")
    _require(registry.get("prior_commit") == PRIOR_COMMIT and registry.get("prior_snapshot_sha256") == PRIOR_SNAPSHOT_SHA256,
             "moment_prior_source_mismatch")
    _require(registry.get("prior_charged_seconds") == PRIOR and registry.get("additional_limit_seconds") == ADDITIONAL
             and registry.get("global_limit_seconds") == GLOBAL and PRIOR + ADDITIONAL <= GLOBAL,
             "moment_global_allocation_mismatch")
    for key in ("run_id", "claim_id"):
        _require(isinstance(registry.get(key), str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", registry[key]), "moment_claim_identity_invalid")
    for key in ("plan_sha256", "npz_sha256"):
        _require(isinstance(registry.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", registry[key]), "moment_input_identity_invalid")
    spent = registry.get("charged_seconds")
    _require(type(spent) in (int, float) and math.isfinite(spent) and spent >= 0, "moment_charge_invalid")
    _require(registry.get("phase") in ("initializing", "ready"), "moment_registry_phase_invalid")
    if registry.get("initial_ack_commit") is not None:
        _require(re.fullmatch(r"[0-9a-f]{40}", registry["initial_ack_commit"]), "moment_claim_ack_invalid")


def validate_snapshot(manifest, files):
    _require(manifest.get("version") == VERSION and set(files) == set(manifest.get("files", {})), "moment_snapshot_schema_invalid")
    _require({"global_registry.json", "plan.json", "ledger.json"} <= set(files), "moment_registry_snapshot_incomplete")
    _require(sum(len(x) for x in files.values()) <= core.MAX_SNAPSHOT_BYTES, "moment_snapshot_too_large")
    for path, payload in files.items():
        record = manifest["files"][path]
        _require(_approved(path) and isinstance(payload, bytes) and len(payload) <= core.MAX_FILE_BYTES
                 and record == {"bytes": len(payload), "sha256": core.sha256(payload)}, "moment_file_path_or_hash_invalid")
    registry, ledger = json.loads(files["global_registry.json"]), json.loads(files["ledger.json"])
    validate_registry(registry)
    _require(registry == manifest["registry"] and ledger["charged_seconds"] == registry["charged_seconds"]
             and ledger["limit_seconds"] == ADDITIONAL and ledger.get("active_session") == registry.get("active_session"),
             "moment_registry_ledger_mismatch")
    _require(core.sha256(core.canonical(json.loads(files["plan.json"]))) == registry["plan_sha256"], "moment_plan_hash_mismatch")
    identity = json.loads(files["manifest.json"]) if "manifest.json" in files else None
    if identity is not None:
        _require(identity["run_id"] == registry["run_id"] and identity["claim_id"] == registry["claim_id"]
                 and identity["plan_sha256"] == registry["plan_sha256"] and identity["npz_sha256"] == registry["npz_sha256"],
                 "moment_scientific_claim_or_input_mismatch")
        if identity.get("data_fingerprint") is not None:
            _require("data_admission.json" in files and core.sha256(core.canonical(json.loads(files["data_admission.json"])))
                     == identity["data_fingerprint"], "moment_admission_fingerprint_mismatch")
    _require(manifest.get("scientific_identity") == _scientific_identity(files)
             and manifest.get("progress") == _model_progress(files), "moment_progress_descriptor_mismatch")
    if registry["phase"] == "ready":
        required = {"manifest.json", "data_admission.json", "state.json", "runtime.json", "controls.json", "controls/validation_predictions.csv"}
        required |= {name + "/" + part for name in NAMES for part in ("last.pt", "last.pt.sha256")}
        _require(required <= set(files), "moment_scientific_snapshot_incomplete")
        identity = json.loads(files["manifest.json"])
        _require(identity["run_id"] == registry["run_id"] and identity["claim_id"] == registry["claim_id"]
                 and identity["plan_sha256"] == registry["plan_sha256"] and identity.get("initialized") is True,
                 "moment_scientific_identity_mismatch")
    for name in NAMES:
        for filename in ("last.pt", "best.pt"):
            path = name + "/" + filename
            if path in files:
                _require(path + ".sha256" in files and core.sha256(files[path]) in files[path + ".sha256"].decode().splitlines(),
                         "moment_checkpoint_receipt_invalid")
    if registry.get("active_session"):
        core._now(registry["active_session"]["last_wall_utc"])
        _require(manifest.get("lease_owner") and core._now(manifest["lease_until_utc"]) > core._now(manifest["lease_claimed_at_utc"]),
                 "moment_active_lease_invalid")
    return registry


def _scientific_identity(files):
    if "manifest.json" not in files:
        return None
    identity = json.loads(files["manifest.json"])
    return {key: identity.get(key) for key in ("run_id", "claim_id", "plan_sha256", "npz_sha256", "source_hashes", "data_fingerprint")}


def _model_progress(files):
    if "manifest.json" not in files:
        return None
    identity = json.loads(files["manifest.json"])
    state = json.loads(files.get("state.json", b'{"status":"initializing","models":{}}'))
    _require(state.get("status") in TERMINAL_STATUSES | {"initializing", "initialized", "paused", "running", "interrupted"},
             "moment_state_status_invalid")
    models = {}
    for name in NAMES:
        row = state.get("models", {}).get(name, {})
        epoch, complete = row.get("epoch", 0), row.get("complete", False)
        _require(type(epoch) is int and 0 <= epoch <= 40 and type(complete) is bool, "moment_model_progress_invalid")
        models[name] = {"epoch": epoch, "complete": complete}
    if state["status"] == "completed":
        _require(identity.get("initialized") is True
                 and all(row["complete"] and row["epoch"] >= 1 for row in models.values()),
                 "moment_completed_state_has_incomplete_recipes")
    return {"initialized": identity.get("initialized") is True, "state_status": state["status"], "models": models}


def _snapshot(registry, files, owner, now):
    registry = dict(registry)
    files = dict(files)
    files["global_registry.json"] = core.canonical(registry) + b"\n"
    manifest = {"version": VERSION, "registry": registry,
                "files": {p: {"bytes": len(b), "sha256": core.sha256(b)} for p, b in sorted(files.items())},
                "scientific_identity": _scientific_identity(files), "progress": _model_progress(files)}
    if registry.get("active_session"):
        stamp = core._now(now)
        remaining = max(0, min(ADDITIONAL - registry["charged_seconds"], GLOBAL - PRIOR - registry["charged_seconds"]))
        manifest.update(lease_owner=owner, lease_claimed_at_utc=stamp.isoformat(),
                        lease_until_utc=(stamp + timedelta(seconds=remaining + 60)).isoformat())
    validate_snapshot(manifest, files)
    return manifest, files


def _claimable(manifest, owner, now):
    if manifest["registry"].get("active_session") and manifest.get("lease_owner") != owner and core._now(now) < core._now(manifest["lease_until_utc"]):
        raise StorageConflict("moment_foreign_active_claim_inspection_only")


def _progress(previous, current):
    for key in ("campaign_id", "run_id", "claim_id", "prior_commit", "prior_snapshot_sha256", "prior_charged_seconds",
                "additional_limit_seconds", "global_limit_seconds", "plan_sha256", "npz_sha256"):
        _require(previous[key] == current[key], "moment_global_binding_cannot_change")
    _require(current["charged_seconds"] >= previous["charged_seconds"], "moment_global_charge_cannot_reset")
    if previous.get("initial_ack_commit"):
        _require(current.get("initial_ack_commit") == previous["initial_ack_commit"], "moment_initial_claim_cannot_change")
    _require(not (previous["phase"] == "ready" and current["phase"] != "ready"), "moment_ready_run_cannot_reinitialize")


def _scientific_progress(previous, current):
    old_identity, new_identity = previous.get("scientific_identity"), current.get("scientific_identity")
    if old_identity is None:
        return
    _require(new_identity is not None, "moment_scientific_identity_cannot_disappear")
    for key in old_identity:
        if old_identity[key] is not None:
            _require(new_identity[key] == old_identity[key], "moment_scientific_identity_cannot_change")
    old, new = previous["progress"], current["progress"]
    if old["state_status"] in TERMINAL_STATUSES:
        _require(current["registry"] == previous["registry"] and current["files"] == previous["files"]
                 and new == old, "moment_terminal_snapshot_is_immutable")
    _require(not old["initialized"] or new["initialized"], "moment_initialization_cannot_regress")
    _require(old["state_status"] not in TERMINAL_STATUSES or new["state_status"] == old["state_status"],
             "moment_terminal_run_cannot_reopen")
    for name in NAMES:
        a, b = old["models"][name], new["models"][name]
        _require(b["epoch"] >= a["epoch"] and (not a["complete"] or b["complete"]), "moment_model_progress_cannot_regress")
    if current["registry"]["charged_seconds"] == previous["registry"]["charged_seconds"]:
        _require(new["models"] == old["models"], "moment_model_progress_without_accounted_execution")
        for path, record in current["files"].items():
            if path.split("/")[0] in NAMES and previous["files"].get(path) != record:
                raise StorageConflict("moment_model_artifact_changed_without_accounted_progress")
    if old["initialized"]:
        for path in ("controls.json", "controls/validation_predictions.csv", "plan.json", "data_admission.json", "runtime.json"):
            _require(previous["files"].get(path) == current["files"].get(path),
                     "moment_fixed_scientific_input_cannot_change")


class MomentGitTransport(core.GitTransport):
    """Reuse v1 credential isolation and non-force Git, with a separate namespace."""
    def __init__(self, token, **kwargs):
        super().__init__(token, **kwargs)
        self.branch, self._ref = BRANCH, "refs/heads/" + BRANCH

    def read_snapshot(self, head):
        head = self._sha(head)
        manifest = json.loads(self._run(["show", head + ":" + MANIFEST_PATH]).stdout)
        files = {}
        for path in manifest.get("files", {}):
            _require(_approved(path), "moment_remote_path_refused")
            files[path] = self._run(["show", head + ":" + PREFIX + "/" + path]).stdout
        validate_snapshot(manifest, files)
        return manifest, files

    def publish(self, expected_head, manifest, files):
        validate_snapshot(manifest, files)
        if self.head() != expected_head:
            raise StorageConflict("moment_remote_head_changed")
        index = self._private / "moment-index"
        index.unlink(missing_ok=True)
        env = {"GIT_INDEX_FILE": str(index)}
        if expected_head:
            tree = self._run(["show", "-s", "--format=%T", self._sha(expected_head)]).stdout.decode().strip()
            self._run(["read-tree", tree], env=env)
        else:
            self._run(["read-tree", "--empty"], env=env)
        payloads = {PREFIX + "/" + path: data for path, data in files.items()}
        payloads[MANIFEST_PATH] = core.canonical(manifest) + b"\n"
        lines = []
        for path, data in sorted(payloads.items()):
            blob = self._sha(self._run(["hash-object", "-w", "--stdin"], data=data).stdout.decode().strip())
            lines.append(b"100644 " + blob.encode() + b"\t" + path.encode() + b"\0")
        self._run(["update-index", "-z", "--index-info"], data=b"".join(lines), env=env)
        tree = self._sha(self._run(["write-tree"], env=env).stdout.decode().strip())
        args = ["commit-tree", tree] + (["-p", self._sha(expected_head)] if expected_head else [])
        commit = self._sha(self._run(args, data=b"Save single moment campaign and cumulative development charge\n").stdout.decode().strip())
        result = self._run(["push", "--porcelain", "origin", commit + ":" + self._ref], allow_failure=True)
        if result.returncode:
            raise StorageError("moment_git_push_unconfirmed")
        return commit


class MomentStore:
    def __init__(self, transport, token=None):
        self.transport, self.token = transport, token

    def read(self):
        head = self.transport.head()
        if head is None:
            return None
        manifest, files = self.transport.read_snapshot(head)
        validate_snapshot(manifest, files)
        return head, manifest, files

    def publish(self, registry, files, expected_head, owner, now):
        manifest, files = _snapshot(registry, files, owner, now)
        if self.token:
            _require(all(self.token.encode() not in value for value in files.values()), "moment_credential_in_snapshot_refused")
        previous = self.read()
        if previous and previous[1] == manifest and previous[2] == files:
            return previous[0]
        if (previous[0] if previous else None) != expected_head:
            raise StorageConflict("moment_remote_head_changed")
        if previous:
            _progress(previous[1]["registry"], registry)
            _scientific_progress(previous[1], manifest)
            _claimable(previous[1], owner, now)
        try:
            return self.transport.publish(expected_head, manifest, files)
        except StorageError:
            confirmed = self.read()
            if confirmed and confirmed[1] == manifest and confirmed[2] == files:
                return confirmed[0]
            raise


class V2ReferenceTransport(core.GitTransport):
    """Read sealed references without Git-fetching any predecessor weight blob.

    Git ls-remote establishes live branch authority. Public pinned raw URLs
    provide only allowlisted text bytes and carry no credential. The canonical
    snapshot digest and each member digest establish their exact identity.
    """
    def __init__(self, token, **kwargs):
        super().__init__(token, **kwargs)
        self.branch, self._ref = PRIOR_BRANCH, "refs/heads/" + PRIOR_BRANCH
        self._reference_cache = None

    def head(self):
        refs = self._run(["ls-remote", "--exit-code", "origin", "refs/heads/" + core.BRANCH, self._ref], allow_failure=True)
        pairs = [line.split() for line in refs.stdout.decode().splitlines()]
        _require(refs.returncode == 0 and all(len(row) == 2 for row in pairs),
                 "authoritative_prior_heads_unavailable")
        observed = {row[1]: row[0] for row in pairs}
        _require(observed.get("refs/heads/" + core.BRANCH) == V1_COMMIT,
                 "authoritative_v1_source_advanced_or_unavailable")
        _require(observed.get(self._ref) == PRIOR_COMMIT,
                 "authoritative_v2_source_advanced_or_unavailable")
        return PRIOR_COMMIT

    def _read_pinned(self, path):
        _require(path in REFERENCE_FILES | {"snapshot_manifest.json"}, "moment_reference_path_not_allowlisted")
        url = "https://raw.githubusercontent.com/" + core.REPOSITORY + "/" + PRIOR_COMMIT + "/" + PRIOR_PREFIX + "/" + path
        query = request.Request(url, headers={"User-Agent": "E3 moment pilot text-only reference verifier"})
        try:
            with request.urlopen(query, timeout=45) as response:
                payload = response.read(core.MAX_FILE_BYTES + 1)
        except (OSError, error.URLError):
            raise StorageError("moment_public_reference_download_failed") from None
        _require(len(payload) <= core.MAX_FILE_BYTES, "moment_reference_text_too_large")
        return payload

    def read_snapshot(self, head):
        head = self._sha(head)
        _require(head == PRIOR_COMMIT, "moment_reference_commit_not_pinned")
        if self._reference_cache is None:
            raw = self._read_pinned("snapshot_manifest.json")
            _require(core.sha256(raw) == PRIOR_SNAPSHOT_SHA256, "moment_reference_snapshot_changed")
            files = {path: self._read_pinned(path) for path in sorted(REFERENCE_FILES)}
            self._reference_cache = (raw, files)
        raw, cached = self._reference_cache
        manifest, files = json.loads(raw), dict(cached)
        _validate_reference(manifest, files)
        return manifest, files

    def publish(self, *args, **kwargs):
        raise StorageError("moment_prior_transport_is_read_only")


def _validate_reference(manifest, files):
    _require(core.sha256(core.canonical(manifest) + b"\n") == PRIOR_SNAPSHOT_SHA256,
             "moment_reference_snapshot_changed")
    _require(set(files) == REFERENCE_FILES, "moment_reference_text_files_incomplete_or_unexpected")
    for path, payload in files.items():
        _require(isinstance(payload, bytes) and manifest["files"].get(path)
                 == {"bytes": len(payload), "sha256": core.sha256(payload)},
                 "moment_reference_text_digest_mismatch")
    registry = json.loads(files["global_registry.json"])
    ledger, summary, state = (json.loads(files[p]) for p in ("ledger.json", "summary.json", "state.json"))
    _require(manifest.get("version") == "e3-innovation-github-snapshot-v1" and manifest.get("registry") == registry,
             "moment_reference_registry_invalid")
    _require(registry.get("run_id") == PRIOR_RUN_ID and registry.get("v1_commit") == V1_COMMIT
             and registry.get("prior_charged_seconds") == V1_CHARGED
             and registry.get("charged_seconds") == V2_CHARGED and V1_CHARGED + V2_CHARGED == PRIOR
             and registry.get("active_session") is None and registry.get("phase") == "ready",
             "moment_reference_cumulative_budget_or_identity_changed")
    _require(ledger.get("charged_seconds") == V2_CHARGED and ledger.get("active_session") is None
             and summary.get("charged_seconds") == V2_CHARGED and summary.get("status") == "completed"
             and summary.get("run_id") == PRIOR_RUN_ID and state.get("status") == "completed"
             and manifest.get("progress", {}).get("state_status") == "completed",
             "moment_reference_not_terminal")
    for name in PRIOR_NAMES:
        _require(state.get("models", {}).get(name) == {"complete": True, "epoch": 40}
                 and summary.get("models", {}).get(name, {}).get("complete") is True
                 and summary["models"][name].get("completed_epochs") == 40,
                 "moment_reference_neural_recipe_incomplete")
    return registry


def _prior_terminal(transport):
    _require(transport.head() == PRIOR_COMMIT, "authoritative_v2_source_advanced_or_unavailable")
    manifest, files = transport.read_snapshot(PRIOR_COMMIT)
    _validate_reference(manifest, files)
    return manifest, files


def _restore_reference(manifest, files, destination, snapshot_manifest_path):
    """Materialize verified text only; existing files must match exactly."""
    _validate_reference(manifest, files)
    destination, snapshot_manifest_path = Path(destination), Path(snapshot_manifest_path)
    _require(not destination.is_symlink() and not snapshot_manifest_path.is_symlink(),
             "moment_reference_symlink_refused")
    destination.mkdir(parents=True, exist_ok=True)
    actual = {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()}
    _require(actual <= REFERENCE_FILES and not any(p.is_symlink() for p in destination.rglob("*")),
             "moment_reference_unexpected_local_file")
    for name, payload in files.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            _require(target.read_bytes() == payload, "moment_reference_existing_file_mismatch")
        else:
            target.write_bytes(payload)
    snapshot = core.canonical(manifest) + b"\n"
    if snapshot_manifest_path.exists():
        _require(snapshot_manifest_path.read_bytes() == snapshot, "moment_reference_existing_manifest_mismatch")
    else:
        snapshot_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_manifest_path.write_bytes(snapshot)


def _capture(run_dir, registry):
    files = {}
    for path in Path(run_dir).rglob("*"):
        relative = path.relative_to(run_dir).as_posix()
        _require(not path.is_symlink(), "moment_snapshot_symlink_refused")
        if path.is_dir():
            _require(relative in set(NAMES) | {"controls"}, "moment_snapshot_directory_refused")
            continue
        if relative in IGNORED or path.name.endswith(".tmp"):
            continue
        _require(_approved(relative), "moment_local_file_not_allowlisted")
        files[relative] = path.read_bytes()
    ledger = json.loads(files["ledger.json"])
    registry = {**registry, "charged_seconds": ledger["charged_seconds"], "active_session": ledger.get("active_session")}
    required = {"manifest.json", "data_admission.json", "state.json", "runtime.json", "controls.json", "controls/validation_predictions.csv"}
    required |= {n + "/" + x for n in NAMES for x in ("last.pt", "last.pt.sha256")}
    initialized = "manifest.json" in files and json.loads(files["manifest.json"]).get("initialized") is True
    registry["phase"] = "ready" if required <= set(files) and initialized else "initializing"
    return registry, files


def _restore(manifest, files, destination):
    validate_snapshot(manifest, files)
    destination = Path(destination)
    _require(not destination.exists() and not destination.is_symlink(), "moment_restore_destination_must_be_new")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".moment-restore-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "run"; staging.mkdir()
        for path, payload in files.items():
            target = staging / path; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(payload)
        os.rename(staging, destination)


def _child_environment(environment, token):
    _require(isinstance(environment, dict), "moment_explicit_child_environment_required")
    return {key: str(value) for key, value in environment.items()
            if not re.search(r"(?:^|_)(?:TOKEN|PAT|SECRET|PASSWORD)(?:_|$)", key.upper())
            and (not token or token not in str(value))}


def _run_child(argv, environment):
    process = subprocess.Popen(argv, env=environment, start_new_session=True)
    try:
        return process.wait()
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL); process.wait()
        except ProcessLookupError:
            pass
        raise


def _claim(registry, head):
    return {key: registry[key] for key in ("version", "campaign_id", "run_id", "claim_id", "prior_commit", "prior_snapshot_sha256", "prior_charged_seconds",
             "additional_limit_seconds", "global_limit_seconds", "plan_sha256", "npz_sha256")} | {"registry_commit": head}


def _publish_ack(store, registry, files, head, owner, now):
    commit = store.publish(registry, files, head, owner, now)
    print(json.dumps({"event": "moment_global_snapshot_acknowledged", "remote_commit": commit,
        "run_id": registry["run_id"], "prior_charged_seconds": PRIOR,
        "moment_charged_seconds": registry["charged_seconds"], "cumulative_charged_seconds": PRIOR + registry["charged_seconds"],
        "moment_limit_seconds": ADDITIONAL, "global_limit_seconds": GLOBAL}), flush=True)
    return commit


@contextmanager
def _context(token, run_dir, transport, prior_transport):
    run_dir = Path(run_dir)
    _require(not run_dir.is_symlink(), "moment_run_directory_symlink_refused")
    with _lock(run_dir.parent / ("." + run_dir.name + ".github_storage.lock")):
        actual = transport or MomentGitTransport(token)
        prior = None
        try:
            prior = prior_transport or V2ReferenceTransport(token)
            yield MomentStore(actual, token), prior
        finally:
            if transport is None: actual.close()
            if prior_transport is None and prior is not None: prior.close()


def _prepare(store, prior, *, token, npz_path, expected_sha256, run_dir, plan_path, code_root, python, environment,
             device, command_runner, now, reference_dir, snapshot_manifest_path):
    baseline, baseline_files = _prior_terminal(prior)
    plan = _read(plan_path); plan_sha = core.sha256(core.canonical(plan))
    remote = store.read()
    if remote is None:
        _require(not run_dir.exists(), "moment_local_state_without_remote_claim_refused")
        registry = {"version": REGISTRY_VERSION, "campaign_id": CAMPAIGN, "run_id": uuid.uuid4().hex,
            "claim_id": uuid.uuid4().hex, "prior_commit": PRIOR_COMMIT, "prior_snapshot_sha256": PRIOR_SNAPSHOT_SHA256,
            "prior_charged_seconds": PRIOR, "additional_limit_seconds": ADDITIONAL, "global_limit_seconds": GLOBAL,
            "plan_sha256": plan_sha, "npz_sha256": expected_sha256, "charged_seconds": 0.0,
            "phase": "initializing", "initial_ack_commit": None,
            "active_session": {"session_id": "moment-init-" + uuid.uuid4().hex, "last_wall_utc": core._now(now).isoformat()}}
        owner = OWNERS.setdefault(registry["run_id"], uuid.uuid4().hex)
        ledger = {"limit_seconds": ADDITIONAL, "charged_seconds": 0.0, "events": [], "active_session": registry["active_session"]}
        files = {"plan.json": core.canonical(plan) + b"\n", "ledger.json": core.canonical(ledger) + b"\n"}
        head = _publish_ack(store, registry, files, None, owner, now)
        registry["initial_ack_commit"] = head
    else:
        head, remote_manifest, remote_files = remote
        registry = dict(remote_manifest["registry"])
        _require(registry["plan_sha256"] == plan_sha and registry["npz_sha256"] == expected_sha256,
                 "moment_plan_or_data_cannot_change")
        owner = OWNERS.setdefault(registry["run_id"], uuid.uuid4().hex)
        registry["initial_ack_commit"] = registry.get("initial_ack_commit") or head
        if not run_dir.exists() and "manifest.json" in remote_files:
            _restore(remote_manifest, remote_files, run_dir)
        _claimable(remote_manifest, owner, now)
        if run_dir.exists():
            local_registry, files = _capture(run_dir, registry)
            _progress(registry, local_registry)
            registry = local_registry
            head = _publish_ack(store, registry, files, head, owner, now)
        else:
            files = dict(remote_files)
    controls = run_dir.parent / ("." + run_dir.name + ".control")
    if "state.json" in files and json.loads(files["state.json"]).get("status") in TERMINAL_STATUSES:
        _require(registry.get("active_session") is None, "moment_terminal_worker_left_active_claim")
        # A terminal failure can precede complete fitted initialization. Do not
        # mint another ACTIVE reservation merely because phase is initializing.
        return registry, head, owner, controls
    claim = _claim(registry, head)
    _write(controls / "global_claim.json", claim)
    if registry["phase"] == "initializing":
        # A retried fitted initialization needs a new acknowledged ACTIVE claim,
        # without refreshing an earlier uncharged ACTIVE recovery timestamp.
        ledger = json.loads(files["ledger.json"])
        if ledger.get("active_session") is None:
            ledger["active_session"] = {"session_id": "moment-init-" + uuid.uuid4().hex,
                                       "last_wall_utc": core._now(now).isoformat()}
        registry["active_session"] = ledger["active_session"]
        files["ledger.json"] = core.canonical(ledger) + b"\n"
        if remote is not None:
            head = _publish_ack(store, registry, files, head, owner, now)
        _write(controls / "global_claim.json", _claim(registry, head))
        if (run_dir / "manifest.json").is_file():
            # Resume reads the existing ledger directly. The same ACTIVE
            # receipt must exist locally only after its remote ACK succeeds.
            _write(run_dir / "ledger.json", ledger)
        reference_dir = Path(reference_dir) if reference_dir else controls / "reference_v2"
        snapshot_manifest_path = Path(snapshot_manifest_path) if snapshot_manifest_path else controls / "v2_snapshot_manifest.json"
        _restore_reference(baseline, baseline_files, reference_dir, snapshot_manifest_path)
        _write(controls / "initial_ledger.json", ledger)
        _prior_terminal(prior)  # The prior is still sealed immediately before any fitted initialization.
        command = "resume" if (run_dir / "manifest.json").is_file() else "preflight"
        argv = [str(python), str(Path(code_root) / "scripts/e3_run_moments.py"), command, "--npz", str(npz_path),
            "--expected-sha256", expected_sha256, "--run-dir", str(run_dir), "--device", device,
            "--global-claim", str(controls / "global_claim.json")]
        if command == "preflight":
            argv += ["--plan", str(plan_path), "--initial-ledger", str(controls / "initial_ledger.json"),
                     "--reference-dir", str(reference_dir), "--snapshot-manifest-path", str(snapshot_manifest_path)]
        else:
            argv += ["--stop-after-epochs", "1", "--reference-dir", str(reference_dir),
                     "--snapshot-manifest-path", str(snapshot_manifest_path)]
        failure = None
        try:
            code = command_runner(argv, environment)
        except BaseException as error:
            failure, code = error, 1
        if run_dir.is_dir() and (run_dir / "ledger.json").is_file():
            registry, files = _capture(run_dir, registry)
            head = _publish_ack(store, registry, files, head, owner, now)
        if failure is not None: raise failure
        exhausted = (run_dir / "state.json").is_file() and _read(run_dir / "state.json")["status"] == "budget_exhausted"
        _require(not code and (registry["phase"] == "ready" or exhausted) and registry.get("active_session") is None,
                 "moment_initialization_failed_claim_preserved")
    return registry, head, owner, controls


def prepare_github_moments(*, token, npz_path, expected_sha256, run_dir, plan_path, code_root, python=None,
                             environment=None, device="auto", transport=None, prior_transport=None, command_runner=None,
                             now=None, reference_dir=None, snapshot_manifest_path=None):
    _require(python is not None, "moment_explicit_isolated_python_required")
    environment = _child_environment(environment, token); run_dir = Path(run_dir)
    with _context(token, run_dir, transport, prior_transport) as (store, prior):
        registry, head, _, _ = _prepare(store, prior, token=token, npz_path=npz_path, expected_sha256=expected_sha256,
            run_dir=run_dir, plan_path=plan_path, code_root=code_root, python=python, environment=environment,
            device=device, command_runner=command_runner or _run_child, now=now,
            reference_dir=reference_dir, snapshot_manifest_path=snapshot_manifest_path)
        return {"remote_commit": head, "run_id": registry["run_id"], "ready": registry["phase"] == "ready",
                "prior_charged_seconds": PRIOR, "moment_charged_seconds": registry["charged_seconds"],
                "cumulative_charged_seconds": PRIOR + registry["charged_seconds"], "moment_limit_seconds": ADDITIONAL,
                "status": _read(Path(run_dir) / "state.json")["status"]}


def run_github_moments(*, token, npz_path, expected_sha256, run_dir, plan_path, code_root, python=None,
                         environment=None, device="auto", transport=None, prior_transport=None, command_runner=None,
                         now=None, reference_dir=None, snapshot_manifest_path=None, max_chunks=40, stop_after_epochs=12):
    _require(python is not None and type(max_chunks) is int and 1 <= max_chunks <= 40
             and type(stop_after_epochs) is int and 1 <= stop_after_epochs <= 120, "moment_execution_limits_invalid")
    environment = _child_environment(environment, token); run_dir = Path(run_dir); runner = command_runner or _run_child
    with _context(token, run_dir, transport, prior_transport) as (store, prior):
        registry, head, owner, controls = _prepare(store, prior, token=token, npz_path=npz_path, expected_sha256=expected_sha256,
            run_dir=run_dir, plan_path=plan_path, code_root=code_root, python=python, environment=environment,
            device=device, command_runner=runner, now=now, reference_dir=reference_dir, snapshot_manifest_path=snapshot_manifest_path)
        for _ in range(max_chunks):
            with _lock(run_dir / ".run.lock"):
                registry, files = _capture(run_dir, registry)
                state, ledger = _read(run_dir / "state.json"), _read(run_dir / "ledger.json")
                if state["status"] in TERMINAL_STATUSES or ledger["charged_seconds"] >= ADDITIONAL or PRIOR + ledger["charged_seconds"] >= GLOBAL:
                    return {"status": state["status"], "remote_commit": head, "cumulative_charged_seconds": PRIOR + ledger["charged_seconds"]}
                _prior_terminal(prior)
                current = store.read()
                _require(current is not None and current[0] == head, "moment_global_head_changed_before_fit")
                _claimable(current[1], owner, now)
                if ledger.get("active_session") is None:
                    ledger["active_session"] = {"session_id": "moment-" + uuid.uuid4().hex, "last_wall_utc": core._now(now).isoformat()}
                _write(run_dir / "ledger.json", ledger)
                registry, files = _capture(run_dir, registry)
                head = _publish_ack(store, registry, files, head, owner, now)
                _write(controls / "global_claim.json", _claim(registry, head))
            _prior_terminal(prior)
            command = "run" if state["status"] == "initialized" else "resume"
            argv = [str(python), str(Path(code_root) / "scripts/e3_run_moments.py"), command,
                "--npz", str(npz_path), "--expected-sha256", expected_sha256, "--run-dir", str(run_dir),
                "--device", device, "--global-claim", str(controls / "global_claim.json"),
                "--stop-after-epochs", str(stop_after_epochs),
                "--reference-dir", str(Path(reference_dir) if reference_dir else controls / "reference_v2"),
                "--snapshot-manifest-path", str(Path(snapshot_manifest_path) if snapshot_manifest_path
                                                 else controls / "v2_snapshot_manifest.json")]
            failure = None
            try:
                code = runner(argv, environment)
            except BaseException as error:
                failure, code = error, 1
            with _lock(run_dir / ".run.lock"):
                registry, files = _capture(run_dir, registry)
                head = _publish_ack(store, registry, files, head, owner, now)
            if failure is not None: raise failure
            _require(not code and registry.get("active_session") is None, "moment_worker_failed_snapshot_preserved")
        return {"status": _read(run_dir / "state.json")["status"], "remote_commit": head,
                "cumulative_charged_seconds": PRIOR + registry["charged_seconds"]}


def pull_github_moments(*, token, run_dir, transport=None, **_):
    actual = transport or MomentGitTransport(token)
    try:
        snapshot = MomentStore(actual, token).read()
        _require(snapshot is not None, "moment_snapshot_missing")
        _restore(snapshot[1], snapshot[2], Path(run_dir))
        return {"remote_commit": snapshot[0], "run_id": snapshot[1]["registry"]["run_id"]}
    finally:
        if transport is None: actual.close()
