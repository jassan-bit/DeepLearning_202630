"""One audited reopening of the first E2 worker's pre-step metadata failure.

The pure builder performs no model work or publication. The launcher accepts
only a remotely published authorization and preserves the failed local folder.
"""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import uuid

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_e2_json_repair_bootstrap_" + uuid.uuid4().hex, HERE / "bootstrap.py")
b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b)

FAILED_REQUEST = f"jobs/{b.FAILED_JOB_ID}/fit/request.json"
FAILED_LOG = "logs/eec1cdd8aaaa4b6c9a9579606bbbc8c9.log"
FAILED_REQUEST_SHA256 = "4956d6687a0e380bc10844cac142bd882e0ca8712d87774f2203143de3651bbf"
FAILED_LOG_SHA256 = "164dfb07f59a268fec64c19cfd2eb4b0b9b23d002693226505d743ef95e01a6f"
FIXED_WORKER_SHA256 = "6d4f7067818c8760646dad7390d8a29e13b54c72b3c9cb7c58ff80596d030a5e"
ORIGINAL_FILES = {FAILED_REQUEST, FAILED_LOG, "plan.json", "run_manifest.json", "runtime.json", "state.json", "ledger.json", "ledger.jsonl"}


def _bytes(payloads):
    return {p: v.encode() if isinstance(v, str) else v for p, v in payloads.items()}


def validate_failed(payloads, head):
    """Reject every progressed, divergent or differently failed allocation."""
    b.require(head == b.FAILED_HEAD, "Repair requires the exact reviewed failed remote head")
    payloads = _bytes(payloads)
    snapshot = payloads.pop("snapshot.json", None)
    b.require(set(payloads) == ORIGINAL_FILES and snapshot is not None
        and b.sha(snapshot) == b.FAILED_SNAPSHOT_SHA256, "Repair requires the exact request-only failed snapshot")
    b.validate_snapshot(json.loads(snapshot), payloads)
    manifest = json.loads(payloads["run_manifest.json"])
    b.require(manifest["run_id"] == b.FAILED_RUN_ID
        and b.sha(payloads["run_manifest.json"]) == b.FAILED_MANIFEST_SHA256
        and b.sha(payloads["ledger.jsonl"]) == b.FAILED_LEDGER_SHA256
        and b.sha(payloads[FAILED_REQUEST]) == FAILED_REQUEST_SHA256
        and b.sha(payloads[FAILED_LOG]) == FAILED_LOG_SHA256, "Reviewed failed evidence hash differs")
    ledger, state = json.loads(payloads["ledger.json"]), json.loads(payloads["state.json"])
    b.require(ledger["charged_seconds"] == b.FAILED_CHARGED_SECONDS and ledger["active_reservation"] is None
        and state["status"] == "failed" and not state["jobs"], "Failed allocation already progressed or changed cost")
    return payloads, manifest, snapshot


def build_repair_payloads(failed_payloads, new_code_hashes, *, expected_head=b.FAILED_HEAD,
                          source_commit=None, now=None):
    """Pure, reviewable migration; the old ledger remains a byte-exact prefix."""
    old, old_manifest, original_snapshot = validate_failed(failed_payloads, expected_head)
    b.require(set(new_code_hashes) == set(old_manifest["code_hashes"]) | {"repair.py"}
        and all(re.fullmatch(r"[a-f0-9]{64}", value) for value in new_code_hashes.values()), "Repair code binding is incomplete")
    for name in ("plan.json", "prepare.py", "aggregate.py"):
        b.require(new_code_hashes[name] == old_manifest["code_hashes"][name], "Repair cannot change plan, preparation or aggregation")
    b.require(new_code_hashes["worker.py"] == FIXED_WORKER_SHA256
        and new_code_hashes["bootstrap.py"] == b.sha((HERE / "bootstrap.py").read_bytes())
        and new_code_hashes["repair.py"] == b.sha(Path(__file__).read_bytes()), "Repair adapter code hash differs")
    if source_commit is not None:
        b.require(bool(re.fullmatch(r"[a-f0-9]{40}", source_commit)), "Invalid repair source commit")
    stamp = now or datetime.now(timezone.utc).isoformat()
    authorization = {"version": "e2-first-step-json-repair-v1", "repair_id": b.REPAIR_ID,
        "scope": "TorchVersion converted to plain str before original finite-JSON metadata admission; no scientific recipe change",
        "failed_head": b.FAILED_HEAD, "run_id": b.FAILED_RUN_ID,
        "plan_fingerprint": old_manifest["plan_fingerprint"], "old_manifest_sha256": b.FAILED_MANIFEST_SHA256,
        "original_snapshot_sha256": b.FAILED_SNAPSHOT_SHA256, "original_ledger_sha256": b.FAILED_LEDGER_SHA256,
        "failed_request_sha256": FAILED_REQUEST_SHA256, "failed_log_sha256": FAILED_LOG_SHA256,
        "charged_seconds_preserved": b.FAILED_CHARGED_SECONDS, "limit_seconds_preserved": b.LIMIT,
        "evidence": "request only; no phase identity, prepared data, events, optimizer or checkpoint",
        "old_code_hashes": old_manifest["code_hashes"], "new_code_hashes": dict(new_code_hashes),
        "source_commit": source_commit, "created_at_utc": stamp}
    authorization_bytes = b.canonical(authorization) + b"\n"
    manifest = {**old_manifest, "code_hashes": dict(new_code_hashes),
        "repair_authorization_sha256": b.sha(authorization_bytes)}
    original = {**{p: v.decode() for p, v in old.items()}, "snapshot.json": original_snapshot.decode()}
    result = {p: v for p, v in old.items() if p not in {FAILED_REQUEST, FAILED_LOG}}
    result.update({b.REPAIR_ROOT + "/authorization.json": authorization_bytes,
        b.REPAIR_ROOT + "/failed_request.json": old[FAILED_REQUEST],
        b.REPAIR_ROOT + "/failed_worker.log": old[FAILED_LOG],
        b.REPAIR_ROOT + "/original_run_manifest.json": old["run_manifest.json"],
        b.REPAIR_ROOT + "/original_snapshot.json": b.canonical(original) + b"\n",
        "run_manifest.json": b.canonical(manifest) + b"\n"})
    events = b.events_from(old["ledger.jsonl"])
    body = {"version": "e2-extension-event-v1", "event": "reopen", "sequence": 2,
        "previous_sha256": events[-1]["sha256"], "run_id": b.FAILED_RUN_ID,
        "plan_fingerprint": manifest["plan_fingerprint"], "job_id": b.FAILED_JOB_ID, "phase": "fit",
        "repair_id": b.REPAIR_ID, "failed_head": b.FAILED_HEAD, "authorization_sha256": b.sha(authorization_bytes),
        "charged_seconds_preserved": b.FAILED_CHARGED_SECONDS, "utc": stamp}
    event = {**body, "sha256": b.sha(b.canonical(body))}
    events.append(event)
    result["ledger.jsonl"] = old["ledger.jsonl"] + b.canonical(event) + b"\n"
    result["ledger.json"] = b.canonical(b.ledger_summary(events, manifest)) + b"\n"
    result["state.json"] = b.canonical(b.state_projection(events, manifest)) + b"\n"
    b.validate_snapshot(b.make_snapshot(result), result)
    return result


def _verified_job_directory(path):
    """The original store may have left only an empty phase lock/directory."""
    allowed_directories = {"fit", "fit/phases", "fit/phases/supervised"}
    for item in path.rglob("*"):
        relative = item.relative_to(path).as_posix()
        b.require(not item.is_symlink(), "Symlink in failed local job refused")
        if item.is_dir():
            b.require(relative in allowed_directories, "Local failed job has an unreviewed phase directory")
        elif relative == "fit/request.json":
            b.require(b.sha(item.read_bytes()) == FAILED_REQUEST_SHA256, "Local failed request changed")
        else:
            b.require(relative == "fit/phases/supervised/.writer.lock" and item.is_file() and item.stat().st_size == 0,
                "Local job has phase, optimizer, checkpoint or other unreviewed artifacts")
    b.require((path / "fit/request.json").is_file(), "Original failed local request is missing")


def install_remote_repair(workspace, remote):
    """Reconcile an exact old local copy or a torn metadata migration, no fit."""
    workspace, (head, snapshot, payloads) = Path(workspace), remote
    b.validate_snapshot(snapshot, payloads)
    manifest = json.loads(payloads["run_manifest.json"])
    b.require(manifest.get("repair_authorization_sha256") is not None and len(b.events_from(payloads["ledger.jsonl"])) >= 3,
        "Reviewed repair must already be acknowledged remotely")
    directory = workspace / "run"
    original = {p: v.encode() for p, v in json.loads(payloads[b.REPAIR_ROOT + "/original_snapshot.json"]).items() if p != "snapshot.json"}
    if directory.exists():
        local = b.collect(directory)
        local_manifest = json.loads(local["run_manifest.json"])
        local_events = b.events_from(local["ledger.jsonl"])
        remote_events = b.events_from(payloads["ledger.jsonl"])
        if (local_manifest == manifest and len(local_events) >= 3
                and local_events[2]["sha256"] == remote_events[2]["sha256"]):
            # Ordinary resume owns valid local work beyond the migration. In
            # particular, never replace a completed child after a lost Git ACK.
            b.require(local["ledger.jsonl"].startswith(payloads["ledger.jsonl"])
                or payloads["ledger.jsonl"].startswith(local["ledger.jsonl"]), "Repaired local ledger diverged")
            for name in b.REPAIR_FILES:
                path = b.REPAIR_ROOT + "/" + name
                b.require(local.get(path) == payloads[path], "Local repair evidence differs")
            return head
        if local != payloads:
            # Before moving any local folder, every file must be an exact old or
            # exact new metadata byte string. Unknown progress is never erased.
            b.require(set(local) <= set(original) | set(payloads), "Unreviewed local files prevent repair reconciliation")
            for name, value in local.items():
                b.require(value == original.get(name) or value == payloads.get(name), "Divergent local progress preserved")
            b.require(local.get("ledger.jsonl") in {original["ledger.jsonl"], payloads["ledger.jsonl"]}, "Local ledger cannot be reconciled")
            jobs_folder = directory / "jobs"
            b.require(not jobs_folder.exists() or all(p.name == b.FAILED_JOB_ID for p in jobs_folder.iterdir()),
                "Other local jobs prevent request-only failure repair")
            job = jobs_folder / b.FAILED_JOB_ID
            archived = workspace / "failed_preflight_copies" / (b.FAILED_JOB_ID + "-" + FAILED_REQUEST_SHA256[:12])
            if job.exists():
                _verified_job_directory(job)
                b.require(not archived.exists(), "A prior failed-folder copy already exists; both preserved")
                archived.parent.mkdir(parents=True, exist_ok=True)
                os.rename(job, archived)
            elif archived.exists():
                _verified_job_directory(archived)
            else:
                b.require(FAILED_REQUEST not in local, "Failed job folder unexpectedly disappeared")
            old_log = directory / FAILED_LOG
            if old_log.exists():
                b.require(b.sha(old_log.read_bytes()) == FAILED_LOG_SHA256, "Failed worker log changed")
                old_log.unlink()  # exact bytes already present in the remote archive
    b.restore(directory, payloads)
    b.require(b.collect(directory) == payloads, "Repaired local installation differs from remote bytes")
    b.write_json(workspace / "ack.json", {"head": head, "run_id": manifest["run_id"],
        "plan_fingerprint": manifest["plan_fingerprint"], "snapshot_sha256": b.sha(b.canonical(snapshot)),
        "ledger_sha256": b.sha(payloads["ledger.jsonl"])})
    return head


def run(workspace=None):
    """Use the published repair, then continue the same reserved 52-job run."""
    plan = b.read_json(HERE / "plan.json")
    workspace = Path(workspace or plan["storage"]["workspace"])
    secret = b.token()
    core_path = b.download({"commit": b.CORE_COMMIT, "path": "scripts/e3_github_storage.py", "sha256": b.CORE_SHA},
        workspace / "private/e3_github_storage.py")
    core = b.load_private("e2_repair_git_core", core_path)
    client = b.NativeGit(core, secret)
    try:
        os.environ.pop("GITHUB_TOKEN", None)
        os.environ.pop("GH_TOKEN", None)
        with b.coordinator_lock(workspace):
            remote = client.read()
            b.require(remote is not None, "Existing extension allocation is required")
            install_remote_repair(workspace, remote)
    finally:
        client.close()
    original_reader = b.token
    b.token = lambda: secret
    try:
        return b.run(workspace)
    finally:
        b.token = original_reader
        secret = ""
