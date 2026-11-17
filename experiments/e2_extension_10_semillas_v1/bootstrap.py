"""One sequential, remotely reserved E2 extension; scientific work stays in children."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from getpass import getpass
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import time
from urllib.request import Request, urlopen
import uuid

HERE = Path(__file__).resolve().parent
REPO = "jassan-bit/DeepLearning_202630"
BRANCH = "e2-extension-10-semillas-v1"
PREFIX = "runs/e2_extension_10_semillas_v1"
INITIAL_PARENT = "510c3bddfdc9926322e3210ac61ee1f5eeffad2e"
CORE_COMMIT = "cf891213f1acb6967aef5e0857fa0ac63768f836"
CORE_SHA = "1de75f622c35e0a9375e96197a620663411243c87c93b1bc2e23d45032f0d632"
LIMIT = 7200.0
HISTORIC_E3_SECONDS = 948.439625042
SEEDS = (271807, 271808, 271809, 271810)
TERMINAL = {"completed", "failed", "budget_exhausted", "unreconciled"}
MAX_BLOB = 90 * 1024**2
MAX_SNAPSHOT = 2 * 1024**3
BASE_FILES = {"plan.json", "run_manifest.json", "ledger.jsonl", "ledger.json", "state.json",
              "all_fits_remote_ack.json", "summary.json", "runtime.json", "aggregate_receipt.json", "combined_predictors.json", "per_seed_metrics.csv", "comparison.csv"}
ARTIFACT_NAMES = {"checkpoint.pt", "scaler.json", "calibrator.json", "fit_receipt.json", "evaluation.json",
                  "history.json", "history.csv", "predictions.csv", "metrics.json", "fit_remote_ack.json"}
AGGREGATE_NAMES = {"summary.json", "per_seed_metrics.csv", "family_summary.csv", "predictions.csv",
                   "aggregate_receipt.json", "provenance.json", "comparison.json"}

# The only permitted terminal-state reopening is the audited first-job metadata
# failure, before any phase or optimizer artifact existed. Later failures halt.
REPAIR_ID = "finite_json_before_first_step_v1"
REPAIR_ROOT = "repairs/" + REPAIR_ID
FAILED_RUN_ID = "e2ext10-20261117-4f4fa4687ce5"
FAILED_HEAD = "f4a16e33291e6e175c8d1fff0e2d5f3470a585e7"
FAILED_JOB_ID = "759a1b80df7f02c7-271807"
FAILED_LEDGER_SHA256 = "b125c5d6d6e20ef14735c9c4b790aa96c2d8da6f31009cf710d4c04cfb4b174e"
FAILED_MANIFEST_SHA256 = "b317c319e1fb43ee19a1902c673e5dce03ac88c31a7292798e027b511e5a17d5"
FAILED_SNAPSHOT_SHA256 = "793d07a5fc0475e009e7710e8d2d32159b831b7825a672a2ffe5a00eeb6865b6"
FAILED_CHARGED_SECONDS = 3.62394042599999
REPAIR_FILES = {"authorization.json", "failed_request.json", "failed_worker.log", "original_run_manifest.json", "original_snapshot.json"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def utc():
    return datetime.now(timezone.utc)


def atomic(path, payload):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Symlinked path refused")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_json(path, value):
    atomic(path, canonical(value) + b"\n")


def safe_path(value):
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and str(path) == value


def jobs_for(plan):
    jobs = plan["jobs"]
    require(len(jobs) == 52 and len({j["id"] for j in jobs}) == 52, "Exactly 52 unique frozen fits required")
    require({int(j["seed"]) for j in jobs} == set(SEEDS), "Only the four declared new seeds may run")
    models = {j["model"] for j in jobs}
    require(len(models) == 13 and {(j["model"], int(j["seed"])) for j in jobs}
            == {(m, s) for m in models for s in SEEDS}, "Frozen 13-model by four-seed grid differs")
    require(all(re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", j["id"]) for j in jobs), "Unsafe job identifier")
    return jobs


def initial_files(plan, code_hashes, run_id=None):
    """Pure allocation builder: no ACTIVE reservation, model or runtime creation."""
    jobs_for(plan)
    require(all(re.fullmatch(r"[a-f0-9]{64}", v) for v in code_hashes.values()) and bool(code_hashes), "Code hashes required")
    manifest = {"version": "e2-extension-run-v1", "run_id": run_id or uuid.uuid4().hex,
        "plan_fingerprint": sha(canonical(plan)), "jobs": plan["jobs"], "input_bindings": plan.get("inputs", {}),
        "code_hashes": code_hashes, "source_archive": plan.get("source_archive", {}),
        "source_binding_sha256": plan.get("source_binding_sha256"), "budget_stage": "new_e2_extension_52_fits", "limit_seconds": LIMIT,
        "historic_e3_charged_seconds": HISTORIC_E3_SECONDS, "historic_e3_closed_commit": INITIAL_PARENT}
    state = state_projection([], manifest)
    ledger = ledger_summary([], manifest)
    return {name: canonical(value) + b"\n" for name, value in {"plan.json": plan, "run_manifest.json": manifest,
            "state.json": state, "ledger.json": ledger}.items()} | {"ledger.jsonl": b""}


def events_from(payload):
    events, previous = [], "0" * 64
    for index, line in enumerate(payload.splitlines()):
        event = json.loads(line)
        digest = event.get("sha256")
        body = {k: v for k, v in event.items() if k != "sha256"}
        require(body.get("sequence") == index and body.get("previous_sha256") == previous
                and digest == sha(canonical(body)), "Ledger hash chain invalid")
        events.append(event)
        previous = digest
    return events


def ledger_summary(events, manifest):
    active, charged, finished = None, 0.0, set()
    valid_jobs = {j["id"] for j in manifest["jobs"]} | {"aggregate"}
    for event in events:
        require(event["run_id"] == manifest["run_id"] and event["plan_fingerprint"] == manifest["plan_fingerprint"]
                and event["job_id"] in valid_jobs and event["phase"] in {"fit", "evaluate", "aggregate"}, "Ledger identity differs")
        if event["event"] == "reserve":
            require(active is None and event["reservation_id"] not in finished, "Overlapping or reused reservation")
            cap = event["cap_seconds"]
            require(type(cap) in (int, float) and math.isfinite(cap) and 0 < cap <= LIMIT - charged + 1e-8,
                    "Reservation exceeds remaining shared budget")
            require(cap <= {"fit": 1800, "evaluate": 300, "aggregate": 120}[event["phase"]], "Phase cap exceeded")
            active = event
        elif event["event"] == "reopen":
            prefix = b"".join(canonical(row) + b"\n" for row in events[:2])
            require(event["sequence"] == 2 and manifest["run_id"] == FAILED_RUN_ID
                and event["job_id"] == FAILED_JOB_ID and event["phase"] == "fit"
                and event.get("repair_id") == REPAIR_ID and event.get("failed_head") == FAILED_HEAD
                and sha(prefix) == FAILED_LEDGER_SHA256 and active is None
                and charged == event.get("charged_seconds_preserved") == FAILED_CHARGED_SECONDS
                and event.get("authorization_sha256") == manifest.get("repair_authorization_sha256")
                and bool(re.fullmatch(r"[a-f0-9]{64}", event.get("authorization_sha256", ""))),
                "Reopening is restricted to the audited metadata-only first failure")
        elif event["event"] == "finish":
            require(active is not None and event["reservation_id"] == active["reservation_id"]
                    and event["job_id"] == active["job_id"] and event["phase"] == active["phase"], "Finish without matching reservation")
            elapsed = event["elapsed_worker_wall_seconds"]
            require(type(elapsed) in (int, float) and math.isfinite(elapsed) and 0 <= elapsed <= active["cap_seconds"] + 10,
                    "Invalid parent-measured worker wall time")
            require(event["outcome"] in {"success", "failed", "timeout", "lost"}, "Unknown finish outcome")
            if event["outcome"] == "lost":
                require(elapsed == active["cap_seconds"], "Lost worker must consume its full reserved cap")
            charged += elapsed
            finished.add(event["reservation_id"])
            active = None
        else:
            raise RuntimeError("Unknown ledger event")
    return {"version": "e2-extension-ledger-v1", "run_id": manifest["run_id"], "limit_seconds": LIMIT,
        "charged_seconds": charged, "remaining_seconds": max(0.0, LIMIT - charged), "events": len(events),
        "chain_head_sha256": events[-1]["sha256"] if events else "0" * 64,
        "active_reservation": active, "historic_e3_charged_seconds": HISTORIC_E3_SECONDS,
        "historic_cost_is_separate_closed_stage": True}


def state_projection(events, manifest):
    state = {"version": "e2-extension-state-v1", "run_id": manifest["run_id"], "status": "new", "jobs": {}}
    for event in events:
        if event["event"] == "reopen":
            require(state["status"] == "failed" and not state["jobs"], "Only the first pre-step failure can reopen")
            state["prior_failure"] = state.pop("failure")
            state["repair_id"], state["status"] = REPAIR_ID, "running"
            continue
        if event["event"] != "finish":
            continue
        if event["outcome"] == "success":
            state["jobs"].setdefault(event["job_id"], {})[event["phase"]] = {
                "receipt_path": event["receipt_path"], "receipt_sha256": event["receipt_sha256"],
                "finish_event_sha256": event["sha256"]}
            state["status"] = "completed" if event["phase"] == "aggregate" else "running"
        else:
            state["status"] = "unreconciled" if event["outcome"] == "lost" else "failed"
            state["failure"] = {"job_id": event["job_id"], "phase": event["phase"], "outcome": event["outcome"],
                "nonresumable_partial_fit_preserved": event["phase"] == "fit"}
    if state["status"] not in TERMINAL and ledger_summary(events, manifest)["remaining_seconds"] <= 0:
        state["status"] = "budget_exhausted"
    return state


def repair_projections(directory):
    manifest = read_json(directory / "run_manifest.json")
    events = events_from((directory / "ledger.jsonl").read_bytes())
    write_json(directory / "ledger.json", ledger_summary(events, manifest))
    write_json(directory / "state.json", state_projection(events, manifest))


def append_event(directory, body):
    manifest = read_json(directory / "run_manifest.json")
    path = directory / "ledger.jsonl"
    events = events_from(path.read_bytes())
    body = {**body, "version": "e2-extension-event-v1", "run_id": manifest["run_id"],
        "plan_fingerprint": manifest["plan_fingerprint"], "sequence": len(events),
        "previous_sha256": events[-1]["sha256"] if events else "0" * 64, "utc": utc().isoformat()}
    event = {**body, "sha256": sha(canonical(body))}
    summary = ledger_summary([*events, event], manifest)
    with path.open("ab") as stream:
        stream.write(canonical(event) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    write_json(directory / "ledger.json", summary)
    write_json(directory / "state.json", state_projection([*events, event], manifest))
    return event


def approved_path(name, manifest):
    if not safe_path(name):
        return False
    parts = PurePosixPath(name).parts
    jobs = {j["id"] for j in manifest["jobs"]}
    if name in BASE_FILES:
        return True
    if name.startswith(REPAIR_ROOT + "/"):
        return manifest.get("run_id") == FAILED_RUN_ID and name[len(REPAIR_ROOT) + 1:] in REPAIR_FILES
    if len(parts) == 2 and parts[0] == "remote_acks":
        return parts[1].endswith(".json") and parts[1][:-5] in jobs
    if len(parts) >= 3 and parts[0] == "jobs" and parts[1] in jobs:
        relative = "/".join(parts[2:])
        if relative in ARTIFACT_NAMES | {"calibration_predictions.json", "evaluation.json", "test_predictions.json"}:
            return True
        # Exact original fitter archive locations; no arbitrary datasets/source.
        if len(parts) == 4 and parts[2] == "fit":
            return parts[3] in {"request.json", "job_result.json", "selected_model.pt", "worker_spec.json", "worker.log"}
        if len(parts) >= 6 and parts[2:4] == ("fit", "phases") and parts[4] in {"supervised", "ae1", "ae2"}:
            tail = parts[5:]
            if len(tail) in (1, 2) and (len(tail) == 1 or tail[0] in {"checkpoints", "artifacts"}) and re.fullmatch(r"\.pending-[a-z0-9_]{8}", tail[-1]):
                return True
            return (len(tail) == 1 and tail[0] in {"identity.json", "prepared.json", "manifest.json", "events.jsonl",
                "superseded_checkpoint_hashes.json", "curves.csv", "worker_spec.json", "worker.log", "history.csv"}
                or len(tail) == 2 and tail[0] in {"checkpoints", "artifacts"} and bool(re.fullmatch(r"[A-Za-z0-9_.-]+\.(?:pt|csv|json)", tail[1])))
    return bool(len(parts) == 2 and parts[0] == "logs" and re.fullmatch(r"[a-f0-9]{32}\.log", parts[1]))


def collect(directory):
    manifest = read_json(directory / "run_manifest.json")
    payloads = {}
    for path in sorted(directory.rglob("*")):
        require(not path.is_symlink(), "Symlinked artifact refused")
        if path.is_file():
            name = path.relative_to(directory).as_posix()
            if path.name.endswith((".tmp", ".lock")):
                continue
            require(approved_path(name, manifest), "Unapproved artifact: " + name)
            require(path.stat().st_size <= (1024**2 if name.startswith("logs/") else MAX_BLOB), "Artifact exceeds upload cap")
            payloads[name] = path.read_bytes()
    require(sum(map(len, payloads.values())) <= MAX_SNAPSHOT, "Snapshot exceeds 2 GiB")
    return payloads


def make_snapshot(payloads):
    manifest = json.loads(payloads["run_manifest.json"])
    return {"version": "e2-extension-snapshot-v1", "run_id": manifest["run_id"],
        "plan_fingerprint": manifest["plan_fingerprint"],
        "files": {p: {"sha256": sha(v), "bytes": len(v)} for p, v in sorted(payloads.items())}}


def validate_snapshot(snapshot, payloads):
    require(snapshot == make_snapshot(payloads), "Snapshot path, bytes or identity mismatch")
    manifest, plan = json.loads(payloads["run_manifest.json"]), json.loads(payloads["plan.json"])
    require(sha(canonical(plan)) == manifest["plan_fingerprint"] and jobs_for(plan) == manifest["jobs"], "Immutable plan changed")
    require(manifest["limit_seconds"] == LIMIT and manifest["historic_e3_charged_seconds"] == HISTORIC_E3_SECONDS
            and manifest["historic_e3_closed_commit"] == INITIAL_PARENT, "Budget stage provenance changed")
    require(sum(map(len, payloads.values())) <= MAX_SNAPSHOT, "Snapshot exceeds cap")
    for name, value in payloads.items():
        require(approved_path(name, manifest) and len(value) <= (1024**2 if name.startswith("logs/") else MAX_BLOB), "Snapshot allowlist or size violation")
    events = events_from(payloads["ledger.jsonl"])
    require(json.loads(payloads["ledger.json"]) == ledger_summary(events, manifest), "Budget ledger projection changed")
    state = json.loads(payloads["state.json"])
    require(state == state_projection(events, manifest), "State differs from append-only ledger progress")
    if manifest.get("repair_authorization_sha256") is not None:
        authorization_path = REPAIR_ROOT + "/authorization.json"
        require(authorization_path in payloads and sha(payloads[authorization_path]) == manifest["repair_authorization_sha256"],
                "Repair authorization bytes differ")
        authorization = json.loads(payloads[authorization_path])
        require(authorization.get("repair_id") == REPAIR_ID and authorization.get("failed_head") == FAILED_HEAD
                and authorization.get("run_id") == FAILED_RUN_ID and manifest["run_id"] == FAILED_RUN_ID
                and authorization.get("old_manifest_sha256") == FAILED_MANIFEST_SHA256
                and authorization.get("new_code_hashes") == manifest["code_hashes"], "Repair authorization identity differs")
        old_manifest_bytes = payloads[REPAIR_ROOT + "/original_run_manifest.json"]
        require(sha(old_manifest_bytes) == FAILED_MANIFEST_SHA256, "Archived original manifest changed")
        old_manifest = json.loads(old_manifest_bytes)
        require({k: v for k, v in manifest.items() if k not in {"code_hashes", "repair_authorization_sha256"}}
                == {k: v for k, v in old_manifest.items() if k != "code_hashes"}, "Repair changed frozen science or budget identity")
        original = json.loads(payloads[REPAIR_ROOT + "/original_snapshot.json"])
        old_payloads = {p: v.encode() for p, v in original.items() if p != "snapshot.json"}
        require(sha(original["snapshot.json"].encode()) == FAILED_SNAPSHOT_SHA256,
                "Archived original snapshot differs")
        validate_snapshot(json.loads(original["snapshot.json"]), old_payloads)
        require(sha(old_payloads["ledger.jsonl"]) == FAILED_LEDGER_SHA256
                and payloads["ledger.jsonl"].startswith(old_payloads["ledger.jsonl"])
                and payloads[REPAIR_ROOT + "/failed_request.json"] == old_payloads[f"jobs/{FAILED_JOB_ID}/fit/request.json"]
                and payloads[REPAIR_ROOT + "/failed_worker.log"] == old_payloads["logs/eec1cdd8aaaa4b6c9a9579606bbbc8c9.log"],
                "Repair lost the failed ledger/request/log evidence")
        require(len(events) >= 3 and events[2]["event"] == "reopen", "Repair lacks its append-only reopening record")
    completed = {"fit": set(), "evaluate": set(), "aggregate": set()}
    gate = json.loads(payloads["all_fits_remote_ack.json"]) if "all_fits_remote_ack.json" in payloads else None
    for event in events:
        phase, jid = event["phase"], event["job_id"]
        if event["event"] == "reserve":
            require(jid not in completed[phase], "Completed scientific job cannot run twice")
            if phase == "evaluate":
                require(len(completed["fit"]) == 52 and gate is not None, "Evaluation before all 52 acknowledged fits")
            elif phase == "aggregate":
                require(len(completed["evaluate"]) == 52, "Aggregate before all 52 evaluations")
        elif event["event"] == "finish" and event["outcome"] == "success":
            path = event["receipt_path"]
            require(path in payloads and sha(payloads[path]) == event["receipt_sha256"], "Completed receipt changed")
            validate_receipt(json.loads(payloads[path]), manifest, jid, phase, payloads=payloads)
            completed[phase].add(jid)
    if gate is not None:
        require(gate.get("version") == "e2-seed-extension-v1" and gate.get("run_id") == manifest["run_id"]
            and gate.get("plan_fingerprint") == manifest["plan_fingerprint"] and len(completed["fit"]) == 52,
            "All-fits freeze identity or completed set differs")
        rows = gate.get("fits", [])
        require(len(rows) == 52 and {r["job_id"] for r in rows} == completed["fit"], "All-fits freeze requires 52 unique receipts")
        for row in rows:
            jid = row["job_id"]
            receipt_path, ack_path = f"jobs/{jid}/fit_receipt.json", f"remote_acks/{jid}.json"
            require(row["fit_receipt_sha256"] == sha(payloads[receipt_path])
                    and row["remote_ack_sha256"] == sha(payloads[ack_path]), "All-fits freeze receipt or remote ACK changed")
            ack = json.loads(payloads[ack_path])
            require(ack.get("job_id") == jid and ack.get("status") == "acknowledged"
                and ack.get("fit_receipt_sha256") == row["fit_receipt_sha256"]
                and bool(re.fullmatch(r"[a-f0-9]{40}", ack.get("ack_commit", ""))), "Fit remote ACK binding differs")
    return snapshot


def restore(directory, payloads):
    for name, value in payloads.items():
        require(safe_path(name), "Unsafe restore path")
        atomic(directory / name, value)


class NativeGit:
    """Pinned private askpass core, sparse tree, non-force native pushes and readback."""
    def __init__(self, core, token, *, test_remote=None):
        core.BRANCH = BRANCH
        options = {} if test_remote is None else {"_test_remote": test_remote, "_allow_local_for_tests": True}
        self.transport = core.GitTransport(token, repository=REPO, branch=BRANCH, **options)
        self.token = token
        self.blobs = {}
        self.hashed = {}

    def head(self):
        for attempt in range(2):
            try:
                return self.transport.head()
            except RuntimeError:
                if attempt:
                    raise
                time.sleep(1)

    def read(self):
        head = self.head()
        if head is None:
            return None
        records = self.transport._run(["ls-tree", "-r", "-z", head]).stdout.split(b"\0")
        objects = {}
        for row in filter(None, records):
            meta, name = row.split(b"\t", 1)
            mode, kind, object_sha = meta.decode().split()
            require(mode == "100644" and kind == "blob", "Nonregular remote snapshot entry")
            objects[name.decode()] = self.transport._sha(object_sha)
        names, marker = set(objects), PREFIX + "/snapshot.json"
        require(marker in names and all(p.startswith(PREFIX + "/") for p in names), "Foreign sparse snapshot tree")
        def blob(name):
            object_sha = objects[name]
            if object_sha not in self.blobs:
                self.blobs[object_sha] = self.transport._run(["cat-file", "blob", object_sha]).stdout
            return self.blobs[object_sha]
        snapshot = json.loads(blob(marker))
        require(names == {PREFIX + "/" + p for p in snapshot["files"]} | {marker}, "Unmanifested remote file")
        payloads = {p: blob(PREFIX + "/" + p) for p in snapshot["files"]}
        validate_snapshot(snapshot, payloads)
        return head, snapshot, payloads

    def publish(self, snapshot, payloads, expected):
        validate_snapshot(snapshot, payloads)
        require(not any(self.token.encode() in v for v in payloads.values()), "Credential in snapshot refused")
        actual = self.read()
        if actual and actual[1] == snapshot and actual[2] == payloads:
            return actual[0]
        require((actual[0] if actual else None) == expected, "Native Git writer conflict; no overwrite")
        parent = expected or INITIAL_PARENT
        if expected is None:
            self.transport._run(["fetch", "--depth=1", "--no-tags", "origin", parent])
        index = self.transport._private / ("extension-index-" + uuid.uuid4().hex)
        env = {"GIT_INDEX_FILE": str(index)}
        self.transport._run(["read-tree", "--empty"], env=env)
        records = []
        for name, value in sorted({**payloads, "snapshot.json": canonical(snapshot) + b"\n"}.items()):
            digest = sha(value)
            if digest not in self.hashed:
                self.hashed[digest] = self.transport._sha(self.transport._run(
                    ["hash-object", "-w", "--stdin"], data=value).stdout.decode().strip())
            blob = self.hashed[digest]
            self.blobs[blob] = value
            records.append(b"100644 " + blob.encode() + b"\t" + (PREFIX + "/" + name).encode() + b"\0")
        self.transport._run(["update-index", "-z", "--index-info"], data=b"".join(records), env=env)
        tree = self.transport._sha(self.transport._run(["write-tree"], env=env).stdout.decode().strip())
        commit = self.transport._sha(self.transport._run(["commit-tree", tree, "-p", parent],
            data=b"Preserve fixed E2 four-seed extension\n").stdout.decode().strip())
        for attempt in range(2):
            current = self.head()
            if current == commit:
                return self._confirm(commit, snapshot, payloads)
            require(current == expected, "Native Git writer conflict; no force push")
            try:
                self.transport._run(["push", "--porcelain", "origin", commit + ":" + self.transport._ref], allow_failure=True)
            except RuntimeError:
                pass
            current = self.head()
            if current == commit:
                return self._confirm(commit, snapshot, payloads)
            require(current == expected, "Native Git writer conflict after ambiguous push")
            if attempt == 0:
                time.sleep(1)
        raise RuntimeError("Native Git publication unconfirmed; completed local work preserved")

    def _confirm(self, commit, snapshot, payloads):
        remote = self.read()
        require(remote and remote == (commit, snapshot, payloads), "Remote byte readback ACK differs")
        return commit

    def close(self):
        self.token = ""
        self.blobs.clear()
        self.hashed.clear()
        self.transport.close()


def publish(client, directory, ack_path, expected):
    payloads = collect(directory)
    snapshot = make_snapshot(payloads)
    validate_snapshot(snapshot, payloads)
    head = client.publish(snapshot, payloads, expected)
    remote = client.read()
    require(remote == (head, snapshot, payloads), "Remote ACK readback missing")
    write_json(ack_path, {"head": head, "run_id": snapshot["run_id"], "plan_fingerprint": snapshot["plan_fingerprint"],
        "snapshot_sha256": sha(canonical(snapshot)), "ledger_sha256": sha(payloads["ledger.jsonl"])})
    return head


def reconcile(client, directory, ack_path, expected_manifest):
    remote = client.read()
    require(remote is not None and json.loads(remote[2]["run_manifest.json"]) == expected_manifest, "Remote run identity changed")
    if not directory.exists():
        restore(directory, remote[2])
    repair_projections(directory)
    local = collect(directory)
    if local != remote[2]:
        if remote[2]["ledger.jsonl"].startswith(local["ledger.jsonl"]) and remote[2]["ledger.jsonl"] != local["ledger.jsonl"]:
            validate_snapshot(make_snapshot(local), local)
            for event in events_from(local["ledger.jsonl"]):
                if event["event"] == "finish" and event["outcome"] == "success":
                    require(remote[2].get(event["receipt_path"]) == local[event["receipt_path"]], "Remote completed artifact conflict")
            restore(directory, remote[2])
            local = collect(directory)
    if local != remote[2]:
        # Local continuation must have observed this exact predecessor ACK, retain
        # every remote event, and preserve every already-completed artifact.
        ack = read_json(ack_path) if ack_path.is_file() else {}
        require(ack.get("head") == remote[0], "Unacknowledged divergent local state preserved; no new worker")
        require(local["run_manifest.json"] == remote[2]["run_manifest.json"] and local["plan.json"] == remote[2]["plan.json"]
                and local["ledger.jsonl"].startswith(remote[2]["ledger.jsonl"]), "Local ledger rollback or divergence refused")
        for event in events_from(remote[2]["ledger.jsonl"]):
            if event["event"] == "finish" and event["outcome"] == "success":
                path = event["receipt_path"]
                require(local.get(path) == remote[2][path], "Completed fit cannot be replaced")
        publish(client, directory, ack_path, remote[0])
        remote = client.read()
    write_json(ack_path, {"head": remote[0], "run_id": expected_manifest["run_id"],
        "plan_fingerprint": expected_manifest["plan_fingerprint"], "snapshot_sha256": sha(canonical(remote[1])),
        "ledger_sha256": sha(remote[2]["ledger.jsonl"])})
    return remote


def validate_receipt(receipt, manifest, job_id, phase, *, directory=None, payloads=None):
    require(receipt.get("run_id") == manifest["run_id"] and receipt.get("plan_fingerprint") == manifest["plan_fingerprint"]
            and receipt.get("job_id") == job_id and receipt.get("phase") == phase and receipt.get("status") == "completed",
            "Scientific receipt identity or completion differs")
    artifacts = receipt.get("artifacts", [])
    if isinstance(artifacts, dict):
        artifacts = list(artifacts.values())
    if phase == "fit":
        require(isinstance(artifacts, list) and bool(artifacts), "Successful fit needs hashed artifacts")
        paths = {r["path"] for r in artifacts}
        require(all(f"jobs/{job_id}/" + receipt[k] in paths for k in ("checkpoint", "scaler", "calibrator")),
                "Fit lacks checkpoint/scaler/calibrator")
    elif phase == "aggregate":
        require(isinstance(artifacts, list) and bool(artifacts), "Aggregate needs hashed outputs")
    elif phase == "evaluate":
        require(bool(receipt.get("fit_receipt_sha256")) and receipt.get("n_dates") == 608, "Evaluation fit binding or cohort differs")
        fit_path = f"jobs/{job_id}/fit_receipt.json"
        fit_bytes = payloads.get(fit_path) if payloads is not None else (directory / fit_path).read_bytes()
        require(fit_bytes is not None and sha(fit_bytes) == receipt["fit_receipt_sha256"], "Evaluation fitted predictor changed")
    for record in artifacts:
        path = record["path"]
        require(approved_path(path, manifest), "Receipt artifact not allowlisted")
        value = payloads.get(path) if payloads is not None else (directory / path).read_bytes()
        require(value is not None and record["sha256"] == sha(value) and record["bytes"] == len(value), "Receipt artifact hash differs")
    return receipt


def filtered_environment(environment):
    return {**{k: v for k, v in environment.items() if not k.startswith("GIT_")
        and not re.search(r"(?:TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY|ASKPASS|(?:^|_)PAT(?:_|$))", k.upper())}, "PYTHONDONTWRITEBYTECODE": "1"}


def trim_log(path):
    if path.is_file() and path.stat().st_size > 1024**2:
        with path.open("rb") as stream:
            stream.seek(-1024**2, os.SEEK_END)
            tail = stream.read()
        atomic(path, tail)


def show_worker_failure(path, client):
    if not path.is_file():
        return
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 8192))
        tail = stream.read().decode("utf-8", errors="replace")
    secret = getattr(client, "token", "")
    if secret:
        tail = tail.replace(secret, "[redacted]")
    tail = re.sub(r"(?:gh[pousr]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+)", "[redacted]", tail)
    print("Última salida del proceso (se conserva el registro completo acotado):\n" + tail, flush=True)


@contextmanager
def scientific_environment():
    original = dict(os.environ)
    os.environ.clear()
    os.environ.update(filtered_environment(original))
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(original)


def execute_worker(command, environment, cap, log_path):
    started = time.monotonic()
    process, outcome = None, "failed"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # File output avoids memory growth; retain only the final MiB after exit.
    with log_path.open("wb") as output:
        try:
            process = subprocess.Popen(command, env=filtered_environment(environment), stdout=output,
                stderr=subprocess.STDOUT, start_new_session=True)
            result = process.wait(timeout=max(0.0, cap - (time.monotonic() - started)))
            outcome = "success" if result == 0 else "failed"
        except subprocess.TimeoutExpired:
            outcome = "timeout"
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        except BaseException:
            if process and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        finally:
            elapsed = time.monotonic() - started
    trim_log(log_path)
    return outcome, elapsed


@contextmanager
def coordinator_lock(workspace):
    workspace.mkdir(parents=True, exist_ok=True)
    with (workspace / "coordinator.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another coordinator owns this VM") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def finish_job(directory, reservation, outcome, elapsed, receipt_path=None):
    manifest = read_json(directory / "run_manifest.json")
    receipt_hash = None
    if outcome == "success":
        validate_receipt(read_json(directory / receipt_path), manifest, reservation["job_id"], reservation["phase"], directory=directory)
        receipt_hash = sha((directory / receipt_path).read_bytes())
    event = append_event(directory, {"event": "finish", "job_id": reservation["job_id"], "phase": reservation["phase"],
        "reservation_id": reservation["reservation_id"], "outcome": outcome, "elapsed_worker_wall_seconds": elapsed,
        "receipt_path": receipt_path, "receipt_sha256": receipt_hash})
    return event


def run_coordinator(workspace, plan, client, runtime, command_builder, *, executor=execute_worker, max_jobs=None):
    """Dependency injection is only for deterministic lifecycle/local Git tests."""
    workspace, directory = Path(workspace), Path(workspace) / "run"
    ack_path = workspace / "ack.json"
    remote = client.read()
    require(remote is not None, "Run must be registered remotely before this launcher starts")
    manifest = json.loads(remote[2]["run_manifest.json"])
    require(manifest["plan_fingerprint"] == sha(canonical(plan)), "Registered immutable plan differs")
    remote = reconcile(client, directory, ack_path, manifest)
    owner_path = workspace / "owner.json"
    if not owner_path.exists():
        write_json(owner_path, {"owner_id": uuid.uuid4().hex})
    owner = read_json(owner_path)["owner_id"]
    count = 0
    while True:
        remote = reconcile(client, directory, ack_path, manifest)
        ledger, state = read_json(directory / "ledger.json"), read_json(directory / "state.json")
        if state["status"] in TERMINAL:
            return remote[0]
        active = ledger["active_reservation"]
        if active:
            require(active["owner_id"] == owner or datetime.fromisoformat(active["lease_until_utc"]) <= utc(),
                    "Another VM owns the ACTIVE worker lease")
            finish_job(directory, active, "lost", active["cap_seconds"])
            publish(client, directory, ack_path, remote[0])
            raise RuntimeError("Unreconciled nonresumable worker: reserved cap charged; partial artifacts preserved")
        # Recover a missing local remote-ACK marker from the already published
        # fit receipt; this never starts the original nonresumable fitter twice.
        added_ack = False
        for job in jobs_for(plan):
            jid = job["id"]
            row = state["jobs"].get(jid, {}).get("fit")
            ack = directory / "remote_acks" / (jid + ".json")
            if row and not ack.exists():
                require(remote[1]["files"][row["receipt_path"]]["sha256"] == row["receipt_sha256"], "Fit receipt ACK missing")
                write_json(ack, {"version": "e2-seed-extension-v1", "status": "acknowledged", "job_id": jid,
                    "run_id": manifest["run_id"], "plan_fingerprint": manifest["plan_fingerprint"],
                    "fit_receipt_sha256": row["receipt_sha256"], "ack_commit": remote[0]})
                added_ack = True
        if added_ack:
            publish(client, directory, ack_path, remote[0])
            remote = client.read()
        fit_jobs = [j for j in jobs_for(plan) if "fit" not in state["jobs"].get(j["id"], {})]
        if fit_jobs:
            job_id, phase = fit_jobs[0]["id"], "fit"
        else:
            binding = []
            for job in jobs_for(plan):
                jid = job["id"]
                row = state["jobs"][jid]["fit"]
                receipt = read_json(directory / row["receipt_path"])
                validate_receipt(receipt, manifest, jid, "fit", directory=directory)
                ack = directory / "remote_acks" / (jid + ".json")
                binding.append({"job_id": jid, "fit_receipt_sha256": row["receipt_sha256"], "remote_ack_sha256": sha(ack.read_bytes())})
            gate_path = directory / "all_fits_remote_ack.json"
            if not gate_path.exists():
                write_json(gate_path, {"version": "e2-seed-extension-v1", "run_id": manifest["run_id"],
                    "plan_fingerprint": manifest["plan_fingerprint"], "fit_count": 52,
                    "ack_commit": remote[0], "fits": binding})
                publish(client, directory, ack_path, remote[0])
                remote = client.read()
            gate = read_json(gate_path)
            require(gate["fits"] == binding and gate["fit_count"] == 52 and gate["ack_commit"], "All-fits test gate differs")
            pending_scores = [j for j in jobs_for(plan) if "evaluate" not in state["jobs"].get(j["id"], {})]
            job_id, phase = (pending_scores[0]["id"], "evaluate") if pending_scores else ("aggregate", "aggregate")
        remaining = ledger["remaining_seconds"]
        if remaining <= 0:
            state["status"] = "budget_exhausted"
            write_json(directory / "state.json", state)
            return publish(client, directory, ack_path, remote[0])
        cap = min({"fit": 1800.0, "evaluate": 300.0, "aggregate": 120.0}[phase], remaining)
        reservation = append_event(directory, {"event": "reserve", "job_id": job_id, "phase": phase,
            "reservation_id": uuid.uuid4().hex, "cap_seconds": cap, "owner_id": owner,
            "lease_until_utc": (utc() + timedelta(seconds=cap + 60)).isoformat()})
        head = publish(client, directory, ack_path, remote[0])
        claim = {"version": "e2-seed-extension-v1", "run_id": manifest["run_id"], "job_id": job_id, "phase": phase,
            "plan_fingerprint": manifest["plan_fingerprint"], "reservation_id": reservation["reservation_id"],
            "cap_seconds": cap, "ledger_event_sha256": reservation["sha256"], "ack_commit": head}
        claim_path = workspace / "claim.json"
        write_json(claim_path, claim)
        receipt_path = "aggregate_receipt.json" if phase == "aggregate" else f"jobs/{job_id}/" + ("fit_receipt.json" if phase == "fit" else "evaluation.json")
        print(f"ACK {head[:12]} · {phase} {job_id} · {ledger['charged_seconds']:.1f}/{LIMIT:.0f} s", flush=True)
        outcome, elapsed = "failed", 0.0
        started = time.monotonic()
        failure = None
        try:
            outcome, elapsed = executor(command_builder(phase, job_id, claim_path), runtime[1], cap,
                directory / "logs" / (reservation["reservation_id"] + ".log"))
            if outcome == "success":
                validate_receipt(read_json(directory / receipt_path), manifest, job_id, phase, directory=directory)
        except BaseException as error:
            outcome, elapsed, failure = "failed", time.monotonic() - started, error
        log_path = directory / "logs" / (reservation["reservation_id"] + ".log")
        trim_log(log_path)
        finish_job(directory, reservation, outcome, elapsed, receipt_path if outcome == "success" else None)
        if outcome != "success":
            show_worker_failure(log_path, client)
        # This must finish before any new child: a lost ACK only republishes bytes.
        publish(client, directory, ack_path, head)
        if failure:
            raise failure
        if outcome != "success":
            raise RuntimeError(f"{phase} {job_id} {outcome}; partial artifacts preserved without retry")
        count += 1
        if max_jobs is not None and count >= max_jobs:
            return client.read()[0]


def load_private(name, path):
    spec = importlib.util.spec_from_file_location(name + "_" + uuid.uuid4().hex, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def download(record, destination):
    destination = Path(destination)
    if destination.exists():
        require(sha(destination.read_bytes()) == record["sha256"], "Pinned local input changed: " + destination.name)
        return destination
    url = record.get("url") or f"https://raw.githubusercontent.com/{REPO}/{record['commit']}/{record['path']}"
    request = Request(url, headers={"User-Agent": "E2-frozen-seed-extension"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=60) as response:
                value = response.read(256 * 1024**2 + 1)
            require(len(value) <= 256 * 1024**2 and sha(value) == record["sha256"], "Pinned download hash differs")
            atomic(destination, value)
            return destination
        except (TimeoutError, ConnectionError, OSError):
            if attempt == 2:
                raise RuntimeError("Pinned input unavailable; no worker started") from None
            time.sleep(attempt + 1)


def token():
    value = os.environ.get("GITHUB_TOKEN", "").strip()
    if not value:
        try:
            from google.colab import userdata
            value = (userdata.get("GITHUB_TOKEN") or "").strip()
        except Exception:
            value = getpass("GITHUB_TOKEN (entrada oculta): ").strip()
    require(bool(value), "GITHUB_TOKEN required")
    return value


def show_results(remote):
    head, _, payloads = remote
    state, ledger = json.loads(payloads["state.json"]), json.loads(payloads["ledger.json"])
    fits = sum("fit" in row for row in state["jobs"].values())
    evaluations = sum("evaluate" in row for row in state["jobs"].values())
    print(f"Estado: {state['status']} · ajustes {fits}/52 · evaluaciones {evaluations}/52")
    print(f"Cómputo nuevo E2: {ledger['charged_seconds']:.2f}/{LIMIT:.0f} s; E3 cerrado conserva {HISTORIC_E3_SECONDS:.9f} s.")
    print(f"Resultados: https://github.com/{REPO}/tree/{head}/{PREFIX}", flush=True)


def run(workspace=None):
    plan = read_json(HERE / "plan.json")
    workspace = Path(workspace or plan.get("storage", {}).get("workspace", "/content/e2_extension_10_semillas_v1"))
    with coordinator_lock(workspace):
        core_path = download({"commit": CORE_COMMIT, "path": "scripts/e3_github_storage.py", "sha256": CORE_SHA},
            workspace / "private" / "e3_github_storage.py")
        core = load_private("e2_native_core", core_path)
        client = NativeGit(core, token())
        try:
            remote = client.read()
            require(remote is not None, "Run must already be registered; no implicit second allocation")
            manifest = json.loads(remote[2]["run_manifest.json"])
            require(manifest["plan_fingerprint"] == sha(canonical(plan)), "Pinned launcher plan differs")
            for name, expected in manifest["code_hashes"].items():
                require(safe_path(name) and sha((HERE / name).read_bytes()) == expected, "Pinned launcher code differs: " + name)
            os.environ.pop("GITHUB_TOKEN", None)
            os.environ.pop("GH_TOKEN", None)
            remote = reconcile(client, workspace / "run", workspace / "ack.json", manifest)
            if json.loads(remote[2]["state.json"])["status"] in TERMINAL:
                show_results(remote)
                return remote[0]
            for name, record in plan.get("inputs", {}).items():
                require(safe_path(name), "Unsafe input name")
                download(record, workspace / "inputs" / name)
            archive = download(plan["source_archive"], workspace / "inputs" / "02_codigo_modelos.zip")
            prepare = load_private("e2_prepare", HERE / "prepare.py")
            source = prepare.prepare_source(archive, workspace)
            require(source["source_binding_sha256"] == plan["source_binding_sha256"], "Complete source binding differs")
            parent_spec = plan["parent_runtime"]
            parent_source = workspace / "parent_source"
            for name, digest in parent_spec["files"].items():
                download({"commit": parent_spec["commit"], "path": name, "sha256": digest}, parent_source / name)
            helper = load_private("e2_verified_parent", parent_source / "notebooks/preparar_e3_innovacion_runtime.py")
            with scientific_environment():
                parent_python, parent_env, _ = helper.preparar_entorno(Path(parent_spec["workspace"]), device="cuda")
            runtime = prepare.prepare_runtime(workspace, parent_python, filtered_environment(parent_env))
            runtime_path = workspace / "run/runtime.json"
            runtime_record = {"version": "e2-seed-extension-v1", "run_id": manifest["run_id"],
                "source_binding_sha256": source["source_binding_sha256"], "identity": runtime[2]}
            if runtime_path.exists():
                require(read_json(runtime_path) == runtime_record, "Existing admitted runtime differs")
            else:
                write_json(runtime_path, runtime_record)
            def command(phase, job_id, claim_path):
                common = ["--inputs", str(workspace / "inputs"), "--run-dir", str(workspace / "run"),
                    "--phase", phase, "--job-id", job_id, "--claim-path", str(claim_path)]
                if phase == "aggregate":
                    return [str(runtime[0]), "-B", str(HERE / "aggregate.py"), *common,
                        "--plan", str(workspace / "run/plan.json")]
                return [str(runtime[0]), "-B", str(HERE / "worker.py"), *common,
                    "--bundle-root", source["package_root"], "--device", "cuda:0"]
            head = run_coordinator(workspace, plan, client, runtime, command)
            show_results(client.read())
            return head
        finally:
            client.close()
