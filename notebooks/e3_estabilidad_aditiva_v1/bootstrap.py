"""GitHub coordinator for one fixed ten-seed stability allocation.

The coordinator never imports Torch, fits a model, or opens NPZ members. Each
scientific chunk needs an acknowledged remote ACTIVE reservation and must be
published before another chunk starts. Credentials never enter the worker.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
import csv
from datetime import datetime, timedelta, timezone
import fcntl
from getpass import getpass
import hashlib
from http.client import RemoteDisconnected
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from types import SimpleNamespace
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen
import uuid

HERE = Path(__file__).resolve().parent
REPO = "jassan-bit/DeepLearning_202630"
BRANCH = "e3-estabilidad-aditiva-v1"
PREFIX = "runs/estabilidad_aditiva_v1"
INITIAL_PARENT = "4cd455d90fb72fe78beeb885dc5dadc4ea9edeb4"
PLAN_FINGERPRINT = "0b2dbfea9f14f1506f09fd42b8e74fc299a91438b2a6c01fec6e4181fa7afa8c"
IDENTITY = {"name": "E3 development storage", "email": "e3@users.noreply.github.com"}
TERMINAL = {"completed", "budget_exhausted", "failed"}
SEEDS = tuple(range(271801, 271811))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def utc():
    return datetime.now(timezone.utc)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_bytes())


def atomic(path, payload):
    path = Path(path)
    require(not path.is_symlink(), "Symlinked state refused")
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
    rel = PurePosixPath(value)
    return bool(value and not rel.is_absolute() and ".." not in rel.parts
                and str(rel) == value and "\\" not in value)


def download(commit, path, destination, expected):
    destination = Path(destination)
    require(safe_path(path), "Unsafe source path")
    require(not any(p.is_symlink() for p in (destination, *destination.parents)), "Symlinked download refused")
    if destination.exists():
        payload = destination.read_bytes()
        require(sha(payload) == expected, "Local input differs; preserved: " + str(destination))
        return payload
    request = Request(f"https://raw.githubusercontent.com/{REPO}/{commit}/{path}",
                      headers={"User-Agent": "E3-seed-stability"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=45) as response:
                payload = response.read(32 * 1024 * 1024 + 1)
            require(len(payload) <= 32 * 1024 * 1024 and sha(payload) == expected,
                    "Downloaded input SHA256 mismatch: " + path)
            atomic(destination, payload)
            return payload
        except (RemoteDisconnected, TimeoutError, ConnectionError, URLError):
            if attempt == 2:
                raise RuntimeError("Input download unconfirmed; no scientific chunk started") from None
            time.sleep(attempt + 1)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HTTPError(req.full_url, code, "API redirect refused", headers, fp)


def error_detail(error, token):
    try:
        record = json.loads(error.read(8192))
        values = [record.get("message", "")] if isinstance(record, dict) else []
        for item in record.get("errors", [])[:4] if isinstance(record, dict) and isinstance(record.get("errors", []), list) else []:
            if isinstance(item, dict):
                values.extend(str(item[k]) for k in ("resource", "field", "code") if isinstance(item.get(k), str))
        value = " | ".join(v for v in values if isinstance(v, str))
        value = value.replace(token, "[redacted]") if token else value
        return re.sub(r"(?:gh[pousr]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+)", "[redacted]", " ".join(value.split()))[:600]
    except (ValueError, UnicodeError, OSError):
        return ""


class GitHub:
    """Retry immutable Git objects; reconcile every ambiguous ref write by reading."""
    def __init__(self, token):
        self.token, self.cache = token, {}

    def api(self, method, path, body=None, *, missing=False, retry=True):
        for attempt in range(3 if retry else 1):
            req = Request(f"https://api.github.com/repos/{REPO}/{path}", method=method,
                data=None if body is None else canonical(body), headers={
                    "Authorization": "Bearer " + self.token, "Accept": "application/vnd.github+json",
                    "Content-Type": "application/json", "User-Agent": "E3-seed-stability",
                    "X-GitHub-Api-Version": "2022-11-28"})
            try:
                with build_opener(NoRedirect()).open(req, timeout=45) as response:
                    return json.load(response)
            except HTTPError as error:
                if missing and error.code == 404:
                    return None
                if error.code in (429, 500, 502, 503, 504) and retry and attempt < 2:
                    time.sleep(attempt + 1)
                    continue
                detail = error_detail(error, self.token)
                raise RuntimeError(f"GitHub {method} {path}: HTTP {error.code}; {detail}") from None
            except (RemoteDisconnected, TimeoutError, ConnectionError, URLError):
                if not retry or attempt == 2:
                    raise ConnectionError("GitHub operation unconfirmed: " + method + " " + path) from None
                time.sleep(attempt + 1)

    def head(self):
        value = self.api("GET", "git/ref/heads/" + BRANCH, missing=True)
        return None if value is None else value["object"]["sha"]

    def blob(self, object_sha):
        if object_sha not in self.cache:
            record = self.api("GET", "git/blobs/" + object_sha)
            require(record["encoding"] == "base64", "Unexpected blob encoding")
            self.cache[object_sha] = base64.b64decode(record["content"])
        return self.cache[object_sha]

    def read(self):
        head = self.head()
        if head is None:
            return None
        commit = self.api("GET", "git/commits/" + head)
        tree = self.api("GET", "git/trees/" + commit["tree"]["sha"] + "?recursive=1")
        require(not tree.get("truncated"), "Incomplete remote tree")
        blobs = {r["path"]: r["sha"] for r in tree["tree"] if r["type"] == "blob"}
        marker = PREFIX + "/snapshot.json"
        require(marker in blobs and all(p.startswith(PREFIX + "/") for p in blobs), "Foreign or incomplete stability branch")
        snapshot = json.loads(self.blob(blobs[marker]))
        require(set(blobs) == {PREFIX + "/" + p for p in snapshot["files"]} | {marker}, "Unmanifested remote file")
        payloads = {p: self.blob(blobs[PREFIX + "/" + p]) for p in snapshot["files"]}
        validate_snapshot(snapshot, payloads)
        return head, commit["tree"]["sha"], snapshot, payloads

    def publish(self, snapshot, payloads, expected):
        validate_snapshot(snapshot, payloads)
        all_payloads = {**payloads, "snapshot.json": canonical(snapshot) + b"\n"}
        actual = self.read()
        if actual and actual[2] == snapshot and actual[3] == payloads:
            return actual[0]
        require((actual[0] if actual else None) == expected, "Remote head changed; no overwrite")
        entries = []
        for name, payload in sorted(all_payloads.items()):
            object_sha = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
            if object_sha not in self.cache:
                record = self.api("POST", "git/blobs", {"encoding": "base64", "content": base64.b64encode(payload).decode()})
                require(record["sha"] == object_sha, "Git blob identity mismatch")
                self.cache[object_sha] = payload
            entries.append({"path": PREFIX + "/" + name, "mode": "100644", "type": "blob", "sha": object_sha})
        tree = self.api("POST", "git/trees", {"tree": entries})
        stamp = utc().isoformat()
        identity = {**IDENTITY, "date": stamp}
        commit = self.api("POST", "git/commits", {"message": "Preserve E3 fixed ten-seed stability snapshot",
            "tree": tree["sha"], "parents": [expected or INITIAL_PARENT],
            "author": identity, "committer": identity})["sha"]
        for attempt in range(3):
            current = self.head()
            if current == commit:
                return commit
            require(current == expected, "Remote writer conflict; no force update")
            try:
                if expected is None:
                    self.api("POST", "git/refs", {"ref": "refs/heads/" + BRANCH, "sha": commit}, retry=False)
                else:
                    self.api("PATCH", "git/refs/heads/" + BRANCH, {"sha": commit, "force": False}, retry=False)
            except (ConnectionError, RuntimeError) as error:
                if self.head() == commit:
                    return commit
                require(self.head() == expected, "Remote writer conflict after unconfirmed write")
                if isinstance(error, RuntimeError):
                    raise
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
                continue
            require(self.head() == commit, "Remote publication ACK was not confirmed")
            return commit
        raise RuntimeError("Publication not acknowledged; no further scientific chunk allowed")


def progress(payloads):
    ledger, state = json.loads(payloads["ledger.json"]), json.loads(payloads["state.json"])
    value = {"charged": ledger["charged_seconds"], "status": state["status"], "seeds": {}}
    require(not isinstance(value["charged"], bool) and isinstance(value["charged"], (int, float)) and math.isfinite(value["charged"]) and value["charged"] >= 0,
            "Invalid charged budget")
    for seed in SEEDS:
        row = state.get("seeds", {}).get(str(seed), {})
        meta_path = f"seeds/{seed}/checkpoint.meta.json"
        if meta_path in payloads:
            row = json.loads(payloads[meta_path])
        value["seeds"][str(seed)] = [int(bool(row.get("initialized", False))), int(row.get("epoch", 0)), int(row.get("batch_cursor", 0))]
    return value


def dominates(new, old):
    return (new["charged"] >= old["charged"] and all(new["seeds"][k] >= v for k, v in old["seeds"].items())
            and (old["status"] not in TERMINAL or new == old))


def validate_snapshot(snapshot, payloads):
    require(snapshot["version"] == "e3-stability-snapshot-v1" and set(snapshot["files"]) == set(payloads), "Snapshot format mismatch")
    for name, payload in payloads.items():
        require(safe_path(name) and snapshot["files"][name] == {"sha256": sha(payload), "bytes": len(payload)}, "Snapshot byte hash mismatch")
    manifest = json.loads(payloads["manifest.json"])
    require(snapshot["scientific_identity"] == manifest, "Snapshot scientific identity mismatch")
    require(sha(canonical(json.loads(payloads["plan.json"]))) == manifest["plan_sha256"] == PLAN_FINGERPRINT, "Plan changed")
    ledger, registry = json.loads(payloads["ledger.json"]), json.loads(payloads["registry.json"])
    require(ledger["limit_seconds"] == registry["additional_limit_seconds"] == 500.0
            and ledger["prior_charged_seconds"] == registry["prior_charged_seconds"] == 650.3773435349993
            and ledger["global_limit_seconds"] == registry["global_limit_seconds"] == 1200.0,
            "Budget identity changed")
    require(registry["charged_seconds"] == ledger["charged_seconds"] and registry["run_id"] == ledger["run_id"] == manifest["run_id"]
            and registry["prior_result_commit"] == INITIAL_PARENT
            and registry["claim_id"] == manifest["claim_id"] and registry["scientific_identity"] == manifest,
            "Registry budget mismatch")
    active, lease = ledger.get("active_session"), registry.get("active_lease")
    require((active is None and lease is None) or (isinstance(active, dict) and isinstance(lease, dict)
            and active["session_id"] == lease["session_id"] and bool(lease["owner_id"])), "ACTIVE lease mismatch")
    for name in payloads:
        if name.endswith("/checkpoint.pt"):
            receipt = payloads.get(name + ".sha256", b"").decode().splitlines()
            require(sha(payloads[name]) in receipt, "Checkpoint transaction digest invalid")
        if name.endswith("/checkpoint.meta.json"):
            row = json.loads(payloads[name])
            require(row["run_id"] == manifest["run_id"] and row["plan_sha256"] == PLAN_FINGERPRINT
                    and 0 <= row["epoch"] <= 40 and 0 <= row["batch_cursor"] <= 532
                    and row["charged_seconds_checkpoint"] <= ledger["charged_seconds"], "Checkpoint metadata regressed or changed identity")
            checkpoint = name.removesuffix("checkpoint.meta.json") + "checkpoint.pt"
            require(checkpoint in payloads and row["checkpoint_sha256"] == sha(payloads[checkpoint])
                    and row["checkpoint_bytes"] == len(payloads[checkpoint]),
                    "Torn checkpoint metadata; local files preserved, no inconsistent snapshot published")
    require(snapshot["progress"] == progress(payloads), "Snapshot progress mismatch")


def collect(directory):
    result = {}
    for path in sorted(directory.rglob("*")):
        require(not path.is_symlink(), "Symlinked run file refused")
        if path.is_file() and not path.name.endswith((".tmp", ".lock")):
            relative = path.relative_to(directory).as_posix()
            require(relative != "snapshot.json", "Unexpected local remote-manifest file")
            result[relative] = path.read_bytes()
    return result


def make_snapshot(payloads):
    return {"version": "e3-stability-snapshot-v1", "scientific_identity": json.loads(payloads["manifest.json"]),
            "files": {p: {"sha256": sha(v), "bytes": len(v)} for p, v in payloads.items()}, "progress": progress(payloads)}


def update_registry(directory, owner=None, phase=None, training_snapshot=None):
    ledger, registry = read_json(directory / "ledger.json"), read_json(directory / "registry.json")
    registry["charged_seconds"] = ledger["charged_seconds"]
    active = ledger.get("active_session")
    registry["active_lease"] = None if not active else {
        "owner_id": owner, "session_id": active["session_id"], "phase": phase,
        "until_utc": (utc() + timedelta(seconds=max(0.0, 500 - ledger["charged_seconds"]) + 60)).isoformat(),
        "training_snapshot_commit": training_snapshot}
    write_json(directory / "registry.json", registry)


def restore(directory, payloads):
    directory.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        atomic(directory / name, payload)


def initial_files(plan, code_hashes, run_id=None, claim_id=None):
    """Build the immutable allocation descriptor, without initialization or fitting."""
    require(sha(canonical(plan)) == PLAN_FINGERPRINT, "Fixed plan fingerprint mismatch")
    run_id, claim_id = run_id or uuid.uuid4().hex, claim_id or uuid.uuid4().hex
    manifest = {"version": "e3-additive-stability-manifest-v1", "run_id": run_id, "claim_id": claim_id,
        "plan_sha256": PLAN_FINGERPRINT, "npz_sha256": plan["inputs"]["current_cohort_arrays.npz"]["sha256"],
        "data_fingerprint": "3d53f39ad2b738cae39449b0615292353e32060a73dba18397789e14f8db6317",
        "source_hashes": plan["source"]["files"], **code_hashes}
    require(all(re.fullmatch(r"[0-9a-f]{64}", manifest.get(k, "")) for k in
                ("worker_sha256", "bootstrap_sha256", "reference_sha256")), "Code SHA256 missing")
    registry = {"version": "e3-seed-stability-registry-v1", "run_id": run_id, "claim_id": claim_id,
        "prior_result_commit": INITIAL_PARENT, "prior_charged_seconds": 650.3773435349993,
        "additional_limit_seconds": 500.0, "global_limit_seconds": 1200.0,
        "charged_seconds": 0.0, "active_lease": None, "scientific_identity": manifest}
    ledger = {"version": "e3-seed-stability-ledger-v1", "run_id": run_id,
        "charged_seconds": 0.0, "limit_seconds": 500.0, "prior_charged_seconds": 650.3773435349993,
        "global_limit_seconds": 1200.0, "events": [], "active_session": None}
    state = {"status": "new", "run_id": run_id, "plan_sha256": PLAN_FINGERPRINT, "seeds": {}}
    return {n: canonical(v) + b"\n" for n, v in {"plan.json": plan, "manifest.json": manifest,
        "registry.json": registry, "ledger.json": ledger, "state.json": state}.items()}


def reconcile(client, directory, ack_path, manifest):
    remote = client.read()
    require(remote is not None and remote[2]["scientific_identity"] == manifest, "Remote run identity mismatch")
    if not directory.exists():
        restore(directory, remote[3])
    local = collect(directory)
    require(json.loads(local["manifest.json"]) == manifest, "Local scientific identity differs")
    if local != remote[3]:
        lp, rp = progress(local), remote[2]["progress"]
        ack = read_json(ack_path) if ack_path.is_file() else {}
        if ack.get("head") == remote[0] and dominates(lp, rp):
            old_lease = json.loads(remote[3]["registry.json"]).get("active_lease") or {}
            update_registry(directory, old_lease.get("owner_id"), old_lease.get("phase"), old_lease.get("training_snapshot_commit"))
            local = collect(directory)
            head = client.publish(make_snapshot(local), local, remote[0])
            remote = client.read()
            require(remote[0] == head, "Local recovery publication not acknowledged")
        elif dominates(rp, lp):
            restore(directory, remote[3])
        else:
            raise RuntimeError("Divergent local/remote progress; both preserved, no rollback")
    write_json(ack_path, {"head": remote[0], "run_id": manifest["run_id"], "plan_sha256": PLAN_FINGERPRINT,
                          "progress": remote[2]["progress"]})
    return remote


def token():
    value = os.environ.get("GITHUB_TOKEN", "").strip()
    if not value:
        try:
            from google.colab import userdata
            value = (userdata.get("GITHUB_TOKEN") or "").strip()
        except Exception:
            value = getpass("GITHUB_TOKEN (entrada oculta): ").strip()
    require(bool(value), "GITHUB_TOKEN is required before allocating compute")
    return value


def show_results(payloads, head):
    state, ledger = json.loads(payloads["state.json"]), json.loads(payloads["ledger.json"])
    summary = json.loads(payloads.get("summary.json", b"{}"))
    complete = sum(bool(v.get("complete")) for v in state.get("seeds", {}).values())
    print(f"\nEstado: {state['status']} · entrenamientos terminados {complete}/10")
    print(f"Consumo nuevo: {ledger['charged_seconds']:.2f}/500 s · "
          f"acumulado: {650.3773435349993 + ledger['charged_seconds']:.2f}/1200 s")
    if state["status"] == "completed":
        require(summary.get("comparison_complete") is True and "per_seed_metrics.csv" in payloads,
                "Completed snapshot is missing its complete ten-seed results")
        rows = list(csv.DictReader(io.StringIO(payloads["per_seed_metrics.csv"].decode())))
        require([int(row["seed"]) for row in rows] == list(SEEDS), "Final seed table differs from prescribed seeds")
        print(f"{'Semilla':>10} {'Mejor época':>12} {'RMSE':>13} {'MAE':>13}")
        for row in rows:
            print(f"{row['seed']:>10} {row['best_epoch']:>12} "
                  f"{float(row['RMSE']):>13.9f} {float(row['MAE']):>13.9f}")
        low, high = summary["RMSE_bootstrap_CI95"]
        print(f"RMSE medio de las diez semillas: {summary['RMSE_mean']:.9f}")
        print(f"Desviación estándar muestral: {summary['RMSE_sample_SD']:.9f}")
        print(f"IC95% bootstrap por semilla: [{low:.9f}, {high:.9f}], condicionado al calendario histórico conocido.")
        print("La semilla histórica 20261116 se conserva por separado. Esta etapa no acredita R3.")
    else:
        print("Campaña incompleta: no se presenta una media de diez semillas ni superioridad.")
    print("Resultados GitHub:", f"https://github.com/{REPO}/tree/{head}/{PREFIX}", flush=True)


@contextmanager
def coordinator_lock(workspace):
    workspace.mkdir(parents=True, exist_ok=True)
    with (workspace / "coordinator.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another coordinator is active in this Colab VM") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def run(workspace=None):
    plan = read_json(HERE / "plan.json")
    require(sha(canonical(plan)) == PLAN_FINGERPRINT, "Fixed plan fingerprint mismatch")
    workspace = Path(plan["storage"]["workspace"] if workspace is None else workspace)
    with coordinator_lock(workspace):
        return _run(workspace, plan)


def _run(workspace, plan):
    inputs, source, directory = workspace / "inputs", workspace / "source", workspace / "run"
    ack_path, claim_path = workspace / "ack.json", workspace / "claim.json"
    for name, record in plan["inputs"].items():
        download(record["commit"], record["path"], inputs / name, record["sha256"])
    for path, digest in plan["source"]["files"].items():
        download(plan["source"]["commit"], path, source / path, digest)
    prior = read_json(inputs / "prior_moment_summary.json")
    require(prior["status"] == "completed" and prior["global_charged_seconds"] == 650.3773435349993,
            "Prior closed allocation changed")
    owner_path = workspace / "owner.json"
    if not owner_path.is_file():
        write_json(owner_path, {"owner_id": uuid.uuid4().hex})
    owner = read_json(owner_path)["owner_id"]
    code_identity = {"plan_sha256": PLAN_FINGERPRINT, "worker_sha256": sha((HERE / "worker.py").read_bytes()),
        "bootstrap_sha256": sha(Path(__file__).read_bytes()), "reference_sha256": sha((HERE / "reference.py").read_bytes()),
        "source_hashes": plan["source"]["files"]}
    require(code_identity["reference_sha256"] == plan["reference"]["sha256"], "Reference source changed")
    client = GitHub(token())
    try:
        prior_heads = {"e3-desarrollo-resultados-v1": "f7826d120f3a6eb8e9a0952dd9842cd77f152f34",
            "e3-desarrollo-resultados-v2": "1ca029e869a517cb8de704e573c78ca1e0cbad3c",
            "e3-desarrollo-resultados-momentos": INITIAL_PARENT}
        for branch, expected in prior_heads.items():
            require(client.api("GET", "git/ref/heads/" + branch)["object"]["sha"] == expected,
                    "A closed predecessor branch changed; no new compute allocation")
        remote = client.read()
        if remote is None:
            require(not directory.exists() or not any(directory.iterdir()), "Remote campaign absent but local progress exists; no new allocation")
            payloads = initial_files(plan, {k: code_identity[k] for k in
                ("worker_sha256", "bootstrap_sha256", "reference_sha256")})
            manifest = json.loads(payloads["manifest.json"])
            run_id = manifest["run_id"]
            restore(directory, payloads)
            payloads = collect(directory)
            head = client.publish(make_snapshot(payloads), payloads, None)
            write_json(ack_path, {"head": head, "run_id": run_id, "plan_sha256": PLAN_FINGERPRINT,
                                  "progress": progress(payloads)})
        else:
            manifest = remote[2]["scientific_identity"]
            require(all(manifest.get(k) == v for k, v in code_identity.items()), "Permanent campaign identity differs; no second allocation")
        remote = reconcile(client, directory, ack_path, manifest)
        if remote[2]["progress"]["status"] in TERMINAL:
            show_results(remote[3], remote[0])
            return remote[0]
        helper_path = source / "notebooks/preparar_e3_innovacion_runtime.py"
        spec = importlib.util.spec_from_file_location("stability_runtime", helper_path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        previous = Path("/content/e3_momentos_github/runtime")
        runtime = previous if (previous / "runtime_identity.json").is_file() else workspace / "runtime"
        python, environment, _ = helper.preparar_entorno(runtime, device="cuda")
        environment = {k: v for k, v in environment.items() if not re.search(r"(?:^|_)(?:TOKEN|PAT|SECRET|PASSWORD|CREDENTIAL|API_KEY)(?:_|$)", k.upper())}
        while True:
            remote = reconcile(client, directory, ack_path, manifest)
            state, ledger = read_json(directory / "state.json"), read_json(directory / "ledger.json")
            if state["status"] in TERMINAL:
                break
            registry = read_json(directory / "registry.json")
            lease = registry.get("active_lease")
            if lease and lease["owner_id"] != owner:
                require(datetime.fromisoformat(lease["until_utc"]) <= utc(), "Another VM owns the active compute lease; no concurrent execution")
            phase = "evaluate" if state["status"] == "trained" else "train"
            training_head = ((lease or {}).get("training_snapshot_commit") or remote[0]) if phase == "evaluate" else None
            checkpoint_hashes = None
            if phase == "evaluate":
                require(all(state.get("seeds", {}).get(str(seed), {}).get("complete") is True for seed in SEEDS),
                        "All ten training trajectories must be published before evaluation")
                checkpoint_hashes = {str(seed): remote[2]["files"][f"seeds/{seed}/checkpoint.pt"]["sha256"] for seed in SEEDS}
            if ledger.get("active_session") is None:
                ledger["active_session"] = {"session_id": uuid.uuid4().hex, "last_wall_utc": utc().isoformat()}
                write_json(directory / "ledger.json", ledger)
            update_registry(directory, owner, phase, training_head)
            payloads = collect(directory)
            acknowledged = client.publish(make_snapshot(payloads), payloads, remote[0])
            write_json(ack_path, {"head": acknowledged, "run_id": manifest["run_id"],
                "plan_sha256": PLAN_FINGERPRINT, "progress": progress(payloads)})
            claim = {"version": "e3-additive-stability-claim-v1", "run_id": manifest["run_id"],
                "claim_id": manifest["claim_id"], "plan_sha256": PLAN_FINGERPRINT,
                "npz_sha256": manifest["npz_sha256"],
                "active_session_id": ledger["active_session"]["session_id"], "ack_commit": acknowledged,
                "additional_limit_seconds": 500.0, "prior_charged_seconds": 650.3773435349993,
                "global_limit_seconds": 1200.0, "phase": phase, "training_snapshot_commit": training_head,
                "checkpoint_sha256": checkpoint_hashes}
            write_json(claim_path, claim)
            print(f"ACK GitHub {acknowledged[:12]} · fase {phase} · consumo nuevo {ledger['charged_seconds']:.2f}/500 s", flush=True)
            failure = None
            try:
                subprocess.run([str(python), str(HERE / "worker.py"), "--source-root", str(source),
                    "--inputs", str(inputs), "--run-dir", str(directory), "--claim-path", str(claim_path),
                    "--device", "cuda", "--phase", phase, "--max-epochs-this-call", str(plan["training"]["max_epochs_this_call"])], env=environment, check=True)
            except BaseException as error:
                failure = error
            update_registry(directory, owner, phase, training_head)
            payloads = collect(directory)
            require(dominates(progress(payloads), remote[2]["progress"]), "Worker progress or cost regressed")
            head = client.publish(make_snapshot(payloads), payloads, acknowledged)
            write_json(ack_path, {"head": head, "run_id": manifest["run_id"],
                "plan_sha256": PLAN_FINGERPRINT, "progress": progress(payloads)})
            if failure:
                raise failure
        final = client.read()
        show_results(final[3], final[0])
        return final[0]
    finally:
        client.token = ""
