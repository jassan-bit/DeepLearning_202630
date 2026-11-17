"""Recover the fixed stability campaign through native Git, without changing science."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import time
import uuid

PINS = {
    "bootstrap.py": "4ff6a3702f9ab90b5d8e283e47cadd0569bad12dea1bc3949bb39c8a74c58f9e",
    "worker.py": "739efe735d180e4aa6f2e28623632b21ace93bf21d18c958edd0649e9a7f07d8",
    "reference.py": "a79a4017d919e1ccad3f182a87b3bdb4f48016e4a463993e881808cf126a9613",
    "plan.json": "57de7c087113163b18fb384ffc73fed5dbf6045b37b42eae43c6261b1f76b148",
}
CORE_COMMIT = "cf891213f1acb6967aef5e0857fa0ac63768f836"
CORE_SHA = "1de75f622c35e0a9375e96197a620663411243c87c93b1bc2e23d45032f0d632"


def load_private(name, path):
    spec = importlib.util.spec_from_file_location(name + "_" + uuid.uuid4().hex, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_client_class(b, core, clients, *, test_remote=None):
    """Adapt the tested private askpass transport to the unchanged snapshot protocol."""
    core.BRANCH = b.BRANCH

    class NativeGit(b.GitHub):
        def __init__(self, token):
            super().__init__(token)
            options = {} if test_remote is None else {
                "_test_remote": test_remote, "_allow_local_for_tests": True}
            self.transport = core.GitTransport(token, repository=b.REPO, branch=b.BRANCH, **options)
            clients.append(self)

        def head(self):
            for attempt in range(2):
                try:
                    return self.transport.head()
                except core.StorageError:
                    if attempt:
                        raise
                    time.sleep(1)

        def read(self):
            head = self.head()
            if head is None:
                return None
            tree = self.transport._run(["show", "-s", "--format=%T", head]).stdout.decode().strip()
            names = self.transport._run(["ls-tree", "-r", "--name-only", "-z", head]).stdout.decode().split("\0")
            names = {p for p in names if p}
            marker = b.PREFIX + "/snapshot.json"
            b.require(marker in names and all(p.startswith(b.PREFIX + "/") for p in names),
                      "Foreign native-Git snapshot tree")
            snapshot = json.loads(self.transport._run(["show", head + ":" + marker]).stdout)
            b.require(all(b.safe_path(p) for p in snapshot["files"]), "Unsafe native snapshot path")
            b.require(names == {b.PREFIX + "/" + p for p in snapshot["files"]} | {marker},
                      "Unmanifested native-Git snapshot file")
            payloads = {p: self.transport._run(["show", head + ":" + b.PREFIX + "/" + p]).stdout
                        for p in snapshot["files"]}
            b.validate_snapshot(snapshot, payloads)
            return head, tree, snapshot, payloads

        def publish(self, snapshot, payloads, expected):
            b.validate_snapshot(snapshot, payloads)
            b.require(not any(self.token.encode() in value for value in payloads.values()),
                      "Credential found in snapshot; publication refused")
            actual = self.read()
            if actual and actual[2] == snapshot and actual[3] == payloads:
                return actual[0]
            b.require((actual[0] if actual else None) == expected, "Native-Git writer conflict; no overwrite")
            parent = expected or b.INITIAL_PARENT
            if expected is None:
                self.transport._run(["fetch", "--depth=1", "--no-tags", "origin", parent])
            index = self.transport._private / ("stability-index-" + uuid.uuid4().hex)
            environment = {"GIT_INDEX_FILE": str(index)}
            self.transport._run(["read-tree", "--empty"], env=environment)
            all_payloads = {**payloads, "snapshot.json": b.canonical(snapshot) + b"\n"}
            records = []
            for name, value in sorted(all_payloads.items()):
                blob = self.transport._sha(self.transport._run(
                    ["hash-object", "-w", "--stdin"], data=value).stdout.decode().strip())
                records.append(b"100644 " + blob.encode() + b"\t" +
                               (b.PREFIX + "/" + name).encode() + b"\0")
            self.transport._run(["update-index", "-z", "--index-info"],
                                data=b"".join(records), env=environment)
            tree = self.transport._sha(self.transport._run(["write-tree"], env=environment).stdout.decode().strip())
            commit = self.transport._sha(self.transport._run(
                ["commit-tree", tree, "-p", parent],
                data=b"Preserve E3 stability snapshot via native Git transport\n").stdout.decode().strip())
            for attempt in range(2):
                current = self.head()
                if current == commit:
                    return commit
                b.require(current == expected, "Native-Git writer conflict; no force push")
                try:
                    self.transport._run(["push", "--porcelain", "origin", commit + ":" + self.transport._ref],
                                        allow_failure=True)
                except core.StorageError:
                    pass
                current = self.head()
                if current == commit:
                    return commit
                b.require(current == expected, "Native-Git writer conflict after unconfirmed push")
                if attempt == 0:
                    time.sleep(1)
            raise RuntimeError("Native Git publication unconfirmed; local progress preserved, no next training chunk")

        def close(self):
            self.token = ""
            self.transport.close()

    return NativeGit


def run_recovery(workspace=None):
    import hashlib
    workspace = Path("/content/e3_estabilidad_aditiva_v1") if workspace is None else Path(workspace)
    launcher = workspace / "launcher"
    for name, expected in PINS.items():
        path = launcher / name
        if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Missing or changed original pinned source; preserved: " + name)
    b = load_private("stability_original_bootstrap", launcher / "bootstrap.py")
    plan = b.read_json(launcher / "plan.json")
    b.require(b.sha(b.canonical(plan)) == b.PLAN_FINGERPRINT, "Recovery plan differs")
    directory = workspace / "run"
    manifest = b.read_json(directory / "manifest.json")
    b.require(all(manifest[k] == PINS[name] for k, name in (
        ("bootstrap_sha256", "bootstrap.py"), ("worker_sha256", "worker.py"), ("reference_sha256", "reference.py"))),
        "Recovery scientific identity differs")
    payloads = b.collect(directory)
    b.validate_snapshot(b.make_snapshot(payloads), payloads)
    core_path = workspace / "repair_git_private" / "e3_github_storage.py"
    b.download(CORE_COMMIT, "scripts/e3_github_storage.py", core_path, CORE_SHA)
    core = load_private("stability_git_private", core_path)
    clients = []
    b.GitHub = native_client_class(b, core, clients)
    record = {"version": "e3-stability-transport-recovery-v1", "scope": "publication_transport_only",
        "repair_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "original_git_core_sha256": CORE_SHA, "scientific_pins": PINS,
        "worker_plan_budget_and_sessions_modified": False}
    record_path = directory / "transport_recovery.json"
    terminal = b.read_json(directory / "state.json")["status"] in b.TERMINAL
    if terminal:
        print("Campaña cerrada: se conserva su registro y se reconcilian los resultados pendientes.", flush=True)
    elif record_path.exists():
        b.require(b.read_json(record_path) == record, "Different transport recovery record; preserved")
    else:
        b.write_json(record_path, record)
    print("Transporte Git nativo preparado. Se conserva el cálculo y se publica el progreso pendiente antes de continuar.", flush=True)
    try:
        return b.run(workspace)
    finally:
        for client in clients:
            client.close()
