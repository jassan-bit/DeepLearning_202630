"""Exercise native publication against a local bare Git repository only."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("E3_EVAL_SOURCE_ROOT", str(HERE.parents[1]))).resolve()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class NativeGitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.remote = self.root / "remote.git"
        subprocess.run(["git", "init", "--bare", str(self.remote)], check=True, capture_output=True)
        self.b = module("repair_test_bootstrap", HERE / "bootstrap.py")
        self.core = module("repair_test_private_core", ROOT / "scripts/e3_github_storage.py")
        self.repair = module("repair_test_adapter", HERE / "repair_git_transport.py")
        self.assertEqual(hashlib.sha256((ROOT / "scripts/e3_github_storage.py").read_bytes()).hexdigest(), self.repair.CORE_SHA)
        self.clients = []
        Client = self.repair.native_client_class(self.b, self.core, self.clients, test_remote=self.remote)
        self.client = Client("test-secret-not-to-be-published")
        transport = self.client.transport
        blob = transport._run(["hash-object", "-w", "--stdin"], data=b"fixture root\n").stdout.decode().strip()
        tree = transport._run(["mktree"], data=f"100644 blob {blob}\tREADME\n".encode()).stdout.decode().strip()
        parent = transport._run(["commit-tree", tree], data=b"Fixture initial parent\n").stdout.decode().strip()
        transport._run(["push", "origin", parent + ":refs/heads/fixture-parent"])
        self.b.INITIAL_PARENT = parent
        self.parent = parent
        plan = json.loads((HERE / "plan.json").read_bytes())
        self.payloads = self.b.initial_files(plan, {"worker_sha256": "a" * 64,
            "bootstrap_sha256": "b" * 64, "reference_sha256": plan["reference"]["sha256"]},
            "run12345678", "claim12345678")
        # This fixture uses a reachable local parent, while production pins4cd.
        registry = json.loads(self.payloads["registry.json"])
        registry["prior_result_commit"] = parent
        self.payloads["registry.json"] = self.b.canonical(registry) + b"\n"
        self.snapshot = self.b.make_snapshot(self.payloads)

    def tearDown(self):
        for client in self.clients: client.close()
        self.temporary.cleanup()

    def advanced(self, cost=1):
        payloads = dict(self.payloads)
        ledger, registry = json.loads(payloads["ledger.json"]), json.loads(payloads["registry.json"])
        ledger["charged_seconds"] = registry["charged_seconds"] = cost
        state = json.loads(payloads["state.json"])
        state["status"] = "paused"
        state["seeds"] = {"271801": {"initialized": True, "epoch": 10, "batch_cursor": 0}}
        for name, value in (("ledger.json", ledger), ("registry.json", registry), ("state.json", state)):
            payloads[name] = self.b.canonical(value) + b"\n"
        return payloads

    def test_sparse_binary_roundtrip_idempotence_and_no_rest_writes(self):
        binary = b"opaque checkpoint fixture\x00\xff"
        self.payloads["seeds/271801/checkpoint.pt"] = binary
        self.payloads["seeds/271801/checkpoint.pt.sha256"] = (self.b.sha(binary) + "\n").encode()
        snapshot = self.b.make_snapshot(self.payloads)
        with patch.object(self.b.GitHub, "api", side_effect=AssertionError("REST write attempted")):
            head = self.client.publish(snapshot, self.payloads, None)
            self.assertEqual(self.client.read()[3], self.payloads)
            self.assertEqual(self.client.publish(snapshot, self.payloads, head), head)
        names = self.client.transport._run(["ls-tree", "-r", "--name-only", head]).stdout.decode().splitlines()
        self.assertTrue(all(name.startswith(self.b.PREFIX + "/") for name in names))
        self.assertNotIn("README", names)
        commit_headers = self.client.transport._run(["cat-file", "-p", head]).stdout.decode().split("\n\n", 1)[0]
        self.assertIn("parent " + self.parent, commit_headers.splitlines())

    def test_completed_local_chunk_is_uploaded_without_any_worker_or_cost_reset(self):
        head = self.client.publish(self.snapshot, self.payloads, None)
        run, ack = self.root / "run", self.root / "ack.json"
        self.b.restore(run, self.payloads)
        self.b.write_json(ack, {"head": head})
        advanced = self.advanced(cost=51.8)
        self.b.restore(run, advanced)
        with patch.object(self.b.subprocess, "run", side_effect=AssertionError("Scientific worker repeated")):
            result = self.b.reconcile(self.client, run, ack, json.loads(self.payloads["manifest.json"]))
        self.assertEqual(result[3], advanced)
        self.assertEqual(self.b.progress(result[3])["charged"], 51.8)
        self.assertEqual(self.b.progress(result[3])["seeds"]["271801"][1], 10)

    def test_lost_push_ack_reads_back_same_commit_and_never_force_pushes(self):
        head = self.client.publish(self.snapshot, self.payloads, None)
        advanced = self.advanced()
        original = self.client.transport._run
        pushed = []

        def uncertain(args, **kwargs):
            result = original(args, **kwargs)
            if args[0] == "push":
                pushed.append(args)
                raise self.core.StorageError("git_operation_unconfirmed")
            return result

        with patch.object(self.client.transport, "_run", side_effect=uncertain):
            new = self.client.publish(self.b.make_snapshot(advanced), advanced, head)
        self.assertEqual(self.client.head(), new)
        self.assertEqual(len(pushed), 1)
        self.assertFalse(any("force" in item or item.startswith("+") for args in pushed for item in args))

    def test_stale_writer_rejected_and_credentials_cleared_on_close(self):
        head = self.client.publish(self.snapshot, self.payloads, None)
        advanced = self.advanced()
        self.client.publish(self.b.make_snapshot(advanced), advanced, head)
        other = self.advanced(cost=2)
        with self.assertRaisesRegex(RuntimeError, "writer conflict"):
            self.client.publish(self.b.make_snapshot(other), other, head)
        environment = self.client.transport._environment
        self.assertIn("E3_GITHUB_TOKEN", environment)
        self.client.close()
        self.assertNotIn("E3_GITHUB_TOKEN", environment)
        self.assertEqual(self.client.token, "")


if __name__ == "__main__":
    unittest.main()
