"""Coordinator verification with mocked GitHub only; never starts a model."""
from copy import deepcopy
from contextlib import redirect_stdout
import csv
import hashlib
from http.client import RemoteDisconnected
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("stability_bootstrap_test", HERE / "bootstrap.py")
b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(b)
PLAN = b.read_json(HERE / "plan.json")
CODE_HASHES = {"worker_sha256": "a" * 64, "bootstrap_sha256": "b" * 64,
               "reference_sha256": PLAN["reference"]["sha256"]}


def initial():
    return b.initial_files(PLAN, CODE_HASHES, "run12345678", "claim12345678")


def changed(payloads, cost=1., epoch=1):
    payloads = dict(payloads)
    ledger = json.loads(payloads["ledger.json"])
    ledger["charged_seconds"] = cost
    registry = json.loads(payloads["registry.json"])
    registry["charged_seconds"] = cost
    state = json.loads(payloads["state.json"])
    state["status"] = "paused"
    state["seeds"]["271801"] = {"initialized": True, "epoch": epoch, "batch_cursor": 0}
    for name, value in (("ledger.json", ledger), ("registry.json", registry), ("state.json", state)):
        payloads[name] = b.canonical(value) + b"\n"
    return payloads


class MemoryRemote:
    def __init__(self, payloads):
        self.payloads = payloads
        self.current = "1" * 40
        self.publications = 0

    def read(self):
        return self.current, "tree", b.make_snapshot(self.payloads), dict(self.payloads)

    def publish(self, snapshot, payloads, expected):
        if expected != self.current: raise RuntimeError("writer conflict")
        b.validate_snapshot(snapshot, payloads)
        self.publications += 1
        self.current = str(self.publications + 1) * 40
        self.payloads = dict(payloads)
        return self.current


class StorageTests(unittest.TestCase):
    def test_initial_allocation_is_unique_bound_and_without_active_work(self):
        payloads = initial()
        b.validate_snapshot(b.make_snapshot(payloads), payloads)
        self.assertEqual(json.loads(payloads["ledger.json"])["charged_seconds"], 0)
        self.assertIsNone(json.loads(payloads["ledger.json"])["active_session"])
        self.assertEqual(json.loads(payloads["manifest.json"])["worker_sha256"], CODE_HASHES["worker_sha256"])

    def test_cost_or_seed_rollback_rejected(self):
        old = b.progress(changed(initial(), 12, 2))
        self.assertFalse(b.dominates(b.progress(changed(initial(), 11, 3)), old))
        self.assertFalse(b.dominates(b.progress(changed(initial(), 13, 1)), old))
        self.assertTrue(b.dominates(b.progress(changed(initial(), 13, 3)), old))

    def test_registry_cannot_reset_ledger_cost(self):
        payloads = changed(initial(), 5)
        registry = json.loads(payloads["registry.json"])
        registry["charged_seconds"] = 0
        payloads["registry.json"] = b.canonical(registry)
        with self.assertRaisesRegex(RuntimeError, "Registry budget"):
            b.validate_snapshot(b.make_snapshot(payloads), payloads)

    def test_remote_restore_and_local_completed_chunk_upload_without_worker(self):
        remote = MemoryRemote(initial())
        manifest = json.loads(remote.payloads["manifest.json"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, ack = root / "run", root / "ack.json"
            b.reconcile(remote, directory, ack, manifest)
            self.assertEqual(b.collect(directory), remote.payloads)
            advanced = changed(b.collect(directory), 4, 2)
            b.restore(directory, advanced)
            b.reconcile(remote, directory, ack, manifest)
            self.assertEqual(remote.publications, 1)
            self.assertEqual(remote.payloads, advanced)
            self.assertEqual(b.read_json(ack)["head"], remote.current)

    def test_stale_ack_cannot_discard_higher_local_progress(self):
        remote = MemoryRemote(initial())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            b.restore(root / "run", changed(initial(), 10, 3))
            b.write_json(root / "ack.json", {"head": "different"})
            with self.assertRaisesRegex(RuntimeError, "Divergent"):
                b.reconcile(remote, root / "run", root / "ack.json", json.loads(initial()["manifest.json"]))
            self.assertEqual(b.progress(b.collect(root / "run"))["charged"], 10)

    def test_abandoned_active_timestamp_preserved_for_budget_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            b.restore(run, initial())
            ledger = b.read_json(run / "ledger.json")
            stamp = "2026-11-16T00:00:00+00:00"
            ledger["active_session"] = {"session_id": "session123", "last_wall_utc": stamp}
            b.write_json(run / "ledger.json", ledger)
            b.update_registry(run, "new-owner", "train")
            self.assertEqual(b.read_json(run / "ledger.json")["active_session"]["last_wall_utc"], stamp)
            b.validate_snapshot(b.make_snapshot(b.collect(run)), b.collect(run))

    def test_torn_checkpoint_metadata_is_preserved_but_not_published(self):
        payloads = initial()
        checkpoint = "seeds/271801/checkpoint.pt"
        payloads[checkpoint] = b"opaque checkpoint bytes"
        payloads[checkpoint + ".sha256"] = (b.sha(payloads[checkpoint]) + "\n").encode()
        meta = {"run_id": "run12345678", "plan_sha256": b.PLAN_FINGERPRINT,
                "epoch": 1, "batch_cursor": 0, "initialized": True,
                "charged_seconds_checkpoint": 0, "checkpoint_sha256": "stale",
                "checkpoint_bytes": len(payloads[checkpoint])}
        payloads["seeds/271801/checkpoint.meta.json"] = b.canonical(meta)
        with self.assertRaisesRegex(RuntimeError, "Torn checkpoint"):
            b.validate_snapshot(b.make_snapshot(payloads), payloads)
        meta["checkpoint_sha256"] = b.sha(payloads[checkpoint])
        payloads["seeds/271801/checkpoint.meta.json"] = b.canonical(meta)
        b.validate_snapshot(b.make_snapshot(payloads), payloads)

    def test_final_display_lists_all_seeds_and_incomplete_has_no_ten_seed_mean(self):
        payloads = initial()
        output = io.StringIO()
        with redirect_stdout(output): b.show_results(payloads, "1" * 40)
        self.assertIn("incompleta", output.getvalue())
        self.assertNotIn("RMSE medio", output.getvalue())
        state = json.loads(payloads["state.json"])
        state.update(status="completed", seeds={str(s): {"complete": True} for s in b.SEEDS})
        payloads["state.json"] = b.canonical(state)
        payloads["summary.json"] = b.canonical({"comparison_complete": True, "RMSE_mean": .003,
            "RMSE_sample_SD": .0001, "RMSE_bootstrap_CI95": [.0029, .0031]})
        rows = io.StringIO()
        writer = csv.DictWriter(rows, fieldnames=["seed", "best_epoch", "RMSE", "MAE"])
        writer.writeheader()
        writer.writerows({"seed": s, "best_epoch": 38, "RMSE": .003, "MAE": .002} for s in b.SEEDS)
        payloads["per_seed_metrics.csv"] = rows.getvalue().encode()
        output = io.StringIO()
        with redirect_stdout(output): b.show_results(payloads, "1" * 40)
        self.assertIn("10/10", output.getvalue())
        self.assertIn("IC95%", output.getvalue())
        self.assertIn("calendario histórico conocido", output.getvalue())
        for seed in b.SEEDS: self.assertIn(str(seed), output.getvalue())


class WireTests(unittest.TestCase):
    def test_read_connection_retry_is_bounded(self):
        with patch.object(b, "build_opener") as opener, patch.object(b.time, "sleep"):
            opener.return_value.open.side_effect = [RemoteDisconnected(), io.BytesIO(b'{"ok":true}')]
            self.assertEqual(b.GitHub("hidden").api("GET", "git/ref/heads/test"), {"ok": True})
            self.assertEqual(opener.return_value.open.call_count, 2)
        with patch.object(b, "build_opener") as opener, patch.object(b.time, "sleep"):
            opener.return_value.open.side_effect = RemoteDisconnected()
            with self.assertRaises(ConnectionError):
                b.GitHub("hidden").api("GET", "git/ref/heads/test")
            self.assertEqual(opener.return_value.open.call_count, 3)

    def test_lost_ref_ack_is_read_back_without_second_write(self):
        class Fake(b.GitHub):
            def __init__(self, race=False, rejection=False):
                super().__init__("hidden")
                self.current, self.ref_writes, self.race, self.rejection = "1" * 40, 0, race, rejection
            def read(self): return ("1" * 40, "tree", {}, {})
            def head(self): return "3" * 40 if self.race else self.current
            def api(self, method, path, body=None, **kwargs):
                if path == "git/blobs":
                    import base64
                    content = base64.b64decode(body["content"])
                    return {"sha": hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()}
                if path == "git/trees": return {"sha": "tree"}
                if path == "git/commits":
                    self.commit_body = body
                    return {"sha": "2" * 40}
                if path.startswith("git/refs/"):
                    self.ref_writes += 1
                    if self.rejection: raise RuntimeError("GitHub HTTP 422")
                    self.current = body["sha"]
                    raise ConnectionError("ACK lost")
                raise AssertionError(path)
        payloads = initial()
        client = Fake()
        self.assertEqual(client.publish(b.make_snapshot(payloads), payloads, "1" * 40), "2" * 40)
        self.assertEqual(client.ref_writes, 1)
        self.assertEqual(client.commit_body["author"]["date"], client.commit_body["committer"]["date"])
        conflict = Fake(race=True)
        with self.assertRaisesRegex(RuntimeError, "writer conflict"):
            conflict.publish(b.make_snapshot(payloads), payloads, "1" * 40)
        self.assertEqual(conflict.ref_writes, 0)
        rejected = Fake(rejection=True)
        with self.assertRaisesRegex(RuntimeError, "HTTP 422"):
            rejected.publish(b.make_snapshot(payloads), payloads, "1" * 40)
        self.assertEqual(rejected.ref_writes, 1)


if __name__ == "__main__":
    unittest.main()
