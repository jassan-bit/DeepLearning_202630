"""Mock GitHub + synthetic byte fixtures only; no network, tokens or training."""
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from seoi_v5 import runner
from seoi_v5.github_store import GithubStore, StorageError, classify_http, _git_blob_sha, _zip

TOKEN = "synthetic-token-for-unit-tests-only"
SECRET = "synthetic-independent-backup-passphrase-32-characters"
SOURCE = "c" * 40


class FakeGit:
    """Small in-memory Git object/ref service, never an external request."""
    def __init__(self):
        self.head = None
        self.blobs = {}
        self.trees = {"0" * 40: {}}
        self.commits = {SOURCE: {"tree": {"sha": "0"*40}, "parents": []}}
        self.calls = []
        self.conflict_on_ref = False

    def request(self, method, path, payload=None, *, allow_missing=False):
        self.calls.append((method, path, payload))
        if "/git/ref/heads/" in path:
            return None if self.head is None else {"object": {"sha": self.head}}
        if "/contents/" in path:
            name, commit = path.split("/contents/", 1)[1].split("?ref=")
            tree = self.trees[self.commits[commit]["tree"]["sha"]]
            if name not in tree:
                if allow_missing:
                    return None
                raise StorageError("mock_missing_file")
            sha = tree[name]
            return {"type": "file", "sha": sha, "size": len(self.blobs[sha])}
        if method == "GET" and "/git/blobs/" in path:
            raw = self.blobs[path.rsplit("/", 1)[1]]
            return {"content": base64.b64encode(raw).decode(), "encoding": "base64"}
        if method == "GET" and "/git/commits/" in path:
            return self.commits[path.rsplit("/", 1)[1]]
        if method == "POST" and path.endswith("/git/blobs"):
            raw = base64.b64decode(payload["content"])
            sha = _git_blob_sha(raw); self.blobs[sha] = raw
            return {"sha": sha}
        if method == "POST" and path.endswith("/git/trees"):
            tree = dict(self.trees[payload["base_tree"]])
            for entry in payload["tree"]:
                tree[entry["path"]] = entry["sha"]
            sha = hashlib.sha1(runner.canonical(tree)).hexdigest(); self.trees[sha] = tree
            return {"sha": sha}
        if method == "POST" and path.endswith("/git/commits"):
            sha = hashlib.sha1(runner.canonical(payload)).hexdigest()
            self.commits[sha] = {"tree": {"sha": payload["tree"]}, "parents": payload["parents"]}
            return {"sha": sha}
        if (method == "POST" and path.endswith("/git/refs")) or (method == "PATCH" and "/git/refs/heads/" in path):
            if self.conflict_on_ref:
                raise StorageError("github_ref_conflict status=422")
            if method == "PATCH":
                assert payload["force"] is False
                if self.commits[payload["sha"]]["parents"] != [self.head]:
                    raise StorageError("github_ref_conflict")
            elif self.head is not None:
                raise StorageError("github_ref_conflict")
            self.head = payload["sha"]
            return {"object": {"sha": self.head}}
        raise AssertionError((method, path))


class StoreTests(unittest.TestCase):
    def store(self, service=None, secret=SECRET):
        obj = GithubStore(TOKEN, secret, source_commit=SOURCE, sleep=lambda n: None)
        if service is not None:
            obj._request = service.request
        return obj

    def case(self, folder, *, open_session=False):
        case = Path(folder)/"private"; case.mkdir()
        job = {"recipe": "seoi_mse", "seed": 20261127, "job_id": "seed20261127/seoi_mse"}
        plan = {"case_id": "seoi-v5-dev-synthetic-fixture", "jobs": [job], "version": runner.VERSION}
        runner.write_json(case/"plan.json", plan)
        runner.write_json(case/"prepared.json", {"plan_sha256": runner.digest(plan)})
        runner.write_json(case/"github_session.json", {"session_open": open_session})
        (case/"inputs").mkdir(); (case/"inputs/cohort.npz").write_bytes(b"RAW NOT TO BE UPLOADED")
        directory = case/"jobs"/job["job_id"]; directory.mkdir(parents=True)
        latest = "snapshot_" + "a"*32 + ".pt"
        raw = b"FAKE WEIGHTS OPTIMIZER AND RNG; SYNTHETIC BYTES NOT PICKLED"
        (directory/latest).write_bytes(raw)
        (directory/("snapshot_"+"b"*32+".pt")).write_bytes(b"UNUSED OLD SNAPSHOT")
        runner.write_json(directory/"checkpoint.json", {"file": latest, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        runner.write_json(directory/"curve.json", [{"epoch": 1, "selection_mse_scaled": .2}])
        runner.write_json(directory/"job.json", {"status": "completed_validated_fit"})
        runner.write_json(case/".github_controller_receipt.json", {"not": "part of the archive"})
        return case, plan, directory/latest

    def test_AES_roundtrip_tamper_and_independent_secret(self):
        store = self.store()
        context = {"case": "synthetic", "source_commit": SOURCE}
        first = store._encrypt(b"synthetic zip bytes", context)
        second = store._encrypt(b"synthetic zip bytes", context)
        self.assertNotEqual(first, second)
        self.assertEqual(store._decrypt(first, context), b"synthetic zip bytes")
        tampered = first[:-1] + bytes([first[-1] ^ 1])
        with self.assertRaisesRegex(StorageError, "decryption_failed"):
            store._decrypt(tampered, context)
        with self.assertRaisesRegex(StorageError, "decryption_failed"):
            self.store(secret="another independent passphrase long enough 32 chars")._decrypt(first, context)
        with self.assertRaisesRegex(StorageError, "context_"):
            store._decrypt(first, {"case": "different"})
        self.assertNotIn(TOKEN, repr(store)); self.assertNotIn(SECRET, repr(store))
        with self.assertRaises(StorageError):
            GithubStore(TOKEN, TOKEN, source_commit=SOURCE)

    def test_compact_archive_only_latest_PT_and_no_raw_or_PT_public(self):
        with tempfile.TemporaryDirectory() as folder:
            case, plan, latest = self.case(folder)
            context, private, public, counts = self.store()._archives(case)
            private_files = self.store()._validated_members(private, context)
            self.assertIn(latest.relative_to(case).as_posix(), private_files)
            self.assertEqual(len([n for n in private_files if n.endswith(".pt")]), 1)
            self.assertFalse(any(n.endswith(".npz") or n.endswith(".py") or "controller_receipt" in n for n in private_files))
            with zipfile.ZipFile(io.BytesIO(public)) as archive:
                names = archive.namelist()
                self.assertFalse(any(n.endswith((".pt", ".npz", ".npy")) for n in names))
                self.assertIn("plan.json", names)
            self.assertEqual(counts["completed_neural_fits"], 1)

    def test_atomic_three_file_backup_and_verified_restore(self):
        git = FakeGit(); store = self.store(git)
        self.assertFalse(store.status()["exists"])
        with tempfile.TemporaryDirectory() as folder:
            case, _, latest = self.case(folder, open_session=True)
            original = latest.read_bytes()
            receipt = store.backup(case, stage={"event": "job_completed"})
            self.assertTrue(receipt["remote_verified"])
            self.assertEqual(sum(m == "POST" and p.endswith("/git/blobs") for m,p,_ in git.calls), 3)
            tree = git.trees[git.commits[git.head]["tree"]["sha"]]
            self.assertEqual(len(tree), 3)
            self.assertFalse((case/".github_pending").exists())
            destination = Path(folder)/"restored"
            restored = store.restore(destination)
            self.assertEqual((destination/latest.relative_to(case)).read_bytes(), original)
            self.assertFalse((destination/"inputs").exists())
            recovery = runner.read_json(destination/"remote_restore.json")
            self.assertTrue(recovery["unobserved_remote_compute_possible"])
            self.assertFalse(recovery["budget_comparability_verified"])
            self.assertEqual(len(runner.read_json(destination/"recovery_history.json")), 1)
            self.assertEqual(runner.read_json(destination/".github_backup_receipt.json")["commit"], receipt["commit"])
            with self.assertRaisesRegex(StorageError, "fresh_restore"):
                store.restore(destination)
            self.assertEqual(restored["case_id"], receipt["case_id"])

    def test_ref_conflict_preserves_exact_pending_payload(self):
        git = FakeGit(); git.conflict_on_ref = True
        store = self.store(git)
        with tempfile.TemporaryDirectory() as folder:
            case, _, _ = self.case(folder)
            with self.assertRaisesRegex(StorageError, "ref_conflict"):
                store.backup(case)
            pending = case/".github_pending/checkpoint.aesgcm"
            before = pending.read_bytes()
            git.conflict_on_ref = False
            receipt = store.backup(case)
            encrypted = store._file_bytes("checkpoint.aesgcm", receipt["commit"], 32*1024*1024)
            self.assertEqual(before, encrypted)
            self.assertTrue(receipt["remote_verified"])

    def test_remote_newer_than_local_receipt_is_never_overwritten(self):
        git = FakeGit(); store = self.store(git)
        with tempfile.TemporaryDirectory() as folder:
            case, _, _ = self.case(folder)
            first = store.backup(case)
            stale_receipt = runner.read_json(case/".github_backup_receipt.json")
            runner.write_json(case/"fit_status.json", {"progress": 2})
            store.backup(case)
            runner.write_json(case/".github_backup_receipt.json", stale_receipt)
            runner.write_json(case/"fit_status.json", {"progress": 3})
            before = git.head
            with self.assertRaisesRegex(StorageError, "stale_local_case"):
                store.backup(case)
            self.assertEqual(git.head, before)
            self.assertNotEqual(before, first["commit"])

    def test_credential_text_in_public_metadata_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            case, _, _ = self.case(folder)
            runner.write_json(case/"failures.json", [{"accidental_secret": TOKEN}])
            with self.assertRaisesRegex(StorageError, "credential_material"):
                self.store()._archives(case)

    def test_authenticated_archive_traversal_or_symlink_is_refused(self):
        store = self.store(); context = {"fixture": True}
        for name, mode in (("../escape", 0), ("plan.json", 0o120777)):
            raw = io.BytesIO()
            with zipfile.ZipFile(raw, "w") as archive:
                info = zipfile.ZipInfo(name)
                info.external_attr = mode << 16
                archive.writestr(info, b"synthetic")
            with self.assertRaisesRegex(StorageError, "unsafe_or_symlink"):
                store._validated_members(raw.getvalue(), context)

    def test_missing_remote_with_prior_receipt_is_not_recreated(self):
        git = FakeGit(); store = self.store(git)
        with tempfile.TemporaryDirectory() as folder:
            case, _, _ = self.case(folder)
            store.backup(case)
            git.head = None
            with self.assertRaisesRegex(StorageError, "disappeared"):
                store.backup(case)

    def test_partial_pending_creation_never_installs_broken_transaction(self):
        git = FakeGit(); store = self.store(git)
        with tempfile.TemporaryDirectory() as folder:
            case, _, _ = self.case(folder)
            original = runner.atomic_bytes
            writes = [0]
            def crash(path, raw, **kwargs):
                if ".github-staging-" in str(path):
                    writes[0] += 1
                    if writes[0] == 2:
                        raise KeyboardInterrupt("synthetic staging crash")
                return original(path, raw, **kwargs)
            with mock.patch.object(runner, "atomic_bytes", side_effect=crash):
                with self.assertRaises(KeyboardInterrupt):
                    store.backup(case)
            self.assertFalse((case/".github_pending").exists())
            self.assertFalse(any(m != "GET" for m,p,v in git.calls))
            self.assertTrue(store.backup(case)["remote_verified"])

    def test_cleanup_failure_leaves_no_pending_and_next_backup_succeeds(self):
        git = FakeGit(); store = self.store(git)
        with tempfile.TemporaryDirectory() as folder:
            case, _, _ = self.case(folder)
            with mock.patch("seoi_v5.github_store.shutil.rmtree", side_effect=OSError("synthetic cleanup interruption")):
                first = store.backup(case)
            self.assertTrue(first["remote_verified"])
            self.assertFalse((case/".github_pending").exists())
            self.assertEqual(len(list(case.glob(".github-completed-cleanup-*"))), 1)
            self.assertTrue(store.backup(case)["remote_verified"])

    def test_wrong_remote_key_fails_before_pending_creation_then_correct_key_works(self):
        git = FakeGit(); good = self.store(git)
        with tempfile.TemporaryDirectory() as folder:
            case, _, _ = self.case(folder)
            good.backup(case)
            initial = git.head
            writes = sum(m != "GET" for m,p,v in git.calls)
            wrong = self.store(git, secret="mistyped independent backup secret longer than32")
            with self.assertRaisesRegex(StorageError, "decryption_failed"):
                wrong.backup(case)
            self.assertEqual(git.head, initial)
            self.assertFalse((case/".github_pending").exists())
            self.assertEqual(sum(m != "GET" for m,p,v in git.calls), writes)
            self.assertTrue(self.store(git).backup(case)["remote_verified"])

    def test_wrong_pending_key_cannot_publish_even_after_remote_authentication(self):
        git = FakeGit(); good = self.store(git)
        with tempfile.TemporaryDirectory() as folder:
            case, _, _ = self.case(folder)
            good.backup(case)
            initial = git.head
            writes = sum(m != "GET" for m,p,v in git.calls)
            wrong = self.store(git, secret="different independent key for corrupt pending fixture")
            wrong._pending(case, "synthetic malformed-key staging")
            with self.assertRaisesRegex(StorageError, "decryption_failed"):
                self.store(git).backup(case)
            self.assertEqual(git.head, initial)
            self.assertEqual(sum(m != "GET" for m,p,v in git.calls), writes)


class TransportTests(unittest.TestCase):
    def test_rate_limit_and_auth_are_distinguished(self):
        self.assertEqual(classify_http(403, "You have exceeded a secondary rate limit", {"Retry-After": "60"})["kind"], "secondary_rate_limit")
        self.assertEqual(classify_http(403, "rate limit", {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1050"}, now=1000)["retry_after_seconds"], 51)
        self.assertFalse(classify_http(403, "Resource not accessible by token")["retryable"])
        self.assertEqual(classify_http(422, "Reference update failed")["kind"], "ref_conflict")
        self.assertTrue(classify_http(503, "timeout")["retryable"])

    def test_http_retry_uses_same_payload_and_paces_writes(self):
        now = [1000.]
        sleeps, bodies, calls = [], [], [0]
        def sleep(n):
            sleeps.append(n); now[0] += n
        class Response:
            def __enter__(self): return self
            def __exit__(self,*a): pass
            def read(self,n): return b'{"ok":true}'
        def opener(request, timeout):
            bodies.append(request.data); calls[0] += 1
            if calls[0] == 1:
                raise urllib.error.HTTPError(request.full_url,403,"Forbidden",{"Retry-After":"3"},io.BytesIO(b'{"message":"secondary rate limit"}'))
            return Response()
        store = GithubStore(TOKEN, SECRET, source_commit=SOURCE, sleep=sleep, clock=lambda: now[0], opener=opener)
        self.assertEqual(store._request("POST","/synthetic",{"fixed":"payload"}),{"ok":True})
        store._request("POST","/synthetic",{"second":"write"})
        self.assertEqual(bodies[0], bodies[1])
        self.assertGreaterEqual(sum(sleeps),4.25)


if __name__ == "__main__":
    unittest.main()
