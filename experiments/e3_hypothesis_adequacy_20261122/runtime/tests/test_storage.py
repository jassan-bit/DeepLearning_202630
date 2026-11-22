"""Offline Git REST and synthetic-byte fixtures; no token, real data or fits."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import urllib.error
from urllib.parse import unquote
from unittest import mock
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import storage

TOKEN = "synthetic-token-for-software-tests-only"
SECRET = "synthetic-independent-passphrase-at-least-32-characters"
SOURCE = "c" * 40


def write(path, value):
    storage._atomic(path, storage._canonical(value))


class FakeGit:
    """Git objects with immediate (hierarchical) trees and non-force refs."""
    def __init__(self):
        self.head = None
        self.blobs = {}
        self.trees = {"0" * 40: {}}
        self.commits = {SOURCE: {"tree": {"sha": "0" * 40}, "parents": []}}
        self.calls = []
        self.conflict_on_ref = False
        self.lose_ref_reply = False
        self.corrupt_read = None

    def tree(self, commit):
        result = {}
        def walk(sha, prefix=""):
            for name, entry in self.trees[sha].items():
                path = prefix + name
                if entry["type"] == "tree":
                    walk(entry["sha"], path + "/")
                else:
                    result[path] = entry["sha"]
        walk(self.commits[commit]["tree"]["sha"])
        return result

    def request(self, method, path, payload=None, *, allow_missing=False):
        self.calls.append((method, path, payload))
        if "/git/ref/heads/" in path:
            return None if self.head is None else {"object": {"sha": self.head}}
        if "/contents/" in path:
            name, commit = path.split("/contents/", 1)[1].split("?ref=")
            name = unquote(name)
            tree = self.tree(commit)
            if name not in tree:
                if allow_missing:
                    return None
                raise storage.StorageError("mock_missing_file")
            sha = tree[name]
            return {"type": "file", "sha": sha, "size": len(self.blobs[sha])}
        if method == "GET" and "/git/blobs/" in path:
            sha = path.rsplit("/", 1)[1]
            raw = self.blobs[sha]
            if self.corrupt_read == sha:
                raw = raw[:-1] + bytes([raw[-1] ^ 1])
            return {"content": base64.b64encode(raw).decode(), "encoding": "base64"}
        if method == "GET" and "/git/commits/" in path:
            return self.commits[path.rsplit("/", 1)[1]]
        if method == "GET" and "/git/trees/" in path:
            values = self.trees[path.rsplit("/", 1)[1]]
            return {"tree": [{"path": n, **v} for n, v in values.items()], "truncated": False}
        if method == "POST" and path.endswith("/git/blobs"):
            raw = base64.b64decode(payload["content"])
            assert len(raw) <= storage.MAX_BLOB
            sha = storage._git_blob_sha(raw)
            self.blobs[sha] = raw
            return {"sha": sha}
        if method == "POST" and path.endswith("/git/trees"):
            assert len(payload["tree"]) <= 128
            tree = dict(self.trees[payload["base_tree"]]) if "base_tree" in payload else {}
            for entry in payload["tree"]:
                assert "/" not in entry["path"]
                if entry["sha"] is None:
                    tree.pop(entry["path"], None)
                else:
                    tree[entry["path"]] = {k: v for k, v in entry.items() if k != "path"}
            sha = hashlib.sha1(storage._canonical(tree)).hexdigest()
            self.trees[sha] = tree
            return {"sha": sha}
        if method == "POST" and path.endswith("/git/commits"):
            sha = hashlib.sha1(storage._canonical(payload)).hexdigest()
            self.commits[sha] = {"tree": {"sha": payload["tree"]}, "parents": payload["parents"]}
            return {"sha": sha}
        if (method == "POST" and path.endswith("/git/refs")) or (method == "PATCH" and "/git/refs/heads/" in path):
            if self.conflict_on_ref:
                raise storage.StorageError("github_ref_conflict status=422")
            if method == "PATCH":
                assert payload["force"] is False
                if self.commits[payload["sha"]]["parents"] != [self.head]:
                    raise storage.StorageError("github_ref_conflict")
            elif self.head is not None:
                raise storage.StorageError("github_ref_conflict")
            self.head = payload["sha"]
            if self.lose_ref_reply:
                self.lose_ref_reply = False
                raise storage.StorageError("github_transport_retry_same_pending_snapshot")
            return {"object": {"sha": self.head}}
        raise AssertionError((method, path))


class StoreTests(unittest.TestCase):
    def store(self, git=None, secret=SECRET, **kwargs):
        result = storage.GithubStore(TOKEN, secret, "owner/repository", "r3-plan-fixture", source_commit=SOURCE,
                                     prefix="runs/r3/fixture", sleep=lambda _: None, **kwargs)
        if git is not None:
            result._request = git.request
        return result

    def case(self, base, final=False, size=200):
        case = Path(base)/"case"
        case.mkdir()
        job = {"id": "F00-m00-s00", "model": "input_only_mse", "seed": 123, "panel": "F00"}
        plan = {"case_id": "r3-engine-" + ("final" if final else "plan") + "-fixture", "jobs": [job],
                "protocol_sha256": "a"*64, "source_commit": SOURCE, "seeds": [123],
                "data_panels": [{"id": "F00", "data_sha256": "b"*64}]}
        write(case/"plan.json", plan)
        write(case/"session.json", {"session_open": True})
        jobdir = case/"jobs"/job["id"]
        jobdir.mkdir(parents=True)
        filename = "snapshot_" + "d"*32 + ".pt"
        raw = b"synthetic-model-optimizer-RNG-" * (size // 30 + 1)
        (jobdir/filename).write_bytes(raw)
        pointer = {"file": filename, "bytes": len(raw), "sha256": storage._sha(raw)}
        write(jobdir/"checkpoint.json", {"latest": pointer, "best": pointer})
        write(jobdir/"done.json", {"status": "completed"})
        (jobdir/("snapshot_" + "e"*32 + ".pt")).write_bytes(b"old-unused-not-in-inventory")
        (case/"inputs").mkdir()
        (case/"inputs/cohort.npz").write_bytes(b"PRIVATE-MARKET-DATA-NEVER-UPLOAD")
        return case, plan, jobdir/filename

    def mutations(self, git):
        return sum(method != "GET" for method, _, _ in git.calls)

    def test_AES_fresh_randomness_tamper_and_context_binding(self):
        obj = self.store()
        context = {"source": SOURCE, "plan_sha256": "a"*64}
        first = obj._encrypt(b"synthetic", context)
        second = obj._encrypt(b"synthetic", context)
        self.assertNotEqual(first, second)
        self.assertEqual(obj._decrypt(first, context), b"synthetic")
        with self.assertRaisesRegex(storage.StorageError, "decryption_failed"):
            obj._decrypt(first[:-1] + bytes([first[-1] ^ 1]), context)
        with self.assertRaisesRegex(storage.StorageError, "context_or_KDF"):
            obj._decrypt(first, {"source": SOURCE, "plan_sha256": "b"*64})
        with self.assertRaisesRegex(storage.StorageError, "decryption_failed"):
            self.store(secret="different-independent-secret-longer-than-32")._decrypt(first, context)
        self.assertNotIn(TOKEN, repr(obj))
        self.assertNotIn(SECRET, repr(obj))

    def test_credentials_cannot_enter_public_storage_identity_or_stage(self):
        with self.assertRaisesRegex(storage.StorageError, "credentials_in_storage_identity"):
            storage.GithubStore(TOKEN, SECRET, "owner/repo", TOKEN, source_commit=SOURCE, prefix="runs/r3")
        with self.assertRaisesRegex(storage.StorageError, "github_token_required"):
            storage.GithubStore("bad token " * 3, SECRET, "owner/repo", "branch", source_commit=SOURCE, prefix="runs/r3")
        git = FakeGit()
        with tempfile.TemporaryDirectory() as base:
            case, _, _ = self.case(base)
            # Scan before truncation, including a secret that starts after the
            # public stage's 100-character bound.
            with self.assertRaisesRegex(storage.StorageError, "credential_material"):
                self.store(git).backup(case, stage="a" * 90 + SECRET)
            self.assertEqual(self.mutations(git), 0)

    def test_atomic_public_allowlist_and_authenticated_restore(self):
        git = FakeGit()
        obj = self.store(git)
        with tempfile.TemporaryDirectory() as base:
            case, plan, latest = self.case(base)
            public = {"metrics/planning.json": b'{"synthetic":true}', "docs/readme.md": b"fixture"}
            receipt = obj.backup(case, public, stage="initial-before-any-fit")
            self.assertTrue(receipt["remote_verified"])
            tree = git.tree(git.head)
            self.assertTrue(all(n.startswith(obj.prefix + "/") for n in tree))
            self.assertIn(obj.prefix + "/checkpoint.aesgcm", tree)
            self.assertIn(obj.prefix + "/status.json", tree)
            self.assertIn(obj.prefix + "/public_results.zip", tree)
            self.assertTrue(all(n.endswith(".aesgcm") or n.endswith(("status.json", "public_results.zip")) for n in tree))
            self.assertNotIn(b"PRIVATE-MARKET", b"".join(git.blobs.values()))
            with zipfile.ZipFile(io.BytesIO(obj._file_bytes("public_results.zip", git.head))) as archive:
                self.assertEqual(set(archive.namelist()), set(public))
            target = Path(base)/"restored"
            restored = self.store(git).restore(target)
            self.assertTrue(restored["remote_verified"])
            self.assertEqual((target/latest.relative_to(case)).read_bytes(), latest.read_bytes())
            self.assertEqual(len(list(target.rglob("*.pt"))), 1)
            self.assertFalse((target/"inputs").exists())
            self.assertTrue(json.loads((target/"remote_restore.json").read_text())["unobserved_remote_compute_possible"])
            self.assertFalse((case/".github_pending").exists())
            self.assertEqual(obj.status()["case_id"], plan["case_id"])

    def test_sharding_exceeds_old_20MiB_limit_and_reuses_unchanged_chunks(self):
        git = FakeGit()
        obj = self.store(git)
        with tempfile.TemporaryDirectory() as base:
            case, _, latest = self.case(base)
            latest.write_bytes(os.urandom(21 * 1024 * 1024 + 17))
            pointer = {"file": latest.name, "bytes": latest.stat().st_size, "sha256": storage._sha(latest.read_bytes())}
            write(latest.parent/"checkpoint.json", {"latest": pointer, "best": pointer})
            first = obj.backup(case)
            manifest = obj._remote_manifest
            big = next(r for r in manifest["files"] if r["path"].endswith(".pt"))
            self.assertEqual(len(big["chunks"]), 3)
            self.assertTrue(all(c["cipher_bytes"] <= 8 * 1024 * 1024 for c in big["chunks"]))
            initial_blobs = sum(m == "POST" and p.endswith("/git/blobs") for m,p,_ in git.calls)
            write(case/"status.json", {"new": "progress"})
            obj.backup(case)
            later_blobs = sum(m == "POST" and p.endswith("/git/blobs") for m,p,_ in git.calls) - initial_blobs
            self.assertEqual(later_blobs, 3)  # changed private status + encrypted index + public status
            self.assertNotEqual(first["commit"], git.head)
            restored = Path(base)/"restored"
            self.store(git).restore(restored)
            self.assertEqual((restored/latest.relative_to(case)).read_bytes(), latest.read_bytes())

    def test_many_unchanged_job_pointers_reuse_authenticated_metadata_without_download(self):
        git=FakeGit();obj=self.store(git)
        with tempfile.TemporaryDirectory() as base:
            case,plan,latest=self.case(base)
            first_job=plan["jobs"][0]
            for index in range(1,12):
                job={**first_job,"id":f"F00-m00-s{index:02d}","seed":123+index}
                plan["jobs"].append(job)
                directory=case/"jobs"/job["id"]
                directory.mkdir(parents=True)
                raw=(f"synthetic-job-{index}-model-optimizer-RNG"*8).encode()
                (directory/latest.name).write_bytes(raw)
                pointer={"file":latest.name,"bytes":len(raw),"sha256":storage._sha(raw)}
                write(directory/"checkpoint.json",{"latest":pointer,"best":pointer})
                write(directory/"done.json",{"status":"completed"})
            write(case/"plan.json",plan)
            obj.backup(case)
            pointer_shas={chunk["git_sha"] for record in obj._remote_manifest["files"]
                          if record["path"].startswith("jobs/") and Path(record["path"]).name in storage.POINTER_NAMES
                          for chunk in record["chunks"]}
            self.assertEqual(len(pointer_shas),12)
            git.calls.clear()
            write(case/"status.json",{"progress":2})
            result=obj.backup(case)
            pointer_gets=[path for method,path,_ in git.calls if method=="GET" and "/git/blobs/" in path
                          and path.rsplit("/",1)[1] in pointer_shas]
            self.assertEqual(pointer_gets,[])
            self.assertTrue(result["remote_verified"])
            # Warm cache never disables full byte verification on restoration.
            git.corrupt_read=next(iter(pointer_shas))
            with self.assertRaisesRegex(storage.StorageError,"readback_mismatch"):
                obj.restore(Path(base)/"corrupt-restore")
            self.assertFalse((Path(base)/"corrupt-restore").exists())

    def test_cached_pointer_still_checks_current_checkpoint_inventory_and_new_head_paths(self):
        git=FakeGit();obj=self.store(git)
        with tempfile.TemporaryDirectory() as base:
            case,_,_=self.case(base)
            obj.backup(case)
            manifest=json.loads(json.dumps(obj._remote_manifest));context=manifest["context"]
            checkpoint=next(r for r in manifest["files"] if r["path"].endswith(".pt"))
            checkpoint["sha256"]="0"*64
            with self.assertRaisesRegex(storage.StorageError,"restored_checkpoint_pointer_hash_differs"):
                obj._verify_files(manifest,context,git.head,local=Path(base))
            # A commit unfamiliar to this store still needs path/SHA readback,
            # although the authenticated plaintext pointer itself is cached.
            old=git.head;new="f"*40
            git.commits[new]={"tree":git.commits[old]["tree"],"parents":[old]};git.head=new
            pointer_record=next(r for r in obj._remote_manifest["files"] if Path(r["path"]).name=="checkpoint.json")
            paths={obj.prefix+"/"+c["path"] for c in pointer_record["chunks"]}
            git.calls.clear();obj.status()
            content_reads=[unquote(path.split("/contents/",1)[1].split("?ref=")[0])
                           for method,path,_ in git.calls if method=="GET" and "/contents/" in path]
            self.assertTrue(paths.issubset(content_reads))

    def test_large_complete_prediction_csv_stays_private_and_public_metadata_is_bounded(self):
        from engine import controller
        git=FakeGit();obj=self.store(git)
        with tempfile.TemporaryDirectory() as base:
            case,_,_=self.case(base)
            (case/"predictions").mkdir()
            # More than the old public ZIP/raw-member limit, with no real rows.
            raw=b"synthetic_payload\n"+base64.b64encode(os.urandom(9*1024*1024))+b"\n"
            prediction=case/"predictions/planning_predictions.csv";prediction.write_bytes(raw)
            self.assertGreater(len(raw),storage.MAX_BLOB)
            admissions={p:{"rights":{"allow_derived_publication":True}} for p in controller.PANELS}
            public=controller._public(case,admissions)
            self.assertFalse(any(name.startswith("predictions/") for name in public))
            receipt=obj.backup(case,public)
            self.assertTrue(receipt["remote_verified"])
            public_zip=obj._file_bytes("public_results.zip",git.head)
            self.assertLess(len(public_zip),2*1024*1024)
            with zipfile.ZipFile(io.BytesIO(public_zip)) as archive:
                self.assertFalse(any(name.startswith("predictions/") for name in archive.namelist()))
                inventory=json.loads(archive.read("receipts/private_artifact_inventory.json"))
            item=next(r for r in inventory["artifact_inventory"] if r["path"]=="predictions/planning_predictions.csv")
            self.assertEqual(item["sha256"],storage._sha(raw));self.assertEqual(item["bytes"],len(raw))
            target=Path(base)/"full-restore"
            self.store(git).restore(target)
            self.assertEqual((target/"predictions/planning_predictions.csv").read_bytes(),raw)
            denied={p:{"rights":{"allow_derived_publication":False}} for p in controller.PANELS}
            self.assertEqual(controller._public(case,denied),{})

    def test_ref_failure_retries_exact_pending_snapshot(self):
        git = FakeGit()
        git.conflict_on_ref = True
        obj = self.store(git)
        with tempfile.TemporaryDirectory() as base:
            case, _, _ = self.case(base)
            with self.assertRaisesRegex(storage.StorageError, "ref_conflict"):
                obj.backup(case)
            pending = (case/".github_pending/checkpoint.aesgcm").read_bytes()
            git.conflict_on_ref = False
            obj.backup(case)
            self.assertEqual(obj._file_bytes("checkpoint.aesgcm", git.head), pending)

    def test_lost_successful_ref_reply_reconciles(self):
        git = FakeGit()
        git.lose_ref_reply = True
        with tempfile.TemporaryDirectory() as base:
            case, _, _ = self.case(base)
            self.assertTrue(self.store(git).backup(case)["remote_verified"])

    def test_crash_after_ref_before_receipt_reconciles_new_process(self):
        git = FakeGit()
        obj = self.store(git)
        original = storage._atomic
        def crash(path, raw):
            if Path(path).name == ".github_backup_receipt.json":
                raise KeyboardInterrupt("synthetic crash")
            return original(path, raw)
        with tempfile.TemporaryDirectory() as base:
            case, _, _ = self.case(base)
            with mock.patch.object(storage, "_atomic", side_effect=crash):
                with self.assertRaises(KeyboardInterrupt):
                    obj.backup(case)
            before = git.head
            self.assertTrue(self.store(git).backup(case)["remote_verified"])
            self.assertEqual(git.head, before)

    def test_stale_VM_and_disappeared_branch_are_blocked(self):
        git = FakeGit()
        obj = self.store(git)
        with tempfile.TemporaryDirectory() as base:
            case, _, _ = self.case(base)
            first = obj.backup(case)
            write(case/"status.json", {"progress": 1})
            second = obj.backup(case)
            write(case/".github_backup_receipt.json", first)
            write(case/"status.json", {"progress": 2})
            before = self.mutations(git)
            with self.assertRaisesRegex(storage.StorageError, "stale_local_case"):
                obj.backup(case)
            self.assertEqual(self.mutations(git), before)
            self.assertEqual(git.head, second["commit"])
            shutil.rmtree(case/".github_pending")
            git.head = None
            with self.assertRaisesRegex(storage.StorageError, "disappeared"):
                obj.backup(case)

    def test_wrong_key_never_rotates_remote_or_creates_pending(self):
        git = FakeGit()
        with tempfile.TemporaryDirectory() as base:
            case, _, _ = self.case(base)
            self.store(git).backup(case)
            head, mutations = git.head, self.mutations(git)
            with self.assertRaisesRegex(storage.StorageError, "decryption_failed"):
                self.store(git, secret="wrong-independent-secret-longer-than-32-characters").backup(case)
            self.assertFalse((case/".github_pending").exists())
            self.assertEqual(git.head, head)
            self.assertEqual(self.mutations(git), mutations)

    def test_remote_blob_corruption_blocks_restore_without_destination(self):
        git = FakeGit()
        with tempfile.TemporaryDirectory() as base:
            case, _, _ = self.case(base)
            obj = self.store(git)
            obj.backup(case)
            chunk = next(c for r in obj._remote_manifest["files"] for c in r["chunks"])
            git.corrupt_read = chunk["git_sha"]
            destination = Path(base)/"restore"
            with self.assertRaisesRegex(storage.StorageError, "readback_mismatch"):
                self.store(git).restore(destination)
            self.assertFalse(destination.exists())

    def test_private_pointer_traversal_symlink_and_hash_rejected(self):
        for mode in ("traversal", "symlink", "hash"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as base:
                case, _, latest = self.case(base)
                if mode == "traversal":
                    write(latest.parent/"checkpoint.json", {"file": "../../escape.pt", "bytes": 1, "sha256": "a"*64})
                elif mode == "symlink":
                    latest.unlink()
                    latest.symlink_to(case/"plan.json")
                else:
                    latest.write_bytes(b"bad")
                with self.assertRaises(storage.StorageError):
                    self.store(FakeGit()).backup(case)

    def test_only_new_case_ids_and_complete_plan_AAD(self):
        with tempfile.TemporaryDirectory() as base:
            case, plan, _ = self.case(base)
            plan["case_id"] = "seoi-v5-dev-old-case"
            write(case/"plan.json", plan)
            with self.assertRaisesRegex(storage.StorageError, "new_r3"):
                self.store(FakeGit()).backup(case)
        obj = self.store()
        a = {"case_id": "r3-engine-plan-fixture", "source_commit": SOURCE, "seeds": [1], "protocol_sha256": "a"*64, "data_panels": [{"data_sha256":"a"*64}]}
        for key, value in (("seeds", [2]), ("protocol_sha256", "b"*64), ("data_panels", [{"data_sha256":"b"*64}])):
            self.assertNotEqual(obj._context(a), obj._context({**a,key:value}))

    def test_final_predictions_require_matching_sealed_receipt(self):
        with tempfile.TemporaryDirectory() as base:
            case, plan, _ = self.case(base, final=True)
            obj = self.store(FakeGit())
            with self.assertRaisesRegex(storage.StorageError, "sealed"):
                obj.backup(case, {"predictions/arbitrary.csv": b"yhat\n1\n"})
            write(case/"evaluation_receipt.json", {"sealed": True, "case_id": "other"})
            with self.assertRaisesRegex(storage.StorageError, "sealed"):
                obj.backup(case, {"predictions/arbitrary.csv": b"yhat\n1\n"})
            write(case/"evaluation_receipt.json", {"sealed": True, "case_id": plan["case_id"]})
            self.assertTrue(obj.backup(case, {"predictions/arbitrary.csv": b"yhat\n1\n"})["remote_verified"])

    def test_public_raw_weights_source_unsafe_and_credentials_are_refused(self):
        with tempfile.TemporaryDirectory() as base:
            case, _, _ = self.case(base)
            obj = self.store(FakeGit())
            for name, raw in (("raw/data.csv", b"data"), ("metrics/weights.pt", b"weights"),
                              ("docs/source.py", b"source"), ("../metrics/a.json", b"{}"),
                              ("metrics/secret.json", TOKEN.encode())):
                with self.subTest(name=name), self.assertRaises(storage.StorageError):
                    obj.backup(case, {name: raw})
            write(case/"failures.json", [{"accident": SECRET}])
            with self.assertRaisesRegex(storage.StorageError, "credential_material"):
                obj.backup(case)

    def test_restore_missing_and_existing_destinations(self):
        obj = self.store(FakeGit())
        with tempfile.TemporaryDirectory() as base:
            self.assertIsNone(obj.restore(Path(base)/"missing"))
            with self.assertRaisesRegex(storage.StorageError, "fresh_restore"):
                obj.restore(base)

    def test_staging_crash_never_installs_partial_pending_or_writes_remote(self):
        git = FakeGit()
        obj = self.store(git)
        original = storage._atomic
        count = [0]
        def crash(path, raw):
            if ".github-staging-" in str(path):
                count[0] += 1
                if count[0] == 2:
                    raise KeyboardInterrupt("synthetic staging crash")
            return original(path, raw)
        with tempfile.TemporaryDirectory() as base:
            case, _, _ = self.case(base)
            with mock.patch.object(storage, "_atomic", side_effect=crash):
                with self.assertRaises(KeyboardInterrupt):
                    obj.backup(case)
            self.assertFalse((case/".github_pending").exists())
            self.assertEqual(self.mutations(git), 0)
            self.assertTrue(obj.backup(case)["remote_verified"])

    def test_manifest_path_traversal_and_duplicate_records_rejected(self):
        obj = self.store()
        context = {"case_id":"fixture"}
        record = {"path":"../escape", "bytes":0, "sha256":storage._sha(b""), "chunks":[]}
        manifest = {"format":"chunked-private-files-1", "context":context,"files":[record],
                    "public_results_bytes":0,"public_results_sha256":storage._sha(b"")}
        with self.assertRaisesRegex(storage.StorageError, "member_invalid"):
            obj._validate_manifest(manifest, context)
        record["path"] = "plan.json"
        manifest["files"] = [record,record]
        with self.assertRaisesRegex(storage.StorageError, "member_invalid"):
            obj._validate_manifest(manifest, context)

    def test_initial_metadata_only_backup_before_any_fit(self):
        git = FakeGit()
        with tempfile.TemporaryDirectory() as base:
            case, _, latest = self.case(base)
            shutil.rmtree(case/"jobs")
            obj = self.store(git)
            self.assertTrue(obj.backup(case)["remote_verified"])
            self.assertFalse(any(r["path"].endswith(".pt") for r in obj._remote_manifest["files"]))
            self.assertEqual(len(obj._remote_manifest["files"]), 2)

    def test_wrong_pending_key_and_extra_plaintext_member_never_publish(self):
        for mode in ("key", "extra"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as base:
                git = FakeGit()
                case, _, _ = self.case(base)
                good = self.store(git)
                good.backup(case)
                remote = good.status()
                if mode == "key":
                    wrong = self.store(git, secret="different-independent-pending-key-over-32-characters")
                    wrong._remote_manifest = good._remote_manifest
                    wrong._pending(case, {}, "fixture", remote)
                else:
                    folder, record, _ = good._pending(case, {}, "fixture", remote)
                    (folder/"plaintext.pt").write_bytes(b"NEVER PUBLISH THIS")
                    record["sha256"]["plaintext.pt"] = storage._sha(b"NEVER PUBLISH THIS")
                    write(folder/"publication.json", record)
                before = self.mutations(git)
                with self.assertRaises(storage.StorageError):
                    good.backup(case)
                self.assertEqual(self.mutations(git), before)

    def test_remote_status_public_hash_cannot_be_rebound_without_authentication(self):
        git = FakeGit()
        with tempfile.TemporaryDirectory() as base:
            case, _, _ = self.case(base)
            obj = self.store(git)
            obj.backup(case)
            status_path = obj.prefix + "/status.json"
            old_sha = git.tree(git.head)[status_path]
            metadata = json.loads(git.blobs[old_sha])
            metadata["public_results_sha256"] = "a" * 64
            raw = storage._canonical(metadata)
            new_sha = storage._git_blob_sha(raw)
            git.blobs[new_sha] = raw
            # Simulates an independently changed Git head that did not know
            # the key for the authenticated index.
            tree = obj._tree_update(git.commits[git.head]["tree"]["sha"], {status_path:new_sha})
            git.head = git.request("POST", obj._root+"/git/commits", {"tree":tree,"parents":[git.head]})["sha"]
            with self.assertRaisesRegex(storage.StorageError, "authenticated_identity"):
                self.store(git).status()

    def test_tree_payloads_bounded_with_large_existing_source(self):
        git = FakeGit()
        # Existing large root is never copied into create-tree request bodies.
        git.trees["0"*40] = {f"old-{i}":{"sha":"f"*40,"mode":"100644","type":"blob"} for i in range(5000)}
        with tempfile.TemporaryDirectory() as base:
            case, _, _ = self.case(base)
            self.store(git).backup(case)
            calls = [body for method,path,body in git.calls if method == "POST" and path.endswith("/git/trees")]
            self.assertTrue(all(len(p["tree"]) <= 128 for p in calls))
            self.assertEqual(len(git.trees[git.commits[git.head]["tree"]["sha"]]), 5001)


class TransportTests(unittest.TestCase):
    def test_classification(self):
        self.assertEqual(storage.classify_http(403,"secondary rate limit",{"Retry-After":"60"})["kind"],"secondary_rate_limit")
        self.assertEqual(storage.classify_http(403,"limit",{"X-RateLimit-Remaining":"0","X-RateLimit-Reset":"1050"},now=1000)["retry_after_seconds"],51)
        self.assertFalse(storage.classify_http(403,"Resource not accessible by token")["retryable"])
        self.assertEqual(storage.classify_http(422,"Reference update failed")["kind"],"ref_conflict")
        self.assertTrue(storage.classify_http(503,"unavailable")["retryable"])

    def test_identical_retry_payload_and_mutation_pacing(self):
        now, calls, bodies, waits = [1000.], [0], [], []
        def sleep(n):
            waits.append(n)
            now[0] += n
        class Response:
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def read(self,n): return b'{"ok":true}'
        def opener(request,timeout):
            bodies.append(request.data)
            calls[0] += 1
            if calls[0] == 1:
                raise urllib.error.HTTPError(request.full_url,403,"Forbidden",{"Retry-After":"3"},io.BytesIO(b'{"message":"secondary rate limit"}'))
            return Response()
        obj = storage.GithubStore(TOKEN,SECRET,"owner/repo","branch",source_commit=SOURCE,prefix="runs/r3",
                                  opener=opener,sleep=sleep,clock=lambda:now[0])
        self.assertEqual(obj._request("POST","/synthetic",{"fixed":"payload"}),{"ok":True})
        obj._request("POST","/synthetic",{"next":"write"})
        self.assertEqual(bodies[0],bodies[1])
        self.assertGreaterEqual(sum(waits),4.25)
        self.assertTrue(all(n <= 30 for n in waits))


if __name__ == "__main__":
    unittest.main()
