"""Synthetic transport fixtures only: never contacts GitHub or trains a model."""
from copy import deepcopy
import base64
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
import io
from urllib.parse import parse_qs, unquote, urlsplit
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_reduced_github_backup as backup


class FakeGitHub(backup.GitHubBackup):
    def __init__(self):
        super().__init__("synthetic-token-for-unit-tests")
        self.head = "a" * 40
        self.branch_exists = False
        self.blobs, self.trees, self.commits = {}, {"b" * 40: {}}, {self.head: {"tree": {"sha": "b" * 40}, "parents": []}}
        self.release = None
        self.remote_assets = {}
        self.asset_bytes = {}
        self.calls = []
        self.counter = 0
        self.conflict_once = False
        self.readonly = False
        self.asset_write_denied = False
        self.bad_asset_digest = False

    def _request(self, method, path, payload=None, *, upload=None, accept=None):
        self.calls.append((method, path))
        if self.readonly and method in {"POST", "PATCH"}:
            raise backup.BackupPause("github_request_failed", status=403)
        if method == "GET" and path == "/git/ref/heads/main":
            return {"object": {"sha": "a" * 40}}
        if path == "/git/ref/heads/" + backup.BRANCH:
            if not self.branch_exists:
                raise backup.BackupPause("github_request_failed", status=404)
            return {"object": {"sha": self.head}}
        if method == "POST" and path == "/git/refs":
            self.branch_exists, self.head = True, payload["sha"]
            return {"object": {"sha": self.head}}
        if method == "GET" and path.startswith("/releases/tags/"):
            if self.release is None:
                raise backup.BackupPause("github_request_failed", status=404)
            return deepcopy(self.release)
        if method == "POST" and path == "/releases":
            self.release = {**payload, "id": 1}
            return deepcopy(self.release)
        if method == "GET" and path.startswith("/releases/1/assets?"):
            return [deepcopy(x) for x in self.remote_assets.values()]
        if method == "POST" and path.startswith("/releases/1/assets?"):
            if self.asset_write_denied:
                raise backup.BackupPause("github_request_failed", status=403)
            name = parse_qs(urlsplit(path).query)["name"][0]
            if name in self.remote_assets:
                raise backup.BackupPause("github_request_failed", status=422)
            data = Path(upload).read_bytes()
            asset = {"id": len(self.remote_assets) + 1, "name": name, "size": len(data),
                "digest": "sha256:" + ("0" * 64 if self.bad_asset_digest else backup.digest_bytes(data)),
                "browser_download_url": "https://github.com/" + backup.REPOSITORY + "/releases/download/" + backup.RELEASE_TAG + "/" + name}
            self.remote_assets[name] = asset
            self.asset_bytes[name] = data
            return deepcopy(asset)
        if method == "GET" and path.startswith("/contents/"):
            parsed = urlsplit(path)
            relative = unquote(parsed.path[len("/contents/"):])
            head = parse_qs(parsed.query)["ref"][0]
            tree = self.trees[self.commits[head]["tree"]["sha"]]
            if relative not in tree:
                raise backup.BackupPause("github_request_failed", status=404)
            sha = tree[relative]
            return {"sha": sha, "content": base64.b64encode(self.blobs[sha]).decode()}
        if method == "GET" and path.startswith("/git/commits/"):
            return deepcopy(self.commits[path.rsplit("/", 1)[1]])
        if method == "POST" and path == "/git/blobs":
            data = base64.b64decode(payload["content"])
            sha = backup.git_blob_sha(data)
            self.blobs[sha] = data
            return {"sha": sha}
        if method == "POST" and path == "/git/trees":
            tree = dict(self.trees[payload["base_tree"]])
            tree.update({row["path"]: row["sha"] for row in payload["tree"]})
            sha = hashlib.sha1(backup.canonical(tree)).hexdigest()
            self.trees[sha] = tree
            return {"sha": sha}
        if method == "POST" and path == "/git/commits":
            self.counter += 1
            sha = "%040x" % self.counter
            self.commits[sha] = {"tree": {"sha": payload["tree"]}, "parents": payload["parents"]}
            return {"sha": sha}
        if method == "PATCH" and path == "/git/refs/heads/" + backup.BRANCH:
            if payload.get("force") is not False:
                raise AssertionError("A backup must never force a ref")
            if self.conflict_once:
                self.conflict_once = False
                data = b"another independent writer\n"
                blob = backup.git_blob_sha(data)
                self.blobs[blob] = data
                previous = self.head
                tree = dict(self.trees[self.commits[previous]["tree"]["sha"]])
                tree["runs/other-writer/result.txt"] = blob
                tree_sha = hashlib.sha1(backup.canonical(tree)).hexdigest()
                self.trees[tree_sha] = tree
                self.counter += 1
                self.head = "%040x" % self.counter
                self.commits[self.head] = {"tree": {"sha": tree_sha}, "parents": [previous]}
                raise backup.BackupPause("github_request_failed", status=422)
            if self.commits[payload["sha"]]["parents"] != [self.head]:
                raise backup.BackupPause("github_request_failed", status=422)
            self.head = payload["sha"]
            return {"object": {"sha": self.head}}
        raise AssertionError("Unexpected synthetic request %s %s" % (method, path))


def fixture(directory, *, suffix=".pt"):
    root = Path(directory)
    (root / "curves.csv").write_text("epoch,train_loss\n1,0.2\n")
    (root / ("weights" + suffix)).write_bytes(b"SYNTHETIC TRANSPORT FIXTURE ONLY; not scientific weights")
    return {"attempt_id": "synthetic-fit-001", "status": "completed", "synthetic_fixture": True,
        "artifacts_sha256": {p.name: backup.digest_file(p) for p in sorted(root.iterdir()) if p.is_file()}}


class GitHubBackupTests(unittest.TestCase):
    def test_preflight_proves_branch_and_new_asset_write_without_training(self):
        client = FakeGitHub()
        report = client.preflight()
        self.assertEqual(report["status"], "backup_write_access_verified")
        self.assertFalse(report["scientific_training_started"])
        self.assertTrue(client.branch_exists)
        self.assertEqual(len(client.remote_assets), 1)
        self.assertTrue(client._checked)

    def test_readonly_or_asset_write_failure_blocks_admission(self):
        for setting in ("readonly", "asset_write_denied"):
            client = FakeGitHub()
            setattr(client, setting, True)
            with self.assertRaises(backup.BackupPause) as caught:
                client.preflight()
            self.assertFalse(client._checked)
            self.assertFalse(caught.exception.retry_training)

    def test_completed_artifacts_and_weights_are_verified_and_idempotent(self):
        client = FakeGitHub()
        with tempfile.TemporaryDirectory() as folder:
            receipt = fixture(folder)
            first = client.persist_result(folder, receipt)
            calls = len(client.calls)
            second = client.persist_result(folder, receipt)
            self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
            self.assertTrue(second["existing_identical_result"])
            self.assertFalse(second["retry_scientific_fit"])
            self.assertFalse(any(method in {"POST", "PATCH"} for method, _ in client.calls[calls:]))
            self.assertEqual(len(client.remote_assets), 2)  # one transport probe + one weight
            self.assertTrue((Path(folder) / "github_backup_receipt.json").exists())

    def test_supervisor_wrapper_and_ARIMA_pickle_use_same_transport(self):
        client = FakeGitHub()
        with tempfile.TemporaryDirectory() as folder:
            receipt = {"result": fixture(folder, suffix=".pkl"), "elapsed_seconds": 0.1}
            result = client.persist_result(folder, receipt)
            self.assertEqual(result["artifact_count"], 2)
            self.assertTrue(any(x.endswith(".pkl") for x in client.remote_assets))

    def test_ref_conflict_merges_without_losing_other_writer(self):
        client = FakeGitHub()
        client.preflight()
        client.conflict_once = True
        with tempfile.TemporaryDirectory() as folder:
            client.persist_result(folder, fixture(folder))
            paths = client.trees[client.commits[client.head]["tree"]["sha"]]
            self.assertIn("runs/other-writer/result.txt", paths)
            self.assertIn("runs/synthetic-fit-001/manifest.json", paths)

    def test_many_metadata_files_use_one_deterministic_lossless_archive(self):
        client = FakeGitHub()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            receipt = fixture(folder)
            for index in range(backup.METADATA_ARCHIVE_THRESHOLD + 1):
                path = root / ("dated-event-%03d.json" % index)
                path.write_text(json.dumps({"synthetic_fixture": True, "index": index}) + "\n")
                receipt["artifacts_sha256"][path.name] = backup.digest_file(path)
            first = client.persist_result(folder, receipt)
            current = client.trees[client.commits[client.head]["tree"]["sha"]]
            run_files = [path for path in current if path.startswith("runs/synthetic-fit-001/")]
            self.assertEqual(len(run_files), 2)  # receipt and manifest, not34 blobs
            manifest_blob = current[first["manifest_path"]]
            manifest = json.loads(client.blobs[manifest_blob])
            transport = manifest["archive_transport"]
            raw = client.asset_bytes[transport["name"]]
            self.assertEqual(backup.digest_bytes(raw), transport["sha256"])
            metadata = [r for r in manifest["artifacts"] if r["storage"] == "release_archive_member"]
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                self.assertEqual(len(archive.namelist()), len(metadata))
                for row in metadata:
                    self.assertEqual(backup.digest_bytes(archive.read(row["archive_member"])), row["sha256"])
                    self.assertEqual(archive.read(row["archive_member"]), (root / row["path"]).read_bytes())
            original_count = len(client.remote_assets)
            second = client.persist_result(folder, receipt)
            self.assertTrue(second["existing_identical_result"])
            self.assertEqual(len(client.remote_assets), original_count)
            self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])

    def test_mutation_and_raw_market_data_are_rejected_without_upload(self):
        for content in (b"open,high,low,close,volume\n1,2,1,2,10\n", b"synthetic-token-for-unit-tests"):
            client = FakeGitHub()
            with tempfile.TemporaryDirectory() as folder:
                receipt = fixture(folder)
                target = Path(folder) / "curves.csv"
                target.write_bytes(content)
                receipt["artifacts_sha256"]["curves.csv"] = backup.digest_file(target)
                with self.assertRaises(backup.BackupPause):
                    client.persist_result(folder, receipt)
                self.assertFalse(client.calls)
                self.assertTrue((Path(folder) / "github_backup_pending.json").exists())

    def test_local_hash_failure_preserves_fit_and_pause_marker(self):
        client = FakeGitHub()
        with tempfile.TemporaryDirectory() as folder:
            receipt = fixture(folder)
            weights = Path(folder) / "weights.pt"
            weights.write_bytes(b"tampered after receipt")
            with self.assertRaises(backup.BackupPause):
                client.persist_result(folder, receipt)
            self.assertEqual(weights.read_bytes(), b"tampered after receipt")
            self.assertFalse(client.calls)

    def test_different_same_attempt_receipt_never_overwrites_existing_result(self):
        client = FakeGitHub()
        with tempfile.TemporaryDirectory() as folder:
            receipt = fixture(folder)
            client.persist_result(folder, receipt)
            changed = {**receipt, "altered_metadata": True}
            with self.assertRaisesRegex(backup.BackupPause, "different_content"):
                client.persist_result(folder, changed)

    def test_bad_release_digest_pauses_without_retraining(self):
        client = FakeGitHub()
        client.bad_asset_digest = True
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(backup.BackupPause) as caught:
                client.persist_result(folder, fixture(folder))
            self.assertFalse(caught.exception.retry_training)
            self.assertTrue(caught.exception.fit_result_preserved)
            self.assertTrue((Path(folder) / "weights.pt").exists())

    def test_auth_is_only_in_header_and_public_redirects_reject_other_hosts(self):
        client = backup.GitHubBackup("synthetic-token-for-unit-tests")
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = b"{}"
        client._opener = Mock()
        client._opener.open.return_value = response
        client._request("GET", "/git/ref/heads/main")
        request = client._opener.open.call_args.args[0]
        self.assertNotIn(client._token, request.full_url)
        self.assertEqual(request.get_header("Authorization"), "Bearer " + client._token)
        self.assertNotIn(client._token, repr(client))
        for url in ("http://github.com/file", "https://evil.example/asset", "https://user:pass@github.com/file"):
            with self.assertRaises(backup.BackupPause):
                backup._public_url(url)


if __name__ == "__main__":
    unittest.main()
