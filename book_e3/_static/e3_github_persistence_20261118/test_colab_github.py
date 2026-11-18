"""Storage contracts with a fake GitHub API; no network, GPU or credentials."""
from __future__ import annotations

import base64
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock
import urllib.error
import warnings
import zipfile

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("colab_github_under_test", HERE / "colab_github.py")
storage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(storage)
SOURCE, BASE_TREE, NEW_TREE, COMMIT = (letter * 40 for letter in "abcd")
RUN_ID = "20261118T150022983069Z"
TOKEN = "test_secret_never_persist_this"
BRANCH = "e3-colab-receipts-" + RUN_ID


class FakeGitHub:
    def __init__(self, *, bad_blob=False, drop_old=False, lost_ack=False):
        self.calls, self.blobs, self.ref = [], {}, None
        self.bad_blob, self.drop_old, self.lost_ack = bad_blob, drop_old, lost_ack
        self.old = [{"path": "README.md", "type": "blob", "mode": "100644",
                     "sha": storage._git_sha(b"existing source"), "size": 15}]
        self.tree, self.commit = [], None

    def request(self, method, path, token, payload=None, *, allow_missing=False):
        assert token == TOKEN
        self.calls.append((method, path, deepcopy(payload)))
        if method == "GET" and path == "/user":
            return {"login": "jassan-bit"}
        if method == "GET" and path == "/repos/" + storage.REPOSITORY:
            return {"full_name": storage.REPOSITORY, "permissions": {"push": True}}
        if method == "GET" and path.endswith("/git/commits/" + SOURCE):
            return {"sha": SOURCE, "tree": {"sha": BASE_TREE}}
        if method == "GET" and path.endswith("/git/commits/" + COMMIT):
            return deepcopy(self.commit)
        if method == "GET" and path.endswith("/git/trees/" + BASE_TREE + "?recursive=1"):
            return {"sha": BASE_TREE, "tree": deepcopy(self.old), "truncated": False}
        if method == "GET" and path.endswith("/git/trees/" + NEW_TREE + "?recursive=1"):
            return {"sha": NEW_TREE, "tree": deepcopy(self.tree), "truncated": False}
        if method == "GET" and path.endswith("/git/ref/heads/" + BRANCH):
            if self.ref is None and not allow_missing:
                raise AssertionError("Missing required ref")
            return deepcopy(self.ref)
        if method == "POST" and path.endswith("/git/blobs"):
            assert payload["encoding"] == "base64"
            raw = base64.b64decode(payload["content"], validate=True)
            sha = storage._git_sha(raw)
            self.blobs[sha] = raw
            return {"sha": "e" * 40 if self.bad_blob else sha}
        if method == "POST" and path.endswith("/git/trees"):
            assert payload["base_tree"] == BASE_TREE
            self.tree = ([] if self.drop_old else deepcopy(self.old)) + deepcopy(payload["tree"])
            return {"sha": NEW_TREE}
        if method == "POST" and path.endswith("/git/commits"):
            self.commit = {"sha": COMMIT, "tree": {"sha": payload["tree"]},
                           "parents": [{"sha": p} for p in payload["parents"]]}
            return deepcopy(self.commit)
        if method == "POST" and path.endswith("/git/refs"):
            assert payload == {"ref": "refs/heads/" + BRANCH, "sha": COMMIT}
            assert self.ref is None
            self.ref = {"ref": payload["ref"], "object": {"type": "commit", "sha": COMMIT}}
            if self.lost_ack:
                raise storage.StorageError("github_network_error_retry_same_snapshot")
            return deepcopy(self.ref)
        raise AssertionError("Unexpected API request: " + method + " " + path)


def fixture(root):
    output = root / "output"
    output.mkdir()
    receipt = {"repository_HEAD": SOURCE, "status": "passed_synthetic_only",
               "GPU_execution_verified": True, "new_real_data_fits": 0,
               "new_market_predictions": 0, "new_hypothesis_tests": 0}
    (output / "preflight_receipt.json").write_text(json.dumps(receipt) + "\n")
    (output / "models").mkdir()
    (output / "models/worker.log").write_text("Synthetic profile passed.\n")
    (output / "models/synthetic_state.pt").write_bytes(b"not selected")
    (output / "source_package").mkdir()
    (output / "source_package/private.json").write_text("not selected")
    archive = root / "receipts.zip"
    make_zip(output, archive)
    return output, archive


def make_zip(output, archive, *, replacements=None, extra=None):
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob("*")):
            relative = path.relative_to(output).as_posix()
            if path.is_file() and "source_package" not in path.parts and path.suffix in (".json", ".log"):
                name, raw = (replacements or {}).get(relative, (relative, path.read_bytes()))
                bundle.writestr(name, raw)
        if extra:
            for name, raw in extra:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    bundle.writestr(name, raw)


class GitHubPersistenceContracts(unittest.TestCase):
    def save(self, output, archive, api):
        with mock.patch.object(storage, "_request", side_effect=api.request):
            return storage.save_run(output, archive, TOKEN, SOURCE, RUN_ID)

    def test_atomic_results_branch_preserves_source_and_exact_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, archive = fixture(Path(temporary))
            api = FakeGitHub()
            result = self.save(output, archive, api)
            self.assertTrue(result["saved"])
            self.assertFalse(result["main_updated"])
            self.assertFalse(result["already_saved"])
            self.assertEqual(result["branch"], BRANCH)
            self.assertEqual(result["persisted_files"], 4)
            self.assertTrue(result["run_url"].endswith("runs/colab_preflight/" + RUN_ID))
            refs = [i for i, call in enumerate(api.calls) if call[0] == "POST" and call[1].endswith("/git/refs")]
            self.assertEqual(len(refs), 1)
            self.assertTrue(all(call[0] != "PATCH" for call in api.calls))
            self.assertTrue(any(i < refs[0] and call[0] == "GET" and "/git/commits/" + COMMIT in call[1]
                                for i, call in enumerate(api.calls)))
            self.assertNotIn(TOKEN, json.dumps([payload for _, _, payload in api.calls]))
            manifest = next(raw for raw in api.blobs.values() if b'"schema_version"' in raw)
            decoded = json.loads(manifest)
            self.assertEqual(result["manifest_sha256"], storage._sha(manifest))
            self.assertEqual(decoded["files"]["receipts.zip"]["sha256"], storage._sha(archive.read_bytes()))
            self.assertIn((output / "models/worker.log").read_bytes(), api.blobs.values())
            self.assertNotIn(b"not selected", api.blobs.values())

    def test_identical_retry_and_lost_ack_do_not_require_another_gpu_run(self):
        for lost_ack in (False, True):
            with self.subTest(lost_ack=lost_ack), tempfile.TemporaryDirectory() as temporary:
                output, archive = fixture(Path(temporary))
                api = FakeGitHub(lost_ack=lost_ack)
                first = self.save(output, archive, api)
                boundary = len(api.calls)
                second = self.save(output, archive, api)
                self.assertTrue(second["already_saved"])
                self.assertEqual(first["commit_sha"], second["commit_sha"])
                self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
                self.assertFalse(any(method != "GET" for method, _, _ in api.calls[boundary:]))

    def test_changed_existing_run_is_refused_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, archive = fixture(Path(temporary))
            api = FakeGitHub()
            self.save(output, archive, api)
            before = deepcopy(api.ref)
            (output / "models/worker.log").write_text("Different saved result.\n")
            make_zip(output, archive)
            boundary = len(api.calls)
            with self.assertRaises(storage.StorageConflict):
                self.save(output, archive, api)
            self.assertEqual(api.ref, before)
            self.assertFalse(any(method != "GET" for method, _, _ in api.calls[boundary:]))

    def test_zip_cannot_hide_extra_or_different_receipts_or_credentials(self):
        linked = zipfile.ZipInfo("models/worker.log")
        linked.create_system = 3
        linked.external_attr = (stat.S_IFLNK | 0o777) << 16
        variants = (
            ({"models/worker.log": (linked, b"Synthetic profile passed.\n")}, None),
            ({"models/worker.log": ("../worker.log", b"Synthetic profile passed.\n")}, None),
            ({"models/worker.log": ("models/worker.log", b"changed")}, None),
            (None, [("secret.json", TOKEN.encode())]),
            (None, [("source_package/private.json", b"hidden")]),
            (None, [("checkpoint.pt", b"pickle")]),
            (None, [("models/worker.log", b"duplicate")]),
        )
        for replacements, extra in variants:
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as temporary:
                output, archive = fixture(Path(temporary))
                make_zip(output, archive, replacements=replacements, extra=extra)
                api = FakeGitHub()
                with self.assertRaises(storage.StorageError):
                    self.save(output, archive, api)
                self.assertEqual(api.calls, [])

    def test_symlinks_and_literal_secret_fail_before_any_api_request(self):
        for kind in ("symlink", "secret"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                output, archive = fixture(root)
                if kind == "symlink":
                    outside = root / "outside.json"
                    outside.write_text("{}")
                    (output / "linked.json").symlink_to(outside)
                else:
                    (output / "models/worker.log").write_text(TOKEN)
                    make_zip(output, archive)
                api = FakeGitHub()
                with self.assertRaises(storage.StorageError) as error:
                    self.save(output, archive, api)
                self.assertNotIn(TOKEN, str(error.exception))
                self.assertEqual(api.calls, [])

    def test_blob_or_tree_integrity_failure_never_creates_a_branch(self):
        for options in ({"bad_blob": True}, {"drop_old": True}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as temporary:
                output, archive = fixture(Path(temporary))
                api = FakeGitHub(**options)
                with self.assertRaises(storage.StorageError):
                    self.save(output, archive, api)
                self.assertIsNone(api.ref)
                self.assertFalse(any(method == "POST" and path.endswith("/git/refs")
                                     for method, path, _ in api.calls))

    def test_read_only_role_check_does_not_claim_token_write_scope(self):
        api = FakeGitHub()
        with mock.patch.object(storage, "_request", side_effect=api.request):
            access = storage.validate_credentials(TOKEN)
        self.assertTrue(access["authenticated"])
        self.assertFalse(access["token_contents_write_verified"])
        self.assertTrue(all(method == "GET" for method, _, _ in api.calls))
        error = urllib.error.HTTPError("https://api.github.com/user", 403, TOKEN, {}, io.BytesIO(TOKEN.encode()))
        with mock.patch.object(storage.urllib.request, "build_opener") as opener:
            opener.return_value.open.side_effect = error
            with self.assertRaises(storage.StorageError) as caught:
                storage.validate_credentials(TOKEN)
        self.assertNotIn(TOKEN, str(caught.exception))

    def test_size_limits_fail_closed_without_uploading_a_partial_result(self):
        for constant in ("MAX_TEXT_BYTES", "MAX_ZIP_BYTES", "MAX_TOTAL_BYTES"):
            with self.subTest(constant=constant), tempfile.TemporaryDirectory() as temporary:
                output, archive = fixture(Path(temporary))
                api = FakeGitHub()
                with mock.patch.object(storage, constant, 1), self.assertRaises(storage.StorageError):
                    self.save(output, archive, api)
                self.assertEqual(api.calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
