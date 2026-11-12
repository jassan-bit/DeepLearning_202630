"""Local bare-Git tests; no GitHub requests, credentials or training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "frozen")]
from e2_optimized_recovery_git_v1 import (
    BRANCH, REPOSITORY, MAX_RECEIPT_BYTES, RecoveryGitTransport,
)
from e2_github_campaign import CampaignSyncError


class RecoveryGitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.remote = Path(self.tmp.name) / "remote.git"
        subprocess.run(["git", "init", "--bare", "--quiet", str(self.remote)], check=True)
        self.env = dict(os.environ, GIT_AUTHOR_NAME="Recovery test", GIT_AUTHOR_EMAIL="test@example.invalid",
                        GIT_COMMITTER_NAME="Recovery test", GIT_COMMITTER_EMAIL="test@example.invalid")
        tree = self.git("mktree", data=b"").strip().decode()
        self.main = self.git("commit-tree", tree, data=b"Unrelated main branch\n").strip().decode()
        self.git("update-ref", "refs/heads/main", self.main)

    def git(self, *args, data=None):
        return subprocess.run(["git", "--git-dir=" + str(self.remote), *args],
            input=data, check=True, capture_output=True, env=self.env).stdout

    def client(self):
        obj = RecoveryGitTransport(token="opaque-synthetic-test-secret",
            _test_remote=self.remote, _allow_local_for_tests=True)
        self.addCleanup(obj.close)
        return obj

    @staticmethod
    def payload(owner):
        return (json.dumps({"owner_nonce": owner, "phase": "started"}, sort_keys=True) + "\n").encode()

    def test_create_is_immutable_and_does_not_touch_main(self):
        c = self.client()
        authorization = b'{"authorization": "registered"}\n'
        self.assertIsNone(c.read("authorization.json"))
        self.assertEqual(c.create("authorization.json", authorization), authorization)
        first = c.head
        self.assertEqual(c.create("started.json", self.payload("owner-a")), self.payload("owner-a"))
        self.assertNotEqual(c.head, first)
        self.assertEqual(c.create("authorization.json", b'{"authorization":"changed"}\n'), authorization)
        self.assertEqual(c.read("authorization.json"), authorization)
        self.assertEqual(self.git("rev-parse", "refs/heads/main").strip().decode(), self.main)
        paths = self.git("ls-tree", "-r", "--name-only", "refs/heads/" + BRANCH).decode().splitlines()
        self.assertEqual(paths, [BRANCH + "/authorization.json", BRANCH + "/started.json"])
        self.assertEqual(self.git("show", "refs/heads/" + BRANCH + ":" + BRANCH + "/authorization.json"), authorization)

    def test_racing_claims_return_one_remote_owner_without_overwrite(self):
        seed = self.client()
        seed.create("authorization.json", b'{"authorization":"registered"}\n')
        clients = [self.client(), self.client()]
        barrier = threading.Barrier(2)
        for c in clients:
            original = c._git._push
            def concurrent_push(sha, *, create, original=original):
                barrier.wait(timeout=10)
                return original(sha, create=create)
            c._git._push = concurrent_push
        offers = [self.payload("owner-a"), self.payload("owner-b")]
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda pair: pair[0].create("started.json", pair[1]), zip(clients, offers)))
        actual = seed.read("started.json")
        self.assertIn(actual, offers)
        self.assertEqual(results, [actual, actual])
        history = self.git("rev-list", "--count", "refs/heads/" + BRANCH).strip()
        self.assertEqual(history, b"2")
        self.assertEqual(self.git("rev-parse", "refs/heads/main").strip().decode(), self.main)

    def test_racing_initial_authorizations_have_one_root(self):
        clients = [self.client(), self.client()]
        barrier = threading.Barrier(2)
        for c in clients:
            original = c._git._push
            def concurrent_push(sha, *, create, original=original):
                barrier.wait(timeout=10)
                return original(sha, create=create)
            c._git._push = concurrent_push
        offers = [b'{"authorization":"a"}\n', b'{"authorization":"b"}\n']
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda pair: pair[0].create("authorization.json", pair[1]), zip(clients, offers)))
        actual = self.client().read("authorization.json")
        self.assertIn(actual, offers)
        self.assertEqual(results, [actual, actual])
        self.assertEqual(self.git("rev-list", "--count", "refs/heads/" + BRANCH).strip(), b"1")

    def test_paths_branches_and_repository_are_fixed(self):
        c = self.client()
        for path in ["../main", "status.json", "main", "", BRANCH + "/started.json", "started.json\x00extra"]:
            with self.subTest(path=path):
                with self.assertRaises(CampaignSyncError):
                    c.create(path, b"{}\n")
                with self.assertRaises(CampaignSyncError):
                    c.read(path)
        for args in [(REPOSITORY, "main"), ("other/project", BRANCH)]:
            with self.assertRaises(CampaignSyncError):
                RecoveryGitTransport(*args, token="opaque-synthetic-test-secret")
        c.branch = "main"
        with self.assertRaises(CampaignSyncError):
            c.read("started.json")

    def test_nonfinite_duplicate_keys_invalid_utf8_and_credentials_are_refused(self):
        c = self.client()
        values = [b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e309}', b'{"x":1,"x":2}',
                  b"[]", b"\xff", b"", b"{}" + b" " * MAX_RECEIPT_BYTES,
                  b'{"secret":"opaque-synthetic-test-secret"}']
        for payload in values:
            with self.subTest(payload=payload[:40]):
                with self.assertRaises(CampaignSyncError):
                    c.create("authorization.json", payload)
        self.assertIsNone(c.read("authorization.json"))

    def test_unconfirmed_ack_does_not_cause_second_write(self):
        c = self.client()
        original = c._git._push
        calls = []
        def lose_ack(sha, *, create):
            calls.append(sha)
            original(sha, create=create)
            raise CampaignSyncError("simulated_lost_ack")
        c._git._push = lose_ack
        self.assertEqual(c.create("authorization.json", b'{"authorization":"a"}\n'), b'{"authorization":"a"}\n')
        c._git._push = original
        self.assertEqual(c.create("authorization.json", b'{"authorization":"different"}\n'), b'{"authorization":"a"}\n')
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.git("rev-list", "--count", "refs/heads/" + BRANCH).strip(), b"1")

    def test_foreign_content_on_recovery_branch_is_refused(self):
        self.git("update-ref", "refs/heads/" + BRANCH, self.main)
        # Place an unexpected path on the dedicated branch using test plumbing.
        blob = self.git("hash-object", "-w", "--stdin", data=b"unrelated\n").strip().decode()
        tree = self.git("mktree", data=("100644 blob " + blob + "\tREADME.md\n").encode()).strip().decode()
        commit = self.git("commit-tree", tree, "-p", self.main, data=b"Foreign tree\n").strip().decode()
        self.git("update-ref", "refs/heads/" + BRANCH, commit)
        c = self.client()
        with self.assertRaises(CampaignSyncError):
            c.create("authorization.json", b"{}\n")
        self.assertEqual(self.git("rev-parse", "refs/heads/" + BRANCH).strip().decode(), commit)


if __name__ == "__main__":
    unittest.main()
