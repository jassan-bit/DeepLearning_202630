"""Independent, synthetic-only review of PACK publication and restoration."""
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile

STORE = Path(__file__).parent / "git_checkpoint_store.py"
if not STORE.is_file():
    STORE = Path(__file__).parents[1] / "launcher" / "git_checkpoint_store.py"
spec = importlib.util.spec_from_file_location("qa_pilot_git_store", STORE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def git(*args, cwd=None, data=None):
    env = os.environ.copy()
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1",
               GIT_AUTHOR_NAME="Synthetic QA", GIT_COMMITTER_NAME="Synthetic QA",
               GIT_AUTHOR_EMAIL="qa@example.invalid", GIT_COMMITTER_EMAIL="qa@example.invalid")
    return subprocess.run(["git", *map(str, args)], cwd=cwd, env=env, input=data,
                          check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode().strip()


class GitStoreReview(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="v3-independent-store-")
        self.root = Path(self.temporary.name)
        self.remote = self.root / "remote.git"
        self.work = self.root / "work"
        git("init", "--bare", "--quiet", self.remote)
        git("--git-dir", self.remote, "config", "uploadpack.allowFilter", "true")
        git("--git-dir", self.remote, "config", "uploadpack.allowAnySHA1InWant", "true")
        git("init", "--quiet", self.work)
        (self.work / "source.txt").write_text("Synthetic immutable source. No market data.\n")
        git("add", "source.txt", cwd=self.work)
        git("commit", "--quiet", "-m", "Synthetic source", cwd=self.work)
        self.source = git("rev-parse", "HEAD", cwd=self.work)
        git("remote", "add", "origin", self.remote.as_uri(), cwd=self.work)
        git("push", "--quiet", "origin", "HEAD:refs/heads/main", "HEAD:refs/heads/e3-old-audit-sentinel", cwd=self.work)
        self.protected = self.refs()
        self.run_id = "20261120T010203000001Z"
        self.token = "QA_FAKE_SECRET_NEVER_A_REAL_CREDENTIAL_0123456789"
        self.output = self.root / "output"
        self.output.mkdir()
        (self.output / "plan.json").write_text('{"scope":"synthetic_software_test_only"}\n')
        self.store = self.new_store("cache")

    def tearDown(self):
        self.temporary.cleanup()

    def new_store(self, name):
        return module.GitCheckpointStore(self.root / name, self.source, self.run_id,
                                         self.token, repo_url=self.remote.as_uri(), allow_local=True)

    def refs(self):
        return {line.split()[1]: line.split()[0] for line in
                git("--git-dir", self.remote, "show-ref").splitlines()}

    def assert_protected(self):
        actual = self.refs()
        for ref, sha in self.protected.items():
            self.assertEqual(actual[ref], sha)

    def test_incremental_publish_and_fresh_partial_clone_restore(self):
        first = self.store.publish(self.output)
        self.assertEqual(first["sequence"], 1)
        checkpoint = self.output / "jobs" / "a" / "checkpoint.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"synthetic checkpoint bytes without market arrays")
        second = self.store.publish(self.output)
        self.assertEqual(second["sequence"], 2)
        self.assertNotEqual(first["git_commit"], second["git_commit"])
        fresh = self.new_store("fresh-blob-free-cache")
        fresh._fetch(second["git_commit"])
        manifest_path = fresh.prefix + "/snapshot_manifest.json"
        obj = fresh._git("rev-parse", second["git_commit"] + ":" + manifest_path).decode().strip()
        missing = fresh._git("cat-file", "-e", obj, check=False)
        self.assertNotEqual(missing.returncode, 0, "Fixture must really omit the manifest blob")
        restored = self.root / "restored"
        receipt = fresh.restore(restored)
        self.assertEqual(receipt["git_commit"], second["git_commit"])
        self.assertEqual((restored / "plan.json").read_bytes(), (self.output / "plan.json").read_bytes())
        self.assertEqual((restored / "jobs/a/checkpoint.pt").read_bytes(), checkpoint.read_bytes())
        self.assertEqual(fresh.publish(restored)["sequence"], 2)
        self.assert_protected()

    def test_removed_saved_file_rejected_without_remote_movement(self):
        prior = self.store.publish(self.output)
        (self.output / "plan.json").unlink()
        (self.output / "replacement.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Previously saved"):
            self.store.publish(self.output)
        self.assertEqual(self.store.remote_head(), prior["git_commit"])
        self.assert_protected()

    def test_concurrent_writer_conflict_preserves_every_remote_ref(self):
        competing = self.new_store("competing")
        prior = self.store.publish(self.output)
        before = self.refs()
        with self.assertRaisesRegex(ValueError, "conflict"):
            competing.publish(self.output)
        self.assertEqual(self.refs(), before)
        self.assertEqual(self.store.remote_head(), prior["git_commit"])
        self.assert_protected()

    def test_credentials_in_raw_or_compressed_evidence_are_rejected(self):
        leak = self.output / "leak.log"
        leak.write_text(self.token)
        with self.assertRaisesRegex(ValueError, "Credential"):
            self.store.publish(self.output)
        leak.unlink()
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("checkpoint/data.pkl", self.token * 20)
        compressed = archive.getvalue()
        self.assertNotIn(self.token.encode(), compressed, "Fixture must hide raw token bytes")
        (self.output / "checkpoint.pt").write_bytes(compressed)
        with self.assertRaisesRegex(ValueError, "Credential"):
            self.store.publish(self.output)
        self.assertIsNone(self.store.remote_head())
        self.assert_protected()

    def test_lost_push_acknowledgement_reconciles_exact_commit(self):
        original = self.store._git
        injected = [False]

        def uncertain(*args, **kwargs):
            result = original(*args, **kwargs)
            if "push" in args and not injected[0]:
                injected[0] = True
                raise RuntimeError("Synthetic lost ACK after successful remote mutation")
            return result

        self.store._git = uncertain
        receipt = self.store.publish(self.output)
        self.assertTrue(injected[0])
        self.assertEqual(receipt["git_commit"], self.store.remote_head())
        self.assertFalse(self.store.pending_path.exists())
        self.assertEqual(self.store.publish(self.output)["sequence"], 1)
        self.assert_protected()

    def test_nonempty_restore_target_and_symlinks_rejected(self):
        self.store.publish(self.output)
        fresh = self.new_store("fresh")
        target = self.root / "existing"
        target.mkdir()
        (target / "old.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "empty output"):
            fresh.restore(target)
        self.assertEqual((target / "old.json").read_text(), "{}")
        (self.output / "link.json").symlink_to(self.output / "plan.json")
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.store.publish(self.output)
        self.assert_protected()


if __name__ == "__main__":
    unittest.main(verbosity=2)
