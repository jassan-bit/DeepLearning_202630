"""Exact-byte preparation in temporary checkout copies; no model/data execution."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import prepare_e2_pilot_snapshot as snapshot


def checkout(root,protocol_bytes):
    exact=(ROOT/snapshot.SNAPSHOT_PATH).read_bytes()
    for relative,data in ((snapshot.SNAPSHOT_PATH,exact),(snapshot.PROTOCOL_PATH,protocol_bytes)):
        path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    return root/snapshot.PROTOCOL_PATH


class PilotSnapshotPreparationTests(unittest.TestCase):
    def test_exact_registered_bytes_are_idempotent_without_any_write_or_backup(self):
        exact=(ROOT/snapshot.SNAPSHOT_PATH).read_bytes()
        self.assertEqual(hashlib.sha256(exact).hexdigest(),snapshot.REGISTERED_SHA256)
        self.assertEqual(hashlib.sha256(snapshot._canonical(exact)).hexdigest(),snapshot.CANONICAL_SHA256)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);target=checkout(root,exact);before=target.stat().st_mtime_ns
            for _ in range(2):
                receipt=snapshot.prepare(root)
                self.assertFalse(receipt["modified"])
                self.assertFalse(receipt["backup_created"])
                self.assertEqual(receipt["normalized_sha256"],snapshot.REGISTERED_SHA256)
            self.assertEqual(target.read_bytes(),exact)
            self.assertEqual(target.stat().st_mtime_ns,before)
            self.assertFalse((root/snapshot.BACKUP_PATH).exists())
            self.assertFalse((root/"work").exists())

    def test_known_newline_only_copy_gets_exclusive_backup_and_exact_snapshot(self):
        exact=(ROOT/snapshot.SNAPSHOT_PATH).read_bytes();github=exact[:-1]
        self.assertEqual(hashlib.sha256(github).hexdigest(),snapshot.GITHUB_CHECKOUT_SHA256)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);target=checkout(root,github)
            receipt=snapshot.prepare(root)
            self.assertTrue(receipt["modified"])
            self.assertTrue(receipt["backup_created"])
            backup=root/snapshot.BACKUP_PATH
            self.assertEqual(backup.read_bytes(),github)
            self.assertEqual(target.read_bytes(),exact)
            self.assertEqual(snapshot._canonical(target.read_bytes()),snapshot._canonical(github))
            again=snapshot.prepare(root)
            self.assertFalse(again["modified"])
            self.assertEqual(backup.read_bytes(),github)
            target.write_bytes(github)
            with self.assertRaises(FileExistsError):
                snapshot.prepare(root)
            self.assertEqual(target.read_bytes(),github)
            self.assertEqual(backup.read_bytes(),github)

    def test_semantic_changes_or_unknown_byte_copies_fail_before_any_write(self):
        exact=(ROOT/snapshot.SNAPSHOT_PATH).read_bytes()
        changed=json.loads(exact);changed["models"][0]["model"]="SEMANTIC_CHANGE"
        choices=[json.dumps(changed,sort_keys=True).encode(),exact+b"\n"]
        for content in choices:
            with self.subTest(content_hash=hashlib.sha256(content).hexdigest()),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);target=checkout(root,content)
                before=target.stat().st_mtime_ns
                with self.assertRaises(ValueError):
                    snapshot.prepare(root)
                self.assertEqual(target.read_bytes(),content)
                self.assertEqual(target.stat().st_mtime_ns,before)
                self.assertFalse((root/"work").exists())
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);target=checkout(root,exact[:-1])
            (root/snapshot.SNAPSHOT_PATH).write_bytes(exact+b"\n")
            with self.assertRaisesRegex(ValueError,"Snapshot differs"):
                snapshot.prepare(root)
            self.assertEqual(target.read_bytes(),exact[:-1])
            self.assertFalse((root/"work").exists())


if __name__=="__main__":
    unittest.main()

