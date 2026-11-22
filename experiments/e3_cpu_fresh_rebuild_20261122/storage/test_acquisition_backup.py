"""Offline, synthetic metadata; actual frozen S2 cryptography + fake Git REST."""
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import zipfile

HERE = Path(__file__).resolve().parent
RUNTIME = Path("/workspace/e3_hypothesis_adequacy_20261122/runtime")
sys.path.insert(0, str(RUNTIME))
sys.path.insert(0, str(HERE))
from engine import storage
from acquisition_backup import MetadataJournal, JournalError, SOURCE_COMMIT

spec = importlib.util.spec_from_file_location("s2_storage_test_fixture", RUNTIME / "tests/test_storage.py")
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name) / "work"
        self.work.mkdir()
        self.git = fixture.FakeGit()
        self.git.commits[SOURCE_COMMIT] = self.git.commits[fixture.SOURCE]

    def tearDown(self):
        self.tmp.cleanup()

    def journal(self, work=None, secret=fixture.SECRET):
        work = self.work if work is None else work
        value = MetadataJournal(storage, token=fixture.TOKEN, backup_secret=secret,
            repository="owner/repository", branch="data-acquisition-fixture",
            prefix="private-state/acquisition", work_root=work, journal_dir=work / "private/journal",
            case_id="data-acquisition-fixture", sleep=lambda _: None)
        value.store._request = self.git.request
        return value

    def metadata(self, name="private/inputs/owner.json", raw=b'{"actor":"synthetic-only","consent":"prior-authorization-reconstructed"}\n'):
        path = self.work / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return path

    def test_early_backup_real_s2_crypto_private_bytes_and_exact_restore(self):
        path = self.metadata()
        original = path.read_bytes()
        journal = self.journal()
        receipt = journal.checkpoint([path], event="owner_metadata_before_acquisition")
        self.assertTrue(receipt["remote_verified"])
        plan = json.loads((journal.case / "plan.json").read_bytes())
        self.assertEqual(plan["kind"], "development_acquisition_journal")
        self.assertEqual(plan["jobs"], [])
        self.assertEqual(plan["market_fits"], 0)
        self.assertFalse(plan["test_opened"])
        for raw in self.git.blobs.values():
            self.assertNotIn(original, raw)
            self.assertNotIn(fixture.TOKEN.encode(), raw)
            self.assertNotIn(fixture.SECRET.encode(), raw)
        tree = self.git.tree(self.git.head)
        public = self.git.blobs[tree["private-state/acquisition/public_results.zip"]]
        self.assertEqual(zipfile.ZipFile(io.BytesIO(public)).namelist(), [])
        second_work = Path(self.tmp.name) / "restored"
        restored = self.journal(second_work)
        result = restored.restore()
        self.assertTrue(result["remote_verified"])
        self.assertEqual((second_work / "private/inputs/owner.json").read_bytes(), original)
        self.assertFalse(result["raw_inputs_restored"])
        self.assertNotIn("r3-engine-plan-", result["case_id"])

    def test_after_panel_capsule_cumulative_and_rerun_exact(self):
        owner = self.metadata()
        j = self.journal()
        j.checkpoint([owner], event="owner_metadata_before_acquisition")
        recovery = self.metadata("datasets/F00/recovery_metadata.json", b'{"kind":"synthetic-recovery","source_sha":"' + b"0" * 64 + b'"}')
        j.checkpoint([recovery], event="after_admission_F00")
        self.assertEqual(len(j._read_capsules()), 2)
        j.checkpoint([owner, recovery], event="after_admission_F00")
        second_work = Path(self.tmp.name) / "restored"
        restored = self.journal(second_work)
        restored.restore()
        self.assertEqual((second_work / recovery.relative_to(self.work)).read_bytes(), recovery.read_bytes())
        self.assertEqual((second_work / owner.relative_to(self.work)).read_bytes(), owner.read_bytes())

    def test_restore_conflicts_leave_all_owner_metadata_untouched(self):
        one = self.metadata("private/inputs/first.json", b'{"first":true}')
        two = self.metadata("private/inputs/second.json", b'{"second":true}')
        self.journal().checkpoint([one, two], event="owner_metadata_before_acquisition")
        target = Path(self.tmp.name) / "restored"
        conflict = target / two.relative_to(self.work)
        conflict.parent.mkdir(parents=True)
        conflict.write_bytes(b"conflict")
        with self.assertRaisesRegex(JournalError, "restore_metadata_conflict"):
            self.journal(target).restore()
        self.assertFalse((target / one.relative_to(self.work)).exists())
        self.assertEqual(conflict.read_bytes(), b"conflict")

    def test_raw_arrays_zip_csv_and_secret_payload_refused_before_writes(self):
        j = self.journal()
        for suffix in (".npz", ".npy", ".zip", ".csv", ".pt"):
            with self.assertRaisesRegex(JournalError, "nonmetadata_file_refused"):
                j.checkpoint([self.metadata("private/blob" + suffix, b"synthetic")], event="before_data")
        with self.assertRaisesRegex(storage.StorageError, "credential_material_detected"):
            j.checkpoint([self.metadata("private/credential.json", json.dumps({"token":fixture.TOKEN}).encode())], event="before_data")
        self.assertFalse(any(call[0] != "GET" for call in self.git.calls))

    def test_changed_metadata_and_symlink_and_escape_refused(self):
        path = self.metadata()
        j = self.journal()
        j.checkpoint([path], event="before_data")
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(JournalError, "original_metadata_change"):
            j.checkpoint([path], event="after_data")
        with self.assertRaisesRegex(JournalError, "unsafe_metadata_path"):
            j.checkpoint(["../escape.json"], event="after_data")
        link = self.work / "private/link.json"
        link.symlink_to(path)
        with self.assertRaisesRegex(JournalError, "symlink"):
            j.checkpoint([link], event="after_data")

    def test_wrong_secret_remote_refuses_before_metadata_restore(self):
        path = self.metadata()
        self.journal().checkpoint([path], event="before_data")
        with self.assertRaisesRegex(storage.StorageError, "wrong_secret_or_tampering"):
            self.journal(Path(self.tmp.name)/"restore", secret="another-independent-secret-1234567890").restore()

    def test_storage_allowlist_rejects_array_even_if_manually_inserted(self):
        path = self.metadata()
        j = self.journal()
        j.checkpoint([path], event="before_data")
        manifest = dict(j.store._remote_manifest)
        manifest["files"] = [dict(x) for x in manifest["files"]]
        manifest["files"][0]["path"] = "receipts/raw.npz"
        with self.assertRaisesRegex(storage.StorageError, "backup_manifest_member_invalid"):
            j.store._validate_manifest(manifest, manifest["context"])

    def test_pending_retry_is_completed_before_new_capsules(self):
        first = self.metadata()
        j = self.journal()
        self.git.conflict_on_ref = True
        with self.assertRaises(storage.StorageError):
            j.checkpoint([first], event="before_data")
        self.assertTrue((j.case / ".github_pending").exists())
        self.git.conflict_on_ref = False
        second = self.metadata("datasets/F00/admission.json", b'{"synthetic":true}')
        result = j.checkpoint([second], event="f00_admitted")
        self.assertTrue(result["remote_verified"])
        target = Path(self.tmp.name) / "restored"
        self.journal(target).restore()
        self.assertEqual((target / second.relative_to(self.work)).read_bytes(), second.read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
