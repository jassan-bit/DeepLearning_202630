"""Integrity and isolation checks for the private E1 evidence handoff."""
import hashlib
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from restore_e1_private_artifacts import check_destination, digest, relative_file, verify_archive


class PrivateArtifactRestorationTests(unittest.TestCase):
    def archive(self, root, records):
        path = root / "evidence.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            for name, payload, kind in records:
                item = tarfile.TarInfo(name)
                if kind == "file":
                    item.size = len(payload)
                    archive.addfile(item, io.BytesIO(payload))
                else:
                    item.type = tarfile.SYMTYPE
                    item.linkname = payload.decode()
                    archive.addfile(item)
        return path

    def test_original_directory_and_existing_results_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            with self.assertRaises(ValueError):
                check_destination(source, source)
            with self.assertRaises(ValueError):
                check_destination(source, source / "replica")
            destination = root / "replica"
            destination.mkdir()
            (destination / "raw_access_started.json").write_text("existing evidence")
            with self.assertRaises(ValueError):
                check_destination(source, destination)
            self.assertEqual((destination / "raw_access_started.json").read_text(), "existing evidence")

    def test_empty_external_destination_is_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            destination = root / "replica"
            destination.mkdir()
            self.assertEqual(check_destination(source, destination), (source.resolve(), destination.resolve()))

    def test_private_bundle_cannot_target_an_unrelated_git_checkout_or_public_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "private-package"
            source.mkdir()
            checkout = root / "checkout"
            checkout.mkdir()
            (checkout / ".git").mkdir()
            with self.assertRaisesRegex(ValueError, "Git checkout"):
                check_destination(source, checkout / "replica")
            for name in ("_static", "_build", "gh-pages"):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "publication/build"):
                    check_destination(source, root / name / "replica")

    def test_paths_reject_traversal_absolute_windows_and_noncanonical_names(self):
        for value in ("../out", "/outside", "runs/../../out", "runs\\out", "C:/outside", "runs//file", "runs/./file", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                relative_file(value)
        self.assertEqual(relative_file("runs/model/checkpoint.pt"), Path("runs/model/checkpoint.pt"))

    def test_verified_regular_file_is_restored_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = b"frozen checkpoint bytes, not an executable model call"
            path = self.archive(root, [("runs/model/model.pt", payload, "file")])
            wanted = {"runs/model/model.pt": hashlib.sha256(payload).hexdigest()}
            destination = root / "restored"
            self.assertEqual(verify_archive(path, digest(path), wanted, destination=destination), 1)
            self.assertEqual((destination / "runs/model/model.pt").read_bytes(), payload)

    def test_archive_checksum_failure_precedes_any_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.archive(root, [("runs/a", b"data", "file")])
            destination = root / "restored"
            with self.assertRaisesRegex(ValueError, "Archive SHA"):
                verify_archive(path, "0"*64, {"runs/a": hashlib.sha256(b"data").hexdigest()}, destination=destination)
            self.assertFalse(destination.exists())

    def test_member_checksum_failure_precedes_any_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.archive(root, [("runs/a", b"changed", "file")])
            destination = root / "restored"
            with self.assertRaisesRegex(ValueError, "member SHA"):
                verify_archive(path, digest(path), {"runs/a": hashlib.sha256(b"original").hexdigest()}, destination=destination)
            self.assertFalse(destination.exists())

    def test_symbolic_links_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.archive(root, [("runs/a", b"/outside", "symlink")])
            with self.assertRaisesRegex(ValueError, "non-regular"):
                verify_archive(path, digest(path), {"runs/a": hashlib.sha256(b"").hexdigest()})

    def test_duplicate_unexpected_or_missing_members_are_rejected(self):
        for scenario in ("duplicate", "unexpected", "missing"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                entries = [("runs/a", b"a", "file")]
                wanted = {"runs/a": hashlib.sha256(b"a").hexdigest()}
                if scenario == "duplicate": entries.append(entries[0])
                if scenario == "unexpected": entries.append(("runs/b", b"b", "file"))
                if scenario == "missing": wanted["runs/b"] = hashlib.sha256(b"b").hexdigest()
                path = self.archive(root, entries)
                with self.assertRaises(ValueError):
                    verify_archive(path, digest(path), wanted)

    def test_an_existing_restored_file_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.archive(root, [("runs/a", b"a", "file")])
            destination = root / "restored"
            (destination / "runs").mkdir(parents=True)
            (destination / "runs/a").write_bytes(b"previous evidence")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                verify_archive(path, digest(path), {"runs/a": hashlib.sha256(b"a").hexdigest()}, destination=destination)
            self.assertEqual((destination / "runs/a").read_bytes(), b"previous evidence")


if __name__ == "__main__":
    unittest.main()
