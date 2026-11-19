"""Regressions for frozen v1 verification after current documentation evolves."""
import io
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest

import verify_historical_bundle as check


class HistoricalBundle(unittest.TestCase):
    def fixture(self, root):
        destination = Path(root) / check.FOLDER
        destination.mkdir(parents=True)
        for name in (check.ARCHIVE_NAME, check.MANIFEST_NAME):
            shutil.copyfile(check.ROOT / check.FOLDER / name, destination / name)
        return destination

    def test_published_bytes_and_deterministic_repack(self):
        receipt = check.verify()
        self.assertEqual(receipt["files"], 31)
        self.assertEqual(receipt["bytes"], 70337)
        self.assertTrue(receipt["deterministic_repack_identical"])

    def test_current_cards_and_generators_are_not_snapshot_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = self.fixture(folder)
            for relative in ("MODEL_CARD.md", "DATA_STATEMENT.md"):
                (Path(folder) / relative).write_text("Current, legitimate new version.\n")
            # No source/generator tree is needed: only the published two artifacts.
            receipt = check.verify(folder)
            self.assertFalse(receipt["current_source_files_consulted"])
            (Path(folder) / "DATA_STATEMENT.md").write_text("Later new version.\n")
            self.assertEqual(receipt, check.verify(folder))

    def test_corruption_or_rewritten_manifest_rejected(self):
        for changed in (check.ARCHIVE_NAME, check.MANIFEST_NAME):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as folder:
                destination = self.fixture(folder)
                path = destination / changed
                raw = bytearray(path.read_bytes()); raw[-1] ^= 1; path.write_bytes(raw)
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    check.verify(folder)

    def test_unsafe_tar_paths_and_links_rejected_before_extraction(self):
        for name, kind in ((check.PREFIX + "/../escape", tarfile.REGTYPE),
                           ("/tmp/escape", tarfile.REGTYPE),
                           (check.PREFIX + "/a//b", tarfile.REGTYPE),
                           (check.PREFIX + "/link", tarfile.SYMTYPE),
                           (check.PREFIX + "/hard", tarfile.LNKTYPE)):
            with self.subTest(name=name, kind=kind):
                buffer = io.BytesIO()
                with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
                    member = tarfile.TarInfo(name)
                    member.type = kind; member.linkname = "../../outside"
                    archive.addfile(member)
                with self.assertRaisesRegex(ValueError, "unsafe|non-regular"):
                    check.member_payloads(buffer.getvalue())

    def test_duplicate_members_rejected(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for _ in range(2):
                archive.addfile(tarfile.TarInfo(check.PREFIX + "/same"))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            check.member_payloads(buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
