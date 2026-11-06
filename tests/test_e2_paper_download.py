"""Offline acquisition tests: integrity, cache reuse and archive containment."""
import copy
from hashlib import sha256
import importlib.util
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/download_e2_paper_sources.py"
spec = importlib.util.spec_from_file_location("e2_paper_download", SCRIPT)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def make_zip(entries):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in entries:
            archive.writestr(name, value)
    return stream.getvalue()


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="e2-source-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = b"date,OT\n2000-01-01,1\n"
        self.assets = {}
        self.design = {"dataset": {"url": "https://raw.githubusercontent.com/example/data/" + "a" * 40 + "/ETTh1.csv",
                                  "filename": "ETTh1.csv", "sha256": sha256(self.data).hexdigest()},
                       "sources": {}}
        self.assets[self.design["dataset"]["url"]] = self.data
        for role, letter in [("TimeMixer", "b"), ("controls", "c")]:
            directory = role + "-" + letter * 40
            code = (role + " source\n").encode()
            payload = make_zip([(directory + "/model.py", code), (directory + "/LICENSE", b"license\n")])
            url = "https://codeload.github.com/example/" + role + "/zip/" + letter * 40
            self.assets[url] = payload
            self.design["sources"][role] = {"repository": "example/" + role, "commit": letter * 40,
                "archive_url": url, "archive_filename": role + ".zip",
                "archive_sha256": sha256(payload).hexdigest(), "extract_directory": directory,
                "files": [{"path": "model.py", "sha256": sha256(code).hexdigest()},
                          {"path": "LICENSE", "sha256": sha256(b"license\n").hexdigest()}]}

    def fetch(self, url):
        return self.assets[url]

    def acquire(self):
        return module.acquire(self.design, self.root, self.fetch)

    def replace_archive(self, entries):
        source = self.design["sources"]["TimeMixer"]
        payload = make_zip(entries)
        self.assets[source["archive_url"]] = payload
        source["archive_sha256"] = sha256(payload).hexdigest()

    def test_valid_acquisition_and_cache_are_deterministic(self):
        first = self.acquire()
        original = (self.root / module.RECEIPT_NAME).read_bytes()
        def forbidden(_): raise AssertionError("Verified cache must avoid network")
        second = module.acquire(self.design, self.root, forbidden)
        self.assertEqual(first, second)
        self.assertEqual(original, (self.root / module.RECEIPT_NAME).read_bytes())
        self.assertEqual(first["model_fits"], 0)
        self.assertEqual(len(first["sources"]), 2)

    def test_download_hash_mismatch_is_not_written(self):
        self.assets[self.design["dataset"]["url"]] = b"wrong"
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"): self.acquire()
        self.assertFalse((self.root / "ETTh1.csv").exists())

    def test_corrupt_cached_dataset_is_not_replaced(self):
        (self.root / "ETTh1.csv").write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"): self.acquire()
        self.assertEqual((self.root / "ETTh1.csv").read_bytes(), b"different")

    def test_corrupt_cached_archive_is_not_replaced(self):
        (self.root / "TimeMixer.zip").write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"): self.acquire()
        self.assertEqual((self.root / "TimeMixer.zip").read_bytes(), b"different")

    def test_archive_traversal_is_rejected_after_archive_hash_check(self):
        prefix = self.design["sources"]["TimeMixer"]["extract_directory"]
        self.replace_archive([(prefix + "/../escaped.py", b"bad")])
        with self.assertRaisesRegex(ValueError, "Unsafe relative path"): self.acquire()
        self.assertFalse((self.root / "escaped.py").exists())

    def test_absolute_and_backslash_members_are_rejected(self):
        for name in ["/absolute.py", "TimeMixer\\escaped.py", "C:/escaped.py"]:
            with self.subTest(name=name):
                temp = Path(self.temp.name) / sha256(name.encode()).hexdigest()
                source = copy.deepcopy(self.design["sources"]["TimeMixer"])
                payload = make_zip([(name, b"bad")])
                with self.assertRaisesRegex(ValueError, "Unsafe relative path"):
                    module.extract_verified(temp, payload, source)

    def test_symlink_zip_member_is_rejected(self):
        source = self.design["sources"]["TimeMixer"]
        link = zipfile.ZipInfo(source["extract_directory"] + "/link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        self.replace_archive([(link, b"../../escape")])
        with self.assertRaisesRegex(ValueError, "Symlink"): self.acquire()

    def test_destination_symlink_is_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        prefix = self.design["sources"]["TimeMixer"]["extract_directory"]
        (self.root / prefix).symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlinks"): self.acquire()
        self.assertEqual(list(outside.iterdir()), [])

    def test_wrong_extraction_root_is_rejected(self):
        self.replace_archive([("other/model.py", b"wrong")])
        with self.assertRaisesRegex(ValueError, "unexpected source root"): self.acquire()

    def test_pinned_source_file_hash_mismatch_prevents_extraction(self):
        source = self.design["sources"]["TimeMixer"]
        source["files"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Pinned source file"): self.acquire()
        self.assertFalse((self.root / source["extract_directory"]).exists())

    def test_changed_extracted_source_is_not_overwritten(self):
        self.acquire()
        source = self.design["sources"]["TimeMixer"]
        target = self.root / source["extract_directory"] / "model.py"
        target.write_bytes(b"local edit")
        with self.assertRaisesRegex(ValueError, "Existing source differs"): self.acquire()
        self.assertEqual(target.read_bytes(), b"local edit")

    def test_archive_expansion_is_bounded(self):
        with patch.object(module, "MAX_EXTRACTED_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "expands beyond"): self.acquire()

    def test_design_requires_immutable_archive_url_and_distinct_paths(self):
        self.design["sources"]["TimeMixer"]["archive_url"] = "https://codeload.github.com/example/TimeMixer/zip/main"
        with self.assertRaisesRegex(ValueError, "pinned repository and commit"): self.acquire()

    def test_cli_reuses_verified_offline_cache(self):
        self.acquire()
        path = self.root / "design.json"
        path.write_text(json.dumps(self.design))
        with patch.object(module, "fetch_bytes", side_effect=AssertionError("No network")):
            self.assertEqual(module.main(["--design", str(path), "--output-dir", str(self.root)]), 0)


if __name__ == "__main__":
    unittest.main()
