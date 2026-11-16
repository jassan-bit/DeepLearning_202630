"""Portable controller guards; synthetic files only, no scientific execution."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "scripts/e2_portable_campaign.py"
SPEC = importlib.util.spec_from_file_location("portable_campaign_under_test", SOURCE)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class PortableGuards(unittest.TestCase):
    def test_package_path_traversal_and_absolute_paths_fail(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for path in ("../outside", "/tmp/outside", "a/../b", "a\\b", "", "./a"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    RUNNER.relative_path(root, path, exists=False)

    def test_symlink_input_is_rejected_even_when_its_target_is_inside(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "real").write_text("fixture")
            (root / "alias").symlink_to(root / "real")
            with self.assertRaises(ValueError):
                RUNNER.relative_path(root, "alias")

    def test_changed_bytes_fail_verification(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / "input"
            path.write_bytes(b"known fixture")
            record = RUNNER.file_record(root, path)
            self.assertEqual(RUNNER.verify_record(root, record), path)
            path.write_bytes(b"altered bytes")
            with self.assertRaises(ValueError):
                RUNNER.verify_record(root, record)

    def test_check_mode_never_starts_a_worker(self):
        args = type("Args", (), {"mode": "check", "output": None})()
        with patch.object(RUNNER.subprocess, "run", side_effect=AssertionError("unexpected worker")), \
             patch.object(RUNNER, "runtime", return_value={"fixture": True}), \
             patch("builtins.print"):
            self.assertEqual(RUNNER.run(args, Path("."), {"files": []}), 0)

    def test_computational_mode_requires_an_explicit_execution_and_limit(self):
        args = type("Args", (), {"mode": "train-fixed", "execute": False, "output": None, "cap_seconds": None})()
        with patch.object(RUNNER.subprocess, "run", side_effect=AssertionError("unexpected worker")):
            with self.assertRaises(ValueError):
                RUNNER.run(args, Path("."), {})

    def test_exclusive_receipts_preserve_first_attempt(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "receipt.json"
            RUNNER.write(path, {"returncode": 1})
            with self.assertRaises(FileExistsError):
                RUNNER.write(path, {"returncode": 0})
            self.assertEqual(json.loads(path.read_text())["returncode"], 1)


if __name__ == "__main__":
    unittest.main()
