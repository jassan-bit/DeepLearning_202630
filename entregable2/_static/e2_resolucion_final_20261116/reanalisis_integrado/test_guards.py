"""Synthetic controller guards; no native scientific module is executed."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

spec = importlib.util.spec_from_file_location("integrated_runner", Path(__file__).with_name("run.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Guards(unittest.TestCase):
    def test_unsafe_paths_and_symlinks_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("../outside", "/tmp/outside", "a/../b", "a\\b", ""):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    runner.inside(root, name, file=False)
            (root / "real").write_text("fixture")
            (root / "link").symlink_to(root / "real")
            with self.assertRaises(ValueError):
                runner.inside(root, "link")

    def test_traversal_archive_is_not_extracted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("../escape", "fixture")
            with self.assertRaises(ValueError):
                runner.safe_extract(archive, root / "output")
            self.assertFalse((root / "escape").exists())

    def test_zero_exit_code_cannot_override_failed_documentary_check(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "result.json").write_text(json.dumps({"status": "verified", "checks": [{"correcto": False}]}))
            task = {"id": "D04", "scope": "fixture", "result_json": "result.json", "required_equal": {"status": "verified"}}
            with self.assertRaises(ValueError):
                runner.expected_result(task, root)

    def test_model_inventory_must_not_fall_back_to_forward_default(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tasks = [{"id": i, "archive": i + ".zip", "script": runner.SCRIPTS[i], "arguments": []} for i in runner.IDS]
            (root / "catalog.json").write_text(json.dumps({"version": runner.VERSION, "tasks": tasks}))
            with self.assertRaises(ValueError):
                runner.catalog(root)


if __name__ == "__main__":
    unittest.main()
