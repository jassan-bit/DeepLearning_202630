"""Offline source-admission and parent-preserving overlay checks."""
import io
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile, ZipInfo

import prepare


def fixture_zip(extra=()):
    stream = io.BytesIO()
    author_rows = []
    payloads = {}
    for i in range(76):
        model = sorted(prepare.AUTHOR_MODELS)[i % 6]
        path = "LICENSE" if i < 6 else "source_" + str(i) + ".py"
        raw = (model + ":" + str(i)).encode()
        author_rows.append({"model": model, "path": path, "bytes": len(raw),
                            "sha256": prepare.sha256(raw).hexdigest()})
        payloads["D02/authors/" + model + "/" + path] = raw
    payloads.update({"D02/code/" + name: b"{}" for name in prepare.METADATA})
    payloads["D02/code/" + prepare.METADATA[0]] = json.dumps({"author_sources": author_rows}).encode()
    payloads["D02/code/scripts/scientific.py"] = b"# unchanged source\n"
    payloads["D02/requirements_cuda.txt"] = b"# fixture\n"
    payloads["D02/code/old_scores.csv"] = b"excluded"
    payloads["D02/trained_weights.pkl"] = b"excluded"
    payloads["D02/code/results/market.csv"] = b"excluded"
    with ZipFile(stream, "w") as archive:
        for name, raw in payloads.items():
            archive.writestr(name, raw)
        for name, raw in extra:
            archive.writestr(name, raw)
    return stream.getvalue()


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def source(self, raw):
        path = self.root / "source.zip"
        path.write_bytes(raw)
        return path

    def test_positive_selection_hash_manifest_and_idempotence(self):
        source = self.source(fixture_zip())
        expected = prepare.file_sha(source)
        with patch.object(prepare, "source_preflight", return_value={"status": "verified"}):
            report = prepare.prepare_source(source, self.root / "one", expected)
            self.assertEqual(report, prepare.prepare_source(source, self.root / "one", expected))
            other = prepare.prepare_source(source, self.root / "two", expected)
        self.assertEqual(len(report["members"]), 85)
        self.assertEqual(report["source_binding_sha256"], other["source_binding_sha256"])
        for row in report["members"]:
            path = self.root / "one/bundle" / row["path"]
            self.assertEqual(prepare.file_sha(path), row["sha256"])
            self.assertEqual(path.stat().st_size, row["bytes"])
        self.assertFalse((self.root / "one/bundle/D02/trained_weights.pkl").exists())
        self.assertFalse((self.root / "one/bundle/D02/code/old_scores.csv").exists())
        self.assertFalse((self.root / "one/bundle/D02/code/results").exists())

    def test_wrong_zip_hash_writes_nothing(self):
        source = self.source(fixture_zip())
        with self.assertRaisesRegex(ValueError, "SHA256"):
            prepare.prepare_source(source, self.root / "work", "0" * 64)
        self.assertFalse((self.root / "work").exists())

    def test_traversal_absolute_and_windows_paths_rejected(self):
        for bad in ("../escape", "/absolute", "D02/code/../escape", "D02\\escape", "C:/escape"):
            with self.subTest(bad=bad):
                with ZipFile(io.BytesIO(fixture_zip(((bad, b"bad"),)))) as archive:
                    with self.assertRaisesRegex(ValueError, "Unsafe"):
                        prepare.selected_members(archive)

    def test_symlink_and_duplicate_members_rejected(self):
        link = ZipInfo("D02/link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        with ZipFile(io.BytesIO(fixture_zip(((link, b"../../escape"),)))) as archive:
            with self.assertRaisesRegex(ValueError, "symlinks"):
                prepare.selected_members(archive)
        with ZipFile(io.BytesIO(fixture_zip((("D02/requirements_cuda.txt", b"duplicate"),)))) as archive:
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                prepare.selected_members(archive)

    def test_author_pin_and_existing_source_tamper_rejected(self):
        raw = fixture_zip()
        stream = io.BytesIO()
        with ZipFile(io.BytesIO(raw)) as original, ZipFile(stream, "w") as changed:
            for info in original.infolist():
                payload = original.read(info.filename)
                changed.writestr(info.filename, b"tampered" if info.filename.endswith("source_6.py") else payload)
        with ZipFile(io.BytesIO(stream.getvalue())) as archive:
            with self.assertRaisesRegex(ValueError, "Pinned author"):
                prepare.selected_members(archive)
        source = self.source(raw)
        work = self.root / "work"
        with patch.object(prepare, "source_preflight", return_value={}):
            prepare.prepare_source(source, work, prepare.file_sha(source))
            target = work / "bundle/D02/code/scripts/scientific.py"
            target.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "Existing source bytes"):
                prepare.prepare_source(source, work, prepare.file_sha(source))
        self.assertEqual(target.read_bytes(), b"tampered")


class OverlayTests(unittest.TestCase):
    def test_child_precedence_and_parent_bytes_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "parent"
            env = prepare.clean_environment()
            subprocess.run([sys.executable, "-B", "-m", "venv", "--without-pip", str(parent)],
                           env=env, check=True, capture_output=True)
            parent_python = parent / "bin/python"
            parent_site = Path(prepare.run_python(parent_python,
                "import json,sysconfig;print(json.dumps(sysconfig.get_path('purelib')))", env))
            (parent_site / "e2_overlay_fixture.py").write_text("VALUE='parent'\n")
            before = {str(p.relative_to(parent)): prepare.file_sha(p)
                      for p in parent.rglob("*") if p.is_file() and not p.is_symlink()}
            child_python, child_site = prepare.create_overlay(parent_python, root / "child", [str(parent_site)], env)
            probe = "import e2_overlay_fixture as f,json,sys;print(json.dumps([f.VALUE,sys.prefix]))"
            value, prefix = prepare.run_python(child_python, probe, env)
            self.assertEqual(value, "parent")
            self.assertEqual(prefix, str(root / "child"))
            (child_site / "e2_overlay_fixture.py").write_text("VALUE='child'\n")
            self.assertEqual(prepare.run_python(child_python, probe, env)[0], "child")
            after = {str(p.relative_to(parent)): prepare.file_sha(p)
                     for p in parent.rglob("*") if p.is_file() and not p.is_symlink()}
            self.assertEqual(before, after)
            self.assertFalse((parent_site / "__pycache__").exists())

    def test_environment_removes_python_and_installer_contamination(self):
        env = prepare.clean_environment({"PATH": "/bin", "PYTHONPATH": "/wrong", "PYTHONHOME": "/wrong",
            "PIP_TARGET": "/parent", "UV_CACHE_DIR": "/parent", "VIRTUAL_ENV": "/parent",
            "GITHUB_TOKEN": "secret", "GIT_ASKPASS": "/wrong", "SERVICE_API_KEY": "secret"})
        for key in ("PYTHONPATH", "PYTHONHOME", "PIP_TARGET", "UV_CACHE_DIR", "VIRTUAL_ENV",
                    "GITHUB_TOKEN", "GIT_ASKPASS", "SERVICE_API_KEY"):
            self.assertNotIn(key, env)
        self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(env["PIP_CONFIG_FILE"], prepare.os.devnull)

    def test_gpu_core_mismatch_fails_before_child_created(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(prepare, "read_lock", return_value={"torch": "2.8.0+cu126"}), \
                 patch.object(prepare, "runtime_probe", return_value={"packages": {"torch": "2.8.0+cpu"}}):
                with self.assertRaisesRegex(ValueError, "Parent Torch"):
                    prepare.prepare_runtime(temporary, sys.executable)
            self.assertFalse((Path(temporary) / "runtime").exists())

    def test_installer_targets_child_and_parent_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = {"torch": "2.8.0+cu126", "einops": "0.8.1"}
            parent = {"packages": {"torch": "2.8.0+cu126", "einops": None},
                "prefix": "/parent", "site_paths": ["/parent/site-packages"],
                "modules": {"torch": "/parent/torch/__init__.py"},
                "python": "3.12.12", "torch": "2.8.0+cu126", "torch_cuda": "12.6",
                "devices": [{"name": "A100"}], "base_prefix": "/base-python"}
            child = {**parent, "prefix": str(root / "runtime/venv-e2-extension"),
                "packages": expected}

            def fake_overlay(parent_python, child_root, paths, env):
                child_site = child_root / "lib/python3.12/site-packages"
                child_site.mkdir(parents=True)
                return child_root / "bin/python", child_site

            changed = {**parent, "packages": {"torch": "changed", "einops": None}}
            with patch.object(prepare, "read_lock", return_value=expected), \
                 patch.object(prepare, "runtime_probe", side_effect=[parent, child, changed]), \
                 patch.object(prepare, "create_overlay", side_effect=fake_overlay), \
                 patch.object(prepare, "source_preflight", return_value={}), \
                 patch.object(prepare, "file_sha", return_value="frozen-lock"), \
                 patch.object(prepare.subprocess, "run") as installer:
                with self.assertRaisesRegex(ValueError, "Parent runtime identity changed"):
                    prepare.prepare_runtime(root, "/parent/bin/python")
            command = installer.call_args.args[0]
            target = command[command.index("--target") + 1]
            self.assertTrue(target.startswith(str(root / "runtime/venv-e2-extension")))
            self.assertIn("--no-deps", command)
            self.assertIn("--ignore-installed", command)
            self.assertIn("-B", command)
            self.assertFalse((root / "runtime_identity.json").exists())

    def test_actual_source_lock_and_source_inventory(self):
        root = Path(__file__).parent
        lock = root / "bundle/D02/requirements_cuda.txt"
        if not lock.exists():
            self.skipTest("Pinned source bundle is not present")
        self.assertEqual(len(prepare.read_lock(lock)), 43)
        report = prepare.source_preflight(root / "bundle/D02")
        self.assertEqual(report["author_files_verified"], 76)
        self.assertEqual(report["original_constructor_files_verified"], 12)
        self.assertEqual(len(report["models"]), 13)


if __name__ == "__main__":
    unittest.main()
