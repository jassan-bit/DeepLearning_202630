"""Setup contracts with fake files/network only; never install or train."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import bootstrap_colab as bootstrap


class Response:
    def __init__(self, body): self.body = body
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self, limit): return self.body[:limit]


class BootstrapTests(unittest.TestCase):
    def test_hash_bound_download(self):
        body = b"synthetic source"
        with patch("urllib.request.urlopen", return_value=Response(body)):
            self.assertEqual(bootstrap.download("https://example.test/file", hashlib.sha256(body).hexdigest(), max_bytes=100), body)
            with self.assertRaises(RuntimeError):
                bootstrap.download("https://example.test/file", "0" * 64, max_bytes=100)

    def test_download_size_bound(self):
        with patch("urllib.request.urlopen", return_value=Response(b"too many bytes")):
            with self.assertRaises(RuntimeError):
                bootstrap.download("https://example.test/file", "0" * 64, max_bytes=2)

    def test_unsafe_source_paths(self):
        for value in ("../secret", "/absolute", "a/../b", "a//b", "a\\b", "./x"):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                bootstrap.safe_member(value)

    def test_manifest_duplicate_rejected(self):
        item = {"path": "protocol.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()}
        manifest = json.dumps({"files": [item, item]}).encode()
        def fake(url, *args, **kwargs):
            return manifest if url.endswith("SOURCE_MANIFEST.json") else b"{}"
        with tempfile.TemporaryDirectory() as folder, patch.object(bootstrap, "download", side_effect=fake):
            with self.assertRaisesRegex(RuntimeError, "Duplicate"):
                bootstrap.acquire_source("1" * 40, "2" * 64, folder)

    def fake_runtime(self, folder):
        cache = Path(folder) / "cache"
        source = cache / "source"
        source.mkdir(parents=True)
        lock = b"synthetic requirements\n"
        (source / "requirements-cpu.lock.txt").write_bytes(lock)
        runtime = cache / ("venv-cpu-" + hashlib.sha256(lock).hexdigest()[:16] + "-py" + str(sys.version_info.major) + str(sys.version_info.minor))
        (runtime / "bin").mkdir(parents=True)
        (runtime / "bin/python").write_bytes(b"fixture, not an executable")
        return cache, source

    def test_first_setup_leaves_fresh_case_for_prepare_and_never_runs_training(self):
        calls = []
        with tempfile.TemporaryDirectory() as folder:
            cache, source = self.fake_runtime(folder)
            case = Path(folder) / "persistent" / "case"
            def fake_command(settings, name, *args):
                calls.append(name)
                self.assertEqual(name, "prepare")
                self.assertFalse(case.exists(), "prepare requires a fresh case")
                case.mkdir(parents=True)
                (case / "prepared.json").write_text("{}")
            with patch.object(bootstrap, "acquire_source", return_value=source), \
                 patch.object(bootstrap, "download", return_value=b"fixture NPZ bytes"), \
                 patch.object(bootstrap, "command", side_effect=fake_command), \
                 patch("subprocess.run"):
                result = bootstrap.setup(source_commit="1" * 40, manifest_sha256="2" * 64,
                                         case_root=case, cache_root=cache, install_runtime=False)
            self.assertEqual(calls, ["prepare"])
            self.assertEqual(result["case_root"], str(case))

    def test_existing_prepared_setup_only_checks_status(self):
        with tempfile.TemporaryDirectory() as folder:
            cache, source = self.fake_runtime(folder)
            case = Path(folder) / "persistent" / "case"
            case.mkdir(parents=True)
            (case / "prepared.json").write_text("{}")
            with patch.object(bootstrap, "acquire_source", return_value=source), \
                 patch.object(bootstrap, "download", side_effect=AssertionError("do not redownload data")), \
                 patch.object(bootstrap, "command") as command, \
                 patch("subprocess.run"):
                bootstrap.setup(source_commit="1" * 40, manifest_sha256="2" * 64,
                                case_root=case, cache_root=cache, install_runtime=False)
                self.assertEqual(command.call_args.args[1], "status")
                self.assertEqual(command.call_count, 1)

    def test_command_uses_isolated_cpu_environment(self):
        settings = {"python": "/fixture/venv/bin/python", "source_root": "/fixture/source", "case_root": "/fixture/case"}
        with patch("subprocess.run") as run:
            bootstrap.command(settings, "status")
            args, kwargs = run.call_args
            self.assertEqual(args[0][:4], [settings["python"], "-m", "seoi_v5.runner", "status"])
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(kwargs["cwd"], settings["source_root"])


if __name__ == "__main__":
    unittest.main()
