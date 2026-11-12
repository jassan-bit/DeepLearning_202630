"""Transport-only launch tests. No network, actual GPU, optimizer or market."""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


launcher = module("github_only_launcher", "notebooks/run_e2_reduced_github_colab.py")
driver = module("github_only_driver", "scripts/run_e2_reduced_github_delivery.py")


class LauncherTests(unittest.TestCase):
    def test_extension_is_pinned_and_tampered_local_file_is_preserved(self):
        payloads = {name: (name + " source").encode() for name in launcher.EXTENSION_FILES}
        record = {"protocol_id": "e2-reduced-delivery-v1", "engine_commit": launcher.ENGINE_COMMIT,
            "files_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()}}
        payloads[launcher.EXTENSION_REGISTRATION] = json.dumps(record).encode()
        with tempfile.TemporaryDirectory() as directory, patch.object(launcher, "_public_bytes",
                side_effect=lambda commit, name: payloads[name]):
            root = launcher._extension("a" * 40, Path(directory))
            changed = root / launcher.EXTENSION_FILES[0]
            changed.write_bytes(b"modified")
            with self.assertRaisesRegex(RuntimeError, "conserva"):
                launcher._extension("a" * 40, Path(directory))
            self.assertEqual(changed.read_bytes(), b"modified")

    def test_launch_has_only_local_scratch_and_github_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / "python"
            python.write_text("synthetic")
            commands = []
            with patch.object(launcher, "LOCAL_WORKSPACE", root), \
                    patch.object(launcher, "_run"), patch.object(launcher, "_assert_idle"), \
                    patch.object(launcher, "_checkout", return_value=root / "engine"), \
                    patch.object(launcher, "_extension", return_value=root / "extension"), \
                    patch.object(launcher, "_stream", side_effect=lambda command, *a: commands.append(command) or 0), \
                    patch.dict(os.environ, {"GITHUB_TOKEN": "synthetic-test-secret"}):
                result = launcher.run_reduced_github_colab("a" * 40,
                    namespace={"GPU_PYTHON": str(python)}, execute=True)
            self.assertEqual(result["persistent_destination"], "github")
            self.assertEqual(len(commands), 1)
            self.assertIn("run_e2_reduced_github_delivery.py", str(commands[0][1]))
            self.assertIn("--execute", commands[0])
            self.assertIn(launcher.ENGINE_COMMIT, commands[0])
            self.assertNotIn("drive", str(commands[0]).lower())
            source = (ROOT / "notebooks/run_e2_reduced_github_colab.py").read_text()
            self.assertNotIn("google.colab import drive", source)
            self.assertNotIn("/content/drive", source)

    def test_restore_precedes_ledger_and_execution_and_failure_cannot_train(self):
        calls = []
        args = SimpleNamespace(execute=True, stage="all", output=Path("/temporary-only"),
            data_root="dev", source_roots="roots", runtime_profile="runtime", budget_evidence="budget",
            migration="migration", cache_audit="audit", reconciliation_root="recon",
            migration_registration_commit=launcher.ENGINE_COMMIT, inherited_ledger=[])
        parser = SimpleNamespace(parse_args=lambda unused: args)
        scientific = SimpleNamespace(parser=lambda: parser, read_json=lambda p: {"file": p},
            execute_delivery=lambda *a: calls.append("execute"))
        class Backup:
            def preflight(self): calls.append("preflight")
        class State:
            fail = False
            def __init__(self, *a): pass
            def restore(self):
                calls.append("restore")
                if self.fail: raise RuntimeError("unknown reservation")
            @contextmanager
            def budget_factory(self, *a, **kw):
                calls.append("ledger")
                yield object()
                calls.append("closed")
            def snapshot(self, reason): calls.append("snapshot")
        modules = {"run_e2_reduced_delivery": scientific,
            "e2_reduced_protocol": SimpleNamespace(verify_reduced_registration=lambda *a, **kw: None),
            "e2_reduced_github_backup": SimpleNamespace(GitHubBackup=Backup),
            "e2_reduced_github_state": SimpleNamespace(GitHubState=State)}
        argv = ["--engine-root", "/engine", "--extension-root", "/extension", "--extension-commit", "a" * 40]
        with patch.dict(sys.modules, modules), patch.object(driver, "_engine_root", return_value=Path("/engine")), \
                patch.object(driver, "_extension_root", return_value=Path("/extension")):
            self.assertEqual(driver.main(argv), 0)
            self.assertEqual(calls, ["preflight", "restore", "ledger", "snapshot", "execute", "closed"])
            calls.clear(); State.fail = True
            self.assertEqual(driver.main(argv), 3)
            self.assertEqual(calls, ["preflight", "restore"])


if __name__ == "__main__":
    unittest.main()
