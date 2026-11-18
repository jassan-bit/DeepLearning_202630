"""CPU checks of isolated-runtime routing; no GPU or CUDA wheel is simulated as executed."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("isolated_runtime_under_test", HERE / "colab_runtime.py")
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)
LOCK = HERE.parents[2] / "requirements-e2-training-gpu.lock.txt"


class IsolatedRuntimeContracts(unittest.TestCase):
    def test_complete_existing_lock_and_sanitized_environment(self):
        raw = LOCK.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), runtime.LOCK_SHA256)
        packages = runtime._expected(raw)
        for name in ("pandas", "scipy", "statsmodels", "einops", "reformer-pytorch", "torch", "nvidia-cudnn-cu12"):
            self.assertIn(name, packages)
        with mock.patch.dict(os.environ, {"PYTHONPATH": "/host/site-packages", "PIP_TARGET": "/host", "TEST_SECRET": "fixture", "GH_TOKEN": "fixture"}):
            env = runtime._environment(Path("/tmp/isolated-fixture"))
            self.assertNotIn("PYTHONPATH", env)
            self.assertNotIn("PIP_TARGET", env)
            self.assertNotIn("TEST_SECRET", env)
            self.assertNotIn("GH_TOKEN", env)
            self.assertEqual(os.environ["PIP_TARGET"], "/host")
            self.assertEqual(env["UV_PYTHON_INSTALL_DIR"], "/tmp/isolated-fixture/managed-python")
            self.assertEqual(env["PYTHONNOUSERSITE"], "1")

    def test_missing_gpu_stops_before_network_or_installation(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "new-runtime"
            with mock.patch.object(runtime.subprocess, "check_output", side_effect=FileNotFoundError), mock.patch.object(runtime, "_bytes") as download:
                with self.assertRaisesRegex(RuntimeError, "No hay GPU visible"):
                    runtime.prepare_runtime(workspace)
            download.assert_not_called()
            self.assertFalse(workspace.exists())

    def test_existing_bootstrap_routes_to_isolated_interpreter_and_checks_receipt(self):
        # These deliberately fake helper functions test routing only. No GPU,
        # package installation or genuine runtime compatibility is asserted.
        helper = b'''from pathlib import Path
def _assert_no_controller(): pass
def _bootstrap_python(workspace, env):
    path=workspace/'venv-py312-cu126/bin/python'
    path.parent.mkdir(parents=True)
    path.write_text('synthetic routing fixture; never executed')
    return path
def _gpu_environment(python, source, env):
    assert str(python).endswith('/venv-py312-cu126/bin/python')
    assert (source/'requirements-e2-training-gpu.lock.txt').is_file()
    return {'fixture_only':True}
'''
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "isolated-runtime"
            lock = LOCK.read_bytes()
            def source(relative, expected):
                return helper if relative == runtime.HELPER_PATH else lock
            def check(command, **kwargs):
                if command[0] == "nvidia-smi":
                    return "Fixture GPU; no hardware execution\n"
                self.assertEqual(command[0], str(workspace / "venv-py312-cu126/bin/python"))
                self.assertEqual(kwargs["env"]["PYTHONNOUSERSITE"], "1")
                return json.dumps({"python": command[0], "packages_verified": len(runtime._expected(lock))})
            with mock.patch.object(runtime, "_bytes", side_effect=source), mock.patch.object(runtime.subprocess, "check_output", side_effect=check):
                python, environment, receipt = runtime.prepare_runtime(workspace)
                self.assertEqual(python, workspace / "venv-py312-cu126/bin/python")
                self.assertFalse(receipt["base_Torch_changed"])
                self.assertFalse(receipt["training_started"])
                self.assertFalse(receipt["models_executed"])
                self.assertTrue((workspace / "e3_preflight_runtime_receipt.json").is_file())
                # A receipt from a different lock must fail, preserving bytes.
                path = workspace / "e3_preflight_runtime_receipt.json"
                changed = json.loads(path.read_text())
                changed["lock_sha256"] = "altered"
                path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(RuntimeError, "different lock"):
                    runtime.prepare_runtime(workspace)
                self.assertEqual(json.loads(path.read_text())["lock_sha256"], "altered")


if __name__ == "__main__":
    unittest.main(verbosity=2)
