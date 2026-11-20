"""Synthetic launcher contract checks; no Colab, GPU, repository network or fits."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

LAUNCHER = Path(__file__).parent / "launch_colab.py"
if not LAUNCHER.is_file():
    LAUNCHER = Path(__file__).parents[1] / "launcher" / "launch_colab.py"
spec = importlib.util.spec_from_file_location("qa_pilot_launcher", LAUNCHER)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherReview(unittest.TestCase):
    def test_scientific_environment_removes_credentials_and_injects_determinism(self):
        fake = "SYNTHETIC_SECRET_not_a_real_credential"
        with patch.dict(os.environ, {"GITHUB_TOKEN": fake, "SERVICE_SECRET": "other",
                                    "GIT_ASKPASS": "/bad", "UNUSUAL": fake,
                                    "PYTHONPATH": "/bad", "PIP_EXTRA_INDEX_URL": "bad"}):
            env = launcher.environment({"API_TOKEN": "other", "SAFE_SETTING": "yes"}, fake)
        self.assertFalse(any(launcher.SECRET.search(k) for k in env))
        self.assertNotIn(fake, env.values())
        self.assertNotIn("GIT_ASKPASS", env)
        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("PIP_EXTRA_INDEX_URL", env)
        self.assertEqual(env["SAFE_SETTING"], "yes")
        self.assertEqual(env["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")

    def test_local_session_reuses_exact_source_and_run(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(launcher, "_git", return_value=b""):
                first, restoring = launcher.choose_run(d, "a" * 40, None)
            self.assertFalse(restoring)
            self.assertEqual(launcher.choose_run(d, "a" * 40, None), (first, False))
            self.assertEqual(launcher.choose_run(d, "a" * 40, first), (first, True))
            with self.assertRaises(ValueError):
                launcher.choose_run(d, "b" * 40, None)

    def test_plan_is_published_before_worker_and_all_engine_paths_exist(self):
        events = []
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            source = base / "source"
            names = [launcher.SCOPE + "/launch_colab.py", launcher.SCOPE + "/git_checkpoint_store.py",
                     launcher.SCOPE + "/pilot_engine.py", launcher.SCOPE + "/plugin_control.py",
                     *launcher.FROZEN_FILES, "src/original_model.py", "src/e3_development.py",
                     "src/e3_innovation_model.py", "src/e3_overlap_model_v3.py"]
            for name in names:
                p = source / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("# synthetic fixture; no executable model\n")
            npz = base / "SYNTHETIC_ARCHIVE.npz"
            npz.write_bytes(b"synthetic only")

            class Store:
                def __init__(self, *args):
                    self.state = {"sequence": 0}
                    self.branch = "synthetic-branch"
                def publish(self, output):
                    self.state["sequence"] += 1
                    events.append(("publish", (output / "plan.json").is_file(),
                                   (output / "publication_receipt.json").is_file()))
                    return {"git_commit": "b" * 40, "source_commit": "a" * 40,
                            "master_run_id": run_id, "run_url": "synthetic://only"}

            run_id = "20261120T010203000002Z"

            def loaded(path, name):
                if path.name == "git_checkpoint_store.py":
                    return types.SimpleNamespace(GitCheckpointStore=Store)
                if path.name == "colab_runtime.py":
                    return types.SimpleNamespace(prepare_runtime=lambda p: (Path("/synthetic/python"), {}, {"fixture": True}))
                if path.name == "pilot_engine.py":
                    return types.SimpleNamespace(digest=lambda x: hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest())
                raise AssertionError(path)

            def prepare(command, **kwargs):
                self.assertTrue(Path(command[1]).is_file(), command)
                self.assertEqual(command[-1], "--prepare-only")
                self.assertNotIn("SYNTHETIC_CREDENTIAL_123456789", kwargs["env"].values())
                output = Path(command[command.index("--output") + 1])
                launcher.atomic(output / "plan.json", {"master_run_id": run_id, "scope": "synthetic_only"})
                launcher.atomic(output / "admission.json", {"synthetic_fixture": True})
                events.append(("prepare",))

            def worker(command, **kwargs):
                events.append(("worker",))
                self.assertIn("--run", command)
                self.assertTrue((kwargs["output"] / "publication_receipt.json").is_file())
                self.assertGreaterEqual(kwargs["store"].state["sequence"], 2)
                self.assertTrue(Path(command[1]).is_file())
                return {"run_url": "synthetic://only"}

            with patch.object(launcher, "checkout", return_value=source), \
                 patch.object(launcher, "choose_run", return_value=(run_id, False)), \
                 patch.object(launcher, "load", side_effect=loaded), \
                 patch.object(launcher, "hardware_profile", return_value={"synthetic_fixture": True}), \
                 patch.object(launcher, "admit_npz", return_value=npz), \
                 patch.object(launcher.subprocess, "run", side_effect=prepare), \
                 patch.object(launcher, "run_worker", side_effect=worker), \
                 contextlib.redirect_stdout(io.StringIO()):
                launcher.main("a" * 40, base=base, token="SYNTHETIC_CREDENTIAL_123456789")
            self.assertEqual([event[0] for event in events], ["prepare", "publish", "publish", "worker"])
            self.assertEqual(events[1][1:], (True, False))
            self.assertEqual(events[2][1:], (True, True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
