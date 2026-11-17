"""Exercise the coordinator and real worker together, with synthetic prices only."""
from contextlib import redirect_stdout
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("E3_EVAL_SOURCE_ROOT", str(HERE.parents[1]))).resolve()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Integration(unittest.TestCase):
    def test_full_seed_campaign_then_idempotent_reentry(self):
        fixtures = module("integration_worker_fixture", HERE / "test_worker_synthetic.py")
        w = fixtures.w
        b = module("integration_coordinator", HERE / "bootstrap.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_root = root / "fixture"
            fixture_root.mkdir()
            sample_inputs, sample_run = fixtures.fixture(fixture_root)
            plan = json.loads((sample_run / "plan.json").read_text())
            b.PLAN_FINGERPRINT = w.PLAN_FINGERPRINT
            w.MAX_EPOCHS = 2  # Synthetic-only shortening, never production inputs.
            b.HERE = fixture_root / "code"
            (b.HERE / "plan.json").write_bytes(b.canonical(plan) + b"\n")
            source_inputs = {record["path"]: sample_inputs / name
                             for name, record in plan["inputs"].items()}

            def offline_download(commit, relative, destination, expected):
                original = source_inputs.get(relative, ROOT / relative)
                payload = original.read_bytes()
                self.assertEqual(b.sha(payload), expected)
                b.atomic(destination, payload)
                return payload

            class Remote:
                token = "unused"
                def __init__(self):
                    self.current, self.payloads, self.count = None, None, 0
                def head(self):
                    return self.current
                def api(self, method, path, *args, **kwargs):
                    self_outer.assertEqual(method, "GET")
                    branches = {
                        "e3-desarrollo-resultados-momentos": "4cd455d90fb72fe78beeb885dc5dadc4ea9edeb4",
                        "e3-desarrollo-resultados-v2": "1ca029e869a517cb8de704e573c78ca1e0cbad3c",
                        "e3-desarrollo-resultados-v1": "f7826d120f3a6eb8e9a0952dd9842cd77f152f34",
                        "e3-evaluacion-aditiva-v2": "c27f892c7743a6a80f6b49b3a6930b18e2e20943",
                    }
                    return {"object": {"sha": branches[path.rsplit("/", 1)[-1]]}}
                def read(self):
                    if self.current is None:
                        return None
                    return self.current, "tree", b.make_snapshot(self.payloads), dict(self.payloads)
                def publish(self, snapshot, payloads, expected):
                    self_outer.assertEqual(expected, self.current)
                    b.validate_snapshot(snapshot, payloads)
                    self.count += 1
                    self.current = f"{self.count:040x}"
                    self.payloads = dict(payloads)
                    return self.current

            self_outer = self
            remote = Remote()
            phases = []
            real_spec = importlib.util.spec_from_file_location
            class RuntimeLoader:
                def create_module(self, spec):
                    return None
                def exec_module(self, target):
                    target.preparar_entorno = lambda *args, **kwargs: (
                        "python", {"OMP_NUM_THREADS": "2", "GITHUB_TOKEN": "must-be-filtered"}, {})
            def fake_runtime_spec(name, path, *args, **kwargs):
                if str(path).endswith("preparar_e3_innovacion_runtime.py"):
                    return importlib.machinery.ModuleSpec(name, RuntimeLoader())
                return real_spec(name, path, *args, **kwargs)

            def worker(arguments, **kwargs):
                self.assertNotIn("GITHUB_TOKEN", kwargs["env"])
                def argument(name):
                    return arguments[arguments.index(name) + 1]
                phase = argument("--phase")
                phases.append(phase)
                if phase == "evaluate":
                    saved = remote.read()
                    self.assertEqual(saved[2]["progress"]["status"], "trained")
                    self.assertTrue(all(json.loads(saved[3][f"seeds/{seed}/checkpoint.meta.json"])["complete"]
                                        for seed in w.SEEDS))
                old_modules = {name: obj for name, obj in sys.modules.items()
                               if name == "src" or name.startswith("src.")}
                old_path = list(sys.path)
                for name in old_modules:
                    del sys.modules[name]
                try:
                    return w.execute(argument("--source-root"), argument("--inputs"),
                        argument("--run-dir"), argument("--claim-path"), phase,
                        int(argument("--max-epochs-this-call")), "cpu")
                finally:
                    for name in list(sys.modules):
                        if name == "src" or name.startswith("src."):
                            del sys.modules[name]
                    sys.modules.update(old_modules)
                    sys.path[:] = old_path

            workspace = root / "colab"
            with patch.object(b, "download", side_effect=offline_download), \
                 patch.object(b, "token", return_value="unused"), \
                 patch.object(b, "GitHub", return_value=remote), \
                 patch.object(b, "subprocess", SimpleNamespace(run=worker)), \
                 patch("importlib.util.spec_from_file_location", side_effect=fake_runtime_spec), \
                 redirect_stdout(io.StringIO()):
                head = b.run(workspace)
                self.assertEqual(phases, ["train", "evaluate"])
                summary = json.loads(remote.payloads["summary.json"])
                self.assertEqual(summary["status"], "completed")
                self.assertEqual(summary["complete_training_seeds"], 10)
                self.assertEqual(len(summary["per_seed_metrics"]), 10)
                self.assertGreater(summary["charged_seconds"], 0)
                self.assertLess(summary["charged_seconds"], 500)
                self.assertEqual(b.run(workspace), head)
                self.assertEqual(phases, ["train", "evaluate"])


if __name__ == "__main__":
    unittest.main()
