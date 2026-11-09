"""Notebook transport, admission and immutable-source checks; no installation/GPU."""
from pathlib import Path
from types import SimpleNamespace
import ast
import json
import re
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT/"notebooks/entregable2_gpu.ipynb"


def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def sources():
    return [cell["source"] for cell in notebook()["cells"] if cell["cell_type"] == "code"]


def assignment(source, name):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("Missing assignment: "+name)


class GPUNotebookTests(unittest.TestCase):
    def test_standard_notebook_has_compilable_unexecuted_cells(self):
        value = notebook()
        self.assertEqual(value["nbformat"], 4)
        self.assertEqual(value["metadata"]["language_info"]["version"], "3.12")
        ids = [cell["id"] for cell in value["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cell_id) for cell_id in ids))
        code_cells = [cell for cell in value["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(len(code_cells), 8)
        for cell in code_cells:
            ast.parse(cell["source"])
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])

    def test_published_code_pin_is_a_full_commit_after_freeze(self):
        source = assignment(sources()[0], "SOURCE_COMMIT")
        if source == "GPU_CODE_COMMIT_TO_BE_SET_BEFORE_PUBLICATION":
            self.skipTest("Parent is creating the immutable GPU code commit; replace before publication")
        self.assertRegex(source, r"^[0-9a-f]{40}$")
        self.assertNotIn(source, {"main", "master"})
        self.assertEqual(assignment(sources()[0], "DATA_COMMIT"), "0b079b50625a936456a70695532853fc13298f31")

    def test_default_is_fourteen_registered_references_with_launch_disabled(self):
        config = sources()[0]
        self.assertIsNone(assignment(config, "MODELS"))
        self.assertIs(assignment(config, "RUN_PILOT"), False)
        self.assertIs(assignment(config, "GPU_READY"), False)
        self.assertEqual(assignment(config, "COHORTS"), ["small"])
        self.assertEqual(assignment(config, "DEVICE"), "cuda:0")
        self.assertIn('os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"', config)
        self.assertNotRegex(config, r"(?:^|\n)\s*(?:from torch|import torch)")

    def test_sparse_checkout_uses_real_newlines_and_never_materializes_results(self):
        source = next(s for s in sources() if s.startswith("SPARSE_PATHS"))
        paths = assignment(source, "SPARSE_PATHS")
        self.assertIn("/scripts/", paths)
        self.assertIn("/requirements-e2-training-gpu.lock.txt", paths)
        self.assertIn("/experiments/e2_gpu_pilot_20261109/", paths)
        self.assertIn("/book_e2/_static/search_protocol_20261109/protocol.json", paths)
        self.assertFalse(any("results" in path or ".csv" in path or path == "/" for path in paths))
        self.assertIn('"--filter=blob:none"', source)
        self.assertIn('"--depth=1"', source)
        calls = []
        pin = "a"*40
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)/"source"
            def fake_run(args, **kwargs):
                calls.append((args, kwargs))
                return SimpleNamespace(stdout=pin+"\n" if "rev-parse" in args else "")
            namespace = {"REPO":repo, "SOURCE_COMMIT":pin, "REPOSITORY_URL":"https://example.invalid/source.git", "run":fake_run}
            exec(compile(source, str(NOTEBOOK), "exec"), namespace)
            self.assertFalse((repo/"results").exists())
        sparse = next(kwargs["input_text"] for args, kwargs in calls if "--stdin" in args)
        self.assertEqual(sparse, "\n".join(paths)+"\n")
        self.assertNotIn("\\n", sparse)

    def test_cuda_probe_and_both_model_suites_precede_any_market_fetch(self):
        code = sources()
        probe_index = next(i for i, s in enumerate(code) if '"--check-environment"' in s)
        smoke_index = next(i for i, s in enumerate(code) if "test_e2_cuda_models.py" in s)
        data_index = next(i for i, s in enumerate(code) if "fetch_e2_development_sources.py" in s)
        launch_index = next(i for i, s in enumerate(code) if "completed = subprocess.run" in s)
        self.assertLess(probe_index, smoke_index)
        self.assertLess(smoke_index, data_index)
        self.assertLess(data_index, launch_index)
        self.assertGreater(code[probe_index].index("GPU_READY = True"), code[probe_index].index('"--check-environment"'))
        smoke = code[smoke_index]
        self.assertIn("test_e2_cuda_training.py", smoke)
        self.assertLess(smoke.index("fetch_e2_itransformer_sources.py"), smoke.index("test_e2_cuda_models.py"))
        self.assertIn("E2_ITRANSFORMER_SOURCE_ROOT", smoke)
        self.assertIn("if not GPU_READY:", smoke)

    def test_market_download_and_launch_require_both_gates(self):
        code = sources()
        data = next(s for s in code if "fetch_e2_development_sources.py" in s)
        launch = next(s for s in code if "completed = subprocess.run" in s)
        for source in (data, launch):
            tree = ast.parse(source)
            gated = next(node for node in tree.body if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "RUN_PILOT")
            self.assertTrue(any(isinstance(node, ast.If) and "GPU_READY" in ast.unparse(node.test) for node in gated.body))
        # The plan is durable before even the raw byte fetch.
        self.assertLess(data.index('open("x"'), data.index("fetch_e2_development_sources.py"))
        self.assertIn("--expected-registration-sha256", launch)
        self.assertIn("--registration-commit", launch)
        self.assertNotIn('"--epochs"', launch)
        self.assertIn('if MODELS is not None:', launch)

    def test_failed_repeated_cuda_preflight_clears_a_previous_ready_flag(self):
        source = next(s for s in sources() if '"--check-environment"' in s)
        def failed_probe(args, **kwargs):
            if "--check-environment" in args:
                raise RuntimeError("synthetic CUDA unavailable")
            return SimpleNamespace(stdout="")
        with tempfile.TemporaryDirectory() as directory:
            namespace={"GPU_READY":True, "VENV":Path(directory), "REPO":Path(directory)/"source",
                       "run":failed_probe, "sys":SimpleNamespace(executable="python"), "DEVICE":"cuda:0"}
            with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                exec(compile(source, str(NOTEBOOK), "exec"), namespace)
            self.assertIs(namespace["GPU_READY"], False)

    def test_gpu_environment_never_includes_the_historical_cpu_torch(self):
        lines=(ROOT/"requirements-e2-training-gpu.lock.txt").read_text().splitlines()
        active=[line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
        self.assertIn("torch==2.8.0+cu126", active)
        self.assertIn("--extra-index-url https://download.pytorch.org/whl/cu126", active)
        self.assertFalse(any("+cpu" in line or "/whl/cpu" in line or line.startswith("-r") for line in active))
        self.assertIn("nvidia-cuda-runtime-cu12==12.6.77", active)
        self.assertIn("nvidia-cudnn-cu12==9.10.2.21", active)
        self.assertIn("triton==3.4.0", active)

    def test_result_cell_displays_only_the_scientific_report_without_push(self):
        joined="\n".join(sources())
        self.assertIn('"scientific_report.json"', sources()[-1])
        self.assertNotRegex(joined, r'(?i)git\s+push|["\']push["\']|files\.upload|files\.download')
        self.assertNotIn("os.environ.items()", joined)


if __name__ == "__main__":
    unittest.main()
