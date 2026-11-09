"""Notebook transport, admission and immutable-source checks; no installation/GPU."""
from pathlib import Path
from types import SimpleNamespace
import ast
from contextlib import redirect_stdout
from io import StringIO
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


def execute(source, namespace):
    """Execute a notebook cell with supplied fakes and keep unittest output quiet."""
    with redirect_stdout(StringIO()):
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)


class BootstrapHarness:
    """A filesystem-only model of the pinned installer; never starts a process."""

    def __init__(self, directory, *, uv_version="0.9.5", python_version=(3, 12, 12),
                 failure=None, managed_outside=False, install_uv=True):
        self.workspace = Path(directory)/"workspace"
        self.uv_binary = self.workspace/"bootstrap/uv-0.9.5/bin/uv"
        self.python = self.workspace/"venv-py312-cu126/bin/python"
        self.managed = self.workspace/"managed-python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12"
        self.calls = []
        self.uv_version = uv_version
        self.python_version = python_version
        self.failure = failure
        self.managed_outside = managed_outside
        self.install_uv = install_uv
        config = sources()[0]
        names = ("UV_VERSION", "MANAGED_PYTHON_VERSION", "PIP_VERSION", "UV_WHEEL_SHA256", "PIP_WHEEL_SHA256")
        self.namespace = {name: assignment(config, name) for name in names}
        self.namespace.update(Path=Path, json=json, sys=SimpleNamespace(executable="host-python313", version_info=(3, 13, 9)),
                              WORKSPACE=self.workspace, VENV=self.python.parents[1], REPO=self.workspace/"source",
                              HARDWARE_READY=True, GPU_READY=True, DATA_READY=True, DEVICE="cuda:0", run=self.run)

    @staticmethod
    def touch(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def run(self, args, **kwargs):
        args = [str(arg) for arg in args]
        self.calls.append((args, kwargs))
        output = ""
        if args[0] == "host-python313":
            if self.failure == "bootstrap":
                raise RuntimeError("synthetic bootstrap download failed")
            if self.install_uv:
                self.touch(self.uv_binary)
        elif args[0] == str(self.uv_binary):
            if not self.uv_binary.is_file():
                raise FileNotFoundError("synthetic uv executable missing")
            if args[1:] == ["--version"]:
                output = "uv "+self.uv_version+"\n"
            elif args[1:3] == ["python", "install"]:
                if self.failure == "python-download":
                    raise RuntimeError("synthetic managed Python download failed")
                self.touch(self.managed)
            elif args[1:3] == ["python", "find"]:
                selected = self.workspace.parent/"outside/bin/python" if self.managed_outside else self.managed
                output = str(selected)+"\n"
            elif args[1] == "venv":
                self.touch(self.python)
            elif args[1:3] == ["pip", "install"] and self.failure == "pip":
                raise RuntimeError("synthetic pinned pip install failed")
        elif args[0] == str(self.python):
            if not self.python.is_file():
                raise FileNotFoundError("synthetic venv Python missing")
            if len(args) > 2 and "json.dumps" in args[2]:
                output = json.dumps({"version": list(self.python_version)})
            if "--check-environment" in args and self.failure == "cuda":
                raise RuntimeError("synthetic CUDA unavailable")
        else:
            raise AssertionError("Unexpected command: "+repr(args))
        return SimpleNamespace(stdout=output)

    def execute_cell(self):
        execute(next(s for s in sources() if '"--check-environment"' in s), self.namespace)


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
        self.assertEqual(source, "728277517be653be0d0ef10fb677a68adec22bb6")
        self.assertEqual(assignment(sources()[0], "DATA_COMMIT"), "0b079b50625a936456a70695532853fc13298f31")
        self.assertEqual(assignment(sources()[0], "EXPECTED_GPU_REGISTRATION_SHA256"),
                         "892ad148e1a261d16b178371a8f8c71ea3af352cc4c6ac4fea934df7dc22ba87")

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
        with tempfile.TemporaryDirectory() as directory:
            harness = BootstrapHarness(directory, failure="cuda")
            with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                harness.execute_cell()
            self.assertIs(harness.namespace["GPU_READY"], False)
            self.assertIs(harness.namespace["DATA_READY"], False)

    def test_hardware_preflight_accepts_a_python313_colab_kernel(self):
        calls = []
        def fake_run(args, **kwargs):
            calls.append((args, kwargs))
            return SimpleNamespace(stdout="Tesla T4, 15360 MiB, 580.95.05\n")
        with tempfile.TemporaryDirectory() as directory:
            namespace = {"re":re, "SOURCE_COMMIT":"728277517be653be0d0ef10fb677a68adec22bb6",
                         "platform":SimpleNamespace(system=lambda:"Linux", machine=lambda:"x86_64"),
                         "shutil":SimpleNamespace(which=lambda name:"/fake/nvidia-smi"),
                         "sys":SimpleNamespace(version_info=(3, 13, 9)), "WORKSPACE":Path(directory)/"workspace",
                         "run":fake_run, "HARDWARE_READY":False, "GPU_READY":True, "DATA_READY":True}
            execute(sources()[1], namespace)
            self.assertTrue(namespace["WORKSPACE"].is_dir())
            self.assertIs(namespace["HARDWARE_READY"], True)
            self.assertIs(namespace["GPU_READY"], False)
            self.assertIs(namespace["DATA_READY"], False)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0][0], "nvidia-smi")

    def test_bootstrap_pins_and_local_cache_are_explicit(self):
        config = sources()[0]
        self.assertEqual(assignment(config, "UV_VERSION"), "0.9.5")
        self.assertEqual(assignment(config, "MANAGED_PYTHON_VERSION"), "3.12.12")
        self.assertEqual(assignment(config, "PIP_VERSION"), "25.3")
        self.assertEqual(assignment(config, "UV_WHEEL_SHA256"),
                         "6507bbbcd788553ec4ad5a96fa19364dc0f58b023e31d79868773559a83ec181")
        self.assertEqual(assignment(config, "PIP_WHEEL_SHA256"),
                         "9655943313a94722b7774661c21049070f6bbb0a1516bf02f7c8d5d9201514cd")
        for setting in ('UV_CACHE_DIR=str(WORKSPACE/"uv-cache")',
                        'UV_PYTHON_INSTALL_DIR=str(WORKSPACE/"managed-python")',
                        'UV_NATIVE_TLS="true"', 'CUDA_ENV.pop("PYTHONPATH", None)'):
            self.assertIn(setting, config)

    def test_python313_host_uses_only_managed_python312_after_bootstrap(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = BootstrapHarness(directory)
            harness.execute_cell()
            self.assertEqual(harness.namespace["GPU_PYTHON"], harness.python)
            self.assertIs(harness.namespace["GPU_READY"], True)
            self.assertIs(harness.namespace["DATA_READY"], False)
            calls = [args for args, _ in harness.calls]
            host = [args for args in calls if args[0] == "host-python313"]
            self.assertEqual(len(host), 1)
            self.assertIn("--require-hashes", host[0])
            self.assertIn("--target", host[0])
            self.assertEqual(calls[0], host[0])
            install = next(args for args in calls if args[1:3] == ["python", "install"])
            self.assertIn("3.12.12", install)
            self.assertIn("--no-bin", install)
            find = next(args for args in calls if args[1:3] == ["python", "find"])
            self.assertIn("--managed-python", find)
            self.assertIn("--no-python-downloads", find)
            venv = next(args for args in calls if args[1] == "venv")
            self.assertEqual(venv[venv.index("--python")+1], str(harness.managed))
            self.assertIn("--no-python-downloads", venv)
            pinned_pip = next(args for args in calls if args[1:3] == ["pip", "install"])
            self.assertIn("--require-hashes", pinned_pip)
            self.assertEqual(pinned_pip[pinned_pip.index("--python")+1], str(harness.python))
            lock = next(args for args in calls if any(arg.endswith("requirements-e2-training-gpu.lock.txt") for arg in args))
            self.assertEqual(lock[0], str(harness.python))
            self.assertFalse(any("fetch_e2_development_sources" in " ".join(args) for args in calls))
            self.assertEqual((harness.workspace/"bootstrap/uv.requirements.txt").read_text(),
                             "uv==0.9.5 --hash=sha256:"+harness.namespace["UV_WHEEL_SHA256"]+"\n")
            self.assertEqual((harness.workspace/"bootstrap/pip.requirements.txt").read_text(),
                             "pip==25.3 --hash=sha256:"+harness.namespace["PIP_WHEEL_SHA256"]+"\n")

    def test_bootstrap_reuses_the_pinned_local_uv_instead_of_the_host_uv(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = BootstrapHarness(directory)
            harness.touch(harness.uv_binary)
            harness.execute_cell()
            self.assertFalse(any(args[0] == "host-python313" for args, _ in harness.calls))
            self.assertTrue(any(args[0] == str(harness.uv_binary) for args, _ in harness.calls))

    def test_incomplete_existing_uv_directory_is_rejected_without_installing(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = BootstrapHarness(directory)
            harness.uv_binary.parents[1].mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "bootstrap uv está incompleto"):
                harness.execute_cell()
            self.assertEqual(harness.calls, [])
            self.assertIs(harness.namespace["GPU_READY"], False)
            self.assertIs(harness.namespace["DATA_READY"], False)

    def test_wrong_local_uv_is_rejected_before_python_or_cuda_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = BootstrapHarness(directory, uv_version="0.12.23")
            with self.assertRaisesRegex(RuntimeError, "versión local de uv"):
                harness.execute_cell()
            self.assertEqual(len(harness.calls), 2)
            self.assertIs(harness.namespace["GPU_READY"], False)
            self.assertIs(harness.namespace["DATA_READY"], False)

    def test_missing_uv_executable_after_bootstrap_does_not_enable_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = BootstrapHarness(directory, install_uv=False)
            with self.assertRaisesRegex(FileNotFoundError, "uv executable missing"):
                harness.execute_cell()
            self.assertIs(harness.namespace["GPU_READY"], False)
            self.assertIs(harness.namespace["DATA_READY"], False)

    def test_managed_python_outside_the_workspace_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = BootstrapHarness(directory, managed_outside=True)
            with self.assertRaisesRegex(RuntimeError, "fuera del directorio gestionado"):
                harness.execute_cell()
            self.assertFalse(harness.python.exists())
            self.assertIs(harness.namespace["GPU_READY"], False)
            self.assertIs(harness.namespace["DATA_READY"], False)

    def test_wrong_venv_python_is_rejected_before_installing_cuda_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = BootstrapHarness(directory, python_version=(3, 13, 9))
            with self.assertRaisesRegex(RuntimeError, "venv no usa CPython 3.12.12"):
                harness.execute_cell()
            self.assertFalse(any("requirements-e2-training-gpu.lock.txt" in " ".join(args) for args, _ in harness.calls))
            self.assertIs(harness.namespace["GPU_READY"], False)
            self.assertIs(harness.namespace["DATA_READY"], False)

    def test_bootstrap_and_download_failures_clear_both_previous_ready_flags(self):
        for failure, message in (("bootstrap", "bootstrap download failed"),
                                 ("python-download", "managed Python download failed"),
                                 ("pip", "pinned pip install failed")):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                harness = BootstrapHarness(directory, failure=failure)
                with self.assertRaisesRegex(RuntimeError, message):
                    harness.execute_cell()
                self.assertIs(harness.namespace["GPU_READY"], False)
                self.assertIs(harness.namespace["DATA_READY"], False)

    def test_bootstrap_requires_hardware_admission_before_any_download(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = BootstrapHarness(directory)
            harness.namespace["HARDWARE_READY"] = False
            with self.assertRaisesRegex(RuntimeError, "Comprueba primero Linux y la GPU"):
                harness.execute_cell()
            self.assertEqual(harness.calls, [])
            self.assertIs(harness.namespace["GPU_READY"], False)
            self.assertIs(harness.namespace["DATA_READY"], False)

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
