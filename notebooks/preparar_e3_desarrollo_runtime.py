"""Prepare an isolated Colab E3 runtime without loading data or fitting models.

Usage from the notebook::

    python, environment, receipt = preparar_entorno(local_workspace, device="cuda")

The returned interpreter runs the CLI in a subprocess. No kernel restart is
needed. Keep ``local_workspace`` on Colab's local disk; only experiment outputs
belong on Drive. This helper never imports an E2 launcher or changes a budget.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys


UV_VERSION = "0.9.5"
PYTHON_VERSION = "3.12.12"
PIP_VERSION = "25.3"
UV_WHEEL_SHA256 = "6507bbbcd788553ec4ad5a96fa19364dc0f58b023e31d79868773559a83ec181"
PIP_WHEEL_SHA256 = "9655943313a94722b7774661c21049070f6bbb0a1516bf02f7c8d5d9201514cd"
LOCK_NAME = "requirements-e3-development-gpu.lock.txt"


def _run(args, env, *, capture=False):
    return subprocess.run([str(value) for value in args], env=env, check=True,
                          text=True, capture_output=capture)


def _environment(workspace):
    environment = dict(os.environ)
    # Prevent the notebook's package paths/configuration from entering the venv.
    for key in list(environment):
        if key.startswith(("PIP_", "UV_")) or key in {
            "GITHUB_TOKEN", "GH_TOKEN", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"
        }:
            environment.pop(key, None)
    environment.update(
        CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONNOUSERSITE="1",
        OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
        PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK="1",
        UV_NATIVE_TLS="true", UV_NO_CONFIG="1",
        UV_CACHE_DIR=str(workspace / "uv-cache"),
        UV_PYTHON_INSTALL_DIR=str(workspace / "managed-python"),
        UV_PYTHON_BIN_DIR=str(workspace / "managed-python-bin"),
    )
    return environment


def _expected_packages(lock):
    expected = {}
    for line in lock.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+)", line.strip())
        if match:
            expected[match.group(1)] = match.group(2)
    if expected.get("torch") != "2.8.0+cu126" or expected.get("numpy") != "2.3.5":
        raise RuntimeError("El lock de desarrollo no contiene las versiones esperadas.")
    return expected


@contextmanager
def _setup_lock(workspace):
    with (workspace / ".runtime-setup.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Ya hay otra preparación de este entorno en curso.") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _bootstrap_python(workspace, env):
    bootstrap = workspace / "bootstrap"
    bootstrap.mkdir(exist_ok=True)
    uv_home = bootstrap / ("uv-" + UV_VERSION)
    uv_binary = uv_home / "bin/uv"
    uv_requirements = bootstrap / "uv.requirements.txt"
    uv_requirements.write_text(
        f"uv=={UV_VERSION} --hash=sha256:{UV_WHEEL_SHA256}\n", encoding="utf-8")
    if not uv_binary.is_file():
        if uv_home.exists():
            raise RuntimeError("El bootstrap uv está incompleto; se conserva en " + str(uv_home))
        _run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--no-deps",
              "--only-binary=:all:", "--require-hashes", "--index-url", "https://pypi.org/simple",
              "--target", uv_home, "-r", uv_requirements], env)
    version = _run([uv_binary, "--version"], env, capture=True).stdout.strip().split()
    if version[:2] != ["uv", UV_VERSION]:
        raise RuntimeError("El ejecutable uv no coincide con la versión fijada.")
    managed = workspace / "managed-python"
    _run([uv_binary, "python", "install", PYTHON_VERSION, "--install-dir", managed,
          "--no-bin", "--no-config"], env)
    executable = Path(_run(
        [uv_binary, "python", "find", PYTHON_VERSION, "--managed-python",
         "--no-python-downloads", "--no-project", "--no-config"], env, capture=True
    ).stdout.strip())
    if managed.resolve() not in executable.resolve().parents:
        raise RuntimeError("uv eligió un intérprete fuera del directorio aislado.")
    venv = workspace / "venv-py312-cu126"
    if venv.exists():
        raise RuntimeError("Ya existe un entorno parcial; se conserva en " + str(venv))
    _run([uv_binary, "venv", "--python", executable, "--no-python-downloads",
          "--no-project", "--no-config", venv], env)
    python = venv / "bin/python"
    pip_requirements = bootstrap / "pip.requirements.txt"
    pip_requirements.write_text(
        f"pip=={PIP_VERSION} --hash=sha256:{PIP_WHEEL_SHA256}\n", encoding="utf-8")
    _run([uv_binary, "pip", "install", "--python", python, "--no-deps",
          "--only-binary=:all:", "--require-hashes", "--index-url", "https://pypi.org/simple",
          "--no-config", "-r", pip_requirements], env)
    return python


def _probe(python, env, expected, device):
    code = """
import importlib.metadata as metadata
import json
import platform
import sys
import torch
expected = EXPECTED
assert tuple(sys.version_info[:3]) == (3, 12, 12), 'Python incompatible'
for name, version in expected.items():
    assert metadata.version(name) == version, 'Paquete incompatible: ' + name
assert metadata.version('pip') == '25.3', 'pip incompatible'
assert str(torch.__version__) == '2.8.0+cu126', 'Torch incompatible'
assert torch.version.cuda == '12.6', 'CUDA de Torch incompatible'
requested = DEVICE
if requested == 'cuda':
    assert torch.cuda.is_available(), 'No hay GPU CUDA disponible; revise el tipo de entorno Colab'
devices = []
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append({'index': index, 'name': properties.name,
                        'total_memory_bytes': properties.total_memory,
                        'capability': list(torch.cuda.get_device_capability(index))})
print(json.dumps({'python': platform.python_version(), 'torch': str(torch.__version__),
                  'numpy': metadata.version('numpy'), 'torch_cuda': torch.version.cuda,
                  'cudnn': torch.backends.cudnn.version(), 'cuda_available': torch.cuda.is_available(),
                  'requested_device': requested, 'selected_device': 'cuda:0' if requested == 'cuda' else 'cpu',
                  'devices': devices, 'packages': expected}))
""".replace("EXPECTED", repr(expected)).replace("DEVICE", repr(device))
    return json.loads(_run([python, "-c", code], env, capture=True).stdout)


def preparar_entorno(workspace, device="cuda"):
    """Return ``(python, environment, runtime)``; never read data or train.

    ``device`` is ``cuda`` (GPU required) or an explicitly chosen ``cpu``. A
    complete existing environment is verified and reused. A partial or altered
    environment is preserved and rejected; choose another *local runtime*
    folder if needed, keeping the same experiment-output folder on Drive.
    """
    if device not in {"cuda", "cpu"}:
        raise ValueError("device debe ser 'cuda' o 'cpu'.")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Esta preparación requiere Linux x86_64, como Colab.")
    workspace = Path(workspace).expanduser().absolute()
    if any(part.is_symlink() for part in (workspace, *workspace.parents)):
        raise RuntimeError("La carpeta local del entorno no puede ser un enlace simbólico.")
    if "drive" in {part.lower() for part in workspace.parts}:
        raise RuntimeError("El entorno debe estar en disco local, no en Drive.")
    lock = Path(__file__).resolve().parents[1] / LOCK_NAME
    expected = _expected_packages(lock)
    identity = {"python": PYTHON_VERSION, "pip": PIP_VERSION, "uv": UV_VERSION,
                "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest()}
    workspace.mkdir(parents=True, exist_ok=True)
    env = _environment(workspace)
    with _setup_lock(workspace):
        if device == "cuda":
            try:
                visible_gpu = _run(
                    ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                    env, capture=True).stdout.strip()
            except (OSError, subprocess.CalledProcessError) as error:
                raise RuntimeError("Colab no muestra una GPU. Seleccione Entorno de ejecución > "
                                   "Cambiar tipo de entorno > GPU y ejecute esta celda de nuevo.") from error
            if not visible_gpu:
                raise RuntimeError("nvidia-smi no encontró una GPU disponible.")
            print("GPU visible:", visible_gpu, flush=True)
        python = workspace / "venv-py312-cu126/bin/python"
        receipt_path = workspace / "runtime_identity.json"
        if receipt_path.exists():
            try:
                matching_receipt = json.loads(receipt_path.read_text(encoding="utf-8")) == identity
            except (OSError, ValueError):
                matching_receipt = False
            if not matching_receipt:
                raise RuntimeError("El registro del entorno pertenece a otro lock o está dañado; "
                                   "se conserva sin sobrescribir: " + str(workspace))
        if python.is_file():
            if not receipt_path.is_file():
                raise RuntimeError("El entorno existente está incompleto o pertenece a otro lock; "
                                   "se conserva sin sobrescribir: " + str(python.parent.parent))
        else:
            print("Preparando Python 3.12.12 / Torch 2.8.0+cu126 en disco local. "
                  "La primera instalación puede tardar varios minutos.", flush=True)
            python = _bootstrap_python(workspace, env)
            _run([python, "-m", "pip", "install", "--only-binary=:all:", "-r", lock], env)
        _run([python, "-m", "pip", "check"], env)
        runtime = _probe(python, env, expected, device)
        if not receipt_path.exists():
            temporary = workspace / "runtime_identity.json.tmp"
            temporary.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, receipt_path)
    runtime.update(identity)
    runtime.update(checked_at_utc=datetime.now(timezone.utc).isoformat(),
                   kernel_python=platform.python_version(), training_started=False)
    print("Entorno aislado verificado:", json.dumps(
        {key: value for key, value in runtime.items() if key != "packages"}, ensure_ascii=False), flush=True)
    return python, env, runtime
