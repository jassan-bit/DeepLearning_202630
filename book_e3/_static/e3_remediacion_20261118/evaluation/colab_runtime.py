"""Prepare an isolated CUDA runtime using the existing pinned E2 bootstrap.

Only environment preparation occurs here. No market data, model, fitting,
controller, budget change or credential is required. The notebook's Python and
Torch stay unchanged; the returned Python executes the synthetic preflight.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import urllib.request

SOURCE_COMMIT = "79f4ee4719346a9c00d795ed6f030ec69a2f0203"
HELPER_PATH = "notebooks/run_e2_pilots_colab.py"
HELPER_SHA256 = "bd65ecfe77ba017f5610404f038b2d6edf1dd3468163abb481c9cf355575551e"
LOCK_PATH = "requirements-e2-training-gpu.lock.txt"
LOCK_SHA256 = "5201f687d8c600ae9458444dbc46e5bc97bdedf938ddb21801887f6f9d4508d9"
REPOSITORY_RAW = "https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/"


def _bytes(relative, expected):
    url = REPOSITORY_RAW + SOURCE_COMMIT + "/" + relative
    with urllib.request.urlopen(url, timeout=45) as response:
        raw = response.read(131073)
    if len(raw) > 131072 or hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError("Pinned environment source differs: " + relative)
    return raw


def _environment(workspace):
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("PIP_", "UV_")) or key in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"} or re.search(r"(?:^|_)(?:TOKEN|PAT|SECRET|PASSWORD)(?:_|$)", key.upper()):
            env.pop(key, None)
    env.update(CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONNOUSERSITE="1", OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PIP_CONFIG_FILE=os.devnull,
        PIP_DISABLE_PIP_VERSION_CHECK="1", UV_NATIVE_TLS="true", UV_NO_CONFIG="1",
        UV_CACHE_DIR=str(workspace / "uv-cache"), UV_PYTHON_INSTALL_DIR=str(workspace / "managed-python"),
        UV_PYTHON_BIN_DIR=str(workspace / "managed-python-bin"))
    return env


def _save_exact(path, raw):
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError("Preserved source changed; choose a new runtime directory: " + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)


def _expected(lock):
    packages = {}
    for line in lock.decode().splitlines():
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+)", line.strip())
        if match:
            packages[match.group(1)] = match.group(2)
    if packages.get("torch") != "2.8.0+cu126" or packages.get("statsmodels") != "0.14.5" or packages.get("reformer-pytorch") != "1.4.4":
        raise RuntimeError("The reviewed complete E2 CUDA profile is required")
    return packages


def prepare_runtime(workspace):
    """Return (isolated_python, sanitized_environment, receipt); GPU required.

    Works even when the notebook kernel runs Python 3.13: the existing bootstrap
    obtains its own CPython 3.12.12. No base-kernel pip install or restart is used.
    GPU wheels can require several GB and several minutes on first preparation.
    """
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Use a Linux x86_64 Colab GPU runtime")
    workspace = Path(workspace).expanduser().absolute()
    if any(part.is_symlink() for part in (workspace, *workspace.parents)):
        raise RuntimeError("The isolated runtime directory cannot traverse a symlink")
    if "drive" in {p.lower() for p in workspace.parts}:
        raise RuntimeError("Use Colab local /content storage for the runtime")
    env = _environment(workspace)
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], env=env, text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("No hay GPU visible. En Colab: Entorno de ejecución > Cambiar tipo > GPU > T4 (o disponible). No se descargaron wheels CUDA.") from exc
    if not gpu:
        raise RuntimeError("nvidia-smi returned no GPU; CUDA package installation was not started")
    workspace.mkdir(parents=True, exist_ok=True)
    # The existing helper only defines functions when imported under this name.
    helper = _bytes(HELPER_PATH, HELPER_SHA256)
    lock = _bytes(LOCK_PATH, LOCK_SHA256)
    expected = _expected(lock)
    source = workspace / ("pinned-source-" + SOURCE_COMMIT)
    _save_exact(source / HELPER_PATH, helper)
    _save_exact(source / LOCK_PATH, lock)
    module = {"__name__": "e3_preflight_existing_E2_environment_functions_only"}
    exec(compile(helper, str(source / HELPER_PATH), "exec"), module)
    module["_assert_no_controller"]()
    isolated_python = workspace / "venv-py312-cu126/bin/python"
    receipt_path = workspace / "e3_preflight_runtime_receipt.json"
    if isolated_python.is_file():
        if not receipt_path.is_file():
            raise RuntimeError("Existing partial runtime preserved; choose a new local runtime folder: " + str(workspace))
        previous = json.loads(receipt_path.read_text())
        if previous.get("lock_sha256") != LOCK_SHA256 or previous.get("helper_sha256") != HELPER_SHA256:
            raise RuntimeError("Existing runtime belongs to a different lock; it was not altered")
    elif isolated_python.parent.parent.exists():
        raise RuntimeError("Partial runtime preserved; choose a new local runtime folder: " + str(workspace))
    print("GPU visible:", gpu, flush=True)
    print("Preparando un Python 3.12.12 aislado con el perfil E2 CUDA completo. La primera descarga de wheels puede sumar varios GB y tardar varios minutos. El Torch base de Colab no cambia.", flush=True)
    if not isolated_python.is_file():
        isolated_python = module["_bootstrap_python"](workspace, env)
    runtime = module["_gpu_environment"](isolated_python, source, env)
    # Recheck every exact package after installation, including author imports.
    code = ("import importlib.metadata as m,json,sys\nexpected=" + repr(expected)
        + "\nassert sys.version_info[:3]==(3,12,12)\n"
          "mismatch={k:(m.version(k),v) for k,v in expected.items() if m.version(k)!=v}\n"
          "assert not mismatch,mismatch\n"
          "import reformer_pytorch,einops,statsmodels\n"
          "print(json.dumps({'python':sys.executable,'packages_verified':len(expected)}))")
    verified = json.loads(subprocess.check_output([str(isolated_python), "-c", code], env=env, text=True))
    receipt = {"source_commit": SOURCE_COMMIT, "helper_path": HELPER_PATH, "helper_sha256": HELPER_SHA256,
        "lock_path": LOCK_PATH, "lock_sha256": LOCK_SHA256, "isolated_python": str(isolated_python),
        "kernel_python": sys.executable, "kernel_python_version": platform.python_version(),
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "runtime": runtime,
        "package_check": verified, "base_Torch_changed": False, "kernel_restarted": False,
        "models_executed": False, "market_data_read": False, "training_started": False}
    if not receipt_path.exists():
        _save_exact(receipt_path, (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode())
    return Path(isolated_python), env, receipt
