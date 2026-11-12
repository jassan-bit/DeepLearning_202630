"""Prepare only the isolated, pinned E2 CUDA environment and recheck it.

Run with exec(..., globals()) in Colab. The host kernel is not modified or
restarted. No repository checkout, market file, optimizer, fit or budget
extension is requested. GPU_PYTHON is set only after environment verification.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import urllib.request

_E2_ENV_SOURCE_COMMIT = "79f4ee4719346a9c00d795ed6f030ec69a2f0203"
_E2_ENV_PINNED_FILES = {
    "notebooks/run_e2_pilots_colab.py": "bd65ecfe77ba017f5610404f038b2d6edf1dd3468163abb481c9cf355575551e",
    "requirements-e2-training-gpu.lock.txt": "5201f687d8c600ae9458444dbc46e5bc97bdedf938ddb21801887f6f9d4508d9",
    "notebooks/comprobar_e2_reducido_celda.py": "dd2952d42c1c388d0ebbd260566ca9f4df8568f82d6dc98f74d5b6f747b078c2",
}


def _e2_env_download(relative):
    expected = _E2_ENV_PINNED_FILES[relative]
    url = ("https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/"
           + _E2_ENV_SOURCE_COMMIT + "/" + relative)
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read(131073)
    if len(data) > 131072 or hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError("Los bytes de la preparación no coinciden con la fuente fijada: " + relative)
    return data


def _e2_env_existing_matches(python, lock, env):
    expected = {}
    for line in lock.decode().splitlines():
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+)", line.strip())
        if match:
            expected[match.group(1)] = match.group(2)
    code = ("import importlib.metadata as m,json,sys\nexpected=" + repr(expected)
        + "\nmissing=[]\nfor name,version in expected.items():\n"
          " try:\n  actual=m.version(name)\n except m.PackageNotFoundError:\n  actual=None\n"
          " if actual!=version:missing.append(name)\n"
          "print(json.dumps({'python':list(sys.version_info[:3]),'mismatched':missing}))")
    result = subprocess.run([str(python), "-c", code], env=env, capture_output=True, text=True, check=True)
    value = json.loads(result.stdout)
    if value["python"] != [3, 12, 12] or value["mismatched"]:
        raise RuntimeError("El entorno aislado existente está incompleto o es incompatible; se conserva sin sobrescribir: "
            + str(python.parent.parent) + ". Python=" + str(value["python"])
            + "; paquetes distintos=" + str(value["mismatched"]))


def preparar_entorno_e2_reducido(namespace, *, workspace=None):
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Esta preparación CUDA requiere Colab Linux x86_64.")
    helper = _e2_env_download("notebooks/run_e2_pilots_colab.py")
    lock = _e2_env_download("requirements-e2-training-gpu.lock.txt")
    diagnostic = _e2_env_download("notebooks/comprobar_e2_reducido_celda.py")
    bootstrap = {"__name__": "e2_isolated_environment_only"}
    exec(compile(helper, "e2_pinned_environment_functions", "exec"), bootstrap)
    bootstrap["_assert_no_controller"]()
    # A fresh versioned destination preserves the old kernel and any prior venv.
    workspace = Path(workspace or "/content/entregable2_gpu_workspace/reduced-runtime-py312-cu126-v1")
    if any(part.is_symlink() for part in (workspace, *workspace.parents)):
        raise RuntimeError("La carpeta del entorno no puede ser un enlace simbólico.")
    env = dict(os.environ)
    for key in ("GITHUB_TOKEN", "GH_TOKEN", "PYTHONPATH"):
        env.pop(key, None)
    env.update(CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONNOUSERSITE="1", OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", UV_NATIVE_TLS="true",
        UV_CACHE_DIR=str(workspace / "uv-cache"), UV_PYTHON_INSTALL_DIR=str(workspace / "managed-python"),
        UV_PYTHON_BIN_DIR=str(workspace / "managed-python-bin"))
    python = workspace / "venv-py312-cu126/bin/python"
    if python.is_file():
        _e2_env_existing_matches(python, lock, env)
    elif python.parent.parent.exists():
        raise RuntimeError("Ya existe un entorno parcial; se conserva sin sobrescribir: " + str(python.parent.parent))
    workspace.mkdir(parents=True, exist_ok=True)
    source = workspace / ("pinned-source-" + _E2_ENV_SOURCE_COMMIT)
    source.mkdir(exist_ok=True)
    lock_path = source / "requirements-e2-training-gpu.lock.txt"
    if lock_path.exists():
        if lock_path.read_bytes() != lock:
            raise RuntimeError("El lock local cambió; no se reemplazó.")
    else:
        with lock_path.open("xb") as stream:
            stream.write(lock)
    print("Preparando Python3.12.12 / Torch2.8.0+cu126 en una carpeta aislada. El kernel actual permanece abierto.", flush=True)
    python = bootstrap["_bootstrap_python"](workspace, env)
    runtime = bootstrap["_gpu_environment"](python, source, env)
    namespace["GPU_PYTHON"] = str(python)
    print("Entorno aislado preparado. Comprobando otra vez la GPU y el saldo registrado; no se inicia entrenamiento.", flush=True)
    exec(compile(diagnostic, "e2_registered_runtime_recheck", "exec"), namespace)
    result = {"GPU_PYTHON": str(python), "kernel_python": sys.version.split()[0],
        "runtime": runtime, "source_commit": _E2_ENV_SOURCE_COMMIT,
        "training_started": False, "budget_extended": False, "kernel_restarted": False}
    namespace["E2_REDUCED_RUNTIME_PREPARATION"] = result
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    preparar_entorno_e2_reducido(globals())
