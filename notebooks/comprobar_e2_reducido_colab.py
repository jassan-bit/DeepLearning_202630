"""Short Colab diagnostics; no installation, checkout, market data or training.

The release cell supplies immutable source/checker hashes. The current kernel
may be3.13 while the already-prepared CUDA interpreter is3.12; inspect the latter
without changing or restarting the kernel. This file starts no controller.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.request

REPOSITORY = "jassan-bit/DeepLearning_202630"


def choose_python(namespace=None):
    namespace = {} if namespace is None else namespace
    candidates = [namespace.get("GPU_PYTHON"),
        "/content/entregable2_gpu_workspace/venv-py312-cu126/bin/python",
        "/content/entregable2_gpu_workspace/pilot-python312/bin/python", sys.executable]
    for candidate in candidates:
        if candidate is not None:
            path = Path(candidate)
            if path.is_file() and os.access(path, os.X_OK):
                return str(path), str(path) != sys.executable
    raise RuntimeError("No existe un intérprete accesible; no se instaló ni cambió el kernel.")


def download_pinned(source_commit, relative, expected_sha256):
    if (not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", source_commit)
            or not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)):
        raise ValueError("Usa el commit y SHA256 inmutables publicados para esta comprobación.")
    allowed = ("scripts/check_e2_reduced_environment.py",)
    if relative not in allowed and not re.fullmatch(r"experiments/e2_reduced_protocol_20261112/[A-Za-z0-9_.-]+\.json", relative):
        raise ValueError("La comprobación sólo admite su script y recibos públicos fijados.")
    url = "https://raw.githubusercontent.com/" + REPOSITORY + "/" + source_commit + "/" + relative
    with urllib.request.urlopen(url, timeout=30) as response:
        raw = response.read(1048577)
    if len(raw) > 1048576 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("La descarga no coincide con sus bytes registrados; no se ejecutó.")
    return raw


def run_colab_check(source_commit, checker_sha256, *, namespace=None,
                    budget_relative_path=None, budget_sha256=None, ledger=None, output_directory=None):
    """Run only the pinned diagnostic in an existing Python environment."""
    raw = download_pinned(source_commit, "scripts/check_e2_reduced_environment.py", checker_sha256)
    python, isolated = choose_python(namespace)
    drive = Path("/content/drive/MyDrive")
    if output_directory is None:
        directory = (drive / "DeepLearning_202630/E2/reduced_protocol/runtime" if drive.is_dir()
            else Path("/content/entregable2_gpu_workspace/diagnostics/reduced_protocol"))
    else:
        directory = Path(output_directory)
    if directory.is_symlink():
        raise ValueError("La salida de diagnóstico no puede ser un enlace simbólico.")
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    script = directory / ("checker-" + checker_sha256 + ".py")
    if script.exists():
        if script.read_bytes() != raw:
            raise ValueError("El verificador local cambió; conserva el archivo para revisar.")
    else:
        with script.open("xb") as stream:
            stream.write(raw)
    report = directory / ("runtime-" + stamp + ".json")
    args = [python, str(script), "--output", str(report)]
    if budget_relative_path is not None:
        budget = directory / ("budget-" + str(budget_sha256) + ".json")
        data = download_pinned(source_commit, budget_relative_path, budget_sha256)
        if budget.exists() and budget.read_bytes() != data:
            raise ValueError("El recibo presupuestario local cambió; no se reemplazó.")
        if not budget.exists():
            with budget.open("xb") as stream:
                stream.write(data)
        args.extend(["--budget-evidence", str(budget)])
    if ledger is not None:
        args.extend(["--ledger", str(ledger)])
    env = dict(os.environ)
    for name in ("GITHUB_TOKEN", "GH_TOKEN", "PYTHONPATH"):
        env.pop(name, None)
    env.update(CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONNOUSERSITE="1")
    print("Kernel:", sys.version.split()[0], "; comprobación sin entrenamiento con Python:", python,
          "; intérprete distinto del kernel:", isolated, flush=True)
    result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=120)
    if result.stderr:
        print(result.stderr[-3000:], flush=True)
    if not report.is_file():
        raise RuntimeError("No se generó el informe. Código: " + str(result.returncode))
    value = json.loads(report.read_text())
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)
    print("Informe:", report, "; conservado en Drive:", report.resolve().is_relative_to(drive.resolve()), flush=True)
    return {"exit_code": result.returncode, "report": str(report), "runtime": value.get("runtime"),
        "budget": value.get("budget"), "training_started": False, "budget_extended": False,
        "source_commit": source_commit, "checker_sha256": checker_sha256,
        "kernel_python": sys.version.split()[0], "kernel_python_executable": sys.executable,
        "diagnostic_python_executable": python}
