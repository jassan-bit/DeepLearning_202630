"""One Colab call exports already cached E2 diagnostics inputs; no GPU work."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.request

ENGINE_COMMIT = "1e8745497b484518f766b6da5c19be8362d9add7"
LOCAL_WORKSPACE = Path("/content/entregable2_gpu_workspace")
REGISTRATION_PATH = "experiments/e2_f2a_audit_20261113/registration.json"
FILES = ("scripts/export_e2_f2a_cached_data.py", "notebooks/export_e2_f2a_cached_colab.py")
EXISTING_PYTHON = LOCAL_WORKSPACE / "reduced-runtime-py312-cu126-v1/venv-py312-cu126/bin/python"


def _download(commit, relative):
    url = "https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/" + commit + "/" + relative
    with urllib.request.urlopen(url, timeout=40) as response:
        raw = response.read(4 * 1024**2 + 1)
    if len(raw) > 4 * 1024**2:
        raise RuntimeError("El archivo de exportación excede su límite")
    return raw


def _write_once(path, payload):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise RuntimeError("La exportación no admite enlaces simbólicos")
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError("Se conserva una fuente de exportación diferente para revisión")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def _materialize(commit):
    registration_raw = _download(commit, REGISTRATION_PATH)
    registration = json.loads(registration_raw)
    hashes = registration.get("files_sha256", {})
    if (registration.get("engine_commit") != ENGINE_COMMIT
            or registration.get("protocol_id") != "e2-reduced-delivery-v1"
            or not set(FILES).issubset(hashes)):
        raise RuntimeError("El registro no corresponde al exportador de la campaña existente")
    payloads = {}
    for relative in FILES:
        payload = _download(commit, relative)
        if hashlib.sha256(payload).hexdigest() != hashes[relative]:
            raise RuntimeError("La fuente de exportación no coincide con su registro")
        payloads[relative] = payload
    root = LOCAL_WORKSPACE / ("f2a-readonly-source-" + commit)
    for relative, payload in payloads.items():
        _write_once(root / relative, payload)
    _write_once(root / "registration.json", registration_raw)
    return root


def run_e2_f2a_cached_export_colab(commit, *, namespace=None, execute=False):
    """Reuse the existing Python 3.12, source checkout, daily cache and token."""
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Usa el commit inmutable del exportador F2a")
    if type(execute) is not bool:
        raise TypeError("execute debe ser booleano")
    if not execute:
        return {"status": "dry_run", "scientific_fit": False, "model_inference": False,
            "benchmark_repeated": False, "runtime_installation": False,
            "restore_requested": False, "requested_action": "export_cached_derived_inputs"}
    namespace = globals() if namespace is None else namespace
    python = Path(namespace.get("GPU_PYTHON") or EXISTING_PYTHON)
    source = LOCAL_WORKSPACE / ("reduced-source-" + ENGINE_COMMIT)
    output = LOCAL_WORKSPACE / "github-primary-output-v1"
    if not python.is_file() or not source.is_dir() or not (output / "final").is_dir():
        raise RuntimeError("Faltan el entorno o los archivos originales; este exportador no instala ni restaura")
    # This secret stays in the notebook process and subprocess stdin only.
    token = namespace.get("GITHUB_TOKEN", "")
    if not token:
        from google.colab import userdata
        token = userdata.get("GITHUB_TOKEN") or ""
    if not isinstance(token, str) or not token or any(c.isspace() for c in token):
        raise RuntimeError("Activa el secreto GITHUB_TOKEN existente para esta sesión")
    root = _materialize(commit)
    command = [str(python), "-B", str(root / FILES[0]),
        "--source-root", str(source), "--output-root", str(output),
        "--destination", str(LOCAL_WORKSPACE / "f2a-readonly-export-v1")]
    env = dict(os.environ)
    for key in ("GITHUB_TOKEN", "GH_TOKEN", "PYTHONPATH"):
        env.pop(key, None)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    completed = subprocess.run(command, input=token + "\n", text=True, capture_output=True, env=env)
    if completed.returncode:
        # Never echo a child environment, command payload, market value or secret.
        reason = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else ""
        if token in reason:
            reason = ""
        allowed = ("ValueError:", "FileNotFoundError:", "BackupPause:")
        detail = reason if reason.startswith(allowed) else "falló una validación de archivos existentes"
        raise RuntimeError("La exportación se detuvo sin repetir el benchmark: " + detail)
    try:
        result = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise RuntimeError("El exportador no devolvió un recibo de respaldo verificable") from None
    if result.get("status") != "backed_up":
        raise RuntimeError("El exportador no confirmó el respaldo de los datos derivados")
    return {**result, "scientific_fit": False, "model_inference": False,
        "benchmark_repeated": False, "export_source_commit": commit}
