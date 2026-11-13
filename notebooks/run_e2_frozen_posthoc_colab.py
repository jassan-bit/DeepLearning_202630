"""Run bounded frozen-predictor diagnostics in the preserved Colab A100 session."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import urllib.request

ENGINE_COMMIT = "1e8745497b484518f766b6da5c19be8362d9add7"
WORKSPACE = Path("/content/entregable2_gpu_workspace")
EXISTING_PYTHON = WORKSPACE / "reduced-runtime-py312-cu126-v1/venv-py312-cu126/bin/python"
REGISTRATION = "experiments/e2_frozen_posthoc_20261113/registration.json"
FILES = ("scripts/e2_frozen_posthoc_gpu.py", "scripts/export_e2_f2a_cached_data.py",
         "scripts/e2_posthoc_fit_reproduction.py",
         "notebooks/run_e2_frozen_posthoc_colab.py")


def _download(commit, relative):
    url = "https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/"+commit+"/"+relative
    with urllib.request.urlopen(url, timeout=40) as response:
        raw = response.read(4*1024**2+1)
    if len(raw) > 4*1024**2:
        raise ValueError("Registered source exceeds the metadata limit")
    return raw


def _write_once(path, raw):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Posthoc source must not use symlinks")
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("Existing posthoc source differs; preserved for review")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def materialize(commit):
    registration_raw = _download(commit, REGISTRATION)
    registration = json.loads(registration_raw)
    hashes = registration.get("files_sha256", {})
    if (registration.get("engine_commit") != ENGINE_COMMIT
            or registration.get("protocol_id") != "e2-reduced-delivery-v1"
            or registration.get("training_performed") is not False
            or not set(FILES).issubset(hashes)):
        raise ValueError("Registered diagnostics do not match the preserved E2 campaign")
    payloads = {relative: _download(commit, relative) for relative in FILES}
    for relative, raw in payloads.items():
        if hashlib.sha256(raw).hexdigest() != hashes[relative]:
            raise ValueError("Posthoc source SHA256 differs from registration")
    root = WORKSPACE / ("frozen-posthoc-source-"+commit)
    for relative, raw in payloads.items():
        _write_once(root / relative, raw)
    _write_once(root / "registration.json", registration_raw)
    return root


def run_e2_frozen_posthoc_colab(commit, *, namespace=None, execute=False, wall_seconds=5400):
    """No install, restore or training; stream heartbeat lines to the notebook."""
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Use the immutable diagnostics source commit")
    if type(execute) is not bool or type(wall_seconds) is not int or not 60 <= wall_seconds <= 5400:
        raise ValueError("Explicit execute boolean and wall cap 60..5400 seconds required")
    if not execute:
        return {"status": "dry_run", "training_performed": False,
                "model_inference": True, "batch_size": 1, "wall_cap_seconds": wall_seconds,
                "persistent_destination": "github", "original_score_modified": False,
                "requested_action": "frozen_predictor_posthoc_diagnostics"}
    raise ValueError("Use la celda del lanzador unificado de cierre E2: la ejecución independiente está deshabilitada para conservar un único presupuesto remoto")
    namespace = globals() if namespace is None else namespace
    python = Path(namespace.get("GPU_PYTHON") or EXISTING_PYTHON)
    source = WORKSPACE / ("reduced-source-"+ENGINE_COMMIT)
    output = WORKSPACE / "github-primary-output-v1"
    if not python.is_file() or not source.is_dir() or not (output / "final").is_dir():
        raise ValueError("Preserved original runtime/source/cache/weights required; no installation or restore")
    token = namespace.get("GITHUB_TOKEN", "")
    if not token:
        from google.colab import userdata
        token = userdata.get("GITHUB_TOKEN") or ""
    if not isinstance(token, str) or not token or any(c.isspace() for c in token):
        raise ValueError("Enable the existing GITHUB_TOKEN secret for this notebook")
    root = materialize(commit)
    destination = WORKSPACE / "e2-guide-closure-posthoc-v1/robustness"
    command = [str(python), "-B", str(root / FILES[0]), "--source-root", str(source),
               "--output-root", str(output), "--destination", str(destination),
               "--wall-seconds", str(wall_seconds)]
    env = dict(os.environ)
    for key in ("GITHUB_TOKEN", "GH_TOKEN", "PYTHONPATH"):
        env.pop(key, None)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    # Stdout is inherited to keep model/seed/perturbation progress visible.
    # Credentials travel only over the private stdin pipe; stderr is sanitized.
    with tempfile.TemporaryFile(mode="w+t") as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors, text=True, env=env)
        try:
            process.communicate(token+"\n", timeout=wall_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise RuntimeError("Posthoc wall cap reached; preserve partial receipts and reconcile reserved cost") from None
        if process.returncode:
            errors.seek(0)
            lines = errors.read().strip().splitlines()
            reason = lines[-1] if lines else ""
            permitted = ("ValueError:", "FileNotFoundError:", "TimeoutError:", "BackupPause:")
            detail = reason if reason.startswith(permitted) and token not in reason else "validation or bounded diagnostics stopped"
            raise RuntimeError("Original score preserved; inspect completed posthoc backups: "+detail)
    receipt = json.loads((destination / "github_backup_receipt.json").read_bytes())
    if receipt.get("status") != "backed_up":
        raise RuntimeError("A completed GitHub backup receipt is required")
    return {k: receipt[k] for k in ("status", "attempt_id", "remote_commit", "manifest_path")}
