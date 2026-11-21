"""Hash-bound CPU setup. setup() downloads without a token and never trains.

The GitHub bridge takes private credentials through stdin only.
Training requires an explicit run action. No Render or base-Torch mutation.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import venv

REPO = "jassan-bit/DeepLearning_202630"
PREFIX = "experiments/e3_seoi_v5_pilot_20261121"
MAX_SOURCE_BYTES = 2 * 1024 * 1024
NPZ_URL = "https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/8f65eac34aaf49293924ae8328c235cedbdd2866/book_e2/_static/e2_scientific_closure_20261116/data_audit/current_cohort_arrays.npz"
NPZ_SHA256 = "186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683"


def require(ok, message):
    if not ok: raise RuntimeError(message)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def emit(message, **fields):
    print(message, json.dumps(fields, ensure_ascii=False) if fields else "", flush=True)


def download(url, expected_hash, *, max_bytes):
    require(bool(re.fullmatch(r"[0-9a-f]{64}", expected_hash)), "Expected SHA256 required")
    error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "E3-SE-OI-V5-public-source/1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read(max_bytes + 1)
            require(len(body) <= max_bytes, "Download exceeded frozen size bound")
            require(sha(body) == expected_hash, "Downloaded bytes do not match the pinned version")
            return body
        except urllib.error.HTTPError as exc:
            error = exc
            if exc.code not in (429, 500, 502, 503, 504): raise
            delay = min(30, 3 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError) as exc:
            error = exc
            delay = 3 * (attempt + 1)
        if attempt < 2:
            emit("Descarga pública temporalmente indisponible; reintento", attempt=attempt + 1, seconds=delay)
            time.sleep(delay)
    raise RuntimeError("Pinned download failed after three attempts: " + str(error))


def safe_member(value):
    require(isinstance(value, str) and value and "\\" not in value, "Invalid source member")
    p = PurePosixPath(value)
    require(not p.is_absolute() and str(p) == value and not any(part in (".", "..") for part in p.parts), "Unsafe source member")
    return value


def atomic_bytes(path, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), "Symlink destination refused")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".partial-", delete=False) as f:
        f.write(body)
        f.flush()
        os.fsync(f.fileno())
        temp = Path(f.name)
    os.replace(temp, path)


def acquire_source(source_commit, manifest_sha256, cache_root):
    require(bool(re.fullmatch(r"[0-9a-f]{40}", source_commit)), "A full immutable source commit is required")
    base = f"https://raw.githubusercontent.com/{REPO}/{source_commit}/{PREFIX}/"
    body = download(base + "SOURCE_MANIFEST.json", manifest_sha256, max_bytes=MAX_SOURCE_BYTES)
    manifest = json.loads(body)
    require(isinstance(manifest.get("files"), list) and manifest["files"], "Empty source manifest")
    root = Path(cache_root) / ("source-" + source_commit)
    require(not root.is_symlink(), "Source-root symlink refused")
    root.mkdir(parents=True, exist_ok=True)
    seen = set()
    for item in manifest["files"]:
        relative = safe_member(item["path"])
        require(relative not in seen, "Duplicate source manifest member")
        seen.add(relative)
        require(type(item["bytes"]) is int and 0 <= item["bytes"] <= MAX_SOURCE_BYTES, "Invalid declared source size")
        path = root / relative
        for parent in path.parents:
            if parent == root.parent: break
            require(not parent.is_symlink(), "Source directory symlink refused")
        if path.exists():
            require(not path.is_symlink() and path.is_file() and path.stat().st_size == item["bytes"]
                    and sha(path.read_bytes()) == item["sha256"], "Existing pinned source changed; preserve cache and use a fresh cache directory")
        else:
            file_body = download(base + relative, item["sha256"], max_bytes=item["bytes"])
            require(len(file_body) == item["bytes"], "Downloaded source size mismatch")
            atomic_bytes(path, file_body)
    required = {"seoi_v5/model.py", "seoi_v5/data.py", "seoi_v5/runner.py", "protocol.json", "requirements-cpu.lock.txt"}
    require(required <= seen, "Required runtime files missing from manifest")
    atomic_bytes(root / "SOURCE_MANIFEST.json", body)
    atomic_bytes(root / "PUBLIC_SOURCE_ANCHOR.json", (json.dumps({
        "source_commit": source_commit, "manifest_sha256": manifest_sha256,
        "source_readback_at_UTC": datetime.now(timezone.utc).isoformat(),
        "kind": "immutable public code and pilot specification; not a preregistration of old experiments",
    }, indent=2) + "\n").encode())
    return root


def cpu_environment():
    result = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
              "OPENBLAS_NUM_THREADS": "1", "PYTHONHASHSEED": "0", "MPLBACKEND": "Agg"}
    result.pop("PYTHONPATH", None)
    result.pop("PYTHONHOME", None)
    return result


def setup(*, source_commit, manifest_sha256, case_root,
          cache_root="/content/e3-seoi-v5-cache", install_runtime=True, prepare_case=True):
    """Download/check code/data, isolate CPU packages, prepare a case. Zero fits."""
    require((3, 11) <= sys.version_info[:2] <= (3, 13), "Use a Python3.11–3.13 Colab runtime")
    case = Path(case_root).expanduser().absolute()
    cache = Path(cache_root).expanduser().absolute()
    require(case != cache and cache not in case.parents and case not in cache.parents,
            "Persistent case and disposable source/runtime cache must be separate directories")
    emit("Preparando V5 en CPU; esta celda no entrena", case=str(case))
    source = acquire_source(source_commit, manifest_sha256, cache)
    lock = source / "requirements-cpu.lock.txt"
    lock_hash = sha(lock.read_bytes())
    runtime = cache / ("venv-cpu-" + lock_hash[:16] + "-py" + str(sys.version_info.major) + str(sys.version_info.minor))
    python = runtime / "bin/python"
    marker = runtime / "E3_RUNTIME_LOCK.json"
    if install_runtime:
        if not python.exists():
            emit("Creando entorno CPU aislado; la primera instalación descarga aproximadamente250MB")
            # Base pip can target an empty venv; no ensurepip/apt/CUDA dependency.
            venv.EnvBuilder(with_pip=False, system_site_packages=False, symlinks=True).create(runtime)
        if marker.exists():
            require(json.loads(marker.read_text())["lock_sha256"] == lock_hash, "Runtime lock mismatch")
        else:
            subprocess.run([sys.executable, "-m", "pip", "--python", str(python), "install",
                            "--disable-pip-version-check", "-r", str(lock)], env=cpu_environment(), check=True)
            atomic_bytes(marker, (json.dumps({"lock_sha256": lock_hash, "created_at_UTC": datetime.now(timezone.utc).isoformat()}) + "\n").encode())
        subprocess.run([sys.executable, "-m", "pip", "--python", str(python), "check"], env=cpu_environment(), check=True)
        version_check = ("import importlib.metadata as m, pathlib, sys; "
                         "pins=[line.strip().split('==') for line in pathlib.Path(sys.argv[1]).read_text().splitlines() "
                         "if '==' in line and not line.startswith('--')]; "
                         "bad=[(name,version,m.version(name)) for name,version in pins if m.version(name)!=version]; "
                         "assert not bad, 'Installed package lock differs: '+str(bad)")
        subprocess.run([str(python), "-c", version_check, str(lock)], env=cpu_environment(), check=True)
    else:
        require(python.is_file(), "Runtime installation skipped but environment is absent")
    emit("Comprobando el programa con fixtures sintéticos; no se entrenan modelos")
    subprocess.run([str(python), "-m", "unittest", "discover", "-s", "tests", "-v"],
                   cwd=source, env=cpu_environment(), check=True)
    case.parent.mkdir(parents=True, exist_ok=True)
    require(not case.is_symlink(), "Case symlink refused")
    if not prepare_case:
        data_file = cache / "data/current_cohort_arrays.npz"
        if data_file.exists():
            require(not data_file.is_symlink() and sha(data_file.read_bytes()) == NPZ_SHA256,
                    "Cached pinned input changed; do not overwrite")
        else:
            emit("Verificando los datos antes de restaurar o preparar el caso de GitHub")
            atomic_bytes(data_file, download(NPZ_URL, NPZ_SHA256, max_bytes=460090))
        return {"python": str(python), "source_root": str(source), "case_root": str(case),
                "data_file": str(data_file)}
    if case.exists():
        require((case / "prepared.json").is_file(), "Existing case lacks its prepared receipt; preserve it and use a fresh case name")
        settings = {"python": str(python), "source_root": str(source), "case_root": str(case)}
        command(settings, "status")
        emit("Caso existente verificado; se conserva para reanudar. Esta celda no entrena.")
        return settings
    data_file = cache / "data/current_cohort_arrays.npz"
    if data_file.exists():
        require(not data_file.is_symlink() and sha(data_file.read_bytes()) == NPZ_SHA256, "Persistent input changed; do not overwrite")
    else:
        emit("Verificando el archivo de datos; sólo se admitirán tres arrays training_*")
        atomic_bytes(data_file, download(NPZ_URL, NPZ_SHA256, max_bytes=460090))
    settings = {"python": str(python), "source_root": str(source), "case_root": str(case)}
    command(settings, "prepare", "--npz", str(data_file))
    emit("Piloto preparado: 30 ajustes previstos, 0 ejecutados por esta celda. Ejecuta la celda de entrenamiento cuando corresponda.")
    return settings


def command(settings, subcommand, *extra):
    require(subcommand in ("prepare", "run", "assess", "export", "status"), "Unknown pilot command")
    return subprocess.run([settings["python"], "-m", "seoi_v5.runner", subcommand,
                           "--case-root", settings["case_root"], *extra],
                          cwd=settings["source_root"], env=cpu_environment(), check=True)


def github_command(settings, action, *, source_commit, token, backup_secret,
                   repository="jassan-bit/DeepLearning_202630", branch="e3-seoi-v5-run-01",
                   prefix="runs/e3_seoi_v5/run-01", output=None):
    """Pass secrets through stdin only; never source files, arguments or output."""
    require(action in ("setup", "run", "assess", "status", "export"), "Unknown GitHub pilot action")
    args = [settings["python"], "-m", "seoi_v5.github_controller", "--source-root", settings["source_root"],
            "--case-root", settings["case_root"], "--source-commit", source_commit,
            "--repository", repository, "--branch", branch, "--prefix", prefix]
    if settings.get("data_file"):
        args.extend(["--data-file", settings["data_file"]])
    if output is not None:
        args.extend(["--output", str(output)])
    args.append(action)
    child_env = cpu_environment()
    for key in ("GITHUB_TOKEN", "GH_TOKEN", "E3_V5_BACKUP_KEY"):
        child_env.pop(key, None)
    return subprocess.run(args, input=json.dumps({"token": token, "backup_secret": backup_secret}),
                          text=True, cwd=settings["source_root"], env=child_env, check=True)
