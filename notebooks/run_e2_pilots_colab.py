"""Colab launcher for the registered E2 pilots, with Drive-backed artifacts.

The source commit is supplied explicitly by the release cell. Preflight is the
default and reads no development CSVs. Execution uses a real CUDA runtime and
the runner's reconciled 64-hour budget; this launcher never increases it.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

REPOSITORY = "https://github.com/jassan-bit/DeepLearning_202630.git"
UV_VERSION, PYTHON_VERSION, PIP_VERSION = "0.9.5", "3.12.12", "25.3"
UV_WHEEL_SHA256 = "6507bbbcd788553ec4ad5a96fa19364dc0f58b023e31d79868773559a83ec181"
PIP_WHEEL_SHA256 = "9655943313a94722b7774661c21049070f6bbb0a1516bf02f7c8d5d9201514cd"
CONTROLLERS = (
    "run_e2_cuda_search.py", "run_e2_optimized_selection.py",
    "run_e2_optimized_selection_v2.py", "e2_guide_schedule.py",
    "run_e2_cuda_recovery.py", "run_e2_amended_pilots.py",
    "supervise_e2_within_56_hours.py",
)
SPARSE_PATHS = (
    "/scripts/", "/tests/", "/experiments/", "/requirements*.txt",
    "/.github/workflows/publish-book.yml", "/data/manifests/",
    "/book_e2/_static/search_protocol_20261109/",
    "/book/_static/e1_608_remediation_20261107/chronology.json",
    "/notebooks/run_e2_pilots_colab.py",
)


def _run(args, *, env, cwd=None, capture=False):
    return subprocess.run([str(value) for value in args], cwd=cwd, env=env,
                          check=True, text=True, capture_output=capture)


def _sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _controller_pids(proc_root=Path("/proc")):
    found = []
    for directory in Path(proc_root).iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            arguments = (directory / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for raw in arguments:
            argument = raw.decode("utf-8", errors="replace")
            # Shell/Python code strings are not executable script arguments.
            if argument and not any(character.isspace() for character in argument):
                if Path(argument).name in CONTROLLERS:
                    found.append(int(directory.name))
                    break
    return found


def _assert_no_controller():
    found = _controller_pids()
    if found:
        raise RuntimeError("Hay otro controlador activo (PID " + ", ".join(map(str, found))
                           + "). Espera su cierre normal; no se inició otro proceso.")


@contextmanager
def _local_launch_lock(workspace):
    workspace.mkdir(parents=True, exist_ok=True)
    with (workspace / "amended-pilots-launch.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Ya hay un lanzador de pilotos activo en esta VM.") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _mount_drive():
    root = Path("/content/drive/MyDrive")
    if os.path.ismount("/content/drive") and root.is_dir():
        return root.resolve()
    try:
        from google.colab import drive
    except ImportError:
        raise RuntimeError("Este lanzador requiere Google Colab y Google Drive montado.") from None
    drive.mount("/content/drive", force_remount=False)
    if not root.is_dir():
        raise RuntimeError("No se pudo verificar MyDrive; no se usarán resultados transitorios.")
    return root.resolve()


def _token():
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        try:
            from google.colab import userdata
            token = (userdata.get("GITHUB_TOKEN") or "").strip()
        except Exception:
            token = ""
    if token and any(character.isspace() for character in token):
        raise RuntimeError("El secreto GITHUB_TOKEN tiene un formato inválido.")
    return token


def _checkout(source_commit, workspace, env):
    if not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("Usa el SHA publicado de 40 caracteres; no main ni una rama móvil.")
    repo = workspace / ("pilot-source-" + source_commit)
    if not repo.exists():
        repo.mkdir()
        _run(["git", "init", "--quiet", repo], env=env)
        _run(["git", "-C", repo, "remote", "add", "origin", REPOSITORY], env=env)
        _run(["git", "-C", repo, "config", "remote.origin.promisor", "true"], env=env)
        _run(["git", "-C", repo, "config", "remote.origin.partialclonefilter", "blob:none"], env=env)
        _run(["git", "-C", repo, "fetch", "--filter=blob:none", "--depth", "1", "origin", source_commit], env=env)
        _run(["git", "-C", repo, "sparse-checkout", "init", "--no-cone"], env=env)
        (repo / ".git/info/sparse-checkout").write_text("\n".join(SPARSE_PATHS) + "\n")
        _run(["git", "-C", repo, "checkout", "--detach", "--quiet", source_commit], env=env)
    remote = _run(["git", "-C", repo, "remote", "get-url", "origin"], env=env, capture=True).stdout.strip()
    head = _run(["git", "-C", repo, "rev-parse", "HEAD"], env=env, capture=True).stdout.strip()
    dirty = _run(["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"],
                 env=env, capture=True).stdout.strip()
    if remote != REPOSITORY or head != source_commit or dirty:
        raise RuntimeError("El checkout de pilotos no coincide con sus fuentes inmutables; se conserva para inspección.")
    if (repo / "results/eda_development/daily_market_development.csv").exists():
        raise RuntimeError("El checkout de código contiene CSV de mercado; utiliza el checkout parcial publicado.")
    proof_path = repo / "experiments/e2_protocol_amendment_20261112/git_registration.json"
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    registration_commit = proof.get("registration_commit", "")
    if not re.fullmatch(r"[0-9a-f]{40}", registration_commit):
        raise RuntimeError("No hay un commit de registro válido en el código seleccionado.")
    present = subprocess.run(["git", "-C", str(repo), "cat-file", "-e", registration_commit + "^{commit}"],
                             env=env, capture_output=True, text=True)
    if present.returncode:
        _run(["git", "-C", repo, "fetch", "--filter=blob:none", "--depth", "1", "origin", registration_commit], env=env)
    return repo


def _bootstrap_python(workspace, env):
    bootstrap = workspace / "bootstrap"
    bootstrap.mkdir(parents=True, exist_ok=True)
    uv_home = bootstrap / ("uv-" + UV_VERSION)
    uv_binary = uv_home / "bin/uv"
    requirements = bootstrap / "uv.requirements.txt"
    requirements.write_text("uv==" + UV_VERSION + " --hash=sha256:" + UV_WHEEL_SHA256 + "\n")
    if not uv_binary.is_file():
        if uv_home.exists():
            raise RuntimeError("El bootstrap uv está incompleto; conserva esa carpeta para inspección.")
        _run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir",
              "--no-deps", "--only-binary=:all:", "--require-hashes", "--index-url", "https://pypi.org/simple",
              "--target", uv_home, "-r", requirements], env=env)
    version = _run([uv_binary, "--version"], env=env, capture=True).stdout.strip().split()
    if version[:2] != ["uv", UV_VERSION]:
        raise RuntimeError("La versión de uv no coincide con la fijada.")
    managed = workspace / "managed-python"
    _run([uv_binary, "python", "install", PYTHON_VERSION, "--install-dir", managed,
          "--no-bin", "--no-config"], env=env)
    executable = Path(_run([uv_binary, "python", "find", PYTHON_VERSION, "--managed-python",
        "--no-python-downloads", "--no-project", "--no-config"], env=env, capture=True).stdout.strip())
    if managed.resolve() not in executable.resolve().parents:
        raise RuntimeError("uv seleccionó un Python fuera del directorio gestionado.")
    venv = workspace / "venv-py312-cu126"
    python = venv / "bin/python"
    if not python.is_file():
        _run([uv_binary, "venv", "--python", executable, "--no-python-downloads",
              "--no-project", "--no-config", venv], env=env)
    probe = json.loads(_run([python, "-c", "import json,sys; print(json.dumps(list(sys.version_info[:3])))"],
                             env=env, capture=True).stdout)
    if probe != [3, 12, 12]:
        raise RuntimeError("El Python aislado no es 3.12.12; no se modificó ese entorno.")
    pip_requirements = bootstrap / "pip.requirements.txt"
    pip_requirements.write_text("pip==" + PIP_VERSION + " --hash=sha256:" + PIP_WHEEL_SHA256 + "\n")
    _run([uv_binary, "pip", "install", "--python", python, "--no-deps", "--only-binary=:all:",
          "--require-hashes", "--index-url", "https://pypi.org/simple", "--no-config", "-r", pip_requirements], env=env)
    return python


def _gpu_environment(python, repo, env):
    lock = repo / "requirements-e2-training-gpu.lock.txt"
    expected = {}
    for line in lock.read_text().splitlines():
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+)", line.strip())
        if match:
            expected[match.group(1)] = match.group(2)
    check = ("import importlib.metadata as m,json\nexpected=" + repr(expected)
             + "\nmissing=[]\nfor name,version in expected.items():\n"
               " try:\n  actual=m.version(name)\n except m.PackageNotFoundError:\n  actual=None\n"
               " if actual!=version:missing.append(name)\nprint(json.dumps(missing))")
    missing = json.loads(_run([python, "-c", check], env=env, capture=True).stdout)
    if missing:
        print("Preparando las dependencias CUDA fijadas; el kernel de Colab no se reinicia.")
        _run([python, "-m", "pip", "install", "-r", lock], env=env)
    _run([python, "-m", "pip", "check"], env=env)
    probe = "import json,sys,torch; assert sys.version_info[:3]==(3,12,12); assert torch.__version__=='2.8.0+cu126'; assert torch.version.cuda=='12.6'; assert torch.cuda.is_available(); print(json.dumps({'python':'.'.join(map(str,sys.version_info[:3])),'torch':str(torch.__version__),'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0)}))"
    runtime = json.loads(_run([python, "-c", probe], env=env, capture=True).stdout)
    print("Entorno GPU verificado:", json.dumps(runtime, ensure_ascii=False))
    return runtime


def _stream_runner(args, *, cwd, env, log_path, token):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen([str(value) for value in args], cwd=cwd, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        try:
            for line in process.stdout:
                if token:
                    line = line.replace(token, "[REDACTED]")
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            return process.wait()
        except BaseException:
            print("El controlador conserva sus recibos; no se lanzó una repetición automática.")
            raise


def run_colab_pilots(source_commit, *, stage="all", execute=False, output=None,
                     budget_evidence=None):
    """Prepare once, preflight without CSVs, then execute only when admitted."""
    if stage not in ("noise", "power", "all") or type(execute) is not bool:
        raise ValueError("Usa stage noise, power o all y una decisión execute booleana.")
    if not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("Usa el SHA inmutable de la publicación, no main ni un marcador pendiente.")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("El entorno CUDA fijado requiere Linux x86_64.")
    _assert_no_controller()
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("No hay GPU NVIDIA visible. Este lanzador no tiene fallback CPU.")
    workspace = Path("/content/entregable2_gpu_workspace")
    with _local_launch_lock(workspace):
        drive = _mount_drive()
        base = drive / "DeepLearning_202630/E2"
        target = Path(output) if output is not None else base / "amended-pilots-v1"
        target = target.absolute()
        if not target.resolve().is_relative_to(drive):
            raise ValueError("Guarda --output dentro de MyDrive; /content es transitorio.")
        assets = base / "amended-pilots-assets"
        assets.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        for key in ("GITHUB_TOKEN", "GH_TOKEN", "PYTHONPATH"):
            env.pop(key, None)
        env.update(CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONNOUSERSITE="1",
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                   UV_CACHE_DIR=str(workspace / "uv-cache"),
                   UV_PYTHON_INSTALL_DIR=str(workspace / "managed-python"),
                   UV_PYTHON_BIN_DIR=str(workspace / "managed-python-bin"), UV_NATIVE_TLS="true")
        repo = _checkout(source_commit, workspace, env)
        print("Fuente inmutable:", source_commit)
        python = _bootstrap_python(workspace, env)
        runtime = _gpu_environment(python, repo, env)
        roots = assets / "author_sources/source_roots.json"
        _run([python, repo / "scripts/fetch_e2_published_sources.py", "--output", roots.parent], env=env, cwd=repo)
        token = _token()
        runner_env = dict(env)
        if token:
            runner_env["GITHUB_TOKEN"] = token
        data_root = assets / "development_cache"
        args = [python, repo / "scripts/run_e2_amended_pilots.py", "--output", target,
                "--source-roots", roots, "--data-root", data_root, "--stage", stage]
        if budget_evidence is not None:
            args.extend(["--budget-evidence", Path(budget_evidence)])
        started = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log = assets / "launch_logs" / (started + "-" + str(os.getpid()) + ".log")
        (assets / ("launch-" + started + "-" + str(os.getpid()) + ".json")).write_text(
            json.dumps({"source_commit": source_commit, "stage": stage,
                "execution_requested": execute, "output": str(target), "log": str(log),
                "runtime": runtime, "requirements_sha256": _sha256(repo / "requirements-e2-training-gpu.lock.txt")},
                ensure_ascii=False, indent=2) + "\n")
        print("Comprobando plan, GPU y saldo conciliado antes de abrir los CSV de desarrollo.")
        _assert_no_controller()
        code = _stream_runner(args, cwd=repo, env=runner_env, log_path=log, token=token)
        preflight_path = target / "preflight.json"
        preflight = json.loads(preflight_path.read_text()) if preflight_path.is_file() else {}
        if code != 0 or preflight.get("ready") is not True:
            print("No se inició entrenamiento. Revisa:", preflight_path)
            return {"exit_code": code, "training_started_by_launcher": False,
                    "preflight": str(preflight_path), "log": str(log)}
        if not execute:
            print("Preparación aprobada; execute=False conserva los pilotos sin iniciar.")
            return {"exit_code": 0, "training_started_by_launcher": False,
                    "preflight": str(preflight_path), "log": str(log)}
        _run([python, repo / "scripts/fetch_e2_development_sources.py", "--output", data_root], env=env, cwd=repo)
        _assert_no_controller()
        print("Solicitando ejecución; el controlador revalidará el saldo. Artefactos:", target)
        code = _stream_runner(args + ["--execute"], cwd=repo, env=runner_env, log_path=log, token=token)
        (assets / "last_launch.json").write_text(json.dumps({"source_commit": source_commit,
            "stage": stage, "exit_code": code, "output": str(target), "log": str(log),
            "runtime": runtime, "requirements_sha256": _sha256(repo / "requirements-e2-training-gpu.lock.txt")},
            ensure_ascii=False, indent=2) + "\n")
        print("Código de salida:", code, "; artefactos:", target)
        return {"exit_code": code, "execution_requested": True,
                "training_completion_claimed": False, "output": str(target), "log": str(log)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--stage", choices=("noise", "power", "all"), default="all")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--budget-evidence", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result = run_colab_pilots(args.source_commit, stage=args.stage, execute=args.execute,
                              output=args.output, budget_evidence=args.budget_evidence)
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
