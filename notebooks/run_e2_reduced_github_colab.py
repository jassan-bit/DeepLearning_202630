"""Run the reduced E2 plan with GitHub as its only persistent destination.

The pinned scientific engine remains unchanged. Local /content files are a
scratch workspace, restored from GitHub control snapshots on reconnection.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.request

REPOSITORY = "https://github.com/jassan-bit/DeepLearning_202630.git"
IMPLEMENTATION = "experiments/e2_reduced_execution_20261112/registration.json"
MIGRATION = "experiments/e2_reduced_execution_20261112/budget_migration.json"
CONTROLLERS = {"run_e2_cuda_search.py", "run_e2_optimized_selection.py",
    "run_e2_optimized_selection_v2.py", "run_e2_amended_pilots.py",
    "run_e2_cuda_recovery.py", "e2_guide_schedule.py", "run_e2_reduced_selection.py",
    "run_e2_reduced_delivery.py", "run_e2_reduced_github_delivery.py"}


def _run(command, env, *, cwd=None, capture=False):
    return subprocess.run([str(p) for p in command], env=env, cwd=cwd,
        check=True, text=True, capture_output=capture)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _assert_idle():
    for item in Path("/proc").glob("[0-9]*/cmdline"):
        if int(item.parent.name) == os.getpid():
            continue
        try:
            args = item.read_bytes().decode().split("\0")
        except (OSError, UnicodeError):
            continue
        if any(word and not any(c.isspace() for c in word)
               and Path(word).name in CONTROLLERS for word in args):
            raise RuntimeError("Ya existe un controlador activo: PID " + item.parent.name)


@contextmanager
def _launch_lock(workspace):
    workspace.mkdir(parents=True, exist_ok=True)
    with (workspace / "reduced-launch.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Ya hay un lanzador reducido activo en esta VM.") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _checkout(commit, workspace, env):
    repo = workspace / ("reduced-source-" + commit)
    if not repo.exists():
        repo.mkdir()
        _run(["git", "init", "--quiet", repo], env)
        _run(["git", "-C", repo, "remote", "add", "origin", REPOSITORY], env)
        _run(["git", "-C", repo, "config", "remote.origin.promisor", "true"], env)
        _run(["git", "-C", repo, "config", "remote.origin.partialclonefilter", "blob:none"], env)
        _run(["git", "-C", repo, "fetch", "--filter=blob:none", "--depth", "1", "origin", commit], env)
        _run(["git", "-C", repo, "sparse-checkout", "init", "--no-cone"], env)
        patterns = ["/scripts/", "/tests/", "/experiments/", "/requirements*.txt",
            "/.github/workflows/publish-book.yml", "/data/manifests/",
            "/book_e2/_static/books_alignment_20261108.json",
            "/book_e2/_static/search_protocol_20261109/",
            "/results/filipe_multihistory_20261104/test/raw_test_audit.json",
            "/notebooks/*.py"]
        (repo / ".git/info/sparse-checkout").write_text("\n".join(patterns) + "\n")
        _run(["git", "-C", repo, "checkout", "--detach", "--quiet", commit], env)
    head = _run(["git", "-C", repo, "rev-parse", "HEAD"], env, capture=True).stdout.strip()
    remote = _run(["git", "-C", repo, "remote", "get-url", "origin"], env, capture=True).stdout.strip()
    dirty = _run(["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"], env, capture=True).stdout.strip()
    if head != commit or remote != REPOSITORY or dirty:
        raise RuntimeError("La fuente local no coincide con la versión fijada; se conserva para inspección.")
    # Fetch the actual registration objects and parents needed by the frozen
    # verifier; this does not switch the checkout or replace source files.
    for receipt in ("experiments/e2_reduced_protocol_20261112/git_registration.json",
                    "experiments/e2_protocol_amendment_20261112/git_registration.json"):
        proof = json.loads((repo / receipt).read_text())
        registered = proof["registration_commit"]
        _run(["git", "-C", repo, "fetch", "--filter=blob:none", "--depth", "2", "origin", registered], env)
    record = json.loads((repo / IMPLEMENTATION).read_text())
    if record.get("protocol_id") != "e2-reduced-delivery-v1":
        raise RuntimeError("El registro no corresponde al protocolo reducido autorizado.")
    for relative, expected in record["files_sha256"].items():
        target = repo / relative
        if not target.resolve().is_relative_to(repo.resolve()) or _sha(target) != expected:
            raise RuntimeError("Una fuente registrada cambió: " + relative)
    return repo


def _stream(command, env, cwd, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as output:
        child = subprocess.Popen([str(p) for p in command], env=env, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in child.stdout:
                secret = env.get("GITHUB_TOKEN")
                if secret:
                    line = line.replace(secret, "[REDACTED]")
                print(line, end="", flush=True)
                output.write(line)
                output.flush()
            return child.wait()
        except BaseException:
            print("Interrupción del lanzador. Revisa los recibos antes de reanudar; no se repetirá un ajuste automáticamente.", flush=True)
            raise


ENGINE_COMMIT = "1e8745497b484518f766b6da5c19be8362d9add7"
LOCAL_WORKSPACE = Path("/content/entregable2_gpu_workspace")
EXTENSION_REGISTRATION = "experiments/e2_github_primary_20261112/registration.json"
EXTENSION_FILES = ("scripts/e2_reduced_github_state.py", "scripts/e2_reduced_github_restore.py",
    "scripts/run_e2_reduced_github_delivery.py")


def _public_bytes(commit, relative):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("La extensión necesita un commit inmutable.")
    url = "https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/" + commit + "/" + relative
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = response.read(2 * 1024**2 + 1)
    if len(payload) > 2 * 1024**2:
        raise RuntimeError("Una fuente de extensión supera el límite admitido.")
    return payload


def _extension(commit, workspace):
    raw = _public_bytes(commit, EXTENSION_REGISTRATION)
    record = json.loads(raw)
    if record.get("protocol_id") != "e2-reduced-delivery-v1" or record.get("engine_commit") != ENGINE_COMMIT:
        raise RuntimeError("La extensión no corresponde al motor científico registrado.")
    hashes = record.get("files_sha256", {})
    if not set(EXTENSION_FILES).issubset(hashes):
        raise RuntimeError("El registro no vincula los dos componentes de persistencia.")
    root = workspace / ("github-primary-extension-" + commit)
    for relative in EXTENSION_FILES:
        payload = _public_bytes(commit, relative)
        if hashlib.sha256(payload).hexdigest() != hashes[relative]:
            raise RuntimeError("La fuente de extensión no coincide: " + relative)
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != payload:
                raise RuntimeError("Una extensión local cambió; se conserva para revisión.")
        else:
            with target.open("xb") as stream:
                stream.write(payload)
    proof = root / "registration.json"
    if proof.exists() and proof.read_bytes() != raw:
        raise RuntimeError("El registro local de la extensión cambió.")
    if not proof.exists():
        with proof.open("xb") as stream:
            stream.write(raw)
    return root


def run_reduced_github_colab(extension_commit, *, namespace=None, execute=False):
    """Prepare unchanged engine E, restore GitHub state, then run reduced fits."""
    if not isinstance(extension_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", extension_commit):
        raise ValueError("Se requiere el commit inmutable de la extensión GitHub.")
    if type(execute) is not bool:
        raise TypeError("execute debe ser booleano.")
    namespace = globals() if namespace is None else namespace
    python = Path(namespace.get("GPU_PYTHON") or
        "/content/entregable2_gpu_workspace/reduced-runtime-py312-cu126-v1/venv-py312-cu126/bin/python")
    if not python.is_file():
        raise RuntimeError("Falta el Python aislado preparado previamente.")
    env = dict(os.environ)
    for key in ("GITHUB_TOKEN", "GH_TOKEN", "PYTHONPATH"):
        env.pop(key, None)
    env.update(CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONNOUSERSITE="1",
        OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    _run([python, "-c", "import sys,torch; assert sys.version_info[:3]==(3,12,12); assert torch.__version__=='2.8.0+cu126'; assert torch.version.cuda=='12.6'; assert torch.cuda.is_available(); print('Entorno CUDA compatible; todavía sin entrenamiento')"], env)
    _assert_idle()
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        from google.colab import userdata
        token = (userdata.get("GITHUB_TOKEN") or "").strip()
    if not token or any(c.isspace() for c in token):
        raise RuntimeError("Activa el secreto GITHUB_TOKEN con escritura en tu repositorio; no lo pegues en el chat.")
    workspace = LOCAL_WORKSPACE
    with _launch_lock(workspace):
        repo = _checkout(ENGINE_COMMIT, workspace, env)
        extension = _extension(extension_commit, workspace)
        target = workspace / "github-primary-output-v1"
        assets = target / "assets"
        assets.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        log = assets / "launch_logs" / (stamp + ".log")
        runtime = assets / ("runtime-" + ENGINE_COMMIT + ".json")
        _run([python, repo / "scripts/run_e2_reduced_selection.py", "--write-runtime-profile", runtime], env, cwd=repo)
        _run([python, repo / "scripts/fetch_e2_development_sources.py", "--output", repo], env, cwd=repo)
        roots = workspace / "reduced-author-sources/source_roots.json"
        _run([python, repo / "scripts/fetch_e2_published_sources.py", "--output", roots.parent], env, cwd=repo)
        recon = workspace / "reduced-historical-reconciliation"
        runner_env = dict(env, GITHUB_TOKEN=token)
        code = ("from pathlib import Path; from e2_amended_pilot_budget import fetch_reconciliation; "
            + "r=fetch_reconciliation(Path(" + repr(str(recon)) + ")); "
            + "print('Saldo histórico comprobado:',r['remaining_seconds_before_pilots'])")
        _run([python, "-c", code], dict(runner_env, PYTHONPATH=str(repo / "scripts")), cwd=repo)
        audit = assets / ("cache-audit-" + ENGINE_COMMIT + ".json")
        _run([python, repo / "scripts/audit_e2_reduced_cache.py",
            "--reconciliation-root", recon, "--repo-root", repo,
            "--domains", repo / "experiments/e2_reduced_protocol_20261112/configurations.json",
            "--verify-prepared", "--runtime-profile", runtime, "--output", audit], env, cwd=repo)
        command = [python, extension / "scripts/run_e2_reduced_github_delivery.py",
            "--engine-root", repo, "--extension-root", extension,
            "--extension-commit", extension_commit, "--output", target,
            "--data-root", repo, "--source-roots", roots, "--runtime-profile", runtime,
            "--budget-evidence", recon / "budget_evidence.json", "--migration", repo / MIGRATION,
            "--cache-audit", audit, "--reconciliation-root", recon,
            "--migration-registration-commit", ENGINE_COMMIT, "--allow-final-download"]
        if execute:
            command.append("--execute")
        _assert_idle()
        print("GitHub es el único destino persistente. Iniciando restauración y control de presupuesto.", flush=True)
        exit_code = _stream(command, runner_env, repo, log)
        result = {"engine_commit": ENGINE_COMMIT, "extension_commit": extension_commit,
            "execute_requested": execute, "exit_code": exit_code, "temporary_workspace": str(target),
            "persistent_destination": "github", "log": str(log), "budget_extended": False}
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
