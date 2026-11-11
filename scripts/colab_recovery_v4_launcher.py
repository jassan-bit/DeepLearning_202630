#!/usr/bin/env python3
"""Recover the sealed campaign in a prepared Colab session, including a new VM.

The importing cell supplies RECOVERY_COMMIT and
EXPECTED_RECOVERY_AUTHORIZATION_SHA256. This file deliberately has no pin to
its own commit. A missing campaign cache is restored and validated only by the
controller; this launcher never initializes an empty scientific campaign. The
default invocation is an inert, standard-library preview.
"""
from __future__ import annotations

import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import uuid


_A3_SOURCE = "564ba4762f84a800b02fa4c734f5752a169f5223"
_A3_REGISTRATION = "78c3c5b8ca042a347aaa80d642f3affd8b5911e8697d8747a2de915d1c3194a9"
_A3_PROTOCOL = "e0030c990a4afa15665aec254211b0d9e680e9ac5d92dfb8094e43e965152abe"
_SEALED_PLAN = "4eae2d2ff9ef6f83a2f9eb0dc5ba647208d695e1665a8e522d812fed67893bda"
_REPOSITORY = "jassan-bit/DeepLearning_202630"
_CAMPAIGN = "e2-cuda-selection-v1"
_CAMPAIGN_BRANCH = "campaigns/" + _CAMPAIGN
_AUTHORIZATION_PATH = "experiments/e2_recovery_20261111_r4/authorization.json"
_INITIAL_CAMPAIGN_HOURS = 24
_REGISTERED_EFFECTIVE_CAMPAIGN_HOURS = 48
_CONTROLLER_PATHS = (
    "scripts/e2_manual_reconciliation_v4.py",
    "scripts/e2_recovery_receipts_v4.py",
    "scripts/run_e2_cuda_recovery_v4.py",
    "tests/test_e2_manual_reconciliation_v4.py",
    "tests/test_e2_recovery_receipts_v4.py",
    "tests/test_e2_cuda_recovery_v4.py",
    "scripts/colab_recovery_v4_launcher.py",
    "tests/test_e2_colab_launcher_v4.py",
    _AUTHORIZATION_PATH,
)
_DATA_HASHES = {
    "results/eda_development/daily_market_development.csv": "bf6bb5160af924aef6a15c3684aeae932df3017e8639888bb7e91ce787900171",
    "results/eda_development/daily_volatility_development.csv": "20b96374a659eb87dadfa68aa3d6b6b5fb6bea5b10eebe6f2c186ad69861844a",
}


class LauncherBlocked(RuntimeError):
    """A preparation or single-writer admission failed before launch."""


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _no_symlinks(path):
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise LauncherBlocked("La caché preparada contiene un enlace simbólico no permitido.")


def _json_file(path, maximum=4 * 1024 * 1024):
    _no_symlinks(path)
    with Path(path).open("rb") as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise LauncherBlocked("Un archivo de control supera el tamaño esperado.")
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise LauncherBlocked("Un archivo de control no contiene un objeto JSON.")
    return result


def _clean_environment():
    result = dict(os.environ)
    for key in ("GITHUB_TOKEN", "GH_TOKEN", "PYTHONPATH"):
        result.pop(key, None)
    result.update(CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONNOUSERSITE="1",
                  OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                  GIT_TERMINAL_PROMPT="0")
    return result


def _run(args, environment, *, cwd=None, capture=False, input_text=None):
    try:
        return subprocess.run([str(item) for item in args], cwd=cwd, env=environment,
                              check=True, text=True, capture_output=capture, input=input_text)
    except (subprocess.CalledProcessError, OSError):
        # Subprocess exception representations can include arguments or output.
        raise LauncherBlocked("Falló una comprobación de preparación; no se inició la recuperación.") from None


def _read_only_campaign_lock(campaign):
    """Hold the existing ledger lock without creating or changing a file."""
    path = campaign / ".campaign.lock"
    _no_symlinks(path)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return None
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise LauncherBlocked("runner_already_active: la campaña ya tiene un ejecutor activo.") from None
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _verify_campaign_cache_if_present(campaign):
    """An absent cache is admissible; an unrelated existing path is not.

    The controller must restore the remote journal and verify its sealed head,
    plan, extension and recovery receipt before claiming any physical attempt.
    Never mkdir or write campaign.json here to make admission appear satisfied.
    """
    _no_symlinks(campaign)
    if not campaign.exists():
        return False
    if not campaign.is_dir() or not (campaign / "campaign.json").is_file():
        raise LauncherBlocked("La ruta de caché existente no contiene una campaña válida; no se reemplazó ni se creó otra campaña.")
    _no_symlinks(campaign / "campaign.json")
    return True


def _unlock(descriptor):
    if descriptor is not None:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _launcher_lock(workspace):
    path = workspace / ".recovery-v4-launcher.lock"
    _no_symlinks(path)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise LauncherBlocked("runner_already_active: otra celda de recuperación sigue activa.") from None
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _verify_engine(engine, environment):
    if _run(["git", "-C", engine, "rev-parse", "HEAD"], environment, capture=True).stdout.strip() != _A3_SOURCE:
        raise LauncherBlocked("La fuente preparada debe conservar exactamente el commit científico A3.")
    registration = engine / "experiments/e2_cuda_search_20261109/registration.json"
    if _sha256(registration) != _A3_REGISTRATION:
        raise LauncherBlocked("El registro científico preparado cambió.")
    registered = _json_file(registration)
    hashes = registered.get("implementation_sha256", {})
    if not isinstance(hashes, dict) or len(hashes) != 61:
        raise LauncherBlocked("El cierre científico A3 está incompleto.")
    for relative, expected in hashes.items():
        path = engine / relative
        _no_symlinks(path)
        if _sha256(path) != expected:
            raise LauncherBlocked("Cambió un archivo de la implementación científica A3.")
    protocol = "book_e2/_static/search_protocol_20261109/protocol.json"
    if _sha256(engine / protocol) != _A3_PROTOCOL:
        raise LauncherBlocked("Falta el protocolo normalizado de la preparación anterior.")
    changed = _run(["git", "-C", engine, "diff", "--name-only", "HEAD"], environment, capture=True).stdout.splitlines()
    if any(name != protocol for name in changed):
        raise LauncherBlocked("Hay cambios adicionales en la fuente científica preparada.")
    untracked = _run(["git", "-C", engine, "ls-files", "--others", "--exclude-standard", "scripts"], environment, capture=True).stdout.splitlines()
    if any(name.endswith(".py") for name in untracked):
        raise LauncherBlocked("Hay módulos Python adicionales en la fuente científica.")


def _verify_prepared_files(workspace, engine):
    python = workspace / "venv-py312-cu126/bin/python"
    data = workspace / "development_cache"
    if not engine.is_dir() or not python.is_file() or not data.is_dir():
        raise LauncherBlocked("Falta la preparación anterior (A3, Python aislado o caché de desarrollo). Ejecuta las secciones 1–7 de preparación; este lanzador no instaló paquetes ni descargó datos.")
    return python, data


def _verify_hardware(python, engine, environment):
    probe = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], environment, capture=True)
    names = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
    if not names or names[0] != "Tesla T4":
        raise LauncherBlocked("La campaña requiere una Tesla T4 disponible; no se reclamó ningún intento.")
    _run([python, "-c", "import sys,torch; assert sys.version_info[:3]==(3,12,12); assert torch.__version__=='2.8.0+cu126'; assert torch.version.cuda=='12.6'"], environment, capture=True)
    _run([python, engine / "scripts/run_e2_gpu_pilot.py", "--check-environment", "--device", "cuda:0"], environment, cwd=engine, capture=True)


def _checkout_controller(controller, commit, authorization_sha, environment):
    _no_symlinks(controller)
    if not controller.exists():
        controller.mkdir()
        _run(["git", "init", controller], environment, capture=True)
        _run(["git", "-C", controller, "remote", "add", "origin", "https://github.com/" + _REPOSITORY + ".git"], environment)
        _run(["git", "-C", controller, "config", "core.sparseCheckout", "true"], environment)
        _run(["git", "-C", controller, "config", "remote.origin.promisor", "true"], environment)
        _run(["git", "-C", controller, "config", "remote.origin.partialclonefilter", "blob:none"], environment)
        _run(["git", "-C", controller, "sparse-checkout", "set", "--no-cone", "--stdin"], environment,
             input_text="\n".join("/" + path for path in _CONTROLLER_PATHS) + "\n")
        _run(["git", "-C", controller, "fetch", "--filter=blob:none", "--depth=1", "origin", commit], environment, capture=True)
        _run(["git", "-C", controller, "checkout", "--detach", "FETCH_HEAD"], environment, capture=True)
    if _run(["git", "-C", controller, "rev-parse", "HEAD"], environment, capture=True).stdout.strip() != commit:
        raise LauncherBlocked("recovery-v4-source pertenece a otro commit; no se modificó su contenido.")
    if _run(["git", "-C", controller, "diff", "--name-only", "HEAD"], environment, capture=True).stdout.strip():
        raise LauncherBlocked("Cambió el controlador de recuperación v4.")
    untracked = _run(["git", "-C", controller, "ls-files", "--others", "--exclude-standard", "scripts"], environment, capture=True).stdout.splitlines()
    if any(name.endswith(".py") for name in untracked):
        raise LauncherBlocked("Hay módulos no registrados junto al controlador v4.")
    authorization = controller / _AUTHORIZATION_PATH
    if _sha256(authorization) != authorization_sha:
        raise LauncherBlocked("La autorización v4 no coincide con el SHA256 publicado.")
    auth = _json_file(authorization)
    if (auth.get("scientific_source_commit") != _A3_SOURCE or
            auth.get("scientific_registration_sha256") != _A3_REGISTRATION or
            auth.get("plan_sha256") != _SEALED_PLAN):
        raise LauncherBlocked("La autorización v4 no corresponde al plan científico sellado.")
    hashes = auth.get("implementation_sha256", {})
    if not isinstance(hashes, dict) or set(hashes) != set(_CONTROLLER_PATHS[:3]):
        raise LauncherBlocked("Falta el cierre de los tres módulos de recuperación v4.")
    for relative, expected in hashes.items():
        if _sha256(controller / relative) != expected:
            raise LauncherBlocked("Cambió una ayuda de recuperación v4.")
    return authorization


def _command(python, controller, engine, commit, auth, auth_sha, campaign, *, execute, report=None, data=None):
    result = [str(python), str(controller / "scripts/run_e2_cuda_recovery_v4.py"), "--execute" if execute else "--plan-only",
              "--engine-root", str(engine), "--controller-source-commit", commit,
              "--authorization", str(auth), "--expected-authorization-sha256", auth_sha,
              "--source-commit", _A3_SOURCE, "--expected-registration-sha256", _A3_REGISTRATION,
              "--output", str(campaign), "--session-hours", "8", "--campaign-hours", str(_INITIAL_CAMPAIGN_HOURS), "--storage-gib", "10",
              "--device", "cuda:0", "--github-repository", _REPOSITORY,
              "--github-branch", _CAMPAIGN_BRANCH, "--github-prefix", _CAMPAIGN_BRANCH]
    if execute:
        result += ["--retry-lost-once", "--repair-only", "--data-root", str(data),
                   "--source-root-itransformer", str(engine / "work/author_sources/iTransformer"),
                   "--external-report", str(report)]
    return result


def _verify_cached_inputs(engine, data):
    for relative, expected in _DATA_HASHES.items():
        path = data / relative
        _no_symlinks(path)
        if not path.is_file() or _sha256(path) != expected:
            raise LauncherBlocked("Falta un derivado de desarrollo preparado o cambió su SHA256; no se descargó ningún CSV.")
    manifest_source = engine / "scripts/fetch_e2_itransformer_sources.py"
    manifest = next((ast.literal_eval(node.value) for node in ast.parse(manifest_source.read_text()).body
                     if isinstance(node, ast.Assign) and len(node.targets) == 1
                     and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "MANIFEST"), None)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise LauncherBlocked("Falta el manifiesto científico de iTransformer.")
    source = engine / "work/author_sources/iTransformer"
    _no_symlinks(source)
    for relative, expected in manifest["files"].items():
        path = source / relative
        _no_symlinks(path)
        if not path.is_file() or path.stat().st_size > 1024 * 1024 or _sha256(path) != expected:
            raise LauncherBlocked("La caché preparada de iTransformer está incompleta; no se descargó código.")
    for path in source.rglob("*"):
        _no_symlinks(path)
        if path.is_file() and path.suffix in {".py", ".pyc", ".pyo", ".pth", ".so"} and path.relative_to(source).as_posix() not in manifest["files"]:
            if "__pycache__" not in path.relative_to(source).parts or path.suffix != ".pyc":
                raise LauncherBlocked("La caché iTransformer contiene código adicional.")
    expected_receipt = {"status": "complete", "repository": manifest["repository"], "commit": manifest["commit"],
                        "verified_files": manifest["files"], "weights_downloaded": False, "market_data_read": False}
    if _json_file(source / "source_receipt.json", maximum=1024 * 1024) != expected_receipt:
        raise LauncherBlocked("El recibo preparado de iTransformer cambió.")


def _github_environment(clean):
    import urllib.error
    import urllib.request
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        try:
            from google.colab import userdata
            token = userdata.get("GITHUB_TOKEN").strip()
        except Exception:
            raise LauncherBlocked("Habilita el secreto GITHUB_TOKEN en este mismo notebook de Colab.") from None
    if not token or any(character.isspace() for character in token):
        raise LauncherBlocked("El secreto GITHUB_TOKEN está vacío o tiene un formato inválido.")
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, request, response, code, message, headers, new_url):
            return None
    request = urllib.request.Request("https://api.github.com/repos/" + _REPOSITORY,
        headers={"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28"}, method="GET")
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=30) as response:
            raw = response.read(65537)
        if len(raw) > 65536:
            raise ValueError("response too large")
        payload = json.loads(raw)
    except urllib.error.HTTPError as error:
        raise LauncherBlocked("GitHub rechazó la comprobación de acceso (HTTP " + str(error.code) + ").") from None
    except Exception:
        raise LauncherBlocked("No se pudo comprobar el acceso a GitHub; no se inició la recuperación.") from None
    if not isinstance(payload, dict) or payload.get("full_name", "").lower() != _REPOSITORY.lower():
        raise LauncherBlocked("GitHub no confirmó el repositorio autorizado.")
    return dict(clean, GITHUB_TOKEN=token)


def _send_interrupt(process):
    try:
        process.send_signal(signal.SIGINT)
    except ProcessLookupError:
        pass


def _wait_gracefully(process, *, interrupted=False):
    def notify(message):
        try:
            print(message)
        except (Exception, KeyboardInterrupt):
            # A broken or interrupted notebook display is not permission to
            # abandon the controller while it is closing a worker/ledger.
            pass
    while True:
        try:
            return process.wait(), interrupted
        except KeyboardInterrupt:
            if not interrupted:
                interrupted = True
                _send_interrupt(process)
                notify("Interrupción solicitada: se envió SIGINT al controlador; esperando el cierre del trabajador y del diario.")
            else:
                notify("El cierre sigue pendiente; se conserva la espera sin matar el proceso.")


def run_launcher(namespace):
    """Run once, preserving every invocation's independent report and PID."""
    run = namespace.get("RUN_SEARCH", False)
    retry = namespace.get("RECOVER_PENDING_ONCE", False)
    if type(run) is not bool or type(retry) is not bool:
        raise LauncherBlocked("RUN_SEARCH y RECOVER_PENDING_ONCE deben ser booleanos explícitos.")
    if not (run and retry):
        preview = {"version": "e2-recovery-v4-launcher-v1", "status": "preview_only",
                   "RUN_SEARCH": run, "RECOVER_PENDING_ONCE": retry,
                   "session_hours": 8, "campaign_hours": _INITIAL_CAMPAIGN_HOURS,
                   "initial_campaign_hours": _INITIAL_CAMPAIGN_HOURS,
                   "registered_effective_campaign_hours": _REGISTERED_EFFECTIVE_CAMPAIGN_HOURS,
                   "budget_extension": False,
                   "budget_scope": "24 horas iniciales selladas; extensión previa registrada a 48 horas, que el controlador verifica al restaurar. No se solicita otra extensión.",
                   "missing_campaign_cache": "Admitida sólo mediante restauración remota validada por el controlador; no se crea una campaña vacía.",
                   "required_preparation": "Secciones 1–7: workspace, fuente A3, Python aislado, desarrollo e iTransformer.",
                   "network_used": False, "credentials_read": False, "gpu_used": False,
                   "scientific_fits_executed": False, "recovery_claimed": False}
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return preview
    commit = namespace.get("RECOVERY_COMMIT", "")
    auth_sha = namespace.get("EXPECTED_RECOVERY_AUTHORIZATION_SHA256", "")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit) or commit == _A3_SOURCE:
        raise LauncherBlocked("RECOVERY_COMMIT debe ser el commit inmutable publicado de recuperación v4.")
    if not isinstance(auth_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", auth_sha):
        raise LauncherBlocked("Falta el SHA256 publicado de la autorización v4.")
    # Old launcher globals are still meaningful if the same kernel opened R4
    # while an R1/R2 controller may be alive or its launch is unconfirmed.
    for prefix in ("E2_RECOVERY_V4", "E2_RECOVERY_V3", "E2_RECOVERY_V2", "E2_RECOVERY_V1", "E2_GUIDE"):
        active = namespace.get(prefix + "_ACTIVE_PROCESS")
        if active is not None and active.poll() is None:
            raise LauncherBlocked("runner_already_active: la celda anterior sigue ejecutándose; conserva su PID y su informe.")
        if namespace.get(prefix + "_LAUNCH_UNCONFIRMED", False):
            raise LauncherBlocked("launch_unconfirmed: revisa los procesos y el diario antes de volver a lanzar la celda.")
    history = namespace.get("E2_RECOVERY_V4_RUN_HISTORY", [])
    if not isinstance(history, list):
        raise LauncherBlocked("El historial de esta celda no es válido; se conservaron sus informes.")
    workspace = Path(namespace.get("WORKSPACE", "/content/entregable2_gpu_workspace")).absolute()
    campaign = workspace / "search-campaigns" / _CAMPAIGN
    _no_symlinks(workspace)
    if not workspace.is_dir():
        raise LauncherBlocked("Falta el entorno preparado de Colab. Ejecuta las secciones 1–7 de preparación; no se creó un workspace ni se instaló nada.")
    campaign_cache_present = _verify_campaign_cache_if_present(campaign)
    campaign_lock = _read_only_campaign_lock(campaign)
    launcher_lock = None
    environment = None
    process = None
    try:
        launcher_lock = _launcher_lock(workspace)
        clean = _clean_environment()
        engine = workspace / "search-source"
        python, data = _verify_prepared_files(workspace, engine)
        _verify_engine(engine, clean)
        _verify_hardware(python, engine, clean)
        controller = workspace / "recovery-v4-source"
        authorization = _checkout_controller(controller, commit, auth_sha, clean)
        preview = _run(_command(python, controller, engine, commit, authorization, auth_sha, campaign, execute=False), clean,
                       cwd=controller, capture=True)
        inspection = json.loads(preview.stdout)
        if not campaign_cache_present:
            print("La caché local de campaña no existe en esta VM. El controlador restaurará y validará la campaña remota antes de cualquier intento; no se creó una campaña vacía.")
        print(json.dumps(inspection, ensure_ascii=False, indent=2, allow_nan=False))
        test_environment = dict(clean, PYTHONPATH=os.pathsep.join([str(controller / "scripts"), str(engine / "scripts")]), E2_ENGINE_ROOT=str(engine))
        for pattern in ("test_e2_manual_reconciliation_v4.py", "test_e2_recovery_receipts_v4.py", "test_e2_cuda_recovery_v4.py", "test_e2_colab_launcher_v4.py"):
            _run([python, "-m", "unittest", "discover", "-s", "tests", "-p", pattern, "-v"], test_environment, cwd=controller)
        _verify_cached_inputs(engine, data)
        environment = _github_environment(clean)
        # Recheck both immutable roots after tests, before any recovery claim.
        _verify_engine(engine, clean)
        _checkout_controller(controller, commit, auth_sha, clean)
        invocation = uuid.uuid4().hex
        report = campaign.parent / ("recovery-v4-" + invocation + ".json")
        if report.exists():
            raise LauncherBlocked("El informe único ya existe; se conservaron sus bytes.")
        command = _command(python, controller, engine, commit, authorization, auth_sha, campaign,
                           execute=True, report=report, data=data)
        # The controller must acquire its own ledger lock. Our launcher lock
        # stays held until graceful process completion to prevent a second cell.
        _unlock(campaign_lock)
        campaign_lock = None
        record = {"invocation_id": invocation, "pid": None, "external_report": str(report),
                  "controller_source_commit": commit, "authorization_sha256": auth_sha, "status": "running"}
        process = None
        namespace["E2_RECOVERY_V4_LAUNCH_PENDING"] = record
        namespace["E2_RECOVERY_V4_LAUNCH_UNCONFIRMED"] = True
        try:
            process = subprocess.Popen(command, cwd=controller, env=environment, text=True, start_new_session=True)
            namespace["E2_RECOVERY_V4_ACTIVE_PROCESS"] = process
            namespace["E2_RECOVERY_V4_LAST_PID"] = process.pid
            namespace["E2_RECOVERY_V4_LAUNCH_UNCONFIRMED"] = False
            record["pid"] = process.pid
            namespace["E2_RECOVERY_V4_RUN_HISTORY"] = history
            history.append(record)
            print("Controlador PID:", process.pid, "Informe único:", report, flush=True)
            returncode, interrupted = _wait_gracefully(process)
        except KeyboardInterrupt:
            # Covers interruption during post-Popen bookkeeping/output as well
            # as the normal blocking wait. Retain admission state until exit.
            if process is None:
                record["status"] = "launch_unconfirmed"
                raise LauncherBlocked("launch_unconfirmed: la interrupción impidió confirmar el PID; revisa los procesos sin repetir el ajuste.") from None
            namespace["E2_RECOVERY_V4_ACTIVE_PROCESS"] = process
            namespace["E2_RECOVERY_V4_LAST_PID"] = process.pid
            namespace["E2_RECOVERY_V4_LAUNCH_UNCONFIRMED"] = False
            _send_interrupt(process)
            returncode, interrupted = _wait_gracefully(process, interrupted=True)
        except OSError:
            if process is None:
                namespace["E2_RECOVERY_V4_LAUNCH_UNCONFIRMED"] = False
            raise LauncherBlocked("No se pudo iniciar o esperar al controlador; conserva el PID si aparece y revisa su estado.") from None
        physical = _json_file(report) if report.is_file() else {}
        result = {"version": "e2-recovery-v4-launcher-v1", "status": physical.get("status", "unconfirmed"),
                  "returncode": returncode, "interrupted": interrupted, "pid": process.pid,
                  "external_report": str(report),
                  "error_code": physical.get("error_code", None if physical else "external_report_missing"),
                  "error_phase": physical.get("error_phase"),
                  "pending_job_ids": physical.get("pending_job_ids"),
                  "logical_completed_jobs": physical.get("logical_completed_jobs"),
                  "campaign_cache_present_at_admission": campaign_cache_present,
                  "initial_campaign_hours": _INITIAL_CAMPAIGN_HOURS,
                  "registered_effective_campaign_hours": _REGISTERED_EFFECTIVE_CAMPAIGN_HOURS,
                  "additional_budget_extension_requested": False}
        record.update(result)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return result
    finally:
        if environment is not None:
            environment.pop("GITHUB_TOKEN", None)
        if process is not None and process.poll() is None:
            # An output/bookkeeping error must not leave an unobserved live
            # controller behind while releasing the launcher admission lock.
            _send_interrupt(process)
            _wait_gracefully(process, interrupted=True)
        _unlock(campaign_lock)
        _unlock(launcher_lock)


if __name__ == "__main__":
    run_launcher(globals())
