"""CPU-only Colab handoff for the existing five-model public Render client.

Prepared with disclosed AI assistance. No network operation runs on import.
The operator must supply the actual deployment SHA from Render's deploy log.
This client neither trains a model nor recalibrates an interval.
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
import zipfile

REPOSITORY = "https://github.com/jassan-bit/DeepLearning_202630.git"
URL = "https://deeplearning-e3-dashboard.onrender.com"
CLIENT = "experiments/e3_cierre_integral_20261118/acceptance/smoke_render_v2.py"
HELPER = "experiments/e3_bloques_fg_20261117/smoke_render.py"
EXPECTED_CLIENTS = {
    CLIENT: "54ba00ebf49308fe525f7f4b27b4befa5f98912f2abdc61fc2d818eb314583ad",
    HELPER: "4dbbc83603d28224cfef4d1f8159eab55fa8dbef7c23081b5feebccf5e039190",
}
PATTERNS = (
    "/" + CLIENT, "/" + HELPER,
    "/data/e3_dashboard_v1/examples.json", "/data/e3_dashboard_v1/predictions.json",
    "/artifacts/e3_dashboard_v1/", "/artifacts/e3_serving_extensions_v2/",
    "/artifacts/e3_serving_calibration_v2/",
    "/experiments/e3_cierre_integral_20261118/serving/export_capabilities/tsmixer/cpu_examples.json",
    "/MODEL_CARD.md", "/DATA_STATEMENT.md", "/assets/vendor/flatly-5.3.6.min.css",
)
CLIENT_PACKAGES = ("playwright==1.56.0", "greenlet==3.5.6", "pyee==13.0.1")


def safe(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("No se admiten rutas con enlaces simbólicos.")
    return path


def clean_environment():
    secret = re.compile(r"TOKEN|SECRET|PASSWORD|CREDENTIAL|(?:^|_)PAT(?:_|$)", re.I)
    env = {k: v for k, v in os.environ.items()
           if not secret.search(k) and not k.startswith(("GIT_", "PIP_", "UV_"))
           and k not in {"SSH_ASKPASS", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
               GIT_TERMINAL_PROMPT="0", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK="1")
    return env


def call(command, *, cwd=None, env=None, data=None):
    return subprocess.run(command, cwd=cwd, env=env, input=data, check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90).stdout


def checkout_source(deployment_commit, root, env):
    """Preserve an interrupted cache before retrying the same immutable source."""
    source = safe(root / ("e3-render-client-" + deployment_commit))
    if source.exists():
        if not source.is_dir():
            raise ValueError("La ruta del cliente conservada no es una carpeta; preserva sus archivos.")
        try:
            actual = call(["git", "rev-parse", "HEAD"], cwd=source, env=env).decode().strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            actual = None
        if actual is not None and actual != deployment_commit:
            raise ValueError("El checkout conservado tiene otro HEAD; preserva sus archivos.")
        intact = actual is not None
        if intact:
            try:
                call(["git", "diff", "--exit-code", "HEAD"], cwd=source, env=env)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                intact = False
            intact = intact and all((source / name).is_file() for name in (CLIENT, HELPER))
        if not intact:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            preserved = source.with_name(source.name + ".incomplete-" + stamp)
            source.rename(preserved)
            print("Caché de fuente incompleta conservada aparte:", str(preserved), flush=True)
    if not source.exists():
        source.mkdir()
        call(["git", "init", "--quiet"], cwd=source, env=env)
        call(["git", "remote", "add", "origin", REPOSITORY], cwd=source, env=env)
        call(["git", "config", "remote.origin.promisor", "true"], cwd=source, env=env)
        call(["git", "config", "remote.origin.partialclonefilter", "blob:none"], cwd=source, env=env)
        call(["git", "sparse-checkout", "init", "--no-cone"], cwd=source, env=env)
        call(["git", "sparse-checkout", "set", "--no-cone", "--stdin"], cwd=source,
             env=env, data=("\n".join(PATTERNS) + "\n").encode())
        call(["git", "fetch", "--quiet", "--no-tags", "--depth=1", "--filter=blob:none",
              "origin", deployment_commit], cwd=source, env=env)
        call(["git", "checkout", "--quiet", "--detach", deployment_commit], cwd=source, env=env)
    actual = call(["git", "rev-parse", "HEAD"], cwd=source, env=env).decode().strip()
    if actual != deployment_commit:
        raise ValueError("El checkout no corresponde al SHA declarado; preserva sus archivos.")
    call(["git", "diff", "--exit-code", "HEAD"], cwd=source, env=env)
    return source


def main(deployment_commit, *, base="/content", download=True):
    """Check the operator-confirmed deployment; return the local evidence ZIP."""
    if not isinstance(deployment_commit, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", deployment_commit):
        raise ValueError("Pega los 40 caracteres del SHA efectivo mostrado por Render.")
    deployment_commit = deployment_commit.lower()
    root = safe(base)
    root.mkdir(parents=True, exist_ok=True)
    env = clean_environment()
    source = checkout_source(deployment_commit, root, env)
    actual = deployment_commit
    for name, expected in EXPECTED_CLIENTS.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError("El cliente desplegado difiere del V2 revisado: " + name)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = safe(root / ("e3-render-public-smoke-" + stamp))
    output.mkdir()
    runtime = safe(root / "e3-render-browser-client-v1")
    python = runtime / "bin/python"
    if not python.exists():
        call([sys.executable, "-m", "venv", str(runtime)], env=env)
    env["PLAYWRIGHT_BROWSERS_PATH"] = str(runtime / "browsers")
    subprocess.run([str(python), "-m", "pip", "install", *CLIENT_PACKAGES], env=env, check=True)
    subprocess.run([str(python), "-m", "playwright", "install", "--with-deps", "chromium"], env=env, check=True)
    commands = [str(python), str(source / CLIENT), "--url", URL,
                "--deployment-commit", deployment_commit, "--output", str(output / "report.json"),
                "--screenshots-dir", str(output / "captures")]
    context = {"scope": "existing_V2_public_browser_client", "operator_confirmed_deployment_commit": deployment_commit,
               "operator_confirmation_required": "SHA copied from the actual Render deployment log, not inferred from /healthz",
               "source_checkout": actual, "client_sha256": EXPECTED_CLIENTS,
               "client_packages": list(CLIENT_PACKAGES), "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "new_fits": 0, "new_calibrations": 0, "GPU_required": False,
               "full_796_campaign_reproduced": False, "AI_assistance_declared": True}
    (output / "execution_context.json").write_text(json.dumps(context, ensure_ascii=False, indent=2) + "\n")
    with (output / "client_stdout.log").open("wb") as stream:
        run = subprocess.run(commands, cwd=source, env=env, stdout=stream, stderr=subprocess.STDOUT)
    report = json.loads((output / "report.json").read_text()) if (output / "report.json").exists() else None
    (output / "command_result.json").write_text(json.dumps({"exit_code": run.returncode,
          "receipt_present": report is not None, "render_public_smoke_passed": bool(report and report.get("render_public_smoke_passed")),
          "finished_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        for file in sorted(output.rglob("*")):
            if file.is_file():
                zipped.write(file, output.name + "/" + str(file.relative_to(output)))
    print(json.dumps({"exit_code": run.returncode, "render_public_smoke_passed": bool(report and report.get("render_public_smoke_passed")),
                      "output": str(output), "archive": str(archive),
                      "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}, ensure_ascii=False))
    if download:
        from google.colab import files
        files.download(str(archive))
    if run.returncode:
        print("La aceptación falló; conserva el recibo y el ZIP. No se marca Render como aprobado.")
    return archive


if __name__ == "__main__":
    main(input("SHA efectivo de Render (40 caracteres, copiado del log de despliegue): ").strip())
