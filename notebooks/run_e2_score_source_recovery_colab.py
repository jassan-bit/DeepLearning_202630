"""Resume only frozen-predictor evaluation using the registered source fallback."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import urllib.request

ENGINE_COMMIT = "1e8745497b484518f766b6da5c19be8362d9add7"
FUNDING_CONTINUATION_COMMIT = "41da88cf600aadc0319214465c94a85335e2b1b7"
PRIOR_RESUME_COMMIT = "caf650b67a68226d350ccd6a1f96b10940e0c6a9"
PRIOR_ARIMA_COMMIT = "f12cc0904efefde6854342ef18a6c447d5572b3e"
RESOURCE_COMMIT = "b352ce7f27cd7b55e4141fbfb1548f506ed80987"
PRIOR_LAUNCHER_PATH = "notebooks/run_e2_arima_resume_colab.py"
PRIOR_LAUNCHER_SHA256 = "2ba77c678ee5d42ff803179dc41c9f7ca864e27622bc9b455f4871b04e3f0ba8"
REGISTRATION_PATH = "experiments/e2_score_source_recovery_20261113/registration.json"
POLICY_PATH = "experiments/e2_score_source_recovery_20261113/policy.json"
FILES = ("notebooks/run_e2_score_source_recovery_colab.py",
    "scripts/run_e2_score_source_recovery_delivery.py",
    "scripts/e2_score_source_recovery.py", "scripts/e2_score_archive_transport.py",
    "scripts/e2_score_source_worker.py", POLICY_PATH)


def _download(commit, relative):
    url = "https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/" + commit + "/" + relative
    with urllib.request.urlopen(url, timeout=40) as response:
        raw = response.read(4 * 1024**2 + 1)
    if len(raw) > 4 * 1024**2:
        raise RuntimeError("Una fuente de recuperación supera su límite")
    return raw


def _prior_launcher():
    raw = _download(PRIOR_ARIMA_COMMIT, PRIOR_LAUNCHER_PATH)
    if hashlib.sha256(raw).hexdigest() != PRIOR_LAUNCHER_SHA256:
        raise RuntimeError("El lanzador ARIMA anterior difiere")
    module = {"__name__": "e2_preserved_arima_score_launcher"}
    exec(compile(raw, PRIOR_LAUNCHER_PATH, "exec"), module)
    return module


def _materialize(commit, workspace):
    raw = _download(commit, REGISTRATION_PATH)
    record = json.loads(raw)
    required = {"engine_commit": ENGINE_COMMIT, "protocol_id": "e2-reduced-delivery-v1",
        "funding_continuation_commit": FUNDING_CONTINUATION_COMMIT,
        "prior_resume_commit": PRIOR_RESUME_COMMIT, "prior_arima_commit": PRIOR_ARIMA_COMMIT,
        "resource_commit": RESOURCE_COMMIT}
    if (any(record.get(k) != v for k, v in required.items())
            or not set(FILES).issubset(record.get("files_sha256", {}))):
        raise RuntimeError("El registro de evaluación no corresponde a la campaña existente")
    root = Path(workspace) / ("github-score-source-recovery-" + commit)
    if root.is_symlink():
        raise RuntimeError("La extensión no puede ser un enlace simbólico")
    for relative in FILES:
        payload = _download(commit, relative)
        if hashlib.sha256(payload).hexdigest() != record["files_sha256"][relative]:
            raise RuntimeError("Una fuente de evaluación difiere: " + relative)
        target = root / relative
        if (not target.resolve().is_relative_to(root.resolve())
                or any(p.is_symlink() for p in (target, *target.parents) if p.is_relative_to(root))):
            raise RuntimeError("Ruta de recuperación local no segura")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != payload:
            raise RuntimeError("Se conserva una extensión de evaluación modificada para revisión")
        if not target.exists():
            with target.open("xb") as stream:
                stream.write(payload)
    target = root / "registration.json"
    if target.is_symlink() or target.exists() and target.read_bytes() != raw:
        raise RuntimeError("El registro local de evaluación difiere")
    if not target.exists():
        with target.open("xb") as stream:
            stream.write(raw)
    return root


def _instrument_chain(module, keys, recovery_commit):
    """Insert below every preserved launcher so its final command is visible."""
    if keys:
        key, *remaining = keys
        factory = module[key]
        def nested():
            return _instrument_chain(factory(), remaining, recovery_commit)
        module[key] = nested
        return module
    stream = module["_stream"]
    def score_stream(command, *args):
        command = list(command)
        if Path(command[1]).name != "run_e2_arima_resume_delivery.py":
            raise RuntimeError("Ruta inesperada del controlador de evaluación")
        root = _materialize(recovery_commit, module["LOCAL_WORKSPACE"])
        command[1] = root / "scripts/run_e2_score_source_recovery_delivery.py"
        command.extend(["--score-source-root", root, "--score-source-commit", recovery_commit])
        return stream(command, *args)
    module["_stream"] = score_stream
    module["CONTROLLERS"].update({"run_e2_score_source_recovery_delivery.py", "e2_score_source_worker.py"})
    return module


def run_e2_score_source_recovery_colab(recovery_commit, *, namespace=None, execute=False):
    if not isinstance(recovery_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", recovery_commit):
        raise ValueError("Usa el commit inmutable de la recuperación de evaluación")
    if type(execute) is not bool:
        raise TypeError("execute debe ser booleano")
    if not execute:
        return {"training_started": False, "new_budget_authorization": False,
            "requested_stages": ["score"], "predictor_refit_requested": False}
    prior = _instrument_chain(_prior_launcher(),
        ["_prior_launcher", "_prior_launcher", "_prior_launcher", "_funding_launcher", "_base_launcher"],
        recovery_commit)
    result = prior["run_e2_arima_resume_colab"](PRIOR_ARIMA_COMMIT,
        namespace=globals() if namespace is None else namespace, execute=True)
    if isinstance(result, dict):
        result = {**result, "score_source_recovery_commit": recovery_commit,
            "new_budget_authorization": False, "predictor_refit_requested": False,
            "requested_stages": ["score"]}
    return result
