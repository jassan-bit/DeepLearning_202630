"""Resume the unchanged source-recovery evaluation with a metadata receipt fix."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import urllib.request

ENGINE_COMMIT = "1e8745497b484518f766b6da5c19be8362d9add7"
RESOURCE_COMMIT = "b352ce7f27cd7b55e4141fbfb1548f506ed80987"
FUNDING_CONTINUATION_COMMIT = "41da88cf600aadc0319214465c94a85335e2b1b7"
PRIOR_RESUME_COMMIT = "caf650b67a68226d350ccd6a1f96b10940e0c6a9"
PRIOR_ARIMA_COMMIT = "f12cc0904efefde6854342ef18a6c447d5572b3e"
PRIOR_SCORE_SOURCE_COMMIT = "24344bd6db20f0546b26167eda4c3076a38c0f4b"
PRIOR_LAUNCHER_PATH = "notebooks/run_e2_score_source_recovery_colab.py"
PRIOR_LAUNCHER_SHA256 = "36d4f48933b8fe2cf33db340074d2774bace6d99dbcc4dbee9119c6699d5267f"
REGISTRATION_PATH = "experiments/e2_score_receipt_compat_20261113/registration.json"
FILES = ("notebooks/run_e2_score_receipt_compat_colab.py",
    "scripts/run_e2_score_receipt_compat_delivery.py", "scripts/e2_score_receipt_compat.py")


def _download(commit, relative):
    url = "https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/" + commit + "/" + relative
    with urllib.request.urlopen(url, timeout=40) as response:
        raw = response.read(4 * 1024**2 + 1)
    if len(raw) > 4 * 1024**2:
        raise RuntimeError("Una fuente de compatibilidad supera su límite")
    return raw


def _prior_launcher():
    raw = _download(PRIOR_SCORE_SOURCE_COMMIT, PRIOR_LAUNCHER_PATH)
    if hashlib.sha256(raw).hexdigest() != PRIOR_LAUNCHER_SHA256:
        raise RuntimeError("El lanzador de evaluación anterior difiere")
    module = {"__name__": "e2_preserved_score_source_receipt_launcher"}
    exec(compile(raw, PRIOR_LAUNCHER_PATH, "exec"), module)
    return module


def _materialize(commit, workspace):
    raw = _download(commit, REGISTRATION_PATH)
    record = json.loads(raw)
    required = {"engine_commit": ENGINE_COMMIT, "protocol_id": "e2-reduced-delivery-v1",
        "resource_commit": RESOURCE_COMMIT, "funding_continuation_commit": FUNDING_CONTINUATION_COMMIT,
        "prior_resume_commit": PRIOR_RESUME_COMMIT, "prior_arima_commit": PRIOR_ARIMA_COMMIT,
        "prior_score_source_commit": PRIOR_SCORE_SOURCE_COMMIT}
    if (any(record.get(k) != v for k,v in required.items())
            or not set(FILES).issubset(record.get("files_sha256", {}))):
        raise RuntimeError("El registro del recibo no corresponde a la campaña existente")
    root = Path(workspace) / ("github-score-receipt-compat-" + commit)
    if root.is_symlink():
        raise RuntimeError("La extensión no puede ser un enlace simbólico")
    for relative in FILES:
        payload = _download(commit, relative)
        if hashlib.sha256(payload).hexdigest() != record["files_sha256"][relative]:
            raise RuntimeError("Una fuente de compatibilidad difiere: " + relative)
        target = root / relative
        if (not target.resolve().is_relative_to(root.resolve())
                or any(p.is_symlink() for p in (target, *target.parents) if p.is_relative_to(root))):
            raise RuntimeError("Ruta local de compatibilidad no segura")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != payload:
            raise RuntimeError("Se conserva una extensión local modificada para revisión")
        if not target.exists():
            with target.open("xb") as stream:
                stream.write(payload)
    target = root / "registration.json"
    if target.is_symlink() or target.exists() and target.read_bytes() != raw:
        raise RuntimeError("El registro local del recibo difiere")
    if not target.exists():
        with target.open("xb") as stream:
            stream.write(raw)
    return root


def _instrument_chain(module, keys, repair_commit):
    if keys:
        key, *remaining = keys
        factory = module[key]
        def nested():
            return _instrument_chain(factory(), remaining, repair_commit)
        module[key] = nested
        return module
    stream = module["_stream"]
    def receipt_stream(command, *args):
        command = list(command)
        if Path(command[1]).name != "run_e2_score_source_recovery_delivery.py":
            raise RuntimeError("Ruta inesperada del controlador de recibos")
        root = _materialize(repair_commit, module["LOCAL_WORKSPACE"])
        command[1] = root / "scripts/run_e2_score_receipt_compat_delivery.py"
        command.extend(["--receipt-compat-root", root, "--receipt-compat-commit", repair_commit])
        return stream(command, *args)
    module["_stream"] = receipt_stream
    module["CONTROLLERS"].add("run_e2_score_receipt_compat_delivery.py")
    return module


def run_e2_score_receipt_compat_colab(repair_commit, *, namespace=None, execute=False):
    if not isinstance(repair_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", repair_commit):
        raise ValueError("Usa el commit inmutable de la compatibilidad del recibo")
    if type(execute) is not bool:
        raise TypeError("execute debe ser booleano")
    if not execute:
        return {"training_started": False, "new_budget_authorization": False,
            "requested_stages": ["score"], "predictor_refit_requested": False}
    prior = _instrument_chain(_prior_launcher(),
        ["_prior_launcher", "_prior_launcher", "_prior_launcher", "_prior_launcher",
            "_funding_launcher", "_base_launcher"], repair_commit)
    result = prior["run_e2_score_source_recovery_colab"](PRIOR_SCORE_SOURCE_COMMIT,
        namespace=globals() if namespace is None else namespace, execute=True)
    if isinstance(result, dict):
        result = {**result, "receipt_compat_commit": repair_commit,
            "new_budget_authorization": False, "predictor_refit_requested": False,
            "requested_stages": ["score"]}
    return result
