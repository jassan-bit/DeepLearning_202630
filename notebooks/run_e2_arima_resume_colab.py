"""Continue classical comparators and scoring under the existing 80-hour consent."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import urllib.request

ENGINE_COMMIT = "1e8745497b484518f766b6da5c19be8362d9add7"
FUNDING_CONTINUATION_COMMIT = "41da88cf600aadc0319214465c94a85335e2b1b7"
PRIOR_RESUME_COMMIT = "caf650b67a68226d350ccd6a1f96b10940e0c6a9"
RESOURCE_COMMIT = "b352ce7f27cd7b55e4141fbfb1548f506ed80987"
PRIOR_LAUNCHER_PATH = "notebooks/run_e2_final80_resume_colab.py"
PRIOR_LAUNCHER_SHA256 = "f21fe7be8f380c4862db7f3b3e0d843ba86df8d47870468284d961b76ff83464"
REGISTRATION_PATH = "experiments/e2_arima_compat_20261113/registration.json"
POLICY_PATH = "experiments/e2_arima_compat_20261113/policy.json"
FILES = ("notebooks/run_e2_arima_resume_colab.py", "scripts/run_e2_arima_resume_delivery.py",
    "scripts/e2_arima_prediction_compat.py", "scripts/e2_arima_compat_worker.py",
    "scripts/e2_arima_classical_continuation.py", POLICY_PATH)


def _download(commit, relative):
    url = "https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/" + commit + "/" + relative
    with urllib.request.urlopen(url, timeout=40) as response:
        raw = response.read(4 * 1024**2 + 1)
    if len(raw) > 4 * 1024**2:
        raise RuntimeError("Una fuente de continuación supera su límite")
    return raw


def _prior_launcher():
    raw = _download(PRIOR_RESUME_COMMIT, PRIOR_LAUNCHER_PATH)
    if hashlib.sha256(raw).hexdigest() != PRIOR_LAUNCHER_SHA256:
        raise RuntimeError("El lanzador de reanudación anterior difiere")
    module = {"__name__": "e2_preserved_final80_resume_launcher"}
    exec(compile(raw, PRIOR_LAUNCHER_PATH, "exec"), module)
    return module


def _materialize(commit, workspace):
    raw = _download(commit, REGISTRATION_PATH)
    record = json.loads(raw)
    required = {"engine_commit": ENGINE_COMMIT, "protocol_id": "e2-reduced-delivery-v1",
        "funding_continuation_commit": FUNDING_CONTINUATION_COMMIT,
        "prior_resume_commit": PRIOR_RESUME_COMMIT, "resource_commit": RESOURCE_COMMIT}
    if (any(record.get(k) != v for k, v in required.items())
            or not set(FILES).issubset(record.get("files_sha256", {}))):
        raise RuntimeError("El registro ARIMA no corresponde a la campaña existente")
    root = Path(workspace) / ("github-arima-compat-" + commit)
    if root.is_symlink():
        raise RuntimeError("La extensión no puede ser un enlace simbólico")
    for relative in FILES:
        payload = _download(commit, relative)
        if hashlib.sha256(payload).hexdigest() != record["files_sha256"][relative]:
            raise RuntimeError("Una fuente ARIMA difiere: " + relative)
        target = root / relative
        if (not target.resolve().is_relative_to(root.resolve())
                or any(p.is_symlink() for p in (target, *target.parents) if p.is_relative_to(root))):
            raise RuntimeError("Ruta de continuación local no segura")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != payload:
            raise RuntimeError("Se conserva una extensión ARIMA modificada para revisión")
        if not target.exists():
            with target.open("xb") as stream:
                stream.write(payload)
    target = root / "registration.json"
    if target.is_symlink() or target.exists() and target.read_bytes() != raw:
        raise RuntimeError("El registro local ARIMA difiere")
    if not target.exists():
        with target.open("xb") as stream:
            stream.write(raw)
    return root


def run_e2_arima_resume_colab(repair_commit, *, namespace=None, execute=False):
    if not isinstance(repair_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", repair_commit):
        raise ValueError("Usa el commit inmutable de la corrección ARIMA")
    if type(execute) is not bool:
        raise TypeError("execute debe ser booleano")
    if not execute:
        return {"training_started": False, "new_budget_authorization": False,
            "requested_stages": ["classical", "freeze", "score"]}
    prior = _prior_launcher()
    previous_prior = prior["_prior_launcher"]
    def f80_launcher():
        f80 = previous_prior()
        previous_q = f80["_prior_launcher"]
        def q_launcher():
            q = previous_q()
            previous_funding = q["_funding_launcher"]
            def funding_launcher():
                funding = previous_funding()
                previous_base = funding["_base_launcher"]
                def base_launcher():
                    base = previous_base()
                    stream = base["_stream"]
                    def arima_stream(command, *args):
                        command = list(command)
                        if Path(command[1]).name != "run_e2_final80_resume_delivery.py":
                            raise RuntimeError("Ruta inesperada del controlador de comparadores")
                        root = _materialize(repair_commit, base["LOCAL_WORKSPACE"])
                        command[1] = root / "scripts/run_e2_arima_resume_delivery.py"
                        command.extend(["--arima-root", root, "--arima-commit", repair_commit])
                        return stream(command, *args)
                    base["_stream"] = arima_stream
                    base["CONTROLLERS"].update({"run_e2_arima_resume_delivery.py", "e2_arima_compat_worker.py"})
                    return base
                funding["_base_launcher"] = base_launcher
                return funding
            q["_funding_launcher"] = funding_launcher
            return q
        f80["_prior_launcher"] = q_launcher
        return f80
    prior["_prior_launcher"] = f80_launcher
    result = prior["run_e2_final80_resume_colab"](PRIOR_RESUME_COMMIT,
        namespace=globals() if namespace is None else namespace, execute=True)
    if isinstance(result, dict):
        result = {**result, "arima_repair_commit": repair_commit,
            "new_budget_authorization": False, "neural_final_fits_requested": False,
            "requested_stages": ["classical", "freeze", "score"]}
    return result
