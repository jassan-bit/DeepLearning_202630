"""Resume final E2 stages only after an explicit 80-hour confirmation in Colab.

GitHub remains the sole persistent destination. This launcher does not authorize
the proposed increase by itself: the user must enter 80 before it can execute.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import urllib.request

ENGINE_COMMIT = "1e8745497b484518f766b6da5c19be8362d9add7"
PRIOR_COMMIT = "72c04635b7a746d06066273ce61e891975feb510"
PRIOR_LAUNCHER = "notebooks/run_e2_a100_recovery_colab.py"
PRIOR_SHA256 = "53b4fb5a18585269b4ac8c492bb8a76acd5d17bdacd2a91efbe0a13d6aa15b47"
REGISTRATION_PATH = "experiments/e2_budget_continuation_20261113/registration.json"
PROPOSAL_PATH = "experiments/e2_budget_continuation_20261113/proposal.json"
FILES = ("notebooks/run_e2_final_continuation_colab.py",
    "scripts/run_e2_final_continuation_delivery.py",
    "scripts/e2_confirmed_budget_continuation.py", PROPOSAL_PATH)


def _public_bytes(commit, relative):
    url = "https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/" + commit + "/" + relative
    with urllib.request.urlopen(url, timeout=30) as response:
        raw = response.read(2 * 1024**2 + 1)
    if len(raw) > 2 * 1024**2:
        raise RuntimeError("Una fuente de continuación supera el límite admitido")
    return raw


def _prior_launcher():
    raw = _public_bytes(PRIOR_COMMIT, PRIOR_LAUNCHER)
    if hashlib.sha256(raw).hexdigest() != PRIOR_SHA256:
        raise RuntimeError("La fuente del controlador A100 anterior difiere")
    module = {"__name__": "e2_pinned_a100_continuation_parent"}
    exec(compile(raw, PRIOR_LAUNCHER, "exec"), module)
    return module


def _materialize(commit, workspace):
    raw = _public_bytes(commit, REGISTRATION_PATH)
    registration = json.loads(raw)
    if (registration.get("engine_commit") != ENGINE_COMMIT
            or registration.get("protocol_id") != "e2-reduced-delivery-v1"
            or registration.get("prior_controller_commit") != PRIOR_COMMIT
            or not set(FILES).issubset(registration.get("files_sha256", {}))):
        raise RuntimeError("La propuesta no corresponde a esta campaña registrada")
    root = Path(workspace) / ("github-final-continuation-" + commit)
    if root.is_symlink():
        raise RuntimeError("La extensión local no puede ser un enlace simbólico")
    for relative in FILES:
        payload = _public_bytes(commit, relative)
        if hashlib.sha256(payload).hexdigest() != registration["files_sha256"][relative]:
            raise RuntimeError("Una fuente de continuación difiere: " + relative)
        target = root / relative
        if (not target.resolve().is_relative_to(root.resolve())
                or any(p.is_symlink() for p in (target, *target.parents) if p.is_relative_to(root))):
            raise RuntimeError("Una ruta de continuación local no es segura")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != payload:
            raise RuntimeError("Se conserva una extensión local modificada para revisión")
        if not target.exists():
            with target.open("xb") as stream:
                stream.write(payload)
    target = root / "registration.json"
    if target.exists() and target.read_bytes() != raw:
        raise RuntimeError("El registro local de continuación difiere")
    if not target.exists():
        with target.open("xb") as stream:
            stream.write(raw)
    return root


def _approval(root, commit, confirmed_text, observed_at_utc):
    spec = importlib.util.spec_from_file_location("e2_confirmed_funding_for_launcher",
        root / "scripts/e2_confirmed_budget_continuation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_approval_receipt(continuation_commit=commit,
        confirmed_text=confirmed_text, observed_at_utc=observed_at_utc)


def verified_continuation_summary(result, continuation_commit):
    if not isinstance(result, dict):
        return result
    result = dict(result)
    result["continuation_commit"] = continuation_commit
    result["budget_continuation_applied"] = False
    try:
        anchor = json.loads((Path(result["temporary_workspace"])
            / "shared_budget/budget_continuation_80.json").read_text())
        unsigned = dict(anchor)
        checksum = unsigned.pop("sha256")
        canonical = json.dumps(unsigned, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()
        if (hashlib.sha256(canonical).hexdigest() != checksum
                or anchor.get("effective_total_hours") != 80
                or anchor.get("continuation_registration_commit") != continuation_commit):
            return result
    except (OSError, ValueError, KeyError, TypeError):
        return result
    result.update(effective_total_budget_hours=80, budget_continuation_applied=True,
        funding_summary_status="verified_80_hour_continuation_anchor")
    return result


def run_e2_final_continuation_colab(continuation_commit, *, namespace=None,
        execute=False, input_fn=None):
    if not isinstance(continuation_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", continuation_commit):
        raise ValueError("Usa el commit inmutable de la continuación")
    if type(execute) is not bool:
        raise TypeError("execute debe ser booleano")
    if not execute:
        return {"training_started": False, "budget_continuation_applied": False,
            "proposed_total_hours": 80, "persistent_destination": "github"}
    answer = (input if input_fn is None else input_fn)(
        "Autorizar 80 horas TOTALES (+8 al límite de 72), conservando todos los cargos y resultados. "
        "No compra unidades de Colab. Escribe 80 para continuar; vacío para cancelar: ").strip()
    if answer != "80":
        print("Cancelado: no se amplió el presupuesto ni se inició entrenamiento.", flush=True)
        return {"training_started": False, "budget_continuation_applied": False,
            "cancelled": True, "persistent_destination": "github"}
    approved_at = datetime.now(timezone.utc).isoformat()
    namespace = globals() if namespace is None else namespace
    prior = _prior_launcher()
    previous_funding_launcher = prior["_funding_launcher"]
    def funding_launcher():
        funding = previous_funding_launcher()
        previous_base = funding["_base_launcher"]
        def base_launcher():
            base = previous_base()
            original_stream = base["_stream"]
            def continued_stream(command, *args):
                command = list(command)
                if Path(command[1]).name != "run_e2_a100_recovery_delivery.py":
                    raise RuntimeError("Ruta inesperada del controlador de continuación")
                root = _materialize(continuation_commit, base["LOCAL_WORKSPACE"])
                approval = _approval(root, continuation_commit, answer, approved_at)
                command[1] = root / "scripts/run_e2_final_continuation_delivery.py"
                command.extend(["--continuation-root", root, "--continuation-commit", continuation_commit,
                    "--approval-receipt", json.dumps(approval, ensure_ascii=False)])
                return original_stream(command, *args)
            base["_stream"] = continued_stream
            base["CONTROLLERS"].add("run_e2_final_continuation_delivery.py")
            original_json = base["json"]
            class SummaryJSON:
                def __getattr__(self, name):
                    return getattr(original_json, name)
                def dumps(self, value, *args, **kwargs):
                    if isinstance(value, dict) and "temporary_workspace" in value:
                        value = verified_continuation_summary(value, continuation_commit)
                    return original_json.dumps(value, *args, **kwargs)
            base["json"] = SummaryJSON()
            return base
        funding["_base_launcher"] = base_launcher
        return funding
    prior["_funding_launcher"] = funding_launcher
    result = prior["run_e2_a100_recovery_colab"](PRIOR_COMMIT,
        namespace=namespace, execute=True)
    return verified_continuation_summary(result, continuation_commit)
