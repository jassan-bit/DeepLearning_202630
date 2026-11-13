"""Resume the already-authorized 80-hour E2 campaign without new consent."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re
import time
import urllib.request
import zipfile

ENGINE_COMMIT = "1e8745497b484518f766b6da5c19be8362d9add7"
FUNDING_CONTINUATION_COMMIT = "41da88cf600aadc0319214465c94a85335e2b1b7"
PRIOR_CONTROLLER_COMMIT = "72c04635b7a746d06066273ce61e891975feb510"
PRIOR_LAUNCHER_PATH = "notebooks/run_e2_final_continuation_colab.py"
PRIOR_LAUNCHER_SHA256 = "5a04b0787083c45e5b619dbedd18330be2ec396e1d0b730d95a64e24af21bddd"
REGISTRATION_PATH = "experiments/e2_final80_resume_fix_20261113/registration.json"
FILES = ("notebooks/run_e2_final80_resume_colab.py", "scripts/run_e2_final80_resume_delivery.py",
    "scripts/e2_sealed_transition_resume.py", "scripts/e2_verified_local_restore.py")
HEAD_URL = ("https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/"
    "results/e2-reduced-delivery-v1/control/github-primary-v1/head.json")
WORKSPACE = Path("/content/entregable2_gpu_workspace")
ANCHOR_PATH = "shared_budget/budget_continuation_80.json"
APPROVAL_PATH = "shared_budget/budget_continuation_80_approval.json"


def _download(url, limit=4 * 1024**2):
    request = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})
    with urllib.request.urlopen(request, timeout=40) as response:
        raw = response.read(limit + 1)
    if len(raw) > limit:
        raise RuntimeError("Un archivo remoto supera su límite registrado")
    return raw


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()


def _checked(raw):
    value = json.loads(raw)
    unsigned = dict(value)
    checksum = unsigned.pop("sha256", None)
    if hashlib.sha256(_canonical(unsigned)).hexdigest() != checksum:
        raise RuntimeError("El consentimiento o ancla de 80 horas difiere")
    return value


def _bound(raw, identity):
    return len(raw) == identity.get("bytes") and hashlib.sha256(raw).hexdigest() == identity.get("sha256")


def existing_approval(workspace=WORKSPACE):
    """Read-only: require the original consent bound by the latest GitHub head."""
    head = json.loads(_download(HEAD_URL + "?e2_resume=" + str(time.time_ns())))
    body = {key: value for key, value in head.items() if key != "state_id"}
    if (head.get("scientific_commit") != ENGINE_COMMIT
            or head.get("version") != "e2-github-primary-state-v1"
            or head.get("state_id") != hashlib.sha256(_canonical(body)).hexdigest()
            or head.get("reason") != "shared_budget_closed"
            or head.get("persistent_destination") != "GitHub"):
        raise RuntimeError("La campaña remota no está cerrada y preparada para reanudarse")
    required = (ANCHOR_PATH, APPROVAL_PATH)
    if any(path not in head.get("files", {}) for path in required):
        raise RuntimeError("GitHub todavía no contiene la autorización previa de 80 horas")
    found = {}
    output = Path(workspace) / "github-primary-output-v1"
    for relative in required:
        path = output / relative
        if path.is_file() and not path.is_symlink():
            raw = path.read_bytes()
            if _bound(raw, head["files"][relative]):
                found[relative] = raw
    if len(found) != 2:
        print("Verificando desde GitHub el consentimiento ya registrado; no se solicita otra ampliación.", flush=True)
        for delta in reversed(head.get("metadata_deltas", [])):
            url = ("https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/"
                "results/e2-reduced-delivery-v1/" + delta["path"])
            payload = _download(url, 16 * 1024**2)
            if len(payload) != delta.get("bytes") or hashlib.sha256(payload).hexdigest() != delta.get("sha256"):
                raise RuntimeError("Una copia de control de GitHub difiere")
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                for relative in required:
                    if relative in found or relative not in archive.namelist():
                        continue
                    if archive.getinfo(relative).file_size != head["files"][relative]["bytes"]:
                        continue
                    raw = archive.read(relative)
                    if _bound(raw, head["files"][relative]):
                        found[relative] = raw
            if len(found) == 2:
                break
    if len(found) != 2:
        raise RuntimeError("No se pudo recuperar el consentimiento original de 80 horas")
    anchor = _checked(found[ANCHOR_PATH])
    approval = _checked(found[APPROVAL_PATH])
    if (anchor.get("version") != "e2-confirmed-budget-continuation-v1"
            or anchor.get("effective_total_hours") != 80
            or anchor.get("previous_total_hours") != 72
            or anchor.get("budget_reset") is not False
            or anchor.get("continuation_registration_commit") != FUNDING_CONTINUATION_COMMIT
            or anchor.get("approval_sha256") != approval.get("sha256")
            or anchor.get("approval_receipt") != approval
            or approval.get("approved_total_hours") != 80
            or approval.get("confirmed_text") != "80"
            or approval.get("source") != "explicit_colab_confirmation"
            or approval.get("continuation_registration_commit") != FUNDING_CONTINUATION_COMMIT):
        raise RuntimeError("La autorización persistida no corresponde a esta continuación de 80 horas")
    return approval


def _prior_launcher():
    url = ("https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/"
        + FUNDING_CONTINUATION_COMMIT + "/" + PRIOR_LAUNCHER_PATH)
    raw = _download(url)
    if hashlib.sha256(raw).hexdigest() != PRIOR_LAUNCHER_SHA256:
        raise RuntimeError("El lanzador original de 80 horas difiere")
    module = {"__name__": "e2_original_80_hour_launcher"}
    exec(compile(raw, PRIOR_LAUNCHER_PATH, "exec"), module)
    return module


def _materialize(commit, workspace):
    prefix = "https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/" + commit + "/"
    raw = _download(prefix + REGISTRATION_PATH)
    record = json.loads(raw)
    if (record.get("engine_commit") != ENGINE_COMMIT
            or record.get("protocol_id") != "e2-reduced-delivery-v1"
            or record.get("funding_continuation_commit") != FUNDING_CONTINUATION_COMMIT
            or record.get("prior_controller_commit") != PRIOR_CONTROLLER_COMMIT
            or not set(FILES).issubset(record.get("files_sha256", {}))):
        raise RuntimeError("El registro de reparación no corresponde a esta campaña")
    root = Path(workspace) / ("github-final80-resume-" + commit)
    if root.is_symlink():
        raise RuntimeError("La extensión no puede ser un enlace simbólico")
    for relative in FILES:
        payload = _download(prefix + relative)
        if hashlib.sha256(payload).hexdigest() != record["files_sha256"][relative]:
            raise RuntimeError("Una fuente de reparación difiere: " + relative)
        target = root / relative
        if (not target.resolve().is_relative_to(root.resolve())
                or any(p.is_symlink() for p in (target, *target.parents) if p.is_relative_to(root))):
            raise RuntimeError("Ruta de reparación local no segura")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != payload:
            raise RuntimeError("Se conserva una reparación local modificada para revisión")
        if not target.exists():
            with target.open("xb") as stream:
                stream.write(payload)
    registration = root / "registration.json"
    if registration.is_symlink() or registration.exists() and registration.read_bytes() != raw:
        raise RuntimeError("El registro local de reparación difiere")
    if not registration.exists():
        with registration.open("xb") as stream:
            stream.write(raw)
    return root


def run_e2_final80_resume_colab(fix_commit, *, namespace=None, execute=False):
    if not isinstance(fix_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", fix_commit):
        raise ValueError("Usa el commit inmutable de la reparación")
    if type(execute) is not bool:
        raise TypeError("execute debe ser booleano")
    if not execute:
        return {"training_started": False, "new_budget_authorization": False,
            "funding_continuation_commit": FUNDING_CONTINUATION_COMMIT}
    approval = existing_approval()
    print("Autorización existente de 80 horas verificada; se conserva su fecha y presupuesto original.", flush=True)
    old = _prior_launcher()
    previous_prior = old["_prior_launcher"]
    def approved_receipt(root, commit, confirmed_text, observed_at_utc):
        if commit != FUNDING_CONTINUATION_COMMIT or confirmed_text != "80":
            raise RuntimeError("La autoridad presupuestaria original debe conservarse")
        return deepcopy(approval)
    old["_approval"] = approved_receipt
    def prior_launcher():
        prior = previous_prior()
        previous_funding = prior["_funding_launcher"]
        def funding_launcher():
            funding = previous_funding()
            previous_base = funding["_base_launcher"]
            def base_launcher():
                base = previous_base()
                stream = base["_stream"]
                def fixed_stream(command, *args):
                    command = list(command)
                    if Path(command[1]).name != "run_e2_final_continuation_delivery.py":
                        raise RuntimeError("Ruta inesperada de la continuación de 80 horas")
                    root = _materialize(fix_commit, base["LOCAL_WORKSPACE"])
                    command[1] = root / "scripts/run_e2_final80_resume_delivery.py"
                    command.extend(["--resume-root", root, "--resume-commit", fix_commit])
                    return stream(command, *args)
                base["_stream"] = fixed_stream
                base["CONTROLLERS"].add("run_e2_final80_resume_delivery.py")
                return base
            funding["_base_launcher"] = base_launcher
            return funding
        prior["_funding_launcher"] = funding_launcher
        return prior
    old["_prior_launcher"] = prior_launcher
    # This is the verified, persisted consent, not a new affirmative response.
    result = old["run_e2_final_continuation_colab"](FUNDING_CONTINUATION_COMMIT,
        namespace=globals() if namespace is None else namespace, execute=True,
        input_fn=lambda unused_prompt: approval["confirmed_text"])
    if isinstance(result, dict):
        result = {**result, "resume_commit": fix_commit,
            "consent_reused": True, "new_budget_authorization": False,
            "original_approval_sha256": approval["sha256"]}
    return result
