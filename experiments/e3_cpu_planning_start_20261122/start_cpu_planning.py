"""Start existing S2 CPU planning after verified COIN-M admission repair.

Load the SHA256-verified bytes with exec(..., globals()), then explicitly call
start_cpu_planning(globals()). No __file__ assumption, training on import, new
source identity, final activation, data acquisition, or author declarations.
AI assistance: implementation and software review. Repository code terms apply.
"""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
import urllib.parse

VERSION = "s2-existing-cpu-planning-start-1"
S2_COMMIT = "db74afd0763596f71bcbafb4b9150a44a2e83339"
S2_MANIFEST = "2c7dc052d2a53f212688340b50b65bed5fa367ff16fb5ed0a2474d417696715b"
CM_COMMIT = "6c247cf0914cf28532bcfa5ceb07267a35c28c48"
CM_MANIFEST = "21ce8729f7db6c9e683adf57dfcb94e97eeb27c081c973a0791af5a39e9ac0a2"
REPAIR_VERSION = "coinm-header-operational-resume-v1"
PANELS = ("F00", "F01", "F02", "F03")


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _last_json(result):
    _require(getattr(result, "returncode", None) == 0, "El comando CPU no terminó correctamente; conservar evidencia.")
    for line in reversed(getattr(result, "stdout", "").splitlines()):
        try:
            value = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("El comando no devolvió un resultado JSON verificable.")


def _regular(path, root):
    path, root = Path(path), Path(root).resolve()
    _require(not any(p.is_symlink() for p in (path, *path.parents)) and
             path.resolve().is_relative_to(root) and path.is_file(), "Conservar rutas originales sin symlinks.")
    return path


_CPU_CHECK = r'''
import json, sys, torch
from pathlib import Path
from engine import data
payload = json.load(sys.stdin)
assert torch.version.cuda is None, 'Conservar el interprete PyTorch CPU original'
result = {}
for panel, directory in payload.items():
    path = Path(directory)
    admission = data.read_admission(path)
    assert admission['status'] == 'development_admitted'
    assert set(admission['partition_paths']) == {'fit','selection','planning','calibration'}
    for part in ('fit','selection','planning','calibration'):
        data.load_partition(path, part)
    result[panel] = {'admission_sha256':admission['admission_sha256'],
        'admission_file_sha256':data.sha256_bytes((path/'admission.json').read_bytes()),
        'recovery_metadata_sha256':data.sha256_bytes((path/'recovery_metadata.json').read_bytes())}
print(json.dumps({'cpu_only':True,'panels':result,'new_market_fits':0,'test_opened':False}))
'''


def _validate(ns):
    needed = ("SOURCE", "SOURCE_COMMIT", "MANIFEST_SHA256", "SETTINGS", "PRIVATE", "PLANNING_CASE",
              "DEVELOPMENT_MAP", "repair_dir", "CM_REPAIR_COMMIT", "bootstrap", "run_cli", "remote_args",
              "REPOSITORY", "PLANNING_BRANCH", "PLANNING_PREFIX",
              "CPU_PLANNING_START_COMMIT", "CPU_PLANNING_START_SHA256", "CPU_PLANNING_START_BYTES")
    _require(all(k in ns for k in needed), "Conserva la misma sesión Colab S2 y ejecuta el loader publicado.")
    _require(ns["SOURCE_COMMIT"] == S2_COMMIT and ns["MANIFEST_SHA256"] == S2_MANIFEST,
             "Los pins originales S2 deben conservarse.")
    settings = ns["SETTINGS"]
    _require(isinstance(settings, dict) and settings.get("source_commit") == S2_COMMIT,
             "Conservar SETTINGS del entorno CPU original.")
    source, private, case, bundle = (Path(ns[k]).resolve() for k in ("SOURCE", "PRIVATE", "PLANNING_CASE", "repair_dir"))
    _require(Path(settings.get("source_root", "")).resolve() == source and Path(settings.get("python", "")).is_file(),
             "Conservar la fuente y el intérprete CPU S2.")
    _require(case.parent == private.parent / "cases" and case.name.startswith("planning-") and
             not source.is_relative_to(private) and not private.is_relative_to(source),
             "Conservar el caso de planificación y sus rutas originales.")
    _require(ns["CM_REPAIR_COMMIT"] == CM_COMMIT and not bundle.is_relative_to(source) and
             not source.is_relative_to(bundle), "Corrector COIN-M o ruta diferente.")
    _require(re.fullmatch(r"[0-9a-f]{40}", ns["CPU_PLANNING_START_COMMIT"]) and
             re.fullmatch(r"[0-9a-f]{64}", ns["CPU_PLANNING_START_SHA256"]),
             "El loader debe fijar commit y SHA256 propios antes de ejecutar.")
    _require(isinstance(ns["CPU_PLANNING_START_BYTES"], bytes) and
             _sha(ns["CPU_PLANNING_START_BYTES"]) == ns["CPU_PLANNING_START_SHA256"],
             "El loader debe conservar los bytes verificados del helper para respaldarlos.")
    manifest_path = _regular(source / "SOURCE_MANIFEST.json", source)
    _require(_sha(manifest_path.read_bytes()) == S2_MANIFEST, "Manifest S2 alterado.")
    anchor = _read(_regular(source / "PUBLIC_SOURCE_ANCHOR.json", source))
    _require(anchor.get("source_commit") == S2_COMMIT and anchor.get("manifest_sha256") == S2_MANIFEST,
             "Ancla S2 diferente.")
    files = _read(manifest_path)["files"]
    _require(len(files) == 304, "Inventario S2 diferente.")
    for name, digest in files.items():
        _require(_sha(_regular(source / name, source).read_bytes()) == digest, "Archivo S2 alterado: " + name)
    cm_manifest_path = _regular(bundle / "ops/REPAIR_MANIFEST.json", bundle)
    _require(_sha(cm_manifest_path.read_bytes()) == CM_MANIFEST, "Manifest del corrector diferente.")
    cm_manifest = _read(cm_manifest_path)
    _require(cm_manifest["parent_source_commit"] == S2_COMMIT and cm_manifest["parent_manifest_sha256"] == S2_MANIFEST,
             "Padre del corrector diferente.")
    for name, digest in cm_manifest["files"].items():
        _require(_sha(_regular(bundle / name, bundle).read_bytes()) == digest, "Archivo del corrector diferente: " + name)
    development_map = _regular(ns["DEVELOPMENT_MAP"], private)
    _require(development_map.resolve() == private / "development-panels.json", "Conservar mapa original.")
    panels = _read(development_map)
    _require(set(panels) == set(PANELS) and all(Path(panels[p]).resolve() ==
             private / "panels" / p / "development-v1" for p in PANELS), "Conservar las cuatro admisiones originales.")
    env = ns["bootstrap"].cpu_env(); env["PYTHONDONTWRITEBYTECODE"] = "1"
    check = subprocess.run([settings["python"], "-B", "-c", _CPU_CHECK], cwd=source,
        env=env, text=True, input=json.dumps(panels), capture_output=True)
    integrity = _last_json(check)
    _require(integrity.get("cpu_only") is True and set(integrity.get("panels", {})) == set(PANELS),
             "Validación CPU y admisiones incompletas.")
    admissions = integrity["panels"]
    result = ns.get("repair_result")
    receipt_dir = private / "operational-repairs" / REPAIR_VERSION
    receipts = sorted(receipt_dir.glob("*.json"))
    _require(receipts, "Conservar los recibos operacionales auténticos del corrector.")
    completed = []
    for path in receipts:
        receipt = _read(_regular(path, private))
        claim = receipt.pop("event_sha256", None)
        encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
        _require(claim == _sha(encoded), "Recibo operacional alterado; conservar para revisión.")
        _require(receipt.get("repair_manifest_sha256") == CM_MANIFEST and receipt.get("parent_source_commit") == S2_COMMIT,
                 "Recibo operacional de otra versión.")
        if receipt.get("kind") == "completed" and receipt.get("preserved_existing_files_verified") is True:
            _require(receipt.get("admissions") == admissions, "Las admisiones cambiaron después del corrector.")
            completed.append(path)
    _require(completed, "Falta un recibo completo del corrector para los cuatro paneles.")
    if result is not None:
        _require(isinstance(result, dict) and result.get("status") ==
            "four_development_panels_verified_with_header_overlay" and result.get("panels") == admissions,
            "repair_result no corresponde a las cuatro admisiones actuales.")
        _require(Path(result.get("receipt", "")).resolve() in [p.resolve() for p in completed],
                 "repair_result no enlaza un recibo completo conservado.")
    repository, branch, prefix = (ns[k] for k in ("REPOSITORY", "PLANNING_BRANCH", "PLANNING_PREFIX"))
    _require(repository == "jassan-bit/DeepLearning_202630" and isinstance(branch, str) and
             branch.startswith("e3-r3-crypto-engine-v2-planning-") and isinstance(prefix, str) and
             prefix.startswith("runs/e3_r3_crypto_engine_v2/planning-"), "Conservar la identidad remota S2.")
    return dict(source=source, private=private, case=case, bundle=bundle, admissions=admissions,
        receipts=receipts, cm_manifest=cm_manifest, repository=repository, branch=branch, prefix=prefix)


def _prepare(ns, context):
    case, source = context["case"], context["source"]
    if not (case / "plan.json").is_file():
        _last_json(ns["run_cli"](["prepare", *ns["remote_args"](case, panels=ns["DEVELOPMENT_MAP"]),
            "--stage", "planning", "--protocol", str(source / "configs/protocol.json")]))
    plan = _read(_regular(case / "plan.json", case))
    _require(plan.get("stage") == "planning" and plan.get("source_commit") == S2_COMMIT and
        plan.get("source_manifest_sha256") == S2_MANIFEST and plan.get("original_case_root") == str(case) and
        plan.get("software_fixture") is False and plan.get("test_opened") is False and
        len(plan.get("jobs", [])) == 340 and len(plan.get("seeds", [])) == 5,
        "Conservar el diseño S2 de 340 trabajos y cinco semillas; no iniciar otra campaña.")
    _require({p["id"]:p["data_sha256"] for p in plan["data_panels"]} ==
             {p:context["admissions"][p]["admission_sha256"] for p in PANELS},
             "El plan no corresponde a las admisiones verificadas.")
    context["plan"] = plan
    context["plan_sha256"] = _sha(_canonical(plan))
    return plan


def _capsule(ns, context):
    """Allowlisted JSON preserves exact bytes, including Python overlay files."""
    bundle, private, case = context["bundle"], context["private"], context["case"]
    raw_helper = ns["CPU_PLANNING_START_BYTES"]
    _require(isinstance(raw_helper, bytes) and _sha(raw_helper) == ns["CPU_PLANNING_START_SHA256"],
             "Bytes del helper alterados antes del respaldo.")
    payloads = {"planning_start/start_cpu_planning.py": dict(bytes=len(raw_helper),
        sha256=_sha(raw_helper), base64=base64.b64encode(raw_helper).decode("ascii"))}
    names = ["ops/REPAIR_MANIFEST.json", *context["cm_manifest"]["files"]]
    for name in names:
        raw = _regular(bundle / name, bundle).read_bytes()
        payloads["addon/" + name] = dict(bytes=len(raw), sha256=_sha(raw), base64=base64.b64encode(raw).decode("ascii"))
    for path in context["receipts"]:
        raw = _regular(path, private).read_bytes()
        name = "operational_receipts/" + path.name
        payloads[name] = dict(bytes=len(raw), sha256=_sha(raw), base64=base64.b64encode(raw).decode("ascii"))
    value = dict(kind="s2_cpu_planning_operational_addon_capsule", version=VERSION,
        case_id=context["plan"]["case_id"], plan_sha256=context["plan_sha256"], source_commit=S2_COMMIT,
        source_manifest_sha256=S2_MANIFEST, repair_commit=CM_COMMIT, repair_manifest_sha256=CM_MANIFEST,
        helper_commit=ns["CPU_PLANNING_START_COMMIT"], helper_sha256=ns["CPU_PLANNING_START_SHA256"],
        helper_integrity_origin="Published loader verified script SHA256 before exec in existing Colab globals",
        execution_identity="Immutable S2 CPU planning, with separately archived CM header overlay required for raw replay",
        restore_behavior="Restore preserves this capsule JSON; verify and extract addon before COIN-M raw replay. No automatic execution.",
        original_bundle_directory=str(bundle), original_receipts_directory=str(private / "operational-repairs" / REPAIR_VERSION),
        files=payloads, new_fits_by_packaging=0, test_opened=False,
        AI_assistance="Implementation and software review; existing real author declarations preserved, no new signatures")
    raw = _canonical(value) + b"\n"
    _require(len(raw) < 7 * 1024 * 1024, "Cápsula excede el límite de metadatos; conservar sin iniciar entrenamiento.")
    target = case / "receipts/operational_addons" / ("cm-header-" + _sha(raw) + ".json")
    _require(not any(p.is_symlink() for p in (target, *target.parents)), "Ruta de cápsula alterada.")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        _require(target.read_bytes() == raw, "Cápsula inmutable diferente.")
    else:
        with target.open("xb") as stream:
            stream.write(raw)
    context["capsule"] = target
    return target


def _sync(ns, context):
    result = _last_json(ns["run_cli"](["sync", *ns["remote_args"](context["case"],
        panels=ns["DEVELOPMENT_MAP"])], remote=True))
    backup = result.get("backup", {})
    _require(result.get("status") == "SAVED_CASE_REMOTE_VERIFIED" and backup.get("remote_verified") is True and
        backup.get("source_commit") == S2_COMMIT and backup.get("case_id") == context["plan"]["case_id"] and
        backup.get("plan_sha256") == context["plan_sha256"] and re.fullmatch(r"[0-9a-f]{40}", backup.get("commit", "")),
        "El respaldo remoto no quedó autenticado para este caso; no iniciar ajustes.")
    return result


def start_cpu_planning(ns):
    context = _validate(ns)
    _prepare(ns, context)
    capsule = _capsule(ns, context)
    synced = _sync(ns, context)
    url = ("https://github.com/" + context["repository"] + "/tree/" +
           urllib.parse.quote(context["branch"], safe="") + "/" + context["prefix"])
    print(json.dumps(dict(status="cpu_planning_prepared_and_backed_up_no_fits", case_id=context["plan"]["case_id"],
        plan_sha256=context["plan_sha256"], capsule_sha256=_sha(capsule.read_bytes()),
        remote_commit=synced["backup"]["commit"], results_url=url, new_market_fits=0, test_opened=False)), flush=True)
    acceptance = input("Planificación CPU: 340 trabajos, 320 ajustes neuronales ×120s nominales =10h40, más ARIMA, E/S y guardado. Escribe PLANIFICAR 340 para activar: ").strip()
    if acceptance != "PLANIFICAR 340":
        print("Planificación conservada y respaldada; entrenamiento no activado.")
        return dict(status="CPU_PLANNING_SAVED_NOT_ACTIVATED", results_url=url, new_market_fits=0, test_opened=False)
    result = _last_json(ns["run_cli"](["run", *ns["remote_args"](context["case"],
        panels=ns["DEVELOPMENT_MAP"]), "--activate"], remote=True))
    return dict(status="CPU_PLANNING_COMMAND_COMPLETED", engine_result=result, results_url=url,
        source_commit=S2_COMMIT, export_or_dimension_activated=False, final_test_activated=False)
