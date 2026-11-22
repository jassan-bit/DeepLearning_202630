"""Explicit fresh development rebuild after confirmed loss of Colab VM files.

No import side effects, market fitting, final evaluation or invented old hashes.
Existing session authorization is reused; private identifying fields complete
new records. AI assistance: implementation/review, never a human signature.
"""
from __future__ import annotations
import datetime as dt
import getpass
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

VERSION = "fresh-cpu-development-rebuild-1"
T2_COMMIT = "b3d73a750da2643ecfc7b2802904fd54c1efd4e8"
T2_SHA = "bf7b124289dc6258323b2d32a8c03f56adf890dcce445cbe2b250bf3e6566be5"
S2_COMMIT = "db74afd0763596f71bcbafb4b9150a44a2e83339"
S2_MANIFEST = "2c7dc052d2a53f212688340b50b65bed5fa367ff16fb5ed0a2474d417696715b"
CM_COMMIT = "6c247cf0914cf28532bcfa5ceb07267a35c28c48"
CM_MANIFEST = "21ce8729f7db6c9e683adf57dfcb94e97eeb27c081c973a0791af5a39e9ac0a2"
PRIOR_REPORT_SHA = "7e4dd2b4f2d90dc2c4b3eee528d09d99b45575d2529b1d6501d185263151e567"
PRIOR_ACTOR_PAIR_DIGEST = "c8dacfcd000d3db15b2a76adc9c4a6e240bd47ddcafc0fe1ae7ff6d57f1fd758"
PANELS = ("F00", "F01", "F02", "F03")
SCOPE = ("Academic noncommercial development, selection, planning and calibration only; "
         "publication of permitted derived outputs with source attribution and source license conditions. "
         "No raw redistribution and no final evaluation authorization.")


def _fresh_require(value, message):
    if not value:
        raise RuntimeError(message)


def _fresh_sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _fresh_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fresh_new_json(path, value):
    path = Path(path)
    _fresh_require(not any(p.is_symlink() for p in (path, *path.parents)), "Conservar rutas sin symlinks.")
    raw = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n"
    if path.exists():
        _fresh_require(path.read_bytes() == raw, "Metadatos inmutables diferentes; no sobrescribir.")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)


def initialize(ns, notebook_bytes, before_metadata_callback=None):
    """Execute only the first three code cells of the exact published T2."""
    _fresh_require(isinstance(notebook_bytes, bytes) and _fresh_sha(notebook_bytes) == T2_SHA, "Notebook T2 diferente.")
    book = json.loads(notebook_bytes)
    pins = book["metadata"]["e3_source_pins"]
    _fresh_require(pins["source_commit"] == S2_COMMIT and pins["manifest_sha256"] == S2_MANIFEST, "Pins T2 diferentes.")
    codes = [c["source"] for c in book["cells"] if c["cell_type"] == "code"]
    codes = ["".join(c) if isinstance(c, list) else c for c in codes]
    _fresh_require(len(codes) >= 3 and codes[0].startswith("from pathlib import Path") and
            codes[2].startswith("PREPARE_CRYPTO_DEVELOPMENT = False"), "Estructura T2 diferente.")
    already = (ns.get("SOURCE_COMMIT") == S2_COMMIT and ns.get("MANIFEST_SHA256") == S2_MANIFEST and
               isinstance(ns.get("SETTINGS"), dict) and ns["SETTINGS"].get("source_commit") == S2_COMMIT and
               all(k in ns for k in ("definitions", "run_cli", "remote_args", "bootstrap", "PRIVATE")) and
               Path(ns.get("SOURCE", "/nonexistent")).is_dir())
    if not already:
        for index in (0, 1):
            exec(compile(codes[index], "<pinned T2 setup cell %d>" % index, "exec"), ns)
    if before_metadata_callback is not None:
        before_metadata_callback(ns)
    if not already:
        exec(compile(codes[2], "<pinned T2 metadata cell>", "exec"), ns)
    _fresh_require(ns["SOURCE_COMMIT"] == S2_COMMIT and ns["MANIFEST_SHA256"] == S2_MANIFEST and
            tuple(r["panel"] for r in ns["definitions"]) == PANELS, "Conservar fuente y cuatro paneles T2.")
    return ns


def private_identity():
    def field(secret, prompt):
        try:
            from google.colab import userdata
            value = userdata.get(secret)
        except Exception:
            value = None
        return (value or getpass.getpass(prompt)).strip()
    return dict(actor_name=field("E3_AUTHOR_NAME", "MISMO nombre del autor anterior (privado; nuevos metadatos): "),
                actor_contact=field("E3_AUTHOR_CONTACT", "MISMO contacto anterior (privado): "), actor_role="Autor")


def create_declarations(ns, evidence_hash=PRIOR_REPORT_SHA, identity_provider=None):
    _fresh_require(evidence_hash == PRIOR_REPORT_SHA, "Conservar la evidencia de autorización previa fijada.")
    rows = ns["definitions"]
    _fresh_require(tuple(r["panel"] for r in rows) == PANELS, "Conservar las cuatro propuestas.")
    owners = []
    for row in rows:
        path = Path(row["folder"]) / "owner-development.json"
        owner = _fresh_read(path)
        if owner.get("actor_name"):
            _fresh_require(owner.get("kind") == "prior_authorization_reused_during_user_requested_rebuild" and
                    owner.get("prior_authorization_report_sha256") == evidence_hash and
                    owner.get("AI_completed_signature") is False and owner.get("authorize_test_preparation") is False,
                    "Declaración existente de otra reconstrucción; conservarla para revisión.")
        owners.append(owner)
    existing = [o for o in owners if o.get("actor_name")]
    identity = ({k:existing[0][k] for k in ("actor_name", "actor_contact", "actor_role")} if existing else
                (identity_provider or private_identity)())
    _fresh_require(identity.get("actor_name") and identity.get("actor_contact"), "Nombre y contacto reales necesarios; no se publican.")
    normalized_pair = {"actor_name":" ".join(identity["actor_name"].split()).casefold(),
                       "actor_contact":identity["actor_contact"].strip().casefold()}
    pair_digest = _fresh_sha(json.dumps(normalized_pair, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
    _fresh_require(pair_digest == PRIOR_ACTOR_PAIR_DIGEST,
            "Nombre/contacto no coinciden con los campos declarados anteriormente; conservar metadatos sin atribuir autorización.")
    _fresh_require(all(o["actor_name"] == identity["actor_name"] and o["actor_contact"] == identity["actor_contact"]
                for o in existing), "Conservar la identidad de las declaraciones existentes.")
    recorded = dt.datetime.now(dt.timezone.utc).isoformat()
    for row, owner in zip(rows, owners):
        folder, p = Path(row["folder"]), row["proposal"]
        if not owner.get("actor_name"):
            _fresh_new_json(folder / "owner-development.blank-template.json", owner)
            owner = dict(owner, kind="prior_authorization_reused_during_user_requested_rebuild",
                actor_name=identity["actor_name"], actor_contact=identity["actor_contact"], actor_role="Autor",
                statement_utc=recorded, statement_utc_semantics="Current reconstruction record timestamp; not the unknown original human statement time",
                reconstruction_recorded_utc=recorded, metadata_reconstruction_utc=recorded, original_statement_utc=None,
                statement_time_semantics="current_user_requested_rebuild_record_not_original_consent_time",
                prior_report_bytes_available_in_runtime=False,
                prior_authorization_report_sha256=evidence_hash, prior_authorization_literal="ACEPTO DESARROLLO",
                prior_authorization_reused=True, newly_obtained_authorization=False,
                evidence_origin="Prior real user authorization, traceable by private report SHA256; current user-requested fresh reconstruction. No new interactive acceptance or signature is claimed.",
                human_identity_cryptographically_verified=False,
                prior_self_declared_actor_fields_match=True, actor_field_match_is_not_identity_verification=True,
                acknowledge_prior_D00_D03_exposed=True, all_data_declared_exposed_development=True,
                observation_cutoff_utc=(dt.datetime.combine(dt.date.fromisoformat(p["proposed_development_end"]) + dt.timedelta(days=1),
                    dt.time(), dt.timezone.utc) - dt.timedelta(milliseconds=1)).isoformat(),
                observation_cutoff_semantics="authorized_development_data_boundary_not_proof_of_historical_access",
                source_terms_accepted_by_real_actor=True,
                allowed_processing_scope=SCOPE, allow_derived_publication=True, allow_raw_redistribution=False,
                AI_completed_signature=False, authorize_test_preparation=False,
                reconstruction_scope="New exposed-development metadata after VM loss; no old admission/hash restoration")
            target = folder / "owner-development.json"
            _fresh_require(_fresh_read(target).get("actor_name") is None and not (folder / "exposure-development.json").exists(),
                    "No reemplazar declaraciones o recibos existentes.")
            # Only the exact untouched source template is replaced; its bytes
            # are retained above. New declaration records have their own time.
            target.write_text(json.dumps(owner, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        rights = dict(status="approved", actor_name=owner["actor_name"], actor_contact=owner["actor_contact"],
            evidence_path=owner["source_license_evidence_path"], evidence_sha256=owner["source_license_evidence_sha256"],
            scope=SCOPE, source_url=p["rights_source_url"], license=p["license"],
            allow_derived_publication=True, allow_raw_redistribution=False, statement_utc=owner["statement_utc"],
            statement_utc_semantics=owner["statement_utc_semantics"], prior_authorization_report_sha256=evidence_hash,
            decision_origin="Existing real user authorization reused during requested reconstruction; no new acceptance/signature claimed")
        _fresh_new_json(folder / "rights.json", rights)
    return dict(status="new_development_records_from_prior_authorization", prior_authorization_report_sha256=evidence_hash,
                old_hashes_restored=False, new_market_fits=0, test_opened=False)


def run_worker(ns, operation, panel=None):
    kit = Path(ns["FRESH_KIT_DIR"])
    path = kit / "ops/fresh_worker.py"
    _fresh_require(_fresh_sha(path.read_bytes()) == ns["FRESH_WORKER_SHA256"], "Worker diferente del publicado.")
    env = ns["bootstrap"].cpu_env(); env["PYTHONDONTWRITEBYTECODE"] = "1"
    payload = dict(operation=operation, panel=panel, source=str(ns["SOURCE"]), private=str(ns["PRIVATE"]),
                   bundle=str(ns["repair_dir"]), worker_sha256=ns["FRESH_WORKER_SHA256"])
    command = [ns["SETTINGS"]["python"], "-B", "-u", str(path)]
    process = subprocess.Popen(command, cwd=ns["SOURCE"], env=env, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    process.stdin.write(json.dumps(payload)); process.stdin.close()
    result = None
    for line in process.stdout:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        print(json.dumps(event, ensure_ascii=False), flush=True)
        if event.get("status") == "fresh_worker_complete":
            result = event
    _fresh_require(process.wait() == 0 and result is not None,
            "Reconstrucción detenida; conservar archivos y revisar el diagnóstico mostrado. No se ha entrenado.")
    return result


def run_development(ns, backup_callback, worker_callback=run_worker):
    worker_callback(ns, "record")
    states = worker_callback(ns, "check")["panels"]
    checkpoint = backup_callback(ns, "before_acquisition")
    _fresh_require(isinstance(checkpoint, dict) and checkpoint.get("remote_verified") is True,
            "Verificar respaldo de declaraciones antes de adquirir; no continuar sin respaldo auténtico.")
    actions = {p:"existing_admission_verified" for p in PANELS if states[p]["admitted"]}
    for panel in PANELS:
        if not states[panel]["admitted"]:
            exported = Path(ns["PRIVATE"]) / "recovery-exports" / (panel + ".json")
            operation = "reacquire" if exported.exists() else "acquire_prepare"
            worker_callback(ns, operation, panel)
            actions[panel] = operation
            checkpoint = backup_callback(ns, "after_admission_" + panel)
            _fresh_require(isinstance(checkpoint, dict) and checkpoint.get("remote_verified") is True,
                    "Verificar respaldo de la admisión antes del panel siguiente.")
    final = worker_callback(ns, "check")
    _fresh_require(all(s["admitted"] for s in final["panels"].values()), "Completar cuatro admisiones reales antes del caso CPU.")
    final["recovery_actions"] = actions
    return final


def fresh_rebuild(ns, notebook_bytes, backup_callback, before_metadata_callback=None):
    initialize(ns, notebook_bytes, before_metadata_callback=before_metadata_callback)
    create_declarations(ns)
    ns["CM_REPAIR_COMMIT"] = CM_COMMIT
    result = run_development(ns, backup_callback)
    # Produce the existing standardized four-admission operational receipt so
    # the separately pinned CPU starter can bind these NEW admission hashes.
    manifest = Path(ns["repair_dir"]) / "ops/REPAIR_MANIFEST.json"
    _fresh_require(_fresh_sha(manifest.read_bytes()) == CM_MANIFEST, "Manifest CM diferente.")
    spec = importlib.util.spec_from_file_location("e3_fresh_cm_verifier", Path(ns["repair_dir"]) / "ops/resume_development.py")
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    ns["repair_result"] = helper.resume(ns, ns["repair_dir"], CM_MANIFEST, verify_only=True)
    checkpoint = backup_callback(ns, "all_four_verified_with_cm_overlay")
    _fresh_require(isinstance(checkpoint, dict) and checkpoint.get("remote_verified") is True,
            "No declarar reconstrucción respaldada sin verificar el recibo remoto final.")
    print("Cuatro paneles de desarrollo verificados y respaldados. No se entrenó. Ejecuta el inicio CPU publicado para preparar/sincronizar el caso antes de PLANIFICAR 340.")
    return dict(status="FRESH_DEVELOPMENT_REBUILT_AND_BACKED_UP_NO_FITS", first_lost_vm_hashes_restored=False,
                new_market_fits=0, test_opened=False, panels=result["panels"], recovery_actions=result["recovery_actions"])
