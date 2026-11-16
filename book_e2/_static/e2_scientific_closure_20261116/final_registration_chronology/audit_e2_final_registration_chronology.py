"""Verify original final-fit chronology from Git-bound requests and phase journals.

Metadata only. No market arrays, weights, model fitting or hypothesis tests.
Git mode may retrieve missing original objects; portable mode uses included files.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_e2_registration_chronology import validate_journal, canonical, require, write_csv, write_json

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = "50cb634e9eb163b09f9607982842144ba9f74c87"
BASE = ROOT/"book_e2/_static/e2_scientific_closure_20261116"
DEFAULT = BASE/"final_registration_chronology"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(path):
    return subprocess.check_output(["git", "show", ARCHIVE+":"+path], cwd=ROOT)


def manifest(bundle):
    files = {p.relative_to(bundle).as_posix(): {"sha256": sha(p.read_bytes()), "bytes": p.stat().st_size}
             for p in sorted(bundle.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    (bundle/"manifest.json").write_text(json.dumps({"files": files}, indent=2)+"\n")


def analyse(bundle, output):
    require(not output.exists(), "Preserve earlier audit output")
    bound = json.loads((bundle/"manifest.json").read_text())["files"]
    required = {"phase_resources.csv", "original_final_freeze.json", "original_data_admission.json", "registration_records.json"}
    require(required.issubset(bound), "Required chronology inputs not bound")
    for name, item in bound.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe bundle input")
        path = bundle/relative
        require(path.is_file() and not path.is_symlink(), "Missing regular bound input")
        raw = path.read_bytes()
        require(sha(raw) == item["sha256"] and len(raw) == item["bytes"], "Input changed: "+name)
    registrations = json.loads((bundle/"registration_records.json").read_text())
    prior = {r["id"]: r for r in registrations}
    for record in registrations:
        raw = (bundle/"registrations"/(record["id"]+Path(record["path"]).suffix)).read_bytes()
        require(sha(raw) == record["blob_sha256"], "Original registration source differs")
    resources = list(csv.DictReader((bundle/"phase_resources.csv").open()))
    resources = [r for r in resources if r["stage"] == "final"]
    freeze = json.loads((bundle/"original_final_freeze.json").read_text())
    frozen = {(r["model"], str(r["seed"])): r for r in freeze["models"]}
    admission = json.loads((bundle/"original_data_admission.json").read_text())
    admission_end = datetime.fromisoformat(admission["completed_at_utc"]).timestamp()
    jobs, phases = {}, []
    for resource in resources:
        attempt = resource["source_attempt_id"]
        directory = bundle/"fits"/attempt
        request_raw, fit_raw = (directory/"request.json").read_bytes(), (directory/"job_result.json").read_bytes()
        request, fit = json.loads(request_raw), json.loads(fit_raw)
        require(fit["status"] == "completed" and fit["training_performed"]
                and not fit["synthetic"] and not fit["test_only_cpu_execution"], "Not an original real completed fit")
        require(request["name"] == fit["model_id"] == resource["model"]
                and request["seed"] == fit["seed"] == int(resource["seed"]), "Fit/seed identity differs")
        require(sha(canonical(request)) == resource["request_sha256"] == fit["request_sha256"], "Request binding differs")
        require(request["sources"]["source_commit"] == prior["reduced_engine"]["commit"], "Wrong original source commit")
        checkpoint = fit["selected_checkpoint"]["sha256"]
        require(checkpoint == resource["checkpoint_sha256"] == frozen[resource["model"], resource["seed"]]["checkpoint_sha256"],
                "Final-fit checkpoint does not bind to the scored frozen model")
        phase = resource["phase"]
        journal_raw = gzip.decompress((bundle/"journals"/attempt/(phase+".jsonl.gz")).read_bytes())
        require(sha(journal_raw) == fit["artifacts_sha256"]["phases/"+phase+"/events.jsonl"], "Journal not bound to original fit")
        checked = validate_journal(journal_raw)
        require(abs(float(fit["phase_records"][phase]["elapsed_seconds"])-float(resource["elapsed_seconds"])) < 1e-9,
                "Original recorded phase cost differs")
        first, last = checked["earliest_epoch_seconds"], datetime.fromisoformat(checked["last_utc"]).timestamp()
        row = {"model": resource["model"], "seed": resource["seed"], "attempt_id": attempt, "phase": phase,
               "archive_commit": ARCHIVE, "journal_sha256": sha(journal_raw), "journal_bytes": len(journal_raw),
               "request_sha256": resource["request_sha256"], "request_file_sha256": sha(request_raw),
               "fit_result_sha256": sha(fit_raw), "scored_checkpoint_sha256": checkpoint, **checked,
               "protocol_precedes_phase": prior["reduced_protocol_first"]["committer_epoch_seconds"] < first,
               "engine_precedes_phase": prior["reduced_engine"]["committer_epoch_seconds"] < first,
               "phase_before_reported_data_admission_completion": last <= admission_end}
        phases.append(row)
        key = resource["model"], resource["seed"]
        jobs.setdefault(key, {"model": key[0], "seed": key[1], "attempt_id": attempt, "first_phase_utc": checked["first_utc"],
                              "last_phase_utc": checked["last_utc"], "phase_count": 0})
        jobs[key]["first_phase_utc"] = min(jobs[key]["first_phase_utc"], checked["first_utc"])
        jobs[key]["last_phase_utc"] = max(jobs[key]["last_phase_utc"], checked["last_utc"])
        jobs[key]["phase_count"] += 1
    require(len(jobs) == 78 and len(phases) == 90, "Wrong original final-fit/phase inventory")
    output.mkdir(parents=True)
    write_csv(output/"final_fit_chronology_78.csv", sorted(jobs.values(), key=lambda r: (r["model"], r["seed"])))
    write_csv(output/"final_phase_chronology_90.csv", phases)
    summary = {
        "status": "verified_original_final_metadata_only", "archive_commit": ARCHIVE,
        "final_neural_fits": 78, "phase_journals": 90, "scored_checkpoint_bindings_verified": 78,
        "original_real_fits_rerun": 0, "new_model_inference": False, "new_hypothesis_tests": 0,
        "all_phases_protocol_precedes_execution": all(r["protocol_precedes_phase"] for r in phases),
        "all_phases_engine_precedes_execution": all(r["engine_precedes_phase"] for r in phases),
        "all_phases_before_reported_data_admission_completion": all(r["phase_before_reported_data_admission_completion"] for r in phases),
        "first_final_phase_utc": min(r["first_utc"] for r in phases),
        "last_final_phase_utc": max(r["last_utc"] for r in phases),
        "reported_admission_completed_at_utc": admission["completed_at_utc"],
        "freeze_analysis_frozen_flag": freeze["analysis_frozen"],
        "freeze_weights_before_current_data_flag": freeze["weights_scalers_and_calibrators_frozen_before_final_data"],
        "full_historical_preregistration_or_blinding_certified": False,
        "scope": "Original current78final fits; cached earlier HPO and prior E1exposure retain separate historical scope",
        "timing_limits": [
            "Git dates and worker UTC are recorded timestamps, not an independent trusted timestamp authority.",
            "The end of data admission is an observed upper endpoint, not a verified time of first access.",
            "The freeze flags document controller order; the freeze JSON has no independent trusted wall-clock timestamp.",
            "No claim of original selection, every old cached fit, or inherited known test being prospectively blind.",
        ]}
    write_json(output/"summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def build(bundle):
    require(not bundle.exists(), "Use a new evidence directory")
    bundle.mkdir(parents=True)
    prior = BASE/"preregistration"
    records = json.loads((prior/"summary.json").read_text())["registrations"]
    write_json(bundle/"registration_records.json", records)
    for record in records:
        name = record["id"]+Path(record["path"]).suffix
        target = bundle/"registrations"/name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(prior/name, target)
    copies = {
        "phase_resources.csv": ROOT/"book_e2/_static/e2_efficiency_closure_20261113/phase_resources.csv",
        "original_final_freeze.json": ROOT/"book_e2/_static/e2_benchmark_final_20261113/final_freeze.json",
        "original_data_admission.json": ROOT/"book_e2/_static/e2_benchmark_final_20261113/data_admission.json",
    }
    for name, path in copies.items():
        shutil.copyfile(path, bundle/name)
    phases = [r for r in csv.DictReader((bundle/"phase_resources.csv").open()) if r["stage"] == "final"]
    require(len(phases) == 90, "Wrong phase inventory")
    fetched = set()
    for row in phases:
        attempt, phase = row["source_attempt_id"], row["phase"]
        source = "runs/"+attempt+"/artifacts/fit/"
        if attempt not in fetched:
            for name in ("request.json", "job_result.json"):
                target = bundle/"fits"/attempt/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(git(source+name))
            fetched.add(attempt)
        target = bundle/"journals"/attempt/(phase+".jsonl.gz")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(gzip.compress(git(source+"phases/"+phase+"/events.jsonl"), mtime=0))
    shutil.copyfile(__file__, bundle/"audit_e2_final_registration_chronology.py")
    shutil.copyfile(ROOT/"scripts/audit_e2_registration_chronology.py", bundle/"audit_e2_registration_chronology.py")
    (bundle/"run.py").write_text(
        "from pathlib import Path\nimport runpy,sys\nr=Path(__file__).resolve().parent\n"
        "sys.argv=[str(r/'audit_e2_final_registration_chronology.py'),'--verify-bundle',str(r),'--output',str(r/'output')]\n"
        "runpy.run_path(sys.argv[0],run_name='__main__')\n")
    manifest(bundle)
    analyse(bundle, bundle/"audit_results")
    (bundle/"README.md").write_text(
        "# Anterioridad documentada de los78ajustes finales originales\n\n"
        "Se ligan90journals originales,78solicitudes/resultados y los checkpoints puntuados. "
        "La comparación usa UTC de eventos y fechas Git, sin fabricar sellos externos. "
        "La congelación y el fin de admisión conservan sus propios límites, explicados en summary.json. "
        "No se certifica cegamiento ni toda la selección histórica.\n\n"
        "Para verificar sin Git, red o paquetes: python run.py. "
        "Se genera output/ con tablas y resumen; preserve esa carpeta antes de repetir. "
        "El modo mantenedor requiere el repositorio y puede recuperar objetos originales de origin. "
        "No ejecuta modelos ni pruebas estadísticas.\n", encoding="utf-8")
    manifest(bundle)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--verify-bundle", type=Path)
    args = parser.parse_args(argv)
    if args.verify_bundle:
        analyse(args.verify_bundle, args.output)
    else:
        build(args.output)


if __name__ == "__main__":
    main()
