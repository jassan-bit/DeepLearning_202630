"""Audit existing Git registrations against actual HPO event journals.

Metadata only: no retrieval, fitting, inference, test, or resampling is run.
Historical registrations, missing reports, and non-received exports are read-only.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
HPO = ROOT / "book_e2/_static/e2_corrections_20261115/hpo"
OUT = ROOT / "book_e2/_static/e2_scientific_closure_20261116/preregistration"
ARCHIVE = "6a8000022d2f374731faacc8a5c66f9b9d46b465"
REGISTRATIONS = (
    ("early_margin", "599a29f87f8ebb1d67cd925ec09bf6bdbc732fa0", "experiments/e2_remediation_20261106/preregistration.json"),
    ("historical_full_protocol", "ef2be289903517e9d6f3b70bd72981990d58479a", "experiments/e2_remediation_20261106/preregistration.json"),
    ("reduced_protocol_first", "c056c1cfb8ac771b32b13bf1382b9afc5e813912", "experiments/e2_reduced_protocol_20261112/protocol.json"),
    ("reduced_engine", "1e8745497b484518f766b6da5c19be8362d9add7", "scripts/e2_reduced_final_analysis.py"),
)


def require(test, message):
    if not test:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def utc(stamp):
    return datetime.fromtimestamp(stamp, timezone.utc).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def write_csv(path, rows):
    require(bool(rows), "Do not emit an invented empty evidence table")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_journal(raw):
    rows = [json.loads(line) for line in raw.splitlines()]
    require(bool(rows), "Missing journal events")
    previous = "0"*64
    identity = rows[0]["identity_sha256"]
    stamps = []
    for i, original in enumerate(rows):
        row = dict(original)
        checksum = row.pop("sha256")
        require(checksum == sha(canonical(row)), "Event content checksum differs")
        require(row["sequence"] == i and row["previous_sha256"] == previous,
                "Event sequence/hash chain differs")
        require(row["identity_sha256"] == identity, "Mixed phase identities")
        stamp = float(row["utc_unix_seconds"])
        require(0 < stamp < float("inf"), "Invalid event timestamp")
        stamps.append(stamp)
        previous = checksum
    return {"event_rows": len(rows), "first_utc": utc(min(stamps)),
            "last_utc": utc(max(stamps)), "earliest_epoch_seconds": min(stamps),
            "phase_identity_sha256": identity, "last_event_sha256": previous}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "Use a new output; preserve previous audits")
    args.output.mkdir(parents=True)
    inputs, registrations = {}, []
    for name, commit, path in REGISTRATIONS:
        raw = git("show", commit+":"+path)
        times = git("show", "-s", "--format=%aI%n%cI", commit).decode().splitlines()
        commit_timestamp = datetime.fromisoformat(times[1]).timestamp()
        registrations.append({"id": name, "commit": commit, "path": path,
                              "blob_sha256": sha(raw), "bytes": len(raw),
                              "author_time": times[0], "committer_time": times[1],
                              "committer_epoch_seconds": commit_timestamp})
        (args.output/(name+Path(path).suffix)).write_bytes(raw)
    first = next(r for r in registrations if r["id"] == "reduced_protocol_first")
    engine = next(r for r in registrations if r["id"] == "reduced_engine")
    old = json.loads(git("show", REGISTRATIONS[0][1]+":"+REGISTRATIONS[0][2]))
    reduced = json.loads(git("show", first["commit"]+":"+first["path"]))
    require(old["inherited_contract"]["absolute_delta"] == reduced["contract"]["delta"],
            "Inherited margin changed")
    paths = git("ls-tree", "-r", "--name-only", ARCHIVE).decode().splitlines()
    available = set(paths)
    opportunities_path = HPO/"opportunities_234.csv"
    opportunities_raw = opportunities_path.read_bytes()
    inputs[str(opportunities_path.relative_to(ROOT))] = {"sha256": sha(opportunities_raw), "bytes": len(opportunities_raw)}
    opportunities = list(csv.DictReader(opportunities_raw.decode().splitlines()))
    require(len(opportunities) == 234, "Wrong HPO opportunity cohort")
    phase_rows, job_rows = [], []
    for opportunity in opportunities:
        if opportunity["new_training_receipt_present"].lower() != "true":
            continue
        result_path = HPO/"sources"/opportunity["result_path"]
        result_raw = result_path.read_bytes()
        require(sha(result_raw) == opportunity["result_sha256"], "HPO result source changed")
        result = json.loads(result_raw)
        attempt = result["attempt_id"]
        require(attempt == opportunity["attempt_id"], "Attempt binding changed")
        event_paths = sorted(p for p in available if p.startswith("runs/"+attempt+"/artifacts/phases/")
                             and p.endswith("/events.jsonl"))
        require(event_paths, "A fresh fit has no physical archived phase journal")
        stamps = []
        for path in event_paths:
            raw = git("show", ARCHIVE+":"+path)
            artifact_name = path.split("/artifacts/", 1)[1]
            require(result["artifacts_sha256"][artifact_name] == sha(raw), "Journal not bound to result")
            checked = validate_journal(raw)
            stamps.append(checked["earliest_epoch_seconds"])
            phase_rows.append({"attempt_id": attempt, "model": opportunity["model"],
                               "configuration_id": opportunity["configuration_id"],
                               "fold_origin": opportunity["fold_origin"],
                               "archive_commit": ARCHIVE, "archive_path": path,
                               "journal_sha256": sha(raw), "journal_bytes": len(raw), **checked})
        first_phase = min(stamps)
        job_rows.append({"attempt_id": attempt, "model": opportunity["model"],
                         "configuration_id": opportunity["configuration_id"],
                         "fold_origin": opportunity["fold_origin"], "first_phase_utc": utc(first_phase),
                         "reduced_protocol_before_journal": first["committer_epoch_seconds"] < first_phase,
                         "analysis_engine_before_journal": engine["committer_epoch_seconds"] < first_phase,
                         "minimum_protocol_lead_seconds": first_phase-first["committer_epoch_seconds"],
                         "minimum_engine_lead_seconds": first_phase-engine["committer_epoch_seconds"],
                         "result_sha256": sha(result_raw), "phase_journals": len(event_paths)})
    require(len(job_rows) == 170, "Fresh measured HPO cohort differs from the recorded inventory")
    write_csv(args.output/"fresh_hpo_chronology_170.csv", job_rows)
    write_csv(args.output/"phase_journal_bindings.csv", phase_rows)
    fields = [
        {"field": "primary metric and absolute delta", "first_evidence": "early_margin", "scope": "Historical contract existed; current delta is identical", "full_prospective_claim": False},
        {"field": "delta_g", "first_evidence": "early_margin", "scope": "Margin existed on6Nov; direction/procedure version changes remain separate", "full_prospective_claim": False},
        {"field": "current16systems,6HPOslots,3origins,6finalseeds", "first_evidence": "reduced_protocol_first", "scope": "Registered before the verified170freshHPO journals; cached historical fits are separate", "full_prospective_claim": False},
        {"field": "20primary contrasts,Holm20,delta,14dayblocks", "first_evidence": "reduced_protocol_first", "scope": "Explicit in first reduced protocol; implementation_verified flag was stillfalse", "full_prospective_claim": False},
        {"field": "concrete primary/t/signflip implementation", "first_evidence": "reduced_engine", "scope": "Published source predates the verified fresh journals; runtime freeze has its own receipt", "full_prospective_claim": False},
        {"field": "blinded unseen evaluation", "first_evidence": "historical disclosures", "scope": "Never recovered: inherited E1calendar was alreadyknown, as required and disclosed", "full_prospective_claim": False},
    ]
    write_csv(args.output/"registration_field_scope.csv", fields)
    reports_path = ROOT/"references/literature_e2/prisma_received_v1/full_reports_compact.jsonl"
    reports_raw = reports_path.read_bytes()
    missing = [r for line in reports_raw.splitlines() if (r := json.loads(line))["final_disposition"] == "report_not_retrieved"]
    require(len(missing) == 174, "Do not silently change the reported unresolved full-report cohort")
    requests = [{"group_id": r["group_id"], "title": r["title"], "DOIs": json.dumps(r["DOIs"]),
                 "arxiv_ids": json.dumps(r["arxiv_ids"]), "known_source_urls": json.dumps(r["source_urls"]),
                 "required_input": "legal primary methods/results text with version and hash",
                 "current_state": "not_retrieved_not_a_scientific_exclusion"} for r in missing]
    write_csv(args.output/"primary_report_recovery_requests_174.csv", requests)
    inputs[str(reports_path.relative_to(ROOT))] = {"sha256": sha(reports_raw), "bytes": len(reports_raw)}
    write_json(args.output/"native_export_recovery_requirements.json", {
        "Scopus": {"known_total": 7874, "received_original": 7500, "missing_count": 374,
                   "missing_record_identifiers": None,
                   "required_input": "Exact original native query/history,filters,date,and export covering remaining IDs/ranges",
                   "generic_new_search_would_close_original_query": False},
        "WoS": {"required_input": "Native query history plus totals and batch start/end ranges corresponding to received exports",
                "received_files_do_not_supply_all_query_ranges": True},
        "full_report_recovery_cohort": 174, "original_search_exhaustiveness_closed": False})
    summary = {"version": "e2-registration-chronology-audit-v1",
               "observed_at_utc": datetime.now(timezone.utc).isoformat(),
               "registrations": registrations, "original_HPO_archive_commit": ARCHIVE,
               "fresh_HPO_jobs_with_verified_journal_bindings": len(job_rows),
               "phase_journals_with_content_and_chain_verification": len(phase_rows),
               "all170_reduced_protocol_precedes_journal": all(r["reduced_protocol_before_journal"] for r in job_rows),
               "all170_engine_commit_precedes_journal": all(r["analysis_engine_before_journal"] for r in job_rows),
               "cached_outcomes_not_relabelled_fresh": sum(r["cached_outcome_reused"].lower()=="true" for r in opportunities),
               "fresh_HPO_first_utc": min(r["first_phase_utc"] for r in job_rows),
               "fresh_HPO_last_utc": max(r["last_utc"] for r in phase_rows),
               "complete_pre_training_preregistration_certified": False,
               "blinding_recovered": False, "bibliographic_coverage_repaired": False,
               "missing_reports_retrieved_by_this_audit": 0,
               "new_fits": 0, "new_model_inference": False, "new_statistical_tests": 0,
               "inputs": inputs,
               "limits": ["Git dates and worker UTC fields are stored timestamps, not an independent trusted timestamp authority.",
                          "Hash chains verify content consistency and sequence, not wall-clock accuracy or absence of prior decisions.",
                          "No inference that the first registration preceded all historical/cached fits or exposure to the inherited E1test.",
                          "The pending implementation_verified flag in the first protocol is retained, not rewritten.",
                          "Original374Scopus IDs/ranges and WoShistory cannot be inferred from generic new searches.",
                          "The174unretrieved reports remain access/eligibility uncertainty; the list is a concrete recovery request, not retrieval."]}
    write_json(args.output/"summary.json", summary)
    (args.output/"README.md").write_text(
        "# Auditoría de anterioridad por campo y etapa\n\n"
        "Se ligan registros Git originales a170ajustes HPO nuevos mediante sus journals físicos, "
        "huellas de resultado y cadenas de eventos. La fecha de los commits se compara con UTC de los journals, "
        "sin inventar una autoridad de tiempo externa. Los61resultados reutilizados conservan alcance histórico.\n\n"
        "El primer protocolo reducido es c056c1cf (12/11/2026); la implementación es1e874549. "
        "La tabla informa el resultado de cada comparación; la existencia de contenido previo para esos "
        "ajustes no acredita preregistro integral antes de toda la historia ni recupera cegamiento.\n\n"
        "Se entregan solicitudes exactas para174textos primarios y los campos que faltan en las "
        "exportaciones Scopus/WoS. No se borran informes inaccesibles ni se declara exhaustividad reparada.\n\n"
        "Repetición: python scripts/audit_e2_registration_chronology.py --output NUEVA_CARPETA. "
        "Requiere el repositorio y los commits Git originales enumerados; no usa red ni entrena modelos.\n",
        encoding="utf-8")
    files = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
             for p in sorted(args.output.iterdir()) if p.is_file()}
    write_json(args.output/"manifest.json", {"files": files, "script_sha256": sha(Path(__file__).read_bytes())})
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, (list, dict))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
