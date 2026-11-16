"""Freeze the disposition of four E2 observations without changing their evidence.

This is a documentary build. It never trains, scores, bootstraps, requests data,
or changes previously published downloads. A closed investigation is not a
passed scientific acceptance gate.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "book_e2/_static/e2_resolucion_final_20261116"
FOLLOWUP = "book_e2/_static/e2_closure_followup_20261116"
SCIENCE = "book_e2/_static/e2_scientific_closure_20261116"
SOURCES = {
    "negative_controls.json": f"{FOLLOWUP}/audit/summary.json",
    "negative_trace_review.json": f"{FOLLOWUP}/audit/trace_review.json",
    "hpo_design.json": f"{FOLLOWUP}/hpo/descriptive_summary.json",
    "hpo_identifiability.csv": f"{FOLLOWUP}/hpo/configuration_pair_identifiability_195.csv",
    "portable_reproduction.json": f"{FOLLOWUP}/reproduction/summary.json",
    "conditional_inference.json": f"{FOLLOWUP}/inference/recovery_v3/result_v1/summary.json",
    "preregistration.json": f"{SCIENCE}/preregistration/summary.json",
    "final_registration.json": f"{SCIENCE}/final_registration_chronology/audit_results/summary.json",
    "appendix_B_previous.json": f"{FOLLOWUP}/delivery/appendix_B_current.json",
    "budget_before.json": f"{FOLLOWUP}/delivery/budget_final.json",
    "scopus_scope.json": "book_e2/_static/e2_literature_scope_7500_20261116/scope_decision.json",
    "scopus_closure.json": "book_e2/_static/e2_literature_scope_7500_20261116/closure_state.json",
    "original_A2.csv": "book_e2/_static/e2_benchmark_final_20261113/benchmark_A2.csv",
}


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def decisions() -> list[dict]:
    return [
        {
            "id": "C01", "observation": "Aceptación integral de la auditoría",
            "management_status": "closed_with_unestablished_acceptance",
            "execution_complete_in_declared_scope": True,
            "full_guide_acceptance_established": False,
            "disposition": "Expediente ejecutado y cerrado; B03 no se acredita íntegramente.",
            "basis": "13 controles originales y 6 adicionales ejecutados. TOST/Holm2 no establece cercanía a la constante. La revisión estática no identifica una ruta explícita de etiquetas futuras; la causa de la señal retenida permanece indeterminada.",
            "retained_result": "A2 y los análisis condicionales se conservan como resultados retrospectivos reproducibles, sin certificación integral de Tabla 10.1.",
            "claims_not_supported": ["fuga demostrada por este control", "ausencia universal de fuga", "auditoría integral aprobada"],
            "evidence_files": ["negative_controls.json", "negative_trace_review.json"],
            "active_corrective_tasks": [],
        },
        {
            "id": "C02", "observation": "Cobertura e importancia HPO",
            "management_status": "closed_with_reduced_design_nonconformity",
            "execution_complete_in_declared_scope": True,
            "full_guide_acceptance_established": False,
            "disposition": "Selección reducida cerrada; cobertura mínima integral e importancia independiente no acreditadas.",
            "basis": "234 oportunidades, 231 resultados completos y 13 ganadores. Seis configuraciones por red acreditan igualdad de plazas. Sólo un par de 195 es unifactor; las asociaciones publicadas son descriptivas. La propuesta posterior de 16 tuplas no fue ejecutada y se archiva como diseño futuro.",
            "retained_result": "Ranking condicionado a las recetas efectivamente exploradas y costes observados con ámbitos separados.",
            "claims_not_supported": ["búsqueda HPO exhaustiva", "importancia independiente de todos los hiperparámetros", "igual coste homogéneo integral entre familias"],
            "evidence_files": ["hpo_design.json", "hpo_identifiability.csv"],
            "active_corrective_tasks": [],
        },
        {
            "id": "C03", "observation": "Reproducción integral",
            "management_status": "closed_with_verified_reanalysis_and_historical_limit",
            "execution_complete_in_declared_scope": True,
            "full_guide_acceptance_established": False,
            "disposition": "Se integra y verifica el reanálisis disponible; B21 no se acredita para toda la campaña histórica.",
            "basis": "La reproducción original, repetición CPU de recetas fijas y portabilidad tienen recibos propios. El nuevo comando integra módulos disponibles desde entradas congeladas; no se confunde con repetir HPO, todos los entrenamientos ni todas las figuras.",
            "retained_result": "Reproducción numérica e inferencia ya verificadas y un comando independiente para los reanálisis incluidos.",
            "claims_not_supported": ["campaña histórica completa regenerada por el nuevo comando", "paridad CPU/A100", "nuevo entrenamiento integral ejecutado en este cierre"],
            "evidence_files": ["portable_reproduction.json", "integrated_reanalysis.json"],
            "active_corrective_tasks": [],
        },
        {
            "id": "C04", "observation": "Anterioridad y alcance inferencial/bibliográfico",
            "management_status": "closed_with_retrospective_scope",
            "execution_complete_in_declared_scope": True,
            "full_guide_acceptance_established": False,
            "disposition": "Cronología y comparaciones resueltas documentalmente; las afirmaciones se limitan al diseño observado.",
            "basis": "Registro acreditado para 170 HPO nuevos y 78 ajustes finales, sin certificar toda la historia reutilizada. OOF249-prueba608 está ejecutado pero no constituye validación externa independiente del procedimiento seleccionado. Scopus se cierra con la cohorte original de 7500, conservando los límites de procedencia y cobertura.",
            "retained_result": "Familia primaria registrada con inferencia retrospectiva condicionada, RQ2/RQ3 posteriores y descomposición finita; 9559 grupos multibase, 42 estudios y 174 informes no recuperados conservados.",
            "claims_not_supported": ["todo el proyecto preregistrado antes de entrenar", "cegamiento restaurado", "validación externa del procedimiento", "exhaustividad bibliográfica demostrada"],
            "evidence_files": ["preregistration.json", "final_registration.json", "conditional_inference.json", "scopus_scope.json", "scopus_closure.json"],
            "active_corrective_tasks": [],
        },
    ]


VERIFY = '''"""Check the final E2 disposition offline, without running models or statistics."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent

def validate_state(state):
    if state.get("management_closed") is not True or state.get("active_corrective_tasks") != []:
        raise ValueError("The frozen scope must have a recorded terminal management decision")
    for key in ("full_guide_compliance_certified", "full_audit_accepted", "full_historical_campaign_retrained_by_this_closure", "historical_preregistration_integral_certified", "independent_external_validation_established", "leakage_demonstrated_by_negative_control"):
        if state.get(key) is not False:
            raise ValueError("Unsupported scientific acceptance: " + key)
    if state.get("NAS_required") is not False or state.get("Scopus_original_records") != 7500:
        raise ValueError("The fixed architecture/literature scope changed")
    rows = state.get("observations", [])
    if {r.get("id") for r in rows} != {"C01", "C02", "C03", "C04"} or len(rows) != 4:
        raise ValueError("Exactly four final dispositions are required")
    for row in rows:
        if not row.get("management_status", "").startswith("closed_") or row.get("active_corrective_tasks") != []:
            raise ValueError("Unresolved management disposition")
        if row.get("full_guide_acceptance_established") is not False:
            raise ValueError("Closing a task cannot approve a missing acceptance criterion")
    return state

def verify(root=ROOT):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        rel = PurePosixPath(record["path"])
        if rel.is_absolute() or ".." in rel.parts or not rel.parts:
            raise ValueError("Unsafe manifest path")
        path = root / str(rel)
        if path.is_symlink() or not path.is_file():
            raise ValueError("Missing or nonregular input: " + str(rel))
        raw = path.read_bytes()
        if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("Changed package input: " + str(rel))
    state = validate_state(json.loads((root / "decisions.json").read_text(encoding="utf-8")))
    prior = json.loads((root / "evidence/appendix_B_previous.json").read_text(encoding="utf-8"))
    current = json.loads((root / "appendix_B_final.json").read_text(encoding="utf-8"))
    if {r["id"]: r["status"] for r in prior["items"]} != {r["id"]: r["status"] for r in current["items"]}:
        raise ValueError("Documentary closure must preserve scientific Appendix B statuses")
    inputs = root / "evidence"
    negative = json.loads((inputs / "negative_controls.json").read_text(encoding="utf-8"))
    hpo = json.loads((inputs / "hpo_design.json").read_text(encoding="utf-8"))
    inference = json.loads((inputs / "conditional_inference.json").read_text(encoding="utf-8"))
    integrated = json.loads((inputs / "integrated_reanalysis.json").read_text(encoding="utf-8"))
    if negative["completed_fits"] != 6 or negative["original13_model_audit_accepted"] is not False:
        raise ValueError("Negative-control evidence does not support the disposition")
    if hpo["isolated_hyperparameter_importance_established"] is not False or hpo["completed_fold_outcomes"] != 231:
        raise ValueError("HPO evidence changed")
    if inference["independent_post_selection_validation_established"] is not False:
        raise ValueError("Independent validation was not established by this comparison")
    if integrated.get("status") != "passed" or integrated.get("full_historical_campaign_retrained") is not False:
        raise ValueError("Integrated replay must pass its own limited scope")
    return {"status": "passed", "management_closed": True, "observations": 4, "active_corrective_tasks": [], "scientific_acceptance_changed": False, "full_guide_compliance_certified": False, "checked_files": len(manifest["files"]), "new_fits": 0, "new_forwards": 0, "new_bootstrap_draws": 0, "network_calls": 0}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("verification_result.json"))
    args = parser.parse_args()
    result = verify()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
'''


README = """# Resolución final de las cuatro observaciones de E2

Esta versión cierra el seguimiento de los cuatro puntos con la evidencia disponible y las limitaciones identificadas. **Cierre de gestión no significa aprobación científica.** Los requisitos parciales del Apéndice B conservan su estado. Una evaluación posterior puede mantener descuentos por esas limitaciones; este archivo no promete una nota ni certifica el cumplimiento integral de la guía.

1. Extraer todo el ZIP en una carpeta propia.
2. Ejecutar `python verify.py` (Python 3.12 recomendado, sin dependencias externas ni red).
3. Abrir `decisions.json`, `observaciones_finales.csv`, `appendix_B_final.csv` y el informe generado `verification_result.json`.

El verificador comprueba hashes y la coherencia de los dictámenes con resúmenes ya publicados. **No entrena, no hace inferencia ni vuelve a calcular estadísticos.** El reanálisis numérico integrado tiene su ZIP independiente, con dependencias e instrucciones propias, enlazado desde §67. Este archivo pequeño incluye el recibo de aquella ejecución, no su software/datos completos.

## Decisiones terminales

- C01: controles negativos ejecutados; cercanía al azar y aceptación integral no establecidas. Causa de la señal indeterminada; no se ha demostrado fuga mediante este control.
- C02: HPO reducido ejecutado y conservado; cobertura mínima integral e importancia independiente no acreditadas. La propuesta de ampliación se archiva como diseño no ejecutado, sin tarea activa en esta entrega. NAS no aplica.
- C03: reanálisis incluidos integrados y verificados con un comando. No se acredita reconstrucción automática de toda la campaña histórica ni repetición integral de HPO y todas sus figuras.
- C04: evidencia anterior a 170 trabajos HPO nuevos y 78 ajustes finales; no acredita toda la historia o cegamiento. OOF249–prueba608 conserva alcance condicional. Scopus se cierra con 7.500 registros originales, sin pedir más exportaciones.

No se ordenan nuevas corridas, recuperaciones o exportaciones como obligaciones de este corte congelado. Una futura ampliación sería otro estudio, con plan y recursos propios; no convierte estos resultados en aprobados retrospectivamente. Las métricas, márgenes, p-valores, intervalos y decisiones de A2 permanecen intactos.
"""


def build(integrated_summary: Path) -> None:
    integrated = json.loads(integrated_summary.read_text(encoding="utf-8"))
    if integrated.get("status") != "passed" or integrated.get("full_historical_campaign_retrained") is not False:
        raise ValueError("A verified, explicitly scoped integrated replay is required")
    evidence = DEST / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    records = []
    for name, relative in SOURCES.items():
        source = ROOT / relative
        shutil.copyfile(source, evidence / name)
        records.append({"snapshot": "evidence/" + name, "source_path": relative,
                        "sha256": digest(source), "bytes": source.stat().st_size})
    shutil.copyfile(integrated_summary, evidence / "integrated_reanalysis.json")
    records.append({"snapshot": "evidence/integrated_reanalysis.json",
                    "source_path": integrated_summary.relative_to(ROOT).as_posix(),
                    "sha256": digest(integrated_summary), "bytes": integrated_summary.stat().st_size})
    observations = decisions()
    state = {
        "schema_version": 1, "version": "e2-final-disposition-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviewed_source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "author_instruction": "Solucionar y cerrar los cuatro puntos con la información disponible, sin repetirlos como tareas abiertas.",
        "management_closed": True, "active_corrective_tasks": [],
        "full_guide_compliance_certified": False, "full_audit_accepted": False,
        "full_historical_campaign_retrained_by_this_closure": False,
        "historical_preregistration_integral_certified": False,
        "independent_external_validation_established": False,
        "leakage_demonstrated_by_negative_control": False,
        "NAS_required": False, "Scopus_original_records": 7500,
        "scope": "Frozen retrospective E2; terminal management dispositions retain scientific nonacceptance and conditional claims.",
        "observations": observations,
        "original_A2_sha256": digest(evidence / "original_A2.csv"),
        "scientific_results_modified": False,
        "evaluation_implication": "Operational closure does not erase limitations or require an evaluator to remove corresponding deductions.",
    }
    write_json(DEST / "decisions.json", state)
    write_json(DEST / "source_manifest.json", {"files": records})
    previous = json.loads((evidence / "appendix_B_previous.json").read_text(encoding="utf-8"))
    linked = {"B01": "C04", "B03": "C01", "B06": "C02", "B09": "C02", "B21": "C03"}
    items = []
    for original in previous["items"]:
        row = dict(original)
        row["management_status"] = "closed_with_documented_limit" if row["id"] in linked else "reviewed_no_active_correction"
        row["final_observation_id"] = linked.get(row["id"], "")
        row["active_corrective_tasks"] = []
        items.append(row)
    appendix = {
        "schema_version": 1, "version": "e2-appendix-B-final-disposition-v1",
        "source_guide_sha256": previous["source_guide_sha256"],
        "counts": previous["counts"], "scientific_statuses_changed": False,
        "management_closed": True, "active_corrective_tasks": [],
        "scope": "21 literal requirements; scientific statuses preserved. Terminal disposition is a separate field and grants no acceptance.",
        "items": items,
    }
    write_json(DEST / "appendix_B_final.json", appendix)
    columns = ["id", "requirement", "status", "management_status", "final_observation_id", "evidence", "evidence_paths", "remaining_limit"]
    with (DEST / "appendix_B_final.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(items)
    columns = ["id", "observation", "management_status", "full_guide_acceptance_established", "disposition", "basis", "retained_result"]
    with (DEST / "observaciones_finales.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(observations)
    (DEST / "verify.py").write_text(VERIFY, encoding="utf-8")
    (DEST / "LEEME.md").write_text(README, encoding="utf-8")
    package_files = ["decisions.json", "source_manifest.json", "appendix_B_final.json", "appendix_B_final.csv", "observaciones_finales.csv", "verify.py", "LEEME.md"]
    package_files += ["evidence/" + path.name for path in sorted(evidence.iterdir()) if path.is_file()]
    write_json(DEST / "manifest.json", {"schema_version": 1, "files": [
        {"path": name, "bytes": (DEST / name).stat().st_size, "sha256": digest(DEST / name)}
        for name in sorted(package_files)]})
    archive = DEST / "resolucion_cuatro_observaciones_v1.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for name in sorted(package_files + ["manifest.json"]):
            handle.write(DEST / name, "resolucion_final/" + name)
    print(json.dumps({"archive": archive.relative_to(ROOT).as_posix(), "bytes": archive.stat().st_size, "sha256": digest(archive), "new_fits": 0, "scientific_results_modified": False}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integrated-summary", type=Path, required=True)
    args = parser.parse_args()
    build(args.integrated_summary.resolve())
