#!/usr/bin/env python3
"""Close the user-authorized original7500 Scopus scope; metadata operations only.

Never edits prior PRISMA, screening, quality, benchmark or ZIP evidence. Produces
a new decision and a standard-library standalone verifier without raw abstracts.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "book_e2/_static/e2_literature_scope_7500_20261116"
SOURCE = Path("/workspace/attachments/98533e5e-91b4-411a-b61c-befd8cbe3d14/export_dcb22fe7-ce2a-4457-ab38-2a1021da5766_2026-11-06T011951.464806988.csv")
SOURCE_SHA = "387f673a2146902ce12888ab3d662c095abc33d2cf5b54a168e9e2a12099f7de"
BASELINES = [
    "book_e2/_static/e2_prisma_received_v1/received_flow.json",
    "references/literature_e2/prisma_received_v1/received_groups_9559.csv",
    "references/literature_e2/prisma_received_v1/quality_applied_studies.csv",
    "references/literature_e2/prisma_received_v1/manifest.json",
]
VERSION = "e2-original-scopus7500-operational-scope-v1"
FIELDS = ["original_row_index", "EID", "title", "DOI", "year", "source_title"]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read_source(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 7500 or len({r["EID"].strip() for r in rows}) != 7500:
        raise ValueError("The chosen original cohort must retain7500unique nonempty EIDs")
    if any(not r["EID"].strip() for r in rows):
        raise ValueError("Empty source EID")
    return [{"original_row_index": i, "EID": r["EID"], "title": r["Title"],
             "DOI": r["DOI"], "year": r["Year"], "source_title": r["Source title"]}
            for i, r in enumerate(rows, 1)]


def inventory_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


RUNNER = '''#!/usr/bin/env python3
"""Verify the frozen7500cohort with standard Python; no repository/network needed."""
import argparse,csv,hashlib,io,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
FIELDS=['original_row_index','EID','title','DOI','year','source_title']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--original-export',type=Path,help='Optional private original CSV; never required for the default check')
 a=p.parse_args();manifest=json.loads((ROOT/'manifest.json').read_text())
 for row in manifest['files']:
  path=ROOT/row['path']
  if not path.is_file() or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('Changed package file: '+row['path'])
 with (ROOT/'inventory_7500.csv').open(encoding='utf-8',newline='') as f:
  reader=csv.DictReader(f)
  if reader.fieldnames!=FIELDS:raise ValueError('Inventory schema changed')
  rows=list(reader)
 if len(rows)!=7500 or len({r['EID'] for r in rows if r['EID'].strip()})!=7500:raise ValueError('7500unique nonempty EIDs required')
 if [int(r['original_row_index']) for r in rows]!=list(range(1,7501)):raise ValueError('Original row order changed')
 decision=json.loads((ROOT/'scope_decision.json').read_text());state=json.loads((ROOT/'closure_state.json').read_text())
 flow=json.loads((ROOT/'received_flow_baseline.json').read_text())
 baseline=(flow['received_groups_screened'],flow['unique_studies_included_applied_narrative'],flow['report_not_retrieved_rows'])
 if baseline!=(9559,42,174):raise ValueError('Preserved PRISMA denominator changed')
 if sha(ROOT/'received_flow_baseline.json')!=state['preserved_PRISMA']['public_flow_sha256']:raise ValueError('Preserved flow bytes changed')
 if not state['operational_scope_closed'] or state['active_Scopus_export_recovery_tasks']!=[]:raise ValueError('Scope decision state changed')
 raw_checked=False
 if a.original_export:
  if sha(a.original_export)!=decision['selected_original_export']['sha256']:raise ValueError('Wrong raw export: use original dcb22 cohort')
  with a.original_export.open(encoding='utf-8-sig',newline='') as f:original=list(csv.DictReader(f))
  compact=[{'original_row_index':i,'EID':r['EID'],'title':r['Title'],'DOI':r['DOI'],'year':r['Year'],'source_title':r['Source title']} for i,r in enumerate(original,1)]
  stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=FIELDS,lineterminator='\\n');writer.writeheader();writer.writerows(compact)
  if hashlib.sha256(stream.getvalue().encode('utf-8')).hexdigest()!=sha(ROOT/'inventory_7500.csv'):raise ValueError('Raw-to-inventory projection differs')
  raw_checked=True
 dois=[r['DOI'].lower().strip() for r in rows if r['DOI'].strip()]
 result={'status':'passed','inventory_rows':len(rows),'unique_EIDs':len({r['EID'] for r in rows}),
  'nonempty_DOI_rows':len(dois),'unique_nonempty_DOIs':len(set(dois)),
  'preserved_PRISMA_counts':list(baseline),'operational_scope_closed':True,'active_Scopus_export_recovery_tasks':[],
  'original_private_export_raw_bytes_reverified_this_run':raw_checked,
  'inventory_and_bundled_flow_hashes_verified':True,'new_screening_performed':False,'new_records_incorporated':0,
  'original_native_query_exhaustiveness_established':False,'model_operations':0,'network_calls':0}
 (ROOT/'verification_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\\n')
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
'''


README = """# Cierre operativo de Scopus sobre los 7.500 registros originales

Por instrucción expresa del usuario —«pues cierra eso con los 7500 registros»— se fija como corte de Scopus inicial el archivo original `export_dcb22fe7-ce2a-4457-ab38-2a1021da5766_2026-11-06T011951.464806988.csv`, SHA-256 `387f673a2146902ce12888ab3d662c095abc33d2cf5b54a168e9e2a12099f7de`. La decisión es posterior al análisis y no se presenta como preregistro original.

**Estado: cerrado operativamente dentro del alcance congelado; cero tareas activas de recuperación de exportaciones Scopus para esta versión.** Los 7.500 registros son el componente inicial Scopus, no el número total de estudios o grupos. Las otras rondas ya admitidas de WoS, Scopus, IEEE y arXiv conservan exactamente su función. El flujo multibase mantiene **9.559 grupos, 42 estudios aplicados y 174 informes no recuperados**, sin nuevo cribado, adjudicación, deduplicación o cálculo del PRISMA.

## Alcance y límites conservados

- La diferencia histórica entre 7.874 resultados mostrados y 7.500 recibidos sigue siendo 374. No se afirma haber recuperado esos identificadores ni se los convierte en exclusiones científicas. Su recuperación no condiciona el cierre de este corte.
- El total actual de 7.936 mostrado en la conversación no equivale a un conjunto recibido íntegramente. El nuevo CSV de 7.500 comparte 7.436 EID con el original; tiene 64 diferentes y omite 64 originales. El archivo posterior de 436 repite EID ya presentes en ese nuevo CSV. Son actualizaciones **no incorporadas** a esta versión. No se afirma haber cribado nuevamente esas actualizaciones ni haber sumado sus EID al flujo.
- La cadena nativa exacta, todos los filtros, la fecha de ejecución y los rangos de la consulta Scopus original no están íntegramente acreditados. Las cadenas WoS Q1/Q2 y las consultas arXiv conservadas no sustituyen esa cadena Scopus. Los límites históricos de correspondencia y cobertura WoS se mantienen.
- Este cierre no certifica exhaustividad de las bases, cobertura total de consultas, elegibilidad conocida de los 174 informes inaccesibles, mérito global óptimo del Top10 ni cumplimiento metodológico integral de la guía. Tampoco cambia A2, modelos, inferencia, datos de mercado o la decisión de NAS no aplicable.

Los formularios, colas y estados anteriores de recuperación se conservan como **archivos históricos**. No deben interpretarse como una solicitud vigente de más exportaciones Scopus para esta versión. Una ampliación futura requeriría otra versión con su recepción, identidad, cribado y flujo propios.

## Archivos y verificación independiente

`inventory_7500.csv` contiene sólo índice original, EID, título, DOI, año y fuente. No redistribuye el CSV nativo, resúmenes, referencias completas o PDF. Los campos y el orden proceden exclusivamente del archivo original admitido por hash.

Descargar **`cierre_scopus_7500.zip`**, extraerlo y ejecutar:

```bash
python run.py
```

Sólo requiere Python estándar. No necesita repositorio, Git, red, cuentas, Torch ni datos de mercado. Verifica los hashes del paquete, los 7.500 EID únicos, el orden y los conteos del flujo histórico copiado; escribe `verification_result.json`. No vuelve a ejecutar el cribado ni el PRISMA.

El archivo privado original no se incluye. El modo predeterminado verifica el inventario derivado y el flujo incluido; no afirma recalcular el hash de un archivo original ausente. Quien disponga de ese original puede comprobar además sus bytes y su proyección exacta:

```bash
python run.py --original-export /ruta/al/export_dcb22_original.csv
```

`verification_report.json` acredita la admisión local del original por SHA y la conservación de los archivos previos antes/después de esta operación de metadatos. `standalone_qa.json` acredita la ejecución del ZIP extraído fuera del repositorio. Los manifiestos nuevos no reescriben los recibos históricos.
"""


def metadata_updates():
    records = []
    for name, path, expected_rows, note in [
        ("current7500", Path("/workspace/attachments/2f373051-7294-46fe-8510-dea386914cf3/export_e8599f2b-6b3e-4394-aad3-fa54dd6e2d4c_2026-11-16T124511.800102557.csv"), 7500,
         "Updated7500 snapshot,64EIDs different from original; not substituted into frozen analysis"),
        ("current436", Path("/workspace/attachments/344a4e83-fdeb-4ffa-9980-03259f21f11b/export_25f282f4-86f1-4d03-8b2c-2bcd482dc09d_2026-11-16T130545.75085967.csv"), 436,
         "All436EIDs already present in current7500; not a distinct final-range supplement")]:
        if not path.is_file():
            raise ValueError("Previously observed update missing; do not simulate its receipt")
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == expected_rows
        records.append({"id": name, "filename": path.name, "bytes": path.stat().st_size,
                        "sha256": sha(path), "rows": len(rows), "unique_EIDs": len({r['EID'] for r in rows}),
                        "admitted_to_current_review": False, "new_screening_performed_by_this_closure": False,
                        "note": note})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(); out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("New scope output must be empty; never overwrite an existing closure")
    if sha(args.source) != SOURCE_SHA or args.source.stat().st_size != 19722283:
        raise ValueError("Use the admitted original7500 export, not the later7500 update")
    previous = [{"path": name, "sha256": sha(ROOT / name), "bytes": (ROOT / name).stat().st_size}
                for name in BASELINES]
    flow = json.loads((ROOT / BASELINES[0]).read_text())
    assert (flow["received_groups_screened"], flow["unique_studies_included_applied_narrative"], flow["report_not_retrieved_rows"]) == (9559, 42, 174)
    inventory = read_source(args.source)
    update_records = metadata_updates()
    out.mkdir(parents=True, exist_ok=True)
    raw = inventory_bytes(inventory)
    (out / "inventory_7500.csv").write_bytes(raw)
    now = datetime.now(timezone.utc).isoformat()
    decision = {
        "version": VERSION, "decided_at_utc": now,
        "authorization": {"source": "Explicit user instruction in this session", "verbatim": "pues cierra eso con los 7500 registros"},
        "decision_timing": "Posterior scope closure; not a reconstructed original preregistration",
        "selected_original_export": {"filename": args.source.name, "bytes": args.source.stat().st_size,
                                     "sha256": SOURCE_SHA, "rows": 7500, "unique_EIDs": 7500},
        "scope": "Freeze original initial Scopus7500 contribution within the existing9559-group multibase review; retain other previously admitted rounds unchanged",
        "active_Scopus_export_recovery_tasks": [],
        "additional_Scopus_exports_required_for_this_version": False,
        "historical_displayed_total": 7874, "historical_difference_not_recovered": 374,
        "reported_current_UI_total": 7936,
        "current_UI_total_provenance": "Reported in current exchange; no complete7936unique-ID export admitted",
        "later_updates": update_records,
        "original_exact_native_query_and_filters_fully_verified": False,
        "WoS_query_batch_total_correspondence_fully_verified": False,
        "database_or_original_query_exhaustiveness_certified": False,
        "new_screening_or_PRISMA_recomputation": False,
        "new_study_inclusions": 0, "benchmark_changed": False, "NAS": "not_applicable_defined_bodies",
    }
    write_json(out / "scope_decision.json", decision)
    state = {
        "version": VERSION, "operational_scope_closed": True,
        "active_Scopus_export_recovery_tasks": [], "active_Scopus_export_recovery_task_count": 0,
        "additional_exports_are_closure_prerequisite": False,
        "selected_initial_Scopus_records": 7500,
        "preserved_PRISMA": {"received_groups": 9559, "included_applied_studies": 42,
                             "report_not_retrieved": 174, "public_flow_sha256": previous[0]["sha256"],
                             "files": previous, "recomputed": False},
        "historical_374_difference": "Accepted coverage limitation, not recovered records or scientific exclusions",
        "later_updates_admitted": False, "later_updates_newly_screened": False,
        "original_native_query_exhaustiveness_established": False,
        "old_recovery_forms_and_backlogs": "Historical snapshots; superseded as active Scopus recovery requests for this scoped version",
        "full_guide_compliance_certified": False,
    }
    write_json(out / "closure_state.json", state)
    shutil.copyfile(ROOT / BASELINES[0], out / "received_flow_baseline.json")
    (out / "run.py").write_text(RUNNER, encoding="utf-8")
    (out / "README.md").write_text(README, encoding="utf-8")
    (out / "LEEME.md").write_text(README, encoding="utf-8")
    after = [{"path": name, "sha256": sha(ROOT / name), "bytes": (ROOT / name).stat().st_size}
             for name in BASELINES]
    assert previous == after
    dois = [r["DOI"].lower().strip() for r in inventory if r["DOI"].strip()]
    report = {"version": VERSION, "status": "passed", "private_original_bytes_admitted_by_sha256": True,
              "inventory_rows": len(inventory), "unique_EIDs": len({r["EID"] for r in inventory}),
              "inventory_sha256": sha(out / "inventory_7500.csv"), "nonempty_DOI_rows": len(dois),
              "unique_nonempty_DOIs": len(set(dois)), "prior_baseline_hashes_unchanged": previous == after,
              "baseline_before": previous, "baseline_after": after,
              "model_operations": 0, "network_calls": 0, "new_screening_decisions": 0,
              "old_artifacts_modified": False, "raw_export_or_abstracts_redistributed": False}
    write_json(out / "verification_report.json", report)
    members = ["inventory_7500.csv", "scope_decision.json", "closure_state.json", "received_flow_baseline.json",
               "verification_report.json", "run.py", "README.md", "LEEME.md"]
    manifest = {"version": VERSION, "files": [{"path": name, "bytes": (out / name).stat().st_size,
                     "sha256": sha(out / name)} for name in members],
                "default_requires_repository_or_network": False,
                "private_original_export_included": False, "new_screening_or_models_run": False}
    write_json(out / "manifest.json", manifest)
    package = out / "cierre_scopus_7500.zip"
    with zipfile.ZipFile(package, "x", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in [*members, "manifest.json"]:
            archive.write(out / name, name)
    with tempfile.TemporaryDirectory(prefix="e2-scopus7500-standalone-", dir="/tmp") as folder:
        with zipfile.ZipFile(package) as archive:
            archive.extractall(folder)
        default = subprocess.run([sys.executable, "-I", "run.py"], cwd=folder, check=True,
                                 text=True, capture_output=True, timeout=15)
        default_result = json.loads((Path(folder) / "verification_result.json").read_text())
        optional = subprocess.run([sys.executable, "-I", "run.py", "--original-export", str(args.source.resolve())],
                                  cwd=folder, check=True, text=True, capture_output=True, timeout=15)
        optional_result = json.loads((Path(folder) / "verification_result.json").read_text())
        assert default_result["original_private_export_raw_bytes_reverified_this_run"] is False
        assert optional_result["original_private_export_raw_bytes_reverified_this_run"] is True
        write_json(out / "standalone_qa.json", {"version": VERSION, "status": "passed",
                   "outside_repository": True, "isolated_python": True, "ZIP_sha256": sha(package),
                   "ZIP_bytes": package.stat().st_size, "default_returncode": default.returncode,
                   "default_requires_private_source": False, "default_result": default_result,
                   "optional_original_verification_returncode": optional.returncode,
                   "optional_original_result": optional_result})
    assert all(sha(ROOT / row["path"]) == row["sha256"] for row in previous)
    write_json(out / "delivery_manifest.json", {"version": VERSION, "producer": str(Path(__file__).relative_to(ROOT)),
               "producer_sha256": sha(__file__), "files": [{"path": p.name, "bytes": p.stat().st_size,
               "sha256": sha(p)} for p in sorted(out.iterdir()) if p.is_file()]})
    print(json.dumps({"status": "operational_scope_closed", "records": 7500, "active_recovery_tasks": 0,
                      "PRISMA": [9559,42,174], "old_baseline_unchanged": True,
                      "ZIP": str(package), "ZIP_bytes": package.stat().st_size, "ZIP_sha256": sha(package)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
