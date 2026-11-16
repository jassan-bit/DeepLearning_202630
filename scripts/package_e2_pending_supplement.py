"""Build an independent stdlib supplement; keep the twelve v2 ZIPs immutable."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
AREA = "book_e2/_static/e2_pending_closure_20261116"
RUNNER = r'''"""Verify this supplement and reproduce its algebraic deduction."""
from pathlib import Path
import hashlib, html, json, subprocess, sys
ROOT=Path(__file__).resolve().parent
def main():
    manifest=json.loads((ROOT/"source_manifest.json").read_text(encoding="utf-8"))
    for row in manifest["files"]:
        relative=Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts: raise ValueError("Unsafe manifest path")
        path=ROOT/relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size!=row["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest()!=row["sha256"]:
            raise ValueError("Changed or missing input: "+str(relative))
    output=ROOT/"output"
    if output.exists(): raise ValueError("Preserve previous output; rename its folder before repeating")
    output.mkdir()
    subprocess.run([sys.executable,str(ROOT/"scripts/e2_rq3_necessary_component.py"),"--output",str(output/"RQ3")],check=True)
    process=subprocess.run([sys.executable,str(ROOT/"scripts/reproduce_e2_campaign.py"),"--check","--output",str(output/"readiness")],check=False)
    if process.returncode not in (0,2): raise RuntimeError("Readiness validation failed")
    status=json.loads((output/"readiness/reproduction_status.json").read_text())
    proof=json.loads((output/"RQ3/necessary_component_proof.json").read_text())
    report={"source_commit":manifest["source_commit"],"included_files_verified":len(manifest["files"]),
            "RQ3_all_cost_bounds_equal_one":proof["all_ten_lower_bounds_equal_one"],
            "full_CPU_IUT_executed":False,"full_campaign_passed":False,
            "readiness_status":status["status"],"new_training_or_inference":False,
            "network_downloads":False,"readiness_exit_code":process.returncode}
    (output/"execution_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    body="<html lang='es'><meta charset='utf-8'><title>Seguimiento E2</title><h1>Seguimiento E2</h1><p>Archivos verificados y deducción de RQ3 reproducida. No se establece mejora conjunta en el CPU medido.</p><p>La auditoría integral, los controles reales y toda la campaña de entrenamiento conservan sus bloqueos. La salida parcial es deliberada.</p><pre>"+html.escape(json.dumps(report,ensure_ascii=False,indent=2))+"</pre></html>"
    (output/"informe.html").write_text(body,encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False))
    return 0
if __name__=="__main__": raise SystemExit(main())
'''
README = """# Seguimiento de E2 — 16/11/2026

Extraiga toda la carpeta y ejecute: python run.py (Python 3.10 o posterior).
Abra output/informe.html. Usa biblioteca estándar, sin red, Git, Torch ni otros
paquetes. Verifica los archivos, repite la deducción algebraica de RQ3 y genera
un diagnóstico de disponibilidad. Conserve o renombre output/ antes de repetir.

Funciona por separado para esas tareas. No incluye los doce ZIP v2 ni los OHLCV
ausentes; no entrena modelos ni ejecuta controles de fuga. Los bloqueos se muestran
aunque la verificación del suplemento sea correcta.

Para orquestar los análisis de los doce entregables, descargue los ZIP v2 del
inventario reproduction_inputs.json incluido y ejecute:

    python scripts/reproduce_e2_campaign.py --replay --archives-dir RUTA_ZIPS --output SALIDA_NUEVA

Los entornos se indican con --python, --model-python, --analysis-python y
--book-python. Consulte --help y el README de reproducción incluido para sus
versiones. No se instalan dependencias automáticamente. La salida 2 del
orquestador significa parcial; 1 indica error. La reproducción analítica no
repite toda la selección, entrenamientos, ablaciones e inferencias de mercado.

Los tiempos de fases y cargos de campaña tienen ámbitos solapados y no se suman.
NAS no aplica. La IUT original sigue sin ejecutarse.
"""


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def package(source, output):
    if subprocess.check_output(["git","rev-parse",source+"^{commit}"],cwd=ROOT,text=True).strip()!=source:
        raise ValueError("Use the exact full source commit")
    blobs={}
    for row in subprocess.check_output(["git","ls-tree","-rz",source],cwd=ROOT).split(b"\0"):
        if row:
            meta,name=row.split(b"\t",1); _,kind,oid=meta.split()
            if kind==b"blob": blobs[name.decode()]=oid.decode()
    inputs=json.loads((ROOT/AREA/"reproduction_inputs.json").read_text())
    proof=json.loads((ROOT/AREA/"rq3_necessary_component/v2/necessary_component_proof.json").read_text())
    paths={r["path"] for r in inputs["evidence_files"]}
    paths.update(r["repository_path"] for r in proof["source_records"].values())
    paths.update({"scripts/reproduce_e2_campaign.py","scripts/e2_corrections_budget.py",
                  "scripts/e2_rq3_necessary_component.py","scripts/e2_pending_data_status.py",
                  "scripts/download_e2_known_market.py",
                  "data/manifests/test_raw_reserve_20250101_20260831.json",
                  "book_e2/_static/e2_corrections_20261115/compute_ledger.jsonl",
                  AREA+"/compute_ledger.jsonl",
                  "book_e2/sections/64_correcciones_metodologicas.md"})
    for path in (ROOT/AREA).rglob("*"):
        if path.is_file() and path.suffix in {".json",".csv",".md",".py",".txt"}:
            relative=path.relative_to(ROOT).as_posix()
            if relative in blobs: paths.add(relative)
    payload={}
    for relative in sorted(paths):
        path=ROOT/relative
        if path.is_symlink() or not path.is_file(): raise ValueError("Missing source: "+relative)
        raw=path.read_bytes()
        if blobs.get(relative)!=hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest():
            raise ValueError("Source differs from announced commit: "+relative)
        payload[relative]=raw
    payload.update({"run.py":RUNNER.encode(),"LEEME.md":README.encode(),
                    "ejecutar.cmd":b"@echo off\r\npy -3 run.py\r\npause\r\n",
                    "ejecutar.sh":b"#!/bin/sh\nset -eu\ncd \"$(dirname \"$0\")\"\npython3 run.py\n"})
    manifest={"version":"e2-pending-supplement-v1","source_commit":source,
              "created_at_utc":datetime.now(timezone.utc).isoformat(),
              "full_scientific_campaign_repeated":False,
              "default_scope":"stdlib validation, algebraic RQ3 deduction and readiness report",
              "files":[{"path":p,"bytes":len(b),"sha256":sha(b)} for p,b in sorted(payload.items())]}
    payload["source_manifest.json"]=(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n").encode()
    if output.exists(): raise ValueError("Use a new immutable supplement path")
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name,raw in sorted(payload.items()):
            info=zipfile.ZipInfo("SeguimientoE2/"+name,(1980,1,1,0,0,0))
            info.external_attr=(0o100755 if name.endswith(".sh") else 0o100644)<<16
            info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,raw,compresslevel=9)
    record={"file":output.name,"bytes":output.stat().st_size,"sha256":sha(output.read_bytes()),
            "source_commit":source,"included_files":len(payload),"default_command":"python run.py",
            "requires_other_deliverables_for_default":False,
            "requires_all12_v2_archives_for_optional_analytical_replay":True}
    print(json.dumps(record,ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    package(args.source_commit,args.output)
