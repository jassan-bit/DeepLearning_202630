"""Package existing E2 evidence and report sources without fitting or rescoring."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "book_e2/_static/delivery_20261112"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build(budget_evidence):
    PUBLIC.mkdir(parents=True, exist_ok=True)
    for name in ("selection_snapshot.json", "trials.csv", "source_index.json"):
        shutil.copyfile(ROOT / "experiments/e2_delivery_20261112" / name, PUBLIC / name)
    shutil.copyfile(budget_evidence, PUBLIC / "budget_reconciliation.json")
    e1 = ROOT / "results/e1_608_remediation_20261107/metric_summary.csv"
    (PUBLIC / "e1_reference_metrics.csv").write_bytes(e1.read_bytes())
    provenance = {"scope": "Historical E1 on608 already-known dates; not the sixteen-system E2 benchmark",
        "source_path": str(e1.relative_to(ROOT)), "sha256": sha(e1.read_bytes()),
        "architectures_source": "results/e1_608_remediation_20261107/architecture_summary.csv",
        "replicates_belong_to_E1_only": True, "new_training_executed": False}
    (PUBLIC / "e1_reference_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    members = {}
    def include(path, name=None):
        members[name or str(path.relative_to(ROOT))] = path.read_bytes()
    for path in sorted((ROOT / "book_e2").glob("*.md")):
        include(path)
    for path in sorted((ROOT / "book_e2/sections").glob("*.md")):
        include(path)
    for name in ("_config.yml", "_toc.yml", "references.bib"):
        include(ROOT / "book_e2" / name)
    for name in ("selection_snapshot.json", "trials.csv", "source_index.json", "budget_reconciliation.json",
                 "e1_reference_metrics.csv", "e1_reference_provenance.json"):
        include(PUBLIC / name)
    for folder in ("experiments/e2_published_architectures_20261112", "experiments/e2_protocol_amendment_20261112"):
        for path in sorted((ROOT / folder).glob("*")):
            if path.is_file():
                include(path)
    for name in ("replication_summary.json", "replication_seed_metrics.csv", "pilot_regime_inference.csv", "provenance.json"):
        include(ROOT / "book_e2/_static/paper_guide_results" / name)
    include(ROOT / "book_e2/_static/paper_guide_evidence/training_curve_summary.csv")
    include(ROOT / "book_e2/_static/optimized_progress_20261111_213112/registered_hpo_configs.json")
    include(ROOT / "results/e1_608_remediation_20261107/architecture_summary.csv")
    for path in sorted(ROOT.glob("requirements*.txt")):
        include(path)
    for name in ("export_e2_delivery_snapshot.py", "package_e2_existing_delivery.py"):
        include(ROOT / "scripts" / name)
    members["LEEME.txt"] = ("Entregable2: cierre con evidencia disponible, 12/11/2026.\n"
        "Leer book_e2/sections/47_entrega_consolidada.md y la portada.\n"
        "Sitio completo con figuras/enlaces: https://jassan-bit.github.io/DeepLearning_202630/entregable2/intro.html\n"
        "Este paquete contiene fuentes del informe y evidencia seleccionada; no es una copia offline completa del sitio.\n"
        "3507 registros historicos no equivalen a modelos finales ni a cumplimiento integral.\n"
        "No se ejecutaron nuevas corridas durante el cierre. R/k actuales permanecen pendientes.\n"
        "No contiene pesos privados; configuraciones, recibos y hashes no sustituyen esos archivos.\n"
        "Las metricas E1/ETTh1 conservan su propio alcance y no rellenan el benchmark E2.\n").encode()
    rows = [{"path": name, "bytes": len(data), "sha256": sha(data)} for name, data in sorted(members.items())]
    inventory = {"version": "e2-existing-delivery-bundle-v1", "files": rows,
        "scope": "Consolidated existing evidence and report, not a completed sixteen-system benchmark",
        "historical_records": 3507, "optimized_results": 1705,
        "new_training_executed": False, "final_benchmark_complete": False,
        "public_weights_included": False, "R": None, "k": None}
    members["MANIFIESTO.json"] = json.dumps(inventory, indent=2, ensure_ascii=False).encode() + b"\n"
    target = PUBLIC / "Entregable2_evidencia_disponible.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 12, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise ValueError("Delivery archive CRC mismatch")
        for row in rows:
            if sha(archive.read(row["path"])) != row["sha256"]:
                raise ValueError("Delivery manifest mismatch")
    manifest = {**inventory, "bundle": {"path": target.name, "bytes": target.stat().st_size,
        "sha256": sha(target.read_bytes())}}
    (PUBLIC / "delivery_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return {"files": len(rows), "bundle": manifest["bundle"], "new_training_executed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-evidence", type=Path, required=True)
    print(json.dumps(build(parser.parse_args().budget_evidence)))
