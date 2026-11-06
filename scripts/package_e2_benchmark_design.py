#!/usr/bin/env python3
"""Package the documentary E2 section-4 proposal, without any training assets.

The archive is self-contained for the standard-library metadata auditor/tests.
It is a proposed, blocked design, not an executable DL benchmark or a frozen
preregistration. Only explicit public metadata paths are admitted.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "book_e2/_static/benchmark_design_download"
ARCHIVE_NAME = "Diseno_E2_propuesta_auditoria.zip"
FIXED_DATE = (2026, 11, 5, 0, 0, 0)
PROPOSAL = "experiments/e2_benchmark_design_20261105/proposal.json"
AUDIT_DIR = "book_e2/_static/benchmark_design"
AUDIT_FILES = ("proposal.json", "report.json", "provenance.json", "requirements.csv",
               "model_readiness.csv", "nested_calendar.csv")
CONTRACT_FILES = ("book_e2/_static/e1_reference.json", "book_e2/_static/dataset_card.json",
                  "book_e2/_static/guia_e2_recibida.json")
SOURCE_RECEIPTS = (
    "results/gru_lstm_tcn_planned_20261105/selected_recipes.json",
    "results/gru_lstm_tcn_planned_20261105/metric_summary.csv",
    "results/gru_lstm_tcn_planned_20261105/dimensioning.json",
    "results/gru_lstm_tcn_planned_20261105/design.json",
    "data/manifests/binance_spot_BTCUSDT_1m_20230101_20261030.json",
    "book/sections/09_estudio_planificado.md",
)
FILES = (
    PROPOSAL,
    "scripts/audit_e2_benchmark_readiness.py",
    "scripts/package_e2_benchmark_design.py",
    "tests/test_e2_benchmark_readiness.py",
    "book_e2/sections/04_implementacion.md",
    *CONTRACT_FILES,
    *SOURCE_RECEIPTS,
    *(f"{AUDIT_DIR}/{name}" for name in AUDIT_FILES),
)
README_TEMPLATE = """# Entregable 2, sección 4: propuesta de diseño y auditoría documental

Paquete autocontenido del capítulo 4 de
https://github.com/jassan-bit/DeepLearning_202630.
Extraer conservando todos los directorios. No exige datos de mercado, paquetes de
Deep Learning, acceso de red ni una copia adicional del repositorio para ejecutar
el auditor y sus pruebas; utiliza Python 3.12 y la biblioteca estándar.

## Estado de esta entrega

Estado: proposed_not_frozen. Hay {blockers} prerrequisitos bloqueantes y cero
ajustes de Deep Learning nuevos. La propuesta no está lista para entrenar ni
acredita un test confirmatorio nuevo. No contiene adaptadores D7 validados,
pipeline de entrenamiento, pesos ni predicciones individuales. El test de E1 ya
era conocido; esta cronología se conserva.

La propuesta distingue diez modelos nuevos de las tres recetas E1 fijas, una
referencia no profunda y una trivial: al menos 15 configuraciones. Ocho candidatos
de hiperparámetros por modelo no equivalen a ocho entrenamientos. La contabilidad
propuesta es de {evaluations} evaluaciones de validación y {fits} ajustes del
optimizador por modelo nuevo, antes del piloto y del ajuste definitivo; para los
diez modelos nuevos son {all_fits} ajustes propuestos, no ejecutados. El presupuesto
común de cómputo, los adaptadores, el piloto, la potencia y el cierre del protocolo
siguen pendientes. No se atribuye HPO nuevo a las tres anclas E1 sin volver a
ajustarlas y registrar esas variantes por separado.

El calendario anidado contiene 30 filas de fechas propuestas, no índices tsxv ni
ventanas supervisadas auditadas. Se conserva el embargo, la escala de la métrica,
el margen absoluto y las demás condiciones heredadas de E1.

## Ejecutar la auditoría y las pruebas

Desde la carpeta extraída:

    python scripts/audit_e2_benchmark_readiness.py
    python tests/test_e2_benchmark_readiness.py
    python scripts/audit_e2_benchmark_readiness.py --require-ready

La primera orden valida la coherencia documental y escribe seis derivados
JSON/CSV en book_e2/_static/benchmark_design. Su éxito no significa que se pueda
entrenar. La opción --require-ready devuelve código 2 mientras existen bloqueos;
es el resultado esperado para esta propuesta válida y bloqueada. Una incoherencia
del contrato o un hash alterado devuelve código 1. El auditor no importa librerías
de DL, no abre archivos de mercado y no entrena ningún modelo.

Las secciones de la guía pendientes de recibir se identifican en el capítulo y en
la propuesta. Este paquete permite revisar lo ya recibido, sin inventar el
contenido restante ni declarar completada toda la guía de E2.

## Verificar y regenerar el paquete

MANIFIESTO.json contiene tamaño y SHA-256 de cada archivo. Se incluyen seis
fuentes públicas de E1 identificadas por los recibos fijados en proposal.json:
recetas, resumen agregado, dimensionamiento, diseño, manifiesto de datos y
capítulo metodológico. El manifiesto de datos describe archivos; no contiene los
datos de mercado. No se incluyen archivos crudos, pesos ni predicciones individuales.

    python scripts/package_e2_benchmark_design.py --check
    python scripts/package_e2_benchmark_design.py

--check comprueba CRC, inventario, hashes, fechas internas fijas y correspondencia
exacta con las fuentes presentes. Si se ha vuelto a ejecutar el auditor con otra
versión de Python, su observación de entorno puede cambiar; regenerar el paquete
después de esa actualización. El mismo conjunto de archivos genera el mismo ZIP.
SHA256SUMS, publicado junto al ZIP, identifica el contenedor completo.

Las fuentes públicas E1 se conservan como evidencia heredada y no se seleccionan
modelos con sus resultados. Su inclusión no constituye una repetición del
experimento ni transforma el test histórico en un test desconocido.
"""


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def file_bytes(root, name):
    path = root / name
    if Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink():
        raise ValueError("Only explicit repository-relative regular files are permitted")
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Publication input is missing or outside the root: {name}")
    return path.read_bytes()


def verify_source_snapshot(contents):
    """Do not package a receipt mismatch or stale documentary audit as current."""
    proposal = json.loads(contents[PROPOSAL])
    receipts = proposal["e1_source_receipts"]
    if set(receipts) != set(SOURCE_RECEIPTS):
        raise ValueError("The six allowlisted inherited E1 source receipts differ")
    for name, checksum in receipts.items():
        if sha256(contents[name]).hexdigest() != checksum:
            raise ValueError(f"Pinned E1 receipt mismatch: {name}")
    provenance = json.loads(contents[f"{AUDIT_DIR}/provenance.json"])
    if sha256(contents["scripts/audit_e2_benchmark_readiness.py"]).hexdigest() != provenance["generator_sha256"]:
        raise ValueError("Audit outputs are stale relative to the auditor")
    for name, checksum in provenance["input_hashes"].items():
        if name not in contents or sha256(contents[name]).hexdigest() != checksum:
            raise ValueError(f"Audit input changed after generation: {name}")
    for name, checksum in provenance["output_hashes"].items():
        if sha256(contents[f"{AUDIT_DIR}/{name}"]).hexdigest() != checksum:
            raise ValueError(f"Audit output hash mismatch: {name}")
    if provenance["source_E1_receipts"] != receipts:
        raise ValueError("Audit provenance differs from pinned E1 receipts")
    report = json.loads(contents[f"{AUDIT_DIR}/report.json"])
    if not report["proposal_valid"] or report["ready_to_train"] or report["new_deep_learning_fits"] != 0:
        raise ValueError("This package is scoped to the valid, blocked proposal with zero fits")
    if proposal["status"] != "proposed_not_frozen" or not report["blockers"]:
        raise ValueError("The proposed, not frozen state or unresolved prerequisites changed")
    return proposal, report


def validate_archive(payload, manifest_bytes):
    manifest = json.loads(manifest_bytes)
    with zipfile.ZipFile(io.BytesIO(payload)) as stream:
        if stream.testzip() is not None:
            raise ValueError("ZIP CRC verification failed")
        expected = set(FILES) | {"LEEME.md", "MANIFIESTO.json"}
        if set(stream.namelist()) != expected or len(stream.namelist()) != len(expected):
            raise ValueError("ZIP contains missing, duplicate or non-allowlisted files")
        if stream.read("MANIFIESTO.json") != manifest_bytes:
            raise ValueError("Embedded and published manifests differ")
        if set(manifest["files"]) != expected - {"MANIFIESTO.json"}:
            raise ValueError("Manifest does not cover the allowlisted files")
        for name, record in manifest["files"].items():
            data = stream.read(name)
            if len(data) != record["bytes"] or sha256(data).hexdigest() != record["sha256"]:
                raise ValueError(f"ZIP size or SHA-256 mismatch: {name}")
        if any(info.date_time != FIXED_DATE for info in stream.infolist()):
            raise ValueError("ZIP timestamp is not fixed")


def build_payload(root=ROOT):
    contents = {name: file_bytes(root, name) for name in FILES}
    proposal, report = verify_source_snapshot(contents)
    fits = report["proposed_optimizer_fit_count_inventory"]
    evaluations = report["proposed_validation_evaluation_count_inventory"]
    readme = README_TEMPLATE.format(
        blockers=len(report["blockers"]),
        evaluations=evaluations["subtotal_per_tuned_family_before_pilot_and_final"],
        fits=fits["subtotal_optimizer_fits_per_tuned_family_before_pilot_and_final"],
        all_fits=fits["top10_subtotal_optimizer_fits_before_pilot_and_final"],
    ).encode("utf-8")
    contents["LEEME.md"] = readme
    manifest = {
        "schema_version": 1,
        "administrative_date_colombia": "2026-11-05",
        "scope": "E2 section4 documentary proposal and audit; not a training pipeline",
        "proposal_status": proposal["status"],
        "proposal_valid": report["proposal_valid"],
        "ready_to_train": False,
        "ready_to_claim_confirmatory_test": False,
        "blocking_prerequisites": len(report["blockers"]),
        "new_deep_learning_fits": 0,
        "new_Bitcoin_results": 0,
        "raw_market_files_included": False,
        "model_weights_included": False,
        "individual_predictions_included": False,
        "inherited_source_commit": proposal["inherited_scientific_contract"]["source_commit"],
        "pinned_E1_source_receipts": proposal["e1_source_receipts"],
        "files": {name: {"bytes": len(data), "sha256": sha256(data).hexdigest()}
                  for name, data in sorted(contents.items())},
    }
    manifest_bytes = json_bytes(manifest)
    contents["MANIFIESTO.json"] = manifest_bytes
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=FIXED_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            stream.writestr(info, data)
    archive_bytes = buffer.getvalue()
    validate_archive(archive_bytes, manifest_bytes)
    return {ARCHIVE_NAME: archive_bytes, "MANIFIESTO.json": manifest_bytes, "LEEME.md": readme,
            "SHA256SUMS": f"{sha256(archive_bytes).hexdigest()}  {ARCHIVE_NAME}\n".encode()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the package exactly matches current inputs")
    args = parser.parse_args(argv)
    payload = build_payload()
    if args.check:
        for name, data in payload.items():
            path = OUTPUT / name
            if not path.is_file() or path.read_bytes() != data:
                raise SystemExit(f"Package is stale or missing: {path.relative_to(ROOT)}")
    else:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for name, data in payload.items():
            (OUTPUT / name).write_bytes(data)
    print(json.dumps({"archive": (OUTPUT / ARCHIVE_NAME).relative_to(ROOT).as_posix(),
                      "verified": True, "mode": "check" if args.check else "build",
                      "entries": len(FILES) + 2, "bytes": len(payload[ARCHIVE_NAME]),
                      "sha256": sha256(payload[ARCHIVE_NAME]).hexdigest()}))


if __name__ == "__main__":
    main()
