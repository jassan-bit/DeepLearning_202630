#!/usr/bin/env python3
"""Package E2 section-6 analysis tools and templates, never benchmark results.

Only own review code, explicit metadata and public analysis templates are admitted.
No training, network access or experiment is performed by this packager. The
inherited D7 readiness blockers remain separate from statistical-tool tests.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "book_e2/_static/benchmark_analysis_download"
ARCHIVE_NAME = "Benchmark_E2_analisis_plantillas.zip"
FIXED_DATE = (2026, 11, 6, 0, 0, 0)
ANALYSIS_PROPOSAL = "experiments/e2_benchmark_analysis_20261106/proposal.json"
DESIGN_PROPOSAL = "experiments/e2_benchmark_design_20261105/proposal.json"
ANALYSIS_DIR = "book_e2/_static/benchmark_analysis"
DESIGN_DIR = "book_e2/_static/benchmark_design"
CONTRACT_FILES = ("book_e2/_static/e1_reference.json", "book_e2/_static/dataset_card.json",
                  "book_e2/_static/guia_e2_recibida.json")
DESIGN_FILES = ("proposal.json", "report.json", "provenance.json", "requirements.csv",
                "model_readiness.csv", "nested_calendar.csv")
SOURCE_RECEIPTS = (
    "results/gru_lstm_tcn_planned_20261105/selected_recipes.json",
    "results/gru_lstm_tcn_planned_20261105/metric_summary.csv",
    "results/gru_lstm_tcn_planned_20261105/dimensioning.json",
    "results/gru_lstm_tcn_planned_20261105/design.json",
    "data/manifests/binance_spot_BTCUSDT_1m_20230101_20261030.json",
    "book/sections/09_estudio_planificado.md",
)
FILES = (
    "scripts/analyze_e2_benchmark.py",
    "scripts/audit_e2_benchmark_readiness.py",
    "scripts/package_e2_benchmark_analysis.py",
    "tests/test_e2_benchmark_analysis.py",
    "tests/test_e2_benchmark_readiness.py",
    "requirements-benchmark-analysis.lock.txt",
    "book_e2/sections/06_benchmark_final.md",
    "book_e2/references.bib",
    ANALYSIS_PROPOSAL,
    DESIGN_PROPOSAL,
    *CONTRACT_FILES,
    *(f"{DESIGN_DIR}/{name}" for name in DESIGN_FILES),
    *SOURCE_RECEIPTS,
)
PUBLIC_SUFFIXES = {".json", ".csv", ".md", ".png", ".svg", ".txt"}
PUBLIC_METADATA_CSV_NAMES = {"requirements.csv"}
ANALYSIS_FILES = {"report.json", "table_template.csv", "requirements.csv", "predictions_template.csv",
                  "runs_template.csv", "costs_template.csv", "schemas.json", "proposal.json",
                  "provenance.json", "README.md"}
TABLE_FIELDS = ["model", "role", "status", "eligible_origins", "runs", "RMSE_mean", "RMSE_SD",
                "RMSE_CI95_low", "RMSE_CI95_high", "MAE_mean", "MSE_mean", "R2_mean", "MAPE_percent",
                "MAPE_coverage", "SMAPE_percent", "SMAPE_coverage", "MASE_mean", "training_seconds",
                "latency_ms_per_window", "peak_memory_bytes", "parameters", "FLOPs_per_window", "scope"]
TABLE_METADATA_FIELDS = {"model", "role", "status", "scope"}
PRIVATE_PATH = re.compile(r"(?:/workspace/|/mnt/data/|/tmp/|/home/|sediment://|file://)")
PREDICTION_COLUMNS = {"y_true", "y_pred", "prediction", "predictions", "target", "targets", "close"}
README = """# Entregable 2, sección 6: herramientas y plantillas de análisis

Suplemento del repositorio https://github.com/jassan-bit/DeepLearning_202630.
Este paquete contiene el capítulo 6, la propuesta de análisis estadístico, su
código y pruebas, plantillas públicas y el auditor documental del diseño D7.
No contiene resultados finales del benchmark Bitcoin, nuevas predicciones,
pesos, datos originales ni un pipeline de entrenamiento.

## Estado y límites

**Archivo v1:** las quince filas y bloqueos de este suplemento describen la propuesta inicial. El protocolo vigente de veinte configuraciones, guía completa y P0/OOF registrado está en `experiments/e2_remediation_20261106/` y en el paquete Remediacion_E2_protocolo_codigo_evidencias.zip. Sus contadores no son el total de experimentos posteriores.

La propuesta permanece abierta. Las pruebas verifican funciones estadísticas e
integridad con ejemplos sintéticos; no prueban que los quince comparadores se
hayan entrenado. Continúan los prerrequisitos del diseño D7, incluidos los
adaptadores, la auditoría de fuga, el piloto propio y la potencia. Diez modelos
nuevos se añaden a GRU/LSTM/TCN y dos referencias: al menos 15 configuraciones.
Los resultados ETTh1 de la sección 5 no son resultados Bitcoin ni se reutilizan
en estas plantillas como si lo fueran.

El contrato heredado conserva RMSE primaria, la unidad temporal dependiente, la
partición y el margen absoluto de E1. El test histórico ya conocido no se
renombra como una prueba confirmatoria desconocida. La propuesta de análisis
explicita comparaciones, tamaños de efecto, remuestreo y multiplicidad, sin
convertir una declaración posterior en un diseño previo al test histórico.

## Revisar y ejecutar las pruebas sin entrenar

Extraer conservando la estructura de directorios. El analizador utiliza NumPy y
SciPy; requirements-benchmark-analysis.lock.txt fija sus versiones observadas y
no instala un framework de entrenamiento. En Python3.12:

    python -m pip install -r requirements-benchmark-analysis.lock.txt

    python scripts/analyze_e2_benchmark.py
    python scripts/analyze_e2_benchmark.py --help
    python tests/test_e2_benchmark_analysis.py
    python tests/test_e2_benchmark_readiness.py
    python scripts/audit_e2_benchmark_readiness.py
    python scripts/audit_e2_benchmark_readiness.py --require-ready

--require-ready devuelve código 2 para una propuesta documental válida con
bloqueos. Es un resultado esperado, no autorización para entrenar. Un contrato
incoherente o un hash E1 alterado devuelve código 1. El auditor solo lee metadatos
y sus seis fuentes públicas fijadas; no abre archivos de mercado ni pesos.

La orden sin argumentos solo regenera los diez derivados documentales bajo
book_e2/_static/benchmark_analysis; usa biblioteca estándar y no puntúa modelos.
Su table_template.csv enumera quince modelos pendientes y requirements.csv
registra requisitos. Los CSV de predicciones, ejecuciones y costos se entregan
vacíos, con encabezados; no se rellenan con resultados ficticios. schemas.json
define sus columnas y unidades. Las pruebas sintéticas permanecen en tests.

La CLI opcional --panel, --runs, --costs, --calendar, --scope synthetic_fixture y
--output-dir admite verificar la coherencia de un panel sintético y sus hashes
en una carpeta separada. Ese modo no certifica causalidad, potencia, suficiencia
del presupuesto o desempeño científico, ni produce rankings Bitcoin reales.
El modo real permanece bloqueado por la propuesta no congelada. --help describe
los insumos para cuando existan ejecuciones válidas; no declara que existan hoy.

## Integridad, reconstrucción y procedencia

Se incluyen los tres JSON del contrato, la propuesta D7 y sus seis derivados de
auditoría, además de seis fuentes públicas E1 identificadas por SHA-256. Son
recetas, resumen agregado, diseño/dimensionamiento, manifiesto y documentación;
el manifiesto de datos no contiene las observaciones de mercado.

book_e2/references.bib permite resolver las citas del capítulo, incluidas las
referencias metodológicas de los contrastes. Esas referencias de métodos no se
cuentan como nuevos estudios aplicados del corpus ni acreditan consultas en
bases bibliográficas. No se redistribuyen sus PDF ni se incluye un entorno de
construcción completo del Jupyter Book; este paquete permite revisión y ejecución
de las herramientas y pruebas indicadas.

MANIFIESTO.json identifica cada archivo incluido por tamaño y SHA-256. El ZIP
tiene fechas internas fijas al 06/11/2026. Desde la carpeta extraída, reconstruir
el contenedor y comprobar su correspondencia exacta con esas fuentes:

    python scripts/package_e2_benchmark_analysis.py
    python scripts/package_e2_benchmark_analysis.py --check

El contenedor no se incluye a sí mismo. SHA256SUMS, publicado junto al ZIP,
permite contrastar además el archivo completo. Ninguna de estas órdenes entrena,
descarga datos o sustituye las evidencias experimentales todavía pendientes.
"""


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def read_input(root, name):
    relative = Path(name)
    path = root / relative
    if relative.is_absolute() or ".." in relative.parts or path.is_symlink():
        raise ValueError("Only explicit regular repository-relative inputs are permitted")
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Publication input is missing or outside the root: {name}")
    return path.read_bytes()


def check_public_json(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {"y_true", "y_pred", "predictions", "targets"} and isinstance(item, list) and item:
                raise ValueError("Individual forecast/target arrays cannot be published in this template package")
            check_public_json(item)
    elif isinstance(value, list):
        for item in value:
            check_public_json(item)
    elif isinstance(value, str) and PRIVATE_PATH.search(value):
        raise ValueError("Private filesystem path or uploaded-file pointer in public material")


def check_public_csv(name, payload):
    text = payload.decode("utf-8-sig")
    if PRIVATE_PATH.search(text):
        raise ValueError("Private filesystem path in CSV metadata")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError(f"Public CSV lacks a schema: {name}")
    rows = list(reader)
    if (None in reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames)
            or any(None in row or any(value is None for value in row.values()) for row in rows)):
        raise ValueError(f"Malformed public CSV: {name}")
    if PREDICTION_COLUMNS.intersection(field.lower() for field in reader.fieldnames) and rows:
        raise ValueError("Prediction/target template CSV must contain headers only")
    if (rows and Path(name).name not in PUBLIC_METADATA_CSV_NAMES
            and "template" not in Path(name).stem.lower() and "schema" not in Path(name).stem.lower()):
        raise ValueError("Nonempty CSV requires an explicitly named template/schema in this results-free package")


def csv_table(payload):
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    return reader.fieldnames, list(reader)


def reject_ranking_payload(value):
    """Allow status flags and method descriptions, never a published rank table."""
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = key.lower().replace("-", "_")
            rank_key = normalized in {"rank", "ranks", "ranking", "rankings", "model_rank",
                                      "model_ranks", "model_ranking", "model_rankings", "descriptive_ranks"}
            if rank_key and ((isinstance(item, (list, dict)) and item)
                             or (isinstance(item, (int, float)) and not isinstance(item, bool))):
                raise ValueError("A numerical/synthetic ranking payload cannot be published in section6 templates")
            reject_ranking_payload(item)
    elif isinstance(value, list):
        for item in value:
            reject_ranking_payload(item)


def validate_results_free_snapshot(contents):
    """Require blank measures and factual non-execution flags, not file labels.

    Public section6 permits exactly the ten documentary outputs. A synthetic
    fixture validation report or numerical ranking cannot become public merely
    by being renamed as a template. E1 aggregate receipts remain separate.
    """
    original = json.loads(contents[ANALYSIS_PROPOSAL])
    copied = json.loads(contents[f"{ANALYSIS_DIR}/proposal.json"])
    if original != copied or original.get("status") != "proposed_not_frozen" or original.get("new_Bitcoin_results") != 0:
        raise ValueError("The analysis proposal must remain identical, not frozen and without new Bitcoin results")
    report = json.loads(contents[f"{ANALYSIS_DIR}/report.json"])
    reject_ranking_payload(report)
    if report.get("status") != "proposed_not_frozen":
        raise ValueError("The public analysis report is no longer a proposal")
    for field in ("benchmark_executed", "real_predictions_loaded", "numerical_rankings_published",
                  "actual_test_score_table_available"):
        if report.get(field) is not False:
            raise ValueError(f"Section6 must explicitly retain false non-execution/result flags: {field}")
    for field in ("new_Bitcoin_results", "D7_models_fitted_by_this_tool"):
        if type(report.get(field)) is not int or report[field] != 0:
            raise ValueError(f"Section6 cannot contain new results or fitted models: {field}")
    if report.get("external_ETTh1_section5_not_Bitcoin_results") is not True:
        raise ValueError("External ETTh1 evidence must remain distinct from Bitcoin results")

    model_source = json.loads(contents[DESIGN_PROPOSAL])["models"]
    models = {row["id"]: row for row in model_source}
    if len(models) != 15 or len(model_source) != 15 or report.get("benchmark_models") != 15:
        raise ValueError("Expected exactly fifteen distinct pending comparators")
    fields, rows = csv_table(contents[f"{ANALYSIS_DIR}/table_template.csv"])
    if fields != TABLE_FIELDS or len(rows) != 15 or {row.get("model") for row in rows} != set(models):
        raise ValueError("The pending table must retain the exact schema and fifteen-comparator registry")
    for row in rows:
        if row["role"] != models[row["model"]]["role"]:
            raise ValueError("A table row changed the inherited/proposed comparator role")
        if row["status"] != "pending_not_trained_in_D7_benchmark":
            raise ValueError("Every public table row must explicitly remain pending")
        if row["scope"] != "D7 benchmark template; not external ETTh1 replica results":
            raise ValueError("The pending table cannot reuse external results or a synthetic ranking")
        for field in set(TABLE_FIELDS) - TABLE_METADATA_FIELDS:
            if row[field].strip().lower() not in {"", "null"}:
                raise ValueError(f"Public pending table contains a measure/count/rank instead of an empty cell: {field}")

    schemas = json.loads(contents[f"{ANALYSIS_DIR}/schemas.json"])
    for filename, key in (("predictions_template.csv", "prediction_panel"),
                          ("runs_template.csv", "runs"), ("costs_template.csv", "costs")):
        fields, rows = csv_table(contents[f"{ANALYSIS_DIR}/{filename}"])
        if fields != schemas[key]["fields"] or rows:
            raise ValueError(f"Input template must match its declared schema and contain no observation rows: {filename}")
    fields, rows = csv_table(contents[f"{ANALYSIS_DIR}/requirements.csv"])
    if fields != ["id", "requirement", "status"] or not rows or any(
            row["status"] != "pending_benchmark_execution_and_protocol_closure" for row in rows):
        raise ValueError("The requirement registry must remain documentary and pending")

    provenance = json.loads(contents[f"{ANALYSIS_DIR}/provenance.json"])
    generator = "scripts/analyze_e2_benchmark.py"
    if provenance.get("generator") != generator or provenance.get("generator_sha256") != sha256(contents[generator]).hexdigest():
        raise ValueError("The template snapshot is stale relative to the analyzer")
    for field in ("market_files_opened", "real_benchmark_predictions_loaded", "model_fits"):
        if type(provenance.get(field)) is not int or provenance[field] != 0:
            raise ValueError("The public template generator must not report actual data access or model fitting")
    for name, checksum in provenance["input_hashes"].items():
        if name not in contents or sha256(contents[name]).hexdigest() != checksum:
            raise ValueError(f"Template input changed after its generation: {name}")
    if set(provenance["output_hashes"]) != ANALYSIS_FILES - {"provenance.json"}:
        raise ValueError("The template provenance does not cover exactly the nine documentary outputs")
    for name, checksum in provenance["output_hashes"].items():
        if sha256(contents[f"{ANALYSIS_DIR}/{name}"]).hexdigest() != checksum:
            raise ValueError(f"Template output hash mismatch: {name}")


def source_contents(root=ROOT):
    contents = {name: read_input(root, name) for name in FILES}
    directory = root / ANALYSIS_DIR
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Public benchmark-analysis templates are not yet available")
    paths = sorted(path for path in directory.rglob("*") if path.is_file())
    if {path.relative_to(directory).as_posix() for path in paths} != ANALYSIS_FILES:
        raise ValueError("The public section6 folder must contain exactly the ten documentary outputs; no fixture results/rankings")
    for path in paths:
        name = path.relative_to(root).as_posix()
        if path.name != "SHA256SUMS" and path.suffix not in PUBLIC_SUFFIXES:
            raise ValueError(f"Non-public payload extension: {name}")
        payload = read_input(root, name)
        if path.suffix == ".json":
            check_public_json(json.loads(payload))
        elif path.suffix == ".csv":
            check_public_csv(name, payload)
        elif path.suffix in {".md", ".txt", ".svg"} and PRIVATE_PATH.search(payload.decode("utf-8")):
            raise ValueError("Private filesystem path in public template material")
        contents[name] = payload
    proposal = json.loads(contents[DESIGN_PROPOSAL])
    if set(proposal["e1_source_receipts"]) != set(SOURCE_RECEIPTS):
        raise ValueError("The six allowlisted inherited E1 receipts changed")
    for name, checksum in proposal["e1_source_receipts"].items():
        if sha256(contents[name]).hexdigest() != checksum:
            raise ValueError(f"Pinned E1 receipt mismatch: {name}")
    report = json.loads(contents[f"{DESIGN_DIR}/report.json"])
    if report["ready_to_train"] or report["new_deep_learning_fits"] != 0 or not report["blockers"]:
        raise ValueError("This template package must preserve the blocked, unexecuted D7 design")
    validate_results_free_snapshot(contents)
    contents["LEEME.md"] = README.encode("utf-8")
    return contents, report


def validate_archive(payload, manifest_bytes):
    manifest = json.loads(manifest_bytes)
    with zipfile.ZipFile(io.BytesIO(payload)) as stream:
        if stream.testzip() is not None:
            raise ValueError("ZIP CRC verification failed")
        expected = set(manifest["files"]) | {"MANIFIESTO.json"}
        if set(stream.namelist()) != expected or len(stream.namelist()) != len(expected):
            raise ValueError("ZIP differs from the manifest or contains duplicate entries")
        if stream.read("MANIFIESTO.json") != manifest_bytes:
            raise ValueError("Embedded and published manifests differ")
        for name, record in manifest["files"].items():
            if name not in FILES and name != "LEEME.md" and not name.startswith(ANALYSIS_DIR + "/"):
                raise ValueError("ZIP file is outside the explicit publication roots")
            data = stream.read(name)
            if len(data) != record["bytes"] or sha256(data).hexdigest() != record["sha256"]:
                raise ValueError(f"ZIP size/SHA-256 mismatch: {name}")
        if any(info.date_time != FIXED_DATE for info in stream.infolist()):
            raise ValueError("ZIP timestamps are not fixed")


def build_payload(root=ROOT):
    contents, report = source_contents(root)
    manifest = {
        "schema_version": 1,
        "client_date_colombia": "2026-11-06",
        "scope": "E2 section6 own statistical tools and empty templates; no final Bitcoin benchmark results",
        "analysis_proposal_repository_path": ANALYSIS_PROPOSAL,
        "analysis_proposal_sha256": sha256(contents[ANALYSIS_PROPOSAL]).hexdigest(),
        "D7_design_ready_to_train": False,
        "D7_design_blocking_prerequisites": len(report["blockers"]),
        "new_deep_learning_fits": 0,
        "new_Bitcoin_results": 0,
        "ETTh1_results_used_as_Bitcoin_results": False,
        "raw_market_data_included": False,
        "model_weights_included": False,
        "individual_predictions_included": False,
        "inherited_E1_aggregates_included_as_receipts_only": True,
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
    archive = buffer.getvalue()
    validate_archive(archive, manifest_bytes)
    return {ARCHIVE_NAME: archive, "MANIFIESTO.json": manifest_bytes, "LEEME.md": contents["LEEME.md"],
            "SHA256SUMS": f"{sha256(archive).hexdigest()}  {ARCHIVE_NAME}\n".encode()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify archive matches the current public inputs")
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
    manifest = json.loads(payload["MANIFIESTO.json"])
    print(json.dumps({"archive": (OUTPUT / ARCHIVE_NAME).relative_to(ROOT).as_posix(),
                      "verified": True, "mode": "check" if args.check else "build",
                      "entries": len(manifest["files"]) + 1, "bytes": len(payload[ARCHIVE_NAME]),
                      "sha256": sha256(payload[ARCHIVE_NAME]).hexdigest()}))


if __name__ == "__main__":
    main()
