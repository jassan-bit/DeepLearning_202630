#!/usr/bin/env python3
"""Build a deterministic public E2 literature-review supplement.

Only allowlisted repository derivatives and review scripts are packaged. Native
WoS/Scopus exports, full abstracts/references, PDFs and downloaded author code
remain outside this publication. Re-run after editorial updates; --check verifies
that the delivered archive exactly matches the present repository materials.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "book_e2/_static/literature_delivery"
ARCHIVE_NAME = "Literatura_E2_codigo_matrices_diagramas.zip"
FIXED_DATE = (2026, 11, 5, 0, 0, 0)
FILES = (
    "scripts/import_e2_wos.py",
    "scripts/import_e2_scopus.py",
    "scripts/compile_e2_fulltext_review.py",
    "scripts/build_e2_literature_selection.py",
    "scripts/package_e2_literature.py",
    "tests/test_e2_wos_import.py",
    "tests/test_e2_scopus_import.py",
    "tests/test_e2_fulltext_review.py",
    "requirements-eda.lock.txt",
    "book_e2/sections/03_literatura.md",
    "book_e2/references.bib",
    "book_e2/_static/literature_scopus/original_model_records.csv",
)
DIRECTORIES = (
    "references/literature_e2",
    "book_e2/_static/literature_selection",
    "book_e2/_static/literature_fulltext",
)
ALLOWED_SUFFIXES = {".py", ".json", ".csv", ".bib", ".md", ".png", ".svg", ".txt"}
README = """# Literatura del Entregable 2: código, matrices y diagramas

Suplemento descargable del capítulo 3. Conserva la estructura de directorios del
repositorio https://github.com/jassan-bit/DeepLearning_202630 y permite auditar
la selección bibliográfica, las cifras publicadas y sus localizadores.

Contiene derivados bibliográficos mínimos de dos fuentes recibidas (WoS y
Scopus), la selección propuesta de diez arquitecturas, diez diagramas propios,
las fichas y matrices extraídas de diez PDF completos (215 páginas), bibliografía,
recibos SHA-256 y los scripts de importación, compilación y empaquetado.

Alcance: 8.802 ocurrencias recibidas producen 7.833 grupos bibliográficos después
de deduplicar. El CSV Scopus recibido contiene 7.500 de los 7.874 resultados
mostrados; los 374 no exportados no son exclusiones. La lectura dirigida de los
diez originales no equivale al cribado completo de esos 7.833 grupos. Las cifras
MSE/MAE publicadas mantienen dataset, variante, horizonte, contexto, página y
tabla; no forman un ranking comparable de Bitcoin.

Los diez modelos son adicionales a GRU, LSTM y TCN heredados, una referencia no
profunda y una trivial: el benchmark completo requiere al menos 15
configuraciones. Este paquete no entrena modelos ni contiene nuevos resultados
de Bitcoin, implementaciones D7 ya validadas o reproducciones numéricas.

## Verificar y reconstruir los derivados congelados

Extraer conservando directorios sobre una copia del repositorio. El compilador
de lectura completa y los tests utilizan Python 3.12 y biblioteca estándar:

    python scripts/compile_e2_fulltext_review.py --check
    python tests/test_e2_wos_import.py
    python tests/test_e2_scopus_import.py
    python tests/test_e2_fulltext_review.py

Para regenerar las fichas y los diagramas, crear un entorno con las dependencias
de requirements-eda.lock.txt (incluye matplotlib) y ejecutar:

    python scripts/build_e2_literature_selection.py
    python scripts/package_e2_literature.py
    python scripts/package_e2_literature.py --check

Los importadores de WoS y Scopus están incluidos con sus tests sintéticos y su
interfaz --help. Volver a importar los datos nativos exige los archivos originales
obtenidos mediante acceso autorizado; no se incluyen en este ZIP. Los derivados
versionados permiten auditar la recepción, la vinculación y el recuento sin
redistribuir exportaciones completas de las bases.

MANIFIESTO.json identifica cada archivo incluido por tamaño y SHA-256. El archivo
SHA256SUMS publicado junto al ZIP permite verificar además el contenedor. Las
fechas internas del ZIP son fijas para permitir regeneración determinista.

## Procedencia y derechos

Las fuentes bibliográficas son WoS, Scopus, los originales suministrados por el
usuario y los repositorios públicos identificados en sus recibos. Se incluyen
paráfrasis propias, metadatos mínimos, métricas y localizadores; no se incluyen
PDF, resúmenes completos, listas CR, direcciones, exportaciones nativas ni copias
del código de autores. Consultar los originales mediante sus DOI/enlaces y sus
condiciones de acceso. Una URL pública no implica permiso de reutilización.

El registro de código conserva el estado histórico de esa etapa; la matriz
fulltext_20261105 documenta la revisión actual de los diez originales. NR
significa no reportado en el PDF y alcance examinados; NV significa no verificado.
No se equiparan ausencia de reporte y cero, ni código disponible y reproducción.
"""


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def source_files(root=ROOT):
    """Fail closed if an unexpected extension or path enters an allowed folder."""
    paths = {root / name for name in FILES}
    for folder in DIRECTORIES:
        directory = root / folder
        if not directory.is_dir():
            raise ValueError(f"Missing review directory: {folder}")
        paths.update(path for path in directory.rglob("*") if path.is_file())
    for path in sorted(paths):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Publication source must remain inside the repository")
        if not path.is_file():
            raise ValueError(f"Missing review material: {path.relative_to(root)}")
        if path.name != "SHA256SUMS" and path.suffix not in ALLOWED_SUFFIXES:
            raise ValueError(f"Unexpected publication extension: {path.relative_to(root)}")
        if path.name.startswith(("savedrecs", "export_")) or "__pycache__" in path.parts:
            raise ValueError("Native source exports and caches are not publication inputs")
    return sorted(paths)


def build_payload(root=ROOT):
    contents = {"LEEME.md": README.encode("utf-8")}
    for path in source_files(root):
        contents[path.relative_to(root).as_posix()] = path.read_bytes()
    manifest = {
        "schema_version": 1,
        "administrative_date": "2026-11-05",
        "scope": "E2 literature review; frozen derivatives and review code; repository supplement",
        "native_sources_received": ["Web of Science", "Scopus"],
        "native_occurrences_received": 8802,
        "deduplicated_bibliographic_groups": 7833,
        "scopus_records_received": 7500,
        "scopus_results_displayed": 7874,
        "scopus_records_not_exported": 374,
        "full_original_PDFs_reviewed": 10,
        "physical_PDF_pages_reviewed": 215,
        "proposed_additional_deep_learning_models": 10,
        "minimum_total_benchmark_configurations": 15,
        "global_screening_complete": False,
        "new_deep_learning_fits": 0,
        "numerical_reproductions": 0,
        "new_Bitcoin_results": 0,
        "native_exports_included": False,
        "full_abstracts_references_or_PDFs_included": False,
        "downloaded_author_code_copies_included": False,
        "files": {
            name: {"bytes": len(payload), "sha256": sha256(payload).hexdigest()}
            for name, payload in sorted(contents.items())
        },
    }
    manifest_bytes = json_bytes(manifest)
    contents["MANIFIESTO.json"] = manifest_bytes
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        for name, payload in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=FIXED_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            stream.writestr(info, payload)
    archive_bytes = buffer.getvalue()
    validate_archive(archive_bytes, manifest_bytes)
    checksum = f"{sha256(archive_bytes).hexdigest()}  {ARCHIVE_NAME}\n"
    return {
        ARCHIVE_NAME: archive_bytes,
        "MANIFIESTO.json": manifest_bytes,
        "LEEME.md": contents["LEEME.md"],
        "SHA256SUMS": checksum.encode(),
    }


def validate_archive(payload, manifest_bytes):
    manifest = json.loads(manifest_bytes)
    with zipfile.ZipFile(io.BytesIO(payload)) as stream:
        if stream.testzip() is not None:
            raise ValueError("Archive CRC verification failed")
        expected = set(manifest["files"]) | {"MANIFIESTO.json"}
        if set(stream.namelist()) != expected or len(stream.namelist()) != len(expected):
            raise ValueError("Archive content differs from the manifest")
        if stream.read("MANIFIESTO.json") != manifest_bytes:
            raise ValueError("Embedded manifest differs")
        for name, record in manifest["files"].items():
            data = stream.read(name)
            if len(data) != record["bytes"] or sha256(data).hexdigest() != record["sha256"]:
                raise ValueError(f"Archive SHA-256 mismatch: {name}")
        if any(info.date_time != FIXED_DATE for info in stream.infolist()):
            raise ValueError("Non-deterministic ZIP timestamps")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the existing package matches current sources")
    args = parser.parse_args(argv)
    payload = build_payload()
    if args.check:
        for name, content in payload.items():
            path = OUTPUT / name
            if not path.is_file() or path.read_bytes() != content:
                raise SystemExit(f"Package is stale or missing: {path.relative_to(ROOT)}")
    else:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for name, content in payload.items():
            (OUTPUT / name).write_bytes(content)
    manifest = json.loads(payload["MANIFIESTO.json"])
    print(json.dumps({
        "archive": (OUTPUT / ARCHIVE_NAME).relative_to(ROOT).as_posix(),
        "verified": True,
        "mode": "check" if args.check else "build",
        "entries": len(manifest["files"]) + 1,
        "bytes": len(payload[ARCHIVE_NAME]),
        "sha256": sha256(payload[ARCHIVE_NAME]).hexdigest(),
    }))


if __name__ == "__main__":
    main()
