#!/usr/bin/env python3
"""Build a deterministic, public-only supplement for the complete E2 CPU workflow.

Packaging performs no acquisition, import of model code, fitting or scoring.
The archive needs a matching Git checkout and separately retained data/sources.
"""
from __future__ import annotations

import argparse
import ast
import csv
from hashlib import sha256
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("book_e2/_static/completion_download")
ARCHIVE = "completion_bundle.zip"
FIXED_DATE = (2026, 11, 6, 0, 0, 0)
EXPERIMENT = "experiments/e2_completion_20261106"
CONFIGS = ("design.json", "hpo.json", "candidate_registry.json", "architecture_review.json",
           "analysis_additions.json", "preregistration.json", "workflow.md")
ENTRY_SCRIPTS = ("run_e2_final_benchmark.py", "download_e2_completion_sources.py",
                 "preflight_e2_final_models.py", "e2_final_models.py", "e2_final_models_mlp.py",
                 "e2_final_models_attention.py", "e2_final_statistics.py", "e2_final_uncertainty.py",
                 "e2_final_diagnostics.py", "e2_final_test_data.py", "e2_predictability_reference.py",
                 "e2_compute_profile.py", "e2_leakage_controls.py", "package_e2_completion.py")
TESTS = ("test_e2_final_benchmark.py", "test_e2_completion_acquisition.py",
         "test_e2_final_models.py", "test_e2_final_models_mlp.py", "test_e2_final_models_attention.py",
         "test_e2_final_statistics.py", "test_e2_final_uncertainty.py", "test_e2_final_diagnostics.py",
         "test_e2_final_test_data.py", "test_e2_predictability_reference.py", "test_e2_compute_profile.py",
         "test_e2_leakage_controls.py", "test_e2_completion_package.py")
HELPERS = ("dl_data.py", "dl_data_v2.py", "dl_models.py", "dl_models_gru_lstm_tcn.py",
           "e2_protocol_plan.py", "e2_temporal_baselines.py", "e2_predictive_uncertainty.py",
           "analyze_e2_benchmark.py", "planned_study_data.py", "run_e2_remediation_pilot.py")
LOCKS = ("requirements-e2-completion.lock.txt", "requirements-e2-remediation.lock.txt",
         "requirements-paper-guide.lock.txt", "requirements-eda.lock.txt")
INHERITED_CONFIGS = ("route.json", "architecture_decisions.json", "hpo.json", "preregistration.json",
                     "pilot_design.json", "uncertainty_errors.json")
PUBLIC_DIRS = ("book_e2/_static/completion_protocol", "book_e2/_static/completion_preflight",
               "book_e2/_static/completion_results")
SUPPORT = ("book_e2/_static/e1_reference.json", "book_e2/_static/dataset_card.json",
           "book_e2/_static/dataset_manifest_e1.json", "book_e2/_static/guia_e2_recibida.json",
           "book_e2/_static/remediation_protocol/candidate_registry.json",
           "book_e2/_static/remediation_error_results/training_taxonomy.json",
           "references/literature_e2/top10_20261105/primary_source_registry.json",
           "data/manifests/binance_spot_BTCUSDT_1m_20230101_20261030.json")
OMITTED_DATA = ("results/eda_development/daily_market_development.csv",
                "results/eda_development/daily_volatility_development.csv")
PUBLIC_SUFFIXES = {".json", ".csv", ".md", ".txt", ".bib", ".svg", ".png"}
PRIVATE_PATH = re.compile(r"(?:/workspace/|/mnt/data/|/tmp/|/home/|sediment://|file://)")
FORBIDDEN_FIELDS = {"ab", "abstract", "full_abstract", "raw_abstract", "cr", "email", "e_mail",
                    "fulltext_content", "full_text_content", "source_pdf_text", "raw_records",
                    "y_true", "y_pred", "truth", "predictions", "targets", "individual_errors",
                    "per_origin_errors", "private_stratified_audit_sample", "audit_cases", "frontier_cases",
                    "observed_values", "forecast_values", "actual_values", "raw_prices", "raw_ohlcv"}
FORBIDDEN_COLUMNS = FORBIDDEN_FIELDS | {"target", "prediction", "open", "high", "low", "close",
                                       "volume", "sigma_7", "sigma7", "log_return", "returns"}
CONTRACT_KEYS = ("source_commit", "registered_baseline", "base_model_recipes", "recipe_architectures",
                 "primary_metric", "practical_margin", "independence", "partition")
E1_SHA256 = "1464fa79d472f6755651774989b140659ee1f21daeb30032fc650a8770f4fdc4"

README = """# E2: procedimiento completo, código y evidencia pública

Suplemento de https://github.com/jassan-bit/DeepLearning_202630. Contiene el
procedimiento CPU de veinte configuraciones, quince recetas ajustables con
T=12 y asignación comparable, P1, potencia, ajustes primarios/A53, calibración,
errores OOF, costes, ablaciones y análisis. El código o el presupuesto asignado
no prueban que esas etapas hayan terminado ni que sus conclusiones sean
favorables. MANIFIESTO.json identifica exactamente los artefactos disponibles.
La evidencia de ejecución es únicamente la que figure en recibos públicos;
la ausencia de resultados no se rellena con métricas ficticias.

## Usar un checkout correspondiente del repositorio

Este ZIP no es un repositorio Git completo, un libro compilado ni un ejecutable
autónomo con datos. Usar el commit de publicación que lo acompaña y mantener
sus rutas relativas. El registro exige un checkout Git limpio y las identidades
de código, configuración, datos y fuentes. Extraer este suplemento por sí solo
no proporciona ese contexto. Desde la raíz del checkout, en Linux/Bash:

    python3.12 -m venv .venv-e2
    .venv-e2/bin/python -m pip install -r requirements-e2-completion.lock.txt
    .venv-e2/bin/python scripts/run_e2_final_benchmark.py --phase plan

El lock y el ejecutor son CPU. No hay ejecución GPU equivalente acreditada por
este paquete. Todos los trabajadores de ajuste usan un hilo numérico; el
ejecutor permite de uno a cuatro trabajadores. No reducir arquitecturas,
candidatos o repeticiones para presentar una ejecución parcial como completa.
Las comprobaciones de forma o memorización sintética no son ajustes de Bitcoin.

## Fuentes y datos conservados por separado

Los dos CSV diarios de desarrollo están versionados en el checkout bajo
results/eda_development; no se incluyen en el ZIP. design.json conserva sus
SHA-256. Conservar también los ZIP de mercado del manifiesto público y sus
identidades; el paquete no descarga, distribuye ni abre esos datos. No abrir
las reservas futuras antes de las puertas de fase. La adquisición de fuentes
descarga únicamente los archivos de autores fijados por commit/hash; no papers,
datasets ni pesos. Sus licencias y avisos deben conservarse:

    .venv-e2/bin/python scripts/download_e2_completion_sources.py --output-dir ../e2_run/author_sources

El archivo ../e2_run/author_sources/source_roots.json resultante contiene las
rutas privadas; no publicarlo. El manifiesto de fuentes incluido identifica
archivos ajenos pero no redistribuye sus contenidos.

## Ejecutar por fases

Consultar experiments/e2_completion_20261106/workflow.md antes de ejecutar.
Las órdenes siguientes conservan la misma salida privada y fuentes fijadas:

    .venv-e2/bin/python scripts/run_e2_final_benchmark.py --phase preflight --source-roots-json ../e2_run/author_sources/source_roots.json --output-dir ../e2_run/new_run --workers 4
    .venv-e2/bin/python scripts/run_e2_final_benchmark.py --phase register --source-roots-json ../e2_run/author_sources/source_roots.json --output-dir ../e2_run/new_run --raw-root ../market_raw --workers 4

Una vez registrado, repetir la misma base de orden con --phase hpo,
oof-errors, p1, power, final-fit, calibration, cost y ablation, en ese orden.
El scoring final es una fase explícita:

    .venv-e2/bin/python scripts/run_e2_final_benchmark.py --phase score --source-roots-json ../e2_run/author_sources/source_roots.json --output-dir ../e2_run/new_run --raw-root ../market_raw --workers 4
    .venv-e2/bin/python scripts/run_e2_final_benchmark.py --phase analyze --source-roots-json ../e2_run/author_sources/source_roots.json --output-dir ../e2_run/new_run --workers 4

La opción --phase all implementa esa secuencia y también exige --raw-root.
Usar una salida nueva fuera del checkout; no mezclar ni sobrescribir otro
registro. HPO completo cierra las quince recetas ajustables antes de P1. P1
evalúa 53 fechas del 8nov–30dic2024 con diez semillas nuevas y estima la potencia
del procedimiento; diez semillas no acreditan por sí solas potencia suficiente.
R se fija por esa etapa, con tope200 y declaración explícita si no se alcanza.

Las veinte cohortes A53 conservan cinco semillas prefijadas (una trayectoria
en referencias deterministas), con selección/fit/escalas hasta31oct2024 y
calibración8nov–30dic2024. No son intervalos del ajuste primario hasta24dic2024.
Todos los pesos primarios y A se cierran antes de calibrar; todos los
calibradores se cierran antes de puntuar la reserva conocida de60fechas2026.
No se afirma cobertura temporal exacta ni una nueva confirmación independiente.
Las ablaciones se realizan sobre desarrollo53 y no seleccionan componentes con
el test. Los OOF de las tres anclas heredadas declaran la exposición histórica
de su selección E1; no se presentan como selección anidada imparcial.

## Comprobación sin datos privados

Después de instalar el lock, las pruebas sintéticas no necesitan datos de
mercado ni pesos reales. Algunas comprobaciones de paridad con autores exigen
las fuentes fijadas y declaran su ausencia cuando corresponda:

    .venv-e2/bin/python -m unittest discover -s tests -p 'test_e2_final_*.py'
    .venv-e2/bin/python tests/test_e2_completion_package.py
    python scripts/package_e2_completion.py --check

El ZIP contiene helpers propios transitivos y metadatos; no contiene futuros
precios, CSV diarios, abstracts nativos, PDF, código descargado de autores,
checkpoints, predicciones por fecha o ledgers de errores individuales. Solo se
admiten derivados públicos autorizados. La selección global/PRISMA, los374
registros Scopus restantes y la evidencia editorial pendiente siguen teniendo
sus límites propios, aunque el código experimental esté disponible.

MANIFIESTO.json declara tamaños y SHA-256; SHA256SUMS identifica el contenedor.
Las fechas internas son fijas. Reconstruir no ejecuta ciencia:

    python scripts/package_e2_completion.py
    python scripts/package_e2_completion.py --check
"""


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def strict_json(payload):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key in public material: " + key)
            result[key] = value
        return result
    return json.loads(payload, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON constant: " + value)))


def safe_name(name):
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError("Invalid repository-relative path")
    relative = PurePosixPath(name)
    if relative.is_absolute() or relative.as_posix() != name or any(p in {".", ".."} or ":" in p for p in relative.parts):
        raise ValueError("Unsafe or non-normalized repository-relative path")
    return relative


def file_payload(root, name):
    root = Path(root).resolve()
    path = root
    for part in safe_name(name).parts:
        path /= part
        if path.is_symlink():
            raise ValueError("Symlinks are not permitted in package inputs")
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError("Required regular input missing or outside checkout: " + name)
    return path.read_bytes()


def check_json(value):
    if isinstance(value, dict):
        for key, item in value.items():
            empty = item is None or item is False or item == "" or isinstance(item, (list, dict)) and not item
            if key.lower() in FORBIDDEN_FIELDS and not empty:
                raise ValueError("Raw/individual fields are not permitted in public JSON: " + key)
            check_json(item)
    elif isinstance(value, list):
        for item in value:
            check_json(item)
    elif isinstance(value, str) and PRIVATE_PATH.search(value):
        raise ValueError("Private filesystem path or upload pointer in public JSON")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite public JSON measurement")


def check_public(name, payload):
    path = safe_name(name)
    if path.name.lower().startswith(("savedrecs", "export_", "private_", "predictions")):
        raise ValueError("Private/native output name cannot enter the package")
    if any(token in path.name.lower() for token in ("per_origin", "error_cases", "error_ledger", "daily_market", "daily_volatility")):
        raise ValueError("Daily/individual payload names cannot enter the package")
    if path.suffix not in PUBLIC_SUFFIXES:
        raise ValueError("Unsupported public payload type: " + name)
    if path.suffix == ".json":
        check_json(strict_json(payload))
    elif path.suffix == ".csv":
        text = payload.decode("utf-8-sig")
        if PRIVATE_PATH.search(text):
            raise ValueError("Private path in public CSV")
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Missing/duplicate CSV headers")
        fields = {k.lower() for k in reader.fieldnames}
        if fields & FORBIDDEN_COLUMNS:
            raise ValueError("Raw/native/individual CSV columns are not permitted")
        dates = fields & {"origin_utc", "outcome_day", "market_day_utc", "date", "case_id"}
        if dates and fields & {"error", "residual", "absolute_error", "category", "value", "values", "forecast", "price", "inclusion_weight"}:
            raise ValueError("Per-date observations/error ledgers must remain private")
        if any(None in row or any(v is None for v in row.values()) for row in reader):
            raise ValueError("Malformed public CSV")
    elif path.suffix in {".md", ".txt", ".svg", ".bib"} and PRIVATE_PATH.search(payload.decode("utf-8")):
        raise ValueError("Private path or upload pointer in public documentation")


def local_import_closure(root, names):
    """Include own scripts referenced even by function-local imports."""
    root = Path(root)
    pending = list(names)
    result = set()
    while pending:
        name = pending.pop()
        if name in result:
            continue
        source = file_payload(root, name)
        result.add(name)
        for node in ast.walk(ast.parse(source, filename=name)):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) else
                       [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            for module in modules:
                if not module or "." in module:
                    continue
                dependency = "scripts/" + module + ".py"
                if (root / dependency).is_file() and dependency not in result:
                    pending.append(dependency)
    return result


def source_contents(root=ROOT):
    root = Path(root).resolve()
    design = json.loads(file_payload(root, EXPERIMENT + "/design.json"))
    if set(design.get("development_sources", {})) != set(OMITTED_DATA):
        raise ValueError("The design must bind exactly the two omitted development CSVs")
    names = {f"{EXPERIMENT}/{name}" for name in CONFIGS}
    names.update(f"experiments/e2_remediation_20261106/{name}" for name in INHERITED_CONFIGS)
    names.update(SUPPORT)
    names.update(LOCKS)
    seeds = {"scripts/" + name for name in (*ENTRY_SCRIPTS, *HELPERS)}
    names.update("tests/" + name for name in TESTS)
    # Include source-bound own files; data payloads and downloaded foreign files
    # cannot be admitted merely by adding them to the design.
    for name in design.get("source_files", []):
        relative = safe_name(name)
        if relative.parts[0] in {"scripts", "tests"} and relative.suffix == ".py":
            seeds.add(name)
        elif name in LOCKS or name in SUPPORT or relative.parts[:2] == ("third_party", "e2") and relative.suffix in {".txt", ".md"}:
            names.add(name)
        elif relative.parts[:2] == ("experiments", "e2_completion_20261106") and relative.name in CONFIGS:
            names.add(name)
        else:
            raise ValueError("A source-bound input is outside the package allowlist: " + name)
    names.update(local_import_closure(root, seeds | {"tests/" + name for name in TESTS}))
    names.update("book_e2/sections/" + path.name for path in sorted((root / "book_e2/sections").glob("*.md")))
    for name in ("intro.md", "references.bib", "_config.yml", "_toc.yml"):
        if (root / "book_e2" / name).is_file() and Path(name).suffix != ".yml":
            names.add("book_e2/" + name)
    names.add("third_party/e2/NOTICE.md")
    names.add("third_party/e2/Apache-2.0.txt")
    included, absent = [], []
    for folder in PUBLIC_DIRS:
        directory = root / folder
        if not directory.exists():
            absent.append(folder)
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("Public evidence directory must be regular")
        included.append(folder)
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise ValueError("Symlinks are not permitted under public evidence roots")
            if path.is_file():
                names.add(path.relative_to(root).as_posix())
    contents = {}
    for name in sorted(names):
        if name in OMITTED_DATA or Path(name).suffix not in PUBLIC_SUFFIXES | {".py"}:
            raise ValueError("Data/private payload accidentally selected")
        payload = file_payload(root, name)
        if Path(name).suffix != ".py":
            check_public(name, payload)
        contents[name] = payload
    reference = json.loads(contents["book_e2/_static/e1_reference.json"])
    if sha256(json_bytes({key: reference[key] for key in CONTRACT_KEYS})).hexdigest() != E1_SHA256:
        raise ValueError("Inherited E1 scientific identity differs")
    contents["LEEME.md"] = README.encode()
    return contents, design, included, absent


def validate_archive(payload, manifest_bytes):
    manifest = json.loads(manifest_bytes)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        expected = set(manifest["files"]) | {"MANIFIESTO.json"}
        if len(archive.namelist()) != len(expected) or set(archive.namelist()) != expected:
            raise ValueError("ZIP member set/duplicates differ from the manifest")
        if archive.testzip() is not None or archive.read("MANIFIESTO.json") != manifest_bytes:
            raise ValueError("ZIP CRC or embedded manifest differs")
        for name, record in manifest["files"].items():
            safe_name(name)
            data = archive.read(name)
            if len(data) != record["bytes"] or sha256(data).hexdigest() != record["sha256"]:
                raise ValueError("ZIP member content differs from manifest: " + name)
        if any(info.date_time != FIXED_DATE or info.external_attr >> 16 != 0o100644 for info in archive.infolist()):
            raise ValueError("ZIP timestamps/regular file modes are not fixed")


def build_payload(root=ROOT):
    contents, design, included, absent = source_contents(root)
    manifest = {"schema_version": 1, "scope": "E2 complete CPU procedure supplement; execution completion requires actual phase receipts",
                "protocol_id": design["protocol_id"], "scientific_E1_contract_sha256": E1_SHA256,
                "standalone_without_repository_checkout": False,
                "repository_Git_context_and_tracked_development_CSVs_required": True,
                "omitted_development_sources_sha256": design["development_sources"],
                "included_public_evidence_directories": included,
                "public_evidence_directories_not_present": absent,
                "market_data_or_native_abstracts_included": False, "paper_PDFs_included": False,
                "downloaded_author_code_included": False, "weights_included": False,
                "individual_predictions_or_error_cases_included": False,
                "new_fits_performed_by_packager": 0, "GPU_execution_implemented_by_this_package": False,
                "author_source_manifest_only_count": len(design.get("author_sources", [])),
                "files": {name: {"bytes": len(data), "sha256": sha256(data).hexdigest()}
                          for name, data in sorted(contents.items())}}
    manifest_bytes = json_bytes(manifest)
    contents["MANIFIESTO.json"] = manifest_bytes
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, FIXED_DATE)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    payload = buffer.getvalue()
    validate_archive(payload, manifest_bytes)
    return {ARCHIVE: payload, "MANIFIESTO.json": manifest_bytes, "LEEME.md": contents["LEEME.md"],
            "SHA256SUMS": f"{sha256(payload).hexdigest()}  {ARCHIVE}\n".encode()}


def deliver(payloads, output):
    """Validate all bytes first, then replace each delivered file atomically."""
    output = Path(output)
    if any(parent.is_symlink() for parent in (output, *output.parents)):
        raise ValueError("Output directory cannot be a symlink")
    validate_archive(payloads[ARCHIVE], payloads["MANIFIESTO.json"])
    output.mkdir(parents=True, exist_ok=True)
    if {path.name for path in output.iterdir()} - set(payloads):
        raise ValueError("Unexpected existing files under the public download directory")
    for name, data in payloads.items():
        safe_name(name)
        target = output / name
        if target.is_symlink():
            raise ValueError("Output files cannot be symlinks")
    with tempfile.TemporaryDirectory(prefix=".completion-package-", dir=output) as temporary:
        temporary = Path(temporary)
        for name, data in payloads.items():
            (temporary / name).write_bytes(data)
        for name in payloads:
            (temporary / name).replace(output / name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    payloads = build_payload()
    output = ROOT / OUTPUT
    if args.check:
        if {path.name for path in output.iterdir()} != set(payloads):
            raise SystemExit("Public download directory contains missing or unexpected files")
        for name, data in payloads.items():
            if file_payload(ROOT, (OUTPUT / name).as_posix()) != data:
                raise SystemExit("Package is stale: " + name)
    else:
        deliver(payloads, output)
    manifest = json.loads(payloads["MANIFIESTO.json"])
    print(json.dumps({"archive": (OUTPUT / ARCHIVE).as_posix(), "verified": True,
                      "mode": "check" if args.check else "build", "entries": len(manifest["files"]) + 1,
                      "bytes": len(payloads[ARCHIVE]), "sha256": sha256(payloads[ARCHIVE]).hexdigest(),
                      "market_payloads_opened": 0, "fits": 0}))


if __name__ == "__main__":
    main()
