#!/usr/bin/env python3
"""Package current E2 remediation policy, own code and permitted public evidence.

This is a repository supplement for the registered P0/OOF development workflow,
not the optimized twenty-model benchmark. No data acquisition, inference,
training, scientific-generator execution or source mutation occurs here.
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
OUTPUT = ROOT / "book_e2/_static/remediation_download"
ARCHIVE_NAME = "Remediacion_E2_protocolo_codigo_evidencias.zip"
FIXED_DATE = (2026, 11, 6, 0, 0, 0)
REGISTERED_CORE_COMMIT = "599a29f87f8ebb1d67cd925ec09bf6bdbc732fa0"
SCIENTIFIC_E1_SHA256 = "1464fa79d472f6755651774989b140659ee1f21daeb30032fc650a8770f4fdc4"
EXPERIMENT = "experiments/e2_remediation_20261106"
CONFIG_NAMES = ("route.json", "architecture_decisions.json", "hpo.json", "preregistration.json",
                "pilot_design.json", "uncertainty_errors.json")
CORE_SCRIPTS = ("e2_protocol_plan.py", "e2_temporal_baselines.py", "e2_temporal_models.py",
                "e2_predictive_uncertainty.py", "e2_remediation_data_audit.py",
                "run_e2_remediation_pilot.py", "execute_e2_development_pilot.py",
                "analyze_e2_remediation_pilot.py", "screen_e2_literature.py", "package_e2_remediation.py")
TEST_NAMES = ("test_e2_protocol_plan.py", "test_e2_temporal_baselines.py", "test_e2_remediation_pilot.py",
              "test_e2_remediation_statistics.py", "test_e2_literature_screening.py", "test_e2_predictive_uncertainty.py")
CHAPTERS = ("01_punto_partida.md", "02_eda.md", "03_literatura.md", "04_implementacion.md",
            "06_benchmark_final.md", "07_ruta_temporal.md", "09_protocolo_registrado.md", "12_incertidumbre_errores.md")
NEW_CHAPTER_PREFIXES = ("08_", "10_", "11_", "13_", "14_")
PUBLIC_DIRECTORIES = ("book_e2/_static/remediation_protocol", "book_e2/_static/literature_remediation",
                      "book_e2/_static/remediation_pilot_results", "book_e2/_static/remediation_error_results",
                      "book_e2/_static/remediation_preflight")
# Minimal public bibliographic inputs required by the screening CLI or protocol.
# They contain identifiers/title/DOI/curated facts, not native abstracts or PDFs.
SUPPORT_FILES = (
    "book_e2/_static/literature_scopus/combined_records.csv",
    "book_e2/_static/literature_scopus/original_model_records.csv",
    "book_e2/_static/literature_scopus/summary.json",
    "book_e2/_static/literature_selection/representative_studies.csv",
    "book_e2/_static/literature_selection/model_selection.csv",
    "book_e2/_static/literature_fulltext/models.json",
    "book_e2/_static/literature_fulltext/fulltext_manifest.json",
    "references/literature_e2/top10_20261105/primary_source_registry.json",
)
FIXED_FILES = (
    *(f"scripts/{name}" for name in CORE_SCRIPTS),
    *(f"tests/{name}" for name in TEST_NAMES),
    *(f"{EXPERIMENT}/{name}" for name in CONFIG_NAMES),
    *(f"book_e2/sections/{name}" for name in CHAPTERS),
    "scripts/dl_data.py", "scripts/dl_data_v2.py", "scripts/dl_models.py", "scripts/dl_models_gru_lstm_tcn.py",
    "scripts/plot_e2_remediation_results.py", "requirements-eda.lock.txt", "scripts/download_e2_paper_sources.py", "requirements-e2-remediation.lock.txt", "requirements-paper-guide.lock.txt",
    "book_e2/references.bib", "book_e2/_static/e1_reference.json", "book_e2/_static/dataset_card.json",
    "book_e2/_static/guia_e2_recibida.json", "book_e2/_static/current_protocol.json",
    "book_e2/_static/dataset_manifest_e1.json", *SUPPORT_FILES,
)
PUBLIC_SUFFIXES = {".json", ".csv", ".md", ".png", ".svg", ".txt", ".bib"}
PRIVATE_PATH = re.compile(r"(?:/workspace/|/mnt/data/|/tmp/|/home/|sediment://|file://)")
FORBIDDEN_COLUMNS = {"ab", "abstract", "full_abstract", "raw_abstract", "cr", "c1", "email", "e_mail",
                     "y_true", "y_pred", "truth", "prediction", "predictions", "target", "targets",
                     "open", "high", "low", "close", "volume", "sigma_7", "log_return", "return", "returns"}
FORBIDDEN_JSON_FIELDS = {"ab", "abstract", "full_abstract", "raw_abstract", "cr", "fulltext_content",
                         "full_text_content", "source_pdf_text", "y_true", "y_pred", "predictions",
                         "targets", "individual_errors", "per_origin_errors", "private_stratified_audit_sample",
                         "private_high_loss_frontier_origins", "audit_cases", "frontier_cases"}
CONTRACT_KEYS = ("source_commit", "registered_baseline", "base_model_recipes", "recipe_architectures",
                 "primary_metric", "practical_margin", "independence", "partition")

README = """# Remediación E2: protocolo, código y evidencias públicas

Suplemento del repositorio https://github.com/jassan-bit/DeepLearning_202630.
Conserva el protocolo F2a actual, el código propio, los tests, el cribado
bibliográfico y los agregados públicos disponibles al empaquetar. El núcleo
científico se publicó antes de los nuevos ajustes en el commit
599a29f87f8ebb1d67cd925ec09bf6bdbc732fa0. Los SHA-256 del manifiesto identifican
esta entrega; una repetición posterior debe registrar su propia fecha y fuentes.

## Alcance del procedimiento

El catálogo final planificado tiene veinte configuraciones y quince familias
ajustables. Sus T=12 candidatos y presupuesto de cómputo son asignaciones, no una
búsqueda ya ejecutada. El flujo que puede repetirse con una orden es **P0/OOF de
desarrollo**, con recetas fijas: TimeMixer, iTransformer, PatchTST y GRU, semillas
660101–660105. Son veinte trayectorias noviembre–diciembre y doscientas OOF
mensuales enero–octubre2024: 220 trayectorias, 440 etapas de optimización
(selección de época y refit). No es el HPO, P1, la prueba de potencia definitiva,
la selección de los tres mejores finales ni el benchmark optimizado de veinte
modelos. Los otros adaptadores y las puertas de fase siguen su revisión propia.

Las configuraciones congeladas dicen cero resultados nuevos en el momento de su
definición. Esos contadores históricos no se convierten en el total actual de
P0/OOF. current_protocol.json y los recibos/agregados incluidos describen la
ejecución observada disponible. Si una carpeta de resultados aún no existe, su
ausencia se declara en el manifiesto, no se sustituye por una tabla ficticia.
La conclusión sobre gaps o incertidumbre es la que arrojen los datos registrados;
este paquete no promete significancia, mejora ni potencia suficiente.

La herencia científica E1 permanece: D7×5, σ7, h1, RMSE media entre trayectorias,
δ y calendario. El flujo P0 no abre 2025–2026 ni restaura un test desconocido.
La calibración predictiva A es secundaria: fit/escalas/selección hasta24oct2024,
calibración1nov–30dic con ese mismo predictor congelado, sin cobertura temporal
exacta afirmada. Los errores OOF cuentan fechas únicas y mantienen separados el
diagnóstico P0 y los mejores modelos finales.

## Instalación y fuentes: usar un checkout del repositorio

El ZIP no es una copia completa del repositorio ni del Jupyter Book compilado.
Clonar el proyecto, usar el commit de publicación correspondiente y, si se
extrae este suplemento, conservar sus subdirectorios. No sustituir el código
científico congelado por una versión incompatible. Desde la raíz del checkout:

    python3.12 -m venv .venv-e2
    .venv-e2/bin/python -m pip install -r requirements-e2-remediation.lock.txt

El lock incluye el de paper-guide para las dependencias de los módulos de
autores; usarlo no convierte el presente P0 Bitcoin en la réplica ETTh1. Los dos
CSV diarios de desarrollo2023–2024 ya están versionados en el checkout bajo
results/eda_development. **No se incluyen esos datos diarios en este ZIP**. Sus
nombres y hashes se conservan en pilot_design.json; el ejecutor los verifica y
limita toda decodificación a desarrollo. La documentación data/README.md del
checkout describe la fuente y adquisición originales. No se redistribuyen
archivos crudos de mercado.

Los dos repositorios de autores se adquieren por separado. Sus URLs de ZIP,
commits, licencias y hashes de archivo/archivos interiores están en
pilot_design.json. Este ejemplo utiliza el descargador seguro propio incluido,
descarga solo esos repositorios y **no descarga ETTh1**:

```bash
.venv-e2/bin/python - <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from download_e2_paper_sources import asset, extract_verified, fetch_bytes, MAX_DOWNLOAD_BYTES
plan = json.loads(Path('experiments/e2_remediation_20261106/pilot_design.json').read_text())
folder = Path('../e2_reproduction/author_sources')
folder.mkdir(parents=True, exist_ok=True)
for source in plan['author_sources'].values():
    payload = asset(folder, source['archive_filename'], source['archive_url'], source['archive_sha256'], MAX_DOWNLOAD_BYTES, fetch_bytes)
    extract_verified(folder, payload, source)
PY
```

El ejemplo está escrito para Bash/Linux. Los ZIP y el código de autores quedan
fuera del repositorio y no entran en este paquete. Se conservan Apache2.0 para
TimeMixer y MIT para Time-Series-Library; consultar sus avisos originales.

## Repetir P0 y OOF registrados con una orden

Con los dos CSV presentes en el checkout y las fuentes verificadas:

    .venv-e2/bin/python scripts/execute_e2_development_pilot.py --stage all --source-root ../e2_reproduction/author_sources/TimeMixer-e24610583b36fdd8c76cc17a8df4e65759a5f460 --control-source-root ../e2_reproduction/author_sources/Time-Series-Library-4e938a1767106324dd753b2a44832bf870a0252e --output-dir ../e2_reproduction/new_run --workers 2

La orden registra antes de ajustar, conserva artefactos privados, cierra todos
los checkpoints programados antes del scoring de su fase y analiza predicciones
guardadas. Ejecuta hasta440 ajustes, no solo una demostración de forma. Usar una
salida propia fuera del checkout; no mezclarla con recibos de una ejecución
anterior ni reemplazar semillas fallidas/desfavorables. --stage preflight solo
valida datos/soportes, y las etapas register/fit/score/analyze permiten inspección
explícita. El registro exige contexto Git y bytes científicos versionados.
Extraer el ZIP sin un checkout no recrea ese contexto ni los CSV excluidos.

La incertidumbre y el análisis de errores tienen funciones/CLI propias en
e2_predictive_uncertainty.py. Sus instrucciones están en el capítulo12: requieren
OOF y taxonomía causal, al menos cien fechas elegibles únicas por modelo y salida
privada nueva. No interpretar recetas P0 como Top3 finales ni calibrar el ajuste
primario con sus errores de entrenamiento de noviembre–diciembre.

## Revisión sin fuentes ni datos de mercado

Los tests sintéticos pueden ejecutarse después de instalar el lock, sin los
CSV originales, fuentes de autores o pesos reales:

    .venv-e2/bin/python tests/test_e2_protocol_plan.py
    .venv-e2/bin/python tests/test_e2_temporal_baselines.py
    .venv-e2/bin/python tests/test_e2_remediation_pilot.py
    .venv-e2/bin/python tests/test_e2_remediation_statistics.py
    .venv-e2/bin/python tests/test_e2_literature_screening.py
    .venv-e2/bin/python tests/test_e2_predictive_uncertainty.py

Las pruebas de ARIMA ajustan ejemplos sintéticos pequeños; no son corridas ni
métricas de Bitcoin. Las pruebas no sustituyen ausencia de fuga, potencia o
eficacia del procedimiento final. El validador estándar puede inspeccionarse con
`python -S scripts/e2_protocol_plan.py validate`, sin ajuste ni scoring.

El cribado de literatura contiene el ledger de7.833 grupos, títulos/DOI, reglas,
huellas y decisiones provisionales. No contiene abstracts completos ni los
exports nativos WoS/Scopus. Volver a ejecutar screen_e2_literature.py exige el
private_combined_records.json autorizado, conservado por separado; sus tests
usan fixtures. El texto público distingue consultas locales de nuevas búsquedas
nativas, los374 Scopus no recibidos y la elegibilidad final todavía abierta.

## Integridad y archivos excluidos

MANIFIESTO.json registra cada tamaño y SHA-256. SHA256SUMS junto al ZIP identifica
el contenedor. Los timestamps internos son fijos al06/11/2026. Para reconstruirlo
desde sus fuentes públicas y verificarlo, sin ejecutar ciencia:

    python scripts/package_e2_remediation.py
    python scripts/package_e2_remediation.py --check

El ZIP no se contiene a sí mismo. Los metadatos de preregistro se mantienen como
capturas históricas; sus hashes no certifican por sí solos corrección científica.
Los ZIP anteriores permanecen como archivos de otras versiones, no como el
protocolo vigente. No se ejecutan sus generadores antiguos para reescribirlos.
Se excluyen precios, CSV diarios, abstracts, PDFs, código descargado de autores,
checkpoints, pronósticos por fecha y ledgers individuales de errores. Solo se
admiten agregados por modelo/semilla/fold, recibos y figuras públicos permitidos.
"""


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode()


def file_payload(root, name):
    relative = Path(name); path = root/relative
    if relative.is_absolute() or ".." in relative.parts or path.is_symlink():
        raise ValueError("Only explicit regular repository-relative inputs are permitted")
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Publication input missing or outside repository: {name}")
    return path.read_bytes()


def check_json(value):
    if isinstance(value, dict):
        for key,item in value.items():
            empty = item is None or item is False or item == "" or (isinstance(item,(list,dict)) and not item)
            if key.lower() in FORBIDDEN_JSON_FIELDS and not empty:
                raise ValueError(f"Raw/individual payload is not public package material: {key}")
            check_json(item)
    elif isinstance(value, list):
        for item in value: check_json(item)
    elif isinstance(value, str) and PRIVATE_PATH.search(value):
        raise ValueError("Private filesystem path or uploaded-file pointer in public metadata")


def check_csv(name, payload):
    text = payload.decode("utf-8-sig")
    if PRIVATE_PATH.search(text):
        raise ValueError(f"Private filesystem path in public CSV: {name}")
    reader = csv.DictReader(io.StringIO(text)); fields = reader.fieldnames
    if not fields or len(set(fields)) != len(fields):
        raise ValueError(f"Missing or duplicate CSV headers: {name}")
    normalized = {field.lower() for field in fields}
    if normalized & FORBIDDEN_COLUMNS:
        raise ValueError(f"Native text, prices or per-origin prediction fields cannot be packaged: {name}")
    individual_key = bool(normalized & {"origin_utc", "outcome_day", "market_day_utc", "date", "case_id"})
    if individual_key and normalized & {"category", "error", "residual", "absolute_error", "inclusion_weight", "inverse_inclusion_weight"}:
        raise ValueError(f"Individual error ledgers must stay private: {name}")
    if individual_key and normalized & {"price", "prices", "price_close", "close_price", "sigma7", "value", "values", "forecast"}:
        raise ValueError(f"Daily observation/forecast values must stay outside this supplement: {name}")
    if any(None in row or any(v is None for v in row.values()) for row in reader):
        raise ValueError(f"Malformed public aggregate CSV: {name}")


def check_public(name, payload):
    path = Path(name)
    if path.name.startswith(("savedrecs", "export_", "private_")) or path.name == "predictions.csv":
        raise ValueError(f"Private/native payload name: {name}")
    if path.name != "SHA256SUMS" and path.suffix not in PUBLIC_SUFFIXES:
        raise ValueError(f"Only public aggregate/metadata extensions are permitted: {name}")
    if path.suffix == ".json": check_json(json.loads(payload))
    elif path.suffix == ".csv": check_csv(name,payload)
    elif path.suffix in {".md", ".txt", ".svg", ".bib"} and PRIVATE_PATH.search(payload.decode("utf-8")):
        raise ValueError(f"Private filesystem path in public documentation: {name}")


def source_contents(root=ROOT):
    names = set(FIXED_FILES)
    for prefix in NEW_CHAPTER_PREFIXES:
        matches = sorted((root/"book_e2/sections").glob(prefix+"*.md"))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one current chapter with prefix {prefix}")
        names.add(matches[0].relative_to(root).as_posix())
    included = []; absent = []
    for folder in PUBLIC_DIRECTORIES:
        directory = root/folder
        if not directory.exists():
            absent.append(folder); continue
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("Public output roots must be regular directories")
        included.append(folder)
        names.update(path.relative_to(root).as_posix() for path in directory.rglob("*") if path.is_file())
    # The screening CLI defaults to this curated annotation mirror. It remains
    # public and small; include its other public derivatives if present.
    reference = root/"references/literature_e2/remediation_20261106"
    if reference.exists():
        if reference.is_symlink() or not reference.is_dir():
            raise ValueError("Curated literature reference directory must be regular")
        names.update(path.relative_to(root).as_posix() for path in reference.rglob("*") if path.is_file())
    contents = {}
    for name in sorted(names):
        payload = file_payload(root,name)
        # Source code/tests may contain literal negative-test paths. Public
        # metadata/data/documentation cannot contain real private filesystem paths.
        if Path(name).suffix != ".py": check_public(name,payload)
        contents[name] = payload
    reference = json.loads(contents["book_e2/_static/e1_reference.json"])
    protected = {key:reference[key] for key in CONTRACT_KEYS}
    if sha256(json_bytes(protected)).hexdigest() != SCIENTIFIC_E1_SHA256:
        raise ValueError("Protected E1 scientific identity changed")
    current = json.loads(contents["book_e2/_static/current_protocol.json"])
    if current.get("master_registered_source_commit") != REGISTERED_CORE_COMMIT:
        raise ValueError("Current protocol no longer identifies the registered pre-fit core")
    if current.get("main_twenty_model_benchmark_completed") is not False:
        raise ValueError("This development supplement cannot claim completion of the twenty-model benchmark")
    plan = json.loads(contents[f"{EXPERIMENT}/pilot_design.json"])
    if plan["scope"] != "P0_development_feasibility_variance_gap_not_final_benchmark":
        raise ValueError("This package is restricted to development P0/OOF")
    completed_summary = "book_e2/_static/remediation_pilot_results/summary.json"
    if completed_summary in contents:
        observed = json.loads(contents[completed_summary])
        if {"pilot","oof"}.issubset(observed.get("phases",[])):
            planned = len(plan["models"])*len(plan["seeds"])*(1+len(plan["calendar"]["OOF_origins"]))
            if current.get("P0_and_OOF_closed_trajectories_snapshot") != planned:
                raise ValueError("Completed P0/OOF aggregates require an updated current-protocol closed-trajectory snapshot")
    # Config zero-results flags are historical definitions. Do not reject the
    # later aggregate P0/OOF results merely because development fits now exist.
    contents["LEEME.md"] = README.encode("utf-8")
    return contents,current,plan,included,absent


def validate_archive(payload, manifest_bytes):
    manifest = json.loads(manifest_bytes)
    with zipfile.ZipFile(io.BytesIO(payload)) as stream:
        if stream.testzip() is not None: raise ValueError("ZIP CRC verification failed")
        expected = set(manifest["files"])|{"MANIFIESTO.json"}
        if set(stream.namelist()) != expected or len(stream.namelist()) != len(expected):
            raise ValueError("ZIP differs from its manifest or contains duplicate entries")
        if stream.read("MANIFIESTO.json") != manifest_bytes: raise ValueError("Embedded and external manifests differ")
        for name,record in manifest["files"].items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe ZIP member path")
            data = stream.read(name)
            if len(data) != record["bytes"] or sha256(data).hexdigest() != record["sha256"]:
                raise ValueError(f"ZIP size/SHA-256 mismatch: {name}")
        if any(info.date_time != FIXED_DATE for info in stream.infolist()):
            raise ValueError("ZIP timestamps are not fixed")


def build_payload(root=ROOT):
    contents,current,plan,included,absent = source_contents(root)
    nmodels,nseeds = len(plan["models"]),len(plan["seeds"])
    planned_trajectories = nmodels*nseeds*(1+len(plan["calendar"]["OOF_origins"])) if plan["OOF"]["enabled"] else nmodels*nseeds
    manifest = {
        "schema_version":1,"client_date_colombia":"2026-11-06",
        "scope":"Registered E2 remediation development P0/OOF, policy and public aggregates; not final optimized twenty-model benchmark",
        "registered_core_source_commit":REGISTERED_CORE_COMMIT,
        "scientific_E1_contract_sha256":SCIENTIFIC_E1_SHA256,
        "planned_P0_OOF_trajectories":planned_trajectories,"planned_optimizer_fit_stages":2*planned_trajectories,
        "closed_trajectories_public_snapshot":current.get("P0_and_OOF_closed_trajectories_snapshot"),
        "included_public_output_directories":included,"public_directories_not_yet_present":absent,
        "main_twenty_model_benchmark_complete":False,"P0_is_final_HPO_or_power_demonstration":False,
        "old_zero_execution_flags_scope":"frozen policy definitions before fits, not the total later observed development work",
        "raw_daily_or_market_data_included":False,"native_abstract_exports_or_PDFs_included":False,
        "downloaded_author_code_included":False,"individual_predictions_or_error_ledgers_included":False,
        "checkpoints_included":False,"repository_checkout_with_tracked_development_CSVs_required":True,
        "files":{name:{"bytes":len(data),"sha256":sha256(data).hexdigest()} for name,data in sorted(contents.items())},
    }
    manifest_bytes = json_bytes(manifest); contents["MANIFIESTO.json"] = manifest_bytes
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer,"w",compression=zipfile.ZIP_DEFLATED) as stream:
        for name,data in sorted(contents.items()):
            info = zipfile.ZipInfo(name,date_time=FIXED_DATE); info.compress_type=zipfile.ZIP_DEFLATED
            info.create_system=3; info.external_attr=0o100644<<16; stream.writestr(info,data)
    archive=buffer.getvalue();validate_archive(archive,manifest_bytes)
    return {ARCHIVE_NAME:archive,"MANIFIESTO.json":manifest_bytes,"LEEME.md":contents["LEEME.md"],
            "SHA256SUMS":f"{sha256(archive).hexdigest()}  {ARCHIVE_NAME}\n".encode()}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true",help="Verify delivered container exactly matches its current public sources")
    args=parser.parse_args(argv);payload=build_payload()
    if args.check:
        for name,data in payload.items():
            path=OUTPUT/name
            if not path.is_file() or path.read_bytes()!=data:
                raise SystemExit(f"Package is stale or missing: {path.relative_to(ROOT)}")
    else:
        OUTPUT.mkdir(parents=True,exist_ok=True)
        for name,data in payload.items():(OUTPUT/name).write_bytes(data)
    manifest=json.loads(payload["MANIFIESTO.json"])
    print(json.dumps({"archive":(OUTPUT/ARCHIVE_NAME).relative_to(ROOT).as_posix(),"verified":True,
                      "mode":"check" if args.check else "build","entries":len(manifest["files"])+1,
                      "bytes":len(payload[ARCHIVE_NAME]),"sha256":sha256(payload[ARCHIVE_NAME]).hexdigest()}))


if __name__=="__main__":main()
