"""Package the seven public E1 products without fitting or reading private evidence."""
import argparse
import hashlib
import json
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from literature_crypto_stage import load_crypto_stage

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_BOOK = "https://jassan-bit.github.io/DeepLearning_202630/"
RAW = "https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/entregables/"
D7_STUDY = "e1_filipe_confirmatory_d7_20261105"
CRYPTO_STAGE = "crypto_extension_20261105"
ITEMS = [
    ("8.1", "8_1_informe_html.zip", "Informe HTML", "informe"),
    ("8.2", "8_2_codigo.zip", "Código EDA, modelos y análisis", "codigo"),
    ("8.3", "8_3_dataset_instrucciones.zip", "Instrucciones y adquisición del dataset", "dataset"),
    ("8.4", "8_4_literatura.zip", "Síntesis de literatura y flujo", "literatura"),
    ("8.5", "8_5_benchmark.zip", "Benchmark e inferencia agregada", "benchmark"),
    ("8.6", "8_6_configuraciones.zip", "Dependencias, semillas y configuraciones", "configuraciones"),
    ("8.7", "8_7_declaracion.zip", "Declaración y cronología", "declaracion"),
]
EXCLUDED_NAMES = {
    "predictions.csv", "daily_market_development.csv", "daily_volatility_development.csv",
    "causal_daily_profile.csv", "label_reconstruction.csv", "validation_predictions_residuals.csv",
    "model_parameters.csv", "reference_configurations.csv",
    "preprocessing.json", "splits.json", "history_inner.csv", "history_refit.csv",
}
ARCHIVE_OR_WEIGHTS = {".gz", ".tar", ".zip", ".npz", ".npy", ".pt", ".pth", ".pkl", ".pickle"}
TEXT_SUFFIXES = {".md", ".json", ".csv", ".txt", ".bib", ".svg", ".png", ".py"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def add_file(files, path, name=None):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected regular file: {path}")
    files[name or path.relative_to(ROOT).as_posix()] = path.read_bytes()


def public_tree(files, directory, excluded_directories=()):
    for path in sorted(directory.rglob("*")):
        if any(path.is_relative_to(excluded) for excluded in excluded_directories):
            continue
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            if path.name in EXCLUDED_NAMES:
                raise ValueError(f"Unexpected individual evidence in public tree: {path}")
            add_file(files, path)


def allowlisted_tree(files, directory, manifest_name="public_allowlist_manifest.json", include=None):
    manifest = directory / manifest_name
    if directory.is_symlink() or manifest.is_symlink():
        raise ValueError(f"Expected a regular public aggregate export: {directory}")
    record = json.loads(manifest.read_text())
    for name, expected in record["files_sha256"].items():
        relative = Path(name)
        path = directory / relative
        if (relative.is_absolute() or ".." in relative.parts or path.is_symlink()
                or not path.resolve().is_relative_to(directory.resolve())
                or relative.name in EXCLUDED_NAMES
                or relative.suffix.lower() not in TEXT_SUFFIXES):
            raise ValueError(f"Unexpected private or unsafe allowlisted artifact: {name}")
        data = path.read_bytes()
        if digest(data) != expected:
            raise ValueError(f"Changed public aggregate: {path}")
        if include is None or include(name):
            add_file(files, path)
    add_file(files, manifest)


def package_crypto_stage(packages, stage):
    """Package only the validated stage manifest and its listed public metadata."""
    directory = stage["public_directory"]
    names = set(stage["manifest"]["files_sha256"])
    if any(Path(name).suffix.lower() not in {".json", ".csv", ".svg", ".bib", ".md"}
           for name in names):
        raise ValueError("The crypto literature stage must contain metadata only")
    erratum = stage["summary"].get("harnet_erratum_file", "harnet_erratum.json")
    declarations = {"summary.json", erratum}
    if not declarations.issubset(names):
        raise ValueError("Missing crypto literature summary or HARNet erratum")
    allowlisted_tree(packages["8.4"], directory, "manifest.json")
    for name in sorted(declarations | {"manifest.json"}):
        add_file(packages["8.7"], directory / name)


def literature_readmes(stage):
    if stage is None:
        return (
            "Se conserva el alcance científico del libro: seis textos del corpus BTC y39 referencias. "
            "Miura2019 respalda MLP/CNN independientes como familias. Bucci2020 aporta un original "
            "de revista con FNN financiera; HARNet aporta código autor Conv1D; DeepVol tiene identidad "
            "editorial en CV, con original pendiente. Estas fuentes metodológicas no añaden estudios BTC "
            "ni revelan la dimensionalidad no reportada de la CNN de Miura. No son réplicas de las recetas. ",
            "La síntesis BTC de seis estudios, frecuencias y Tabla1 de Miura2019 están en "
            "book/_static/literature_e1/recovery_20261105/. El catálogo vigente de39 referencias y la "
            "ampliación metodológica T1/T2/T3 están en journal_update_20261105/: artículo Bucci leído, "
            "código HARNet verificado e identidad DeepVol con original pendiente. La indexación de "
            "revistas es secundaria; no se acreditan registros individuales o Q1. No sumar fuentes "
            "metodológicas al denominador BTC. No se incluyen manuscritos de terceros.",
        )
    counts = stage["counts"]
    total = counts["crypto_full_text_studies"]
    common = (
        f"La etapa cripto conserva {counts['bibliographic_references']} referencias y {total} estudios "
        f"con texto completo: {counts['BTC_full_text_studies']} incluyen Bitcoin y "
        f"{counts['studies_with_non_BTC_crypto']} incluye también otras criptomonedas; estos "
        "subconjuntos se superponen. Las frecuencias de comparadores independientes son "
        f"MLP {counts['MLP_baseline_studies']}/{total}, CNN genérica "
        f"{counts['CNN_generic_baseline_studies']}/{total} y CNN1D explícita "
        f"{counts['CNN1D_explicit_baseline_studies']}/{total}. S9 añade una LSTM propuesta, "
        "sin contarla como otro baseline. Bucci conserva su original financiero leído; HARNet "
        "aporta solo código primario con un erratum explícito y DeepVol mantiene identidad "
        "editorial en CV, con original pendiente. Esas transferencias no aumentan las frecuencias "
        "cripto ni revelan la dimensionalidad no reportada de la CNN de Miura. Las recetas propias "
        "siguen siendo adaptaciones. "
    )
    detail = (
        f"El catálogo vigente de {counts['bibliographic_references']} referencias, la síntesis de "
        f"{total} textos cripto, sus frecuencias y la inclusión S9 están en "
        f"book/_static/literature_e1/{CRYPTO_STAGE}/. De ellos, "
        f"{counts['BTC_full_text_studies']} incluyen BTC y "
        f"{counts['studies_with_non_BTC_crypto']} incluye además otras criptomonedas; no se suman "
        "esos subconjuntos como estudios distintos. S9 presenta una LSTM propuesta, no un nuevo "
        "comparador baseline: MLP y CNN genérica permanecen en "
        f"{counts['MLP_baseline_studies']}/{total} y "
        f"{counts['CNN_generic_baseline_studies']}/{total}; CNN1D explícita queda en "
        f"{counts['CNN1D_explicit_baseline_studies']}/{total}. Los cortes anteriores de seis textos "
        "BTC y39 referencias se conservan como historia. Bucci sigue como original financiero "
        "leído, HARNet como código de autor corregido mediante erratum y DeepVol como identidad "
        "editorial con manuscrito original pendiente. Las listas de revistas son evidencia "
        "secundaria, sin acreditar registros individuales o Q1. La carpeta nueva se limita a la "
        "metadata enumerada y verificada por manifest.json; no incluye manuscritos de terceros."
    )
    return common, detail


def report_files():
    build = ROOT / "book/_build/html"
    if not (build / "intro.html").exists():
        raise ValueError("Build the book before packaging the report")
    # Historical public artifacts are not copied into the new distribution.
    excluded = set()
    sensitive_summaries = {
        "_static/eda/results/summary.json", "_static/eda_temporal_reference/summary.json"
    }
    summary_hashes = {digest((build / name).read_bytes()) for name in sensitive_summaries}
    for path in build.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(build).as_posix()
        if (relative.startswith("_static/e1_downloads/")
                or path.name in EXCLUDED_NAMES or path.suffix.lower() in ARCHIVE_OR_WEIGHTS
                or relative in sensitive_summaries
                or (path.name == "summary.json" and digest(path.read_bytes()) in summary_hashes)):
            excluded.add(relative)
    files = {}
    rewritten = 0
    for path in sorted(build.rglob("*")):
        if not path.is_file() or path.relative_to(build).as_posix() in excluded:
            continue
        if path.is_symlink():
            raise ValueError(f"Unexpected report symlink: {path}")
        relative = path.relative_to(build).as_posix()
        data = path.read_bytes()
        if path.suffix == ".html":
            def link(match):
                nonlocal rewritten
                parts = urlsplit(match.group(2))
                if parts.scheme or parts.netloc:
                    return match.group(0)
                target = (path.parent / unquote(parts.path)).resolve()
                if target.is_relative_to(build.resolve()):
                    name = target.relative_to(build.resolve()).as_posix()
                    if name in excluded:
                        rewritten += 1
                        suffix = ("#" + parts.fragment) if parts.fragment else ""
                        return match.group(1) + PUBLIC_BOOK + name + suffix + match.group(3)
                return match.group(0)
            data = re.sub(r'((?:href|src)=[\"\'])([^\"\']+)([\"\'])', link, data.decode()).encode()
        files["informe/" + relative] = data
    files["ARTEFACTOS_EXTERNOS.json"] = encode({
        "scope": "Rendered report, figures and aggregate attachments; individual data/parameters and the download packages themselves are external links only, avoiding recursive archives",
        "excluded_from_this_archive": sorted(excluded), "rewritten_links": rewritten,
        "external_math_dependency": "MathJax from its existing CDN may need an Internet connection",
    })
    return files


def verify_report(files):
    class Page(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids, self.links = set(), []
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if "id" in attrs:
                self.ids.add(attrs["id"])
            self.links += [attrs[key] for key in ("href", "src") if key in attrs]
    pages = {}
    import posixpath
    for name, data in files.items():
        if name.endswith(".html") and "/_static/" not in name:
            parser = Page()
            parser.feed(data.decode())
            pages[name] = parser
    checked = 0
    for name, page in pages.items():
        for link in page.links:
            parts = urlsplit(link)
            if parts.scheme or parts.netloc:
                continue
            target = posixpath.normpath(posixpath.join(posixpath.dirname(name), unquote(parts.path))) if parts.path else name
            if target not in files and target + "/index.html" in files:
                target += "/index.html"
            if target not in files:
                raise ValueError(f"Missing report target: {name}: {link}")
            if parts.fragment and target in pages and unquote(parts.fragment) not in pages[target].ids:
                raise ValueError(f"Missing report anchor: {name}: {link}")
            checked += 1
    return {"HTML_pages": len(pages), "local_links_and_assets_checked": checked}


def archive(path, files, item, title, revision, readme):
    files = dict(files)
    files["LEEME.md"] = readme.encode()
    files["MANIFIESTO.json"] = encode({
        "item": item, "title": title, "source_commit": revision,
        "scope": "Public documentation, code, metadata and aggregate results; no new fits or forecasts",
        "files": {name: {"bytes": len(data), "sha256": digest(data)} for name, data in sorted(files.items())},
    })
    with ZipFile(path, "w", compression=ZIP_DEFLATED, compresslevel=6) as output:
        for name, data in sorted(files.items()):
            info = ZipInfo(name, date_time=(2026, 11, 4, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            output.writestr(info, data)
    with ZipFile(path) as saved:
        if saved.testzip() is not None:
            raise ValueError(f"Corrupt archive: {path}")
    return {"item": item, "title": title, "file": path.name, "bytes": path.stat().st_size,
            "sha256": digest(path.read_bytes()), "members": len(files), "url": RAW + path.name}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.is_relative_to(ROOT) or (output.exists() and any(output.iterdir())):
        raise ValueError("Use an empty output directory outside the checkout")
    output.mkdir(parents=True, exist_ok=True)
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip():
        raise ValueError("Commit the source changes before recording the package revision")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    static = ROOT / "book/_static"
    crypto_stage = load_crypto_stage(ROOT)
    packages = {item: {} for item, *_ in ITEMS}
    packages["8.1"] = report_files()
    report_check = verify_report(packages["8.1"])
    tracked = subprocess.check_output(["git", "ls-files", "scripts", "tests"], cwd=ROOT, text=True).splitlines()
    for name in tracked:
        if name.endswith(".py"):
            add_file(packages["8.2"], ROOT / name)
    for path in ROOT.glob("requirements-*.txt"):
        add_file(packages["8.2"], path)
    for name in ["BUILD.md", "DESCARGAS.md", "README.md"]:
        add_file(packages["8.2"], ROOT / name)
    for name in ["data/README.md", "scripts/download_binance_spot.py"]:
        add_file(packages["8.3"], ROOT / name)
    public_tree(packages["8.3"], ROOT / "data/manifests")
    public_tree(packages["8.3"], static / "data_source")
    public_tree(packages["8.4"], static / "literature_e1",
                excluded_directories=(static / "literature_e1" / CRYPTO_STAGE,))
    public_tree(packages["8.4"], static / "e1_audit_improvements_20261104/literature")
    add_file(packages["8.4"], ROOT / "book/references.bib")
    if crypto_stage is not None:
        package_crypto_stage(packages, crypto_stage)
    allowlisted_tree(packages["8.5"], static / "filipe_multihistory_20261104")
    allowlisted_tree(packages["8.5"], static / "e1_audit_improvements_20261104")
    allowlisted_tree(packages["8.5"], static / "e1_filipe_pilot_20261105", "public_manifest.json")
    for name in ["statistical_scope.json", "statistical_scope.md"]:
        add_file(packages["8.5"], static / "e1_deliverables_20261104" / name)
        add_file(packages["8.7"], static / "e1_deliverables_20261104" / name)
    public_tree(packages["8.6"], static / "e1_deliverables_20261104")
    for name in ["runtime.json", "final_fit_summary.csv", "final_models_manifest.json"]:
        add_file(packages["8.6"], static / "filipe_multihistory_20261104" / name)
    public_tree(packages["8.6"], static / "e1_audit_improvements_20261104/traceability")
    for name in ["configuration_catalog.csv", "protocol.json", "runtime.json", "code_manifest_before_fitting.json"]:
        add_file(packages["8.6"], static / "e1_filipe_pilot_20261105" / name)
    for name in ["protocol.json", "freeze.json"]:
        add_file(packages["8.7"], static / "filipe_multihistory_20261104" / name)
    add_file(packages["8.7"], static / "e1_deliverables_20261104/declaration_and_chronology.json")
    for name in ["academic_margin_policy.json", "next_study_pilot_DRAFT.json", "protocol.json"]:
        add_file(packages["8.7"], static / "e1_audit_improvements_20261104" / name)
    for name in ["protocol.json", "preregistration_git_receipt.json", "execution_summary.json", "analysis/summary.json"]:
        add_file(packages["8.7"], static / "e1_filipe_pilot_20261105" / name)
    d7 = static / D7_STUDY
    d7_summary = None
    if d7.exists():
        # Verify the complete export even when a product uses only a subset.
        # The complete source allowlist remains in each subset for provenance;
        # MANIFIESTO.json describes the members actually present in that ZIP.
        allowlisted_tree(packages["8.5"], d7, "public_manifest.json")
        d7_summary = json.loads((d7 / "execution_overview.json").read_text())
        final_analysis = json.loads((d7 / "analysis/summary.json").read_text())
        verification = json.loads((d7 / "verification/revision_filipe_confirmatory_d7_20261105.json").read_text())
        if (d7_summary.get("study_id") != D7_STUDY
                or final_analysis.get("status") != "definitive_conditional_seed_analysis_complete"
                or verification.get("status") != "PASS"
                or verification.get("phase") != "definitive"):
            raise ValueError("Only package the completed and independently verified D7 export")
        roots = {"README.md", "protocol.json", "runtime.json", "code_manifest_before_fitting.json",
                 "preregistration_git_receipt.json", "locked_design.json",
                 "locked_design_publication_receipt.json", "execution_overview.json"}
        phase_records = {phase + "/" + name
                         for phase in ("variance_extension", "definitive")
                         for name in ("execution_summary.json", "run_started.json")}
        configuration_records = roots | phase_records | {
            "configuration_catalog.csv", "planning/variance_summary.json",
            "planning/paired_dimensioning.csv", "analysis/summary.json"}
        chronology_records = roots | phase_records | {
            "planning/variance_summary.json", "analysis/summary.json", "tables/chronology.md"}
        required = configuration_records | chronology_records
        allowlist = json.loads((d7 / "public_manifest.json").read_text())["files_sha256"]
        if not required.issubset(allowlist):
            raise ValueError("Missing registered D7 configuration or chronology records")
        allowlisted_tree(packages["8.6"], d7, "public_manifest.json",
                         include=lambda name: name in configuration_records)
        allowlisted_tree(packages["8.7"], d7, "public_manifest.json",
                         include=lambda name: name in chronology_records or name.startswith("verification/"))
    d7_note = (
        " El estudio D7 registrado se entrega por separado: ampliación de20 semillas por modelo "
        "y definitiva de" + str(d7_summary["definitive"]["seeds_per_model"]) +
        " semillas nuevas por modelo, con R publicado antes de esa fase. Ambas fases se "
        "condicionan al calendario2024 conocido y no reabren el test histórico."
    ) if d7_summary else " El R definitivo del piloto histórico aún no se ejecutó."
    literature_common, literature_detail = literature_readmes(crypto_stage)
    common = (
        "\n\n" + literature_common +
        "Se conservan el suplemento final post-hoc y el piloto histórico registrado de60 trayectorias."
        + d7_note + " La política académica futura no "
        "se presenta como anterior al test. Pesos, escaladores y predicciones individuales "
        "requieren la entrega privada descrita en BUILD.md; este ZIP no los contiene.\n"
        "No se asigna una licencia general al proyecto. El aviso MIT de timeseries-cv se "
        "refiere a ese proyecto; los términos de Binance se consultan en el paquete del dataset.\n"
    )
    notes = {
        "8.1": "Abrir informe/intro.html después de descomprimir. Alternativa: `python -m http.server 8000 --directory informe`. MathJax y los enlaces externos pueden necesitar Internet. ARTEFACTOS_EXTERNOS.json identifica las descargas individuales históricas no incluidas.",
        "8.2": "El código conserva scripts/ y tests/. Leer BUILD.md: entornos separados, adquisición, preparación, entrenamiento y análisis. Para recalcular la ejecución original se necesita la evidencia privada; no ejecutar los originales sobre un registro firmado ni borrar sus marcadores.",
        "8.3": "Leer data/README.md. El script scripts/download_binance_spot.py obtiene los archivos oficiales y verifica checksums. Los manifiestos identifican fuente, fechas, bytes y SHA-256. Este paquete no contiene datos brutos de Binance.",
        "8.4": literature_detail,
        "8.5": "Abrir los CSV con un lector compatible con UTF-8. analysis_cv/ contiene desarrollo; test/analysis/ contiene test. Los suplementos inference/ y descriptive/ conservan su registro post-hoc. e1_filipe_pilot_20261105/ contiene el nuevo piloto sobre validación2024 y su dimensionamiento condicional. Los manifiestos verifican los agregados originales.",
        "8.6": "Los locks y los catálogos identifican 1.440 segmentos modelo-mes de CV original,120 ajustes finales y720 segmentos del nuevo piloto. Las filas son metadatos, sin pesos ni valores de escaladores. No mezclar el entorno CV con el de entrenamiento.",
        "8.7": "Leer declaration_and_chronology.json y statistical_scope.json: distinguen evidencia histórica exploratoria, piloto condicional y requisitos futuros no registrados ni ejecutados. Consultar la congelación anterior al test, la política futura y el registro del nuevo piloto e1_filipe_pilot_20261105. Este piloto fue registrado antes de sus propios ajustes, después del benchmark histórico. next_study_pilot_DRAFT.json conserva el borrador de otra evaluación independiente futura, no ejecutada.",
    }
    if d7_summary:
        notes["8.5"] += " " + D7_STUDY + "/ reúne todos los agregados, el dimensionamiento, las figuras, los recibos y las auditorías enumerados en su public_manifest.json."
        notes["8.6"] += " " + D7_STUDY + "/ contiene además el catálogo público por fase de " + str(d7_summary["configuration_records"]) + " configuraciones D7, con hashes y semillas, sin valores de escaladores o pesos."
        notes["8.7"] += " " + D7_STUDY + "/ conserva protocolo, R fijado, ambas publicaciones previas, resúmenes de fases, cronología y auditorías. Su allowlist de procedencia es completa; el ZIP8.5 entrega todos sus archivos."
    if crypto_stage is not None:
        notes["8.7"] += (" La etapa " + CRYPTO_STAGE + "/ añade summary.json, el erratum de HARNet "
                          "y manifest.json; el ZIP8.4 contiene su metadata pública completa por allowlist.")
    license_path = static / "licenses/timeseries-cv-LICENSE.txt"
    records = []
    for item, filename, title, _ in ITEMS:
        add_file(packages[item], ROOT / "BUILD.md", "BUILD.md")
        add_file(packages[item], license_path, "AVISOS/timeseries-cv-LICENSE.txt")
        readme = f"# {item}. {title}\n\nFuente versionada: {revision}.\n\n{notes[item]}{common}"
        records.append(archive(output / filename, packages[item], item, title, revision, readme))
    combined = {record["file"]: (output / record["file"]).read_bytes() for record in records}
    combined["CATALOGO_PRODUCTOS.json"] = encode(records)
    complete = archive(output / "entregable1_completo.zip", combined, "8.1–8.7", "Entregable 1 completo público", revision,
                       "# Entregable 1: siete productos\n\nDescomprimir este paquete y después cada ZIP según el ítem. "
                       "Cada producto tiene LEEME.md y MANIFIESTO.json con hashes.\n" + common)
    catalog = {"source_commit": revision, "packages": records, "complete_package": complete,
               "report_verification": report_check, "new_model_fits": 0, "new_forecasts": 0,
               "raw_market_archive_reads": 0,
               "limits": "Public deliverables only; private evidence is not redistributed; scientific gaps are preserved"}
    if d7_summary:
        catalog["registered_d7_study"] = d7_summary
        catalog["execution_counter_scope"] = "Zero counters describe packaging only; completed study fits are recorded in registered_d7_study by phase"
    if crypto_stage is not None:
        catalog["literature_crypto_extension"] = {
            "summary": crypto_stage["summary"],
            "source_sha256": crypto_stage["source_sha256"],
        }
    (output / "catalogo.json").write_bytes(encode(catalog))
    paths = sorted(output.glob("*.zip")) + [output / "catalogo.json"]
    (output / "SHA256SUMS.txt").write_text("".join(f"{digest(path.read_bytes())}  {path.name}\n" for path in paths))
    (output / "README.md").write_text((ROOT / "DESCARGAS.md").read_text())
    print(json.dumps({"packages": len(records), "complete_bytes": complete["bytes"], "source_commit": revision,
                      "report_verification": report_check, "output_dir": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
