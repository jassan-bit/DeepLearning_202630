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

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_BOOK = "https://jassan-bit.github.io/DeepLearning_202630/"
RAW = "https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/entregables/"
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


def public_tree(files, directory):
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            if path.name in EXCLUDED_NAMES:
                raise ValueError(f"Unexpected individual evidence in public tree: {path}")
            add_file(files, path)


def allowlisted_tree(files, directory):
    manifest = directory / "public_allowlist_manifest.json"
    record = json.loads(manifest.read_text())
    for name, expected in record["files_sha256"].items():
        path = directory / name
        data = path.read_bytes()
        if digest(data) != expected:
            raise ValueError(f"Changed public aggregate: {path}")
        add_file(files, path)
    add_file(files, manifest)


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
    public_tree(packages["8.4"], static / "literature_e1")
    public_tree(packages["8.4"], static / "e1_audit_improvements_20261104/literature")
    add_file(packages["8.4"], ROOT / "book/references.bib")
    allowlisted_tree(packages["8.5"], static / "filipe_multihistory_20261104")
    allowlisted_tree(packages["8.5"], static / "e1_audit_improvements_20261104")
    public_tree(packages["8.6"], static / "e1_deliverables_20261104")
    for name in ["runtime.json", "final_fit_summary.csv", "final_models_manifest.json"]:
        add_file(packages["8.6"], static / "filipe_multihistory_20261104" / name)
    public_tree(packages["8.6"], static / "e1_audit_improvements_20261104/traceability")
    for name in ["protocol.json", "freeze.json"]:
        add_file(packages["8.7"], static / "filipe_multihistory_20261104" / name)
    add_file(packages["8.7"], static / "e1_deliverables_20261104/declaration_and_chronology.json")
    for name in ["academic_margin_policy.json", "next_study_pilot_DRAFT.json", "protocol.json"]:
        add_file(packages["8.7"], static / "e1_audit_improvements_20261104" / name)
    common = (
        "\n\nSe conserva el alcance científico del libro: Miura2019 respalda MLP/CNN independientes "
        "como familias; no declara dimensionalidad CNN ni demuestra una réplica exacta o uso amplio. "
        "suplemento final post-hoc y piloto futuro sin ejecutar. La política académica futura no "
        "se presenta como anterior al test. Pesos, escaladores y predicciones individuales "
        "requieren la entrega privada descrita en BUILD.md; este ZIP no los contiene.\n"
        "No se asigna una licencia general al proyecto. El aviso MIT de timeseries-cv se "
        "refiere a ese proyecto; los términos de Binance se consultan en el paquete del dataset.\n"
    )
    notes = {
        "8.1": "Abrir informe/intro.html después de descomprimir. Alternativa: `python -m http.server 8000 --directory informe`. MathJax y los enlaces externos pueden necesitar Internet. ARTEFACTOS_EXTERNOS.json identifica las descargas individuales históricas no incluidas.",
        "8.2": "El código conserva scripts/ y tests/. Leer BUILD.md: entornos separados, adquisición, preparación, entrenamiento y análisis. Para recalcular la ejecución original se necesita la evidencia privada; no ejecutar los originales sobre un registro firmado ni borrar sus marcadores.",
        "8.3": "Leer data/README.md. El script scripts/download_binance_spot.py obtiene los archivos oficiales y verifica checksums. Los manifiestos identifican fuente, fechas, bytes y SHA-256. Este paquete no contiene datos brutos de Binance.",
        "8.4": "La síntesis vigente de seis estudios, frecuencias y Tabla1 de Miura2019 están en book/_static/literature_e1/recovery_20261105/. La bibliografía principal reúne36referencias. Los archivos anteriores conservan sus cortes históricos; no sumar etapas sin deduplicación. No se incluyen manuscritos de terceros.",
        "8.5": "Abrir los CSV con un lector compatible con UTF-8. analysis_cv/ contiene desarrollo; test/analysis/ contiene test. Los suplementos inference/ y descriptive/ conservan su registro post-hoc. Los manifiestos de listas permitidas verifican los agregados originales.",
        "8.6": "Los locks y los catálogos identifican 1.440 segmentos modelo-mes de CV y 120 ajustes finales. Las filas son metadatos, sin pesos ni valores de escaladores. No mezclar el entorno CV con el de entrenamiento.",
        "8.7": "Leer declaration_and_chronology.json, la congelación anterior al test y la política futura. Los registros conservan su momento real; next_study_pilot_DRAFT.json es un borrador, no un piloto ejecutado.",
    }
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
    (output / "catalogo.json").write_bytes(encode(catalog))
    paths = sorted(output.glob("*.zip")) + [output / "catalogo.json"]
    (output / "SHA256SUMS.txt").write_text("".join(f"{digest(path.read_bytes())}  {path.name}\n" for path in paths))
    (output / "README.md").write_text((ROOT / "DESCARGAS.md").read_text())
    print(json.dumps({"packages": len(records), "complete_bytes": complete["bytes"], "source_commit": revision,
                      "report_verification": report_check, "output_dir": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
