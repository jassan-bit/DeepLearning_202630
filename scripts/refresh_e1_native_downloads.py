"""Refresh E1 literature downloads while retaining the current scientific products.

After committing the source and rebuilding ``jupyter-book build book``, run::

    python scripts/refresh_e1_native_downloads.py --output-dir /tmp/e1-native-downloads

The destination must be empty and outside this checkout. This command uses the
committed current ZIPs as its baseline, not the historical package generator.
It reads public documentation, code and explicitly allowlisted literature only;
it does not train, forecast, read market archives, or publish any files.
"""
from __future__ import annotations

import argparse
import copy
import csv
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
from zipfile import ZipFile

import package_e1_downloads as legacy


ROOT = Path(__file__).resolve().parents[1]
ITEMS = {item: filename for item, filename, *_ in legacy.ITEMS}
PRESERVED_ITEMS = frozenset({"8.3", "8.5", "8.6"})
STAGES = {
    "native_search_20261106": {
        "README.md", "source_manifest.json", "summary.json", "records.csv",
        "occurrences.csv", "deduplication.csv", "screening.csv", "review_queue.csv", "flow.svg",
    },
    "primary_reviews_20261106": {
        "README.md", "source_manifest.json", "summary.json", "evidence_matrix.json",
        "evidence_matrix.csv", "metrics.csv", "references.bib",
    },
}
CHRONOLOGY_NOTE = (
    "La ampliación bibliográfica del 6 de noviembre de 2026 es posterior a los "
    "experimentos documentados. No constituye un prerregistro retrospectivo, no "
    "modifica sus configuraciones, resultados ni cronologías y no añade ajustes "
    "o pronósticos. Los estratos históricos de literatura conservan su identidad."
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PRIVATE_FIELDS = {
    "abstract", "abstracts", "abstracttext", "email", "emails", "emailaddress",
    "authoremail", "authoremails", "correspondence", "correspondenceaddress",
}


def safe_member(name):
    """Accept canonical relative POSIX filenames, never paths to extract."""
    if not isinstance(name, str):
        raise ValueError("Archive/allowlist path must be text")
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or ".." in path.parts or "\\" in name
            or ":" in name or any(ord(char) < 32 for char in name)
            or path.as_posix() != name or name == "."):
        raise ValueError("Unsafe archive/allowlist path: " + repr(name))
    return path


def verify_archive(path):
    """Validate CRC, canonical paths and the complete member hash manifest."""
    with ZipFile(path) as saved:
        names = saved.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP member: " + str(path))
        for info in saved.infolist():
            safe_member(info.filename)
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("ZIP contains a directory or symlink: " + info.filename)
        if saved.testzip() is not None:
            raise ValueError("ZIP CRC failure: " + str(path))
        manifest = json.loads(saved.read("MANIFIESTO.json"))
        mapping = manifest.get("files")
        if not isinstance(mapping, dict) or set(names) != set(mapping) | {"MANIFIESTO.json"}:
            raise ValueError("ZIP members differ from MANIFIESTO.json: " + str(path))
        for name, metadata in mapping.items():
            safe_member(name)
            if (not isinstance(metadata, dict) or type(metadata.get("bytes")) is not int
                    or metadata["bytes"] < 0
                    or not isinstance(metadata.get("sha256"), str)
                    or not _SHA256.fullmatch(metadata["sha256"])):
                raise ValueError("Invalid ZIP member metadata: " + name)
            data = saved.read(name)
            if len(data) != metadata["bytes"] or legacy.digest(data) != metadata["sha256"]:
                raise ValueError("ZIP member hash mismatch: " + name)
    return {"CRC": "PASS", "member_hashes": "PASS", "safe_paths": "PASS", "members": len(names)}


def _regular_file(path, boundary=ROOT):
    """Reject symlink components and filesystem escapes before reading."""
    path, boundary = Path(path), Path(boundary)
    if not path.is_relative_to(boundary):
        raise ValueError("Source is outside its public directory: " + str(path))
    if not path.resolve().is_relative_to(boundary.resolve()):
        raise ValueError("Source escapes its public directory: " + str(path))
    relative = path.relative_to(boundary)
    if boundary.is_symlink() or any((boundary / Path(*relative.parts[:index])).is_symlink()
                                    for index in range(1, len(relative.parts) + 1)):
        raise ValueError("Symlink in public source: " + str(path))
    if not path.is_file():
        raise ValueError("Expected regular public source: " + str(path))
    return path.read_bytes()


def _field_is_private(name):
    return re.sub(r"[^a-z]", "", str(name).lower()) in _PRIVATE_FIELDS


def validate_public_bytes(name, data):
    """Do not republish exported abstracts or contact fields as metadata."""
    suffix = PurePosixPath(name).suffix.lower()
    text = data.decode("utf-8-sig")
    if suffix == ".csv":
        headers = next(csv.reader(io.StringIO(text)), [])
        if any(_field_is_private(header) for header in headers):
            raise ValueError("Private/abstract field in public CSV: " + name)
    elif suffix == ".json":
        def visit(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if _field_is_private(key):
                        raise ValueError("Private/abstract field in public JSON: " + name)
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)
        visit(json.loads(text))
    elif suffix == ".bib" and re.search(
            r"^\s*(?:abstract|e-?mail|correspondence)\s*=", text, flags=re.I | re.M):
        raise ValueError("Private/abstract field in public BibTeX: " + name)


def public_stage(directory, expected_names):
    """Read only reviewed metadata names with hashes in a committed manifest."""
    directory = Path(directory)
    manifest_data = _regular_file(directory / "manifest.json", ROOT)
    manifest = json.loads(manifest_data)
    if "files_sha256" in manifest and "files" in manifest:
        raise ValueError("Ambiguous literature manifest schema: " + str(directory))
    sized = "files" in manifest
    mapping = manifest.get("files" if sized else "files_sha256")
    if not isinstance(mapping, dict) or set(mapping) != set(expected_names):
        raise ValueError("Literature manifest must list exactly the approved public files: " + str(directory))
    files = {}
    for name, metadata in sorted(mapping.items()):
        relative = safe_member(name)
        expected = metadata.get("sha256") if sized and isinstance(metadata, dict) else metadata
        if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
            raise ValueError("Invalid literature SHA256: " + name)
        data = _regular_file(directory / str(relative), directory)
        if sized and (type(metadata.get("bytes")) is not int or metadata["bytes"] != len(data)):
            raise ValueError("Literature allowlist byte count mismatch: " + name)
        if legacy.digest(data) != expected:
            raise ValueError("Literature allowlist hash mismatch: " + name)
        validate_public_bytes(name, data)
        files[name] = data
    validate_public_bytes("manifest.json", manifest_data)
    files["manifest.json"] = manifest_data
    summary = json.loads(files["summary.json"])
    if not isinstance(summary, dict):
        raise ValueError("Literature summary must be an object: " + str(directory))
    return files, {"directory": directory.relative_to(ROOT).as_posix(),
                   "manifest_sha256": legacy.digest(manifest_data),
                   "public_files": len(files), "summary": summary}


def source_revision(requested=None):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if requested is not None:
        resolved = subprocess.check_output(
            ["git", "rev-parse", "--verify", requested + "^{commit}"], cwd=ROOT, text=True).strip()
        if resolved != revision:
            raise ValueError("--source-commit must identify the current HEAD")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT, text=True).strip()
    if dirty:
        raise ValueError("Commit final source changes before refreshing the downloads")
    return revision


def tracked_files():
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    return {name for name in names if name}


def verify_build(tracked):
    """Require the final E1 build to follow all committed book input edits."""
    intro = ROOT / "book/_build/html/intro.html"
    _regular_file(intro, ROOT)
    inputs = [ROOT / name for name in tracked
              if name.startswith("book/") and not name.startswith(
                  ("book/_build/", "book/_static/e1_downloads/"))]
    if inputs and intro.stat().st_mtime_ns < max(path.stat().st_mtime_ns for path in inputs):
        raise ValueError("Rebuild E1 after the final book source changes: jupyter-book build book")


def baseline_products(directory):
    """Authenticate every baseline product before reusing any member."""
    raw = _regular_file(directory / "catalogo.json", ROOT)
    catalog = json.loads(raw)
    rows = catalog.get("packages")
    if (not isinstance(rows, list) or len(rows) != len(ITEMS)
            or {row.get("item") for row in rows} != set(ITEMS)):
        raise ValueError("Baseline catalog must contain exactly the seven E1 products")
    rows = {row["item"]: row for row in rows}
    content, verifications = {}, {}
    for item, filename in ITEMS.items():
        row = rows[item]
        if row.get("file") != filename:
            raise ValueError("Unexpected baseline product filename: " + item)
        data = _regular_file(directory / filename, ROOT)
        if len(data) != row.get("bytes") or legacy.digest(data) != row.get("sha256"):
            raise ValueError("Baseline ZIP does not match catalog: " + filename)
        verification = verify_archive(directory / filename)
        if verification["members"] != row.get("members"):
            raise ValueError("Baseline ZIP member count differs from catalog: " + filename)
        with ZipFile(io.BytesIO(data)) as saved:
            content[item] = {name: saved.read(name) for name in saved.namelist()}
        verifications[filename] = verification
    return catalog, rows, content, verifications, legacy.digest(raw)


def _readme(item, title, revision, previous):
    introduction = (
        "Abrir `informe/intro.html` después de descomprimir. Los enlaces a descargas "
        "históricas excluidas se dirigen al libro público; no se incluyen ZIP recursivos.\n\n"
        if item == "8.1" else ""
    )
    return (f"# {item}. {title}\n\nFuente de esta actualización: `{revision}`.\n\n"
            + introduction + CHRONOLOGY_NOTE + "\n\n"
            "Los productos 8.3, 8.5 y 8.6 conservan exactamente los ZIP de la entrega base. "
            "Los manifiestos internos identifican los bytes realmente incluidos.\n\n"
            "## Notas conservadas de la entrega base\n\n" + previous.decode("utf-8"))


def package(output_dir, requested_revision=None):
    output = Path(output_dir).expanduser().resolve()
    if output.is_relative_to(ROOT.resolve()) or (output.exists()
            and (not output.is_dir() or any(output.iterdir()))):
        raise ValueError("Use an empty output directory outside the checkout")
    revision = source_revision(requested_revision)
    tracked = tracked_files()
    verify_build(tracked)
    base = ROOT / "book/_static/e1_downloads"
    catalog, base_rows, products, _, base_catalog_sha = baseline_products(base)
    stages, stage_info = {}, {}
    for stage, expected_names in STAGES.items():
        directory = ROOT / "book/_static/literature_e1" / stage
        files, info = public_stage(directory, expected_names)
        for name in files:
            if (directory / name).relative_to(ROOT).as_posix() not in tracked:
                raise ValueError("Literature source is not committed: " + stage + "/" + name)
        stages[stage], stage_info[stage] = files, info
    # The historical generator is used only for its HTML exclusion/link logic.
    # Do not call its main(), load_crypto_stage(), or any experimental exporter.
    report = legacy.report_files()
    report_check = legacy.verify_report(report)
    for name in report:
        safe_member(name)
        if PurePosixPath(name).suffix.lower() in legacy.ARCHIVE_OR_WEIGHTS:
            raise ValueError("Nested archive/weights in the rendered report: " + name)
    original_readmes = {item: files["LEEME.md"] for item, files in products.items()}
    for item, files in products.items():
        if item not in PRESERVED_ITEMS:
            files.pop("LEEME.md")
            files.pop("MANIFIESTO.json")
    products["8.1"] = report
    # Retain original configuration/protocol members, but never ship obsolete code.
    products["8.2"] = {name: data for name, data in products["8.2"].items()
                       if not name.startswith(("scripts/", "tests/"))}
    for name in sorted(tracked):
        if name.startswith(("scripts/", "tests/")) and name.endswith(".py"):
            safe_member(name)
            products["8.2"][name] = _regular_file(ROOT / name, ROOT)
    for name in ("BUILD.md", "README.md"):
        if name not in tracked:
            raise ValueError("Required source is not committed: " + name)
        products["8.2"][name] = _regular_file(ROOT / name, ROOT)
    if not {"book/references.bib", "book/_static/literature_e1/README.md"}.issubset(tracked):
        raise ValueError("Bibliography and literature index must be committed")
    bibliography = _regular_file(ROOT / "book/references.bib", ROOT)
    validate_public_bytes("book/references.bib", bibliography)
    products["8.4"]["book/references.bib"] = bibliography
    literature_readme = "book/_static/literature_e1/README.md"
    products["8.4"][literature_readme] = _regular_file(ROOT / literature_readme, ROOT)
    for stage, files in stages.items():
        prefix = "book/_static/literature_e1/" + stage + "/"
        for item in ("8.4", "8.7"):
            products[item] = {name: data for name, data in products[item].items()
                              if not name.startswith(prefix)}
        for name, data in files.items():
            products["8.4"][prefix + name] = data
            if name in {"summary.json", "source_manifest.json", "manifest.json"}:
                products["8.7"][prefix + name] = data
    # User-confirmed query families and year filters are a separate, reproducible
    # input to the importer; they must travel with its code and derived ledger.
    context = stage_info["native_search_20261106"]["summary"].get("search_context")
    if context:
        context_name = "references/literature_e1/native_search_context_20261106.json"
        if context_name not in tracked:
            raise ValueError("The confirmed search context must be committed")
        context_data = _regular_file(ROOT / context_name, ROOT)
        if legacy.digest(context_data) != context["sha256"]:
            raise ValueError("Search context differs from the imported provenance")
        validate_public_bytes(context_name, context_data)
        for item in ("8.2", "8.4", "8.7"):
            products[item][context_name] = context_data
    refresh = {
        "date": "2026-11-06", "source_commit": revision,
        "baseline_source_commit": catalog.get("source_commit"),
        "baseline_catalog_sha256": base_catalog_sha,
        "stages": stage_info,
        "preserved_archive_sha256": {ITEMS[item]: base_rows[item]["sha256"]
                                     for item in sorted(PRESERVED_ITEMS)},
        "original_experiments_changed": False, "retrospective_preregistration": False,
        "chronology_note": CHRONOLOGY_NOTE,
    }
    products["8.7"]["literature_refresh_20261106.json"] = legacy.encode(refresh)
    products["8.7"]["AMPLIACION_LITERARIA_20261106.md"] = (
        "# Ampliación literaria del 6 de noviembre de 2026\n\n" + CHRONOLOGY_NOTE + "\n\n"
        "El producto 8.4 contiene los metadatos públicos completos de las nuevas etapas; "
        "aquí se conservan sus resúmenes, manifiestos y procedencia. No se redistribuyen "
        "los archivos de búsqueda originales, resúmenes editoriales ni PDF de terceros.\n"
    ).encode()
    # Nothing is written until all source metadata and baseline archives pass.
    output.mkdir(parents=True, exist_ok=True)
    records, verifications = [], {}
    for item, filename in ITEMS.items():
        row = copy.deepcopy(base_rows[item])
        if item in PRESERVED_ITEMS:
            shutil.copyfile(base / filename, output / filename)
            if legacy.digest((output / filename).read_bytes()) != row["sha256"]:
                raise ValueError("Preserved archive changed while copying: " + filename)
        else:
            for name in products[item]:
                safe_member(name)
            row.update(legacy.archive(output / filename, products[item], item, row["title"], revision,
                                      _readme(item, row["title"], revision, original_readmes[item])))
        verifications[filename] = verify_archive(output / filename)
        row["url"] = legacy.PUBLIC_BOOK + "_static/e1_downloads/" + filename
        records.append(row)
    combined = {row["file"]: (output / row["file"]).read_bytes() for row in records}
    combined["CATALOGO_PRODUCTOS.json"] = legacy.encode(records)
    complete = legacy.archive(output / "entregable1_completo.zip", combined, "8.1–8.7",
        catalog["complete_package"]["title"], revision,
        "# Entregable 1: siete productos vigentes\n\nDescomprimir este ZIP y después cada producto. "
        "Cada producto conserva su LEEME.md y MANIFIESTO.json.\n\n" + CHRONOLOGY_NOTE + "\n")
    complete["url"] = legacy.PUBLIC_BOOK + "_static/e1_downloads/" + complete["file"]
    verifications[complete["file"]] = verify_archive(output / complete["file"])
    result = copy.deepcopy(catalog)
    result.update(source_commit=revision, packages=records, complete_package=complete,
                  report_verification=report_check, archive_verification=verifications,
                  literature_refresh=refresh, scientific_limitations_preserved=True,
                  new_model_fits_by_packaging=0, new_checkpoint_forecasts_by_packaging=0,
                  raw_market_archive_reads_by_packaging=0, publications_by_packaging=0)
    (output / "catalogo.json").write_bytes(legacy.encode(result))
    (output / "README.md").write_text(
        "# Descargas vigentes del Entregable 1\n\nLos siete productos conservan los nombres "
        "de la unidad 8. La descarga completa reúne esos siete ZIP y su catálogo.\n\n"
        + CHRONOLOGY_NOTE + "\n\nLos ZIP 8.3, 8.5 y 8.6 son idénticos a la entrega base. "
        "`catalogo.json` conserva modelos, fases, entrypoints y limitaciones científicas. "
        "Verificar integridad con `sha256sum -c SHA256SUMS.txt`; los manifiestos internos "
        "verifican bytes y SHA256 por miembro.\n", encoding="utf-8")
    paths = sorted(output.glob("*.zip")) + [output / "catalogo.json", output / "README.md"]
    (output / "SHA256SUMS.txt").write_text(
        "".join(f"{legacy.digest(path.read_bytes())}  {path.name}\n" for path in paths), encoding="utf-8")
    if source_revision(revision) != revision:
        raise ValueError("Source revision changed during packaging")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", help="Current source commit; defaults to git HEAD")
    args = parser.parse_args()
    catalog = package(args.output_dir, args.source_commit)
    print(json.dumps({"source_commit": catalog["source_commit"], "products": len(catalog["packages"]),
                      "output_dir": str(args.output_dir.resolve()), "preserved_products": sorted(PRESERVED_ITEMS),
                      "archive_verification": "CRC, member hashes and safe paths PASS",
                      "new_model_fits": 0, "new_forecasts": 0, "raw_market_archive_reads": 0},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
