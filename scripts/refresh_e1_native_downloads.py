"""Refresh E1 downloads with reviewed evidence and a post-hoc statistics supplement.

After committing the source and rebuilding ``jupyter-book build book``, run::

    python scripts/refresh_e1_native_downloads.py --output-dir /tmp/e1-native-downloads

The destination must be empty and outside this checkout. This command uses the
committed current ZIPs as its baseline, not the historical package generator.
It reads public documentation, code and explicitly allowlisted evidence only;
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
import struct
import subprocess
import zlib
from zipfile import ZipFile

import package_e1_downloads as legacy


ROOT = Path(__file__).resolve().parents[1]
ITEMS = {item: filename for item, filename, *_ in legacy.ITEMS}
PRESERVED_ITEMS = frozenset({"8.3"})
STAGES = {
    "native_search_20261106": {
        "README.md", "source_manifest.json", "summary.json", "records.csv",
        "occurrences.csv", "deduplication.csv", "screening.csv", "review_queue.csv", "flow.svg",
    },
    "primary_reviews_20261106": {
        "README.md", "source_manifest.json", "summary.json", "evidence_matrix.json",
        "evidence_matrix.csv", "metrics.csv", "references.bib",
    },
    "editorial_screening_20261106": {
        "README.md", "decisions.csv", "summary.json", "identity_review.csv", "flow.svg",
    },
    "selection_synthesis_20261106": {
        "README.md", "decision_matrix.csv", "search_evidence_ledger.csv", "summary.json",
    },
}
# These stages contain reviewed nested metadata. Their committed manifests must
# explicitly approve every member; unlisted neighbouring uploads are never read.
MANIFEST_STAGES = (
    "search_recovery_20261106", "internet_closure_20261106",
    "open_access_replacement_20261107",
)
PUBLICATION_SCHEMA = "e1_public_metadata_allowlist_v1"
SUPPLEMENTS = {
    "dataset_comparison": {
        "directory": "book/_static/e1_dataset_comparison_20261106",
        "manifest": "manifest.json", "item": "8.7",
        "files": {"README.md", "comparison.csv", "margin_sensitivity.csv", "summary.json"},
    },
    "statistics": {
        "directory": "book/_static/e1_statistics_supplement_20261106",
        "manifest": "source_manifest.json", "item": "8.5",
        "files": {"README.md", "summary.json", "friedman_global.json", "pairwise_sensitivity.csv",
                  "qq_diagnostics.csv", "qq_coordinates.csv", "qq_differences.png",
                  "artifact_availability.json"},
    },
    "academic_margin": {
        "directory": "book/_static/e1_academic_margin_20261106",
        "manifest": "source_manifest.json", "item": "8.7",
        "files": {"README.md", "summary.json", "margin_context.csv", "paired_development.csv"},
    },
    "curves": {
        "directory": "book/_static/e1_curve_reproduction_20261106",
        "manifest": "manifest.json", "item": "8.5",
        "files": {"README.md", "summary.json", "reproduction_registration.json", "execution.json",
                  "runtime.json", "fit_summary.csv", "learning_curves_by_run.csv",
                  "learning_curve_summary.csv", "epoch_summary.csv", "artifact_hash_comparison.csv",
                  "learning_curves_final_recipes.png", "learning_curves_seed8101.png"},
    },
}
REPRODUCTION_PROTOCOL_FILES = {
    "experiments/e1_curve_reproduction_20261106/protocol.json",
    "experiments/e1_curve_reproduction_20261106/README.md",
}
CHRONOLOGY_NOTE = (
    "La recuperación bibliográfica y la fundamentación académica del margen son posteriores "
    "a los experimentos iniciales. La reproducción de curvas tiene un protocolo y registro "
    "propios y realiza nuevos ajustes de desarrollo; no sustituye los originales históricos "
    "ni modifica sus resultados, recetas, selección o cronologías. Los contrastes suplementarios "
    "anteriores conservan su carácter posterior. Esta actualización no constituye un prerregistro "
    "retrospectivo. Empaquetar consume las evidencias guardadas y no ejecuta ajustes o pronósticos."
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PRIVATE_FIELDS = {
    "abstract", "abstracts", "abstracttext", "email", "emails", "emailaddress",
    "authoremail", "authoremails", "correspondence", "correspondenceaddress",
    "rawabstract", "exportedabstract", "abstractbody", "rawfulltext", "originalfulltext",
    "fulltext", "rawhtml", "contactemail", "contactemails",
}
_PRIVATE_BASENAMES = legacy.EXCLUDED_NAMES | {
    "model.pt", "model.pth", "checkpoint.json", "weights.json", "scaler.json",
}
PUBLIC_EVIDENCE_SUFFIXES = frozenset({".csv", ".json", ".bib", ".md", ".svg", ".png"})


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
    """Validate allowlisted metadata and plots, never exported private fields."""
    suffix = PurePosixPath(name).suffix.lower()
    if suffix == ".png":
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Invalid public PNG: " + name)
        offset, chunks = 8, []
        while offset < len(data):
            if offset + 12 > len(data):
                raise ValueError("Truncated public PNG: " + name)
            length = struct.unpack(">I", data[offset:offset + 4])[0]
            end = offset + 12 + length
            if end > len(data):
                raise ValueError("Truncated public PNG: " + name)
            kind = data[offset + 4:offset + 8]
            payload = data[offset + 8:offset + 8 + length]
            checksum = struct.unpack(">I", data[offset + 8 + length:end])[0]
            if zlib.crc32(kind + payload) & 0xffffffff != checksum:
                raise ValueError("Public PNG CRC mismatch: " + name)
            chunks.append(kind)
            offset = end
            if kind == b"IEND":
                if length or offset != len(data):
                    raise ValueError("Trailing data in public PNG: " + name)
                break
        if not chunks or chunks[0] != b"IHDR" or chunks[-1] != b"IEND" or b"IDAT" not in chunks:
            raise ValueError("Incomplete public PNG: " + name)
        return
    if suffix not in PUBLIC_EVIDENCE_SUFFIXES:
        raise ValueError("Unsupported public evidence extension: " + name)
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


def public_stage(directory, expected_names, manifest_name="manifest.json"):
    """Read only reviewed metadata names with hashes in a committed manifest."""
    directory = Path(directory)
    safe_member(manifest_name)
    manifest_data = _regular_file(directory / manifest_name, ROOT)
    manifest = json.loads(manifest_data)
    if "files_sha256" in manifest and "files" in manifest:
        raise ValueError("Ambiguous literature manifest schema: " + str(directory))
    sized = "files" in manifest
    mapping = manifest.get("files" if sized else "files_sha256")
    if expected_names is None:
        approved = manifest.get("approved_public_files")
        if (manifest.get("publication_schema") != PUBLICATION_SCHEMA or not sized
                or not isinstance(approved, list) or not all(isinstance(name, str) for name in approved)
                or approved != sorted(set(approved)) or not isinstance(mapping, dict)
                or set(approved) != set(mapping) or not {"README.md", "summary.json"} <= set(approved)
                or manifest_name in approved):
            raise ValueError("Nested evidence needs an explicit closed public approval manifest: " + str(directory))
        expected_names = set(approved)
    if not isinstance(mapping, dict) or set(mapping) != set(expected_names):
        raise ValueError("Literature manifest must list exactly the approved public files: " + str(directory))
    files = {}
    for name, metadata in sorted(mapping.items()):
        relative = safe_member(name)
        if relative.name in _PRIVATE_BASENAMES:
            raise ValueError("Private experimental payload is not public evidence: " + name)
        if relative.suffix.lower() not in PUBLIC_EVIDENCE_SUFFIXES:
            raise ValueError("Unsupported public evidence extension: " + name)
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
    validate_public_bytes(manifest_name, manifest_data)
    files[manifest_name] = manifest_data
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
            "El producto 8.3 conserva exactamente el ZIP de la entrega base. "
            "8.5 conserva sus archivos científicos originales y añade estadística y curvas "
            "de una reproducción identificada; 8.6 conserva sus configuraciones originales "
            "y añade el protocolo y metadatos de esa reproducción. 8.7 incorpora contexto "
            "documental, margen y procedencia. "
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
    for stage, expected_names in [*STAGES.items(), *((name, None) for name in MANIFEST_STAGES)]:
        directory = ROOT / "book/_static/literature_e1" / stage
        files, info = public_stage(directory, expected_names)
        for name in files:
            if (directory / name).relative_to(ROOT).as_posix() not in tracked:
                raise ValueError("Literature source is not committed: " + stage + "/" + name)
        stages[stage], stage_info[stage] = files, info
    supplements, supplement_info = {}, {}
    for label, spec in SUPPLEMENTS.items():
        directory = ROOT / spec["directory"]
        files, info = public_stage(directory, spec["files"], spec["manifest"])
        for name in files:
            if (directory / name).relative_to(ROOT).as_posix() not in tracked:
                raise ValueError("Supplement source is not committed: " + label + "/" + name)
        supplements[label], supplement_info[label] = files, {**info, "product": spec["item"]}
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
    for supplement_lock in ("requirements-e1-supplement.lock.txt", "requirements-e1-curves.lock.txt"):
        if supplement_lock in tracked:
            data = _regular_file(ROOT / supplement_lock, ROOT)
            products["8.2"][supplement_lock] = data
            if supplement_lock == "requirements-e1-curves.lock.txt":
                products["8.6"][supplement_lock] = data
    for name in sorted(REPRODUCTION_PROTOCOL_FILES):
        if name not in tracked:
            raise ValueError("Reproduction protocol source is not committed: " + name)
        data = _regular_file(ROOT / name, ROOT)
        validate_public_bytes(name, data)
        for item in ("8.6", "8.7"):
            products[item][name] = data
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
    for label, files in supplements.items():
        spec = SUPPLEMENTS[label]
        prefix = spec["directory"] + "/"
        for item in {spec["item"], "8.7"} | ({"8.6"} if label == "curves" else set()):
            products[item] = {name: data for name, data in products[item].items()
                              if not name.startswith(prefix)}
        for name, data in files.items():
            products[spec["item"]][prefix + name] = data
            if name in {"summary.json", "source_manifest.json", "manifest.json", "artifact_availability.json"}:
                products["8.7"][prefix + name] = data
            if label == "curves" and name in {"reproduction_registration.json", "runtime.json", "execution.json"}:
                products["8.6"][prefix + name] = data
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
        "supplements": supplement_info,
        "preserved_archive_sha256": {ITEMS[item]: base_rows[item]["sha256"]
                                     for item in sorted(PRESERVED_ITEMS)},
        "original_experiments_changed": False, "retrospective_preregistration": False,
        "new_analysis_scope": "Reviewed source recovery and development-only margin context; separately registered curve reproduction. Packaging itself performs no training or daily forecasts.",
        "chronology_note": CHRONOLOGY_NOTE,
    }
    products["8.7"]["literature_refresh_20261106.json"] = legacy.encode(refresh)
    products["8.7"]["AMPLIACION_LITERARIA_20261106.md"] = (
        "# Ampliación literaria del 6 de noviembre de 2026\n\n" + CHRONOLOGY_NOTE + "\n\n"
        "El producto 8.4 contiene los metadatos públicos completos de las nuevas etapas; "
        "aquí se conservan sus resúmenes, manifiestos y procedencia. No se redistribuyen "
        "los archivos de búsqueda originales, resúmenes editoriales ni PDF de terceros.\n"
    ).encode()
    products["8.7"]["SUPLEMENTO_EVIDENCIAS_20261106.md"] = (
        "# Evidencias y análisis posteriores del 6 de noviembre de 2026\n\n"
        + CHRONOLOGY_NOTE + "\n\n"
        "8.7 añade la comparación documental y el contexto académico del margen; 8.4 incluye "
        "cribado, recuperación de originales y evidencia de búsquedas con sus límites. "
        "8.5 añade estadística posterior y curvas de una reproducción separada. Su protocolo, "
        "registro y entorno también se incluyen en 8.6. Los resultados y fuentes históricos "
        "no se reescriben; los vectores diarios, pesos y escaladores permanecen fuera de "
        "la entrega pública. Los nuevos ajustes de curvas no son los originales recuperados.\n"
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
    if "presentation_refresh" in result:
        result.setdefault("prior_presentation_refreshes", []).append(result.pop("presentation_refresh"))
    result.update(source_commit=revision, packages=records, complete_package=complete,
                  report_verification=report_check, archive_verification=verifications,
                  literature_refresh=refresh, evidence_refresh=refresh,
                  scientific_limitations_preserved=True,
                  new_model_fits_by_packaging=0, new_checkpoint_forecasts_by_packaging=0,
                  raw_market_archive_reads_by_packaging=0, publications_by_packaging=0)
    (output / "catalogo.json").write_bytes(legacy.encode(result))
    (output / "README.md").write_text(
        "# Descargas vigentes del Entregable 1\n\nLos siete productos conservan los nombres "
        "de la unidad 8. La descarga completa reúne esos siete ZIP y su catálogo.\n\n"
        + CHRONOLOGY_NOTE + "\n\nEl ZIP 8.3 es idéntico a la entrega base. "
        "8.5 y 8.6 conservan sus archivos científicos originales y añaden evidencia y "
        "configuración de la reproducción identificada; 8.7 incorpora comparación documental, "
        "margen y procedencia. "
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
