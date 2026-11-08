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
import math
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
    "open_access_replacement_20261107", "provisional_native_log_20261107",
    "homogeneous_synthesis_20261107", "native_execution_20261107",
    "free_model_expansion_20261107", "homogeneous_expansion_20261107",
    "core_selection_20261107", "native_620_screening_20261108",
)
PUBLICATION_SCHEMA = "e1_public_metadata_allowlist_v1"
REFRESH_DATE = "2026-11-08"
REFRESH_TIMEZONE = "UTC"
# The UTC edition date is distinct from the existing run's Colombia date.
# Preserve every earlier edition and scientific execution record unchanged.
REFRESH_ID = "20261108_evidence_closure"
REMEDIATION_STUDY = "e1_608_remediation_20261107"
REMEDIATION_SOURCE = "98e0cbcf4bb09dba863289261677dd6e7534d259"
NATIVE_SCREENING_SOURCE_ALIASES = {
    **{"review_inputs/" + name: "references/reviews/native_620_screening_20261108/" + name
       for name in ("protocol.json", "completion.json", "primary_reviews.json", "decisions_chunk1.json",
                    "decisions_chunk2.json", "decisions_chunk3.json", "base_bibliography.bib")},
    "review_inputs/prior_adjudications.json":
        "references/literature_e1/homogeneous_expansion_20261107/adjudications.json",
}
REMEDIATION_PUBLIC_FILES = frozenset({
    "README.md", "summary.json", "registration.json", "source_manifest.json", "selection.json",
    "plan.json", "plan_publication.json", "chronology.json", "runtime.json", "hpo_summary.csv",
    "hpo_run_metrics.csv", "pilot_metrics.csv", "final_fit_summary.csv", "architecture_summary.csv",
    "learning_curves_by_run.csv", "learning_curve_summary.csv", "learning_curves.png",
    "metric_summary.csv", "run_metrics.csv", "pairwise_rmse.csv", "dm_pairs.csv",
    "temporal_bootstrap.csv", "diagnostics.csv", "analysis_summary.json", "friedman_rmse.csv",
    "wilcoxon_rmse.csv", "qq_data.csv", "qq_rmse_differences.png", "first_seed_learning_curves.csv",
    "first_seed_learning_curves.png", "run_costs.csv", "cost_summary.csv", "pilot_summary.csv",
})
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
        "configuration_files": {"reproduction_registration.json", "runtime.json", "execution.json"},
    },
    "replay_forensics": {
        "directory": "book/_static/e1_replay_forensics_20261107",
        "manifest": "public_allowlist_manifest.json", "item": "8.5", "files": None,
    },
    "replay_repeatability": {
        "directory": "book/_static/e1_replay_repeatability_20261107",
        "manifest": "public_allowlist_manifest.json", "item": "8.5", "files": None,
        "configuration_files": {"registration.json", "runtime.json", "execution.json",
                                "publication_verification.json", "manifest.json",
                                "public_allowlist_manifest.json"},
    },
    "historical_core": {
        "directory": "book/_static/e1_historical_core_20261107",
        "manifest": "manifest.json", "item": "8.5", "files": None,
    },
    "primary_608": {
        "directory": "book/_static/e1_primary_608_20261107",
        "manifest": "manifest.json", "item": "8.5", "files": None,
        "configuration_files": {"summary.json", "execution_provenance.json", "source_manifest.json"},
    },
    "core_mlp_extension": {
        "directory": "book/_static/e1_core_mlp_extension_20261107",
        "manifest": "manifest.json", "item": "8.5", "files": None,
        "configuration_files": {"registration.json", "selection.json", "plan.json", "plan_publication.json",
                                "chronology.json", "runtime.json", "source_manifest.json"},
    },
    "remediation_608": {
        "directory": "book/_static/" + REMEDIATION_STUDY,
        "manifest": "manifest.json", "item": "8.5", "files": None,
        "configuration_files": {"registration.json", "selection.json", "plan.json", "plan_publication.json",
                                "chronology.json", "runtime.json", "source_manifest.json", "architecture_summary.csv"},
    },
    "frozen_error_analysis": {
        "directory": "book/_static/e1_frozen_error_analysis_20261108",
        "manifest": "manifest.json", "item": "8.5", "files": None,
        "configuration_files": {"summary.json", "source_manifest.json", "manifest.json"},
    },
}
REPRODUCTION_PROTOCOL_FILES = {
    "experiments/e1_curve_reproduction_20261106/protocol.json",
    "experiments/e1_curve_reproduction_20261106/README.md",
    "experiments/e1_replay_audit_20261107/protocol.json",
    "experiments/e1_replay_audit_20261107/README.md",
    "experiments/e1_core_mlp_extension_20261107/design.json",
    "experiments/e1_core_mlp_extension_20261107/README.md",
    "experiments/" + REMEDIATION_STUDY + "/design.json",
    "experiments/" + REMEDIATION_STUDY + "/README.md",
}
MANUAL_REACQUISITION_FILES = {"notebooks/descargar_datos_originales_e1.ipynb", "notebooks/descargar_datos_608_e1.ipynb"}
CHRONOLOGY_NOTE = (
    "La recuperación bibliográfica y la fundamentación académica del margen son posteriores "
    "a los experimentos iniciales. La reproducción de curvas tiene un protocolo y registro "
    "propios y realiza nuevos ajustes de desarrollo; no sustituye los originales históricos "
    "ni modifica sus resultados, recetas, selección o cronologías. Los contrastes suplementarios "
    "anteriores conservan su carácter posterior. La auditoría forense y la repetibilidad actual "
    "tienen alcance separado de la identidad histórica; las búsquedas nuevas conservan su "
    "propia procedencia. La búsqueda web dirigida de trece obras y su conciliación "
    "acumulada conservan los ámbitos, papeles y niveles de lectura; no convierten "
    "la cuota de selección por arquitectura en frecuencia del corpus. "
    "El benchmark principal D7 usa una ejecución nueva GRU32–LSTM32–MLP16, HPO igualado y R calculado desde su piloto, sobre608 fechas históricas conocidas. La fuente y el plan calculado preceden a sus respectivos ajustes, y todos los pesos cierran antes del acceso a precios y scoring. El anterior benchmark de608 fechas y diez semillas conserva sus metadatos íntegros como registro fechado. La extensión LSTM–MLP–GRU de sesenta fechas conserva como "
    "exploratoria la evaluación de septiembre–octubre de 2026 ya observada; "
    "los resultados anteriores de TCN permanecen complementarios y separados. "
    "El cribado de la ronda nativa recibida y el análisis de errores agregado se añaden "
    "después del cierre del benchmark, sin reentrenar, seleccionar otras recetas ni modificar "
    "sus métricas finales. Las comparaciones por segmento son descriptivas y posteriores. "
    "Esta actualización no constituye un prerregistro "
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


def native_screening_closure(files, native_files):
    """Require one finished decision per received group, tied to its source."""
    summary = json.loads(files["summary.json"])
    sources = json.loads(files["source_manifest.json"])
    required = {"stage_id": "native_620_screening_20261108",
        "source_campaign_id": "E1-NATIVE-20261106-COLOMBIA",
        "reviewed_groups": 620, "screened_groups": 620,
        "screening_complete": True, "new_native_search_executions": 0,
        "benchmark_results_changed": False}
    if any(type(summary.get(key)) is not type(value) or summary.get(key) != value
           for key, value in required.items()):
        raise ValueError("Native screening needs actual closed620-group decisions, without altered benchmark results")
    expected_sources = {name: legacy.digest(native_files[name])
                        for name in ("records.csv", "occurrences.csv")}
    if sources.get("native_execution_public_files_sha256") != expected_sources:
        raise ValueError("Native screening source hashes differ from the received campaign")
    occurrences = list(csv.DictReader(io.StringIO(native_files["occurrences.csv"].decode("utf-8-sig"))))
    expected = {row["group_id"] for row in occurrences if row["origin"] == "new_native_export"}
    decisions = list(csv.DictReader(io.StringIO(files["decisions.csv"].decode("utf-8-sig"))))
    actual = [row.get("group_id") for row in decisions]
    if len(expected) != 620 or len(actual) != 620 or len(set(actual)) != 620 or set(actual) != expected:
        raise ValueError("Native screening decisions must cover exactly the620 received groups")
    if any(not row.get("screening_decision", "").strip() or not row.get("reason", "").strip()
           for row in decisions):
        raise ValueError("Every received group needs a documented screening decision and reason")
    return {"stage_id": required["stage_id"], "screened_groups": len(actual),
        "benchmark_results_changed": False,
        "source_files_sha256": expected_sources}


def native_review_source_aliases(files):
    """Rebuild canonical compiler input paths from approved metadata snapshots."""
    sources = json.loads(files["source_manifest.json"]).get("sources_sha256", {})
    result = {}
    for approved, canonical in NATIVE_SCREENING_SOURCE_ALIASES.items():
        safe_member(approved)
        safe_member(canonical)
        if approved not in files or legacy.digest(files[approved]) != sources.get(canonical):
            raise ValueError("Native review canonical aliases need every approved source snapshot with its original hash")
        validate_public_bytes(canonical, files[approved])
        result[canonical] = files[approved]
    return result


def frozen_error_closure(files, original_files, current_catalog):
    """Admit post-hoc aggregates only when all original result bytes are pinned."""
    summary = json.loads(files["summary.json"])
    sources = json.loads(files["source_manifest.json"])
    validation = json.loads(files["validation.json"])
    current = current_catalog["current_study"]
    r, n = current["R"], current["n_dates"]
    required = {"supplement_id": "e1_frozen_error_analysis_20261108",
        "parent_study_id": REMEDIATION_STUDY,
        "posthoc_after_original_analysis": True, "original_metrics_unchanged": True,
        "model_fits": 0, "new_checkpoint_forecasts": 0, "raw_market_archive_reads": 0,
        "weights_or_scalers_loaded": 0, "neural_runs_per_model": r,
        "neural_runs": 3*r, "deterministic_reference_runs": 1, "common_dates": n,
        "neural_prediction_rows_read": 3*r*n, "public_daily_observations_or_forecasts": False,
        "test_driven_model_selection": False, "new_confirmatory_hypothesis_tests": False}
    if any(type(summary.get(key)) is not type(value) or summary.get(key) != value
           for key, value in required.items()):
        raise ValueError("Error supplement requires closed post-hoc aggregates without fitting, forecasting or changed final metrics")
    expected_sources = {name: legacy.digest(data) for name, data in original_files.items()}
    if sources.get("original_public_files_sha256") != expected_sources:
        raise ValueError("Error supplement does not preserve every original public result file")
    scientific_sources = json.loads(original_files["source_manifest.json"])["sources_sha256"]
    required_validation = {"original_public_files_verified_per_mirror": len(original_files),
        "registered_scientific_sources_verified": len(scientific_sources),
        "all_original_hashes_unchanged": True, "matched_runs": 3*r+1, "metrics_per_run": 6}
    if any(type(validation.get(key)) is not type(value) or validation.get(key) != value
           for key, value in required_validation.items()):
        raise ValueError("Error supplement lacks its complete frozen-metric/hash validation")
    private_columns = {"outcomeday", "ytrue", "ypred", "persistencepred", "residual",
        "observed", "prediction", "predictions", "rawprices", "weights", "statedict",
        "scaler", "scalers", "masescale", "xmean", "xscale", "yscale"}
    tables, plots = 0, 0
    for name, data in files.items():
        if name.endswith(".csv"):
            fields = next(csv.reader(io.StringIO(data.decode("utf-8-sig"))), [])
            if any(re.sub(r"[^a-z]", "", field.lower()) in private_columns for field in fields):
                raise ValueError("Error supplement cannot contain private daily observations or forecasts")
            tables += 1
        elif name.endswith(".png"):
            plots += 1
    if tables < 1 or plots < 1:
        raise ValueError("Closed error supplement needs actual aggregate tables and plots")
    return {"supplement_id": required["supplement_id"], "parent_study_id": REMEDIATION_STUDY,
        "original_metrics_unchanged": True, "original_public_files_verified": len(original_files),
        "matched_runs": 3*r+1, "common_dates": n, "aggregate_tables": tables, "plots": plots}


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
            "8.5 conserva sus archivos científicos originales y añade el núcleo histórico "
            "compatible y el benchmark principal D7 de 608 fechas, con la ejecución de sesenta fechas complementaria, además de estadística, curvas, "
            "auditoría forense y repetibilidad actual con alcances identificados; 8.6 conserva "
            "sus configuraciones originales y añade los protocolos y metadatos correspondientes. 8.7 incorpora contexto "
            "documental, margen y procedencia. Esta edición añade el cribado de los grupos "
            "nativos recibidos y gráficos y errores agregados del benchmark cerrado, "
            "sin volver a entrenar o puntuar modelos. "
            "Los manifiestos internos identifican los bytes realmente incluidos.\n\n"
            "## Notas conservadas de la entrega base\n\n" + previous.decode("utf-8"))


def complementary_core_catalog(files):
    """Validate the closed 60-date complementary export and preserve its metadata."""
    summary, selection, plan = (json.loads(files[name]) for name in
                                ("summary.json", "selection.json", "plan.json"))
    families = {"GRU", "LSTM", "MLP"}
    signature = summary.get("execution_id")
    r = plan.get("R_final")
    if (set(summary.get("core_models", ())) != families
            or set(selection.get("recipes", {})) != families
            or any(recipe.get("model") != family or recipe.get("history_days") != 7
                   for family, recipe in selection.get("recipes", {}).items())
            or selection.get("baseline_model") not in families
            or summary.get("baseline_selected_before_reserve") != selection["baseline_model"]
            or not isinstance(signature, str) or not _SHA256.fullmatch(signature)
            or any(row.get("registration_signature") != signature for row in (selection, plan))
            or type(r) is not int or r < 465 or plan.get("R") != r
            or summary.get("R_final") != r or summary.get("definitive_trajectories") != 3 * r
            or summary.get("study_id") != "e1_core_mlp_extension_20261107"
            or not re.fullmatch(r"[0-9a-f]{40}", str(summary.get("source_commit", "")))
            or summary.get("known_reserve_exploratory") is not True
            or summary.get("fresh_confirmatory_holdout") is not False):
        raise ValueError("The current core catalog requires one complete, consistently identified GRU/LSTM/MLP export")
    baseline = selection["baseline_model"]
    return {
        "models": ["LSTM", "MLP", "GRU"], "complementary_models": ["TCN"],
        "active_entrypoint": "scripts/run_e1_core_mlp_extension.py",
        "active_entrypoints": {phase: "scripts/run_e1_core_mlp_extension.py"
                               for phase in ("hpo", "pilot", "fit", "score", "analyze")},
        "active_entrypoint_scope": "Current common GRU/LSTM/MLP execution; historical entrypoints are preserved separately",
        "baseline_selected_by_cv": {"model": baseline, **selection["recipes"][baseline]},
        "scope": "Current common LSTM/MLP/GRU development tuning, pilot and final fits; exploratory scoring on an already observed 60-day historical reserve. Previous TCN studies remain complementary.",
        "current_study": {
            "study_id": summary["study_id"], "execution_id": signature,
            "source_commit": summary["source_commit"], "history_days": 7,
            "selected_recipes": selection["recipes"], "definitive_seeds_per_family": r,
            "reserve": {"start": "2026-09-01", "end": "2026-10-30", "days": 60},
            "known_reserve_exploratory": True, "fresh_confirmatory_holdout": False,
            "public_metadata_directory": SUPPLEMENTS["core_mlp_extension"]["directory"],
        },
    }


def current_core_catalog(files, historical_files):
    """Promote only the original D7/608-day scores, with their saved source proof."""
    summary = json.loads(files["summary.json"])
    recipes = summary.get("selected_recipes", {})
    expected = {"GRU": (32, 0.003, 0.1, 3777), "LSTM": (32, 0.003, 0.1, 5025),
                "MLP": (16, 0.001, 0.1, 593)}
    phases = {"cv": {"start": "2024-01-01", "end": "2024-12-31", "n_dates": 366},
              "test": {"start": "2025-01-01", "end": "2026-08-31", "n_dates": 608}}
    if (summary.get("study_id") != "e1_primary_608_20261107"
            or set(summary.get("models", ())) != set(expected)
            or set(recipes) != set(expected) or summary.get("baseline_model") != "GRU"
            or summary.get("R") != 10 or summary.get("R_per_neural") != 10
            or summary.get("n_dates") != 608 or summary.get("dates_per_trajectory") != 608
            or summary.get("phases") != phases
            or summary.get("primary_scope") != "historical_known_fixed_recipes"
            or summary.get("HPO_not_equal") is not True
            or summary.get("editorial_promotion_after_results_seen") is not True
            or summary.get("new_model_fits") != 0 or summary.get("new_forecasts") != 0
            or summary.get("source_summary_sha256") != legacy.digest(historical_files["summary.json"])
            or summary.get("source_score_rows_sha256") != legacy.digest(historical_files["run_metrics.csv"])):
        raise ValueError("Primary catalog requires the original D7/608-date, R=10 cohort and its saved source proof")
    for model, (width, lr, dropout, parameters) in expected.items():
        recipe = recipes[model]
        if (recipe.get("model") != model or recipe.get("history_days") != 7
                or tuple(recipe.get(key) for key in ("width", "learning_rate", "dropout", "parameters"))
                   != (width, lr, dropout, parameters)):
            raise ValueError("Primary catalog recipe differs from the original D7 cohort: " + model)
    if summary.get("baseline_recipe") != recipes["GRU"]:
        raise ValueError("Primary catalog baseline differs from its original GRU recipe")
    rows = list(csv.DictReader(io.StringIO(files["run_metrics.csv"].decode("utf-8"))))
    historical_rows = {(row["phase"], row["model"], row["seed"]): row
                       for row in csv.DictReader(io.StringIO(historical_files["run_metrics.csv"].decode("utf-8")))
                       if row.get("history_days") == "7"}
    observed = set()
    for row in rows:
        phase, model = row.get("phase"), row.get("model")
        seed = row.get("seed", "")
        identity = (phase, model, seed)
        if (phase not in phases or model not in {*expected, "Persistence"}
                or row.get("history_days") != "7"
                or row.get("n_dates") != str(phases[phase]["n_dates"])
                or identity in observed
                or (model == "Persistence" and seed != "")
                or (model in expected and (seed not in {str(n) for n in range(4101, 4111)}
                    or row.get("parameters") != str(expected[model][3])))):
            raise ValueError("Primary score rows mix dates, recipes, seeds or models")
        if model in expected:
            original = historical_rows.get(identity)
            if original is None or any(row.get(key) != value for key, value in original.items()):
                raise ValueError("Primary scores differ from their saved historical source rows")
        observed.add(identity)
    required = {(phase, model, str(seed)) for phase in phases for model in expected
                for seed in range(4101, 4111)} | {(phase, "Persistence", "") for phase in phases}
    if observed != required:
        raise ValueError("Primary score rows require the complete D7/608-date R=10 cohort and Persistence")
    return {
        "models": ["LSTM", "MLP", "GRU"], "complementary_models": ["TCN"],
        "active_entrypoint": "scripts/build_e1_primary_608.py",
        "active_entrypoints": {"assemble": "scripts/build_e1_primary_608.py",
                               "verify": "scripts/build_e1_primary_608.py --check"},
        "active_entrypoint_scope": "Assemble and verify existing original saved metrics; no new model fits or forecasts",
        "baseline_selected_by_cv": {"model": "GRU", **recipes["GRU"]},
        "scope": "Primary original fixed-recipe D7 LSTM/MLP/GRU benchmark on 608 already observed dates, with Persistence; unequal historical HPO and R=10 training-seed budget",
        "current_study": {**copy.deepcopy(summary),
            "history_days": 7, "definitive_seeds_per_family": 10,
            "reserve": {"start": "2025-01-01", "end": "2026-08-31", "days": 608},
            "known_reserve_exploratory": True, "fresh_confirmatory_holdout": False,
            "public_metadata_directory": SUPPLEMENTS["primary_608"]["directory"],
            "public_summary_sha256": legacy.digest(files["summary.json"]),
            "public_manifest_sha256": legacy.digest(files["manifest.json"])},
    }


def remediation_core_catalog(files):
    """Admit only the actual closed export of the published new608 execution.

    This verifier consumes public aggregate metadata only. It neither fabricates
    missing closure records nor falls back to an earlier study's results.
    """
    if set(files) != REMEDIATION_PUBLIC_FILES | {"manifest.json"}:
        raise ValueError("New primary requires exactly33 public payloads and its closed manifest")
    private = {"ytrue", "ypred", "persistencepred", "outcomeday", "observed",
               "prediction", "predictions", "statedict", "weights", "scaler", "scalers", "preprocessing",
               "masescale", "masetrainscale", "yscale", "xmean", "xscale", "rawprices", "open", "high", "low", "close", "volume"}
    def public_key(key):
        return re.sub(r"[^a-z]", "", str(key).lower()) not in private
    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if not public_key(key):
                    raise ValueError("New primary metadata contains private prices, vectors, weights or scalers")
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    records, tables = {}, {}
    for name, data in files.items():
        if name.endswith(".json"):
            records[name] = json.loads(data)
            visit(records[name])
        elif name.endswith(".csv"):
            reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
            def public_column(key):
                normalized = re.sub(r"[^a-z]", "", str(key).lower())
                return public_key(key) and not any(part in normalized for part in
                    ("price", "prediction", "scaler", "weights", "ytrue", "ypred", "dailyclose", "dailyopen", "dailyhigh", "dailylow", "dailyvolume"))
            if not all(public_column(key) for key in reader.fieldnames or ()):
                raise ValueError("New primary CSV contains private prices, vectors, weights or scalers")
            tables[name] = list(reader)
    summary, registration, selection, plan, sources = (records[name] for name in
        ("summary.json", "registration.json", "selection.json", "plan.json", "source_manifest.json"))
    families = {"GRU", "LSTM", "MLP"}
    metrics = {"rmse", "mae", "mse", "r2", "mape", "mase"}
    signature, source_signature, r = summary.get("execution_id"), summary.get("source_signature"), plan.get("R")
    if (summary.get("study_id") != REMEDIATION_STUDY or summary.get("source_commit") != REMEDIATION_SOURCE
            or set(summary.get("core_models", ())) != families or set(summary.get("models", ())) != families
            or not isinstance(signature, str) or not _SHA256.fullmatch(signature)
            or not isinstance(source_signature, str) or not _SHA256.fullmatch(source_signature)
            or registration.get("registration_signature") != signature or registration.get("source_signature") != source_signature
            or registration.get("study_id") != REMEDIATION_STUDY or registration.get("source_commit") != REMEDIATION_SOURCE
            or sources.get("source_commit") != REMEDIATION_SOURCE
            or sources.get("sources_sha256") != registration.get("sources_sha256")
            or sources.get("design_sha256") != registration.get("design_sha256")
            or legacy.digest(legacy.encode({"sources_sha256": registration.get("sources_sha256"),
                "design_sha256": registration.get("design_sha256"), "source_commit": REMEDIATION_SOURCE})) != source_signature):
        raise ValueError("New primary source and execution identity must match the published608 source")
    if (type(r) is not int or type(plan.get("R_required")) is not int or not 30 <= r <= 10000
            or r != max(30, plan["R_required"]) or plan.get("R_final") != r
            or plan.get("seeds") != list(range(18101, 18101+r))
            or any(record.get("registration_signature") != signature for record in (selection, plan, records["plan_publication.json"]))
            or summary.get("R_final") != r or summary.get("R_required") != plan["R_required"]
            or summary.get("dates_per_trajectory") != 608 or summary.get("neural_prediction_rows") != 1824*r
            or summary.get("definitive_trajectories") != 3*r or summary.get("definitive_fit_stages") != 6*r
            or summary.get("hpo_trajectories") != 24 or summary.get("hpo_blocks") != 240 or summary.get("hpo_fit_stages") != 480
            or summary.get("pilot_trajectories") != 30 or summary.get("pilot_fit_stages") != 60
            or summary.get("known_reserve_exploratory") is not True or summary.get("fresh_confirmatory_holdout") is not False
            or summary.get("selection_using_holdout") is not False or summary.get("private_payloads_exported") is not False
            or any(summary.get(key) is not True for key in ("source_publication_precedes_new_fits",
                "plan_publication_precedes_definitive_fits", "scoring_after_all_new_weights_frozen"))):
        raise ValueError("New primary requires its complete608 cohort, dynamic pilot-derived R and registered phase counts")
    design = registration.get("design", {})
    reserve = design.get("reserve", {})
    if (reserve.get("start") != "2025-01-01" or reserve.get("end") != "2026-08-31"
            or design.get("models") != ["GRU", "LSTM", "MLP"] or design.get("history_days") != 7):
        raise ValueError("New primary design mixes the608 historical outcome calendar or families")
    recipes, baseline = selection.get("recipes", {}), selection.get("baseline_model")
    if set(recipes) != families or baseline not in families or summary.get("baseline_selected_before_reserve") != baseline:
        raise ValueError("New primary baseline must be the family frozen by its own development CV")
    for model, width, parameters in (("GRU", 32, 3777), ("LSTM", 32, 5025), ("MLP", 16, 593)):
        recipe = recipes[model]
        if (recipe.get("model") != model or recipe.get("history_days") != 7
                or recipe.get("width") != width or recipe.get("parameters") != parameters
                or recipe.get("candidate_id") not in {"c00", "c01", "c02", "c03"}
                or recipe.get("learning_rate") not in {.001, .003} or recipe.get("dropout") not in {0., .1}):
            raise ValueError("New primary recipes must be the actual GRU32/LSTM32/MLP16 HPO selections")
    def panel(name, seeds, *, reference=False, identity=True):
        rows = tables[name]
        required = {(model, str(seed)) for model in families for seed in seeds}
        actual = [(row.get("model"), row.get("seed")) for row in rows if row.get("model") != "Persistence"]
        if set(actual) != required or len(actual) != len(required):
            raise ValueError("New primary panel mixes seeds, models or phase: " + name)
        references = [row for row in rows if row.get("model") == "Persistence"]
        if len(references) != int(reference):
            raise ValueError("New primary requires exactly one deterministic Persistence reference")
        for row in rows:
            if identity and any(row.get(key) != expected for key, expected in
                    (("study_id", REMEDIATION_STUDY), ("execution_id", signature), ("source_signature", source_signature))):
                raise ValueError("New primary rows mix registered source or execution identity: " + name)
            if "phase" in row and row["phase"] != {"pilot_metrics.csv": "pilot", "final_fit_summary.csv": "final"}.get(name, "test"):
                raise ValueError("New primary rows mix phase identity: " + name)
        return rows
    runs = panel("run_metrics.csv", plan["seeds"], reference=True)
    panel("pilot_metrics.csv", range(17101, 17111))
    panel("final_fit_summary.csv", plan["seeds"])
    costs = panel("run_costs.csv", plan["seeds"], identity=False)
    for row in runs:
        if row.get("n_dates") != "608" or not metrics <= set(row) or any(not math.isfinite(float(row[key])) for key in metrics):
            raise ValueError("New primary requires all six observed metrics on608 dates per trajectory")
    if any(row.get("n_dates") != "608" or not all(math.isfinite(float(row[key])) and float(row[key]) >= 0
            for key in ("training_seconds", "inference_seconds")) for row in costs):
        raise ValueError("New primary costs must come from the current608-date fit and inference")
    tuning = tables["hpo_run_metrics.csv"]
    expected = {(model, f"c{candidate:02d}", str(seed)) for model in families for candidate in range(4) for seed in (16101, 16102)}
    actual = [(row.get("model"), row.get("candidate_id"), row.get("trajectory_seed")) for row in tuning]
    if len(actual) != 24 or set(actual) != expected or any(any(row.get(key) != value for key, value in
            (("study_id", REMEDIATION_STUDY), ("execution_id", signature), ("source_signature", source_signature))) for row in tuning):
        raise ValueError("New primary requires all24 current HPO trajectories")
    pairs = {("GRU", "LSTM"), ("GRU", "MLP"), ("LSTM", "MLP")}
    for name in ("pairwise_rmse.csv", "wilcoxon_rmse.csv"):
        rows = tables[name]
        if len(rows) != 3 or {(row.get("model_a"), row.get("model_b")) for row in rows} != pairs or any(row.get("n_paired") != str(r) for row in rows):
            raise ValueError("New primary registered paired statistics require all three current R panels")
    if len(tables["qq_data.csv"]) != 3*r or len(tables["friedman_rmse.csv"]) != 1:
        raise ValueError("New primary requires its observed QQ and Friedman diagnostics")
    if (tables["friedman_rmse.csv"][0].get("n_paired") != str(r)
            or tables["friedman_rmse.csv"][0].get("n_models") != "3"
            or {(row.get("model_a"), row.get("model_b")) for row in tables["qq_data.csv"]} != pairs
            or any(row.get("n_paired") != str(r) for row in tables["qq_data.csv"])):
        raise ValueError("New primary rank and QQ diagnostics mix a different study panel")
    qq_keys = [(row.get("model_a"), row.get("model_b"), row.get("order_index")) for row in tables["qq_data.csv"]]
    if set(qq_keys) != {(a, b, str(index)) for a, b in pairs for index in range(1, r+1)}:
        raise ValueError("New primary QQ coordinates require every ordered observation in all three pairs")
    temporal_pairs = pairs | {(model, "Persistence") for model in families}
    for name, expected_rows in (("dm_pairs.csv", 6), ("temporal_bootstrap.csv", 18)):
        rows = tables[name]
        if (len(rows) != expected_rows or {(row.get("model_a"), row.get("model_b")) for row in rows} != temporal_pairs
                or any(row.get("n_dates") != "608" or row.get("baseline_selected_before_reserve") != baseline for row in rows)):
            raise ValueError("New primary temporal statistics mix a different608 cohort or frozen baseline")
    if any(row.get("n_residuals") != "608" for row in tables["diagnostics.csv"]) or len(tables["diagnostics.csv"]) != 9*r:
        raise ValueError("New primary residual diagnostics must use every current608-date trajectory")
    residual_keys = [(row.get("model"), row.get("seed"), row.get("diagnostic"), row.get("lag"))
                     for row in tables["diagnostics.csv"]]
    if set(residual_keys) != {(model, str(seed), diagnostic, lag) for model in families for seed in plan["seeds"]
            for diagnostic, lag in (("BDS", "2"), ("Ljung-Box", "1"), ("Ljung-Box", "5"))}:
        raise ValueError("New primary residual diagnostic rows mix models, seeds or registered lags")
    for name in ("learning_curves_by_run.csv", "first_seed_learning_curves.csv"):
        rows = tables[name]
        seeds = [18101] if name.startswith("first_seed") else plan["seeds"]
        expected = {(model, str(seed), stage) for model in families for seed in seeds for stage in ("inner", "refit")}
        if {(row.get("model"), row.get("seed"), row.get("stage")) for row in rows} != expected:
            raise ValueError("New primary requires observed inner and refit curves for every final seed")
        if any(row["stage"] == "refit" and row.get("validation_mse_scaled_eval") not in ("", "nan", "NaN") for row in rows):
            raise ValueError("New primary refit curves cannot contain held-out validation losses")
    return {"models": ["LSTM", "MLP", "GRU"], "complementary_models": ["TCN"],
        "active_entrypoint": "scripts/run_e1_608_remediation.py",
        "active_entrypoints": {phase: "scripts/run_e1_608_remediation.py" for phase in ("hpo", "pilot", "plan", "fit", "freeze", "score", "analyze")},
        "active_entrypoint_scope": "New equal-HPO and pilot-sized GRU32/LSTM32/MLP16 execution on608 known historical dates",
        "baseline_selected_by_cv": {"model": baseline, **copy.deepcopy(recipes[baseline])},
        "scope": "Current registered-method historical exploratory608-date study; seed uncertainty is conditional on known dates, without temporal power or a fresh confirmatory holdout",
        "current_study": {**copy.deepcopy(summary), "R": r, "n_dates": 608, "history_days": 7,
            "selected_recipes": copy.deepcopy(recipes), "definitive_seeds_per_family": r,
            "reserve": {"start": "2025-01-01", "end": "2026-08-31", "days": 608},
            "public_metadata_directory": SUPPLEMENTS["remediation_608"]["directory"],
            "public_summary_sha256": legacy.digest(files["summary.json"]),
            "public_manifest_sha256": legacy.digest(files["manifest.json"])}}


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
        if not set(spec.get("configuration_files", ())).issubset(files):
            raise ValueError("Supplement configuration files must be publicly approved: " + label)
        for name in files:
            if (directory / name).relative_to(ROOT).as_posix() not in tracked:
                raise ValueError("Supplement source is not committed: " + label + "/" + name)
        supplements[label], supplement_info[label] = files, {**info, "product": spec["item"]}
    previous_primary_catalog = current_core_catalog(supplements["primary_608"], supplements["historical_core"])
    current_catalog = remediation_core_catalog(supplements["remediation_608"])
    complementary_catalog = complementary_core_catalog(supplements["core_mlp_extension"])
    screening_closure = native_screening_closure(stages["native_620_screening_20261108"],
                                               stages["native_execution_20261107"])
    native_review_inputs = native_review_source_aliases(stages["native_620_screening_20261108"])
    error_closure = frozen_error_closure(supplements["frozen_error_analysis"],
                                        supplements["remediation_608"], current_catalog)
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
    for name in sorted(MANUAL_REACQUISITION_FILES):
        if name not in tracked:
            raise ValueError("Manual acquisition helper must be committed: " + name)
        data = _regular_file(ROOT / name, ROOT)
        validate_public_bytes(str(PurePosixPath(name).with_suffix(".json")), data)
        notebook = json.loads(data)
        if (notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list)
                or any(cell.get("outputs") or cell.get("execution_count") is not None
                       for cell in notebook["cells"])):
            raise ValueError("Manual acquisition notebook must be unexecuted and have no outputs")
        products["8.2"][name] = data
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
    for name, data in native_review_inputs.items():
        if name in products["8.4"] and products["8.4"][name] != data:
            raise ValueError("Refuse to replace an existing canonical reviewed input with different bytes: " + name)
        products["8.4"][name] = data
    for label, files in supplements.items():
        spec = SUPPLEMENTS[label]
        prefix = spec["directory"] + "/"
        for item in {spec["item"], "8.7"} | ({"8.6"} if spec.get("configuration_files") else set()):
            products[item] = {name: data for name, data in products[item].items()
                              if not name.startswith(prefix)}
        for name, data in files.items():
            products[spec["item"]][prefix + name] = data
            if name in {"summary.json", "source_manifest.json", "manifest.json", "artifact_availability.json", spec["manifest"]}:
                products["8.7"][prefix + name] = data
            if name in spec.get("configuration_files", ()):
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
        "edition_id": REFRESH_ID, "date": REFRESH_DATE, "date_timezone": REFRESH_TIMEZONE,
        "colombia_date_context": "2026-11-07", "source_commit": revision,
        "baseline_source_commit": catalog.get("source_commit"),
        "baseline_catalog_sha256": base_catalog_sha,
        "stages": stage_info,
        "supplements": supplement_info,
        "preserved_archive_sha256": {ITEMS[item]: base_rows[item]["sha256"]
                                     for item in sorted(PRESERVED_ITEMS)},
        "original_experiments_changed": False, "retrospective_preregistration": False,
        "new_analysis_scope": "Post-hoc screening of received native-search records and aggregate error diagnosis from the already closed GRU32/LSTM32/MLP16 R309/608-date predictions. Current final metrics, recipes, baseline and source records remain unchanged; original R10/608-date and R465/60-date studies are preserved. Packaging itself performs no training, searches or daily forecasts.",
        "current_benchmark_results_changed": False,
        "native_screening_closure": screening_closure,
        "native_review_canonical_input_aliases": sorted(native_review_inputs),
        "frozen_error_closure": error_closure,
        "chronology_note": CHRONOLOGY_NOTE,
    }
    products["8.7"]["literature_refresh_" + REFRESH_ID + ".json"] = legacy.encode(refresh)
    products["8.7"]["AMPLIACION_LITERARIA_" + REFRESH_ID + ".md"] = (
        "# Ampliación literaria del 8 de noviembre de 2026 (UTC)\n\n"
        "Fecha correspondiente en Colombia: 7 de noviembre de 2026.\n\n" + CHRONOLOGY_NOTE + "\n\n"
        "El producto 8.4 contiene los metadatos públicos completos de las nuevas etapas; "
        "aquí se conservan sus resúmenes, manifiestos y procedencia. No se redistribuyen "
        "los archivos de búsqueda originales, resúmenes editoriales ni PDF de terceros.\n"
    ).encode()
    products["8.7"]["SUPLEMENTO_EVIDENCIAS_" + REFRESH_ID + ".md"] = (
        "# Evidencias y análisis posteriores del 8 de noviembre de 2026 (UTC)\n\n"
        "Fecha correspondiente en Colombia: 7 de noviembre de 2026.\n\n"
        + CHRONOLOGY_NOTE + "\n\n"
        "8.7 añade la comparación documental y el contexto académico del margen; 8.4 incluye "
        "cribado, recuperación de originales y evidencia de búsquedas con sus límites. "
        "8.5 añade estadística posterior, curvas de una reproducción separada, auditoría "
        "forense y repetibilidad diagnóstica actual, además del diagnóstico agregado "
        "de errores y gráficos observado/pronosticado del estudio R309 cerrado. "
        "Los protocolos y metadatos de ejecución "
        "también se incluyen en 8.6. La guía exige procedimiento reproducible, semillas, "
        "dependencias y declarar el alcance del determinismo; no exige identidad histórica "
        "bit a bit. Las discrepancias encontradas conservan sus límites científicos. Los resultados y fuentes históricos "
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
    if "historical_study_metadata" not in result:
        historical_keys = ("source_commit", "models", "active_entrypoint", "active_entrypoints",
                           "active_entrypoint_scope", "baseline_selected_by_cv", "scope",
                           "public_aggregate_manifest_sha256", "declaration_manifest_sha256", "planned_D7_phase")
        result["historical_study_metadata"] = {
            key: copy.deepcopy(catalog[key]) for key in historical_keys if key in catalog}
    for key in ("public_aggregate_manifest_sha256", "declaration_manifest_sha256", "planned_D7_phase"):
        result.pop(key, None)
    if "complementary_study_metadata" not in result:
        # Preserve the full prior 60-day catalog fields and its earlier snapshot.
        keys = ("source_commit", "models", "complementary_models", "active_entrypoint",
                "active_entrypoints", "active_entrypoint_scope", "baseline_selected_by_cv",
                "scope", "current_study", "historical_study_metadata")
        previous = {key: copy.deepcopy(catalog[key]) for key in keys if key in catalog}
        if catalog.get("current_study", {}).get("study_id") != "e1_core_mlp_extension_20261107":
            previous.update(copy.deepcopy(complementary_catalog))
        previous["public_summary"] = json.loads(supplements["core_mlp_extension"]["summary.json"])
        previous["public_manifest_sha256"] = legacy.digest(supplements["core_mlp_extension"]["manifest.json"])
        result["complementary_study_metadata"] = previous
    if "previous_primary_study_metadata" not in result:
        if catalog.get("current_study", {}).get("study_id") == "e1_primary_608_20261107":
            previous = copy.deepcopy(catalog)
        else:
            previous = copy.deepcopy(previous_primary_catalog)
            previous["historical_study_metadata"] = copy.deepcopy(result["historical_study_metadata"])
        previous["public_summary"] = json.loads(supplements["primary_608"]["summary.json"])
        previous["public_manifest_sha256"] = legacy.digest(supplements["primary_608"]["manifest.json"])
        result["previous_primary_study_metadata"] = previous
    records_by_study = result.setdefault("dated_study_records", {})
    records_by_study.setdefault("e1_primary_608_20261107", copy.deepcopy(result["previous_primary_study_metadata"]))
    records_by_study.setdefault("e1_core_mlp_extension_20261107", copy.deepcopy(result["complementary_study_metadata"]))
    # Freeze the exact preceding current-study record as well. Supplementary
    # screening and plots must never silently replace its scientific identity.
    snapshot_keys = ("source_commit", "models", "complementary_models", "active_entrypoint",
        "active_entrypoints", "active_entrypoint_scope", "baseline_selected_by_cv", "scope", "current_study")
    prior_current = catalog if catalog.get("current_study", {}).get("study_id") == REMEDIATION_STUDY else current_catalog
    records_by_study.setdefault(REMEDIATION_STUDY, {
        key: copy.deepcopy(prior_current[key]) for key in snapshot_keys if key in prior_current})
    previous_refresh = catalog.get("evidence_refresh", catalog.get("literature_refresh"))
    if previous_refresh:
        history = result.setdefault("prior_evidence_refreshes", [])
        if previous_refresh not in history:
            history.append(copy.deepcopy(previous_refresh))
    result.update(current_catalog)
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
        "configuración de las reproducciones y auditorías identificadas; 8.7 incorpora comparación documental, "
        "margen y procedencia. "
        "`catalogo.json` identifica el núcleo actual LSTM–MLP–GRU y conserva modelos, "
        "fases y entrypoints anteriores en `historical_study_metadata`; el anterior primario R=10 permanece en `previous_primary_study_metadata`, y la ejecución de sesenta fechas y su historial en `complementary_study_metadata`. Ambos se conservan también como registros fechados en `dated_study_records`. "
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
