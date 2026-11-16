#!/usr/bin/env python3
"""Build the maintainer's deterministic D01 payload from an explicit Git commit.

This command packages an already compiled book. It does not compile the book,
download dependencies, execute notebooks, fit models or perform inference.
Run it only after committing the selected sources and compiling that snapshot.
The delivered reader needs Python, but does not need Git or a repository clone.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import io
import json
import lzma
import os
from pathlib import Path, PurePosixPath
import re
import runpy
import shutil
import subprocess
import sys
import tarfile
import tempfile
from urllib.parse import unquote, urlsplit
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "bundle_manifest.json"
MATHJAX_PREFIX = "site/_static/vendor/mathjax/"
MATHJAX_VERSION = "3.2.2"
GITHUB_FILE_LIMIT = 100_000_000
MAX_FILES = 30_000
MAX_EXPANDED_BYTES = 2 * 1024**3
SOURCE_PATHS = (
    "book_e2",
    "requirements-book.lock.txt",
    "requirements-book.txt",
    "BUILD.md",
    "scripts/check_book_links.py",
    "scripts/e2_standalone_book.py",
    "scripts/build_e2_standalone_book_payload.py",
)
DELIVERABLE_ZIPS = frozenset((
    "01_informe_jupyter_book.zip", "02_codigo_modelos.zip",
    "03_datos_contrato.zip", "04_preregistro_auditoria.zip",
    "05_ficha_ruta_F2a.zip", "06_benchmark_comparaciones.zip",
    "07_registro_experimentos.zip", "08_techo_incertidumbre.zip",
    "09_arquitecturas_no_aplica.zip", "10_paper_guia_replica.zip",
    "11_reproducibilidad.zip", "12_presentacion_defensa.zip",
))
EXCLUDED_DIRS = frozenset((
    ".git", "work", "outputs", "__pycache__", ".ipynb_checkpoints",
    "_build", ".venv", "venv", ".cache",
))
WEIGHT_SUFFIXES = frozenset((".pt", ".pth", ".ckpt", ".safetensors"))
RECURSIVE_PAYLOADS = frozenset((
    "payload.tar.xz", "e2-standalone-book-payload.tar.xz",
))
OPTIONAL_DOWNLOAD_PREFIX = "https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630"
BUILDER_SOURCE_PATH = "scripts/build_e2_standalone_book_payload.py"
OPTIONAL_ASSET_PINS = {
    "book_e2/_static/e2_scientific_closure_20261116/preregistration_bundle.zip": {
        "bytes": 13_159_582,
        "sha256": "51add9f3236b2109d9b01702ea420e9b1ed1d89a7ad1a133d8a18984a66b2cf6",
    },
    "book_e2/_static/e2_scientific_closure_20261116/final_registration_bundle.zip": {
        "bytes": 4_545_962,
        "sha256": "3068baac29ef94f1e0a387550377f9bea005b3bd900f84c84c305b80dc961b1a",
    },
}
OPTIONAL_JOURNAL_ROOTS = {
    "book_e2/_static/e2_scientific_closure_20261116/preregistration/journals/": {
        "bundle_repository_path": "book_e2/_static/e2_scientific_closure_20261116/preregistration_bundle.zip",
        "member_prefix": "Preregistro_E2/journals/", "expected_count": 206,
    },
    "book_e2/_static/e2_scientific_closure_20261116/final_registration_chronology/journals/": {
        "bundle_repository_path": "book_e2/_static/e2_scientific_closure_20261116/final_registration_bundle.zip",
        "member_prefix": "Registro_Final_E2/journals/", "expected_count": 90,
    },
}
JOURNAL_RELATIVE_PATTERN = r"[0-9a-f]{64}/(?:ae1|ae2|supervised)\.jsonl\.gz"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def git_blob_digest(path: Path, object_id: str) -> str:
    algorithm = {40: "sha1", 64: "sha256"}.get(len(object_id))
    if algorithm is None:
        raise ValueError("Formato de identificador Git no admitido")
    result = hashlib.new(algorithm)
    result.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def relative_name(name: str) -> Path:
    if not name or "\\" in name or "\x00" in name or ":" in name:
        raise ValueError(f"Ruta de archivo no permitida: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("..", ".git") for part in path.parts):
        raise ValueError(f"Ruta fuera del paquete: {name!r}")
    return Path(*path.parts)


def optional_candidates(path: Path, optional_downloads: dict | None = None) -> list[dict]:
    """Source/static paths are exact; Sphinx filenames need content matching."""
    parts = path.parts[1:] if path.parts[:1] == ("site",) else path.parts
    result = []
    for original, record in (optional_downloads or {}).items():
        site_path = Path(*PurePosixPath(original).parts[1:])
        if path == Path("source") / original or Path(*parts) == site_path:
            return [record]
        if (len(parts) == 3 and parts[0] == "_downloads"
                and re.fullmatch(r"[0-9a-f]{32}", parts[1])
                and parts[2] == site_path.name):
            result.append(record)
    return result


def optional_record(path: Path, optional_downloads: dict | None = None) -> dict | None:
    candidates = optional_candidates(path, optional_downloads)
    return candidates[0] if candidates else None


def exclusion(path: Path, optional_downloads: dict | None = None) -> str | None:
    if any(part in EXCLUDED_DIRS for part in path.parts):
        return "generated_or_private_directory"
    if path.name in DELIVERABLE_ZIPS:
        return "independent_deliverable_zip_and_sphinx_copy"
    if path.suffix.lower() in WEIGHT_SUFFIXES:
        return "model_weights"
    if path.name in RECURSIVE_PAYLOADS:
        return "recursive_payload"
    if optional_record(path, optional_downloads) is not None:
        return "optional_scientific_download_available_at_immutable_https"
    return None


def check_size(size: int, name: str) -> None:
    if size >= GITHUB_FILE_LIMIT:
        raise ValueError(f"{name}: {size} bytes alcanza el límite de 100.000.000 bytes")


def git(*arguments: str) -> bytes:
    return subprocess.check_output(["git", *arguments], cwd=ROOT)


def optional_download_catalogue(commit: str | None, source_inventory: dict) -> dict:
    """Opt in only after both immutable Git snapshots match the pinned bytes."""
    if commit is None:
        return {}
    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise ValueError("--optional-download-commit debe ser un SHA completo de 40 caracteres")
    commit = commit.lower()
    if git("rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip() != commit:
        raise ValueError("El commit de descargas debe identificar directamente un commit")
    result = {}
    for path, pin in OPTIONAL_ASSET_PINS.items():
        if path not in source_inventory:
            raise ValueError(f"El commit de fuente no contiene el descargable opcional: {path}")
        tree = git("ls-tree", "-z", commit, "--", path).split(b"\0")
        entries = [row for row in tree if row]
        if len(entries) != 1:
            raise ValueError(f"El commit de descargas no contiene un archivo único: {path}")
        metadata, encoded = entries[0].split(b"\t", 1)
        mode, kind, blob = metadata.decode("ascii").split()
        if encoded.decode("utf-8") != path or kind != "blob" or mode not in ("100644", "100755"):
            raise ValueError(f"Descargable opcional no regular: {path}")
        for snapshot, object_id in (("fuente", source_inventory[path]["git_blob"]),
                                    ("descarga", blob)):
            raw = git("cat-file", "blob", object_id)
            if len(raw) != pin["bytes"] or hashlib.sha256(raw).hexdigest() != pin["sha256"]:
                raise ValueError(f"Bytes/SHA del descargable no coinciden en {snapshot}: {path}")
        result[path] = {"repository_path": path, **pin, "download_commit": commit,
                        "url": f"{OPTIONAL_DOWNLOAD_PREFIX}/{commit}/{path}",
                        "reason": "Redundant scientific ZIP; reports and book resources remain selected locally; archive download requires Internet."}
    return result


def read_git_asset(commit: str, path: str) -> bytes:
    rows = [row for row in git("ls-tree", "-z", commit, "--", path).split(b"\0") if row]
    if len(rows) != 1:
        raise ValueError(f"Falta un archivo único en el commit opcional: {path}")
    metadata, name = rows[0].split(b"\t", 1)
    mode, kind, blob = metadata.decode("ascii").split()
    if name.decode("utf-8") != path or kind != "blob" or mode not in ("100644", "100755"):
        raise ValueError(f"Archivo opcional no regular en Git: {path}")
    return git("cat-file", "blob", blob)


def source_archive_catalogue(commit: str, inventory: dict) -> dict:
    """Internal archive identities for rejecting dirty site copies by default."""
    present = set(OPTIONAL_ASSET_PINS) & inventory.keys()
    if not present:
        return {}
    if present != set(OPTIONAL_ASSET_PINS):
        raise ValueError("El cierre científico requiere sus dos ZIP fuente fijados")
    result = {}
    for path, pin in OPTIONAL_ASSET_PINS.items():
        raw = git("cat-file", "blob", inventory[path]["git_blob"])
        if len(raw) != pin["bytes"] or hashlib.sha256(raw).hexdigest() != pin["sha256"]:
            raise ValueError(f"El ZIP fuente no coincide con la identidad científica fijada: {path}")
        result[path] = {"repository_path": path, **pin, "download_commit": commit,
                        "url": f"{OPTIONAL_DOWNLOAD_PREFIX}/{commit}/{path}"}
    return result


def scientific_journal_catalogue(downloads: dict, inventory: dict) -> dict:
    """Bind the exact 206+90 gzip members to Git-tracked source ZIPs.

    ZIP members are read in memory; the inner gzip journals are never decoded,
    and no archive or individual journal source file is created. Individual
    journal paths are deliberately absent from the selected Git snapshot.
    """
    result, hashes = {}, set()
    for prefix, specification in OPTIONAL_JOURNAL_ROOTS.items():
        if any(path.startswith(prefix) for path in inventory):
            raise ValueError("No se admiten journals individuales como fuentes Git en el modo ZIP-member")
        bundle = downloads[specification["bundle_repository_path"]]
        raw_bundle = read_git_asset(bundle["download_commit"], bundle["repository_path"])
        if len(raw_bundle) != bundle["bytes"] or hashlib.sha256(raw_bundle).hexdigest() != bundle["sha256"]:
            raise ValueError("El ZIP C0 de journals no coincide con su identidad fijada")
        source_blob = inventory[bundle["repository_path"]]["git_blob"]
        source_bundle = git("cat-file", "blob", source_blob)
        if len(source_bundle) != bundle["bytes"] or hashlib.sha256(source_bundle).hexdigest() != bundle["sha256"]:
            raise ValueError("El ZIP fuente C1 difiere del ZIP de descarga fijado")
        member_prefix = specification["member_prefix"]
        selected = set()
        with zipfile.ZipFile(io.BytesIO(raw_bundle)) as archive:
            for member in archive.infolist():
                if member.is_dir() or not member.filename.startswith(member_prefix):
                    continue
                relative = member.filename[len(member_prefix):]
                if not re.fullmatch(JOURNAL_RELATIVE_PATTERN, relative):
                    raise ValueError(f"Miembro journal fuera del catálogo autorizado: {member.filename}")
                path = prefix + relative
                if path in selected or not 0 < member.file_size < GITHUB_FILE_LIMIT:
                    raise ValueError(f"Journal repetido o de tamaño inválido: {path}")
                selected.add(path)
                raw = archive.read(member)
                sha = hashlib.sha256(raw).hexdigest()
                if sha in hashes:
                    raise ValueError("Los journals deben conservar identidades de contenido únicas para las copias Sphinx")
                hashes.add(sha)
                result[path] = {
                    "repository_path": path, "bytes": len(raw), "sha256": sha,
                    "download_commit": bundle["download_commit"], "url": bundle["url"],
                    "bundle_repository_path": bundle["repository_path"],
                    "bundle_bytes": bundle["bytes"], "bundle_sha256": bundle["sha256"],
                    "member_path": member.filename,
                    "source_archive_git_blob": source_blob,
                    "source_individual_git_present": False,
                    "source_binding": "Git-tracked ZIP member; no individual journal Git source is claimed",
                    "reason": "Journal available as this exact member of the pinned scientific ZIP. Only explicit opt-in permits omitting recognized residual site copies; none are fabricated as source files.",
                }
        if len(selected) != specification["expected_count"]:
            raise ValueError(f"El catálogo journal debe contener exactamente {specification['expected_count']} rutas: {prefix}")
    return result


def optional_journal_catalogue(enabled: bool, downloads: dict, inventory: dict) -> dict:
    if not enabled:
        return {}
    if not downloads:
        raise ValueError("--optional-scientific-journals requiere --optional-download-commit")
    return scientific_journal_catalogue(downloads, inventory)


def canonical_journal_path(path: Path) -> str | None:
    parts = path.parts[1:] if path.parts[:1] == ("site",) else path.parts
    candidate = "book_e2/" + Path(*parts).as_posix()
    for prefix in OPTIONAL_JOURNAL_ROOTS:
        if candidate.startswith(prefix) and re.fullmatch(JOURNAL_RELATIVE_PATTERN, candidate[len(prefix):]):
            return candidate
    return None


def verify_optional_bytes(path: Path, record: dict) -> None:
    if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
        raise ValueError(f"La copia opcional no coincide con el archivo fijado: {path.name}")


def verified_site_optional(path: Path, logical: Path, catalogue: dict | None) -> dict | None:
    candidates = optional_candidates(logical, catalogue)
    if not candidates:
        return None
    parts = logical.parts[1:] if logical.parts[:1] == ("site",) else logical.parts
    sphinx = len(parts) == 3 and parts[0] == "_downloads"
    size, sha = path.stat().st_size, digest(path)
    matches = [record for record in candidates if record["bytes"] == size and record["sha256"] == sha]
    if len(matches) == 1:
        return matches[0]
    if not matches and sphinx and all("bundle_repository_path" in record for record in candidates):
        return None  # A same-named unrelated gzip is retained, never omitted.
    raise ValueError(f"La copia opcional no coincide de forma única con el archivo fijado: {path.name}")


def check_optional_source_links(stage: Path, optional_downloads: dict) -> dict:
    """Reject local MyST/Markdown dependencies on omitted download archives."""
    book = stage / "source/book_e2"
    targets = {(stage / "source" / name).resolve(): record
               for name, record in optional_downloads.items()}
    seen = set()
    checked = 0
    for path in sorted(book.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".md", ".rst", ".ipynb"):
            continue
        raw = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".ipynb":
            notebook = json.loads(raw)
            raw = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", [])
                            if cell.get("cell_type") == "markdown")
        references = re.findall(r"\]\(\s*<?([^\s)>]+)>?(?:\s+[^)]*)?\)", raw)
        references += re.findall(r"^\s*\[[^]]+\]:\s*<?([^\s>]+)", raw, re.MULTILINE)
        references += re.findall(r"(?:href|src)\s*=\s*[\"']([^\"']+)[\"']", raw)
        for role in re.findall(r"(?:\{download\}|:download:)`([^`]+)`", raw):
            references.append(role.rsplit("<", 1)[-1].rstrip(">") if "<" in role else role)
        for link in references:
            parts = urlsplit(link)
            if parts.scheme or parts.netloc:
                for record in optional_downloads.values():
                    if unquote(parts.path).endswith("/" + record["repository_path"]):
                        if link != record["url"]:
                            raise ValueError(f"Descarga opcional sin URL HTTPS inmutable fijada: {path.relative_to(book)}")
                        seen.add(record["repository_path"])
                continue
            if not parts.path:
                continue
            local = ((book / unquote(parts.path).lstrip("/")) if parts.path.startswith("/")
                     else path.parent / unquote(parts.path)).resolve()
            if local in targets:
                raise ValueError(f"Referencia local MyST/Markdown a un archivo opcional omitido: {path.relative_to(book)}: {link}")
            checked += 1
    return {"local_source_references_checked": checked,
            "optional_archives_required_locally": 0,
            "canonical_optional_links_seen": sorted(seen),
            "scope": "Markdown, RST download roles and notebook Markdown; source bytes are not rewritten."}


def source_inventory(commit: str) -> dict[str, dict]:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise ValueError("--source-commit debe ser un SHA completo de 40 caracteres")
    resolved = git("rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip()
    if resolved != commit.lower():
        raise ValueError("--source-commit debe identificar directamente un commit")
    entries = {}
    raw = git("ls-tree", "-r", "-z", commit, "--", *SOURCE_PATHS)
    for item in raw.split(b"\0"):
        if not item:
            continue
        metadata, encoded_name = item.split(b"\t", 1)
        mode, kind, blob = metadata.decode("ascii").split()
        name = encoded_name.decode("utf-8")
        relative_name(name)
        if kind != "blob" or mode not in ("100644", "100755"):
            raise ValueError(f"Fuente no regular, enlace o submódulo: {name}")
        entries[name] = {"git_blob": blob, "git_mode": mode}
    for requested in SOURCE_PATHS:
        if not any(name == requested or name.startswith(requested + "/") for name in entries):
            raise ValueError(f"La fuente {commit} no contiene {requested}")
    return entries


def verify_executed_builder(inventory: dict) -> None:
    """The packaging code being executed must itself belong to the snapshot."""
    path = Path(__file__)
    record = inventory.get(BUILDER_SOURCE_PATH)
    if path.is_symlink() or not path.is_file() or not isinstance(record, dict):
        raise ValueError("El constructor ejecutado debe ser un archivo regular inventariado en el commit de fuente")
    expected = record["git_blob"]
    if git_blob_digest(path, expected) != expected:
        raise ValueError("Los bytes del constructor ejecutado no coinciden con el commit de fuente")


def copy_source(commit: str, stage: Path, inventory: dict, excluded: list,
                optional_downloads: dict | None = None) -> None:
    # git archive reads only committed objects, never working-tree corrections.
    with tempfile.TemporaryFile() as stream:
        subprocess.run(["git", "archive", "--format=tar", "--prefix=source/",
                        commit, "--", *SOURCE_PATHS], cwd=ROOT, stdout=stream, check=True)
        stream.seek(0)
        included = set()
        with tarfile.open(fileobj=stream, mode="r:") as archive:
            for member in archive:
                path = relative_name(member.name)
                if not path.parts or path.parts[0] != "source":
                    raise ValueError(f"Raíz de fuente inesperada: {member.name}")
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError(f"Archivo fuente no regular: {member.name}")
                original = Path(*path.parts[1:]).as_posix()
                if original not in inventory or original in included:
                    raise ValueError(f"Archivo fuente no inventariado o repetido: {original}")
                reason = exclusion(path, optional_downloads)
                if reason:
                    optional = optional_record(path, optional_downloads)
                    if optional is not None:
                        stream = archive.extractfile(member)
                        if stream is None:
                            raise ValueError(f"No se puede leer {member.name}")
                        with stream:
                            raw = stream.read()
                        if len(raw) != optional["bytes"] or hashlib.sha256(raw).hexdigest() != optional["sha256"]:
                            raise ValueError(f"git archive modificó el descargable opcional: {original}")
                    excluded.append({"path": path.as_posix(), "reason": reason,
                                     "bytes": member.size, **inventory[original],
                                     **({"optional_download": optional} if optional else {})})
                    continue
                check_size(member.size, member.name)
                target = stage / path
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"No se puede leer {member.name}")
                with source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
                if git_blob_digest(target, inventory[original]["git_blob"]) != inventory[original]["git_blob"]:
                    raise ValueError(f"git archive modificó los bytes de la fuente: {original}")
                target.chmod(0o755 if inventory[original]["git_mode"] == "100755" else 0o644)
                included.add(original)
        expected = {name for name in inventory if exclusion(Path("source") / name, optional_downloads) is None}
        if included != expected:
            raise ValueError(f"git archive no coincide con ls-tree: {sorted(expected - included)}")


def copy_site(site: Path, stage: Path, excluded: list,
              optional_downloads: dict | None = None,
              scientific_journals: dict | None = None) -> dict:
    if site.is_symlink() or not site.is_dir() or not (site / "intro.html").is_file():
        raise ValueError("--site-root debe ser la raíz HTML compilada con intro.html")
    inputs = {}
    for current, dirs, files in os.walk(site, followlinks=False):
        parent = Path(current)
        for name in sorted(dirs + files):
            path = parent / name
            if path.is_symlink():
                raise ValueError(f"No se admiten enlaces simbólicos en HTML: {path}")
        kept_dirs = []
        for name in sorted(dirs):
            path = (parent / name).relative_to(site)
            if canonical_journal_path(path) or optional_record(path, optional_downloads) is not None:
                raise ValueError(f"La copia opcional debe ser un archivo regular, no una carpeta: {path}")
            reason = exclusion(path, optional_downloads)
            if reason:
                excluded.append({"path": (Path("site") / path).as_posix(), "reason": reason})
            else:
                kept_dirs.append(name)
        dirs[:] = kept_dirs
        for name in sorted(files):
            original = parent / name
            path = relative_name(original.relative_to(site).as_posix())
            if not original.is_file():
                raise ValueError(f"Recurso HTML no regular: {path}")
            size = original.stat().st_size
            journal_path = canonical_journal_path(path)
            if journal_path is not None:
                if journal_path not in (scientific_journals or {}):
                    raise ValueError(f"Journal canónico no seleccionado por Git en el HTML: {path}; compile desde una fuente limpia")
                if journal_path not in (optional_downloads or {}):
                    raise ValueError(f"Journal residual en el HTML: {path}; compile limpio o use --optional-scientific-journals explícitamente")
            elif optional_candidates(path, scientific_journals):
                recognized = verified_site_optional(original, path, scientific_journals)
                if recognized is not None and recognized["repository_path"] not in (optional_downloads or {}):
                    raise ValueError(f"Copia Sphinx residual de un journal C0 en el HTML: {path}; compile limpio o use --optional-scientific-journals")
            reason = exclusion(path, optional_downloads)
            optional = None
            if optional_record(path, optional_downloads) is not None:
                optional = verified_site_optional(original, path, optional_downloads)
                if optional is None:
                    reason = None
            if reason:
                excluded.append({"path": (Path("site") / path).as_posix(),
                                 "reason": reason, "bytes": size,
                                 **({"optional_download": optional} if optional else {})})
                continue
            check_size(size, path.as_posix())
            target = stage / "site" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            target.chmod(0o644)
            inputs[path.as_posix()] = {"bytes": size, "sha256": digest(target)}
    return {"files": len(inputs), "bytes": sum(row["bytes"] for row in inputs.values()),
            "inventory_sha256": hashlib.sha256(json_text(inputs).encode()).hexdigest()}


def check_compiled_sources(stage: Path) -> dict:
    compiled = stage / "site/_sources"
    if not (compiled / "intro.md").is_file():
        raise ValueError("Falta site/_sources/intro.md para vincular el HTML con el commit")
    checked = 0
    for path in sorted(compiled.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(compiled)
        source = stage / "source/book_e2" / relative
        if not source.is_file() or digest(path) != digest(source):
            raise ValueError(f"HTML compilado ajeno a --source-commit: _sources/{relative}")
        checked += 1
    return {"compiled_source_files_matched": checked,
            "scope": "Every included Sphinx _sources file matches its selected Git source; "
                     "this is not an independent reproduction of the HTML build."}


def copy_mathjax(donor: Path, stage: Path) -> dict:
    if donor.is_symlink() or not donor.is_file():
        raise ValueError("--mathjax-donor debe ser un payload .tar.xz regular")
    target = stage / MATHJAX_PREFIX
    # An existing cache must not supply extra or stale MathJax files.
    if target.exists():
        shutil.rmtree(target)
    with tarfile.open(donor, "r:xz") as archive:
        members = archive.getmembers()
        manifest_members = [member for member in members if member.name == MANIFEST]
        if len(manifest_members) != 1 or not manifest_members[0].isfile():
            raise ValueError("El donante no tiene un manifiesto único regular")
        if manifest_members[0].size > 32 * 1024**2:
            raise ValueError("Manifiesto del donante demasiado grande")
        source = archive.extractfile(manifest_members[0])
        if source is None:
            raise ValueError("No se puede leer el manifiesto del donante")
        with source:
            manifest = json.load(source)
        records = manifest.get("files")
        provenance = manifest.get("mathjax")
        if not isinstance(records, dict) or not isinstance(provenance, dict):
            raise ValueError("El donante carece del inventario y procedencia MathJax")
        if provenance.get("version") != MATHJAX_VERSION:
            raise ValueError("El donante debe acreditar MathJax 3.2.2")
        expected = {name for name in records if name.startswith(MATHJAX_PREFIX)}
        required = {MATHJAX_PREFIX + name for name in
                    ("es5/tex-mml-chtml.js", "LICENSE", "PROVENANCE.json")}
        if not required <= expected or len(expected) != 108:
            raise ValueError("MathJax donante debe contener los 108 archivos de la distribución 3.2.2 fijada")
        seen = set()
        total = 0
        for member in members:
            if not member.name.startswith(MATHJAX_PREFIX):
                continue
            path = relative_name(member.name)
            if member.isdir():
                continue
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError(f"Recurso MathJax no regular, repetido o no inventariado: {member.name}")
            record = records[member.name]
            check_size(member.size, member.name)
            if member.size != record["bytes"]:
                raise ValueError(f"Tamaño MathJax distinto del inventario: {member.name}")
            total += member.size
            if total > MAX_EXPANDED_BYTES:
                raise ValueError("MathJax donante excede el límite expandido")
            destination = stage / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"No se puede leer {member.name}")
            with source, destination.open("xb") as output:
                shutil.copyfileobj(source, output)
            destination.chmod(0o644)
            if digest(destination) != record["sha256"]:
                raise ValueError(f"SHA-256 MathJax distinto del inventario: {member.name}")
            seen.add(member.name)
        if seen != expected:
            raise ValueError(f"Recursos MathJax ausentes: {sorted(expected - seen)}")
        copied_provenance = json.loads((target / "PROVENANCE.json").read_text(encoding="utf-8"))
        if copied_provenance != provenance:
            raise ValueError("La procedencia MathJax no coincide con el manifiesto donante")
    return {"files": len(seen), "bytes": total, "provenance": provenance,
            "donor_sha256": digest(donor)}


def readme(commit: str, optional_downloads: dict | None = None,
           optional_journals: dict | None = None) -> str:
    optional_note = ""
    archive_note = "Los ZIP históricos de evidencia científica se conservan cuando forman parte\nde la fuente seleccionada."
    if optional_downloads:
        archive_note = "Dos ZIP científicos redundantes se ofrecen como descargas opcionales HTTPS;\nno están incluidos en esta carpeta. Los demás recursos seleccionados se conservan."
        optional_note = "\n## Descargas científicas opcionales\n\n"
        optional_note += "La lectura sin conexión y la reconstrucción del libro no necesitan estos ZIP.\nDescargarlos requiere Internet; su disponibilidad remota no se prueba por red al\nconstruir este paquete. Sus bytes se verificaron en ambos commits de Git.\n\n"
        for record in optional_downloads.values():
            optional_note += (f"- [{Path(record['repository_path']).name}]({record['url']}): "
                              f"{record['bytes']} bytes; SHA-256 `{record['sha256']}`.\n")
    if optional_journals:
        optional_note += (f"\nEl catálogo describe {len(optional_journals)} journals gzip "
                          f"({sum(record['bytes'] for record in optional_journals.values())} bytes) "
                          "presentes como miembros ZIP, no como archivos fuente Git individuales. "
                          "El opt-in permite omitir únicamente sus copias residuales reconocidas "
                          "en un HTML anterior; en una compilación limpia puede no haber ninguna. "
                          "El catálogo `optional_scientific_journals` del manifiesto identifica "
                          "cada ruta, SHA-256, tamaño, ZIP C0 y miembro exacto. Los journals y "
                          "ambos ZIP se conservan en los entregables científicos D04/D11.\n\n"
                          "D01 permite leer y reconstruir el libro; no contiene todos los insumos "
                          "para ejecutar las auditorías científicas de esos journals sin obtener "
                          "el suplemento. No se sustituyen por archivos vacíos ni se modifican "
                          "sus manifiestos científicos originales.\n")
    return f"""# D01 · Libro completo y fuentes seleccionadas del Entregable 2

Con Python 3.10 o posterior, ejecute `python run.py` y abra `output/index.html`.
El modo de lectura usa Python estándar, sin Git, clon del repositorio ni paquetes
adicionales. `python run.py --serve --port 0` ofrece un servidor local opcional.

`site/` contiene el HTML compilado y los recursos de lectura, incluido MathJax
3.2.2, LICENSE y PROVENANCE.json. Las referencias externas y los doce entregables
independientes conservan enlaces que necesitan Internet si se visitan. Estos
doce ZIP no se incluyen dentro del libro: D01 no se anida dentro de sí mismo.
{archive_note} El manifiesto enumera todas las exclusiones.

`source/` contiene archivos seleccionados, sin modificación, del commit
`{commit}`: `book_e2`, requisitos del libro, BUILD y los scripts de reconstrucción
y verificación enumerados en `bundle_manifest.json`. La bibliografía necesaria
está en `source/book_e2/references.bib`. No es una copia completa del repositorio;
no incorpora `.git`, salidas privadas, pesos, cachés ni los doce ZIP anidados.
Sólo se incluyen fuentes seleccionadas de Git: no se añaden journals individuales
desde un árbol de trabajo. Los insumos interiores de los ZIP científicos mantienen
su identidad de miembros y no se anuncian como archivos fuente individuales.

Para reconstruir el libro, instale las dependencias con
`python -m pip install -r output/source/requirements-book.lock.txt` y ejecute
`python run.py --build`. Instalar dependencias puede necesitar Internet. La
reconstrucción conserva `execute_notebooks=off`; el runner usa MathJax local.
BUILD.md se conserva como documentación del repositorio completo; los libros
E1/E3 y sus comandos no forman parte de este paquete de E2.

El modo predeterminado comprueba hashes, enlaces y recursos HTML. No entrena,
abre checkpoints, genera predicciones ni realiza nuevos análisis estadísticos.
El HTML suministrado sólo se adapta para resolver MathJax localmente; el
inventario identifica esa adaptación y los archivos exactos incluidos.
{optional_note}"""


def inventory_files(stage: Path) -> dict:
    files = {}
    total = 0
    for path in sorted(stage.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Enlace no permitido en staging: {path}")
        if path.is_file():
            size = path.stat().st_size
            check_size(size, path.relative_to(stage).as_posix())
            total += size
            files[path.relative_to(stage).as_posix()] = {"bytes": size, "sha256": digest(path)}
    if len(files) > MAX_FILES or total > MAX_EXPANDED_BYTES:
        raise ValueError("El payload excede los límites de extracción del runner")
    return files


def write_archive(stage: Path, destination: Path) -> None:
    with lzma.open(destination, "wb", format=lzma.FORMAT_XZ,
                   filters=[{"id": lzma.FILTER_LZMA2, "preset": 6,
                             "dict_size": 64 * 1024 * 1024}]) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            for path in sorted(stage.rglob("*"), key=lambda item: item.relative_to(stage).as_posix()):
                name = path.relative_to(stage).as_posix()
                member = tarfile.TarInfo(name)
                member.uid = member.gid = member.mtime = 0
                member.uname = member.gname = ""
                if path.is_dir():
                    member.type = tarfile.DIRTYPE
                    member.mode = 0o755
                    archive.addfile(member)
                elif path.is_file() and not path.is_symlink():
                    member.mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
                    member.size = path.stat().st_size
                    with path.open("rb") as stream:
                        archive.addfile(member, stream)
                else:
                    raise ValueError(f"Entrada no regular: {name}")


def build(args: argparse.Namespace) -> dict:
    commit = args.source_commit.lower()
    entries = source_inventory(commit)
    verify_executed_builder(entries)
    optional_downloads = optional_download_catalogue(
        getattr(args, "optional_download_commit", None), entries)
    journals_enabled = getattr(args, "optional_scientific_journals", False)
    if journals_enabled and not optional_downloads:
        raise ValueError("--optional-scientific-journals requiere --optional-download-commit")
    archive_sources = optional_downloads or source_archive_catalogue(commit, entries)
    scientific_journals = scientific_journal_catalogue(archive_sources, entries) if archive_sources else {}
    optional_journals = scientific_journals if journals_enabled else {}
    optional_resources = {**optional_downloads, **optional_journals}
    site = args.site_root.absolute()
    output = args.output.absolute()
    donor = args.mathjax_donor.absolute()
    if output.is_symlink() or output.exists():
        raise ValueError("--output debe ser un archivo nuevo; no se sobrescribe un payload")
    if not output.name.endswith(".tar.xz"):
        raise ValueError("--output debe terminar en .tar.xz")
    if output.resolve().is_relative_to(site.resolve()):
        raise ValueError("--output no puede estar dentro de --site-root")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="e2-d01-payload-") as directory:
        stage = Path(directory)
        excluded = []
        copy_source(commit, stage, entries, excluded, optional_resources)
        compiled = copy_site(site, stage, excluded, optional_resources, scientific_journals)
        source_check = check_compiled_sources(stage)
        if optional_downloads or scientific_journals:
            source_check["optional_download_dependencies"] = check_optional_source_links(
                stage, {**optional_resources, **scientific_journals})
        mathjax = copy_mathjax(donor, stage)
        runner = runpy.run_path(str(stage / "source/scripts/e2_standalone_book.py"))
        if "source_commit" not in inspect.signature(runner["verify_files"]).parameters:
            raise ValueError("El runner del commit debe admitir verify_files(root, source_commit=None)")
        if optional_downloads and "optional_downloads" not in runner:
            raise ValueError("El commit de fuente debe contener el lector de catálogos opcionales v3")
        if optional_journals and "optional_scientific_journals" not in runner:
            raise ValueError("El commit de fuente debe contener el lector del catálogo de 296 journals")
        localized = runner["localize_mathjax"](stage / "site", stage / "site")
        try:
            html = runner["validate_html"](stage / "site")
        except ValueError as error:
            raise ValueError("El HTML no cumple la lectura autónoma. Las exclusiones no autorizan "
                             "enlaces locales rotos; convierta los enlaces opcionales a HTTPS "
                             "en la fuente antes de fijar el commit y compilar.\n" + str(error)) from error
        (stage / "LEEME_D01.md").write_text(readme(commit, optional_downloads, optional_journals), encoding="utf-8")
        files = inventory_files(stage)
        selected = {name: metadata for name, metadata in entries.items()
                    if exclusion(Path("source") / name, optional_resources) is None}
        manifest = {
            "version": "e2-standalone-book-payload-v3" if optional_downloads else "e2-standalone-book-payload-v2", "source_commit": commit,
            "compiled_snapshot": compiled, "compiled_source_check": source_check,
            "scientific_artifacts_modified": False,
            "site_mathjax_links_localized": localized, "mathjax": mathjax["provenance"],
            "mathjax_donor": {key: value for key, value in mathjax.items() if key != "provenance"},
            "html_validation": html, "source_paths": list(SOURCE_PATHS),
            "source_files_git": selected,
            "excluded": sorted(excluded, key=lambda row: row["path"]),
            "exclusion_policy": {
                "directories": sorted(EXCLUDED_DIRS), "weights": sorted(WEIGHT_SUFFIXES),
                "independent_deliverable_zips": sorted(DELIVERABLE_ZIPS),
                "recursive_payloads": sorted(RECURSIVE_PAYLOADS),
                "historical_scientific_archives": "retained except the two explicitly catalogued optional downloads" if optional_downloads else "retained",
                "other_repository_paths": "not selected; source_paths is an explicit allowlist",
            },
            "archive_normalization": {"mtime": 0, "uid": 0, "gid": 0,
                                      "file_mode": "0644, or 0755 for Git executable sources",
                                      "directory_mode": "0755", "order": "POSIX path lexical"},
            "github_per_file_limit_bytes": GITHUB_FILE_LIMIT, "files": files,
        }
        if optional_downloads:
            manifest["optional_downloads"] = list(optional_downloads.values())
        if optional_journals:
            manifest["optional_scientific_journals"] = list(optional_journals.values())
            manifest["exclusion_policy"]["scientific_journals"] = "only recognized residual site/static/Sphinx copies of 206+90 ZIP-member identities; no individual journal Git source is included or claimed"
        (stage / MANIFEST).write_text(json_text(manifest), encoding="utf-8")
        # Check the runtime contract before creating any distributable archive.
        verification = runner["verify_files"](stage, source_commit=commit)
        all_files = inventory_files(stage)
        if set(all_files) != set(files) | {MANIFEST}:
            raise ValueError("El manifiesto no cubre exactamente los archivos del payload")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(prefix=".d01-payload-", suffix=".tar.xz",
                                             dir=output.parent, delete=False) as handle:
                temporary = Path(handle.name)
            write_archive(stage, temporary)
            check_size(temporary.stat().st_size, output.name)
            report = {
                "version": "e2-standalone-book-payload-build-v1", "source_commit": commit,
                "payload": str(output), "payload_bytes": temporary.stat().st_size,
                "payload_sha256": digest(temporary), "expanded_files": len(all_files),
                "expanded_bytes": sum(row["bytes"] for row in all_files.values()),
                "manifest_sha256": digest(stage / MANIFEST),
                "source_selected_files": len(selected), "excluded_count": len(excluded),
                "excluded": manifest["excluded"], "compiled_snapshot": compiled,
                "compiled_source_check": source_check, "mathjax": mathjax,
                "html_validation": html, "runtime_verification": verification,
                "github_per_file_limit_bytes": GITHUB_FILE_LIMIT,
                "book_built": False, "scientific_experiments_executed": False,
            }
            if optional_downloads:
                report["optional_downloads"] = list(optional_downloads.values())
            if optional_journals:
                report["optional_scientific_journals"] = list(optional_journals.values())
            # Exclusive publication avoids accidentally replacing an existing release.
            os.link(temporary, output)
            return report
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True,
                        help="Immutable 40-character commit containing the selected sources")
    parser.add_argument("--site-root", required=True, type=Path,
                        help="HTML root already compiled from that snapshot")
    parser.add_argument("--mathjax-donor", type=Path,
                        default=Path("/tmp/e2-standalone-book-payload.tar.xz"),
                        help="Existing payload; only its inventoried MathJax 3.2.2 is extracted")
    parser.add_argument("--output", required=True, type=Path, help="New .tar.xz payload")
    parser.add_argument("--optional-download-commit",
                        help="Explicit immutable 40-character Git commit for the two pinned scientific ZIPs; opt in to exclude their verified copies")
    parser.add_argument("--optional-scientific-journals", action="store_true",
                        help="Also omit only the exact 206+90 journals bound to members of the two ZIPs; requires --optional-download-commit")
    args = parser.parse_args(argv)
    if args.optional_scientific_journals and not args.optional_download_commit:
        parser.error("--optional-scientific-journals requiere --optional-download-commit")
    try:
        report = build(args)
    except (ValueError, OSError, subprocess.CalledProcessError, tarfile.TarError, zipfile.BadZipFile,
            KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json_text(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
