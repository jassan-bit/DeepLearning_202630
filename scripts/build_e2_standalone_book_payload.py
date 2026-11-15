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


def exclusion(path: Path) -> str | None:
    if any(part in EXCLUDED_DIRS for part in path.parts):
        return "generated_or_private_directory"
    if path.name in DELIVERABLE_ZIPS:
        return "independent_deliverable_zip_and_sphinx_copy"
    if path.suffix.lower() in WEIGHT_SUFFIXES:
        return "model_weights"
    if path.name in RECURSIVE_PAYLOADS:
        return "recursive_payload"
    return None


def check_size(size: int, name: str) -> None:
    if size >= GITHUB_FILE_LIMIT:
        raise ValueError(f"{name}: {size} bytes alcanza el límite de 100.000.000 bytes")


def git(*arguments: str) -> bytes:
    return subprocess.check_output(["git", *arguments], cwd=ROOT)


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


def copy_source(commit: str, stage: Path, inventory: dict, excluded: list) -> None:
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
                reason = exclusion(path)
                if reason:
                    excluded.append({"path": path.as_posix(), "reason": reason,
                                     "bytes": member.size, **inventory[original]})
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
        expected = {name for name in inventory if exclusion(Path("source") / name) is None}
        if included != expected:
            raise ValueError(f"git archive no coincide con ls-tree: {sorted(expected - included)}")


def copy_site(site: Path, stage: Path, excluded: list) -> dict:
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
            reason = exclusion(path)
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
            reason = exclusion(path)
            if reason:
                excluded.append({"path": (Path("site") / path).as_posix(),
                                 "reason": reason, "bytes": size})
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


def readme(commit: str) -> str:
    return f"""# D01 · Libro completo y fuentes seleccionadas del Entregable 2

Con Python 3.10 o posterior, ejecute `python run.py` y abra `output/index.html`.
El modo de lectura usa Python estándar, sin Git, clon del repositorio ni paquetes
adicionales. `python run.py --serve --port 0` ofrece un servidor local opcional.

`site/` contiene el HTML compilado y los recursos de lectura, incluido MathJax
3.2.2, LICENSE y PROVENANCE.json. Las referencias externas y los doce entregables
independientes conservan enlaces que necesitan Internet si se visitan. Estos
doce ZIP no se incluyen dentro del libro: D01 no se anida dentro de sí mismo.
Los ZIP históricos de evidencia científica sí se conservan cuando forman parte
de la fuente seleccionada. El manifiesto enumera todas las exclusiones.

`source/` contiene archivos seleccionados, sin modificación, del commit
`{commit}`: `book_e2`, requisitos del libro, BUILD y los scripts de reconstrucción
y verificación enumerados en `bundle_manifest.json`. La bibliografía necesaria
está en `source/book_e2/references.bib`. No es una copia completa del repositorio;
no incorpora `.git`, salidas privadas, pesos, cachés ni los doce ZIP anidados.

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
"""


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
        copy_source(commit, stage, entries, excluded)
        compiled = copy_site(site, stage, excluded)
        source_check = check_compiled_sources(stage)
        mathjax = copy_mathjax(donor, stage)
        runner = runpy.run_path(str(stage / "source/scripts/e2_standalone_book.py"))
        if "source_commit" not in inspect.signature(runner["verify_files"]).parameters:
            raise ValueError("El runner del commit debe admitir verify_files(root, source_commit=None)")
        localized = runner["localize_mathjax"](stage / "site", stage / "site")
        try:
            html = runner["validate_html"](stage / "site")
        except ValueError as error:
            raise ValueError("El HTML no cumple la lectura autónoma. Las exclusiones no autorizan "
                             "enlaces locales rotos; convierta los doce enlaces opcionales a HTTPS "
                             "en la fuente antes de fijar el commit y compilar.\n" + str(error)) from error
        (stage / "LEEME_D01.md").write_text(readme(commit), encoding="utf-8")
        files = inventory_files(stage)
        selected = {name: metadata for name, metadata in entries.items()
                    if exclusion(Path("source") / name) is None}
        manifest = {
            "version": "e2-standalone-book-payload-v2", "source_commit": commit,
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
                "historical_scientific_archives": "retained",
                "other_repository_paths": "not selected; source_paths is an explicit allowlist",
            },
            "archive_normalization": {"mtime": 0, "uid": 0, "gid": 0,
                                      "file_mode": "0644, or 0755 for Git executable sources",
                                      "directory_mode": "0755", "order": "POSIX path lexical"},
            "github_per_file_limit_bytes": GITHUB_FILE_LIMIT, "files": files,
        }
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
    args = parser.parse_args(argv)
    try:
        report = build(args)
    except (ValueError, OSError, subprocess.CalledProcessError, tarfile.TarError,
            KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json_text(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
