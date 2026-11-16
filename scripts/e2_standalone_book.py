#!/usr/bin/env python3
"""D01: extract, inspect, serve or rebuild a self-contained E2 book.

Default operation uses only Python's standard library.  It never downloads data,
installs packages, trains models or executes notebooks.  The adjacent payload is
an immutable book/source snapshot; a requested source rebuild needs Jupyter Book.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
from html import escape
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from urllib.parse import unquote, urlsplit

PAYLOAD_NAME = "payload.tar.xz"
MANIFEST_NAME = "bundle_manifest.json"
MAX_FILES = 30000
MAX_EXPANDED_BYTES = 2 * 1024 ** 3
MATHJAX_VERSION = "3.2.2"
MATHJAX_RELATIVE = Path("_static/vendor/mathjax/es5/tex-mml-chtml.js")
MATHJAX_CDN = re.compile(
    r"https://cdn\.jsdelivr\.net/npm/mathjax@(?:3|3\.2\.2)/es5/tex-mml-chtml\.js"
)
OPTIONAL_DOWNLOAD_PREFIX = "https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/"
OPTIONAL_DOWNLOAD_PATHS = frozenset((
    "book_e2/_static/e2_scientific_closure_20261116/preregistration_bundle.zip",
    "book_e2/_static/e2_scientific_closure_20261116/final_registration_bundle.zip",
))
OPTIONAL_JOURNAL_ROOTS = {
    "book_e2/_static/e2_scientific_closure_20261116/preregistration/journals/": (
        "book_e2/_static/e2_scientific_closure_20261116/preregistration_bundle.zip",
        "Preregistro_E2/journals/", 206),
    "book_e2/_static/e2_scientific_closure_20261116/final_registration_chronology/journals/": (
        "book_e2/_static/e2_scientific_closure_20261116/final_registration_bundle.zip",
        "Registro_Final_E2/journals/", 90),
}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def safe_relative(name: str) -> Path:
    if not name or "\\" in name or "\x00" in name or ":" in name:
        raise ValueError(f"Ruta no permitida en el paquete: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("..", ".git") for part in path.parts):
        raise ValueError(f"Ruta fuera del paquete: {name!r}")
    if not path.parts or path.parts[0] not in ("site", "source", MANIFEST_NAME, "LEEME_D01.md"):
        raise ValueError(f"Raíz de archivo inesperada: {name!r}")
    return Path(*path.parts)


def confined(root: Path, relative: Path) -> Path:
    candidate = root / relative
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"La ruta sale de la carpeta de salida: {relative}")
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"No se admiten enlaces simbólicos: {relative}")
    return candidate


def reject_symlinks(root: Path) -> None:
    if root.is_symlink():
        raise ValueError("La salida no puede ser un enlace simbólico")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Enlace simbólico no permitido en la salida: {path.relative_to(root)}")


def advertised_source_commit() -> str:
    """The extracted package declares its immutable source, without Git."""
    package = Path(__file__).resolve().parent / "package.json"
    value = json.loads(package.read_text(encoding="utf-8")).get("source_commit")
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("package.json debe declarar un commit completo de fuente")
    return value


def optional_downloads(manifest: dict) -> list[dict]:
    """Validate a v3 catalogue without fetching any optional resource."""
    records = manifest.get("optional_downloads", [])
    if not records:
        if manifest.get("version") == "e2-standalone-book-payload-v3":
            raise ValueError("El manifiesto v3 requiere su catálogo opcional explícito")
        return []  # Payload v1/v2 readers keep their existing behavior.
    if manifest.get("version") != "e2-standalone-book-payload-v3" or not isinstance(records, list):
        raise ValueError("Catálogo opcional sólo admitido en un manifiesto v3")
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Cada descarga opcional debe ser un registro explícito")
        path, commit = record.get("repository_path"), record.get("download_commit")
        if (path not in OPTIONAL_DOWNLOAD_PATHS or path in seen
                or not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit)
                or record.get("url") != f"{OPTIONAL_DOWNLOAD_PREFIX}{commit}/{path}"
                or not isinstance(record.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
                or type(record.get("bytes")) is not int or not 0 < record["bytes"] < 100_000_000
                or not isinstance(record.get("reason"), str) or not record["reason"]):
            raise ValueError("Catálogo opcional con ruta, identidad o HTTPS inmutable inválido")
        seen.add(path)
    if seen != OPTIONAL_DOWNLOAD_PATHS:
        raise ValueError("El catálogo opcional debe identificar exactamente los dos ZIP autorizados")
    return records


def optional_scientific_journals(manifest: dict, downloads: list[dict]) -> list[dict]:
    """Validate optional journal identities/member aliases without opening ZIPs."""
    records = manifest.get("optional_scientific_journals", [])
    if not records:
        return []
    if (manifest.get("version") != "e2-standalone-book-payload-v3"
            or not downloads or not isinstance(records, list)):
        raise ValueError("El catálogo journal requiere v3 y sus dos ZIP opcionales")
    archives = {record["repository_path"]: record for record in downloads}
    counts = {prefix: 0 for prefix in OPTIONAL_JOURNAL_ROOTS}
    paths, hashes = set(), set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("repository_path"), str):
            raise ValueError("Registro journal opcional inválido")
        path = record["repository_path"]
        prefix = next((root for root in counts if path.startswith(root)), None)
        if prefix is None:
            raise ValueError("Journal fuera de las dos carpetas autorizadas")
        bundle_path, member_prefix, _ = OPTIONAL_JOURNAL_ROOTS[prefix]
        bundle = archives[bundle_path]
        relative = path[len(prefix):]
        if (not re.fullmatch(r"[0-9a-f]{64}/(?:ae1|ae2|supervised)\.jsonl\.gz", relative)
                or path in paths or not isinstance(record.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
                or record["sha256"] in hashes
                or type(record.get("bytes")) is not int or not 0 < record["bytes"] < 100_000_000
                or record.get("bundle_repository_path") != bundle_path
                or record.get("member_path") != member_prefix + relative
                or record.get("bundle_bytes") != bundle["bytes"]
                or record.get("bundle_sha256") != bundle["sha256"]
                or record.get("download_commit") != bundle["download_commit"]
                or record.get("url") != bundle["url"]
                or record.get("source_individual_git_present") is not False
                or not isinstance(record.get("source_archive_git_blob"), str)
                or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", record["source_archive_git_blob"])
                or record.get("source_binding") != "Git-tracked ZIP member; no individual journal Git source is claimed"
                or not isinstance(record.get("reason"), str) or not record["reason"]):
            raise ValueError("Identidad, miembro ZIP o ruta journal opcional inválidos")
        counts[prefix] += 1
        paths.add(path)
        hashes.add(record["sha256"])
    if any(counts[prefix] != specification[2] for prefix, specification in OPTIONAL_JOURNAL_ROOTS.items()):
        raise ValueError("El catálogo journal debe contener exactamente 206+90 rutas")
    return records


def verify_files(root: Path, source_commit: str | None = None) -> dict:
    reject_symlinks(root)
    manifest_path = root / MANIFEST_NAME
    if manifest_path.is_symlink():
        raise ValueError("El manifiesto no puede ser un enlace simbólico")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_commit = source_commit or advertised_source_commit()
    if manifest.get("source_commit") != expected_commit:
        raise ValueError("El paquete no corresponde al commit de fuente anunciado")
    downloads = optional_downloads(manifest)
    journals = optional_scientific_journals(manifest, downloads)
    entries = manifest.get("files")
    if not isinstance(entries, dict) or not entries or len(entries) > MAX_FILES:
        raise ValueError("Inventario de archivos inválido")
    for record in downloads + journals:
        site_path = PurePosixPath(*PurePosixPath(record["repository_path"]).parts[1:])
        if ("source/" + record["repository_path"] in entries
                or "site/" + site_path.as_posix() in entries):
            raise ValueError("Un archivo declarado opcional no puede estar incluido a la vez")
    total = 0
    for name, record in entries.items():
        path = confined(root, safe_relative(name))
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise ValueError(f"Archivo ausente o tamaño distinto: {name}")
        if digest(path) != record["sha256"]:
            raise ValueError(f"Contenido modificado: {name}")
        total += record["bytes"]
    report = {"verified_files": len(entries), "verified_bytes": total,
              "source_commit": expected_commit, "mathjax_version": MATHJAX_VERSION}
    if downloads:
        report["optional_downloads"] = downloads
    if journals:
        report["optional_scientific_journals"] = journals
    return report


def extract_payload(payload: Path, output: Path, source_commit: str | None = None) -> dict:
    if not payload.is_file() or payload.is_symlink():
        raise ValueError(f"Falta el archivo regular {payload}")
    if output.is_symlink():
        raise ValueError("La carpeta de salida no puede ser un enlace simbólico")
    if output.exists():
        if not output.is_dir() or not (output / MANIFEST_NAME).is_file():
            raise ValueError("La salida ya existe y no es una extracción D01; elija otra carpeta")
        return verify_files(output, source_commit)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".d01-extract-", dir=output.parent))
    try:
        seen, total, file_count = set(), 0, 0
        with tarfile.open(payload, "r:xz") as archive:
            for member in archive:
                relative = safe_relative(member.name)
                if member.name in seen:
                    raise ValueError(f"Entrada repetida: {member.name}")
                seen.add(member.name)
                if len(seen) > 2 * MAX_FILES:
                    raise ValueError("Demasiadas entradas de archivo/directorio")
                if not member.isdir() and not member.isfile():
                    raise ValueError(f"Tipo de archivo no permitido: {member.name}")
                target = confined(temporary, relative)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                file_count += 1
                total += member.size
                if member.size < 0 or total > MAX_EXPANDED_BYTES or file_count > MAX_FILES:
                    raise ValueError("El paquete excede el límite de extracción")
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"No se pudo leer: {member.name}")
                with stream, target.open("xb") as destination:
                    shutil.copyfileobj(stream, destination)
                target.chmod(0o755 if member.mode & 0o111 else 0o644)
        report = verify_files(temporary, source_commit)
        temporary.rename(output)
        return report
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.links, self.remote_resources = set(), [], []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "id" in values:
            self.ids.add(values["id"])
        for attribute in ("href", "src"):
            if attribute not in values:
                continue
            link = values[attribute]
            self.links.append(link)
            parts = urlsplit(link)
            if (parts.scheme or parts.netloc) and tag in ("script", "img", "link", "iframe"):
                self.remote_resources.append(link)


class ConfinedHTTPHandler(SimpleHTTPRequestHandler):
    """Retain the stdlib server while refusing paths through symlinks."""

    def send_head(self):
        relative = unquote(urlsplit(self.path).path).lstrip("/")
        if "\\" in relative or ".." in PurePosixPath(relative).parts:
            self.send_error(403, "Ruta fuera del libro")
            return None
        try:
            root = Path(self.directory).resolve()
            candidate = confined(root, Path(relative))
            if candidate.is_dir():
                for name in ("index.html", "index.htm"):
                    confined(root, Path(relative) / name)
        except ValueError:
            self.send_error(403, "Enlace fuera del libro")
            return None
        return super().send_head()


def validate_html(site: Path) -> dict:
    site = site.resolve()
    pages = {}
    for path in sorted(site.rglob("*.html")):
        if "_static" in path.relative_to(site).parts:
            continue  # Theme assets include unrendered HTML template fragments.
        page = Page()
        page.feed(path.read_text(encoding="utf-8"))
        pages[path.resolve()] = page
    if not pages or not (site / "intro.html").is_file():
        raise ValueError("El libro extraído no contiene su portada HTML")
    checked, errors = 0, []
    for path, page in pages.items():
        for resource in page.remote_resources:
            errors.append(f"{path.relative_to(site)}: recurso externo necesario: {resource}")
        for link in page.links:
            parts = urlsplit(link)
            if parts.scheme or parts.netloc:
                continue  # Bibliographic/external navigation links remain external.
            target = (path.parent / unquote(parts.path)).resolve() if parts.path else path
            if target.is_dir():
                target /= "index.html"
            if not target.is_relative_to(site):
                errors.append(f"{path.relative_to(site)}: ruta fuera del libro: {link}")
            elif not target.is_file():
                errors.append(f"{path.relative_to(site)}: falta {link}")
            elif parts.fragment and target in pages and unquote(parts.fragment) not in pages[target].ids:
                errors.append(f"{path.relative_to(site)}: falta ancla {link}")
            checked += 1
    # Fonts and images referenced by CSS are part of offline readability too.
    css_checked = 0
    for css in site.rglob("*.css"):
        for raw in re.findall(r"url\(\s*([^)]*?)\s*\)", css.read_text(encoding="utf-8", errors="replace")):
            link = raw.strip("\"' ")
            parts = urlsplit(link)
            if not link or parts.scheme == "data" or not parts.path:
                continue
            if parts.scheme or parts.netloc:
                errors.append(f"{css.relative_to(site)}: recurso CSS externo: {link}")
                continue
            target = (css.parent / unquote(parts.path)).resolve()
            if not target.is_relative_to(site) or not target.is_file():
                errors.append(f"{css.relative_to(site)}: recurso CSS ausente: {link}")
            css_checked += 1
    if errors:
        raise ValueError("\n".join(errors[:50]))
    return {"html_pages": len(pages), "local_links_and_assets_checked": checked,
            "css_resources_checked": css_checked, "remote_required_resources": 0}


def localize_mathjax(site: Path, bundled_site: Path) -> int:
    target = site / "_static/vendor/mathjax"
    if not target.exists():
        shutil.copytree(bundled_site / "_static/vendor/mathjax", target)
    changed = 0
    for page in site.rglob("*.html"):
        original = page.read_text(encoding="utf-8")
        relative = os.path.relpath(site / MATHJAX_RELATIVE, page.parent).replace(os.sep, "/")
        replaced, count = MATHJAX_CDN.subn(relative, original)
        if count:
            page.write_text(replaced, encoding="utf-8")
            changed += count
    return changed


def rebuild(output: Path) -> Path:
    if importlib.util.find_spec("jupyter_book") is None:
        requirements = output / "source/requirements-book.lock.txt"
        raise ValueError("--build requiere Jupyter Book instalado. Preparación opcional: "
                         f'"{sys.executable}" -m pip install -r "{requirements}". '
                         "El modo de lectura no necesita estos paquetes.")
    destination = output / "rebuild"
    # Jupyter Book 1.0.4 (the included lock) has no package __main__.  Use
    # the same entry point as its installed jupyter-book console script.
    entrypoint = "from jupyter_book.cli.main import main\nmain()"
    subprocess.run([sys.executable, "-c", entrypoint, "build",
                    str(output / "source/book_e2"), "--path-output", str(destination)], check=True)
    site = destination / "_build/html"
    if not site.is_dir():
        raise ValueError(f"No se encontró la salida de Jupyter Book: {site}")
    localize_mathjax(site, output / "site")
    validate_html(site)
    return site


def write_entry(output: Path, report: dict, rebuilt: Path | None = None) -> None:
    reject_symlinks(output)
    rebuilt_link = (f'<p><a href="{escape(rebuilt.relative_to(output).as_posix())}/intro.html">'
                    "Abrir la reconstrucción desde las fuentes</a></p>" if rebuilt else "")
    downloads = report.get("optional_downloads", [])
    optional_links = ""
    if downloads:
        optional_links = ("<h2>Descargas científicas opcionales</h2><p>El libro, sus figuras y las "
                          "fuentes seleccionadas para reconstruirlo se leen sin estos ZIP. "
                          "Descargarlos requiere Internet.</p><ul>" + "".join(
            f'<li><a href="{escape(record["url"], quote=True)}">'
            f'{escape(Path(record["repository_path"]).name)}</a>: {record["bytes"]} bytes; '
            f'SHA-256 <code>{escape(record["sha256"])}</code>.</li>' for record in downloads) + "</ul>")
    journals = report.get("optional_scientific_journals", [])
    if journals:
        optional_links += (f"<p>El catálogo describe {len(journals)} journals gzip como miembros "
                           "de los ZIP opcionales; sus rutas individuales no son fuentes Git incluidas. "
                           "El opt-in sólo permite omitir copias residuales reconocidas en el HTML. "
                           "El manifiesto indica la ruta, SHA-256 y miembro de los ZIP anteriores "
                           "para recuperar cada uno. D04/D11 conservan esos insumos científicos; "
                           "la lectura y reconstrucción de este libro no los ejecuta ni los necesita.</p>")
    document = f"""<!doctype html><html lang="es"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>D01 · Libro E2 autónomo</title>
<style>body{{font:18px/1.6 system-ui,sans-serif;max-width:860px;margin:3rem auto;padding:0 1.2rem}}
a{{color:#125b91}}code{{background:#f0f3f6;padding:.15em .3em}}li{{margin:.6em 0}}</style>
<h1>Libro del Entregable 2</h1><p>El libro y sus fuentes seleccionadas están en esta carpeta.
La lectura del contenido, las figuras, la navegación y las fórmulas no necesita el sitio web.</p>
<p><strong><a href="site/intro.html">Abrir el libro completo</a></strong></p>{rebuilt_link}
<ul><li>Instantánea de fuente: <code>{report['source_commit']}</code>.</li>
<li>Archivos verificados: {report['verified_files']}. Páginas HTML: {report['html_pages']}.</li>
<li>Las fórmulas usan MathJax {MATHJAX_VERSION}, incluido con su
<a href="site/_static/vendor/mathjax/LICENSE">licencia</a> y
<a href="site/_static/vendor/mathjax/PROVENANCE.json">procedencia</a> en el paquete.</li>
<li>Las referencias bibliográficas y servicios externos mantienen sus enlaces; visitarlos puede requerir Internet.</li>
<li>Para servir este contenido localmente: <code>python run.py --serve</code>.
Sólo escucha en <code>127.0.0.1</code>; puede elegir puerto con <code>--port 0</code>.</li>
<li>Las fuentes exactas están en <code>source/</code>. La opción <code>--build</code> reconstruye
el libro si Jupyter Book ya está instalado. Sus dependencias se indican en
<code>source/requirements-book.lock.txt</code>.</li>
<li>Esta entrega no entrena modelos, no ejecuta notebooks y no convierte la reproducción editorial en un experimento nuevo.</li></ul>
{optional_links}<p>Instrucciones y alcance: <a href="LEEME_D01.md">LEEME_D01.md</a>.</p></html>"""
    contents = {"index.html": document,
                "extraction_report.json": json.dumps(report, ensure_ascii=False, indent=2) + "\n"}
    for name, content in contents.items():
        target = confined(output, Path(name))
        descriptor, temporary = tempfile.mkstemp(prefix=".d01-entry-", dir=output)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(content)
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--extract", action="store_true", help="Extraer y validar; también es el modo predeterminado")
    parser.add_argument("--serve", action="store_true", help="Servir el libro en 127.0.0.1")
    parser.add_argument("--port", type=int, default=8765, help="Puerto local; 0 elige uno libre")
    parser.add_argument("--build", action="store_true", help="Reconstruir desde source con Jupyter Book ya instalado")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("El puerto debe estar entre 0 y 65535")
    try:
        output = args.output.expanduser().absolute()
        report = extract_payload(Path(__file__).resolve().parent / PAYLOAD_NAME, output)
        report.update(validate_html(output / "site"))
        report["payload_sha256"] = digest(Path(__file__).resolve().parent / PAYLOAD_NAME)
        rebuilt = rebuild(output) if args.build else None
        if rebuilt:
            report["rebuilt_html"] = str(rebuilt.relative_to(output))
        write_entry(output, report, rebuilt)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print(f"Libro local: {(output / 'index.html').as_uri()}", flush=True)
        if args.serve:
            reject_symlinks(output)
            handler = functools.partial(ConfinedHTTPHandler, directory=str(output))
            with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
                print(f"Servidor local: http://127.0.0.1:{server.server_port}/index.html", flush=True)
                try:
                    server.serve_forever()
                except KeyboardInterrupt:
                    pass
        return 0
    except (OSError, ValueError, tarfile.TarError, subprocess.CalledProcessError) as error:
        print(f"D01: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
