"""Assemble three independently built Jupyter Books without changing their sources.

E1 retains its existing root URL. E2 and E3 have their own compiled trees below
entregable2/ and entregable3/. Only a new or empty output directory is accepted;
this command never replaces a previously populated directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import shutil
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
BOOKS = (
    ("entregable1", "book", "", "Informe y estudios del Entregable 1"),
    ("entregable2", "book_e2", "entregable2", "Dieciséis sistemas evaluados; reproducción en entorno limpio y tres ablaciones completadas; límites metodológicos documentados"),
    ("entregable3", "book_e3", "entregable3", "Libro inicial; guía específica pendiente"),
)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def validate_inputs(project_root, output_dir):
    root, output = Path(project_root).resolve(), Path(output_dir).resolve()
    if output == root or root.is_relative_to(output):
        raise ValueError("Output must not be the repository or one of its parent directories")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Use a new or empty output directory; existing contents are never replaced")
    inputs = []
    for book_id, source_name, prefix, status in BOOKS:
        source = root / source_name
        if output.is_relative_to(source.resolve()) or source.resolve().is_relative_to(output):
            raise ValueError("Output must not contain or be inside a source book")
        html = source / "_build" / "html"
        if not html.is_dir() or not html.resolve().is_relative_to(source.resolve()):
            raise ValueError("Build the independent source book first: " + source_name)
        if any(path.is_symlink() for path in html.rglob("*")):
            raise ValueError("Compiled books must contain regular local files, not symlinks")
        for name in ("intro.html", "index.html", "objects.inv", "searchindex.js"):
            if not (html / name).is_file():
                raise ValueError(f"Incomplete HTML output: {source_name}/{name}")
        config_path, toc_path = source / "_config.yml", source / "_toc.yml"
        if not config_path.is_file() or not toc_path.is_file():
            raise ValueError("Each book needs its own configuration and table of contents")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not isinstance(config.get("title"), str):
            raise ValueError("Each independent book must declare its own title")
        if not prefix and any((html / name).exists() for name in
                              ("entregable2", "entregable3", "entregables.html", "books_manifest.json")):
            raise ValueError("The E1 output collides with reserved multi-book paths")
        inputs.append({"id": book_id, "source": source_name, "prefix": prefix,
            "html": html, "title": config["title"], "status": status,
            "config_sha256": digest(config_path), "toc_sha256": digest(toc_path),
            "intro_sha256": digest(html / "intro.html")})
    return root, output, inputs


def portal_html(books):
    cards = []
    for book in books:
        link = (book["prefix"] + "/" if book["prefix"] else "") + "intro.html"
        cards.append(f'<li><h2><a href="{escape(link, quote=True)}">'
                     f'{escape(book["id"].replace("entregable", "Entregable "))}</a></h2>'
                     f'<p>{escape(book["title"])}</p><p>{escape(book["status"])}</p></li>')
    return ("<!doctype html><html lang=\"es\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            "<title>Deep Learning · Tres entregables</title>"
            "<style>body{font:18px/1.6 system-ui,sans-serif;max-width:900px;margin:3rem auto;"
            "padding:0 1.5rem;color:#1f2937}a{color:#075985}ul{list-style:none;padding:0}"
            "li{border:1px solid #cbd5e1;border-radius:10px;margin:1.2rem 0;padding:1rem 1.5rem}"
            "h2{margin:.2rem 0}p{margin:.4rem 0}</style></head><body>"
            "<h1>Deep Learning · Entregables 1, 2 y 3</h1>"
            "<p>Tres Jupyter Books con índices, configuraciones y direcciones propias, "
            "publicados desde un mismo repositorio.</p><ul>" + "".join(cards) +
            "</ul><p><a href=\"books_manifest.json\">Versión y procedencia de esta publicación</a>"
            "</p></body></html>\n")


def assemble(project_root=ROOT, output_dir=None, source_commit=None):
    output_dir = Path(output_dir) if output_dir is not None else Path(project_root) / "_build" / "site"
    root, output, inputs = validate_inputs(project_root, output_dir)
    if source_commit is not None and (len(source_commit) != 40
            or any(char not in "0123456789abcdef" for char in source_commit)):
        raise ValueError("A supplied source commit must be a full lowercase Git SHA")
    # Validate everything before creating output or touching an existing empty target.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="assemble-books-", dir=output.parent) as temporary:
        staging = Path(temporary) / "site"
        records = []
        for book in inputs:
            destination = staging / book["prefix"] if book["prefix"] else staging
            shutil.copytree(book["html"], destination)
            intro = destination / "intro.html"
            if digest(intro) != book["intro_sha256"]:
                raise ValueError("A copied book introduction differs from its compiled source")
            files = [path for path in book["html"].rglob("*") if path.is_file()]
            records.append({key: value for key, value in book.items() if key != "html"} |
                {"entrypoint": (book["prefix"] + "/" if book["prefix"] else "") + "intro.html",
                 "compiled_files": len(files),
                 "html_pages": sum(path.suffix == ".html" and "_static" not in
                                   path.relative_to(book["html"]).parts for path in files)})
        (staging / "entregables.html").write_text(portal_html(inputs), encoding="utf-8")
        manifest = {"schema_version": 1, "source_commit": source_commit,
            "source_commit_scope": "Supplied checkout commit; null for an unversioned local preview",
            "assembled_at_utc": datetime.now(timezone.utc).isoformat(),
            "books": records, "e1_root_url_preserved": True,
            "new_model_fits": 0, "new_model_forecasts": 0,
            "new_model_counts_scope": "Only fits and forecasts performed by this editorial assembly; excludes the previously executed E1, external ETTh1 and registered Bitcoin development P0/OOF experiments whose saved artifacts are published."}
        (staging / "books_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (staging / ".nojekyll").touch()
        if output.exists():
            output.rmdir()  # Only a still-empty directory can be removed; never a populated tree.
        staging.rename(output)
    return {"output_dir": str(output), "books": len(records),
            "entrypoints": [book["entrypoint"] for book in records],
            "e1_intro_sha256_preserved": records[0]["intro_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    result = assemble(args.project_root, args.output_dir, args.source_commit)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
