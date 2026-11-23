"""Reconstruye el HTML del Jupyter Book en el entorno Python activo."""
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version
import sys

root = Path(__file__).resolve().parent
try:
    installed = version("jupyter-book")
except PackageNotFoundError:
    raise SystemExit("Falta Jupyter Book. Ejecuta: python -m pip install -r requirements.txt")
if installed != "1.0.4.post1":
    raise SystemExit(
        "Este proyecto usa Jupyter Book 1.0.4.post1. "
        "Ejecuta: python -m pip install -r requirements.txt"
    )

from jupyter_book.cli.main import main

main(["build", str(root), "--all", "--warningiserror"], prog_name="jupyter-book")
