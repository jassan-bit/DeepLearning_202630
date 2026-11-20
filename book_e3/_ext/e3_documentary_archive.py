"""Prepare the hash-verified documentary download during the book build.

AI assistance declared. No network, model, data fitting, or protocol changes.
"""
from pathlib import Path
import json
import runpy
from sphinx.util import logging


def prepare_download(app):
    root = Path(app.srcdir).resolve().parent
    script = root / "experiments/e3_evidence_close_20261120/release/assemble_archive.py"
    receipt = runpy.run_path(str(script))["assemble"](root)
    logging.getLogger(__name__).info("E3 documentary download: %s", json.dumps(receipt, sort_keys=True))


def setup(app):
    app.connect("builder-inited", prepare_download)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
