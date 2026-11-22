"""Explicit, separately pinned operational repair for an existing S2 Colab.

Imports in the notebook kernel use only the standard library. The original S2
source is never edited; its CPU interpreter runs the independently pinned worker.
No credentials, new author declarations, fitting, final preparation or test IO.
AI assistance: implementation and software review; no human signatures supplied.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

VERSION = "coinm-header-operational-resume-v1"
SOURCE_COMMIT = "db74afd0763596f71bcbafb4b9150a44a2e83339"
SOURCE_MANIFEST_SHA256 = "2c7dc052d2a53f212688340b50b65bed5fa367ff16fb5ed0a2474d417696715b"
SOURCE_DATA_SHA256 = "1e64a500a62b3ec803471f87a2a81161a81d0e73d2c4cce7fd17d73b2b044e60"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def prepare_payload(namespace, bundle_dir, bundle_manifest_sha256):
    """Read-only validation; do not rewrite globals, metadata or prompt anyone."""
    needed = ("SOURCE", "SOURCE_COMMIT", "MANIFEST_SHA256", "SETTINGS", "PRIVATE",
              "definitions", "DEVELOPMENT_MAP", "bootstrap")
    require(all(key in namespace for key in needed), "Conserva la sesión y las etapas 1–2 del cuaderno S2.")
    require(namespace["SOURCE_COMMIT"] == SOURCE_COMMIT and
            namespace["MANIFEST_SHA256"] == SOURCE_MANIFEST_SHA256,
            "Esta reparación sólo corresponde a la fuente S2 fijada; no reemplazar sus pins.")
    settings = namespace["SETTINGS"]
    require(isinstance(settings, dict) and settings.get("source_commit") == SOURCE_COMMIT,
            "SETTINGS debe conservar el intérprete CPU y la fuente S2.")
    source, private = Path(namespace["SOURCE"]).resolve(), Path(namespace["PRIVATE"]).resolve()
    require(Path(settings.get("source_root", "")).resolve() == source,
            "SETTINGS apunta a otra fuente.")
    require(Path(settings.get("python", "")).is_file(), "El intérprete CPU original ya no está disponible.")
    require(not private.is_relative_to(source) and not source.is_relative_to(private),
            "Las carpetas privada y de fuente deben permanecer separadas.")
    bundle = Path(bundle_dir).resolve()
    require(not bundle.is_relative_to(source) and not source.is_relative_to(bundle),
            "Instalar el addon en una carpeta separada de SOURCE.")
    manifest_path = bundle / "ops" / "REPAIR_MANIFEST.json"
    require(sha(manifest_path) == bundle_manifest_sha256, "Manifest del addon diferente.")
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("version") == VERSION and manifest.get("parent_source_commit") == SOURCE_COMMIT and
            manifest.get("parent_manifest_sha256") == SOURCE_MANIFEST_SHA256 and
            manifest.get("parent_data_sha256") == SOURCE_DATA_SHA256,
            "Identidad o padre del addon diferente.")
    require(manifest.get("scope") == "resume_existing_F02_F03_development_only", "Alcance del addon diferente.")
    for relative, expected in manifest["files"].items():
        path = bundle / relative
        require(path.resolve().is_relative_to(bundle) and not path.is_symlink() and sha(path) == expected,
                "Archivo del addon diferente: " + relative)
    require(sha(Path(__file__)) == manifest["files"]["ops/resume_development.py"], "Launcher no fijado.")
    require(tuple(row["panel"] for row in namespace["definitions"]) == ("F00", "F01", "F02", "F03"),
            "Conservar exactamente los cuatro paneles originales.")
    for row in namespace["definitions"]:
        panel = row["panel"]
        for key, expected in (("folder", private / "inputs" / panel),
                              ("raw", private / "raw" / panel / "development-v1"),
                              ("panel_dir", private / "panels" / panel / "development-v1")):
            require(Path(row[key]).resolve() == expected, "Ruta original diferente: " + panel)
    require(Path(namespace["DEVELOPMENT_MAP"]).resolve() == private / "development-panels.json",
            "Mapa original diferente.")
    return dict(source=str(source), private=str(private), bundle=str(bundle),
                repair_manifest_sha256=bundle_manifest_sha256)


def resume(namespace, bundle_dir, bundle_manifest_sha256, *, verify_only=False):
    """Resume F02/F03 under already recorded author permission; never asks again.

    Explicitly calling this function activates only the previously authorized
    development acquisition. verify_only=True performs no acquisition/admission.
    It does not activate the notebook's planning flag or change its source pins.
    """
    payload = prepare_payload(namespace, bundle_dir, bundle_manifest_sha256)
    payload["operation"] = "verify-only" if verify_only else "resume-F02-F03"
    env = namespace["bootstrap"].cpu_env()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    worker = Path(bundle_dir) / "ops" / "resume_worker.py"
    command = [namespace["SETTINGS"]["python"], "-B", "-u", str(worker)]
    # No getpass/input, token-bearing subprocess arguments, pip or environment changes.
    process = subprocess.Popen(command, cwd=namespace["SOURCE"], env=env,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, text=True)
    process.stdin.write(json.dumps(payload))
    process.stdin.close()
    summary = None
    for line in process.stdout:
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            # The pinned worker contract is JSON summaries only; never echo a
            # surprise library traceback, private author fields or provider URL.
            continue
        print(json.dumps(event, ensure_ascii=False), flush=True)
        if event.get("status") == "four_development_panels_verified_with_header_overlay":
            summary = event
    process.stdout.close()
    # stderr is discarded rather than echoed; typed diagnostics and safe
    # reasons are emitted by the pinned worker and saved in its private receipts.
    require(process.wait() == 0 and summary is not None,
            "Reanudación detenida. Conserva esta sesión y los archivos; revisa el recibo operacional indicado. No se ha entrenado.")
    return summary
