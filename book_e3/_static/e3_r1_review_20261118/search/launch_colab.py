"""Single-cell Colab launcher: stdlib CPU, native searches, immutable GitHub save.

The caller must set R1_SOURCE_COMMIT to the published full commit before execution.
AI assistance declared, 2026-11-18.
"""
from datetime import datetime, timezone
from getpass import getpass
from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys

SOURCE = globals().get("R1_SOURCE_COMMIT", globals().get("E3_R1_COMMIT", ""))
if len(SOURCE) != 40 or any(c not in "0123456789abcdef" for c in SOURCE):
    raise ValueError("Set R1_SOURCE_COMMIT to the published full commit")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def github_token():
    value = os.environ.get("GITHUB_TOKEN", "").strip()
    if not value:
        try:
            from google.colab import userdata
            value = (userdata.get("GITHUB_TOKEN") or "").strip()
        except Exception:
            value = ""
    value = value or globals().get("_E3_R1_GITHUB_TOKEN", "").strip()
    return value or getpass("GITHUB_TOKEN (Contents: Read and write): ").strip()


_E3_R1_GITHUB_TOKEN = github_token()
context = globals().get("_E3_R1_SEARCH_CONTEXT")
if not (context and context["source"] == SOURCE and (context["output"] / "search_receipt.json").is_file()):
    base = Path("/content") if Path("/content").is_dir() else Path.cwd()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    checkout = base / ("e3-r1-source-" + run_id)
    subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout",
                    "https://github.com/jassan-bit/DeepLearning_202630.git", str(checkout)], check=True)

    def git(*args):
        return subprocess.check_output(["git", "-C", str(checkout), *args], text=True).strip()

    git("sparse-checkout", "init", "--cone")
    git("fetch", "--depth", "1", "origin", SOURCE)
    git("checkout", "--detach", SOURCE)
    git("sparse-checkout", "set", "experiments/e3_r1_review_20261118/search",
        "experiments/e3_remediacion_20261118/evaluation")
    if git("rev-parse", "HEAD") != SOURCE:
        raise RuntimeError("source_commit_identity_mismatch")
    api = module("e3_r1_api_" + run_id, checkout / "experiments/e3_remediacion_20261118/evaluation/colab_github.py")
    api.validate_credentials(_E3_R1_GITHUB_TOKEN)
    here = checkout / "experiments/e3_r1_review_20261118/search"
    collector = module("e3_r1_collect_" + run_id, here / "collect.py")
    storage = module("e3_r1_store_" + run_id, here / "github_store.py")
    output = base / ("e3-r1-search-" + run_id)
    context = {"source": SOURCE, "run_id": run_id, "api": api, "storage": storage, "output": output}
    collector.run(output, SOURCE)
    _E3_R1_SEARCH_CONTEXT = context
else:
    print("Reutilizando consultas locales; se reintenta únicamente el guardado.")
try:
    R1_SEARCH_PUBLICATION = context["storage"].save(context["output"], _E3_R1_GITHUB_TOKEN,
                                                  SOURCE, context["run_id"], context["api"])
except Exception:
    print("Conserva la sesión. Reejecutar esta celda reintenta guardar el mismo resultado.")
    raise
print("Búsqueda guardada en GitHub:", R1_SEARCH_PUBLICATION["run_url"])
receipt = json.loads((context["output"] / "search_receipt.json").read_text())
print("Bases con seis respuestas válidas:", receipt["effective_bases_all_six_queries"])
print("Cribado pendiente. Revisión humana pendiente. R1 no acreditada por este recolector.")
