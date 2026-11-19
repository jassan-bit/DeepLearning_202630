"""Separate A100 40 GB audit; operational launcher prepared with disclosed AI.

The scientific checkout and its cohort verification are unchanged. All models
are fitted again in a new output directory. The earlier 80 GB run is retained.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import urllib.request

SOURCE = "5a07d6157a9796ca54416dc5dea1e1527e731597"
MASTER = "20261119T132443278380Z"
GPU = "NVIDIA A100-SXM4-40GB"
ADMISSION = (
    "https://github.com/jassan-bit/DeepLearning_202630/tree/"
    "fe061b59678741d5de06eebb46526748af017fd4/"
    "runs/e3_closure_cpu/20261119T031104515520Z"
)
LAUNCHER_SHA256 = "d78e223e19a74ffd624b1a5baf7c0654c3e78132c0330d1a226c78633385c30f"
MISSING = "Requested audit master was not found; refuse to start an accidental new campaign"
SCOPE = "experiments/e3_evaluacion_ampliada_20261119/orchestration"
ALLOWED_READ = re.compile(
    r"/repos/jassan-bit/DeepLearning_202630/git/"
    r"(?:trees/[0-9a-f]{40}(?:\?recursive=1)?|"
    r"ref/heads/e3-extended-audit-[0-9]{8}T[0-9]{12}Z)"
)


def scope_note():
    return {
        "kind": "e3_separate_GPU_cohort_reexecution",
        "source_commit": SOURCE,
        "master_run_id": MASTER,
        "requested_GPU": GPU,
        "admission_reuse_url": ADMISSION,
        "previous_80GB_master": "20261119T031104515520Z",
        "previous_80GB_anchor_commit": "fa68f195d2e9f9bee23aab84239f2567a14b122d",
        "previous_80GB_completed_records_at_anchor": 252,
        "previous_results_known": True,
        "previous_weights_or_results_imported": False,
        "cross_GPU_result_pooling_allowed": False,
        "comparison": "all nineteen systems fitted/evaluated under this new frozen GPU profile",
        "training_budget_changed": False,
        "scientific_code_changed": False,
        "old_protocol_changed": False,
        "mode": "exploratory reexecution after a Colab hardware allocation change",
        "blind_or_independent_market_replication_claimed": False,
        "prospective_full_battery_power_claimed": False,
        "R1_accredited": False,
        "R3_accredited": False,
        "AI_assistance_declared": True,
    }


def _write_scope(base):
    output = base / ("e3-extended-audit-" + MASTER)
    path = output / "GPU_cohort_reexecution.json"
    if any(p.is_symlink() for p in (path, output, base, *base.parents)):
        raise ValueError("Symlink cohort evidence refused")
    raw = (json.dumps(scope_note(), ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    output.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != raw:
            raise ValueError("Existing cohort declaration differs; nothing was replaced")
        return
    with tempfile.NamedTemporaryFile(dir=output, prefix=".partial-cohort-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != raw:
                raise ValueError("Existing cohort declaration differs; nothing was replaced")
    finally:
        temporary.unlink(missing_ok=True)


def _missing_master(error, base):
    if type(error) is not ValueError or str(error) != MISSING:
        return False
    expected = (base / ("e3-extended-source-" + SOURCE) / SCOPE / "controller.py").resolve()
    trace = error.__traceback__
    while trace is not None:
        frame = trace.tb_frame
        values = frame.f_locals
        if (frame.f_code.co_name == "run"
                and Path(frame.f_code.co_filename).resolve() == expected
                and values.get("source_commit") == SOURCE
                and values.get("master_run_id") == MASTER
                and values.get("requested") == MASTER
                and values.get("start_new") is False
                and values.get("previous") is None
                and values.get("known_local") is False):
            return True
        trace = trace.tb_next
    return False


def _read_retry(error, base):
    if type(error).__name__ != "StorageError":
        return None
    message = str(error)
    if message.startswith("github_transport_network_read_failed "):
        return "corte de lectura de red"
    if message != "github_transport_http_error operation=git_trees_read status=404 attempts=1":
        return None
    expected = (base / ("e3-extended-source-" + SOURCE) / SCOPE / "github_transport.py").resolve()
    trace = error.__traceback__
    while trace is not None:
        frame = trace.tb_frame
        values = frame.f_locals
        route = values.get("path")
        if (frame.f_code.co_name == "_perform"
                and Path(frame.f_code.co_filename).resolve() == expected
                and values.get("method") == "GET"
                and type(values.get("status")) is int and values["status"] == 404
                and values.get("allow_missing") is False
                and isinstance(route, str) and ALLOWED_READ.fullmatch(route)):
            return route
        trace = trace.tb_next
    return None


def main(namespace, *, base=None):
    base = Path("/content") if base is None else Path(base)
    observed = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip()
    print("GPU asignada:", observed, flush=True)
    if observed != GPU:
        raise ValueError("Esta nueva campaña requiere " + GPU + "; no se ha iniciado ningún ajuste.")
    print("Campaña separada de 40 GB:", MASTER, flush=True)
    print("Se conservan los 252 resultados de 80 GB; no se incorporan a esta comparación.", flush=True)
    _write_scope(base)
    url = ("https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/"
           + SOURCE + "/" + SCOPE + "/launch_colab.py")
    with urllib.request.urlopen(url, timeout=60) as response:
        launcher = response.read()
    if hashlib.sha256(launcher).hexdigest() != LAUNCHER_SHA256:
        raise ValueError("Immutable scientific launcher hash differs")
    context = {
        "__name__": "__main__",
        "E3_EXTENDED_SOURCE_COMMIT": SOURCE,
        "E3_EXTENDED_MASTER_RUN": MASTER,
        "E3_EXTENDED_START_NEW": False,
        "E3_EXTENDED_ADMISSION_REUSE_URL": ADMISSION,
    }
    if namespace.get("_E3_GITHUB_TOKEN"):
        context["_E3_GITHUB_TOKEN"] = namespace["_E3_GITHUB_TOKEN"]
    creation_used = False
    for attempt in range(1, 5):
        try:
            try:
                exec(compile(launcher, "e3_a10040_original_launcher", "exec"), context)
            except ValueError as error:
                if creation_used or not _missing_master(error, base):
                    raise
                creation_used = True
                context["E3_EXTENDED_START_NEW"] = True
                print("Se crea el nuevo protocolo; los modelos comenzarán con pesos nuevos.", flush=True)
                exec(compile(launcher, "e3_a10040_original_launcher", "exec"), context)
            result = context.get("_E3_EXTENDED_AUDIT_RESULT")
            namespace["_E3_A10040_AUDIT_RESULT"] = result
            return result
        except Exception as error:
            context["E3_EXTENDED_START_NEW"] = False
            reason = _read_retry(error, base)
            if reason is None or attempt == 4:
                raise
            print("Lectura de GitHub interrumpida:", reason, flush=True)
            print(f"Reanudación de la misma campaña en 30 s ({attempt}/3).", flush=True)
            time.sleep(30)


if __name__ == "__main__":
    main(globals())
