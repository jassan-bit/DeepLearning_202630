"""Reproduce the frozen, posterior E3 statistics on CPU; never fit a model."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/e3-statistics-matplotlib")

HERE = Path(__file__).resolve().parent
PLAN_SHA256 = "41dbf0d10ba50b4a4f495c6525d8bb2b4e59328a84dce620c51724225f0eb2c7"
PLAN_COMMIT = "4a21b0329b99e864ec64fc2df49260aa874141bf"
CODE_FILES = ("analysis.py", "inputs.py", "mcs.py", "plots.py", "run.py", "plan.json", "requirements.txt")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()+b"\n"


def save(path, value):
    raw = canonical(value)
    if path.exists() and path.read_bytes() != raw:
        raise RuntimeError("Existing result differs; preserve and choose a new output directory: "+str(path))
    path.write_bytes(raw)


def csv_output(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    raw = stream.getvalue().encode()
    if path.exists() and path.read_bytes() != raw:
        raise RuntimeError("Existing table differs: "+str(path))
    path.write_bytes(raw)


def execute(data, output, download=False):
    import numpy as np
    import scipy
    import matplotlib
    import analysis
    import inputs
    import mcs
    if np.__version__ != "2.3.5" or scipy.__version__ != "1.17.0" or matplotlib.__version__ != "3.10.8":
        raise RuntimeError("Use the versions fixed in requirements.txt")
    if sha(HERE/"plan.json") != PLAN_SHA256:
        raise RuntimeError("The published analysis plan bytes changed")
    plan = json.loads((HERE/"plan.json").read_bytes())
    if download:
        inputs.download_inputs(data)
    panel = inputs.load_inputs(data)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"result.json").exists():
        raise RuntimeError("An existing result is preserved; verify it rather than rerunning into this directory")
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    def timeout(signum, frame):
        raise TimeoutError("Declared CPU analysis limit reached; no new model work is allowed")
    previous_handler = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(plan["runtime"]["scientific_worker_seconds_limit"])
    try:
        result = analysis.analyze(panel, plan)
        auxiliary = mcs.run_mcs(panel, plan)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
    elapsed = time.monotonic()-started
    provenance = {"version": "e3-posterior-statistics-execution-v1", "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(), "scientific_elapsed_seconds": elapsed,
        "plan_commit": PLAN_COMMIT, "plan_sha256": PLAN_SHA256,
        "code_sha256": {name: sha(HERE/name) for name in CODE_FILES},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__, "platform": platform.platform(), "device": "CPU",
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")},
        "source_inputs": plan["inputs"], "fits_performed": 0, "new_model_inference": 0,
        "old_E2_charged_seconds": 5130.970719594005, "old_E3_charged_seconds": 948.439625042,
        "old_counters_reset": False, "not_confirmatory_preregistration": True}
    result["execution"] = provenance
    save(output/"result.json", result)
    save(output/"mcs.json", auxiliary)
    save(output/"input_validation.json", panel["provenance"])
    save(output/"execution.json", provenance)
    csv_output(output/"metrics.csv", result["metrics"])
    csv_output(output/"primary_contrasts.csv", result["primary_contrasts"])
    csv_output(output/"effects.csv", result["effects"])
    csv_output(output/"verdicts.csv", result["per_rival_verdicts"])
    csv_output(output/"mae_secondary.csv", result["MAE_secondary_contrasts"])
    csv_output(output/"seed_subgroups.csv", result["seed_subgroups_descriptive"])
    csv_output(output/"sensitivities.csv", [row for setting in result["calendar_sensitivities"] for row in setting["contrasts"]])
    from plots import make_figures
    make_figures(result, output)
    manifest = {name.name: {"sha256": sha(name), "bytes": name.stat().st_size}
        for name in sorted(output.iterdir()) if name.is_file() and name.name != "manifest.json"}
    save(output/"manifest.json", {"files": manifest, "plan_commit": PLAN_COMMIT,
        "plan_sha256": PLAN_SHA256, "code_sha256": provenance["code_sha256"]})
    print(json.dumps({"status": "completed", "scientific_elapsed_seconds": elapsed,
        "primary_rejections": sum(row["reject"] for row in result["primary_contrasts"]),
        "family_size": 48, "all16_all3": result["all_16_all_three_reject"],
        "MCS_included": auxiliary["included_models"], "R3_verified": False}, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE/"data")
    parser.add_argument("--output", type=Path, default=HERE/"results")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    execute(args.data, args.output, args.download)


if __name__ == "__main__":
    main()
