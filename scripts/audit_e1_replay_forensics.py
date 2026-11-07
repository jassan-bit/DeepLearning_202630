"""Forensic metadata and preregistered development-only E1 repeatability check.

inspect/prepare do not fit models. register requires an already published clean
source commit. Each fit invocation creates a fresh spawned pool in an empty
replicate directory; it never scores or reads the historical reserve.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
from pathlib import Path
import platform
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUNNER = "scripts/audit_e1_replay_forensics.py"
PROTOCOL = "experiments/e1_replay_audit_20261107/protocol.json"
OUTPUT = "results/e1_replay_forensics_20261107"
ORIGINAL = "results/gru_lstm_tcn_planned_20261105"
REPLAY = "results/e1_curve_reproduction_20261106"
PINNED = {"torch": "2.15.0+cpu", "numpy": "2.5.4", "pandas": "3.1.0",
          "scipy": "1.18.1", "statsmodels": "0.15.0", "matplotlib": "3.11.2"}
HEADER = ("epoch", "stage", "optimization_mse_scaled", "train_mse_scaled_eval",
          "validation_mse_scaled_eval", "best_epoch_so_far", "learning_rate")
INTEGER_COLUMNS = {"epoch", "best_epoch_so_far"}
RTOL, ATOL = 1e-6, 1e-8


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encoded(value))


def write_csv_new(path, rows):
    rows = list(rows)
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def feed(digest, value):
    """Length-prefix each element, avoiding ambiguous concatenation."""
    digest.update(struct.pack("<Q", len(value)))
    digest.update(value)


def csv_value_hash(path):
    """e1-csv-values-v1: typed cells in original column and row order.

    Header is exact; integers are signed int64 LE, finite numeric values are
    IEEE754 binary64 LE after Python's correctly rounded decimal parse. Only
    empty validation cells mean missing. UTF8 stage strings are length-prefixed.
    This ignores decimal spelling, quoting and line endings, not numeric bits.
    """
    digest = hashlib.sha256(b"e1-csv-values-v1\0")
    count = 0
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != HEADER:
            raise ValueError("Unexpected curve header: " + str(path))
        for name in HEADER:
            feed(digest, name.encode())
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed curve row")
            count += 1
            for name in HEADER:
                cell = row[name]
                if name == "stage":
                    if cell not in {"inner", "refit"}:
                        raise ValueError("Unknown fitting stage")
                    feed(digest, b"s" + cell.encode())
                elif name in INTEGER_COLUMNS:
                    feed(digest, b"i" + struct.pack("<q", int(cell)))
                elif cell == "" and name == "validation_mse_scaled_eval":
                    feed(digest, b"missing")
                else:
                    number = float(cell)
                    if not math.isfinite(number):
                        raise ValueError("Nonfinite curve value")
                    feed(digest, b"f" + struct.pack("<d", number))
    feed(digest, struct.pack("<Q", count))
    return {"schema": "e1-csv-values-v1", "sha256": digest.hexdigest(), "rows": count}


def tensor_value_hash(path):
    """e1-state-dict-v1 hashes names/dtypes/shapes and contiguous tensor bits.

    Checkpoint metadata other than state_dict has its own canonical JSON hash.
    Tensor storage identifiers, ZIP metadata, strides and device are excluded.
    Only dense CPU-readable tensors and a little-endian host are admitted.
    """
    import torch
    if sys.byteorder != "little":
        raise ValueError("v1 tensor hash requires a little-endian host")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    digest = hashlib.sha256(b"e1-state-dict-v1\0")
    state = checkpoint["state_dict"]
    if not isinstance(state, dict) or not state:
        raise ValueError("Nonempty state_dict required")
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided or tensor.is_quantized:
            raise ValueError("Only dense, unquantized tensors are supported")
        values = tensor.detach().cpu().resolve_conj().resolve_neg().contiguous()
        feed(digest, name.encode("utf-8"))
        feed(digest, str(values.dtype).encode())
        feed(digest, json.dumps(list(values.shape), separators=(",", ":")).encode())
        feed(digest, values.reshape(-1).view(torch.uint8).numpy().tobytes())
    metadata = {key: value for key, value in checkpoint.items() if key != "state_dict"}
    meta = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {"schema": "e1-state-dict-v1", "sha256": digest.hexdigest(),
            "tensors": len(state), "metadata_sha256": hashlib.sha256(meta).hexdigest()}


def numerical_comparison(left, right, artifact):
    """Symmetric, fixed a-priori tolerance; no fitted or data-dependent limit."""
    import numpy as np
    pairs = []
    structure_equal = True
    if artifact == "model.pt":
        import torch
        checkpoints = [torch.load(path, map_location="cpu", weights_only=True) for path in (left, right)]
        metadata = [{key: value for key, value in item.items() if key != "state_dict"} for item in checkpoints]
        states = [item["state_dict"] for item in checkpoints]
        structure_equal = metadata[0] == metadata[1] and set(states[0]) == set(states[1])
        if structure_equal:
            for name in sorted(states[0]):
                a, b = states[0][name], states[1][name]
                if a.shape != b.shape or a.dtype != b.dtype:
                    structure_equal = False
                    break
                pairs.append((a.detach().cpu().to(torch.float64).numpy().reshape(-1),
                              b.detach().cpu().to(torch.float64).numpy().reshape(-1)))
    else:
        tables = []
        for path in (left, right):
            csv_value_hash(path)
            with Path(path).open(newline="") as stream:
                tables.append(list(csv.DictReader(stream)))
        structure_equal = len(tables[0]) == len(tables[1])
        if structure_equal:
            left_values, right_values = [], []
            for a, b in zip(*tables):
                if a["stage"] != b["stage"] or any(int(a[key]) != int(b[key]) for key in INTEGER_COLUMNS):
                    structure_equal = False
                for key in HEADER:
                    if key == "stage" or key in INTEGER_COLUMNS:
                        continue
                    if a[key] == "" or b[key] == "":
                        if a[key] != b[key]:
                            structure_equal = False
                    else:
                        left_values.append(float(a[key]))
                        right_values.append(float(b[key]))
            pairs.append((np.asarray(left_values), np.asarray(right_values)))
    within = structure_equal
    maximum_absolute, maximum_relative = 0.0, 0.0
    for a, b in pairs:
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            within = False
            continue
        difference = np.abs(a - b)
        magnitude = np.maximum(np.abs(a), np.abs(b))
        within = within and bool(np.all(difference <= ATOL + RTOL * magnitude))
        maximum_absolute = max(maximum_absolute, float(difference.max(initial=0.0)))
        maximum_relative = max(maximum_relative,
                               float((difference / np.maximum(magnitude, 1e-300)).max(initial=0.0)))
    return {"structure_equal": structure_equal, "within_tolerance": within,
            "max_abs_difference": maximum_absolute if structure_equal else None,
            "max_symmetric_relative_difference": maximum_relative if structure_equal else None}


def discordances():
    historical = read(ROOT / ORIGINAL / "final_freeze_provenance.json")
    old = {(job["model"], int(job["trajectory_seed"])): job for job in historical["jobs"]}
    with (ROOT / REPLAY / "fit_summary.csv").open(newline="") as stream:
        current = list(csv.DictReader(stream))
    if len(old) != 1395 or len(current) != 1395:
        raise ValueError("Complete historical/replay epoch panels are required")
    rows = []
    seen = set()
    for job in current:
        key = job["model"], int(job["seed"])
        if key in seen:
            raise ValueError("Duplicate replay trajectory")
        seen.add(key)
        before, after = int(old[key]["selected_epoch"]), int(job["selected_epoch"])
        if before != after:
            rows.append({"model": key[0], "seed": key[1], "original_selected_epoch": before,
                "replay_selected_epoch": after, "delta": after - before,
                "absolute_delta": abs(after - before)})
    return sorted(rows, key=lambda row: (-row["absolute_delta"], row["model"], row["seed"]))


def panel():
    """Three fixed family controls plus three largest historical epoch changes."""
    return ([{"model": family, "seed": 8101, "reason": "fixed first registered seed"}
             for family in ("GRU", "LSTM", "TCN")]
            + [{"model": row["model"], "seed": row["seed"],
                "reason": "top three absolute epoch changes; ties model then seed"}
               for row in discordances()[:3]])


def checked_private(path):
    path = Path(path).resolve()
    if path.is_relative_to(ROOT):
        raise ValueError("Private artifacts must remain outside the source checkout")
    for parent in (path, *path.parents):
        if (parent / ".git").exists() and parent not in {Path("/workspace"), Path("/tmp")}:
            raise ValueError("Private artifacts cannot be written into another Git checkout")
    return path


@lru_cache(maxsize=None)
def binary_sha(path):
    return sha(path)


def capture_runtime():
    import torch
    cpu = Path("/proc/cpuinfo").read_text()
    cpu_names = sorted({line.split(":", 1)[1].strip() for line in cpu.splitlines()
                        if line.startswith("model name")})
    return {"python": platform.python_version(), "platform": platform.platform(),
        "versions": {name: importlib.metadata.version(name) for name in PINNED},
        "torch_config": torch.__config__.show(), "cpu_models": cpu_names,
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
        "torch_extension_sha256": sha(torch._C.__file__),
        "torch_cpu_library_sha256": binary_sha(str(Path(torch.__file__).parent / "lib" / "libtorch_cpu.so")),
        "environment": {name: os.environ.get(name) for name in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONHASHSEED",
             "MKL_CBWR", "DNNL_MAX_CPU_ISA", "ONEDNN_MAX_CPU_ISA")},
        "actual_torch_threads": torch.get_num_threads(),
        "actual_torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "default_dtype": str(torch.get_default_dtype())}


def verify_pins():
    runtime = capture_runtime()
    if runtime["python"] != "3.12.14" or runtime["versions"] != PINNED:
        raise ValueError("The exact preregistered historical package versions are required")
    return runtime


def verify_prior_artifact(path, family, seed, name):
    """Pin existing private payloads to the already published replay hashes."""
    with (ROOT / REPLAY / "artifact_hash_comparison.csv").open(newline="") as stream:
        matches = [row for row in csv.DictReader(stream)
                   if row["model"] == family and int(row["seed"]) == seed and row["artifact"] == name]
    if len(matches) != 1 or sha(path) != matches[0]["reproduced_sha256"]:
        raise ValueError("Existing replay artifact differs from its published hash")


def inspect(private, output):
    """Read existing fits, write only hashes/configuration/epoch diagnostics."""
    private, output = checked_private(private), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Forensic outputs must have a new empty destination")
    old = read(ROOT / ORIGINAL / "final_freeze_provenance.json")
    with (ROOT / REPLAY / "artifact_hash_comparison.csv").open(newline="") as stream:
        replay_hashes = {(row["model"], int(row["seed"]), row["artifact"]): row["reproduced_sha256"]
                         for row in csv.DictReader(stream)}
    if len(replay_hashes) != 8370:
        raise ValueError("A complete published prior-replay artifact map is required")
    comparisons, canonical = [], []
    for family in ("GRU", "LSTM", "TCN"):
        for seed in range(8101, 8566):
            for name in ("history_inner.csv", "history_refit.csv", "model.pt",
                         "configuration.json", "preprocessing.json", "split.json"):
                relative = f"final/{family}_s{seed}/{name}"
                path = private / relative
                value = sha(path)
                if value != replay_hashes[family, seed, name]:
                    raise ValueError("Existing replay artifact differs from its published hash: " + relative)
                comparisons.append({"model": family, "seed": seed, "artifact": name,
                    "original_sha256": old["artifact_sha256"][relative], "replay_sha256": value,
                    "byte_equal": value == old["artifact_sha256"][relative]})
                if name.endswith(".csv") or name == "model.pt":
                    result = csv_value_hash(path) if name.endswith(".csv") else tensor_value_hash(path)
                    canonical.append({"model": family, "seed": seed, "artifact": name,
                        "schema": result["schema"], "canonical_sha256": result["sha256"],
                        "row_or_tensor_count": result.get("rows", result.get("tensors")),
                        "checkpoint_metadata_sha256": result.get("metadata_sha256", "")})
    mismatches = discordances()
    counts = {name: {"compared": sum(row["artifact"] == name for row in comparisons),
                    "matched": sum(row["artifact"] == name and row["byte_equal"] for row in comparisons)}
              for name in sorted({row["artifact"] for row in comparisons})}
    summary = {"schema_version": 1, "created_at_utc": stamp(), "new_fits": 0,
        "reserve_price_payloads_read": 0, "forecasts_generated": 0,
        "original_private_payloads_available": False, "exact_cause_identified": False,
        "original_and_replay_start_method": "spawn", "original_workers": 3, "replay_workers": 4,
        "history_csv_contains_timing_columns": False,
        "selected_epochs_compared": 1395, "selected_epoch_mismatches": len(mismatches),
        "selected_epoch_mismatch_families": sorted({row["model"] for row in mismatches}),
        "artifact_byte_comparisons": counts,
        "canonical_historical_comparison_possible": False,
        "reason": "Historical SHA256 digests cannot reconstruct values or canonical tensor hashes",
        "unrecorded_original_details": ["CPU model and ISA dispatch", "Torch build/extension digest",
            "effective BLAS environment", "actual worker runtime snapshots", "initialization/gradient tensors"],
        "observed_execution_differences_not_established_causes": ["three vs four workers",
            "original respects preexisting OMP/MKL/OPENBLAS environment; replay forces those values to one"],
        "conclusion": "At least the 29 epoch changes are numerical/algorithmic divergence; its root cause is unresolved. Other byte mismatches alone do not identify value differences.",
        "diagnostic_panel": panel(), "runtime_observed_now_not_historical": verify_pins()}
    write_csv_new(output / "artifact_byte_comparison.csv", comparisons)
    write_csv_new(output / "replay_canonical_hashes.csv", canonical)
    write_csv_new(output / "epoch_discordances.csv", mismatches)
    write_new(output / "summary.json", summary)
    write_new(output / "manifest.json", {"new_fits": 0,
        "sources_sha256": {RUNNER: sha(ROOT / RUNNER),
            ORIGINAL + "/final_freeze_provenance.json": sha(ROOT / ORIGINAL / "final_freeze_provenance.json"),
            REPLAY + "/fit_summary.csv": sha(ROOT / REPLAY / "fit_summary.csv")},
        "outputs_sha256": {path.name: sha(path) for path in output.iterdir() if path.is_file()}})
    return {"new_fits": 0, "output": str(output), "epoch_mismatches": len(mismatches)}


def prepare():
    historical = read(ROOT / ORIGINAL / "registration.json")
    sources = dict(historical["sources_sha256"])
    for name, expected in sources.items():
        if sha(ROOT / name) != expected:
            raise ValueError("Historical scientific source changed: " + name)
    names = (RUNNER, "tests/test_e1_replay_forensics.py",
             "experiments/e1_replay_audit_20261107/README.md", "requirements-e1-curves.lock.txt", ORIGINAL + "/selected_recipes.json",
             ORIGINAL + "/final_freeze_provenance.json", REPLAY + "/fit_summary.csv",
             REPLAY + "/artifact_hash_comparison.csv")
    sources.update({name: sha(ROOT / name) for name in names})
    plan = {"schema_version": 1, "prepared_at_utc": stamp(), "new_fits_at_preparation": 0,
        "purpose": "Current independent-process numerical repeatability of a fixed diagnostic subpanel",
        "chronology": "Post-original and post-full-replay diagnostic; publish protocol and runner before these new fits",
        "claim_limit": "Six diagnostic cells only; neither full R465 repeatability nor exact original recovery",
        "original_artifacts_unchanged": True, "baseline_unchanged": True,
        "panel": panel(), "panel_rule": "GRU/LSTM/TCN at first registered seed8101 plus three greatest absolute historical→replay selected-epoch changes; ties by model then seed. Chosen after old replay, before any new repeatability fits; all results retained.",
        "replicates": ["a", "b"], "planned_trajectories": 12, "planned_fit_stages": 24,
        "history_days": 7, "workers_per_process_pool": 3, "start_method": "spawn",
        "replicate_processes": "Separate CLI invocations, fresh pools, independent empty private directories; run a then b",
        "job_order": "Published panel list in order; each job resets Python/NumPy/Torch seeds and independent shuffle generator",
        "environment": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONHASHSEED": "0"},
        "worker_settings": {"torch_threads": 1, "torch_interop_threads": 1,
            "deterministic_algorithms": True, "mkldnn_enabled": True,
            "float32_matmul_precision": "highest", "default_dtype": "torch.float32"},
        "runtime_expected": {"python": "3.12.14", "versions": PINNED},
        "data_scope": {"calendar": ["2023-01-01", "2024-12-31"],
            "train_samples": 709, "inner_train_samples": 619, "inner_validation_samples": 83,
            "reserve_price_payloads_allowed": 0, "reserve_scoring_allowed": False,
            "new_forecasts_allowed": False},
        "settings": historical["settings"],
        "recipes": read(ROOT / ORIGINAL / "selected_recipes.json")["recipes"],
        "primary_acceptance": "All six cells have identical selected_epoch, e1-csv-values-v1 hashes of both histories, e1-state-dict-v1 tensor hashes and checkpoint metadata hashes between a and b. Exact equality; no fitted tolerance.",
        "secondary_numerical_equivalence": {"rtol": RTOL, "atol": ATOL,
            "elementwise_rule": "abs(a-b) <= atol + rtol*max(abs(a),abs(b)); symmetric and fixed before new fits",
            "values": "Every numeric loss/LR CSV cell and every tensor element; no observations or predictions",
            "structure_rule": "Equal selected_epoch, curve row counts, epoch/stage/best_epoch, missingness, tensor names/dtypes/shapes and checkpoint metadata; otherwise no whole-fit equivalence",
            "interpretation": "Separate secondary numerical classification, never relabel a strict canonical failure as an exact pass"},
        "secondary_comparisons": "Report all a/b vs existing full replay raw/canonical hashes and historical raw hashes, without optimizing settings to improve agreement",
        "failure_policy": "Stop, retain every partial output; no seed substitution, tuning, retry, cache reuse or extension of scope. Any further fits require a new published protocol.",
        "public_outputs": ["hashes", "runtime metadata", "selected epochs", "pass/fail per cell", "aggregate counts"],
        "excluded_public_outputs": ["weights", "learned scaler values", "observations", "daily predictions"],
        "sources_sha256": sources, "runtime_observed_at_preparation": verify_pins()}
    write_new(ROOT / PROTOCOL, plan)
    return {"protocol": PROTOCOL, "new_fits": 0, "planned_fit_stages": 24}


def protocol():
    plan = read(ROOT / PROTOCOL)
    if plan["panel"] != panel() or plan["replicates"] != ["a", "b"] or plan["planned_fit_stages"] != 24:
        raise ValueError("The published fixed diagnostic panel and two repeats are required")
    for name, expected in plan["sources_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Protocol scientific source or input changed: " + name)
    verify_pins()
    return plan


def register(private, published_commit):
    plan = protocol()
    revision = git("rev-parse", "HEAD")
    if published_commit != revision or git("status", "--porcelain"):
        raise ValueError("Register from the clean, already-published exact source commit")
    for name in (PROTOCOL, *plan["sources_sha256"]):
        committed = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=ROOT)
        if committed != (ROOT / name).read_bytes():
            raise ValueError("Protocol/source must be committed unchanged before registration")
    private = checked_private(private)
    if private.exists() and any(private.iterdir()):
        raise ValueError("Registration requires a new empty private root")
    record = {"registered_at_utc": stamp(), "source_commit": revision,
        "publication_url": f"https://github.com/jassan-bit/DeepLearning_202630/commit/{revision}",
        "publication_verification": "Caller must independently verify this remote commit exists before invoking register; this CLI argument is not independent publication evidence",
        "protocol_sha256": sha(ROOT / PROTOCOL), "protocol": plan,
        "runtime": capture_runtime(), "new_fits_before_registration": 0,
        "reserve_price_payloads_read": 0, "forecasts_generated": 0}
    record["signature"] = hashlib.sha256(encoded(record)).hexdigest()
    write_new(private / "registration.json", record)
    return {"registered_at_utc": record["registered_at_utc"], "source_commit": revision, "new_fits": 0}


def registered(private):
    private = checked_private(private)
    record = read(private / "registration.json")
    unsigned = {key: value for key, value in record.items() if key != "signature"}
    if hashlib.sha256(encoded(unsigned)).hexdigest() != record["signature"]:
        raise ValueError("Private registration changed")
    if record["protocol"] != protocol() or record["protocol_sha256"] != sha(ROOT / PROTOCOL):
        raise ValueError("Protocol differs from private registration")
    if record["source_commit"] != git("rev-parse", "HEAD") or git("status", "--porcelain"):
        raise ValueError("Repeat from the clean registered immutable source commit")
    return private, record


def worker_init(root, output, record):
    import torch
    import run_gru_lstm_tcn_planned as original
    torch.backends.mkldnn.enabled = True
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    original.worker_init(root, output, record)
    observed = capture_runtime()
    if (observed["actual_torch_threads"] != 1 or observed["actual_torch_interop_threads"] != 1
            or not observed["deterministic_algorithms"] or mp.get_start_method() != "spawn"):
        raise ValueError("Worker runtime differs from the registered numerical controls")
    preparation = record["protocol"]["runtime_observed_at_preparation"]
    for key in ("versions", "cpu_models", "cpu_capability", "torch_config", "torch_extension_sha256", "torch_cpu_library_sha256"):
        if observed[key] != preparation[key]:
            raise ValueError("Worker runtime changed since preparation: " + key)
    for key, expected in record["protocol"]["environment"].items():
        if observed["environment"][key] != expected:
            raise ValueError("Worker environment differs from protocol: " + key)
    for key in ("MKL_CBWR", "DNNL_MAX_CPU_ISA", "ONEDNN_MAX_CPU_ISA"):
        if observed["environment"][key] != preparation["environment"][key]:
            raise ValueError("Worker ISA/BLAS environment differs from preparation: " + key)
    write_new(Path(output) / f"worker_runtime_{os.getpid()}.json",
              {"pid": os.getpid(), "start_method": mp.get_start_method(), "runtime": observed})


def fit(private, replicate):
    private, record = registered(private)
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("Launch each fresh Python CLI with PYTHONHASHSEED=0")
    output = private / replicate
    if output.exists():
        raise ValueError("Each replicate needs a fresh directory; cached/partial fits are never reused")
    if replicate == "b" and not (private / "a" / "finished.json").is_file():
        raise ValueError("Complete independent process a before launching process b")
    import run_gru_lstm_tcn_planned as original
    output.mkdir()
    started = time.perf_counter()
    write_new(output / "started.json", {"at_utc": stamp(), "parent_pid": os.getpid(),
        "replicate": replicate, "registration_signature": record["signature"], "new_jobs_planned": 6})
    registration = {**record, "registration_signature": record["signature"], "settings": record["protocol"]["settings"]}
    jobs = [{**record["protocol"]["recipes"][cell["model"]], "phase": "final",
             "trajectory_seed": cell["seed"], "seed": cell["seed"]} for cell in record["protocol"]["panel"]]
    with ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context("spawn"),
            initializer=worker_init, initargs=(str(ROOT), str(output), registration)) as pool:
        completed = list(pool.map(original.fixed_job, jobs))
    write_new(output / "finished.json", {"at_utc": stamp(), "parent_pid": os.getpid(),
        "replicate": replicate, "models": len(completed), "new_fit_stages": 2 * len(completed),
        "wall_seconds": time.perf_counter() - started, "reserve_price_payloads_read": 0,
        "forecasts_generated": 0})
    return {"replicate": replicate, "models": len(completed), "new_fit_stages": 2 * len(completed)}


def compare(private, prior, output):
    private, record = registered(private)
    prior, output = checked_private(prior), Path(output)
    if read(private / "a" / "started.json")["parent_pid"] == read(private / "b" / "started.json")["parent_pid"]:
        raise ValueError("Independent fresh CLI processes are required")
    for replicate in ("a", "b"):
        finish = read(private / replicate / "finished.json")
        if finish["models"] != 6 or finish["new_fit_stages"] != 12:
            raise ValueError("Incomplete diagnostic replicate")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use a new empty public comparison directory")
    rows = []
    historical = read(ROOT / ORIGINAL / "final_freeze_provenance.json")
    historical_epochs = {(job["model"], job["trajectory_seed"]): int(job["selected_epoch"])
                         for job in historical["jobs"]}
    for cell in record["protocol"]["panel"]:
        family, seed = cell["model"], cell["seed"]
        relative = f"final/{family}_s{seed}"
        epochs = {rep: int(read(private / rep / relative / "summary.json")["selected_epoch"])
                  for rep in ("a", "b")}
        epochs["prior"] = int(read(prior / relative / "summary.json")["selected_epoch"])
        for artifact in ("history_inner.csv", "history_refit.csv", "model.pt"):
            files = {rep: private / rep / relative / artifact for rep in ("a", "b")}
            files["prior"] = prior / relative / artifact
            verify_prior_artifact(files["prior"], family, seed, artifact)
            hashing = tensor_value_hash if artifact == "model.pt" else csv_value_hash
            hashes = {rep: hashing(path) for rep, path in files.items()}
            numeric = numerical_comparison(files["a"], files["b"], artifact)
            rows.append({"model": family, "seed": seed, "artifact": artifact,
                "original_selected_epoch": historical_epochs[family, seed], "prior_selected_epoch": epochs["prior"],
                "selected_epoch_a": epochs["a"], "selected_epoch_b": epochs["b"],
                "a_canonical_sha256": hashes["a"]["sha256"], "b_canonical_sha256": hashes["b"]["sha256"],
                "prior_canonical_sha256": hashes["prior"]["sha256"],
                "a_b_canonical_equal": hashes["a"] == hashes["b"],
                "a_prior_canonical_equal": hashes["a"] == hashes["prior"],
                "b_prior_canonical_equal": hashes["b"] == hashes["prior"],
                "a_b_raw_equal": sha(files["a"]) == sha(files["b"]),
                "a_b_structure_equal": numeric["structure_equal"],
                "a_b_within_tolerance": numeric["within_tolerance"],
                "a_b_max_abs_difference": numeric["max_abs_difference"],
                "a_b_max_symmetric_relative_difference": numeric["max_symmetric_relative_difference"],
                "a_historical_raw_equal": sha(files["a"]) == historical["artifact_sha256"][relative + "/" + artifact],
                "b_historical_raw_equal": sha(files["b"]) == historical["artifact_sha256"][relative + "/" + artifact]})
    accepted = all(row["a_b_canonical_equal"] and row["selected_epoch_a"] == row["selected_epoch_b"] for row in rows)
    equivalent_cells = sum(all(row["a_b_within_tolerance"] and row["selected_epoch_a"] == row["selected_epoch_b"]
                              for row in rows if (row["model"], row["seed"]) == (cell["model"], cell["seed"]))
                           for cell in record["protocol"]["panel"])
    write_csv_new(output / "comparison.csv", rows)
    summary = {"schema_version": 1, "created_at_utc": stamp(), "source_commit": record["source_commit"],
        "protocol_sha256": record["protocol_sha256"], "primary_acceptance_passed": accepted,
        "secondary_numerical_equivalent_cells": equivalent_cells, "fixed_rtol": RTOL, "fixed_atol": ATOL,
        "cells": 6, "independent_replicates": 2, "new_fit_stages": 24,
        "artifacts_compared": len(rows), "canonical_matches_a_b": sum(row["a_b_canonical_equal"] for row in rows),
        "canonical_matches_a_prior": sum(row["a_prior_canonical_equal"] for row in rows),
        "canonical_matches_b_prior": sum(row["b_prior_canonical_equal"] for row in rows),
        "scope": record["protocol"]["claim_limit"], "reserve_price_payloads_read": 0,
        "forecasts_generated": 0, "original_private_payloads_recovered": False}
    write_new(output / "summary.json", summary)
    runtimes = {rep: [read(path) for path in sorted((private / rep).glob("worker_runtime_*.json"))]
                for rep in ("a", "b")}
    write_new(output / "runtime.json", runtimes)
    write_new(output / "execution.json", {rep: {name: read(private / rep / name)
        for name in ("started.json", "finished.json")} for rep in ("a", "b")})
    write_new(output / "manifest.json", {"outputs_sha256": {
        path.name: sha(path) for path in output.iterdir() if path.is_file()}})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("inspect", "prepare", "register", "fit", "compare"))
    parser.add_argument("--private-dir", type=Path)
    parser.add_argument("--prior-private-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--published-commit")
    parser.add_argument("--replicate", choices=("a", "b"))
    args = parser.parse_args()
    if args.phase == "prepare":
        result = prepare()
    elif args.phase == "inspect":
        if not args.private_dir or not args.output_dir:
            parser.error("inspect requires --private-dir and --output-dir")
        result = inspect(args.private_dir, args.output_dir)
    elif args.phase == "register":
        if not args.private_dir or not args.published_commit:
            parser.error("register requires --private-dir and --published-commit")
        result = register(args.private_dir, args.published_commit)
    elif args.phase == "fit":
        if not args.private_dir or not args.replicate:
            parser.error("fit requires --private-dir and --replicate")
        result = fit(args.private_dir, args.replicate)
    else:
        if not args.private_dir or not args.prior_private_dir or not args.output_dir:
            parser.error("compare requires --private-dir, --prior-private-dir and --output-dir")
        result = compare(args.private_dir, args.prior_private_dir, args.output_dir)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
