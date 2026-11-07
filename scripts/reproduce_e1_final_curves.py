"""New development-only replay of E1's frozen final recipes.

This is not recovery of the original private run and never scores the reserve.
Publish ``protocol.json`` and this runner before using register, then fit.
All weights, scalers and observations remain in the external private directory.
"""
from __future__ import annotations

import os
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_variable] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "experiments/e1_curve_reproduction_20261106/protocol.json"
RUNNER = "scripts/reproduce_e1_final_curves.py"
LOCK = "requirements-e1-curves.lock.txt"
ORIGINAL = "results/gru_lstm_tcn_planned_20261105"
MODELS = ("GRU", "LSTM", "TCN")
SEEDS = tuple(range(8101, 8566))
SCIENCE_FILES = ("history_inner.csv", "history_refit.csv", "configuration.json",
                 "preprocessing.json", "split.json", "model.pt")
DIRECT = {"torch": "2.15.0+cpu", "numpy": "2.5.4", "pandas": "3.1.0",
          "scipy": "1.18.1", "statsmodels": "0.15.0", "matplotlib": "3.11.2"}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encode(value))


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def runtime():
    import torch
    return {"python": platform.python_version(), "platform": platform.platform(),
        "versions": {name: importlib.metadata.version(name) for name in DIRECT},
        "device": "cpu", "torch_mkldnn_enabled": torch.backends.mkldnn.enabled,
        "cpu_affinity_count": len(os.sched_getaffinity(0)),
        "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
        "torch_threads_per_worker": 1, "torch_interop_threads": 1,
        "deterministic_algorithms": True}


def check_runtime():
    observed = runtime()
    if observed["python"] != "3.12.14" or observed["versions"] != DIRECT:
        raise ValueError("Use the preregistered reproduction environment; do not silently change pins")
    return observed


def original_identities():
    manifest = read(ROOT / ORIGINAL / "source_manifest.json")
    mapping = {**manifest["registered_scientific_source_sha256"],
               **manifest["canonical_public_records_sha256"]}
    for name, expected in mapping.items():
        if sha(ROOT / name) != expected:
            raise ValueError("Original frozen source changed: " + name)
    return mapping


def prepare():
    """Prepare a plan, not an empirical fit or a claim of public registration."""
    import run_gru_lstm_tcn_planned as original
    from dl_data import load_development
    from dl_data_v2 import fit_relative_preprocessing
    data = load_development(ROOT)
    split = original.split_at(data, "2025-01-01")
    # Array/shape and causal data checks only; no model or optimizer is created.
    for key in ("train_indices", "inner_train_indices"):
        fit_relative_preprocessing(data, 7, split[key])
    selection = read(ROOT / ORIGINAL / "selected_recipes.json")
    original_hashes = original_identities()
    names = (*original.CODE, *original.DEVELOPMENT, original.DESIGN,
             ORIGINAL + "/selected_recipes.json", ORIGINAL + "/final_freeze_provenance.json",
             ORIGINAL + "/source_manifest.json", RUNNER, LOCK)
    document = {
        "schema_version": 1, "prepared_at_utc": stamp(),
        "purpose": "New reproduction of the final D7 development fits and learning curves only",
        "chronology": "Prepared after original evaluation; publication must precede these new fits",
        "original_records_are_not_modified": True,
        "models": list(MODELS), "history_days": 7, "target_returns_days": 7,
        "forecast_days": 1, "seeds": list(SEEDS), "R_per_family": len(SEEDS),
        "planned_trajectories": 1395, "planned_fit_stages": 2790,
        "representative_figure_seed": 8101, "workers": 4,
        "data_scope": {"calendar": ["2023-01-01", "2024-12-31"],
            "fit_origin": "2025-01-01", "inner_validation_days": 90,
            "train_samples": len(split["train_indices"]),
            "inner_train_samples": len(split["inner_train_indices"]),
            "inner_validation_samples": len(split["inner_validation_indices"]),
            "reserve_price_payloads_allowed": 0, "reserve_scoring_allowed": False,
            "new_forecasts_allowed": False},
        "recipes": selection["recipes"], "baseline_model_unchanged": selection["baseline_model"],
        "settings": original.effective_settings(read(ROOT / original.DESIGN)),
        "execution": {"runner": "Unmodified run_gru_lstm_tcn_planned.worker_init and fixed_job",
            "stage": "inner validation epoch selection followed by fresh fixed-duration refit",
            "seeds_and_order": "All 8101–8565 for each family; no outcome-based stopping or replacement",
            "failure": "Stop on a failed job; preserve incomplete output and same planned seed on resume",
            "runtime_changes": "Exact six historical package pins; hardware/runtime differences are recorded"},
        "comparison": {"against": ORIGINAL + "/final_freeze_provenance.json",
            "files_per_trajectory": list(SCIENCE_FILES),
            "interpretation": "Hash equality supports byte-identical reproduction, not recovered originals; all mismatches remain reported",
            "original_curve_files_expected": 2790},
        "public_outputs_allowed": ["aggregate and seed-level loss curves", "fit summaries without scaler values",
            "artifact hashes and match status", "figures", "runtime and provenance"],
        "public_outputs_excluded": ["observations", "daily predictions", "weights", "learned scaler values"],
        "sources_sha256": {name: sha(ROOT / name) for name in dict.fromkeys(names)},
        "original_registered_identities_sha256": original_hashes,
        "runtime_expected": check_runtime(), "new_fits_at_preparation": 0,
    }
    write_new(ROOT / PROTOCOL, document)
    return document


def protocol():
    value = read(ROOT / PROTOCOL)
    if (value["models"] != list(MODELS) or tuple(value["seeds"]) != SEEDS
            or value["planned_trajectories"] != 1395 or value["workers"] != 4):
        raise ValueError("The complete predeclared three-family 465-seed replay is required")
    for name, expected in value["sources_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Reproduction source or input changed: " + name)
    original_identities()
    check_runtime()
    return value


def private_destination(path):
    destination = Path(path).resolve()
    if destination.is_relative_to(ROOT):
        raise ValueError("Private output must be outside the source checkout")
    if any((parent / ".git").exists() and parent not in {Path('/workspace'), Path('/tmp')}
           for parent in (destination, *destination.parents)):
        raise ValueError("Private output cannot be placed inside another Git checkout")
    return destination


def register(out, published_commit):
    plan = protocol()
    revision = git("rev-parse", "HEAD")
    if published_commit != revision or git("status", "--porcelain"):
        raise ValueError("Register from the clean, already-published source commit")
    names = (PROTOCOL, *plan["sources_sha256"])
    for name in names:
        recorded = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=ROOT)
        if recorded != (ROOT / name).read_bytes():
            raise ValueError("Source must be included unchanged in the published commit: " + name)
    out = private_destination(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("A new registration requires an empty private directory")
    registration = {"registered_at_utc": stamp(), "source_commit": revision,
        "published_source_url": f"https://github.com/jassan-bit/DeepLearning_202630/commit/{revision}",
        "protocol_sha256": sha(ROOT / PROTOCOL), "protocol": plan, "runtime": runtime(),
        "settings": plan["settings"], "new_fits_before_registration": 0,
        "reserve_price_payloads_read": 0, "forecasts_generated": 0,
        "scope": "New development reproduction; not original private histories or a new holdout benchmark"}
    registration["registration_signature"] = hashlib.sha256(encode(registration)).hexdigest()
    write_new(out / "registration.json", registration)
    return registration


def registration(out):
    out = private_destination(out)
    record = read(out / "registration.json")
    signature = record["registration_signature"]
    unsigned = {k: v for k, v in record.items() if k != "registration_signature"}
    if hashlib.sha256(encode(unsigned)).hexdigest() != signature:
        raise ValueError("Private registration was changed")
    if record["protocol_sha256"] != sha(ROOT / PROTOCOL) or record["protocol"] != protocol():
        raise ValueError("Published protocol differs from the private registered plan")
    if record["source_commit"] != git("rev-parse", "HEAD"):
        raise ValueError("Train from the exact registered source commit")
    return out, record


def fit(out):
    out, record = registration(out)
    if git("status", "--porcelain"):
        raise ValueError("Keep the immutable reproduction checkout clean")
    import run_gru_lstm_tcn_planned as original
    jobs = [{**recipe, "phase": "final", "trajectory_seed": seed, "seed": seed}
            for recipe in record["protocol"]["recipes"].values() for seed in SEEDS]
    if not (out / "fit_started.json").exists():
        write_new(out / "fit_started.json", {"started_at_utc": stamp(),
            "registration_signature": record["registration_signature"], "planned_jobs": len(jobs)})
    started = time.perf_counter()
    completed = []
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn"),
            initializer=original.worker_init, initargs=(ROOT, out, record)) as pool:
        pending = {pool.submit(original.fixed_job, job): job for job in jobs}
        for future in as_completed(pending):
            completed.append(future.result())
            if len(completed) % 15 == 0 or len(completed) == len(jobs):
                print(json.dumps({"completed": len(completed), "planned": len(jobs),
                    "elapsed_seconds": round(time.perf_counter() - started, 2)}), flush=True)
    import pandas as pd
    pd.DataFrame(completed).sort_values(["model", "trajectory_seed"]).to_csv(out / "fit_metrics.csv", index=False)
    result = {"finished_at_utc": stamp(), "registration_signature": record["registration_signature"],
        "models": len(completed), "new_fit_stages": 2 * len(completed),
        "wall_seconds": time.perf_counter() - started,
        "reserve_price_payloads_read": 0, "forecasts_generated": 0,
        "fit_metrics_sha256": sha(out / "fit_metrics.csv")}
    write_new(out / "fit_finished.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "register", "fit"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--published-commit")
    args = parser.parse_args()
    if args.phase == "prepare":
        result = prepare()
        print(json.dumps({"protocol": PROTOCOL, "models": result["planned_trajectories"], "new_fits": 0}))
    else:
        if args.output_dir is None:
            parser.error("--output-dir is required")
        if args.phase == "register":
            if args.published_commit is None:
                parser.error("--published-commit is required")
            result = register(args.output_dir, args.published_commit)
            print(json.dumps({"registered_at_utc": result["registered_at_utc"],
                "source_commit": result["source_commit"], "new_fits": 0}))
        else:
            print(json.dumps(fit(args.output_dir)))


if __name__ == "__main__":
    main()
