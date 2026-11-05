"""Portable reproduction adapter for the current LSTM--GRU--TCN benchmark.

Default validation never trains. Manual ``run`` creates a new registration
before fresh development fits. ``close``, ``fit`` and ``score`` dispatch the
separate frozen final workflow. Building the Jupyter Book invokes none of them.

This publication adapter postdates the recorded experiment; its source hash is
not the pre-training hash of those archived fits. It does not retroactively
register them or promise a prospective pilot, 80% power, or improved results.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
import sys
sys.dont_write_bytecode = True

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd

HISTORIES = (7, 14, 21, 28)
FAMILIES = ("LSTM", "GRU", "TCN")
DEFAULT_SEEDS = tuple(range(4101, 4111))
DEFAULT_MODELS = {"LSTM": {"width": 32, "learning_rate": .003},
                  "GRU": {"width": 32, "learning_rate": .003},
                  "TCN": {"width": 16, "learning_rate": .001}}
DEVELOPMENT_SOURCES = (
    "results/eda_development/daily_market_development.csv",
    "results/eda_development/daily_volatility_development.csv",
)
FOLD_SOURCES = {7: "results/dl_r2_cv_20261104/folds.json",
    **{d: f"results/filipe_multihistory_20261104/D{d}/folds.json" for d in (14, 21, 28)}}
CODE_FILES = (
    "scripts/run_gru_lstm_tcn.py", "scripts/dl_models_gru_lstm_tcn.py",
    "scripts/gru_lstm_tcn_final.py", "scripts/dl_models.py", "scripts/dl_data.py",
    "scripts/dl_data_v2.py", "scripts/train_dl_benchmark.py",
    "scripts/train_filipe_multihistory.py", "scripts/final_fit_filipe_multihistory.py",
    "scripts/load_filipe_reserved_test.py",
    "scripts/analyze_gru_lstm_tcn_run.py", "scripts/build_gru_lstm_tcn_evidence.py",
    "scripts/gru_lstm_tcn_diagnostics.py",
)
_HISTORICAL = None


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as stream:
        stream.write(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def project_path(root, relative):
    """Resolve a portable source/artifact name without escaping its root."""
    root, relative = Path(root).resolve(), Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Manifest paths must be relative and cannot contain '..'")
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError("A source/artifact symlink escapes the selected root")
    return result


def verify_mapping(mapping, base):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("A nonempty relative-path SHA256 mapping is required")
    for name, expected in mapping.items():
        expected = expected.get("sha256") if isinstance(expected, dict) else expected
        if (not isinstance(expected, str) or len(expected) != 64
                or any(c not in "0123456789abcdef" for c in expected.lower())):
            raise ValueError("Invalid SHA256 for " + str(name))
        path = project_path(base, name)
        if not path.is_file() or digest(path) != expected.lower():
            raise ValueError("Missing or modified artifact: " + str(name))
    return len(mapping)


def verify_evidence_manifest(path, base=None):
    """Verify a portable publication/restoration copy, without fitting models.

    Raw historical manifests with absolute private paths are deliberately not
    treated as portable publication manifests.
    """
    path = Path(path).resolve()
    document = json.loads(path.read_text())
    mapping = next((document[key] for key in (
        "artifact_sha256", "file_sha256", "files_sha256", "files") if key in document), None)
    count = verify_mapping(mapping, Path(base).resolve() if base is not None else path.parent)
    return {"status": "portable_evidence_verified", "artifacts_verified": count,
        "manifest_sha256": digest(path), "training_performed": False,
        "provenance_limit": "Integrity of this publication copy; not a claim that this adapter trained the archived experiment"}


def configure(project_root, output_dir):
    root, out = Path(project_root).resolve(), Path(output_dir).resolve()
    sys.path.insert(0, str(root / "scripts"))
    models = importlib.import_module("dl_models_gru_lstm_tcn")
    fitting = importlib.import_module("train_dl_benchmark")
    historical = importlib.import_module("train_filipe_multihistory")
    for module in (models, fitting, historical):
        if Path(module.__file__).resolve().parent != root / "scripts":
            raise RuntimeError("Switching project roots requires a fresh Python process")
    fitting.build_model = models.build_model
    historical.ROOT, historical.OUT = root, out / "results"
    historical.PROTOCOL = out / "protocol.json"
    historical.NEW_HISTORIES = HISTORIES
    historical.architecture_spec = models.architecture_spec
    return fitting, historical, models


def validate_inputs(project_root):
    root = Path(project_root).resolve()
    missing = [name for name in (*DEVELOPMENT_SOURCES, *FOLD_SOURCES.values(), *CODE_FILES)
               if not project_path(root, name).is_file()]
    if missing:
        raise FileNotFoundError("Prepare the documented development CSVs, archived Filipe folds and source package first; missing: " + ", ".join(missing))
    fitting, historical, _ = configure(root, root / "results/gru_lstm_tcn_reproduction")
    from dl_data import load_development
    data = load_development(root)
    folds = {}
    expected_dates = np.flatnonzero(data["calendar"].year == 2024).tolist()
    for history, name in FOLD_SOURCES.items():
        manifest = json.loads(project_path(root, name).read_text())
        if (manifest["history_days"] != history or manifest["target_returns_days"] != 7
                or manifest["forecast_days"] != 1 or manifest["monthly_fold_count"] != 12
                or manifest["library"]["version"] != "0.1.5"):
            raise ValueError("Unexpected archived Filipe fold contract: " + name)
        covered = [index for fold in manifest["folds"]
                   for index in historical.validate_fold(data, history, fold).tolist()]
        if covered != expected_dates or len(covered) != 366:
            raise ValueError("Each history requires exactly one chronological validation prediction per2024day")
        folds[history] = manifest
    return folds


def _source_commit(root):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None  # A restored source ZIP need not contain .git.


def register(project_root, output_dir, workers=3):
    root, out = Path(project_root).resolve(), Path(output_dir).resolve()
    if (out / "protocol.json").exists():
        return load_registered(root, out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("A fresh reproduction needs an empty output directory; never overwrite the published evidence")
    if isinstance(workers, bool) or int(workers) != workers or workers < 1:
        raise ValueError("workers must be a positive integer")
    folds = validate_inputs(root)
    fitting, _, _ = configure(root, out)
    sources = {name: digest(project_path(root, name)) for name in (*DEVELOPMENT_SOURCES, *FOLD_SOURCES.values())}
    code_names = list(CODE_FILES)
    diagnostics = "scripts/gru_lstm_tcn_diagnostics.py"
    if diagnostics not in code_names and project_path(root, diagnostics).is_file():
        code_names.append(diagnostics)
    code = {name: digest(project_path(root, name)) for name in code_names}
    training = {"optimizer": "AdamW", "weight_decay": .0001, "dropout": .1,
        "batch_size": 32, "max_epochs": 100, "gradient_clip_norm": 1.,
        "early_stopping": "Minimum inner-validation scaledMSE", "patience": 12,
        "min_delta": .00001, "refit": "Fresh model on all permitted monthly training windows at selected inner epochs",
        "inner_seed": "runseed", "refit_seed": "runseed+10000000",
        "batch_order": "Generator seeded with stage seed+17; shuffled training windows only",
        "device": "cpu", "torch_threads": 1, "torch_interop_threads": 1,
        "strict_deterministic_algorithms": True}
    plan = {"schema_version": 1, "adapter_kind": "public_post_training_reproduction_adapter",
        "registered_at_utc": stamp(), "source_commit_if_available": _source_commit(root),
        "source_provenance": "This is a NEW reproduction registration, not the original local experiment's pre-training source hash",
        "histories": list(HISTORIES), "variant": "relative_window", "models": DEFAULT_MODELS,
        "seeds": list(DEFAULT_SEEDS), "workers": int(workers), "training": training,
        "sources": sources, "code_sha256": code,
        "data_contract": {"raw_frequency": "1minute", "model_frequency": "dailyUTC",
            "features": ["open", "high", "low", "close", "volume"],
            "target": "PopulationSD(ddof0) of seven daily logClose returns; unannualized",
            "target_returns_days": 7, "forecast_days": 1, "target_lags_added": False,
            "input_support": "[outcome_day-D,outcome_day); oldest first",
            "missing_data": "No imputation or calendar compression; exclude dependent incomplete windows"},
        "calendar": {"initial_training": "2023", "forward_validation": "2024",
            "old_test": "2025-01-01..2026-08-31", "september_october2026": "unassigned; never read"},
        "cv_policy": {"folds": "Archived indices generated by timeseries-cv0.1.5 forwardChaining",
            "monthly_blocks": 12, "inner_validation_calendar_days": 90,
            "embargo_days": "D", "hyperparameter_search": "None; the three fixed recipes are not claimed globally optimal"},
        "statistics_scope": "Exploratory fixed-calendar, fixed-recipe reproduction; ten seeds are a screening budget, not a prospective power calculation",
        "pilot_status": "Prospective pilot and power-based dimensioning for this procedure remain pending; no assertion of80percent power",
        "test_source": {"daily_csv": "results/filipe_multihistory_20261104/test/daily_market_test.csv",
            "audit_json": "results/filipe_multihistory_20261104/test/raw_test_audit.json",
            "raw_reserve_manifest": "data/manifests/test_raw_reserve_20250101_20260831.json"},
        "test_policy": "Manual score only after complete frozen final models. The old period is already observed and exploratory; no test tuning or automatic acquisition",
        "publication_policy": "This runner has no GitHub/JBook publishing operation and is never invoked by a book build"}
    out.mkdir(parents=True, exist_ok=True)
    for history, manifest in folds.items():
        destination = out / "results" / f"D{history}" / "folds.json"
        destination.parent.mkdir(parents=True)
        shutil.copyfile(project_path(root, FOLD_SOURCES[history]), destination)
    write_json(out / "protocol.json", plan, exclusive=True)
    runtime = fitting.runtime()
    signature = hashlib.sha256(json.dumps({"protocol": digest(out / "protocol.json"),
        "code": code, "versions": runtime["versions"]}, sort_keys=True).encode()).hexdigest()
    context = {"registered_before_first_reproduction_fit_at_utc": stamp(),
        "protocol_sha256": digest(out / "protocol.json"), "code_sha256": code,
        "code_signature": signature, "runtime": runtime, "test_files_opened": 0,
        "provenance": "New post-publication reproduction context; original recorded training retained separately"}
    write_json(out / "results/code_manifest_before_fitting.json", context, exclusive=True)
    write_json(out / "results/runtime.json", runtime, exclusive=True)
    return plan, context, folds


def load_registered(project_root, output_dir, require_cv_complete=False):
    root, out = Path(project_root).resolve(), Path(output_dir).resolve()
    plan = json.loads((out / "protocol.json").read_text())
    context = json.loads((out / "results/code_manifest_before_fitting.json").read_text())
    if (plan.get("adapter_kind") != "public_post_training_reproduction_adapter"
            or plan.get("histories") != list(HISTORIES) or plan.get("models") != DEFAULT_MODELS
            or plan.get("seeds") != list(DEFAULT_SEEDS) or plan.get("variant") != "relative_window"):
        raise ValueError("This output directory is not a matching fresh public reproduction")
    if digest(out / "protocol.json") != context["protocol_sha256"]:
        raise ValueError("The reproduction protocol changed after registration")
    verify_mapping(plan["sources"], root)
    verify_mapping(plan["code_sha256"], root)
    fitting, _, _ = configure(root, out)
    if fitting.runtime()["versions"] != context["runtime"]["versions"]:
        raise ValueError("Software versions changed after registration; restore the locked environment or use a fresh output directory")
    folds = {}
    for history in HISTORIES:
        path = out / "results" / f"D{history}" / "folds.json"
        if digest(path) != plan["sources"][FOLD_SOURCES[history]]:
            raise ValueError("A copied CV fold manifest changed")
        folds[history] = json.loads(path.read_text())
    if require_cv_complete:
        verify_cv_outputs(root, out, plan, context)
    return plan, context, folds


def _initialize_cv_worker(project_root, output_dir, plan, folds, context):
    global _HISTORICAL
    _, historical, _ = configure(project_root, output_dir)
    historical.init_worker(plan, folds, context)
    _HISTORICAL = historical


def _run_cv_job(job):
    return _HISTORICAL.run_trajectory(job)


def _jobs(plan):
    return [{"variant": "relative_window", "model": family, "seed": seed,
        "history_days": history, "width": recipe["width"], "learning_rate": recipe["learning_rate"]}
        for history in plan["histories"] for seed in plan["seeds"]
        for family, recipe in plan["models"].items()]


def aggregate_cv(project_root, output_dir, plan, context):
    out = Path(output_dir).resolve()
    _, historical, _ = configure(project_root, out)
    records, frames, monthly = [], [], []
    for path in sorted((out / "results/runs").glob("*/summary.json")):
        record = historical.verify_completion(path.parent, context["code_signature"], "summary.json")
        records.append(record)
        frames.append(pd.read_csv(path.parent / "predictions.csv"))
        monthly.append(pd.read_csv(path.parent / "fold_metrics.csv"))
    expected = {(j["history_days"], j["model"], j["seed"]) for j in _jobs(plan)}
    observed = {(r["history_days"], r["model"], r["seed"]) for r in records}
    if len(records) != 120 or observed != expected:
        raise ValueError("All120current-family CV trajectories are required for aggregate closure")
    annual = pd.DataFrame(records).sort_values(["history_days", "model", "seed"])
    predictions, folds = pd.concat(frames, ignore_index=True), pd.concat(monthly, ignore_index=True)
    if len(predictions) != 43920 or len(folds) != 1440:
        raise ValueError("CV completeness differs from43920predictions/1440monthlysegments")
    for frame, name in ((annual, "runs.csv"), (predictions, "predictions.csv"), (folds, "fold_metrics.csv")):
        frame.to_csv(out / "results" / name, index=False)
    summary = {"completed_at_utc": stamp(), "trajectories": 120,
        "all_registered_trajectories_complete": True, "completed_project_fits": int(annual.model_fits.sum()),
        "fold_model_segments": 1440, "actual_cv_folds": 12, "prediction_dates_per_trajectory": 366,
        "seeds_per_history_model": 10, "histories": list(HISTORIES), "models": list(FAMILIES),
        "test_files_opened": 0, "D7_policy": "Fresh fits likeD14/D21/D28; no inherited reuse label",
        "scope": "New exploratory reproduction on the already observed2024calendar"}
    write_json(out / "results/execution_summary.json", summary)
    return summary


def verify_cv_outputs(project_root, output_dir, plan=None, context=None):
    root, out = Path(project_root).resolve(), Path(output_dir).resolve()
    if plan is None or context is None:
        plan, context, _ = load_registered(root, out)
    _, historical, _ = configure(root, out)
    paths = list((out / "results/runs").glob("*/summary.json"))
    if len(paths) != 120:
        raise ValueError("A complete CV reproduction needs120signed trajectory summaries")
    expected = {(j["history_days"], j["model"], j["seed"]) for j in _jobs(plan)}
    records = [historical.verify_completion(p.parent, context["code_signature"], "summary.json") for p in paths]
    if {(r["history_days"], r["model"], r["seed"]) for r in records} != expected:
        raise ValueError("Unexpected CV model/history/seed identities")
    execution = json.loads((out / "results/execution_summary.json").read_text())
    if (execution.get("trajectories") != 120 or execution.get("completed_project_fits") != 2880
            or not execution.get("all_registered_trajectories_complete") or execution.get("test_files_opened") != 0):
        raise ValueError("CV execution manifest is incomplete")
    return {"status": "complete_cv_verified", "trajectories": 120, "optimization_fits": 2880,
            "training_performed_by_verifier": False}


def run_cv(project_root, output_dir, workers=None, limit_trajectories=None):
    root, out = Path(project_root).resolve(), Path(output_dir).resolve()
    plan, context, folds = (load_registered(root, out) if (out / "protocol.json").is_file()
                            else register(root, out, workers=3 if workers is None else workers))
    jobs = _jobs(plan)
    if limit_trajectories is not None:
        if not 1 <= limit_trajectories <= len(jobs):
            raise ValueError("Invalid technical trajectory limit")
        jobs = jobs[:limit_trajectories]
    with ProcessPoolExecutor(max_workers=plan["workers"], mp_context=mp.get_context("spawn"),
            initializer=_initialize_cv_worker, initargs=(root, out, plan, folds, context)) as executor:
        futures = [executor.submit(_run_cv_job, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), 1):
            record = future.result()
            print(json.dumps({"completed_in_invocation": count, "submitted": len(jobs),
                "model": record["model"], "history_days": record["history_days"],
                "seed": record["seed"], "rmse": record["rmse"]}), flush=True)
    count = len(list((out / "results/runs").glob("*/summary.json")))
    if count == 120:
        result = aggregate_cv(root, out, plan, context)
    else:
        result = {"status": "partial technical reproduction; no configuration closure", "completed_trajectories": count}
    load_registered(root, out)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--phase", choices=("validate", "register", "run", "verify", "restoreverify", "close", "fit", "score"), default="validate")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--limit-trajectories", type=int, help="Technical partial run only; cannot close incomplete CV")
    parser.add_argument("--evidence-manifest", type=Path, help="Portable publication-copy SHA manifest; verifier never trains")
    parser.add_argument("--test-daily-csv", type=Path, help="Explicit previously prepared daily test derivative, frozen atclose")
    parser.add_argument("--test-audit-json", type=Path, help="Metadata declaring the daily derivative's expectedSHA256")
    parser.add_argument("--raw-root", type=Path, help="Already acquired monthly test ZIPs; read only byscore after final models freeze")
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = args.output_dir.resolve() if args.output_dir else root / "results/gru_lstm_tcn_reproduction"
    if args.evidence_manifest:
        if args.phase not in ("verify", "restoreverify"):
            parser.error("--evidence-manifest is valid only with a verification phase")
        result = verify_evidence_manifest(args.evidence_manifest, args.output_dir)
    elif args.phase == "validate":
        if (out / "protocol.json").exists():
            plan, context, _ = load_registered(root, out)
        else:
            validate_inputs(root)
        result = {"status": "development_inputs_validated", "models": list(FAMILIES),
            "histories": list(HISTORIES), "planned_cv_trajectories": 120,
            "planned_cv_fits": 2880, "training_performed": False, "test_payload_opened": False}
    elif args.phase == "register":
        plan, context, _ = register(root, out, workers=3 if args.workers is None else args.workers)
        result = {"status": "new_reproduction_registered", "code_signature": context["code_signature"], "training_performed": False}
    elif args.phase == "run":
        result = run_cv(root, out, args.workers, args.limit_trajectories)
    elif args.phase in ("verify", "restoreverify"):
        result = verify_cv_outputs(root, out)
    else:
        sys.path.insert(0, str(root / "scripts"))
        from gru_lstm_tcn_final import run_final_phase
        result = run_final_phase(root, out, args.phase, test_daily_csv=args.test_daily_csv,
            test_audit_json=args.test_audit_json, raw_root=args.raw_root)
    print(json.dumps(clean(result), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
