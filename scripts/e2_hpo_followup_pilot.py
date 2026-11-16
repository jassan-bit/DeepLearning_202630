#!/usr/bin/env python3
"""Three capped, full-policy CPU HPO timing attempts on the first inner fold.

--prepare only freezes an independent pilot plan. --admit checks source/runtime
and development admission without fitting. --run needs that admission and charges
each worker to the continued ledger. No test/calibration values are decoded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
VERSION = "e2-three-full-HPO-timing-pilot-v2"
DEFAULT_OUT = ROOT / "book_e2/_static/e2_closure_followup_20261116/hpo/pilot_v2"
MODELS = (("MLP16", 60), ("iTransformer", 90), ("TimesNet", 270))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def module(root, name):
    sys.path.insert(0, str(root / "scripts"))
    value = __import__(name)
    if Path(value.__file__).resolve() != root / "scripts" / (name + ".py"):
        raise ValueError("Scientific module shadowed: " + name)
    return value


def budget_module(root):
    path = root / "scripts/e2_corrections_budget.py"
    spec = importlib.util.spec_from_file_location("_hpo_followup_budget", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def prepare(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a fresh pilot directory; previous plans/results stay immutable")
    source = args.source_root.resolve()
    registered = source / "book_e2/_static/e2_corrections_20261115/hpo/sources/configurations_registered.json"
    configs = json.loads(registered.read_text())
    bound = [
        source / "results/eda_development/daily_market_development.csv",
        source / "results/eda_development/daily_volatility_development.csv",
        source / "experiments/dl_r2_cv_20261104/protocol.json",
        registered,
        source / "experiments/e2_published_architectures_20261112/architecture_policy.json",
    ]
    bound += sorted((source / "scripts").glob("e2_*.py"))
    rows = [{"path": str(p.relative_to(source)), "sha256": sha(p), "bytes": p.stat().st_size}
            for p in dict.fromkeys(bound)]
    previous = source / "book_e2/_static/e2_scientific_closure_20261116/reproduction/v2/plan.json"
    authors = [r for r in json.loads(previous.read_text())["author_files"]
               if r["model"] in {m for m, _ in MODELS}]
    jobs = []
    for index, (name, cap) in enumerate(MODELS, 1):
        selected = next(r for r in configs[name] if r["configuration_id"] == "C03")
        assert selected["parent_trial_index"] == 8
        jobs.append({"id": f"pilot-{index:02d}", "model": name, "cap_seconds": cap,
                     "seed": 271211, "fold_origin": "2024-01-01", "configuration_id": "C03",
                     "parent_trial_index": 8, "trial": selected["trial"],
                     "configuration_source_sha256": sha(registered),
                     "selection_reason": "Prespecified third registered position for all3models; not chosen from fitness/winner"})
    plan = {
        "version": VERSION, "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha(__file__), "source_root": str(source), "author_root": str(args.author_root.resolve()),
        "bound_sources": rows, "author_files": authors,
        "budget_source_sha256": sha(source / "scripts/e2_corrections_budget.py"),
        "ledger_path": str(args.ledger.resolve()), "budget_reset": False,
        "accounting": "Each worker charged once including imports, setup, failure and timeout; controller wall not added to nested workers",
        "group_cap_seconds_assigned": 450, "worker_cap_sum_seconds": 420,
        "admission_setup_cap_seconds": 30, "main_jobs": jobs,
        "data_admission": {
            "source_files": "Two original hash-pinned2023-2024derivatives",
            "last_numerical_date_decoded": "2023-12-24", "calendar_placeholder_end": "2024-01-01",
            "train_count": 253, "validation_count": 83,
            "train_label_end": "2023-09-25", "validation_start": "2023-10-03", "validation_end": "2023-12-24",
            "scaler": "Fit only253innertrain observations; validation uses same scaler",
            "calibration53_numeric_values_decoded": False, "test608_accessed": False,
        },
        "runtime": {"python": "3.12", "torch": "2.8.0+cpu", "numpy": "2.3.5", "pandas": "2.2.3",
                    "threads": 2, "interop_threads": 2, "deterministic_algorithms": True,
                    "device": "cpu", "dtype": "float32", "parallel_workers": 1},
        "HPO_policy": {"max_epochs": 200, "patience": 20, "fixed_epochs": None,
                       "scheduler_horizon_epochs": 200, "optimizer": "unchanged AdamW/groups, cosine/warmup,clip1",
                       "epoch_validation": "Original batch-size-specific validation used for stopping; persisted selected-checkpoint scoring uses batch1",
                       "process_cap_includes_imports_setup_and_scoring": True},
        "timeout_policy": "Record censored attempt and measured charge; partial epochs/steps never treated as a completed HPO fitness or full cost. No automatic retry/replacement.",
        "limits": ["Posterior timing pilot of3representatives, not all13bodies",
                   "Does not estimate selection variance or isolated hyperparameter importance",
                   "Different batches/epochs/body costs prohibit exact whole-study extrapolation",
                   "A censored fit gives no finite complete-fit cost bound",
                   "Historical costs and earlier failed jobs are unchanged"],
        "supersedes_only_conceptual_timing_candidate": "Earlier unexecuted F07proposal requires new registry admission; this pilot uses exact existing C03 and does not execute the new16tuple domain",
    }
    write(args.output / "plan.json", plan)
    print(json.dumps({"plan": str(args.output / "plan.json"), "sha256": sha(args.output / "plan.json"),
                      "source_files": len(rows), "author_files": len(authors), "new_fits": 0}))


def load(args):
    plan = json.loads(args.plan.read_text())
    if plan["version"] != VERSION or plan["script_sha256"] != sha(__file__):
        raise ValueError("Registered pilot source changed")
    source = Path(plan["source_root"])
    for row in plan["bound_sources"]:
        path = source / row["path"]
        if path.is_symlink() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError("Registered source changed: " + row["path"])
    for row in plan["author_files"]:
        path = Path(plan["author_root"]) / row["model"] / row["path"]
        if path.is_symlink() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError("Author source changed: " + row["model"] + "/" + row["path"])
    if str(args.ledger.resolve()) != plan["ledger_path"]:
        raise ValueError("Use the continued registered ledger, never a replacement")
    return plan, source


def runtime(plan):
    import torch
    for name in ("torch", "numpy", "pandas"):
        actual = torch.__version__ if name == "torch" else importlib.metadata.version(name)
        if actual != plan["runtime"][name]:
            raise ValueError(f"Runtime differs: {name} {actual}")
    if not sys.version.startswith(plan["runtime"]["python"] + "."):
        raise ValueError("Python major.minor differs")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    return torch


def first_fold(plan, source):
    """Reuse original hash/parser/scaler without decoding later numerical cells."""
    import numpy as np
    import pandas as pd
    data = module(source, "e2_development_data")
    temporal = module(source, "e2_temporal_training")
    protocol_raw, protocol_sha = data._read_registered(source, data.PROTOCOL_PATH, data.EXPECTED_PROTOCOL_SHA256)
    protocol = json.loads(protocol_raw)
    if any(protocol["sources"].get(k) != v for k, v in data.EXPECTED_SOURCES.items()):
        raise ValueError("Development source map changed")
    decoded, hashes, parsing = {}, {}, {}
    for path, expected in data.EXPECTED_SOURCES.items():
        raw, hashes[path] = data._read_registered(source, path, expected)
        dates, decoded[path], parsing[path] = data._prefix_values(raw, path, pd.Timestamp("2023-12-24", tz="UTC"))
    features, target = decoded[data.MARKET_PATH], decoded[data.TARGET_PATH][:, 0]
    rebuilt = pd.Series(np.log(features[:, 3]), index=dates).diff().rolling(7, min_periods=7).std(ddof=0).to_numpy()
    if not np.allclose(target, rebuilt, rtol=1e-9, atol=1e-13, equal_nan=True):
        raise ValueError("Population sigma7 target reconstruction changed")
    calendar = pd.date_range("2023-01-01", "2024-01-01", tz="UTC")
    x, y, quality = np.full((len(calendar), 5), np.nan), np.full(len(calendar), np.nan), np.zeros(len(calendar), dtype=bool)
    x[:len(features)], y[:len(target)] = features, target
    quality[:len(features)] = np.isfinite(features).all(axis=1)
    panel = temporal.DatedPanel(calendar, x, y, quality)
    prepared = temporal.prepare_b(panel, "2024-01-01", "2024-01-01")
    train, valid = prepared.inner_train, prepared.inner_validation
    checks = (len(train.x) == 253, len(valid.x) == 83,
              str(train.dates[-1].date()) == "2023-09-25",
              str(valid.dates[0].date()) == "2023-10-03", str(valid.dates[-1].date()) == "2023-12-24",
              train.scaler.record() == valid.scaler.record())
    if not all(checks):
        raise ValueError("First inner fold support/scaler differs")
    receipt = {"source_protocol_sha256": protocol_sha, "source_sha256": hashes, "parsing": parsing,
               "last_numerical_date_decoded": "2023-12-24", "train_count": 253, "validation_count": 83,
               "train_signature": train.signature, "validation_signature": valid.signature,
               "scaler": train.scaler.record(), "target_reconstruction_matches": True,
               "calibration53_numeric_values_decoded": False, "test608_accessed": False}
    return train, valid, receipt


def admit(args, plan, source):
    runtime(plan)
    _, _, data = first_fold(plan, source)
    registry = module(source, "e2_published_registry")
    for job in plan["main_jobs"]:
        if job["trial"] not in registry.registered_trials(job["model"]):
            raise ValueError("Pilot trial is outside exact registered recipe")
    value = {"version": VERSION, "plan_sha256": sha(args.plan), "source_admission_passed": True,
             "data": data, "new_fits": 0, "model_forward_passes": 0,
             "recorded_at_utc": datetime.now(timezone.utc).isoformat()}
    write(args.output / "admission.json", value)
    print(json.dumps({"source_admission_passed": True, "models": 3, "train": 253, "validation": 83, "new_fits": 0}))


def worker(args, plan, source):
    budget = budget_module(source)
    with budget.locked_ledger(args.ledger) as (rows, _):
        reservation = budget.summarize(rows)["open_reservations"].get(args.reservation_id)
    job = next(j for j in plan["main_jobs"] if j["id"] == args.job)
    phase = "followup-HPO-timing:" + job["id"]
    if not reservation or reservation["phase"] != phase or reservation["cap_seconds"] != job["cap_seconds"]:
        raise ValueError("An assigned active exact-job reservation is mandatory")
    admission = json.loads((args.output / "admission.json").read_text())
    if not admission["source_admission_passed"] or admission["plan_sha256"] != sha(args.plan):
        raise ValueError("Successful frozen source/data admission required")
    started = time.monotonic()
    destination = args.output / job["id"]
    destination.mkdir(parents=True, exist_ok=False)
    write(destination / "job_identity.json", {"version": VERSION, "job": job, "plan_sha256": sha(args.plan),
                "reservation_id": args.reservation_id, "started_at_utc": datetime.now(timezone.utc).isoformat()})
    outcome = {"version": VERSION, "job": job["id"], "model": job["model"], "status": "failed",
               "plan_sha256": sha(args.plan), "partial_fit_is_complete_fitness": False}
    try:
        torch = runtime(plan)
        train, valid, data = first_fold(plan, source)
        write(destination / "data_admission.json", data)
        engine = module(source, "e2_staged_training")
        factory = module(source, "e2_reduced_fit")
        roots = {name: str(Path(plan["author_root"]) / name) for name in {r["model"] for r in plan["author_files"]}}
        engine._seed(job["seed"])
        model = factory._builder(job["model"], job["trial"], job["seed"], roots)
        parameters = sum(p.numel() for p in model.parameters())
        stage_wall = job["cap_seconds"] - (time.monotonic() - started) - 7
        if stage_wall <= 0:
            raise TimeoutError("Pilot setup consumed worker cap; no optimizer was started")
        t = job["trial"]
        config = engine.TrainConfig(learning_rate=t["learning_rate"], weight_decay=t["weight_decay"],
                    batch_size=t["batch_size"], max_epochs=200, patience=20, seed=job["seed"],
                    warmup_fraction=t.get("warmup_fraction", 0), wall_limit_seconds=stage_wall,
                    memory_limit_bytes=12 * 1024 ** 3)
        write(destination / "fit_request.json", {"config": config.__dict__, "fixed_epochs": None,
              "parameters": parameters, "training_signature": train.signature,
              "validation_signature": valid.signature, "constructed_before_optimizer": True})
        tick = time.monotonic()
        result = engine.fit_supervised(model, train.x, train.y, valid.x, valid.y,
                                       train.scaler.target_scale, config, fixed_epochs=None)
        fit_seconds = time.monotonic() - tick
        write(destination / "phase_result.json", result.to_dict())
        checkpoint = destination / "selected_model.pt"
        torch.save({"state_dict": result.state_dict, "model": job["model"], "seed": job["seed"],
                    "trial": job["trial"], "scaler": train.scaler.record(), "plan_sha256": sha(args.plan)}, checkpoint)
        model.load_state_dict(result.state_dict, strict=True)
        model.eval()
        predictions = []
        with torch.inference_mode():
            for x in valid.x:
                if time.monotonic() - started >= job["cap_seconds"] - 2:
                    raise TimeoutError("Capped pilot stopped before completing selected-checkpoint validation scoring")
                value = model(x.unsqueeze(0)).reshape(-1)
                if value.shape != (1,) or not bool(torch.isfinite(value).all()):
                    raise ValueError("Invalid causal batch1 validation prediction")
                predictions.append(value.item() * train.scaler.target_scale)
        rmse = (sum((a-b)**2 for a, b in zip(predictions, valid.y_original)) / len(predictions)) ** 0.5
        write(destination / "validation_predictions.json", {"dates": [str(d.date()) for d in valid.dates],
              "prediction_original": predictions, "target_original": list(valid.y_original),
              "RMSE": rmse, "batch_size": 1, "used_for_any_recipe_selection": False})
        outcome.update(status="completed", full_HPO_policy_fit_completed=True,
                       epochs_completed=result.epochs_completed, best_epoch=result.best_epoch,
                       optimizer_steps=result.optimizer_steps, stop_reason=result.stop_reason,
                       fit_seconds=fit_seconds, selected_checkpoint_sha256=sha(checkpoint),
                       validation_count=len(predictions), validation_RMSE=rmse,
                       timer_scope="worker includes loading/setup, full200/patience20HPO fit, checkpoint and selected validation scoring")
    except Exception as error:
        receipt = getattr(error, "receipt", None)
        if receipt is not None:
            write(destination / "failed_stage_receipt.json", receipt)
        timeout = isinstance(error, TimeoutError) or "wall limit" in str(error).lower() or "time limit" in str(error).lower()
        outcome.update(status="censored_timeout" if timeout else "failed", error_type=type(error).__name__,
                       reason=str(error), full_HPO_policy_fit_completed=False,
                       failed_stage_receipt_saved=receipt is not None,
                       completed_fit_cost_estimated=False, retry_performed=False)
    outcome["worker_elapsed_seconds_before_final_receipt"] = time.monotonic() - started
    write(destination / "outcome.json", outcome)
    print(json.dumps({"job": job["id"], "model": job["model"], "status": outcome["status"],
                      "elapsed_seconds": outcome["worker_elapsed_seconds_before_final_receipt"]}))
    return 0 if outcome["status"] == "completed" else 2


def run(args, plan, source):
    admission = json.loads((args.output / "admission.json").read_text())
    if not admission["source_admission_passed"] or admission["plan_sha256"] != sha(args.plan):
        raise ValueError("Admission missing/changed; do not launch pilot")
    budget = budget_module(source)
    summary = []
    for job in plan["main_jobs"]:
        if (args.output / job["id"]).exists():
            raise ValueError("No automatic replay/retry of prior pilot attempts")
        phase = "followup-HPO-timing:" + job["id"]
        reservation_id = budget.reserve(phase, job["cap_seconds"], path=args.ledger)
        started, returncode = time.monotonic(), 1
        process_timeout = False
        try:
            command = [str(args.python), str(Path(__file__).resolve()), "--worker", "--plan", str(args.plan),
                       "--output", str(args.output), "--ledger", str(args.ledger),
                       "--job", job["id"], "--reservation-id", reservation_id]
            result = subprocess.run(command, timeout=job["cap_seconds"], check=False)
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            returncode, process_timeout = 124, True
        finally:
            elapsed = time.monotonic() - started
            budget.finish(reservation_id, elapsed, returncode, path=args.ledger)
        result_path = args.output / job["id"] / "outcome.json"
        outcome = json.loads(result_path.read_text()) if result_path.exists() else {
            "status": "censored_process_timeout" if process_timeout else "failed_without_worker_receipt",
            "full_HPO_policy_fit_completed": False, "completed_fit_cost_estimated": False}
        row = {"job": job["id"], "model": job["model"], "cap_seconds": job["cap_seconds"],
               "reservation_id": reservation_id, "observed_charged_worker_seconds": elapsed,
               "returncode": returncode, "outcome": outcome}
        write(args.output / (job["id"] + "_worker_charge.json"), row)
        summary.append(row)
    total = sum(r["observed_charged_worker_seconds"] for r in summary)
    write(args.output / "pilot_summary.json", {"version": VERSION, "plan_sha256": sha(args.plan),
          "jobs": summary, "worker_charges_seconds": total,
          "completed_full_policy_fits": sum(r["outcome"]["status"] == "completed" for r in summary),
          "attempts": 3, "calibration_test_accessed": False, "old_outcomes_changed": False,
          "whole13model_HPO_cost_forecast_established": False, "selection_variance_established": False,
          "isolated_importance_established": False, "NAS": "not_applicable_defined_bodies",
          "projection": "Completed representatives give observed job costs for exact recipes/fold; censored attempts only lower-bound required complete cost. Other10models and broader domains need their own calibrated bounds; no full-campaign numerical forecast is identified."})
    print(json.dumps({"pilot_attempts": 3, "worker_charges_seconds": total,
                      "summary": str(args.output / "pilot_summary.json")}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "admit", "run", "worker"):
        mode.add_argument("--" + name, action="store_true")
    p.add_argument("--source-root", type=Path, default=ROOT)
    p.add_argument("--author-root", type=Path, default=Path("/workspace/work/e2architecture_sources"))
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--plan", type=Path, default=DEFAULT_OUT / "plan.json")
    p.add_argument("--ledger", type=Path, default=ROOT / "book_e2/_static/e2_closure_followup_20261116/compute_ledger.jsonl")
    p.add_argument("--python", type=Path, default=Path(sys.executable))
    p.add_argument("--job")
    p.add_argument("--reservation-id", help=argparse.SUPPRESS)
    args = p.parse_args()
    args.output = args.output.resolve(); args.plan = args.plan.resolve()
    if args.prepare:
        prepare(args); return
    plan, source = load(args)
    if args.admit:
        admit(args, plan, source)
    elif args.worker:
        raise SystemExit(worker(args, plan, source))
    else:
        run(args, plan, source)


if __name__ == "__main__":
    main()
