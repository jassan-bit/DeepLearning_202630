"""Durable, sequential CUDA pilots for the registered sixteen-system amendment.

Default invocation is a metadata/GPU/budget preflight. Market training requires
--execute, the reconciled shared allowance, and an exclusive persistent ledger.
Neither an incomplete pilot nor a synthetic fixture can choose scientific R/k.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
VERSION = "e2-amended-artifact-pilots-v1"
MODELS = ("TimeMixer", "PatchTST", "GRU32")
NOISE_SEEDS = tuple(range(271501, 271506))
POWER_SEEDS = tuple(range(271701, 271706))
HPO_SEEDS = (271201, 271202)
STAGE_WALL = 7200
WORKER_RESERVATION = 7500  # Full original stage cap plus startup/persistence.
LIVE_PILOT_EXECUTION_ENABLED = False  # Submission closes with existing evidence.


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    """Exclusive durable metadata, without hardlinks (Drive/FUSE compatible)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != canonical(value) + b"\n":
            raise FileExistsError("An existing immutable pilot record differs: " + str(path))
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def execution_spec():
    return {
        "version": VERSION, "parent_protocol": "e2-protocol-amendment-v1",
        "noise": {"model": "TiDE", "origin": "2023-10-01",
            "last_label": "2023-09-23", "architectures": 12,
            "seeds": list(NOISE_SEEDS), "max_epochs": 200},
        "october_hpo_prerequisites": {"models": list(MODELS), "origin": "2024-10-01",
            "last_label": "2024-09-23", "trials": 18, "seeds": list(HPO_SEEDS),
            "logical_fits": 108, "closes_common_HPO": False},
        "power": {"models": list(MODELS), "seeds": list(POWER_SEEDS),
            "origin": "2024-11-08", "end": "2024-12-30", "days": 53,
            "train_label_end": "2024-10-31", "logical_fits": 15,
            "epochs": "ceil(median(two October winner best epochs)); fresh fixed-epoch refit",
            "early_stopping_on_power_labels": False},
        "resources": {"stage_wall_seconds": STAGE_WALL,
            "whole_worker_reservation_seconds": WORKER_RESERVATION,
            "rss_limit_bytes": 12 * 1024**3, "precision": "float32",
            "amp": False, "tf32": False, "deterministic": True,
            "automatic_retry": False, "automatic_budget_extension": False},
        "artifacts": ["configuration", "source/data/scaler identities", "initial weights",
            "epoch curves/resources", "best/latest weights and optimizer/RNG states",
            "dated predictions", "terminal receipts", "charged time"],
        "power_limits": "Conditional on five seeds and53 known development days; not final evidence or a guarantee for all20 tests/future regimes",
    }


def job_plan(stage="all"):
    """Enumerate complete registered domains; no market source is opened."""
    from e2_published_registry import registered_trials
    from e2_training_models import protocol
    from e2_search_plan import noise_pilot_architectures
    jobs = []
    if stage in ("noise", "all"):
        for pattern in noise_pilot_architectures(protocol()):
            for seed in NOISE_SEEDS:
                jobs.append({"stage": "noise", "model": "TiDE", "seed": seed,
                    "origin": "2023-10-01", "architecture": pattern,
                    "trial": {"learning_rate": .001, "weight_decay": .0001,
                              "batch_size": 32, "warmup_fraction": 0}})
    if stage in ("power", "all"):
        for model in MODELS:
            for trial in registered_trials(model):
                for seed in HPO_SEEDS:
                    jobs.append({"stage": "hpo", "model": model, "seed": seed,
                        "origin": "2024-10-01", "trial": trial})
        for model in MODELS:
            for seed in POWER_SEEDS:
                jobs.append({"stage": "power", "model": model, "seed": seed,
                    "origin": "2024-11-08", "end": "2024-12-30"})
    return jobs


def source_identity():
    # All executable repository sources, not only modules imported so far.
    files = sorted((ROOT / "scripts").glob("*.py"))
    return {str(path.relative_to(ROOT)): file_sha(path) for path in files}


def current_source_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def require_no_other_controller():
    markers = {"run_e2_cuda_search.py", "run_e2_cuda_recovery.py", "e2_guide_schedule.py",
        "run_e2_optimized_selection_v2.py", "run_e2_optimized_selection.py", "run_e2_amended_pilots.py"}
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        if int(path.parent.name) == os.getpid():
            continue
        try:
            arguments = path.read_bytes().decode().split("\0")
        except (OSError, UnicodeError):
            continue
        if any(word and not any(c.isspace() for c in word) and Path(word).name in markers
               for word in arguments[1:]):
            raise RuntimeError("Another training controller is active (PID " + path.parent.name + ")")


def preflight(args):
    from e2_protocol_amendment import verify_registration
    from e2_cuda_training import validate_cuda_device
    from e2_amended_pilot_budget import fetch_reconciliation
    require_no_other_controller()
    verify_registration(ROOT, require_git=True)
    import torch
    device = validate_cuda_device("cuda:0")
    if str(torch.__version__) != "2.8.0+cu126":
        raise ValueError("Use the registered isolated Torch2.8.0+cu126 environment")
    evidence = fetch_reconciliation(args.output / "budget_evidence", token=os.environ.get("GITHUB_TOKEN"))
    if isinstance(evidence, (str, Path)):
        evidence = json.loads(Path(evidence).read_text())
    if args.budget_evidence:
        supplied = json.loads(args.budget_evidence.read_text())
        if supplied != evidence:
            raise ValueError("Supplied reconciliation is stale or differs from current remote journals")
    properties = torch.cuda.get_device_properties(device)
    from e2_amended_pilot_budget import PilotBudgetLedger
    with PilotBudgetLedger(args.output / "ledger", evidence) as ledger:
        balance = ledger.report()
    budget_ready = balance["remaining_seconds"] >= WORKER_RESERVATION
    ready = budget_ready and LIVE_PILOT_EXECUTION_ENABLED
    result = {"version": VERSION, "ready": ready,
        "status": "paused_budget_insufficient" if not budget_ready else "prepared_cuda_acceptance_pending",
        "execution_enabled": LIVE_PILOT_EXECUTION_ENABLED,
        "device": str(device), "device_name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "budget": evidence, "pilot_balance": balance, "next_reservation_seconds": WORKER_RESERVATION,
        "source_commit": current_source_commit(), "spec_sha256": sha(execution_spec()),
        "market_data_read": False, "training_started": False, "R": None, "k": None}
    args.output.mkdir(parents=True, exist_ok=True)
    # Each invocation gets its own immutable record; preflight.json is a view.
    stamp = str(time.time_ns())
    write_json(args.output / "preflights" / (stamp + ".json"), result)
    temporary = args.output / "preflight.json.tmp"
    temporary.write_bytes(canonical(result) + b"\n")
    os.replace(temporary, args.output / "preflight.json")
    print(json.dumps({k: result[k] for k in ("status", "ready", "device_name", "next_reservation_seconds")}
        | {"remaining_seconds": balance["remaining_seconds"], "market_training_started": False}, ensure_ascii=False), flush=True)
    return result


def _validate_worker_spec(spec):
    job = spec["job"]
    if spec.get("version") != VERSION or spec.get("spec_sha256") != sha(execution_spec()):
        raise ValueError("Unregistered pilot execution specification")
    if spec["source_sha256"] != source_identity():
        raise ValueError("Pilot source files changed")
    bare = {k: v for k, v in job.items() if k not in ("trial", "fixed_epochs", "winner_receipt_sha256")}
    if job["stage"] == "power":
        allowed = [{k: v for k, v in x.items() if k != "trial"} for x in job_plan("power") if x["stage"] == "power"]
        if bare not in allowed or type(job.get("fixed_epochs")) is not int or not 1 <= job["fixed_epochs"] <= 200:
            raise ValueError("Power pilot requires its frozen October recipe and epoch count")
        from e2_published_registry import registered_trials
        if job["trial"] not in registered_trials(job["model"]):
            raise ValueError("Power recipe is not an October HPO point")
        freeze = Path(spec["output"]).parents[1] / "october_winners.json"
        receipt = json.loads(freeze.read_text())
        winner = receipt["winners"][job["model"]]
        if (sha(receipt) != job["winner_receipt_sha256"] or job["trial"] != winner["trial"]
                or job["fixed_epochs"] != winner["fixed_epochs"]):
            raise ValueError("Power fit differs from its frozen October winner")
    elif job not in job_plan("all"):
        raise ValueError("Worker outside registered pilot/HPO domain")


def run_worker(spec):
    """One bounded fit, no retries and no parent-budget mutations in the child."""
    _validate_worker_spec(spec)
    import numpy as np
    import torch
    from e2_amended_pilot_artifacts import PilotArtifactStore
    from e2_amended_pilot_data import load_pilot_panel
    from e2_published_pilot_training import fit_pilot, predict_pilot
    from e2_published_registry import build_train_model
    from e2_development_models import build_pilot_model
    from e2_staged_training import TrainConfig, state_sha256
    from e2_temporal_training import prepare_b
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    job, output = spec["job"], Path(spec["output"])
    roots = json.loads(Path(spec["source_roots"]).read_text())
    data_stage = {"noise": "noise", "hpo": "hpo", "power": "power_fit"}[job["stage"]]
    panel, data_receipt = load_pilot_panel(spec["data_root"], data_stage)
    prepared = prepare_b(panel, job["origin"], job["origin"])
    train = prepared.refit_train if job["stage"] == "power" else prepared.inner_train
    valid = None if job["stage"] == "power" else prepared.inner_validation
    trial = job["trial"]
    config = TrainConfig(learning_rate=trial["learning_rate"], weight_decay=trial["weight_decay"],
        batch_size=trial["batch_size"], max_epochs=200, patience=20, seed=job["seed"],
        warmup_fraction=trial.get("warmup_fraction", 0), wall_limit_seconds=STAGE_WALL,
        memory_limit_bytes=12 * 1024**3)
    identity = {"job": job, "version": VERSION, "sources": spec["source_sha256"],
        "data": data_receipt, "prepared_signature": train.signature,
        "scaler": train.scaler.record(), "runtime": spec["runtime_identity"],
        "execution_spec_sha256": spec["spec_sha256"]}
    store = PilotArtifactStore(output / "artifacts", spec["attempt_id"], identity)
    store.prepare({"config": asdict(config), "source": spec["source_sha256"],
        "data": {"receipt": data_receipt, "train": train.record(),
            "validation": None if valid is None else valid.record()},
        "seed": job["seed"], "resources": spec["runtime_identity"]})
    if job["stage"] == "noise":
        model = build_pilot_model("TiDE", job["architecture"], seed=job["seed"], source_roots=roots)
    else:
        model = build_train_model(job["model"], trial, seed=job["seed"], source_roots=roots)
    write_json(output / "architecture.json", model.architecture_metadata)
    outcome = fit_pilot(model, train.x, train.y, config, store,
        X_val=None if valid is None else valid.x, y_val=None if valid is None else valid.y,
        fixed_epochs=job.get("fixed_epochs"), device="cuda:0",
        target_scale=train.scaler.target_scale,
        gpu_memory_limit_bytes=int(spec["runtime_identity"]["total_memory_bytes"] * .8))
    result = {"version": VERSION, "status": "completed", "job": job,
        "attempt_id": spec["attempt_id"], "worker_spec_sha256": sha(spec),
        "identity_sha256": sha(identity), "best_epoch": outcome.best_epoch,
        "epochs_completed": outcome.epochs_completed, "optimizer_steps": outcome.optimizer_steps,
        "synthetic": False, "test_only_cpu_execution": False,
        "prepared_signature": train.signature,
        "data_sha256": sha(data_receipt["source_sha256"]),
        "model_source_sha256": sha(spec["source_sha256"]),
        "execution_policy_sha256": sha(spec["runtime_identity"]),
        "recipe_id": model.architecture_metadata.get("training_recipe_id", "TiDE-noise-fixed-v1")}
    if valid is not None:
        prediction = predict_pilot(model, valid.x, batch_size=1, device="cuda:0")
        values = np.asarray(prediction, dtype=float) * train.scaler.target_scale
        truth = np.asarray(valid.y_original)
        result["validation_rmse"] = float(np.sqrt(np.mean((values - truth)**2)))
        write_json(output / "predictions.json", {"dates": [str(d.date()) for d in valid.dates],
            "predictions": values.tolist(), "truths": truth.tolist(), "data_role": "inner_validation"})
    else:
        result["power_scoring_complete"] = False
        result["scope"] = "frozen_power_weights_only; score only after all fifteen weights close"
        best = outcome.receipt["checkpoint_receipts"]["best"]
        selected = output / "artifacts" / spec["attempt_id"] / best["path"]
        result["selected_checkpoint"] = {"path": str(selected.relative_to(output)), "sha256": file_sha(selected)}
    store.complete({"optimizer_steps": outcome.optimizer_steps, "result": result})
    store.close()
    # Bind every artifact byte, not just reported scores, before allowing reuse.
    artifacts = {str(p.relative_to(output)): file_sha(p) for p in output.rglob("*") if p.is_file()
                 and p.name not in ("worker_spec.json", "worker.log", "result.json")}
    result["artifacts_sha256"] = artifacts
    write_json(output / "result.json", result)
    return result


def verified_result(directory, spec):
    path = Path(directory) / "result.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if result.get("status") != "completed" or result.get("worker_spec_sha256") != sha(spec):
        raise ValueError("An existing attempt is failed/unknown or belongs to another identity")
    if result.get("synthetic") is not False or result.get("test_only_cpu_execution") is not False:
        raise ValueError("Synthetic/CPU fixtures cannot close a scientific pilot")
    for relative, expected in result["artifacts_sha256"].items():
        path = Path(directory) / relative
        if path.is_symlink() or not path.is_file() or file_sha(path) != expected:
            raise ValueError("Saved pilot artifact is missing or changed: " + relative)
    return result


def freeze_october(results, output):
    from e2_amended_metrics import select_local_hpo
    from e2_published_registry import registered_trials
    winners = {}
    for name in MODELS:
        rows, epochs = [], {}
        for trial in registered_trials(name):
            paired = [r for r in results if r["job"]["stage"] == "hpo" and r["job"]["model"] == name
                      and r["job"]["trial"]["trial_id"] == trial["trial_id"]]
            if len(paired) != 2 or {r["job"]["seed"] for r in paired} != set(HPO_SEEDS):
                raise ValueError("October winner cannot be frozen before all108 prerequisites complete")
            rows.append({"trial_id": trial["trial_id"], "status": "completed",
                "seed_rmse": {str(r["job"]["seed"]): r["validation_rmse"] for r in paired}})
            epochs[trial["trial_id"]] = int(math.ceil(statistics.median(r["best_epoch"] for r in paired)))
        choice = select_local_hpo(rows, anchor="2024-10-01",
            registered_trial_ids=[t["trial_id"] for t in registered_trials(name)])
        trial = next(t for t in registered_trials(name) if t["trial_id"] == choice["winner_trial_id"])
        winners[name] = {"trial": trial, "fixed_epochs": epochs[trial["trial_id"]], "selection": choice}
    receipt = {"version": VERSION, "winners": winners, "prerequisite_results_sha256": sha(results),
        "power_pilot_labels_used": False, "common_HPO_complete": False}
    write_json(output / "october_winners.json", receipt)
    return receipt


def close_noise(results, output, *, scope="registered_market_pilot"):
    from e2_amended_metrics import choose_nas_k
    from e2_training_models import protocol
    from e2_search_plan import noise_pilot_architectures
    patterns = noise_pilot_architectures(protocol())
    noise = [r for r in results if r["job"]["stage"] == "noise"]
    if len(noise) != 60:
        return None
    if scope not in ("registered_market_pilot", "synthetic_fixture"):
        raise ValueError("Declare the scientific or synthetic-fixture noise scope")
    if scope == "registered_market_pilot" and any(r.get("synthetic") is not False
            or r.get("test_only_cpu_execution") is not False or r.get("status") != "completed"
            or not r.get("artifacts_sha256") for r in noise):
        raise ValueError("Only sixty completed artifact-verified CUDA market fits can close k")
    scores = []
    for seed in NOISE_SEEDS:
        scores.append([next(r["validation_rmse"] for r in noise if r["job"]["seed"] == seed
            and r["job"]["architecture"] == pattern) for pattern in patterns])
    decision = choose_nas_k({"TiDE": scores})
    decision.update(version=VERSION, synthetic=scope == "synthetic_fixture", actual_completed_fits=60 if scope != "synthetic_fixture" else 0,
        artifact_results_sha256=sha(noise), scores={"TiDE": scores})
    if scope == "synthetic_fixture":
        decision["fixture_k_candidate"], decision["k"] = decision["k"], None
    write_json(output / "k_decision.json", decision)
    return decision


def wait_worker(process, ledger, attempt_id, start, description):
    """Bound the whole child, including startup/IO, by its existing reservation."""
    while True:
        remaining = WORKER_RESERVATION - (time.monotonic() - start) - 4
        if remaining <= 0:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return 124
        try:
            return process.wait(timeout=min(20, remaining))
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            ledger.record_progress(attempt_id, elapsed, 0)
            print(json.dumps({"event": "worker_progress", "description": description,
                "elapsed_seconds": elapsed}), flush=True)


def score_worker(spec):
    """Separate read-only-in-weights scoring, after all fifteen fits are sealed."""
    if spec["source_sha256"] != source_identity() or spec["spec_sha256"] != sha(execution_spec()):
        raise ValueError("Power-scoring sources/specification changed")
    from e2_published_pilot_scoring import score_frozen_pilots
    from e2_amended_pilot_power import analyze_pilot_power
    payload, context = score_frozen_pilots(spec["power_results"], output=Path(spec["output"]),
        data_root=spec["data_root"], source_roots=spec["source_roots"],
        source_identity=spec["source_sha256"], runtime_identity=spec["runtime_identity"],
        winner_receipt=spec["winner_receipt"], device="cuda:0")
    write_json(Path(spec["output"]) / "power_payload.json", payload)
    write_json(Path(spec["output"]) / "power_context.json", context)
    decision = analyze_pilot_power(payload, expected_context=context)
    write_json(Path(spec["output"]) / "R_decision.json", decision)


def execute(args, readiness):
    from e2_amended_pilot_budget import PilotBudgetLedger
    ledger = PilotBudgetLedger(args.output / "ledger", readiness["budget"])
    sources, results, winner_receipt = source_identity(), [], None
    runtime_identity = {k: readiness[k] for k in ("device", "device_name", "total_memory_bytes")}
    runtime_identity.update(torch="2.8.0+cu126", cuda="12.6", dtype="float32", amp=False, tf32=False,
        deterministic=True, whole_worker_seconds=WORKER_RESERVATION)
    write_json(args.output / "execution_spec.json", execution_spec())
    try:
        for raw_job in job_plan(args.stage):
            job = dict(raw_job)
            if job["stage"] == "power":
                if winner_receipt is None:
                    winner_receipt = freeze_october(results, args.output)
                winner = winner_receipt["winners"][job["model"]]
                job.update(trial=winner["trial"], fixed_epochs=winner["fixed_epochs"],
                    winner_receipt_sha256=sha(winner_receipt))
            attempt_id = sha({"job": job, "sources": sources, "runtime": runtime_identity})
            directory = args.output / "attempts" / attempt_id
            spec = {"version": VERSION, "job": job, "attempt_id": attempt_id,
                "output": str(directory), "data_root": str(args.data_root.resolve()),
                "source_roots": str(args.source_roots.resolve()), "source_sha256": sources,
                "runtime_identity": runtime_identity, "spec_sha256": sha(execution_spec())}
            existing = verified_result(directory, spec)
            if existing is not None:
                results.append(existing)
                continue
            if directory.exists():
                raise ValueError("Existing unfinished attempt requires reconciliation; no automatic retry")
            ledger.reserve(attempt_id, WORKER_RESERVATION)
            start = time.monotonic()
            directory.mkdir(parents=True, exist_ok=False)
            write_json(directory / "worker_spec.json", spec)
            print(json.dumps({"event": "new_fit_started", "stage": job["stage"], "model": job["model"],
                "seed": job["seed"], "attempt_id": attempt_id}), flush=True)
            with (directory / "worker.log").open("xb") as log:
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                    "--worker", str(directory / "worker_spec.json")], stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                code = wait_worker(process, ledger, attempt_id, start,
                    job["stage"] + "/" + job["model"] + "/" + str(job["seed"]))
            elapsed = time.monotonic() - start
            ledger.charge(attempt_id, elapsed, status="completed" if code == 0 else "failed")
            if code:
                raise RuntimeError("Worker failed/paused; inspect " + str(directory / "worker.log")
                    + "; no automatic retry was performed")
            result = verified_result(directory, spec)
            if result is None:
                raise RuntimeError("Worker exited without a verified terminal result")
            results.append(result)
            close_noise(results, args.output)
            print(json.dumps({"event": "fit_completed", "stage": job["stage"],
                "model": job["model"], "completed_this_plan": len(results)}), flush=True)
        power = [r for r in results if r["job"]["stage"] == "power"]
        if power:
            score_spec = {"version": VERSION, "power_results": power, "output": str(args.output),
                "data_root": str(args.data_root.resolve()), "source_roots": str(args.source_roots.resolve()),
                "source_sha256": sources, "runtime_identity": runtime_identity,
                "winner_receipt": winner_receipt, "spec_sha256": sha(execution_spec())}
            scoring_id = "score-" + sha(score_spec)
            if (args.output / "R_decision.json").exists():
                raise ValueError("Scoring already exists; verify its frozen bundle before any new invocation")
            ledger.reserve(scoring_id, WORKER_RESERVATION, metadata={"stage": "frozen_power_scoring_only"})
            started = time.monotonic()
            score_path = args.output / "power_score_spec.json"
            write_json(score_path, score_spec)
            with (args.output / "power_score_worker.log").open("xb") as stream:
                child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                    "--score-spec", str(score_path)], stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                code = wait_worker(child, ledger, scoring_id, started, "frozen_power_scoring_and_analysis")
            ledger.charge(scoring_id, time.monotonic() - started, status="completed" if code == 0 else "failed")
            if code:
                raise RuntimeError("Power scoring/analysis did not close; inspect power_score_worker.log")
        return 0
    finally:
        ledger.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-roots", type=Path)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--budget-evidence", type=Path)
    parser.add_argument("--stage", choices=("noise", "power", "all"), default="all")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--score-spec", type=Path)
    args = parser.parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if (args.execute or args.worker or args.score_spec) and not LIVE_PILOT_EXECUTION_ENABLED:
        print(json.dumps({"status": "prepared_execution_disabled", "training_started": False,
            "reason": "Current delivery uses existing results; optional pilot controller needs real-GPU acceptance and budget/cache integration review before activation"}))
        return 3
    if args.worker:
        run_worker(json.loads(args.worker.read_text()))
        return 0
    if args.score_spec:
        score_worker(json.loads(args.score_spec.read_text()))
        return 0
    if args.output is None or args.source_roots is None:
        parser.error("--output and --source-roots are required")
    try:
        readiness = preflight(args)
        if not readiness["ready"]:
            return 3
        return execute(args, readiness) if args.execute else 0
    except Exception as error:
        from e2_amended_pilot_budget import BudgetPause
        print(json.dumps({"status": "blocked_or_failed", "reason": str(error),
            "training_completion_claimed": False, "automatic_retry": False}, ensure_ascii=False), flush=True)
        return 3 if isinstance(error, BudgetPause) else 2


if __name__ == "__main__":
    raise SystemExit(main())
