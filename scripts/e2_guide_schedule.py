"""Explicit three-origin execution amendment; immutable A3 fits remain unchanged.

This is a scheduler, not a replacement model implementation or recovery tool.
Existing job identities and outcomes are reused within the original ledger.
An unresolved interrupted attempt is NEVER retried by this launcher.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

AMENDMENT = "E2-GUIDE-SCHEDULE-20261111-v1"
SOURCE = "564ba4762f84a800b02fa4c734f5752a169f5223"
PLAN_SHA256 = "4eae2d2ff9ef6f83a2f9eb0dc5ba647208d695e1665a8e522d812fed67893bda"
REGISTRATION = "78c3c5b8ca042a347aaa80d642f3affd8b5911e8697d8747a2de915d1c3194a9"
ORIGINS = ("2024-01-01", "2024-10-01", "2025-01-01")
BRANCH = "campaigns/e2-cuda-selection-v1"


def summary():
    return {"amendment": AMENDMENT, "origins": list(ORIGINS),
            "hpo_attempts": 14*36*3, "nas1_attempts": 6*1000*3,
            "nas2_attempts_upper_bound": 6*41*5*3,
            "selection_attempts_upper_bound": 23202,
            "previous_upper_bound": 85074,
            "reduction_fraction": 1-3/11,
            "models_changed": False, "nas_domain_changed": False,
            "epochs_changed": False, "k": 40, "final_R": None,
            "training_started": False,
            "scope": "amended_three_origin_neural_selection_not_final_benchmark"}


def validate_plan(plan, digest):
    if digest(plan) != PLAN_SHA256:
        raise ValueError("amendment_requires_exact_original_campaign_plan")
    if plan["source_commit"] != SOURCE or not set(ORIGINS) <= set(plan["origins"]):
        raise ValueError("amendment_source_or_origins_mismatch")
    if plan["remote_storage"]["branch"] != BRANCH:
        raise ValueError("amendment_requires_existing_campaign")


def run_selection(plan, session, panel, *, worker, mirror=None, progress=None):
    """Schedule a declared subset, while hashing each job with the FULL old plan.

    In particular, do not change plan['origins']: doing so would invalidate all
    original job IDs and silently retrain the already completed first origin.
    """
    import run_e2_cuda_search as runner
    import e2_training_models as registry
    from e2_training_bindings import run_neural_selection
    from e2_temporal_training import prepare_b

    validate_plan(plan, runner.admission.digest)
    selections = [] if progress is None else progress

    def sync():
        if mirror is not None and session.pending_record_paths:
            paths = session.pending_record_paths
            mirror.push(session.directory, paths=paths,
                        message="Confirm three-origin amended scientific search state")
            session.mark_records_synced(paths)

    for origin in ORIGINS:
        prepared = prepare_b(panel, origin, origin)
        for model in plan["models"]:
            class DurableAdapter:
                wall = 7200
                short_wall = 1800
                memory = 12*1024**3
                full_epochs = 200
                short_epochs = 50
                patience = 20

                def fit_candidate(self, candidate, seed, train, validation):
                    if (train.signature != prepared.inner_train.signature
                            or validation.signature != prepared.inner_validation.signature):
                        raise ValueError("selection_callback_split_changed")
                    job = runner.job_payload(plan, prepared, model, candidate, seed)
                    identity = runner.admission.digest({
                        "plan_sha256": runner.admission.digest(plan), "job": job})
                    phases = 3 if model == "LSTM-autoencoder con predictor" else 1
                    seconds = (1800 if job["stage"] == "nas1" else 7200)*phases+300
                    cached = session.get_result(identity, job) is not None
                    if mirror is not None and not cached:
                        mirror.check_capacity(required_requests=32)

                    def call(deadline):
                        sync()
                        return worker(job, deadline)

                    value = session.run_job(identity, job, call, reservation_seconds=seconds)
                    sync()
                    if not cached:
                        print(json.dumps({"amendment": AMENDMENT, "model": model,
                              "origin": job["origin"], "stage": job["stage"],
                              "status": value["status"], "seed": seed}), flush=True)
                    return runner.restore_trial(value)

            protocol = registry.protocol()
            architectures = (runner.admission.planner.enumerate_architectures(protocol, model)
                             if model in registry.MODEL_IDS else [])
            selected, _ = run_neural_selection(
                prepared, model, DurableAdapter(), hpo_trials=registry.registered_trials(model),
                architectures=architectures, hpo_seeds=plan["seed_pools"]["hpo"],
                phase1_seeds=plan["seed_pools"]["nas1"],
                phase2_seeds=plan["seed_pools"]["nas2"], k=40,
                delta=protocol["contract"]["delta"])
            selections.append({"model": model, "origin": origin,
                "winner_id": selected.winner.candidate_id, "winner": selected.winner.payload,
                "fixed_epochs": selected.fixed_epochs, "selection_signature": selected.signature,
                "attempts": selected.attempts, "selection_rule": selected.selection_rule})
    return selections


def amended_report(original, *args, **kwargs):
    state = original(*args, **kwargs)
    state["scope"] = "amended_three_origin_neural_selection_20261111_v1"
    state["coverage"] = {**state["coverage"], "selected_origins": 3, "partial_origins": True}
    if state["status"] == "completed_selected_registered_domains":
        state["status"] = "completed_amended_three_origin_selection"
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--frozen-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--source-root-itransformer", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(summary(), ensure_ascii=False, indent=2), flush=True)
    if args.plan_only:
        return 0
    if any(v is None for v in (args.frozen_root, args.output, args.data_root,
                               args.source_root_itransformer)):
        parser.error("execution requires frozen root, output, data and author-source paths")
    root = args.frozen_root.resolve()
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
    if head != SOURCE:
        raise ValueError("use_original_A3_checkout_not_main")
    # Refuse a second controller in this runtime. Never terminate another job.
    processes = subprocess.run(["ps", "-eo", "pid=,args="], check=True,
                               capture_output=True, text=True).stdout.splitlines()
    import os
    own = os.getpid()
    for row in processes:
        pid, _, command = row.strip().partition(" ")
        if pid.isdigit() and int(pid) != own and (
                "run_e2_cuda_search.py" in command or "e2_guide_schedule.py --execute" in command
                or "run_e2_cuda_recovery_v3.py" in command):
            raise RuntimeError("another_search_process_is_active_do_not_launch_twice")
    sys.path.insert(0, str(root/"scripts"))
    runner = importlib.import_module("run_e2_cuda_search")
    runner.run_selection = run_selection
    original_report = runner.report
    runner.report = lambda *a, **kw: amended_report(original_report, *a, **kw)
    return runner.main(["--execute", "--source-commit", SOURCE,
        "--expected-registration-sha256", REGISTRATION,
        "--output", str(args.output), "--data-root", str(args.data_root),
        "--source-root-itransformer", str(args.source_root_itransformer),
        "--session-hours", "8", "--campaign-hours", "24", "--storage-gib", "10",
        "--github-branch", BRANCH, "--github-prefix", BRANCH])


if __name__ == "__main__":
    raise SystemExit(main())
