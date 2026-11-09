"""Verify completed development receipts and publish a reproducible pilot report.

This command fits no predictors. It refuses incomplete paired data for the
conditional variance analysis and keeps unexecuted full-search gates open.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from e2_pilot_decisions import (decide_k, decide_R, budget_report,
                                conditional_power_analysis)
from run_e2_development_pilot import read_events, registration, write_json_new
from run_e2_reference_variance import (REGISTRATION_COMMIT, REGISTRATION_SHA256,
                                      MODELS, SEEDS, safe_load_checkpoint,
                                      replay_vectors, median_epoch_count)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "book_e2/_static/search_protocol_20261109/protocol.json"
AMENDMENT_COMMIT = "d511a8de3b9e9d5fa559d761ce9f2ac732921a80"
AMENDMENT_SHA256 = "025fe220405050a5cb8246bf22638e8bebc301093fcc571e878ebbdffa17ebc1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_checks(receipt):
    sources = receipt.get("source_sha256", receipt.get("source_sha256_before", {})) or {}
    for name, expected in sources.items():
        if sha(ROOT / name) != expected:
            raise ValueError("Source changed after fitting: " + name)
    return len(sources)


def build(resources, variance, registered, output):
    resources, variance, output = map(Path, (resources, variance, output))
    registered = Path(registered)
    registration(registered)
    if sha(registered) != REGISTRATION_SHA256:
        raise ValueError("Use the exact published pre-fit registration")
    reg = json.loads(registered.read_text())
    amendment_path = ROOT / "experiments/e2_development_pilot_20261109/analysis_amendment.json"
    if not amendment_path.is_file():
        amendment_path = ROOT / "analysis_amendment.json"
    if sha(amendment_path) != AMENDMENT_SHA256:
        raise ValueError("Use the exact secondary-analysis amendment published before held replay")
    protocol = json.loads(PROTOCOL.read_text())
    if sha(PROTOCOL) != reg["source_protocol_sha256"]:
        raise ValueError("The source protocol changed after registration")
    resource = json.loads((resources / "summary.json").read_text())
    reference = json.loads((variance / "summary.json").read_text())
    if (not resource["all_requested_jobs_attempted"]
            or not resource["full_registered_job_set"] or len(resource["results"]) != 15):
        raise ValueError("The report requires all15 registered resource attempts")
    if reference["registration_file_sha256"] != REGISTRATION_SHA256:
        raise ValueError("Variance receipts refer to a different registration")
    read_events(resources / "events.jsonl")
    table, checked_sources, ledgers, checkpoints = [], 0, {}, []
    for row in resource["results"]:
        worker = row.get("worker_profile") or {}
        checked_sources += _source_checks(worker)
        before = worker.get("source_sha256_before", {})
        after = worker.get("source_sha256_after", {})
        if before and after and any(after.get(p) != digest for p, digest in before.items()):
            raise ValueError("Source changed during a resource attempt")
        receipt = worker.get("fit_receipt")
        if receipt and receipt["protocol_file_sha256"] != reg["source_protocol_sha256"]:
            raise ValueError("Successful resource receipt used a different protocol")
        events_path = resources / ("worker-%02d" % row["index"]) / "events.jsonl"
        events = read_events(events_path)
        for event in events["events"]:
            for name, digest in event.get("source_sha256", {}).items():
                if sha(ROOT / name) != digest:
                    raise ValueError("Source changed after a censored resource attempt")
        ledgers["resources/worker-%02d" % row["index"]] = events
        table.append({"model": row["model"], "case": row["case"],
                      "status": row["status"], "error_code": row["error_code"],
                      "elapsed_seconds": row["elapsed_wall_seconds"],
                      "sampled_peak_rss_bytes": row["sampled_peak_process_group_rss_bytes"],
                      "optimizer_steps": row["optimizer_steps"],
                      "unknown_steps": row["optimizer_steps_censored"],
                      "phases": {name: {key: stage[key] for key in
                                 ("best_epoch", "epochs_completed", "optimizer_steps", "stop_reason")}
                                 for name, stage in worker.get("phase_receipts", {}).items()}})

    matrices, included_names, truth, dates = [], [], None, None
    counts = {"selection": 0, "refit": 0, "completed_optimizer_steps": 0,
              "attempts": 0, "failed_attempts": 0, "failed_known_optimizer_steps": 0,
              "unknown_optimizer_step_workers": 0, "safe_checkpoints": 0}
    all_complete = reference["paired_3x5_complete"]
    for name in MODELS:
        model = reference["models"][name]
        include = all(len(model[phase]) == 5
                      and {row["seed"] for row in model[phase]} == set(SEEDS)
                      and all(row["status"] == "completed" for row in model[phase])
                      for phase in ("selection", "refit"))
        if include and median_epoch_count(model["selection"]) != reference["fixed_epoch_counts"][name]:
            raise ValueError("Frozen epochs differ from the five-seed selection median")
        rows_by_seed = {}
        for phase in ("selection", "refit"):
            for row in model[phase]:
                counts["attempts"] += 1
                worker = row.get("worker_profile") or {}
                checked_sources += _source_checks(worker)
                if row["status"] != "completed":
                    counts["failed_attempts"] += 1
                    steps = row.get("optimizer_steps")
                    if type(steps) is int and steps >= 0:
                        counts["failed_known_optimizer_steps"] += steps
                    else:
                        counts["unknown_optimizer_step_workers"] += 1
                    continue
                counts[phase] += 1
                counts["completed_optimizer_steps"] += worker["optimizer_steps"]
                if worker["registration_file_sha256"] != REGISTRATION_SHA256:
                    raise ValueError("Worker registration mismatch")
                if not worker["fresh_optimizer"]:
                    raise ValueError("Worker reused optimizer state")
                worker_dir = (variance / row["worker_result_file"]).parent
                checkpoint = worker["checkpoint"]
                path = worker_dir / checkpoint["relative_path"]
                safe_load_checkpoint(path, checkpoint["sha256"], checkpoint["state_sha256"])
                counts["safe_checkpoints"] += 1
                checkpoints.append({"model": name, "seed": row["seed"], "phase": phase,
                                    **checkpoint})
                expected_end = "2024-06-25" if phase == "selection" else "2024-10-31"
                if not worker["training_split"]["dates"][-1].startswith(expected_end):
                    raise ValueError("Training support crossed its registered cutoff")
                if phase == "refit":
                    vectors = worker["evaluation"]
                    checked = replay_vectors(worker_dir / "replay/events.jsonl",
                        name + "/seed" + str(row["seed"]), vectors["dates"])
                    if checked != vectors:
                        raise ValueError("Replayed predictions differ from worker receipt")
                    if dates is None:
                        dates, truth = vectors["dates"], vectors["truths"]
                    if vectors["dates"] != dates or vectors["truths"] != truth:
                        raise ValueError("Paired models/seeds do not share dates and truth")
                    if len(dates) != 53 or worker["validation_split"] is not None:
                        raise ValueError("Refit used validation or incomplete pilot support")
                    if worker["fixed_epoch_count"] != reference["fixed_epoch_counts"][name]:
                        raise ValueError("Refit epochs changed after freezing")
                    rows_by_seed[row["seed"]] = vectors["predictions"]
                    ledgers[name + "/seed" + str(row["seed"])] = [
                        json.loads(line) for line in (worker_dir / "replay/events.jsonl").read_text().splitlines()]
        if include:
            if set(rows_by_seed) != set(SEEDS):
                raise ValueError("Do not discard or replace a missing seed")
            matrices.append([rows_by_seed[seed] for seed in SEEDS])
            included_names.append(name)
    if all_complete and (counts["selection"] != 15 or counts["refit"] != 15):
        raise ValueError("A complete paired reference pilot needs all30 distinct fits")
    counts["known_optimizer_steps"] = counts["completed_optimizer_steps"] + counts["failed_known_optimizer_steps"]
    counts["blocked_refits_not_attempted"] = sum(5-len(model["refit"]) for model in reference["models"].values())
    power = conditional_power_analysis(np.asarray(matrices), truth, protocol["contract"]["delta"],
        m=153, model_names=included_names, seed_ids=list(SEEDS), dates=dates) if len(included_names) >= 2 else {
            "status": "blocked_incomplete_paired_reference_pilot", "achieved_final_power": None,
            "reference_pilot_R_recommendation": None}
    power.update({"primary_three_model_pilot_complete": all_complete,
                  "analysis_role": "primary_reference_planning" if all_complete else "secondary_completely_observed_models_only",
                  "included_models": included_names,
                  "unobserved_or_incomplete_models": [name for name in MODELS if name not in included_names],
                  "analysis_amendment_commit": AMENDMENT_COMMIT,
                  "analysis_amendment_sha256": AMENDMENT_SHA256,
                  "incomplete_model_variance_estimated": False})
    decisions = {"k": decide_k(protocol, []),
                 "original_postselection_R_gate": decide_R(protocol, power_runs=[], reference_analysis=power),
                 "reference_preselection_power": power,
                 "budget_scenarios": [budget_report(protocol, 40, R) for R in (6, 10, 20, 30)],
                 "final_seed_count_set": False, "full_noise_pilot_fits_executed": 0,
                 "full_HPO_or_NAS_executed": False}
    if output.exists():
        raise FileExistsError("Report directory must be new")
    output.mkdir(parents=True)
    artifacts = {}
    for name, value in {"resources_full.json.gz": resource, "variance_full.json.gz": reference,
                        "ledgers.json.gz": ledgers}.items():
        raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()
        compressed = gzip.compress(raw, mtime=0)
        (output / name).write_bytes(compressed)
        artifacts[name] = {"bytes": len(compressed), "sha256": sha(output / name),
                           "uncompressed_bytes": len(raw),
                           "uncompressed_sha256": hashlib.sha256(raw).hexdigest()}
    write_json_new(output / "decisions.json", decisions)
    write_json_new(output / "resources_table.json", table)
    write_json_new(output / "checkpoints.json", checkpoints)
    for name in ("decisions.json", "resources_table.json", "checkpoints.json"):
        artifacts[name] = {"sha256": sha(output / name), "bytes": (output / name).stat().st_size}
    summary = {"version": "e2-real-development-pilot-report-v1", "created_at_utc":
               datetime.now(timezone.utc).isoformat(), "registration_commit": REGISTRATION_COMMIT,
               "registration_sha256": REGISTRATION_SHA256, "protocol_sha256": sha(PROTOCOL),
               "analysis_amendment_commit": AMENDMENT_COMMIT,
               "analysis_amendment_sha256": AMENDMENT_SHA256,
               "remote_provider": "Codex Cloud", "device": "cpu", "gpu_visible": False,
               "development_market_training_executed": True, "final_market_data_read": False,
               "resource_attempts": len(table),
               "resource_successes": sum(row["status"] == "success" for row in table),
               "resource_known_optimizer_steps": resource["known_optimizer_steps_sum"],
               "resource_unknown_optimizer_step_workers": resource["unknown_optimizer_step_workers"],
               "resource_measured_worker_sum_seconds": resource["elapsed_workers_sum_seconds"],
               "reference_pilot": counts, "reference_paired_3x5_complete": all_complete,
               "conditional_analysis_models": included_names,
               "reference_fixed_epochs": reference["fixed_epoch_counts"],
               "reference_measured_worker_sum_seconds": sum(row["elapsed_wall_seconds"]
                   for model in reference["models"].values() for phase in model.values() for row in phase),
               "checkpoints_safely_reloaded_for_report": counts["safe_checkpoints"],
               "source_hash_observations_checked": checked_sources,
               "k_operational": decisions["k"]["k_recommendation"],
               "reference_R_recommendation": power["reference_pilot_R_recommendation"],
               "final_R": None, "achieved_final_power": None,
               "full_HPO_or_NAS_executed": False, "full_campaign_hours_estimate": None,
               "currency_cost_estimate": None, "artifacts": artifacts,
               "weights_published": False,
               "limits": ["Reference-only pilot is not the complete scientific campaign.",
                          "Measured pilot time is not a validated full-campaign extrapolation.",
                          "No failed attempt is replaced and unknown steps are not set to zero.",
                          "Inference checkpoint hashes are published; synthetic/development weights are not hosted."]}
    write_json_new(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--variance", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.resources, args.variance, args.registration, args.output),
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
