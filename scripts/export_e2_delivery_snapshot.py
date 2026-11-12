#!/usr/bin/env python3
"""Export existing, replay-verified E2 selection metadata; never fit or score.

The output is an evidence inventory, not a common final benchmark. Prediction
vectors stay in their immutable source receipts. No market labels are loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path

from e2_amended_pilot_budget import (
    LEGACY_BRANCH, LEGACY_PLAN, OPTIMIZED_BRANCH, OPTIMIZED_PLAN,
    canonical, digest, reconcile_snapshots, replay_campaign,
)

VERSION = "e2-delivery-existing-evidence-v1"
COLUMNS = (
    "model", "origin", "stage", "candidate_id", "seed", "status",
    "optimizer_steps", "model_parameters", "charged_wall_seconds",
    "callback_wall_seconds", "epochs_completed", "best_epoch",
    "selected_state_sha256", "prepared_signature", "candidate_payload_sha256",
    "prediction_count", "prediction_date_start", "prediction_date_end",
    "job_id", "result_sha256", "source_path", "source_git_blob_sha",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_descriptor(path, relative):
    raw = Path(path).read_bytes()
    return {"path": relative, "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "git_blob_sha": hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()}


def finite_number(value, name, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid_nonnegative_" + name)
    if integer and int(value) != value:
        raise ValueError("invalid_integer_" + name)
    return value


def trial_row(record, descriptor):
    outcome = record["outcome"]
    use = outcome["resource_use"]
    identity = use["identity"]
    phase = use.get("phase_records", {}).get("supervised", {})
    dates = outcome.get("prediction_dates", [])
    # Only date/count metadata is exported. Predictions are neither scored nor
    # converted into market metrics by this exporter.
    if len(dates) != len(outcome.get("predictions_original", [])):
        raise ValueError("prediction_date_count_mismatch")
    if dates != sorted(set(dates)):
        raise ValueError("prediction_dates_not_unique_and_sorted")
    steps = finite_number(use["optimizer_steps"], "optimizer_steps", integer=True)
    if use.get("optimizer_steps_unknown") is not False:
        raise ValueError("unknown_optimizer_steps_require_reconciliation")
    if outcome.get("status") not in ("completed", "failed"):
        raise ValueError("unclassified_result_status")
    if outcome.get("status") == "failed" and not outcome.get("reason"):
        raise ValueError("failed_result_requires_reason")
    return {
        "model": identity["model"], "origin": identity["origin"],
        "stage": identity["stage"], "candidate_id": identity["candidate_id"],
        "seed": identity["seed"], "status": outcome["status"],
        "optimizer_steps": steps,
        "model_parameters": finite_number(use["model_parameters"], "model_parameters", integer=True),
        "charged_wall_seconds": finite_number(record["charged_wall_seconds"], "charged_wall_seconds"),
        "callback_wall_seconds": finite_number(record.get("callback_wall_seconds", 0), "callback_wall_seconds"),
        "epochs_completed": phase.get("epochs_completed"), "best_epoch": phase.get("best_epoch"),
        "selected_state_sha256": phase.get("selected_state_sha256"),
        "prepared_signature": identity.get("prepared_signature"),
        "candidate_payload_sha256": identity["payload_sha256"],
        "prediction_count": len(dates), "prediction_date_start": dates[0] if dates else None,
        "prediction_date_end": dates[-1] if dates else None,
        "job_id": record["job_id"], "result_sha256": record["sha256"],
        "source_path": descriptor["path"], "source_git_blob_sha": descriptor["git_blob_sha"],
    }


def summarize_rows(rows, selections, *, reused_rows=()):
    """Count actual records and validate each published local selection receipt."""
    ids = [row["job_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate_job_id")
    groups = defaultdict(list)
    for row in [*rows, *reused_rows]:
        groups[(row["model"], row["origin"])].append(row)
    closed = {}
    for selection in selections:
        key = (selection["model"], selection["origin"])
        if key in closed:
            raise ValueError("duplicate_selection")
        group = groups[key]
        # These historical HPO domains have 18 candidates × 2 seeds. A status
        # claiming closure must be supported by all its fit receipts.
        if selection.get("attempts") != 36 or len(group) != 36:
            raise ValueError("selection_missing_36_fit_receipts")
        counts = Counter(row["candidate_id"] for row in group)
        if len(counts) != 18 or set(counts.values()) != {2}:
            raise ValueError("selection_missing_18_paired_trials")
        seeds = selection["winner"]["trial"]["optimization_seeds"]
        if any(sorted(row["seed"] for row in group if row["candidate_id"] == candidate) != sorted(seeds)
               for candidate in counts):
            raise ValueError("selection_trial_missing_registered_seed_pair")
        winner = [row for row in group if row["candidate_id"] == selection["winner_id"]]
        if (len(winner) != 2 or sorted(row["seed"] for row in winner) != sorted(seeds)
                or any(row["status"] != "completed" for row in winner)):
            raise ValueError("selection_winner_missing_completed_seed_pair")
        closed[key] = selection
    coverage = []
    for key, group in sorted(groups.items()):
        selection = closed.get(key)
        coverage.append({
            "model": key[0], "origin": key[1], "available_fit_receipts": len(group),
            "new_fit_receipts": sum(row in rows for row in group),
            "legacy_reused_fit_receipts": sum(row in reused_rows for row in group),
            "completed_fit_receipts": sum(row["status"] == "completed" for row in group),
            "expected_hpo_fit_receipts": 36,
            "local_hpo_selection_closed": selection is not None,
            "winner_id": selection["winner_id"] if selection else None,
            "fixed_epochs": selection.get("fixed_epochs") if selection else None,
            "selection_signature": selection.get("selection_signature") if selection else None,
        })
    return {
        "new_fit_receipts": len(rows),
        "new_completed_fit_receipts": sum(row["status"] == "completed" for row in rows),
        "new_failed_fit_receipts": sum(row["status"] == "failed" for row in rows),
        "known_optimizer_steps_new_fits": sum(row["optimizer_steps"] for row in rows),
        "legacy_reused_fit_receipts_in_current_selection": len(reused_rows),
        "local_hpo_selections_closed": len(closed),
        "closed_selections_by_model": dict(sorted(Counter(key[0] for key in closed).items())),
        "coverage_by_model_origin": coverage,
    }


def _load_trials(directory, branch, head):
    rows, descriptors = [], []
    for path in sorted((directory / "results").glob("*.json")):
        relative = branch + "/" + path.relative_to(directory).as_posix()
        descriptor = file_descriptor(path, relative)
        descriptor["head_sha"] = head
        row = trial_row(read_json(path), descriptor)
        rows.append(row)
        descriptors.append({**descriptor, "job_id": row["job_id"], "result_sha256": row["result_sha256"]})
    return rows, descriptors


def export_delivery(evidence_root, output):
    root, output = Path(evidence_root), Path(output)
    report = read_json(root / "budget_evidence.json")
    heads = report["historical_heads"]
    replays, directories, public_journals = {}, {}, []
    for branch, plan in ((LEGACY_BRANCH, LEGACY_PLAN), (OPTIMIZED_BRANCH, OPTIMIZED_PLAN)):
        head = heads[branch]
        directory = root / "snapshots" / branch.rsplit("/", 1)[1] / head
        replay = replay_campaign(directory, branch=branch, head_sha=head,
                                 expected_plan_sha256=plan, verify_inventory=True)
        stored = read_json(root / (branch.rsplit("/", 1)[1] + "_replay.json"))
        if replay != stored:
            raise ValueError("stored_replay_changed")
        replays[branch], directories[branch] = replay, directory
        manifest = read_json(directory / "snapshot.json")
        events = [read_json(p) for p in sorted((directory / "events").glob("*.json"))]
        public_journals.append({
            "repository": "jassan-bit/DeepLearning_202630", "branch": branch,
            "head_sha": head, "plan_sha256": plan,
            "journal_replayed_and_merkle_verified": True,
            "snapshot_manifest_sha256": manifest["sha256"],
            "inventory_algorithm": manifest["inventory_algorithm"],
            "inventory_root_sha256": manifest["inventory_root_sha256"],
            "record_count": manifest["record_count"], "record_bytes": manifest["record_bytes"],
            "replay_sha256": replay["sha256"], "event_count": len(events),
            "last_event_sha256": events[-1]["sha256"], "last_event_utc": events[-1]["utc"],
            "status": replay["status"],
            "source_files": [file_descriptor(directory / name, branch + "/" + name)
                             for name in ("campaign.json", "snapshot.json", "status.json")],
        })
    reconstructed = reconcile_snapshots(replays[LEGACY_BRANCH], replays[OPTIMIZED_BRANCH], observed_heads=heads)
    if reconstructed != report:
        raise ValueError("budget_evidence_changed")
    rows, descriptors = _load_trials(directories[OPTIMIZED_BRANCH], OPTIMIZED_BRANCH, heads[OPTIMIZED_BRANCH])
    legacy_rows, legacy_descriptors = _load_trials(directories[LEGACY_BRANCH], LEGACY_BRANCH, heads[LEGACY_BRANCH])
    status = read_json(directories[OPTIMIZED_BRANCH] / "status.json")
    new_keys = {(row["model"], row["origin"]) for row in rows}
    required_keys = {(s["model"], s["origin"]) for s in status["completed_selections"]} - new_keys
    reused = [row for row in legacy_rows if (row["model"], row["origin"]) in required_keys]
    reused_ids = {row["job_id"] for row in reused}
    summary = summarize_rows(rows, status["completed_selections"], reused_rows=reused)
    if (summary["new_fit_receipts"] != status["completed_jobs"]
            or summary["new_failed_fit_receipts"] != status["failed_jobs"]
            or summary["known_optimizer_steps_new_fits"] != status["known_optimizer_steps"]
            or len(reused) != status["reused_jobs"]):
        raise ValueError("status_counts_disagree_with_receipts")
    historical = replays[LEGACY_BRANCH]["completed_jobs"] + len(rows)
    if status["total_available_unique_jobs"] != historical:
        raise ValueError("historical_unique_counter_disagrees")
    summary.update({
        "version": VERSION, "scope": "existing_historical_selection_evidence_only",
        "source_cutoff_utc": public_journals[-1]["last_event_utc"],
        "historical_result_receipts": historical,
        "historical_legacy_result_receipts": replays[LEGACY_BRANCH]["completed_jobs"],
        "historical_counter_is_not_current_recipe_benchmark_count": True,
        "benchmark_complete": False, "R": None, "k": None,
        "new_models_trained": False, "market_labels_loaded_by_exporter": False,
        "final_market_evaluation_performed": False,
        "campaign_status_at_source_cutoff": status["status"],
        "current_training_activity_verified": False,
        "training_activity_note": "A pinned, paused historical journal does not establish a live Colab process.",
        "budget": {k: report[k] for k in (
            "authorized_combined_hours", "authorized_combined_seconds", "historical_charged_seconds",
            "remaining_seconds_before_pilots", "budget_reset", "conservative_historical_charges_preserved")},
        "source_budget_evidence_sha256": report["sha256"],
        "source_heads": heads,
        "fit_artifacts": {
            "prediction_vectors_preserved_in_immutable_source_receipts": True,
            "prediction_vectors_copied_into_public_snapshot": False,
            "epoch_loss_curves_available_in_selection_receipts": False,
            "metadata_exported": ["epochs_completed", "best_epoch", "optimizer_steps", "model_parameters",
                                  "charged_wall_seconds", "selected_state_sha256", "prediction_date_coverage"],
            "checkpoint_weights_available_from_selection_receipts": False,
        },
        "limitations": [
            "Local historical HPO selections are not final refits or final-test scores.",
            "The 1802 legacy receipts include historical and superseded recipes; they cannot all count toward the amended catalog.",
            "No scores from old SAE, TimesNet, TimeMixer or PatchTST recipes are transferred to the newly restored bodies.",
            "The current sixteen-system common benchmark, NAS, R and k remain unclosed.",
        ],
    })
    rows.sort(key=lambda row: (row["origin"], row["model"], row["candidate_id"], row["seed"]))
    csv_stream = io.StringIO(newline="")
    writer = csv.DictWriter(csv_stream, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    csv_bytes = csv_stream.getvalue().encode()
    summary["trials_csv_sha256"] = hashlib.sha256(csv_bytes).hexdigest()
    summary["sha256"] = digest(summary)
    index = {
        "version": VERSION, "journals": public_journals,
        "result_sources_new_fits": descriptors,
        "result_sources_reused_for_local_closures": [d for d in legacy_descriptors if d["job_id"] in reused_ids],
        "selection_snapshot_sha256": summary["sha256"],
        "trials_csv_sha256": summary["trials_csv_sha256"],
        "proof_scope": "Full local journal replay and byte-level Merkle inventory against the pinned source manifests; no assertion of a newer remote head.",
    }
    index["sha256"] = digest(index)
    output.mkdir(parents=True, exist_ok=True)
    (output / "trials.csv").write_bytes(csv_bytes)
    (output / "selection_snapshot.json").write_bytes(canonical(summary) + b"\n")
    (output / "source_index.json").write_bytes(canonical(index) + b"\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = export_delivery(args.evidence_root, args.output)
    print(json.dumps({k: report[k] for k in (
        "new_fit_receipts", "historical_result_receipts", "known_optimizer_steps_new_fits",
        "local_hpo_selections_closed", "benchmark_complete", "R", "k", "sha256")}, sort_keys=True))


if __name__ == "__main__":
    main()
