"""Read-only data and raw-HPO cache audit for the six-slot E2 amendment.

The audit never chooses a winner, fits, opens the 608-date evaluation, retries an
attempt, or restores weights. Historical validation predictions may be reused
only after their input signatures and the new worker runtime are verified.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path

from e2_amended_pilot_budget import (
    LEGACY_BRANCH, LEGACY_PLAN, OPTIMIZED_BRANCH, OPTIMIZED_PLAN,
    digest, reconcile_snapshots, replay_campaign,
)

VERSION = "e2-six-slot-cache-audit-v1"
ANCHORS = ("2024-01-01", "2024-07-01", "2024-10-01")
SLOTS = (5, 7, 8, 9, 10, 15)
SEED = 271201
UNMODIFIED = frozenset(("FCN-1D", "ResNet-1D", "Inception-1D", "TSMixer",
                       "iTransformer", "MLP16", "LSTM32", "GRU32"))
RESTORED = frozenset(("LSTM-autoencoder con predictor", "TimesNet", "TimeMixer", "PatchTST"))
NEURAL = ("FCN-1D", "ResNet-1D", "Inception-1D", "LSTM-autoencoder con predictor",
          "TimesNet", "TSMixer", "TiDE", "TimeMixer", "PatchTST", "iTransformer",
          "MLP16", "LSTM32", "GRU32")
PROTOCOL_PATH = "experiments/dl_r2_cv_20261104/protocol.json"
PROTOCOL_SHA = "e8da0f0098acd24a58664b83d0ba499b43aa9d597a36c52da6b3325c2c3c7bec"
DATA_SOURCES = {
    "results/eda_development/daily_market_development.csv": "bf6bb5160af924aef6a15c3684aeae932df3017e8639888bb7e91ce787900171",
    "results/eda_development/daily_volatility_development.csv": "20b96374a659eb87dadfa68aa3d6b6b5fb6bea5b10eebe6f2c186ad69861844a",
}
RUNTIME_KEYS = ("device", "device_name", "torch", "torch_cuda", "cudnn", "total_memory_bytes",
                "effective_memory_limit_bytes", "cpu_threads", "interop_threads", "dtype",
                "amp", "tf32", "deterministic", "cuda_kernel_probe_passed")
RUNTIME_FACT_KEYS = ("device", "device_name", "torch", "torch_cuda", "cudnn",
                     "total_memory_bytes", "effective_memory_limit_bytes", "cuda_kernel_probe_passed")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_payload(model, trial):
    return {"model": model, "trial": trial, "architecture": None, "stage": "hpo"}


def candidate_sha(model, trial):
    # This is the frozen CandidateSpec's encoding, distinct from ledger JSON.
    return hashlib.sha256(json.dumps(candidate_payload(model, trial), sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def _regular(root, relative):
    root = Path(root).absolute()
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("regular_in_root_source_required")
    if not path.is_file():
        raise ValueError("source_missing:" + str(relative))
    return path


def verify_development_data(repo_root, *, expected_sources=None, protocol_path=PROTOCOL_PATH,
                            protocol_sha=PROTOCOL_SHA):
    """Verify bytes of the established development-only derivatives, not cells."""
    expected_sources = DATA_SOURCES if expected_sources is None else expected_sources
    protocol = _regular(repo_root, protocol_path)
    if file_sha(protocol) != protocol_sha:
        raise ValueError("development_source_protocol_hash_changed")
    registered = read_json(protocol).get("sources", {})
    records = []
    for relative, expected in expected_sources.items():
        if registered.get(relative) != expected:
            raise ValueError("development_protocol_source_binding_changed")
        path = _regular(repo_root, relative)
        actual = file_sha(path)
        if actual != expected:
            raise ValueError("development_derivative_hash_changed:" + relative)
        records.append({"path": relative, "sha256": actual, "bytes": path.stat().st_size})
    return {"status": "development_bytes_verified", "protocol_path": protocol_path,
            "protocol_sha256": protocol_sha, "sources": records,
            "numerical_cells_decoded_by_hash_audit": False,
            "evaluation_608_dates_opened": False}


def build_development_context(repo_root):
    """Reconstruct frozen HPO input signatures from 2023--2024 development only.

    Both historical CUDA controllers call prepare_b(origin, origin). The global
    three-fold scoring policy changes selection, not any of these input tensors.
    Torch/pandas are imported only for this explicit data verification path.
    """
    from run_e2_cuda_search import prepare_panel
    from e2_temporal_training import prepare_b
    panel, data_receipt = prepare_panel(repo_root)
    folds = {}
    for anchor in ANCHORS:
        prepared = prepare_b(panel, anchor, anchor)
        folds[anchor] = {"prepared_signature": prepared.signature,
            "training_signature": prepared.inner_train.signature,
            "validation_signature": prepared.inner_validation.signature,
            "train_scaler_sha256": digest(prepared.inner_train.scaler.record()),
            "validation_dates": [str(d.date()) for d in prepared.inner_validation.dates],
            "train_last_label": str(prepared.inner_train.dates[-1].date()),
            "validation_first_label": str(prepared.inner_validation.dates[0].date()),
            "validation_last_label": str(prepared.inner_validation.dates[-1].date()),
            "outer_end_used_for_historical_input_identity": anchor}
    return {"scope": "development_signatures_only_no_model_fit", "folds": folds,
            "data_receipt_sha256": digest(data_receipt), "numerical_admission_end": "2024-12-30",
            "evaluation_608_dates_opened": False, "models_constructed": False,
            "development_values_decoded_for_signatures": True}


def validate_domains(domains):
    """Six shared slots, not a ranking/selection copied from the old winners."""
    if isinstance(domains, dict) and "models" in domains:
        domains = domains["models"]
    if not isinstance(domains, dict) or set(domains) != set(NEURAL):
        raise ValueError("exact_thirteen_model_domains_required")
    for model, rows in domains.items():
        if not isinstance(rows, list) or len(rows) != 6:
            raise ValueError("six_slots_per_model_required")
        for index, (row, parent_index) in enumerate(zip(rows, SLOTS)):
            trial = row.get("trial")
            if (row.get("configuration_id") != "C%02d" % (index + 1)
                    or row.get("parent_trial_index") != parent_index
                    or not isinstance(trial, dict)
                    or row.get("parent_trial_id") != trial.get("trial_id")
                    or row.get("optimization_seed", SEED) != SEED):
                raise ValueError("six_slot_definition_changed")
            if model == "TiDE" and row.get("execution_allowed") is not False:
                raise ValueError("TiDE_architecture_gate_must_remain_explicit")
    return domains


def _day(value):
    if len(value) == 10:
        return date.fromisoformat(value)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0) or parsed.time().replace(tzinfo=None).isoformat() != "00:00:00":
        raise ValueError("validation_dates_must_be_UTC_midnights")
    return parsed.date()


def inspect_record(record, plan, *, model, anchor, slot, context=None, runtime_profile=None):
    """Check one outcome's raw fit identity; no winner or final score is created."""
    raw = {k: v for k, v in record.items() if k != "sha256"}
    if digest(raw) != record.get("sha256") or digest(record.get("outcome")) != record.get("outcome_sha256"):
        raise ValueError("cached_record_hash_changed")
    outcome, trial = record["outcome"], slot["trial"]
    use = outcome.get("resource_use", {})
    identity = use.get("identity", {})
    expected = {"model": model, "origin": anchor, "candidate_id": "hpo:" + trial["trial_id"],
                "payload_sha256": candidate_sha(model, trial), "seed": SEED, "stage": "hpo"}
    if any(identity.get(k) != v for k, v in expected.items()):
        raise ValueError("cached_candidate_identity_changed")
    if (outcome.get("status") != "completed" or outcome.get("fatal") is not False
            or use.get("optimizer_steps_unknown") is not False
            or use.get("partial_optimizer_update_may_have_occurred") is not False
            or type(use.get("optimizer_steps")) is not int or use["optimizer_steps"] <= 0):
        raise ValueError("cached_fit_not_successfully_closed")
    dates, values = outcome.get("prediction_dates"), outcome.get("predictions_original")
    wanted_start, wanted_end = date.fromisoformat(anchor) - timedelta(days=90), date.fromisoformat(anchor) - timedelta(days=8)
    wanted = [wanted_start + timedelta(days=i) for i in range((wanted_end - wanted_start).days + 1)]
    if not isinstance(dates, list) or [_day(d) for d in dates] != wanted:
        raise ValueError("cached_validation_calendar_changed")
    if (not isinstance(values, list) or len(values) != len(dates)
            or any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in values)):
        raise ValueError("cached_validation_predictions_invalid")
    phase = use.get("phase_records", {}).get("supervised", {})
    if (type(phase.get("best_epoch")) is not int or not 1 <= phase["best_epoch"] <= 200
            or type(phase.get("epochs_completed")) is not int
            or not phase["best_epoch"] <= phase["epochs_completed"] <= 200):
        raise ValueError("cached_training_epochs_invalid")
    checks = {"candidate_payload": True, "recipe_identity": True,
              "complete_validation_calendar": True, "known_completed_updates": True,
              "prepared_inputs": False, "job_payload": False, "worker_runtime": False}
    if context is not None:
        if identity.get("prepared_signature") != context.get("prepared_signature"):
            raise ValueError("cached_prepared_signature_changed")
        job = {**expected, "candidate": candidate_payload(model, trial),
            "prepared_signature": context["prepared_signature"],
            "training_signature": context["training_signature"],
            "validation_signature": context["validation_signature"],
            "source_commit": plan["source_commit"], "protocol_sha256": plan["protocol_sha256"],
            "hardware_sha256": digest(plan["hardware_binding"])}
        if digest(job) != record.get("job_payload_sha256"):
            raise ValueError("cached_full_job_payload_changed")
        if digest({"plan_sha256": record["plan_sha256"], "job": job}) != record.get("job_id"):
            raise ValueError("cached_job_id_changed")
        checks.update(prepared_inputs=True, job_payload=True)
    runtime_facts_match, fit_policy_verified, fit_policy_match = None, False, False
    if runtime_profile is not None:
        actual = runtime_profile.get("runtime", runtime_profile.get("profile", runtime_profile))
        runtime_facts_match = all(actual.get(k) == plan["hardware_binding"].get(k) for k in RUNTIME_FACT_KEYS)
        # Observed kernel defaults are not the policy inside a fitting context.
        # A matrix probe cannot attest to the executed fit policy of 13 bodies.
        fit_policy_verified = runtime_profile.get("fit_execution_policy_verified") is True
        fit_policy_match = (isinstance(plan.get("execution"), dict)
                            and runtime_profile.get("fit_execution_policy") == plan["execution"])
        checks["worker_runtime"] = runtime_facts_match and fit_policy_verified and fit_policy_match
    ready = all(checks.values())
    state = "reusable_validation_metadata" if ready else (
        "reusable_after_runtime_check" if checks["prepared_inputs"] else "requires_prepared_and_runtime_checks")
    if runtime_profile is not None and not checks["worker_runtime"]:
        state = ("reusable_after_execution_policy_attestation" if runtime_facts_match and not fit_policy_verified
                 else "runtime_mismatch_not_reusable")
    return {"model_id": model, "origin": anchor, "configuration_id": slot["configuration_id"],
        "parent_trial_id": trial["trial_id"], "seed": SEED, "state": state,
        "reuse_approved": ready, "checks": checks, "historical_prepared_signature": identity.get("prepared_signature"),
        "runtime_facts_match": runtime_facts_match, "fit_execution_policy_attested": fit_policy_verified,
        "fit_execution_policy_match": fit_policy_match,
        "job_id": record["job_id"], "result_sha256": record["sha256"],
        "candidate_payload_sha256": expected["payload_sha256"],
        "validation_prediction_sha256": digest(values), "validation_dates_sha256": digest(dates),
        "validation_prediction_count": len(values), "best_epoch": phase["best_epoch"],
        "scope": "raw_inner_validation_HPO_outcome_only", "weights_reusable": False,
        "old_local_winner_reused": False, "final_metrics_reused": False}


def audit(reconciliation_root, repo_root, domains, *, prepared_context=None, runtime_profile=None):
    domains = validate_domains(domains)
    data = verify_development_data(repo_root)
    root = Path(reconciliation_root)
    reconciliation = read_json(root / "budget_evidence.json")
    heads = reconciliation["historical_heads"]
    all_records = defaultdict(list)
    replays, sources = {}, {}
    for branch, plan_sha in ((LEGACY_BRANCH, LEGACY_PLAN), (OPTIMIZED_BRANCH, OPTIMIZED_PLAN)):
        directory = root / "snapshots" / branch.rsplit("/", 1)[-1] / heads[branch]
        replay = replay_campaign(directory, branch=branch, head_sha=heads[branch],
                                 expected_plan_sha256=plan_sha, verify_inventory=True)
        if replay != read_json(root / (branch.rsplit("/", 1)[-1] + "_replay.json")):
            raise ValueError("stored_journal_replay_changed")
        replays[branch] = replay
        plan = read_json(directory / "campaign.json")["plan"]
        changed = [relative for relative, expected in plan["source_sha256"].items()
                   if not (Path(repo_root) / relative).is_file()
                   or file_sha(_regular(repo_root, relative)) != expected]
        sources[branch] = {"source_commit": plan["source_commit"], "file_count": len(plan["source_sha256"]),
            "all_frozen_source_bytes_match": not changed, "changed_or_missing": changed,
            "source_binding_sha256": digest(plan["source_sha256"]), "journal_replay_sha256": replay["sha256"],
            "snapshot_manifest_sha256": replay["snapshot_manifest_sha256"]}
        for path in sorted((directory / "results").glob("*.json")):
            record = read_json(path)
            identity = record.get("outcome", {}).get("resource_use", {}).get("identity", {})
            key = (identity.get("model"), identity.get("origin"), identity.get("candidate_id"), identity.get("seed"))
            all_records[key].append((record, plan, branch, path))
    if reconcile_snapshots(replays[LEGACY_BRANCH], replays[OPTIMIZED_BRANCH], observed_heads=heads) != reconciliation:
        raise ValueError("reconciliation_report_changed")
    folds = {} if prepared_context is None else prepared_context.get("folds", prepared_context)
    rows = []
    for model, slots in domains.items():
        for anchor in ANCHORS:
            for slot in slots:
                row = {"model_id": model, "origin": anchor, "configuration_id": slot["configuration_id"],
                    "parent_trial_id": slot["trial"]["trial_id"], "seed": SEED,
                    "reuse_approved": False, "weights_reusable": False}
                if model == "TiDE":
                    rows.append({**row, "state": "blocked_architecture_adjudication"})
                    continue
                matches = all_records.get((model, anchor, "hpo:" + slot["trial"]["trial_id"], SEED), [])
                if not matches:
                    rows.append({**row, "state": "no_current_recipe_outcome"})
                    continue
                if len(matches) != 1:
                    raise ValueError("duplicate_physical_results_for_reduced_fit")
                record, plan, branch, path = matches[0]
                if model not in UNMODIFIED:
                    rows.append({**row, "state": "restored_recipe_requires_new_fit"})
                    continue
                checked = inspect_record(record, plan, model=model, anchor=anchor, slot=slot,
                    context=folds.get(anchor), runtime_profile=runtime_profile)
                if not sources[branch]["all_frozen_source_bytes_match"]:
                    checked.update(state="source_mismatch_not_reusable", reuse_approved=False)
                rows.append({**checked, "source_branch": branch, "source_head": heads[branch],
                    "source_path": branch + "/results/" + path.name, "file_sha256": file_sha(path)})
    matched = [row for row in rows if "result_sha256" in row]
    by_model = {model: {"planned_HPO_fits": 18,
        "matching_historical_raw_outcomes": sum(row["model_id"] == model for row in matched),
        "reuse_approved": sum(row["model_id"] == model and row["reuse_approved"] for row in rows)} for model in domains}
    report = {"version": VERSION, "scope": "read_only_six_slot_data_cache_audit",
        "domains_sha256": digest(domains), "data": data, "historical_heads": heads,
        "budget_evidence_sha256": reconciliation["sha256"], "sources": sources,
        "planned_HPO_fits": len(rows), "matching_historical_raw_outcomes": len(matched),
        "reuse_approved": sum(row["reuse_approved"] for row in rows),
        "state_counts": dict(sorted(Counter(row["state"] for row in rows).items())),
        "by_model": by_model, "fits": rows,
        "prepared_context_sha256": None if prepared_context is None else digest(prepared_context),
        "development_values_decoded_for_signatures": (False if prepared_context is None
            else prepared_context.get("development_values_decoded_for_signatures", False)),
        "runtime_profile_supplied": runtime_profile is not None,
        "cache_weights_reusable": 0, "old_local_winners_reused": 0,
        "final_benchmark_scores_created": False, "new_market_training_executed": False,
        "evaluation_608_dates_opened": False,
        "limitations": ["Matching historical outcomes are raw HPO validation predictions, not final weights or test metrics.",
            "GPU runtime compatibility must be checked on the actual Colab worker before approving mixed-cache execution.",
            "Observed kernel flags and a synthetic CUDA matrix probe do not attest the policy actually applied during fits.",
            "All six slots and all three folds must close before choosing one global recipe; old local winners are not adopted.",
            "TiDE remains blocked until its architecture policy is resolved; old NAS variants cannot impersonate restored bodies."]}
    report["sha256"] = digest(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconciliation-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--domains", type=Path, required=True)
    parser.add_argument("--verify-prepared", action="store_true")
    parser.add_argument("--runtime-profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    domains = read_json(args.domains)
    context = build_development_context(args.repo_root) if args.verify_prepared else None
    report = audit(args.reconciliation_root, args.repo_root, domains,
        prepared_context=context, runtime_profile=None if args.runtime_profile is None else read_json(args.runtime_profile))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if read_json(args.output) != report:
            raise FileExistsError("Existing immutable cache audit differs")
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"output": str(args.output), "matching_historical_raw_outcomes": report["matching_historical_raw_outcomes"],
        "reuse_approved": report["reuse_approved"], "state_counts": report["state_counts"], "new_market_training_executed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
