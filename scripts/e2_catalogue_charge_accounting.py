#!/usr/bin/env python3
"""Audit mutually exclusive observed wall charges for the recorded E2 HPO grid.

This reads metadata only. Historical reused-fit charges and current-worker
charges cover different registered opportunities. Their descriptive sum mixes
timing boundaries and hardware; it is not pure training time, GPU-active time,
money, energy, or the entire historical selection campaign. Unknown elapsed
time remains null and its conservative budget allocation remains separate.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
HPO = Path("book_e2/_static/e2_corrections_20261115/hpo")
PHASE_SUMMARY = Path("book_e2/_static/e2_scientific_closure_20261116/hpo_cost_reconciliation/summary.json")
VERSION = "e2-catalogue-charge-accounting-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode())


def nonnegative(value):
    result = float(value)
    require(math.isfinite(result) and result >= 0, "Invalid wall-charge seconds")
    return result


class Inputs:
    def __init__(self):
        self.files = {}

    def read(self, relative, expected=None):
        relative = Path(relative)
        require(not relative.is_absolute() and ".." not in relative.parts,
                "Input must be repository-relative")
        path = ROOT / relative
        require(path.is_file() and not path.is_symlink(), f"Missing regular input: {relative}")
        raw = path.read_bytes()
        actual = sha(raw)
        require(expected is None or actual == expected, f"Source SHA changed: {relative}")
        record = {"sha256": actual, "bytes": len(raw)}
        key = relative.as_posix()
        require(key not in self.files or self.files[key] == record, f"Input changed: {relative}")
        self.files[key] = record
        return raw

    def json(self, relative, expected=None):
        return json.loads(self.read(relative, expected))

    def reverify(self):
        for path, record in list(self.files.items()):
            self.read(path, record["sha256"])


def logical_key(row):
    return row["model"], row["trial_id"], row["fold_origin"], str(row["seed"])


def validate_exclusive_records(rows):
    """Reject duplicate opportunities/physical receipts and invented unknown time."""
    attempts, logical, physical, payloads = set(), set(), set(), set()
    for row in rows:
        require(row["attempt_id"] not in attempts, "Duplicate logical attempt")
        attempts.add(row["attempt_id"])
        key = logical_key(row)
        require(key not in logical, "Duplicate registered opportunity")
        logical.add(key)
        identity = row["physical_job_id"]
        require(identity not in physical, "Physical job charged more than once, including across sources")
        physical.add(identity)
        pair = row["physical_job_id"], row["physical_payload_sha256"]
        require(pair not in payloads, "Duplicate physical job/payload receipt")
        payloads.add(pair)
        measured = row["observed_wall_charge_seconds"]
        allocation = row["unmeasured_conservative_allocation_seconds"]
        if row["status"] == "outcome_unknown":
            require(measured is None, "Unknown elapsed time cannot become measured cost")
            require(allocation is not None and nonnegative(allocation) > 0,
                    "Unknown outcome must retain its separate allocation")
        else:
            require(measured is not None and allocation is None,
                    "Known observed cost must be exclusive of an unknown allocation")
            nonnegative(measured)


def verify_ledger(events):
    previous = "0" * 64
    charges, reservations = {}, {}
    for sequence, event in enumerate(events):
        require(event["sequence"] == sequence and event["previous_sha256"] == previous,
                "Campaign ledger chronology changed")
        require(digest({k: v for k, v in event.items() if k != "sha256"}) == event["sha256"],
                "Campaign ledger event hash changed")
        previous = event["sha256"]
        if event["kind"] in ("charged", "reserved"):
            target = charges if event["kind"] == "charged" else reservations
            require(event["attempt_id"] not in target, "Duplicate campaign reservation/charge")
            target[event["attempt_id"]] = event
    return charges, reservations


def source_job_matches(row, job):
    expected = {"logical_job_id": row["attempt_id"], "model_id": row["model"],
                "configuration_id": row["configuration_id"], "configuration_sha256": row["configuration_sha256"],
                "fold_origin": row["fold_origin"], "seed": int(row["seed"]),
                "parent_trial_id": row["trial_id"], "stage": "hpo"}
    require(all(job.get(k) == v for k, v in expected.items()), "Source job differs from registered opportunity")
    require(digest({k: v for k, v in job.items() if k != "logical_job_id"}) == row["attempt_id"],
            "Logical job hash changed")


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def reconcile(output):
    started, clock = datetime.now(timezone.utc).isoformat(), time.monotonic()
    require(not output.exists(), "Preserve prior evidence: use a new output directory")
    inputs = Inputs()
    script_sha = sha(inputs.read(Path(__file__).resolve().relative_to(ROOT)))
    # Record the source implementations which define the two timing boundaries.
    for name in ("e2_search_session.py", "e2_reduced_supervision.py", "e2_reduced_budget.py",
                 "run_e2_reduced_selection.py", "audit_e2_reduced_cache.py"):
        inputs.read(Path("scripts") / name)
    source_rows = list(csv.DictReader(io.StringIO(inputs.read(HPO / "opportunities_234.csv").decode())))
    require(len(source_rows) == 234, "Expected the frozen 234-opportunity catalogue")
    ledger_raw = inputs.read(HPO / "sources/metadata/shared_budget/pilot_budget.jsonl")
    ledger_sha = sha(ledger_raw)
    charges, reservations = verify_ledger([json.loads(line) for line in ledger_raw.decode().splitlines()])
    phase_summary = inputs.json(PHASE_SUMMARY)
    rows, incidents = [], []
    for source in source_rows:
        attempt = source["attempt_id"]
        folder = Path(source["selection_row_path"]).parent
        for path_field, sha_field in (("selection_row_path", "selection_row_sha256"),
                ("result_path", "result_sha256"), ("cached_source_path", "cached_source_sha256"),
                ("terminal_failure_receipt_path", "terminal_failure_receipt_sha256")):
            if source[path_field]:
                inputs.read(HPO / "sources" / source[path_field], source[sha_field])
        require(source["budget_ledger_sha256"] == ledger_sha, "Wrong source ledger binding")
        selection = inputs.json(HPO / "sources" / source["selection_row_path"], source["selection_row_sha256"])
        cached = bool(source["cached_source_path"])
        current = bool(source["campaign_budget_charge_seconds"])
        require(cached != current, "Each opportunity needs exactly one original-cache or current charge source")
        row = {key: source[key] for key in ("model", "configuration_id", "trial_id", "fold_origin", "seed",
                "attempt_id", "configuration_sha256", "status", "selection_row_path", "selection_row_sha256",
                "result_path", "result_sha256", "cached_source_path", "cached_source_sha256")}
        row.update(physical_job_id=None, physical_payload_sha256=None, charge_source=None,
                   source_field=None, timing_scope=None, observed_wall_charge_seconds=None,
                   unmeasured_conservative_allocation_seconds=None, nested_callback_seconds_not_added=None,
                   original_started_event_sha256=None, campaign_charge_event_sha256=None,
                   device_recorded_in_current_request=None, terminal_failure_class=None,
                   optimizer_steps_exact_at_failure=None)
        if cached:
            require(source["status"] == "completed" and source["cached_outcome_reused"] == "True",
                    "Cached outcome must be a completed reuse")
            require(attempt not in charges and attempt not in reservations,
                    "Reused historical fit also received a new campaign charge")
            result = inputs.json(HPO / "sources" / source["result_path"], source["result_sha256"])
            source_job_matches(source, result["job"])
            require(result.get("cache_reuse") is True and result.get("training_performed") is False
                    and result.get("optimizer_steps") == 0, "Reuse performed a new fit")
            receipt = inputs.json(HPO / "sources" / source["cached_source_path"], source["cached_source_sha256"])
            require(digest({k: v for k, v in receipt.items() if k != "sha256"}) == receipt["sha256"],
                    "Original cached envelope self-hash changed")
            require(digest(receipt["outcome"]) == receipt["outcome_sha256"], "Cached outcome hash changed")
            identity = receipt["outcome"]["resource_use"]["identity"]
            expected = {"model": source["model"], "origin": source["fold_origin"], "seed": int(source["seed"]),
                        "stage": "hpo", "candidate_id": "hpo:" + source["trial_id"]}
            require(all(identity.get(k) == v for k, v in expected.items()), "Original cache maps to another opportunity")
            candidate = {"model": source["model"], "trial": result["job"]["trial"], "architecture": None, "stage": "hpo"}
            candidate_sha = sha(json.dumps(candidate, sort_keys=True, allow_nan=False).encode())
            require(identity["payload_sha256"] == candidate_sha, "Original cache candidate payload changed")
            charged, callback = nonnegative(receipt["charged_wall_seconds"]), nonnegative(receipt["callback_wall_seconds"])
            require(charged == nonnegative(source["original_cached_fit_charge_seconds"])
                    and callback == nonnegative(source["original_cached_fit_callback_seconds"])
                    and callback <= charged, "Cached wall-charge or nested callback differs")
            require(selection["run_receipt_sha256"] == source["result_sha256"], "Cached selection/result binding changed")
            row.update(physical_job_id=receipt["job_id"], physical_payload_sha256=receipt["job_payload_sha256"],
                       charge_source="historical_original_cached_fit", source_field="cached_source_record.charged_wall_seconds",
                       timing_scope="search-session wall charge from prior charge point through callback completion; includes prior gap",
                       observed_wall_charge_seconds=charged, nested_callback_seconds_not_added=callback,
                       original_started_event_sha256=receipt["started_event_sha256"])
        else:
            charge, reservation = charges[attempt], reservations[attempt]
            require(charge["sha256"] == source["campaign_charge_event_sha256"]
                    and nonnegative(charge["elapsed_seconds"]) == nonnegative(source["campaign_budget_charge_seconds"]),
                    "Current charge differs from source ledger")
            spec_sha = reservation["metadata"]["worker_spec_sha256"]
            require(charge["resource_use"]["worker_spec_sha256"] == spec_sha, "Charge/reservation worker identity differs")
            row.update(physical_job_id=attempt, physical_payload_sha256=spec_sha,
                       charge_source="current_campaign_worker", source_field="campaign_charged_event.elapsed_seconds",
                       timing_scope="reduced-budget elapsed wall from reservation/supervision to terminal charge; max with supervised worker timer",
                       campaign_charge_event_sha256=charge["sha256"],
                       device_recorded_in_current_request=source["current_device"] or None)
            if source["status"] == "outcome_unknown":
                require(source["campaign_charge_basis"] == "conservative_reserved_allocation_not_measured_elapsed"
                        and charge["resource_use"].get("elapsed_seconds_measured") is False
                        and charge["resource_use"].get("measured_elapsed_seconds") is None,
                        "Unknown reserved allocation cannot be classified as observed elapsed")
                row.update(unmeasured_conservative_allocation_seconds=nonnegative(charge["elapsed_seconds"]),
                           terminal_failure_class="expired_runtime_outcome_unknown")
            else:
                require(source["campaign_charge_basis"] == "measured_closed_campaign_charge", "Unexpected measured-charge basis")
                row["observed_wall_charge_seconds"] = nonnegative(charge["elapsed_seconds"])
                if source["status"] == "completed":
                    result = inputs.json(HPO / "sources" / source["result_path"], source["result_sha256"])
                    source_job_matches(source, result["job"])
                    require(result.get("training_performed") is True and not result.get("cache_reuse")
                            and result["worker_spec_sha256"] == spec_sha,
                            "Current successful receipt is not a distinct new worker fit")
                    require(charge["resource_use"]["result_sha256"] == source["result_sha256"]
                            and selection["run_receipt_sha256"] == source["result_sha256"],
                            "Current result/charge/selection binding changed")
                    request_raw = inputs.read(HPO / "sources" / folder / "request.json")
                    require(digest(json.loads(request_raw)) == result["request_sha256"],
                            "Current worker canonical request hash changed")
                else:
                    require(source["status"] == "known_failed", "Unexpected current disposition")
                    failure = inputs.json(HPO / "sources" / source["terminal_failure_receipt_path"],
                                          source["terminal_failure_receipt_sha256"])
                    source_job_matches(source, failure["job"])
                    require(failure["worker_spec_sha256"] == spec_sha
                            and failure["supervision"]["budget_receipt"] == charge,
                            "Terminal failure has another physical worker or charge")
                    steps = selection["resource_use"]["optimizer_steps"]
                    row["optimizer_steps_exact_at_failure"] = steps
                    row["terminal_failure_class"] = "pre_model_setup_failure" if steps == 0 else "synchronized_phase_timeout"
                    require((steps == 0 and source["fold_origin"] == "2024-01-01")
                            or (steps == 749 and source["fold_origin"] == "2024-07-01"),
                            "TimesNet failure classification changed")
            if source["status"] != "completed":
                incidents.append(row)
        rows.append(row)
    validate_exclusive_records(rows)
    measured = [r for r in rows if r["observed_wall_charge_seconds"] is not None]
    cached_rows = [r for r in rows if r["charge_source"] == "historical_original_cached_fit"]
    current_rows = [r for r in measured if r["charge_source"] == "current_campaign_worker"]
    unknown = [r for r in rows if r["status"] == "outcome_unknown"]
    require((len(rows), len(measured), len(cached_rows), len(current_rows), len(unknown)) == (234, 233, 61, 172, 1),
            "Unexpected exclusive accounting coverage")
    total = math.fsum(r["observed_wall_charge_seconds"] for r in measured)
    require(abs(total - 25833.89891848399) < 1e-8, "Observed catalogue charge total changed")
    require(math.fsum(r["unmeasured_conservative_allocation_seconds"] for r in unknown) == 1860,
            "Unknown allocation changed")
    model_rows = []
    for model in sorted({r["model"] for r in rows}):
        subset = [r for r in rows if r["model"] == model]
        model_rows.append({"model": model, "registered_opportunities": len(subset),
            "completed": sum(r["status"] == "completed" for r in subset),
            "known_failed": sum(r["status"] == "known_failed" for r in subset),
            "outcome_unknown": sum(r["status"] == "outcome_unknown" for r in subset),
            "observed_charge_receipts": sum(r["observed_wall_charge_seconds"] is not None for r in subset),
            "historical_cached_receipts": sum(r["charge_source"] == "historical_original_cached_fit" for r in subset),
            "observed_wall_charge_seconds_mixed_scopes": math.fsum(r["observed_wall_charge_seconds"] for r in subset if r["observed_wall_charge_seconds"] is not None),
            "unmeasured_conservative_allocation_seconds": math.fsum(r["unmeasured_conservative_allocation_seconds"] for r in subset if r["unmeasured_conservative_allocation_seconds"] is not None)})
    summary = {"version": VERSION, "scope": "recorded reduced HPO catalogue only; heterogeneous observed wall charges",
        "registered_opportunities": 234, "completed_outcomes": 231, "known_failed_outcomes": 2, "unknown_outcomes": 1,
        "current_campaign_charge_receipts": 173, "current_campaign_observed_charge_receipts": 172,
        "original_cached_charge_receipts": 61, "observed_charge_receipts_union": 233,
        "physical_jobs_double_counted": 0,
        "current_campaign_observed_wall_charge_seconds": math.fsum(r["observed_wall_charge_seconds"] for r in current_rows),
        "historical_original_cached_wall_charge_seconds": math.fsum(r["observed_wall_charge_seconds"] for r in cached_rows),
        "recorded_catalogue_observed_wall_charge_seconds_mixed_scopes": total,
        "recorded_catalogue_observed_wall_charge_hours_mixed_scopes": total / 3600,
        "unmeasured_conservative_allocation_seconds_not_added": 1860.0,
        "unknown_actual_elapsed_seconds": None,
        "phase_breakdown_available_jobs_separate_scope": phase_summary["jobs_with_phase_receipts_union"],
        "completed_jobs_without_phase_breakdown_separate_scope": phase_summary["completed_jobs_without_phase_durations"],
        "two_known_failures": [{k: row[k] for k in ("model", "attempt_id", "fold_origin", "terminal_failure_class",
                              "optimizer_steps_exact_at_failure", "observed_wall_charge_seconds")}
                              for row in incidents if row["status"] == "known_failed"],
        "limitations": ["Different historical search-session and reduced-budget timer boundaries and mixed hardware; descriptive accounting, not comparable pure training speed.",
            "Only the 234 recorded reduced-catalogue opportunities; earlier discarded trials, outer controller overhead, source recovery, final fitting, CPU supplements and all project-wide costs are outside this sum.",
            "One lost worker has no observed elapsed duration; its 1860-second conservative allocation is not added to observed time and is not an upper bound on its actual wall time.",
            "The 51 missing phase breakdowns do not remove existing original wall-charge receipts; phase totals remain a distinct incomplete breakdown.",
            "No GPU-active time, monetary cost, energy, complete historical selection cost, or end-to-end campaign duration is inferred."],
        "do_not_add_to_this_total": ["phase elapsed times", "cached callbacks", "failed-worker nested elapsed", "unknown reserved allocation", "new CPU reproduction cost or other campaigns without separate scope accounting"]}
    inputs.reverify()
    output.mkdir(parents=True)
    write_csv(output / "catalogue_charge_opportunities_234.csv", rows)
    write_csv(output / "catalogue_charge_by_model_13.csv", model_rows)
    write_json(output / "summary.json", summary)
    write_json(output / "inputs_manifest.json", {"version": VERSION, "files": inputs.files})
    write_json(output / "execution_receipt.json", {"version": VERSION, "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_metadata_audit_seconds": time.monotonic() - clock,
        "script_sha256": script_sha, "source_files_reverified": len(inputs.files),
        "input_manifest_sha256": sha((output / "inputs_manifest.json").read_bytes()),
        "new_model_fits": 0, "new_model_inference": False, "new_HPO": False, "NAS_executed": False,
        "statistical_tests": 0, "budget_phase": "catalogue_charge_accounting_metadata_audit",
        "budget_ledger": "book_e2/_static/e2_scientific_closure_20261116/compute_ledger.jsonl"})
    write_json(output / "manifest.json", {"version": VERSION, "files": {p.name: {"bytes": p.stat().st_size,
        "sha256": sha(p.read_bytes())} for p in sorted(output.iterdir()) if p.is_file()}})
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reconcile(args.output)


if __name__ == "__main__":
    main()
