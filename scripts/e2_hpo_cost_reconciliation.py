#!/usr/bin/env python3
"""Reconcile recorded E2 costs without fits, inference or statistical analysis.

The historical 170-job report is retained. Ten selected cached jobs contribute
additional recorded phase durations; 29 selected jobs overlap and are not added
again. The resulting union mixes explicitly labelled historical timing scopes.
It is not an end-to-end campaign timer or a new training reproduction.
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
PHASES = Path("book_e2/_static/e2_efficiency_closure_20261113/phase_resources.csv")
REGISTER = Path("book_e2/_static/e2_data_training_uncertainty_current_v1/model_training_register.csv")
VERSION = "e2-hpo-cost-reconciliation-v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def number(value):
    result = float(value)
    require(math.isfinite(result) and result >= 0, "Invalid nonnegative recorded cost")
    return result


def total(rows, field):
    values = [number(row[field]) for row in rows if row.get(field) not in (None, "")]
    return math.fsum(values) if values else None


class Inputs:
    def __init__(self):
        self.records = {}

    def read(self, relative, expected=None):
        relative = Path(relative)
        require(not relative.is_absolute() and ".." not in relative.parts,
                "Inputs must be repository-relative")
        path = ROOT / relative
        require(path.is_file() and not path.is_symlink(), f"Missing regular input: {relative}")
        raw = path.read_bytes()
        digest = sha(raw)
        require(expected is None or digest == expected, f"Input SHA mismatch: {relative}")
        key = relative.as_posix()
        record = {"bytes": len(raw), "sha256": digest}
        require(key not in self.records or self.records[key] == record, f"Input changed: {relative}")
        self.records[key] = record
        return raw

    def csv(self, relative):
        return list(csv.DictReader(io.StringIO(self.read(relative).decode("utf-8"))))

    def json(self, relative, expected=None):
        return json.loads(self.read(relative, expected))

    def reverify(self):
        for relative, record in list(self.records.items()):
            require(len(self.read(relative, record["sha256"])) == record["bytes"],
                    f"Input size changed: {relative}")


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def job_key(row):
    return row["model"], row["trial_id"], row["fold_origin"], str(row["seed"])


def reconcile(output):
    started, clock = datetime.now(timezone.utc).isoformat(), time.monotonic()
    require(not output.exists(), "Use a NEW output directory; existing evidence is preserved")
    inputs = Inputs()
    script_relative = Path(__file__).resolve().relative_to(ROOT)
    script_sha = sha(inputs.read(script_relative))
    opportunities = inputs.csv(HPO / "opportunities_234.csv")
    historical = inputs.json(HPO / "summary.json")
    inputs.json(HPO / "analysis_plan.json")
    inputs.json(HPO / "execution_receipt.json")
    phases = inputs.csv(PHASES)
    inputs.json(PHASES.parent / "source_evidence.json")
    register = inputs.csv(REGISTER)
    require(len(register) == 13 and len(opportunities) == 234, "Unexpected registered design")
    require(len({job_key(row) for row in opportunities}) == 234, "Duplicate HPO job identity")
    require(len({row["attempt_id"] for row in opportunities}) == 234, "Duplicate HPO attempt")
    models = [row["model"] for row in register]
    require(set(models) == {row["model"] for row in opportunities}, "Model catalog differs")
    ledger_raw = inputs.read(HPO / "sources/metadata/shared_budget/pilot_budget.jsonl")
    ledger_sha = sha(ledger_raw)
    charges = {}
    for line in ledger_raw.decode().splitlines():
        event = json.loads(line)
        if event["kind"] == "charged":
            require(event["attempt_id"] not in charges, "Duplicate campaign charge identity")
            charges[event["attempt_id"]] = event
    selected = {}
    for row in phases:
        if row["stage"] == "selected_hpo":
            selected.setdefault(job_key(row), []).append(row)
    require(len(selected) == 39, "Expected 39 selected HPO jobs, not 45 jobs")
    identity_index = {job_key(row): row for row in opportunities}
    require(set(selected) <= set(identity_index), "Selected phase has no registered HPO identity")
    for group in selected.values():
        require(len({row["phase"] for row in group}) == len(group), "Duplicate phase within selected job")
    known, missing, cached_identities = [], [], set()
    old_jobs = additions = overlaps = 0
    for row in opportunities:
        require(row["model"] in models, "Unexpected model")
        for path_field, sha_field in (
                ("selection_row_path", "selection_row_sha256"), ("result_path", "result_sha256"),
                ("cached_source_path", "cached_source_sha256"),
                ("terminal_failure_receipt_path", "terminal_failure_receipt_sha256")):
            if row[path_field]:
                inputs.read(HPO / "sources" / row[path_field], row[sha_field])
        require(row["budget_ledger_sha256"] == ledger_sha, "Wrong budget-ledger binding")
        if row["campaign_budget_charge_seconds"]:
            event = charges.get(row["attempt_id"])
            require(event is not None and event["sha256"] == row["campaign_charge_event_sha256"],
                    "Campaign charge identity differs")
            require(number(event["elapsed_seconds"]) == number(row["campaign_budget_charge_seconds"]),
                    "Campaign charge duration differs")
        cached = None
        if row["cached_source_path"]:
            cached = inputs.json(HPO / "sources" / row["cached_source_path"], row["cached_source_sha256"])
            identity = cached["job_id"], cached["job_payload_sha256"]
            require(identity not in cached_identities, "Historical cached fit counted twice")
            cached_identities.add(identity)
            for column, field in (("original_cached_fit_charge_seconds", "charged_wall_seconds"),
                                  ("original_cached_fit_callback_seconds", "callback_wall_seconds")):
                require(number(row[column]) == number(cached[field]), "Cached duration differs from source")
        groups = selected.get(job_key(row), [])
        selected_seconds = math.fsum(number(item["elapsed_seconds"]) for item in groups) if groups else None
        measured = number(row["measured_phase_elapsed_seconds"]) if row["measured_phase_elapsed_seconds"] else None
        if measured is not None:
            old_jobs += 1
            result = inputs.json(HPO / "sources" / row["result_path"], row["result_sha256"])
            source_phases = result["phase_records"]
            require(source_phases and all(x.get("elapsed_seconds") is not None for x in source_phases.values()),
                    "Incomplete original phase breakdown")
            require(abs(math.fsum(number(x["elapsed_seconds"]) for x in source_phases.values()) - measured) < 1e-9,
                    "Phase duration differs from its result receipt")
            if groups:
                require(abs(selected_seconds - measured) < 1e-9, "Overlapping phase receipts disagree")
                overlaps += 1
            duration, origin = measured, "historical_hpo_report_170_jobs"
            timing_scope = "recorded_result_phase_records; see original source receipts"
        elif groups:
            require(cached is not None and row["cached_outcome_reused"] == "True",
                    "Additional selected job must bind a historical reused fit")
            additions += 1
            duration, origin = selected_seconds, "additional_selected_cached_phase_receipt"
            timing_scope = ";".join(sorted({item["phase_timing_scope"] for item in groups}))
        else:
            duration = None
        identity = {key: row[key] for key in ("model", "configuration_id", "trial_id", "fold_origin", "seed",
                                             "attempt_id", "configuration_sha256", "selection_row_path", "selection_row_sha256")}
        if duration is not None:
            require(row["status"] == "completed", "A failed/unknown job acquired an invented fit duration")
            known.append({**identity, "recorded_phase_seconds": duration, "duration_source": origin,
                          "timing_scope": timing_scope, "selected_phase_table": PHASES.as_posix() if groups else "",
                          "selected_source_attempt_id": groups[0]["source_attempt_id"] if groups else "",
                          "result_path": row["result_path"], "result_sha256": row["result_sha256"],
                          "cached_source_path": row["cached_source_path"], "cached_source_sha256": row["cached_source_sha256"]})
        elif cached is not None:
            missing.append({**identity, "historical_job_id": cached["job_id"],
                            "historical_job_payload_sha256": cached["job_payload_sha256"],
                            "cached_source_path": row["cached_source_path"], "cached_source_sha256": row["cached_source_sha256"],
                            "original_cached_fit_charge_seconds": row["original_cached_fit_charge_seconds"],
                            "original_cached_fit_callback_seconds": row["original_cached_fit_callback_seconds"],
                            "missing_measurement": "phase-level elapsed for this historical reused job; wall charge is available"})
    require((old_jobs, additions, overlaps, len(known), len(missing), len(cached_identities)) == (170, 10, 29, 180, 51, 61),
            "Unexpected union design; preserve evidence and review instead of silently changing scope")
    union_seconds = math.fsum(row["recorded_phase_seconds"] for row in known)
    require(abs(union_seconds - 20784.46532408601) < 1e-8, "Unexpected recorded union sum")
    require(abs(total(opportunities, "measured_phase_elapsed_seconds") - historical["known_training_phase_seconds"]) < 1e-8,
            "Historical 170-job report no longer agrees")
    table = []
    for model in models:
        rows = [row for row in opportunities if row["model"] == model]
        costs = [row for row in known if row["model"] == model]
        final = [row for row in phases if row["stage"] == "final" and row["model"] == model]
        require(len({row["seed"] for row in final}) == 6, "Missing final seed timing")
        require(len({(row["seed"], row["phase"]) for row in final}) == len(final), "Duplicate final phase")
        actual_charges = [row for row in rows if row["campaign_charge_basis"] == "measured_closed_campaign_charge"]
        allocations = [row for row in rows if row["campaign_charge_basis"] == "conservative_reserved_allocation_not_measured_elapsed"]
        table.append({"model": model, "planned_HPO_jobs": len(rows),
                      "completed_HPO_jobs": sum(row["status"] == "completed" for row in rows),
                      "known_failed_jobs": sum(row["status"] == "known_failed" for row in rows),
                      "unknown_outcome_jobs": sum(row["status"] == "outcome_unknown" for row in rows),
                      "jobs_with_original_phase_receipts": sum(bool(row["measured_phase_elapsed_seconds"]) for row in rows),
                      "additional_selected_cached_jobs": sum(row["duration_source"] == "additional_selected_cached_phase_receipt" for row in costs),
                      "jobs_with_phase_receipts_union": len(costs),
                      "original_170_report_phase_seconds": total(rows, "measured_phase_elapsed_seconds"),
                      "additional_cached_phase_seconds": total([row for row in costs if row["duration_source"] == "additional_selected_cached_phase_receipt"], "recorded_phase_seconds"),
                      "recorded_phase_seconds_union_mixed_scopes": total(costs, "recorded_phase_seconds"),
                      "cached_jobs_without_phase_durations": sum(row["model"] == model for row in missing),
                      "current_measured_charge_receipts": len(actual_charges),
                      "current_campaign_measured_charge_seconds": total(actual_charges, "campaign_budget_charge_seconds"),
                      "conservative_allocation_count": len(allocations),
                      "conservative_allocation_seconds_not_measured": total(allocations, "campaign_budget_charge_seconds"),
                      "historical_cached_charge_receipts": sum(bool(row["original_cached_fit_charge_seconds"]) for row in rows),
                      "historical_cached_fit_charge_seconds": total(rows, "original_cached_fit_charge_seconds"),
                      "historical_cached_callback_seconds_nested_in_charge": total(rows, "original_cached_fit_callback_seconds"),
                      "failed_worker_wall_seconds_nested_in_campaign_charge": total(rows, "failed_worker_wall_seconds"),
                      "original_final_fit_jobs": 6, "original_final_phase_rows": len(final),
                      "original_final_recorded_phase_seconds": total(final, "elapsed_seconds"),
                      "full_selection_wall_seconds": None, "full_end_to_end_reproduction_seconds": None})
    summary = {
        "version": VERSION, "scope": "Recorded metadata reconciliation; no new fit, inference or statistical calculation",
        "opportunities": 234, "completed": 231, "known_failed": 2, "outcome_unknown": 1,
        "old_jobs_with_phase_receipts": old_jobs, "selected_jobs": len(selected),
        "selected_overlapping_jobs_not_added_again": overlaps, "additional_selected_cached_jobs": additions,
        "jobs_with_phase_receipts_union": len(known), "completed_jobs_without_phase_durations": len(missing),
        "original_170_report_phase_seconds": historical["known_training_phase_seconds"],
        "additional_selected_cached_phase_seconds": total([row for row in known if row["duration_source"] == "additional_selected_cached_phase_receipt"], "recorded_phase_seconds"),
        "recorded_phase_seconds_union_mixed_scopes": union_seconds,
        "current_campaign_measured_charge_seconds": total([row for row in opportunities if row["campaign_charge_basis"] == "measured_closed_campaign_charge"], "campaign_budget_charge_seconds"),
        "conservative_reserved_allocation_seconds_not_measured": total([row for row in opportunities if row["campaign_charge_basis"] == "conservative_reserved_allocation_not_measured_elapsed"], "campaign_budget_charge_seconds"),
        "historical_cached_fit_charge_seconds": total(opportunities, "original_cached_fit_charge_seconds"),
        "historical_cached_fit_charge_unique_jobs": len(cached_identities),
        "original_final_neural_jobs": 78, "original_final_phase_rows": sum(row["original_final_phase_rows"] for row in table),
        "original_final_recorded_phase_seconds": math.fsum(row["original_final_recorded_phase_seconds"] for row in table),
        "classical_resources": {model: {"recorded_phase_seconds": None, "full_selection_seconds": None,
                                        "reason": "No comparable recorded resource timing; not zero"}
                                for model in ("Persistence", "SeasonalNaive", "ARIMA")},
        "count_unit": "HPO job/opportunity, not individual optimization phases; SAE has three phases per job",
        "timing_union_scope": "Non-overlapping recorded job phase durations, with distinct historical checkpoint policies and potentially different hardware. Descriptive accounted duration, not a homogeneous GPU speed comparison.",
        "do_not_add": ["Union phases + selected_hpo phases: 29 jobs overlap",
                       "Recorded phase elapsed + campaign worker/budget charges: nested boundaries",
                       "Historical cached callback + cached charged wall: nested boundaries",
                       "Failed worker elapsed + campaign charge: nested boundaries",
                       "Conservative lost-runtime allocation as measured training time",
                       "Any of these fields as a complete GPU-active, energy, monetary or end-to-end campaign total"],
        "remaining": ["Phase breakdown unavailable for 51 historical reused jobs, whose charged/callback walls remain recorded",
                      "Lost TSMixer C02/2024-07-01 outcome and measured elapsed remain unknown; 1860 s is an allocation",
                      "TimesNet has two known setup failures; their charges do not create successful fits",
                      "Full selection and end-to-end timing remain unknown for every model; three classical timings missing"],
        "old_reports_modified": False, "new_model_fits": 0, "new_model_inference": False,
        "new_HPO": False, "NAS_executed": False, "statistical_tests": 0, "bootstrap_draws": 0,
        "full_campaign_reproduction_performed": False,
    }
    inputs.reverify()
    output.mkdir(parents=True)
    write_csv(output / "hpo_costs_by_model_13.csv", table)
    write_csv(output / "known_hpo_job_costs_180.csv", known)
    write_csv(output / "cached_hpo_jobs_without_phase_cost_51.csv", missing)
    write_json(output / "summary.json", summary)
    write_json(output / "inputs_manifest.json", {"version": VERSION, "files": inputs.records,
                                                "script": {"repository_path": script_relative.as_posix(), "sha256": script_sha}})
    (output / "README.md").write_text(
        "# Conciliación de costes HPO registrados\n\n"
        "La unión conserva los170jobs del informe anterior y añade10jobs cacheados con fases publicadas en la selección ganadora;29jobs seleccionados ya estaban incluidos. Son180de231jobs completados, no180fases optimizadoras. SAE conserva sus tres etapas.\n\n"
        "El total descriptivo de fases registradas es20784.46532408601s. Mezcla ámbitos históricos de checkpoint/hardware explícitos; no es una medida homogénea de velocidad GPU ni el coste integral del proyecto. La tabla de13modelos mantiene separados fases, cargos actuales, cargos históricos de caché y asignaciones conservadoras. No sumarlos entre sí. Los51cachés sin fases conservan identidad, SHA y sus cargos existentes.\n\n"
        "Los78ajustes neuronales finales tienen90filas de fases y5765.086219800991s registrados. Los tres clásicos no tienen tiempos comparables; desconocido no significa cero. Este trabajo sólo lee metadatos, comprueba hashes y concilia identidades. No entrena, predice, ejecuta HPO/NAS, remuestrea ni sustituye informes anteriores.\n",
        encoding="utf-8")
    write_json(output / "execution_receipt.json", {
        "version": VERSION, "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_metadata_reconciliation_seconds": time.monotonic() - clock,
        "script_sha256": script_sha, "input_files_reverified": len(inputs.records),
        "input_manifest_sha256": sha((output / "inputs_manifest.json").read_bytes()),
        "identity_checks": {"unique_opportunities": 234, "unique_cached_source_jobs": 61,
                            "overlapping_selected_jobs_agree": 29, "additional_selected_cached_jobs": 10},
        "new_model_fits": 0, "new_model_inference": False, "new_HPO": False, "NAS_executed": False,
        "statistical_tests": 0, "bootstrap_draws": 0,
        "budget_runner": "scripts/e2_corrections_budget.py",
        "budget_phase": "hpo_cost_reconciliation",
        "budget_ledger": "book_e2/_static/e2_scientific_closure_20261116/compute_ledger.jsonl"})
    write_json(output / "manifest.json", {"version": VERSION, "files": {
        path.name: {"bytes": path.stat().st_size, "sha256": sha(path.read_bytes())}
        for path in sorted(output.iterdir()) if path.is_file()}})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reconcile(args.output)


if __name__ == "__main__":
    main()
