#!/usr/bin/env python3
"""Audit the documentary E2 proposal without loading market data or ML packages.

This is a metadata preflight, not a training pipeline or an execution permit.
A valid proposal can have blocking prerequisites. --require-ready exits 2 for
those blockers; a broken scientific contract exits 1. No market archives or
individual forecasts are loaded; known E1 aggregate metadata is hashed, not used
to select models.
"""
from __future__ import annotations

import argparse
import csv
import calendar
from datetime import date, timedelta
from hashlib import sha256
import io
import itertools
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
PROPOSAL = "experiments/e2_benchmark_design_20261105/proposal.json"
OUTPUT = "book_e2/_static/benchmark_design"
SOURCE_COMMIT = "94a4b0324e93361e8928d708fdfeb50110e3686f"
DELTA = 0.00025772842664981106
# Pin the scientific fields themselves, independently of both editable copies.
# Metadata-only additions to e1_reference.json do not enter scientific_snapshot.
INHERITED_SCIENTIFIC_SHA256 = "1464fa79d472f6755651774989b140659ee1f21daeb30032fc650a8770f4fdc4"
TOP10 = {"TSMixer": "MLP/mixing", "TiDE": "MLP/mixing", "TimeMixer": "MLP/mixing",
         "TimesNet": "Convolution", "PatchTST": "Attention", "Informer": "Attention",
         "Autoformer": "Attention", "iTransformer": "Attention", "Crossformer": "Attention",
         "S-Mamba": "State-space"}
ANCHORS = {"GRU/D7-c03", "LSTM/D7-c02", "TCN/D7-c02"}
CONTRACT_FIELDS = ("source_commit", "registered_baseline", "base_model_recipes",
                   "recipe_architectures", "primary_metric", "practical_margin",
                   "independence", "partition")
PENDING_SECTIONS = ("7", "9.2", "10", "11.1", "13.2", "14")
ARTIFACTS = {
    "adapter_architecture_definitions": "Definiciones exactas de los adaptadores D7 y aplicabilidad de NAS",
    "D7_contract_and_causality_tests": "Pruebas de forma, información disponible y causalidad D7",
    "nested_indices_audit": "Índices y soportes de la nueva CV anidada propuestos, todavía no generados",
    "training_environment_lock": "Lock E2 compatible con las diez implementaciones",
    "common_hardware_profile": "Perfil común de CPU/GPU/RAM y coste por ajuste sin usar test",
    "leakage_audit": "Auditoría formal de fuga de sección 10",
    "pilot_and_power": "Piloto propio de E2 y dimensionamiento de sección 9.2",
    "frozen_protocol": "Protocolo definitivo y comparación equitativa cerrados antes de ejecutar E2",
}


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def scientific_snapshot(reference):
    return {key: reference[key] for key in CONTRACT_FIELDS}


def digest(value):
    return sha256(value).hexdigest()


def local_path(root, name):
    """Only access explicit repository metadata paths, never a raw data root."""
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or "\\" in str(name):
        raise ValueError("Metadata paths must stay relative to the repository")
    result = root / path
    if result.is_symlink() or not result.resolve().is_relative_to(root.resolve()):
        raise ValueError("Metadata path escapes the repository")
    return result


def load(root, name):
    return json.loads(local_path(root, name).read_bytes())



def proposed_nested_calendar(proposal):
    """Generate dates only, not supervised windows or executable CV indices.

    For validation origin q the inherited nominal inner90 rule is [q-90,q-8]
    inclusive (83 potential calendar dates), with stop-training outcomes <=q-98.
    Missing days, target support and actual eligibility are deliberately unaudited.
    """
    hpo = proposal["hpo"]["nested_validation_proposal"]
    first_label = date(2023, 1, 8)
    rows = []
    for month in range(1, 11):
        origin = date(2024, month, 1)
        outer_max = origin - timedelta(days=8)
        beginning = outer_max - timedelta(days=89)
        for block in range(3):
            inner_start = beginning + timedelta(days=30 * block)
            inner_end = inner_start + timedelta(days=29)
            train_max = inner_start - timedelta(days=8)
            stop_start = inner_start - timedelta(days=90)
            stop_end = train_max
            early_max = inner_start - timedelta(days=98)
            if early_max < first_label or inner_end > outer_max:
                raise ValueError("Nested calendar exceeds admissible chronological supports")
            if (stop_end - stop_start).days + 1 != 83:
                raise ValueError("The inherited nominal90/embargo7 stop interval differs")
            rows.append({"outer_origin_UTC": origin.isoformat(),
                         "outer_validation_end_inclusive_UTC": date(2024, month, calendar.monthrange(2024, month)[1]).isoformat(),
                         "outer_train_outcome_max_UTC": outer_max.isoformat(),
                         "inner_block": block + 1,
                         "inner_validation_start_UTC": inner_start.isoformat(),
                         "inner_validation_end_inclusive_UTC": inner_end.isoformat(),
                         "inner_train_outcome_max_UTC": train_max.isoformat(),
                         "early_stop_validation_start_UTC": stop_start.isoformat(),
                         "early_stop_validation_end_inclusive_UTC": stop_end.isoformat(),
                         "early_stop_potential_calendar_days": 83,
                         "early_stop_training_outcome_max_UTC": early_max.isoformat(),
                         "first_potential_training_label_UTC": first_label.isoformat(),
                         "calendar_state": "proposed_dates_only",
                         "actual_eligible_windows_audited": False,
                         "tsxv_indices_generated": False})
    return rows


def validate(proposal, root=ROOT):
    """Fail on contract corruption; return factual, unresolved prerequisites."""
    root = Path(root)
    if proposal.get("status") != "proposed_not_frozen":
        raise ValueError("The experimental proposal is not a frozen preregistration")
    reference = load(root, "book_e2/_static/e1_reference.json")
    card = load(root, "book_e2/_static/dataset_card.json")
    guide = load(root, "book_e2/_static/guia_e2_recibida.json")
    original = scientific_snapshot(reference)
    if original["source_commit"] != SOURCE_COMMIT or original["practical_margin"]["absolute_delta"] != DELTA:
        raise ValueError("The E1 reference has changed its pinned scientific identity or absolute delta")
    if digest(json_bytes(original)) != INHERITED_SCIENTIFIC_SHA256:
        raise ValueError("The E1 reference differs from its pinned scientific snapshot")
    if proposal.get("inherited_scientific_contract") != original:
        raise ValueError("Proposal changed the inherited E1 scientific contract, metric, recipes or calendar")
    if proposal.get("data_contract") != {"schema": card["schema"], "label_protocol": card["label_protocol"],
                                       "representation": card["representation"]}:
        raise ValueError("Proposal changed the inherited D7×5 input, target or preprocessing contract")
    receipts = proposal.get("e1_source_receipts", {})
    expected = {name: record["sha256"] for name, record in reference["source_files"].items()}
    if receipts != expected or len(receipts) != 6:
        raise ValueError("The six pinned E1 source receipts are missing or differ")
    for name, checksum in receipts.items():
        source = local_path(root, name)
        if not source.is_file() or digest(source.read_bytes()) != checksum:
            raise ValueError("Pinned E1 source hash mismatch: " + name)
    chronology = proposal.get("evaluation", {})
    if chronology.get("new_unseen_test_assigned") is not False or chronology.get("confirmatory") is not False:
        raise ValueError("Known E1 evaluations cannot be renamed new or confirmatory tests")
    if chronology.get("test_selection_allowed") is not False or chronology.get("maximum_future_scoring_events") != 1:
        raise ValueError("Test-based selection or repeated adaptive scoring is not allowed")
    if chronology.get("historical_test_is_unseen") is not False:
        raise ValueError("A frozen hash does not restore unseen-test status")
    models = proposal.get("models", [])
    ids = [model.get("id") for model in models]
    if len(models) != 15 or len(set(ids)) != len(ids):
        raise ValueError("Exactly 15 distinct proposed configurations are required in this manifest")
    top = [model for model in models if model.get("role") == "top10_candidate"]
    if {m["id"]: m.get("family") for m in top} != TOP10:
        raise ValueError("The proposed Top10 identities or their four families differ")
    if {m["id"] for m in models if m.get("role") == "inherited_fixed_anchor"} != ANCHORS:
        raise ValueError("The three inherited fixed recipes must remain separate references")
    if set(ids) != set(TOP10) | ANCHORS | {"Ridge", "Persistence"}:
        raise ValueError("Benchmark references differ from the documented minimum 15")
    hpo = proposal["hpo"]
    if hpo.get("status") != "proposed_not_frozen" or hpo.get("declared_equal_budget_for_all15") is not False:
        raise ValueError("Do not declare a closed or equal HPO budget for frozen E1 references")
    grid = hpo["proposed_hpo_grid"]
    if not grid.get("D7_adapter_validity_pending"):
        raise ValueError("Grid does not establish validity of the D7 architectures")
    axes = grid["axes"]
    candidates = list(itertools.product(axes["learning_rate"], axes["dropout"], axes["d_model"]))
    if not candidates or len(set(candidates)) != hpo["candidates_per_family"]:
        raise ValueError("Candidate count does not match the distinct proposed grid")
    if any(lr <= 0 or not 0 <= drop < 1 or not isinstance(width, int) or width <= 0 for lr, drop, width in candidates):
        raise ValueError("Invalid proposed optimizer/capacity grid")
    seeds = hpo["search_seed_ids"]
    if len(seeds) != 2 or len(set(seeds)) != 2 or any(not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("HPO requires the same two distinct registered proposal seeds")
    nested = hpo["nested_validation_proposal"]
    if (nested["outer_monthly_blocks"], nested["inner_blocks_per_origin"],
        nested["inner_block_calendar_days"], nested["embargo_days"]) != (10, 3, 30, 7):
        raise ValueError("Nested-validation proposal no longer matches its documented calendar")
    if nested.get("indices_generated") is not False or nested.get("nested_execution_already_performed") is not False:
        raise ValueError("This documentary proposal does not contain generated or executed nested folds")
    if nested.get("outer_start") != original["partition"]["hpo_forward_chaining"]["start"] or nested.get("outer_end_inclusive") != original["partition"]["hpo_forward_chaining"]["end_inclusive"]:
        raise ValueError("Nested outer origins changed the inherited validation calendar")
    nas = proposal["architecture_search_condition"]
    if nas.get("all_adapters_exempt_from_section14") is not False or nas.get("section14_received") is not False:
        raise ValueError("Do not exempt D7 adapters from an unread conditional NAS rule")
    repetitions = proposal["repetitions"]
    if repetitions.get("minimum_guide_runs") != 5 or repetitions.get("final_R") is not None:
        raise ValueError("The guide minimum is five; E2 power has not determined final R")
    if repetitions.get("E1_R465_is_E2_power") is not False:
        raise ValueError("E1 R465 is not evidence of E2 power")
    if proposal.get("new_deep_learning_fits") != 0 or proposal.get("new_Bitcoin_results") != 0:
        raise ValueError("No E2 training or Bitcoin scoring receipt is present in this proposal")
    if tuple(proposal["required_guide_sections_pending"]) != PENDING_SECTIONS:
        raise ValueError("Do not clear guide prerequisites by editing readiness flags")
    blockers = [{"id": "training_executor_missing", "reason": "Este auditor no implementa ni ejecuta el pipeline de entrenamiento E2."}]
    for section in PENDING_SECTIONS:
        blockers.append({"id": "guide_" + section, "reason": "Falta revisar la sección " + section + " completa y su efecto sobre el protocolo."})
    blockers.append({"id": "fair_HPO_scope_unresolved", "reason": hpo["fairness_resolution_pending"]})
    if hpo.get("common_compute_budget_seconds") is None:
        blockers.append({"id": "compute_budget_not_profiled", "reason": "Presupuesto común de tiempo no dimensionado con un perfil de hardware."})
    else:
        if not isinstance(hpo["common_compute_budget_seconds"], (int, float)) or hpo["common_compute_budget_seconds"] <= 0:
            raise ValueError("Compute budget must be positive when proposed")
        blockers.append({"id": "compute_budget_unvalidated", "reason": "Un número propuesto no acredita un perfil de cómputo ejecutado."})
    for key, description in ARTIFACTS.items():
        receipt = proposal.get("pretraining_receipts", {}).get(key, {})
        path, checksum = receipt.get("path"), receipt.get("sha256")
        if path is None:
            blockers.append({"id": key, "reason": description + ": evidencia pendiente."})
        else:
            artifact = local_path(root, path)
            if not artifact.is_file() or not checksum or digest(artifact.read_bytes()) != checksum:
                raise ValueError("Pretraining receipt missing or hash mismatch: " + key)
            blockers.append({"id": key + "_needs_scope_review", "reason": description + ": existencia y hash no certifican suficiencia metodológica."})
    model_rows = []
    for model in models:
        adapter = model.get("e2_adapter_path")
        if adapter is not None:
            path = local_path(root, adapter)
            if not path.is_file() or digest(path.read_bytes()) != model.get("e2_adapter_sha256"):
                raise ValueError("Adapter missing or hash mismatch: " + model["id"])
        model_rows.append({"model": model["id"], "role": model["role"], "family": model["family"],
                           "original_architecture_identified": model["original_architecture_identified"],
                           "e2_adapter_path": adapter or "pending",
                           "D7_executable_adapter_verified": False,
                           "training_lock_verified": False,
                           "HPO_budget_closed": False,
                           "executed_E2_runs": 0,
                           "status": "pending_E2_contract_validation"})
    if proposal.get("declared_ready_to_train"):
        blockers.append({"id": "self_declaration_ignored", "reason": "La declaración de listo no sustituye adaptadores, auditorías y protocolo."})
    requirements = []
    for item in guide.get("section_4_requirements", []):
        status = "documented_proposal_not_executed"
        if item["id"] == "4.1.partition": status = "known_test_limitation_unsatisfied"
        if item["id"] == "4.1.undefined_architecture": status = "conditional_section14_pending"
        if item["id"] == "4.3.repetitions": status = "pilot_and_power_pending"
        requirements.append({"id": item["id"], "requirement": item["requirement"], "status": status,
                             "execution_evidence_available": False})
    if not requirements:
        raise ValueError("Section4 requirements have not been received")
    count = hpo["candidates_per_family"] * len(seeds)
    if proposal["training_proposal"].get("optimizer_fits_per_validation_evaluation") != 2:
        raise ValueError("Each planned evaluation must count early-stop plus full admissible-training refit")
    if proposal["training_proposal"].get("early_stop_nominal_lookback_days") != 90:
        raise ValueError("The inherited early-stop rule uses a nominal90 origin lookback")
    evaluations = {"nested_inner_evaluations_per_tuned_family": count * nested["inner_blocks_per_origin"] * 10,
                   "outer_selected_evaluations_per_tuned_family": len(seeds) * 10,
                   "final_recipe_search_evaluations_per_tuned_family": count * 10}
    evaluations["subtotal_per_tuned_family_before_pilot_and_final"] = sum(evaluations.values())
    costs = {"nested_inner_optimizer_fits_per_tuned_family": 2 * evaluations["nested_inner_evaluations_per_tuned_family"],
             "outer_selected_optimizer_fits_per_tuned_family": 2 * evaluations["outer_selected_evaluations_per_tuned_family"],
             "final_recipe_search_optimizer_fits_per_tuned_family": 2 * evaluations["final_recipe_search_evaluations_per_tuned_family"]}
    costs["subtotal_optimizer_fits_per_tuned_family_before_pilot_and_final"] = sum(costs.values())
    costs["top10_subtotal_optimizer_fits_before_pilot_and_final"] = costs["subtotal_optimizer_fits_per_tuned_family_before_pilot_and_final"] * 10
    costs["scope"] = "Propuesta, no ajustes ejecutados. Cada evaluación exige dos ajustes optimizador (early-stop y refit); no incluye Ridge, anclas, NAS condicional, piloto, definitivo, robustez o ablaciones."
    dates = proposed_nested_calendar(proposal)
    report = {"schema_version": 1, "archived_proposal_scope": "Initial15-row proposal only; superseded by experiments/e2_remediation_20261106 and scripts/e2_protocol_plan.py. Its pending-guide blockers describe the original snapshot, not current guide receipt.", "audit_kind": "documentary_pretraining_readiness_not_training_pipeline",
              "proposal_valid": True, "proposal_status": "proposed_not_frozen", "ready_to_train": False,
              "ready_to_claim_confirmatory_test": False, "new_deep_learning_fits": 0, "new_Bitcoin_results": 0,
              "minimum_total_configurations": 15, "top10_candidate_count": 10,
              "top10_family_count": 4, "inherited_source_commit": SOURCE_COMMIT,
              "inherited_scientific_contract_sha256": digest(json_bytes(original)),
              "pinned_E1_source_hashes_verified": len(receipts),
              "source_guide_requirements_received": len(requirements), "blockers": blockers,
              "proposed_validation_evaluation_count_inventory": evaluations,
              "proposed_optimizer_fit_count_inventory": costs,
              "proposed_nested_calendar_rows": len(dates), "actual_eligible_windows_audited": False,
              "receipt_scope": "Existencia y hash son comprobaciones documentales; este auditor no certifica implementación, fuga, potencia o ejecución.",
              "execution_environment_observation": {"python": platform.python_version(), "platform": platform.system(),
                                                  "training_environment_validated": False}}
    return report, model_rows, requirements


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode()


def audit(root=ROOT, proposal_name=PROPOSAL, output_name=OUTPUT):
    root = Path(root)
    proposal = load(root, proposal_name)
    report, models, requirements = validate(proposal, root)
    output = local_path(root, output_name); output.mkdir(parents=True, exist_ok=True)
    contents = {"proposal.json": json_bytes(proposal), "report.json": json_bytes(report),
                "model_readiness.csv": csv_bytes(models), "requirements.csv": csv_bytes(requirements),
                "nested_calendar.csv": csv_bytes(proposed_nested_calendar(proposal))}
    provenance = {"generator": "scripts/audit_e2_benchmark_readiness.py",
                  "generator_sha256": digest(Path(__file__).read_bytes()),
                  "proposal_repository_path": proposal_name,
                  "input_hashes": {name: digest(local_path(root, name).read_bytes()) for name in
                                   [proposal_name, "book_e2/_static/e1_reference.json", "book_e2/_static/dataset_card.json",
                                    "book_e2/_static/guia_e2_recibida.json"]},
                  "output_hashes": {name: digest(data) for name, data in contents.items()},
                  "source_E1_receipts": proposal["e1_source_receipts"],
                  "market_archives_opened": 0, "DL_packages_imported": 0, "model_fits": 0,
                  "non_training_audit": True}
    contents["provenance.json"] = json_bytes(provenance)
    for name, data in contents.items(): (output / name).write_bytes(data)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", default=PROPOSAL, help="Repository-relative proposal JSON")
    parser.add_argument("--output-dir", default=OUTPUT, help="Repository-relative metadata output directory")
    parser.add_argument("--require-ready", action="store_true", help="Exit2 if actual prerequisites remain incomplete")
    args = parser.parse_args(argv)
    try: report = audit(proposal_name=args.proposal, output_name=args.output_dir)
    except (ValueError, KeyError, OSError, TypeError) as error:
        print("Invalid E2 proposal: " + str(error), file=sys.stderr); return 1
    print(json.dumps({"proposal_valid": report["proposal_valid"], "ready_to_train": report["ready_to_train"],
                      "blockers": len(report["blockers"]), "models": 15,
                      "new_deep_learning_fits": 0}, ensure_ascii=False))
    return 2 if args.require_ready and not report["ready_to_train"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
