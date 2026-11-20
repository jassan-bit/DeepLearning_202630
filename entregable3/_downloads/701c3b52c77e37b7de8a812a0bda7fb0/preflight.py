"""CPU-only, resumable protocol readiness adapter. This file cannot train.

It evaluates concrete evidence bindings before a future V3 engine can be
considered. Importing or executing it never imports torch, controller or worker.
Prepared receipts are new files; the two historical masters are read-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def evidence_bound(protocol, protocol_path, name):
    binding = protocol.get("evidence_bindings", {}).get(name)
    if not isinstance(binding, dict):
        return False
    relative, expected = binding.get("path"), binding.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        return False
    root = protocol_path.parent.resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        return False
    return digest(candidate) == expected


def assess(protocol, protocol_path):
    family = protocol.get("family", {})
    science = protocol.get("scientific_decision", {})
    data = protocol.get("data", {})
    estimand = protocol.get("estimand", {})
    sampling = protocol.get("sampling", {})
    budget = protocol.get("budget", {})
    freeze = protocol.get("freeze", {})
    adapter = protocol.get("adapter", {})
    tests = protocol.get("proposed_F2_test_plan", {})
    rivals = family.get("rivals", [])
    checks = {
        "scientific_go_and_reason_bound": science.get("confirmation_go") is True
            and evidence_bound(protocol, protocol_path, "renewed_scientific_decision"),
        "original_failed_pilot_preserved": science.get("original_pilot_heuristic_passed") is False
            and science.get("original_pilot_rule_preserved") is True
            and science.get("automatic_promotion_allowed") is False,
        "complete_main_family19_57": isinstance(rivals, list) and len(rivals) == len(set(rivals)) == 19
            and family.get("candidate") == "atom_crps" and family.get("all_system_count") == 20
            and family.get("main_component_count") == 57 and family.get("procedures_per_rival") == 3,
        "unexposed_admission_and_endpoints_bound": data.get("independent_confirmation_admission_complete") is True
            and data.get("labels_unknown_until_lock") is True
            and isinstance(data.get("new_main_panel"), dict)
            and len(data.get("new_additional_panels", [])) == 3
            and all(isinstance(x, dict) for x in data.get("new_additional_panels", []))
            and evidence_bound(protocol, protocol_path, "unexposed_admission"),
        "margins_justified_before_outcomes": finite_positive(estimand.get("BTC_delta_inherited_reference"))
            and all(finite_positive(estimand.get("additional_dataset_deltas", {}).get(k)) for k in ("D01", "D02", "D03"))
            and evidence_bound(protocol, protocol_path, "operational_margins"),
        "full_family_prospective_power_bound": type(sampling.get("selected_final_R")) is int
            and sampling["selected_final_R"] >= 10
            and isinstance(sampling.get("final_seed_ids"), list)
            and len(set(sampling["final_seed_ids"])) == len(sampling["final_seed_ids"]) == sampling["selected_final_R"]
            and sampling.get("current_pilot_seed_pairs", 0) >= 5
            and sampling.get("current_pilot_complete_family_variances") is True
            and evidence_bound(protocol, protocol_path, "prospective_full_family_power"),
        "temporal_battery_estimand_review_bound": tests.get("method_review_approved") is True
            and tests.get("primary_estimand_alignment_approved") is True
            and evidence_bound(protocol, protocol_path, "F2_battery_review_and_tests"),
        "author_budget_capacity_and_fairness_bound": budget.get("authorized") is True
            and finite_positive(budget.get("approved_GPU_hours"))
            and finite_positive(budget.get("cost_upper_template_seconds"))
            and evidence_bound(protocol, protocol_path, "author_budget_and_fairness"),
        "V3_engine_version_contract_bound": adapter.get("V3_runner_contract_reviewed") is True
            and isinstance(adapter.get("V3_runner_source_commit"), str)
            and bool(re.fullmatch(r"[0-9a-f]{40}", adapter.get("V3_runner_source_commit", "")))
            and evidence_bound(protocol, protocol_path, "V3_engine_contract_and_code"),
        "annotated_protocol_freeze_bound": freeze.get("tag_annotated") is True
            and freeze.get("all_fields_final") is True
            and bool(freeze.get("tag")) and bool(freeze.get("date"))
            and isinstance(freeze.get("commit"), str)
            and bool(re.fullmatch(r"[0-9a-f]{40}", freeze.get("commit", "")))
            and evidence_bound(protocol, protocol_path, "annotated_tag_verification"),
    }
    return {"status": "evidence_structurally_ready_for_separate_authorized_engine_review" if all(checks.values()) else "BLOCKED_PREPARATION_ONLY",
        "checks": checks, "blockers": [name for name, ok in checks.items() if not ok],
        "structural_readiness_only": True, "scientific_and_statistical_validity_not_inferred_from_JSON_fields": True,
        "this_adapter_can_train": False, "training_started": False,
        "new_model_fits": 0, "new_model_predictions": 0}


def prepare(protocol_path, state_path):
    protocol = json.loads(protocol_path.read_text())
    identity = {"protocol_sha256": digest(protocol_path), "adapter_sha256": digest(Path(__file__)),
        "kind": "CPU_V3_preparation_only", "historical_masters_preserved": ["20261119T031104515520Z", "20261120T031424176339Z"]}
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if previous.get("identity") != identity:
            raise ValueError("Preparation identity changed; preserve this receipt and choose a new output path")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "identity": identity,
        **assess(protocol, protocol_path)}
    temporary = state_path.with_suffix(state_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(state_path)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.DRAFT.json"))
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true", help="Successful CPU preparation exits zero even when readiness is blocked; never enables training")
    args = parser.parse_args()
    result = prepare(args.protocol, args.receipt)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if args.prepare_only or not result["blockers"] else 2)
