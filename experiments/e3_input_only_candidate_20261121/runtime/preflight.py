"""Read-only, standard-library admission audit; never a confirmation runner.

Run ``python -m runtime.preflight --root PACKET [--output audit/preflight.json]``.
The optional output is relative to PACKET and cannot replace a packet input.
An existing derived audit report can be regenerated atomically.
No candidate module, optimizer, data loader, or reserved payload is imported or
opened. A successful planning audit is not evidence that R3 was satisfied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any


CANDIDATE_ID = "input-only-overlap-mse-1"
SOURCE_COMMIT = "c4ca86f1431afeaa958ff0b544b453ab17a4d7b8"
ENTRYPOINT = "runtime.candidate.InputOnlyOverlapModel"
CONTROL_PATHS = (
    "CANDIDATE_IDENTITY.json", "SOURCE_MANIFEST.json",
    "confirmation/CONFIRMATION_PROTOCOL_DRAFT.json",
    "confirmation/DATA_ADMISSION_TEMPLATE.json", "confirmation/POWER_PLAN.json",
)
REQUIRED_SOURCE_PATHS = (
    "CANDIDATE_IDENTITY.json", "runtime/candidate.py", "runtime/preflight.py",
    "runtime/vendor/ORIGIN.json", "runtime/vendor/seoi_v5/__init__.py",
    "runtime/vendor/seoi_v5/data.py", "runtime/vendor/seoi_v5/model.py",
    "runtime/vendor/seoi_v5/frozen_original.py", "requirements-cpu.lock.txt",
    "receipts/remote_source.json", "inputs/status.json", "inputs/public_results.zip",
    "confirmation/FINAL_SEED_BANK.json",
) + CONTROL_PATHS[2:]
GATES = ("integrity", "data", "protocol", "methods", "power")
SHA256 = re.compile(r"[a-f0-9]{64}\Z")
RIVALS = (
    "FCN-1D", "ResNet-1D", "Inception-1D", "LSTM-autoencoder con predictor",
    "TimesNet", "TSMixer", "TiDE", "TimeMixer", "PatchTST", "iTransformer",
    "MLP16", "LSTM32", "GRU32", "Persistence", "SeasonalNaive", "ARIMA",
    "CNN-LSTM-atencion-aditiva E3", "floor_only", "floor_plus_mean_excess", "global_mse",
)
PANEL_IDS = ("F00", "F01", "F02", "F03")
READINESS_IDS = (
    "candidate_identity", "full_comparator_recipes", "four_panels_admitted",
    "conditional_method_derivation", "F2_guide_correspondence", "fresh_method_validation",
    "planning_inputs_complete", "margin_justification", "power_and_final_design",
    "budget_and_recovery", "public_preexecution_record", "one_time_evaluation_seal",
    "adequacy_and_robustness_plan",
)


class UnsafePath(ValueError):
    """A packet path would escape the local, non-symlink namespace."""


def local_path(root: Path, relative: str, *, allow_missing: bool = False) -> Path:
    """Validate before opening; do not resolve a symlink into an allowed path."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise UnsafePath("Expected a nonempty relative POSIX path")
    rel = PurePosixPath(relative)
    if rel.is_absolute() or any(p in ("", ".", "..") for p in relative.split("/")):
        raise UnsafePath("Absolute paths, empty segments and traversal are forbidden")
    if ":" in rel.parts[0]:
        raise UnsafePath("Drive and URI paths are forbidden")
    path = root
    for part in rel.parts:
        path = path / part
        if path.is_symlink():
            raise UnsafePath("Symlinks are forbidden")
    if not allow_missing and not path.is_file():
        raise FileNotFoundError(relative)
    return path


def packet_root(root: str | Path) -> Path:
    root = Path(os.path.abspath(root))
    for path in (root, *root.parents):
        if path.is_symlink():
            raise UnsafePath("Packet root cannot contain a symlink")
    if not root.is_dir():
        raise ValueError("Packet root must be an existing directory")
    return root


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _json(root: Path, relative: str) -> dict:
    with local_path(root, relative).open("r", encoding="utf-8") as stream:
        result = json.load(stream, parse_constant=lambda x: (_ for _ in ()).throw(
            ValueError("Nonfinite JSON constant: " + x)))
    if not isinstance(result, dict):
        raise ValueError("JSON object required: " + relative)
    return result


def _positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _integer(value: Any, minimum: int = 1) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _nulls(value: Any, prefix: str = "") -> list[str]:
    if value is None:
        return [prefix or "<root>"]
    if isinstance(value, dict):
        return [p for key, item in value.items() for p in _nulls(item, f"{prefix}.{key}".lstrip("."))]
    if isinstance(value, list):
        return [p for i, item in enumerate(value) for p in _nulls(item, f"{prefix}[{i}]")]
    return []


def _payload_paths(value: Any) -> set[str]:
    """Collect payload declarations without reading any of their contents."""
    result: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"payload_path", "payload_relative_path", "reserved_payload_path", "data_path", "local_data_path"} and isinstance(item, str):
                result.add(item)
            elif key in {"payload_paths", "payload_relative_paths", "reserved_payload_paths"} and isinstance(item, list):
                result.update(x for x in item if isinstance(x, str))
            result.update(_payload_paths(item))
    elif isinstance(value, list):
        for item in value:
            result.update(_payload_paths(item))
    return result


class Audit:
    def __init__(self, root: Path):
        self.root = root
        self.blockers: list[dict] = []
        self.observations: dict[str, Any] = {}
        self.manifest: dict[str, str] = {}
        self.protected: set[str] = set()
        self.seed_bank: list[int] = []

    def block(self, gate: str, code: str, detail: Any = None):
        item = {"gate": gate, "code": code}
        if detail is not None:
            item["detail"] = detail
        if item not in self.blockers:
            self.blockers.append(item)

    def need(self, condition: bool, gate: str, code: str, detail: Any = None):
        if not condition:
            self.block(gate, code, detail)

    def read(self, relative: str, gate: str) -> dict:
        try:
            if self.is_payload(relative):
                raise UnsafePath("Declared data payload cannot be read by preflight")
            return _json(self.root, relative)
        except (OSError, ValueError, TypeError) as exc:
            self.block(gate, "unreadable_control", {"path": relative, "error": str(exc)})
            return {}

    def is_payload(self, relative: str) -> bool:
        parts = PurePosixPath(relative).parts
        return relative in self.protected or any(p.lower() in {
            "reserved", "reserved_test", "test_payload", "datasets", "raw_data"
        } for p in parts)

    def evidence(self, holder: dict, path_key: str, hash_key: str, gate: str) -> dict:
        """Metadata receipt must be local and bound by the frozen source manifest."""
        relative, expected = holder.get(path_key), holder.get(hash_key)
        if not (isinstance(relative, str) and isinstance(expected, str) and SHA256.fullmatch(expected)):
            self.block(gate, "missing_content_bound_evidence", path_key)
            return {}
        if not relative.startswith("confirmation/evidence/") or not relative.endswith(".json"):
            self.block(gate, "evidence_must_be_local_metadata_json", relative)
            return {}
        try:
            local_path(self.root, relative)
        except (OSError, ValueError) as exc:
            self.block(gate, "unsafe_or_missing_evidence", {"path": relative, "error": str(exc)})
            return {}
        if self.is_payload(relative):
            self.block(gate, "evidence_is_data_payload_not_read", relative)
            return {}
        self.need(self.manifest.get(relative) == expected, gate, "evidence_not_manifest_bound", relative)
        result = self.read(relative, gate)
        self.need(not _nulls(result), gate, "evidence_contains_nulls", relative)
        return result

    def integrity(self, identity: dict, manifest: dict):
        for key, expected in (("candidate_id", CANDIDATE_ID), ("source_commit", SOURCE_COMMIT),
                              ("runtime_entrypoint", ENTRYPOINT), ("historical_recipe", "input_only_mse")):
            self.need(identity.get(key) == expected, "integrity", "identity_mismatch", key)
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            self.block("integrity", "missing_source_manifest_entries")
            files = {}
        self.manifest = files
        self.need(manifest.get("algorithm") == "sha256", "integrity", "manifest_algorithm")
        identity_hashes = identity.get("runtime_source_sha256", {})
        self.need(isinstance(identity_hashes, dict) and bool(identity_hashes), "integrity", "identity_runtime_hashes_missing")
        for relative, expected in identity_hashes.items() if isinstance(identity_hashes, dict) else []:
            self.need(files.get(relative) == expected, "integrity", "identity_runtime_hash_mismatch", relative)
        self.need(isinstance(identity.get("requirements_sha256"), str) and
                  files.get("requirements-cpu.lock.txt") == identity.get("requirements_sha256"),
                  "integrity", "identity_dependency_hash_mismatch")
        training = identity.get("training", {})
        selection = training.get("final_seed_selection", {}) if isinstance(training, dict) else {}
        selection = selection if isinstance(selection, dict) else {}
        bank_path, bank_hash = selection.get("bank_path"), selection.get("bank_sha256")
        self.need(bank_path == "confirmation/FINAL_SEED_BANK.json" and
                  isinstance(bank_hash, str) and bool(SHA256.fullmatch(bank_hash)) and files.get(bank_path) == bank_hash,
                  "integrity", "candidate_seed_bank_not_bound")
        bank = self.read("confirmation/FINAL_SEED_BANK.json", "integrity")
        ids = bank.get("seed_ids", [])
        capacity = selection.get("bank_capacity")
        bank_valid = (isinstance(ids, list) and _integer(capacity, 10) and len(ids) == capacity
                      and all(_integer(s, 0) and s < 2 ** 63 for s in ids)
                      and len(set(ids)) == len(ids) and bank.get("prefix_length_capacity") == capacity
                      and bank.get("observed_duplicate_IDs") == 0)
        self.need(bank_valid, "integrity", "invalid_candidate_seed_bank")
        if bank_valid:
            self.seed_bank = ids
        for required in REQUIRED_SOURCE_PATHS:
            self.need(required in files, "integrity", "manifest_required_path_missing", required)
        checked = 0
        for relative, expected in files.items():
            if not isinstance(relative, str) or not isinstance(expected, str) or not SHA256.fullmatch(expected):
                self.block("integrity", "malformed_manifest_entry", str(relative))
                continue
            if self.is_payload(relative):
                self.block("integrity", "payload_in_source_manifest_not_read", relative)
                continue
            try:
                path = local_path(self.root, relative)
                actual = file_sha256(path)
                self.need(actual == expected, "integrity", "source_hash_mismatch", relative)
                sizes = manifest.get("sizes", {})
                if not isinstance(sizes, dict):
                    self.block("integrity", "manifest_sizes_not_object")
                    sizes = {}
                if relative in sizes:
                    self.need(sizes[relative] == path.stat().st_size, "integrity", "source_size_mismatch", relative)
                checked += 1
            except (OSError, ValueError, TypeError) as exc:
                self.block("integrity", "unsafe_or_missing_manifest_file", {"path": relative, "error": str(exc)})
        self.observations["source_files_hashed"] = checked
        origin = self.read("runtime/vendor/ORIGIN.json", "integrity")
        receipt = self.read("receipts/remote_source.json", "integrity")
        self.need(origin.get("source_commit") == SOURCE_COMMIT, "integrity", "vendor_source_commit")
        remote_files = {x.get("path"): x for x in receipt.get("files", []) if isinstance(x, dict)}
        vendor_files = origin.get("files", [])
        self.need(isinstance(vendor_files, list) and len(vendor_files) == 4, "integrity", "vendor_origin_files")
        for item in vendor_files if isinstance(vendor_files, list) else []:
            if not isinstance(item, dict):
                self.block("integrity", "malformed_vendor_origin")
                continue
            rel = item.get("local_path")
            original = remote_files.get(item.get("path"), {})
            self.need(isinstance(rel, str) and self.manifest.get(rel) == item.get("sha256") == original.get("sha256"),
                      "integrity", "vendor_remote_source_mismatch", rel)
        status = self.read("inputs/status.json", "integrity")
        self.need(status.get("source_commit") == SOURCE_COMMIT, "integrity", "historical_source_commit")
        self.need(status.get("completed_neural_fits") == 30, "integrity", "historical_fit_count")
        self.need(status.get("prefix") == "runs/e3_seoi_v5/run-01", "integrity", "historical_result_prefix")
        expected_zip = status.get("public_results_sha256")
        self.need(isinstance(expected_zip, str) and bool(SHA256.fullmatch(expected_zip)) and
                  files.get("inputs/public_results.zip") == expected_zip,
                  "integrity", "historical_saved_zip_hash")
        try:
            zpath = local_path(self.root, "inputs/public_results.zip")
            self.need(_integer(status.get("public_results_bytes")) and
                      zpath.stat().st_size == status.get("public_results_bytes"),
                      "integrity", "historical_saved_zip_size")
        except (OSError, ValueError) as exc:
            self.block("integrity", "historical_saved_zip_missing", str(exc))


def run_preflight(root: str | Path) -> dict:
    """Return evidence about local admission, without changing packet inputs."""
    root = packet_root(root)
    audit = Audit(root)
    # Read metadata before walking the manifest so declared data are never hashed.
    data = audit.read(CONTROL_PATHS[3], "data")
    audit.protected.update(_payload_paths(data))
    # Admission receipts contain only metadata. Discover payload declarations
    # before hashing any manifest entry, including receipt-only declarations.
    panels = data.get("potential_panels", [])
    for panel in panels if isinstance(panels, list) else []:
        relative = panel.get("admission_receipt_path") if isinstance(panel, dict) else None
        if isinstance(relative, str) and relative.startswith("confirmation/evidence/") and relative.endswith(".json"):
            receipt = audit.read(relative, "data")
            audit.protected.update(_payload_paths(receipt))
    protocol = audit.read(CONTROL_PATHS[2], "protocol")
    audit.protected.update(_payload_paths(protocol))
    power = audit.read(CONTROL_PATHS[4], "power")
    identity = audit.read(CONTROL_PATHS[0], "integrity")
    manifest = audit.read(CONTROL_PATHS[1], "integrity")
    audit.integrity(identity, manifest)
    _admit_documents(audit, data, protocol, power)
    counts = {gate: sum(item["gate"] == gate for item in audit.blockers) for gate in GATES}
    result = {
        "schema_version": 1,
        "kind": "read_only_candidate_admission_preflight",
        "candidate_id": CANDIDATE_ID,
        "admission_ready": not audit.blockers,
        "confirmation_ready": False,
        "R3_accredited": False,
        "training_executed": False,
        "reserved_payloads_opened": 0,
        "gate_counts": counts,
        "pending_gate_categories": sum(value > 0 for value in counts.values()),
        "blocker_count": len(audit.blockers),
        "blockers": audit.blockers,
        "observations": audit.observations,
        "schedule": None,
        "cost_estimate": None,
        "scope": "Admission and bounded planning only; confirmation runner, checkpoint/resume, remote backup and one-shot evaluation are not implemented here.",
    }
    if not audit.blockers:
        result.update(_schedule(protocol, audit.observations["measured_costs"]))
        if not (result["cost_estimate"]["within_admitted_time_ceiling"] and
                result["cost_estimate"]["within_admitted_money_ceiling"]):
            audit.block("protocol", "measured_cost_exceeds_admitted_budget")
            result["admission_ready"] = False
            result["blocker_count"] = len(audit.blockers)
            result["gate_counts"]["protocol"] += 1
            result["pending_gate_categories"] = 1
    return result


def _admit_documents(audit: Audit, data: dict, protocol: dict, power: dict):
    # Document-specific validators are deliberately separate from file integrity.
    # A frozen checksum can faithfully preserve an incomplete scientific draft.
    for gate, document in (("data", data), ("protocol", protocol), ("power", power)):
        audit.need(document.get("status") == "ADMITTED_FROZEN", gate, "not_admitted_frozen")
    audit.need(protocol.get("is_frozen_protocol") is True, "protocol", "protocol_not_frozen")
    audit.need(protocol.get("candidate_recipe") == "input_only_mse", "protocol", "wrong_candidate_recipe")
    reference = protocol.get("candidate_identity_reference", {})
    audit.need(isinstance(reference, dict) and reference.get("path") == "CANDIDATE_IDENTITY.json"
               and reference.get("sha256") == audit.manifest.get("CANDIDATE_IDENTITY.json"),
               "protocol", "candidate_identity_reference_not_bound")
    family = protocol.get("family", {})
    family = family if isinstance(family, dict) else {}
    audit.need(family.get("primary_rivals") == list(RIVALS) and family.get("primary_rival_count") == len(RIVALS)
               and family.get("total_systems") == len(RIVALS) + 1,
               "protocol", "incomplete_or_changed_rival_family")
    audit.need(family.get("no_performance_exclusions") is True, "protocol", "performance_exclusion_policy")
    multiplicity = protocol.get("multiplicity", {})
    multiplicity = multiplicity if isinstance(multiplicity, dict) else {}
    audit.need(multiplicity.get("primary_p_slots") == 6 * len(RIVALS) and multiplicity.get("alpha") == .05,
               "methods", "multiplicity_family_incomplete")
    panels = data.get("potential_panels", [])
    panels = panels if isinstance(panels, list) else []
    ids = [p.get("id") for p in panels if isinstance(p, dict)]
    audit.need(ids == list(PANEL_IDS), "data", "four_panels_required")
    admitted = 0
    for panel in panels:
        if not isinstance(panel, dict):
            audit.block("data", "malformed_panel")
            continue
        panel_id = panel.get("id")
        audit.need(panel.get("status") == "ADMITTED" and panel.get("selected_for_final_campaign") is True,
                   "data", "panel_not_admitted", panel_id)
        audit.need(panel.get("is_reserved_test") is True and panel.get("exposure_status") == "VERIFIED_UNEXPOSED",
                   "data", "reserved_status_not_verified", panel_id)
        audit.need(not _nulls(panel), "data", "panel_unresolved_fields", panel_id)
        receipt = audit.evidence(panel, "admission_receipt_path", "admission_receipt_sha256", "data")
        required_fields = data.get("required_final_receipt_fields", [])
        audit.need(isinstance(required_fields, list) and bool(required_fields), "data", "receipt_requirements_missing")
        for key in required_fields if isinstance(required_fields, list) else []:
            audit.need(isinstance(key, str) and bool(receipt.get(key)), "data", "panel_receipt_field_missing",
                       {"panel": panel_id, "field": key})
        audit.need(receipt.get("panel_id") == panel_id and receipt.get("admission_status") == "ADMITTED"
                   and receipt.get("rights_decision") == "PERMITTED"
                   and receipt.get("exposure_decision") == "VERIFIED_UNEXPOSED"
                   and receipt.get("raw_support_disjoint") is True and receipt.get("test_unopened") is True,
                   "data", "panel_receipt_not_verified", panel_id)
        # Reserved arrays are authenticated by a custodian receipt, never opened here.
        payloads = receipt.get("payload_paths", [])
        audit.need(isinstance(payloads, list) and bool(payloads), "data", "panel_payload_declarations_missing", panel_id)
        for relative in payloads if isinstance(payloads, list) else []:
            try:
                local_path(audit.root, relative, allow_missing=True)
                audit.need(relative not in audit.manifest, "data", "payload_must_not_be_in_source_manifest", relative)
            except (OSError, ValueError, TypeError) as exc:
                audit.block("data", "unsafe_payload_declaration", str(exc))
        if panel.get("status") == "ADMITTED":
            admitted += 1
    audit.observations["panels_claimed_admitted"] = admitted
    ppanels = protocol.get("panels", {})
    audit.need(isinstance(ppanels, dict) and ppanels.get("potential_panel_ids") == list(PANEL_IDS)
               and ppanels.get("admitted_count") == 4, "data", "protocol_panels_not_admitted")
    design = protocol.get("final_design", {})
    design = design if isinstance(design, dict) else {}
    audit.need(not _nulls(design), "protocol", "final_design_unresolved_fields", _nulls(design))
    R, T, B, seeds = design.get("R"), design.get("T_by_panel"), design.get("B"), design.get("final_seed_ids")
    audit.need(_integer(R, 10), "protocol", "final_seed_count_not_admitted")
    audit.need(isinstance(seeds, list) and _integer(R, 10) and len(seeds) == R and
               all(_integer(x, 0) for x in seeds) and len(set(seeds)) == R,
               "protocol", "final_seed_ids_not_admitted")
    audit.need(isinstance(seeds, list) and _integer(R, 10) and R <= len(audit.seed_bank)
               and seeds == audit.seed_bank[:R], "protocol", "candidate_seed_bank_prefix_mismatch")
    audit.need(isinstance(T, dict) and set(T) == set(PANEL_IDS) and all(_integer(n) for n in T.values()),
               "protocol", "final_test_lengths_not_admitted")
    audit.need(_integer(B) and 1 / (B + 1) < .05 / (6 * len(RIVALS)),
               "methods", "bootstrap_tail_resolution_insufficient")
    audit.need(_integer(R, 10) and 2.0 ** (1 - R) < .05 / (6 * len(RIVALS)),
               "methods", "signed_rank_tail_resolution_insufficient")
    for panel in panels:
        if isinstance(panel, dict) and isinstance(T, dict):
            audit.need(panel.get("final_test_T") == T.get(panel.get("id")), "data", "panel_final_T_mismatch", panel.get("id"))
    gates = protocol.get("readiness_gates", [])
    gates = gates if isinstance(gates, list) else []
    audit.need([g.get("id") for g in gates if isinstance(g, dict)] == list(READINESS_IDS),
               "protocol", "readiness_gate_roster_incomplete")
    for gate in gates:
        if not isinstance(gate, dict):
            audit.block("protocol", "malformed_readiness_gate")
            continue
        name = gate.get("id")
        category = "methods" if name in {"conditional_method_derivation", "F2_guide_correspondence", "fresh_method_validation"} else "protocol"
        audit.need(gate.get("pass") is True and gate.get("status") == "ADMITTED", category, "readiness_gate_pending", name)
        evidence = audit.evidence(gate, "evidence_path", "evidence_sha256", category)
        audit.need(evidence.get("gate_id") == name and evidence.get("status") == "PASS",
                   category, "readiness_evidence_not_passed", name)
    methods = protocol.get("methods", [])
    methods = methods if isinstance(methods, list) else []
    audit.need([m.get("id") for m in methods if isinstance(m, dict)] == ["C1", "C2", "C3"],
               "methods", "three_method_roster_incomplete")
    for method in methods:
        if not isinstance(method, dict):
            audit.block("methods", "malformed_method")
            continue
        mid = method.get("id")
        audit.need(method.get("current_status") == "ADMITTED_VALIDATED", "methods", "method_not_admitted", mid)
        evidence = audit.evidence(method, "evidence_path", "evidence_sha256", "methods")
        audit.need(evidence.get("method_id") == mid and evidence.get("status") == "PASS"
                   and evidence.get("deterministic_fixtures_passed") is True
                   and evidence.get("null_validation_complete") is True
                   and evidence.get("scope_supported") is True,
                   "methods", "method_validation_missing", mid)
    audit.need(power.get("final_R") == R and power.get("final_T_by_panel") == T and power.get("final_B") == B,
               "power", "power_final_design_mismatch")
    _power_evidence(audit, protocol, power)
    _cost_evidence(audit, design)


def _power_evidence(audit: Audit, protocol: dict, power: dict):
    receipt = audit.evidence(power, "completed_power_receipt_path", "completed_power_receipt_sha256", "power")
    audit.need(receipt.get("kind") == "completed_joint_design_simulation" and receipt.get("status") == "COMPLETE"
               and receipt.get("no_reserved_data_used") is True and _integer(receipt.get("actual_MC_replicates")),
               "power", "completed_power_simulation_missing")
    design = protocol.get("final_design", {})
    design = design if isinstance(design, dict) else {}
    audit.need(receipt.get("design_sha256") == canonical_sha256({
        "R": design.get("R"), "T_by_panel": design.get("T_by_panel"), "B": design.get("B"),
        "final_seed_ids": design.get("final_seed_ids"), "rivals": list(RIVALS), "panels": list(PANEL_IDS),
    }), "power", "power_evidence_design_mismatch")
    contrasts = receipt.get("per_contrast", {})
    contrasts = contrasts if isinstance(contrasts, dict) else {}
    expected = {f"{panel}::{rival}" for panel in PANEL_IDS for rival in RIVALS}
    audit.need(set(contrasts) == expected, "power", "full_family_power_evidence_missing")
    for name in sorted(expected):
        item = contrasts.get(name, {})
        item = item if isinstance(item, dict) else {}
        audit.need(_event_rate_valid(item, .9), "power", "contrast_power_not_demonstrated", name)
    joint = receipt.get("joint_four_panel_decision", {})
    audit.need(isinstance(joint, dict) and _event_rate_valid(joint), "power", "joint_power_not_reported")
    audit.need(power.get("power_per_contrast") == contrasts and power.get("joint_power") == joint
               and design.get("power_per_contrast") == contrasts and design.get("joint_power") == joint,
               "power", "power_summaries_not_bound_to_actual_evidence")
    audit.observations["power_contrasts_required"] = len(expected)
    audit.observations["power_contrasts_demonstrated"] = sum(_event_rate_valid(v, .9) for v in contrasts.values() if isinstance(v, dict))


def _event_rate_valid(item: dict, minimum: float = 0.) -> bool:
    n, successes = item.get("replicates"), item.get("successes")
    estimate, low, high = item.get("estimate"), item.get("simultaneous_CI_lower"), item.get("simultaneous_CI_upper")
    if not (_integer(n) and _integer(successes, 0) and successes <= n):
        return False
    if not all(isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x)
               for x in (estimate, low, high)):
        return False
    return 0 <= low <= estimate <= high <= 1 and low >= minimum and abs(estimate - successes / n) < 1e-12


def _cost_evidence(audit: Audit, design: dict):
    evidence = audit.evidence(design, "timing_evidence_path", "timing_evidence_sha256", "protocol")
    audit.need(evidence.get("kind") == "measured_development_costs" and evidence.get("status") == "COMPLETE",
               "protocol", "measured_cost_evidence_missing")
    for key in ("neural_fit_seconds_p95", "arima_fit_seconds_p95", "analytic_trajectory_seconds_p95", "export_seconds"):
        audit.need(_positive(evidence.get(key)), "protocol", "measured_cost_missing", key)
    audit.need(_integer(evidence.get("timing_samples_per_system")), "protocol", "timing_sample_count_missing")
    for key in ("monetary_rate_per_hour", "reserve_fraction"):
        value = evidence.get(key)
        audit.need(isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0,
                   "protocol", "cost_parameter_missing", key)
    audit.need(_nonempty(evidence.get("currency")), "protocol", "currency_missing")
    audit.need(_positive(design.get("budget_wall_seconds")) and _positive(design.get("budget_money")),
               "protocol", "final_budget_not_admitted")
    # Store already-validated metadata, not any dataset or fitted model.
    audit.observations["measured_costs"] = evidence


def _schedule(protocol: dict, costs: dict) -> dict:
    # Called only after every integrity/science admission gate has passed.
    design = protocol["final_design"]
    R = design["R"]
    schedule = {"R": R, "panel_ids": list(PANEL_IDS), "rivals": list(RIVALS),
                "final_seed_ids": design["final_seed_ids"], "T_by_panel": design["T_by_panel"],
                "neural_systems_including_candidate": 16, "neural_fits": 16 * 4 * R,
                "ARIMA_fits": 4, "analytic_trajectories": 16,
                "fits_exclude_tuning_failures_and_extra_phases": True,
                "execution_enabled": False}
    active_seconds = (schedule["neural_fits"] * costs["neural_fit_seconds_p95"] +
                      schedule["ARIMA_fits"] * costs["arima_fit_seconds_p95"] +
                      schedule["analytic_trajectories"] * costs["analytic_trajectory_seconds_p95"] +
                      costs["export_seconds"])
    budgeted_seconds = active_seconds * (1 + costs["reserve_fraction"])
    money = budgeted_seconds / 3600 * costs["monetary_rate_per_hour"]
    return {"schedule": schedule, "cost_estimate": {
        "active_compute_seconds": active_seconds, "with_reserve_seconds": budgeted_seconds,
        "money": money, "currency": costs["currency"],
        "within_admitted_time_ceiling": budgeted_seconds <= design["budget_wall_seconds"],
        "within_admitted_money_ceiling": money <= design["budget_money"],
        "basis": "Measured development p95 timings under the admitted deterministic fit policy; excludes tuning and additional phases.",
        "execution_enabled": False,
    }}


def write_report(root: Path, relative: str, report: dict):
    """Atomically write a derived report under audit/, never a frozen input."""
    if not relative.startswith("audit/") or relative in CONTROL_PATHS:
        raise UnsafePath("Output must be a relative path under audit/")
    path = local_path(root, relative, allow_missing=True)
    try:
        manifest = _json(root, "SOURCE_MANIFEST.json")
    except (OSError, ValueError):
        manifest = {}
    files = manifest.get("files", {})
    if isinstance(files, dict) and relative in files:
        raise UnsafePath("Output cannot replace a manifest-bound packet input")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Revalidate after creating parents. Derived reports can be regenerated.
    path = local_path(root, relative, allow_missing=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".preflight-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", help="Optional derived report path under audit/, relative to --root")
    args = parser.parse_args(argv)
    try:
        root = packet_root(args.root)
        report = run_preflight(root)
        if args.output:
            write_report(root, args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0 if report["admission_ready"] else 2
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"schema_version": 1, "admission_ready": False,
                          "confirmation_ready": False, "R3_accredited": False,
                          "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
