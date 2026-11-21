"""Admission tests use synthetic metadata/files only; no training imports."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from runtime import preflight as p


class PacketFixture:
    def __init__(self, root):
        self.root = Path(root)
        self.documents = {}
        for name in p.REQUIRED_SOURCE_PATHS:
            if name.endswith(".json"):
                self.put(name, {})
            else:
                path = self.root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(("synthetic software fixture: " + name).encode())
        vendor = []
        for name in ("__init__.py", "data.py", "model.py", "frozen_original.py"):
            local = "runtime/vendor/seoi_v5/" + name
            vendor.append({"path": "seoi_v5/" + name, "local_path": local,
                           "sha256": p.file_sha256(self.root / local)})
        self.put("runtime/vendor/ORIGIN.json", {"source_commit": p.SOURCE_COMMIT, "files": vendor})
        self.put("receipts/remote_source.json", {"files": vendor})
        self.put("inputs/status.json", {
            "source_commit": p.SOURCE_COMMIT, "completed_neural_fits": 30,
            "prefix": "runs/e3_seoi_v5/run-01",
            "public_results_sha256": p.file_sha256(self.root / "inputs/public_results.zip"),
            "public_results_bytes": (self.root / "inputs/public_results.zip").stat().st_size,
        })
        bank_ids = list(range(20261201, 20261233))
        self.put("confirmation/FINAL_SEED_BANK.json", {
            "seed_ids": bank_ids, "prefix_length_capacity": len(bank_ids), "observed_duplicate_IDs": 0,
        })
        self.put("CANDIDATE_IDENTITY.json", {
            "candidate_id": p.CANDIDATE_ID, "source_commit": p.SOURCE_COMMIT,
            "runtime_entrypoint": p.ENTRYPOINT, "historical_recipe": "input_only_mse",
            "runtime_source_sha256": {"runtime/candidate.py": p.file_sha256(self.root / "runtime/candidate.py")},
            "requirements_sha256": p.file_sha256(self.root / "requirements-cpu.lock.txt"),
            "training": {"final_seed_selection": {"bank_path": "confirmation/FINAL_SEED_BANK.json",
                "bank_sha256": p.file_sha256(self.root / "confirmation/FINAL_SEED_BANK.json"),
                "bank_capacity": len(bank_ids)}},
        })
        R, B = 13, 250000
        seeds = list(range(20261201, 20261201 + R))
        T = {panel: 100 for panel in p.PANEL_IDS}
        event = {"replicates": 10000, "successes": 9500, "estimate": .95,
                 "simultaneous_CI_lower": .91, "simultaneous_CI_upper": .98}
        contrasts = {f"{panel}::{rival}": copy.deepcopy(event) for panel in p.PANEL_IDS for rival in p.RIVALS}
        joint = {"replicates": 10000, "successes": 5000, "estimate": .5,
                 "simultaneous_CI_lower": .45, "simultaneous_CI_upper": .55}
        panels = []
        for panel_id in p.PANEL_IDS:
            receipt = {"panel_id": panel_id, "admission_status": "ADMITTED", "rights_decision": "PERMITTED",
                       "exposure_decision": "VERIFIED_UNEXPOSED", "raw_support_disjoint": True,
                       "test_unopened": True, "payload_paths": ["private/" + panel_id + ".csv"]}
            path, digest = self.evidence("panel-" + panel_id, receipt)
            panels.append({"id": panel_id, "status": "ADMITTED", "selected_for_final_campaign": True,
                           "is_reserved_test": True, "exposure_status": "VERIFIED_UNEXPOSED", "final_test_T": 100,
                           "admission_receipt_path": path, "admission_receipt_sha256": digest})
        self.put(p.CONTROL_PATHS[3], {"status": "ADMITTED_FROZEN", "potential_panels": panels,
                                    "required_final_receipt_fields": ["rights_decision", "exposure_decision"]})
        gates = []
        for name in p.READINESS_IDS:
            path, digest = self.evidence("gate-" + name, {"gate_id": name, "status": "PASS"})
            gates.append({"id": name, "status": "ADMITTED", "pass": True, "evidence_path": path, "evidence_sha256": digest})
        methods = []
        for mid in ("C1", "C2", "C3"):
            path, digest = self.evidence("method-" + mid, {
                "method_id": mid, "status": "PASS", "deterministic_fixtures_passed": True,
                "null_validation_complete": True, "scope_supported": True,
            })
            methods.append({"id": mid, "current_status": "ADMITTED_VALIDATED", "evidence_path": path, "evidence_sha256": digest})
        timing_path, timing_hash = self.evidence("timings", {
            "kind": "measured_development_costs", "status": "COMPLETE", "timing_samples_per_system": 5,
            "neural_fit_seconds_p95": 120, "arima_fit_seconds_p95": 10,
            "analytic_trajectory_seconds_p95": 1, "export_seconds": 60,
            "monetary_rate_per_hour": 2, "reserve_fraction": .2, "currency": "USD",
        })
        design = {"R": R, "T_by_panel": T, "B": B, "final_seed_ids": seeds,
                  "power_per_contrast": contrasts, "joint_power": joint,
                  "budget_wall_seconds": 200000, "budget_money": 200,
                  "timing_evidence_path": timing_path, "timing_evidence_sha256": timing_hash}
        self.put(p.CONTROL_PATHS[2], {
            "status": "ADMITTED_FROZEN", "is_frozen_protocol": True, "candidate_recipe": "input_only_mse",
            "candidate_identity_reference": {"path": "CANDIDATE_IDENTITY.json",
                "sha256": p.file_sha256(self.root / "CANDIDATE_IDENTITY.json")},
            "family": {"primary_rivals": list(p.RIVALS), "primary_rival_count": len(p.RIVALS),
                       "total_systems": len(p.RIVALS) + 1, "no_performance_exclusions": True},
            "multiplicity": {"primary_p_slots": 120, "alpha": .05},
            "panels": {"potential_panel_ids": list(p.PANEL_IDS), "admitted_count": 4},
            "readiness_gates": gates, "methods": methods, "final_design": design,
        })
        power_path, power_hash = self.evidence("power", {
            "kind": "completed_joint_design_simulation", "status": "COMPLETE", "no_reserved_data_used": True,
            "actual_MC_replicates": 10000,
            "design_sha256": p.canonical_sha256({"R": R, "T_by_panel": T, "B": B,
                                                  "final_seed_ids": seeds, "rivals": list(p.RIVALS), "panels": list(p.PANEL_IDS)}),
            "per_contrast": contrasts, "joint_four_panel_decision": joint,
        })
        self.put(p.CONTROL_PATHS[4], {
            "status": "ADMITTED_FROZEN", "final_R": R, "final_T_by_panel": T, "final_B": B,
            "power_per_contrast": contrasts, "joint_power": joint,
            "completed_power_receipt_path": power_path, "completed_power_receipt_sha256": power_hash,
        })
        self.freeze()

    def put(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        self.documents[name] = value

    def evidence(self, name, value):
        path = "confirmation/evidence/" + name + ".json"
        self.put(path, value)
        return path, p.file_sha256(self.root / path)

    def freeze(self):
        files = {str(path.relative_to(self.root)): p.file_sha256(path)
                 for path in self.root.rglob("*") if path.is_file() and
                 path.name != "SOURCE_MANIFEST.json" and not str(path.relative_to(self.root)).startswith("private/")}
        self.put("SOURCE_MANIFEST.json", {"algorithm": "sha256", "files": files})


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.packet = PacketFixture(self.tmp.name)
        self.root = self.packet.root

    def report(self):
        return p.run_preflight(self.root)

    def codes(self, result):
        return {item["code"] for item in result["blockers"]}

    def test_complete_synthetic_receipts_allow_planning_only(self):
        result = self.report()
        self.assertEqual(result["blockers"], [])
        self.assertTrue(result["admission_ready"])
        self.assertFalse(result["confirmation_ready"])
        self.assertFalse(result["R3_accredited"])
        self.assertEqual(result["schedule"]["neural_fits"], 16 * 4 * 13)
        self.assertEqual(result["schedule"]["ARIMA_fits"], 4)
        self.assertEqual(result["schedule"]["analytic_trajectories"], 16)
        self.assertEqual(result["cost_estimate"]["active_compute_seconds"], 16 * 4 * 13 * 120 + 4 * 10 + 16 + 60)

    def test_current_draft_documents_block_without_modification(self):
        source = Path(__file__).resolve().parents[1]
        before = {}
        for relative in p.CONTROL_PATHS[2:]:
            path = source / relative
            self.assertTrue(path.is_file(), relative)
            content = path.read_bytes()
            before[relative] = content
            self.packet.put(relative, json.loads(content))
        self.packet.freeze()
        result = self.report()
        self.assertFalse(result["admission_ready"])
        self.assertIsNone(result["schedule"])
        self.assertIsNone(result["cost_estimate"])
        self.assertGreater(result["gate_counts"]["data"], 0)
        self.assertGreater(result["gate_counts"]["methods"], 0)
        self.assertGreater(result["gate_counts"]["power"], 0)
        for relative, content in before.items():
            self.assertEqual((source / relative).read_bytes(), content)

    def test_missing_rival_cannot_reduce_family_or_manufacture_sample_size(self):
        protocol = self.packet.documents[p.CONTROL_PATHS[2]]
        protocol["family"]["primary_rivals"].remove("iTransformer")
        protocol["family"]["primary_rival_count"] -= 1
        protocol["final_design"]["R"] = 1000000
        self.packet.put(p.CONTROL_PATHS[2], protocol)
        self.packet.freeze()
        result = self.report()
        self.assertIn("incomplete_or_changed_rival_family", self.codes(result))
        self.assertIn("final_seed_ids_not_admitted", self.codes(result))
        self.assertIn("power_evidence_design_mismatch", self.codes(result))
        self.assertIsNone(result["schedule"])

    def test_tampering_source_or_manifest_is_detected(self):
        (self.root / "runtime/candidate.py").write_text("tampered")
        result = self.report()
        self.assertIn("source_hash_mismatch", self.codes(result))
        self.assertFalse(result["admission_ready"])
        self.packet.freeze()
        result = self.report()
        self.assertIn("identity_runtime_hash_mismatch", self.codes(result))

    def test_reserved_payload_in_manifest_is_never_opened(self):
        secret = self.root / "private/F00.csv"
        secret.parent.mkdir()
        secret.write_text("DO NOT OPEN FINAL DATA")
        manifest = self.packet.documents["SOURCE_MANIFEST.json"]
        manifest["files"]["private/F00.csv"] = "0" * 64
        self.packet.put("SOURCE_MANIFEST.json", manifest)
        original = Path.open

        def guard(path, *args, **kwargs):
            if Path(path) == secret:
                raise AssertionError("Reserved test opened")
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", guard):
            result = self.report()
        self.assertEqual(result["reserved_payloads_opened"], 0)
        self.assertIn("payload_in_source_manifest_not_read", self.codes(result))

    def test_claimed_power_without_completed_receipt_does_not_admit(self):
        power = self.packet.documents[p.CONTROL_PATHS[4]]
        power.pop("completed_power_receipt_path")
        power.pop("completed_power_receipt_sha256")
        self.packet.put(p.CONTROL_PATHS[4], power)
        self.packet.freeze()
        result = self.report()
        self.assertFalse(result["admission_ready"])
        self.assertIn("completed_power_simulation_missing", self.codes(result))
        self.assertIsNone(result["cost_estimate"])

    def test_twenty_rivals_four_panels_required_in_power_receipt(self):
        power = self.packet.documents[p.CONTROL_PATHS[4]]
        receipt_path = power["completed_power_receipt_path"]
        receipt = self.packet.documents[receipt_path]
        receipt["per_contrast"].pop("F03::global_mse")
        self.packet.put(receipt_path, receipt)
        power["completed_power_receipt_sha256"] = p.file_sha256(self.root / receipt_path)
        self.packet.put(p.CONTROL_PATHS[4], power)
        self.packet.freeze()
        self.assertIn("full_family_power_evidence_missing", self.codes(self.report()))

    def test_final_seeds_must_be_frozen_bank_prefix(self):
        protocol = self.packet.documents[p.CONTROL_PATHS[2]]
        protocol["final_design"]["final_seed_ids"].reverse()
        self.packet.put(p.CONTROL_PATHS[2], protocol)
        self.packet.freeze()
        result = self.report()
        self.assertIn("candidate_seed_bank_prefix_mismatch", self.codes(result))
        self.assertIsNone(result["schedule"])

    def test_symlinks_and_escaping_paths_block_before_read(self):
        link = self.root / "runtime/alias.py"
        link.symlink_to("candidate.py")
        manifest = self.packet.documents["SOURCE_MANIFEST.json"]
        manifest["files"]["runtime/alias.py"] = "0" * 64
        manifest["files"]["../outside.py"] = "0" * 64
        self.packet.put("SOURCE_MANIFEST.json", manifest)
        self.assertIn("unsafe_or_missing_manifest_file", self.codes(self.report()))

    def test_cli_json_and_nonzero_for_pending_draft(self):
        protocol = self.packet.documents[p.CONTROL_PATHS[2]]
        protocol["status"] = "DRAFT"
        self.packet.put(p.CONTROL_PATHS[2], protocol)
        self.packet.freeze()
        result = subprocess.run([sys.executable, "-m", "runtime.preflight", "--root", str(self.root),
                                 "--output", "audit/preflight.json"], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(json.loads((self.root / "audit/preflight.json").read_text()), report)
        self.assertFalse(report["training_executed"])
        self.assertFalse(report["confirmation_ready"])

    def test_output_cannot_overwrite_input_or_follow_symlink(self):
        report = self.report()
        with self.assertRaises(p.UnsafePath):
            p.write_report(self.root, "confirmation/POWER_PLAN.json", report)
        (self.root / "audit").symlink_to(self.root / "confirmation", target_is_directory=True)
        with self.assertRaises(p.UnsafePath):
            p.write_report(self.root, "audit/elsewhere.json", report)

    def test_derived_output_can_be_regenerated_but_manifest_input_cannot(self):
        report = self.report()
        p.write_report(self.root, "audit/preflight.json", report)
        p.write_report(self.root, "audit/preflight.json", report)
        self.assertEqual(json.loads((self.root / "audit/preflight.json").read_text()), report)
        manifest = self.packet.documents["SOURCE_MANIFEST.json"]
        manifest["files"]["audit/preflight.json"] = p.file_sha256(self.root / "audit/preflight.json")
        self.packet.put("SOURCE_MANIFEST.json", manifest)
        with self.assertRaises(p.UnsafePath):
            p.write_report(self.root, "audit/preflight.json", report)


if __name__ == "__main__":
    unittest.main()
