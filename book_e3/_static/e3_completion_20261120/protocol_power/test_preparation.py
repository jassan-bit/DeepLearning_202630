"""Meaningful boundary tests for the fail-closed CPU preparation adapter."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cpu_preflight", ROOT / "preflight.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


class PreparationBoundaries(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((ROOT / "protocol.DRAFT.json").read_text())

    def test_current_draft_cannot_enable_training(self):
        report = preflight.assess(self.protocol, ROOT / "protocol.DRAFT.json")
        self.assertFalse(report["this_adapter_can_train"])
        self.assertFalse(report["training_started"])
        self.assertIn("scientific_go_and_reason_bound", report["blockers"])
        self.assertIn("full_family_prospective_power_bound", report["blockers"])
        self.assertEqual(report["status"], "BLOCKED_PREPARATION_ONLY")

    def test_changing_booleans_without_bound_evidence_does_not_pass(self):
        trial = copy.deepcopy(self.protocol)
        trial["scientific_decision"]["confirmation_go"] = True
        trial["budget"]["authorized"] = True
        trial["budget"]["approved_GPU_hours"] = 999
        trial["budget"]["cost_upper_template_seconds"] = 999
        trial["sampling"]["selected_final_R"] = 10
        trial["sampling"]["final_seed_ids"] = list(range(10))
        trial["sampling"]["current_pilot_seed_pairs"] = 5
        trial["sampling"]["current_pilot_complete_family_variances"] = True
        report = preflight.assess(trial, ROOT / "protocol.DRAFT.json")
        self.assertIn("scientific_go_and_reason_bound", report["blockers"])
        self.assertIn("author_budget_capacity_and_fairness_bound", report["blockers"])
        self.assertIn("full_family_prospective_power_bound", report["blockers"])

    def test_evidence_path_traversal_and_changed_hash_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bundle").mkdir()
            path = root / "bundle/protocol.json"
            outside = root / "outside.txt"
            outside.write_text("cannot bind an arbitrary outside file")
            trial = {"evidence_bindings": {"x": {"path": "../outside.txt", "sha256": preflight.digest(outside)}}}
            self.assertFalse(preflight.evidence_bound(trial, path, "x"))
            inside = root / "bundle/evidence.txt"
            inside.write_text("bound")
            trial["evidence_bindings"]["x"] = {"path": "evidence.txt", "sha256": preflight.digest(inside)}
            self.assertTrue(preflight.evidence_bound(trial, path, "x"))
            inside.write_text("changed")
            self.assertFalse(preflight.evidence_bound(trial, path, "x"))

    def test_resume_keeps_identity_and_refuses_protocol_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = root / "protocol.json"
            state = root / "state.json"
            protocol.write_text(json.dumps(self.protocol))
            first = preflight.prepare(protocol, state)
            second = preflight.prepare(protocol, state)
            self.assertEqual(first["identity"], second["identity"])
            self.protocol["sampling"]["selected_final_R"] = 20
            protocol.write_text(json.dumps(self.protocol))
            before = state.read_bytes()
            with self.assertRaisesRegex(ValueError, "identity changed"):
                preflight.prepare(protocol, state)
            self.assertEqual(before, state.read_bytes())

    def test_no_subset_of_the_main_family_passes(self):
        trial = copy.deepcopy(self.protocol)
        trial["family"]["rivals"] = trial["family"]["rivals"][:16]
        report = preflight.assess(trial, ROOT / "protocol.DRAFT.json")
        self.assertFalse(report["checks"]["complete_main_family19_57"])


if __name__ == "__main__":
    unittest.main()
