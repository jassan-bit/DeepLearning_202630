"""Scientific-integrity checks for the documentary E2 preflight, with no model fits."""
from __future__ import annotations
import copy
from datetime import date, timedelta
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e2_benchmark_readiness", ROOT / "scripts/audit_e2_benchmark_readiness.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReadinessIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proposal = json.loads((ROOT / module.PROPOSAL).read_bytes())

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="e2-readiness-test-")
        self.root = Path(self.tmp.name)
        names = set(self.proposal["e1_source_receipts"]) | {module.PROPOSAL,
            "scripts/audit_e2_benchmark_readiness.py", "book_e2/_static/e1_reference.json",
            "book_e2/_static/dataset_card.json", "book_e2/_static/guia_e2_recibida.json"}
        for name in names:
            dest = self.root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, dest)
        self.config = copy.deepcopy(self.proposal)
        self.addCleanup(self.tmp.cleanup)

    def evaluate(self):
        return module.validate(self.config, self.root)

    def test_valid_documentary_plan_is_blocked_not_an_executed_experiment(self):
        report, models, requirements = self.evaluate()
        self.assertTrue(report["proposal_valid"])
        self.assertFalse(report["ready_to_train"])
        self.assertFalse(report["ready_to_claim_confirmatory_test"])
        self.assertEqual(report["pinned_E1_source_hashes_verified"], 6)
        self.assertEqual(len(models), 15)
        self.assertTrue(requirements)
        self.assertEqual(report["new_deep_learning_fits"], 0)

    def test_inherited_delta_or_calendar_cannot_be_changed(self):
        for key in ("delta", "calendar"):
            with self.subTest(key=key):
                self.config = copy.deepcopy(self.proposal)
                inherited = self.config["inherited_scientific_contract"]
                if key == "delta": inherited["practical_margin"]["absolute_delta"] *= 0.5
                else: inherited["partition"]["phase_II_evaluation"]["start"] = "2026-08-01"
                with self.assertRaisesRegex(ValueError, "inherited E1 scientific contract"):
                    self.evaluate()

    def test_target_lags_or_extra_channels_cannot_enter_silently(self):
        for key in ("lag", "feature"):
            with self.subTest(key=key):
                self.config = copy.deepcopy(self.proposal)
                if key == "lag": self.config["data_contract"]["label_protocol"]["target_lags_added"] = True
                else: self.config["data_contract"]["schema"]["model_features"].append("calendar_day")
                with self.assertRaisesRegex(ValueError, "D7×5"):
                    self.evaluate()

    def test_coordinated_reference_and_proposal_cannot_rewrite_the_inherited_calendar(self):
        reference_path = self.root / "book_e2/_static/e1_reference.json"
        reference = json.loads(reference_path.read_bytes())
        reference["partition"]["phase_II_evaluation"]["start"] = "2026-08-01"
        reference_path.write_bytes(module.json_bytes(reference))
        self.config["inherited_scientific_contract"] = module.scientific_snapshot(reference)
        self.config["declared_ready_to_train"] = True
        with self.assertRaisesRegex(ValueError, "pinned scientific snapshot"):
            self.evaluate()

    def test_prior_test_inspection_cannot_be_erased_by_a_flag(self):
        for key in ("new_unseen_test_assigned", "confirmatory", "historical_test_is_unseen", "test_selection_allowed"):
            with self.subTest(key=key):
                self.config = copy.deepcopy(self.proposal)
                self.config["evaluation"][key] = True
                with self.assertRaises(ValueError): self.evaluate()

    def test_duplicate_or_missing_comparators_are_rejected(self):
        self.config["models"][-1]["id"] = self.config["models"][0]["id"]
        with self.assertRaisesRegex(ValueError, "15 distinct"): self.evaluate()
        self.config = copy.deepcopy(self.proposal)
        self.config["models"] = self.config["models"][:-1]
        with self.assertRaisesRegex(ValueError, "15 distinct"): self.evaluate()

    def test_mutated_pinned_source_is_detected(self):
        name = next(iter(self.proposal["e1_source_receipts"]))
        path = self.root / name
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "source hash mismatch"): self.evaluate()

    def test_self_declared_readiness_does_not_create_missing_evidence(self):
        self.config["declared_ready_to_train"] = True
        for receipt in self.config["pretraining_receipts"].values(): receipt["status"] = "passed"
        report, rows, _ = self.evaluate()
        self.assertFalse(report["ready_to_train"])
        self.assertTrue(any(x["id"] == "self_declaration_ignored" for x in report["blockers"]))
        self.assertTrue(all(not x["D7_executable_adapter_verified"] for x in rows))

    def test_identical_HPO_claim_cannot_include_unchanged_anchors(self):
        self.config["hpo"]["declared_equal_budget_for_all15"] = True
        with self.assertRaisesRegex(ValueError, "frozen E1 references"): self.evaluate()

    def test_candidate_inventory_and_NAS_exemption_are_checked(self):
        self.config["hpo"]["candidates_per_family"] = 7
        with self.assertRaisesRegex(ValueError, "distinct proposed grid"): self.evaluate()
        self.config = copy.deepcopy(self.proposal)
        self.config["architecture_search_condition"]["all_adapters_exempt_from_section14"] = True
        with self.assertRaisesRegex(ValueError, "NAS rule"): self.evaluate()

    def test_E1_repetitions_do_not_establish_E2_power(self):
        self.config["repetitions"]["final_R"] = 465
        with self.assertRaisesRegex(ValueError, "E2 power"): self.evaluate()

    def test_calendar_has_strict_nested_supports_and_nominal90_semantics(self):
        rows = module.proposed_nested_calendar(self.config)
        self.assertEqual(len(rows), 30)
        self.assertEqual(len({r["outer_origin_UTC"] for r in rows}), 10)
        self.assertTrue(any(r["outer_validation_end_inclusive_UTC"] == "2024-02-29" for r in rows))
        for row in rows:
            q = date.fromisoformat(row["inner_validation_start_UTC"])
            self.assertEqual(date.fromisoformat(row["inner_train_outcome_max_UTC"]), q - timedelta(days=8))
            self.assertEqual(date.fromisoformat(row["early_stop_validation_start_UTC"]), q - timedelta(days=90))
            self.assertEqual(date.fromisoformat(row["early_stop_training_outcome_max_UTC"]), q - timedelta(days=98))
            self.assertEqual(row["early_stop_potential_calendar_days"], 83)
            self.assertLess(row["inner_validation_end_inclusive_UTC"], row["outer_origin_UTC"])
            self.assertFalse(row["actual_eligible_windows_audited"])
            self.assertFalse(row["tsxv_indices_generated"])

    def test_optimizer_fits_are_not_confused_with_evaluation_count(self):
        report, _, _ = self.evaluate()
        evaluations = report["proposed_validation_evaluation_count_inventory"]
        fits = report["proposed_optimizer_fit_count_inventory"]
        self.assertEqual(evaluations["subtotal_per_tuned_family_before_pilot_and_final"], 660)
        self.assertEqual(fits["subtotal_optimizer_fits_per_tuned_family_before_pilot_and_final"], 1320)
        self.assertEqual(fits["top10_subtotal_optimizer_fits_before_pilot_and_final"], 13200)
        self.assertEqual(report["new_deep_learning_fits"], 0)

    def test_cli_require_ready_refuses_execution_and_writes_only_metadata(self):
        script = self.root / "scripts/audit_e2_benchmark_readiness.py"
        process = subprocess.run([sys.executable, str(script), "--require-ready"], cwd=self.root,
                                 capture_output=True, text=True)
        self.assertEqual(process.returncode, 2, process.stderr)
        report = json.loads((self.root / module.OUTPUT / "report.json").read_bytes())
        provenance = json.loads((self.root / module.OUTPUT / "provenance.json").read_bytes())
        self.assertFalse(report["ready_to_train"])
        self.assertEqual(provenance["market_archives_opened"], 0)
        self.assertEqual(provenance["DL_packages_imported"], 0)
        self.assertEqual(provenance["model_fits"], 0)
        self.assertEqual(set(p.suffix for p in (self.root / module.OUTPUT).iterdir()), {".json", ".csv"})


if __name__ == "__main__": unittest.main()
