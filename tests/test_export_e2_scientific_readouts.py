"""Protect literal requirements, original primary evidence and study identity."""
import copy
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("readout_export", ROOT / "scripts/export_e2_scientific_readouts.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EvidencePreservationTests(unittest.TestCase):
    def test_A3_rejects_a_primary_p_value_change(self):
        raw = (ROOT / MODULE.OLD / "delivery/a3_primary_contrasts_current.csv").read_bytes()
        MODULE.validate_primary_copy(raw, raw)
        changed = raw.replace(b"0.9617519124043797", b"0.0117519124043797", 1)
        self.assertNotEqual(raw, changed)
        with self.assertRaisesRegex(ValueError, "byte preservation"):
            MODULE.validate_primary_copy(raw, changed)

    def test_B21_literal_cannot_be_edited(self):
        rows = MODULE.parse_csv((ROOT / MODULE.OLD / "delivery/appendix_B_current.csv").read_bytes())
        changed = copy.deepcopy(rows)
        changed[-1]["requirement"] = "Run only a report."
        with self.assertRaisesRegex(ValueError, "literal Appendix B"):
            MODULE.validate_b_identity(rows, changed)

    def test_completed_CPU_fits_cannot_promote_B21_to_whole_campaign_complete(self):
        rows = MODULE.parse_csv((ROOT / MODULE.OLD / "delivery/appendix_B_current.csv").read_bytes())
        changed = copy.deepcopy(rows)
        changed[-1]["status"] = "completo"
        with self.assertRaisesRegex(ValueError, "whole historical campaign"):
            MODULE.validate_b_identity(rows, changed)

    def test_completed_CPU_RQ3_cannot_certify_failed_GPU_parity(self):
        state = {"original_GPU_parity_plan": {"all78_parity_valid": False},
                 "posterior_CPU_test_IUT": {"full_CPU_IUT_executed": True, "original_GPU_parity_gate_passed": False}}
        MODULE.validate_inference_distinction(state)
        state["original_GPU_parity_plan"]["all78_parity_valid"] = True
        with self.assertRaisesRegex(ValueError, "failed original GPU"):
            MODULE.validate_inference_distinction(state)

    def test_adverse_results_do_not_remove_completed_reporting_and_audited_pipeline(self):
        rows = MODULE.parse_csv((ROOT / MODULE.OLD / "delivery/appendix_B_current.csv").read_bytes())
        cpu = {"CPU_study_completed": True, "completed_neural_workers": 78,
               "completed_classical_workers": 3, "completed_scoring_stages": 2}
        actual = {row["id"]: row for row in MODULE.update_b(rows, cpu)}
        for item in ("B05", "B08", "B11", "B12", "B13", "B17"):
            self.assertEqual(actual[item]["status"], "completo")
        self.assertEqual({key for key, row in actual.items() if row["status"] == "parcial"},
                         {"B01", "B03", "B06", "B09", "B21"})
        self.assertIn("cercano al azar", actual["B03"]["remaining_limit"])
        self.assertIn("no demuestra equivalencia", actual["B03"]["remaining_limit"])
        self.assertIn("No se alcanzó potencia 0,80", actual["B08"]["remaining_limit"])


if __name__ == "__main__":
    unittest.main()
