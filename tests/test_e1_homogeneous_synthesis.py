"""Integrity checks against documentary double counting and selective strata."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("homogeneous", ROOT / "scripts/build_e1_homogeneous_synthesis.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class HomogeneousDocumentaryIntegrity(unittest.TestCase):
    def setUp(self):
        base = module.STAGE
        self.corpus = json.loads((base / "part_corpus/adjudications.json").read_text())
        self.protocol = json.loads((base / "part_corpus/protocol.json").read_text())
        self.mechanisms = json.loads((base / "part_mechanisms/mechanisms.json").read_text())

    def test_recomputed_membership_preserves_partial_reading_and_historical_cuts(self):
        result = module.audit(self.corpus, self.protocol, self.mechanisms)
        self.assertTrue(result["passed"])
        self.assertEqual(result["identities"], 47)
        self.assertEqual(self.corpus["summary"]["primary_document_read_identities"], 26)
        self.assertLess(self.corpus["summary"]["primary_document_read_identities"], result["identities"])
        self.assertTrue(result["historical_N7_unchanged"])

    def test_repeated_version_cannot_be_counted_as_another_original(self):
        self.corpus["adjudications"][1]["identity"]["canonical_key"] = self.corpus["adjudications"][0]["identity"]["canonical_key"]
        with self.assertRaisesRegex(ValueError, "Repeated identity"):
            module.audit(self.corpus, self.protocol, self.mechanisms)

    def test_two_documents_reusing_one_experiment_do_not_get_two_votes(self):
        self.corpus["adjudications"][1]["identity"]["dependence_group"] = "S1"
        with self.assertRaisesRegex(ValueError, "Dependent experiment"):
            module.audit(self.corpus, self.protocol, self.mechanisms)

    def test_historical_and_new_insufficient_recipes_receive_same_sensitivity_rule(self):
        rows = self.corpus["adjudications"]
        inclusive = {r["id"] for r in module.membership(rows, "crypto_construct_reference_extractable")}
        strict = {r["id"] for r in module.membership(rows, "crypto_construct_reference_reporting_sensitivity")}
        self.assertEqual(inclusive - strict, {"S8", "E1N138"})
        # Silently filtering one source without re-adjudicating its eligibility fails.
        frequency = next(f for f in self.corpus["frequencies"] if f["id"] == "crypto_construct_reference_reporting_sensitivity")
        frequency["records"].append("E1N138")
        with self.assertRaisesRegex(ValueError, "Wrong stratum membership"):
            module.audit(self.corpus, self.protocol, self.mechanisms)

    def test_positive_family_count_requires_independent_comparator_evidence(self):
        frequency = next(f for f in self.corpus["frequencies"] if f["id"] == "crypto_construct_all")
        frequency["family_counts"]["TCN"]["experiments"].append("S9")
        frequency["family_counts"]["TCN"]["count"] += 1
        with self.assertRaisesRegex(ValueError, "Invalid positive comparator count"):
            module.audit(self.corpus, self.protocol, self.mechanisms)

    def test_no_minimal_or_top_three_selection_claim_is_generated(self):
        outputs = module.compile_outputs()
        summary = json.loads(outputs["summary.json"])
        self.assertEqual(summary["frequency_top_three"], ["LSTM", "GRU", "MLP"])
        self.assertEqual(summary["retained_registered_trio"], ["GRU", "LSTM", "TCN"])
        self.assertFalse(summary["claims"]["three_most_frequent_selected"])
        self.assertFalse(summary["claims"]["scientific_optimality_certified"])
        manifest = json.loads(outputs["manifest.json"])
        self.assertEqual(set(manifest["approved_public_files"]), set(outputs) - {"manifest.json"})
        self.assertTrue(all(not name.lower().endswith((".pdf", ".pt")) for name in outputs))


if __name__ == "__main__":
    unittest.main()
