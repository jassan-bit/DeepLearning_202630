"""Substantive integrity checks for the post-experiment corpus expansion."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_e1_homogeneous_expansion as build


class CumulativeLiteratureIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = build.read(ROOT / "references/literature_e1/homogeneous_synthesis_20261107/part_corpus/adjudications.json")
        cls.evidence = build.read(ROOT / "references/literature_e1/free_model_expansion_20261107/readings.json")

    def merged(self, evidence=None):
        return build.merge_reviewed(self.old, evidence or self.evidence)

    def test_requested_thirteen_add_only_ten_and_preserve_every_historical_row(self):
        rows, duplicates, additions = self.merged()
        self.assertEqual(len(rows), 57)
        self.assertEqual(len(additions), 10)
        self.assertEqual(len(duplicates), 3)
        self.assertEqual(rows[:47], self.old["adjudications"])
        self.assertEqual({x["historical_id"] for x in duplicates}, {"S1", "PR_CVI", "E1OA-T01"})

    def test_a_publication_and_free_preprint_cannot_be_counted_as_two_works(self):
        evidence = copy.deepcopy(self.evidence)
        evidence["records"][-1] = copy.deepcopy(evidence["records"][-2])
        evidence["records"][-1]["id"] = "ANOTHER_FREE_VERSION"
        with self.assertRaisesRegex(ValueError, "identity missing or duplicated"):
            self.merged(evidence)

    def test_new_families_are_all_counted_not_just_the_requested_mlp(self):
        rows, _, _ = self.merged()
        frequencies = {r["id"]: r for r in build.build_frequencies(rows, self.old["frequencies"])}
        main = frequencies["crypto_construct_reference_reporting_sensitivity"]
        descriptive = frequencies["crypto_construct_reference_extractable"]
        self.assertEqual(main["denominator"], 10)
        self.assertEqual(descriptive["denominator"], 13)
        self.assertEqual({f: main["family_counts"][f]["count"] for f in ["LSTM", "MLP", "GRU", "TCN", "RNN_simple"]},
                         {"LSTM": 10, "MLP": 4, "GRU": 2, "TCN": 1, "RNN_simple": 1})
        self.assertEqual(descriptive["family_counts"]["CNN_generic"]["count"], 2)
        for family in ["RBFN", "GFN", "MFN"]:
            self.assertEqual(main["family_counts"][family]["count"], 1)
            self.assertIn(family, build.CONTEXTUAL_NEURAL)

    def test_financial_transfer_cannot_be_relabelled_as_crypto_to_raise_tcn_count(self):
        rows, _, additions = self.merged()
        item = next(r for r in additions if r["id"] == "E1F-T04")
        item["decision"]["continuous_crypto_construct_corpus"] = True
        changed = [item if r["id"] == item["id"] else r for r in rows]
        frequencies = build.build_frequencies(changed, self.old["frequencies"])
        with self.assertRaisesRegex(ValueError, "Crypto construct lacks primary operational evidence"):
            build.audit_new(changed, self.old, additions, frequencies)

    def test_internal_or_unverified_hybrid_component_is_not_a_baseline_vote(self):
        rows, _, additions = self.merged()
        item = next(r for r in additions if r["id"] == "E1F-T07")
        self.assertFalse(item["decision"]["baseline_role_verified"])
        item["independentfamilies"] = ["TCN"]
        changed = [item if r["id"] == item["id"] else r for r in rows]
        frequencies = build.build_frequencies(changed, self.old["frequencies"])
        with self.assertRaisesRegex(ValueError, "Unverified standalone family"):
            build.audit_new(changed, self.old, additions, frequencies)

    def test_reporting_exclusion_applies_to_new_lensky_as_to_historical_partial_recipes(self):
        rows, _, additions = self.merged()
        item = next(r for r in additions if r["id"] == "E1F-M03")
        self.assertTrue(item["decision"]["frequency_eligible"])
        self.assertFalse(item["decision"]["minimum_reporting_adequate_for_candidate_appraisal"])
        item["decision"]["guide31_reporting_sensitivity_eligible"] = True
        changed = [item if r["id"] == item["id"] else r for r in rows]
        frequencies = build.build_frequencies(changed, self.old["frequencies"])
        with self.assertRaisesRegex(ValueError, "Strict reporting inclusion"):
            build.audit_new(changed, self.old, additions, frequencies)

    def test_shared_experiment_cannot_add_an_independent_vote(self):
        rows, _, _ = self.merged()
        item = next(r for r in rows if r["id"] == "E1F-M01")
        item["identity"]["dependence_group"] = "S1"
        with self.assertRaisesRegex(ValueError, "Dependent experiment counted twice"):
            build.build_frequencies(rows, self.old["frequencies"])

    def test_primary_reading_is_not_inferred_from_indexed_open_access_metadata(self):
        rows, _, _ = self.merged()
        vo = next(r for r in rows if r["id"] == "E1F-T05")
        self.assertFalse(vo["decision"]["primary_document_read"])
        counts = build.summary_counts(rows)
        self.assertEqual(counts["primary_document_read_identities"], 35)
        self.assertEqual(counts["complete_mirror_only_identities"], 1)
        self.assertEqual(counts["crypto_construct_independent_experiments"], 15)
        financial = build.member(rows, "financial_transfer_primary_all")
        self.assertNotIn("E1F-T05", {r["id"] for r in financial})


if __name__ == "__main__":
    unittest.main()
