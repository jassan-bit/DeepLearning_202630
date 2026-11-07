"""Integrity checks for evidence units and qualified model roles."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("free_expansion", ROOT / "scripts/build_e1_free_model_expansion.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FreeModelExpansionTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((MODULE.STAGE / "inputs.json").read_text())

    def row(self, ident):
        return next(r for r in self.data["records"] if r["id"] == ident)

    def test_intake_deduplicates_publication_and_author_versions(self):
        self.data["records"][-1]["doi"] = self.data["records"][0]["doi"]
        with self.assertRaises(ValueError):
            MODULE.validate(self.data)

    def test_selected_counts_are_not_prevalence_and_history_stays_distinct(self):
        rows = MODULE.validate(self.data)
        self.assertEqual(sum(not r["existing_identity"] for r in rows), 10)
        self.assertEqual(sum(r["decision"]["continuous_crypto_construct_corpus"] and not r["existing_identity"] for r in rows), 3)

    def test_partial_indexed_primary_cannot_be_promoted_to_full_read(self):
        self.row("E1F-T05")["decision"]["primary_document_read"] = True
        with self.assertRaises(ValueError):
            MODULE.validate(self.data)

    def test_hybrid_component_cannot_become_a_basic_tcn_vote(self):
        self.row("E1F-T07")["independentfamilies"].append("TCN")
        with self.assertRaises(ValueError):
            MODULE.validate(self.data)

    def test_non_searched_comparator_families_must_not_be_dropped(self):
        self.row("E1F-M03")["independentfamilies"] = ["MLP"]
        with self.assertRaises(ValueError):
            MODULE.validate(self.data)

    def test_incomplete_lensky_recipe_retains_reporting_limit(self):
        self.row("E1F-M03")["decision"]["guide31_reporting_sensitivity_eligible"] = True
        with self.assertRaises(ValueError):
            MODULE.validate(self.data)

    def test_dependency_check_is_required_for_shared_author_crypto_paper(self):
        self.row("E1F-M02").pop("dependency_check")
        with self.assertRaises(ValueError):
            MODULE.validate(self.data)

    def test_ge_remains_preprint(self):
        self.row("E1F-T06")["genre"] = "publication"
        with self.assertRaises(ValueError):
            MODULE.validate(self.data)


if __name__ == "__main__":
    unittest.main()
