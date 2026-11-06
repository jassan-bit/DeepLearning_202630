"""Meaningful import/deduplication boundaries for external bibliographic evidence."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from import_e2_wos import (application_scope, clean_cell, deduplicate, model_mentions,
                           normalize_doi, parse_wos_text, publication_metadata)


class WoSImportTests(unittest.TestCase):
    def export(self, body):
        return "\ufeffFN Clarivate Analytics Web of Science\nVR 1.0\n"+body+"\nEF\n"

    def test_continuations_keep_title_abstract_and_reference_lists_separate(self):
        rows, audit = parse_wos_text(self.export("PT J\nTI A wrapped\n   scientific title\nAB Forecasting with Mamba\n   and no contemporary target\nCR Smith A, 2024, PatchTST\n   Doe B, 2023, TimesNet\nUT WOS:1\nER"))
        self.assertEqual(rows[0]["TI"], "A wrapped scientific title")
        self.assertEqual(len(rows[0]["CR"]), 2)
        self.assertEqual(model_mentions(rows[0]), ["Mamba"])
        self.assertEqual(audit["complete_records"], 1)

    def test_truncated_nested_and_orphan_records_fail(self):
        for body in ("PT J\nTI Missing end", "PT J\nTI A\nPT J\nER", "ER"):
            with self.subTest(body=body), self.assertRaises(ValueError):
                parse_wos_text(self.export(body))
        with self.assertRaises(ValueError):
            parse_wos_text("FN Clarivate Analytics Web of Science\nPT J\nTI A\nER\n")

    def test_duplicate_ut_then_doi_with_title_fallback_and_conflict(self):
        records = [
            {"UT":"WOS:1","DI":"10.1/ABC","TI":"The same scientific article long title","PY":"2024"},
            {"UT":"WOS:1","DI":"10.1/abc","TI":"The same scientific article long title","PY":"2024"},
            {"UT":"WOS:2","DI":"https://doi.org/10.1/ABC","TI":"The same scientific article long title","PY":"2024"},
            {"UT":"WOS:3","TI":"Another scientific title with enough detail","PY":"2025"},
            {"UT":"WOS:4","DI":"10.2/new","TI":"Another scientific title with enough detail","PY":"2025"},
            {"UT":"WOS:5","DI":"10.3/different","TI":"The same scientific article long title","PY":"2024"},
        ]
        groups, edges, conflicts = deduplicate(records)
        self.assertEqual(sorted(map(len, groups)), [1,2,3])
        self.assertEqual({x["rule"] for x in edges}, {"exact_UT","normalized_DOI","normalized_title_and_year_fallback"})
        self.assertTrue(conflicts)

    def test_tide_ocean_and_s4_other_meanings_are_not_strict_model_matches(self):
        self.assertNotIn("TiDE", model_mentions({"TI":"Ocean tide forecasting","AB":"Forecast high tide with neural networks"}))
        self.assertIn("TiDE", model_mentions({"TI":"Forecasting with TiDE"}))
        self.assertNotIn("S4", model_mentions({"TI":"Treatment S4 for weather"}))
        self.assertIn("S4", model_mentions({"TI":"S4 structured state space model"}))

    def test_early_access_future_nominal_year_is_preserved_without_false_exclusion(self):
        r=publication_metadata({"PY":"2027","EA":"OCT 2026","TC":"12"})
        self.assertEqual(r["earliest_known_publication_year"],2026)
        self.assertEqual(r["publication_window_status"],"inside_via_early_access_year_needs_date_verification")
        self.assertEqual(r["citations_per_calendar_year"],12)
        self.assertIsNone(publication_metadata({"PY":"2027","TC":"2"})["citations_per_calendar_year"])

    def test_scope_terms_do_not_consume_cited_reference_mentions(self):
        r={"TI":"Rain prediction", "AB":"Weather forecasting", "CR":["Bitcoin volatility Mamba"]}
        self.assertEqual(application_scope(r),"general_forecasting_or_other_domain")
        self.assertEqual(model_mentions(r),[])

    def test_missing_citation_count_is_distinct_from_zero_and_sheet_formula_is_quoted(self):
        self.assertIsNone(publication_metadata({"PY":"2024"})["citations_per_calendar_year"])
        self.assertEqual(publication_metadata({"PY":"2024","TC":"0"})["citations_per_calendar_year"],0)
        self.assertEqual(clean_cell("=1+2"),"'=1+2")
        self.assertEqual(normalize_doi("https://doi.org/10.1000/ABC."),"10.1000/abc")


if __name__ == "__main__":
    unittest.main()
