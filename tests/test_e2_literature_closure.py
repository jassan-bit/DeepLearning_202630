"""Guard against false closure, denominator drift, and fabricated score ranks."""
import importlib.util
import json
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("literature_closure", Path(__file__).resolve().parents[1] / "scripts/e2_literature_closure.py")
closure = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(closure)


class LiteratureClosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = closure.build()

    def test_unknown_performance_is_interval_not_observed_zero(self):
        components = dict(C=1, U=1, P=None, A=1, F=1)
        self.assertEqual(closure.score_interval(components, closure.WEIGHTS["reference"]), (75, 100))
        self.assertIsNone(components["P"])

    def test_overlapping_scores_do_not_manufacture_order(self):
        self.assertEqual(closure.rank_bounds([(20, 80), (30, 70)]), [(1, 2), (1, 2)])
        self.assertEqual(closure.rank_bounds([(90, 100), (20, 30)]), [(1, 1), (2, 2)])

    def test_studies_systems_and_complete_readings_have_distinct_denominators(self):
        flow = json.loads(self.files["bounded_flow.json"])
        self.assertEqual(flow["complete_PDFs_received"], 10)
        self.assertEqual(flow["physical_pages"], 215)
        self.assertEqual(flow["current_Top10_systems"], 10)
        self.assertEqual(flow["current_Top10_distinct_primary_studies"], 9)
        self.assertEqual(len(flow["current_Top10_overlap_systems"]), 6)
        self.assertIsNone(flow["global_included_studies"])
        self.assertFalse(flow["global_review_complete"])

    def test_shared_source_and_new_web_read_have_no_invented_PDF_hash(self):
        rows = {r["model"]: r for r in json.loads(self.files["current10_extraction.json"])}
        self.assertEqual(rows["FCN-1D"]["study_id"], rows["ResNet-1D"]["study_id"])
        sae = rows["LSTM-autoencoder con predictor"]
        self.assertIsNone(sae["source_PDF_sha256"])
        self.assertEqual(sae["value_1"], 24.041)
        self.assertEqual(sae["value_2"], 12.060)
        self.assertIn("complete_primary_main_web_text", sae["read_level"])
        self.assertTrue(all(r["performance_component"] is None for r in rows.values()))

    def test_failed_native_queries_have_unknown_counts(self):
        log = json.loads(self.files["native_query_attempts.json"])
        self.assertEqual(len(log["queries"]), 3)
        for q in log["queries"]:
            self.assertIsNone(q["native_total"])
            self.assertIsNone(q["retrieved_records"])
            self.assertEqual(q["status"], "attempted_not_retrieved")

    def test_current_TiDE_uses_later_published_extension(self):
        tide = next(r for r in json.loads(self.files["current10_extraction.json"]) if r["model"] == "TiDE")
        self.assertEqual(tide["current_recipe"], "TIDE-D7-PUBLISHED-R0-v2")
        self.assertTrue(tide["published_body"])
        self.assertFalse(tide["NAS_required"])

    def test_selected_row_quality_denominator_not_current_catalogue(self):
        rows = json.loads(self.files["quality_coverage_architectural10.json"])
        counts = {r["field"]: r["status_counts"] for r in rows}
        self.assertEqual(counts["equal_hpo_budget"], {"NR": 10})
        self.assertEqual(counts["variance_selected_pair"], {"NR": 9, "reported": 1})
        self.assertTrue(all(sum(r["status_counts"].values()) == r["denominator"] for r in rows))


if __name__ == "__main__":
    unittest.main()
