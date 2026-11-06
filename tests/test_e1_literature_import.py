"""Meaningful parser, identity and disclosure checks for the E1 discovery tier."""

import csv
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("e1_import", ROOT / "scripts/import_e1_literature.py")
importer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = importer
SPEC.loader.exec_module(importer)


def scopus_text(rows):
    out = io.StringIO(newline="")
    columns = ["Title", "Year", "DOI", "EID", "Abstract", "Author Keywords", "Source title", "Document Type"]
    writer = csv.DictWriter(out, columns)
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def record(number, title="Bitcoin volatility prediction", doi="", source_id="", year="2026", abstract=""):
    return importer.Record(f"S01:{number:04d}", "S01", "Scopus", source_id, title, doi, year, abstract=abstract)


class ParserTests(unittest.TestCase):
    def test_scopus_missing_optional_fields_bom_and_quoted_title(self):
        text = '\ufeffTitle,DOI\n"Bitcoin, volatility",https://doi.org/10.1000/ABC\n'
        rows = importer.parse_scopus(text, "S01")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].title, "Bitcoin, volatility")
        self.assertEqual(rows[0].doi, "10.1000/abc")
        self.assertEqual(rows[0].source_id, "")
        self.assertEqual(rows[0].abstract, "")
        self.assertEqual(rows[0].occurrence_id, "S01:0001")

    def test_wos_continuations_repeated_fields_and_eof_record(self):
        text = """FN Clarivate Analytics Web of Science
VR 1.0
PT J
TI Forecasting realized
   volatility in Bitcoin
AB This is a wrapped
   abstract, not a new record.
DE GRU
DE LSTM
PY 2026
DI 10.1000/TEST
UT WOS:0001
ER

PT J
TI Second record with no ER
UT WOS:0002
"""
        rows = importer.parse_wos(text, "W01")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].title, "Forecasting realized volatility in Bitcoin")
        self.assertEqual(rows[0].abstract, "This is a wrapped abstract, not a new record.")
        self.assertEqual(rows[0].keywords.strip(), "GRU LSTM")
        self.assertEqual(rows[0].doi, "10.1000/test")
        self.assertEqual(rows[1].abstract, "")
        self.assertEqual(rows[1].occurrence_id, "W01:0002")

    def test_non_export_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Title column"):
            importer.parse_scopus("a,b\n1,2\n", "S01")
        with self.assertRaisesRegex(ValueError, "No WoS"):
            importer.parse_wos("An ordinary prose document", "W01")


class IdentityTests(unittest.TestCase):
    def test_exact_doi_links_even_when_titles_change(self):
        rows = [record(1, doi="10.1000/a"), record(2, title="Different editorial title", doi="10.1000/a")]
        groups, decisions = importer.deduplicate(rows)
        self.assertEqual(len(groups), 1)
        self.assertEqual(decisions[0]["method"], "exact_doi")
        self.assertEqual(decisions[0]["provisional"], "false")

    def test_title_normalization_is_only_provisional_for_distinct_eids(self):
        rows = [
            record(1, title="Enhancing: Cryptocurrency Price Prediction", source_id="2-s2.0-A", year="2024"),
            record(2, title="ENHANCING Cryptocurrency Price Prediction", source_id="2-s2.0-B", year="2025"),
        ]
        groups, decisions = importer.deduplicate(rows)
        self.assertEqual(len(groups), 1)
        self.assertEqual({r.source_id for r in groups[0]}, {"2-s2.0-A", "2-s2.0-B"})
        self.assertEqual(decisions[0]["method"], "normalized_title_provisional")
        self.assertEqual(decisions[0]["provisional"], "true")
        self.assertEqual(decisions[0]["reason"], "normalized_title_is_not_final_identity")

    def test_same_title_conflicting_dois_never_merges(self):
        groups, decisions = importer.deduplicate([record(1, doi="10.1000/a"), record(2, doi="10.1000/b")])
        self.assertEqual(len(groups), 2)
        self.assertEqual(decisions[0]["decision"], "retain_separate_doi_conflict")
        self.assertEqual(decisions[0]["provisional"], "true")

    def test_missing_titles_do_not_merge_unrelated_records(self):
        groups, decisions = importer.deduplicate([record(1, title=""), record(2, title="")])
        self.assertEqual(len(groups), 2)
        self.assertEqual(decisions, [])

    def test_price_prediction_with_volatility_context_is_not_direct_target(self):
        result = importer.screen_group([record(1, title="Bitcoin price prediction with GRU", abstract="Volatility makes prediction difficult. LSTM is discussed.")])
        self.assertEqual(result["scope_category"], "price_return_risk_title")
        self.assertEqual(result["eligibility_decision"], "undetermined")
        self.assertEqual(result["full_text_review_by_importer"], "not_performed")
        self.assertEqual(result["model_mentions_unverified"], "GRU;LSTM")

    def test_volatile_context_is_not_automatically_volatility_target(self):
        result = importer.screen_group([record(1, title="Forecasting volatile Bitcoin time series")])
        self.assertEqual(result["volatility_signal_title"], "false")
        self.assertNotEqual(result["review_priority"], "1")


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        # The sentinel includes private material that must not appear in any output.
        self.secret = "PRIVATE_ABSTRACT_SENTINEL author@example.invalid affiliation street 123"
        self.csv = self.base / "first.csv"
        self.csv.write_text(scopus_text([
            {"Title": "Bitcoin volatility prediction", "DOI": "10.1000/test", "EID": "2-s2.0-one", "Year": "2027", "Abstract": self.secret + " GRU"},
            {"Title": "Same title without DOI", "EID": "2-s2.0-two", "Year": "2024", "Abstract": self.secret},
            {"Title": "Same title without DOI", "EID": "2-s2.0-three", "Year": "2025", "Abstract": self.secret},
        ]), encoding="utf-8")
        self.wos = self.base / "second.txt"
        self.wos.write_text("PT J\nTI Bitcoin volatility prediction\nDI 10.1000/test\nUT WOS:one\nAB " + self.secret + " LSTM\nER\nEF\n", encoding="utf-8")
        self.specs = [("scopus", self.csv), ("wos", self.wos)]

    def tearDown(self):
        self.temp.cleanup()

    def generate(self):
        output, mirror = self.base / "output", self.base / "mirror"
        summary = importer.import_sources(self.specs, output, mirror, historical_path=None)
        return output, mirror, summary

    def refresh_hash(self, output, name):
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        raw = (output / name).read_bytes()
        manifest["files"][name] = {"sha256": importer.sha256(raw), "bytes": len(raw)}
        manifest_path.write_bytes(importer.json_bytes(manifest))

    def test_outputs_retain_occurrences_without_abstracts_or_final_decisions(self):
        output, mirror, summary = self.generate()
        self.assertEqual(summary["exported_occurrences"], 4)
        self.assertEqual(summary["candidate_groups_provisional"], 2)
        self.assertEqual(summary["groups_with_distinct_ids_requiring_identity_review"], 1)
        self.assertIsNone(summary["new_final_inclusions"])
        self.assertIsNone(summary["final_exclusions"])
        self.assertEqual(summary["historical_e1"]["crypto_full_text_denominator_preserved"], 7)
        self.assertEqual(importer.verify_public(output)["status"], "verified")
        for path in output.iterdir():
            raw = path.read_bytes()
            self.assertNotIn(self.secret.encode(), raw)
            self.assertNotIn(b"author@example.invalid", raw)
            self.assertEqual(raw, (mirror / path.name).read_bytes())
        rows = list(csv.DictReader(io.StringIO((output / "records.csv").read_text(encoding="utf-8"))))
        self.assertEqual(sum(int(r["occurrence_count"]) for r in rows), 4)
        self.assertTrue(any(r["identity_status"] == "provisional_title_link_distinct_ids" for r in rows))
        self.assertTrue(any(r["nominal_year_after_received_year"] == "true" for r in rows))
        sources = json.loads((output / "source_manifest.json").read_text())["sources"]
        self.assertEqual(sources[0]["sha256"], importer.sha256(self.csv.read_bytes()))
        self.assertIsNone(sources[0]["executed_query"])
        self.assertIsNone(sources[0]["applied_filters"])

    def test_regeneration_is_byte_identical_and_verification_needs_no_originals(self):
        output, _, _ = self.generate()
        before = {p.name: p.read_bytes() for p in output.iterdir()}
        importer.import_sources(self.specs, output, historical_path=None)
        self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})
        self.csv.unlink()
        self.wos.unlink()
        self.assertEqual(importer.verify_public(output)["occurrences"], 4)

    def test_verifier_detects_tampered_artifact(self):
        output, _, _ = self.generate()
        with (output / "records.csv").open("a") as f:
            f.write("tampered\n")
        with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
            importer.verify_public(output)

    def test_verifier_rejects_disallowed_abstract_column_even_with_updated_hash(self):
        output, _, _ = self.generate()
        path = output / "occurrences.csv"
        rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
        for r in rows:
            r["Abstract"] = self.secret
        path.write_bytes(importer.csv_bytes(rows, list(rows[0])))
        self.refresh_hash(output, path.name)
        with self.assertRaisesRegex(ValueError, "Unexpected public CSV fields"):
            importer.verify_public(output)

    def test_verifier_checks_cross_file_counts_not_only_hashes(self):
        output, _, _ = self.generate()
        path = output / "summary.json"
        summary = json.loads(path.read_text())
        summary["exported_occurrences"] += 1
        path.write_bytes(importer.json_bytes(summary))
        self.refresh_hash(output, path.name)
        with self.assertRaisesRegex(ValueError, "Summary/table counts differ"):
            importer.verify_public(output)

    def test_verifier_rejects_extra_raw_export_even_when_hashed(self):
        output, _, _ = self.generate()
        raw = self.csv.read_bytes()
        (output / "raw.csv").write_bytes(raw)
        path = output / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["files"]["raw.csv"] = {"sha256": importer.sha256(raw), "bytes": len(raw)}
        path.write_bytes(importer.json_bytes(manifest))
        with self.assertRaisesRegex(ValueError, "Unexpected public artifact names"):
            importer.verify_public(output)


if __name__ == "__main__":
    unittest.main()
