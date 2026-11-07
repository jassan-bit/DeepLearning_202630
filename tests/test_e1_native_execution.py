"""Synthetic-fixture checks for fresh native-search ingestion and disclosure."""
import copy
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ingest_e1_native_execution as native


class NativeExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.manifest_path = self.base / "private_manifest.json"
        self.output = self.base / "review"
        self.now = datetime(2026, 11, 7, 12, tzinfo=timezone.utc)
        self.secret = "PRIVATE_ABSTRACT_SENTINEL author@example.invalid street 123"
        self.historical_rows, _, self.history_hashes = native.load_history()
        self.historical = next(r for r in self.historical_rows if r["doi"])
        self.manifest = {"schema_version": "1.0", "campaign_id": "SYNTHETIC-TEST", "evidence": [], "executions": []}
        for code, database in (("W", "Web of Science"), ("S", "Scopus"), ("I", "IEEE Xplore")):
            xid, eid = "TEST-" + code, "EVIDENCE-" + code
            path = self.base / ("synthetic_evidence_" + code)
            # These bytes deliberately are NOT a real screenshot. The program checks
            # integrity/review assertions, not image content; real reviews are manual.
            path.write_bytes(("SYNTHETIC TEST EVIDENCE " + code).encode())
            self.manifest["evidence"].append({
                "evidence_id": eid, "execution_id": xid, "path": str(path),
                "sha256": native.sha256(path.read_bytes()), "kind": "screenshot",
                "collected_at": "2026-11-07T02:00:00Z", "reviewed_at": "2026-11-07T03:00:00Z",
                "verified_fields": sorted(native.FIELDS | {"export_range"}),
            })
            values = {
                "executed_at": "2026-11-06T20:00:00-05:00", "query_literal": "SYNTHETIC TEST QUERY " + code,
                "interface": "Synthetic test interface", "field_scope": "Synthetic test scope",
                "collection": "Synthetic test collection", "editions": ["SCI-EXPANDED"] if code == "W" else [],
                "filters": {"publication_year_start": 2019, "publication_year_end": 2026,
                            "document_types": ["all"], "other": []},
                "displayed_total": 0 if code == "I" else 1,
            }
            execution = {"execution_id": xid, "database": database, "status": "executed",
                         "verified": {k: {"value": v, "evidence_ids": [eid]} for k, v in values.items()},
                         "user_declared": {}, "exports": [], "attempt_evidence_ids": []}
            if code == "W":
                text = ("FN Clarivate Analytics Web of Science\nVR 1.0\nPT J\n"
                        "TI Synthetic new title matching a historical DOI\nDI " + self.historical["doi"] +
                        "\nPY 2026\nUT WOS:SYNTHETIC001\nAB " + self.secret + "\nER\nEF\n")
                execution["exports"] = [self.export("TEST-W-EXPORT", "wos_tagged", text, eid)]
            elif code == "S":
                text = self.scopus_text("Synthetic new independent volatility study", "10.9999/synthetic-new", "2-s2.0-SYNTHETIC")
                execution["exports"] = [self.export("TEST-S-EXPORT", "scopus_csv", text, eid)]
            self.manifest["executions"].append(execution)

    def scopus_text(self, title, doi, eid):
        out = io.StringIO(newline="")
        writer = csv.DictWriter(out, ["Title", "DOI", "Year", "EID", "Abstract", "Authors", "Correspondence Address"])
        writer.writeheader()
        writer.writerow({"Title": title, "DOI": doi, "Year": "2026", "EID": eid,
                         "Abstract": self.secret, "Authors": self.secret, "Correspondence Address": self.secret})
        return out.getvalue()

    def export(self, source, kind, text, evidence_id):
        path = self.base / (source + ".txt")
        path.write_text(text, encoding="utf-8")
        return {"export_id": source, "path": str(path), "sha256": native.sha256(path.read_bytes()),
                "format": kind, "range_start": 1, "range_end": 1, "range_evidence_ids": [evidence_id]}

    def only(self, code):
        xid = "TEST-" + code
        self.manifest["executions"] = [e for e in self.manifest["executions"] if e["execution_id"] == xid]
        self.manifest["evidence"] = [e for e in self.manifest["evidence"] if e["execution_id"] == xid]
        return self.manifest["executions"][0]

    def ingest(self):
        self.manifest_path.write_bytes(native.json_bytes(self.manifest))
        return native.ingest(self.manifest_path, self.output, now=self.now)

    def output_json(self, name):
        return json.loads((self.output / name).read_text())

    def test_complete_three_database_evidence_and_real_zero_preserve_history(self):
        summary = self.ingest()
        self.assertTrue(summary["all_three_native_metadata_verified"])
        self.assertTrue(summary["all_three_native_exports_complete"])
        self.assertEqual(summary["new_exported_occurrences"], 2)
        self.assertEqual(summary["global_occurrences"], 246)
        self.assertEqual(summary["global_candidate_groups_provisional"], 183)
        self.assertEqual(summary["new_round_flow"]["received_exported_occurrences"], 2)
        self.assertEqual(summary["new_round_flow"]["exact_doi_or_native_id_groups"], 2)
        self.assertEqual(summary["global_overlap"]["historical_groups_with_exact_doi_or_native_id_overlap"], 1)
        self.assertIsNone(summary["new_round_flow"]["new_studies_included"])
        self.assertEqual(summary["historical_crypto_fulltext_denominator_preserved"], 7)
        self.assertFalse(summary["training_performed"])
        self.assertIsNone(summary["new_final_inclusions"])
        run = self.output_json("execution.json")
        ieee = next(e for e in run["executions"] if e["database"] == "IEEE Xplore")
        self.assertEqual(ieee["verified"]["displayed_total"]["value"], 0)
        self.assertEqual(ieee["exports"], [])
        rows = list(csv.DictReader(io.StringIO((self.output / "occurrences.csv").read_text())))
        original = {r["occurrence_id"]: r for r in self.historical_rows}
        retained = [r for r in rows if r["origin"] == "historical_export"]
        self.assertEqual(len(retained), 244)
        for row in retained:
            self.assertEqual(row["historical_group_id"], original[row["occurrence_id"]]["group_id"])
            for field in ("source_key", "database", "source_id", "title", "doi", "year_reported"):
                self.assertEqual(row[field], original[row["occurrence_id"]][field])
        self.assertEqual(native.load_history()[2], self.history_hashes)
        public_manifest = self.output_json("manifest.json")
        for name, metadata in public_manifest["files"].items():
            data = (self.output / name).read_bytes()
            self.assertEqual(metadata, {"sha256": native.sha256(data), "bytes": len(data)})

    def test_private_abstract_contacts_paths_and_images_are_not_republished(self):
        self.ingest()
        self.assertEqual({p.name for p in self.output.iterdir()},
                         {"README.md", "schema.json", "execution.json", "summary.json", "occurrences.csv", "records.csv", "deduplication.csv", "manifest.json"})
        for path in self.output.iterdir():
            data = path.read_bytes()
            self.assertNotIn(self.secret.encode(), data)
            self.assertNotIn(b"author@example.invalid", data)
            self.assertNotIn(str(self.base).encode(), data)
            self.assertNotIn(b"SYNTHETIC TEST EVIDENCE", data)

    def test_offset_is_preserved_and_compared_as_an_instant(self):
        self.ingest()
        self.assertEqual(self.output_json("execution.json")["executions"][0]["verified"]["executed_at"]["value"],
                         "2026-11-06T20:00:00-05:00")

    def test_user_declarations_cannot_fill_missing_verified_fields(self):
        execution = self.only("S")
        execution["user_declared"] = {"collection": "Declared collection", "displayed_total": 999}
        execution["verified"]["collection"] = None
        execution["verified"]["displayed_total"] = None
        summary = self.ingest()
        result = self.output_json("execution.json")["executions"][0]
        self.assertIsNone(result["verified"]["displayed_total"])
        self.assertEqual(result["user_declared"]["displayed_total"], 999)
        self.assertFalse(result["native_metadata_complete"])
        self.assertFalse(result["native_export_complete"])
        self.assertFalse(summary["all_three_native_metadata_verified"])

    def test_declared_and_verified_disagreement_is_explicit(self):
        execution = self.only("S")
        execution["user_declared"] = {"displayed_total": 99}
        self.ingest()
        result = self.output_json("execution.json")["executions"][0]
        self.assertEqual(result["declaration_disagreements"], ["displayed_total"])
        self.assertEqual(result["verified"]["displayed_total"]["value"], 1)

    def test_partial_export_does_not_become_complete_or_estimate_total(self):
        execution = self.only("S")
        execution["verified"]["displayed_total"]["value"] = 4
        self.ingest()
        result = self.output_json("execution.json")["executions"][0]
        self.assertTrue(result["native_metadata_complete"])
        self.assertFalse(result["native_export_complete"])
        self.assertEqual(result["verified"]["displayed_total"]["value"], 4)

    def test_unreviewed_export_range_does_not_accredit_completeness(self):
        execution = self.only("S")
        execution["exports"][0]["range_evidence_ids"] = []
        self.ingest()
        self.assertFalse(self.output_json("execution.json")["executions"][0]["native_export_complete"])

    def test_ieee_access_failure_is_unknown_and_not_zero(self):
        execution = self.only("I")
        execution.update(status="blocked", verified=dict.fromkeys(native.FIELDS), attempt_evidence_ids=["EVIDENCE-I"])
        evidence = self.manifest["evidence"][0]
        evidence.update(kind="browser_error", verified_fields=["access_error"])
        summary = self.ingest()
        result = self.output_json("execution.json")["executions"][0]
        self.assertIsNone(result["verified"]["displayed_total"])
        self.assertFalse(result["native_metadata_complete"])
        self.assertFalse(summary["all_three_native_exports_complete"])

    def test_ieee_failure_cannot_claim_verified_or_declared_zero(self):
        execution = self.only("I")
        execution.update(status="blocked", attempt_evidence_ids=["EVIDENCE-I"])
        self.manifest["evidence"][0]["verified_fields"].append("access_error")
        with self.assertRaisesRegex(ValueError, "blocked or unexecuted"):
            self.ingest()
        execution["verified"] = dict.fromkeys(native.FIELDS)
        execution["user_declared"]["displayed_total"] = 0
        with self.assertRaisesRegex(ValueError, "blocked or unexecuted"):
            self.ingest()
        self.assertFalse(self.output.exists())

    def test_zero_without_query_and_date_cannot_accredit_execution_or_completeness(self):
        execution = self.only("I")
        total = execution["verified"]["displayed_total"]
        execution["verified"] = dict.fromkeys(native.FIELDS)
        execution["verified"]["displayed_total"] = total
        self.ingest()
        result = self.output_json("execution.json")["executions"][0]
        self.assertFalse(result["native_export_complete"])
        self.assertFalse(result["verified_zero_result_execution"])
        self.assertFalse(result["guide_3_1_search_log_complete"])

    def test_blocked_requires_evidence_of_the_actual_access_error(self):
        execution = self.only("I")
        execution.update(status="blocked", verified=dict.fromkeys(native.FIELDS), attempt_evidence_ids=["EVIDENCE-I"])
        with self.assertRaisesRegex(ValueError, "access_error evidence"):
            self.ingest()

    def test_day_and_supplied_literal_satisfy_guide_without_inventing_seconds(self):
        for execution in self.manifest["executions"]:
            total = execution["verified"]["displayed_total"]
            query = execution["verified"]["query_literal"]["value"]
            execution["verified"] = dict.fromkeys(native.FIELDS)
            execution["verified"]["displayed_total"] = total
            execution["user_declared"] = {"executed_on": "2026-11-06", "execution_date_timezone": "America/Bogota", "query_literal": query}
            for export in execution["exports"]:
                export.update(range_start=None, range_end=None, range_evidence_ids=[])
        summary = self.ingest()
        self.assertTrue(summary["guide_3_1_search_log_complete"])
        self.assertFalse(summary["all_three_native_metadata_verified"])
        self.assertFalse(summary["all_three_native_exports_complete"])
        for execution in self.output_json("execution.json")["executions"]:
            self.assertIsNone(execution["verified"]["executed_at"])
            self.assertEqual(execution["guide_3_1_provenance"]["execution_date"], "user_confirmed_day")
            self.assertTrue(execution["guide_3_1_search_log_complete"])

    def test_conversation_image_is_witnessed_without_fabricating_hash_or_bytes(self):
        execution = self.only("S")
        evidence = self.manifest["evidence"][0]
        evidence.update(kind="conversation_image", path=None, sha256=None,
                        conversation_ref="Synthetic test image in this test conversation",
                        review_provenance="Synthetic test witness")
        execution["exports"][0].update(range_start=None, range_end=None, range_evidence_ids=[])
        self.ingest()
        result = self.output_json("execution.json")["evidence"][0]
        self.assertIsNone(result["sha256"])
        self.assertIsNone(result["bytes"])
        self.assertEqual(result["integrity_status"], "conversation_image_witnessed_no_binary_hash_available")
        self.assertNotIn("path", result)

    def test_protected_historical_subdirectory_cannot_be_created_by_ingestion(self):
        self.manifest_path.write_bytes(native.json_bytes(self.manifest))
        with self.assertRaisesRegex(ValueError, "protected historical"):
            native.ingest(self.manifest_path, native.HISTORY / "SYNTHETIC-DO-NOT-CREATE", now=self.now)
        self.assertFalse((native.HISTORY / "SYNTHETIC-DO-NOT-CREATE").exists())

    def test_manifest_matches_the_published_json_schema(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("Optional independent JSON Schema checker is not installed")
        schema = json.loads((native.STAGE / "schema.json").read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(self.manifest)

    def test_user_transcribed_native_heading_is_not_relabelled_as_visual_evidence(self):
        execution = self.only("I")
        execution["verified"] = dict.fromkeys(native.FIELDS)
        execution["user_declared"] = {"query_literal": "SYNTHETIC IEEE QUERY", "executed_on": "2026-11-06", "displayed_total": 32}
        execution["reported_native_total_evidence_ids"] = ["EVIDENCE-I"]
        evidence = self.manifest["evidence"][0]
        evidence.update(kind="conversation_transcription", path=None, sha256=None,
                        conversation_ref="Synthetic user transcribed native heading: Showing 1-25 of 32 results",
                        verified_fields=["displayed_total"])
        self.ingest()
        result = self.output_json("execution.json")["executions"][0]
        self.assertIsNone(result["verified"]["displayed_total"])
        self.assertEqual(result["user_declared"]["displayed_total"], 32)
        self.assertEqual(result["native_displayed_total_provenance"], "user_transcribed_native_result_heading")
        self.assertTrue(result["guide_3_1_search_log_complete"])
        self.assertFalse(result["native_metadata_complete"])

    def test_context_day_is_tied_to_actual_export_filename_and_stays_distinct(self):
        execution = self.only("I")
        execution["verified"] = dict.fromkeys(native.FIELDS)
        execution["user_declared"] = {"query_literal": "SYNTHETIC IEEE QUERY", "displayed_total": 32}
        execution["reported_native_total_evidence_ids"] = ["EVIDENCE-I"]
        evidence = self.manifest["evidence"][0]
        evidence.update(kind="conversation_transcription", path=None, sha256=None,
                        conversation_ref="Synthetic user transcribed native heading: Showing 1-25 of 32 results",
                        verified_fields=["displayed_total"])
        export = self.export("TEST-I-EXPORT", "ieee_plain_text",
            'A. Synthetic, "Synthetic Bitcoin volatility title," in IEEE Access, vol. 1, pp. 1-9, 2026, doi: 10.9999/synthetic.\nkeywords: {private}\n', "EVIDENCE-I")
        path = Path(export["path"])
        named = path.with_name("Synthetic IEEE 2026.11.6.23.18.3.txt")
        path.rename(named)
        export.update(path=str(named), range_start=None, range_end=None, range_evidence_ids=[])
        execution["exports"] = [export]
        execution["execution_day_context"] = {"date": "2026-11-06", "basis": "native_export_filename_and_current_session_receipt", "source_export_id": "TEST-I-EXPORT", "timezone": "America/Bogota"}
        self.ingest()
        result = self.output_json("execution.json")["executions"][0]
        self.assertNotIn("executed_on", result["user_declared"])
        self.assertEqual(result["guide_3_1_provenance"]["execution_date"], "native_export_filename_and_current_session_receipt")
        self.assertTrue(result["guide_3_1_search_log_complete"])

    def test_unreviewed_or_wrong_field_evidence_cannot_verify_claim(self):
        self.only("S")
        self.manifest["evidence"][0]["verified_fields"].remove("query_literal")
        with self.assertRaisesRegex(ValueError, "not been reviewed"):
            self.ingest()
        self.assertFalse(self.output.exists())

    def test_cross_database_evidence_cannot_be_reused_as_proof(self):
        self.manifest["executions"][0]["verified"]["query_literal"]["evidence_ids"] = ["EVIDENCE-S"]
        with self.assertRaisesRegex(ValueError, "not been reviewed"):
            self.ingest()

    def test_chronology_requires_offset_campaign_date_and_no_future(self):
        execution = self.only("S")
        for invalid in ("2026-11-07T01:00:00", "2026-11-06T23:59:59Z", "2026-11-08T01:00:00Z"):
            execution["verified"]["executed_at"]["value"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.ingest()
        execution["verified"]["executed_at"]["value"] = "2026-11-07T02:30:00Z"
        with self.assertRaisesRegex(ValueError, "postdates"):
            self.ingest()

    def test_review_cannot_precede_evidence_collection(self):
        self.only("S")
        self.manifest["evidence"][0]["reviewed_at"] = "2026-11-07T01:00:00Z"
        with self.assertRaisesRegex(ValueError, "precedes collection"):
            self.ingest()

    def test_wos_editions_must_be_explicit(self):
        execution = self.only("W")
        execution["verified"]["editions"]["value"] = []
        with self.assertRaisesRegex(ValueError, "selected editions"):
            self.ingest()

    def test_hash_tampering_is_rejected_before_any_output(self):
        self.only("S")
        Path(self.manifest["evidence"][0]["path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.ingest()
        self.assertFalse(self.output.exists())

    def test_repository_raw_input_is_rejected(self):
        self.only("S")
        evidence = self.manifest["evidence"][0]
        evidence["path"] = str(native.STAGE / "schema.json")
        evidence["sha256"] = native.sha256(Path(evidence["path"]).read_bytes())
        with self.assertRaisesRegex(ValueError, "outside the checkout"):
            self.ingest()

    def test_export_range_must_match_row_count_and_verified_total(self):
        execution = self.only("S")
        export = execution["exports"][0]
        export["range_end"] = 2
        with self.assertRaisesRegex(ValueError, "cardinality"):
            self.ingest()
        export["range_start"] = 2
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.ingest()

    def test_ranges_cannot_overlap_and_export_bytes_cannot_be_counted_twice(self):
        execution = self.only("S")
        execution["verified"]["displayed_total"]["value"] = 2
        second = copy.deepcopy(execution["exports"][0])
        second.update(export_id="TEST-SECOND", range_start=2, range_end=2)
        execution["exports"].append(second)
        with self.assertRaisesRegex(ValueError, "same export bytes"):
            self.ingest()
        second = self.export("TEST-SECOND", "scopus_csv", self.scopus_text("Other synthetic title", "10.9999/other", "other"), "EVIDENCE-S")
        execution["exports"][1] = second
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.ingest()

    def test_source_keys_cannot_reuse_historical_identifiers(self):
        execution = self.only("S")
        execution["exports"][0]["export_id"] = "S01"
        with self.assertRaisesRegex(ValueError, "collision"):
            self.ingest()

    def test_title_conflicting_dois_stay_separate_during_global_deduplication(self):
        execution = self.only("S")
        execution["exports"][0] = self.export("TEST-S-EXPORT", "scopus_csv",
            self.scopus_text(self.historical["title"], "10.9999/conflicting-synthetic", "different"), "EVIDENCE-S")
        summary = self.ingest()
        self.assertEqual(summary["global_candidate_groups_provisional"], 183)
        decisions = (self.output / "deduplication.csv").read_text()
        self.assertIn("retain_separate_doi_conflict", decisions)

    def test_output_cannot_overwrite_an_existing_historical_directory(self):
        self.manifest_path.write_bytes(native.json_bytes(self.manifest))
        with self.assertRaisesRegex(ValueError, "empty"):
            native.ingest(self.manifest_path, native.HISTORY, now=self.now)
        self.assertEqual(native.load_history()[2], self.history_hashes)

    def test_unknown_private_fields_are_rejected(self):
        execution = self.only("S")
        execution["user_declared"]["author_email"] = self.secret
        with self.assertRaisesRegex(ValueError, "unknown object keys"):
            self.ingest()


class IEEEParserTests(unittest.TestCase):
    def test_plain_text_citations_extract_title_doi_year_only(self):
        text = ('A. Private, "Synthetic Bitcoin volatility title," in IEEE Access, vol. 1, pp. 2019-2026, 2025, doi: 10.9999/TEST.\n'
                'keywords: {private author@example.invalid}\n\n')
        record = native.parse_ieee_plain_text(text, "TEST-I")[0]
        self.assertEqual(record.title, "Synthetic Bitcoin volatility title")
        self.assertEqual(record.doi, "10.9999/test")
        self.assertEqual(record.year, "2025")
        self.assertEqual(record.source_id, "")
        self.assertEqual(record.abstract, "")
        self.assertEqual(record.keywords, "")

    def test_native_id_comes_from_link_and_document_identifier_is_not_an_id(self):
        text = ('Document Title,DOI,Publication Year,Document Identifier,PDF Link,Abstract\n'
                'Synthetic title,10.9999/a,2026,Journals,https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=12345,private\n')
        record = native.parse_ieee(text, "TEST-I")[0]
        self.assertEqual(record.source_id, "12345")
        self.assertEqual(record.document_type, "Journals")
        self.assertEqual(record.abstract, "")
        self.assertEqual(record.keywords, "")

    def test_type_without_link_does_not_become_unique_id(self):
        record = native.parse_ieee("Document Title,Document Identifier\nSynthetic title,Journals\n", "TEST-I")[0]
        self.assertEqual(record.source_id, "")


if __name__ == "__main__":
    unittest.main()
