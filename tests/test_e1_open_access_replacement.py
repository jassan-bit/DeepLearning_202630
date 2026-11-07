"""Integrity and scientific-scope checks; no papers, searches or model fitting."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_e1_open_access_replacement as amendment
import refresh_e1_native_downloads as publication


class OpenAccessReplacementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.stage = self.base / "stage"
        for name in amendment.INPUT_FILES:
            destination = self.stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(amendment.STAGE / name, destination)
        self.frozen = self.base / "frozen"
        for name in amendment.FROZEN_FILES:
            destination = self.frozen / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(amendment.BASE / name, destination)
        self.mirror = self.base / "public"

    def build(self):
        return amendment.build(self.stage, self.mirror, self.frozen)

    def reviews(self):
        return {r["group_id"]: r for r in json.loads((self.stage / "reviews.json").read_text())}

    def validate_modified(self, mutate):
        raw, frozen, web, uploads = amendment.load_inputs(self.stage, self.frozen)
        native = {r["group_id"]: r for r in json.loads(frozen["internet_closure_20261106/reviews.json"]) if r["scope"] == "native_22"}
        protocol = json.loads(raw["protocol.json"])
        mutate(web, uploads, native, protocol)
        amendment.validate_records(uploads, web, native, protocol)

    def test_previous_seventeen_reconcile_as_three_uploads_and_fourteen_not_obtained(self):
        summary = self.build()
        self.assertEqual(summary["initial_pending_partition"], {"uploaded_or_reprovided": 3, "not_obtained": 14})
        self.assertEqual(summary["native_queue_size"], 22)
        self.assertEqual(summary["native_status_counts"], {"closed_excluded_context": 8, "not_retrieved_user_reported_paywall": 14})
        self.assertEqual(summary["native_active_seek_at_cutoff"], 0)
        self.assertEqual(summary["native_pending_supplement_ids"], [])
        ledger = amendment.csv_read((self.stage / "native_dispositions.csv").read_bytes())
        self.assertEqual(len(ledger), 22)
        self.assertEqual({r["group_id"] for r in ledger}, amendment.NATIVE_IDS)
        missing = [r for r in ledger if r["group_id"] in amendment.NOT_RETRIEVED_IDS]
        self.assertTrue(all(r["strict_eligibility_closed"] == "false" for r in missing))
        self.assertTrue(all(r["strict_eligibility_decision"] == "not_assessed_original_not_obtained" for r in missing))
        self.assertTrue(all(r["exclusion_for_quality_without_fulltext"] == "false" for r in missing))

    def test_native_reading_is_unique_originals_not_rereads_or_partial_method(self):
        summary = self.build()
        self.assertEqual(summary["native_originals_read_ids"], sorted(amendment.NATIVE_ORIGINAL_IDS))
        self.assertEqual(summary["native_originals_read"], 6)
        self.assertEqual(summary["native_complete_document_read"], 7)
        self.assertEqual(summary["native_complete_mirror_transcription_ids"], ["E1N091"])
        self.assertEqual(summary["native_new_unique_original_ids"], ["E1N057", "E1N138"])
        self.assertEqual(summary["native_original_reread_ids"], ["E1N118"])
        records = self.reviews()
        self.assertFalse(records["E1N091"]["original_read"])
        self.assertFalse(records["E1N170"]["complete_document_read"])
        self.assertTrue(records["E1N170"]["strict_eligibility_closed"])

    def test_thirteen_round_fiches_separate_nine_web_bodies_eight_with_tables_and_partial(self):
        summary = self.build()
        self.assertEqual(summary["catalogue_sources_including_uploads"], 13)
        self.assertEqual(summary["web_catalog_sources"], 10)
        self.assertEqual(summary["web_full_originals_read"], 9)
        self.assertEqual(summary["web_original_body_and_tables_read"], 8)
        self.assertEqual(summary["web_original_body_only_without_tables_ids"], ["E1OA-T03"])
        self.assertEqual(summary["web_partial_only_ids"], ["E1X002"])
        records = self.reviews()
        self.assertFalse(records["E1OA-T03"]["quantitative_extraction_complete"])
        self.assertFalse(records["E1X002"]["original_read"])
        self.assertEqual(len(records), 32)

    def test_access_route_reuse_licence_and_direct_free_url_are_independent(self):
        self.build()
        catalog = {r["group_id"]: r for r in amendment.csv_read((self.stage / "sources.csv").read_bytes())}
        for group in amendment.UPLOADS:
            self.assertEqual(catalog[group]["source_scope"], "user_supplied_original")
            self.assertEqual(catalog[group]["free_fulltext_web_read_this_stage"], "false")
            self.assertEqual(catalog[group]["free_full_text_url"], "")
        self.assertEqual(catalog["E1N118"]["open_reuse_license_verified"], "true")
        self.assertEqual(catalog["E1N057"]["open_reuse_license_verified"], "false")
        self.assertEqual(catalog["E1N138"]["open_reuse_license_verified"], "false")
        self.assertEqual(catalog["E1OA-R02"]["open_reuse_license_verified"], "false")
        self.assertTrue(catalog["E1OA-R02"]["free_full_text_url"].endswith("ASC.pdf"))
        self.assertEqual(catalog["E1X002"]["free_full_text_url"], "")
        self.assertEqual({k for k, r in catalog.items() if r["free_full_text_url"]}, set(amendment.FREE_FULLTEXT_URLS))

    def test_source_reviews_and_frozen_previous_statuses_are_preserved_literally(self):
        historical = {name: (self.frozen / name).read_bytes() for name in amendment.FROZEN_FILES}
        self.build()
        records = self.reviews()
        old = {r["group_id"]: r for r in json.loads(historical["internet_closure_20261106/reviews.json"])}
        for group in amendment.NATIVE_IDS:
            self.assertEqual(records[group]["previous_cut_adjudication"], old[group])
        self.assertEqual(records["E1N118"]["previous_cut_adjudication"]["strict_eligibility_status"], "pending_target_supplement")
        self.assertEqual(records["E1N118"]["strict_eligibility_status"], "closed_excluded_context")
        for group, name in amendment.UPLOADS.items():
            self.assertEqual(records[group]["source_review"], json.loads((self.stage / name).read_text()))
        for name, data in historical.items():
            self.assertEqual(data, (self.frozen / name).read_bytes())

    def test_model_matrix_distinguishes_dccnn_hybrid_and_code_only_from_tcn(self):
        self.build()
        matrix = {(r["group_id"], r["model"]): r for r in amendment.csv_read((self.stage / "model_evidence.csv").read_bytes())}
        self.assertEqual(len(matrix), 32 * 7)
        self.assertIn("autonomous", matrix["E1OA-T02", "DCCNN"]["role_evidence"])
        self.assertIn("not_verified", matrix["E1OA-T02", "TCN"]["role_evidence"])
        self.assertIn("hybrid", matrix["E1OA-T03", "TCN"]["role_evidence"])
        self.assertIn("author_code_only", matrix["E1X002", "TCN"]["role_evidence"])
        self.assertEqual(matrix["E1X002", "TCN"]["full_original_read"], "false")
        self.assertTrue(all(r["strict_volatility_recurrence_increment"] == "0" for r in matrix.values()))
        self.assertTrue(all(r["recipe_for_simple_e1_comparison_verified"] == "false" for r in matrix.values()))

    def test_existing_sources_and_dependent_data_are_not_new_independent_replications(self):
        summary = self.build()
        self.assertEqual(summary["web_existing_source_upgrades"], ["E1T3", "E1X001"])
        self.assertEqual(len(summary["web_new_distinct_source_ids"]), 7)
        self.assertNotIn("E1OA-R02", summary["web_new_global_reference_ids"])
        records = self.reviews()
        self.assertIn("not an independent dataset replication", records["E1OA-R04"]["dependence_note"])
        self.assertTrue(records["E1OA-R03"]["full_text_read"])
        self.assertFalse(records["E1OA-R03"]["include_in_new_stratum"])

    def test_replacement_map_covers_fourteen_roles_without_bijective_papers(self):
        self.build()
        rows = amendment.csv_read((self.stage / "replacement_map.csv").read_bytes())
        self.assertEqual(len(rows), 14)
        self.assertEqual({r["native_id"] for r in rows}, amendment.NOT_RETRIEVED_IDS)
        self.assertTrue(all(r["equivalent_original_replacement_claimed"] == "false" for r in rows))
        self.assertTrue(all(r["mapping_unit"] == "substantive_role_not_one_to_one_paper" for r in rows))
        self.assertTrue(any("partial_only" in r["source_reading_states"] for r in rows))

    def test_historical_counts_frequencies_and_search_claims_are_unchanged(self):
        summary = self.build()
        self.assertEqual((summary["historical_exported_occurrences"], summary["historical_candidate_groups"], summary["historical_N7"]), (244, 182, 7))
        self.assertEqual({k: summary["historical_family_counts"][k] for k in ("LSTM", "GRU", "TCN")}, {"LSTM": 5, "GRU": 3, "TCN": 1})
        self.assertEqual(summary["native_new_strict_inclusions"], 0)
        self.assertFalse(summary["ieee_native_consultation_accredited_by_this_stage"])
        self.assertEqual(summary["native_searches_recovered_by_this_stage"], 0)
        self.assertEqual(summary["new_model_fits"], 0)

    def test_reproducible_exact_allowlist_hashes_bytes_and_mirror(self):
        self.build()
        before = {name: (self.stage / name).read_bytes() for name in amendment.PUBLIC_FILES | {"manifest.json"}}
        self.build()
        manifest = json.loads(before["manifest.json"])
        self.assertEqual(manifest["publication_schema"], publication.PUBLICATION_SCHEMA)
        self.assertEqual(manifest["approved_public_files"], sorted(amendment.PUBLIC_FILES))
        self.assertEqual(set(manifest["files"]), amendment.PUBLIC_FILES)
        for name, data in before.items():
            self.assertEqual((self.stage / name).read_bytes(), data)
            self.assertEqual((self.mirror / name).read_bytes(), data)
            if name != "manifest.json":
                self.assertEqual(manifest["files"][name], amendment.metadata(data))
        amendment.verify(self.stage, self.mirror, self.frozen)

    def test_generated_manifest_passes_existing_privacy_gate(self):
        self.build()
        original_root = publication.ROOT
        try:
            publication.ROOT = self.base
            files, info = publication.public_stage(self.stage, None)
        finally:
            publication.ROOT = original_root
        self.assertEqual(set(files), amendment.PUBLIC_FILES | {"manifest.json"})
        self.assertEqual(info["summary"]["native_not_retrieved"], 14)

    def test_input_and_historical_source_mutations_require_re_review(self):
        name = "part_open_recurrent/README.md"
        (self.stage / name).write_bytes((self.stage / name).read_bytes() + b"\nchanged\n")
        with self.assertRaisesRegex(ValueError, "Reviewed input changed"):
            self.build()
        shutil.copyfile(amendment.STAGE / name, self.stage / name)
        name = "internet_closure_20261106/summary.json"
        (self.frozen / name).write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "Frozen historical evidence changed"):
            self.build()

    def test_private_nested_payload_is_rejected_before_hash_comparison(self):
        path = self.stage / "part_upload_multitarget/review.json"
        source = json.loads(path.read_text())
        source["nested"] = {"abstract": "unapproved native export payload"}
        path.write_bytes(amendment.json_bytes(source))
        with self.assertRaisesRegex(ValueError, "Private/abstract"):
            self.build()

    def test_unapproved_original_pdf_neighbour_and_symlink_are_rejected(self):
        extra = self.stage / "part_open_tcn/original.pdf"
        extra.write_bytes(b"%PDF-private")
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.build()
        extra.unlink()
        self.build()
        target = self.base / "outside.txt"
        target.write_text("unchanged")
        path = self.mirror / "README.md"
        path.unlink()
        path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.build()
        self.assertEqual(target.read_text(), "unchanged")

    def test_modified_outputs_manifest_or_mirror_fail_verify(self):
        self.build()
        (self.mirror / "summary.json").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "mirror byte mismatch"):
            amendment.verify(self.stage, self.mirror, self.frozen)
        self.build()
        (self.stage / "summary.json").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "Compiled evidence/hash"):
            amendment.verify(self.stage, self.mirror, self.frozen)

    def test_duplicate_identity_and_changed_reading_state_cannot_be_promoted(self):
        with self.assertRaisesRegex(ValueError, "unique approved IDs"):
            self.validate_modified(lambda web, *_: web.append(copy.deepcopy(web[0])))
        def duplicate_doi(web, *_):
            with_doi = [r for r in web if r["source_review"].get("doi")]
            with_doi[0]["source_review"]["doi"] = with_doi[1]["source_review"]["doi"]
        with self.assertRaisesRegex(ValueError, "Duplicate web DOI"):
            self.validate_modified(duplicate_doi)
        def promote_partial(web, *_):
            next(r["source_review"] for r in web if r["source_review"]["group_id"] == "E1X002")["full_text_read"] = True
        with self.assertRaisesRegex(ValueError, "reading state"):
            self.validate_modified(promote_partial)


if __name__ == "__main__":
    unittest.main()
