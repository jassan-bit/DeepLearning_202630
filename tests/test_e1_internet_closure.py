"""Evidence-integrity tests; no searches, original documents or model fitting."""
import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_e1_internet_closure as closure
import refresh_e1_native_downloads as publication


class InternetClosureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.stage = self.base / "stage"
        self.stage.mkdir()
        for name in closure.INPUT_FILES:
            destination = self.stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(closure.STAGE / name, destination)
        self.editorial = self.base / "editorial.csv"
        self.native = self.base / "native.csv"
        shutil.copyfile(closure.EDITORIAL, self.editorial)
        shutil.copyfile(closure.NATIVE, self.native)
        self.mirror = self.base / "public"

    def build(self):
        return closure.build(self.stage, self.mirror, self.editorial, self.native, search_stage=None)

    def verify(self):
        closure.verify(self.stage, self.mirror, self.editorial, self.native)

    def change_review(self, group_id, change):
        for name in closure.INPUT_FILES:
            if not name.endswith("/reviews.json"):
                continue
            path = self.stage / name
            rows = json.loads(path.read_text())
            for row in rows:
                if row["group_id"] == group_id:
                    change(row)
                    closure.json_write(path, rows)
                    return
        self.fail("Unknown fixture review")

    def test_counts_separate_reading_eligibility_and_complementary_scope(self):
        summary = self.build()
        self.assertEqual(summary["native_queue_size"], 22)
        self.assertEqual(summary["native_originals_read_ids"], ["E1N053", "E1N075", "E1N118", "E1N163"])
        self.assertEqual(summary["native_complete_document_read"], 5)
        self.assertEqual(summary["native_complete_mirror_transcription_ids"], ["E1N091"])
        self.assertEqual(summary["native_strict_closed_ids"], ["E1N053", "E1N075", "E1N091", "E1N163", "E1N170"])
        self.assertEqual(summary["native_strict_pending"], 17)
        self.assertIn("E1N118", summary["native_strict_pending_ids"])
        self.assertEqual(summary["complementary_ids"], ["E1X001", "E1X002", "E1X003", "E1X004"])
        self.assertEqual(summary["complementary_originals_read_ids"], ["E1X003"])
        self.assertFalse(summary["historical_N7_changed"])
        self.assertEqual(summary["native_new_strict_inclusions"], 0)
        self.verify()

    def test_source_fiches_preserved_but_current_decision_explicit(self):
        self.build()
        rows = {r["group_id"]: r for r in json.loads((self.stage / "reviews.json").read_text())}
        original = json.loads((self.stage / "part_a/reviews.json").read_text())[1]
        self.assertEqual(rows["E1N075"]["source_review"], original)
        self.assertEqual(rows["E1N075"]["strict_eligibility_decision"], "exclude_current_core_retain_context")
        self.assertTrue(rows["E1N075"]["original_read"])
        self.assertFalse(rows["E1N091"]["original_read"])
        self.assertTrue(rows["E1N091"]["complete_mirror_transcription_read"])
        self.assertFalse(rows["E1N170"]["complete_document_read"])
        self.assertTrue(rows["E1N170"]["strict_eligibility_closed"])
        self.assertTrue(rows["E1N118"]["original_read"])
        self.assertFalse(rows["E1N118"]["strict_eligibility_closed"])

    def test_component_and_partial_code_do_not_increase_tcn_recurrence(self):
        self.build()
        rows = closure.read_csv(self.stage / "model_evidence.csv")
        self.assertEqual(len(rows), 26 * 3)
        self.assertTrue(all(r["strict_volatility_recurrence_increment"] == "0" for r in rows))
        tcn = {r["group_id"]: r for r in rows if r["model"] == "TCN"}
        for group in ["E1N091", "E1N163", "E1X003"]:
            self.assertEqual(tcn[group]["autonomous_presence_evidence"], "component_only")
            self.assertEqual(tcn[group]["autonomy_verified_from_original"], "false")
        self.assertEqual(tcn["E1X002"]["autonomy_verified_in_author_code"], "true")
        self.assertEqual(tcn["E1X002"]["autonomy_verified_from_original"], "false")

    def test_exact_queue_and_unique_review_coverage_required(self):
        path = self.stage / "part_a/reviews.json"
        rows = json.loads(path.read_text())
        closure.json_write(path, rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.build()
        closure.json_write(path, rows)
        ledger = closure.read_csv(self.editorial)
        for row in ledger:
            if row["group_id"] == "E1N118":
                row["decision"] = "contextual_only"
        self.editorial.write_bytes(closure.csv_bytes(ledger))
        with self.assertRaisesRegex(ValueError, "exact frozen 22"):
            self.build()

    def test_no_relabeling_partial_or_author_access_without_review(self):
        self.change_review("E1N075", lambda row: row.update(full_text_read=False))
        with self.assertRaisesRegex(ValueError, "reading state changed"):
            self.build()

    def test_complementary_doi_cannot_duplicate_native(self):
        self.change_review("E1X002", lambda row: row.update(doi="10.1007/s10614-023-10392-5"))
        with self.assertRaisesRegex(ValueError, "already occurs"):
            self.build()

    def test_public_allowlist_hashes_and_mirror_are_closed_and_deterministic(self):
        self.build()
        manifest = json.loads((self.stage / "manifest.json").read_text())
        self.assertEqual(manifest["publication_schema"], "e1_public_metadata_allowlist_v1")
        self.assertEqual(manifest["approved_public_files"], sorted(manifest["files"]))
        self.assertEqual(set(manifest["reviewed_public_inputs"]), set(closure.INPUT_FILES))
        before = (self.stage / "manifest.json").read_bytes()
        self.build()
        self.assertEqual(before, (self.stage / "manifest.json").read_bytes())
        for name, saved in manifest["files"].items():
            self.assertEqual(saved, closure.metadata(self.stage / name))
            self.assertEqual((self.stage / name).read_bytes(), (self.mirror / name).read_bytes())
        (self.mirror / "decisions.csv").write_text("modified")
        with self.assertRaisesRegex(ValueError, "mirror byte mismatch"):
            self.verify()

    def test_modified_evidence_and_privacy_payloads_rejected(self):
        self.build()
        (self.stage / "summary.json").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "hash/size"):
            self.verify()
        self.change_review("E1N053", lambda row: row.update(abstract="private export payload"))
        with self.assertRaisesRegex(ValueError, "Private/abstract"):
            self.build()

    def test_unreviewed_neighbour_or_symlink_never_published(self):
        (self.stage / "part_a/original.pdf").write_bytes(b"%PDF-unreviewed")
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.build()
        (self.stage / "part_a/original.pdf").unlink()
        self.build()
        outside = self.base / "outside.txt"
        outside.write_text("unchanged")
        (self.mirror / "README.md").unlink()
        (self.mirror / "README.md").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.build()
        self.assertEqual(outside.read_text(), "unchanged")

    def test_search_schema_upgrade_preserves_evidence_and_mirror(self):
        search = self.base / "search"
        search.mkdir()
        for name in closure.SEARCH_FILES | {"manifest.json"}:
            shutil.copyfile(closure.SEARCH / name, search / name)
        evidence = {name: (search / name).read_bytes() for name in closure.SEARCH_FILES}
        mirror = self.base / "search_public"
        closure.update_search_manifest(search, mirror)
        manifest = json.loads((search / "manifest.json").read_text())
        self.assertEqual(manifest["publication_schema"], closure.PUBLICATION_SCHEMA)
        self.assertEqual(manifest["approved_public_files"], sorted(manifest["files"]))
        for name, data in evidence.items():
            self.assertEqual(data, (search / name).read_bytes())
            self.assertEqual(data, (mirror / name).read_bytes())
        (search / "summary.json").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "changed after its review"):
            closure.update_search_manifest(search, mirror)

    def test_generated_manifest_accepted_by_existing_download_privacy_gate(self):
        self.build()
        # public_stage expects its source within its ROOT; temporary fixture
        # still exercises identical recursive allowlist/privacy/hash checks.
        original_root = publication.ROOT
        try:
            publication.ROOT = self.base
            files, info = publication.public_stage(self.stage, None)
        finally:
            publication.ROOT = original_root
        self.assertEqual(set(files), closure.PUBLIC_FILES | {"manifest.json"})
        self.assertEqual(info["summary"]["native_strict_pending"], 17)


if __name__ == "__main__":
    unittest.main()
