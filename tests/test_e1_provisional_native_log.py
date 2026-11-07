"""Guard the boundary between saved exports and unverified search history."""
import csv
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_e1_provisional_native_log as provisional


class ProvisionalNativeLogTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base = self.root / "sources"
        for name in provisional.INPUT_SHA256:
            destination = self.base / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(provisional.BASE / name, destination)
        self.stage, self.mirror = self.root / "stage", self.root / "public"
        self.raw = provisional.load_inputs(self.base)

    def build(self):
        return provisional.build(self.stage, self.mirror, self.base)

    def test_bounds_count_distinct_native_ids_within_each_export(self):
        rows, _ = provisional.derive_rows(self.raw)
        by_key = {row["source_key"]: row for row in rows if row["source_key"]}
        self.assertEqual({key: row["export_compatible_lower_bound"] for key, row in by_key.items()},
                         {"W02": 8, "W01": 59, "S01": 19, "S02": 158})
        occurrences = provisional.csv_read(self.raw["native_search_20261106/occurrences.csv"])
        for key, row in by_key.items():
            ids = [item["source_id"] for item in occurrences if item["source_key"] == key]
            self.assertTrue(all(ids))
            self.assertEqual(row["exported_occurrences_verified"], len(ids))
            self.assertEqual(row["distinct_source_ids_verified"], len(set(ids)))
            self.assertEqual(row["lower_bound_display"], "≥" + str(len(set(ids))))
        # A repeated native ID must lower the bound even if the row count is unchanged.
        s01 = [r for r in occurrences if r["source_key"] == "S01"]
        s01[1]["source_id"] = s01[0]["source_id"]
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=occurrences[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(occurrences)
        changed = {**self.raw, "native_search_20261106/occurrences.csv": output.getvalue().encode()}
        derived, _ = provisional.derive_rows(changed)
        row = next(r for r in derived if r["source_key"] == "S01")
        self.assertEqual((row["exported_occurrences_verified"], row["export_compatible_lower_bound"]), (19, 18))

    def test_exact_import_and_historical_denominator_remain_separate_from_provisional_bounds(self):
        summary = self.build()
        self.assertEqual(summary["preserved_native_import"], {"exported_occurrences": 244, "candidate_groups_provisional": 182})
        self.assertEqual(summary["historical_e1"], {"crypto_full_text_denominator_preserved": 7,
                                                 "frequency_update_performed": False, "experiments_changed": False})
        self.assertEqual(summary["row_count"], 5)
        self.assertEqual(summary["status_counts"], {"provisional_lower_bound": 4, "planned_not_executed": 1})
        self.assertFalse(summary["objective_3_1_closed"])
        self.assertEqual(summary["log_status"], "PROVISIONAL_EDITABLE")
        self.assertTrue(summary["excluded_from_prisma_totals"])
        for field in ("new_native_search_executions", "verified_native_search_totals_added",
                      "verified_native_search_dates_added", "recovered_historical_query_literals_added"):
            self.assertEqual(summary[field], 0)

    def test_no_provisional_value_claims_verified_query_count_date_or_execution(self):
        summary = self.build()
        for row in summary["rows"]:
            for field in ("native_total_verified", "native_query_literal_verified", "native_search_date_verified",
                          "native_export_range_verified"):
                self.assertIsNone(row[field])
            self.assertFalse(row["native_query_literal_recovered"])
            self.assertFalse(row["native_execution_accredited_this_stage"])
            self.assertTrue(row["exclude_from_verified_native_search_totals"])
            self.assertTrue(row["exclude_from_prisma_totals"])
            self.assertTrue(row["editable"])
        csv_rows = provisional.csv_read((self.stage / "provisional_log.csv").read_bytes())
        self.assertTrue(all(row["native_total_verified"] == "" for row in csv_rows))
        self.assertTrue(all(row["native_query_literal_recovered"] == "false" for row in csv_rows))

    def test_ieee_unknown_is_null_and_never_a_fabricated_zero(self):
        summary = self.build()
        ieee = summary["rows"][-1]
        self.assertEqual(ieee["database"], "IEEE Xplore")
        self.assertEqual(ieee["status"], "planned_not_executed")
        self.assertEqual(ieee["lower_bound_display"], "Sin base para estimar")
        for field in ("exported_occurrences_verified", "distinct_source_ids_verified", "export_compatible_lower_bound",
                      "native_total_verified", "native_search_date_verified", "date_reference"):
            self.assertIsNone(ieee[field])
        self.assertIsNone(summary["ieee"]["export_compatible_lower_bound"])
        self.assertFalse(summary["ieee"]["execution_accredited"])

    def test_reference_dates_keep_their_actual_scope_and_never_add_native_times(self):
        rows, _ = provisional.derive_rows(self.raw)
        for row in rows[:4]:
            self.assertEqual(row["date_reference_display"], "06/11/2026 (referencia)")
            self.assertEqual(row["date_reference"], "2026-11-06")
            self.assertIsNone(row["native_search_date_verified"])
            if row["database"] == "Scopus":
                self.assertIsNone(row["date_reference_timezone"])
                self.assertEqual(row["date_reference_scope"], "export_filename_date_unknown_timezone")
            else:
                self.assertEqual(row["date_reference_scope"], "received_and_user_confirmation_date_not_execution_date")
                self.assertEqual(row["date_reference_timezone"], "America/Bogota (context only)")
        self.assertIsNone(rows[-1]["date_reference_display"])

    def test_proposals_merge_plan_families_in_the_correct_platform_field(self):
        rows, _ = provisional.derive_rows(self.raw)
        plans = {p["proposal_id"]: p for p in provisional.csv_read(self.raw["search_recovery_20261106/query_plan.csv"])}
        for row in rows:
            query = row["query_proposed"]
            self.assertEqual(query.count("("), query.count(")"))
            self.assertEqual(row["query_proposal_status"], "proposed_not_historical")
            self.assertEqual(row["publication_year_filter"], "2019–2026")
            if row["query_family"] == "TCN":
                self.assertEqual(query, plans[row["query_proposal_source_ids"]]["query_proposed"])
            if "GRU/LSTM" in row["query_family"]:
                self.assertIn('GRU OR "gated recurrent unit" OR "gated recurrent units" OR LSTM', query)
            if row["database"] == "Scopus":
                self.assertTrue(query.startswith("TITLE-ABS-KEY("))
                self.assertTrue(query.endswith(") AND PUBYEAR > 2018 AND PUBYEAR < 2027"))
            elif row["database"] == "Web of Science":
                self.assertTrue(query.startswith("TS=("))
                self.assertNotIn("PUBYEAR", query)
            else:
                self.assertEqual(row["field_scope"], "All Metadata")
                for family in ("TCN", "GRU", "LSTM"):
                    self.assertIn(family, query)
        self.assertTrue(all(r["publication_year_filter_status"] == "user_reported_not_independently_verified_in_export" for r in rows[:4]))

    def test_frozen_hashes_are_checked_and_build_never_changes_sources(self):
        before = {name: provisional.metadata((self.base / name).read_bytes()) for name in provisional.INPUT_SHA256}
        self.build()
        self.assertEqual(before, {name: provisional.metadata((self.base / name).read_bytes()) for name in provisional.INPUT_SHA256})
        path = self.base / "search_recovery_20261106/query_plan.csv"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "Frozen source hash mismatch"):
            provisional.build(self.stage, self.mirror, self.base)

    def test_closed_manifest_authenticates_only_derived_public_metadata(self):
        self.build()
        manifest = json.loads((self.stage / "manifest.json").read_bytes())
        self.assertEqual(manifest["publication_schema"], "e1_public_metadata_allowlist_v1")
        self.assertEqual(manifest["approved_public_files"], sorted(provisional.PUBLIC_FILES))
        self.assertEqual(set(manifest["files"]), provisional.PUBLIC_FILES)
        self.assertEqual(provisional.inventory(self.stage), provisional.PUBLIC_FILES | {"manifest.json"})
        for name, meta in manifest["files"].items():
            data = (self.stage / name).read_bytes()
            self.assertEqual(meta, provisional.metadata(data))
            text = data.decode()
            self.assertNotRegex(text, r"https?://|(?i:access_token|api_key|sessionid)=|\b2-s2\.0-|\bWOS:\d")
            self.assertNotRegex(text, r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
            if name.endswith(".json"):
                self.assertNotRegex(text, r'"(?:abstract|email|contact|full_text)"\s*:')
        provenance = json.loads((self.stage / "provenance.json").read_bytes())
        self.assertFalse(provenance["original_exports_read_this_stage"])
        self.assertFalse(provenance["original_exports_republished"])
        self.assertEqual(set(provenance["input_files"]), set(provisional.INPUT_SHA256))

    def test_build_is_idempotent_and_verify_detects_mirror_or_content_tampering(self):
        self.build()
        first = {name: (self.stage / name).read_bytes() for name in provisional.inventory(self.stage)}
        self.build()
        provisional.verify(self.stage, self.mirror, self.base)
        self.assertEqual(first, {name: (self.stage / name).read_bytes() for name in provisional.inventory(self.stage)})
        self.assertEqual(first, {name: (self.mirror / name).read_bytes() for name in provisional.inventory(self.mirror)})
        summary = json.loads((self.mirror / "summary.json").read_bytes())
        summary["ieee"]["native_total_verified"] = 0
        (self.mirror / "summary.json").write_bytes(provisional.json_bytes(summary))
        with self.assertRaisesRegex(ValueError, "hash differs"):
            provisional.verify(self.stage, self.mirror, self.base)

    def test_unapproved_neighbours_and_symlink_destinations_are_rejected_before_writing(self):
        self.stage.mkdir()
        (self.stage / "unreviewed.pdf").write_bytes(b"private")
        with self.assertRaisesRegex(ValueError, "Unapproved neighbouring"):
            self.build()
        self.assertFalse(self.mirror.exists())
        (self.stage / "unreviewed.pdf").unlink()
        target = self.root / "elsewhere"
        target.mkdir()
        self.mirror.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.build()
        self.assertEqual(list(target.iterdir()), [])
        self.assertEqual(list(self.stage.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
