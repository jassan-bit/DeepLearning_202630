"""Meaningful synthetic checks of the new historical-holdout access boundary."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import planned_study_data as loader


class PlannedStudyDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        cls.root = Path(cls.fixture.name)
        cls.raw = cls.root / "raw"
        records = []
        for kind, label, first, last in loader._identities():
            index = pd.date_range(first, pd.Timestamp(last) + pd.Timedelta(days=1), freq="min", inclusive="left", tz="UTC")
            day = (index.normalize() - pd.Timestamp(loader.CONTEXT_START, tz="UTC")).days.to_numpy()
            close = 100.0 * np.exp(.0001 * day ** 2)
            stamp = index.as_unit("us").asi8
            frame = pd.DataFrame({0:stamp, 1:close, 2:close+1, 3:close-1, 4:close,
                5:np.ones(len(index)), 6:stamp+60_000_000-1, 7:np.ones(len(index)),
                8:np.ones(len(index), dtype=int), 9:np.ones(len(index)),
                10:np.ones(len(index)), 11:np.zeros(len(index))})
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(f"BTCUSDT-1m-{label}.csv", frame.to_csv(index=False, header=False))
            payload = buffer.getvalue()
            relative = f"BTCUSDT/1m/{kind}/BTCUSDT-1m-{label}.zip"
            path = cls.raw / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            records.append({"archive":path.name, "kind":kind, "relative_path":relative,
                "url":"https://example.invalid/"+path.name, "checksum_url":"https://example.invalid/checksum",
                "sha256":hashlib.sha256(payload).hexdigest(), "bytes":len(payload), "status":"verified",
                "csv_rows":len(index), "expected_minutes":len(index), "first_open_utc":index[0].isoformat(),
                "last_open_utc":index[-1].isoformat(), "timestamp_units":["us"]})
        metadata = cls.root / loader.MANIFEST
        metadata.parent.mkdir(parents=True)
        metadata.write_text(json.dumps({"archives":records}))

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.output = tempfile.TemporaryDirectory()
        self.addCleanup(self.output.cleanup)
        self.out = Path(self.output.name)
        self.reservation = loader.reserve_metadata(self.root)

    def test_registration_is_metadata_only_and_reserves_exact_calendar(self):
        original = Path.read_bytes
        def guarded(path):
            if path.suffix == ".zip":
                raise AssertionError("Registration opened a raw payload")
            return original(path)
        with patch.object(Path, "read_bytes", guarded):
            reservation = loader.reserve_metadata(self.root, self.raw)
        self.assertEqual(reservation["calendar_days"], 60)
        self.assertEqual(reservation["archive_count"], 31)
        self.assertEqual(reservation["archives"][-1]["archive"], "BTCUSDT-1m-2026-10-30.zip")
        self.assertEqual(reservation["payloads_opened_by_reservation"], 0)

    def test_mixed_monthly_daily_cohort_and_verified_cache(self):
        data, audit = loader.load_reserved_data(self.root, self.raw, self.reservation, self.out)
        self.assertEqual(audit["archives_opened"], 32)
        self.assertEqual(len(data["test_indices"]), 60)
        first = int(data["test_indices"][0])
        self.assertEqual(data["calendar"][first].strftime("%Y-%m-%d"), "2026-09-01")
        closes = data["features"][first-7:first+1, 3]
        self.assertAlmostEqual(data["targets"][first], np.std(np.diff(np.log(closes)), ddof=0), places=12)
        np.testing.assert_equal(data["windows_by_history"][7][first], data["features"][first-7:first])
        with patch.object(loader, "_read_payload", side_effect=AssertionError("Cache reopened a ZIP")):
            restored, _ = loader.load_reserved_data(self.root, self.raw, self.reservation, self.out)
        np.testing.assert_array_equal(restored["features"], data["features"])
        np.testing.assert_array_equal(restored["targets"], data["targets"])

    def test_changed_reserved_identity_fails_before_payload_reads(self):
        altered = deepcopy(self.reservation)
        altered["archives"][0]["sha256"] = "0"*64
        with patch.object(loader, "_read_payload", side_effect=AssertionError("Premature payload read")):
            with self.assertRaisesRegex(ValueError, "reservation differs"):
                loader.load_reserved_data(self.root, self.raw, altered, self.out)
        self.assertFalse((self.out/"raw_access_started.json").exists())

    def test_bad_payload_preserves_access_and_does_not_silently_retry(self):
        with patch.object(loader, "_read_payload", return_value=b"changed"):
            with self.assertRaisesRegex(ValueError, "bytes/SHA256"):
                loader.load_reserved_data(self.root, self.raw, self.reservation, self.out)
        self.assertTrue((self.out/"raw_access_started.json").is_file())
        with patch.object(loader, "_read_payload", side_effect=AssertionError("Unsafe repeat")):
            with self.assertRaisesRegex(RuntimeError, "Interrupted first"):
                loader.load_reserved_data(self.root, self.raw, self.reservation, self.out)

    def test_missing_day_propagates_without_calendar_compression(self):
        daily = pd.DataFrame({"open":100., "high":101., "low":99., "close":100., "volume":1440.}, index=loader.CALENDAR)
        daily.loc[pd.Timestamp("2026-09-10", tz="UTC")] = np.nan
        data, cohort = loader._build_data(daily)
        self.assertEqual(len(data["calendar"]), 91)
        self.assertLess(cohort["n_common_outcomes"], 60)
        self.assertIn("2026-09-10", cohort["excluded_outcome_days_utc"])
        self.assertIn("2026-09-17", cohort["excluded_outcome_days_utc"])

    def test_primary_d7_does_not_require_unused_d28_context(self):
        daily = pd.DataFrame({"open":100., "high":101., "low":99., "close":100., "volume":1440.}, index=loader.CALENDAR)
        daily.loc[pd.Timestamp("2026-08-10", tz="UTC")] = np.nan
        data, cohort = loader._build_data(daily)
        self.assertEqual(cohort["primary_history_days"], 7)
        self.assertEqual(cohort["n_common_outcomes"], 60)
        self.assertLess(cohort["by_history"]["28"], 60)
        self.assertTrue(data["calendar"][data["test_indices"]].equals(loader.TEST_CALENDAR))

        # August24 is outside the first D7 input but is needed for the
        # September1 persistence value. That boundary must remain enforced.
        daily.loc[pd.Timestamp("2026-08-24", tz="UTC")] = np.nan
        affected, missing = loader._build_data(daily)
        first = int(loader.CALENDAR.get_loc(loader.TEST_CALENDAR[0]))
        self.assertTrue(np.isfinite(affected["targets"][first]))
        self.assertFalse(np.isfinite(affected["persistence"][first]))
        self.assertIn("2026-09-01", missing["excluded_outcome_days_utc"])

    def test_directory_escape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "relative"):
            loader._safe_path(self.raw, "../../outside.zip")


if __name__ == "__main__":
    unittest.main()
