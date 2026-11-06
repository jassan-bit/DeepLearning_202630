"""Synthetic metadata/state guards; never decode real historical observations."""
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_final_test_data as state


def fixture(root):
    records = []
    for month in state.SOURCE_MONTHS:
        first = pd.Timestamp(month + "-01", tz="UTC")
        last = first + pd.offsets.MonthEnd(0)
        name = f"BTCUSDT-1m-{month}.zip"
        records.append({"archive": name, "kind": "monthly", "status": "verified",
                        "relative_path": f"BTCUSDT/1m/monthly/{name}", "bytes": 4,
                        "sha256": sha256(b"mock").hexdigest(),
                        "first_open_utc": first.isoformat(),
                        "last_open_utc": (last + pd.Timedelta(hours=23, minutes=59)).isoformat(),
                        "timestamp_units": ["ms"]})
    p = root / state.MANIFEST; p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"symbol": "BTCUSDT", "interval": "1m", "archives": records}))
    return p


class StateDataTests(unittest.TestCase):
    def test_reservation_is_twenty_one_months_and_never_reads_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture(root)
            reserve = state.reserve_final_state_metadata(root)
            self.assertEqual(len(reserve["records"]), 21)
            self.assertEqual(reserve["records"][0]["archive"], "BTCUSDT-1m-2024-12.zip")
            self.assertEqual(reserve["records"][-1]["archive"], "BTCUSDT-1m-2026-08.zip")
            self.assertEqual(reserve["archive_payloads_opened_by_reservation"], 0)
            self.assertIsNone(reserve["stat_verification"])

    def test_missing_month_and_wrong_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = fixture(root)
            original = json.loads(path.read_text()); original["archives"].pop()
            path.write_text(json.dumps(original))
            with self.assertRaises(ValueError):
                state.reserve_final_state_metadata(root)

    def test_seven_return_population_sigma_and_boundary_warmup(self):
        index = pd.date_range("2024-12-01", "2026-08-31", tz="UTC")
        increments = .001 * np.sin(np.arange(len(index)))
        closes = np.exp(4 + np.cumsum(increments))
        history = state.state_from_daily_closes(pd.DataFrame({"close": closes}, index=index))
        self.assertEqual(min(history), date(2024, 12, 25))
        self.assertEqual(max(history), date(2026, 8, 31))
        wanted = np.std(np.diff(np.log(closes[17:25])), ddof=0)
        self.assertAlmostEqual(history[date(2024, 12, 25)], wanted, places=13)

    def test_missing_close_preserves_dates_and_missing_target(self):
        index = pd.date_range("2024-12-01", "2026-08-31", tz="UTC")
        closes = np.exp(4 + .001 * np.arange(len(index)))
        frame = pd.DataFrame({"close": closes}, index=index)
        frame.loc["2025-03-01", "close"] = np.nan
        history = state.state_from_daily_closes(frame)
        self.assertIsNone(history[date(2025, 3, 1)])
        self.assertIsNone(history[date(2025, 3, 8)])
        self.assertIsNotNone(history[date(2025, 3, 9)])
        with self.assertRaises(ValueError):
            state.state_from_daily_closes(frame.dropna())

    def test_wrong_reservation_and_public_output_fail_before_archive_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture(root)
            reserve = state.reserve_final_state_metadata(root)
            with patch.object(state, "_aggregate_payload", side_effect=AssertionError("not allowed")):
                with self.assertRaises(ValueError):
                    state.load_final_state_history(root, root, reserve, root / "public")
                reserve["records"][0]["sha256"] = "f" * 64
                with self.assertRaises(ValueError):
                    state.load_final_state_history(root, root, reserve, root.parent / "private")


if __name__ == "__main__":
    unittest.main()
