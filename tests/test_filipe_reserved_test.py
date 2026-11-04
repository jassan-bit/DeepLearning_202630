"""Synthetic-only checks for the final freeze gate and minute-to-day test path."""
from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import load_filipe_reserved_test as loader
import final_fit_filipe_multihistory as final


def synthetic_minute_frame():
    index = pd.date_range("2025-01-01", "2025-02-01", inclusive="left", freq="min", tz="UTC")
    # Explicit integer microseconds; Python Timestamp.value always uses ns.
    times = np.array([t.value // 1000 for t in index], dtype=np.int64)
    close = 100 + np.arange(len(index)) * .001
    return pd.DataFrame({"open_time": times, "open": close, "high": close + 1,
        "low": close - 1, "close": close, "volume": np.ones(len(index)),
        "close_time": times + 59_999_999, "quote_volume": close,
        "trades": np.ones(len(index), dtype=int), "taker_base": np.zeros(len(index)),
        "taker_quote": np.zeros(len(index)), "ignore": np.zeros(len(index))})


def zip_payload(frame):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("BTCUSDT-1m-2025-01.csv", frame.to_csv(index=False, header=False))
    return buffer.getvalue()


def market_fixture():
    index = pd.date_range("2023-01-01", "2026-09-01", inclusive="left", tz="UTC")
    returns = .002 * np.sin(np.arange(len(index)) * .31) + .0002
    close = 100 * np.exp(np.cumsum(returns))
    market = pd.DataFrame({"open": close * .999, "high": close * 1.01,
        "low": close * .99, "close": close, "volume": 1 + np.arange(len(index))}, index=index)
    return market.loc[:"2024-12-31"].copy(), market.loc["2025-01-01":].copy()


class MinuteAggregationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = synthetic_minute_frame()

    def test_microsecond_bars_form_complete_days_and_expected_close(self):
        payload = zip_payload(self.raw)
        daily, quality = loader.aggregate_month_payload(payload, "2025-01", expected_sha256=hashlib.sha256(payload).hexdigest())
        self.assertEqual(quality["timestamp_unit"], "us")
        self.assertEqual(quality["complete_days"], 31)
        self.assertEqual(quality["missing_minutes"], 0)
        self.assertEqual(daily.iloc[0]["close"], self.raw.iloc[1439]["close"])
        self.assertEqual(daily.iloc[0]["volume"], 1440)

    def test_missing_minute_excludes_entire_day_without_compressing_grid(self):
        daily, quality = loader.aggregate_month_payload(zip_payload(self.raw.drop(index=[100])), "2025-01")
        self.assertEqual(len(daily), 31)
        self.assertEqual(quality["missing_minutes"], 1)
        self.assertEqual(quality["excluded_days_utc"], ["2025-01-01"])
        self.assertTrue(daily.iloc[0].isna().all())
        self.assertTrue(daily.iloc[1].notna().all())

    def test_incorrect_close_timestamp_excludes_day(self):
        changed = self.raw.copy()
        changed.loc[1600, "close_time"] += 1
        daily, quality = loader.aggregate_month_payload(zip_payload(changed), "2025-01")
        self.assertEqual(quality["invalid_close_time_minutes"], 1)
        self.assertTrue(daily.iloc[1].isna().all())

    def test_duplicate_and_off_grid_timestamps_are_rejected(self):
        duplicate = pd.concat([self.raw.iloc[:1], self.raw], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            loader.aggregate_month_payload(zip_payload(duplicate), "2025-01")
        off_grid = self.raw.copy()
        off_grid.loc[0, "open_time"] += 1
        with self.assertRaisesRegex(ValueError, "aligned"):
            loader.aggregate_month_payload(zip_payload(off_grid), "2025-01")

    def test_wrong_hash_and_month_are_rejected(self):
        payload = zip_payload(self.raw)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            loader.aggregate_month_payload(payload, "2025-01", expected_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "outside"):
            loader.aggregate_month_payload(payload, "2025-02")


class CohortAndGateTests(unittest.TestCase):
    def test_all_histories_share_full_calendar_and_inputs_end_before_outcome(self):
        development, test = market_fixture()
        data, eligibility, manifest = loader.build_combined_data(development, test)
        self.assertEqual(manifest["calendar_days"], 608)
        self.assertEqual(manifest["n_common_outcomes"], 608)
        self.assertEqual(len(eligibility), 4 * 608)
        first = data["test_indices"][0]
        for history in loader.HISTORIES:
            np.testing.assert_array_equal(data["windows_by_history"][history][first],
                development.to_numpy()[-history:])
        returns = np.log(pd.concat([development, test]).close).diff()
        self.assertAlmostEqual(data["targets"][first], np.std(returns.iloc[first - 6:first + 1], ddof=0))

    def test_missing_day_propagates_to_targets_and_common_history(self):
        development, test = market_fixture()
        test.loc[pd.Timestamp("2025-01-10", tz="UTC"), :] = np.nan
        data, eligibility, manifest = loader.build_combined_data(development, test)
        excluded = manifest["excluded_common"]
        self.assertIn("2025-01-10", excluded)
        self.assertIn("2025-02-07", excluded)
        self.assertNotIn("2025-02-08", excluded)
        self.assertTrue(any("D28" in reason for reason in excluded["2025-02-07"]))
        day = data["calendar"].get_loc(pd.Timestamp("2025-01-11", tz="UTC"))
        self.assertFalse(np.isfinite(data["targets"][day]))

    def test_frozen_preprocessing_ignores_current_and_future_prices(self):
        development, test = market_fixture()
        original, _, _ = loader.build_combined_data(development, test)
        changed_test = test.copy()
        changed_test.loc["2025-01-20":, ["open", "high", "low", "close"]] *= 100
        changed, _, _ = loader.build_combined_data(development, changed_test)
        day = original["calendar"].get_loc(pd.Timestamp("2025-01-20", tz="UTC"))
        preproc = {"history_days": 28, "x_mean": [0] * 5, "x_scale": [1] * 5}
        before = final.apply_frozen_test_preprocessing(original, 28, [day], preproc)
        after = final.apply_frozen_test_preprocessing(changed, 28, [day], preproc)
        np.testing.assert_array_equal(before, after)

    def test_development_final_training_respects_each_embargo(self):
        development, test = market_fixture()
        full, _, _ = loader.build_combined_data(development, test)
        data = {"calendar": development.index,
            "eligible_by_history": {d: full["eligible_by_history"][d][:731] for d in loader.HISTORIES}}
        for history in loader.HISTORIES:
            train = final.final_training_indices(data, history)
            self.assertEqual(int(train.max()), 731 - history - 1)
            self.assertLess(development.index[train.max()], pd.Timestamp("2025-01-01", tz="UTC"))

    def test_freeze_rejects_modified_code_and_protocol_before_raw_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "script.py").write_text("original\n")
            protocol = root / "protocol.json"
            protocol.write_text("{}\n")
            freeze = root / "freeze.json"
            freeze.write_text(json.dumps({"final_jobs": [{"history_days": 7}],
                "code_sha256": {"script.py": loader.digest(root / "script.py")},
                "protocol_sha256": loader.digest(protocol)}))
            loader.validate_freeze(freeze, root=root, protocol_path=protocol)
            (root / "script.py").write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "Code changed"):
                loader.validate_freeze(freeze, root=root, protocol_path=protocol)
            protocol.write_text('{"changed": true}\n')
            with self.assertRaisesRegex(ValueError, "Protocol differs"):
                loader.validate_freeze(freeze, root=root, protocol_path=protocol)

    def test_archive_paths_cannot_escape_raw_root(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError): loader.safe_archive_path(directory, "../outside.zip")
            with self.assertRaises(ValueError): loader.safe_archive_path(directory, "/tmp/outside.zip")
            self.assertEqual(loader.safe_archive_path(directory, "valid/part.zip"), Path(directory) / "valid/part.zip")

    def test_raw_access_gate_requires_all_frozen_model_artifacts_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "script.py").write_text("original\n")
            protocol = root / "protocol.json"
            protocol.write_text("{}\n")
            freeze = root / "freeze.json"
            jobs = [{"history_days": 7, "model": "MLP", "seed": 4101}]
            freeze.write_text(json.dumps({"final_jobs": jobs,
                "code_sha256": {"script.py": loader.digest(root / "script.py")},
                "protocol_sha256": loader.digest(protocol)}))
            out = root / "out"
            out.mkdir()
            model = out / "model.pt"
            model.write_bytes(b"synthetic checkpoint identity; no training")
            manifest = out / "final_models_manifest.json"
            manifest.write_text(json.dumps({"freeze_sha256": loader.digest(freeze),
                "jobs": jobs, "artifact_sha256": {"model.pt": loader.digest(model)}}))
            loader.validate_final_models(freeze, root=root, out=out, protocol_path=protocol)
            model.write_bytes(b"changed checkpoint")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                loader.validate_final_models(freeze, root=root, out=out, protocol_path=protocol)

    def test_closed_job_matrix_requires_every_seed_history_and_model(self):
        jobs = [{"history_days": d, "model": m, "seed": s, "epochs": 10,
            "width": 16, "learning_rate": .001} for d in loader.HISTORIES
            for m in ("MLP", "LSTM", "CNN1D") for s in range(4101, 4111)]
        final.check_jobs(jobs)
        with self.assertRaises(ValueError): final.check_jobs(jobs[:-1])
        invalid = copy.deepcopy(jobs)
        invalid[0]["epochs"] = 0
        with self.assertRaises(ValueError): final.check_jobs(invalid)


if __name__ == "__main__":
    unittest.main()
