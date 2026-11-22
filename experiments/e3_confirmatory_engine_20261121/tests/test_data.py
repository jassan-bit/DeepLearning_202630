"""Software-only fixtures. No market queries and no final market labels."""
import datetime as dt
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from engine import data


def synthetic_bars(n=140):
    dates = [(dt.date(2020, 1, 1)+dt.timedelta(days=i)).isoformat() for i in range(n)]
    t = np.arange(n)
    close = 100*np.exp(.001*t+.01*np.sin(t/3))
    return {"dates": np.asarray(dates, dtype="U10"),
            "ohlcv": np.column_stack((close*.999, close*1.01, close*.99, close, 1000+t)),
            "split": np.ones(n)}


def zip_kline(date="2025-01-01", unit=1_000_000, volume="3.5"):
    start = int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp())*unit
    row = f"{start},10,12,9,11,{volume},{start+86400*unit-1},38.5,7,1,11,0\n"
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        archive.writestr("synthetic.csv", row)
    return out.getvalue()


class DataTests(unittest.TestCase):
    def test_binance_timestamp_milliseconds_microseconds_and_base_volume(self):
        for date, unit in (("2024-12-31", 1000), ("2025-01-01", 1_000_000)):
            bars = data.parse_binance_zip(zip_kline(date, unit), acquired_utc="2025-01-03T00:00:00+00:00")
            self.assertEqual(bars["dates"].tolist(), [date])
            self.assertEqual(bars["ohlcv"][0, 4], 3.5)
            self.assertEqual(bars["metadata"]["epoch_units_per_second"], [unit])

    def test_partial_daily_candle_rejected(self):
        with self.assertRaisesRegex(data.AdmissionError, "Incomplete"):
            data.parse_binance_zip(zip_kline(), acquired_utc="2025-01-01T12:00:00+00:00")

    def test_stooq_schema_and_duplicates(self):
        raw = b"Date,Open,High,Low,Close,Volume\n2020-01-02,10,12,9,11,100\n"
        self.assertEqual(data.parse_stooq_csv(raw)["ohlcv"].shape, (1, 5))
        with self.assertRaisesRegex(data.AdmissionError, "Duplicate"):
            data.parse_stooq_csv(raw+raw.splitlines(keepends=True)[1])
        with self.assertRaises(data.AdmissionError):
            data.parse_stooq_csv(b"No data")

    def test_alpha_vantage_uses_raw_close_and_raw_volume_not_adjusted(self):
        obj = {"Meta Data": {"2. Symbol": "AAPL", "5. Time Zone": "US/Eastern"},
               "Time Series (Daily)": {"2020-01-02": {"1. open": "100", "2. high": "102", "3. low": "98",
               "4. close": "101", "5. adjusted close": "25", "6. volume": "1234", "7. dividend amount": "0", "8. split coefficient": "4"}}}
        bars = data.parse_alphavantage(json.dumps(obj).encode(), "AAPL")
        self.assertEqual(bars["ohlcv"][0, 3], 101)
        self.assertEqual(bars["ohlcv"][0, 4], 1234)
        self.assertEqual(bars["split"][0], 4)
        with self.assertRaises(data.AdmissionError):
            data.parse_alphavantage(b'{"Information":"premium"}', "AAPL")
        with self.assertRaisesRegex(data.AdmissionError, "Duplicate"):
            data.parse_alphavantage(b'{"x":1,"x":2}', "AAPL")

    def test_wiki_json_rejects_wrong_symbol_and_reports_cursor(self):
        obj = {"datatable": {"columns": [{"name": k} for k in ("ticker", "date", "open", "high", "low", "close", "volume", "split_ratio")],
                             "data": [["IBM", "2010-01-04", 10, 12, 9, 11, 100, 1]]}, "meta": {"next_cursor_id": "next"}}
        bars, cursor = data.parse_nasdaq_wiki(json.dumps(obj).encode(), "IBM")
        self.assertEqual(cursor, "next")
        self.assertEqual(bars["ohlcv"][0, 3], 11)
        with self.assertRaises(data.AdmissionError):
            data.parse_nasdaq_wiki(json.dumps(obj).encode(), "MSFT")

    def test_missing_nonfinite_duplicate_and_incoherent_are_rejected(self):
        bars = synthetic_bars(25)
        data.validate_bars(bars, bars["dates"].tolist())
        for defect in ("missing", "duplicate", "nan", "candle", "negative_volume"):
            broken = {k: v.copy() for k, v in bars.items()}
            if defect == "missing":
                broken = {k: np.delete(v, 10, axis=0) for k, v in broken.items()}
            elif defect == "duplicate":
                broken["dates"][10] = broken["dates"][9]
            elif defect == "nan":
                broken["ohlcv"][10, 3] = np.nan
            elif defect == "candle":
                broken["ohlcv"][10, 1] = 1
            else:
                broken["ohlcv"][10, 4] = -1
            with self.subTest(defect=defect), self.assertRaises(data.AdmissionError):
                data.validate_bars(broken, bars["dates"].tolist())

    def test_weekends_use_market_sessions_and_bar_indices_not_days(self):
        dates = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08",
                 "2020-01-09", "2020-01-10", "2020-01-13", "2020-01-14", "2020-01-15",
                 "2020-01-16", "2020-01-17", "2020-01-21", "2020-01-22", "2020-01-23"]
        bars = synthetic_bars(15)
        bars["dates"] = np.asarray(dates)
        self.assertEqual(data.validate_bars(bars, dates)["missing"], 0)
        arrays, _ = data.build_windows(bars)
        self.assertEqual(arrays["support_start"].tolist(), [0])
        self.assertEqual(arrays["support_end"].tolist(), [14])
        self.assertEqual(arrays["dates"].tolist(), ["2020-01-23"])

    def test_sigma_population_next_bar_alignment_and_overlap_floor(self):
        bars = synthetic_bars(40)
        arrays, _ = data.build_windows(bars)
        np.testing.assert_array_equal(arrays["raw"][0], bars["ohlcv"][7:14])
        closes = bars["ohlcv"][:, 3]
        self.assertAlmostEqual(arrays["y"][0], np.diff(np.log(closes[7:15])).std(ddof=0), places=15)
        expected = [np.diff(np.log(closes[k-7:k+1])).std(ddof=0) for k in range(7, 14)]
        np.testing.assert_allclose(arrays["known_sigma_history"][0], expected)
        six = np.diff(np.log(closes[7:14]))
        self.assertAlmostEqual(arrays["known_floor_A"][0], ((six-six.mean())**2).sum()/7)
        self.assertTrue(np.all(arrays["y"]**2 >= arrays["known_floor_A"]-1e-15))

    def test_no_future_target_or_observed_state_is_constructed_for_development(self):
        bars = synthetic_bars(60)
        cutoff = str(bars["dates"][35])
        a, _ = data.build_windows(bars, target_end=cutoff)
        changed = {k: v.copy() for k, v in bars.items()}
        changed["ohlcv"][36:, :] *= 400
        b, _ = data.build_windows(changed, target_end=cutoff)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        self.assertLess(a["observed_sigma_dates"][-1], cutoff)

    def test_split_support_rule_prespecified_without_adjusted_volume(self):
        bars = synthetic_bars(50)
        bars["split"][20] = 2
        arrays, audit = data.build_windows(bars)
        excluded = set(audit["split_support_excluded_dates"])
        self.assertEqual(excluded, set(bars["dates"][20:35]))
        self.assertNotIn(str(bars["dates"][20]), arrays["dates"].tolist())
        self.assertEqual(arrays["raw"][-1, -1, 4], bars["ohlcv"][-2, 4])

    def _partitions(self):
        arrays, _ = data.build_windows(synthetic_bars(140))
        return data.split_by_support(arrays, {"fit": [0, 34], "selection": [35, 69],
                "planning": [70, 104], "calibration": [105, 139]}, embargo_bars=2)

    def test_support_purges_and_continuous_causal_prefix(self):
        parts, audit = self._partitions()
        self.assertEqual(parts["fit"]["support_end"][0], 14)
        self.assertEqual(parts["selection"]["support_end"][0], 51)
        self.assertEqual(parts["selection"]["support_start"][0], 37)
        self.assertEqual(audit["selection"]["embargo_bars"], 2)
        for part, values in parts.items():
            data.validate_samples(values)
            for i, stop in enumerate(values["history_end_positions"]):
                self.assertEqual(stop+7, values["support_end"][i])
                self.assertLess(values["observed_sigma_dates"][stop-1], values["dates"][i])
        self.assertEqual(parts["selection"]["observed_sigma_bar_indices"][0], 7)
        with self.assertRaisesRegex(data.AdmissionError, "Development"):
            data.split_by_support(data.build_windows(synthetic_bars())[0], {"test": [0, 139]})

    def test_partition_loader_isolated_and_hash_checked(self):
        parts, audit = self._partitions()
        with tempfile.TemporaryDirectory() as tmp:
            record = data.write_panel(tmp, parts, provenance={"fixture": True}, rights={}, exposure={}, support_audit=audit, fixture=True)
            Path(tmp, "selection.npz").write_bytes(b"deliberately corrupted OTHER partition")
            fit = data.load_partition(tmp, "fit")
            self.assertEqual(len(fit["y"]), record["partition_counts"]["fit"])
            with self.assertRaisesRegex(data.AdmissionError, "SHA256"):
                data.load_partition(tmp, "selection")
            Path(tmp, "fit.npz").write_bytes(Path(tmp, "fit.npz").read_bytes()+b"tamper")
            with self.assertRaisesRegex(data.AdmissionError, "SHA256"):
                data.load_partition(tmp, "fit")
            with self.assertRaisesRegex(data.AdmissionError, "sealed"):
                data.load_partition(tmp, "test")

    def test_admission_metadata_hash_and_path_escape(self):
        parts, audit = self._partitions()
        with tempfile.TemporaryDirectory() as tmp:
            record = data.write_panel(tmp, parts, provenance={}, rights={}, exposure={}, support_audit=audit, fixture=True)
            record["partition_counts"]["fit"] += 1
            Path(tmp, "admission.json").write_text(json.dumps(record))
            with self.assertRaisesRegex(data.AdmissionError, "metadata hash"):
                data.load_partition(tmp, "fit")
            with self.assertRaisesRegex(data.AdmissionError, "Unsafe"):
                data._safe_path(Path(tmp), "../secret.npz")

    def test_market_network_disabled_before_any_access_and_real_evidence_required(self):
        with patch.object(data, "_network_get", side_effect=AssertionError("network accessed")):
            with self.assertRaisesRegex(data.AdmissionError, "activation"):
                data.collect_development({}, "/tmp/unused", rights_receipt={}, exposure_receipt={})
            with self.assertRaisesRegex(data.AdmissionError, "Evidence"):
                data.collect_development({}, "/tmp/unused", rights_receipt={}, exposure_receipt={}, enable_market_acquisition=True)
        with self.assertRaisesRegex(data.AdmissionError, "Evidence"):
            data.write_panel("/tmp/unused", self._partitions()[0], provenance={}, rights={}, exposure={}, support_audit={})

    def test_owner_form_not_auto_signed_and_future_reserve_requires_all_support_after_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            form = Path(tmp, "form.json"); evidence = Path(tmp, "author-export.txt"); output = Path(tmp, "receipt.json")
            evidence.write_text("SOFTWARE TEST ACTOR, NOT A REAL AUTHOR DECLARATION")
            form.write_text("{}")
            with self.assertRaisesRegex(data.AdmissionError, "actor_name"):
                data.record_owner_evidence(form, evidence, output)
            value = {"actor_name": "SOFTWARE FIXTURE", "actor_contact": "fixture@example.invalid", "actor_role": "fixture",
                "statement_utc": "2026-11-22T00:00:00+00:00", "protocol_sha256": "a"*64,
                "exposure_inventory_sha256": "b"*64, "reserve_mode": "post_protocol", "evidence_origin": "fixture",
                "acknowledge_prior_D00_D03_exposed": True, "protocol_lock_utc": "2026-11-21T23:59:00+00:00",
                "first_raw_support_session": "2026-11-21"}
            form.write_text(json.dumps(value))
            with self.assertRaisesRegex(data.AdmissionError, "Every final support"):
                data.record_owner_evidence(form, evidence, output)
            value["first_raw_support_session"] = "2026-11-22"; form.write_text(json.dumps(value))
            receipt = data.record_owner_evidence(form, evidence, output)
            self.assertFalse(receipt["cryptographic_signature_verified"])
            self.assertIn("not_identity_verification", receipt["kind"])

    def test_bounded_wiki_collector_pagination_and_no_secret_in_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence = Path(tmp, "fixture-evidence.txt")
            evidence.write_text("SOFTWARE FIXTURE, NO REAL GRANT OR ACTOR")
            common = {"actor_name": "SOFTWARE FIXTURE", "actor_contact": "fixture@example.invalid",
                      "evidence_path": str(evidence), "evidence_sha256": data.sha256_bytes(evidence.read_bytes())}
            rights = dict(common, status="approved", allow_derived_publication=True,
                          allow_raw_redistribution=True, license="SOFTWARE FIXTURE", scope="fixture only", source_url="https://example.invalid")
            exposure = dict(common, status="audited")
            columns = [{"name": k} for k in ("ticker", "date", "open", "high", "low", "close", "volume", "split_ratio")]
            def page(date, cursor):
                return json.dumps({"datatable": {"columns": columns,
                     "data": [["IBM", date, 10, 12, 9, 11, 100, 1]]}, "meta": {"next_cursor_id": cursor}}).encode()
            config = {"provider": "nasdaq_wiki", "instrument": "IBM", "development_start": "2010-01-04",
                      "development_end": "2010-01-05", "first_reserved_raw_session": "2010-02-01"}
            with patch.dict("os.environ", {"NASDAQ_DATA_LINK_API_KEY": "PRIVATE_FIXTURE_SECRET"}), \
                 patch.object(data, "_network_get", side_effect=[page("2010-01-04", "NEXT"), page("2010-01-05", None)]) as network:
                result = data.collect_development(config, Path(tmp, "raw"), rights_receipt=rights,
                       exposure_receipt=exposure, enable_market_acquisition=True)
            self.assertEqual(len(result["files"]), 2)
            self.assertIn("qopts.cursor_id=NEXT", network.call_args_list[1].args[0])
            self.assertNotIn("PRIVATE_FIXTURE_SECRET", json.dumps(result))
            self.assertIn("date.gte=2010-01-04", network.call_args_list[0].args[0])

    def test_development_designation_without_final_protocol_or_reserve_allows_pilot_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            license_path = root/"fixture-primary-notice.txt"
            license_path.write_text("SOFTWARE FIXTURE PRIMARY NOTICE; NOT A REAL LICENSE")
            evidence_path = root/"fixture-development-declaration.txt"
            evidence_path.write_text("SOFTWARE FIXTURE ACTOR designates all supplied dataset observations as exposed development. No final reserve.")
            form = {"actor_name": "SOFTWARE FIXTURE", "actor_contact": "fixture@example.invalid", "actor_role": "fixture",
                    "statement_utc": "2020-05-20T00:00:00+00:00",
                    "reserve_mode": "development_only", "evidence_origin": "software fixture", "acknowledge_prior_D00_D03_exposed": True,
                    "all_data_declared_exposed_development": True, "authorize_test_preparation": False,
                    "observation_cutoff_utc": "2020-05-20T00:00:00+00:00", "dataset_scope": "all synthetic IBM fixture dates 2010-01-01..2010-05-20",
                    "source_url": "https://example.invalid/fixture", "source_license_evidence_path": str(license_path),
                    "source_license_evidence_sha256": data.sha256_bytes(license_path.read_bytes())}
            form_path = root/"development-form.json"; form_path.write_text(json.dumps(form))
            exposure = data.record_owner_evidence(form_path, evidence_path, root/"development-exposure.json")
            self.assertNotIn("protocol_sha256", exposure["form"])
            self.assertNotIn("custodian_seal_sha256", exposure["form"])
            self.assertFalse(exposure["final_reserve_claim"])
            self.assertFalse(exposure["cryptographic_signature_verified"])
            self.assertEqual(exposure["reserved_status"], "not_reserved_all_data_exposed")
            rights = {"status": "approved", "actor_name": "SOFTWARE FIXTURE", "actor_contact": "fixture@example.invalid",
                      "evidence_path": str(license_path), "evidence_sha256": data.sha256_bytes(license_path.read_bytes()),
                      "license": "SOFTWARE FIXTURE", "source_url": form["source_url"], "scope": "fixture software only",
                      "allow_derived_publication": True, "allow_raw_redistribution": True}
            bars = synthetic_bars(140)
            columns = [{"name": k} for k in ("ticker", "date", "open", "high", "low", "close", "volume", "split_ratio")]
            payload = json.dumps({"datatable": {"columns": columns,
                "data": [["IBM", str(date), *row.tolist(), 1] for date, row in zip(bars["dates"], bars["ohlcv"])]},
                "meta": {"next_cursor_id": None}}).encode()
            # No final protocol, test date, power output or reserve date is supplied.
            # Move the synthetic fixture dates into the supported historical feed.
            dates = np.asarray([(dt.date(2010, 1, 1)+dt.timedelta(days=i)).isoformat() for i in range(140)])
            value = json.loads(payload)
            for row, date in zip(value["datatable"]["data"], dates): row[1] = str(date)
            payload = json.dumps(value).encode()
            config = {"provider": "nasdaq_wiki", "instrument": "IBM", "development_start": str(dates[0]), "development_end": str(dates[-1])}
            with patch.dict("os.environ", {"NASDAQ_DATA_LINK_API_KEY": "PRIVATE_FIXTURE_SECRET"}), \
                 patch.object(data, "_network_get", return_value=payload):
                manifest = data.collect_development(config, root/"raw", rights_receipt=rights, exposure_receipt=exposure,
                                                    enable_market_acquisition=True)
            split = {"raw_start": str(dates[0]), "raw_end": str(dates[-1]), "calendar": "24/7", "embargo_bars": 2,
                     "raw_bounds": {"fit": [0, 34], "selection": [35, 69], "planning": [70, 104], "calibration": [105, 139]}}
            admission = data.prepare_development(root/"raw/raw_manifest.json", split, root/"pilot", rights=rights, exposure=exposure)
            self.assertEqual(admission["status"], "development_admitted")
            self.assertEqual(admission["provenance"]["bar_index_origin"], str(dates[0]))
            self.assertEqual(admission["provenance"]["bar_index_convention"], data.BAR_INDEX_CONVENTION)
            self.assertEqual(admission["reserved_status"], "test_not_acquired")
            self.assertEqual(len(data.load_partition(root/"pilot", "planning")["y"]), 19)
            self.assertNotIn("test", admission["partition_paths"])
            with self.assertRaisesRegex(data.AdmissionError, "sealed"):
                data.load_partition(root/"pilot", "test")
            with self.assertRaisesRegex(data.AdmissionError, "not prepared"):
                data.load_partition(root/"pilot", "test", allow_test=True)
            # This same development declaration can never be promoted to a final reserve.
            protocol = root/"protocol.json"; protocol.write_text('{"fixture":true}')
            checkpoints = root/"checkpoints.json"; checkpoints.write_text('{"fixture":true}')
            protocol_hash = data.sha256_bytes(protocol.read_bytes())
            final_manifest = dict(manifest, scope="reserved_final", protocol_sha256=protocol_hash)
            final_manifest.pop("manifest_sha256")
            final_manifest["manifest_sha256"] = data.identity(final_manifest)
            (root/"raw/final-manifest.json").write_text(json.dumps(final_manifest))
            auth = dict(rights, status="authorized", authorize_test_preparation=True, protocol_path=str(protocol),
                        protocol_sha256=protocol_hash, sources_methods_size_frozen=True,
                        selected_checkpoints_manifest_path=str(checkpoints), selected_checkpoints_sha256=data.sha256_bytes(checkpoints.read_bytes()))
            with self.assertRaisesRegex(data.AdmissionError, "Exposure declaration not bound"):
                data.prepare_final_partition(root/"raw/final-manifest.json", {"protocol_sha256": protocol_hash, "raw_bounds": {"test": [0, 139]}},
                       root/"blocked-final", rights=rights, exposure=exposure, authorization=auth, enable_final_preparation=True)
            exposure["form"]["protocol_sha256"] = protocol_hash
            with self.assertRaisesRegex(data.AdmissionError, "Authentic reserve mode missing"):
                data.prepare_final_partition(root/"raw/final-manifest.json", {"protocol_sha256": protocol_hash, "raw_bounds": {"test": [0, 139]}},
                       root/"still-blocked-final", rights=rights, exposure=exposure, authorization=auth, enable_final_preparation=True)
            form["authorize_test_preparation"] = True; form_path.write_text(json.dumps(form))
            with self.assertRaisesRegex(data.AdmissionError, "cannot authorize"):
                data.record_owner_evidence(form_path, evidence_path, root/"invalid-receipt.json")

    def test_final_preparation_explicit_guard_then_synthetic_success(self):
        with self.assertRaisesRegex(data.AdmissionError, "activation"):
            data.prepare_final_partition("not-read", {}, "not-written", rights={}, exposure={}, authorization={})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root/"fixture-declaration.txt"; evidence.write_text("SOFTWARE FIXTURE; NOT AUTHENTIC MARKET AUTHORIZATION")
            common = {"actor_name": "SOFTWARE FIXTURE", "actor_contact": "fixture@example.invalid",
                      "evidence_path": str(evidence), "evidence_sha256": data.sha256_bytes(evidence.read_bytes())}
            rights = dict(common, status="approved", allow_derived_publication=True,
                          allow_raw_redistribution=True, license="SOFTWARE FIXTURE", scope="fixture only", source_url="https://example.invalid")
            exposure = dict(common, status="audited")
            protocol = root/"protocol.json"; protocol.write_text('{"fixture":true}')
            checkpoints = root/"checkpoints.json"; checkpoints.write_text('{"fixture":true}')
            protocol_hash = data.sha256_bytes(protocol.read_bytes())
            exposure["form"] = {"protocol_sha256": protocol_hash, "reserve_mode": "external_custodian",
                                "custodian_seal_sha256": "c"*64, "custodian_access_log_sha256": "d"*64}
            auth = dict(common, status="authorized", authorize_test_preparation=True, protocol_path=str(protocol),
                        protocol_sha256=protocol_hash, selected_checkpoints_manifest_path=str(checkpoints),
                        selected_checkpoints_sha256=data.sha256_bytes(checkpoints.read_bytes()), sources_methods_size_frozen=True)
            bars = synthetic_bars(80)
            auth["development_bar_index_origin"] = str(bars["dates"][0])
            columns = [{"name": k} for k in ("ticker", "date", "open", "high", "low", "close", "volume", "split_ratio")]
            payload = json.dumps({"datatable": {"columns": columns,
                "data": [["IBM", str(date), *row.tolist(), 1] for date, row in zip(bars["dates"], bars["ohlcv"])]}}).encode()
            (root/"synthetic.json").write_bytes(payload)
            manifest = {"scope": "reserved_final", "provider": "nasdaq_wiki", "instrument": "IBM", "protocol_sha256": protocol_hash,
                        "custodian_seal_sha256": "c"*64,
                        "files": [{"path": "synthetic.json", "sha256": data.sha256_bytes(payload)}]}
            manifest["manifest_sha256"] = data.identity(manifest)
            (root/"manifest.json").write_text(json.dumps(manifest))
            split = {"raw_start": str(bars["dates"][0]), "raw_end": str(bars["dates"][-1]), "calendar": "24/7",
                     "protocol_sha256": protocol_hash, "target_start": str(bars["dates"][40]), "target_end": str(bars["dates"][-1]),
                     "raw_bounds": {"test": [40, 79]}, "embargo_bars": 2, "fixed_test_count": 24}
            result = data.prepare_final_partition(root/"manifest.json", split, root/"final", rights=rights,
                        exposure=exposure, authorization=auth, enable_final_preparation=True)
            self.assertEqual(result["partition_counts"], {"test": 24})
            self.assertEqual(result["provenance"]["bar_index_origin"], str(bars["dates"][0]))
            self.assertEqual(result["provenance"]["bar_index_convention"], data.BAR_INDEX_CONVENTION)
            with self.assertRaisesRegex(data.AdmissionError, "sealed"):
                data.load_partition(root/"final", "test")
            final = data.load_partition(root/"final", "test", allow_test=True)
            self.assertEqual(final["support_start"][0], 42)
            self.assertEqual(final["support_end"][0], 56)
            # Valid evidence cannot authorize a different numeric clock. Both
            # failures occur before reserved labels/windows are constructed.
            for origin, diagnostic in ((None, "origin required"), (str(bars["dates"][1]), "origin differs")):
                changed = dict(auth)
                if origin is None:
                    changed.pop("development_bar_index_origin")
                else:
                    changed["development_bar_index_origin"] = origin
                with self.subTest(origin=origin), patch.object(data, "build_windows", side_effect=AssertionError("reserved labels opened")), \
                     self.assertRaisesRegex(data.AdmissionError, diagnostic):
                    data.prepare_final_partition(root/"manifest.json", split, root/"blocked-origin", rights=rights,
                        exposure=exposure, authorization=changed, enable_final_preparation=True)
            # Supplying only a suffix of the raw history cannot silently reset
            # test indices even if its target calendar and prices are identical.
            shifted = json.loads(payload)
            shifted["datatable"]["data"] = shifted["datatable"]["data"][20:]
            shifted_raw = json.dumps(shifted).encode()
            (root/"shifted.json").write_bytes(shifted_raw)
            shifted_manifest = dict(manifest)
            shifted_manifest.pop("manifest_sha256")
            shifted_manifest["files"] = [{"path": "shifted.json", "sha256": data.sha256_bytes(shifted_raw)}]
            shifted_manifest["manifest_sha256"] = data.identity(shifted_manifest)
            (root/"shifted-manifest.json").write_text(json.dumps(shifted_manifest))
            shifted_split = dict(split, raw_start=str(bars["dates"][20]))
            with patch.object(data, "build_windows", side_effect=AssertionError("reserved labels opened")), \
                 self.assertRaisesRegex(data.AdmissionError, "origin differs"):
                data.prepare_final_partition(root/"shifted-manifest.json", shifted_split, root/"blocked-reset", rights=rights,
                    exposure=exposure, authorization=auth, enable_final_preparation=True)


if __name__ == "__main__":
    unittest.main()
